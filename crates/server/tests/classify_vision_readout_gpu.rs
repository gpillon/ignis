//! Does the typed-option readout work when the evidence is a **picture**?
//!
//! `classify_readout_gpu.rs` established the readout on text: one prefill,
//! zero decoded tokens, the last position's logits restricted to the declared
//! options' letter tokens, with the declared options holding a median 99.8% of
//! the distribution. This asks the same mechanical question of a multimodal
//! prompt, where the evidence is an image and only the criterion and the
//! options are text.
//!
//! It matters because TypeSafe's Jev cannot do it: its `state` is
//! `string | object | array`. If the readout holds over an image, a decision
//! endpoint on ignis can take content parts as its evidence, which is the one
//! thing it would do that Jev does not.
//!
//! The seam is already there: `step::prefill_program_multimodal` takes the
//! same `out_logits` buffer as the text path (GitHub #178 kept it), and
//! `support::vision_canary::prefill_prompt` drives the chunk loop.
//!
//! **Four rows are a mechanism check, not a measurement.** They say whether
//! the model still puts its mass on the declared letters when it is looking
//! at a picture; they do not say how accurate a vision decision endpoint
//! would be. The vision canary fixture is what exists with a known answer.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38).

#![cfg(feature = "cuda")]

#[path = "support/mod.rs"]
mod support;

use std::path::Path;

use ignis_artifact::{
    ChatMessage, ChatRenderOptions, ContentPart, CudaDevice, FrontendSet, MessageContent,
    ModelScope, Reader, Role, bind_model_scope_27b_with, materialize,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_options;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step;
use ignis_core::vision::{Multimodal, vision_item_control};
use ignis_core::{KvFormat, RopeScaling, Vision};
use serde_json::json;

use support::vision_canary::{load_canaries, prefill_prompt};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 2048;
/// The served chunk width. Only the *last* chunk's logits are the readout, so
/// a prompt this short is deliberately one chunk; the spanning-chunk case is
/// the vision canary's robustness test, not the served path.
const PREFILL_CHUNK: u32 = 1024;
const LETTERS: &str = "ABCDEFGHIJKLMNOP";

/// `classify_readout_gpu.rs`'s system instruction, unchanged: it already says
/// "the supplied evidence" without saying what kind.
const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

/// One typed decision over a vision canary image: the criterion, the option
/// descriptions in slot order, and the index of the right one.
///
/// The distractors are deliberately close — transposed digits, an adjacent
/// count, a plausible near-miss of the same error string — so a hit is the
/// model reading the picture rather than eliminating nonsense.
struct VisionDecision {
    canary: &'static str,
    criterion: &'static str,
    options: &'static [&'static str],
    label: usize,
}

const DECISIONS: &[VisionDecision] = &[
    VisionDecision {
        canary: "number",
        criterion: "Which number is shown in the image?",
        options: &["42", "47", "74"],
        label: 1,
    },
    VisionDecision {
        canary: "colour",
        criterion: "What colour is the square in the image?",
        options: &["Blue", "Red", "Green"],
        label: 1,
    },
    VisionDecision {
        canary: "circles",
        criterion: "How many circles are in the image?",
        options: &["Two", "Three", "Four"],
        label: 1,
    },
    VisionDecision {
        canary: "text",
        criterion: "What does the text in the image say?",
        options: &[
            "error: config.yaml not found",
            "warning: disk full",
            "error: config.yml missing",
        ],
        label: 0,
    },
];

fn new_pool() -> SeqPool {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: (MAX_CONTEXT / 64) * 2,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 2,
            retained_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"))
}

fn logsumexp(values: impl Iterator<Item = f32> + Clone) -> f64 {
    let maximum = values
        .clone()
        .fold(f64::NEG_INFINITY, |acc, v| acc.max(f64::from(v)));
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + values.map(|v| (f64::from(v) - maximum).exp()).sum::<f64>().ln()
}

fn softmax(values: &[f32]) -> Vec<f64> {
    let total = logsumexp(values.iter().copied());
    values.iter().map(|&v| (f64::from(v) - total).exp()).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn typed_option_logits_are_readable_over_an_image() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let processor = frontend
        .vision_processor()
        .unwrap_or_else(|e| panic!("vision processor: {e}"));
    let canaries = load_canaries(&frontend);
    assert!(!canaries.is_empty(), "the vision canary fixture must carry images");

    // Each letter is one token that decodes back to itself. The text path's
    // third check -- that appending the letter to the rendered prompt extends
    // the tokenization by exactly that token -- has no counterpart here:
    // `prepare_prompt` returns scattered token ids, not a string to append to.
    // The prompt ends in the same `<|im_start|>assistant\n` tail the text path
    // verified clean on 144 of 144 rows, so this is a stated assumption.
    let slots: Vec<u32> = (0..LETTERS.len())
        .map(|index| {
            let letter = &LETTERS[index..index + 1];
            let encoded = frontend
                .tokenizer()
                .encode(letter)
                .unwrap_or_else(|e| panic!("encode {letter}: {e}"));
            assert_eq!(encoded.len(), 1, "slot {letter} must be one token");
            assert_eq!(
                frontend.tokenizer().decode(&encoded).ok().as_deref(),
                Some(letter),
                "slot {letter} must round-trip"
            );
            encoded[0]
        })
        .collect();

    let (plan, handles) = bind_model_scope_27b_with(&reader, ModelScope { draft: None, vision: true })
        .unwrap_or_else(|e| panic!("bind with vision: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(a) => a,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + vision: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b_with_options(
        &reader,
        &artifact,
        &handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::Bf16,
        None,
        Some(Vision::default()),
        RopeScaling::NONE,
    )
    .unwrap_or_else(|e| panic!("ignis_model_load with vision: {e}"));
    let pool = new_pool();
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = vec![0f32; vocab];

    let (mut hits, mut in_slots, mut scored) = (0usize, 0usize, 0usize);
    let mut masses: Vec<f64> = Vec::new();
    for decision in DECISIONS {
        let canary = canaries
            .iter()
            .find(|c| c.id == decision.canary)
            .unwrap_or_else(|| panic!("the fixture is missing canary {}", decision.canary));
        let payload = json!({
            "criterion": decision.criterion,
            "options": decision.options.iter().enumerate().map(|(index, description)| json!({
                "letter": &LETTERS[index..index + 1],
                "description": description,
            })).collect::<Vec<_>>(),
        });
        // The image is the evidence, so it comes first -- the same ordering
        // the text path uses, and the one that would let a shared image
        // become a shared prefix across several questions.
        let messages = [
            ChatMessage {
                role: Role::System,
                content: MessageContent::Text(DIRECT_SYSTEM.to_owned()),
                tool_calls: Vec::new(),
                reasoning_content: None,
            },
            ChatMessage {
                role: Role::User,
                content: MessageContent::Parts(vec![
                    ContentPart::Image { url: None },
                    ContentPart::Text(payload.to_string()),
                ]),
                tool_calls: Vec::new(),
                reasoning_content: None,
            },
        ];
        let prepared = frontend
            .prepare_prompt(
                &processor,
                &messages,
                &[&canary.image],
                ChatRenderOptions { enable_thinking: false, ..Default::default() },
                None,
            )
            .unwrap_or_else(|e| panic!("{}: prepare prompt: {e}", decision.canary));
        let (token_ids, prompt) = Multimodal::from_prepared(prepared);
        let tokens: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        assert!(
            tokens.len() <= PREFILL_CHUNK as usize,
            "{}: {} tokens would span chunks, and only the last chunk's logits are the readout",
            decision.canary,
            tokens.len()
        );
        assert_eq!(prompt.media.len(), 1, "{}: one image", decision.canary);
        let item = &prompt.media[0];
        let control = vision_item_control(item.grid);
        // One embedding may be live at a time, so it is encoded, used and
        // dropped inside this iteration.
        let embedding = step::encode_media(&model, item.grid, &item.patches, &control)
            .unwrap_or_else(|e| panic!("{}: encode: {e}", decision.canary));

        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
        logits.fill(0.0);
        let started = std::time::Instant::now();
        prefill_prompt(
            &model,
            &pool,
            &mut sequence,
            &tokens,
            &prompt,
            &embedding,
            PREFILL_CHUNK,
            &mut logits,
        )
        .unwrap_or_else(|e| panic!("{}: prefill: {e}", decision.canary));
        let elapsed = started.elapsed();
        drop(sequence);
        drop(embedding);

        assert!(
            logits.iter().all(|v| v.is_finite()),
            "{}: the readout's logits must be finite",
            decision.canary
        );
        let used = &slots[..decision.options.len()];
        let option_logits: Vec<f32> = used.iter().map(|&slot| logits[slot as usize]).collect();
        let probabilities = softmax(&option_logits);
        let predicted = argmax(&option_logits);
        let full_vocab_argmax = argmax(&logits) as u32;
        let argmax_in_slots = used.contains(&full_vocab_argmax);
        let allowed_mass =
            (logsumexp(option_logits.iter().copied()) - logsumexp(logits.iter().copied())).exp();

        scored += 1;
        hits += usize::from(predicted == decision.label);
        in_slots += usize::from(argmax_in_slots);
        masses.push(allowed_mass);
        let winner = frontend
            .tokenizer()
            .decode(&[full_vocab_argmax])
            .unwrap_or_else(|_| "<undecodable>".to_owned());
        eprintln!(
            "ignis classify/vision {}: {} tokens ({} image columns), pred={} label={} p={:?} mass={allowed_mass:.4} argmax={winner:?} in_slots={argmax_in_slots} prefill={:.1} ms",
            decision.canary,
            tokens.len(),
            item.token_span.count,
            decision.options[predicted],
            decision.options[decision.label],
            probabilities.iter().map(|p| (p * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    masses.sort_by(f64::total_cmp);
    eprintln!(
        "ignis classify/vision: {hits}/{scored} correct, argmax_in_slots={in_slots}/{scored}, allowed_mass min={:.4} median={:.4}",
        masses[0],
        masses[masses.len() / 2]
    );
    // The mechanism is the claim, not the accuracy: the declared options must
    // still hold the distribution when the evidence is a picture. Four rows
    // cannot support an accuracy claim and this test does not make one.
    assert_eq!(
        in_slots, scored,
        "the unrestricted winner must be a declared option letter on every row"
    );
    assert!(
        masses[0] > 0.5,
        "every row's declared options must hold most of the distribution (min was {:.4})",
        masses[0]
    );
}
