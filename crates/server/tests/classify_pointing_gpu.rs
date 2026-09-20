//! Can a constrained readout produce **coordinates**?
//!
//! The typed readout reads one position, so it yields one symbol — never a
//! pair of numbers. The extension that would is a **constrained decode**: one
//! prefill, then K steps, each of which restricts the next token to its own
//! declared alphabet and forces the winner back in. For a point that alphabet
//! is the digits `0`–`9` (single tokens here, `slot_alphabet.rs`), three per
//! axis on a 0–999 scale, which rescales onto the original image.
//!
//! Two things are measured per scene, in this order, because the second
//! depends on the first:
//!
//!   1. **Free probe** — the model is asked for the button's position with no
//!      constraint at all and decoded greedily. This is what says which scale
//!      it thinks in: Qwen2-VL emitted coordinates normalized to 0–1000,
//!      Qwen3-VL moved to absolute pixels, and the grounding vocabulary alone
//!      (`<|box_start|>`, present here) does not say which. Constraining the
//!      model to three digits when it thinks in pixels would force it out of
//!      its own distribution and the error would look like a pointing failure
//!      rather than a units mismatch.
//!   2. **Constrained readout** — a forced `{"x":` prefix, three digits read
//!      and forced, a forced `,"y":`, three more. Each digit carries its
//!      restricted probability and the mass the digits hold against the whole
//!      vocabulary, which is what says whether the constraint is reading the
//!      model or overruling it.
//!
//! The fixture is three synthetic 4096x4096 app screenshots with a blue button
//! at a known centre, beside distractors of other colours so that "the blue
//! one" is a real choice rather than the only rectangle. Synthetic flat-colour
//! chrome is not a real screenshot and this test does not claim it is.
//!
//! At the default 32,768-token vision budget a 4096x4096 image is **not**
//! downscaled, so the prompt runs to ~16K tokens and the context here is sized
//! for that rather than for the 2K the other vision tests use.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38).

#![cfg(feature = "cuda")]

#[path = "support/mod.rs"]
mod support;

use std::path::{Path, PathBuf};

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
use serde::Deserialize;

use support::vision_canary::{argmax_lowest_id, prefill_prompt};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// A 4096x4096 image is ~16K tokens at the default vision budget, and the
/// forced suffix adds a dozen.
const MAX_CONTEXT: u32 = 20_480;
const PREFILL_CHUNK: u32 = 1024;
/// Digits per axis, and therefore the scale: 0-999.
const DIGITS: usize = 3;
const SCALE: f64 = 999.0;
/// How far the free probe is allowed to run before it is cut.
const PROBE_TOKENS: usize = 48;

/// The scale the prompt declares. Stated rather than left implicit: the free
/// probe reports what the model does unprompted, and this says what it is
/// asked for.
const POINT_SYSTEM: &str = "You are given a screenshot and an instruction. Answer with the position on the screen the instruction refers to. Use a 0-999 scale on each axis, where x=0 is the left edge, x=999 the right edge, y=0 the top edge and y=999 the bottom edge. Reply with only a JSON object of the form {\"x\":NNN,\"y\":NNN}, three digits each.";

const PROBE_QUESTION: &str =
    "Locate the blue Save button in this screenshot and output its bounding box.";
const INSTRUCTION: &str = "click the blue button";

#[derive(Deserialize)]
struct Manifest {
    side: u32,
    scenes: Vec<Scene>,
}

#[derive(Deserialize)]
struct Scene {
    id: String,
    image: String,
    blue_box: [i64; 4],
    blue_centre: [i64; 2],
    blue_size: [i64; 2],
}

/// The pointing scenes, committed beside this test: under
/// `IGNIS_GPU_PROFILE=1` a missing fixture is a hard failure, and the images
/// are not reproducible from `generate.py` alone — the button labels are drawn
/// with whatever font the generating machine had.
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pointing")
}

fn new_pool() -> SeqPool {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT / 64,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
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

/// The first four standalone integers in `text` — enough to pull a box out of
/// the free probe whether it came back as ```` ```json [{"bbox_2d": [a,b,c,d]}]
/// ``` ```` or as a bare `[a, b, c, d]`.
///
/// "Standalone" is the whole difficulty: a run of digits glued to a word is
/// part of an identifier, not a coordinate. Without that rule the `2` of
/// `bbox_2d` is read as the box's first number and every coordinate shifts by
/// one, which reads as the constraint overruling the model when it is only the
/// comparison that is wrong.
///
/// This is a comparison aid, never a parser the endpoint would ship: a
/// constrained readout exists precisely so that nothing has to parse.
fn first_four_integers(text: &str) -> Option<[i64; 4]> {
    let chars: Vec<char> = text.chars().collect();
    let mut found: Vec<i64> = Vec::new();
    let mut index = 0usize;
    while index < chars.len() && found.len() < 4 {
        if !chars[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        let start = index;
        while index < chars.len() && chars[index].is_ascii_digit() {
            index += 1;
        }
        let glued_left = start > 0 && (chars[start - 1].is_alphabetic() || chars[start - 1] == '_');
        let glued_right =
            index < chars.len() && (chars[index].is_alphabetic() || chars[index] == '_');
        if glued_left || glued_right {
            continue;
        }
        found.push(chars[start..index].iter().collect::<String>().parse().ok()?);
    }
    (found.len() == 4).then(|| [found[0], found[1], found[2], found[3]])
}

/// One constrained step: restrict `logits` to `slots`, return the winning
/// index, its restricted probability, and how much of the whole distribution
/// the slots hold.
fn constrained_pick(logits: &[f32], slots: &[u32]) -> (usize, f64, f64) {
    let restricted: Vec<f32> = slots.iter().map(|&slot| logits[slot as usize]).collect();
    let total = logsumexp(restricted.iter().copied());
    let (index, _) = restricted
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        });
    let probability = (f64::from(restricted[index]) - total).exp();
    let mass = (total - logsumexp(logits.iter().copied())).exp();
    (index, probability, mass)
}

fn messages_with(system: &str, text: String) -> [ChatMessage; 2] {
    [
        ChatMessage {
            role: Role::System,
            content: MessageContent::Text(system.to_owned()),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
        ChatMessage {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Image { url: None },
                ContentPart::Text(text),
            ]),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
    ]
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_constrained_readout_points_at_the_blue_button() {
    let dir = fixture_dir();
    let Ok(manifest_text) = std::fs::read_to_string(dir.join("manifest.json")) else {
        if gpu_profile::skip_or_fail(&format!("the pointing fixture is absent: {}", dir.display())) {
            return;
        }
        unreachable!("skip_or_fail panics under the profile");
    };
    let manifest: Manifest =
        serde_json::from_str(&manifest_text).unwrap_or_else(|e| panic!("parse the manifest: {e}"));
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let processor = frontend
        .vision_processor()
        .unwrap_or_else(|e| panic!("vision processor: {e}"));
    let eos = frontend.eos_token_id().expect("eos");
    let tokenizer = frontend.tokenizer();

    let digit_slots: Vec<u32> = (0..10)
        .map(|digit| {
            let text = digit.to_string();
            let ids = tokenizer.encode(&text).unwrap_or_else(|e| panic!("encode {text}: {e}"));
            assert_eq!(ids.len(), 1, "digit {text} must be one token");
            ids[0]
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

    for scene in &manifest.scenes {
        let bytes = std::fs::read(dir.join(&scene.image))
            .unwrap_or_else(|e| panic!("{}: read image: {e}", scene.id));

        // ── 1. the free probe: which scale does it answer in? ────────────
        let probe = messages_with(
            "You are a helpful assistant.",
            PROBE_QUESTION.to_owned(),
        );
        let prepared = frontend
            .prepare_prompt(
                &processor,
                &probe,
                &[&bytes],
                ChatRenderOptions { enable_thinking: false, ..Default::default() },
                None,
            )
            .unwrap_or_else(|e| panic!("{}: prepare probe: {e}", scene.id));
        let (token_ids, prompt) = Multimodal::from_prepared(prepared);
        let tokens: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        assert_eq!(prompt.media.len(), 1, "{}: one image", scene.id);
        let item = &prompt.media[0];
        let control = vision_item_control(item.grid);
        let embedding = step::encode_media(&model, item.grid, &item.patches, &control)
            .unwrap_or_else(|(_, e)| panic!("{}: encode: {e}", scene.id));
        eprintln!(
            "ignis pointing {}: {}x{} image -> {} prompt tokens ({} image columns); blue box {:?} centre {:?} size {:?}",
            scene.id,
            manifest.side,
            manifest.side,
            tokens.len(),
            item.token_span.count,
            scene.blue_box,
            scene.blue_centre,
            scene.blue_size,
        );
        assert!(
            tokens.len() + DIGITS * 2 + 16 < MAX_CONTEXT as usize,
            "{}: {} prompt tokens do not leave room in a {MAX_CONTEXT}-token context",
            scene.id,
            tokens.len()
        );

        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
        let started = std::time::Instant::now();
        logits.fill(0.0);
        prefill_prompt(&model, &pool, &mut sequence, &tokens, &prompt, &embedding, PREFILL_CHUNK, &mut logits)
            .unwrap_or_else(|e| panic!("{}: probe prefill: {e}", scene.id));
        let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut emitted: Vec<u32> = Vec::new();
        let mut position = tokens.len() as u64;
        while emitted.len() < PROBE_TOKENS {
            let next = argmax_lowest_id(&logits);
            if next == eos {
                break;
            }
            emitted.push(next);
            step::prefill_program(&model, &pool, &mut sequence, &[next as i32], position, Some(&mut logits))
                .unwrap_or_else(|e| panic!("{}: probe step: {e}", scene.id));
            position += 1;
        }
        drop(sequence);
        let probe_text = tokenizer.decode(&emitted).unwrap_or_else(|_| "<undecodable>".to_owned());
        eprintln!(
            "ignis pointing {}: prefill {prefill_ms:.0} ms; free probe ({} tokens) -> {probe_text:?}",
            scene.id,
            emitted.len()
        );

        // ── 2. the constrained readout ───────────────────────────────────
        let constrained = messages_with(POINT_SYSTEM, format!("{{\"instruction\":{INSTRUCTION:?}}}"));
        let prepared = frontend
            .prepare_prompt(
                &processor,
                &constrained,
                &[&bytes],
                ChatRenderOptions { enable_thinking: false, ..Default::default() },
                None,
            )
            .unwrap_or_else(|e| panic!("{}: prepare constrained: {e}", scene.id));
        let (token_ids, prompt) = Multimodal::from_prepared(prepared);
        let tokens: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();

        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
        logits.fill(0.0);
        prefill_prompt(&model, &pool, &mut sequence, &tokens, &prompt, &embedding, PREFILL_CHUNK, &mut logits)
            .unwrap_or_else(|e| panic!("{}: constrained prefill: {e}", scene.id));
        let mut position = tokens.len() as u64;

        // Force a literal, read DIGITS constrained digits, and report each
        // one's restricted probability and the mass the digits hold.
        let mut read_axis = |axis: &str,
                             literal: &str,
                             position: &mut u64,
                             sequence: &mut _,
                             logits: &mut Vec<f32>|
         -> (i64, Vec<(u32, f64, f64)>) {
            let forced: Vec<i32> = tokenizer
                .encode(literal)
                .unwrap_or_else(|e| panic!("encode {literal:?}: {e}"))
                .into_iter()
                .map(|id| id as i32)
                .collect();
            step::prefill_program(&model, &pool, sequence, &forced, *position, Some(logits))
                .unwrap_or_else(|e| panic!("{}: force {literal:?}: {e}", scene.id));
            *position += forced.len() as u64;
            let mut value: i64 = 0;
            let mut trace = Vec::with_capacity(DIGITS);
            for step_index in 0..DIGITS {
                let (digit, probability, mass) = constrained_pick(logits, &digit_slots);
                trace.push((digit as u32, probability, mass));
                value = value * 10 + digit as i64;
                let token = [digit_slots[digit] as i32];
                step::prefill_program(&model, &pool, sequence, &token, *position, Some(logits))
                    .unwrap_or_else(|e| panic!("{}: {axis} digit {step_index}: {e}", scene.id));
                *position += 1;
            }
            (value, trace)
        };

        let (x, x_trace) = read_axis("x", "{\"x\":", &mut position, &mut sequence, &mut logits);
        let (y, y_trace) = read_axis("y", ",\"y\":", &mut position, &mut sequence, &mut logits);
        drop(sequence);
        drop(embedding);

        let side = f64::from(manifest.side);
        let px = (x as f64 / SCALE * side).round() as i64;
        let py = (y as f64 / SCALE * side).round() as i64;
        let [cx, cy] = scene.blue_centre;
        let error = (((px - cx) as f64).powi(2) + ((py - cy) as f64).powi(2)).sqrt();
        let [bx0, by0, bx1, by1] = scene.blue_box;
        let inside = px >= bx0 && px <= bx1 && py >= by0 && py <= by1;
        let fmt = |trace: &[(u32, f64, f64)]| {
            trace
                .iter()
                .map(|(d, p, m)| format!("{d}(p={p:.3},mass={m:.3})"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        eprintln!(
            "ignis pointing {}: constrained x={x:03} y={y:03} -> ({px},{py}) px, target ({cx},{cy}), error {error:.0} px ({:.1}% of side), inside_button={inside}",
            scene.id,
            100.0 * error / side
        );
        eprintln!("ignis pointing {}:   x digits {}", scene.id, fmt(&x_trace));
        eprintln!("ignis pointing {}:   y digits {}", scene.id, fmt(&y_trace));

        // Does the constraint *read* the model or overrule it? The free probe
        // answered in the same declared units; its box centre is what the
        // model says when nothing restricts it.
        match first_four_integers(&probe_text) {
            Some([px0, py0, px1, py1]) => {
                let (fx, fy) = ((px0 + px1) / 2, (py0 + py1) / 2);
                eprintln!(
                    "ignis pointing {}:   free-probe box [{px0},{py0},{px1},{py1}] centre ({fx},{fy}) vs constrained ({x},{y}) -- delta ({}, {}) on a 0-{SCALE:.0} scale",
                    scene.id,
                    (x - fx).abs(),
                    (y - fy).abs()
                );
            }
            None => eprintln!(
                "ignis pointing {}:   free probe carried no parseable box, so there is nothing to compare",
                scene.id
            ),
        }
    }
}
