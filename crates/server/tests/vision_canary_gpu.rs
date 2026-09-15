//! The vision canary (GitHub #178, ADR 0014's floor applied to multimodal
//! prompts): a handful of fixed images with short, unambiguous questions,
//! scored by **teacher-forced next-token agreement** against the reference's
//! greedy answers (`tests/fixtures/vision_canary`, recorded by
//! `tools/vision-canary/record.py` with `--vision`, thinking off), first 32
//! answer positions, floor 95%.
//!
//! Each prompt is scored twice: prefilled in one span, and cut into 48-token
//! spans so every image's placeholder run crosses span boundaries -- the
//! embedding is encoded once and its columns placed chunk by chunk. A wrong
//! merge order, patch layout, MRoPE axis or scatter offset is a gross error
//! and craters both.
//!
//! Two more checks ride the same load. Decode rotates at `position +
//! rope_delta` from its own staging (through a captured graph): its greedy run
//! must equal the prefill path's greedy chain up to the turn's end. And the
//! vision reservation holds one embedding at a time: an encode while one is
//! live is refused, and 100 encode/release cycles leave the footprint as it
//! was.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use ignis_artifact::{
    bind_model_scope_27b_with, materialize, ChatMessage, ContentPart, CudaDevice, FrontendSet,
    MessageContent, ModelScope, Reader, Role,
};
use ignis_bench::oracle::{
    meets_g1_floor, overall_teacher_forced_agreement, score_teacher_forced, TeacherForcedResult,
    G1_AGREEMENT_FLOOR,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_options, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    self, MediaEmbedding, MultimodalPrefill, SamplingParams, SpanMediaColumns,
};
use ignis_core::vision::{vision_item_control, Multimodal};
use ignis_core::{KvFormat, Vision};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 2048;
const PREFILL_CHUNK: u32 = 1024;
const FIRST_N: usize = 32;
/// Narrower than every canary image's placeholder run (64..196 tokens).
const SPANNING_CHUNK: u32 = 48;
/// Decode rounds compared against the prefill path's greedy chain.
const DECODE_ROUNDS: usize = 16;

struct Canary {
    id: String,
    image: Vec<u8>,
    question: String,
    expected: Vec<u32>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("vision_canary")
}

/// The answers are short and the reference stopped on its own, so the turn's
/// end is scored too: each expected run closes with the model's EOS.
fn load_canaries(frontend: &FrontendSet) -> Vec<Canary> {
    let eos = frontend.eos_token_id().expect("eos");
    let dir = fixture_dir();
    let text = std::fs::read_to_string(dir.join("fixture.json"))
        .unwrap_or_else(|e| panic!("read the vision canary fixture: {e}"));
    let fixture: serde_json::Value = serde_json::from_str(&text).expect("fixture json");
    fixture["canaries"]
        .as_array()
        .expect("canaries")
        .iter()
        .map(|canary| {
            let field = |name: &str| canary[name].as_str().unwrap_or_else(|| panic!("{name}")).to_owned();
            Canary {
                id: field("id"),
                image: std::fs::read(dir.join(field("image"))).expect("canary image"),
                question: field("question"),
                expected: {
                    let mut ids = frontend.tokenizer().encode(&field("text")).expect("tokenize the answer");
                    ids.push(eos);
                    ids
                },
            }
        })
        .collect()
}

fn argmax_lowest_id(logits: &[f32]) -> u32 {
    let mut best_id = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (id, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_id = id;
        }
    }
    best_id as u32
}

/// Prefill the whole prompt in spans of at most `chunk` tokens (one media
/// item per span, as the scheduler cuts them), leaving the last position's
/// logits in `logits`.
fn prefill_prompt(
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    tokens: &[i32],
    prompt: &Multimodal,
    embedding: &MediaEmbedding<'_>,
    chunk: u32,
    logits: &mut [f32],
) -> Result<(), String> {
    let total = tokens.len() as u32;
    let mut start = 0u32;
    while start < total {
        let len = prompt.cap_chunk(start, chunk.min(total - start));
        let positions = prompt.span_positions(start as usize, len as usize);
        let media = prompt.chunk_media(start, len);
        let last = start + len == total;
        step::prefill_program_multimodal(
            model,
            pool,
            sequence,
            &tokens[start as usize..(start + len) as usize],
            u64::from(start),
            SamplingParams::greedy(),
            MultimodalPrefill {
                positions: &positions,
                rope_delta: prompt.rope_delta,
                media: media.as_ref().map(|media| SpanMediaColumns {
                    embedding,
                    first_column: media.first_column,
                    scatter_indices: &media.scatter_indices,
                }),
            },
            if last { Some(&mut *logits) } else { None },
        )?;
        start += len;
    }
    Ok(())
}

fn new_pool() -> SeqPool {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: (MAX_CONTEXT / 64) * 2,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 2,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"))
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_vision_canary_meets_the_teacher_forced_floor_whole_and_across_chunks() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let processor = frontend.vision_processor().unwrap_or_else(|e| panic!("vision processor: {e}"));
    let eos = frontend.eos_token_id().expect("eos");
    let canaries = load_canaries(&frontend);
    assert!(!canaries.is_empty());

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
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
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
    )
    .unwrap_or_else(|e| panic!("ignis_model_load with vision: {e}"));
    let pool = new_pool();
    // Decode through the captured graphs, which read the rope positions from
    // their staging address.
    step::capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let vocab = ModelConfig::qwen38_27b().vocab as usize;

    let mut results: Vec<TeacherForcedResult> = Vec::new();
    let (mut decode_agree, mut decode_compared) = (0usize, 0usize);
    for canary in &canaries {
        let messages = [ChatMessage {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Image { url: None },
                ContentPart::Text(canary.question.clone()),
            ]),
            tool_calls: Vec::new(),
            reasoning_content: None,
        }];
        let prepared = frontend
            .prepare_prompt(&processor, &messages, &[&canary.image], false, None, None)
            .unwrap_or_else(|e| panic!("{}: prepare prompt: {e}", canary.id));
        let (token_ids, prompt) = Multimodal::from_prepared(prepared);
        let tokens: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        assert_eq!(prompt.media.len(), 1, "{}", canary.id);
        let item = &prompt.media[0];
        eprintln!(
            "vision canary {}: {} prompt tokens, image {}x{} patches ({} merged), rope_delta {}",
            canary.id,
            tokens.len(),
            item.grid.h,
            item.grid.w,
            item.token_span.count,
            prompt.rope_delta
        );
        let control = vision_item_control(item.grid);
        let embedding = step::encode_media(&model, item.grid, &item.patches, &control)
            .unwrap_or_else(|e| panic!("{}: encode: {e}", canary.id));
        assert_eq!(embedding.columns() as usize, item.token_span.count);
        let refused = step::encode_media(&model, item.grid, &item.patches, &control);
        assert!(
            matches!(&refused, Err(e) if e.contains("already live")),
            "one embedding at a time: {:?}",
            refused.as_ref().err()
        );

        let compared = FIRST_N.min(canary.expected.len());
        for (label, chunk) in [("whole", PREFILL_CHUNK), ("spanning", SPANNING_CHUNK)] {
            let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            let mut logits = vec![0f32; vocab];
            prefill_prompt(&model, &pool, &mut sequence, &tokens, &prompt, &embedding, chunk, &mut logits)
                .unwrap_or_else(|e| panic!("{} {label}: prefill: {e}", canary.id));
            let mut position = tokens.len() as u64;
            let mut predictions = Vec::with_capacity(compared);
            for i in 0..compared {
                predictions.push(argmax_lowest_id(&logits));
                let forced = [canary.expected[i] as i32];
                step::prefill_program(&model, &pool, &mut sequence, &forced, position, Some(&mut logits))
                    .unwrap_or_else(|e| panic!("{} {label}: forced position {i}: {e}", canary.id));
                position += 1;
            }
            let result =
                score_teacher_forced(&format!("{}/{label}", canary.id), &canary.expected, &predictions, FIRST_N);
            for m in &result.mismatches {
                eprintln!(
                    "vision canary {} position {}: ours={:?} reference={}",
                    result.id, m.position, m.predicted, m.expected
                );
            }
            eprintln!(
                "vision canary {}: teacher-forced agreement {}/{} = {:.1}%",
                result.id,
                result.agree,
                result.compared,
                result.agreement * 100.0
            );
            results.push(result);
        }

        // Decode at `position + rope_delta` against the prefill path's own
        // greedy chain, both cut at the turn's end.
        let mut chain_sequence = pool.alloc(MAX_CONTEXT).expect("seq alloc");
        let mut logits = vec![0f32; vocab];
        prefill_prompt(&model, &pool, &mut chain_sequence, &tokens, &prompt, &embedding, PREFILL_CHUNK, &mut logits)
            .expect("chain prefill");
        let mut chain = Vec::new();
        let mut position = tokens.len() as u64;
        while chain.len() < DECODE_ROUNDS {
            let next = argmax_lowest_id(&logits);
            chain.push(next);
            if next == eos {
                break;
            }
            step::prefill_program(&model, &pool, &mut chain_sequence, &[next as i32], position, Some(&mut logits))
                .expect("chain step");
            position += 1;
        }
        drop(chain_sequence);
        let mut decode_sequence = pool.alloc(MAX_CONTEXT).expect("seq alloc");
        prefill_prompt(&model, &pool, &mut decode_sequence, &tokens, &prompt, &embedding, PREFILL_CHUNK, &mut logits)
            .expect("decode prefill");
        let mut decoded = Vec::new();
        while decoded.len() < chain.len() {
            let ids = step::decode_program_batch(&model, &pool, &mut [&mut decode_sequence])
                .unwrap_or_else(|e| panic!("{}: decode: {e}", canary.id));
            decoded.push(ids[0] as u32);
        }
        let agree = decoded.iter().zip(&chain).take_while(|(a, b)| a == b).count();
        eprintln!(
            "vision canary {}: decode agrees with the prefill chain on {agree}/{} (decode {:?}, chain {:?})",
            canary.id,
            chain.len(),
            decoded,
            chain
        );
        decode_agree += agree;
        decode_compared += chain.len();
        drop(decode_sequence);
        drop(embedding);
    }

    // The reservation is reused, never grown: 100 encode/release cycles leave
    // the footprint as it was, and each encode finds it free.
    let before = step::program_stats(&model, &pool).expect("stats").vram_bytes;
    let canary = &canaries[0];
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(vec![ContentPart::Image { url: None }, ContentPart::Text(canary.question.clone())]),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let prepared = frontend
        .prepare_prompt(&processor, &messages, &[&canary.image], false, None, None)
        .expect("prepare");
    let item = &prepared.media[0];
    let control = vision_item_control(item.grid);
    for cycle in 0..100 {
        let embedding = step::encode_media(&model, item.grid, &item.patches, &control)
            .unwrap_or_else(|e| panic!("encode cycle {cycle}: {e}"));
        drop(embedding);
    }
    assert_eq!(step::program_stats(&model, &pool).expect("stats").vram_bytes, before);

    let overall = overall_teacher_forced_agreement(&results);
    let agree: usize = results.iter().map(|r| r.agree).sum();
    let compared: usize = results.iter().map(|r| r.compared).sum();
    eprintln!(
        "vision canary OVERALL teacher-forced agreement {agree}/{compared} = {:.1}% (floor {:.0}%); \
         decode vs prefill chain {decode_agree}/{decode_compared}",
        overall * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );
    drop(pool);
    drop(model);
    let _ = artifact.release_arena(&mut device);

    assert!(compared > 0, "the fixture must contribute scored positions");
    // ADR 0014: the floor is the suite's, not each canary's -- a flip between
    // near-equivalent forms of one answer (" red" / "Red") counts as a
    // mismatch and the suite still has to clear 95%.
    assert!(
        meets_g1_floor(overall),
        "vision canary floor: teacher-forced agreement {agree}/{compared} = {:.1}% < {:.0}% -- a \
         gross error in the encoder, the placeholder scatter or the MRoPE rotation",
        overall * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );
    assert_eq!(
        decode_agree, decode_compared,
        "decode at position + rope_delta must continue the prefill path's greedy chain"
    );
}
