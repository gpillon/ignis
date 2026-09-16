//! The vision canary fixture and the multimodal prefill both GPU canary
//! binaries drive it through (GitHub #178, GitHub #195) -- a `support`
//! submodule, reached as `support::vision_canary` like every other one.
//!
//! `tests/fixtures/vision_canary` is the reference's greedy answers to four
//! fixed images (`tools/vision-canary/record.py`, `--vision`, thinking off).
//! `vision_canary_gpu.rs` scores a vision load against it;
//! `vision_dflash2_gpu.rs` scores a load that also carries the DFlash2
//! drafter. One copy, so a change to the fixture's shape lands in one place.
//! (`support/mod.rs`'s own `#![allow(dead_code)]` covers this module too:
//! each binary uses a subset.)

use std::path::PathBuf;

use ignis_artifact::FrontendSet;
use ignis_core::model_load::Model;
use ignis_core::seq::{Seq, SeqPool};
use ignis_core::step::{self, MediaEmbedding, MultimodalPrefill, SamplingParams, SpanMediaColumns};
use ignis_core::vision::Multimodal;

/// One canary: the image bytes, its question, and the reference's answer.
pub struct Canary {
    pub id: String,
    pub image: Vec<u8>,
    pub question: String,
    pub expected: Vec<u32>,
}

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("vision_canary")
}

/// The answers are short and the reference stopped on its own, so the turn's
/// end is scored too: each expected run closes with the model's EOS.
pub fn load_canaries(frontend: &FrontendSet) -> Vec<Canary> {
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

/// The engine's own argmax rule: highest logit, lowest token id on a tie.
pub fn argmax_lowest_id(logits: &[f32]) -> u32 {
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
pub fn prefill_prompt(
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
