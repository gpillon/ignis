//! The Flash-Next program on the real artifact (spec flash-next/04, GitHub
//! #302): the whole load -- non-expert weights on the device, the expert
//! residency and its pinned pool, the n-gram table, the model, a pool with
//! Flash-Next's sections, the round graphs -- and the two programs it runs
//! held to each other.
//!
//! - **A span is deterministic.** The same tokens teacher-forced twice from
//!   position 0 give the same logits, bit for bit, every position.
//! - **Decode agrees with prefill.** Greedy generation through decode rounds
//!   (graphs, one lane and three) and a teacher-forced prefill of the prompt
//!   and the generated tokens pick the same argmax at each generated
//!   position, but where the two routes' logits are a near-tie: the round
//!   runs the MoE decode route and gathered attention, the chunk the prefill
//!   route and dense attention, so their logits differ by summation order.
//!
//! The G1 canaries and the KLD scorers on this engine are
//! `ignis_bench::flash_next` (pack's seam, `FlashNextEngine::span_logits`).
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure. Needs ~38 GB of free RAM for the pinned expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine};
use ignis_core::gpu_profile;

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
/// A near-tie: below this many logits apart, two routes may pick
/// differently.
const NEAR_TIE: f32 = 0.125;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

fn engine() -> Option<FlashNextEngine> {
    let dir = model_dir();
    if !dir.join(ARTIFACT_FILE_NAME).exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact in {}", dir.display()));
        return None;
    }
    let options = EngineOptions { max_context_tokens: 8192, ..EngineOptions::default() };
    match FlashNextEngine::load(&dir, options) {
        Ok(engine) => Some(engine),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine: {e}"));
            None
        }
    }
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Argmax of one BF16 row, and its margin over the runner-up.
fn argmax(row: &[u16]) -> (u32, f32) {
    let (mut best, mut first, mut second) = (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (id, &bits) in row.iter().enumerate() {
        let v = bf16(bits);
        if v > first {
            second = first;
            first = v;
            best = id as u32;
        } else if v > second {
            second = v;
        }
    }
    (best, first - second)
}

/// A deterministic prompt of ordinary token ids (the low vocab is text).
fn prompt(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32).map(|i| 1000 + (i * 7919 + salt * 104_729) % 60_000).collect()
}

fn all_rows(engine: &mut FlashNextEngine, tokens: &[u32]) -> Vec<u16> {
    let vocab = engine.vocab();
    let mut out = vec![0u16; tokens.len() * vocab];
    engine
        .span_logits(tokens, &mut |first, rows| {
            out[first * vocab..first * vocab + rows.len()].copy_from_slice(rows);
            Ok(())
        })
        .unwrap_or_else(|e| panic!("span logits: {e}"));
    out
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn a_span_is_deterministic_and_decode_agrees_with_prefill() {
    let Some(mut engine) = engine() else { return };
    let vocab = engine.vocab();
    println!("graphs ready: {:#b}", engine.graphs_ready());
    if let Some(error) = engine.graph_error() {
        println!("graph capture: {error}");
    }
    assert_ne!(engine.graphs_ready() & 1, 0, "the one-lane round captured a graph: {:?}", engine.graph_error());
    assert_eq!(engine.reserved(), engine.planned(), "the load holds what its plan laid out");

    // A span past one chunk and past the dense threshold (2051 visible tokens).
    let tokens = prompt(2600, 1);
    let first = all_rows(&mut engine, &tokens);
    assert!(first.iter().all(|&b| bf16(b).is_finite()), "every logit is finite");
    let again = all_rows(&mut engine, &tokens);
    let differing = first.chunks_exact(vocab).zip(again.chunks_exact(vocab)).filter(|(a, b)| a != b).count();
    assert_eq!(differing, 0, "{differing} of {} rows differ between two identical spans", tokens.len());

    // Decode against prefill, on one lane and on three.
    for lanes in [1usize, 3] {
        let prompts: Vec<Vec<u32>> = (0..lanes).map(|l| prompt(300 + 17 * l, 2 + l as u32)).collect();
        let generated = engine.generate(&prompts, 24).unwrap_or_else(|e| panic!("generate on {lanes} lanes: {e}"));
        for (lane, (p, g)) in prompts.iter().zip(&generated).enumerate() {
            let fed: Vec<u32> = p.iter().chain(&g[..g.len() - 1]).copied().collect();
            let rows = all_rows(&mut engine, &fed);
            let (mut disagree, mut near_ties) = (0, 0);
            for (i, &token) in g.iter().enumerate() {
                let row = &rows[(p.len() - 1 + i) * vocab..(p.len() + i) * vocab];
                let (best, margin) = argmax(row);
                if best != token {
                    if margin < NEAR_TIE {
                        near_ties += 1;
                    } else {
                        disagree += 1;
                    }
                }
            }
            println!("{lanes} lanes, lane {lane}: {} tokens, {near_ties} near-tie flips, {disagree} disagreements", g.len());
            assert_eq!(disagree, 0, "lane {lane} of {lanes}: decode and prefill disagree away from a near-tie");
        }
    }
    println!("n-gram rows: {:?}", engine.ngram_table().counters());
    println!("residency: {:?}", engine.residency().counters());
}
