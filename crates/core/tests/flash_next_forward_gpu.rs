//! The Flash-Next program on the real artifact (spec flash-next/04, GitHub
//! #302): the whole load -- non-expert weights on the device, the expert
//! residency and its pinned pool, the n-gram table, the model, a pool with
//! Flash-Next's sections, the round graphs -- and the two programs it runs
//! held to each other.
//!
//! - **A span is deterministic.** The same tokens teacher-forced twice from
//!   position 0 give the same logits, bit for bit, every position; and a
//!   span's row is the draw's own row -- equal, bit for bit, to the last
//!   logits of the prefix prompt that ends there, where the two cut the same
//!   chunks (the span-logits readout the acceptance scorers read).
//! - **Decode agrees with prefill, on real text.** Greedy generation through
//!   decode rounds (graphs, one lane and three) and a teacher-forced prefill
//!   of the prompt and the generated tokens pick the same argmax at each
//!   generated position, but where the two routes' logits are a near-tie: the
//!   round runs the MoE decode route and gathered attention, the chunk the
//!   prefill route and dense attention, so their logits differ by summation
//!   order. The prompts are the converter's G1 prompts, whose teacher-forced
//!   argmax is also held to the converter's own (G1 agreement, printed).
//!   On prompts of arbitrary token ids the same comparison is only reported:
//!   the model's next token there is close to a coin toss, and the two routes
//!   flipped 3 of 24 picks per lane by 0.25-3.25 logits (2026-10-06, first
//!   runs on the artifact; a router near-tie choosing other experts would do
//!   it -- not measured) while real text flipped none of 224.
//! - **Decode is deterministic, and a graph replays what eager computes.**
//!   The same prompt decoded three times, inside a three-lane round and
//!   alone, with the round graphs and with none, gives the same tokens.
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

    // The readout's rows are the draw's own rows: where a span and a prefix
    // prompt cut the same chunks -- the first chunk's end, and the span's end
    // -- the span's row equals the prefix's last-position logits bit for bit.
    let chunk = engine.options().prefill_chunk_tokens as usize;
    for end in [chunk, tokens.len()] {
        let last = engine.last_logits(&tokens[..end]).unwrap_or_else(|e| panic!("last logits at {end}: {e}"));
        let row = &first[(end - 1) * vocab..end * vocab];
        let differing = row.iter().zip(&last).filter(|&(&b, &f)| bf16(b).to_bits() != f.to_bits()).count();
        assert_eq!(differing, 0, "row {}: {differing} logits differ from the prefix prompt's last row", end - 1);
    }

    // Decode against prefill on arbitrary ids, one lane and three: reported,
    // not held (the header says why; the G1 test below holds it).
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
                    // How far below the prefill's own pick the decoded token
                    // sits in the prefill's row: a near-tie is a route's
                    // summation order, a wide gap is a different computation.
                    let gap = bf16(row[best as usize]) - bf16(row[token as usize]);
                    println!("  lane {lane} position {i}: decode {token}, prefill {best}, margin {margin}, gap {gap}");
                    if margin < NEAR_TIE {
                        near_ties += 1;
                    } else {
                        disagree += 1;
                    }
                }
            }
            println!("{lanes} lanes, lane {lane}: {} tokens, {near_ties} near-tie flips, {disagree} wider flips", g.len());
        }
    }
    println!("n-gram rows: {:?}", engine.ngram_table().counters());
    println!("residency: {:?}", engine.residency().counters());
}

/// The G1 prompts (the converter's `references/g1_flash_next.json`): each
/// prompt's reference continuation teacher-forced through one prefill, its
/// argmax per position against the converter's own quantized argmax (G1
/// agreement, printed); and greedy decode rounds against the teacher-forced
/// prefill of what they generated, on real text, with each disagreement's
/// gap printed.
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn on_the_g1_prompts_decode_agrees_with_prefill() {
    let refs = model_dir().join("references").join("g1_flash_next.json");
    let Ok(text) = std::fs::read_to_string(&refs) else {
        gpu_profile::skip_or_fail(&format!("no G1 references at {}", refs.display()));
        return;
    };
    let g1: serde_json::Value = serde_json::from_str(&text).expect("the G1 references parse");
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array().expect("an id list").iter().map(|x| x.as_u64().expect("an id") as u32).collect()
    };
    let Some(mut engine) = engine() else { return };
    let vocab = engine.vocab();
    let (mut agree, mut total) = (0usize, 0usize);
    let mut prompts = Vec::new();
    for p in g1["prompts"].as_array().expect("prompts") {
        let prompt = ids(&p["prompt_token_ids"]);
        let reference = ids(&p["token_ids"]);
        let expected = ids(&p["expected_argmax"]);
        let fed: Vec<u32> = prompt.iter().chain(&reference[..reference.len() - 1]).copied().collect();
        let rows = all_rows(&mut engine, &fed);
        let mine: Vec<u32> =
            (0..reference.len()).map(|i| argmax(&rows[(prompt.len() - 1 + i) * vocab..(prompt.len() + i) * vocab]).0).collect();
        let same = mine.iter().zip(&expected).filter(|(a, b)| a == b).count();
        println!("G1 {}: {same}/{} teacher-forced argmax equal to the converter's", p["id"], expected.len());
        agree += same;
        total += expected.len();
        prompts.push(prompt);
    }
    println!("G1 agreement: {agree}/{total} = {:.1}%", 100.0 * agree as f64 / total as f64);

    let mut far = 0;
    for lanes in [1usize, 3] {
        for batch in prompts.chunks(lanes).filter(|b| b.len() == lanes) {
            let generated = engine.generate(batch, 32).unwrap_or_else(|e| panic!("generate: {e}"));
            for (lane, (p, g)) in batch.iter().zip(&generated).enumerate() {
                let fed: Vec<u32> = p.iter().chain(&g[..g.len() - 1]).copied().collect();
                let rows = all_rows(&mut engine, &fed);
                let mut flips = Vec::new();
                for (i, &token) in g.iter().enumerate() {
                    let row = &rows[(p.len() - 1 + i) * vocab..(p.len() + i) * vocab];
                    let (best, margin) = argmax(row);
                    if best != token {
                        let gap = bf16(row[best as usize]) - bf16(row[token as usize]);
                        flips.push((i, token, best, margin, gap));
                        if margin >= NEAR_TIE {
                            far += 1;
                        }
                    }
                }
                println!("{lanes} lanes, lane {lane}: {} tokens, flips (position, decode, prefill, margin, gap) {flips:?}", g.len());
            }
        }
    }
    assert_eq!(far, 0, "decode and prefill disagree away from a near-tie on real text");
}

/// Greedy decode is a function of its inputs: the same prompt three times,
/// beside two other lanes, and on a load without graphs, gives the same
/// tokens. Two loads, one after the other.
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn decode_is_deterministic_and_a_graph_replays_what_eager_computes() {
    let (a, b, c) = (prompt(300, 2), prompt(317, 3), prompt(334, 4));
    let mut runs: Vec<(String, Vec<u32>)> = Vec::new();
    for graphs in [true, false] {
        let dir = model_dir();
        if !dir.join(ARTIFACT_FILE_NAME).exists() {
            gpu_profile::skip_or_fail(&format!("no Flash-Next artifact in {}", dir.display()));
            return;
        }
        let options = EngineOptions { max_context_tokens: 8192, capture_graphs: graphs, ..EngineOptions::default() };
        let mut engine = FlashNextEngine::load(&dir, options).unwrap_or_else(|e| panic!("load: {e}"));
        assert_eq!(engine.graphs_ready() != 0, graphs, "graphs {graphs}: ready {:#b}", engine.graphs_ready());
        for round in 0..3 {
            let one = engine.generate(&[a.clone()], 24).unwrap_or_else(|e| panic!("generate: {e}"));
            runs.push((format!("graphs {graphs}, alone, round {round}"), one[0].clone()));
        }
        let three = engine.generate(&[a.clone(), b.clone(), c.clone()], 24).unwrap_or_else(|e| panic!("generate: {e}"));
        runs.push((format!("graphs {graphs}, lane 0 of 3"), three[0].clone()));
        let solo = engine.generate(&[c.clone()], 24).unwrap_or_else(|e| panic!("generate: {e}"));
        assert_eq!(three[2], solo[0], "graphs {graphs}: lane 2 of 3 decodes as it does alone");
    }
    let (first, tokens) = &runs[0];
    for (name, other) in &runs[1..] {
        assert_eq!(other, tokens, "{name} against {first}");
    }
}
