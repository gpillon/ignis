//! The verify round on Flash-Next's state (spec flash-next/07 phase C,
//! GitHub #307), proven with fake drafters on the real artifact before the
//! MTP head drafts.
//!
//! The load is `SpeculativeBackend::VerifyOnly` at 3 draft tokens and the
//! decode route's 8 rows: one lane verifies 3 drafts a round, three lanes 1.
//! Four drafters run the same prompts for the same number of tokens:
//! - **none** (every lane at extent 0): one committed token a round, through
//!   the verify pass's k + 1 columns;
//! - **reject** (a constant id no greedy step picks): every draft rejected;
//! - **random**: rejected at the first divergence, by chance accepted;
//! - **oracle** (none's own text): every draft accepted.
//!
//! A column's computation does not depend on what the other columns of its
//! round hold -- each op is row-independent at a fixed round shape -- so a
//! rejected column is a column never drafted: **reject and random commit
//! none's text bit for bit, and leave every lane's state as none leaves it**.
//! The state is probed where a later step reads it: one more token prefilled
//! at each lane's frontier, its logits compared bit for bit (KV and pooled
//! indexer blocks, the indexer tail, the GDN states and conv taps, the n-gram
//! conv, the hq residual window). Any component the commit restored wrongly
//! moves that row. The oracle commits several columns a round: in BF16 KV
//! the window's earlier columns are read exactly as the pages would hold
//! them, so it too commits none's text and state bit for bit; under
//! hq-e8-2b a column reads the window's own rows fresh where a later round
//! reads them back from the residual ring, so there it is held to the
//! near-tie rule instead.
//!
//! Spec-on against spec-off (one-token rounds, the plain decode graphs) is
//! held to the near-tie rule: the verify pass runs k + 1 columns per lane
//! through other kernel shapes (attention splits, the indexer's prefill
//! scoring), so its logits differ by summation order.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure. Needs ~38 GB of free RAM for the pinned expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine, LaneRound};
use ignis_core::gpu_profile;
use ignis_core::kv_format::KvFormat;
use ignis_core::speculation::{FlashNextSpeculation, SpeculativeBackend};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const DRAFT_TOKENS: u32 = 3;
const TOKENS: usize = 40;
/// A draft no greedy step picks: the vocabulary's last id, a reserved token.
const REJECTED: u32 = 248_319;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

fn engine(kv_format: KvFormat) -> Option<FlashNextEngine> {
    engine_with(kv_format, SpeculativeBackend::VerifyOnly)
}

fn engine_with(kv_format: KvFormat, backend: SpeculativeBackend) -> Option<FlashNextEngine> {
    let dir = model_dir();
    if !dir.join(ARTIFACT_FILE_NAME).exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact in {}", dir.display()));
        return None;
    }
    if backend == SpeculativeBackend::Mtp && !ignis_core::flash_next_mtp::companion_path(&dir).exists() {
        gpu_profile::skip_or_fail(&format!("no MTP companion in {}", dir.display()));
        return None;
    }
    let speculation = FlashNextSpeculation::new(backend, DRAFT_TOKENS, 0).expect("valid");
    let options =
        EngineOptions { max_context_tokens: 8192, kv_format, speculation: Some(speculation), ..EngineOptions::default() };
    match FlashNextEngine::load(&dir, options) {
        Ok(engine) => Some(engine),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine: {e}"));
            None
        }
    }
}

/// The converter's G1 prompts (real text), or None with a skip.
fn g1_prompts() -> Option<Vec<Vec<u32>>> {
    let refs = model_dir().join("references").join("g1_flash_next.json");
    let Ok(text) = std::fs::read_to_string(&refs) else {
        gpu_profile::skip_or_fail(&format!("no G1 references at {}", refs.display()));
        return None;
    };
    let g1: serde_json::Value = serde_json::from_str(&text).expect("the G1 references parse");
    Some(
        g1["prompts"]
            .as_array()
            .expect("prompts")
            .iter()
            .map(|p| {
                p["prompt_token_ids"].as_array().expect("ids").iter().map(|x| x.as_u64().expect("an id") as u32).collect()
            })
            .collect(),
    )
}

/// A prompt past the dense threshold (2051 visible tokens), so every column
/// attends through the indexer's sparse selection: the G1 prompts and their
/// reference continuations, repeated.
fn long_prompt(prompts: &[Vec<u32>]) -> Vec<u32> {
    let mut out = Vec::new();
    while out.len() < 2600 {
        for p in prompts {
            out.extend_from_slice(p);
        }
    }
    out.truncate(2600);
    out
}

/// What one drafter's run left: each lane's tokens, its rounds, and the
/// probe row after them.
struct Outcome {
    tokens: Vec<Vec<u32>>,
    rounds: Vec<Vec<LaneRound>>,
    probes: Vec<Vec<f32>>,
}

fn run(engine: &FlashNextEngine, prompts: &[Vec<u32>], drafter: Option<&mut dyn FnMut(usize, &[u32], u32) -> Vec<u32>>) -> Outcome {
    let mut run = engine
        .generate_speculative(prompts, TOKENS, drafter)
        .unwrap_or_else(|e| panic!("generate_speculative on {} lanes: {e}", prompts.len()));
    let mut probes = Vec::new();
    for (seq, context) in run.sequences.iter_mut().zip(run.contexts.iter_mut()) {
        probes.push(engine.probe_logits(seq, context).unwrap_or_else(|e| panic!("probe: {e}")));
    }
    Outcome { tokens: run.tokens, rounds: run.rounds, probes }
}

fn same_bits(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn top_two(row: &[f32]) -> (u32, f32, f32) {
    let (mut best, mut first, mut second) = (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (id, &v) in row.iter().enumerate() {
        if v > first {
            second = first;
            first = v;
            best = id as u32;
        } else if v > second {
            second = v;
        }
    }
    (best, first, second)
}

/// One BF16 ulp at `x`'s magnitude (`flash_next_forward_gpu.rs`'s rule).
fn bf16_ulp(x: f32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent - 7)
}

/// `on` against `off`, lane by lane: identical, or identical up to a first
/// divergence where the prefill route's logits after the agreed text put the
/// two picks within one BF16 ulp, or side with `on` (the reference's own
/// decode drifted). Returns a line per divergence; panics on a divergence
/// the prefill route lays at `on`'s door.
fn assert_near_tie(engine: &mut FlashNextEngine, prompts: &[Vec<u32>], off: &[Vec<u32>], on: &[Vec<u32>], what: &str) -> Vec<String> {
    let mut report = Vec::new();
    for (lane, ((prompt, a), b)) in prompts.iter().zip(off).zip(on).enumerate() {
        let Some(at) = a.iter().zip(b).position(|(x, y)| x != y) else {
            assert_eq!(a.len(), b.len(), "{what} lane {lane}: lengths");
            continue;
        };
        let mut text = prompt.clone();
        text.extend_from_slice(&a[..at]);
        let logits = engine.last_logits(&text).unwrap_or_else(|e| panic!("near-tie probe: {e}"));
        let (top, first, _) = top_two(&logits);
        let gap = logits[a[at] as usize] - logits[b[at] as usize];
        let line = format!(
            "{what} lane {lane}: diverges at {at}: off {} on {} (prefill pick {top}), prefill gap off - on = {gap} (ulp {})",
            a[at],
            b[at],
            bf16_ulp(first)
        );
        println!("{line}");
        assert!(gap <= bf16_ulp(first), "{line}: the prefill route sides with spec-off past a near-tie");
        report.push(line);
    }
    report
}

/// Every drafter on one set of prompts, in one KV format.
fn exercise(engine: &mut FlashNextEngine, kv_format: KvFormat, prompts: &[Vec<u32>], what: &str) {
    let lanes = prompts.len() as u32;
    let window = engine.options().speculation.expect("speculation").window(lanes);
    assert!(window > 0, "{what}: a width that verifies");
    let none = run(engine, prompts, Some(&mut |_, _, _| Vec::new()));
    for round in &none.rounds {
        assert!(round.iter().all(|l| l.extent == 0 && l.committed == 1), "{what}: none committed a draft");
    }
    let reject = run(engine, prompts, Some(&mut |_, _, w| vec![REJECTED; w as usize]));
    let mut seed = 0x9e37_79b9_u32;
    let random = run(
        engine,
        prompts,
        Some(&mut |_, _, w| {
            (0..w)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    1000 + seed % 60_000
                })
                .collect()
        }),
    );
    let reference = none.tokens.clone();
    let oracle = run(
        engine,
        prompts,
        Some(&mut |lane, emitted, w| {
            let next = emitted.len() + 1;
            reference[lane][next.min(TOKENS)..(next + w as usize).min(TOKENS)].to_vec()
        }),
    );

    for (name, outcome) in [("reject", &reject), ("random", &random)] {
        assert_eq!(outcome.tokens, none.tokens, "{what}: the {name} drafter committed other text than none");
        for (lane, (p, q)) in outcome.probes.iter().zip(&none.probes).enumerate() {
            assert_eq!(same_bits(p, q), 0, "{what}: lane {lane}'s state after the {name} drafter's rounds is not none's");
        }
    }
    let drafted: u32 = reject.rounds.iter().flatten().map(|l| l.extent).sum();
    assert!(drafted > 0, "{what}: the reject drafter's drafts were never verified");
    assert!(reject.rounds.iter().flatten().all(|l| l.committed == 1), "{what}: a rejected draft was committed");

    // The oracle's drafts are none's text: every one is accepted while the
    // oracle's own text is none's.
    let accepted: u32 = oracle.rounds.iter().flatten().map(|l| l.committed - 1).sum();
    let proposed: u32 = oracle.rounds.iter().flatten().map(|l| l.extent).sum();
    println!(
        "{what} ({kv_format:?}): oracle {} rounds for {} tokens per lane, {accepted}/{proposed} drafts accepted",
        oracle.rounds.len(),
        TOKENS
    );
    if kv_format == KvFormat::Bf16 {
        assert_eq!(accepted, proposed, "{what}: an oracle draft was rejected");
        assert_eq!(oracle.tokens, none.tokens, "{what}: the oracle committed other text than none");
        for (lane, (p, q)) in oracle.probes.iter().zip(&none.probes).enumerate() {
            assert_eq!(same_bits(p, q), 0, "{what}: lane {lane}'s state after the oracle's rounds is not none's");
        }
    } else {
        assert_near_tie(engine, prompts, &none.tokens, &oracle.tokens, &format!("{what} oracle vs none"));
    }

    // Spec-on (the verify round) against spec-off (one-token rounds).
    let off = engine.generate(prompts, TOKENS).unwrap_or_else(|e| panic!("{what}: spec-off generate: {e}"));
    assert_near_tie(engine, prompts, &off, &none.tokens, &format!("{what} spec-on vs spec-off"));
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn a_rejected_draft_is_a_draft_never_made_on_every_state_component() {
    let Some(g1) = g1_prompts() else { return };
    let long = long_prompt(&g1);
    for kv_format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let Some(mut engine) = engine(kv_format) else { return };
        assert_ne!(engine.graphs_ready(), 0, "{kv_format:?}: no round graph captured: {:?}", engine.graph_error());
        exercise(&mut engine, kv_format, &g1[..1], &format!("{kv_format:?} one lane, dense"));
        exercise(&mut engine, kv_format, &g1[..3], &format!("{kv_format:?} three lanes, dense"));
        exercise(&mut engine, kv_format, std::slice::from_ref(&long), &format!("{kv_format:?} one lane, sparse"));
        drop(engine);
    }
}

/// Each draft position's acceptance over `rounds`: of the rounds that
/// verified a draft at position j with every draft before it accepted, the
/// share that accepted it too (a run cut by its budget counts as far as it
/// committed).
fn acceptance(rounds: &[Vec<LaneRound>]) -> Vec<(u32, u32)> {
    let mut at = vec![(0u32, 0u32); DRAFT_TOKENS as usize];
    for lane in rounds.iter().flatten() {
        let accepted = lane.committed - 1;
        for j in 0..lane.extent.min(accepted + 1) {
            at[j as usize].1 += 1;
            if j < accepted {
                at[j as usize].0 += 1;
            }
        }
    }
    at
}

/// Spec flash-next/07 phase D: the MTP head drafts every round from its
/// own entries (prefill's, then each round's alignment and chain), and the
/// verify round keeps the text -- bit for bit what the same load commits
/// without drafts in BF16 KV (a draft-free run of the same load: the trunk
/// never sees the head), up to the near-tie rule under hq-e8-2b and against
/// spec-off. Prints each draft position's acceptance.
#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its MTP companion"]
fn the_mtp_head_drafts_and_the_text_is_kept() {
    let Some(g1) = g1_prompts() else { return };
    let long = long_prompt(&g1);
    for kv_format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let Some(mut engine) = engine_with(kv_format, SpeculativeBackend::Mtp) else { return };
        for (prompts, what) in [
            (g1[..1].to_vec(), format!("{kv_format:?} MTP one lane, dense")),
            (g1[..3].to_vec(), format!("{kv_format:?} MTP three lanes, dense")),
            (vec![long.clone()], format!("{kv_format:?} MTP one lane, sparse")),
        ] {
            let none = run(&engine, &prompts, Some(&mut |_, _, _| Vec::new()));
            let head = run(&engine, &prompts, None);
            let alpha = acceptance(&head.rounds);
            let tokens: u32 = head.rounds.iter().flatten().map(|l| l.committed).sum();
            println!(
                "{what}: {} rounds for {tokens} tokens, acceptance per position (accepted/reached) {alpha:?}",
                head.rounds.len()
            );
            assert!(alpha[0].0 > 0, "{what}: the head's first drafts were never accepted: {alpha:?}");
            if kv_format == KvFormat::Bf16 {
                assert_eq!(head.tokens, none.tokens, "{what}: the head's drafts changed the committed text");
                for (lane, (p, q)) in head.probes.iter().zip(&none.probes).enumerate() {
                    assert_eq!(same_bits(p, q), 0, "{what}: lane {lane}'s state after the head's rounds is not none's");
                }
            } else {
                assert_near_tie(&mut engine, &prompts, &none.tokens, &head.tokens, &format!("{what} head vs none"));
            }
            let off = engine.generate(&prompts, TOKENS).unwrap_or_else(|e| panic!("{what}: spec-off generate: {e}"));
            assert_near_tie(&mut engine, &prompts, &off, &head.tokens, &format!("{what} spec-on vs spec-off"));
        }
        drop(engine);
    }
}
