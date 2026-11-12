//! GPU coverage for the 27B's decode lanes as a load setting (GitHub #336,
//! spec 27b-16gb/04 solution 1, ADR 0063): a leaf loaded at 1, 2 and 8 lanes
//! with DFlash2 on and vision on, as the `rtx5090` profile runs it, holds to
//! the byte what its plan laid out and decodes greedily; a round of a given
//! width commits the same tokens whatever the load's lanes, so a smaller
//! load changes what fits, never what is said.
//!
//! The prompts are the canary suite's (`crates/bench/src/canary.rs`
//! `CANARIES`), in a user turn with thinking closed, as
//! `cuda_leaf_dflash2_gpu.rs` asks: three of them read code.
//!
//! The 8-lane load's greedy token ids, at every width, must equal main's
//! (the spec's "at 8 lanes, greedy outputs identical to today's"), on every
//! run: `fixtures/decode_lanes_greedy_8lanes.json`, recorded by this test
//! at 8 lanes on 2026-10-23 by a binary built at 00:28, before the kernel
//! rebuild (00:51) and the first commit that changed the engine (01:10) on
//! a branch off main 75210360. At 8 lanes this branch's plan is main's to the
//! byte on both shapes `vram_plan_gpu` asks, and its load options cross
//! nothing main's did not. `IGNIS_RECORD_DECODE_LANES_TOKENS=1` writes it
//! again, and is only ever run on a build of main's engine.
//!
//! Its own test binary: it materializes the artifact, and the card fits one
//! at a time.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignis_artifact::{bind_model_scope_27b_with, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::gpu_profile;
use ignis_core::{
    model_load, Compute, DecodeJob, DecodeParams, ExceptionWeights, KvFormat, PrefillJob, RequestId, Speculation,
    SpeculativeBackend, TokenId, Vision,
};
use ignis_runtime::{auto_kv_pool_bytes, CudaLeaf, CudaLeafConfig, Model, ReservedBytes, RuntimeCompute};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 4096;
const PREFILL_CHUNK: u32 = 2048;
/// What each request may hold: its prompt and its generation, whole pages.
const CONTEXT_TOKENS: u32 = 1024;
const MAX_GENERATED: u32 = 64;

/// The canary suite's prompts (`crates/bench/src/canary.rs`).
const PROMPTS: [&str; 4] = [
    "In one sentence, what does `fn main() { println!(\"hi\"); }` do?",
    "What does `let v = vec![3,1,2]; v.sort();` set `v` to, after the call?",
    "Compute, step by step, 2 * 3 + 4 and give the final number on the last line.",
    "Explain in one sentence what `x.reverse()` does to a `Vec<i32>` named `x`.",
];

/// The loads, and the round widths each one decodes at: every load the
/// one-lane width, each its own widest, and the 8-lane load the 2-lane one's
/// too, so each width is compared across the loads that serve it.
const LOADS: [(u32, &[usize]); 3] = [(1, &[1]), (2, &[1, 2]), (8, &[1, 2, 8])];

const RECORD_ENV: &str = "IGNIS_RECORD_DECODE_LANES_TOKENS";

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("decode_lanes_greedy_8lanes.json")
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_load_decodes_at_its_lanes_and_a_width_says_the_same_at_every_load() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    // Token ids by width, one run per prompt in order, per load.
    let mut runs: BTreeMap<u32, BTreeMap<usize, Vec<Vec<TokenId>>>> = BTreeMap::new();
    for (lanes, widths) in LOADS {
        let Some(by_width) = load_and_decode(path, lanes, widths) else {
            return;
        };
        runs.insert(lanes, by_width);
    }

    // A width commits the same tokens at every load that serves it.
    let widest = &runs[&8];
    for (lanes, by_width) in &runs {
        for (width, outputs) in by_width {
            assert_eq!(outputs, &widest[width], "width {width} at {lanes} lanes against 8 lanes");
        }
    }

    // And at eight lanes, every width says what main's engine said.
    let path = fixture_path();
    if std::env::var_os(RECORD_ENV).is_some() {
        let json = serde_json::to_string_pretty(widest).expect("token ids serialize");
        std::fs::write(&path, json).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }
    let recorded = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let recorded: BTreeMap<usize, Vec<Vec<TokenId>>> = serde_json::from_str(&recorded).expect("fixture json");
    assert_eq!(widest, &recorded, "the 8-lane load's greedy token ids against main's");
}

/// Load the leaf at `lanes`, check it holds what its plan laid out, and
/// decode every prompt greedily at each of `widths`: `None` when the profile
/// is off and the card or the artifact is not there.
fn load_and_decode(path: &Path, lanes: u32, widths: &[usize]) -> Option<BTreeMap<usize, Vec<Vec<TokenId>>>> {
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    let prompts: Vec<Vec<TokenId>> = PROMPTS
        .iter()
        .map(|prompt| {
            frontend
                .tokenizer()
                .encode(&format!(
                    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                ))
                .unwrap_or_else(|e| panic!("tokenize prompt: {e}"))
        })
        .collect();

    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7");
    let vision = Vision::new(ignis_core::DEFAULT_VISION_MAX_TOKENS).expect("default envelope");
    let scope = model_load::model_scope(Some(speculation), Some(vision), ExceptionWeights::default());
    let (plan, handles) = bind_model_scope_27b_with(&reader, scope).unwrap_or_else(|e| panic!("bind: {e}"));
    let config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::HqE8_2b,
        kv_pool_bytes: auto_kv_pool_bytes(KvFormat::HqE8_2b, MAX_CONTEXT),
        slot_count: lanes,
        prefill_chunk_tokens: PREFILL_CHUNK,
        speculation: Some(speculation),
        vision: Some(vision),
        retained_device_slots: 0,
        retained_host_slots: 0,
        ..CudaLeafConfig::default()
    };
    let planned = config
        .plan_reservations(&reader, &plan, &handles)
        .unwrap_or_else(|e| panic!("plan the reservations at {lanes} lanes: {e}"));

    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return None;
            }
            unreachable!();
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return None;
            }
            unreachable!();
        }
    };
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config);
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load at {lanes} lanes: {e:?}")));
    let stats = model.stats().unwrap_or_else(|e| panic!("stats: {e:?}"));
    let kv_arena = config
        .kv_pool_arena_bytes(stats.kv_page_count)
        .unwrap_or_else(|e| panic!("plan the pool: {e}"));
    assert_eq!(
        stats.reserved,
        ReservedBytes { kv_pool: kv_arena, ..planned.reserved },
        "the {lanes}-lane load holds what its plan laid out"
    );
    let compute = RuntimeCompute::new(model, eos);

    let mut by_width = BTreeMap::new();
    let mut next_request: RequestId = 1;
    for &width in widths {
        // Every prompt once, `width` of them in each round (the last batch
        // of a wide width cycles the prompts).
        let mut outputs = vec![Vec::new(); PROMPTS.len()];
        let mut start = 0;
        while start < PROMPTS.len() {
            let batch: Vec<usize> = (0..width).map(|i| (start + i) % PROMPTS.len()).collect();
            let generated = decode_batch(&compute, &prompts, &batch, &mut next_request);
            for (slot, &prompt) in batch.iter().enumerate() {
                if outputs[prompt].is_empty() {
                    outputs[prompt] = generated[slot].clone();
                } else {
                    assert_eq!(outputs[prompt], generated[slot], "prompt {prompt} twice in one width-{width} round");
                }
            }
            start += width;
        }
        for (prompt, tokens) in outputs.iter().enumerate() {
            assert!(!tokens.is_empty(), "prompt {prompt} at width {width}, {lanes} lanes: no token");
        }
        by_width.insert(width, outputs);
    }
    Some(by_width)
}

/// Prefill `batch`'s prompts together and decode them in rounds of every
/// running one until each has `MAX_GENERATED` tokens or stops: each one's
/// tokens the rounds committed, in `batch`'s order.
fn decode_batch(
    compute: &RuntimeCompute<CudaLeaf>,
    prompts: &[Vec<TokenId>],
    batch: &[usize],
    next_request: &mut RequestId,
) -> Vec<Vec<TokenId>> {
    let params = DecodeParams { max_tokens: Some(MAX_GENERATED), ..DecodeParams::default() };
    let requests: Vec<RequestId> = batch
        .iter()
        .map(|_| {
            *next_request += 1;
            *next_request
        })
        .collect();
    let jobs: Vec<PrefillJob> = batch
        .iter()
        .zip(&requests)
        .map(|(&prompt, &request)| PrefillJob {
            checkpoint: None,
            capture_checkpoint: None,
            multimodal: None,
            readout: None,
            request,
            tokens: prompts[prompt].clone(),
            context_tokens: CONTEXT_TOKENS,
            start_position: 0,
            params,
            shared_prefix: None,
            publish_prefix: None,
            permitted: None,
            attention: None,
            reasoning_close: None,
            drafter_floor: 0,
            next_prompt_token: None,
        })
        .collect();
    compute.prefill_step(&jobs).unwrap_or_else(|e| panic!("prefill_step: {e}"));

    let mut tokens: Vec<Vec<TokenId>> = vec![Vec::new(); batch.len()];
    // The prefill drew the first token, which the first round commits.
    let mut generated = vec![1u32; batch.len()];
    let mut running: Vec<usize> = (0..batch.len()).collect();
    while !running.is_empty() {
        let jobs: Vec<DecodeJob> = running
            .iter()
            .map(|&i| DecodeJob {
                request: requests[i],
                lane: i,
                params,
                remaining_tokens: MAX_GENERATED - generated[i],
                permitted: None,
                reasoning_close: None,
                anchor_emitted: false,
                checked: false,
            })
            .collect();
        let outcomes = compute.decode_step(&jobs).unwrap_or_else(|e| panic!("decode_step: {e}"));
        assert_eq!(outcomes.len(), running.len(), "one outcome per job");
        let mut still = Vec::new();
        for (&i, outcome) in running.iter().zip(outcomes) {
            generated[i] += outcome.tokens.len() as u32;
            tokens[i].extend(&outcome.tokens);
            if outcome.finish.is_none() && generated[i] < MAX_GENERATED {
                still.push(i);
            } else {
                compute.release(requests[i]);
            }
        }
        running = still;
    }
    tokens
}
