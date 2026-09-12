//! GPU diagnostic for GitHub #72: `rust-sort` and `explain-reverse` diverge
//! from the canary oracle at token 0 while `rust-hello` and `math-greedy`
//! match exactly. The working theory is an argmax near-tie flipped by a
//! residual numeric difference. This test drives the real 64-layer program
//! (`ignis_core::step::prefill_program`, GitHub #72's `out_logits` addition)
//! over the exact prompt the oracle fixture was recorded against --
//! `ArtifactTemplateProvider`, thinking disabled (GitHub #68) -- and reports
//! the top-k logits at token 0, so the oracle's chosen token's rank and its
//! logit gap to ignis's own winner are visible (`--nocapture`).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{decode_program_batch, prefill_program};
use serde::Deserialize;

use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 512;
const TOP_K: usize = 8;

/// The canary ids known to diverge from the oracle at token 0 (GitHub #72's
/// evidence section) -- see `crates/bench/src/canary.rs::CANARIES`.
const DIVERGENT_CANARY_IDS: &[&str] = &["rust-sort", "explain-reverse"];

/// One entry of `crates/bench/tests/fixtures/oracle_canary.json` -- mirrors
/// `ignis_bench::oracle::FixturePrompt` (not reused directly: pulling in
/// `ignis-bench` as a dependency for one struct shape is not worth it here).
#[derive(Deserialize)]
struct FixturePrompt {
    id: String,
    prompt: String,
    token_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Fixture {
    prompts: Vec<FixturePrompt>,
}

/// A canary known to diverge from the oracle at token 0, loaded straight
/// from the recorded fixture -- so the prompt text and the oracle's first
/// token id can never drift from what the G1 gate actually scores against
/// (GitHub #72: "same tokenized prompt ... at the point of first
/// divergence").
struct DivergentCanary {
    id: String,
    prompt: String,
    oracle_first_token: u32,
}

/// Read `crates/bench/tests/fixtures/oracle_canary.json` and pick out
/// [`DIVERGENT_CANARY_IDS`].
fn load_divergent_canaries() -> Vec<DivergentCanary> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("tests")
        .join("fixtures")
        .join("oracle_canary.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read the oracle fixture {}: {e}", path.display()));
    let fixture: Fixture =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse the oracle fixture: {e}"));
    DIVERGENT_CANARY_IDS
        .iter()
        .map(|&id| {
            let entry = fixture
                .prompts
                .iter()
                .find(|p| p.id == id)
                .unwrap_or_else(|| panic!("oracle fixture is missing canary {id}"));
            DivergentCanary {
                id: entry.id.clone(),
                prompt: entry.prompt.clone(),
                oracle_first_token: *entry
                    .token_ids
                    .first()
                    .unwrap_or_else(|| panic!("canary {id}: the oracle recorded no tokens")),
            }
        })
        .collect()
}

/// The top-`k` `(token_id, logit)` pairs, highest logit first.
fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);
    indexed
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn token_zero_top_k_logits_for_the_divergent_canaries() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let provider = ArtifactTemplateProvider::new(frontend);
    // GitHub #68: the oracle fixture was recorded with thinking disabled --
    // the candidate prompt must match, or this compares a template
    // difference instead of a numeric one.
    let thinking = ThinkingOptions {
        enable_thinking: false,
        ..ThinkingOptions::default()
    };

    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind text scope: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize the text scope on the device: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));

    for canary in load_divergent_canaries() {
        let prompt_tokens: Vec<i32> = provider
            .apply_chat_template(&[ChatMessage::text("user", canary.prompt.clone())], &thinking, &[])
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect();
        assert!(!prompt_tokens.is_empty(), "{}: the prompt must template", canary.id);

        let pool = SeqPool::create(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: ignis_core::KvFormat::Bf16,
                kv_page_group_count: 8,
                max_context_tokens: MAX_CONTEXT,
                slot_count: 1,
            },
        )
        .unwrap_or_else(|e| panic!("{}: seq pool create: {e}", canary.id));
        let mut sequence = pool
            .alloc(MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("{}: seq alloc: {e}", canary.id));

        let mut logits = vec![0f32; ModelConfig::qwen38_27b().vocab as usize];
        if let Err(e) = prefill_program(&model, &pool, &mut sequence, &prompt_tokens, 0, Some(&mut logits)) {
            if gpu_profile::skip_or_fail(&format!("{}: prefill_program: {e}", canary.id)) {
                continue;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
        let first_token = match decode_program_batch(&model, &pool, &mut [&mut sequence]) {
            Ok(ids) => ids[0],
            Err(e) => {
                if gpu_profile::skip_or_fail(&format!("{}: decode_program_batch: {e}", canary.id)) {
                    continue;
                }
                unreachable!("skip_or_fail panics under the profile");
            }
        };

        assert!(
            logits.iter().all(|v| v.is_finite()),
            "{}: token 0's logits must be finite",
            canary.id
        );
        let ranked = top_k(&logits, TOP_K);
        assert_eq!(
            ranked[0].0 as i32, first_token,
            "{}: the captured logits' argmax must match the emitted token",
            canary.id
        );

        let oracle_rank = ranked
            .iter()
            .position(|&(id, _)| id as u32 == canary.oracle_first_token);
        eprintln!(
            "ignis #72 {}: emitted={first_token} top{TOP_K}={ranked:?} oracle_first={} oracle_rank={oracle_rank:?}",
            canary.id, canary.oracle_first_token
        );
        if let Some(rank) = oracle_rank {
            if rank > 0 {
                eprintln!(
                    "ignis #72 {}: winner-vs-oracle logit gap = {} (winner={}, oracle={})",
                    canary.id,
                    ranked[0].1 - ranked[rank].1,
                    ranked[0].1,
                    ranked[rank].1
                );
            }
        } else {
            eprintln!(
                "ignis #72 {}: the oracle's first token is not even in the top {TOP_K}",
                canary.id
            );
        }
    }
}
