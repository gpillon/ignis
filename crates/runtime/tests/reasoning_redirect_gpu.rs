//! The reasoning redirect on the card (GitHub #315, ADR 0048, spec server/13
//! acceptance 9), on each leaf through the `Compute` seam the server calls.
//!
//! A draw constrained to the stop id -- a permitted set of one -- with the
//! reasoning block open is redirected: the leaf's pending token becomes the
//! close, the next round emits it, and the sequence goes on generating. The
//! same draw with the block closed stays the stop id, and the round that
//! emits it ends the turn. Both on the prefill's draw and on a decode round's.
//! The verify round's cut is the CTest unit's
//! (`kernel/tests/test_reasoning_redirect.cpp`): forcing an accepted stop
//! draft on the card would need a hand-made drafter.
//!
//! The leaf's cut is blind to which ids it compares, so two ordinary ids
//! stand in for the EOS and `</think>`; naming the real ones is the server's
//! (its CPU tests). Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure. Run alone: each leg loads a whole model.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{bind_text_scope_27b, materialize, CudaDevice, Reader};
use ignis_core::flash_next::EngineOptions;
use ignis_core::gpu_profile;
use ignis_core::{Compute, DecodeJob, DecodeOutcome, DecodeParams, FinishReason, PrefillJob, TokenId};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, FlashNextLeaf, Model, RuntimeCompute};

const ARTIFACT_27B: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
/// The stop id the `Compute` adapter is built with (its EOS), and the close.
const STOP: TokenId = 9_001;
const CLOSE: TokenId = 9_002;
const MAX_TOKENS: u32 = 16;

fn params() -> DecodeParams {
    DecodeParams {
        max_tokens: Some(MAX_TOKENS),
        ..DecodeParams::default()
    }
}

fn prompt() -> Vec<TokenId> {
    (0..48).map(|i| 1000 + (i * 7919) % 60_000).collect()
}

fn prefill(compute: &dyn Compute, request: u64, permitted: Option<&[TokenId]>, close: Option<TokenId>) -> bool {
    let job = PrefillJob {
        request,
        tokens: prompt(),
        context_tokens: 256,
        start_position: 0,
        params: params(),
        shared_prefix: None,
        publish_prefix: None,
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: None,
        readout: None,
        permitted: permitted.map(|ids| ids.to_vec().into()),
        attention: None,
        reasoning_close: close,
    };
    let outcomes = compute.prefill_step(&[job]).unwrap_or_else(|e| panic!("prefill_step {request}: {e}"));
    outcomes[0].reasoning_redirected
}

fn round(compute: &dyn Compute, request: u64, permitted: Option<&[TokenId]>, close: Option<TokenId>) -> DecodeOutcome {
    let job = DecodeJob {
        request,
        lane: 0,
        params: params(),
        remaining_tokens: MAX_TOKENS,
        permitted: permitted.map(|ids| ids.to_vec().into()),
        reasoning_close: close,
    };
    compute
        .decode_step(&[job])
        .unwrap_or_else(|e| panic!("decode_step {request}: {e}"))
        .remove(0)
}

/// Three more rounds, each committing something and none finishing: the
/// sequence goes on after the close.
fn keeps_generating(compute: &dyn Compute, request: u64, label: &str) {
    for at in 0..3 {
        let next = round(compute, request, None, None);
        assert!(!next.tokens.is_empty() && next.finish.is_none(), "{label}: round {at} after the close: {next:?}");
    }
}

fn the_redirect_holds(compute: &dyn Compute, label: &str) {
    // The prefill's draw, the block open: redirected, and the first round
    // emits the close -- its own draw not redirected, the block now closed.
    assert!(prefill(compute, 1, Some(&[STOP]), Some(CLOSE)), "{label}: the prefill's stop draw is redirected");
    let first = round(compute, 1, None, Some(CLOSE));
    assert_eq!(first.tokens.first(), Some(&CLOSE), "{label}: {first:?}");
    assert!(first.finish.is_none() && !first.reasoning_redirected, "{label}: {first:?}");
    keeps_generating(compute, 1, label);
    compute.release(1);

    // The prefill's draw, the block closed: the stop id, and the turn ends.
    assert!(!prefill(compute, 2, Some(&[STOP]), None), "{label}");
    let ended = round(compute, 2, None, None);
    assert_eq!(ended.finish, Some(FinishReason::Stop), "{label}: {ended:?}");
    assert!(ended.tokens.is_empty(), "{label}: the stop is never emitted: {ended:?}");

    // A decode round's draw, the block open.
    assert!(!prefill(compute, 3, None, None), "{label}");
    let drew = round(compute, 3, Some(&[STOP]), Some(CLOSE));
    assert!(drew.reasoning_redirected && drew.finish.is_none(), "{label}: {drew:?}");
    let next = round(compute, 3, None, Some(CLOSE));
    assert_eq!(next.tokens.first(), Some(&CLOSE), "{label}: {next:?}");
    assert!(next.finish.is_none() && !next.reasoning_redirected, "{label}: {next:?}");
    keeps_generating(compute, 3, label);
    compute.release(3);

    // A decode round's draw, the block closed.
    assert!(!prefill(compute, 4, None, None), "{label}");
    let drew = round(compute, 4, Some(&[STOP]), None);
    assert!(!drew.reasoning_redirected, "{label}: {drew:?}");
    let ended = round(compute, 4, None, None);
    assert_eq!(ended.finish, Some(FinishReason::Stop), "{label}: {ended:?}");
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_27b_leaf_redirects_a_stop_drawn_inside_the_reasoning() {
    let path = Path::new(ARTIFACT_27B);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT_27B}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let leaf = CudaLeaf::new(device, reader, artifact, handles, CudaLeafConfig::default());
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    the_redirect_holds(&RuntimeCompute::new(model, STOP), "27B");
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn the_flash_next_leaf_redirects_a_stop_drawn_inside_the_reasoning() {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display()));
        return;
    }
    let options = EngineOptions {
        max_context_tokens: 4096,
        ..EngineOptions::default()
    };
    let leaf = match FlashNextLeaf::open(&path, options) {
        Ok(leaf) => leaf,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("open the Flash-Next leaf: {e}"));
            return;
        }
    };
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    the_redirect_holds(&RuntimeCompute::new(model, STOP), "Flash-Next");
}
