//! #518 on a Flash-Next leaf, MTP off and on: the prompt's last job emits the
//! sequence's pending token through the `Compute` seam, the first round takes
//! it as its anchor without emitting it again, and the text is the one the
//! same prompt gives with every round returning its anchor -- no token twice,
//! none missing -- with the same speculative counters under MTP.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure. Needs ~38 GB of free RAM for the pinned expert pool, one load at
//! a time.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::flash_next::EngineOptions;
use ignis_core::gpu_profile;
use ignis_core::speculation::{FlashNextSpeculation, SpeculativeBackend};
use ignis_core::{Compute, DecodeJob, DecodeOutcome, DecodeParams, PrefillJob, RequestId, SpecCounters, TokenId};
use ignis_runtime::{FlashNextLeaf, Model, RuntimeCompute};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MAX_CONTEXT: u32 = 4096;
const PROMPT: u32 = 300;
const GENERATED: u32 = 24;
/// Unread: every request runs under `ignore_eos`, so the synthetic prompt's
/// text runs to its cap either way.
const NO_EOS: TokenId = TokenId::MAX;

fn leaf(speculation: Option<FlashNextSpeculation>) -> Option<FlashNextLeaf> {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display()));
        return None;
    }
    let options = EngineOptions { max_context_tokens: MAX_CONTEXT, speculation, ..EngineOptions::default() };
    match FlashNextLeaf::open(&path, options) {
        Ok(leaf) => Some(leaf),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("open the Flash-Next leaf: {e}"));
            None
        }
    }
}

/// One request, greedy, to `GENERATED` tokens: its text and its rounds'
/// counters. `first_with_the_prefill`: the way the scheduler serves it since
/// #518; otherwise every round returns its anchor, as before.
fn serve(compute: &RuntimeCompute<FlashNextLeaf>, request: RequestId, first_with_the_prefill: bool) -> (Vec<TokenId>, SpecCounters) {
    let params = DecodeParams { max_tokens: Some(GENERATED), ignore_eos: true, ..DecodeParams::default() };
    let outcomes = compute
        .prefill_step(&[PrefillJob {
            checkpoint: None,
            capture_checkpoint: None,
            multimodal: None,
            readout: None,
            request,
            tokens: (0..PROMPT).map(|i| 1000 + (i * 7919) % 60_000).collect(),
            context_tokens: MAX_CONTEXT,
            start_position: 0,
            params,
            shared_prefix: None,
            publish_prefix: None,
            permitted: None,
            attention: None,
            reasoning_close: None,
            drafter_floor: 0,
            next_prompt_token: None,
        }])
        .unwrap_or_else(|e| panic!("prefill_step: {e}"));
    let mut text = Vec::new();
    if first_with_the_prefill {
        let first = outcomes[0].emitted.clone().unwrap_or_else(|| panic!("the prompt's last job emits nothing"));
        assert_eq!((first.tokens.len(), first.finish), (1, None), "one token, the request goes on: {first:?}");
        text.extend(first.tokens);
    }
    let mut spec = SpecCounters::default();
    let mut round = 0;
    while (text.len() as u32) < GENERATED {
        let out = compute
            .decode_step(&[DecodeJob {
                request,
                lane: 0,
                params,
                remaining_tokens: GENERATED - text.len() as u32,
                permitted: None,
                reasoning_close: None,
                anchor_emitted: first_with_the_prefill && round == 0,
                checked: false,
            }])
            .unwrap_or_else(|e| panic!("decode_step round {round}: {e}"));
        let DecodeOutcome { tokens, finish, spec: counters, .. } = out.into_iter().next().expect("one outcome");
        spec = spec + counters.unwrap_or_default();
        text.extend(tokens);
        round += 1;
        if finish.is_some() {
            break;
        }
    }
    compute.release(request);
    (text, spec)
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn the_first_token_leaves_with_the_prefill_with_mtp_off_and_on() {
    let mtp = FlashNextSpeculation::new(SpeculativeBackend::Mtp, 2, 0).expect("MTP at two drafts");
    for speculation in [None, Some(mtp)] {
        let Some(leaf) = leaf(speculation) else {
            return;
        };
        let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
        let compute = RuntimeCompute::new(model, NO_EOS);

        let (text, spec) = serve(&compute, 1, true);
        let before = serve(&compute, 2, false);
        assert_eq!(text.len() as u32, GENERATED, "MTP {}: {text:?}", speculation.is_some());
        assert_eq!((text, spec), before, "MTP {}: no token twice, none missing", speculation.is_some());
        assert_eq!(spec.rounds > 0, speculation.is_some(), "verify rounds only with MTP: {spec:?}");
    }
}
