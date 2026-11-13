//! GPU integration coverage for a speculative `CudaLeaf` (P5-05, GitHub
//! #155): the DFlash2 drafter driven end to end through the public `Compute`
//! seam `ignis-server` calls. `ignis-core`'s `dflash2_round_gpu.rs` proves
//! the round's text and acceptance; this proves the runtime's side of it --
//! the leaf loads the drafter from the model scope's handles, passes no
//! drafts, and turns each lane's extent and committed run into the round's
//! speculative counters. Since #518 also that the first token leaving with
//! the prefill, and the first verify round not emitting it again, leave the
//! text, the rounds and the acceptance as they were.
//!
//! Its own test binary rather than a second test in `cuda_leaf_gpu.rs`: each
//! materializes the artifact, and the card fits one at a time.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, DraftModule, FrontendSet, Reader};
use ignis_core::gpu_profile;
use ignis_core::{
    Compute, DecodeJob, DecodeOutcome, DecodeParams, KvFormat, PrefillJob, RequestId, Speculation, SpeculativeBackend,
    TokenId,
};
use ignis_runtime::{auto_kv_pool_bytes, CudaLeaf, CudaLeafConfig, Model, RuntimeCompute};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 1024;
const WINDOW: u32 = 7;
const MAX_GENERATED: u32 = 48;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_dflash2_leaf_drafts_inside_the_round_and_reports_each_rounds_counters() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    // A canary in a user turn with thinking closed: the text the drafter
    // was trained to continue.
    let prompt = frontend
        .tokenizer()
        .encode(
            "<|im_start|>user\nIn one sentence, what does `fn main() { println!(\"hi\"); }` do?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"));

    let (plan, handles) = bind_model_scope_27b(&reader, Some(DraftModule::Dflash2))
        .unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize text + dflash2: {e}")) {
                return;
            }
            unreachable!();
        }
    };

    let config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::Bf16,
        kv_pool_bytes: auto_kv_pool_bytes(KvFormat::Bf16, MAX_CONTEXT),
        prefill_chunk_tokens: MAX_CONTEXT,
        speculation: Some(Speculation::new(SpeculativeBackend::Dflash2, WINDOW).unwrap()),
        ..CudaLeafConfig::default()
    };
    // GitHub #213: KV-RAM is one arena pinned at the load, so a leaf without
    // one refuses every spill. A gibibyte costs this load nothing it notices
    // and keeps a leg that starts spilling from failing for the wrong reason.
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config)
        .with_kv_ram_arena(1024 * 1024 * 1024)
        .unwrap_or_else(|e| panic!("pin KV-RAM: {e}"));
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let compute = RuntimeCompute::new(model, eos);

    let params = DecodeParams { max_tokens: Some(MAX_GENERATED), ..DecodeParams::default() };
    // #518: one request served the way the scheduler serves it -- its first
    // token out of the prompt's last job, the first verify round taking it
    // as its anchor and emitting only what it accepted -- and the same prompt
    // served with every round returning its anchor, as before. The text, the
    // rounds and the acceptance are the same.
    let serve = |request: RequestId, first_with_the_prefill: bool| -> (Vec<TokenId>, u32, u32) {
        let outcomes = compute
            .prefill_step(&[PrefillJob {
                checkpoint: None,
                capture_checkpoint: None,
                multimodal: None,
                readout: None,
                request,
                tokens: prompt.clone(),
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
                checked: false,
            }])
            .unwrap_or_else(|e| panic!("prefill_step: {e}"));
        let mut text = Vec::new();
        if first_with_the_prefill {
            let first = outcomes[0].emitted.clone().unwrap_or_else(|| panic!("the prompt's last job emits nothing"));
            assert_eq!((first.tokens.len(), first.finish), (1, None), "one token, the request goes on: {first:?}");
            text.extend(first.tokens);
        }
        let mut accepted = 0u32;
        let mut rounds = 0u32;
        while (text.len() as u32) < MAX_GENERATED {
            let anchor_emitted = first_with_the_prefill && rounds == 0;
            let remaining = MAX_GENERATED - text.len() as u32;
            let out = compute
                .decode_step(&[DecodeJob {
                    request,
                    lane: 0,
                    params,
                    remaining_tokens: remaining,
                    permitted: None,
                    reasoning_close: None,
                    anchor_emitted,
                    checked: false,
                }])
                .unwrap_or_else(|e| panic!("decode_step: {e}"));
            let Some(DecodeOutcome { tokens, finish, spec, .. }) = out.into_iter().next() else {
                break;
            };
            if tokens.is_empty() && finish.is_some() && spec.is_none() {
                // A request already at its cap finishes without a round.
                break;
            }
            let spec = spec.unwrap_or_else(|| panic!("round {rounds}: a DFlash2 leaf's round carried no counters"));
            // The lane's budget counts the anchor it commits again.
            let budget = remaining + u32::from(anchor_emitted);
            assert_eq!(spec.rounds, 1, "round {rounds}: one verify round per decode step");
            assert_eq!(spec.drafted, WINDOW.min(budget - 1), "round {rounds}: the lane's extent");
            assert!(spec.accepted <= spec.drafted, "round {rounds}: {spec:?}");
            if finish.is_none() {
                // No cut: the run is the anchor, unless it left with the
                // prefill, plus every accepted draft.
                let anchor = u32::from(!anchor_emitted);
                assert_eq!(tokens.len() as u32, spec.accepted + anchor, "round {rounds}: run against {spec:?}");
            }
            accepted += spec.accepted;
            text.extend(tokens);
            rounds += 1;
            if finish.is_some() {
                break;
            }
        }
        compute.release(request);
        (text, rounds, accepted)
    };

    let (text, rounds, accepted) = serve(1, true);
    assert!(rounds > 0, "no verify round ran");
    assert!(accepted > 0, "the leaf never landed a draft: the drafter is not proposing inside the round");
    assert!(text.len() as u32 <= MAX_GENERATED, "the runs outgrew max_tokens: {}", text.len());
    let before = serve(2, false);
    assert_eq!((text, rounds, accepted), before, "no token twice, none missing, the same rounds");
}
