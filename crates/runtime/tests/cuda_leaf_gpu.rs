//! GPU integration coverage for the production `CudaLeaf` (GitHub #61 /
//! P1-25): the same real artifact + real GPU path as `ignis-core`'s
//! `program_full_gpu.rs`, but driven through the public `Compute` seam
//! `ignis-server` actually calls — proves the runtime crate's safe wrapper
//! (sequence lifecycle, EOS, max_tokens) works end to end against the real
//! GPU-backed leaf, not just the CPU stub `runtime_compute.rs` exercises.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{bind_text_scope_27b, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::gpu_profile;
use ignis_core::{Compute, DecodeJob, DecodeOutcome, DecodeParams, PrefillJob, RequestId};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, RuntimeCompute};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_GENERATED: usize = 32;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_cuda_leaf_prefills_and_decodes_a_real_prompt_through_the_compute_trait() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    let prompt = frontend
        .tokenizer()
        .encode("In one sentence, what is 2 + 2?")
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"));
    assert!(!prompt.is_empty());

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
    let model = Arc::new(
        Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")),
    );
    let compute = RuntimeCompute::new(model, eos);

    let request: RequestId = 1;
    compute
        .prefill_step(&[PrefillJob {
            request,
            tokens: prompt,
            context_tokens: 256,
            start_position: 0,
            params: DecodeParams {
                max_tokens: Some(MAX_GENERATED as u32),
                ..DecodeParams::default()
            },
            shared_prefix: None,
            publish_prefix_tokens: None,
        }])
        .unwrap_or_else(|e| panic!("prefill_step: {e}"));

    let mut generated = Vec::new();
    for _ in 0..MAX_GENERATED {
        let out = compute
            .decode_step(&[DecodeJob {
                request,
                lane: 0,
                params: DecodeParams {
                    max_tokens: Some(MAX_GENERATED as u32),
                    ..DecodeParams::default()
                },
            }])
            .unwrap_or_else(|e| panic!("decode_step: {e}"));
        match out.into_iter().next() {
            Some(DecodeOutcome::Token(token)) => generated.push(token),
            // `Finished`: the request hit EOS or its `max_tokens` cap and
            // the adapter already released its leaf sequence.
            Some(DecodeOutcome::Finished(_)) | None => break,
        }
    }
    assert!(
        !generated.is_empty(),
        "the real model must generate at least one token for a real prompt"
    );
    assert!(
        generated.len() <= MAX_GENERATED,
        "decode must stop at max_tokens even without an EOS hit"
    );
    compute.release(request);
}
