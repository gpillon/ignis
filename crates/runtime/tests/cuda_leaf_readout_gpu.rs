//! Does the **readout seam** work on the card? (GitHub #237/#238/#239.)
//!
//! Every test of the decision path so far runs against `MockCompute` or a
//! CPU `StubLeaf`. The one GPU test that proves the *technique*
//! (`crates/server/tests/classify_readout_gpu.rs`) calls
//! `step::prefill_program` directly — it bypasses `StepLeaf`,
//! `RuntimeCompute`, `PrefillJob::readout` and the gather entirely. So the
//! path that actually serves `/v1/decide` had never executed on a GPU: in
//! particular `CudaLeaf::prefill` had only ever been called with `None` for
//! `out_logits`, and `prefill_program_sampled` with a non-null logits
//! pointer was an untried FFI call.
//!
//! This closes that. It drives a real `CudaLeaf` through the public
//! `Compute` seam with a readout requested, and checks the three things that
//! could be wrong in a way nothing CPU-side would notice:
//!
//! 1. The buffer is the right width. `RuntimeCompute` sizes it from
//!    `StepLeaf::vocab`, and a short one is a buffer overrun through FFI,
//!    not a wrong answer.
//! 2. The logits are the model's. A buffer the kernel never wrote would come
//!    back all zeros and every probability would be uniform — which a test
//!    that only checked "finite" would pass.
//! 3. The gather reads the *answer* tokens. The seam is checked against the
//!    one oracle that exists on this path: the unrestricted argmax the
//!    readout reports must be the token the leaf's own decode commits for
//!    the same prompt, since a decision's prompt is greedy at its last
//!    position.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile it is a failure.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::decision::AnswerAlphabet;
use ignis_core::gpu_profile;
use ignis_core::{Compute, DecodeJob, DecodeParams, PrefillJob, RequestId};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, RuntimeCompute};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// A prompt whose next token is an answer letter, in the shape `/v1/decide`
/// builds — the measured `DIRECT_SYSTEM` instruction plus the decision as
/// JSON, rendered by the artifact's own chat template with thinking off.
const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_readout_crosses_the_real_compute_seam() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));

    // The answer alphabet, from the loaded tokenizer exactly as the server
    // computes it at load.
    let alphabet = AnswerAlphabet::from_tokenizer(frontend.tokenizer());
    let answers = alphabet
        .take(3)
        .unwrap_or_else(|| panic!("this tokenizer names fewer than three options"))
        .to_vec();
    let labels: Vec<&str> = answers.iter().map(|a| a.label.as_str()).collect();
    assert_eq!(labels, vec!["A", "B", "C"], "the first three answer tokens");

    // The decision in the layout `/v1/decide` sends since GitHub #240: the
    // evidence rides in the **system block**, where a sibling question's
    // retained prefix can reach it, and the user turn carries only the
    // question. `decide::messages_for` says why it is the block and not
    // merely first.
    let evidence = r#"{"evidence":"Help! My payouts have been failing for 3 days."}"#;
    let payload = r#"{"criterion":"Which team should handle this?","options":[{"letter":"A","description":"Payments, invoicing, refunds"},{"letter":"B","description":"Bugs, outages, integrations"},{"letter":"C","description":"Pricing, upgrades, new accounts"}]}"#;
    let rendered = format!(
        "<|im_start|>system\n{DIRECT_SYSTEM}\n\n{evidence}<|im_end|>\n<|im_start|>user\n{payload}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let prompt = frontend
        .tokenizer()
        .encode(&rendered)
        .unwrap_or_else(|e| panic!("tokenize the decision prompt: {e}"));
    assert!(!prompt.is_empty());

    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let leaf = CudaLeaf::new(device, reader, artifact, handles, CudaLeafConfig::default())
        .with_kv_ram_arena(1024 * 1024 * 1024)
        .unwrap_or_else(|e| panic!("pin KV-RAM: {e}"));
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let compute = RuntimeCompute::new(model, eos);

    let params = DecodeParams {
        max_tokens: Some(1),
        ..DecodeParams::default()
    };
    let job = |request: RequestId, readout: Option<Vec<u32>>| PrefillJob {
        request,
        tokens: prompt.clone(),
        context_tokens: 2048,
        start_position: 0,
        params,
        shared_prefix: None,
        publish_prefix: None,
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: None,
        readout: readout.map(Arc::from),
        permitted: None,
        attention: None,
};

    // ── the readout itself ───────────────────────────────────────────────
    let answer_ids: Vec<u32> = answers.iter().map(|a| a.id).collect();
    let outcomes = compute
        .prefill_step(&[job(1, Some(answer_ids.clone()))])
        .unwrap_or_else(|e| panic!("prefill_step with a readout: {e}"));
    let readout = outcomes[0]
        .readout
        .as_ref()
        .unwrap_or_else(|| panic!("the real leaf returned no readout"));

    assert_eq!(readout.logits.len(), answers.len(), "one logit per answer token");
    assert!(
        readout.logits.iter().all(|v| v.is_finite()),
        "the model's logits are finite: {:?}",
        readout.logits
    );
    // A buffer the kernel never touched reads as zeros, and every one of
    // these checks below would still pass on a uniform distribution — this
    // is the one that would not.
    assert!(
        readout.logits.iter().any(|&v| v != 0.0),
        "the kernel wrote into the buffer: {:?}",
        readout.logits
    );
    assert!(
        readout.logits.windows(2).any(|pair| pair[0] != pair[1]),
        "and wrote *different* values per answer token, so this is a \
         distribution rather than an untouched allocation: {:?}",
        readout.logits
    );
    assert!(
        readout.full_log_sum_exp.is_finite(),
        "the full-vocabulary log-sum-exp is finite: {}",
        readout.full_log_sum_exp
    );
    let mass = readout.answer_mass();
    assert!(
        (0.0..=1.0).contains(&mass),
        "answer mass is a probability: {mass}"
    );
    let probabilities = readout.probabilities();
    let sum: f64 = probabilities.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "the restricted softmax sums to one: {sum}");

    eprintln!(
        "ignis readout gpu: logits {:?}, mass {mass:.4}, argmax {} ({:?}), p {:?}",
        readout.logits,
        readout.full_argmax,
        frontend.tokenizer().decode(&[readout.full_argmax]).ok(),
        probabilities
    );

    // The buffer is as wide as the head, not as the tokenizer: an argmax
    // past the vocabulary would mean the gather read uninitialized memory.
    let vocab = ModelConfig::qwen38_27b().vocab as u32;
    assert!(
        readout.full_argmax < vocab,
        "the unrestricted argmax {} is a column of the {vocab}-wide head",
        readout.full_argmax
    );

    // The measured claim, on one row: the model's own unrestricted winner is
    // a declared answer. One row cannot support an accuracy claim and none
    // is made — what this says is that the gather is reading the position
    // the answer is written at, not some other one.
    assert!(
        answer_ids.contains(&readout.full_argmax),
        "the unrestricted winner is a declared option, as it was on 144/144 \
         authored rows: got {} ({:?})",
        readout.full_argmax,
        frontend.tokenizer().decode(&[readout.full_argmax]).ok()
    );
    assert!(
        mass > 0.5,
        "and the declared options hold the distribution (measured median \
         0.998): {mass}"
    );

    // ── the oracle: the same prompt, decoded ─────────────────────────────
    // A second request over the same prompt, greedy, one token. What the
    // leaf commits *is* the unrestricted argmax at that position, so it is
    // the only independent check that the gather read the right column.
    compute
        .prefill_step(&[job(2, None)])
        .unwrap_or_else(|e| panic!("prefill_step without a readout: {e}"));
    let decoded = compute
        .decode_step(&[DecodeJob {
            request: 2,
            lane: 0,
            params,
            remaining_tokens: 1,
            permitted: None,
}])
        .unwrap_or_else(|e| panic!("decode_step: {e}"));
    let committed = decoded
        .into_iter()
        .next()
        .and_then(|outcome| outcome.tokens.first().copied())
        .unwrap_or_else(|| panic!("the decode committed no token"));
    assert_eq!(
        committed, readout.full_argmax,
        "the token the leaf decodes greedily at this position ({:?}) is the \
         argmax the readout reported ({:?}) — if these differ, the gather is \
         reading a different position or a different buffer",
        frontend.tokenizer().decode(&[committed]).ok(),
        frontend.tokenizer().decode(&[readout.full_argmax]).ok()
    );

    compute.release(1);
    compute.release(2);
}
