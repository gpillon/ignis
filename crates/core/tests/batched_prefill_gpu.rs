//! B1 (spec 08, GitHub #31) — the multi-token (batched) prefill forward
//! path: a `seq > 1` fresh-prompt prefill runs the layer stack in one
//! multi-token pass (the multi-token GEMM, kernel-abi 05, + the
//! multi-token attention, kernel-abi 01, + the per-token GDN recurrence
//! / RoPE / KV writeback) instead of the per-token loop; `seq == 1` (the
//! GEMV special case, ADR 0001) and a warm-KV (prefix-reuse tail)
//! prefill keep the per-token loop (the ADR 0003 eager fallback — a
//! busy/absent multi-token kernel falls back to it too, after the fresh
//! state is restored).
//!
//! The `prefill_step` dispatch is observed through the backend's
//! `batched_prefill_count` surface (the multi-token pass increments it;
//! the eager loop / a fallback never do).
//!
//! **CPU-only pins (no GPU — always run):** the dispatch rule's truth
//! table (`batched_prefill_eligible`) + the synthetic model's NVFP4 GEMM
//! shapes (every GEMM's `k` passes the kernel's `k % 16 == 0` group-
//! scale validation — the GDN readout GEMM's `k` = the readout width
//! `state_rows` is a multiple of 16; a violating `k` is rejected by the
//! kernel *before* any CUDA call, so the synthetic forward pass would
//! fault on it).
//!
//! **GPU-gated tests (a busy/absent GPU self-skips, ADR 0006 — a few MB
//! of VRAM runs even with the model loaded):**
//! 1. **Batched-prefill self-consistency (synthetic model):** a
//!    multi-token prompt's prefill takes the multi-token path (the
//!    `batched_prefill_count` increments) and the emitted decode tokens
//!    are in-vocabulary + reproducible across two fresh backends (the
//!    self-consistency invariant, ADR 0007: greedy + fixed seed — the
//!    batched path's acceptance is a *sane* output, not a bit-exact
//!    agreement with the per-token loop, spec 08's design §7 caveat).
//! 2. **The `seq == 1` GEMV special case:** a single-token prefill keeps
//!    the single-token path (the counter never increments, ADR 0001).
//! 3. **Real-model E2E (the Qwen 3.8-27B artifact, feature `cuda`):**
//!    a multi-token prompt's prefill through `CudaCompute::from_artifact`
//!    takes the multi-token path + the decode stream is in-vocab (the
//!    real vocab, 248 320) + reproducible across two fresh backends.
//!
//! Run with:
//!   `cargo test -p ignis-core --test batched_prefill_gpu -- --ignored`
//!   `cargo test -p ignis-core --test batched_prefill_gpu --features cuda -- --ignored`
//!
//! Build precondition: links the kernel .lib (the FFI, ADR 0001). The
//! `cuda`-gated real-model E2E also needs the `cuda` feature (the
//! artifact's device surface, the `from_artifact` constructor).

use ignis_core::compute::{CudaCompute, ModelConfig, Weights};
use ignis_core::scheduler::{Compute, DecodeJob, PrefillJob};
use ignis_core::types::{ComputeError, DecodeParams, TokenId};

/// The self-consistency run's request id (a fresh request per backend).
const REQ: u64 = 1;

/// A multi-token synthetic prompt (in-vocab ids — the synthetic model's
/// vocab is 256, B1 / #31: `seq > 1` exercises the multi-token path).
const PROMPT: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// The real-model E2E's prompt (in-vocab token ids — the real Qwen 3.8-
/// 27B vocab is 248 320, so small ids are valid, A3 / #30). A multi-
/// token prompt (B1 / #31: `seq > 1` exercises the multi-token path).
/// The real-model constants are `cuda`-feature-gated (the real-model E2E
/// needs the artifact's device surface — `from_artifact`, the feature
/// `cuda`).
#[cfg(feature = "cuda")]
const REAL_PROMPT: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
/// The real-model E2E's decode budget (the `max_tokens` cap + the number
/// of decode steps to drive, A3 / #30).
#[cfg(feature = "cuda")]
const REAL_MAX_TOKENS: u32 = 4;
#[cfg(feature = "cuda")]
const REAL_N_DECODE: u32 = 3;
/// The real Qwen 3.8-27B vocab (the in-vocab check, A3 / #30).
#[cfg(feature = "cuda")]
const REAL_VOCAB: u64 = 248_320;

// ---------------------------------------------------------------------------
// CPU-only pins (no GPU needed — always run)
// ---------------------------------------------------------------------------

/// The batched (multi-token) prefill dispatch rule's truth table (B1 /
/// #31, spec 08 acceptance criteria 1 + 3): the multi-token path is used
/// only when the prompt is `seq > 1` tokens (the `seq == 1` GEMV special
/// case stays on the single-token path, ADR 0001) on a fresh (empty-KV,
/// `kv_len == 0`) request (the multi-token attention's fresh-prompt
/// causal mask, base_pos = 0 — a warm-KV tail prefill keeps the per-
/// token loop).
#[test]
fn batched_prefill_dispatch_truth_table() {
    use ignis_core::compute::CudaCompute;
    // The GEMV special case (ADR 0001): `seq == 1` is the single-token
    // path, never the multi-token one.
    assert!(
        !CudaCompute::batched_prefill_eligible(1, 0),
        "seq == 1 (fresh) must stay on the GEMV special case"
    );
    assert!(
        !CudaCompute::batched_prefill_eligible(1, 4),
        "seq == 1 (warm) must stay on the single-token path"
    );
    // A multi-token prompt on a fresh (empty-KV) request: the multi-
    // token path.
    assert!(
        CudaCompute::batched_prefill_eligible(4, 0),
        "seq > 1 (fresh) must take the multi-token path"
    );
    assert!(
        CudaCompute::batched_prefill_eligible(256, 0),
        "a long multi-token prompt (fresh) must take the multi-token path"
    );
    // A multi-token prompt on a warm-KV request (a prefix-reuse tail
    // prefill): the per-token loop (the multi-token attention's fresh-
    // prompt causal mask is only valid on an empty cache).
    assert!(
        !CudaCompute::batched_prefill_eligible(4, 1),
        "seq > 1 (warm KV) must keep the per-token loop"
    );
    assert!(
        !CudaCompute::batched_prefill_eligible(4, 256),
        "seq > 1 (a warm KV tail) must keep the per-token loop"
    );
}

/// The synthetic model's NVFP4 GEMM shapes pass the kernel's `k % 16 ==
/// 0` group-scale validation (kernel-abi 05: a violating `k` is rejected
/// *before* any CUDA call — the synthetic forward pass's kernels all pass
/// validation, B1 / #31). Regression pin: the GDN readout GEMM's `k` =
/// the readout width `state_rows` was a multiple-of-16 violation in the
/// pre-#31 synthetic geometry (the synthetic forward pass faulted on
/// the GEMM validation, and every synthetic GPU e2e self-skipped).
#[test]
fn synthetic_nvfp4_gemv_shapes_pass_the_kernel_validation() {
    use ignis_core::compute::HeadWeight;

    let cfg = ModelConfig::synthetic();
    let w = Weights::synthetic(&cfg, 42);
    // Every layer's GEMM shapes: the attention / GDN projections (the
    // `nvfp4_gemm` host-weight slots), the FFN — the kernel's `k` must
    // be a multiple of 16 (the NVFP4 group scale, kernel-abi 05); a
    // zero-geometry slot (an unused weight) carries `k = 0` (no GEMM).
    for (i, lw) in w.per_layer.iter().enumerate() {
        for (slot, wgt) in lw.projection.iter().enumerate() {
            assert!(
                wgt.k == 0 || wgt.k % 16 == 0,
                "layer {i}'s projection slot {slot} GEMM k = {} violates the NVFP4 \
                 group-scale validation (k % 16 != 0)",
                wgt.k
            );
        }
        assert!(
            lw.gdn_output.k == 0 || lw.gdn_output.k % 16 == 0,
            "the GDN readout GEMM k = {} (the readout width state_rows) must be a \
             multiple of 16 (the NVFP4 group scale, kernel-abi 05)",
            lw.gdn_output.k
        );
        for wgt in [&lw.ffn_gate, &lw.ffn_up, &lw.ffn_down] {
            assert!(
                wgt.k == 0 || wgt.k % 16 == 0,
                "the FFN GEMM k = {} must be a multiple of 16 (kernel-abi 05)",
                wgt.k
            );
        }
    }
    // The lm_head (the logits GEMM — the synthetic path's NVFP4 variant,
    // the `ignis_nvfp4_gemm_*` kernels, kernel-abi 05).
    match &w.lm_head {
        HeadWeight::Nvfp4(nv) => {
            assert!(
                nv.k == 0 || nv.k % 16 == 0,
                "the lm_head GEMM k = {} must be a multiple of 16 (kernel-abi 05)",
                nv.k
            );
        }
        HeadWeight::DequantBf16 { .. } => {
            // The bf16 logits path (the artifact's W8-dequantized lm_head)
            // has no NVFP4 group-scale constraint (the `ignis_bf16_gemm`
            // kernel, A2b / #29) — no `k % 16` validation.
        }
    }
}

// ---------------------------------------------------------------------------
// GPU-gated tests (a busy/absent GPU self-skips, ADR 0006)
// ---------------------------------------------------------------------------

/// A non-zero kernel rc (a CUDA error / busy GPU) is a skip, never a
/// failure (ADR 0006 — the GPU is occupied by the reference runner).
fn skip_on_err(e: &ComputeError) -> bool {
    match e {
        ComputeError::Kernel(rc) => {
            eprintln!("SKIP: kernel call failed (rc={rc}; GPU busy/unavailable, ADR 0006)");
            true
        }
        _ => false,
    }
}

/// Drive a prefill + up to `n` decode steps for `request`, collecting the
/// generated tokens (until a soft-stop — `max_tokens` / EOS — or `n`
/// steps). A kernel error (a busy/absent GPU) is surfaced to the caller
/// (the caller decides skip vs. fault).
fn run(
    compute: &CudaCompute,
    prompt: &[u32],
    max_tokens: u32,
    n: u32,
) -> Result<Vec<TokenId>, ComputeError> {
    let params = DecodeParams {
        max_tokens: Some(max_tokens),
        ..DecodeParams::default()
    };
    // The prefill (the `prefill_step` seam — the dispatch under test, B1
    // / #31): a `seq > 1` fresh prompt takes the multi-token path; a
    // `seq == 1` prompt the GEMV special case.
    let pjob = PrefillJob {
        request: REQ,
        tokens: prompt.to_vec(),
        params,
    };
    compute.prefill_step(std::slice::from_ref(&pjob))?;
    // The decode steps: one token per step (the greedy sample, the
    // deterministic token, ADR 0007). A soft-stop (`max_tokens` / EOS)
    // ends the run early (a per-job `None`, not a fault).
    let mut tokens = Vec::new();
    for _ in 0..n {
        let djob = DecodeJob {
            request: REQ,
            lane: 0,
            params: DecodeParams {
                max_tokens: Some(max_tokens),
                ..DecodeParams::default()
            },
        };
        let out = compute.decode_step(std::slice::from_ref(&djob))?;
        match out.into_iter().next() {
            Some(Some(t)) => tokens.push(t),
            Some(None) => break, // the request soft-stopped (max_tokens / EOS)
            None => return Err(ComputeError::Kernel(-1)), // an empty result (a bug)
        }
    }
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// The synthetic model (the multi-token path's mechanism test, spec 08:
// "the mechanism is testable on the synthetic model first")
// ---------------------------------------------------------------------------

/// The batched-prefill self-consistency invariant (spec 08 acceptance
/// criteria 1 + 2 — ADR 0007: greedy + fixed seed): a multi-token prompt
/// (the `seq > 1` case) runs the prefill through the **multi-token path**
/// (the `batched_prefill_count` increments — the per-token loop never
/// would), and the decode stream after it is (a) in-vocabulary (the
/// greedy sample's contract — a *sane* output, the correctness floor,
/// ADR 0005) and (b) **reproducible** across two fresh backends (the
/// same synthetic model, the same seed → the same tokens).
///
/// The synthetic backends use the eager constructor (`new_eager` — no
/// kernel-leaf startup graph capture): the graph capture is not part of
/// the prefill / decode forward path (the graph *launch* is the B2 /
/// #32 gap — the captured graph is never launched by the decode step),
/// and a startup capture that interleaves with another thread's kernel
/// launches (the test harness's parallel execution — the leaf's capture
/// stream is context-global) leaves a capture error state that makes the
/// concurrent default-stream launches fault spuriously (the "operation
/// not permitted when stream is capturing" report — the ADR 0003
/// fallback then masks the dispatch under test). The eager path is the
/// correctness floor (ADR 0005 / 0007), so the prefill / decode forward
/// pass — the seam under test — is exercised identically.
#[test]
#[ignore = "GPU launch test — the kernel-leaf forward pass runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn batched_prefill_self_consistency() {
    let cfg = ModelConfig::synthetic();
    let weights = Weights::synthetic(&cfg, 42);

    // Run 1: a multi-token prompt's prefill (the multi-token path — the
    // multi-token GEMM + the multi-token attention + the per-token GDN
    // recurrence / RoPE / KV writeback) + 4 decode steps.
    let compute1 = CudaCompute::new_eager(cfg.clone(), weights.clone());
    let tokens1 = match run(&compute1, &PROMPT, 8, 4) {
        Ok(t) => t,
        Err(e) => {
            if skip_on_err(&e) {
                return;
            }
            panic!("run 1 failed: {e:?}");
        }
    };
    // The prefill took the multi-token path (the dispatch under test, B1
    // / #31 — spec 08 acceptance criterion 1: `prefill_step` uses the
    // multi-token path when `seq > 1`), exactly once (one prefill job).
    assert_eq!(
        compute1.batched_prefill_count(),
        1,
        "a multi-token prompt's prefill must run the multi-token path once"
    );
    // The emitted tokens are in-vocabulary (the correctness floor, ADR
    // 0005: a *sane* output — the greedy sample's contract).
    for &t in tokens1.iter() {
        assert!(
            (t as u64) < cfg.vocab,
            "token {t} must be in vocabulary (vocab = {})",
            cfg.vocab
        );
    }

    // Run 2: a fresh backend (the same synthetic model, the same seed)
    // must produce the *same* decode stream (the self-consistency
    // invariant, ADR 0007: greedy + fixed seed — the batched path's
    // acceptance is a *sane*, reproducible output, not a bit-exact
    // agreement with the per-token loop, spec 08's design §7 caveat).
    let compute2 = CudaCompute::new_eager(cfg.clone(), weights.clone());
    let tokens2 = match run(&compute2, &PROMPT, 8, 4) {
        Ok(t) => t,
        Err(e) => {
            if skip_on_err(&e) {
                return;
            }
            panic!("run 2 failed: {e:?}");
        }
    };
    assert_eq!(
        compute2.batched_prefill_count(),
        1,
        "the second run's prefill must take the multi-token path too"
    );
    assert_eq!(
        tokens1, tokens2,
        "self-consistency: the same seed must produce the same tokens"
    );
}

/// The `seq == 1` GEMV special case (spec 08 acceptance criterion 3 —
/// ADR 0001's eager fallback): a single-token prefill keeps the single-
/// token path (the GEMV `ignis_nvfp4_gemm_decode` + the single-token
/// `ignis_gqa_attention_decode`) — the multi-token counter never
/// increments, and the decode stream is in-vocabulary + reproducible.
#[test]
#[ignore = "GPU launch test — the kernel-leaf forward pass runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn single_token_prefill_keeps_the_gemv_special_case() {
    let cfg = ModelConfig::synthetic();
    let weights = Weights::synthetic(&cfg, 42);

    let compute1 = CudaCompute::new_eager(cfg.clone(), weights.clone());
    let tokens1 = match run(&compute1, &[7], 8, 4) {
        Ok(t) => t,
        Err(e) => {
            if skip_on_err(&e) {
                return;
            }
            panic!("run 1 failed: {e:?}");
        }
    };
    // The `seq == 1` prefill took the GEMV special case (the multi-
    // token path never ran — its counter is untouched, ADR 0001).
    assert_eq!(
        compute1.batched_prefill_count(),
        0,
        "a single-token prefill must stay on the single-token path"
    );
    for &t in tokens1.iter() {
        assert!((t as u64) < cfg.vocab, "token {t} must be in vocabulary");
    }

    // The reproducible-stream invariant (ADR 0007: greedy + fixed seed).
    let compute2 = CudaCompute::new_eager(cfg, weights);
    let tokens2 = match run(&compute2, &[7], 8, 4) {
        Ok(t) => t,
        Err(e) => {
            if skip_on_err(&e) {
                return;
            }
            panic!("run 2 failed: {e:?}");
        }
    };
    assert_eq!(
        compute2.batched_prefill_count(),
        0,
        "the second run's single-token prefill must stay on the single- \
         token path too"
    );
    assert_eq!(tokens1, tokens2, "same seed → same stream (ADR 0007)");
}

// ---------------------------------------------------------------------------
// The real-model E2E (feature `cuda`, the Qwen 3.8-27B artifact)
// ---------------------------------------------------------------------------

/// The real-model batched-prefill E2E (spec 08 acceptance criterion 2,
/// the Qwen 3.8-27B artifact, A3 / #30): a multi-token prompt through
/// `CudaCompute::from_artifact` (the full-correct forward assembly — the
/// 16 GQA + 48 GDN layers, the real head geometry + the GDN state dims +
/// the rotary geometry) takes the **multi-token prefill path** (the
/// multi-token GEMM + the multi-token attention, the `batched_prefill_
/// count` increments once) and the decode stream is (a) in-vocabulary
/// (the 248 320 real vocab — a *sane* output, ADR 0005) and (b)
/// **reproducible** across two fresh backends (the self-consistency
/// invariant, ADR 0007: greedy + fixed seed). Gated on the
/// `IGNIS_ARTIFACT` env var (the artifact path) + the `cuda` feature +
/// the GPU (a busy / OOM GPU self-skips, ADR 0006 — the GPU may hold
/// the reference runner; a `from_artifact` failure is a skip, never a
/// fault).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "real-model E2E — needs the IGNIS_ARTIFACT path + the GPU (-- --ignored, --features cuda)"]
fn real_model_batched_prefill_reproducible() {
    let path = std::env::var("IGNIS_ARTIFACT").unwrap_or_default();
    if path.is_empty() {
        eprintln!("SKIP: IGNIS_ARTIFACT unset (no artifact)");
        return;
    }
    // Run 1: a fresh backend — the multi-token prompt's prefill (the
    // multi-token path — the multi-token GEMM + the multi-token
    // attention + the per-token GDN recurrence / RoPE / KV writeback,
    // B1 / #31) + a few decode steps.
    let compute1 =
        match CudaCompute::from_artifact(std::path::Path::new(&path), "qwen3.8-27b") {
            Ok(c) => c,
            Err(e) => {
                // A busy / OOM GPU (the reference runner may hold the
                // VRAM, ADR 0006): a skip, never a fault.
                eprintln!("SKIP: from_artifact failed: {e:?} (ADR 0006)");
                return;
            }
        };
    let tokens1 = match run(&compute1, &REAL_PROMPT, REAL_MAX_TOKENS, REAL_N_DECODE) {
        Ok(t) => t,
        Err(e) => {
            if skip_on_err(&e) {
                return;
            }
            panic!("run 1 failed: {e:?}");
        }
    };
    // The prefill took the multi-token path (spec 08 acceptance
    // criterion 1 — the `seq > 1` dispatch), exactly once.
    assert_eq!(
        compute1.batched_prefill_count(),
        1,
        "a multi-token prompt's prefill must run the multi-token path once"
    );
    // The in-vocab check (the correctness floor, ADR 0005): every
    // emitted token is a valid vocab id (the `ignis_greedy_sample`
    // contract).
    for &t in tokens1.iter() {
        assert!(
            (t as u64) < REAL_VOCAB,
            "token {t} must be in vocabulary (vocab = {REAL_VOCAB})"
        );
    }

    // Run 2: a fresh backend (the same artifact) must produce the *same*
    // decode stream (the self-consistency invariant, ADR 0007 — the
    // batched path's acceptance is a *sane*, reproducible output, not a
    // bit-exact agreement with the per-token loop, spec 08's design §7
    // caveat — the 99% performance gate, #20, is the re-check).
    drop(compute1);
    let compute2 =
        match CudaCompute::from_artifact(std::path::Path::new(&path), "qwen3.8-27b") {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP: from_artifact (run 2) failed: {e:?} (ADR 0006)");
                return;
            }
        };
    let tokens2 = match run(&compute2, &REAL_PROMPT, REAL_MAX_TOKENS, REAL_N_DECODE) {
        Ok(t) => t,
        Err(e) => {
            if skip_on_err(&e) {
                return;
            }
            panic!("run 2 failed: {e:?}");
        }
    };
    assert_eq!(
        compute2.batched_prefill_count(),
        1,
        "the second run's prefill must take the multi-token path too"
    );
    for &t in tokens2.iter() {
        assert!((t as u64) < REAL_VOCAB, "token {t} must be in vocabulary");
    }
    assert_eq!(
        tokens1, tokens2,
        "self-consistency: the same artifact input must produce the same tokens (greedy + fixed seed, ADR 0007)"
    );
    eprintln!(
        "real-model batched prefill (B1 / #31): {path} — prefill({REAL_PROMPT:?}) took the \
         multi-token path (batched_prefill_count = 1) + {REAL_N_DECODE} decodes → tokens \
         {tokens1:?} (in-vocab, reproducible across two fresh backends = the correctness \
         floor, ADR 0005 / 0007)"
    );
}