//! B2 (kernel-abi 09, GitHub #32) — the CUDA-graph decode replay (the decode
//! hot path) over persistent device staging buffers (ADR 0008).
//!
//! The decode graph is captured at construction (the `CudaCompute::new`
//! synthetic path / the `from_artifact` production path) over the leaf's
//! fixed-address device staging buffers; each decode step H2D's the
//! per-step input (the token id), launches the graph (the whole decode DAG
//! runs on the fixed buffers, the `ignis_graph_launch`), and D2H's the
//! logits (ADR 0008). The graph is invariant after the construction-time
//! capture (no per-step capture, no node update — ADR 0008). A decode step
//! whose batch does not match the captured `GraphGeometry` runs the eager
//! sequence (ADR 0003); a busy/absent GPU (or a VRAM shortfall) leaves the
//! graph `None` (the eager fallback, ADR 0006 self-skip).
//!
//! The *representative* decode sequence captured here (the mechanism this
//! ticket delivers — the full per-layer stack + the host pointwise glue as
//! device kernels is the 99%-gate performance material, ADR 0005 / 0007,
//! ticket 20): embed -> GQA attention decode -> GDN step -> final RMSNorm ->
//! lm_head GEMV. The sequence is launched identically for the capture (the
//! graph's DAG) and for the eager reference run, so the replayed logits are
//! bit-identical to the eager logits (the kernel-abi 03 "replay == eager"
//! invariant, ADR 0007).
//!
//! **CPU-only pin (no GPU — always runs):** the eager-fallback wiring (a
//! `new_eager` backend has no decode graph — `uses_graph()` is `false`, the
//! `GraphGeometry` is `None`, the graph launch counter is 0 — the ADR 0003
//! / ADR 0006 eager-fallback wiring, no kernel calls, no GPU needed).
//!
//! **GPU-gated tests (a busy/absent GPU self-skips, ADR 0006 — a few KB of
//! VRAM runs even with a model loaded):**
//! 1. **The decode graph is captured at construction** (`CudaCompute::new`
//!    — the synthetic path — builds the decode graph; on a free GPU the
//!    `GraphGeometry` is set, batch 1, the representative decode geometry,
//!    ADR 0003; on a busy/absent GPU, or a VRAM shortfall, the capture
//!    self-skips (ADR 0006) and the backend falls back to the eager
//!    sequence, ADR 0003).
//! 2. **Graph replay ≡ the eager decode path** (the kernel-abi 03
//!    "replay == eager" invariant, ADR 0007: the replayed logits are
//!    *bit-identical* to the eager-reference run — the same representative
//!    decode sequence, the same staging buffers, the same weights, the same
//!    token, greedy + fixed seed, ADR 0007 — the "replay == eager"
//!    invariant applied to the *actual* staging-buffer decode graph, not
//!    today's empty-capture warm-up).
//! 3. **The hot path uses the graph** (the `decode_step` dispatch, B2 /
//!    #32, ADR 0008: a single-token (representative-batch) step runs the
//!    graph replay (the `graph_launch_count` increments), and a batch that
//!    does not match the captured `GraphGeometry` (a multi-token step)
//!    runs the eager sequence (the counter is untouched — the eager
//!    fallback engaged, ADR 0003)).
//!
//! Run with:
//!   `cargo test -p ignis-core --test decode_graph_gpu -- --ignored`
//!
//! Build precondition: links the kernel .lib (the FFI, ADR 0001). The GPU-
//! gated tests need a GPU (a no-GPU host self-skips, ADR 0006 — the eager
//! fallback, the CPU-only pin still runs).

use ignis_core::compute::{CudaCompute, GraphGeometry, ModelConfig, Weights};
use ignis_core::scheduler::{Compute, DecodeJob};
use ignis_core::types::{ComputeError, DecodeParams};

/// The decode graph's per-step input (the token id H2D'd into the fixed
/// input buffer, ADR 0008).
const TOK: i32 = 5;
/// The request / token ids (a fresh request per run).
const REQ: u64 = 1;
const REQ_B: u64 = 2;

/// The decode params (the `max_tokens` budget + the greedy sample, ADR 0007).
fn params() -> DecodeParams {
    DecodeParams {
        max_tokens: Some(4),
        ..DecodeParams::default()
    }
}

/// A single decode job (the `decode_step` dispatch, B2 / #32).
fn djob(request: u64) -> DecodeJob {
    DecodeJob {
        request,
        lane: 0,
        params: params(),
    }
}

/// A non-zero kernel rc (a CUDA error / busy GPU) is a skip, never a
/// failure (ADR 0006 — the GPU is occupied by the reference runner).
fn skip(e: &ComputeError) -> bool {
    match e {
        ComputeError::Kernel(rc) => {
            eprintln!(
                "SKIP: kernel call failed (rc={rc}; GPU busy/unavailable, ADR 0006)"
            );
            true
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// CPU-only pin (no GPU — always run): the eager-fallback wiring.
// ---------------------------------------------------------------------------

/// A `new_eager` backend (the eager path — no kernel-leaf startup check /
/// CUDA-graph capture, #26) has no decode graph (`uses_graph()` is `false`,
/// the `GraphGeometry` is `None`, the graph launch counter is 0 — the ADR
/// 0003 / ADR 0006 eager-fallback wiring). No GPU needed (the eager
/// construction is CPU-only; the wiring is observed through the accessors,
/// not a kernel call, ADR 0006).
#[test]
fn new_eager_backend_has_no_decode_graph() {
    let cfg = ModelConfig::synthetic();
    let weights = Weights::synthetic(&cfg, 42);
    let compute = CudaCompute::new_eager(cfg, weights);
    // The eager path (no decode graph — the ADR 0003 / ADR 0006 eager
    // fallback): `uses_graph()` is `false`, the `GraphGeometry` is `None`,
    // and the graph launch counter is 0 (the eager sequence never launched
    // the graph, ADR 0008).
    assert!(
        !compute.uses_graph(),
        "a `new_eager` backend must have no decode graph (the eager fallback, ADR 0003)"
    );
    assert!(
        compute.graph_geometry().is_none(),
        "a `new_eager` backend must have no GraphGeometry (the eager fallback, ADR 0003)"
    );
    assert_eq!(
        compute.graph_launch_count(),
        0,
        "a `new_eager` backend's graph launch counter must be 0 (the eager sequence, ADR 0003)"
    );
}

// ---------------------------------------------------------------------------
// GPU-gated tests (a busy/absent GPU self-skips, ADR 0006)
// ---------------------------------------------------------------------------

/// The decode graph is captured at construction (B2 / #32, ADR 0008):
/// `CudaCompute::new` (the synthetic path) builds the decode graph over the
/// leaf's fixed-address device staging buffers (the representative decode
/// geometry, batch 1, the ADR 0003 eager-fallback geometry). On a free GPU
/// (a few KB of VRAM — runs even with a model loaded, the ADR 0006 nuance)
/// the construction-time capture succeeds + the `GraphGeometry` is set; on
/// a busy/absent GPU (or a VRAM shortfall) the capture self-skips (ADR
/// 0006) and the backend falls back to the eager sequence (ADR 0003).
#[test]
#[ignore = "GPU launch test — the decode graph needs a free GPU (ADR 0006 self-skip on a busy GPU): -- --ignored"]
fn decode_graph_captured_at_construction() {
    let cfg = ModelConfig::synthetic();
    let weights = Weights::synthetic(&cfg, 42);
    // The synthetic backend (the decode graph is built at construction,
    // B2 / #32, ADR 0008). A no-GPU host (the `ignis_decode_graph_new`
    // self-skip, ADR 0006) leaves the graph `None` (the eager fallback) —
    // `uses_graph()` is `false` (the test self-skips).
    let compute = CudaCompute::new(cfg, weights);
    if !compute.uses_graph() {
        eprintln!(
            "SKIP: no decode graph (a no-GPU host / a VRAM shortfall, ADR 0006)"
        );
        return;
    }
    // The decode graph is captured at construction (the startup-time
    // capture, ADR 0008) — the `GraphGeometry` is set (batch 1, the
    // representative decode geometry, the ADR 0003 eager-fallback
    // geometry).
    assert_eq!(
        compute.graph_geometry(),
        Some(GraphGeometry { batch: 1 }),
        "the decode graph's GraphGeometry must be set (batch 1, the representative decode geometry, ADR 0008)"
    );
    // The graph launch counter starts at 0 (no decode step has run yet —
    // the decode graph is captured at construction, not launched, ADR 0008).
    assert_eq!(
        compute.graph_launch_count(),
        0,
        "the graph launch counter must be 0 before any decode step (the capture is at construction, ADR 0008)"
    );
}

/// Graph replay ≡ the eager decode path (B2 / #32, the kernel-abi 03
/// "replay == eager" invariant, ADR 0007): the decode graph's replay (the
/// `ignis_graph_launch` over the captured decode DAG, ADR 0008) produces
/// logits that are **bit-identical** to the eager-reference run (the same
/// representative decode sequence, the same staging buffers, the same
/// weights, the same token — greedy + fixed seed, ADR 0007). This is the
/// kernel-abi 03 "replay == eager" invariant applied to the *actual*
/// staging-buffer decode graph (today's empty-capture warm-up is a no-op;
/// this is the real decode graph's logits).
#[test]
#[ignore = "GPU launch test — the decode graph's replay + eager reference run on the GPU (ADR 0006 self-skip on a busy GPU): -- --ignored"]
fn decode_graph_replay_eager_bit_exact() {
    let cfg = ModelConfig::synthetic();
    let weights = Weights::synthetic(&cfg, 42);
    // The synthetic backend (the decode graph is built at construction,
    // B2 / #32, ADR 0008). A no-GPU host (the self-skip, ADR 0006) has no
    // decode graph (the test self-skips).
    let compute = CudaCompute::new(cfg, weights);
    if !compute.uses_graph() {
        eprintln!("SKIP: no decode graph (a no-GPU host / a VRAM shortfall, ADR 0006)");
        return;
    }
    // The replay (the `ignis_graph_launch`, ADR 0008): H2D the token id,
    // launch the graph (the whole decode DAG runs on the fixed staging
    // buffers), D2H the logits.
    let logits_replay = match compute.graph_logits_replay(TOK) {
        Ok(l) => l,
        Err(e) => {
            if skip(&e) {
                return;
            }
            panic!("the graph replay failed: {e:?}");
        }
    };
    // The eager reference (the same representative decode sequence, no
    // graph — the kernel-abi 03 "replay == eager" invariant, ADR 0007):
    // H2D the token id, run the sequence eagerly, D2H the logits.
    let logits_eager = match compute.graph_logits_eager(TOK) {
        Ok(l) => l,
        Err(e) => {
            if skip(&e) {
                return;
            }
            panic!("the eager reference failed: {e:?}");
        }
    };
    // The kernel-abi 03 "replay == eager" invariant (ADR 0007): the
    // replayed logits are bit-identical to the eager-reference logits (the
    // same representative decode sequence, the same staging buffers, the
    // same weights, the same token — greedy + fixed seed, ADR 0007).
    assert_eq!(
        logits_replay, logits_eager,
        "the graph replay's logits must be bit-identical to the eager reference's logits (the kernel-abi 03 \"replay == eager\" invariant, ADR 0007 / ADR 0008)"
    );
    eprintln!(
        "decode graph replay == eager (the kernel-abi 03 invariant, ADR 0007): {vocab} logits, bit-exact",
        vocab = logits_replay.len()
    );
}

/// The hot path uses the graph (B2 / #32, ADR 0008): a single-token
/// (representative-batch) decode step runs the graph replay (the
/// `ignis_graph_launch`, the `graph_launch_count` increments), and a batch
/// that does not match the captured `GraphGeometry` (a multi-token step)
/// runs the eager sequence (the counter is untouched — the eager fallback
/// engaged, ADR 0003).
#[test]
#[ignore = "GPU launch test — the decode graph's hot path + the eager fallback run on the GPU (ADR 0006 self-skip on a busy GPU): -- --ignored"]
fn decode_graph_hot_path_and_eager_fallback() {
    let cfg = ModelConfig::synthetic();
    let weights = Weights::synthetic(&cfg, 42);
    // The synthetic backend (the decode graph is built at construction,
    // B2 / #32, ADR 0008). A no-GPU host (the self-skip, ADR 0006) has no
    // decode graph (the test self-skips — the eager fallback is the
    // correctness floor, ADR 0005).
    let compute = CudaCompute::new(cfg.clone(), weights);
    if !compute.uses_graph() {
        eprintln!("SKIP: no decode graph (a no-GPU host / a VRAM shortfall, ADR 0006)");
        return;
    }
    // A single-token (representative-batch) decode step (the hot path,
    // ADR 0008): runs the graph replay (the `ignis_graph_launch`, the
    // `graph_launch_count` increments once).
    match compute.decode_step(std::slice::from_ref(&djob(REQ))) {
        Ok(out) => {
            assert_eq!(out.len(), 1, "a single decode job returns one result");
            match out.into_iter().next() {
                Some(Some(t)) => {
                    assert!((t as u64) < cfg.vocab, "token {t} must be in vocabulary");
                }
                Some(None) => {} // a soft-stop (a fresh request's first step is not a soft-stop, but a later one may be)
                None => panic!("a single decode job must return a result"),
            }
        }
        Err(e) => {
            if skip(&e) {
                return;
            }
            panic!("the single-token step failed: {e:?}");
        }
    }
    assert_eq!(
        compute.graph_launch_count(),
        1,
        "a single-token decode step must run the graph replay once (the hot path, ADR 0008)"
    );
    // A multi-token decode step (the batch does not match the captured
    // `GraphGeometry` — batch 1 — the ADR 0003 eager fallback): the eager
    // sequence runs (the counter is *untouched*, the eager fallback
    // engaged, ADR 0003).
    let jobs = vec![djob(REQ), djob(REQ_B)];
    let results = match compute.decode_step(&jobs) {
        Ok(out) => out,
        Err(e) => {
            if skip(&e) {
                return;
            }
            panic!("the multi-token step failed: {e:?}");
        }
    };
    assert_eq!(
        compute.graph_launch_count(),
        1,
        "a multi-token decode step must run the eager sequence (the graph launch counter is untouched — the eager fallback, ADR 0003)"
    );
    // The eager fallback (the multi-token step) still produces in-vocab
    // tokens (the correctness floor, ADR 0005 — the greedy sample's
    // contract).
    assert_eq!(results.len(), 2, "a two-job decode step returns two results");
    for result in &results {
        match result {
            Some(t) => assert!((*t as u64) < cfg.vocab, "token {t} must be in vocabulary"),
            None => {} // a soft-stop (a fresh request's first step is not a soft-stop, but a later one may be)
        }
    }
}