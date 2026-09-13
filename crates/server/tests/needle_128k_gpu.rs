//! The G4 needle-retrieval floor at 131,072 tokens against a live
//! `ignis-server` (GitHub #138).
//!
//! Spec 04's gate table makes needle retrieval at 64K/128K a correctness
//! floor reported pass/fail on its own. The 128K cell could not be reported
//! either way for a whole gate session: it failed identically against both
//! engines, every attempt, with `read SSE: error decoding response body` —
//! the harness's own transport giving up ~30 s into a prefill the engine was
//! still computing (`reqwest::blocking::Client::new()`'s undeclared 30 s
//! default). The reference's request log puts the two cells either side of
//! that line: 65,536 tokens answered at 15.0 s, 131,072 tokens cancelled at
//! 30.85 s having generated nothing.
//!
//! `crates/bench/tests/long_ttft.rs` holds the transport's own regression
//! test (in-process, no GPU). This file is the other half: the cell itself,
//! at its real length, against the real engine — the run that proves the
//! floor is measurable again rather than merely that the client no longer
//! times out on a mock.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, corpus, GPU or kernel error is a **skip**; under the
//! profile the same condition is a **hard failure**.

#![cfg(feature = "cuda")]

mod support;

use std::path::Path;
use std::time::{Duration, Instant};

use ignis_bench::client::HttpEndpoint;
use ignis_bench::g4::{self, NEEDLE_CONTEXT_128K, NEEDLE_CONTEXT_64K};
use ignis_bench::ttft::load_corpus;
use ignis_core::gpu_profile;
use ignis_server::runtime::EngineShape;
use support::live_server::LiveServer;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

/// The pre-tokenized bank the gate run cuts its haystacks from — the same
/// `--corpus` the failing `ignis-bench g4` invocation used, so the prompt
/// this test sends is the prompt that failed.
const CORPUS: &str = r"F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids";

/// The context the cells need to fit in (the gate profile's own).
const MAX_CONTEXT: u32 = 262_144;

/// The server-side deadline: generous, so that anything this test catches
/// is the measurement and not a second timeout of its own.
const SERVER_TIMEOUT: Duration = Duration::from_secs(900);

/// The engine shape the cells need: the gate profile's own context, and a
/// KV pool sized for it (the default pool is sized for the default context,
/// which a 131,072-token sequence does not fit in).
fn gate_shape() -> EngineShape {
    EngineShape {
        max_context: MAX_CONTEXT,
        kv_pool_bytes: ignis_runtime::auto_kv_pool_bytes(
            ignis_core::KvFormat::default(),
            MAX_CONTEXT,
        ),
        ..EngineShape::default()
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1 -- ~130K-token prefills, GitHub #138"]
fn the_needle_floor_is_measurable_at_both_gate_lengths() {
    let corpus_path = Path::new(CORPUS);
    if !corpus_path.exists() && gpu_profile::skip_or_fail(&format!("corpus absent: {CORPUS}")) {
        return;
    }
    let corpus = load_corpus(corpus_path).unwrap_or_else(|e| panic!("load the corpus: {e}"));
    let Some(server) = LiveServer::start(ARTIFACT, MODEL, gate_shape(), SERVER_TIMEOUT) else {
        return;
    };
    let ep = HttpEndpoint::new(&server.url);

    // Both gate lengths, in order: 64K passed even with the old deadline, so
    // it is the control — a failure there is a real engine-side regression,
    // not #138's transport bug.
    for context_tokens in [NEEDLE_CONTEXT_64K, NEEDLE_CONTEXT_128K] {
        let start = Instant::now();
        let result = g4::measure_needle_from_corpus(&ep, &server.frontend, &corpus, context_tokens);
        let elapsed = start.elapsed();
        eprintln!(
            "needle@{context_tokens}: retrieved={} in {:.1}s{}",
            result.retrieved,
            elapsed.as_secs_f64(),
            result.error.as_deref().map(|e| format!(" — {e}")).unwrap_or_default()
        );
        // The floor is a *reported* pass/fail (spec 04). A transport that
        // aborts reports neither, which is the state #138 left the 128K cell
        // in: whatever the answer is, the cell must produce one.
        assert!(
            result.error.is_none(),
            "the {context_tokens}-token cell must reach a verdict, not a transport failure: {:?}",
            result.error
        );
        assert!(
            result.retrieved,
            "the {context_tokens}-token needle must come back (the correctness floor)"
        );
    }
}
