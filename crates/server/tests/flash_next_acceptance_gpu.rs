//! Flash-Next's acceptance on the real artifact (spec flash-next/04,
//! GitHub #302): the engine's teacher-forced logits judged against the
//! converter's reference recordings by `ignis_bench::flash_next`.
//!
//! - On BF16 KV (the oracle format, acceptance 7): G1 against the
//!   quantized reference's argmax, at least 95% with every mismatch listed
//!   (acceptance 4); the top-64 KLD per domain on the 2048-token windows
//!   (acceptance 5) and the 8192-token windows, the sparse QSA path
//!   (acceptance 6), each within `max(1.1 q, q + 0.01)` of the converter's
//!   quantization-only figure; the MMLU-Pro proxy at least 71%, paired
//!   against the BF16 and quantized answers (acceptance 8) -- at this
//!   artifact, the owner's accepted miss below ([`ACCEPTED_MMLU_MISS`]).
//! - On hq-e8-2b, the serving default: the same per-domain KLD, reported
//!   beside the BF16 figures (acceptance 7), not judged.
//!
//! The engine is `ignis_core::flash_next::FlashNextEngine::span_logits`
//! (prefill chunks at the serving width, every row's head), wrapped here in
//! the scorers' `SpanLogits` seam. The two loads run one after the other in
//! one test: each pins the ~38 GB expert pool.
//!
//! Machine-local: the packed artifact and its `references/` in
//! `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or `IGNIS_FLASH_NEXT_DIR`), the
//! corpus file with the MMLU marks (`IGNIS_FLASH_NEXT_CHUNKS`). Explicit GPU
//! profile (ADR 0006): a missing input is a skip outside it, a failure
//! under it.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use ignis_artifact::packer::{ARTIFACT_FILE_NAME, sidecar_path};
use ignis_bench::flash_next::kld::{self, DomainVerdict};
use ignis_bench::flash_next::mmlu::AcceptedFloorMiss;
use ignis_bench::flash_next::references::{ConverterRecord, ReferenceSet};
use ignis_bench::flash_next::{RowSink, SpanLogits, run_acceptance, run_set};
use ignis_core::KvFormat;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine};
use ignis_core::gpu_profile;

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const CHUNKS: &str = "F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03/real/ood/chunks.json";

/// Acceptance 8's floor is missed by the artifact itself: its quantized
/// reference (the converter's torch run) answers 197 of the 281 questions,
/// 70.11%, three short of 71%. The owner accepted the 2.5-bit artifact
/// knowing that (2026-10-06: "ok a 2.5 bit se è solo MMLU" -- fine at 2.5
/// bits if only MMLU misses). So at this artifact the floor is reported as the artifact's accepted miss, and the
/// engine is held to the reference instead: not significantly below it
/// (McNemar, more lost than gained at p <= 0.05 fails). A re-converted
/// artifact with any other figure is held to the floor again.
const ACCEPTED_MMLU_MISS: Option<AcceptedFloorMiss> = Some(AcceptedFloorMiss { quantized_correct: 197 });

/// The scorers' view of the engine.
struct Engine(FlashNextEngine);

impl SpanLogits for Engine {
    fn vocab(&self) -> usize {
        self.0.vocab()
    }

    fn span_logits(&mut self, tokens: &[u32], sink: &mut RowSink<'_>) -> Result<(), String> {
        self.0.span_logits(tokens, sink)
    }
}

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

fn load(dir: &Path, kv_format: KvFormat) -> Option<Engine> {
    // 8192-token windows; no decode rounds, so no graphs.
    let options = EngineOptions { max_context_tokens: 8192, kv_format, capture_graphs: false, ..EngineOptions::default() };
    match FlashNextEngine::load(dir, options) {
        Ok(engine) => Some(Engine(engine)),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine on {kv_format:?} KV: {e}"));
            None
        }
    }
}

fn print_kld(title: &str, verdicts: &[DomainVerdict]) {
    eprintln!("{title}: engine / quantization-only / limit (top-64 scorer, nats)");
    for v in verdicts {
        eprintln!("  {:8} {:7} positions  {:.5} / {:.5} / {:.5}", v.domain, v.positions, v.engine, v.quantized, v.limit);
    }
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact and its references"]
fn acceptance_4_to_8_on_bf16_kv_and_the_hq_kld_beside_it() {
    let dir = model_dir();
    let references = dir.join("references");
    let chunks = std::env::var_os("IGNIS_FLASH_NEXT_CHUNKS").map_or_else(|| PathBuf::from(CHUNKS), PathBuf::from);
    for needed in [dir.join(ARTIFACT_FILE_NAME), references.join("test2048").join("manifest.json"), chunks.clone()] {
        if !needed.exists() {
            gpu_profile::skip_or_fail(&format!("{} does not exist", needed.display()));
            return;
        }
    }
    let work = dir.join("work").join("converter.json");
    let record_path = if work.exists() { work } else { sidecar_path(&dir.join(ARTIFACT_FILE_NAME)) };
    let record = ConverterRecord::read(&record_path).unwrap_or_else(|e| panic!("{e}"));

    let acceptance = {
        let Some(mut engine) = load(&dir, KvFormat::Bf16) else { return };
        run_acceptance(&mut engine, &references, &record, &chunks).unwrap_or_else(|e| panic!("BF16 KV: {e}"))
    };
    eprintln!("BF16 KV:\n{}", acceptance.render_with(ACCEPTED_MMLU_MISS));

    let (hq_2048, hq_8192) = {
        let Some(mut engine) = load(&dir, KvFormat::HqE8_2b) else { return };
        let mut judged = |set: &str, figures| {
            let set = ReferenceSet::read(&references.join(set)).unwrap_or_else(|e| panic!("{e}"));
            let run = run_set(&mut engine, &set, None).unwrap_or_else(|e| panic!("hq-e8-2b KV: {e}"));
            kld::judge(&run.kld, figures).unwrap_or_else(|e| panic!("{e}"))
        };
        (judged("test2048", &record.kld.quantized), judged("long8192", &record.kld_long8192.q))
    };
    print_kld("hq-e8-2b KV, 2048-token windows (acceptance 7, reported)", &hq_2048);
    print_kld("hq-e8-2b KV, 8192-token windows (acceptance 7, reported)", &hq_8192);

    assert!(
        acceptance.pass_with(ACCEPTED_MMLU_MISS),
        "Flash-Next acceptance 4-6 and 8 on BF16 KV:\n{}",
        acceptance.render_with(ACCEPTED_MMLU_MISS)
    );
}
