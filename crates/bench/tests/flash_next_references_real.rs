//! The converter's real reference recordings read by Flash-Next's
//! acceptance scorers (spec flash-next/04 acceptance 4-6 and 8), CPU only.
//! Machine-local: each test skips with a note while
//! `F:/ai/models/Qwen3.8-Flash-Next-ignis/references` does not exist (the
//! conversion writes it at its end); `IGNIS_FLASH_NEXT_DIR` and
//! `IGNIS_FLASH_NEXT_CHUNKS` point elsewhere.
//!
//! The quantized stream stands in for the engine. What the set stores
//! settles exactly: each domain's top-1, the 281 MMLU answers of both
//! streams with their McNemar, and the G1 column. The quantized top-64 KLD
//! is known exactly only where the set stores the quantized log-probs at
//! the BF16 ids (`q_lp_at_bf16_ids.f32`, absent from this conversion) or
//! where every BF16 top-64 id is also in the quantized top-64; the test
//! prints that subset's share and figure beside the converter's, and holds
//! them equal only when the share is whole.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ignis_artifact::packer::{ARTIFACT_FILE_NAME, sidecar_path};
use ignis_bench::flash_next::kld;
use ignis_bench::flash_next::mmlu::{self, MmluSet};
use ignis_bench::flash_next::references::{ConverterRecord, DomainFigures, ReferenceSet};
use ignis_bench::flash_next::{RowSink, SpanLogits, g1};
use ignis_bench::oracle::Fixture;

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const CHUNKS: &str = "F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03/real/ood/chunks.json";

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

/// The references directory, or `None` (with a note) while it is missing.
fn references() -> Option<PathBuf> {
    let dir = model_dir().join("references");
    if dir.join("test2048").join("manifest.json").exists() {
        Some(dir)
    } else {
        eprintln!("skip: {} holds no reference sets yet (written at the conversion's end)", dir.display());
        None
    }
}

/// `work/converter.json`, or the sidecar it is merged into once packed.
fn record() -> ConverterRecord {
    let work = model_dir().join("work").join("converter.json");
    let path = if work.exists() { work } else { sidecar_path(&model_dir().join(ARTIFACT_FILE_NAME)) };
    ConverterRecord::read(&path).unwrap()
}

/// The quantized stream scored at every position with a next token, per
/// domain, against the converter's figures for the same set.
fn reproduce(set: &ReferenceSet, figures: &BTreeMap<String, DomainFigures>) {
    // domain -> (positions, top-1 hits, exact positions, KL over them)
    let mut tally: BTreeMap<&str, (u64, u64, u64, f64)> = BTreeMap::new();
    for window in &set.windows {
        for p in 0..window.valid.saturating_sub(1) {
            let (kl, top1) = kld::score_quantized(set, window.first_position + p);
            let entry = tally.entry(window.kind.as_str()).or_default();
            entry.0 += 1;
            entry.1 += top1 as u64;
            if let Some(kl) = kl {
                entry.2 += 1;
                entry.3 += kl;
            }
        }
    }
    assert_eq!(
        tally.keys().copied().collect::<Vec<_>>(),
        figures.keys().map(String::as_str).collect::<Vec<_>>(),
        "{}: domains",
        set.dir.display()
    );
    eprintln!("{}:", set.dir.display());
    eprintln!("  domain  positions  top1 here / converter   exact share  KLD64 exact subset / converter all");
    for (domain, &(n, hits, exact, kl)) in &tally {
        let f = &figures[*domain];
        let top1 = hits as f64 / n as f64;
        let subset = kl / exact.max(1) as f64;
        eprintln!(
            "  {domain:7} {n:9}  {top1:.6} / {:.6}        {:6.2}%      {subset:.5} / {:.5}",
            f.top1,
            100.0 * exact as f64 / n as f64,
            f.kld_top64
        );
        // The converter's top-1 is an fp32 mean of 0/1: exact counts, one
        // fp32 rounding of the quotient.
        assert!((top1 - f.top1).abs() <= 1e-7, "{domain}: top-1 {top1} against the converter's {}", f.top1);
        if exact == n {
            assert!(
                (subset - f.kld_top64).abs() <= 1e-5 + 1e-5 * f.kld_top64,
                "{domain}: KLD {subset} against the converter's {}",
                f.kld_top64
            );
        }
    }
}

#[test]
fn the_stored_2048_and_8192_windows_reproduce_the_converters_quantized_top1_and_kld() {
    let Some(refs) = references() else { return };
    let record = record();
    reproduce(&ReferenceSet::read(&refs.join("test2048")).unwrap(), &record.kld.quantized);
    reproduce(&ReferenceSet::read(&refs.join("long8192")).unwrap(), &record.kld_long8192.q);
}

#[test]
fn the_stored_mmlu_answers_reproduce_the_converters_accuracies_and_mcnemar() {
    let Some(refs) = references() else { return };
    let record = record();
    let set = ReferenceSet::read(&refs.join("test2048")).unwrap();
    let chunks = std::env::var_os("IGNIS_FLASH_NEXT_CHUNKS").map_or_else(|| PathBuf::from(CHUNKS), PathBuf::from);
    let questions = MmluSet::from_chunks(&chunks, &set).unwrap();
    assert_eq!(questions.questions.len(), record.mmlu.n, "questions");
    let bf16 = questions.stored_answers(&set, false).unwrap();
    let quantized = questions.stored_answers(&set, true).unwrap();
    let report = mmlu::judge(&questions, &quantized, &bf16, &quantized).unwrap();
    eprintln!(
        "MMLU {} questions: BF16 {:.4} (converter {:?}), quantized {:.4} (converter {:?}), lost {} gained {} p {:.4}",
        report.n,
        report.bf16_accuracy,
        record.mmlu.bf16,
        report.accuracy,
        record.mmlu.quantized,
        report.vs_bf16.lost,
        report.vs_bf16.gained,
        report.vs_bf16.p
    );
    assert_eq!(Some(report.bf16_accuracy), record.mmlu.bf16);
    assert_eq!(Some(report.accuracy), record.mmlu.quantized);
    let converter = record.mmlu.mcnemar["quantized"];
    assert_eq!((report.vs_bf16.lost, report.vs_bf16.gained), (converter.lost, converter.gained));
    assert!((report.vs_bf16.p - converter.p).abs() <= 1e-12 * converter.p.max(1e-300));
}

/// The quantized stream of the canary set as an engine: each row puts all
/// its weight on the stored argmax.
struct StoredArgmax<'a> {
    set: &'a ReferenceSet,
    vocab: usize,
}

impl SpanLogits for StoredArgmax<'_> {
    fn vocab(&self) -> usize {
        self.vocab
    }

    fn span_logits(&mut self, tokens: &[u32], sink: &mut RowSink<'_>) -> Result<(), String> {
        let window = self
            .set
            .windows
            .iter()
            .find(|w| w.valid == tokens.len() + 1 && self.set.tokens(w).starts_with(tokens))
            .ok_or("no canary window starts with these tokens")?;
        for i in 0..tokens.len() {
            let mut row = vec![0u16; self.vocab];
            row[self.set.quantized_argmax(window.first_position + i) as usize] = 0x3f80;
            sink(i, &row)?;
        }
        Ok(())
    }
}

#[test]
fn the_g1_column_is_the_canary_sets_quantized_argmax_at_the_rows_the_run_reads() {
    let Some(refs) = references() else { return };
    let fixture = Fixture::read(&refs.join("g1_flash_next.json")).unwrap();
    let canary = ReferenceSet::read(&refs.join("canary")).unwrap();
    assert_eq!(fixture.reference.as_deref(), Some("quantized"));
    for prompt in &fixture.prompts {
        let window = canary.windows.iter().find(|w| w.source == prompt.id).unwrap();
        let prompt_tokens = prompt.prompt_token_ids.as_ref().unwrap();
        let fed: Vec<u32> = prompt_tokens.iter().chain(&prompt.token_ids).copied().collect();
        assert_eq!(canary.tokens(window), &fed[..], "{}: the canary window is the prompt and its tokens", prompt.id);
        let rows: Vec<u32> = (0..prompt.token_ids.len())
            .map(|i| canary.quantized_argmax(window.first_position + prompt_tokens.len() - 1 + i))
            .collect();
        assert_eq!(prompt.expected_argmax.as_ref(), Some(&rows), "{}", prompt.id);
    }
    let vocab = 1 + (0..canary.positions()).map(|r| canary.quantized_argmax(r) as usize).max().unwrap();
    let run = g1::run(&mut StoredArgmax { set: &canary, vocab }, &fixture).unwrap();
    let compared: usize = run.results.iter().map(|r| r.compared).sum();
    eprintln!("G1 through the seam on the stored quantized argmax: {compared} positions, overall {}", run.overall);
    assert_eq!(run.overall, 1.0);
    assert!(run.pass);
}

#[test]
fn the_converters_dry_run_record_and_g1_fixture_parse() {
    // The converter's real JSON shapes (a 2-layer dry run kept its record and
    // fixture): extra keys, nulls and the fixture's canary_source_model.
    let dir = std::path::Path::new("F:/ai/models/fn-dryrun-report");
    if !dir.join("run2").join("converter.json").exists() {
        eprintln!("skip: {} holds no dry-run record", dir.display());
        return;
    }
    let record = ConverterRecord::read(&dir.join("run2").join("converter.json")).unwrap();
    assert_eq!(record.status, "dry-run");
    assert_eq!(record.mmlu.n, 281);
    assert!(record.kld.quantized.contains_key("mmlu") && record.kld_long8192.q.contains_key("prose"));
    let fixture = Fixture::read(&dir.join("g1_flash_next.json")).unwrap();
    assert_eq!(fixture.reference.as_deref(), Some("quantized"));
    for prompt in &fixture.prompts {
        assert_eq!(prompt.expected_tokens().unwrap().len(), prompt.token_ids.len(), "{}", prompt.id);
    }
}
