//! Flash-Next's acceptance scorers (spec `flash-next/04`, acceptance 4-6
//! and 8): an engine's teacher-forced logits judged against the references
//! the converter recorded (`docs/specs/flash-next/layout.md` §10-11).
//!
//! - [`kld`]: the converter's top-64 KLD scorer, ported from
//!   `tools/flash-next-converter/scoring.py`, per domain, on the 2048-token
//!   test windows (acceptance 5) and the 8192-token windows (acceptance 6),
//!   with each domain's limit taken from the converter's own
//!   quantization-only figure.
//! - [`mmlu`]: the MMLU-Pro proxy (acceptance 8): the 281 questions' answers
//!   read at their rows of the same pass, paired against the BF16 and the
//!   quantized streams' answers.
//! - [`g1`]: the G1 run (acceptance 4) on the fixture's expected-argmax
//!   column, through [`crate::oracle::score_canary`].
//!
//! The engine is reached through one narrow seam, [`SpanLogits`]: the BF16
//! logits of every position of a teacher-forced token window. It is the
//! 27B's measurement readout (`ignis_core::step::prefill_program_span_logits`,
//! `crates/core/examples/span_logits.rs`, the 2026-09-24 KLD finding), so
//! the Flash-Next program implements it the same way and the scorers here
//! run on a mock until it does. [`run_acceptance`] runs all four on one
//! engine and [`Acceptance::render`] prints the verdicts.

pub mod g1;
pub mod kld;
pub mod mmlu;
pub mod references;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::oracle::Fixture;
use g1::G1Run;
use kld::{DenseRow, DomainVerdict, KldReport, KldTally};
use mmlu::{AcceptedFloorMiss, MmluReport, MmluSet};
use references::{ConverterRecord, ReferenceSet};

/// Where an engine hands its rows: `(first_row, rows)`.
pub type RowSink<'a> = dyn FnMut(usize, &[u16]) -> Result<(), String> + 'a;

/// The engine as the scorers see it: the BF16 logits of every position of a
/// token window prefilled teacher-forced on a fresh sequence.
pub trait SpanLogits {
    /// Columns of one logits row.
    fn vocab(&self) -> usize;

    /// Prefill `tokens` from position 0 on a fresh sequence and hand every
    /// position's logits to `sink`, in order: row `i` is the distribution
    /// after `tokens[..=i]`. `sink(first_row, rows)` receives whole
    /// little-endian BF16 rows (`rows.len()` a multiple of [`Self::vocab`]),
    /// as many at a time as the engine produces; an error from `sink` ends
    /// the window and is returned.
    fn span_logits(&mut self, tokens: &[u32], sink: &mut RowSink<'_>) -> Result<(), String>;
}

/// Feeds `tokens` to `engine` and hands `each` every chunk of rows with its
/// first row, after checking the engine delivers whole rows, in order, one
/// per token, no more and no fewer.
pub fn for_each_chunk(
    engine: &mut dyn SpanLogits,
    tokens: &[u32],
    mut each: impl FnMut(usize, &[u16]) -> Result<(), String>,
) -> Result<(), String> {
    let vocab = engine.vocab();
    if vocab == 0 {
        return Err("the engine reports a vocabulary of 0 columns".into());
    }
    let mut next = 0usize;
    engine.span_logits(tokens, &mut |first, rows| {
        if first != next {
            return Err(format!("the engine sent rows from {first}, expected row {next}"));
        }
        if rows.len() % vocab != 0 {
            return Err(format!("the engine sent {} values, not whole rows of {vocab}", rows.len()));
        }
        let count = rows.len() / vocab;
        if next + count > tokens.len() {
            return Err(format!("the engine sent rows up to {} for {} tokens", next + count, tokens.len()));
        }
        each(first, rows)?;
        next += count;
        Ok(())
    })?;
    if next != tokens.len() {
        return Err(format!("the engine sent {next} rows for {} tokens", tokens.len()));
    }
    Ok(())
}

/// What one pass of an engine over a reference set measured: the KLD per
/// domain and, when the pass carried the MMLU questions, the engine's
/// answer to each, in [`MmluSet::questions`] order.
#[derive(Debug, Clone, PartialEq)]
pub struct SetRun {
    pub kld: KldReport,
    pub answers: Vec<u8>,
}

/// One row's measurements.
struct RowOutcome {
    kl: f64,
    top1: bool,
    answer: Option<(usize, u8)>,
}

/// Runs `engine` over every window of `set` and scores it: each window is
/// fed its tokens up to the last scored position (`valid - 1` of them: the
/// converter scores the positions that have a next token), every row is
/// scored against the BF16 reference's top-64, and the rows of `questions`
/// (the MMLU set of this reference set, if any) give the engine's answers.
pub fn run_set(
    engine: &mut dyn SpanLogits,
    set: &ReferenceSet,
    questions: Option<&MmluSet>,
) -> Result<SetRun, String> {
    let vocab = engine.vocab();
    let mut tally = KldTally::default();
    let mut answers: Vec<Option<u8>> = vec![None; questions.map_or(0, |q| q.questions.len())];
    for window in set.windows.iter().filter(|w| w.valid >= 2) {
        let fed = &set.tokens(window)[..window.valid - 1];
        let asked: BTreeMap<usize, usize> = questions
            .map(|q| q.at_window(window.index))
            .unwrap_or_default();
        for_each_chunk(engine, fed, |first, rows| {
            let outcomes = score_chunk(set, window.first_position + first, first, rows, vocab, &asked, questions)?;
            for outcome in outcomes {
                tally.add(&window.kind, outcome.kl, outcome.top1);
                if let Some((question, answer)) = outcome.answer {
                    answers[question] = Some(answer);
                }
            }
            Ok(())
        })
        .map_err(|e| format!("{}: window {} ({}): {e}", set.dir.display(), window.index, window.kind))?;
    }
    let answers = answers
        .into_iter()
        .enumerate()
        .map(|(i, a)| a.ok_or_else(|| format!("MMLU question {i}: its row was never scored")))
        .collect::<Result<Vec<u8>, String>>()?;
    Ok(SetRun { kld: tally.report(), answers })
}

/// Scores a chunk of rows in parallel (the log-sum-exp over the whole
/// vocabulary is the cost): `first_global` is the chunk's first row in the
/// reference set, `first` its first position in the window.
fn score_chunk(
    set: &ReferenceSet,
    first_global: usize,
    first: usize,
    rows: &[u16],
    vocab: usize,
    asked: &BTreeMap<usize, usize>,
    questions: Option<&MmluSet>,
) -> Result<Vec<RowOutcome>, String> {
    let count = rows.len() / vocab;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 8);
    let per = count.div_ceil(threads).max(1);
    let score = |i: usize| -> Result<RowOutcome, String> {
        let position = first + i;
        let row = DenseRow::new(&rows[i * vocab..(i + 1) * vocab])
            .map_err(|e| format!("position {position}: {e}"))?;
        let reference = set.bf16(first_global + i);
        let score = kld::score_dense(reference, &row).map_err(|e| format!("position {position}: {e}"))?;
        let answer = match (asked.get(&position), questions) {
            (Some(&q), Some(set)) => {
                let question = &set.questions[q];
                Some((q, mmlu::answer_dense(&row, &set.letters, question.options)))
            }
            _ => None,
        };
        Ok(RowOutcome { kl: score.kl, top1: score.top1, answer })
    };
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..count)
            .step_by(per)
            .map(|start| {
                let score = &score;
                scope.spawn(move || (start..(start + per).min(count)).map(score).collect::<Result<Vec<_>, String>>())
            })
            .collect();
        let mut out = Vec::with_capacity(count);
        for handle in handles {
            out.extend(handle.join().map_err(|_| "a scoring thread panicked".to_string())??);
        }
        Ok(out)
    })
}

/// Acceptance 4-6 and 8 for one engine configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Acceptance {
    /// Acceptance 4: the G1 run on the expected-argmax column.
    pub g1: G1Run,
    /// Acceptance 5: the 2048-token test windows, per domain.
    pub kld_2048: Vec<DomainVerdict>,
    /// Acceptance 6: the 8192-token windows (the sparse QSA path), per domain.
    pub kld_8192: Vec<DomainVerdict>,
    /// Acceptance 8: the MMLU-Pro proxy, from the 2048-token pass.
    pub mmlu: MmluReport,
}

impl Acceptance {
    pub fn pass(&self) -> bool {
        self.pass_with(None)
    }

    /// [`Self::pass`], acceptance 8 judged by [`MmluReport::verdict`] with
    /// the owner's accepted floor miss, if any.
    pub fn pass_with(&self, accepted: Option<AcceptedFloorMiss>) -> bool {
        self.g1.pass
            && self.kld_2048.iter().chain(&self.kld_8192).all(|v| v.pass)
            && self.mmlu.verdict(accepted)
    }

    /// The verdicts as plain text: every domain, every G1 mismatch.
    pub fn render(&self) -> String {
        self.render_with(None)
    }

    /// [`Self::render`], with acceptance 8 as [`Self::pass_with`] judges it.
    pub fn render_with(&self, accepted: Option<AcceptedFloorMiss>) -> String {
        let verdict = |pass: bool| if pass { "pass" } else { "FAIL" };
        let mut out = String::new();
        let compared: usize = self.g1.results.iter().map(|r| r.compared).sum();
        let agree: usize = self.g1.results.iter().map(|r| r.agree).sum();
        let _ = writeln!(
            out,
            "G1 (acceptance 4): {agree}/{compared} = {:.2}% against the quantized reference's argmax, floor 95%: {}",
            100.0 * self.g1.overall,
            verdict(self.g1.pass)
        );
        for r in &self.g1.results {
            let _ = writeln!(out, "  {}: {}/{}", r.id, r.agree, r.compared);
            for m in &r.mismatches {
                let _ = writeln!(out, "    position {}: expected {}, engine {:?}", m.position, m.expected, m.predicted);
            }
        }
        for (title, verdicts) in [
            ("KLD on the 2048-token windows (acceptance 5)", &self.kld_2048),
            ("KLD on the 8192-token windows (acceptance 6)", &self.kld_8192),
        ] {
            let _ = writeln!(out, "{title}: engine / quantization-only / limit, top-64 scorer, nats");
            for v in verdicts {
                let _ = writeln!(
                    out,
                    "  {:8} {:7} positions  {:.5} / {:.5} / {:.5}  {}",
                    v.domain,
                    v.positions,
                    v.engine,
                    v.quantized,
                    v.limit,
                    verdict(v.pass)
                );
            }
        }
        let m = &self.mmlu;
        let floor = if m.pass {
            "pass"
        } else if m.floor_miss_accepted(accepted) {
            "missed by the artifact itself, accepted by the owner; judged against the quantized reference"
        } else {
            "FAIL"
        };
        let _ = writeln!(
            out,
            "MMLU-Pro proxy (acceptance 8): {:.2}% of {} (BF16 {:.2}%, quantized {:.2}%), floor 71%: {floor}",
            100.0 * m.accuracy,
            m.n,
            100.0 * m.bf16_accuracy,
            100.0 * m.quantized_accuracy,
        );
        let _ = writeln!(
            out,
            "  vs BF16: lost {} gained {} p {:.4}{}; vs quantized: lost {} gained {} p {:.4}{}",
            m.vs_bf16.lost,
            m.vs_bf16.gained,
            m.vs_bf16.p,
            if m.significantly_below_bf16 { " (significantly below)" } else { "" },
            m.vs_quantized.lost,
            m.vs_quantized.gained,
            m.vs_quantized.p,
            if m.significantly_below_quantized { " (significantly below)" } else { "" },
        );
        let _ = writeln!(out, "acceptance 8: {}", verdict(m.verdict(accepted)));
        out
    }
}

/// Runs acceptance 4-6 and 8 on `engine` from the converter's output:
/// `references` (`<out>/references`, with `test2048/`, `long8192/` and the
/// G1 fixture), its record (`work/converter.json` or the sidecar) and the
/// corpus file holding the MMLU marks (`chunks.json`).
pub fn run_acceptance(
    engine: &mut dyn SpanLogits,
    references: &Path,
    record: &ConverterRecord,
    chunks_json: &Path,
) -> Result<Acceptance, String> {
    let fixture = Fixture::read(&references.join("g1_flash_next.json"))?;
    let g1 = g1::run(engine, &fixture)?;
    let test = ReferenceSet::read(&references.join("test2048"))?;
    let questions = MmluSet::from_chunks(chunks_json, &test)?;
    if questions.questions.len() != record.mmlu.n {
        return Err(format!("{} MMLU questions here, the converter scored {}", questions.questions.len(), record.mmlu.n));
    }
    let run = run_set(engine, &test, Some(&questions))?;
    let kld_2048 = kld::judge(&run.kld, &record.kld.quantized)?;
    let mmlu = mmlu::judge(
        &questions,
        &run.answers,
        &questions.stored_answers(&test, false)?,
        &questions.stored_answers(&test, true)?,
    )?;
    drop(test);
    let long = ReferenceSet::read(&references.join("long8192"))?;
    let kld_8192 = kld::judge(&run_set(engine, &long, None)?.kld, &record.kld_long8192.q)?;
    Ok(Acceptance { g1, kld_2048, kld_8192, mmlu })
}
