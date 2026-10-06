//! The MMLU-Pro proxy of spec flash-next/04 acceptance 8: the 281
//! held-out questions of the converter's test windows, each answered at the
//! row before its answer letter by the option letter the distribution
//! favours, and paired against the BF16 and quantized streams' answers
//! (McNemar, `scoring.py` `mcnemar_p` / `paired`).
//!
//! The questions are not in the reference set: their marks (position, gold
//! letter, option count) live in the corpus file `chunks.json` (the
//! compression study's `ood/`, pinned by sha256 in
//! `tools/flash-next-converter/corpus_manifest.json`), and each chunk is
//! matched to its reference window by token identity.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::Deserialize;

use super::kld::DenseRow;
use super::references::{ReferenceSet, TopRow};

/// Acceptance 8's floor.
pub const MMLU_FLOOR: f64 = 0.71;

/// An artifact the owner accepted although its own quantized reference
/// misses [`MMLU_FLOOR`], named by how many questions that reference answers
/// right: the figure the decision was taken on. At that artifact acceptance 8
/// holds the engine to the reference instead of the floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptedFloorMiss {
    pub quantized_correct: usize,
}

/// The reference-set `source` of the windows that carry questions.
pub const QUESTION_SOURCE: &str = "chunks.json";

/// One question: answered at `position` of reference window `window` (the
/// distribution of the token after it is the answer letter's).
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub window: usize,
    pub position: usize,
    pub gold: u8,
    pub options: u8,
    pub category: String,
}

/// The questions of a reference set and the option letters' token ids
/// (`A`, `B`, ... in order).
#[derive(Debug, Clone, PartialEq)]
pub struct MmluSet {
    pub letters: Vec<u32>,
    pub questions: Vec<Question>,
}

#[derive(Deserialize)]
struct Chunks {
    chunks: Vec<Chunk>,
    letter_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Chunk {
    ids: Vec<u32>,
    test: bool,
    valid: usize,
    #[serde(default)]
    mmlu: Vec<Mark>,
}

/// `[position, gold, options, category]`; the category is optional, as the
/// converter reads it.
#[derive(Deserialize)]
#[serde(untagged)]
enum Mark {
    Named(usize, u8, u8, String),
    Bare(usize, u8, u8),
}

impl MmluSet {
    /// The questions of `set`'s windows from `chunks.json`. A window from
    /// that file whose tokens match no held-out chunk is refused. A mark the
    /// converter never scored (no next token after it) is left out, as the
    /// converter leaves it out.
    pub fn from_chunks(chunks_json: &Path, set: &ReferenceSet) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(chunks_json).map_err(|e| format!("read {}: {e}", chunks_json.display()))?;
        let file: Chunks = serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", chunks_json.display()))?;
        let held_out: HashMap<&[u32], &Chunk> =
            file.chunks.iter().filter(|c| c.test).map(|c| (c.ids.as_slice(), c)).collect();
        let mut questions = Vec::new();
        for window in set.windows.iter().filter(|w| w.source == QUESTION_SOURCE) {
            let chunk = held_out.get(set.tokens(window)).ok_or_else(|| {
                format!("window {}: no held-out chunk of {} has its tokens", window.index, chunks_json.display())
            })?;
            if chunk.valid != window.valid {
                return Err(format!("window {}: valid {} here, {} in the chunk", window.index, window.valid, chunk.valid));
            }
            for mark in &chunk.mmlu {
                let (position, gold, options, category) = match mark {
                    Mark::Named(p, g, o, c) => (*p, *g, *o, c.clone()),
                    Mark::Bare(p, g, o) => (*p, *g, *o, String::new()),
                };
                if gold >= options || options as usize > file.letter_ids.len() {
                    return Err(format!(
                        "window {} position {position}: gold {gold} of {options} options, {} letters",
                        window.index,
                        file.letter_ids.len()
                    ));
                }
                if position + 1 < window.valid {
                    questions.push(Question { window: window.index, position, gold, options, category });
                }
            }
        }
        Ok(MmluSet { letters: file.letter_ids, questions })
    }

    /// The questions asked in window `window`: position → question index.
    pub fn at_window(&self, window: usize) -> BTreeMap<usize, usize> {
        self.questions.iter().enumerate().filter(|(_, q)| q.window == window).map(|(i, q)| (q.position, i)).collect()
    }

    /// Each question's answer by a stored stream of `set` (`quantized`
    /// false: the BF16 stream). Refused when a stored top-64 cannot settle
    /// one ([`answer_top64`]).
    pub fn stored_answers(&self, set: &ReferenceSet, quantized: bool) -> Result<Vec<u8>, String> {
        self.questions
            .iter()
            .map(|q| {
                let row = set.windows[q.window].first_position + q.position;
                let top = if quantized { set.quantized(row) } else { set.bf16(row) };
                answer_top64(top, &self.letters, q.options).ok_or_else(|| {
                    format!(
                        "window {} position {}: the stored {} top-64 cannot settle the answer",
                        q.window,
                        q.position,
                        if quantized { "quantized" } else { "BF16" }
                    )
                })
            })
            .collect()
    }

    /// Whether each answer is the gold letter.
    pub fn correct(&self, answers: &[u8]) -> Vec<bool> {
        self.questions.iter().zip(answers).map(|(q, &a)| a == q.gold).collect()
    }
}

/// The option an engine row picks: the largest logit among the first
/// `options` letters, the first letter on ties (`torch.argmax` over the
/// letters' log-probs, as the converter picks).
pub fn answer_dense(row: &DenseRow<'_>, letters: &[u32], options: u8) -> u8 {
    let mut best = 0usize;
    for k in 1..options as usize {
        if row.logit(letters[k]) > row.logit(letters[best]) {
            best = k;
        }
    }
    best as u8
}

/// The option a stored top-64 row picks, when the row settles it: some
/// option letter is among the 64, and the best of them is strictly above
/// the 64th entry or no option letter is missing (a missing letter is at
/// most the 64th entry's log-prob, so it could only tie).
pub fn answer_top64(row: TopRow<'_>, letters: &[u32], options: u8) -> Option<u8> {
    let floor = *row.lp.last()?;
    let mut best: Option<(usize, f32)> = None;
    let mut missing = false;
    for (k, &letter) in letters.iter().take(options as usize).enumerate() {
        match row.log_prob(letter) {
            Some(lp) if best.is_none_or(|(_, b)| lp > b) => best = Some((k, lp)),
            Some(_) => {}
            None => missing = true,
        }
    }
    match best {
        Some((k, lp)) if !(missing && lp <= floor) => Some(k as u8),
        _ => None,
    }
}

/// Exact two-sided binomial test on the discordant pairs (`scoring.py`
/// `mcnemar_p`).
pub fn mcnemar_p(lost: usize, gained: usize) -> f64 {
    let (m, k) = (lost + gained, lost.min(gained));
    if m == 0 {
        return 1.0;
    }
    let (mut sum, mut comb) = (0.0f64, 1.0f64);
    for i in 0..=k {
        sum += comb;
        comb = comb * (m - i) as f64 / (i + 1) as f64;
    }
    (2.0 * sum / 2f64.powi(m as i32)).min(1.0)
}

/// A variant paired against a reference on the same questions: the ones
/// the reference got right and the variant wrong (`lost`), the reverse
/// (`gained`), and McNemar's p.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct Paired {
    pub lost: usize,
    pub gained: usize,
    pub p: f64,
}

/// `scoring.py` `paired`: the two lists cover the same questions in order.
pub fn paired(reference_ok: &[bool], variant_ok: &[bool]) -> Paired {
    let lost = reference_ok.iter().zip(variant_ok).filter(|&(&r, &v)| r && !v).count();
    let gained = reference_ok.iter().zip(variant_ok).filter(|&(&r, &v)| v && !r).count();
    Paired { lost, gained, p: mcnemar_p(lost, gained) }
}

fn accuracy(ok: &[bool]) -> f64 {
    ok.iter().filter(|&&b| b).count() as f64 / ok.len().max(1) as f64
}

/// Acceptance 8 judged.
#[derive(Debug, Clone, PartialEq)]
pub struct MmluReport {
    pub n: usize,
    pub accuracy: f64,
    pub bf16_accuracy: f64,
    pub quantized_accuracy: f64,
    /// The engine against the BF16 checkpoint's answers.
    pub vs_bf16: Paired,
    /// The engine against the quantized reference's (spec 01's torch run).
    pub vs_quantized: Paired,
    /// The accuracy is at least [`MMLU_FLOOR`].
    pub pass: bool,
    /// More lost than gained against BF16 at p ≤ 0.05 (the converter's own
    /// second condition, reported beside the floor).
    pub significantly_below_bf16: bool,
    /// More lost than gained against the quantized reference at p ≤ 0.05:
    /// the engine, not the artifact, costing answers.
    pub significantly_below_quantized: bool,
}

impl MmluReport {
    /// The questions the quantized reference answers right.
    pub fn quantized_correct(&self) -> usize {
        (self.quantized_accuracy * self.n as f64).round() as usize
    }

    /// Whether `accepted` names this artifact's own miss of the floor.
    pub fn floor_miss_accepted(&self, accepted: Option<AcceptedFloorMiss>) -> bool {
        accepted.is_some_and(|a| a.quantized_correct == self.quantized_correct())
            && self.quantized_accuracy < MMLU_FLOOR
    }

    /// Acceptance 8: the floor, or -- at an artifact whose miss `accepted`
    /// names -- not significantly below the quantized reference. Any other
    /// artifact, one with a different reference figure included, is held
    /// to the floor.
    pub fn verdict(&self, accepted: Option<AcceptedFloorMiss>) -> bool {
        if self.floor_miss_accepted(accepted) {
            !self.significantly_below_quantized
        } else {
            self.pass
        }
    }
}

/// Judges an engine's answers against the stored streams' answers, all in
/// [`MmluSet::questions`] order.
pub fn judge(set: &MmluSet, engine: &[u8], bf16: &[u8], quantized: &[u8]) -> Result<MmluReport, String> {
    let n = set.questions.len();
    if engine.len() != n || bf16.len() != n || quantized.len() != n {
        return Err(format!(
            "{n} questions, {} engine / {} BF16 / {} quantized answers",
            engine.len(),
            bf16.len(),
            quantized.len()
        ));
    }
    let (engine_ok, bf16_ok, quantized_ok) = (set.correct(engine), set.correct(bf16), set.correct(quantized));
    let vs_bf16 = paired(&bf16_ok, &engine_ok);
    let vs_quantized = paired(&quantized_ok, &engine_ok);
    let accuracy = accuracy(&engine_ok);
    Ok(MmluReport {
        n,
        accuracy,
        bf16_accuracy: self::accuracy(&bf16_ok),
        quantized_accuracy: self::accuracy(&quantized_ok),
        vs_bf16,
        vs_quantized,
        pass: accuracy >= MMLU_FLOOR,
        significantly_below_bf16: vs_bf16.lost > vs_bf16.gained && vs_bf16.p <= 0.05,
        significantly_below_quantized: vs_quantized.lost > vs_quantized.gained && vs_quantized.p <= 0.05,
    })
}
