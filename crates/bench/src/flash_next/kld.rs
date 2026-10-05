//! The KLD of spec flash-next/04 acceptance 5 and 6: the converter's top-64
//! scorer (`tools/flash-next-converter/scoring.py` `kl_top64`, layout.md
//! §10) ported line for line, pooled per domain, so an engine's figure and
//! the converter's quantization-only figure are the same measurement.
//!
//! The arithmetic is f64 where the converter's is torch f32: the two agree
//! to about 1e-6 per position, far below any limit here.

use std::collections::BTreeMap;

use super::references::{DomainFigures, ReferenceSet, TopRow};

/// Below this tail mass the tail term is dropped, and the candidate's tail
/// is never taken smaller (`scoring.kl_top64`'s `eps`).
pub const TAIL_EPS: f64 = 1e-12;

/// KL(reference ‖ candidate) in nats from the reference's top-64
/// log-probabilities `ref_lp` and the candidate's log-probabilities at the
/// same ids `cand_lp`; the reference's remaining mass `R` is one bucket
/// scored against the candidate's mass outside those ids:
/// `Σ e^{r_i}(r_i − c_i) + R·(log R − log(1 − Σ e^{c_i}))`. A lower bound
/// of the exact KL.
pub fn kl_top64(ref_lp: &[f32], cand_lp: &[f64]) -> f64 {
    let (mut head, mut ref_mass, mut cand_mass) = (0.0f64, 0.0f64, 0.0f64);
    for (&r, &c) in ref_lp.iter().zip(cand_lp) {
        let r = r as f64;
        let p = r.exp();
        head += p * (r - c);
        ref_mass += p;
        cand_mass += c.exp();
    }
    let ref_tail = (1.0 - ref_mass).max(0.0);
    let cand_tail = (1.0 - cand_mass).max(TAIL_EPS);
    let tail = if ref_tail < TAIL_EPS { 0.0 } else { ref_tail * (ref_tail.max(TAIL_EPS).ln() - cand_tail.ln()) };
    head + tail
}

/// A BF16 value as f32 (exact).
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// One engine row: BF16 logits normalised by their own log-sum-exp, as the
/// converter normalises the checkpoint's (fp32 log-softmax of the BF16
/// head output).
#[derive(Debug)]
pub struct DenseRow<'a> {
    logits: &'a [u16],
    lse: f64,
}

impl<'a> DenseRow<'a> {
    /// Refuses a row holding a NaN or an infinity: a broken forward must
    /// not score as a distribution.
    pub fn new(logits: &'a [u16]) -> Result<Self, String> {
        let mut max = f32::NEG_INFINITY;
        for (id, &v) in logits.iter().enumerate() {
            let x = bf16_to_f32(v);
            if !x.is_finite() {
                return Err(format!("logit {id} is {x}"));
            }
            max = max.max(x);
        }
        let max = max as f64;
        let sum: f64 = logits.iter().map(|&v| (bf16_to_f32(v) as f64 - max).exp()).sum();
        Ok(DenseRow { logits, lse: max + sum.ln() })
    }

    pub fn vocab(&self) -> usize {
        self.logits.len()
    }

    pub fn logit(&self, id: u32) -> f32 {
        bf16_to_f32(self.logits[id as usize])
    }

    pub fn log_prob(&self, id: u32) -> f64 {
        self.logit(id) as f64 - self.lse
    }

    /// The most probable token, the lowest id on ties (the converter's
    /// `torch.argmax` and the engine's sampler both keep the first).
    pub fn argmax(&self) -> u32 {
        let mut best = 0usize;
        for (id, &v) in self.logits.iter().enumerate().skip(1) {
            if bf16_to_f32(v) > bf16_to_f32(self.logits[best]) {
                best = id;
            }
        }
        best as u32
    }
}

/// One position's measurements: the KL and whether the candidate's argmax
/// is the BF16 reference's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowScore {
    pub kl: f64,
    pub top1: bool,
}

/// Scores an engine row against the BF16 reference's row.
pub fn score_dense(reference: TopRow<'_>, row: &DenseRow<'_>) -> Result<RowScore, String> {
    let mut cand = [0.0f64; super::references::TOP];
    for (c, &id) in cand.iter_mut().zip(reference.ids) {
        if id as usize >= row.vocab() {
            return Err(format!("reference token {id} is outside the engine's {} columns", row.vocab()));
        }
        *c = row.log_prob(id as u32);
    }
    Ok(RowScore { kl: kl_top64(reference.lp, &cand[..reference.ids.len()]), top1: row.argmax() == reference.argmax() })
}

/// The quantized stream's own score at a row, from what the set stores:
/// the KL is known when the set stores the quantized log-probs at the BF16
/// ids, or when every BF16 top-64 id is also in the quantized top-64;
/// otherwise it is `None`. The top-1 is always known (`q_argmax`).
pub fn score_quantized(set: &ReferenceSet, row: usize) -> (Option<f64>, bool) {
    let reference = set.bf16(row);
    let top1 = set.quantized_argmax(row) == reference.argmax();
    let cand: Option<Vec<f64>> = match set.quantized_at_bf16(row) {
        Some(at) => Some(at.iter().map(|&c| c as f64).collect()),
        None => {
            let q = set.quantized(row);
            reference.ids.iter().map(|&id| q.log_prob(id as u32).map(f64::from)).collect()
        }
    };
    (cand.map(|c| kl_top64(reference.lp, &c)), top1)
}

/// One domain's pooled figures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DomainKld {
    /// Positions scored.
    pub positions: u64,
    /// The mean top-64 KL over them (nats).
    pub kld_top64: f64,
    /// The fraction whose argmax is the BF16 reference's.
    pub top1: f64,
}

/// Per-domain figures of one pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KldReport {
    pub domains: BTreeMap<String, DomainKld>,
}

/// Running sums per domain: every position weighs the same (the
/// converter's pooled mean, not a mean of window means).
#[derive(Debug, Default)]
pub struct KldTally {
    sums: BTreeMap<String, (u64, f64, u64)>,
}

impl KldTally {
    pub fn add(&mut self, domain: &str, kl: f64, top1: bool) {
        let entry = self.sums.entry(domain.to_string()).or_default();
        entry.0 += 1;
        entry.1 += kl;
        entry.2 += top1 as u64;
    }

    pub fn report(&self) -> KldReport {
        KldReport {
            domains: self
                .sums
                .iter()
                .map(|(d, &(n, kl, top1))| {
                    let n_f = n.max(1) as f64;
                    (d.clone(), DomainKld { positions: n, kld_top64: kl / n_f, top1: top1 as f64 / n_f })
                })
                .collect(),
        }
    }
}

/// Acceptance 5 and 6's limit for a domain whose quantization-only top-64
/// KLD is `quantized`: `max(1.1 × quantized, quantized + 0.01)`.
pub fn kld_limit(quantized: f64) -> f64 {
    (1.1 * quantized).max(quantized + 0.01)
}

/// One domain judged.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainVerdict {
    pub domain: String,
    pub positions: u64,
    pub engine: f64,
    /// The converter's quantization-only top-64 KLD for the domain.
    pub quantized: f64,
    pub limit: f64,
    pub pass: bool,
}

/// Judges every domain of an engine's report against the converter's
/// figures for the same windows; a domain on one side only is refused (the
/// two did not score the same set).
pub fn judge(engine: &KldReport, converter: &BTreeMap<String, DomainFigures>) -> Result<Vec<DomainVerdict>, String> {
    let engine_domains: Vec<&String> = engine.domains.keys().collect();
    let converter_domains: Vec<&String> = converter.keys().collect();
    if engine_domains != converter_domains {
        return Err(format!("the engine scored domains {engine_domains:?}, the converter recorded {converter_domains:?}"));
    }
    Ok(engine
        .domains
        .iter()
        .map(|(domain, e)| {
            let quantized = converter[domain].kld_top64;
            let limit = kld_limit(quantized);
            DomainVerdict {
                domain: domain.clone(),
                positions: e.positions,
                engine: e.kld_top64,
                quantized,
                limit,
                pass: e.kld_top64 <= limit,
            }
        })
        .collect())
}
