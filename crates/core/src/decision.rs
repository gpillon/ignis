//! Decisions: the **readout** and the **answer alphabet** (ADR 0034,
//! GitHub #237).
//!
//! A decision is one question put to the model whose answer is a
//! distribution over a declared set of options rather than generated text.
//! Each option is named by an **answer token** — a single vocabulary entry —
//! and the answer is that token's logit at the prompt's last position. No
//! token is ever sampled, so a decision costs one prefill and generates
//! nothing.
//!
//! Two things live here, and only these two: what a readout *is*
//! ([`Readout`], the value that crosses the `Compute` seam) and where the
//! labels come from ([`AnswerAlphabet`], computed from the loaded
//! tokenizer). The request kind that asks for one is GitHub #238's, and the
//! endpoint that shapes the answer is GitHub #239's.

use std::collections::BTreeSet;

use crate::types::TokenId;

// ---------------------------------------------------------------------------
// The readout
// ---------------------------------------------------------------------------

/// What one **readout** carries back across the `Compute` seam: the logits
/// of the answer tokens the job named, plus the two summaries of everything
/// it did *not* name.
///
/// The full-vocabulary buffer the kernel writes is one f32 per column —
/// 248,320 of them on the 27B, nearly a megabyte per decision — and it never
/// crosses the seam. The gather happens on the adapter's side of it
/// ([`Readout::gather`]), which is why this type holds the answers alone.
///
/// `full_log_sum_exp` and `full_argmax` are what the answers are judged
/// against. Without the first there is no **answer mass**, and a restricted
/// softmax with no answer mass is renormalized noise however plausible its
/// winner looks; without the second there is no way to say whether the
/// model's own unrestricted choice was a declared option at all.
#[derive(Debug, Clone, PartialEq)]
pub struct Readout {
    /// One logit per answer token, in the order the job named them.
    pub logits: Vec<f32>,
    /// `logsumexp` over the *whole* vocabulary at this position, in f64.
    pub full_log_sum_exp: f64,
    /// The unrestricted winner: the whole vocabulary's argmax, which may
    /// well be a token no option named.
    pub full_argmax: TokenId,
}

impl Readout {
    /// Read `answers` out of one position's full-vocabulary `logits`, and
    /// summarize the rest.
    ///
    /// This is the gather the `Compute` seam exists to keep on the GPU side:
    /// `logits` is the kernel's own buffer and is dropped as soon as this
    /// returns. An id past the end of the vocabulary reads
    /// `f32::NEG_INFINITY` rather than panicking or aliasing another
    /// column — a readout runs on the prefill path, where a panic would
    /// take the round down with it. An empty `logits` (a leaf claiming a
    /// zero-wide head, which no real one does) reports token 0 and an
    /// infinite log-sum-exp, so the answer mass is 0: nothing was read.
    pub fn gather(logits: &[f32], answers: &[TokenId]) -> Self {
        let gathered: Vec<f32> = answers
            .iter()
            .map(|&id| logits.get(id as usize).copied().unwrap_or(f32::NEG_INFINITY))
            .collect();
        Self {
            logits: gathered,
            full_log_sum_exp: log_sum_exp(logits),
            full_argmax: argmax(logits).unwrap_or(0) as TokenId,
        }
    }

    /// **Answer mass**: how much of the whole next-token distribution the
    /// answer tokens hold, `exp(logsumexp(answers) - logsumexp(all))`.
    ///
    /// Clamped into `[0, 1]`, which it is by construction and can leave by a
    /// few ulps once the answers hold essentially everything — a readout
    /// that reports a probability of 1.0000001 would be a lie about the
    /// arithmetic rather than about the model.
    pub fn answer_mass(&self) -> f64 {
        if self.logits.is_empty() {
            return 0.0;
        }
        let answers = log_sum_exp(&self.logits);
        if !answers.is_finite() {
            // Every answer token read `-inf`: the options hold nothing.
            return 0.0;
        }
        (answers - self.full_log_sum_exp).exp().clamp(0.0, 1.0)
    }

    /// The restricted softmax: one probability per answer token, summing to
    /// one. This is the distribution a decision answers with — the mass it
    /// threw away to get there is [`Readout::answer_mass`], reported beside
    /// it rather than folded into it.
    pub fn probabilities(&self) -> Vec<f64> {
        let total = log_sum_exp(&self.logits);
        if !total.is_finite() {
            return vec![0.0; self.logits.len()];
        }
        self.logits
            .iter()
            .map(|&logit| (f64::from(logit) - total).exp())
            .collect()
    }

    /// The winning slot: the index of the highest answer logit, or `None`
    /// when no answer token was named at all.
    pub fn winner(&self) -> Option<usize> {
        argmax(&self.logits)
    }
}

/// `logsumexp` in f64 over f32 logits, max-subtracted so a vocabulary-wide
/// sum cannot overflow the exponent.
///
/// Public because it *is* the definition of **answer mass** — the
/// denominator a backend has to produce and a caller has to be able to
/// reproduce — rather than an implementation detail of one caller.
pub fn log_sum_exp(values: &[f32]) -> f64 {
    let maximum = values
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &v| acc.max(f64::from(v)));
    if !maximum.is_finite() {
        return maximum;
    }
    let sum: f64 = values.iter().map(|&v| (f64::from(v) - maximum).exp()).sum();
    maximum + sum.ln()
}

/// The index of the largest value, or `None` when there is none. Ties go to
/// the first, matching the kernel's own argmax.
pub(crate) fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .fold(None::<(usize, f32)>, |best, (index, &value)| match best {
            Some((_, high)) if !(value > high) => best,
            _ => Some((index, value)),
        })
        .map(|(index, _)| index)
}

// ---------------------------------------------------------------------------
// The answer alphabet
// ---------------------------------------------------------------------------

/// What the answer alphabet needs of a tokenizer, and nothing more.
///
/// A trait rather than `ignis_artifact::Tokenizer` directly because the rule
/// the alphabet enforces is *about* tokenization: "is this label one token
/// that decodes back to itself?" A test that cannot state a tokenizer which
/// splits a given label cannot check the rule at all, and the one tokenizer
/// that really splits `BQ` is a 27B artifact nobody has under CI.
///
/// Both methods return `Option` rather than a crate error: the alphabet's
/// only question is whether a label works, and every way of failing is the
/// same answer — it does not.
pub trait LabelTokenizer {
    /// The token ids `text` encodes to, or `None` if it cannot be encoded.
    fn encode(&self, text: &str) -> Option<Vec<TokenId>>;
    /// The text `ids` decode to, or `None` if they cannot be decoded.
    fn decode(&self, ids: &[TokenId]) -> Option<String>;
}

impl LabelTokenizer for ignis_artifact::Tokenizer {
    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        ignis_artifact::Tokenizer::encode(self, text).ok()
    }

    fn decode(&self, ids: &[TokenId]) -> Option<String> {
        ignis_artifact::Tokenizer::decode(self, ids).ok()
    }
}

/// One option's label and the single token that stands for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerToken {
    /// The label as it is written into the prompt.
    pub label: String,
    /// The vocabulary entry whose logit the readout reads.
    pub id: TokenId,
}

/// The ordered pool answer tokens are drawn from: `A`-`Z`, `a`-`z`, `0`-`9`,
/// then the uppercase bigrams `AA`-`ZZ`.
///
/// **Computed from the loaded tokenizer, never compiled in.** A label is
/// admitted only if it encodes to exactly one token that decodes back to
/// itself. That rejects 114 of the 676 uppercase bigrams in the 27B's
/// tokenizer — `BQ` among them — and the rejection is a correctness rule,
/// not a tidiness one: `BQ` encodes to `B` followed by `Q`, so a readout
/// that admitted it would read the logit of `B`, which is some *other*
/// option's answer token. Two slots naming one token cannot be told apart
/// without a second forward pass, which is the one thing a zero-decode
/// decision must not need.
///
/// The obvious alphabet is the one that does not work: numbering options
/// `1`..`255` reads the logit of `1` for options 1, 10 and 100 alike,
/// because only `0`-`9` are single tokens
/// (`docs/findings/2026-09-19-typed-option-logit-readout.md`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnswerAlphabet {
    tokens: Vec<AnswerToken>,
}

impl AnswerAlphabet {
    /// Build the alphabet this tokenizer can actually name, in pool order.
    pub fn from_tokenizer(tokenizer: &impl LabelTokenizer) -> Self {
        let mut tokens = Vec::new();
        // A label's own round-trip already makes its id unique: if two
        // labels encoded to the same single token, that token could only
        // decode back to one of them, so the other was never admitted. The
        // set is kept anyway because "no two slots name one token" is the
        // property the whole technique rests on, and a tokenizer that
        // normalizes on decode is not a hypothesis worth betting a silent
        // wrong answer on.
        let mut seen: BTreeSet<TokenId> = BTreeSet::new();
        for label in candidate_labels() {
            let Some(id) = single_token(tokenizer, &label) else {
                continue;
            };
            if seen.insert(id) {
                tokens.push(AnswerToken { label, id });
            }
        }
        Self { tokens }
    }

    /// Every admitted answer token, in pool order.
    pub fn tokens(&self) -> &[AnswerToken] {
        &self.tokens
    }

    /// How many options this tokenizer can name at once.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the tokenizer can name no option at all.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The first `count` answer tokens — what a decision with `count`
    /// options labels them with. `None` when the alphabet is shorter than
    /// that: a decision with more options than the tokenizer can name is
    /// refused, never served with two options sharing a label.
    pub fn take(&self, count: usize) -> Option<&[AnswerToken]> {
        (count <= self.tokens.len()).then(|| &self.tokens[..count])
    }

    /// Where `label` sits in the pool, if it was admitted.
    pub fn position(&self, label: &str) -> Option<usize> {
        self.tokens.iter().position(|token| token.label == label)
    }
}

/// The labels the alphabet is drawn from, in the order it draws them:
/// single characters first (`A`-`Z`, `a`-`z`, `0`-`9`), then the uppercase
/// bigrams. Which of them survive is the tokenizer's answer, not this
/// function's.
fn candidate_labels() -> impl Iterator<Item = String> {
    let singles = ('A'..='Z')
        .chain('a'..='z')
        .chain('0'..='9')
        .map(|c| c.to_string());
    let bigrams = ('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")));
    singles.chain(bigrams)
}

/// The token `label` stands for, if it is exactly one token that decodes
/// back to itself.
fn single_token(tokenizer: &impl LabelTokenizer, label: &str) -> Option<TokenId> {
    let encoded = tokenizer.encode(label)?;
    let [id] = encoded[..] else {
        return None;
    };
    (tokenizer.decode(&[id])? == label).then_some(id)
}
