//! A split between a decode round's pick and a verify round's pick, judged by
//! the two **round shapes** (GitHub #534, spec #531, glossary Near-tie and
//! Round shape). Included by path from the multimodal speculative acceptance
//! (`vision_dflash2_gpu.rs`) and from the CPU test of these rules
//! (`round_shape_near_tie.rs`). No GPU types: the GPU test computes the two
//! shapes' logits through the prefill, this module says what they mean and
//! which spans the verify shape replays.
//!
//! A decode round is one column wide and a verify round `k + 1`, and the wider
//! shape takes other activation routes (ADR 0016), so the two can rank two
//! near-equal tokens differently. A split is therefore judged by the position
//! computed outside the rounds in each shape ([`judge`]): numerics when the
//! two shapes disagree about the winner, when the verify shape ranks the
//! verify's pick at least as high, or when either shape is within one BF16
//! ulp of a tie at its larger logit's magnitude; a fault of the verify round
//! only when the verify shape sides with the decode round by more than that.
//! The tolerance is the logits' own rounding, never a tuned constant.
//!
//! The text near-tie rule of the core speculative tests
//! (`crates/core/tests/support/near_tie.rs`) is a different rule, unchanged.

// Each including test reads the rules it needs.
#![allow(dead_code)]

use std::fmt;
use std::ops::Range;

/// One shape's logits at the split: the decode round's pick and the verify
/// round's pick, as the probe read them (BF16 values).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub decode_pick: f32,
    pub verify_pick: f32,
}

impl Shape {
    /// How far the shape ranks the decode round's pick above the verify
    /// round's: positive sides with the decode round.
    pub fn gap(self) -> f32 {
        self.decode_pick - self.verify_pick
    }

    /// One BF16 ulp at the larger of the two logits' magnitudes.
    pub fn ulp(self) -> f32 {
        bf16_ulp(self.decode_pick.abs().max(self.verify_pick.abs()))
    }
}

/// Which of the numerics clauses held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Numerics {
    /// One shape ranks the decode round's pick higher and the other the
    /// verify round's. A shape at an exact tie names no winner; it is within
    /// an ulp.
    pub shapes_disagree: bool,
    /// The verify shape ranks the verify round's pick at least as high.
    pub verify_shape_sides_with_verify: bool,
    /// Either shape's gap is within one ulp at its larger logit.
    pub within_ulp: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Numerics(Numerics),
    /// The verify shape sides with the decode round beyond one ulp, and so
    /// does the decode shape: the verify round computed the position wrong.
    VerifyFault,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Numerics(held) => {
                let clauses = [
                    (held.shapes_disagree, "the shapes disagree about the winner"),
                    (held.verify_shape_sides_with_verify, "the verify shape ranks the verify's pick at least as high"),
                    (held.within_ulp, "a shape is within one ulp of a tie"),
                ];
                let held: Vec<&str> = clauses.iter().filter(|(held, _)| *held).map(|(_, clause)| *clause).collect();
                write!(f, "numerics: {}", held.join("; "))
            }
            Verdict::VerifyFault => {
                write!(f, "a verify-round fault: the verify shape sides with the decode round beyond one ulp")
            }
        }
    }
}

pub fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One BF16 ulp at `x`'s magnitude: the step between two BF16 values in its
/// binade, `2^(exponent - 7)` for BF16's 8 significant bits
/// (`crates/core/tests/support/logit_drift.rs`'s rule).
pub fn bf16_ulp(x: f32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent - 7)
}

/// The split judged by its two shapes: `decode`, the position reached one
/// token at a time, and `verify`, the position inside a span as wide as the
/// verify round. Panics on a logit that is not finite, which no numerics
/// explain.
pub fn judge(decode: Shape, verify: Shape) -> Verdict {
    for logit in [decode.decode_pick, decode.verify_pick, verify.decode_pick, verify.verify_pick] {
        assert!(logit.is_finite(), "a probe read a logit of {logit}: decode shape {decode:?}, verify shape {verify:?}");
    }
    let (d, v) = (decode.gap(), verify.gap());
    let held = Numerics {
        shapes_disagree: (d > 0.0 && v < 0.0) || (d < 0.0 && v > 0.0),
        verify_shape_sides_with_verify: v <= 0.0,
        within_ulp: d.abs() <= decode.ulp() || v.abs() <= verify.ulp(),
    };
    if held.shapes_disagree || held.verify_shape_sides_with_verify || held.within_ulp {
        Verdict::Numerics(held)
    } else {
        Verdict::VerifyFault
    }
}

/// One verify round of a speculative run, as the run recorded it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyRound {
    /// The sequence's frontier before the round: its anchor's position, its
    /// first column.
    pub base: u64,
    pub extent: u32,
    /// The tokens it committed: its anchor and the drafts it accepted, the
    /// columns whose KV and GDN state it kept.
    pub committed: usize,
}

/// The spans the verify-shape probe writes after the prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replay {
    /// The round that drew the split token.
    pub round: usize,
    /// Every earlier round, one span from its base over the columns it
    /// committed, in positions.
    pub history: Vec<Range<u64>>,
    /// The split's round: one span as wide as the verify round from its base.
    pub span: Range<u64>,
    /// The row of `span` whose logits drew the split token.
    pub row: usize,
}

/// The replay of `rounds` that reaches the position `drawn_at`, whose logits
/// drew the split token, in a span `columns` wide: the verify round's width
/// whatever the round's extent. A round's column `j` at `base + j` drew the
/// token after it -- an accepted draft or the correction -- so the round
/// that drew the split covers `drawn_at` among its committed columns. `None`
/// when no round does: before the first round's base the prompt's own prefill
/// drew it.
pub fn replay(rounds: &[VerifyRound], columns: u32, drawn_at: u64) -> Option<Replay> {
    let round = rounds.iter().position(|r| r.base <= drawn_at && drawn_at < r.base + r.committed as u64)?;
    let base = rounds[round].base;
    Some(Replay {
        round,
        history: rounds[..round].iter().map(|r| r.base..r.base + r.committed as u64).collect(),
        span: base..base + u64::from(columns),
        row: (drawn_at - base) as usize,
    })
}

/// A span's tokens: `text` at the span's positions, `text[0]` at `first`, and
/// past the text's end its last token again. Only the judged span runs past
/// the text, after the row it reads, and a causal span's row reads no later
/// column.
pub fn span_tokens(text: &[i32], first: u64, span: Range<u64>) -> Vec<i32> {
    let last = *text.last().expect("a span over no text");
    span.map(|position| {
        let index = position.checked_sub(first).expect("a span before the text") as usize;
        text.get(index).copied().unwrap_or(last)
    })
    .collect()
}
