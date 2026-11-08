//! Speculation past 8 verify rows against speculation off, row by row
//! (GitHub #329 item 5, ADR 0056). Included by path from the Flash-Next GPU
//! test that holds a wide budget to the plain decode route, and from the CPU
//! test of these rules; the server's round shapes
//! (`crates/server/tests/support/round_shapes.rs`) read its BF16 helpers.
//!
//! Past 8 rows a verify round's linears and mixes take their tensor-core
//! routes, so its logits are the plain round's up to summation order, not bit
//! for bit: a free-running greedy text can flip at a near-tie and never come
//! back. So the comparison is teacher-forced -- the same tokens through both
//! routes, logits row by row -- and its tolerance is a count of BF16 ulps at
//! the reference row's largest magnitude ([`TOLERANCE_ULPS`]). What stays
//! exact at any budget is the commit: a round commits its own argmax
//! ([`commit_follows_argmax`]).

// Each including test reads the rules it needs.
#![allow(dead_code)]

use ignis_core::speculation::FLASH_NEXT_DEFAULT_VERIFY_ROWS;

/// The teacher-forced tolerance, in BF16 ulps at the reference row's largest
/// magnitude: no logit of a verify round's row may sit further than this from
/// the plain round's. Measured on the RTX 5090 (2026-10-18, ADR 0056): the
/// default's own 8-row rounds reach 40.2 ulps from spec-off, 16-32 rows 58.3
/// (one tail logit, 32 rows hq-e8-2b); 64 is the next power of two above
/// both -- a margin of 1.1x over the wide budgets' widest on this card, 1.6x
/// over the default's. Kept that tight because the discriminator is close:
/// rows on another context (a lane one moved pick off the text) reach 33-202,
/// 89% of them past 64 but only 59% past 96. Another card's tensor-core
/// tiling may move the drift: the RTX PRO 6000 run re-derives it before a
/// profile takes a budget past 8 (the finding's follow-up).
pub const TOLERANCE_ULPS: f32 = 64.0;

/// The near tie a moved pick must sit inside, in the same ulps: a verify
/// round's pick may differ from the plain round's only where the plain row
/// rates the two within this. Measured moved picks sat at gaps of 0-8 ulps in
/// every cell (ADR 0056); 16 is the next power of two above, 2x. Not
/// [`TOLERANCE_ULPS`], which bounds how far any logit may move: 64 ulps at a
/// top logit of 20 is 8 logits, no near tie.
pub const PICK_GAP_ULPS: f32 = 16.0;

/// The teacher-forced mean log-probability's tolerance, in nats a token: the
/// wide route's mean over a cell's compared rows against the plain route's,
/// either way. Measured from -0.0117 to +0.0077 (wide minus spec-off): the
/// verify rows read spec-off's own greedy text both less and more likely
/// than spec-off's rows do, so the check is two-sided; 0.02 is 1.7x the
/// widest.
pub const MEAN_LOG_PROB_NATS: f64 = 0.02;

/// The head's first-position acceptance under which a budget's cell is
/// refused: a wide round's rows or the head's entries gone wrong show as
/// drafts no longer accepted. Measured 0.74-0.94 on these prompts.
pub const HEAD_ACCEPTANCE_FLOOR: f64 = 0.5;

/// The band a wide budget's first-draft acceptance keeps around the same
/// prompts' at the default's 8 rows (the head run on them two lanes at a
/// time, rounds of 8 rows): the owner's "acceptance within a band" (ADR
/// 0056), either way.
pub const ACCEPTANCE_BAND: f64 = 0.1;

pub fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One BF16 ulp at `x`'s magnitude (`flash_next_forward_gpu.rs`'s rule).
pub fn bf16_ulp(x: f32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent - 7)
}

/// A row's pick as the engine draws it -- the lowest id among equal maxima
/// (the vendored argmax's rule) -- and its value.
pub fn argmax(row: &[u16]) -> (u32, f32) {
    let (mut best, mut first) = (0u32, f32::NEG_INFINITY);
    for (id, &bits) in row.iter().enumerate() {
        let v = bf16(bits);
        if v > first {
            first = v;
            best = id as u32;
        }
    }
    (best, first)
}

/// `target`'s log-probability under `row` (a log-softmax, accumulated in
/// f64).
pub fn log_prob(row: &[u16], target: u32) -> f64 {
    let (_, top) = argmax(row);
    let sum: f64 = row.iter().map(|&b| (f64::from(bf16(b)) - f64::from(top)).exp()).sum();
    f64::from(bf16(row[target as usize])) - f64::from(top) - sum.ln()
}

/// One teacher-forced row pair: the plain route's (`off`) and the wide
/// verify round's (`on`) after the same tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowDrift {
    /// The largest |on - off| over the vocabulary, in logits.
    pub max_abs: f32,
    /// `max_abs` in BF16 ulps at `off`'s largest magnitude.
    pub ulps: f32,
    /// The id that moved most (the lowest among equals).
    pub worst: u32,
    pub pick_off: u32,
    pub pick_on: u32,
    /// `off[pick_off] - off[pick_on]` in the same ulps: 0 when the picks agree.
    pub gap_ulps: f32,
}

pub fn row_drift(off: &[u16], on: &[u16]) -> RowDrift {
    assert_eq!(off.len(), on.len(), "rows of different vocabularies");
    let scale = off.iter().map(|&b| bf16(b).abs()).fold(0f32, f32::max);
    let ulp = bf16_ulp(scale);
    let (mut worst, mut max_abs) = (0u32, 0f32);
    for (id, (&a, &b)) in off.iter().zip(on).enumerate() {
        let moved = (bf16(a) - bf16(b)).abs();
        if moved > max_abs {
            (worst, max_abs) = (id as u32, moved);
        }
    }
    let (pick_off, top) = argmax(off);
    let (pick_on, _) = argmax(on);
    RowDrift { max_abs, ulps: max_abs / ulp, worst, pick_off, pick_on, gap_ulps: (top - bf16(off[pick_on as usize])) / ulp }
}

/// Within `tolerance_ulps` every logit, and a different pick only where the
/// plain row rates the two picks within `pick_gap_ulps` (a near tie). A line
/// naming the excess otherwise.
pub fn within(drift: &RowDrift, tolerance_ulps: f32, pick_gap_ulps: f32) -> Result<(), String> {
    if drift.ulps > tolerance_ulps {
        return Err(format!("a logit moved {} ({} ulps, past {tolerance_ulps})", drift.max_abs, drift.ulps));
    }
    if drift.pick_off != drift.pick_on && drift.gap_ulps > pick_gap_ulps {
        return Err(format!(
            "the pick moved from {} to {}, {} ulps apart in the plain row (past {pick_gap_ulps})",
            drift.pick_off, drift.pick_on, drift.gap_ulps
        ));
    }
    Ok(())
}

/// The head's first-draft acceptance at a wide budget, `(accepted, offered)`,
/// within `band` of the same prompts' at 8 rows (`narrow`), either way. A
/// line naming both otherwise, or when either offered no first draft.
pub fn acceptance_within_band(wide: (u32, u32), narrow: (u32, u32), band: f64) -> Result<(), String> {
    if wide.1 == 0 || narrow.1 == 0 {
        return Err(format!(
            "no first draft to compare: {wide:?} against {FLASH_NEXT_DEFAULT_VERIFY_ROWS} rows' {narrow:?}"
        ));
    }
    let (w, n) = (f64::from(wide.0) / f64::from(wide.1), f64::from(narrow.0) / f64::from(narrow.1));
    if (w - n).abs() > band {
        return Err(format!(
            "first-draft acceptance {w:.3} against {FLASH_NEXT_DEFAULT_VERIFY_ROWS} rows' {n:.3} on the same prompts, past {band}"
        ));
    }
    Ok(())
}

/// One lane's share of a verify round: its anchor, the drafts it verified
/// (its extent of them), the run it committed, the token that stood pending
/// after it (`None` once the lane's budget is spent), and the picks of its
/// rows, one a column, `picks[j]` after the column holding the anchor (j = 0)
/// or draft j.
pub struct LaneCommit<'a> {
    pub anchor: u32,
    pub drafts: &'a [u32],
    pub committed: &'a [u32],
    pub next: Option<u32>,
    pub picks: &'a [u32],
}

/// The greedy accept, exact at any budget whatever the routes' numerics: the
/// drafts accepted are the longest prefix each of which is the pick after the
/// column before it, the run is the anchor and those drafts, and what stands
/// pending after it is the pick after the run's last column. A line naming
/// the first disagreement otherwise.
pub fn commit_follows_argmax(lane: &LaneCommit<'_>) -> Result<(), String> {
    if lane.picks.len() <= lane.drafts.len() {
        return Err(format!("{} rows for {} drafts", lane.picks.len(), lane.drafts.len()));
    }
    let accepted = lane.drafts.iter().zip(lane.picks).take_while(|(d, p)| d == p).count();
    let mut run = vec![lane.anchor];
    run.extend_from_slice(&lane.drafts[..accepted]);
    if lane.committed != run.as_slice() {
        return Err(format!(
            "committed {:?} where the rows' picks {:?} accept {run:?} of anchor {} and drafts {:?}",
            lane.committed, lane.picks, lane.anchor, lane.drafts
        ));
    }
    match lane.next {
        Some(next) if next != lane.picks[accepted] => Err(format!(
            "{next} stands pending after the run where its last column's pick is {}",
            lane.picks[accepted]
        )),
        _ => Ok(()),
    }
}
