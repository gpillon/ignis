//! The double-precision oracle the sampled draft's rule is held to (GitHub
//! #575, spec #574). Included by path from the tests that hold an
//! implementation of the rule to it: the host reference's in the CPU suite,
//! and the device kernels' in a GPU test, over the same grid.
//!
//! The rule's draws are deterministic in their uniforms, so the oracle sweeps
//! them instead of sampling: a uniform takes the midpoint of each of `cells`
//! equal cells of [0, 1), and the share of the grid on which an
//! implementation emits a token is that token's probability under the
//! implementation, to the grid's resolution. Shares are counted and summed in
//! f64 and held to the probability the target gives the token.
//!
//! **Resolution.** On one uniform's grid, the share of midpoints inside an
//! interval of [0, 1) is the interval's length to one cell, and to one more
//! when f32 rounding moves an end of the interval across a midpoint: within
//! `2 / cells`. A column's verdict reads two uniforms, the accept's and the
//! refusal's, and emits `A·[x = d] + (1 − A)·R(x)` of each token `x` (`A` the
//! accept's share, `R` the refusal draw's), so each token's share is within
//! `4 / cells` of exact. A lane first draws its draft on its own grid of
//! `draft_cells`, each candidate's share within `2 / draft_cells`, and the
//! drafts' errors add: [`lane_tolerance`]. The teacher-forced verdict (GitHub
//! #577) reads one more uniform, its coin, on a grid of its own: its share
//! is within that lane tolerance plus `2 / coin_cells`.

#![allow(dead_code)] // each test that includes the oracle uses its own part of it

use std::collections::BTreeMap;

use ignis_core::sampled_draft::Verdict;

/// The grid of one uniform: the midpoints of `cells` equal cells of [0, 1),
/// exact in f32 for a power of two up to 2^23.
pub fn grid(cells: usize) -> Vec<f32> {
    assert!(cells.is_power_of_two() && cells <= 1 << 23, "{cells} cells are not exact in f32");
    (0..cells).map(|i| ((i as f64 + 0.5) / cells as f64) as f32).collect()
}

/// What one column emits over the grid of its accept's and its refusal's
/// uniforms, given its draft.
pub struct ColumnShare {
    /// Each token's share of the grid.
    pub emitted: BTreeMap<i32, f64>,
    /// The share on which the draft was taken.
    pub accepted: f64,
}

/// Sweeps one column's two uniforms over `cells` x `cells` points.
/// `verdict_at(i, j)` is the verdict at the accept's `i`-th grid point and
/// the refusal's `j`-th.
pub fn column_share(draft: i32, cells: usize, mut verdict_at: impl FnMut(usize, usize) -> Verdict) -> ColumnShare {
    let mut counts: BTreeMap<i32, u64> = BTreeMap::new();
    let mut accepted = 0u64;
    for i in 0..cells {
        for j in 0..cells {
            let token = match verdict_at(i, j) {
                Verdict::Accepted => {
                    accepted += 1;
                    draft
                }
                Verdict::Refused(token) => token,
            };
            *counts.entry(token).or_default() += 1;
        }
    }
    let total = (cells * cells) as f64;
    ColumnShare {
        emitted: counts.into_iter().map(|(token, n)| (token, n as f64 / total)).collect(),
        accepted: accepted as f64 / total,
    }
}

/// Sweeps one column's two uniforms over `cells` x `cells` points, as
/// [`column_share`] does, keeping each verdict whole (GitHub #577): the share
/// of the grid on which the column took its draft, and on which it refused it
/// and drew each token. The teacher-forced verdict reads the live column's
/// verdict, so its share over the column's grid is a sum over these.
pub fn verdict_share(cells: usize, mut verdict_at: impl FnMut(usize, usize) -> Verdict) -> Vec<(Verdict, f64)> {
    let mut accepted = 0u64;
    let mut refused: BTreeMap<i32, u64> = BTreeMap::new();
    for i in 0..cells {
        for j in 0..cells {
            match verdict_at(i, j) {
                Verdict::Accepted => accepted += 1,
                Verdict::Refused(token) => *refused.entry(token).or_default() += 1,
            }
        }
    }
    let total = (cells * cells) as f64;
    std::iter::once((Verdict::Accepted, accepted))
        .chain(refused.into_iter().map(|(token, n)| (Verdict::Refused(token), n)))
        .filter(|&(_, n)| n > 0)
        .map(|(verdict, n)| (verdict, n as f64 / total))
        .collect()
}

/// Sweeps the draft's own uniform over `cells` points: each candidate
/// index's share of the grid. `draw_at(i)` is the candidate drawn at the
/// `i`-th grid point.
pub fn draw_share(candidates: usize, cells: usize, mut draw_at: impl FnMut(usize) -> usize) -> Vec<f64> {
    let mut counts = vec![0u64; candidates];
    for i in 0..cells {
        counts[draw_at(i)] += 1;
    }
    counts.into_iter().map(|n| n as f64 / cells as f64).collect()
}

/// A lane over the product of its grids: the draft drawn with each
/// candidate's share, then that draft's column. Each token's share.
pub fn lane_share(drafts: &[(i32, f64)], mut column: impl FnMut(i32) -> ColumnShare) -> BTreeMap<i32, f64> {
    let mut emitted: BTreeMap<i32, f64> = BTreeMap::new();
    for &(draft, share) in drafts.iter().filter(|(_, share)| *share > 0.0) {
        for (token, s) in column(draft).emitted {
            *emitted.entry(token).or_default() += share * s;
        }
    }
    emitted
}

/// How far a lane's share of a token may sit from exact: `4 / cells` for
/// the column and `2 / draft_cells` for each of the drafter's `candidates`.
pub fn lane_tolerance(cells: usize, draft_cells: usize, candidates: usize) -> f64 {
    4.0 / cells as f64 + 2.0 * candidates as f64 / draft_cells as f64
}

/// Holds a lane's emission to the target's sampling distribution: every
/// token of the support at the target's probability within `tolerance`, and
/// nothing outside the support on any grid point.
pub fn assert_emits_target(case: &str, target_ids: &[i32], target_probs: &[f32], emitted: &BTreeMap<i32, f64>, tolerance: f64) {
    let total: f64 = target_probs.iter().map(|&p| p as f64).sum();
    let mut worst = 0f64;
    for (&token, &p) in target_ids.iter().zip(target_probs) {
        let expected = p as f64 / total;
        let got = emitted.get(&token).copied().unwrap_or(0.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "{case}: token {token} emitted on {got:.6} of the grid, the target gives it {expected:.6} (tolerance {tolerance:.2e})"
        );
        worst = worst.max((got - expected).abs());
    }
    println!("{case}: worst token off the target by {worst:.2e} (tolerance {tolerance:.2e})");
    for (&token, &share) in emitted {
        assert!(
            target_ids.iter().zip(target_probs).any(|(&id, &p)| id == token && p > 0.0),
            "{case}: token {token} is outside the target's support and was emitted on {share:.6} of the grid"
        );
    }
}
