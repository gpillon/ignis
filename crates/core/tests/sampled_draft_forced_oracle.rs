//! The teacher-forced verdict under the sampled draft (GitHub #577, spec
//! #574) held to the double-precision oracle (`support/draft_oracle.rs`) the
//! rule itself is held to (`sampled_draft_oracle.rs`).
//!
//! A replay forces a recorded token x through a round under the rule. A live
//! round that emits x drafted x and took it with probability
//! min(p(x), q(x)) / p(x), so the forced round must take x with exactly that
//! probability for the rounds a replay walks to be distributed as live ones
//! ([`forced_verdict`]): over the product of the grids of the live draft's
//! uniform, the accept's, the refusal's and the coin's, the share on which the
//! forced verdict takes x is that probability, for every token x of the
//! target's support and of the drafter's candidates, for every one of the
//! drafter distributions that break a wrong rule; a token the target gives
//! nothing is taken with the drafter's own probability of it, and a token
//! neither gives anything never. The tolerance is the rule's lane tolerance
//! plus `2 / COIN_CELLS`.
//!
//! And a recording of the lane's own live column is forced back into that
//! column: on every point of the live grid, with the token the live column
//! emitted as the recording, the forced verdict is the live verdict, whatever
//! the coin.

#[path = "support/draft_oracle.rs"]
mod draft_oracle;

use draft_oracle::{draw_share, grid, lane_tolerance, verdict_share, CELLS, DRAFT_CELLS, TARGET_IDS, TARGET_PROBS};
use ignis_core::sampled_draft::{accept_verdict, draw_draft, forced_verdict, ColumnUniforms, Distribution, Verdict};

/// Cells of the coin's uniform.
const COIN_CELLS: usize = 1 << 12;

/// A token neither distribution holds.
const STRANGER: i32 = 999_999;

/// The rule's oracle drafters: q equal to p, one-hot on the target's top
/// token, one-hot on a 1/64 token, no mass on five of the target's eight,
/// half outside the target's support, and flat over sixteen candidates.
fn drafters() -> Vec<(&'static str, Vec<i32>, Vec<f32>)> {
    vec![
        ("q = p", TARGET_IDS.to_vec(), TARGET_PROBS.to_vec()),
        ("q one-hot on the top token", vec![501], vec![1.0]),
        ("q one-hot on a 1/64 token", vec![2048], vec![1.0]),
        ("q on three of the eight", vec![9030, 501, 77], vec![0.5, 0.25, 0.25]),
        ("q half outside the support", vec![31337, 12, 501], vec![0.5, 0.25, 0.25]),
        (
            "q flat over sixteen",
            vec![88, 640, 501, 9031, 12, 3, 4410, 70000, 77, 1999, 365, 248000, 2048, 5, 9030, 42],
            vec![1.0 / 16.0; 16],
        ),
    ]
}

/// The live column under the rule over its grids: each live draft with its
/// share of the draft's grid, and that draft's verdicts with their shares of
/// the accept's and the refusal's.
fn live_column(drafter_ids: &[i32], drafter_probs: &[f32], cells: usize) -> Vec<(i32, f64, Vec<(Verdict, f64)>)> {
    let target = Distribution::new(&TARGET_IDS, &TARGET_PROBS);
    let drafter = Distribution::new(drafter_ids, drafter_probs);
    let draft_grid = grid(DRAFT_CELLS);
    let drawn = draw_share(drafter_ids.len(), DRAFT_CELLS, |i| draw_draft(drafter_probs, draft_grid[i]));
    let g = grid(cells);
    drafter_ids
        .iter()
        .zip(drawn)
        .filter(|&(_, share)| share > 0.0)
        .map(|(&draft, share)| {
            let verdicts = verdict_share(cells, |i, j| {
                accept_verdict(target, drafter, draft, ColumnUniforms { accept: g[i], refusal: g[j] })
            });
            (draft, share, verdicts)
        })
        .collect()
}

/// What a live round takes a recorded token with: min(p, q) / p, or q where
/// the target gives the token nothing.
fn live_take_probability(drafter_ids: &[i32], drafter_probs: &[f32], token: i32) -> f64 {
    let sum = |ids: &[i32], probs: &[f32]| -> f64 {
        ids.iter().zip(probs).filter(|&(&id, _)| id == token).map(|(_, &p)| p as f64).sum()
    };
    let p = sum(&TARGET_IDS, &TARGET_PROBS);
    let q = sum(drafter_ids, drafter_probs);
    if p > 0.0 { p.min(q) / p } else { q }
}

#[test]
fn a_recorded_token_is_taken_with_the_probability_a_live_round_takes_it() {
    let coins = grid(COIN_CELLS);
    let mut worst = 0f64;
    // The cases where the coin split a column's verdict, on neither side for
    // the whole of its grid.
    let mut coin_split = Vec::new();
    for (case, drafter_ids, drafter_probs) in drafters() {
        let target = Distribution::new(&TARGET_IDS, &TARGET_PROBS);
        let drafter = Distribution::new(&drafter_ids, &drafter_probs);
        let live = live_column(&drafter_ids, &drafter_probs, CELLS);
        let tolerance = lane_tolerance(CELLS, DRAFT_CELLS, drafter_ids.len()) + 2.0 / COIN_CELLS as f64;
        let mut tokens: Vec<i32> = TARGET_IDS.iter().chain(&drafter_ids).copied().chain([STRANGER]).collect();
        tokens.sort_unstable();
        tokens.dedup();
        for recorded in tokens {
            // The product grid, summed over the live column's verdicts: the
            // forced verdict reads the live grid only through them.
            let mut taken = 0f64;
            for (draft, draft_share, verdicts) in &live {
                for &(verdict, verdict_share) in verdicts {
                    let on_coin = coins
                        .iter()
                        .filter(|&&coin| forced_verdict(target, drafter, recorded, *draft, verdict, coin))
                        .count() as f64
                        / COIN_CELLS as f64;
                    if on_coin > 0.0 && on_coin < 1.0 && !coin_split.contains(&case) {
                        coin_split.push(case);
                    }
                    taken += draft_share * verdict_share * on_coin;
                }
            }
            let expected = live_take_probability(&drafter_ids, &drafter_probs, recorded);
            assert!(
                (taken - expected).abs() <= tolerance,
                "{case}: recorded token {recorded} taken on {taken:.6} of the grid, a live round takes it with {expected:.6} \
                 (tolerance {tolerance:.2e})"
            );
            if recorded == STRANGER {
                assert_eq!(taken, 0.0, "{case}: a token neither distribution holds was taken");
            }
            worst = worst.max((taken - expected).abs());
        }
    }
    // A drafter that gives a recorded token less than the target does leaves
    // it to the coin: q on three of the eight, half outside the support, flat.
    assert!(coin_split.len() >= 3, "the coin split a verdict only in {coin_split:?}: not a test of it");
    println!("worst recorded token off a live round's probability by {worst:.2e}; the coin split {coin_split:?}");
}

#[test]
fn a_recording_of_the_live_column_is_forced_back_into_the_live_verdict() {
    let coins = grid(64);
    for (case, drafter_ids, drafter_probs) in drafters() {
        let target = Distribution::new(&TARGET_IDS, &TARGET_PROBS);
        let drafter = Distribution::new(&drafter_ids, &drafter_probs);
        let g = grid(256);
        for &draft in &drafter_ids {
            for i in 0..g.len() {
                for j in 0..g.len() {
                    let live = accept_verdict(target, drafter, draft, ColumnUniforms { accept: g[i], refusal: g[j] });
                    let (emitted, took) = match live {
                        Verdict::Accepted => (draft, true),
                        Verdict::Refused(token) => (token, false),
                    };
                    for &coin in &coins {
                        assert_eq!(
                            forced_verdict(target, drafter, emitted, draft, live, coin),
                            took,
                            "{case}: draft {draft}, live verdict {live:?}, coin {coin}: forcing the live column's own token \
                             is not the live verdict"
                        );
                    }
                }
            }
        }
    }
}
