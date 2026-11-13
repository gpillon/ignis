//! The sampled draft's rule (GitHub #575, spec #574) held to the
//! double-precision oracle (`support/draft_oracle.rs`): a lane that draws its
//! draft from the drafter's distribution q, takes it when `u·q(d) < p(d)`
//! and otherwise draws from what the target's distribution p has left over q
//! emits every token with the target's probability, whatever q is.
//!
//! Each case sweeps the draft's uniform over [`DRAFT_CELLS`] points and the
//! accept's and the refusal's over [`CELLS`] x [`CELLS`]: every token's share
//! of the grid is held to p within `4 / CELLS` plus `2 / DRAFT_CELLS` per
//! candidate, about 4e-3 (the oracle's header derives it). The target is
//! dyadic, so its f32 probabilities sum to exactly 1; its smallest
//! probability, 1/64, sits four times the tolerance above 0. The drafter
//! distributions are the ones that break a wrong rule: q equal to p, one-hot
//! on the target's top token, one-hot on an unlikely one, no mass where p has
//! some, mass outside the target's support, and flat.
//!
//! The draft's own draw is swept alone: each candidate is drawn with its
//! share of q, and none the drafter top-p cut removed; and as the drafter
//! temperature tends to 0 the draft becomes the selector walk's, today's
//! fixed draft ([`Dflash2Proposal::walk`]).

#[path = "support/draft_oracle.rs"]
mod draft_oracle;

use draft_oracle::{
    assert_emits_target, assert_outside_drafts_always_refused, column_share, draw_share, grid, lane_share, lane_tolerance,
    CELLS, DRAFT_CELLS, TARGET_IDS, TARGET_PROBS,
};
use ignis_core::dflash2_readout::{Dflash2Proposal, TOP_K};
use ignis_core::sampled_draft::{accept_verdict, drafter_distribution, draw_draft, ColumnUniforms, Distribution};

/// A lane under the rule over the whole grid, its draft drawn from
/// `drafter` by the reference: each token's share, and the share of the
/// accept's grid on which each draft was taken.
fn lane(drafter_ids: &[i32], drafter_probs: &[f32]) -> (std::collections::BTreeMap<i32, f64>, Vec<(i32, f64)>) {
    let target = Distribution::new(&TARGET_IDS, &TARGET_PROBS);
    let drafter = Distribution::new(drafter_ids, drafter_probs);
    let draft_grid = grid(DRAFT_CELLS);
    let drawn = draw_share(drafter_ids.len(), DRAFT_CELLS, |i| draw_draft(drafter_probs, draft_grid[i]));
    let drafts: Vec<(i32, f64)> = drafter_ids.iter().copied().zip(drawn).collect();
    let g = grid(CELLS);
    let mut accepted = Vec::new();
    let emitted = lane_share(&drafts, |draft| {
        let share = column_share(draft, CELLS, |i, j| {
            accept_verdict(target, drafter, draft, ColumnUniforms { accept: g[i], refusal: g[j] })
        });
        accepted.push((draft, share.accepted));
        share
    });
    (emitted, accepted)
}

fn assert_lane_emits_target(case: &str, drafter_ids: &[i32], drafter_probs: &[f32]) -> Vec<(i32, f64)> {
    let (emitted, accepted) = lane(drafter_ids, drafter_probs);
    let tolerance = lane_tolerance(CELLS, DRAFT_CELLS, drafter_ids.len());
    assert_emits_target(case, &TARGET_IDS, &TARGET_PROBS, &emitted, tolerance);
    accepted
}

#[test]
fn a_drafter_equal_to_the_target_emits_the_target() {
    let accepted = assert_lane_emits_target("q = p", &TARGET_IDS, &TARGET_PROBS);
    // Every draft is taken on the whole grid: min(1, p/q) = 1.
    for (draft, share) in accepted {
        assert_eq!(share, 1.0, "q = p: draft {draft} refused on part of the grid");
    }
}

#[test]
fn a_drafter_one_hot_on_the_targets_top_token_emits_the_target() {
    assert_lane_emits_target("q one-hot on the top token", &[501], &[1.0]);
}

#[test]
fn a_drafter_one_hot_on_an_unlikely_token_emits_the_target() {
    assert_lane_emits_target("q one-hot on a 1/64 token", &[2048], &[1.0]);
}

#[test]
fn a_drafter_with_no_mass_where_the_target_has_some_emits_the_target() {
    // Nothing on 12, 4410, 365, 2048 or 88: the refusal alone emits them.
    assert_lane_emits_target("q on three of the eight", &[9030, 501, 77], &[0.5, 0.25, 0.25]);
}

#[test]
fn a_drafter_with_mass_outside_the_targets_support_emits_the_target() {
    let accepted = assert_lane_emits_target("q half outside the support", &[31337, 12, 501], &[0.5, 0.25, 0.25]);
    assert_outside_drafts_always_refused("q half outside the support", &accepted);
}

#[test]
fn a_flat_drafter_emits_the_target() {
    // Sixteen candidates, the proposal head's count, half of them outside the
    // target's support.
    let ids = [88, 640, 501, 9031, 12, 3, 4410, 70000, 77, 1999, 365, 248000, 2048, 5, 9030, 42];
    let accepted = assert_lane_emits_target("q flat over sixteen", &ids, &[1.0 / 16.0; 16]);
    assert_outside_drafts_always_refused("q flat over sixteen", &accepted);
}

/// A column's scores whose softmax at temperature 1 is 1, 8, 2, 4 and 1
/// sixteenths: out of descending order, and with a tie, so the top-p cut has
/// to sort them and break the tie toward the smaller index.
fn column_scores() -> Vec<f32> {
    [1.0f64, 8.0, 2.0, 4.0, 1.0].iter().map(|w| (w / 16.0).ln() as f32).collect()
}

#[test]
fn the_draft_is_drawn_with_the_drafters_probability_and_never_past_its_top_p_cut() {
    let weights = [1.0f64, 8.0, 2.0, 4.0, 1.0];
    // Closed forms: the softmax of ln w at temperature t is w^(1/t),
    // normalized; the cut keeps the smallest descending prefix holding top-p.
    let cases: [(f32, f32, [f64; 5]); 4] = [
        (1.0, 1.0, weights.map(|w| w / 16.0)),
        // 8, 12, 14 and 15 sixteenths: the fourth reaches 0.9, and of the two
        // ones the first is kept.
        (1.0, 0.9, [1.0 / 15.0, 8.0 / 15.0, 2.0 / 15.0, 4.0 / 15.0, 0.0]),
        // Squared, 64, 16, 4, 1 and 1 of 86: 84/86 is the first prefix past 0.95.
        (0.5, 0.95, [0.0, 64.0 / 84.0, 4.0 / 84.0, 16.0 / 84.0, 0.0]),
        // Square roots: only all five hold 0.95 of the mass.
        (2.0, 0.95, {
            let total: f64 = weights.iter().map(|w| w.sqrt()).sum();
            weights.map(|w| w.sqrt() / total)
        }),
    ];
    let draft_grid = grid(DRAFT_CELLS);
    let tolerance = 2.0 / DRAFT_CELLS as f64 + 1e-6;
    for (temperature, top_p, expected) in cases {
        let q = drafter_distribution(&column_scores(), temperature, top_p);
        let drawn = draw_share(q.len(), DRAFT_CELLS, |i| draw_draft(&q, draft_grid[i]));
        for (candidate, (&share, &want)) in drawn.iter().zip(&expected).enumerate() {
            let case = format!("temperature {temperature}, top-p {top_p}, candidate {candidate}");
            assert!((q[candidate] as f64 - want).abs() <= 1e-6, "{case}: q {} against {want:.6}", q[candidate]);
            if want == 0.0 {
                assert_eq!(share, 0.0, "{case}: cut by the top-p and drawn on {share} of the grid");
            } else {
                assert!((share - want).abs() <= tolerance, "{case}: drawn on {share:.6} of the grid against {want:.6}");
            }
        }
    }
    // Four equal scores at top-p 0.5: the second candidate brings the mass to
    // exactly half, and the prefix that reaches the top-p is kept, the
    // sampler's inclusive cut, ties toward the smaller index.
    assert_eq!(drafter_distribution(&[3.0; 4], 1.0, 0.5), vec![0.5, 0.5, 0.0, 0.0]);
}

/// Three columns whose every row ranks the sixteen successors in a
/// different order, a quarter apart, so that the walk's path matters.
fn lattice() -> Dflash2Proposal {
    let k = 3;
    let mut scores = vec![0f32; k * TOP_K * TOP_K];
    for column in 0..k {
        for pred in 0..TOP_K {
            for succ in 0..TOP_K {
                scores[(column * TOP_K + pred) * TOP_K + succ] = ((column * 7 + pred * 3 + succ * 5) % TOP_K) as f32 * 0.25;
            }
        }
    }
    Dflash2Proposal {
        drafts: vec![0; k],
        candidate_ids: (0..(k * TOP_K) as i32).collect(),
        candidate_logits: vec![0.0; k * TOP_K],
        scores,
        drafter_logits: None,
        q: vec![0.0; k * TOP_K],
    }
}

/// The column's row of transition scores after predecessor `pred`.
fn row(proposal: &Dflash2Proposal, column: usize, pred: usize) -> &[f32] {
    &proposal.scores[(column * TOP_K + pred) * TOP_K..][..TOP_K]
}

#[test]
fn as_the_drafter_temperature_tends_to_0_the_draft_tends_to_the_selector_walks() {
    let proposal = lattice();
    let today = proposal.walk();
    let draft_grid = grid(DRAFT_CELLS);
    let mut before = vec![0f64; today.len()];
    for temperature in [1.0, 0.3, 0.1, 0.03, 0.01] {
        let mut pred = 0;
        for (column, &fixed) in today.iter().enumerate() {
            let q = drafter_distribution(row(&proposal, column, pred), temperature, 1.0);
            let share = draw_share(TOP_K, DRAFT_CELLS, |i| draw_draft(&q, draft_grid[i]))[fixed];
            assert!(
                share > before[column],
                "column {column} at temperature {temperature}: the walk's draft drawn on {share:.6} of the grid, {:.6} before",
                before[column]
            );
            before[column] = share;
            pred = fixed;
        }
    }
    assert_eq!(before, vec![1.0; today.len()], "at temperature 0.01 the walk's draft is drawn on the whole grid");
}

#[test]
fn at_a_drafter_temperature_of_0_the_draft_is_the_selector_walks_even_on_a_tie() {
    let mut proposal = lattice();
    let path = proposal.walk();
    // Tie column 2's best successor with candidate 0: the walk takes 0.
    let best = row(&proposal, 2, path[1])[path[2]];
    proposal.scores[(2 * TOP_K + path[1]) * TOP_K] = best;
    let today = proposal.walk();
    assert_eq!(today[2], 0);
    let mut pred = 0;
    for (column, &fixed) in today.iter().enumerate() {
        let q = drafter_distribution(row(&proposal, column, pred), 0.0, 0.95);
        for u in [0.0, 0.5, 1.0 - f32::EPSILON / 2.0] {
            assert_eq!(draw_draft(&q, u), fixed, "column {column}, uniform {u}");
        }
        pred = fixed;
    }
}
