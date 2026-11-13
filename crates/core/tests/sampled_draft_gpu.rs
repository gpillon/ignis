//! The sampled draft's device kernels (GitHub #576, spec #574, ADR 0090) held
//! to the double-precision oracle the host reference is held to
//! (`support/draft_oracle.rs`, `sampled_draft_oracle.rs`): the same target,
//! the same six drafter distributions and the same grids, the verdicts and
//! the draws computed on the device through the leaf's test-only entries
//! (`ignis_core::sampled_draft::device`).
//!
//! A lane that draws its draft from the drafter's distribution q, takes it
//! when `u·q(d) < p(d)` and otherwise draws from what the target's p has left
//! over q emits every token with the target's probability, whatever q is: on
//! the device, each token's share of the grid of the draft's, the accept's
//! and the refusal's uniforms is p's within `4 / CELLS` plus `2 / DRAFT_CELLS`
//! per candidate, a draft the target gives nothing is refused on the whole
//! grid, and a draft is drawn on its share of q and never past the drafter's
//! top-p cut. The device's q itself is the oracle's closed forms.
//!
//! No model: the entries need a CUDA device alone. Explicit GPU profile (ADR
//! 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing GPU or a kernel
//! error is a **skip**; under the profile, a **hard failure**. Run via
//! `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

#[path = "support/draft_oracle.rs"]
mod draft_oracle;

use std::collections::BTreeMap;

use draft_oracle::{assert_emits_target, column_share, draw_share, grid, lane_share, lane_tolerance};
use ignis_artifact::CudaDevice;
use ignis_core::gpu_profile;
use ignis_core::sampled_draft::{device, Verdict};

/// Cells per uniform of the accept and the refusal, and of the draft's own
/// (the CPU oracle test's).
const CELLS: usize = 1024;
const DRAFT_CELLS: usize = 1 << 20;

/// The oracle test's target: 24, 16, 10, 6, 4, 2, 1 and 1 sixty-fourths, in
/// the sampler's order.
const TARGET_IDS: [i32; 8] = [501, 77, 9030, 12, 4410, 365, 2048, 88];
const TARGET_PROBS: [f32; 8] = [0.375, 0.25, 0.15625, 0.09375, 0.0625, 0.03125, 0.015625, 0.015625];

/// The device's result, or a skip outside the profile.
fn on_device<T>(result: Result<T, String>, what: &str) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("{what}: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    }
}

/// A lane under the rule over the whole grid on the device: each token's
/// share, and the share of the accept's grid on which each draft was taken.
fn device_lane(drafter_ids: &[i32], drafter_probs: &[f32]) -> Option<(BTreeMap<i32, f64>, Vec<(i32, f64)>)> {
    let draft_grid = grid(DRAFT_CELLS);
    let drawn = on_device(device::draws(drafter_probs, &draft_grid), "the device draw")?;
    let shares = draw_share(drafter_ids.len(), DRAFT_CELLS, |i| drawn[i]);
    let drafts: Vec<(i32, f64)> = drafter_ids.iter().copied().zip(shares).collect();
    let g = grid(CELLS);
    let mut columns = BTreeMap::new();
    for &(draft, share) in &drafts {
        if share > 0.0 {
            let verdicts = on_device(
                device::verdicts(&TARGET_IDS, &TARGET_PROBS, drafter_ids, drafter_probs, draft, &g),
                "the device verdicts",
            )?;
            columns.insert(draft, verdicts);
        }
    }
    let mut accepted = Vec::new();
    let emitted = lane_share(&drafts, |draft| {
        let verdicts = &columns[&draft];
        let share = column_share(draft, CELLS, |i, j| verdicts[i * CELLS + j]);
        accepted.push((draft, share.accepted));
        share
    });
    Some((emitted, accepted))
}

fn assert_device_lane_emits_target(case: &str, drafter_ids: &[i32], drafter_probs: &[f32]) -> Option<Vec<(i32, f64)>> {
    let (emitted, accepted) = device_lane(drafter_ids, drafter_probs)?;
    let tolerance = lane_tolerance(CELLS, DRAFT_CELLS, drafter_ids.len());
    assert_emits_target(&format!("device, {case}"), &TARGET_IDS, &TARGET_PROBS, &emitted, tolerance);
    Some(accepted)
}

/// A draft the target gives no probability to is refused on every point of
/// the accept's grid.
fn assert_outside_drafts_always_refused(case: &str, accepted: &[(i32, f64)]) {
    let outside: Vec<_> = accepted.iter().filter(|(draft, _)| !TARGET_IDS.contains(draft)).collect();
    assert!(!outside.is_empty(), "{case}: no draft outside the support was drawn");
    for (draft, share) in outside {
        assert_eq!(*share, 0.0, "{case}: draft {draft}, outside the target's support, was taken on {share} of the grid");
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_device_kernels_emit_the_targets_distribution_and_draw_the_drafters() {
    if let Err(e) = CudaDevice::create(0) {
        if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
            return;
        }
    }

    // --- the six drafter distributions: the lane emits p ---------------------
    let Some(accepted) = assert_device_lane_emits_target("q = p", &TARGET_IDS, &TARGET_PROBS) else { return };
    for (draft, share) in accepted {
        assert_eq!(share, 1.0, "device, q = p: draft {draft} refused on part of the grid");
    }
    let Some(_) = assert_device_lane_emits_target("q one-hot on the top token", &[501], &[1.0]) else { return };
    let Some(_) = assert_device_lane_emits_target("q one-hot on a 1/64 token", &[2048], &[1.0]) else { return };
    let Some(_) = assert_device_lane_emits_target("q on three of the eight", &[9030, 501, 77], &[0.5, 0.25, 0.25]) else {
        return;
    };
    let case = "q half outside the support";
    let Some(accepted) = assert_device_lane_emits_target(case, &[31337, 12, 501], &[0.5, 0.25, 0.25]) else { return };
    assert_outside_drafts_always_refused(case, &accepted);
    let flat = [88, 640, 501, 9031, 12, 3, 4410, 70000, 77, 1999, 365, 248000, 2048, 5, 9030, 42];
    let case = "q flat over sixteen";
    let Some(accepted) = assert_device_lane_emits_target(case, &flat, &[1.0 / 16.0; 16]) else { return };
    assert_outside_drafts_always_refused(case, &accepted);

    // A refusal never emits the refused draft where q holds more of it than p
    // does: with q one-hot on a 1/64 token, every refusal draws another one.
    let g = grid(CELLS);
    let Some(verdicts) = on_device(device::verdicts(&TARGET_IDS, &TARGET_PROBS, &[2048], &[1.0], 2048, &g), "verdicts") else {
        return;
    };
    assert!(
        verdicts.iter().all(|&v| v != Verdict::Refused(2048)),
        "a refusal drew the draft q holds more of than p"
    );

    // --- the drafter's distribution and the draw from it ---------------------
    // The oracle test's column: softmax at temperature 1 of 1, 8, 2, 4 and 1
    // sixteenths, out of order and with a tie.
    let weights = [1.0f64, 8.0, 2.0, 4.0, 1.0];
    let scores: Vec<f32> = weights.iter().map(|w| (w / 16.0).ln() as f32).collect();
    let sqrt_total: f64 = weights.iter().map(|w| w.sqrt()).sum();
    let cases: [(f32, f32, [f64; 5]); 4] = [
        (1.0, 1.0, weights.map(|w| w / 16.0)),
        (1.0, 0.9, [1.0 / 15.0, 8.0 / 15.0, 2.0 / 15.0, 4.0 / 15.0, 0.0]),
        (0.5, 0.95, [0.0, 64.0 / 84.0, 4.0 / 84.0, 16.0 / 84.0, 0.0]),
        (2.0, 0.95, weights.map(|w| w.sqrt() / sqrt_total)),
    ];
    let draft_grid = grid(DRAFT_CELLS);
    let tolerance = 2.0 / DRAFT_CELLS as f64 + 1e-6;
    for (temperature, top_p, expected) in cases {
        let Some(q) = on_device(device::drafter_distribution(&scores, temperature, top_p), "the device q") else { return };
        let Some(drawn) = on_device(device::draws(&q, &draft_grid), "the device draw") else { return };
        let shares = draw_share(q.len(), DRAFT_CELLS, |i| drawn[i]);
        for (candidate, (&share, &want)) in shares.iter().zip(&expected).enumerate() {
            let case = format!("device, temperature {temperature}, top-p {top_p}, candidate {candidate}");
            assert!((q[candidate] as f64 - want).abs() <= 1e-6, "{case}: q {} against {want:.6}", q[candidate]);
            if want == 0.0 {
                assert_eq!(share, 0.0, "{case}: cut by the top-p and drawn on {share} of the grid");
            } else {
                assert!((share - want).abs() <= tolerance, "{case}: drawn on {share:.6} of the grid against {want:.6}");
            }
        }
    }
    let Some(tied) = on_device(device::drafter_distribution(&[3.0; 4], 1.0, 0.5), "the device q") else { return };
    assert_eq!(tied, vec![0.5, 0.5, 0.0, 0.0], "the inclusive cut, ties toward the smaller index");
    // At temperature 0 the draft is the selector walk's, the first maximum,
    // at every uniform.
    let Some(walk) = on_device(device::drafter_distribution(&[0.5, 2.0, -1.0, 2.0], 0.0, 0.95), "the device q") else { return };
    assert_eq!(walk, vec![0.0, 1.0, 0.0, 0.0]);
    let Some(drawn) = on_device(device::draws(&walk, &[0.0, 0.5, 1.0 - f32::EPSILON / 2.0]), "the device draw") else { return };
    assert_eq!(drawn, vec![1, 1, 1], "at temperature 0 every uniform draws the walk's draft");
}
