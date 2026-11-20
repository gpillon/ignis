//! A mirror of the spec's shape rule and shape policies (GitHub #689, #690,
//! spec #684), written from their definitions, for the GPU tests that hold a
//! two-shape DFlash2 load's every round to the shape its lanes' own rounds
//! give it (`round_identity_gpu.rs`, `dflash2_round_shape_gpu.rs`). The leaf's
//! own are pure functions with their tables (`kernel/tests/test_round_shape.cpp`);
//! this is what a test predicts each live round's shape from.
//!
//! The rule: a lane with no history does not ask; after a base round that
//! drafted the base window it asks when the target took at least the entry
//! threshold, and a base round its budget cut short leaves its ask; after an
//! extended round it asks while the round took at least the stay threshold.
//!
//! The policy, at 1 to 4 lanes with `IGNIS_ROUND_SHAPE` unset (a lane the
//! host cuts asks like any other, #691): `all`, extended when every lane asks;
//! `aggregate`, at one lane the lane's ask, and at two or more extended when a
//! lane asks whose last rounds hold no extended round (granted), or else when
//!
//! ```text
//! sum over asking lanes of expected_extra_tokens
//!     > (period ratio at the lane count - 1) * sum over all lanes of expected_tokens
//! ```
//!
//! over each lane's last `ShapeEstimate::rounds()` verify rounds that drafted
//! the base window at least: its tokens a round at the base shape (its accepts
//! up to 7, and the target's own; a full base round's 8 for a lane with none),
//! its accepts past 7 a round over its extended rounds alone, each mean in
//! thousandths of a token, as the leaf compares them.

use ignis_core::{ShapeEstimate, ShapePolicy, ShapeRule, BASE_SHAPE_DRAFT_TOKENS};

/// One lane as the rule and the policy see it: what it asks, and its recent
/// verify rounds (whether each ran extended, and what the target accepted).
#[derive(Debug, Clone, Default)]
pub struct Lane {
    pub asks: bool,
    rounds: Vec<(bool, u32)>,
}

impl Lane {
    /// After a verify round of the lane at `extended` (else the base shape)
    /// of `extent` drafts, of which the target accepted `accepted`.
    pub fn after(&mut self, rule: ShapeRule, extended: bool, extent: u32, accepted: u32) {
        self.asks = if extended {
            accepted >= rule.stay_accepted()
        } else if extent < BASE_SHAPE_DRAFT_TOKENS {
            self.asks
        } else {
            accepted >= rule.enter_accepted()
        };
        if extent >= BASE_SHAPE_DRAFT_TOKENS {
            self.rounds.push((extended, accepted));
        }
    }

    /// Restored from a snapshot, or a claim: the ask the image carries, and no
    /// recent rounds, which it does not.
    pub fn restored(&mut self, asks: bool) {
        self.asks = asks;
        self.rounds.clear();
    }

    /// Its last `window` rounds: their count, tokens at the base shape, the
    /// extended rounds among them and their accepts past the base window.
    fn estimate(&self, window: u32) -> (u64, u64, u64, u64) {
        let recent = &self.rounds[self.rounds.len().saturating_sub(window as usize)..];
        let base = BASE_SHAPE_DRAFT_TOKENS;
        let tokens = recent.iter().map(|&(_, accepted)| u64::from(accepted.min(base) + 1)).sum();
        let extended: Vec<u32> = recent.iter().filter(|&&(extended, _)| extended).map(|&(_, accepted)| accepted).collect();
        let extra = extended.iter().map(|&accepted| u64::from(accepted.saturating_sub(base))).sum();
        (recent.len() as u64, tokens, extended.len() as u64, extra)
    }
}

/// Whether a round of `lanes` runs at the extended shape under `policy`, the
/// switch unset: never past the extended shape's 4 lanes.
pub fn extends(policy: ShapePolicy, estimate: ShapeEstimate, lanes: &[&Lane]) -> bool {
    if lanes.len() > estimate.period_ratios_milli().len() {
        return false;
    }
    match policy {
        ShapePolicy::All => lanes.iter().all(|lane| lane.asks),
        ShapePolicy::Aggregate if lanes.len() == 1 => lanes[0].asks,
        ShapePolicy::Aggregate => {
            let estimates: Vec<(bool, (u64, u64, u64, u64))> = lanes.iter().map(|lane| (lane.asks, lane.estimate(estimate.rounds()))).collect();
            if estimates.iter().any(|&(asks, (_, _, extended, _))| asks && extended == 0) {
                return true;
            }
            let full_base_round = u64::from(BASE_SHAPE_DRAFT_TOKENS + 1) * 1000;
            let tokens: u64 = estimates.iter().map(|&(_, (rounds, tokens, _, _))| if rounds == 0 { full_base_round } else { tokens * 1000 / rounds }).sum();
            let gain: u64 = estimates.iter().filter(|(asks, _)| *asks).map(|&(_, (_, _, extended, extra))| extra * 1000 / extended).sum();
            let ratio = u64::from(estimate.period_ratios_milli()[lanes.len() - 1]);
            estimates.iter().any(|(asks, _)| *asks) && gain * 1000 > (ratio - 1000) * tokens
        }
    }
}
