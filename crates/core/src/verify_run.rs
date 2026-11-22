//! GitHub #712 (spec #711): a verify round's runs, lane by lane, from what
//! the leaf's round wrote through `struct ignis_decode_options`, and each
//! lane's counters for the round. No kernel is needed to read them, so the
//! reading and the counting are held in the CPU suite while
//! [`crate::step`], which makes the leaf call, is `cuda`-only.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use crate::types::{splits_shape_asks, SpecCounters};

/// One lane's run through a verify round (P5-05, GitHub #155): the committed run, and how
/// many drafts the round verified for the lane -- its extent, whether the
/// caller proposed them or the DFlash2 drafter did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneVerifyRun {
    pub tokens: Vec<i32>,
    pub extent: u32,
    /// GitHub #307: on a Flash-Next load with the MTP head, the drafts the
    /// head made for the lane's next round, at the frontier this run left it
    /// at (`window` of them); empty on any other load.
    pub next_drafts: Vec<i32>,
    /// GitHub #315 (ADR 0048): the lane's next pending token is its
    /// `reasoning_close` in place of a stop id -- the next round emits it
    /// first.
    pub redirected: bool,
    /// GitHub #689: the lane ran at its load's extended round shape; false at
    /// the base shape, the only one of a load at 7 drafts or fewer. GitHub
    /// #712: the lane's own shape, which the round reports lane by lane.
    pub extended: bool,
    /// GitHub #690: the round split its lanes' asks of the extended shape
    /// ([`crate::types::splits_shape_asks`]), the same for every lane of it.
    pub split: bool,
}

impl LaneVerifyRun {
    /// GitHub #712: the lane's counters for its round: the drafts it
    /// verified, those of them committed (the run past its anchor, which a
    /// stop cut may leave shorter than what the target accepted), position by
    /// position, at the lane's own shape and in a round that split its lanes'
    /// asks or not.
    pub fn counters(&self) -> SpecCounters {
        let accepted = (self.tokens.len() as u32).saturating_sub(1).min(self.extent);
        SpecCounters::round(self.extent, accepted).at_shape(self.extended).in_split_round(self.split)
    }
}

/// What one verify round's leaf call wrote (`struct ignis_decode_options`),
/// every array lane-major over the round's lanes at the load's window.
pub(crate) struct RoundOutputs<'a> {
    pub window: usize,
    /// `[lanes][window + 1]`: each lane's committed run, `committed` of them.
    pub tokens: &'a [i32],
    pub committed: &'a [i32],
    pub extents: &'a [u32],
    /// `[lanes][window]`: a lane's next drafts where every one is a token.
    pub next_drafts: &'a [i32],
    pub redirected: &'a [i32],
    /// GitHub #712: each lane's round shape, 0 the base shape and 1 the
    /// extended one (`out_lane_shapes`).
    pub lane_shapes: &'a [u32],
    /// GitHub #690: how many of the lanes asked the extended shape going in.
    pub shape_asks: u32,
}

impl RoundOutputs<'_> {
    /// Each lane's run, in the round's lane order.
    pub(crate) fn runs(&self) -> Vec<LaneVerifyRun> {
        let width = self.window;
        let split = splits_shape_asks(self.committed.len(), self.shape_asks);
        self.committed
            .iter()
            .zip(self.extents)
            .enumerate()
            .map(|(index, (&count, &extent))| {
                let start = index * (width + 1);
                let drafts = &self.next_drafts[index * width..(index + 1) * width];
                LaneVerifyRun {
                    tokens: self.tokens[start..start + count as usize].to_vec(),
                    extent,
                    next_drafts: if drafts.iter().all(|&d| d >= 0) { drafts.to_vec() } else { Vec::new() },
                    redirected: self.redirected[index] == 1,
                    extended: self.lane_shapes[index] == 1,
                    split,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SpecCounters;

    /// Three lanes at a window of 3 (4 columns a lane): lane 0 commits its
    /// anchor and 2 of its 3 drafts, lane 1 its anchor alone of 3 drafts and
    /// is redirected, lane 2 its anchor and 1 of 2 drafts.
    const TOKENS: [i32; 12] = [5, 6, 7, -1, 8, -1, -1, -1, 9, 10, -1, -1];
    const COMMITTED: [i32; 3] = [3, 1, 2];
    const EXTENTS: [u32; 3] = [3, 3, 2];
    const NEXT_DRAFTS: [i32; 9] = [-1; 9];
    const REDIRECTED: [i32; 3] = [0, 1, 0];

    fn round(lane_shapes: &[u32], shape_asks: u32) -> Vec<LaneVerifyRun> {
        RoundOutputs {
            window: 3,
            tokens: &TOKENS,
            committed: &COMMITTED,
            extents: &EXTENTS,
            next_drafts: &NEXT_DRAFTS,
            redirected: &REDIRECTED,
            lane_shapes,
            shape_asks,
        }
        .runs()
    }

    fn total(runs: &[LaneVerifyRun]) -> SpecCounters {
        runs.iter().map(LaneVerifyRun::counters).fold(SpecCounters::default(), |sum, lane| sum + lane)
    }

    /// The round reports each lane's shape, and each lane is counted at its
    /// own: the drafts it verified and committed at each position, and the
    /// round at the extended shape or not. A uniform round's report is one
    /// shape for every lane, which counts what main's one shape a round did:
    /// every lane extended, or none.
    #[test]
    fn each_lane_is_counted_at_the_shape_the_round_reports_for_it() {
        let runs = round(&[1, 1, 1], 3);
        assert_eq!(runs.iter().map(|run| run.tokens.clone()).collect::<Vec<_>>(), [vec![5, 6, 7], vec![8], vec![9, 10]]);
        assert_eq!(runs.iter().map(|run| (run.extent, run.redirected)).collect::<Vec<_>>(), [(3, false), (3, true), (2, false)]);
        assert!(runs.iter().all(|run| run.next_drafts.is_empty()), "no lane's next drafts on a DFlash2 round");
        assert_eq!(runs.iter().map(|run| run.extended).collect::<Vec<_>>(), [true, true, true]);
        let extended = total(&runs);
        assert_eq!(
            (extended.rounds, extended.drafted, extended.accepted, extended.extended_rounds, extended.split_rounds),
            (3, 8, 3, 3, 0)
        );
        assert_eq!(extended.drafted_at[..4], [3, 3, 2, 0]);
        assert_eq!(extended.accepted_at[..4], [2, 1, 0, 0]);

        // A base round, one lane of three asking the extended shape: a round
        // that split its lanes' asks.
        let base = total(&round(&[0, 0, 0], 1));
        assert_eq!((base.rounds, base.drafted, base.accepted, base.extended_rounds, base.split_rounds), (3, 8, 3, 0, 3));
        assert_eq!((base.drafted_at, base.accepted_at), (extended.drafted_at, extended.accepted_at));

        // Lane 0 at the extended shape beside two lanes at the base shape.
        let runs = round(&[1, 0, 0], 1);
        assert_eq!(runs.iter().map(|run| run.extended).collect::<Vec<_>>(), [true, false, false]);
        assert_eq!(runs.iter().map(|run| run.counters().extended_rounds).collect::<Vec<_>>(), [1, 0, 0]);
        let mixed = total(&runs);
        assert_eq!((mixed.rounds, mixed.drafted, mixed.accepted, mixed.extended_rounds, mixed.split_rounds), (3, 8, 3, 1, 3));
    }
}
