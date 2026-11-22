//! GitHub #715 (spec #711): the **grant**, which turns the lanes' asks of the
//! shape rule into each lane's shape in a DFlash2 verify round of a load with
//! an extended shape, under each of the three shape policies, and the **cost
//! table** it reads (`spec.shape_period_ratios`).
//!
//! A pure function of what the host holds before a round launches. The leaf
//! runs its twin each round (`kernel/src/round_shape.h` `grant`), where each
//! lane's ask and recent rounds live (`ignis_seq`), and holds it to the same
//! rows (`kernel/tests/test_round_shape.cpp`); this one is the definition the
//! CPU tests hold to their tables and the GPU tests predict a round from.
//!
//! - `all`: every lane extended when every lane asks, at 1 to
//!   [`EXTENDED_SHAPE_MAX_LANES`] lanes; otherwise every lane at the base
//!   shape.
//! - `aggregate`: at one lane, the lane's ask; at 2 to
//!   [`EXTENDED_SHAPE_MAX_LANES`] lanes every lane extended when a lane asks
//!   whose recent rounds hold no extended round, or when the asking lanes'
//!   expected extra tokens together beat what the uniform extended round
//!   costs every lane; otherwise, and past those lanes, every lane at the
//!   base shape.
//! - `mixed`: at one lane, the lane's ask; at 2 to 7 lanes the asking lanes
//!   are granted the extended shape one by one, the highest expected extra
//!   tokens first, at a tie the lane refused longest, a lane with no estimate
//!   (no extended round among its recent rounds) after those with one. Each
//!   grant takes the round from its `w` extended lanes to `w + 1`, while
//!   `16 (w + 1) + 8 (n - 1)` columns stay within 64, and is made when
//!
//!   ```text
//!   expected_extra_tokens(lane)
//!       > (ratio(lanes, w + 1) / ratio(lanes, w) - 1) * sum over all lanes of expected_tokens
//!   ```
//!
//!   or, for a lane with no estimate, whenever the columns allow, so that its
//!   estimate has an extended round to read. A refused lane does not end the
//!   grant; only the columns do. Eight lanes are always at the base shape.
//!
//! Each mean is in thousandths of a token and each ratio a thousandth, and
//! the comparison is cross-multiplied in integers, so the leaf's twin decides
//! every round the same. A mixed round exists only at a window of
//! [`crate::lane_lines::MIXED_ROUND_WINDOW`] (16 columns an extended lane): a
//! load elsewhere runs `mixed` as `aggregate`, which its caller resolves
//! ([`crate::Speculation::shape_policy_run`]).

use crate::lane_lines::MIXES;
use crate::speculation::{ShapePolicy, BASE_SHAPE_DRAFT_TOKENS, EXTENDED_SHAPE_MAX_LANES};

/// The most columns a verify round carries: 8 lanes at the base shape.
const ROUND_COLUMNS: u32 = 64;
/// A lane's columns at the base shape and, at the mixed round's window, at the
/// extended one.
const BASE_COLUMNS: u32 = BASE_SHAPE_DRAFT_TOKENS + 1;
const EXTENDED_COLUMNS: u32 = crate::lane_lines::MIXED_ROUND_WINDOW + 1;
/// The most lanes a round with an extended lane holds: one extended lane beside
/// six base lanes.
pub const MIXED_MAX_LANES: u32 = 7;

/// The cost table (`spec.shape_period_ratios`): at 1 to [`MIXED_MAX_LANES`]
/// lanes, the period of the round with `w` of them extended over the base
/// round's at the same lane count, in thousandths, for each `w` the round's 64
/// columns hold (1 to `min(lanes, 8 - lanes)`). Its uniform entries (`w` the
/// lanes, at 1 to 4) are what `aggregate` reads; `mixed` reads every entry.
/// The one-lane entry is not read: a lane alone runs at the shape it asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeriodTable {
    uniform: [u32; EXTENDED_SHAPE_MAX_LANES as usize],
    mixed: [u32; MIXES.len()],
}

impl PeriodTable {
    /// The narrowest and widest ratio taken: a round with an extended lane
    /// costs at least the base round and at most twice it, for at most twice
    /// its columns.
    pub const RATIO_MILLI_RANGE: std::ops::RangeInclusive<u32> = 1000..=2000;

    /// The NVFP4 body's table on the RTX 5090, the one default every load
    /// reads until the trellis body has its own (GitHub #720) and a
    /// measurement of the final build replaces it (GitHub #721).
    ///
    /// The uniform entries, 1.128, 1.29, 1.363 and 1.344 at 1 to 4 lanes:
    /// measured on the GPU clock (round timing), FP8 exception weights, the
    /// `hq-e8-2b` cache, the default cost levers and route, at a matched
    /// context of about 3.5-5K tokens (the ratio grows with context), pooled
    /// over two orders; the card's 575 W limit lowers the extended round's
    /// clock at 3 and 4 lanes, so 3 lanes reads above 4.
    ///
    /// The mixed entries: 1+1, 1+3, 2+2 and 1+5 are the RTX 5090's measured
    /// periods of those mixes with their graph build, over the base round's at
    /// their lane count (19,091 over 16,529 us; 24,467 and 26,275 over 21,505;
    /// 29,039 over 26,574): 1.155, 1.138, 1.222 and 1.093. The other eight
    /// were not measured: each is the RTX 5090's measured base round of the
    /// mix's total columns over the base round at its lane count, since a
    /// mixed round with its graph build was measured to cost what a base round
    /// of its columns costs (3 to 7 lanes: about 19,200, 21,505, 24,668, 26,574
    /// and 29,770 us); at 64 columns, where no base round of 8 lanes was
    /// measured, one 8-column step past 56 at the step from 48 to 56 (32,966
    /// us). All of them pending GitHub #721.
    pub const NVFP4: Self = Self {
        uniform: [1128, 1290, 1363, 1344],
        // MIXES' order: 1+1, 1+2, 1+3, 1+4, 1+5, 1+6, 2+1, 2+2, 2+3, 2+4, 3+1, 3+2.
        mixed: [1155, 1120, 1138, 1077, 1093, 1107, 1285, 1222, 1207, 1241, 1384, 1336],
    };

    /// The entries a row of `lanes` lanes holds: one for each count of
    /// extended lanes the round's columns hold, 1 to `min(lanes, 8 - lanes)`.
    pub fn row_entries(lanes: u32) -> u32 {
        lanes.min(ROUND_COLUMNS / BASE_COLUMNS - lanes)
    }

    /// The table from its rows (`rows[l]` at `l + 1` lanes, `rows[l][w]` at
    /// `w + 1` of them extended): [`MIXED_MAX_LANES`] rows of
    /// [`Self::row_entries`] ratios each, every one in
    /// [`Self::RATIO_MILLI_RANGE`]. The refusal names the row, or the entry
    /// and its value.
    pub fn from_rows(rows: &[Vec<u32>]) -> Result<Self, String> {
        if rows.len() != MIXED_MAX_LANES as usize {
            return Err(format!("the cost table has {} rows, not one for each of 1 to {MIXED_MAX_LANES} lanes", rows.len()));
        }
        let mut table = Self::NVFP4;
        for (lanes, row) in (1..=MIXED_MAX_LANES).zip(rows) {
            let entries = Self::row_entries(lanes);
            if row.len() != entries as usize {
                return Err(format!(
                    "the cost table's row at {lanes} lane{} has {} ratios, not {entries} (1 to {entries} lanes extended)",
                    plural(lanes),
                    row.len()
                ));
            }
            for (extended, &ratio) in (1..=entries).zip(row) {
                *table.entry_mut(lanes, extended) = ratio;
            }
        }
        table.check()?;
        Ok(table)
    }

    /// The table whose uniform entries are `uniform` (at 1 to 4 lanes, the
    /// four-value form of `spec.shape_period_ratios` before the table), every
    /// mixed entry the default's.
    pub fn with_uniform(uniform: [u32; EXTENDED_SHAPE_MAX_LANES as usize]) -> Result<Self, String> {
        let table = Self { uniform, ..Self::NVFP4 };
        table.check()?;
        Ok(table)
    }

    /// The rows [`Self::from_rows`] takes.
    pub fn rows(&self) -> Vec<Vec<u32>> {
        (1..=MIXED_MAX_LANES).map(|lanes| (1..=Self::row_entries(lanes)).map(|w| self.ratio(lanes, w)).collect()).collect()
    }

    /// The uniform entries, at 1 to 4 lanes.
    pub const fn uniform(&self) -> [u32; EXTENDED_SHAPE_MAX_LANES as usize] {
        self.uniform
    }

    /// The mixed entries, in [`MIXES`]' order.
    pub const fn mixed(&self) -> [u32; MIXES.len()] {
        self.mixed
    }

    /// The period of a round of `lanes` lanes with `extended` of them
    /// extended over the base round's, in thousandths: 1000 for none.
    /// `extended` within the row ([`Self::row_entries`]).
    pub fn ratio(&self, lanes: u32, extended: u32) -> u32 {
        if extended == 0 {
            return 1000;
        }
        if extended == lanes {
            return self.uniform[lanes as usize - 1];
        }
        self.mixed[mix_at(extended, lanes - extended)]
    }

    fn entry_mut(&mut self, lanes: u32, extended: u32) -> &mut u32 {
        if extended == lanes {
            &mut self.uniform[lanes as usize - 1]
        } else {
            &mut self.mixed[mix_at(extended, lanes - extended)]
        }
    }

    fn check(&self) -> Result<(), String> {
        for lanes in 1..=MIXED_MAX_LANES {
            for extended in 1..=Self::row_entries(lanes) {
                let ratio = self.ratio(lanes, extended);
                if !Self::RATIO_MILLI_RANGE.contains(&ratio) {
                    return Err(format!(
                        "the cost table's ratio at {lanes} lane{} with {extended} extended ({}) is not in 1..=2",
                        plural(lanes),
                        f64::from(ratio) / 1000.0
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for PeriodTable {
    fn default() -> Self {
        Self::NVFP4
    }
}

fn plural(lanes: u32) -> &'static str {
    if lanes == 1 { "" } else { "s" }
}

/// The index in [`MIXES`] of `extended` lanes beside `base` lanes.
fn mix_at(extended: u32, base: u32) -> usize {
    MIXES.iter().position(|&mix| mix == (extended, base)).expect("a mix of the table")
}

/// What a lane's recent verify rounds say, summed over the window the policy
/// reads (`kernel/src/round_shape.h` `LaneEstimate`): its rounds, the tokens
/// each would commit at the base shape (its accepts up to 7, and the target's
/// own), its extended rounds and their accepts past position 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaneEstimate {
    pub rounds: u32,
    pub tokens: u32,
    pub extended_rounds: u32,
    pub extra_tokens: u32,
}

impl LaneEstimate {
    /// The tokens the lane is expected to commit a base round, in
    /// thousandths: its rounds' mean, or a full base round's 8 with none (the
    /// most the wider round can cost it).
    fn expected_tokens(&self) -> u64 {
        if self.rounds == 0 {
            u64::from(BASE_COLUMNS) * 1000
        } else {
            u64::from(self.tokens) * 1000 / u64::from(self.rounds)
        }
    }

    /// The drafts past position 7 the lane is expected to take at the
    /// extended shape, in thousandths: its extended rounds' mean, or `None`
    /// with no extended round to read.
    fn expected_extra_tokens(&self) -> Option<u64> {
        (self.extended_rounds != 0).then(|| u64::from(self.extra_tokens) * 1000 / u64::from(self.extended_rounds))
    }
}

/// One lane of a round as the grant reads it: what it asks, its recent
/// rounds, and the verify rounds in a row it asked the extended shape and ran
/// at the base shape (`refused_rounds`), which breaks a tie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GrantLane {
    pub asks: bool,
    pub estimate: LaneEstimate,
    pub refused_rounds: u32,
}

/// Each lane's shape in a round of `lanes` under `policy` and the cost table
/// `table`, `true` where it is extended (the module's header). Every lane at
/// one shape is a uniform round; both shapes, a mixed round, which only
/// `mixed` grants.
pub fn lane_shapes(policy: ShapePolicy, table: &PeriodTable, lanes: &[GrantLane]) -> Vec<bool> {
    let count = lanes.len() as u32;
    let uniform = |extended: bool| vec![extended; lanes.len()];
    if count <= 1 {
        return uniform(lanes.first().is_some_and(|lane| lane.asks));
    }
    match policy {
        ShapePolicy::All => uniform(count <= EXTENDED_SHAPE_MAX_LANES && lanes.iter().all(|lane| lane.asks)),
        ShapePolicy::Aggregate => uniform(count <= EXTENDED_SHAPE_MAX_LANES && aggregate_extends(table, lanes)),
        ShapePolicy::Mixed => granted(table, lanes),
    }
}

/// `aggregate` at 2 to 4 lanes (`kernel/src/round_shape.h` `policy_extends`).
fn aggregate_extends(table: &PeriodTable, lanes: &[GrantLane]) -> bool {
    if !lanes.iter().any(|lane| lane.asks) {
        return false;
    }
    if lanes.iter().any(|lane| lane.asks && lane.estimate.extended_rounds == 0) {
        return true;
    }
    let tokens: u64 = lanes.iter().map(|lane| lane.estimate.expected_tokens()).sum();
    let gain: u64 = lanes.iter().filter(|lane| lane.asks).filter_map(|lane| lane.estimate.expected_extra_tokens()).sum();
    let count = lanes.len() as u32;
    gain * 1000 > u64::from(table.ratio(count, count) - 1000) * tokens
}

/// `mixed` at 2 or more lanes.
fn granted(table: &PeriodTable, lanes: &[GrantLane]) -> Vec<bool> {
    let count = lanes.len() as u32;
    let tokens: i128 = lanes.iter().map(|lane| i128::from(lane.estimate.expected_tokens())).sum();
    // The asking lanes in the grant's order: those with an estimate first, the
    // highest expected extra tokens first, then the lane refused longest, then
    // the caller's order.
    let mut order: Vec<usize> = (0..lanes.len()).filter(|&i| lanes[i].asks).collect();
    order.sort_by_key(|&i| {
        let extra = lanes[i].estimate.expected_extra_tokens();
        (extra.is_none(), std::cmp::Reverse(extra.unwrap_or(0)), std::cmp::Reverse(lanes[i].refused_rounds), i)
    });
    let mut shapes = vec![false; lanes.len()];
    let mut extended = 0u32;
    for i in order {
        if EXTENDED_COLUMNS * (extended + 1) + BASE_COLUMNS * (count - extended - 1) > ROUND_COLUMNS {
            break;
        }
        let (current, next) = (i128::from(table.ratio(count, extended)), i128::from(table.ratio(count, extended + 1)));
        let grant = match lanes[i].estimate.expected_extra_tokens() {
            None => true,
            Some(extra) => i128::from(extra) * current > (next - current) * tokens,
        };
        if grant {
            shapes[i] = true;
            extended += 1;
        }
    }
    shapes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lane that asks (or not) over `rounds` recent rounds committing
    /// `tokens` at the base shape, `extended` of them extended with `extra`
    /// drafts past position 7 between them, refused `refused` rounds in a row.
    fn lane(asks: bool, rounds: u32, tokens: u32, extended: u32, extra: u32, refused: u32) -> GrantLane {
        GrantLane { asks, estimate: LaneEstimate { rounds, tokens, extended_rounds: extended, extra_tokens: extra }, refused_rounds: refused }
    }

    /// A conversation: 16 rounds of 2 tokens, never asking.
    fn conversation() -> GrantLane {
        lane(false, 16, 32, 0, 0, 0)
    }

    /// A lane asking, 16 rounds of 8 tokens, 4 of them extended taking
    /// `extra` drafts past position 7 each, refused `refused` rounds in a row.
    fn asking(extra: u32, refused: u32) -> GrantLane {
        lane(true, 16, 128, 4, 4 * extra, refused)
    }

    fn shapes(policy: ShapePolicy, lanes: &[GrantLane]) -> Vec<bool> {
        lane_shapes(policy, &PeriodTable::NVFP4, lanes)
    }

    /// The default table's rows: at 1 to 7 lanes, an entry for each count of
    /// extended lanes 64 columns hold.
    #[test]
    fn the_cost_table_holds_a_ratio_for_each_lane_count_and_extended_lanes_64_columns_hold() {
        let rows = PeriodTable::NVFP4.rows();
        assert_eq!(rows.iter().map(Vec::len).collect::<Vec<_>>(), [1, 2, 3, 4, 3, 2, 1]);
        assert_eq!(
            rows,
            [
                vec![1128],
                vec![1155, 1290],
                vec![1120, 1285, 1363],
                vec![1138, 1222, 1384, 1344],
                vec![1077, 1207, 1336],
                vec![1093, 1241],
                vec![1107],
            ]
        );
        assert_eq!(PeriodTable::from_rows(&rows), Ok(PeriodTable::NVFP4), "the rows read back");
        assert_eq!(PeriodTable::NVFP4.ratio(4, 0), 1000, "no lane extended: the base round");
    }

    /// The grant's order: the highest expected extra tokens first, so that at
    /// 7 lanes, where 64 columns hold one extended lane, it goes to the lane
    /// that gains most, not to the caller's first.
    #[test]
    fn the_grant_takes_the_highest_expected_extra_tokens_first() {
        // 7 lanes at 2 tokens a round and 1 extended (1.107): 14 tokens, so a
        // lane is granted past 1.498 extra tokens.
        let mut lanes = vec![conversation(); 7];
        lanes[0] = lane(true, 16, 32, 4, 12, 0); // 3 extra a round
        lanes[2] = lane(true, 16, 32, 4, 20, 0); // 5
        assert_eq!(shapes(ShapePolicy::Mixed, &lanes), [false, false, true, false, false, false, false]);
        // At 4 lanes (1+3 at 1.138, 2+2 at 1.222 over it) each pays: both granted,
        // whichever first.
        let four = [asking(5, 0), conversation(), asking(6, 0), conversation()];
        // Tokens 8 + 2 + 8 + 2 = 20: the first grant costs 2.76, the second
        // (1.222 / 1.138 - 1) * 20 = 1.476.
        assert_eq!(shapes(ShapePolicy::Mixed, &four), [true, false, true, false]);
        // Each step costs what it adds over the round before it: 2 extra
        // tokens, short of the first grant's 2.76, pay the second's 1.476
        // once 6 took the first; 1 does not.
        let second = [asking(2, 0), conversation(), asking(6, 0), conversation()];
        assert_eq!(shapes(ShapePolicy::Mixed, &second), [true, false, true, false]);
        let refused = [asking(1, 0), conversation(), asking(6, 0), conversation()];
        assert_eq!(shapes(ShapePolicy::Mixed, &refused), [false, false, true, false], "1 extra token does not pay 1.476");
    }

    /// At a tie the lane refused longest goes first; at a tie of that too, the
    /// caller's order.
    #[test]
    fn at_a_tie_the_lane_refused_longest_goes_first() {
        let mut lanes = vec![conversation(); 7];
        lanes[1] = lane(true, 16, 32, 4, 20, 2);
        lanes[4] = lane(true, 16, 32, 4, 20, 5);
        assert_eq!(shapes(ShapePolicy::Mixed, &lanes), [false, false, false, false, true, false, false]);
        lanes[4].refused_rounds = 2;
        assert_eq!(shapes(ShapePolicy::Mixed, &lanes), [false, true, false, false, false, false, false]);
    }

    /// A lane that asks with no extended round among its recent rounds has no
    /// estimate: it comes after the lanes with one and is granted whenever the
    /// columns allow, whatever its round costs.
    #[test]
    fn a_lane_without_an_estimate_comes_after_and_is_granted_when_the_columns_allow() {
        let fresh = lane(true, 16, 128, 0, 0, 9);
        // 3 lanes: the estimated lane first (1+2), then the fresh one (2+1),
        // whatever that step costs.
        assert_eq!(shapes(ShapePolicy::Mixed, &[fresh, asking(7, 0), conversation()]), [true, true, false]);
        // A lane with no round at all: a full base round's 8 tokens.
        assert_eq!(shapes(ShapePolicy::Mixed, &[lane(true, 0, 0, 0, 0, 0), conversation()]), [true, false]);
        // 7 lanes hold one extended lane: the estimated one, refused longest
        // or not, takes it before the fresh one.
        let mut lanes = vec![conversation(); 7];
        lanes[0] = fresh;
        lanes[3] = lane(true, 16, 32, 4, 20, 0);
        assert_eq!(shapes(ShapePolicy::Mixed, &lanes), [false, false, false, true, false, false, false]);
        // Its estimate refused, the fresh lane is granted.
        lanes[3] = lane(true, 16, 32, 4, 4, 0);
        assert_eq!(shapes(ShapePolicy::Mixed, &lanes), [true, false, false, false, false, false, false]);
    }

    /// Never past 64 columns: 16 an extended lane, 8 a base one.
    #[test]
    fn the_grant_never_passes_64_columns() {
        let gains = || asking(8, 0);
        // 5 lanes: 3+2 is 64 columns, a fourth extended lane would be 72.
        assert_eq!(shapes(ShapePolicy::Mixed, &vec![gains(); 5]), [true, true, true, false, false]);
        // 6 lanes: 2+4.
        assert_eq!(shapes(ShapePolicy::Mixed, &vec![gains(); 6]), [true, true, false, false, false, false]);
        // 4 lanes: every lane, the uniform extended round of 64 columns.
        assert_eq!(shapes(ShapePolicy::Mixed, &vec![gains(); 4]), [true; 4]);
        // 2 lanes: both, 32 columns.
        assert_eq!(shapes(ShapePolicy::Mixed, &vec![gains(); 2]), [true; 2]);
    }

    /// A step to a round that costs less than the current one (3+1 at 1.384
    /// to the uniform 1.344) is granted, even to a lane that gains nothing.
    #[test]
    fn a_step_to_a_cheaper_round_is_granted() {
        let lanes = [asking(8, 0), asking(8, 0), asking(8, 0), asking(0, 0)];
        assert_eq!(shapes(ShapePolicy::Mixed, &lanes), [true; 4]);
    }

    /// One lane runs at the shape it asks under `mixed` and `aggregate`, and
    /// under `all` (every lane asking).
    #[test]
    fn one_lane_runs_at_the_shape_it_asks() {
        for policy in [ShapePolicy::Mixed, ShapePolicy::Aggregate, ShapePolicy::All] {
            assert_eq!(shapes(policy, &[asking(0, 0)]), [true], "{policy:?}: a lane that asks, though it gains nothing");
            assert_eq!(shapes(policy, &[lane(false, 16, 128, 4, 32, 0)]), [false], "{policy:?}: a lane that does not ask");
        }
    }

    /// Eight lanes run at the base shape under every policy.
    #[test]
    fn eight_lanes_run_at_the_base_shape() {
        for policy in [ShapePolicy::Mixed, ShapePolicy::Aggregate, ShapePolicy::All] {
            assert_eq!(shapes(policy, &vec![asking(8, 3); 8]), [false; 8], "{policy:?}");
        }
    }

    /// `all` and `aggregate` give today's uniform shapes on the same lanes,
    /// whose numbers are worked here from #690's rule: never a mixed round.
    #[test]
    fn all_and_aggregate_give_uniform_rounds_on_the_same_lanes() {
        // Two lanes: 6 extra tokens beside a conversation. aggregate: 6 >
        // 0.29 * (8 + 2) = 2.9, extended; all: one lane does not ask, base.
        // mixed: the asking lane alone (1+1: 0.155 * 10 = 1.55).
        let pair = [asking(6, 0), conversation()];
        assert_eq!(shapes(ShapePolicy::Aggregate, &pair), [true, true]);
        assert_eq!(shapes(ShapePolicy::All, &pair), [false, false]);
        assert_eq!(shapes(ShapePolicy::Mixed, &pair), [true, false]);
        // Four lanes: 6 beside three conversations: 6 > 0.344 * 14 = 4.816,
        // extended; 4 is not.
        let four = [asking(6, 0), conversation(), conversation(), conversation()];
        assert_eq!(shapes(ShapePolicy::Aggregate, &four), [true; 4]);
        let four = [asking(4, 0), conversation(), conversation(), conversation()];
        assert_eq!(shapes(ShapePolicy::Aggregate, &four), [false; 4]);
        assert_eq!(shapes(ShapePolicy::Mixed, &four), [true, false, false, false], "4 > 0.138 * 14 = 1.932");
        // A lane with no estimate grants the uniform round under aggregate.
        let fresh = [lane(true, 16, 32, 0, 0, 0), conversation(), conversation()];
        assert_eq!(shapes(ShapePolicy::Aggregate, &fresh), [true; 3]);
        // Every lane asking: all extends at 1 to 4 lanes, not at 5.
        assert_eq!(shapes(ShapePolicy::All, &vec![asking(0, 0); 4]), [true; 4]);
        assert_eq!(shapes(ShapePolicy::All, &vec![asking(8, 0); 5]), [false; 5]);
        assert_eq!(shapes(ShapePolicy::Aggregate, &vec![asking(8, 0); 5]), [false; 5]);
    }
}
