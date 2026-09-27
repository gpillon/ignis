//! `locate` in one pass: the **head vote** (spec
//! `docs/specs/decide/18-locate-by-attention.md` phase B, read as spec 19's
//! track L registered it — `19-a-span-read-from-attention.md` § Phase 3;
//! GitHub #275, ADR 0041).
//!
//! A `locate` asks which **segment** of a text state — a line of a string,
//! an element of an array — an instruction names. The model already knows
//! before it writes anything: at the copy scaffold `{"quote":"`, a few dozen
//! heads of the later GQA layers look at the segment it is about to copy.
//! No one of them is reliable, but they fail on different questions, and a
//! majority of them finds the segment as often as a `choice` over labelled
//! segments does (`docs/findings/2026-09-27-locate-by-head-vote-go.md`).
//!
//! What lives here is the host's whole part of it:
//!
//! - **How rows become shares** ([`segment_shares`]): one head's softmax over
//!   the span, summed per segment.
//! - **How shares become an answer** ([`read_vote`]): each head votes for
//!   the segment its question lifts most above the content-free prefill's,
//!   and the most-voted segment wins.
//!
//! Both are ported from `tools/locate-sets/score.py` (`features`,
//! `vote_reading`) and held to it by golden cases it writes
//! (`crates/core/tests/locate_reading.rs`).

use std::ops::Range;

use crate::identity::ArtifactHash;
use crate::pointing::{PointingHead, layer_head};

/// How many segments a `locate`'s ranking names at most.
pub const RANKING_LEN: usize = 5;

/// What one artifact is calibrated with for `locate`: the heads that vote,
/// best first, and the longest span the vote was measured to hold at.
///
/// The rest of the reading is fixed, not calibrated: the copy scaffold
/// `{"quote":"`, the content-free prefill subtracted, one vote per head. Spec
/// 19 phase 0 chose those on the development sets for every configuration it
/// tried, and a recalibration repeats its procedure
/// (`tools/locate-sets/README.md`) to choose the heads and measure the
/// ceiling again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocateCalibration {
    /// The heads that vote, in the order they were ranked when chosen — the
    /// order a tie is broken in, and the order the leaf's rows come back in.
    pub heads: &'static [PointingHead],
    /// `LOCATE_MAX_KEYS`: the most keys of state a `locate` reads. Spec 18's
    /// rule 3 — the longest measured length whose top-1 stays within five
    /// points of the shortest's — on the set that judged the vote. A longer
    /// span is refused before any prefill, never answered unmeasured.
    pub max_keys: u32,
}

/// The served NVFP4 27B's vote: the 32 heads with the most training hits as
/// one-head readings on sets A+B, fitted by spec 19 phase 0's procedure
/// (`.scratch/locate/phase0/vote-choice.json`, recorded in spec 19 § Phase
/// 3, track L before set D was read), and set D's ceiling
/// (`docs/findings/2026-09-27-locate-by-head-vote-go.md`).
const SERVED_NVFP4_27B_VOTE: LocateCalibration = LocateCalibration {
    heads: &[
        layer_head(39, 12),
        layer_head(47, 20),
        layer_head(59, 16),
        layer_head(55, 17),
        layer_head(59, 17),
        layer_head(59, 7),
        layer_head(59, 6),
        layer_head(55, 0),
        layer_head(59, 8),
        layer_head(59, 10),
        layer_head(63, 5),
        layer_head(39, 15),
        layer_head(59, 2),
        layer_head(55, 21),
        layer_head(39, 23),
        layer_head(55, 9),
        layer_head(39, 10),
        layer_head(63, 0),
        layer_head(35, 16),
        layer_head(55, 14),
        layer_head(51, 6),
        layer_head(47, 17),
        layer_head(51, 12),
        layer_head(59, 12),
        layer_head(55, 18),
        layer_head(47, 15),
        layer_head(43, 13),
        layer_head(59, 9),
        layer_head(63, 1),
        layer_head(59, 14),
        layer_head(47, 18),
        layer_head(43, 9),
    ],
    max_keys: 4_554,
};

/// The calibration table: artifact content hash → the vote that locates.
/// One entry, the served artifact, beside its pointing calibration.
const CALIBRATED: &[([u8; 32], LocateCalibration)] =
    &[(crate::pointing::SERVED_NVFP4_27B, SERVED_NVFP4_27B_VOTE)];

/// The `locate` calibration recorded for `artifact`, or `None` when nobody
/// calibrated it — and then `/v1/decide` refuses a `locate` rather than read
/// heads chosen for another model. Looked up by the full 32-byte hash.
pub fn calibration(artifact: ArtifactHash) -> Option<LocateCalibration> {
    CALIBRATED
        .iter()
        .find(|(hash, _)| hash == artifact.as_bytes())
        .map(|&(_, calibration)| calibration)
}

/// Every artifact hash the table names — for the test that fails when the
/// served artifact changes without a recalibration.
pub fn calibrated_artifacts() -> impl Iterator<Item = ArtifactHash> {
    CALIBRATED.iter().map(|(hash, _)| ArtifactHash::from_bytes(*hash))
}

/// One head's **share** of each segment: the softmax of its scores over the
/// whole span, summed over the segment's keys — `None` for a segment that
/// owns no key. Keys no segment owns (the separators between lines, the
/// quotes and commas of an array) take their part of the softmax and credit
/// nobody, so the shares may sum to less than one.
///
/// In the reference's arithmetic: `exp(s - max s)` in f64, each segment's
/// mass as a difference of the running sum at its two ends, divided by the
/// total, and **stored as an f32** — the precision `score.py` kept them in,
/// and so the one every vote in the findings was cast with.
pub fn segment_shares(scores: &[f32], keys: &[Option<Range<usize>>]) -> Vec<Option<f32>> {
    let top = scores.iter().fold(f64::NEG_INFINITY, |a, &s| a.max(f64::from(s)));
    let mut running = Vec::with_capacity(scores.len() + 1);
    running.push(0.0f64);
    let mut sum = 0.0f64;
    for &score in scores {
        sum += (f64::from(score) - top).exp();
        running.push(sum);
    }
    keys.iter()
        .map(|range| range.as_ref().map(|r| ((running[r.end] - running[r.start]) / sum) as f32))
        .collect()
}

/// What [`read_vote`] reads out of the heads' rows.
#[derive(Debug, Clone, PartialEq)]
pub struct VoteReading {
    /// The segment each head voted for, in the heads' order.
    pub voted: Vec<usize>,
    /// Each segment's votes.
    pub votes: Vec<u32>,
    /// The answer: the most-voted segment, a tie going to the segment the
    /// best-ranked head (the earliest in the heads' order) voted for.
    pub winner: usize,
    /// The winner's share of the votes, in `(0, 1]`. How much the heads
    /// agree — not a calibrated probability.
    pub confidence: f64,
    /// The voted segments with their share of the votes, ordered as the
    /// winner is chosen — by votes, then by the best-ranked voter — at most
    /// [`RANKING_LEN`] of them. A segment no head voted for is not ranked.
    pub ranking: Vec<(usize, f64)>,
}

/// Read the **head vote** over `heads` score rows of one span.
///
/// `question` and `baseline` are the heads' pre-softmax scores, row-major
/// `[heads][span]`, in the order the heads were ranked when they were chosen
/// (best first): `question` at the copy scaffold's last position, `baseline`
/// at the same position of the **content-free** prefill — the same state
/// and scaffold with the instruction replaced by `N/A`. `keys` is each
/// segment's keys as a range of the span, or `None` for a segment that owns
/// none.
///
/// Each head votes for the owned segment whose share of its softmax the
/// question raised most over the baseline's — the first such segment on a
/// tie, even when every difference is negative. The rule
/// `tools/locate-sets/score.py vote_reading` measured, in its arithmetic.
///
/// `None` when the rows are not whole: no heads, rows that are not `heads`
/// by one span, the two prefills disagreeing on the span, a score that is
/// not finite, a segment reaching past the span, or no segment owning a key.
/// A readout the leaf did not bring back whole is not an answer.
pub fn read_vote(
    question: &[f32],
    baseline: &[f32],
    heads: usize,
    keys: &[Option<Range<usize>>],
) -> Option<VoteReading> {
    if heads == 0 || question.len() != baseline.len() || question.len() % heads != 0 {
        return None;
    }
    let span = question.len() / heads;
    if span == 0 || question.iter().chain(baseline).any(|s| !s.is_finite()) {
        return None;
    }
    if keys.iter().flatten().any(|r| r.start >= r.end || r.end > span) || keys.iter().all(Option::is_none) {
        return None;
    }
    let mut voted = Vec::with_capacity(heads);
    for (asked, content_free) in question.chunks_exact(span).zip(baseline.chunks_exact(span)) {
        let lifted = segment_shares(asked, keys);
        let prior = segment_shares(content_free, keys);
        let mut best: Option<(usize, f64)> = None;
        for (segment, (q, na)) in lifted.iter().zip(&prior).enumerate() {
            let (Some(q), Some(na)) = (q, na) else {
                continue;
            };
            let lift = f64::from(*q) - f64::from(*na);
            // Strictly greater: the earlier segment keeps a tie.
            if best.is_none_or(|(_, most)| lift > most) {
                best = Some((segment, lift));
            }
        }
        voted.push(best.expect("a segment owns a key").0);
    }
    let mut votes = vec![0u32; keys.len()];
    let mut first_voter = vec![usize::MAX; keys.len()];
    for (rank, &segment) in voted.iter().enumerate() {
        votes[segment] += 1;
        first_voter[segment] = first_voter[segment].min(rank);
    }
    let mut ranked: Vec<usize> = (0..keys.len()).filter(|&segment| votes[segment] > 0).collect();
    ranked.sort_by_key(|&segment| (std::cmp::Reverse(votes[segment]), first_voter[segment]));
    let share = |segment: usize| f64::from(votes[segment]) / heads as f64;
    let winner = ranked[0];
    Some(VoteReading {
        confidence: share(winner),
        ranking: ranked.iter().take(RANKING_LEN).map(|&segment| (segment, share(segment))).collect(),
        winner,
        votes,
        voted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec 19 § Phase 3, track L: the 32 heads registered before set D
    /// was read, best first, as the spec lists them.
    const REGISTERED: [&str; 32] = [
        "L39.h12", "L47.h20", "L59.h16", "L55.h17", "L59.h17", "L59.h7", "L59.h6", "L55.h0", "L59.h8", "L59.h10",
        "L63.h5", "L39.h15", "L59.h2", "L55.h21", "L39.h23", "L55.h9", "L39.h10", "L63.h0", "L35.h16", "L55.h14",
        "L51.h6", "L47.h17", "L51.h12", "L59.h12", "L55.h18", "L47.h15", "L43.h13", "L59.h9", "L63.h1", "L59.h14",
        "L47.h18", "L43.h9",
    ];

    fn served() -> ArtifactHash {
        crate::pointing::calibrated_artifacts().next().expect("the served artifact is calibrated for point")
    }

    #[test]
    fn the_served_artifact_votes_with_the_registered_heads() {
        let calibration = calibration(served()).expect("the served artifact is calibrated for locate");
        let names: Vec<String> = calibration.heads.iter().map(ToString::to_string).collect();
        assert_eq!(names, REGISTERED, "all 32, in the order they were ranked");
        assert!(calibration.heads.len() <= crate::pointing::MAX_ROW_HEADS);
        // Spec 18's rule 3 on set D (`docs/findings/2026-09-27-locate-by-head-vote-go.md`).
        assert_eq!(calibration.max_keys, 4_554);
    }

    /// Keyed by the whole hash: heads chosen for one model are never read on
    /// another, and a load nobody calibrated has no `locate`.
    #[test]
    fn any_other_artifact_has_no_locate() {
        let mut other = *served().as_bytes();
        other[0] ^= 1;
        assert_eq!(calibration(ArtifactHash::from_bytes(other)), None);
        assert_eq!(calibration(ArtifactHash::UNKNOWN), None);
        assert_eq!(calibrated_artifacts().collect::<Vec<_>>(), vec![served()]);
    }
}
