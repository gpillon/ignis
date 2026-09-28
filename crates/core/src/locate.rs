//! `locate`: which **segment** of a text state — a line of a string, an
//! element of an array — an instruction names.
//!
//! Two methods (ADR 0041, ADR 0042):
//!
//! - **the head vote** (spec `docs/specs/decide/18-locate-by-attention.md`
//!   phase B read as spec 19's track L registered it — GitHub #275, ADR
//!   0041), up to the length it was measured at;
//! - **the shortlist** (spec `docs/specs/decide/22-locate-by-copy-over-a-folded-state.md`,
//!   GitHub #278, ADR 0042): attention heads narrow a very long text to a few
//!   candidates ([`reading`]) — after [`fold`]ing a log into templates, window
//!   by window for prose and record arrays — and a labelled `choice` decides
//!   among them ([`render`]), with no token generated; the same `choice` says
//!   when **nothing** answers ([`found_log`], [`found_by_none`]).
//!
//! **The vote.** The model already knows before it writes anything: at the
//! copy scaffold `{"quote":"`, a few dozen heads of the later GQA layers look
//! at the segment it is about to copy. No one of them is reliable, but they
//! fail on different questions, and a majority of them finds the segment as
//! often as a `choice` over labelled segments does
//! (`docs/findings/2026-09-27-locate-by-head-vote-go.md`). What lives here is
//! the host's whole part of it:
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

pub mod fold;
pub mod reading;
pub mod render;

use std::ops::Range;

use crate::identity::ArtifactHash;
use crate::pointing::{PointingHead, layer_head};

/// How many segments a `locate`'s ranking names at most.
pub const RANKING_LEN: usize = 5;

/// Which reading a `locate`'s text gets (spec 22 § Solution): the kind a
/// caller names, or the one `auto` tells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Log,
    Prose,
    Records,
}

/// How a `locate` is answered (spec 22).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// The heads narrow the text to a few candidates, a labelled `choice`
    /// decides (ADR 0042).
    Shortlist,
    /// The head vote, as served before (ADR 0041).
    Vote,
}

/// What a `locate`'s text is read as (spec 22).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// Folded into templates and their values first ([`fold`]).
    TemplateFold,
    /// Read as it is.
    None,
}

impl Kind {
    /// The wire's spelling.
    pub fn label(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Prose => "prose",
            Self::Records => "records",
        }
    }

    /// The compression a `locate` of this kind gets when it names none: a
    /// log is folded, prose and records are read as they are.
    pub fn default_compression(self) -> Compression {
        match self {
            Self::Log => Compression::TemplateFold,
            Self::Prose | Self::Records => Compression::None,
        }
    }
}

impl Method {
    /// The wire's spelling.
    pub fn label(self) -> &'static str {
        match self {
            Self::Shortlist => "shortlist",
            Self::Vote => "vote",
        }
    }
}

impl Compression {
    /// The wire's spelling.
    pub fn label(self) -> &'static str {
        match self {
            Self::TemplateFold => "template_fold",
            Self::None => "none",
        }
    }
}

/// Templates a fold's level 1 keeps for its labelled `choice`.
pub const SHORTLIST_TEMPLATES: usize = 5;
/// Candidates every other shortlist keeps: a fold's level-2 rows, or lines,
/// sentences or records read without a fold.
pub const SHORTLIST_LEN: usize = 16;
/// The share a candidate of the last `choice` needs to be a **pointer** —
/// chosen on spec 23's prose development split and frozen there.
pub const POINTER_SHARE: f64 = 0.05;
/// `found` below this names no segment (spec 22 § Not found). Not a
/// calibrated probability: the rule the third research round measured.
pub const FOUND_THRESHOLD: f64 = 0.5;
/// Segments with content `auto` folds to tell a log from prose.
pub const AUTO_SEGMENTS: usize = 2_000;
/// The share of those in shared templates at which `auto` says `log`.
pub const AUTO_LOG_SHARE: f64 = 0.5;

/// The option a log's or a record array's last `choice` adds for "not
/// found".
pub const NONE_LINE: &str = "No line of the evidence answers the criterion";
/// The option prose's last `choice` adds for "not found".
pub const NONE_SENTENCE: &str = "No sentence of the evidence answers the criterion";
/// The yes/no a log's last request adds, before the instruction as sent.
pub const FOUND_QUESTION: &str = "Is there a line in the evidence that answers this question: ";

/// `found` for a folded log (spec 22 § Not found): the "none" option's
/// probability `p_none` and the yes/no's `p_yes`, averaged as
/// `(1 - p_none + p_yes) / 2`.
pub fn found_log(p_none: f64, p_yes: f64) -> f64 {
    (1.0 - p_none + p_yes) / 2.0
}

/// `found` for prose and records: `1 - p_none`.
pub fn found_by_none(p_none: f64) -> f64 {
    1.0 - p_none
}

/// What one artifact is calibrated with for `locate`: the heads each method
/// reads, best first, the longest span the vote was measured to hold at, and
/// the window the shortlist's readings are cut into.
///
/// The rest of the reading is fixed, not calibrated: the copy scaffold
/// `{"quote":"`, the content-free prefill subtracted, one vote per head (the
/// vote) or each head's lift standardized and summed (the shortlist). Spec 19
/// phase 0 chose those on the development sets for every configuration it
/// tried, and a recalibration repeats its procedure
/// (`tools/locate-sets/README.md`) to choose the heads and measure the
/// ceiling again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocateCalibration {
    /// The heads that vote, in the order they were ranked when chosen — the
    /// order a tie is broken in, and the order the leaf's rows come back in.
    /// Also the **sum heads**: the shortlist's reading of prose (spec 22).
    pub heads: &'static [PointingHead],
    /// The **end heads** (spec 22): the shortlist's reading of logs and
    /// records, at each segment's closing keys, in the order they were
    /// ranked when chosen.
    pub end_heads: &'static [PointingHead],
    /// `LOCATE_MAX_KEYS`: the most keys of state the **vote** reads. Spec
    /// 18's rule 3 — the longest measured length whose top-1 stays within
    /// five points of the shortest's — on the set that judged the vote. A
    /// longer span is refused before any prefill, never answered unmeasured.
    pub max_keys: u32,
    /// `LOCATE_WINDOW_KEYS` (spec 22): the most keys of text one of the
    /// shortlist's readings reads in one prefill. A longer text is read in
    /// windows of at most this many; the rows' room reserved at load is
    /// sized for it.
    pub window_keys: u32,
}

impl LocateCalibration {
    /// The heads `reading` is read with: the sum heads or the end heads.
    pub fn heads_for(&self, reading: reading::Reading) -> &'static [PointingHead] {
        match reading {
            reading::Reading::Sum => self.heads,
            reading::Reading::End => self.end_heads,
        }
    }
}

/// The served NVFP4 27B's vote: the 32 heads with the most training hits as
/// one-head readings on sets A+B, fitted by spec 19 phase 0's procedure
/// (`.scratch/locate/phase0/vote-choice.json`, recorded in spec 19 § Phase
/// 3, track L before set D was read), and set D's ceiling
/// (`docs/findings/2026-09-27-locate-by-head-vote-go.md`).
///
/// Its end heads are spec 23's (`.scratch/locate/zd/endheads.json`): the 32
/// with the best single-head top-1 on sets A+B at a line's last key, the
/// next line's first key or the separator
/// (`docs/findings/2026-09-28-zero-decode-locate-exploration.md`); its
/// window is the length spec 23 and the records round read one prefill at
/// (`docs/findings/2026-09-28-zero-decode-locate-very-long-logs-and-prose.md`).
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
    end_heads: &[
        layer_head(47, 20),
        layer_head(47, 4),
        layer_head(51, 4),
        layer_head(47, 17),
        layer_head(47, 1),
        layer_head(51, 12),
        layer_head(51, 23),
        layer_head(47, 5),
        layer_head(47, 3),
        layer_head(47, 15),
        layer_head(39, 15),
        layer_head(47, 9),
        layer_head(47, 13),
        layer_head(51, 16),
        layer_head(39, 0),
        layer_head(39, 12),
        layer_head(43, 22),
        layer_head(43, 20),
        layer_head(47, 2),
        layer_head(39, 23),
        layer_head(55, 13),
        layer_head(43, 9),
        layer_head(43, 18),
        layer_head(51, 17),
        layer_head(55, 23),
        layer_head(43, 7),
        layer_head(51, 2),
        layer_head(55, 20),
        layer_head(47, 23),
        layer_head(35, 18),
        layer_head(43, 8),
        layer_head(43, 14),
    ],
    max_keys: 4_554,
    window_keys: 200_000,
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
/// total, and **stored as an f32** — the precision `score.py` kept them in.
/// The calibration read the harness's f16 dumps where the served path reads
/// the leaf's f32 rows, so a near-tie can vote differently from the dump; the
/// served acceptance measured the path as it serves
/// (`docs/findings/2026-09-27-locate-through-decide.md`).
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

    /// Spec 22 § The heads and their readings: the end heads in rank order
    /// (spec 23's `endheads.json`), the sum heads the vote's, a 200,000-key
    /// window — and at most the rows the leaf reads at once.
    #[test]
    fn the_served_artifact_reads_the_shortlist_with_spec_22s_heads() {
        const END_HEADS: [&str; 32] = [
            "L47.h20", "L47.h4", "L51.h4", "L47.h17", "L47.h1", "L51.h12", "L51.h23", "L47.h5", "L47.h3", "L47.h15",
            "L39.h15", "L47.h9", "L47.h13", "L51.h16", "L39.h0", "L39.h12", "L43.h22", "L43.h20", "L47.h2", "L39.h23",
            "L55.h13", "L43.h9", "L43.h18", "L51.h17", "L55.h23", "L43.h7", "L51.h2", "L55.h20", "L47.h23", "L35.h18",
            "L43.h8", "L43.h14",
        ];
        let calibration = calibration(served()).expect("the served artifact is calibrated for locate");
        let names: Vec<String> = calibration.end_heads.iter().map(ToString::to_string).collect();
        assert_eq!(names, END_HEADS);
        assert_eq!(calibration.heads_for(reading::Reading::End), calibration.end_heads);
        assert_eq!(calibration.heads_for(reading::Reading::Sum), calibration.heads);
        assert!(calibration.end_heads.len() <= crate::pointing::MAX_ROW_HEADS);
        assert_eq!(calibration.window_keys, 200_000);
    }

    /// Spec 22 § The wire: a kind names its default compression.
    #[test]
    fn a_log_is_folded_by_default_and_prose_and_records_are_not() {
        assert_eq!(Kind::Log.default_compression(), Compression::TemplateFold);
        assert_eq!(Kind::Prose.default_compression(), Compression::None);
        assert_eq!(Kind::Records.default_compression(), Compression::None);
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
