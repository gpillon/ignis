//! The **head vote** that answers a `locate` (spec 18 phase B read as spec
//! 19's track L registered it, GitHub #275): a pure host function of the
//! vote's heads' score rows at the copy scaffold, the same rows at the
//! content-free prefill, and each segment's keys.
//!
//! Two kinds of case. The **golden** ones are written by
//! `tools/locate-sets/score.py golden` — the arithmetic every number in the
//! findings was measured with — three of them real questions of set D read
//! with the registered heads, the rest planted on the rule's edges. The
//! **table** ones pin the rule by hand.

use std::ops::Range;

use ignis_core::locate::{VoteReading, read_vote, segment_shares};
use serde_json::Value;

struct Case {
    name: String,
    heads: usize,
    keys: Vec<Option<Range<usize>>>,
    q: Vec<f32>,
    na: Vec<f32>,
    voted: Vec<usize>,
    votes: Vec<u32>,
    winner: usize,
    confidence: f64,
    ranking: Vec<(usize, f64)>,
}

/// f16 little-endian bytes, hex, into f32: every f16 is exact in f32, so
/// the port reads the scores the reference read.
fn f16_hex(text: &str) -> Vec<f32> {
    let bytes: Vec<u8> = (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).expect("hex"))
        .collect();
    bytes
        .chunks_exact(2)
        .map(|pair| {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let exponent = i32::from((bits >> 10) & 0x1f);
            let mantissa = f32::from(bits & 0x3ff);
            match exponent {
                0 => sign * mantissa * 2f32.powi(-24),
                31 => panic!("a non-finite f16 in the golden file"),
                _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
            }
        })
        .collect()
}

impl Case {
    fn from_json(case: &Value) -> Self {
        let whole = |value: &Value| value.as_u64().expect("a whole number") as usize;
        Self {
            name: case["name"].as_str().expect("name").to_owned(),
            heads: case["heads"].as_array().expect("heads").len(),
            keys: case["keys"]
                .as_array()
                .expect("keys")
                .iter()
                .map(|k| k.as_array().map(|pair| whole(&pair[0])..whole(&pair[1])))
                .collect(),
            q: f16_hex(case["q"].as_str().expect("q")),
            na: f16_hex(case["na"].as_str().expect("na")),
            voted: case["voted"].as_array().expect("voted").iter().map(whole).collect(),
            votes: case["votes"].as_array().expect("votes").iter().map(|v| whole(v) as u32).collect(),
            winner: whole(&case["winner"]),
            confidence: case["confidence"].as_f64().expect("confidence"),
            ranking: case["ranking"]
                .as_array()
                .expect("ranking")
                .iter()
                .map(|pair| (whole(&pair[0]), pair[1].as_f64().expect("a share")))
                .collect(),
        }
    }
}

#[test]
fn the_vote_reproduces_the_reference() {
    let golden: Value =
        serde_json::from_str(include_str!("fixtures/locate_vote.json")).expect("the golden file parses");
    let cases: Vec<Case> = golden["cases"].as_array().expect("cases").iter().map(Case::from_json).collect();
    assert!(cases.len() >= 9, "three real questions and six edges");
    for case in &cases {
        let reading = read_vote(&case.q, &case.na, case.heads, &case.keys)
            .unwrap_or_else(|| panic!("{}: the vote reads", case.name));
        assert_eq!(reading.voted, case.voted, "{}: each head's vote", case.name);
        assert_eq!(reading.votes, case.votes, "{}: the votes", case.name);
        assert_eq!(reading.winner, case.winner, "{}: the winner", case.name);
        assert!((reading.confidence - case.confidence).abs() < 1e-12, "{}: the confidence", case.name);
        assert_eq!(reading.ranking.len(), case.ranking.len(), "{}: the ranking", case.name);
        for (got, want) in reading.ranking.iter().zip(&case.ranking) {
            assert_eq!(got.0, want.0, "{}: the ranking's order", case.name);
            assert!((got.1 - want.1).abs() < 1e-12, "{}: a ranked share", case.name);
        }
    }
}

/// One head's shares: the softmax over the whole span, summed per segment.
/// Keys no segment owns take their part of the mass and credit nobody, and
/// a segment that owns no key has no share.
#[test]
fn a_share_is_the_segments_part_of_the_softmax_over_the_span() {
    // Five keys: segment 0 owns keys 0-1, key 2 is a separator, segment 2
    // owns keys 3-4; segment 1 owns none. ln 2 on key 3 doubles its weight.
    let ln2 = std::f32::consts::LN_2;
    let scores = [0.0, 0.0, 0.0, ln2, 0.0];
    let keys = [Some(0..2), None, Some(3..5)];
    let shares = segment_shares(&scores, &keys);
    // Weights 1, 1, 1, 2, 1 over a total of 6.
    assert_eq!(shares.len(), 3);
    assert!((f64::from(shares[0].expect("owned")) - 2.0 / 6.0).abs() < 1e-7);
    assert_eq!(shares[1], None, "a segment that owns no key has no share");
    assert!((f64::from(shares[2].expect("owned")) - 3.0 / 6.0).abs() < 1e-7);
}

fn rows(heads: &[&[f32]]) -> Vec<f32> {
    heads.iter().flat_map(|row| row.iter().copied()).collect()
}

/// Three segments of two keys each, back to back.
fn three() -> Vec<Option<Range<usize>>> {
    vec![Some(0..2), Some(2..4), Some(4..6)]
}

#[test]
fn every_head_on_one_segment_is_a_unanimous_answer() {
    let peak: &[f32] = &[0.0, 0.0, 5.0, 5.0, 0.0, 0.0];
    let flat: &[f32] = &[0.0; 6];
    let reading = read_vote(&rows(&[peak, peak, peak]), &rows(&[flat, flat, flat]), 3, &three()).expect("reads");
    assert_eq!(
        reading,
        VoteReading {
            voted: vec![1, 1, 1],
            votes: vec![0, 3, 0],
            winner: 1,
            confidence: 1.0,
            ranking: vec![(1, 1.0)],
        }
    );
}

/// The baseline is subtracted before a head votes: a segment the head
/// favours with no question at all loses what it had without one.
#[test]
fn the_content_free_prefill_is_subtracted_before_a_head_votes() {
    // The question puts most mass on segment 0, but the content-free
    // prefill puts even more there: the question's own lift is segment 2.
    let q: &[f32] = &[4.0, 4.0, 0.0, 0.0, 2.0, 2.0];
    let na: &[f32] = &[6.0, 6.0, 0.0, 0.0, 0.0, 0.0];
    let reading = read_vote(q, na, 1, &three()).expect("reads");
    assert_eq!(reading.winner, 2);
}

/// Every difference negative — the question moved its mass onto keys no
/// segment owns — still names the least-negative segment rather than no
/// answer.
#[test]
fn a_question_that_lost_mass_everywhere_still_votes() {
    // Segments at keys 0-1, 3-4 and 6-7; keys 2 and 5 are separators.
    let keys = [Some(0..2), Some(3..5), Some(6..8)];
    let q: &[f32] = &[0.0, 0.0, 9.0, 0.0, 0.0, 9.0, 1.0, 1.0];
    let na: &[f32] = &[0.0; 8];
    let reading = read_vote(q, na, 1, &keys).expect("reads");
    assert_eq!(reading.voted, vec![2], "the segment the question lost least on");
}

/// Two segments with two votes each: the one the best-ranked head named
/// wins, and the ranking orders the tie the same way.
#[test]
fn a_tie_goes_to_the_segment_the_best_ranked_head_named() {
    let on = |segment: usize| -> Vec<f32> {
        let mut row = vec![0.0; 6];
        row[2 * segment] = 8.0;
        row[2 * segment + 1] = 8.0;
        row
    };
    let flat = [0.0f32; 6];
    let q = rows(&[&on(2), &on(0), &on(0), &on(2)]);
    let na = rows(&[&flat, &flat, &flat, &flat]);
    let reading = read_vote(&q, &na, 4, &three()).expect("reads");
    assert_eq!(reading.votes, vec![2, 0, 2]);
    assert_eq!(reading.winner, 2, "head 0 named segment 2");
    assert_eq!(reading.ranking, vec![(2, 0.5), (0, 0.5)]);
    assert_eq!(reading.confidence, 0.5);
}

/// A flat row credits each segment by its width, and the difference of two
/// flat rows is zero everywhere: the first segment takes the vote.
#[test]
fn a_flat_row_against_a_flat_baseline_votes_for_the_first_segment() {
    let flat = [1.5f32; 6];
    let reading = read_vote(&flat, &flat, 1, &three()).expect("reads");
    assert_eq!(reading.voted, vec![0]);
}

/// The ranking holds at most five segments, every one of them voted for.
#[test]
fn the_ranking_is_at_most_five_voted_segments() {
    let keys: Vec<Option<Range<usize>>> = (0..8).map(|s| Some(s..s + 1)).collect();
    let one_hot = |segment: usize| -> Vec<f32> { (0..8).map(|k| if k == segment { 7.0 } else { 0.0 }).collect() };
    let heads: Vec<Vec<f32>> = [6, 5, 4, 3, 2, 1, 6].iter().map(|&s| one_hot(s)).collect();
    let q: Vec<f32> = heads.iter().flatten().copied().collect();
    let na = vec![0.0f32; q.len()];
    let reading = read_vote(&q, &na, 7, &keys).expect("reads");
    assert_eq!(reading.winner, 6);
    let order: Vec<usize> = reading.ranking.iter().map(|&(segment, _)| segment).collect();
    assert_eq!(order, vec![6, 5, 4, 3, 2], "by votes, then by the best-ranked voter");
}

/// What the leaf did not bring back whole is not read: rows of the wrong
/// size, a non-finite score, no heads, a segment past the span, or no
/// segment that owns a key.
#[test]
fn a_readout_that_is_not_whole_is_not_read() {
    let row = [0.0f32; 6];
    assert!(read_vote(&row, &row, 0, &three()).is_none(), "no heads");
    assert!(read_vote(&row, &row[..5], 1, &three()).is_none(), "the two prefills disagree on the span");
    assert!(read_vote(&row[..5], &row[..5], 2, &three()).is_none(), "rows that are not heads x span");
    let mut bad = row;
    bad[3] = f32::NAN;
    assert!(read_vote(&bad, &row, 1, &three()).is_none(), "a score that is not a number");
    assert!(read_vote(&row, &row, 1, &[Some(0..2), Some(4..7)]).is_none(), "a segment past the span");
    assert!(read_vote(&row, &row, 1, &[None, None]).is_none(), "no segment owns a key");
}
