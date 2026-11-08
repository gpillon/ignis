//! The rule the multimodal speculative acceptance (`vision_dflash2_gpu.rs`,
//! AC 3) judges a split by (`support/round_shapes.rs`, GitHub #534, glossary
//! Near-tie and Round shape), on given logits: no GPU. The logits are BF16
//! values, as both probes read them.

#[path = "support/round_shapes.rs"]
mod round_shapes;

use round_shapes::{bf16_ulp, judge, span_tokens, Numerics, Picks, Shape, Verdict, VerifyRound};

fn shape(decode_pick: f32, verify_pick: f32) -> Shape {
    Shape { decode_pick, verify_pick }
}

/// A probe reads the two picks' logits out of one row, in either form the
/// probes read: the decode round's pick first, whatever their order in the
/// vocabulary.
#[test]
fn the_picks_read_their_shape_from_a_row() {
    let picks = Picks { decode: 2, verify: 0 };
    assert_eq!(picks.shape(&[25.25, 7.0, 28.75]), shape(28.75, 25.25));
    // 25.25 and 28.75 in BF16.
    assert_eq!(picks.shape_bf16(&[0x41CA, 0x40E0, 0x41E6]), shape(28.75, 25.25));
}

/// One BF16 step at the value's binade: 8 significant bits.
#[test]
fn the_ulp_is_one_bf16_step_at_the_values_binade() {
    assert_eq!(bf16_ulp(1.0), 1.0 / 128.0);
    assert_eq!(bf16_ulp(25.0), 0.125);
    assert_eq!(bf16_ulp(28.75), 0.125);
    assert_eq!(bf16_ulp(31.75), 0.125);
    assert_eq!(bf16_ulp(32.0), 0.25);
    assert_eq!(bf16_ulp(-40.0), 0.25);
}

/// The diagnosed split (`colour/detail`, after 3 agreed tokens, decode 2099
/// against verify 8851) under ignis's own GDN prefill: the decode shape reads
/// 28.75 / 25.50 (+3.25), the verify shape 25.00 / 25.25 (-0.25). The two
/// shapes disagree and the verify shape sides with the verify round.
#[test]
fn the_colour_detail_split_is_numerics_under_ignis_gdn_prefill() {
    let decode = shape(28.75, 25.50);
    let verify = shape(25.00, 25.25);
    assert_eq!((decode.gap(), verify.gap()), (3.25, -0.25));
    assert_eq!(
        judge(decode, verify),
        Verdict::Numerics(Numerics { shapes_disagree: true, verify_shape_sides_with_verify: true, within_ulp: false })
    );
}

/// The same split under the vendored GDN prefill: the decode shape is an
/// exact tie (27.625 / 27.625), the verify shape 27.50 / 28.625 (-1.125).
/// A shape at an exact tie names no winner, so the shapes do not disagree;
/// the tie is within an ulp, and the verify shape sides with the verify round.
#[test]
fn the_colour_detail_split_is_numerics_under_the_vendored_gdn_prefill() {
    let decode = shape(27.625, 27.625);
    let verify = shape(27.50, 28.625);
    assert_eq!((decode.gap(), verify.gap()), (0.0, -1.125));
    assert_eq!(
        judge(decode, verify),
        Verdict::Numerics(Numerics { shapes_disagree: false, verify_shape_sides_with_verify: true, within_ulp: true })
    );
}

/// The one split refused: both shapes side with the decode round, each by
/// more than one ulp at its larger logit -- the verify round picked a token
/// its own shape, computed outside the round, ranks lower.
#[test]
fn a_verify_shape_siding_with_the_decode_round_beyond_an_ulp_is_a_fault() {
    assert_eq!(judge(shape(28.75, 25.50), shape(28.0, 25.0)), Verdict::VerifyFault);
    // At 25.25 one ulp is 0.125: a lead of one ulp is a tie within the
    // computation's rounding, a lead of two is not.
    assert_eq!(
        judge(shape(28.75, 25.50), shape(25.25, 25.125)),
        Verdict::Numerics(Numerics { shapes_disagree: false, verify_shape_sides_with_verify: false, within_ulp: true })
    );
    assert_eq!(judge(shape(28.75, 25.50), shape(25.25, 25.0)), Verdict::VerifyFault);
}

/// The tolerance is the ulp at the pair's larger magnitude, so one gap of
/// 0.25 is a tie at 32 and two ulps at 31.75, and a negative pair takes its
/// scale from its magnitude.
#[test]
fn the_ulp_follows_the_larger_logits_magnitude() {
    let decode = shape(28.75, 25.50);
    assert_eq!(
        judge(decode, shape(32.0, 31.75)),
        Verdict::Numerics(Numerics { shapes_disagree: false, verify_shape_sides_with_verify: false, within_ulp: true })
    );
    assert_eq!(judge(decode, shape(31.75, 31.5)), Verdict::VerifyFault);
    assert_eq!(
        judge(decode, shape(-31.75, -32.0)),
        Verdict::Numerics(Numerics { shapes_disagree: false, verify_shape_sides_with_verify: false, within_ulp: true })
    );
}

/// Either shape within an ulp of a tie makes the split numerics: the decode
/// round's own pick was a tie in its shape, whatever the verify shape says.
#[test]
fn a_decode_shape_within_an_ulp_of_a_tie_is_numerics() {
    assert_eq!(
        judge(shape(25.125, 25.0), shape(26.0, 25.0)),
        Verdict::Numerics(Numerics { shapes_disagree: false, verify_shape_sides_with_verify: false, within_ulp: true })
    );
}

/// A verify shape that ranks the verify's pick higher passes even where the
/// decode shape, computed one token at a time, sides with the decode round by
/// many ulps: that is the decode round's own shape, not a fault of the verify.
#[test]
fn a_verify_shape_ranking_the_verify_pick_higher_is_numerics() {
    assert_eq!(
        judge(shape(30.0, 20.0), shape(20.0, 30.0)),
        Verdict::Numerics(Numerics { shapes_disagree: true, verify_shape_sides_with_verify: true, within_ulp: false })
    );
    assert_eq!(
        judge(shape(25.0, 20.0), shape(24.0, 24.0)),
        Verdict::Numerics(Numerics { shapes_disagree: false, verify_shape_sides_with_verify: true, within_ulp: true })
    );
}

/// The GPU test prints the verdict: it names every numerics clause that held.
#[test]
fn the_verdict_names_the_clauses_that_held() {
    assert_eq!(
        judge(shape(28.75, 25.50), shape(25.00, 25.25)).to_string(),
        "numerics: the shapes disagree about the winner; the verify shape ranks the verify's pick at least as high"
    );
    assert_eq!(
        judge(shape(27.625, 27.625), shape(27.50, 28.625)).to_string(),
        "numerics: the verify shape ranks the verify's pick at least as high; a shape is within one ulp of a tie"
    );
    assert_eq!(
        judge(shape(28.75, 25.50), shape(28.0, 25.0)).to_string(),
        "a verify-round fault: the verify shape sides with the decode round beyond one ulp"
    );
}

/// A speculative run of a 90-token prompt (`text[0]` at position 90): round 0
/// commits its anchor and two drafts, round 1 every column, round 2 its
/// anchor and one draft.
fn run() -> Vec<VerifyRound> {
    vec![
        VerifyRound { base: 90, extent: 7, committed: 3 },
        VerifyRound { base: 93, extent: 7, committed: 8 },
        VerifyRound { base: 101, extent: 7, committed: 2 },
    ]
}

/// A round's column `j` at `base + j` drew the token after it, so the row is
/// the drawing position's offset from the round's base. Its last committed
/// column drew its correction, the pending token the next round takes as its
/// anchor: the diagnosed split (`colour/detail`, token 3, drawn at position 92)
/// is round 0's row 2, not round 1's, whose anchor it is.
#[test]
fn the_row_is_the_column_of_the_round_that_drew_the_token() {
    let rounds = run();
    assert_eq!(rounds[0].row(90, 1), Some(0));
    assert_eq!(rounds[0].row(90, 3), Some(2));
    assert_eq!(rounds[0].row(90, 4), None);
    assert_eq!(rounds[1].row(90, 3), None);
    assert_eq!(rounds[1].row(90, 4), Some(0));
    assert_eq!(rounds[1].row(90, 11), Some(7));
    assert_eq!(rounds[1].row(90, 12), None);
    assert_eq!(rounds[2].row(90, 12), Some(0));
    assert_eq!(rounds[2].row(90, 13), Some(1));
    assert_eq!(rounds[2].row(90, 14), None);
}

/// The prompt's last position drew the first token, in both runs alike: no
/// round drew it.
#[test]
fn no_round_drew_token_0() {
    for round in run() {
        assert_eq!(round.row(90, 0), None);
    }
}

/// The decode rounds' text after the prompt: 14 tokens, `text[0]` at
/// position 90.
fn off() -> Vec<i32> {
    (100..114).collect()
}

/// A speculative run's text after a round, its pending token last: the
/// decode text before token `split`, then other tokens through token
/// `len - 1`.
fn departing(split: usize, len: usize) -> Vec<i32> {
    off()[..split].iter().copied().chain((split..len).map(|i| 900 + i as i32)).collect()
}

/// The diagnosed split is round 0's correction: the run's text after round 0,
/// its anchor, two drafts and its pending token, departs at token 3, which
/// round 0's last column drew. Round 0 keeps its snapshot; round 1, whose
/// anchor that token is, does not.
#[test]
fn the_round_whose_correction_departs_keeps_its_snapshot() {
    let rounds = run();
    assert!(rounds[0].drew_the_departure(90, &off(), &departing(3, 4)));
    assert!(!rounds[1].drew_the_departure(90, &off(), &departing(3, 12)));
}

/// A departure at an accepted draft: round 0's draws all agree, so it drops
/// its snapshot; round 1 drew token 10 at its row 6 and keeps its own; round 2
/// comes after it and keeps nothing.
#[test]
fn the_first_round_whose_draws_depart_keeps_its_snapshot() {
    let rounds = run();
    assert!(!rounds[0].drew_the_departure(90, &off(), &departing(4, 4)));
    assert!(rounds[1].drew_the_departure(90, &off(), &departing(10, 12)));
    assert_eq!(rounds[1].row(90, 10), Some(6));
    assert!(!rounds[2].drew_the_departure(90, &off(), &departing(10, 14)));
}

/// A round whose draws all agree with the decode text drops its snapshot,
/// also when its pending token lies past the decode text's end: the decode
/// rounds stopped there, so no token of theirs departs.
#[test]
fn a_round_whose_draws_all_agree_drops_its_snapshot() {
    let rounds = run();
    assert!(!rounds[0].drew_the_departure(90, &off(), &off()[..4]));
    assert!(!rounds[1].drew_the_departure(90, &off(), &off()[..12]));
    assert!(!rounds[0].drew_the_departure(90, &off()[..3], &departing(3, 4)));
}

/// A run whose very first token departs departs at the prompt's own draw: no
/// round keeps a snapshot for it.
#[test]
fn no_round_keeps_a_snapshot_for_token_0() {
    let rounds = run();
    assert!(!rounds[0].drew_the_departure(90, &off(), &departing(0, 4)));
    assert!(!rounds[1].drew_the_departure(90, &off(), &departing(0, 12)));
}

/// A span's tokens are the decode text at its positions, and past the text's
/// end its last token again: a causal span's row never reads a later column.
#[test]
fn a_span_reads_the_decode_text_and_repeats_its_last_token_past_its_end() {
    let text = [10, 11, 12, 13];
    assert_eq!(span_tokens(&text, 90, 90..93), vec![10, 11, 12]);
    assert_eq!(span_tokens(&text, 90, 92..98), vec![12, 13, 13, 13, 13, 13]);
}
