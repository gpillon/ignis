//! The `Compute` seam's **repair** (GitHub #586, spec #287), over the mock.
//!
//! What is pinned here is the contract [`Compute::repair`] states, as
//! [`MockCompute`] models it:
//!
//! - a repair acts on the **waiting draw** — what the prefill or the last
//!   plain round drew, which the next round consumes as its anchor — and the
//!   repaired token **replaces** it: the next round commits the repaired
//!   token where the draw would have been, and the stream after it goes on;
//! - the prefill's own draw, which leaves with the prefill (#518), is
//!   repaired the same way;
//! - the mock's pick is deterministic and documented (the draw when the mask
//!   permits it, the mask's lowest id otherwise), so a scheduler's tests can
//!   say which token a miss becomes;
//! - an early stop is a draw like any other, and a repaired one does not end
//!   the request;
//! - what the leaf refuses, the mock refuses: a request with no waiting draw,
//!   and a draw the leaf redirected itself.

use ignis_core::constrained::PermittedMask;
use ignis_core::mock::MockCompute;
use ignis_core::scheduler::{Compute, DecodeJob, PrefillJob, RepairJob};
use ignis_core::types::{DecodeParams, FinishReason, RequestId, TokenId};

const REQUEST: RequestId = 586;
/// A vocabulary the masks below are built over; the mock's free draws fall
/// past it, so every free draw is a miss against any of them.
const VOCAB: u32 = 4096;

fn prefill_job(permitted: Option<Vec<TokenId>>, reasoning_close: Option<TokenId>) -> PrefillJob {
    PrefillJob {
        request: REQUEST,
        tokens: vec![1, 2, 3],
        context_tokens: 64,
        start_position: 0,
        params: DecodeParams::default(),
        shared_prefix: None,
        publish_prefix: None,
        checkpoint: None,
        capture_checkpoint: None,
        multimodal: None,
        readout: None,
        permitted: permitted.map(Into::into),
        attention: None,
        reasoning_close,
        drafter_floor: 0,
        next_prompt_token: None,
    }
}

fn round(anchor_emitted: bool) -> DecodeJob {
    DecodeJob {
        request: REQUEST,
        lane: 0,
        params: DecodeParams::default(),
        remaining_tokens: 8,
        permitted: None,
        reasoning_close: None,
        anchor_emitted,
        checked: false,
    }
}

fn repair_to(ids: &[TokenId]) -> RepairJob {
    RepairJob {
        request: REQUEST,
        params: DecodeParams::default(),
        permitted: PermittedMask::from_ids(VOCAB, ids).expect("a legal mask"),
    }
}

/// The tokens one round over `REQUEST` emits.
fn emitted(compute: &MockCompute, anchor_emitted: bool) -> Vec<TokenId> {
    compute.decode_step(&[round(anchor_emitted)]).expect("the round")[0].tokens.clone()
}

/// The prefill's free draw, emitted at the prefill (#518).
fn prefill(compute: &MockCompute) -> TokenId {
    let outcomes = compute.prefill_step(&[prefill_job(None, None)]).expect("the prefill");
    outcomes[0].emitted.as_ref().expect("the prompt's last job emits its draw").tokens[0]
}

#[test]
fn a_repair_replaces_the_waiting_draw_the_next_round_consumes() {
    let compute = MockCompute::new();
    assert_eq!(prefill(&compute), compute.token_for(REQUEST, 0));
    assert_eq!(emitted(&compute, true), Vec::<TokenId>::new(), "the anchor left with the prefill");

    // The round drew step 1's token, which the next round would emit: a miss
    // against this mask, so the repair takes the mask's lowest id.
    let repaired = compute.repair(&repair_to(&[300, 41, 2048])).expect("a waiting draw to repair");
    assert_eq!(repaired, 41, "the mock's pick on a miss is the mask's lowest id");
    assert_eq!(emitted(&compute, false), vec![41], "the next round consumes the repaired token as its anchor");
    assert_eq!(
        emitted(&compute, false),
        vec![compute.token_for(REQUEST, 2)],
        "and the stream goes on past the one position the repair replaced"
    );
    let calls = compute.repair_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].request, REQUEST);
    assert_eq!(calls[0].permitted.ids().collect::<Vec<_>>(), vec![41, 300, 2048]);
}

#[test]
fn the_prefills_own_draw_is_repaired_the_same_way() {
    let compute = MockCompute::new();
    let draw = prefill(&compute);
    let repaired = compute.repair(&repair_to(&[7])).expect("the prefill's draw is waiting");
    assert_ne!(draw, repaired);
    assert_eq!(repaired, 7);
    // A scheduler that repairs before it emits sends 7 and marks the anchor
    // emitted; one that sent the draw already gets the repaired anchor here.
    assert_eq!(emitted(&compute, false), vec![7], "the first round's anchor is the repaired draw");
    assert_eq!(emitted(&compute, false), vec![compute.token_for(REQUEST, 1)]);
}

#[test]
fn a_repair_keeps_a_draw_its_mask_permits_and_its_run_reports_no_probability() {
    let compute = MockCompute::new();
    // A constrained draw over small ids, so a mask can permit it.
    let outcomes = compute
        .prefill_step(&[prefill_job(Some(vec![10, 11, 12]), None)])
        .expect("the constrained prefill");
    let draw = outcomes[0].emitted.as_ref().expect("its draw").tokens[0];
    assert!([10, 11, 12].contains(&draw));
    let all: Vec<TokenId> = (0..64).collect();
    assert_eq!(
        compute.repair(&repair_to(&all)).expect("a waiting draw"),
        draw,
        "a draw the mask permits is its own masked argmax"
    );
    let outcome = compute.decode_step(&[round(false)]).expect("the round")[0].clone();
    assert_eq!(outcome.tokens, vec![draw]);
    assert!(
        outcome.probabilities.is_empty(),
        "a repaired draw carries no probability within the set it was first drawn from"
    );
}

#[test]
fn a_second_repair_of_a_position_reads_the_scores_the_draw_read() {
    let compute = MockCompute::new();
    let outcomes = compute
        .prefill_step(&[prefill_job(Some(vec![10, 11, 12]), None)])
        .expect("the constrained prefill");
    let draw = outcomes[0].emitted.as_ref().expect("its draw").tokens[0];
    assert_eq!(compute.repair(&repair_to(&[40, 41])).expect("a miss"), 40);
    assert_eq!(
        compute.repair(&repair_to(&[5, draw])).expect("again"),
        draw,
        "the draw ranks first again -- the first repair's token is not the scores"
    );
    assert_eq!(emitted(&compute, false), vec![draw], "and the last repair is what the round consumes");
}

#[test]
fn a_repaired_early_stop_does_not_end_the_request() {
    let compute = MockCompute::new();
    compute.eos_after(REQUEST, 1);
    prefill(&compute);
    assert_eq!(emitted(&compute, true), Vec::<TokenId>::new());
    // The round drew the EOS at step 1, which the mock has no id for: no
    // mask permits it, and the repair takes the mask's own token.
    assert_eq!(compute.repair(&repair_to(&[125])).expect("the EOS is waiting"), 125);
    let outcome = compute.decode_step(&[round(false)]).expect("the round")[0].clone();
    assert_eq!(outcome.tokens, vec![125]);
    assert_eq!(outcome.finish, None, "the stop was repaired away");
}

#[test]
fn without_a_repair_the_early_stop_ends_it() {
    // The control for the test above: the same script, unrepaired.
    let compute = MockCompute::new();
    compute.eos_after(REQUEST, 1);
    prefill(&compute);
    emitted(&compute, true);
    let outcome = compute.decode_step(&[round(false)]).expect("the round")[0].clone();
    assert_eq!(outcome.finish, Some(FinishReason::Stop));
}

/// The leaf keeps a draw's scores until a later round or prefill draw writes
/// their column, so a repair follows the call that drew; the mock holds a
/// scheduler to that by keeping them until its next call of either kind.
#[test]
fn the_scores_last_until_the_next_call_on_the_backend() {
    let compute = MockCompute::new();
    prefill(&compute);
    compute
        .prefill_step(&[PrefillJob { request: REQUEST + 1, ..prefill_job(None, None) }])
        .expect("another request's prefill");
    assert!(
        compute.repair(&repair_to(&[1])).is_err(),
        "another call came between the draw and its repair"
    );
    emitted(&compute, true);
    assert!(compute.repair(&repair_to(&[1])).is_ok(), "the round's own draw is repaired right after it");
    assert!(compute.repair(&repair_to(&[2])).is_ok(), "and again, from the same scores, before the next call");
}

/// A verify round's draw keeps no scores a repair reads: the verify commit
/// whose cut the host decides repairs those (#590).
#[test]
fn a_speculative_rounds_draw_keeps_no_scores() {
    let compute = MockCompute::with_runs(&[3]);
    prefill(&compute);
    assert_eq!(emitted(&compute, true).len(), 2, "a run of three, its anchor sent with the prefill");
    assert!(compute.repair(&repair_to(&[1])).is_err());
}

#[test]
fn a_repair_is_refused_where_the_leaf_refuses_it() {
    let compute = MockCompute::new();
    assert!(
        compute.repair(&repair_to(&[1])).is_err(),
        "a request never prefilled has no waiting draw"
    );

    // A draw the leaf redirected (GitHub #315) keeps no scores.
    const CLOSE: TokenId = 99;
    compute.eos_after(REQUEST, 0);
    let outcomes = compute
        .prefill_step(&[prefill_job(None, Some(CLOSE))])
        .expect("a prefill inside its reasoning block");
    assert!(outcomes[0].reasoning_redirected, "the EOS it drew became the close");
    assert!(compute.repair(&repair_to(&[1])).is_err(), "a redirected draw is refused");

    compute.release(REQUEST);
    assert!(compute.repair(&repair_to(&[1])).is_err(), "a released request has nothing to repair");
    assert!(compute.repair_calls().is_empty(), "and no refused repair is recorded");
}
