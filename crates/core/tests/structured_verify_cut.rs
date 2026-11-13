//! Speculation stays on for structured output (GitHub #590, spec #287, ADR
//! 0089): a verify round's run is checked against the request's grammar
//! before it is committed, cut at its first draft that does not pass, and
//! that draft waits to be repaired as a plain round's draw is.
//!
//! Over `MockCompute::with_runs`, whose rounds commit runs the way a
//! speculative load's verify rounds do, and whose scripted "model"
//! ([`MockCompute::script`]) writes what the test chooses. The vocabulary is
//! ASCII, one character a token; the mock's repair ranks what the script was
//! going to write next above the rest, then the vocabulary in order.

use std::sync::Arc;

use ignis_core::structured::{AnswerGrammar, StructuredKind};
use ignis_core::{
    ConcreteScheduler, DecodeParams, FinishReason, MockCompute, RequestClass, RequestId, RequestInput, SchedEvent,
    Scheduler, SchedulerConfig, SpecCounters, TokenId, PendingToken,
};
use ignis_grammar::{Grammar, GrammarCache, Vocab, VocabEntry, DEFAULT_CACHE_BYTES};

/// The model's EOS, a control token.
const STOP: TokenId = 5;
const VOCAB: TokenId = 128;

/// Printable ASCII and the line break, each its own code point.
fn json_object() -> Arc<Grammar> {
    let entries = (0..VOCAB)
        .map(|id| match id {
            STOP => VocabEntry::Control,
            10 | 32..=126 => VocabEntry::Text(vec![id as u8]),
            _ => VocabEntry::Absent,
        })
        .collect();
    GrammarCache::new(Vocab::new(entries, &[STOP]), DEFAULT_CACHE_BYTES).json_object()
}

fn encode(text: &str) -> Vec<TokenId> {
    text.bytes().map(TokenId::from).collect()
}

fn text(tokens: &[TokenId]) -> String {
    tokens.iter().map(|&t| char::from(t as u8)).collect()
}

fn scheduler(mock: &Arc<MockCompute>) -> ConcreteScheduler {
    ConcreteScheduler::with_config(SchedulerConfig { model: "m".into(), ..SchedulerConfig::default() }, mock.clone())
}

fn submit(sched: &mut ConcreteScheduler, grammar: Option<&Arc<Grammar>>, max_tokens: u32) -> RequestId {
    sched
        .submit(
            RequestInput {
                decision: None,
                multimodal: None,
                opener_tokens: None,
                user_turn_tokens: None,
                system_block_tokens: None,
                reuse_boundaries: Vec::new(),
                model: "m".into(),
                tokens: vec![1, 2, 3],
                params: DecodeParams { max_tokens: Some(max_tokens), ..DecodeParams::default() },
                constrained: None,
                forced_literal: None,
                grammar: grammar.map(|g| AnswerGrammar::new(g.clone(), StructuredKind::JsonObject)),
                warm_up: false,
            },
            RequestClass::Interactive,
        )
        .expect("submit")
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    for _ in 0..1000 {
        if sched.is_idle() {
            return events;
        }
        events.extend(sched.advance());
    }
    panic!("the scheduler never went idle: {events:?}");
}

fn emitted(events: &[SchedEvent], id: RequestId) -> Vec<TokenId> {
    events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Token { request, token } if *request == id => Some(*token),
            _ => None,
        })
        .collect()
}

fn done(events: &[SchedEvent], id: RequestId) -> (FinishReason, Option<SpecCounters>) {
    events
        .iter()
        .find_map(|e| match e {
            SchedEvent::Done { request, reason, spec, .. } if *request == id => Some((*reason, *spec)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("request {id} never finished: {events:?}"))
}

/// The request's misses, and how many of them were repaired (#593).
fn misses(events: &[SchedEvent], id: RequestId) -> (usize, usize) {
    let repairs: Vec<Option<u32>> = events
        .iter()
        .filter_map(|e| match e {
            SchedEvent::Miss { request, repair_micros } if *request == id => Some(*repair_micros),
            _ => None,
        })
        .collect();
    (repairs.len(), repairs.iter().flatten().count())
}

fn draws(text: &str) -> Vec<PendingToken> {
    encode(text).into_iter().map(PendingToken::Token).collect()
}

/// #287: the prompt's last job of a request held to a grammar is checked --
/// its draw keeps the scores a repair reads -- and no other job is: a free
/// request's prefill does no work for a repair.
#[test]
fn only_a_held_requests_last_prefill_job_is_checked() {
    let grammar = json_object();
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(&mock);
    let held = submit(&mut sched, Some(&grammar), 8);
    let free = submit(&mut sched, None, 8);
    mock.script(held, &encode("{}"));
    run_to_idle(&mut sched);
    let jobs: Vec<_> = mock.prefill_calls().into_iter().flatten().collect();
    let checked = |id: RequestId| jobs.iter().filter(|job| job.request == id).map(|job| job.checked).collect::<Vec<_>>();
    assert_eq!(checked(held), [true], "{jobs:?}");
    assert_eq!(checked(free), [false], "{jobs:?}");
}

/// The script writes `{"a": 7` and then an `x` where the object needs a
/// comma or its close: with runs of four, the round whose anchor is the
/// colon proposes ` `, `7` and the `x`. The host keeps the run up to the
/// miss -- the colon, the space and the 7 commit in that round -- and the
/// `x` waits, is repaired, and the answer goes on from the repaired token.
#[test]
fn a_miss_in_the_middle_of_a_verified_run_is_cut_there_and_repaired_and_the_tokens_before_it_kept() {
    let grammar = json_object();
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let mut sched = scheduler(&mock);
    let id = submit(&mut sched, Some(&grammar), 64);
    mock.script(id, &encode("{\"a\": 7x}"));

    let events = run_to_idle(&mut sched);

    let (reason, spec) = done(&events, id);
    assert_eq!(text(&emitted(&events, id)), "{\"a\": 7}", "the answer, its miss repaired");
    assert_eq!(reason, FinishReason::Stop);
    let cuts = mock.cut_calls();
    assert_eq!(cuts[0], (id, draws("{\"a\""), 4), "a run that never misses is kept whole");
    assert_eq!(
        cuts[1],
        (id, draws(": 7x"), 3),
        "the run is cut at its miss, and the three tokens before it are kept"
    );
    let repairs = mock.repair_calls();
    assert_eq!(repairs.len(), 2, "the x, then the brace after the object closed");
    assert!(
        !repairs[0].permitted.permits(TokenId::from(b'x')) && repairs[0].permitted.permits(TokenId::from(b'}')),
        "the x is repaired where the object stands after the 7"
    );
    assert_eq!(misses(&events, id), (2, 2), "the miss inside the run is reported, as any miss is");
    let spec = spec.expect("the request's rounds were verify rounds");
    assert!(spec.drafted > 0 && spec.accepted > 0, "speculation stayed on for the checked request: {spec:?}");
}

/// Speculation never changes a constrained answer: the same scripted model,
/// with runs and one token a round, writes the same answer, repaired the
/// same way.
#[test]
fn a_constrained_answer_is_the_same_with_runs_and_without() {
    let grammar = json_object();
    let script = "Sure: {\"name\": \"Tom\", \"legs\": 4x, \"toys\": [1, 2]}\n\nDone.";
    let answer = |mock: MockCompute| {
        let mock = Arc::new(mock);
        let mut sched = scheduler(&mock);
        let id = submit(&mut sched, Some(&grammar), 128);
        mock.script(id, &encode(script));
        let events = run_to_idle(&mut sched);
        let repairs = mock.repair_calls().len();
        (text(&emitted(&events, id)), done(&events, id).0, repairs)
    };
    let plain = answer(MockCompute::new());
    let runs = answer(MockCompute::with_runs(&[4, 2, 6]));
    assert_eq!(runs, plain, "the answer, its finish and its repairs");
    assert_eq!(plain.1, FinishReason::Stop, "{plain:?}");
    let object: serde_json::Value =
        serde_json::from_str(&plain.0).unwrap_or_else(|e| panic!("not JSON ({e}): {plain:?}"));
    assert!(object.is_object(), "{plain:?}");
}

/// An unconstrained neighbour keeps its runs: beside a checked lane its
/// rounds commit what they commit alone, and the checked lane's own rounds
/// stay verify rounds.
#[test]
fn an_unconstrained_neighbour_keeps_its_runs_beside_a_checked_lane() {
    let grammar = json_object();
    let alone = {
        let mock = Arc::new(MockCompute::with_runs(&[4]));
        let mut sched = scheduler(&mock);
        let id = submit(&mut sched, None, 24);
        let events = run_to_idle(&mut sched);
        (emitted(&events, id), done(&events, id).1)
    };
    let mock = Arc::new(MockCompute::with_runs(&[4]));
    let mut sched = scheduler(&mock);
    let free = submit(&mut sched, None, 24);
    let held = submit(&mut sched, Some(&grammar), 24);
    mock.script(held, &encode("{\"a\": [1, 2, 3]}"));
    let events = run_to_idle(&mut sched);
    assert_eq!((emitted(&events, free), done(&events, free).1), alone, "the neighbour's tokens and its counters");
    assert!(mock.cut_calls().iter().all(|(request, _, _)| *request == held), "only the checked lane is shown to the host");
    assert!(done(&events, held).1.is_some_and(|spec| spec.drafted > 0), "the checked lane drafted too");
}
