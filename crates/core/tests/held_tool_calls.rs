//! A request's tool calls held to its tools on the scheduler's side (GitHub
//! #611, spec #582, ADR 0094): the content is checked against the tool
//! calls' grammar, which arms at the call's opener -- the prefill's own draw
//! included -- and a verified run is cut at its first miss inside a call,
//! the reasoning's close and the opener in the same run.
//!
//! Over `MockCompute`, plain and `with_runs`, whose scripted "model"
//! ([`MockCompute::script`]) writes what the test chooses. The vocabulary is
//! ASCII, one character a token, with the reasoning's close and the call's
//! two tags as added tokens; the mock's repair ranks what the script was
//! going to write next above the rest, then the vocabulary in order.

use std::sync::Arc;

use ignis_core::structured::AnswerGrammar;
use ignis_core::{
    ConcreteScheduler, DecodeParams, FinishReason, MockCompute, PendingToken, RequestClass, RequestId, RequestInput,
    SchedEvent, Scheduler, SchedulerConfig, TokenId,
};
use ignis_grammar::{DeclaredTool, GrammarCache, ToolCallGrammar, Vocab, VocabEntry, DEFAULT_CACHE_BYTES};

/// The model's EOS.
const STOP: TokenId = 5;
/// The reasoning block's close.
const THINK_END: TokenId = 6;
/// `<tool_call>` and `</tool_call>`.
const OPEN: TokenId = 7;
const CLOSE: TokenId = 8;
const VOCAB: TokenId = 128;

const WRITE_FILE: &str = r#"{"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string","enum":["w","a"]}},"required":["path","mode"],"additionalProperties":false}"#;

fn grammar() -> Arc<ToolCallGrammar> {
    let entries = (0..VOCAB)
        .map(|id| match id {
            STOP | THINK_END | OPEN | CLOSE => VocabEntry::Control,
            10 | 32..=126 => VocabEntry::Text(vec![id as u8]),
            _ => VocabEntry::Absent,
        })
        .collect();
    let added = vec![(THINK_END, b"</think>".to_vec()), (OPEN, b"<tool_call>".to_vec()), (CLOSE, b"</tool_call>".to_vec())];
    let cache = GrammarCache::new(Vocab::new(entries, &[STOP]), DEFAULT_CACHE_BYTES).with_added_texts(added);
    cache.tool_calls(&[DeclaredTool { name: "write_file".into(), strict: Some(WRITE_FILE.into()) }]).unwrap()
}

fn encode(text: &str) -> Vec<TokenId> {
    text.bytes().map(TokenId::from).collect()
}

fn scheduler(mock: &Arc<MockCompute>) -> ConcreteScheduler {
    ConcreteScheduler::with_config(SchedulerConfig { model: "m".into(), ..SchedulerConfig::default() }, mock.clone())
}

fn submit(sched: &mut ConcreteScheduler, thinking: bool, max_tokens: u32) -> RequestId {
    let held = AnswerGrammar::tool_calls(grammar());
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
                params: DecodeParams { max_tokens: Some(max_tokens), starts_in_reasoning: thinking, ..DecodeParams::default() },
                constrained: None,
                forced_literal: None,
                grammar: Some(if thinking { held.after(THINK_END) } else { held }),
                warm_up: false,
            },
            RequestClass::Interactive,
        )
        .expect("submit")
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    for _ in 0..2000 {
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

fn reason(events: &[SchedEvent], id: RequestId) -> FinishReason {
    events
        .iter()
        .find_map(|e| match e {
            SchedEvent::Done { request, reason, .. } if *request == id => Some(*reason),
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

/// The call after its opener: `mode` first, its value `value`.
fn body(value: &str) -> String {
    format!("\n<function=write_file>\n<parameter=mode>\n{value}\n</parameter>\n<parameter=path>\na.txt\n</parameter>\n</function>\n")
}

fn script(parts: &[&[TokenId]]) -> Vec<TokenId> {
    parts.concat()
}

/// Thinking off: the prefill draws the opener, which arms the grammar; the
/// draw needs no scores (free text cannot miss on this vocabulary), and the
/// call's `x`, outside its enum, is repaired from the next round's scores
/// to the script's next token the enum allows.
#[test]
fn a_call_the_first_draw_opens_is_held_from_its_next_token() {
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(&mock);
    let id = submit(&mut sched, false, 256);
    mock.script(id, &script(&[&[OPEN], &encode(&body("x")), &[CLOSE], &encode("\nDone.")]));

    let events = run_to_idle(&mut sched);

    assert_eq!(emitted(&events, id), script(&[&[OPEN], &encode(&body("a")), &[CLOSE], &encode("\nDone.")]));
    assert_eq!(reason(&events, id), FinishReason::Stop, "the stop after the call passes in free text");
    assert_eq!(misses(&events, id), (1, 1));
    let prefill: Vec<_> = mock.prefill_calls().into_iter().flatten().filter(|job| job.request == id).collect();
    assert!(prefill.iter().all(|job| !job.checked), "the first draw keeps no scores: {prefill:?}");
    let repairs = mock.repair_calls();
    assert!(repairs[0].permitted.permits(TokenId::from(b'w')) && !repairs[0].permitted.permits(TokenId::from(b'x')));
}

/// Thinking on, runs of eight: the first round's run holds the reasoning's
/// close, the opener and a byte the call cannot start with. The host keeps
/// the run up to that byte -- the reasoning is free, the opener arms the
/// grammar -- and the byte waits, is repaired, and the call goes on.
#[test]
fn a_verified_run_holding_the_reasoning_close_the_opener_and_a_miss_is_cut_at_the_miss() {
    let mock = Arc::new(MockCompute::with_runs(&[8]));
    let mut sched = scheduler(&mock);
    let id = submit(&mut sched, true, 256);
    let written = script(&[&encode("ok"), &[THINK_END, OPEN], &encode("x"), &encode(&body("w")), &[CLOSE]]);
    mock.script(id, &written);

    let events = run_to_idle(&mut sched);

    let repaired = script(&[&encode("ok"), &[THINK_END, OPEN], &encode("\n"), &encode(&body("w")), &[CLOSE]]);
    assert_eq!(emitted(&events, id), repaired);
    assert_eq!(reason(&events, id), FinishReason::Stop);
    assert_eq!(misses(&events, id), (1, 1));
    let first: Vec<PendingToken> = written[..8].iter().map(|&t| PendingToken::Token(t)).collect();
    assert_eq!(mock.cut_calls()[0], (id, first, 4), "kept: o, k, the close and the opener");
}

/// Text outside a call is never a miss, however it looks, and the stop after
/// it ends the request.
#[test]
fn free_text_is_never_a_miss() {
    let mock = Arc::new(MockCompute::with_runs(&[4, 3]));
    let mut sched = scheduler(&mock);
    let id = submit(&mut sched, false, 256);
    let text = script(&[&encode("Here <function=x> and </parameter>\n"), &[CLOSE, THINK_END], &encode("<tool_cal")]);
    mock.script(id, &text);
    let events = run_to_idle(&mut sched);
    assert_eq!(emitted(&events, id), text);
    assert_eq!((reason(&events, id), misses(&events, id)), (FinishReason::Stop, (0, 0)));
    assert!(!mock.cut_calls().is_empty(), "the content's rounds are checked");
}

/// A stop inside a held call is a miss: the call goes on, to the cap.
#[test]
fn a_stop_inside_a_held_call_is_a_miss_and_the_cap_cuts_it() {
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(&mock);
    let id = submit(&mut sched, false, 60);
    mock.script(id, &script(&[&[OPEN], &encode("\n<function=write_file>\n<parameter=path>\n")]));
    let events = run_to_idle(&mut sched);
    assert_eq!(reason(&events, id), FinishReason::Length);
    assert_eq!(emitted(&events, id).len(), 60);
    let (missed, repaired) = misses(&events, id);
    assert!(missed > 0 && missed == repaired, "{missed} misses");
}
