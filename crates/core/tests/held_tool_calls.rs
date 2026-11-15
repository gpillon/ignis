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

use ignis_core::forced_literal::ForcedLiteral;
use ignis_core::structured::{AnswerGrammar, StructuredKind};
use ignis_core::{
    ConcreteScheduler, DecodeParams, FinishReason, MockCompute, PendingToken, RequestClass, RequestId, RequestInput,
    SchedEvent, Scheduler, SchedulerConfig, TokenId,
};
use ignis_grammar::{DeclaredTool, GrammarCache, ToolCallGrammar, ToolHold, Vocab, VocabEntry, DEFAULT_CACHE_BYTES};

/// The model's EOS.
const STOP: TokenId = 5;
/// The reasoning block's close.
const THINK_END: TokenId = 6;
/// `<tool_call>` and `</tool_call>`.
const OPEN: TokenId = 7;
const CLOSE: TokenId = 8;
const VOCAB: TokenId = 128;

const WRITE_FILE: &str = r#"{"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string","enum":["w","a"]}},"required":["path","mode"],"additionalProperties":false}"#;

fn cache() -> GrammarCache {
    let entries = (0..VOCAB)
        .map(|id| match id {
            STOP | THINK_END | OPEN | CLOSE => VocabEntry::Control,
            10 | 32..=126 => VocabEntry::Text(vec![id as u8]),
            _ => VocabEntry::Absent,
        })
        .collect();
    let added = vec![(THINK_END, b"</think>".to_vec()), (OPEN, b"<tool_call>".to_vec()), (CLOSE, b"</tool_call>".to_vec())];
    GrammarCache::new(Vocab::new(entries, &[STOP]), DEFAULT_CACHE_BYTES).with_added_texts(added)
}

fn grammar() -> Arc<ToolCallGrammar> {
    cache().tool_calls(&[DeclaredTool { name: "write_file".into(), hold: ToolHold::Strict(WRITE_FILE.into()) }]).unwrap()
}

fn encode(text: &str) -> Vec<TokenId> {
    text.bytes().map(TokenId::from).collect()
}

fn scheduler(mock: &Arc<MockCompute>) -> ConcreteScheduler {
    ConcreteScheduler::with_config(SchedulerConfig { model: "m".into(), ..SchedulerConfig::default() }, mock.clone())
}

fn submit(sched: &mut ConcreteScheduler, thinking: bool, max_tokens: u32) -> RequestId {
    let held = AnswerGrammar::tool_calls(grammar());
    submit_held(sched, thinking, max_tokens, if thinking { held.after(THINK_END) } else { held }, None)
}

/// A request whose content is held to `grammar`, with the literal `forced`
/// (GitHub #615).
fn submit_held(
    sched: &mut ConcreteScheduler,
    thinking: bool,
    max_tokens: u32,
    grammar: AnswerGrammar,
    forced: Option<ForcedLiteral>,
) -> RequestId {
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
                forced_literal: forced.map(Arc::new),
                grammar: Some(grammar),
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

/// The two shapes a stop arrives in: the leaf's own stop, which in free
/// text ends the request as it would with no grammar, and the EOS a lane
/// that decodes past it (`ignore_eos`) reports as a token, which passes in
/// free text and leaves the content going on. Inside a held call both miss.
#[test]
fn a_stop_ends_free_text_and_an_eos_decoded_past_passes_it() {
    use ignis_core::structured::{GrammarState, Verdict};
    let mut state = GrammarState::new(&AnswerGrammar::tool_calls(grammar()), false);
    assert_eq!(state.check(PendingToken::Stop), Verdict::Finished);
    assert_eq!(state.check(PendingToken::Token(STOP)), Verdict::Passed);
    assert_eq!(state.check(PendingToken::Token(TokenId::from(b'x'))), Verdict::Passed, "the content goes on");
    assert_eq!(state.check(PendingToken::Token(OPEN)), Verdict::Passed);
    assert_eq!(state.check(PendingToken::Token(STOP)), Verdict::Miss);
    assert_eq!(state.check(PendingToken::Stop), Verdict::Miss);
}

// ── GitHub #615: beside a forced literal and a structured answer ────────

/// `tool_choice` `"required"`'s opening as the server forces it: the call's
/// opener up to `function`, before the name.
fn required_opening() -> Vec<TokenId> {
    script(&[&[OPEN], &encode("\n<function")])
}

/// The call's text after the `"required"` opening, to `write_file` under a
/// name of the model's own, `mode` first, its value `value`.
fn after_opening(value: &str) -> Vec<TokenId> {
    script(&[&encode(&body(value).replacen("\n<function=write_file", "=wrote_file", 1)), &[CLOSE]])
}

/// Thinking off, `"required"`: the forced literal opens the call from the
/// prefill's draw. Each forced token is fed to the grammar and none is a
/// miss, though the model would have written text there; what the model
/// writes after it is held: its own name for the tool and a mode outside
/// the enum, each repaired.
#[test]
fn a_call_a_forced_literal_opens_is_held_after_it_and_its_tokens_are_no_miss() {
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(&mock);
    let forced = ForcedLiteral::at_generation(required_opening()).unwrap();
    let id = submit_held(&mut sched, false, 256, AnswerGrammar::tool_calls(grammar()), Some(forced));
    mock.script(id, &script(&[&encode("Sure, I'll "), &after_opening("x"), &encode("\nDone.")]));

    let events = run_to_idle(&mut sched);

    assert_eq!(emitted(&events, id), script(&[&[OPEN], &encode(&body("a")), &[CLOSE], &encode("\nDone.")]));
    assert_eq!(reason(&events, id), FinishReason::Stop);
    assert_eq!(misses(&events, id), (2, 2), "`o` of `wrote` and the mode `x`, nothing forced");
}

/// Thinking on, `"required"`: the literal is forced after the model's own
/// `</think>`, joined on the line break after the token drawn with the
/// close, and the call it opens is held from the model's first token after
/// it. The reasoning before is free.
#[test]
fn a_call_forced_after_the_reasoning_is_held_after_the_literal() {
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(&mock);
    let forced = ForcedLiteral::after_reasoning(required_opening(), THINK_END, required_opening().len()).unwrap();
    let id = submit_held(&mut sched, true, 256, AnswerGrammar::tool_calls(grammar()).after(THINK_END), Some(forced));
    // Twelve tokens the joiner and the literal are forced in place of.
    let filler = encode(&"y".repeat(required_opening().len() + 1));
    mock.script(id, &script(&[&encode("hmm <tool_call>"), &[THINK_END], &encode("\n"), &filler, &after_opening("x")]));

    let events = run_to_idle(&mut sched);

    let call = script(&[&[OPEN], &encode(&body("a")), &[CLOSE]]);
    assert_eq!(emitted(&events, id), script(&[&encode("hmm <tool_call>"), &[THINK_END], &encode("\n\n"), &call]));
    assert_eq!(reason(&events, id), FinishReason::Stop);
    assert_eq!(misses(&events, id), (2, 2));
}

/// A literal token the grammar would refuse where the content stands is not
/// forced: the literal stops there and the grammar holds the rest. A named
/// function whose name holds `>` (a tool that is not held) cannot be
/// written by the dialect, so the model's own draw replaces it, held to the
/// declared tools.
#[test]
fn a_literal_token_the_grammar_refuses_is_not_forced_and_the_grammar_holds_on() {
    let grammar = cache()
        .tool_calls(&[
            DeclaredTool { name: "write_file".into(), hold: ToolHold::Strict(WRITE_FILE.into()) },
            DeclaredTool { name: "a>b".into(), hold: ToolHold::Free },
        ])
        .unwrap();
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(&mock);
    let forced = ForcedLiteral::at_generation(script(&[&[OPEN], &encode("\n<function=a>b>\n")])).unwrap();
    let id = submit_held(&mut sched, false, 256, AnswerGrammar::tool_calls(grammar), Some(forced));
    mock.script(id, &script(&[&[OPEN], &encode(&body("x")), &[CLOSE]]));

    let events = run_to_idle(&mut sched);

    assert_eq!(emitted(&events, id), script(&[&[OPEN], &encode(&body("a")), &[CLOSE]]));
    assert_eq!(misses(&events, id), (1, 1), "the mode `x` alone");
}

/// A structured answer beside a strict tool: until the answer starts the
/// opener releases the answer's grammar (#592), and the call it opens is
/// held to the tool calls' grammar, as is every call after it in the turn,
/// the text between them free.
#[test]
fn beside_a_structured_answer_the_call_the_opener_starts_is_held_and_so_is_the_next() {
    let schema = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;
    let answer = AnswerGrammar::new(cache().json_schema(schema, true).unwrap(), StructuredKind::JsonSchema);
    let held = answer.admitting_tool_call(OPEN).holding_calls(grammar());
    let mock = Arc::new(MockCompute::new());
    let mut sched = scheduler(&mock);
    let id = submit_held(&mut sched, false, 512, held, None);
    let call = |value: &str| script(&[&[OPEN], &encode(&body(value)), &[CLOSE]]);
    mock.script(id, &script(&[&call("x"), &encode("\nand {not json}\n"), &call("x")]));

    let events = run_to_idle(&mut sched);

    assert_eq!(emitted(&events, id), script(&[&call("a"), &encode("\nand {not json}\n"), &call("a")]));
    assert_eq!(reason(&events, id), FinishReason::Stop);
    assert_eq!(misses(&events, id), (2, 2));
}

/// The same with thinking on, runs of eight: the first run holds the
/// reasoning's close, the opener and a byte the call cannot start with. The
/// host keeps it up to that byte, which is repaired, and the call is held.
#[test]
fn beside_a_structured_answer_a_run_holding_the_opener_and_a_miss_is_cut_at_the_miss() {
    let schema = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;
    let answer = AnswerGrammar::new(cache().json_schema(schema, true).unwrap(), StructuredKind::JsonSchema).after(THINK_END);
    let held = answer.admitting_tool_call(OPEN).holding_calls(grammar());
    let mock = Arc::new(MockCompute::with_runs(&[8]));
    let mut sched = scheduler(&mock);
    let id = submit_held(&mut sched, true, 256, held, None);
    let written = script(&[&encode("ok"), &[THINK_END, OPEN], &encode("x"), &encode(&body("x")), &[CLOSE]]);
    mock.script(id, &written);

    let events = run_to_idle(&mut sched);

    let repaired = script(&[&encode("ok"), &[THINK_END, OPEN], &encode("\n"), &encode(&body("a")), &[CLOSE]]);
    assert_eq!(emitted(&events, id), repaired);
    assert_eq!(misses(&events, id), (2, 2));
    let first: Vec<PendingToken> = written[..8].iter().map(|&t| PendingToken::Token(t)).collect();
    assert_eq!(mock.cut_calls()[0], (id, first, 4), "kept: o, k, the close and the opener");
}

/// Thinking on, runs of five: the first run holds the reasoning's close and
/// the model's own `<tool_call>` and line break, so the literal is joined
/// after the token drawn with them, taken to be its `<`. Where that token is
/// `<` the literal goes on from `function`, and the model's text after it is
/// held; where it is a second line break, which the grammar takes where `<`
/// was due, the literal's `function` is not forced: the literal yields and
/// the model writes the call's head itself, held.
#[test]
fn a_call_the_model_opens_in_the_run_that_closes_its_reasoning_is_continued_or_yielded_to() {
    let opened = script(&[&encode("ok"), &[THINK_END, OPEN], &encode("\n")]);
    let cases = [
        (
            "the literal's `<`",
            script(&[&opened, &encode("<"), &encode(&"y".repeat(8)), &after_opening("x")]),
            script(&[&encode("ok"), &[THINK_END, OPEN], &encode(&body("a")), &[CLOSE]]),
            (2, 2),
        ),
        (
            "a line break more",
            script(&[&opened, &encode("\n"), &encode(&body("x")[1..]), &[CLOSE]]),
            script(&[&encode("ok"), &[THINK_END, OPEN], &encode("\n"), &encode(&body("a")), &[CLOSE]]),
            (1, 1),
        ),
    ];
    for (what, written, held, missed) in cases {
        let mock = Arc::new(MockCompute::with_runs(&[5]));
        let mut sched = scheduler(&mock);
        let forced = ForcedLiteral::after_reasoning(required_opening(), THINK_END, required_opening().len()).unwrap();
        let id = submit_held(&mut sched, true, 256, AnswerGrammar::tool_calls(grammar()).after(THINK_END), Some(forced));
        mock.script(id, &written);

        let events = run_to_idle(&mut sched);

        assert_eq!(emitted(&events, id), held, "{what}");
        assert_eq!(misses(&events, id), missed, "{what}");
        let first: Vec<PendingToken> = written[..5].iter().map(|&t| PendingToken::Token(t)).collect();
        assert_eq!(mock.cut_calls()[0], (id, first, 5), "{what}: the run kept whole");
    }
}

/// A repair before the answer starts may draw the opener (#592): beside the
/// request's strict tools the call it opens is held from there, the mask
/// the answer's set was built for read anew by the tools' grammar.
#[test]
fn an_opener_a_repair_draws_opens_a_held_call() {
    use ignis_core::structured::{GrammarState, Verdict};
    let schema = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;
    let answer = AnswerGrammar::new(cache().json_schema(schema, true).unwrap(), StructuredKind::JsonSchema);
    let mut state = GrammarState::new(&answer.admitting_tool_call(OPEN).holding_calls(grammar()), false);
    let before = state.permitted().unwrap();
    assert!(before.permits(OPEN) && before.permits(TokenId::from(b'{')));
    assert_eq!(state.repaired(OPEN), Ok(false));
    assert!(!state.released() && state.holds());
    assert_eq!(state.check(PendingToken::Token(TokenId::from(b'x'))), Verdict::Miss);
    let after = state.permitted().unwrap();
    assert!(after.permits(TokenId::from(b'\n')) && after.permits(TokenId::from(b'<')), "the call's head");
    assert!(!after.permits(TokenId::from(b'{')) && !after.permits(OPEN), "nothing of the answer");
}
