//! Lineage on the **real frontend** (GitHub #187, ADR 0029): a tool loop
//! keeps two checkpoints, and the new user message after it reuses the
//! turn-opening one.
//!
//! Everything below the frontend is already pinned on synthetic tokens
//! (`crates/core/tests/prompt_checkpoint.rs`). What those tests cannot pin is
//! the thing the whole lineage rule exists for, because it is a property of
//! the *chat template* rather than of the scheduler:
//!
//! > once the human speaks again, the tool loop's own checkpoints stop
//! > matching — the template drops the reasoning of every assistant turn
//! > before the last real user query (#185), so the history those checkpoints
//! > covered is no longer what the next render produces. The turn-opening
//! > checkpoint, taken before any of that reasoning existed, still matches.
//!
//! A synthetic test that makes the next turn diverge by choosing different
//! token ids proves nothing about that: it would pass just as well if the
//! template kept every think block. So this renders a real three-iteration
//! tool loop through the artifact's own template with thinking on, feeds the
//! real token ids to the real scheduler over `MockCompute`, and asks what
//! actually gets reused.
//!
//! CPU-only — the backend is the mock; only the tokenizer and the template
//! are real. Skips when the artifact is not at its machine-local path, the
//! convention `real_frontend.rs` and `generation_opener.rs` follow.

use std::path::Path;
use std::sync::Arc;

use ignis_core::types::{
    DecodeParams, RequestClass, RequestId, RequestInput, SchedEvent, TokenId,
};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::template::{
    ChatMessage, FunctionIn, MessageContent, RenderedPrompt, TemplateProvider, ToolCallIn,
};
use ignis_server::thinking::ThinkingOptions;
use serde_json::json;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

fn provider_or_skip() -> Option<ArtifactTemplateProvider> {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    let reader = ignis_artifact::Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    let frontend = ignis_artifact::FrontendSet::from_reader(&reader).expect("frontend set");
    Some(ArtifactTemplateProvider::new(frontend))
}

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_owned(),
        content: MessageContent::Text(content.to_owned()),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn thinking(role: &str, content: &str, reasoning: &str) -> ChatMessage {
    ChatMessage {
        reasoning_content: Some(reasoning.to_owned()),
        ..message(role, content)
    }
}

fn calling(content: &str, reasoning: &str, path: &str) -> ChatMessage {
    ChatMessage {
        tool_calls: Some(vec![ToolCallIn {
            id: Some("call_1".to_owned()),
            function: FunctionIn {
                name: "read_file".to_owned(),
                arguments: format!("{{\"path\":\"{path}\"}}"),
            },
        }]),
        ..thinking("assistant", content, reasoning)
    }
}

/// The tools qwen-code sends on every request — which is why the system block
/// is never short and why this conversation's prompts are worth caching.
fn tools() -> Vec<serde_json::Value> {
    vec![json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the working tree and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Repo-relative path"}},
                "required": ["path"]
            }
        }
    })]
}

/// `api.rs`'s own construction of a text-only request from a render: the
/// tokens plus the two structural offsets only the renderer knows.
fn request_input(rendered: RenderedPrompt) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: rendered.tokens,
        params: DecodeParams {
            max_tokens: Some(4),
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: rendered.opener_tokens,
        user_turn_tokens: rendered.user_turn_tokens,
        system_block_tokens: rendered.system_block_tokens,
        reuse_boundaries: Vec::new(),
        constrained: None,
    }
}

fn run_to_idle(sched: &mut ConcreteScheduler) -> Vec<SchedEvent> {
    let mut events = Vec::new();
    while !sched.is_idle() {
        events.extend(sched.advance());
    }
    events
}

/// The leading prompt tokens `request` resumed from retained state, or `None`
/// when it resumed from nothing.
fn reused(events: &[SchedEvent], request: RequestId) -> Option<u32> {
    events.iter().find_map(|e| match e {
        SchedEvent::StateReused {
            request: r, tokens, ..
        } if *r == request => Some(*tokens),
        _ => None,
    })
}

/// The retained pool's entries, as their token reach and whether each opens a
/// turn.
fn pool(sched: &ConcreteScheduler) -> Vec<(u32, bool)> {
    sched
        .checkpoint_pool()
        .entries()
        .iter()
        .map(|e| (e.tokens, e.turn_opening))
        .collect()
}

#[test]
fn a_new_user_message_after_a_tool_loop_reuses_the_turn_opening_checkpoint() {
    let Some(provider) = provider_or_skip() else {
        return;
    };
    // Thinking on and `preserve_thinking` off: the owner's actual profile, and
    // the one whose history rendering changes under the loop (#185).
    let options = ThinkingOptions::default();
    assert!(options.enable_thinking && !options.preserve_thinking);
    let tools = tools();
    let render = |messages: &[ChatMessage]| provider.apply_chat_template(messages, &options, &tools);

    let system = message("system", "You are a coding agent. Use the tools you are given.");
    let query = message("user", "What does a.rs contain, and what does b.rs contain?");

    // Iteration 1: the user's question.
    let it1 = vec![system.clone(), query.clone()];
    // Iteration 2: the assistant thought, called a tool, and the tool answered.
    let mut it2 = it1.clone();
    it2.push(calling("Reading a.rs.", "I should look at a.rs first.", "a.rs"));
    it2.push(message("tool", "fn main() {}"));
    // Iteration 3: and again for the second file.
    let mut it3 = it2.clone();
    it3.push(calling("Now b.rs.", "a.rs is a main. Next b.rs.", "b.rs"));
    it3.push(message("tool", "pub fn helper() {}"));
    // The turn ends, and the human speaks again.
    let mut next_turn = it3.clone();
    next_turn.push(thinking(
        "assistant",
        "a.rs holds main, b.rs holds a helper.",
        "Both files are read; I can answer now.",
    ));
    next_turn.push(message("user", "Which of them should I start reading?"));

    let rendered: Vec<RenderedPrompt> = [&it1, &it2, &it3, &next_turn]
        .into_iter()
        .map(|m| render(m).expect("the fixture template renders"))
        .collect();
    for (n, r) in rendered.iter().enumerate() {
        assert!(
            r.opener_tokens.is_some(),
            "render {n} must report a generation opener"
        );
        assert!(
            r.user_turn_tokens.is_some(),
            "render {n} must report a last real user query"
        );
    }
    let opener_1 = rendered[0].opener_tokens.expect("iteration 1's opener");
    println!(
        "checkpoint_lineage: (prompt, opener, last user query) = {:?}",
        rendered
            .iter()
            .map(|r| (r.tokens.len(), r.opener_tokens, r.user_turn_tokens))
            .collect::<Vec<_>>()
    );

    // The property the whole lineage rule turns on, asserted rather than
    // assumed: the new turn's render no longer starts with iteration 3's head
    // (its assistant reasoning is gone), but it does still start with
    // iteration 1's — which is exactly why the turn-opening checkpoint is the
    // one kept.
    let head_of = |n: usize| -> Vec<TokenId> {
        let r = &rendered[n];
        r.tokens[..r.opener_tokens.unwrap() as usize].to_vec()
    };
    let last = &rendered[3].tokens;
    assert!(
        last.starts_with(&head_of(0)),
        "the turn opener's head must survive into the next turn's render"
    );
    assert!(
        !last.starts_with(&head_of(2)),
        "iteration 3's head must NOT survive it — if it does, this test is not \
         exercising the thinking-stripped render and the turn opener is not load-bearing"
    );
    // And the loop really is a loop: the tool results are not new user queries.
    assert!(
        rendered[2].user_turn_tokens.unwrap() < opener_1,
        "a tool result must not read as the human speaking"
    );
    assert!(
        rendered[3].user_turn_tokens.unwrap() >= opener_1,
        "the new user message must lie past iteration 1's opener"
    );

    let compute = Arc::new(MockCompute::new());
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );

    let mut skipped = Vec::new();
    let mut pools = Vec::new();
    for prompt in rendered {
        let id = sched
            .submit(request_input(prompt), RequestClass::Agent)
            .expect("submit");
        let events = run_to_idle(&mut sched);
        skipped.push(reused(&events, id));
        pools.push(pool(&sched));
        println!(
            "checkpoint_lineage: request {id} reused {:?}, pool now {:?}",
            skipped.last().unwrap(),
            pools.last().unwrap()
        );
    }

    // Iteration 1 warms the conversation; 2 and 3 each resume from the one
    // before, so the loop never re-prefills its history.
    assert_eq!(skipped[0], None, "iteration 1 had nothing to resume from");
    assert_eq!(
        skipped[1],
        Some(opener_1),
        "iteration 2 resumes from iteration 1's opener"
    );
    let it2_opener = skipped[2].expect("iteration 3 resumed from something");
    assert!(
        it2_opener > opener_1,
        "iteration 3 resumes from iteration 2's own checkpoint ({it2_opener}), not \
         iteration 1's ({opener_1}) — that is the #187 capture working"
    );
    // And the loop never holds more than the pair, however long it runs: the
    // turn opener, and whatever the latest iteration captured.
    assert_eq!(
        pools[1],
        vec![(opener_1, true), (it2_opener, false)],
        "after iteration 2: the turn opener and iteration 2's own checkpoint"
    );
    assert_eq!(
        pools[2].len(),
        2,
        "after iteration 3 it is still two, not three: {:?}",
        pools[2]
    );
    assert_eq!(
        pools[2][0],
        (opener_1, true),
        "the turn opener is the one that survives the loop"
    );
    assert!(
        pools[2][1].0 > it2_opener && !pools[2][1].1,
        "and iteration 2's was superseded by iteration 3's: {:?}",
        pools[2]
    );

    // The acceptance criterion: the new user message reuses the turn-opening
    // checkpoint — the only entry its re-rendered history still matches, since
    // the loop's own entries covered reasoning this render has dropped — and
    // what it captures opens a turn, retiring the one behind it.
    assert_eq!(
        skipped[3],
        Some(opener_1),
        "the new user message reuses the turn-opening checkpoint"
    );
    assert_eq!(pools[3].len(), 1, "the previous turn is retired: {:?}", pools[3]);
    assert!(pools[3][0].1, "and what is left opens the new turn");
    assert!(
        pools[3][0].0 > opener_1,
        "reaching further than the entry it replaced"
    );
}
