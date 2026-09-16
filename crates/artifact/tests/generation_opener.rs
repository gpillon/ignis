//! The **generation opener** on the real frontend (GitHub #186, ADR 0029):
//! the boundary every prompt checkpoint is taken at, and the one property the
//! whole of cross-request state reuse rests on —
//!
//! > turn N's prompt *up to its opener* is an exact token prefix of turn
//! > N+1's whole prompt.
//!
//! If that is false, every match is a miss and the feature is dead weight; if
//! it is *nearly* true — the head tokenizes to different ids than the same
//! bytes inside the longer prompt — it is worse than dead weight, because a
//! checkpoint would be handed to a request whose history is not the one the
//! state was built from. So this checks the claim on real renders of the real
//! artifact's chat template, for a plain chat, an agent's tool loop, and a
//! thinking-on turn.
//!
//! CPU-only; skips when the artifact is not at its machine-local path (the
//! convention `real_frontend.rs` and `chat_render.rs` follow).

use std::path::{Path, PathBuf};

use ignis_artifact::{
    ChatMessage, ChatRenderOptions, ChatTemplate, FrontendSet, MessageContent, Reader, Role,
    ToolCall,
};
use serde_json::Value;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat_render")
}

fn frontend_or_skip() -> Option<FrontendSet> {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    let reader = Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    Some(FrontendSet::from_reader(&reader).expect("frontend set"))
}

fn user(text: &str) -> ChatMessage {
    ChatMessage::text(Role::User, text)
}

fn assistant(text: &str) -> ChatMessage {
    ChatMessage::text(Role::Assistant, text)
}

fn options(enable_thinking: bool) -> ChatRenderOptions {
    ChatRenderOptions {
        enable_thinking,
        ..Default::default()
    }
}

fn render(frontend: &FrontendSet, messages: &[ChatMessage], enable_thinking: bool) -> String {
    frontend
        .chat_template()
        .render_with_thinking_and_tools(messages, options(enable_thinking), Some(&[]))
        .expect("render")
}

/// The prompt's tokens, and the tokens of its head up to the generation
/// opener — the two things a prompt checkpoint's boundary is made of.
fn tokens_and_head(frontend: &FrontendSet, prompt: &str) -> (Vec<u32>, Vec<u32>) {
    let at = ChatTemplate::generation_opener_offset(prompt)
        .unwrap_or_else(|| panic!("no generation opener in:\n{prompt}"));
    let tokens = frontend.tokenizer().encode(prompt).expect("encode prompt");
    let head = frontend
        .tokenizer()
        .encode(&prompt[..at])
        .expect("encode head");
    (tokens, head)
}

/// Assert `head` really is the leading ids of `whole`, reporting the first id
/// that disagrees rather than "a slice is not a prefix".
fn assert_token_prefix(head: &[u32], whole: &[u32], what: &str) {
    assert!(
        head.len() <= whole.len(),
        "{what}: the head ({}) is longer than the prompt ({})",
        head.len(),
        whole.len()
    );
    if let Some(at) = head.iter().zip(whole).position(|(h, w)| h != w) {
        panic!(
            "{what}: the head is not a token prefix — id {at} is {} in the head and {} in the \
             prompt (a merge spanning the boundary)",
            head[at], whole[at]
        );
    }
}

#[test]
fn the_opener_ends_the_last_assistant_marker_of_every_recorded_render() {
    // Over the reference-recorded renders (the same fixtures #184/#185 pin
    // byte for byte), so this is checked against prompts the *reference*
    // produced, not only ones ignis produced.
    let mut dir: Vec<PathBuf> = std::fs::read_dir(fixtures())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    dir.sort();
    assert_eq!(dir.len(), 7, "every recorded fixture is present");
    for path in dir {
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let name = fixture["case"]["name"].as_str().unwrap().to_owned();
        let text = fixture["expected"]["text"].as_str().unwrap();
        let at = ChatTemplate::generation_opener_offset(text)
            .unwrap_or_else(|| panic!("{name}: no generation opener"));
        assert!(
            text[..at].ends_with(ChatTemplate::GENERATION_OPENER),
            "{name}: the offset does not end the opener"
        );
        assert!(
            !text[at..].contains(ChatTemplate::GENERATION_OPENER),
            "{name}: a later opener exists — the offset took an earlier one"
        );
        // The prompt always continues past the opener (the `<think>` block
        // the reference primes generation with). That is the whole reason
        // the checkpoint is taken *here* rather than at the prompt's end:
        // those last tokens are the ones the next turn re-renders away.
        assert!(
            !text[at..].is_empty(),
            "{name}: nothing follows the opener — the boundary would be the prompt's end"
        );
    }
}

#[test]
fn a_prompt_with_no_assistant_marker_has_no_opener() {
    assert_eq!(ChatTemplate::generation_opener_offset("plain text"), None);
    assert_eq!(ChatTemplate::generation_opener_offset(""), None);
}

#[test]
fn turn_n_head_is_a_token_prefix_of_turn_n_plus_1_in_a_plain_chat() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    let turn_n = [
        ChatMessage::text(Role::System, "You are a careful assistant."),
        user("What is the capital of France?"),
    ];
    let turn_n_plus_1 = [
        ChatMessage::text(Role::System, "You are a careful assistant."),
        user("What is the capital of France?"),
        assistant("Paris."),
        user("And of Italy?"),
    ];

    let (whole_n, head_n) = tokens_and_head(&frontend, &render(&frontend, &turn_n, false));
    let (whole_n1, head_n1) = tokens_and_head(&frontend, &render(&frontend, &turn_n_plus_1, false));

    assert_token_prefix(&head_n, &whole_n, "turn N against its own prompt");
    // The claim the feature rests on.
    assert_token_prefix(&head_n1, &whole_n1, "turn N+1 against its own prompt");
    assert_token_prefix(&head_n, &whole_n1, "turn N's checkpoint against turn N+1");
    assert!(
        head_n.len() < whole_n.len(),
        "the opener is inside the prompt, not its end"
    );
    // And the reuse is worth having: turn N+1 skips all of turn N's history.
    assert!(
        head_n.len() * 2 > whole_n1.len(),
        "turn N's opener ({}) should cover most of turn N+1's prompt ({})",
        head_n.len(),
        whole_n1.len()
    );
}

#[test]
fn turn_n_head_is_a_token_prefix_of_turn_n_plus_1_with_thinking_on() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    // Thinking on is the owner's actual profile, and the case ADR 0029's
    // "state after generation never matches" is really about: the next turn
    // renders the assistant's message without the reasoning it generated.
    let turn_n = [user("Plan a three-step migration.")];
    let mut reply = assistant("Step one, two, three.");
    reply.reasoning_content = Some("Weighing the ordering.".to_owned());
    let turn_n_plus_1 = [
        user("Plan a three-step migration."),
        reply,
        user("Now estimate each step."),
    ];

    let (whole_n, head_n) = tokens_and_head(&frontend, &render(&frontend, &turn_n, true));
    let (whole_n1, head_n1) = tokens_and_head(&frontend, &render(&frontend, &turn_n_plus_1, true));

    assert_token_prefix(&head_n, &whole_n, "turn N against its own prompt");
    assert_token_prefix(&head_n1, &whole_n1, "turn N+1 against its own prompt");
    assert_token_prefix(&head_n, &whole_n1, "turn N's checkpoint against turn N+1");
}

#[test]
fn an_agent_tool_loop_extends_its_own_checkpoint_each_iteration() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    // The owner's workload: iteration N+1 re-sends the whole history plus the
    // tool result. #184 (argument order) and #185 (`preserve_thinking`) are
    // what make the re-render an *extension* rather than a different prompt;
    // this is the property they were fixed for.
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }
    })];
    let mut called = ChatMessage::text(Role::Assistant, "Reading it now.");
    called.tool_calls = vec![ToolCall {
        id: Some("call_1".to_owned()),
        name: "read_file".to_owned(),
        arguments: r#"{"path":"a.rs"}"#.to_owned(),
    }];
    let iteration_1 = [user("What is in a.rs?")];
    let iteration_2 = [
        user("What is in a.rs?"),
        called,
        ChatMessage {
            role: Role::Tool,
            content: MessageContent::Text("fn main() {}".to_owned()),
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
    ];

    let render_with_tools = |messages: &[ChatMessage]| {
        frontend
            .chat_template()
            .render_with_thinking_and_tools(messages, options(true), Some(&tools))
            .expect("render")
    };
    let (whole_1, head_1) = tokens_and_head(&frontend, &render_with_tools(&iteration_1));
    let (whole_2, head_2) = tokens_and_head(&frontend, &render_with_tools(&iteration_2));

    assert_token_prefix(&head_1, &whole_1, "iteration 1 against its own prompt");
    assert_token_prefix(&head_2, &whole_2, "iteration 2 against its own prompt");
    assert_token_prefix(&head_1, &whole_2, "iteration 1's checkpoint against iteration 2");
    assert!(
        head_2.len() > head_1.len(),
        "each iteration's own checkpoint reaches further than the last"
    );
}
