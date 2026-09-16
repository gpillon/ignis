//! The **system block boundary** on the real frontend (GitHub #188, ADR
//! 0029): the point a **retained prefix** is published at, and the one
//! property a subagent burst's reuse rests on —
//!
//! > two subagents spawned from the same parent share their whole system and
//! > tools block, token for token, however different their queries are.
//!
//! The generation opener (#186) is the *last* thing a conversation's turns
//! share; this is the *first* thing two unrelated requests share. Neither
//! subagent's prompt extends the other's, so no prompt checkpoint can ever
//! match between them — the only thing that can is this boundary, and only if
//! it is an exact token prefix of both. If it is merely *nearly* that (the
//! head tokenizes to different ids inside the longer prompt), the retained
//! prefix would hand a request KV pages for history it does not have.
//!
//! So this checks the claim on real renders of the real artifact's chat
//! template, with tools and without, thinking on and off, and on the
//! reference-recorded fixtures #184/#185 pinned byte for byte.
//!
//! CPU-only; skips when the artifact is not at its machine-local path (the
//! convention `real_frontend.rs`, `chat_render.rs` and `generation_opener.rs`
//! follow).

use std::path::{Path, PathBuf};

use ignis_artifact::{
    ChatMessage, ChatRenderOptions, ChatTemplate, FrontendSet, Reader, Role,
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

fn system(text: &str) -> ChatMessage {
    ChatMessage::text(Role::System, text)
}

fn user(text: &str) -> ChatMessage {
    ChatMessage::text(Role::User, text)
}

fn options(enable_thinking: bool) -> ChatRenderOptions {
    ChatRenderOptions {
        enable_thinking,
        ..Default::default()
    }
}

fn render(
    frontend: &FrontendSet,
    messages: &[ChatMessage],
    enable_thinking: bool,
    tools: &[Value],
) -> String {
    frontend
        .chat_template()
        .render_with_thinking_and_tools(messages, options(enable_thinking), Some(tools))
        .expect("render")
}

/// The prompt's tokens and the tokens of its head up to the system block's
/// end — the two things a retained prefix's boundary is made of.
fn tokens_and_head(frontend: &FrontendSet, prompt: &str) -> (Vec<u32>, Vec<u32>) {
    let at = ChatTemplate::system_block_offset(prompt)
        .unwrap_or_else(|| panic!("no system block in:\n{prompt}"));
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

/// A tool definition, so the tools half of the block is real.
fn tool() -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }
    })
}

#[test]
fn the_boundary_closes_the_first_system_block_of_every_recorded_render() {
    // Over the reference-recorded renders (the fixtures #184/#185 pin byte for
    // byte), so this is checked against prompts the *reference* produced, not
    // only ones ignis produced.
    let mut dir: Vec<PathBuf> = std::fs::read_dir(fixtures())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    dir.sort();
    assert_eq!(dir.len(), 7, "every recorded fixture is present");
    let mut with_block = 0;
    let mut opens_with_block = 0;
    for path in dir {
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let name = fixture["case"]["name"].as_str().unwrap().to_owned();
        let text = fixture["expected"]["text"].as_str().unwrap();
        if text.starts_with("<|im_start|>system\n") {
            opens_with_block += 1;
        }
        // Some of the recorded renders carry neither a system message nor
        // tools, and the reference opens those straight on the user turn.
        // They publish nothing, and saying so is half of what this checks:
        // the boundary exists exactly when the render opens with a block.
        let Some(at) = ChatTemplate::system_block_offset(text) else {
            assert!(
                !text.starts_with("<|im_start|>system\n"),
                "{name}: the render opens with a system block but none was reported"
            );
            continue;
        };
        with_block += 1;
        assert!(
            text[..at].starts_with("<|im_start|>system\n"),
            "{name}: the boundary's head is not the system block"
        );
        assert!(
            text[..at].ends_with("<|im_end|>\n"),
            "{name}: the boundary does not close a block"
        );
        assert!(
            !text[..at].contains("<|im_start|>user"),
            "{name}: the boundary reached past the first block into the conversation"
        );
        // It is strictly before the generation opener, and strictly inside the
        // prompt: a retained prefix that covered the whole render would be a
        // prefix of nothing else.
        let opener = ChatTemplate::generation_opener_offset(text)
            .unwrap_or_else(|| panic!("{name}: no generation opener"));
        assert!(
            at < opener,
            "{name}: the system block ({at}) must end before the opener ({opener})"
        );
    }
    // Counted from the fixtures themselves rather than against a number
    // written here: a boundary is reported for exactly the renders that open
    // with one. A fixture added or changed later moves both sides together,
    // which a literal would not.
    assert_eq!(
        with_block, opens_with_block,
        "a boundary is reported for exactly the renders that open with a system block"
    );
    assert!(with_block > 0, "at least one recorded render has one");
}

#[test]
fn a_render_that_does_not_open_with_a_system_block_has_no_boundary() {
    // The boundary is the *first* block and only when the render opens with
    // it. A prompt whose system message arrives later shares nothing with a
    // burst sibling from its own first byte, so there is nothing to publish.
    assert_eq!(ChatTemplate::system_block_offset("plain text"), None);
    assert_eq!(ChatTemplate::system_block_offset(""), None);
    assert_eq!(
        ChatTemplate::system_block_offset("<|im_start|>user\nhi<|im_end|>\n"),
        None
    );
    // An unterminated system block: the render never closed it, so there is no
    // point inside it that a later request provably shares.
    assert_eq!(
        ChatTemplate::system_block_offset("<|im_start|>system\nyou are"),
        None
    );
    // The smallest real one.
    let text = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nq<|im_end|>\n";
    assert_eq!(
        ChatTemplate::system_block_offset(text),
        Some("<|im_start|>system\nS<|im_end|>\n".len())
    );
}

#[test]
fn two_subagents_of_a_burst_share_the_boundary_as_an_exact_token_prefix() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    // The workload the retained prefix exists for: one parent spawns two
    // subagents with the same instructions and the same tools, and asks each a
    // different question. Neither prompt extends the other, so no checkpoint
    // can match — the block they share is all there is.
    let instructions = "You are a subagent. Work only in the worktree you are given.";
    let tools = vec![tool()];
    let first = [system(instructions), user("Summarise crates/core/src/prefix.rs.")];
    let second = [system(instructions), user("List every ADR that mentions eviction.")];

    let (whole_1, head_1) = tokens_and_head(&frontend, &render(&frontend, &first, true, &tools));
    let (whole_2, head_2) = tokens_and_head(&frontend, &render(&frontend, &second, true, &tools));

    assert_token_prefix(&head_1, &whole_1, "subagent 1 against its own prompt");
    assert_token_prefix(&head_2, &whole_2, "subagent 2 against its own prompt");
    assert_eq!(
        head_1, head_2,
        "two subagents of one burst render the same system and tools block"
    );
    // The claim the retained prefix rests on.
    assert_token_prefix(&head_1, &whole_2, "subagent 1's boundary against subagent 2");
    assert!(
        head_1.len() < whole_1.len(),
        "the boundary is inside the prompt, not its end"
    );
    // The safety direction, which is a property of the boundary and not of
    // the questions this test happened to pick: the block never reaches past
    // what the two prompts actually share. Reaching further is the failure
    // that matters — it would hand a subagent pages warmed from history it
    // never sent.
    let common = whole_1
        .iter()
        .zip(&whole_2)
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        head_1.len() <= common,
        "the boundary ({}) reaches past what the two prompts share ({common})",
        head_1.len()
    );
    assert!(
        common < whole_1.len().min(whole_2.len()),
        "the two prompts must really diverge, or this proves nothing"
    );
    // They share a little more than the block — the next turn's opening
    // marker — and that remainder is deliberately left unused: reuse happens
    // at recorded structural points, never at an arbitrary longest common
    // prefix (ADR 0029, Out of Scope), because GDN state exists only where it
    // was captured.
    println!(
        "two subagents share {common} leading ids; the boundary publishes {} of them",
        head_1.len()
    );
}

#[test]
fn the_boundary_is_an_exact_token_prefix_with_and_without_tools() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    // Tools are rendered *inside* the first system block, so they move the
    // boundary rather than sitting outside it. Both renders must still cut at
    // a point their own tokenizer agrees with.
    let messages = [system("Be brief."), user("What is a KV page?")];
    let tools = vec![tool()];

    let (whole_bare, head_bare) =
        tokens_and_head(&frontend, &render(&frontend, &messages, true, &[]));
    let (whole_tools, head_tools) =
        tokens_and_head(&frontend, &render(&frontend, &messages, true, &tools));

    assert_token_prefix(&head_bare, &whole_bare, "no tools");
    assert_token_prefix(&head_tools, &whole_tools, "with tools");
    assert!(
        head_tools.len() > head_bare.len(),
        "the tools block is inside the boundary ({} vs {} tokens)",
        head_tools.len(),
        head_bare.len()
    );
}

#[test]
fn the_boundary_is_an_exact_token_prefix_with_thinking_on_and_off() {
    let Some(frontend) = frontend_or_skip() else {
        return;
    };
    // Thinking is a property of the *generation* end of the prompt, so the two
    // renders may or may not agree on the block — what matters, and what is
    // asserted, is that each one's own boundary is exact for its own prompt.
    let messages = [system("Be careful."), user("Plan a three-step migration.")];
    for thinking in [false, true] {
        let prompt = render(&frontend, &messages, thinking, &[]);
        let (whole, head) = tokens_and_head(&frontend, &prompt);
        assert_token_prefix(&head, &whole, if thinking { "thinking on" } else { "thinking off" });
        assert!(
            head.len() < whole.len(),
            "thinking={thinking}: the boundary is inside the prompt"
        );
    }
}
