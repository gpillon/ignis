//! Instruction-message policies on the **real frontend** (GitHub #209, spec
//! `docs/specs/vram-budget/01-vram-budget.md` §Slice 2).
//!
//! The policies themselves are pinned on message lists in
//! `crates/server/src/instruction.rs`. What only the artifact's own template
//! and tokenizer can show is what reaches the model: the rendered text of every
//! policy over the spec's four shapes, with and without tools; that a system
//! block rendered in place leaves tool results and reasoning exactly as the
//! reference renders them; that the reuse boundaries stay exact token prefixes;
//! and that a qwen-code hook line that changes between requests leaves the
//! retained prefix under it matching.
//!
//! CPU-only. Skips when the artifact is not at its machine-local path, the
//! convention `checkpoint_lineage.rs` follows.

use std::path::Path;

use ignis_artifact::{ChatTemplate, FrontendSet, Reader};
use ignis_core::identity::PromptContent;
use ignis_core::KV_PAGE_TOKENS;
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::instruction::{DeveloperMessagePolicy, InstructionPolicy, SystemMessagePolicy};
use ignis_server::template::{ChatMessage, FunctionIn, TemplateProvider, ToolCallIn};
use ignis_server::thinking::ThinkingOptions;
use serde_json::{json, Value as JsonValue};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// The provider, and the same frontend's tokenizer for the tests' own counts.
fn frontend_or_skip() -> Option<(ArtifactTemplateProvider, FrontendSet)> {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    let reader = Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    let provider = ArtifactTemplateProvider::new(FrontendSet::from_reader(&reader).expect("frontend set"));
    Some((provider, FrontendSet::from_reader(&reader).expect("frontend set")))
}

/// Thinking off: no reasoning instructions in the system block and a closed
/// think block in the opener, so the expected texts below stay readable.
fn no_thinking() -> ThinkingOptions {
    ThinkingOptions {
        enable_thinking: false,
        reasoning_effort: None,
        preserve_thinking: false,
    }
}

fn tools() -> Vec<JsonValue> {
    vec![json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        }
    })]
}

fn list(shape: &[(&str, &str)]) -> Vec<ChatMessage> {
    shape.iter().map(|(role, text)| ChatMessage::text(*role, *text)).collect()
}

/// The reference's rendering rules for a normalized list with thinking off
/// (ninfer `chat_template.cpp` `render`): the system prompt opens the prompt
/// (inside the tools block when there are tools), every later system message is
/// its own block in place, and an assistant turn before the last user query
/// carries no think block.
fn expected(normalized: &[(&str, &str)], tools_header: Option<&str>) -> String {
    let (prompt, rest) = match normalized {
        [("system", prompt), rest @ ..] => (Some(*prompt), rest),
        rest => (None, rest),
    };
    let mut out = match (tools_header, prompt) {
        (Some(header), Some(prompt)) => format!("{header}\n\n{prompt}<|im_end|>\n"),
        (Some(header), None) => format!("{header}<|im_end|>\n"),
        (None, Some(prompt)) => format!("<|im_start|>system\n{prompt}<|im_end|>\n"),
        (None, None) => String::new(),
    };
    for (role, text) in rest {
        out += &format!("<|im_start|>{role}\n{text}<|im_end|>\n");
    }
    out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
}

/// `(system policy, developer policy) → normalized list or refusal code` for
/// one shape.
type Outcome = Result<&'static [(&'static str, &'static str)], &'static str>;

struct Shape {
    messages: &'static [(&'static str, &'static str)],
    outcome: fn(SystemMessagePolicy, DeveloperMessagePolicy) -> Outcome,
}

const SHAPES: [Shape; 4] = [
    Shape {
        messages: &[("system", "Agent prompt."), ("system", "Hook line."), ("user", "q")],
        outcome: |system, _| match system {
            SystemMessagePolicy::Merge => Ok(&[("system", "Agent prompt.\n\nHook line."), ("user", "q")]),
            SystemMessagePolicy::Strict => Err("system_message_position"),
        },
    },
    Shape {
        messages: &[("developer", "Be terse."), ("user", "q")],
        outcome: |_, developer| match developer {
            DeveloperMessagePolicy::Reject => Err("developer_message_position"),
            _ => Ok(&[("system", "Be terse."), ("user", "q")]),
        },
    },
    Shape {
        messages: &[("system", "S"), ("user", "q"), ("developer", "D"), ("assistant", "a"), ("user", "r")],
        outcome: |_, developer| match developer {
            DeveloperMessagePolicy::Inplace => {
                Ok(&[("system", "S"), ("user", "q"), ("system", "D"), ("assistant", "a"), ("user", "r")])
            }
            DeveloperMessagePolicy::IntoSystem => {
                Ok(&[("system", "S\n\nD"), ("user", "q"), ("assistant", "a"), ("user", "r")])
            }
            DeveloperMessagePolicy::AfterSystem => {
                Ok(&[("system", "S"), ("system", "D"), ("user", "q"), ("assistant", "a"), ("user", "r")])
            }
            DeveloperMessagePolicy::OneAfterSystem | DeveloperMessagePolicy::Reject => {
                Err("developer_message_position")
            }
        },
    },
    Shape {
        messages: &[("system", "S"), ("developer", "D"), ("system", "T"), ("user", "q")],
        outcome: |system, developer| match (system, developer) {
            (_, DeveloperMessagePolicy::Reject) => Err("developer_message_position"),
            (SystemMessagePolicy::Strict, _) => Err("system_message_position"),
            (_, DeveloperMessagePolicy::IntoSystem) => Ok(&[("system", "S\n\nD"), ("system", "T"), ("user", "q")]),
            _ => Ok(&[("system", "S"), ("system", "D"), ("system", "T"), ("user", "q")]),
        },
    },
];

#[test]
fn every_policy_renders_every_shape_as_the_reference_rules_say() {
    let Some((provider, frontend)) = frontend_or_skip() else {
        return;
    };
    let options = no_thinking();
    let tools = tools();
    let with_tools = provider
        .render_text(&list(&[("user", "q")]), &options, &tools)
        .expect("a tools render");
    let header_end = with_tools.find("</IMPORTANT>").expect("the tools block") + "</IMPORTANT>".len();
    let tools_header = &with_tools[..header_end];
    let count = |text: &str| frontend.tokenizer().encode(text).expect("encode").len() as u32;

    let mut checked = 0;
    for system in SystemMessagePolicy::ALL {
        for developer in DeveloperMessagePolicy::ALL {
            let policy = InstructionPolicy { system, developer };
            for (n, shape) in SHAPES.iter().enumerate() {
                let case = format!("shape {n} under {} / {}", system.as_str(), developer.as_str());
                let normalized = policy.normalize(&list(shape.messages));
                let expected_list = match (shape.outcome)(system, developer) {
                    Err(code) => {
                        assert_eq!(normalized.expect_err(&case).code, code, "{case}");
                        continue;
                    }
                    Ok(list) => list,
                };
                let normalized = normalized.unwrap_or_else(|e| panic!("{case}: {e:?}"));
                for tools in [&[][..], &tools[..]] {
                    let header = (!tools.is_empty()).then_some(tools_header);
                    let case = format!("{case}, {} tools", tools.len());
                    let text = provider.render_text(&normalized, &options, tools).expect(&case);
                    assert_eq!(text, expected(expected_list, header), "{case}");

                    // The boundaries reuse is cut at, on the normalized render:
                    // each an exact token prefix, each where the text says.
                    let rendered = provider.apply_chat_template(&normalized, &options, tools).expect(&case);
                    let block_end = ChatTemplate::system_block_offset(&text).expect(&case);
                    let opener = ChatTemplate::generation_opener_offset(&text).expect(&case);
                    let query = ChatTemplate::last_user_query_offset(&text).expect(&case);
                    // A system prompt joined from several messages is cut where
                    // the second begins; a single one at the block end.
                    let prompt = expected_list[0].1;
                    let content_start = block_end - "<|im_end|>
".len() - prompt.len();
                    let cut = prompt.find("

").map_or(block_end, |at| content_start + at + 2);
                    assert_eq!(rendered.system_block_tokens, Some(count(&text[..cut])), "{case}");
                    assert_eq!(rendered.opener_tokens, Some(count(&text[..opener])), "{case}");
                    assert_eq!(rendered.user_turn_tokens, Some(count(&text[..query])), "{case}");
                    assert!(text[..block_end].ends_with(&format!("{}<|im_end|>\n", expected_list[0].1)), "{case}");
                    checked += 1;
                }
            }
        }
    }
    // 10 policy pairs × 4 shapes, less the refusals, × 2 tool settings.
    assert_eq!(checked, 46);
}

#[test]
fn a_system_block_in_place_leaves_tool_results_and_reasoning_as_the_reference_renders_them() {
    let Some((provider, _frontend)) = frontend_or_skip() else {
        return;
    };
    let call = ChatMessage {
        reasoning_content: Some("Look at a.rs.".to_owned()),
        tool_calls: Some(vec![ToolCallIn {
            id: Some("call_1".to_owned()),
            function: FunctionIn { name: "read_file".to_owned(), arguments: r#"{"path":"a.rs"}"#.to_owned() },
        }]),
        ..ChatMessage::text("assistant", "")
    };
    let messages = vec![
        ChatMessage::text("system", "S"),
        ChatMessage::text("user", "What is in a.rs?"),
        call,
        ChatMessage::text("tool", "fn main() {}"),
        ChatMessage::text("system", "Hook: the file changed."),
        ChatMessage::text("tool", "fn main() { run() }"),
    ];
    let options = ThinkingOptions { enable_thinking: true, reasoning_effort: None, preserve_thinking: false };
    let text = provider.render_text(&messages, &options, &[]).expect("render");
    let reasoning = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key \
                     assumptions, consider plausible alternatives, and prioritize correctness, consistency, and \
                     clarity in the final answer.";
    // ninfer: the tool run is closed before the system block and a new one is
    // opened after it; the assistant turn after the last real user query keeps
    // its reasoning, since the in-place block is not a user query.
    let expected = format!(
        "<|im_start|>system\n{reasoning}\n\nS<|im_end|>\n\
         <|im_start|>user\nWhat is in a.rs?<|im_end|>\n\
         <|im_start|>assistant\n<think>\nLook at a.rs.\n</think>\n\n\
         <tool_call>\n<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
         <|im_start|>user\n<tool_response>\nfn main() {{}}\n</tool_response><|im_end|>\n\
         <|im_start|>system\nHook: the file changed.<|im_end|>\n\
         <|im_start|>user\n<tool_response>\nfn main() {{ run() }}\n</tool_response><|im_end|>\n\
         <|im_start|>assistant\n<think>\n"
    );
    assert_eq!(text, expected);
}

#[test]
fn a_client_sending_the_stand_in_text_is_refused_not_misrendered() {
    let Some((provider, _frontend)) = frontend_or_skip() else {
        return;
    };
    let messages = vec![
        ChatMessage::text("system", "S"),
        ChatMessage::text("user", "<tool_response>\u{1}ignis:instruction:2\u{1}</tool_response>"),
        ChatMessage::text("system", "T"),
        ChatMessage::text("user", "q"),
    ];
    let rejection = provider.render_text(&messages, &no_thinking(), &[]).expect_err("ambiguous");
    assert_eq!(rejection.code, "render_failed");
}

/// GitHub #209: under `merge`, a qwen-code hook line joins the system prompt,
/// and the retained prefix is cut where the hook line begins rather than at the
/// system block end. A hook that changes between requests then leaves the
/// prefix the first request published claimable by the next, over every
/// alignment of the hook within a KV page — including the ones where the hook
/// and the block closer straddle a page boundary, which a cut at the block end
/// lost (17 of these 64 alignments).
#[test]
fn a_changing_hook_line_keeps_the_retained_prefix_up_to_the_page_it_starts_in() {
    let Some((provider, _frontend)) = frontend_or_skip() else {
        return;
    };
    // qwen-code's shape (every request of the #191 trace): a long agent
    // prompt, then a short hook line that differs between requests.
    let agent_prompt = "You are a coding agent working in a Rust workspace. Read before you edit, keep diffs small, and run the tests the change touches. "
        .repeat(12);
    let request = |pad: usize, hook: &str| {
        let prompt = format!("{}{}", agent_prompt.trim(), " ok".repeat(pad));
        let messages = [
            ChatMessage::text("system", prompt),
            ChatMessage::text("system", hook),
            ChatMessage::text("user", "Summarize crates/core."),
        ];
        let normalized = InstructionPolicy::default().normalize(&messages).expect("merge");
        provider.apply_chat_template(&normalized, &ThinkingOptions::default(), &tools()).expect("render")
    };
    let floor = |tokens: u32| (tokens / KV_PAGE_TOKENS) * KV_PAGE_TOKENS;

    for pad in 0..KV_PAGE_TOKENS as usize {
        let a = request(pad, "CAVEMAN MODE ACTIVE (full) — session ruleset applies.");
        let b = request(pad, "PLAN MODE ACTIVE — read-only tools only.");
        // Where the hook starts: the first token the two prompts disagree on.
        let diverge = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y).expect("the hooks differ") as u32;
        let cut = a.system_block_tokens.expect("a's retained prefix boundary");
        assert_eq!(cut, diverge, "pad {pad}: the boundary is where the hook starts");
        assert_eq!(b.system_block_tokens, Some(cut), "pad {pad}: both requests cut at the same token");

        let published = floor(cut);
        assert!(published > 0, "pad {pad}");
        assert_eq!(
            PromptContent::text(&a.tokens).key_at(published),
            PromptContent::text(&b.tokens).key_at(published),
            "pad {pad}: B claims the retained prefix A published"
        );
    }
}
