//! The schema description on the **real frontend** (GitHub #589, ADR 0089):
//! a `json_schema` request's prompt is the conversation, then the
//! description in a block of its own, then the assistant's opener -- before
//! the think block it opens with thinking on -- and the next turn resumes
//! from where the first turn's conversation ended.
//!
//! The placement and the boundary are pinned on a fixture template in
//! `artifact_template.rs`, and the server's side over the mock in
//! `structured_output_http.rs`. What only the artifact's own template and
//! tokenizer can show is that the description lands in its block exactly
//! there on the 27B's template, that the boundary is an exact token prefix of
//! the real tokenization, and that the real scheduler resumes the next turn
//! from it.
//!
//! CPU-only: the backend is the mock, the tokenizer and the template are
//! real. Skips when the artifact is not at its machine-local path, the
//! convention `checkpoint_lineage.rs` follows.

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::ChatTemplate;
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler, SchedulerConfig};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::template::{ChatMessage, RenderedPrompt, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

/// What the server describes a small strict schema as (`SchemaFormat::describe`).
const DESCRIPTION: &str = "Respond with a JSON object that matches this JSON schema:\n\
{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}},\"required\":[\"name\",\"age\"],\"additionalProperties\":false}";

fn provider_or_skip() -> Option<ArtifactTemplateProvider> {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return None;
    }
    let reader = ignis_artifact::Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    let frontend = ignis_artifact::FrontendSet::from_reader(&reader).expect("frontend set");
    Some(ArtifactTemplateProvider::new(frontend))
}

fn first_turn() -> Vec<ChatMessage> {
    vec![
        ChatMessage::text("system", "You describe animals for a catalogue, one record at a time."),
        ChatMessage::text("user", "Describe the cat in the photo on the left: its name and its age."),
    ]
}

fn options(enable_thinking: bool) -> ThinkingOptions {
    ThinkingOptions { enable_thinking, ..ThinkingOptions::default() }
}

#[test]
fn the_description_sits_between_the_conversation_and_the_assistants_opener() {
    let Some(provider) = provider_or_skip() else {
        return;
    };
    for enable_thinking in [false, true] {
        let options = options(enable_thinking);
        // Today's prompt, cut at the start of its generation opener: the
        // conversation, then what the template opens the assistant's turn with.
        let plain = provider.render_text(&first_turn(), &options, &[]).expect("render");
        let opener = ChatTemplate::generation_opener_offset(&plain).expect("an opener")
            - ChatTemplate::GENERATION_OPENER.len();
        let (conversation, turn) = plain.split_at(opener);
        assert!(turn.starts_with("<|im_start|>assistant\n<think>\n"), "{turn:?}");

        let described = provider
            .apply_chat_template_described(&first_turn(), &options, &[], DESCRIPTION)
            .expect("render");
        let text = provider.render_tokens(&described.tokens);
        assert_eq!(
            text,
            format!("{conversation}<|im_start|>system\n{DESCRIPTION}<|im_end|>\n{turn}"),
            "thinking {enable_thinking}"
        );
        // The checkpoint boundary is the conversation's end, as an exact
        // token prefix of the real tokenization.
        let head = provider.encode_literal(conversation).expect("the tokenizer");
        assert_eq!(described.opener_tokens, Some(head.len() as u32), "thinking {enable_thinking}");
        assert!(described.tokens.starts_with(&head));
        // The description off is today's prompt, byte for byte.
        let off = provider.apply_chat_template(&first_turn(), &options, &[]).expect("render");
        assert_eq!(provider.render_tokens(&off.tokens), plain);
    }
}

/// `api.rs`'s own construction of a text-only request from a render.
fn request_input(rendered: RenderedPrompt) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: rendered.tokens,
        params: DecodeParams { max_tokens: Some(4), ..DecodeParams::default() },
        multimodal: None,
        opener_tokens: rendered.opener_tokens,
        user_turn_tokens: rendered.user_turn_tokens,
        system_block_tokens: rendered.system_block_tokens,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
    }
}

/// The leading prompt tokens `request` resumed from retained state.
fn reused(events: &[SchedEvent], request: RequestId) -> Option<u32> {
    events.iter().find_map(|e| match e {
        SchedEvent::StateReused { request: r, tokens, .. } if *r == request => Some(*tokens),
        _ => None,
    })
}

#[test]
fn the_next_turn_resumes_where_the_described_turns_conversation_ended() {
    let Some(provider) = provider_or_skip() else {
        return;
    };
    let options = options(true);
    let first = provider
        .apply_chat_template_described(&first_turn(), &options, &[], DESCRIPTION)
        .expect("render the first turn");
    let mut next = first_turn();
    next.push(ChatMessage::text("assistant", "{\"name\": \"Tom\", \"age\": 3}"));
    next.push(ChatMessage::text("user", "And the dog on the right?"));
    let second = provider
        .apply_chat_template_described(&next, &options, &[], DESCRIPTION)
        .expect("render the second turn");
    let conversation_end = first.opener_tokens.expect("the first turn's boundary");
    assert!(second.tokens.starts_with(&first.tokens[..conversation_end as usize]));
    assert!(!second.tokens.starts_with(&first.tokens), "the description is not the next turn's history");

    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let mut resumed = Vec::new();
    for prompt in [first, second] {
        let id = sched.submit(request_input(prompt), RequestClass::Interactive).expect("submit");
        let mut events = Vec::new();
        while !sched.is_idle() {
            events.extend(sched.advance());
        }
        resumed.push(reused(&events, id));
    }
    assert_eq!(resumed, [None, Some(conversation_end)], "the second turn resumes from the first turn's conversation");
}
