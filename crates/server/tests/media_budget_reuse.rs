//! An agent's image history past the vision budget keeps its prefix reuse
//! (GitHub #673, ADR 0102). A harness resends the whole history every turn,
//! one image more each time; past the budget the oldest images are dropped a
//! whole block at a time, so the prompt changes at its first dropped image
//! once a block, and between two drops each turn's prompt extends the last.
//!
//! - On the template's output: the placeholder's render of each turn, the
//!   dropped images as their text where they were.
//! - Over the mock engine: what each turn reports reused, `cached_tokens`,
//!   covers the turn before it but on the turns that drop a block.
//!
//! Under [`SimpleTemplateProvider`] a prompt is one token per word and an
//! image is its run of `<|image_pad|>` ids; every image here is 64 x 64, 4
//! merged tokens, under a 16-token budget, so a block is two images.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;

use ignis_artifact::vision::{PreparedMedia, ProcessorOptions};
use ignis_core::mock::MockCompute;
use ignis_core::vision::Multimodal;
use ignis_core::{ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::media::{MediaAcquirer, MediaPolicy, OMITTED_IMAGE};
use ignis_server::template::{
    ChatMessage, ContentPart, ContentRejection, MessageContent, RenderedPrompt, SimpleTemplateProvider,
    TemplateProvider, TemplateRejection,
};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

#[path = "support/mod.rs"]
mod support;
use support::media::{data_uri, processor, shade};

const MODEL: &str = "test-model";
/// Turns of the conversation: twelve images, four blocks dropped.
const TURNS: usize = 12;

fn limits() -> ProcessorOptions {
    ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 1 << 20,
        max_decoded_pixels: 1 << 20,
        max_raw_patches: 1 << 16,
        max_vision_tokens: 16,
    }
}

fn acquirer() -> MediaAcquirer {
    MediaAcquirer::new(Arc::new(processor(limits())), limits(), MediaPolicy::new(false, 1 << 20))
}

/// What the user asks of image `i`: past one 16-token KV page, where the
/// first checkpoint is taken.
fn question(i: usize) -> String {
    format!("read crop {i} of the page and say in a few plain words what it shows and where it sits")
}

/// Image `i` of the conversation, each its own bytes.
fn image(i: usize) -> String {
    data_uri(&shade(64, 64, 10 + i as u8))
}

/// The conversation at turn `n` (1-based): `n` user messages, each a text
/// part then one image, the assistant's answer between them.
fn conversation(n: usize) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    for i in 0..n {
        let part = |kind: &str, text: Option<String>, url: Option<String>| ContentPart {
            kind: Some(kind.to_owned()),
            text,
            url,
            cache_control: None,
        };
        let mut user = ChatMessage::text("user", "");
        user.content =
            MessageContent::Parts(vec![part("text", Some(question(i)), None), part("image_url", None, Some(image(i)))]);
        messages.push(user);
        if i + 1 < n {
            messages.push(ChatMessage::text("assistant", &format!("crop {i} reads {i}")));
        }
    }
    messages
}

#[tokio::test]
async fn between_two_drops_each_turns_prompt_extends_the_last_and_a_drop_changes_it_at_its_first_image() {
    let acquirer = acquirer();
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let omitted = SimpleTemplateProvider
        .apply_chat_template(&[ChatMessage::text("user", OMITTED_IMAGE)], &ThinkingOptions::default(), &[])
        .unwrap()
        .tokens;
    let mut turns: Vec<(usize, Vec<TokenId>, Multimodal)> = Vec::new();
    for n in 1..=TURNS {
        let messages = conversation(n);
        let acquired = acquirer.acquire(&messages, deadline).await.unwrap();
        let rendered = acquired.omit_dropped(&messages);
        let (prompt, multimodal) = SimpleTemplateProvider
            .prepare_multimodal(&rendered, &ThinkingOptions::default(), &[], acquired.media.clone())
            .unwrap();
        assert_eq!(multimodal.media.len() + acquired.dropped, n, "turn {n}");
        let pads = prompt.tokens.iter().filter(|&&t| t == ignis_artifact::vision::IMAGE_PAD_ID).count();
        assert_eq!(pads as u64, acquired.stats.vision_tokens, "turn {n}: placeholders for the kept images alone");
        turns.push((acquired.dropped, prompt.tokens, multimodal));
    }
    assert_eq!(turns.iter().map(|t| t.0).collect::<Vec<_>>(), [0, 0, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8]);
    let mut drops = 0;
    for (turn, pair) in turns.windows(2).enumerate() {
        let ((dropped, before, multimodal), (now_dropped, after, now)) = (&pair[0], &pair[1]);
        let n = turn + 1;
        if dropped == now_dropped {
            // Up to its last image -- here its whole prompt -- each turn is a
            // prefix of the next.
            let last = multimodal.media.last().unwrap().token_span;
            let through = last.begin + last.count;
            assert_eq!(before[..through], after[..through], "turn {n} to {}", n + 1);
            continue;
        }
        drops += 1;
        // A drop: the prompt holds until the first image the new block drops
        // -- this turn's first kept one -- which the next turn reads as the
        // placeholder text instead.
        assert_eq!(now_dropped - dropped, 2, "one block of two");
        let first = multimodal.media[0].token_span.begin;
        assert_eq!(before[..first], after[..first], "turn {n} to {}: unchanged before the drop", n + 1);
        assert_eq!(after[first..first + omitted.len()], omitted[..], "turn {}: the text where the image was", n + 1);
        assert!(now.media[0].token_span.begin > first, "the kept images come after it");
    }
    assert_eq!(drops, 4, "once a block");
}

/// [`SimpleTemplateProvider`] with a generation opener: one token after the
/// conversation, where the checkpoint the next turn resumes from is cut.
struct OpenerTemplate;

impl TemplateProvider for OpenerTemplate {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn prepare_multimodal(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
        media: Vec<PreparedMedia>,
    ) -> Result<(RenderedPrompt, Multimodal), ContentRejection> {
        let mut opened = messages.to_vec();
        opened.push(ChatMessage::text("assistant", "<opener>"));
        let (mut rendered, multimodal) = SimpleTemplateProvider.prepare_multimodal(&opened, options, tools, media)?;
        rendered.opener_tokens = Some(rendered.tokens.len() as u32 - 1);
        Ok((rendered, multimodal))
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        SimpleTemplateProvider.render_tokens(tokens)
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        SimpleTemplateProvider.thinking_capabilities()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        SimpleTemplateProvider.token_decoder()
    }

    fn decoder_starts_in_reasoning(&self, _thinking: &ThinkingOptions) -> bool {
        false
    }
}

/// The request for turn `n`, as a harness sends it.
fn turn_request(n: usize) -> Request<Body> {
    let mut messages = Vec::new();
    for i in 0..n {
        messages.push(json!({"role": "user", "content": [
            {"type": "text", "text": question(i)},
            {"type": "image_url", "image_url": {"url": image(i)}},
        ]}));
        if i + 1 < n {
            messages.push(json!({"role": "assistant", "content": format!("crop {i} reads {i}")}));
        }
    }
    let body = json!({"model": MODEL, "max_tokens": 2, "temperature": 0, "messages": messages});
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn a_growing_image_history_reuses_the_turn_before_but_where_a_block_is_dropped() {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let app = Server::new(Engine::new(Box::new(scheduler)), Box::new(OpenerTemplate))
        .with_request_timeout(Duration::from_secs(10))
        .with_media(Arc::new(acquirer()))
        .app();
    // Each turn: (where its conversation ends, what it reused).
    let mut turns = Vec::new();
    for n in 1..=TURNS {
        let response = app.clone().oneshot(turn_request(n)).await.unwrap();
        let status = response.status().as_u16();
        let body: Value = serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
        assert_eq!(status, 200, "turn {n}: {body}");
        let usage = &body["usage"];
        let prompt = usage["prompt_tokens"].as_u64().unwrap();
        let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
        turns.push((prompt - 1, cached));
    }
    // The turns that drop a block: 5, 7, 9 and 11 (the acquisition's own
    // test works the counts out).
    let drops = [5, 7, 9, 11];
    for n in 2..=TURNS {
        let ((conversation, _), (_, cached)) = (turns[n - 2], turns[n - 1]);
        if drops.contains(&n) {
            assert!(cached < conversation, "turn {n} drops a block: {turns:?}");
        } else {
            assert_eq!(cached, conversation, "turn {n} resumes where turn {} ended: {turns:?}", n - 1);
        }
    }
}
