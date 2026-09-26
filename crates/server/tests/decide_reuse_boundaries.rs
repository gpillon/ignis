//! GitHub #270 (spec `docs/specs/decide/16-reuse-boundaries.md`) — where a
//! decision's `state` can be resumed, over the HTTP surface and a real
//! `ConcreteScheduler` on the deterministic `MockCompute` (ADR 0006).
//!
//! A `state` given as content parts stays in the user turn, so the system
//! block — the one place state used to be kept — never held any of it. The
//! server now predicts three more **reuse boundaries** for such a state and
//! hands them to the scheduler: a caller's **reuse marker**, an **observed
//! fork**, and a fan-out's **head**. What is observable here without a model
//! is where each request's prefill is cut and published, and where a later
//! request starts prefilling — which is the whole of what reuse changes.
//!
//! Under [`SimpleTemplateProvider`] a prompt is one token per word, message by
//! message, and an image is its run of `<|image_pad|>` ids, so every position
//! below is a word count a reader can check by hand. The block is the system
//! message ([`DIRECT_SYSTEM`], [`SYSTEM`] words), which floors to one page.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use tower::ServiceExt;

use ignis_core::decision::{AnswerAlphabet, LabelTokenizer};
use ignis_core::mock::MockCompute;
use ignis_core::types::TokenId;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::Server;
use ignis_server::decide::DIRECT_SYSTEM;
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{
    ChatMessage, ContentRejection, RenderedPrompt, SimpleTemplateProvider, TemplateProvider,
    TemplateRejection,
};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use serde_json::{Value as JsonValue, json};

#[path = "support/mod.rs"]
mod support;

const MODEL: &str = "test-model";
/// The default scheduler's KV page, in tokens.
const PAGE: u32 = 16;
/// [`DIRECT_SYSTEM`]'s words: the system block every readout question opens
/// with.
fn system() -> u32 {
    DIRECT_SYSTEM.split_whitespace().count() as u32
}

/// The page floor of `at`.
fn floor(at: u32) -> u32 {
    at / PAGE * PAGE
}

// ── the fixture ─────────────────────────────────────────────────────────

/// Every single character is one label, which is all a `noul` needs.
struct Letters;

impl LabelTokenizer for Letters {
    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        let mut chars = text.chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) => Some(vec![c as TokenId]),
            _ => None,
        }
    }

    fn decode(&self, ids: &[TokenId]) -> Option<String> {
        ids.iter().map(|&id| char::from_u32(id)).collect()
    }
}

/// [`SimpleTemplateProvider`] reporting what the real template reports: the
/// system block (the leading system message's words) and, when asked, where
/// each content part of the last message ends.
struct PartsTemplate;

fn system_block_of(messages: &[ChatMessage]) -> Option<u32> {
    let first = messages.first().filter(|m| m.role == "system")?;
    let words = first.content.text().split_whitespace().count() as u32;
    (words > 0).then_some(words)
}

impl TemplateProvider for PartsTemplate {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        let mut rendered = SimpleTemplateProvider.apply_chat_template(messages, options, tools)?;
        rendered.system_block_tokens = system_block_of(messages);
        Ok(rendered)
    }

    fn apply_chat_template_with_part_ends(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        let mut rendered =
            SimpleTemplateProvider.apply_chat_template_with_part_ends(messages, options, tools)?;
        rendered.system_block_tokens = system_block_of(messages);
        Ok(rendered)
    }

    fn prepare_multimodal(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
        media: Vec<ignis_artifact::vision::PreparedMedia>,
    ) -> Result<(RenderedPrompt, ignis_core::vision::Multimodal), ContentRejection> {
        let (mut rendered, multimodal) =
            SimpleTemplateProvider.prepare_multimodal(messages, options, tools, media)?;
        rendered.system_block_tokens = system_block_of(messages);
        Ok((rendered, multimodal))
    }

    fn prepare_multimodal_with_part_ends(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
        media: Vec<ignis_artifact::vision::PreparedMedia>,
    ) -> Result<(RenderedPrompt, ignis_core::vision::Multimodal), ContentRejection> {
        let (mut rendered, multimodal) = SimpleTemplateProvider
            .prepare_multimodal_with_part_ends(messages, options, tools, media)?;
        rendered.system_block_tokens = system_block_of(messages);
        Ok((rendered, multimodal))
    }

    fn answer_alphabet(&self) -> AnswerAlphabet {
        AnswerAlphabet::from_tokenizer(&Letters)
    }

    /// One token per character, so a `number`'s digits can be forced.
    fn encode_literal(&self, text: &str) -> Option<Vec<TokenId>> {
        Some(text.chars().map(|c| c as TokenId).collect())
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
}

fn scheduler(compute: Arc<dyn ignis_core::Compute>) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute,
    )
}

/// A `--vision` server over `compute`, images acquired from `data:` URIs by
/// the real processor.
fn server_over(compute: Arc<dyn ignis_core::Compute>) -> Server {
    use ignis_server::media::{MediaAcquirer, MediaPolicy};
    let limits = ignis_artifact::vision::ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 1 << 20,
        max_decoded_pixels: 1 << 20,
        max_raw_patches: 1 << 16,
        max_vision_tokens: 1 << 14,
    };
    Server::new(Engine::new(Box::new(scheduler(compute))), Box::new(PartsTemplate))
        .with_request_timeout(std::time::Duration::from_secs(10))
        .with_media(Arc::new(MediaAcquirer::new(
            Arc::new(support::media::processor(limits.clone())),
            limits,
            MediaPolicy::new(false, 1 << 20),
        )))
}

fn app(compute: Arc<MockCompute>) -> axum::Router {
    server_over(compute).app()
}

async fn decide(app: &axum::Router, body: &str) -> (u16, JsonValue) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decide")
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned().into_bytes()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let json = serde_json::from_str(&text).unwrap_or_else(|e| panic!("not JSON ({e}): {text}"));
    (status, json)
}

// ── bodies ──────────────────────────────────────────────────────────────

/// `n` distinct words tagged `tag`: two calls with different tags share none.
fn words(tag: &str, n: usize) -> String {
    (0..n).map(|i| format!("{tag}{i}")).collect::<Vec<_>>().join(" ")
}

fn text(text: &str) -> JsonValue {
    json!({"type": "text", "text": text})
}

/// A text part carrying a reuse marker.
fn marked(text: &str) -> JsonValue {
    json!({"type": "text", "text": text, "cache_control": {"type": "ephemeral"}})
}

/// A flat 64x64 PNG filled with `fill`, as an image part: two fills are two
/// pictures of one size, so their placeholder ids are the same and only the
/// media identity tells them apart.
fn image(fill: u8) -> JsonValue {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 64, 64);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&vec![fill; 64 * 64 * 3]).unwrap();
    }
    json!({"type": "image_url", "image_url": {"url": support::media::data_uri(&bytes)}})
}

/// One `noul` question over `state`.
fn one_question(state: &[JsonValue]) -> String {
    format!(
        r#"{{"state":{},"questions":{{"q":{{"type":"noul","instructions":"Is there a monster in the view?"}}}}}}"#,
        JsonValue::Array(state.to_vec())
    )
}

// ── what the engine was asked ───────────────────────────────────────────

/// The heads `request` published, in prompt order.
fn published(compute: &MockCompute, request: u64) -> Vec<u32> {
    compute
        .prefill_calls()
        .iter()
        .flatten()
        .filter(|j| j.request == request)
        .filter_map(|j| j.publish_prefix.map(|p| p.tokens))
        .collect()
}

/// Where `request`'s prefill started: 0, or the end of what it resumed from.
fn resumed_at(compute: &MockCompute, request: u64) -> u32 {
    compute
        .prefill_calls()
        .into_iter()
        .flatten()
        .find(|j| j.request == request)
        .map(|j| j.start_position)
        .expect("the request was prefilled")
}

/// The tokens each request was prefilled, by request id.
fn prefilled(compute: &MockCompute) -> BTreeMap<u64, u32> {
    let mut totals = BTreeMap::new();
    for job in compute.prefill_calls().into_iter().flatten() {
        *totals.entry(job.request).or_insert(0u32) += job.tokens.len() as u32;
    }
    totals
}

// ── the reuse marker ────────────────────────────────────────────────────

#[tokio::test]
async fn a_marker_that_is_not_exactly_ephemeral_is_refused_before_any_prefill() {
    // Retention here is by eviction, not by time: a `ttl` would be a promise
    // this server does not keep, so it is refused rather than dropped.
    for cache_control in [
        json!({"type": "ephemeral", "ttl": "5m"}),
        json!({"type": "persistent"}),
        json!("ephemeral"),
        json!({}),
    ] {
        let compute = Arc::new(MockCompute::new());
        let part = json!({"type": "text", "text": "Rules.", "cache_control": cache_control});
        let (status, body) = decide(&app(compute.clone()), &one_question(&[part])).await;
        assert_eq!(status, 422, "{cache_control}: {body}");
        assert_eq!(body["error"]["code"], "malformed_reuse_marker", "{cache_control}: {body}");
        assert!(compute.prefill_calls().is_empty(), "{cache_control}: refused before the GPU");
    }
}

#[tokio::test]
async fn more_than_four_markers_are_refused_and_four_are_served() {
    let compute = Arc::new(MockCompute::new());
    let five: Vec<JsonValue> = (0..5).map(|i| marked(&words(&format!("m{i}_"), 3))).collect();
    let (status, body) = decide(&app(compute.clone()), &one_question(&five)).await;
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["error"]["code"], "too_many_reuse_markers");
    assert!(compute.prefill_calls().is_empty(), "refused before the GPU");

    let (status, body) = decide(&app(compute.clone()), &one_question(&five[..4])).await;
    assert_eq!(status, 200, "{body}");
}

#[tokio::test]
async fn a_null_cache_control_is_no_marker_at_all() {
    // The spelling a client serializing an absent field produces: it asks for
    // nothing, so it is neither honoured nor refused.
    let compute = Arc::new(MockCompute::new());
    let part = json!({"type": "text", "text": words("s", 45), "cache_control": null});
    let (status, body) = decide(&app(compute.clone()), &one_question(&[part])).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(published(&compute, 0), vec![floor(system())], "the block alone");
}

#[tokio::test]
async fn a_marker_puts_a_boundary_at_its_parts_end_claimed_from_the_second_request() {
    let compute = Arc::new(MockCompute::new());
    let app = app(compute.clone());
    let statics = words("s", 45);
    let end = system() + 45;

    let (status, body) = decide(&app, &one_question(&[marked(&statics), text(&words("a", 10))])).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        published(&compute, 0),
        vec![floor(system()), floor(end)],
        "the block, then the marked part's end"
    );

    let (status, body) = decide(&app, &one_question(&[marked(&statics), text(&words("b", 10))])).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(resumed_at(&compute, 1), floor(end), "the second request resumes after the marked part");
}

// ── the observed fork ───────────────────────────────────────────────────

/// `[static text][varying text][varying image]` — the game agent's shape.
fn agent_state(varying: &str, fill: u8) -> Vec<JsonValue> {
    vec![text(&words("s", 45)), text(&words(varying, 10)), image(fill)]
}

#[tokio::test]
async fn the_second_request_repeating_a_static_part_keeps_it_and_the_third_resumes_there() {
    let compute = Arc::new(MockCompute::new());
    let app = app(compute.clone());
    let end = floor(system() + 45);

    for (request, (varying, fill)) in [("a", 10), ("b", 20), ("c", 30), ("d", 40)].into_iter().enumerate() {
        let (status, body) = decide(&app, &one_question(&agent_state(varying, fill))).await;
        assert_eq!(status, 200, "request {request}: {body}");
    }
    assert_eq!(published(&compute, 0), vec![floor(system())], "the first request keeps no fork");
    // From the second request on, the block is the first one's to claim.
    assert_eq!(
        published(&compute, 1),
        vec![end],
        "the second keeps the static part, and the text and image after it move nothing"
    );
    for request in [2, 3] {
        assert_eq!(resumed_at(&compute, request), end, "request {request} resumes after the static part");
        assert_eq!(published(&compute, request), Vec::<u32>::new(), "and keeps nothing more");
    }
}

#[tokio::test]
async fn a_request_whose_first_part_differs_keeps_no_fork() {
    let compute = Arc::new(MockCompute::new());
    let app = app(compute.clone());
    decide(&app, &one_question(&[text(&words("s", 45)), text(&words("a", 10))])).await;
    decide(&app, &one_question(&[text(&words("t", 45)), text(&words("a", 10))])).await;
    assert_eq!(resumed_at(&compute, 1), floor(system()), "the block, claimed");
    assert_eq!(published(&compute, 1), Vec::<u32>::new(), "and nothing kept past it");
}

#[tokio::test]
async fn the_same_parts_under_another_questions_instruction_are_another_run() {
    // A `number`'s system text is not a readout's, so the run is keyed apart.
    let compute = Arc::new(MockCompute::new());
    let app = app(compute.clone());
    let state = JsonValue::Array(vec![text(&words("s", 45)), text(&words("a", 10))]);
    let (status, body) = decide(
        &app,
        &format!(r#"{{"state":{state},"questions":{{"n":{{"type":"number","instructions":"How many?"}}}}}}"#),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    decide(&app, &one_question(&[text(&words("s", 45)), text(&words("b", 10))])).await;
    assert_eq!(published(&compute, 1), vec![floor(system())], "no fork across kinds");
    assert_eq!(resumed_at(&compute, 1), 0, "nor anything else: the block differs too");
}

#[tokio::test]
async fn each_kind_in_a_fan_out_keeps_its_own_fork() {
    // The second request over a repeated state, asking a readout and a
    // `number`: each kind's system text comes first, so each has its own run
    // ends, and the first question of each kind keeps its own fork.
    let compute = Arc::new(MockCompute::new());
    let app = app(compute.clone());
    let number_system = ignis_server::numbers::number_system(ignis_server::numbers::DEFAULT_DIGITS)
        .split_whitespace()
        .count() as u32;
    let body = |varying: &str| {
        let state = JsonValue::Array(vec![text(&words("s", 45)), text(&words(varying, 10))]);
        format!(
            r#"{{"state":{state},"questions":{{"q":{{"type":"noul","instructions":"Is it?"}},"n":{{"type":"number","instructions":"How many?"}}}}}}"#
        )
    };
    for varying in ["a", "b"] {
        let (status, response) = decide(&app, &body(varying)).await;
        assert_eq!(status, 200, "{response}");
    }
    // Requests 0 and 1 are the first decide request's questions, 2 and 3 the
    // second's.
    assert!(published(&compute, 2).contains(&floor(system() + 45)), "{:?}", published(&compute, 2));
    assert!(
        published(&compute, 3).contains(&floor(number_system + 45)),
        "{:?}",
        published(&compute, 3)
    );
}

#[tokio::test]
async fn a_request_with_a_marker_does_not_read_the_fork_history() {
    let compute = Arc::new(MockCompute::new());
    let app = app(compute.clone());
    let statics = words("s", 45);
    decide(&app, &one_question(&[text(&statics), text(&words("a", 20))])).await;
    decide(&app, &one_question(&[text(&statics), marked(&words("b", 20))])).await;
    assert_eq!(
        published(&compute, 1),
        vec![floor(system() + 65)],
        "its marker alone: the static part it repeats earns it no fork"
    );
}

#[tokio::test]
async fn a_request_with_a_marker_does_not_write_the_fork_history() {
    let compute = Arc::new(MockCompute::new());
    let app = app(compute.clone());
    let statics = words("s", 45);
    decide(&app, &one_question(&[text(&statics), marked(&words("a", 20))])).await;
    decide(&app, &one_question(&[text(&statics), text(&words("b", 20))])).await;
    assert_eq!(
        published(&compute, 1),
        Vec::<u32>::new(),
        "the marked request left nothing for it to find"
    );
}

// ── the fan-out head ────────────────────────────────────────────────────

/// Four `noul` questions over `state` whose instructions share their first
/// fifteen words: the head runs through the whole state and past a page
/// boundary into the question.
fn four_questions(state: &[JsonValue]) -> String {
    let asked = (0..4)
        .map(|i| {
            format!(
                r#""q{i}":{{"type":"noul","instructions":"Look at the view carefully and tell me whether there is a monster standing anywhere in region {i}"}}"#
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(r#"{{"state":{},"questions":{{{asked}}}}}"#, JsonValue::Array(state.to_vec()))
}

/// Every request's whole prompt, rebuilt from its prefill jobs: the first
/// request's jobs carry all of its prompt, and a follower's prompt is the
/// head it resumed from — the first request's, by construction — then its own
/// jobs.
fn prompts(compute: &MockCompute, first: u64) -> BTreeMap<u64, Vec<TokenId>> {
    let mut own: BTreeMap<u64, (u32, Vec<TokenId>)> = BTreeMap::new();
    for job in compute.prefill_calls().into_iter().flatten() {
        own.entry(job.request)
            .or_insert((job.start_position, Vec::new()))
            .1
            .extend(&job.tokens);
    }
    let whole = own[&first].1.clone();
    own.into_iter()
        .map(|(request, (start, tokens))| (request, [&whole[..start as usize], &tokens].concat()))
        .collect()
}

#[tokio::test]
async fn four_same_kind_questions_over_an_image_prefill_the_state_once() {
    // Spec 16's acceptance 3, on prefill token counts: every follower
    // prefills at most its own tail — what it does not share with the others
    // — and one page.
    //
    // The questions share fifteen words after the image on purpose: a head
    // reaches past a picture only when what its questions share after it
    // crosses the next page boundary (the next test is the other case).
    let compute = Arc::new(MockCompute::new());
    let state = [text(&words("s", 45)), image(10)];
    let (status, body) = decide(&app(compute.clone()), &four_questions(&state)).await;
    assert_eq!(status, 200, "{body}");

    let prompts = prompts(&compute, 0);
    assert_eq!(prompts.len(), 4, "one request per question");
    let first = &prompts[&0];
    let shared = prompts
        .values()
        .map(|p| first.iter().zip(p).take_while(|(a, b)| a == b).count() as u32)
        .min()
        .unwrap();
    let image_end = system() + 45 + 4;
    assert!(floor(shared) >= image_end, "the fixture's head runs past the image: {shared}");

    let prefilled = prefilled(&compute);
    assert_eq!(prefilled[&0], first.len() as u32, "the first question prefills everything");
    for request in 1..4u64 {
        let tail = prompts[&request].len() as u32 - shared;
        assert!(
            prefilled[&request] <= tail + PAGE,
            "question {request} prefilled {} tokens, its tail is {tail}",
            prefilled[&request]
        );
        assert!(resumed_at(&compute, request) >= image_end, "question {request} resumed past the image");
    }
}

#[tokio::test]
async fn a_head_that_would_end_inside_an_image_stops_before_it() {
    // #193's rule, which a fan-out's head keeps like every other boundary: a
    // shared prefix is whole pages and never half a picture. Here the image
    // spans a page boundary and the questions share one word after it, so the
    // head's page floor lands inside the image and walks back before it — and
    // every follower prefills the picture again.
    let compute = Arc::new(MockCompute::new());
    let state = [text(&words("s", 38)), image(10)];
    let image_begin = system() + 38;
    assert!(floor(image_begin + 4 + 1) > image_begin, "the fixture's floor lands inside the image");
    let asked = (0..4)
        .map(|i| format!(r#""q{i}":{{"type":"noul","instructions":"Question {i}: is there a monster?"}}"#))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(r#"{{"state":{},"questions":{{{asked}}}}}"#, JsonValue::Array(state.to_vec()));
    let (status, response) = decide(&app(compute.clone()), &body).await;
    assert_eq!(status, 200, "{response}");
    for request in 1..4u64 {
        assert_eq!(resumed_at(&compute, request), floor(image_begin), "question {request}");
    }
}

#[tokio::test]
async fn a_choice_and_a_point_share_no_head() {
    // Different kinds put different system text first, so their common prefix
    // is nothing a page could hold: the first question keeps exactly what it
    // would keep asked alone.
    let state = JsonValue::Array(vec![text(&words("s", 45)), image(10)]);
    let choice = r#""c":{"type":"choice","instructions":"Which door?","criteria":{"left":null,"right":null}}"#;
    let point = r#""p":{"type":"point","instructions":"Where is the door?"}"#;

    let alone = Arc::new(MockCompute::new());
    let (status, body) =
        decide(&app(alone.clone()), &format!(r#"{{"state":{state},"questions":{{{choice}}}}}"#)).await;
    assert_eq!(status, 200, "{body}");

    let together = Arc::new(MockCompute::new());
    let (status, body) =
        decide(&app(together.clone()), &format!(r#"{{"state":{state},"questions":{{{choice},{point}}}}}"#)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(published(&together, 0), published(&alone, 0), "no head, no slot");
}

// ── the fan-out head's lifetime ─────────────────────────────────────────

/// `GET /metrics` off the metrics listener's own app.
async fn scrape(metrics: &axum::Router) -> String {
    let request = Request::builder().method("GET").uri("/metrics").body(Body::empty()).unwrap();
    let response = metrics.clone().oneshot(request).await.unwrap();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// The sample `series` (name and labels, exactly as exposed).
fn value(text: &str, series: &str) -> u64 {
    text.lines()
        .find_map(|line| line.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
        .unwrap_or_else(|| panic!("no {series} in:\n{text}"))
}

fn in_use(text: &str) -> u64 {
    value(text, r#"ignis_retained_slots{state="in_use"}"#)
}

/// Scrape until `done` holds: the projection is fed asynchronously, after the
/// responses that caused it.
async fn scrape_until(metrics: &axum::Router, done: impl Fn(&str) -> bool) -> String {
    let mut last = String::new();
    let settled = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            last = scrape(metrics).await;
            if done(&last) {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(settled.is_ok(), "the projection never settled; last scrape:\n{last}");
    last
}

/// A server with metrics over `compute`, after one question over another
/// state has left the system block retained: the slots a fan-out must give
/// back to are the ones held here.
async fn warmed(compute: Arc<dyn ignis_core::Compute>) -> (axum::Router, axum::Router, u64) {
    let server = server_over(compute).with_metrics();
    let metrics = server.metrics_app().expect("metrics are on");
    let app = server.app();
    let (status, body) = decide(&app, &one_question(&[text(&words("w", 45))])).await;
    assert_eq!(status, 200, "{body}");
    let text = scrape_until(&metrics, |t| value(t, "ignis_requests_completed_total") == 1).await;
    let before = in_use(&text);
    assert_eq!(before, 1, "the block");
    (app, metrics, before)
}

fn fan_out_state() -> [JsonValue; 2] {
    [text(&words("s", 45)), image(10)]
}

#[tokio::test]
async fn an_answered_fan_out_gives_its_heads_slot_back() {
    let compute = Arc::new(MockCompute::new());
    let (app, metrics, before) = warmed(compute.clone()).await;
    let (status, body) = decide(&app, &four_questions(&fan_out_state())).await;
    assert_eq!(status, 200, "{body}");
    assert!(resumed_at(&compute, 2) > floor(system()), "the followers had a head to claim");
    scrape_until(&metrics, |t| value(t, "ignis_requests_completed_total") == 5 && in_use(t) == before).await;
}

/// Answers `victim`'s prefill with no readout, so that one question fails
/// and its siblings do not.
struct DropReadout {
    inner: MockCompute,
    victim: u64,
}

impl ignis_core::Compute for DropReadout {
    fn prefill_step(
        &self,
        jobs: &[ignis_core::PrefillJob],
    ) -> Result<Vec<ignis_core::PrefillOutcome>, ignis_core::ComputeError> {
        let mut outcomes = self.inner.prefill_step(jobs)?;
        for (job, outcome) in jobs.iter().zip(outcomes.iter_mut()) {
            if job.request == self.victim {
                outcome.readout = None;
            }
        }
        Ok(outcomes)
    }

    fn decode_step(
        &self,
        jobs: &[ignis_core::DecodeJob],
    ) -> Result<Vec<ignis_core::DecodeOutcome>, ignis_core::ComputeError> {
        self.inner.decode_step(jobs)
    }

    fn release(&self, request: u64) {
        self.inner.release(request);
    }
}

#[tokio::test]
async fn a_fan_out_with_a_failed_question_gives_its_heads_slot_back() {
    // The warm-up is request 0 and the fan-out's first question request 1.
    let compute = Arc::new(DropReadout { inner: MockCompute::new(), victim: 3 });
    let (app, metrics, before) = warmed(compute.clone()).await;
    let (status, body) = decide(&app, &four_questions(&fan_out_state())).await;
    assert_eq!(status, 200, "{body}");
    let failed = body["answers"].as_object().unwrap().values().filter(|a| a["type"] == "error").count();
    assert_eq!(failed, 1, "{body}");
    scrape_until(&metrics, |t| value(t, "ignis_requests_completed_total") == 5 && in_use(t) == before).await;
}

/// Holds the first prefill carrying a request at or past `follower` until
/// the harness says go.
struct HoldFollower {
    inner: MockCompute,
    follower: u64,
    entered: std::sync::mpsc::SyncSender<()>,
    go: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
    held: std::sync::atomic::AtomicBool,
}

impl ignis_core::Compute for HoldFollower {
    fn prefill_step(
        &self,
        jobs: &[ignis_core::PrefillJob],
    ) -> Result<Vec<ignis_core::PrefillOutcome>, ignis_core::ComputeError> {
        let follower = jobs.iter().any(|job| job.request >= self.follower);
        if follower && !self.held.swap(true, std::sync::atomic::Ordering::SeqCst) {
            let _ = self.entered.send(());
            let _ = self.go.lock().unwrap().recv();
        }
        self.inner.prefill_step(jobs)
    }

    fn decode_step(
        &self,
        jobs: &[ignis_core::DecodeJob],
    ) -> Result<Vec<ignis_core::DecodeOutcome>, ignis_core::ComputeError> {
        self.inner.decode_step(jobs)
    }

    fn release(&self, request: u64) {
        self.inner.release(request);
    }
}

#[tokio::test]
async fn a_fan_out_whose_client_left_gives_its_heads_slot_back() {
    use std::sync::mpsc::sync_channel;
    let (entered_tx, entered_rx) = sync_channel(0);
    let (go_tx, go_rx) = sync_channel(0);
    let compute = Arc::new(HoldFollower {
        inner: MockCompute::new(),
        // The warm-up is request 0, the fan-out's first question request 1.
        follower: 2,
        entered: entered_tx,
        go: std::sync::Mutex::new(go_rx),
        held: std::sync::atomic::AtomicBool::new(false),
    });
    let (app, metrics, before) = warmed(compute.clone()).await;

    let body = four_questions(&fan_out_state());
    let client = tokio::spawn(async move { decide(&app, &body).await });
    tokio::task::spawn_blocking(move || entered_rx.recv())
        .await
        .expect("the waiter ran")
        .expect("a follower reached the compute");
    // While the followers are held, the head is kept.
    scrape_until(&metrics, |t| in_use(t) == before + 1).await;

    client.abort();
    support::nudge().await;
    go_tx.send(()).expect("the held prefill is still waiting");
    scrape_until(&metrics, |t| in_use(t) == before).await;
}
