//! GitHub #239 — `POST /v1/decide` over the HTTP surface, end to end
//! through a real `ConcreteScheduler` on the deterministic `MockCompute`
//! (ADR 0006): no GPU anywhere.
//!
//! What the mock can and cannot say. Its readout is deterministic but
//! arbitrary — it models a distribution, not this model's — so nothing here
//! asserts *which* option wins. What it does assert is everything the
//! endpoint is responsible for: the shape Jev's clients expect, that a
//! refusal costs no prefill, and that a decision generates nothing.

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use tower::ServiceExt;

use ignis_core::decision::{AnswerAlphabet, LabelTokenizer};
use ignis_core::mock::MockCompute;
use ignis_core::types::TokenId;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::decoder::TokenDecoder;
use ignis_server::template::{
    ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection,
};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use serde_json::{Value as JsonValue, json};

const MODEL: &str = "test-model";

/// A tokenizer wide enough to reach the 256-option ceiling: every single
/// character and every uppercase bigram, 738 labels.
struct WideTokenizer(Vec<String>);

impl WideTokenizer {
    fn new() -> Self {
        let mut entries: Vec<String> = ('A'..='Z')
            .chain('a'..='z')
            .chain('0'..='9')
            .map(|c| c.to_string())
            .collect();
        for a in 'A'..='Z' {
            for b in 'A'..='Z' {
                entries.push(format!("{a}{b}"));
            }
        }
        Self(entries)
    }
}

impl LabelTokenizer for WideTokenizer {
    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        self.0
            .iter()
            .position(|entry| entry == text)
            .map(|index| vec![index as TokenId])
    }

    fn decode(&self, ids: &[TokenId]) -> Option<String> {
        ids.iter()
            .map(|&id| self.0.get(id as usize).cloned())
            .collect::<Option<Vec<String>>>()
            .map(|pieces| pieces.concat())
    }
}

/// [`SimpleTemplateProvider`] with an answer alphabet, which the real
/// provider gets from the loaded artifact's tokenizer. Without one
/// `/v1/decide` refuses every request, which is the right behaviour for a
/// load that cannot name answers and the wrong fixture for testing one.
///
/// It also reports a **system block** the placeholder does not (GitHub
/// #240). The placeholder renders no chat markers at all, so it has no
/// structure to report and nothing it produces is ever reused — which
/// makes it unable to show the one thing a fan-out is for. Here the block
/// is exactly the first message when it is the system one, which is what
/// the real template's block is too: the instruction and, since #240, the
/// evidence.
struct DecidingTemplate;

/// How many of `messages`' tokens the leading system message accounts for,
/// under [`SimpleTemplateProvider`]'s one-token-per-word rendering.
fn system_block_of(messages: &[ChatMessage]) -> Option<u32> {
    let first = messages.first().filter(|m| m.role == "system")?;
    let words = first.content.text().split_whitespace().count() as u32;
    (words > 0).then_some(words)
}

impl TemplateProvider for DecidingTemplate {
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

    fn answer_alphabet(&self) -> AnswerAlphabet {
        AnswerAlphabet::from_tokenizer(&WideTokenizer::new())
    }

    fn prepare_multimodal(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
        media: Vec<ignis_artifact::vision::PreparedMedia>,
    ) -> Result<(RenderedPrompt, ignis_core::vision::Multimodal), ignis_server::template::ContentRejection>
    {
        let (mut rendered, multimodal) =
            SimpleTemplateProvider.prepare_multimodal(messages, options, tools, media)?;
        rendered.system_block_tokens = system_block_of(messages);
        Ok((rendered, multimodal))
    }

    /// One token per character (GitHub #242): every digit is a single
    /// vocabulary entry, which is what a forced alphabet needs, and an
    /// arbitrary literal has an encoding. The ids are the characters' own,
    /// so a test can read a forced prefix straight out of the prompt.
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

fn app(compute: Arc<MockCompute>) -> axum::Router {
    server(compute).app()
}

fn server(compute: Arc<MockCompute>) -> Server {
    server_over(compute)
}

/// [`server`] over any compute, for the fan-out tests' decorators.
fn server_over(compute: Arc<dyn ignis_core::Compute>) -> Server {
    Server::new(Engine::new(Box::new(scheduler_over(compute))), Box::new(DecidingTemplate))
}

fn scheduler_over(compute: Arc<dyn ignis_core::Compute>) -> ConcreteScheduler {
    scheduler_admitting(compute, SchedulerConfig::default().max_in_flight)
}

fn scheduler_admitting(
    compute: Arc<dyn ignis_core::Compute>,
    max_in_flight: usize,
) -> ConcreteScheduler {
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            max_in_flight,
            ..SchedulerConfig::default()
        },
        compute,
    )
}

/// POST raw JSON text — never a `serde_json::Value`, whose object is sorted
/// in this build and would lose the declared option order on the way in.
async fn post(app: &axum::Router, path: &str, body: &str) -> (u16, JsonValue) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned().into_bytes()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let json = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("the response is JSON ({e}): {text}"));
    (status, json)
}

async fn decide(app: &axum::Router, body: &str) -> (u16, JsonValue) {
    post(app, "/v1/decide", body).await
}

// ── Jev's documented examples over the wire (acceptance 1, 9) ────────────

const JEV_NOUL: &str = r#"{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "test-model",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?",
      "criteria": { "true": "Explicitly time-sensitive", "false": "No urgency expressed" }
    }
  }
}"#;

const JEV_CHOICE: &str = r#"{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "test-model",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    }
  }
}"#;

const JEV_SCORE: &str = r#"{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "test-model",
  "questions": {
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}"#;

#[tokio::test]
async fn jevs_noul_example_is_served_in_jevs_shape() {
    let compute = Arc::new(MockCompute::new());
    let (status, body) = decide(&app(compute), JEV_NOUL).await;

    assert_eq!(status, 200, "{body}");
    assert_eq!(body["model"], MODEL);
    let answer = &body["answers"]["is_urgent"];
    assert_eq!(answer["type"], "noul");
    let noul = answer["noul"].as_f64().expect("a number");
    assert!((0.0..=1.0).contains(&noul), "the yes/no scale: {noul}");
    assert_eq!(
        answer.as_object().expect("an object").len(),
        2,
        "a noul answer carries `type` and `noul` alone: {answer}"
    );
    assert_eq!(
        body["usage"]["output_tokens"], 0,
        "a decision generates nothing, and says so"
    );
    assert!(
        body["usage"]["input_tokens"].as_u64().expect("a number") > 0,
        "the prompt it did pay for is reported"
    );
}

#[tokio::test]
async fn jevs_choice_example_is_served_in_jevs_shape() {
    let compute = Arc::new(MockCompute::new());
    let (status, body) = decide(&app(compute), JEV_CHOICE).await;

    assert_eq!(status, 200, "{body}");
    let answer = &body["answers"]["department"];
    assert_eq!(answer["type"], "choice");
    let probabilities = answer["probabilities"].as_object().expect("a map");
    assert_eq!(
        probabilities.keys().cloned().collect::<Vec<_>>(),
        vec!["billing".to_owned(), "sales".to_owned(), "technical".to_owned()],
        "every declared option is priced"
    );
    let sum: f64 = probabilities.values().map(|p| p.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-9, "the distribution sums to one: {sum}");
    let choice = answer["choice"].as_str().expect("a string");
    assert!(
        probabilities.contains_key(choice),
        "the chosen option is one of the declared ones: {choice}"
    );
    let confidence = answer["confidence"].as_f64().expect("a number");
    assert!((0.0..=1.0).contains(&confidence), "{confidence}");
}

#[tokio::test]
async fn jevs_score_example_is_served_in_jevs_shape() {
    let compute = Arc::new(MockCompute::new());
    let (status, body) = decide(&app(compute), JEV_SCORE).await;

    assert_eq!(status, 200, "{body}");
    let answer = &body["answers"]["frustration"];
    assert_eq!(answer["type"], "score");
    let score = answer["score"].as_f64().expect("a number");
    assert!(
        (0.0..=2.0).contains(&score),
        "the score lands inside the levels it is an expectation over: {score}"
    );
    assert_eq!(
        answer["legend"],
        json!({ "0": "Calm", "1": "Frustrated", "2": "Very angry" }),
        "each level mapped back to what the caller wrote"
    );
    assert_eq!(answer["probabilities"].as_object().expect("a map").len(), 3);
    assert!(answer["confidence"].is_number());
}

#[tokio::test]
async fn the_jev_route_name_reaches_the_same_endpoint() {
    let compute = Arc::new(MockCompute::new());
    let app = app(compute);
    let (decide_status, decide_body) = post(&app, "/v1/decide", JEV_NOUL).await;
    let (jev_status, jev_body) = post(&app, "/v1/systemone", JEV_NOUL).await;
    assert_eq!(decide_status, 200);
    assert_eq!(jev_status, 200, "an unmodified Jev client changes only the host");
    assert_eq!(
        decide_body["answers"]["is_urgent"]["type"],
        jev_body["answers"]["is_urgent"]["type"]
    );
}

#[tokio::test]
async fn many_questions_over_one_state_are_all_answered() {
    let compute = Arc::new(MockCompute::new());
    let body = r#"{
      "state": "Help! My payouts have been failing for 3 days.",
      "questions": {
        "is_urgent": { "type": "noul", "instructions": "Urgent?" },
        "department": { "type": "choice", "instructions": "Which team?", "criteria": { "billing": null, "technical": null } },
        "frustration": { "type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Angry"] }
      }
    }"#;
    let (status, response) = decide(&app(compute), body).await;

    assert_eq!(status, 200, "{response}");
    let answers = response["answers"].as_object().expect("a map");
    assert_eq!(answers.len(), 3, "one answer per question, under its own id");
    assert_eq!(answers["is_urgent"]["type"], "noul");
    assert_eq!(answers["department"]["type"], "choice");
    assert_eq!(answers["frustration"]["type"], "score");
    assert_eq!(response["usage"]["output_tokens"], 0);
}

// ── thinking is refused, never ignored (acceptance 5) ────────────────────

#[tokio::test]
async fn asking_for_thinking_is_refused_with_the_reason() {
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"s","enable_thinking":true,"questions":{"q":{"type":"noul","instructions":"Urgent?"}}}"#;
    let (status, response) = decide(&app(compute.clone()), body).await;

    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "thinking_unsupported");
    let message = response["error"]["message"].as_str().expect("a message");
    assert!(
        message.contains("enable_thinking") && message.contains("one position"),
        "the refusal names the field and why it cannot be honoured: {message}"
    );
    assert!(
        compute.prefill_calls().is_empty(),
        "and it costs no prefill"
    );
}

#[tokio::test]
async fn asking_for_thinking_in_any_shape_this_server_reads_is_refused() {
    // Every field `thinking.rs` resolves thinking from, not only the
    // obvious one — a request that asked through `chat_template_kwargs`, or
    // through an effort level that *implies* thinking, must not be served
    // by a path that happened to look at a different field.
    for body in [
        r#"{"state":"s","chat_template_kwargs":{"enable_thinking":true},"questions":{"q":{"type":"noul","instructions":"Urgent?"}}}"#,
        r#"{"state":"s","reasoning_effort":"high","questions":{"q":{"type":"noul","instructions":"Urgent?"}}}"#,
    ] {
        let compute = Arc::new(MockCompute::new());
        let (status, response) = decide(&app(compute.clone()), body).await;
        assert_eq!(status, 422, "{body} was served: {response}");
        assert_eq!(response["error"]["code"], "thinking_unsupported", "{response}");
        assert!(compute.prefill_calls().is_empty(), "and cost no prefill");
    }
}

#[tokio::test]
async fn thinking_turned_off_explicitly_is_served() {
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"s","enable_thinking":false,"questions":{"q":{"type":"noul","instructions":"Urgent?"}}}"#;
    let (status, response) = decide(&app(compute), body).await;
    assert_eq!(status, 200, "saying `false` agrees with us: {response}");
}

// ── the ceiling (acceptance 6) ───────────────────────────────────────────

fn wide_choice(count: usize) -> String {
    let criteria: Vec<String> = (0..count)
        .map(|index| format!(r#""option{index}":null"#))
        .collect();
    format!(
        r#"{{"state":"s","questions":{{"q":{{"type":"choice","instructions":"Which?","criteria":{{{}}}}}}}}}"#,
        criteria.join(",")
    )
}

#[tokio::test]
async fn the_measured_ceiling_is_served() {
    let compute = Arc::new(MockCompute::new());
    let (status, response) = decide(&app(compute), &wide_choice(256)).await;
    assert_eq!(status, 200, "256 options is measured and served: {response}");
    assert_eq!(
        response["answers"]["q"]["probabilities"]
            .as_object()
            .expect("a map")
            .len(),
        256
    );
}

#[tokio::test]
async fn one_option_past_the_ceiling_is_refused_before_the_gpu() {
    let compute = Arc::new(MockCompute::new());
    let (status, response) = decide(&app(compute.clone()), &wide_choice(257)).await;
    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "too_many_options");
    assert!(compute.prefill_calls().is_empty(), "no prefill was spent");
}

// ── validation is all-or-nothing, and early (acceptance 7) ───────────────

#[tokio::test]
async fn a_malformed_question_refuses_the_whole_request_with_no_prefill() {
    let compute = Arc::new(MockCompute::new());
    // One perfectly good question, and one that is not.
    let body = r#"{
      "state": "s",
      "questions": {
        "good": { "type": "noul", "instructions": "Urgent?" },
        "bad": { "type": "choice", "instructions": "Which?" }
      }
    }"#;
    let (status, response) = decide(&app(compute.clone()), body).await;

    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "missing_criteria");
    assert!(
        response["error"]["message"]
            .as_str()
            .expect("a message")
            .contains("bad"),
        "the refusal names the offending question: {response}"
    );
    assert!(
        response.get("answers").is_none(),
        "and answers nothing at all: {response}"
    );
    assert!(
        compute.prefill_calls().is_empty(),
        "the good question's prefill was never spent"
    );
}

#[tokio::test]
async fn a_body_that_will_not_parse_is_refused_the_same_way() {
    let compute = Arc::new(MockCompute::new());
    let (status, response) = decide(&app(compute.clone()), r#"{"state":"s","questions":7}"#).await;
    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["type"], "invalid_request_error");
    assert!(compute.prefill_calls().is_empty());
}

#[tokio::test]
async fn a_load_that_cannot_name_answers_refuses_rather_than_guesses() {
    // `SimpleTemplateProvider` has no tokenizer to build an alphabet from,
    // which is what a provider without one should say.
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    let app = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider)).app();
    let (status, response) = decide(&app, JEV_NOUL).await;
    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "alphabet_exhausted");
    assert!(compute.prefill_calls().is_empty());
}

// ── a decision takes the Agent lane (spec 02's class, reached from here) ─

#[tokio::test]
async fn a_decision_is_admitted_as_an_agent() {
    let compute = Arc::new(MockCompute::new());
    let (status, _) = decide(&app(compute.clone()), JEV_NOUL).await;
    assert_eq!(status, 200);
    // The scheduler saw exactly one prefill, and it asked for a readout:
    // this went down the decision path rather than the chat one.
    let jobs: Vec<_> = compute.prefill_calls().into_iter().flatten().collect();
    assert!(!jobs.is_empty(), "the decision was prefilled");
    assert!(
        jobs.iter().any(|job| job.readout.is_some()),
        "and one of its chunks read the answer tokens out"
    );
}

// ── the evidence may be an image (acceptance 8) ──────────────────────────

#[path = "support/mod.rs"]
mod support;

#[tokio::test]
async fn an_image_state_is_rendered_into_the_prompt() {
    use ignis_artifact::vision::ProcessorOptions;
    use ignis_server::media::{MediaAcquirer, MediaPolicy};
    use support::media::{data_uri, png, processor};

    let limits = ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 1 << 20,
        max_decoded_pixels: 1 << 20,
        max_raw_patches: 1 << 16,
        max_vision_tokens: 1 << 14,
    };
    let compute = Arc::new(MockCompute::new());
    let acquirer = MediaAcquirer::new(
        Arc::new(processor(limits.clone())),
        limits,
        MediaPolicy::new(false, 1 << 20),
    );
    let with_image = server(compute.clone())
        .with_request_timeout(std::time::Duration::from_secs(10))
        .with_media(Arc::new(acquirer))
        .app();

    // The `state` is content parts carrying an image: the one thing Jev
    // cannot do, since its own `state` is `string | object | array`.
    let body = format!(
        r#"{{"state":[{{"type":"image_url","image_url":{{"url":"{}"}}}}],
            "questions":{{"which":{{"type":"choice","instructions":"Which number?",
            "criteria":{{"42":null,"47":null}}}}}}}}"#,
        data_uri(&png(64, 64))
    );
    let (status, response) = decide(&with_image, &body).await;

    assert_eq!(status, 200, "{response}");
    assert_eq!(response["answers"]["which"]["type"], "choice");

    // The image really reached the prompt: the request carries a multimodal
    // part, and its prompt is far longer than the words around it because
    // the placeholders expanded into the image's own token run.
    let jobs: Vec<_> = compute.prefill_calls().into_iter().flatten().collect();
    assert!(!jobs.is_empty(), "the decision was prefilled");
    let multimodal = jobs
        .iter()
        .find_map(|job| job.multimodal.clone())
        .expect("the prefill carries the image's placeholders");
    assert_eq!(multimodal.media.len(), 1, "one image, acquired once");
    let vision_tokens = multimodal.media[0].grid.vision_tokens() as usize;
    assert!(vision_tokens > 0, "the image has a token run of its own");

    // The run is really *in* the prompt, not merely attached to it: the
    // prompt carries exactly that many `<|image_pad|>` ids.
    //
    // Counted in the prompt rather than measured against the same decision
    // without an image, which is what this used to do: since GitHub #240 a
    // text `state` rides in the system message and an image `state` cannot
    // (`check_content_parts` refuses media there), so the two prompts differ
    // by the evidence as well as by the image and their lengths no longer
    // subtract.
    let pads = jobs
        .iter()
        .flat_map(|job| job.tokens.iter())
        .filter(|&&token| token == ignis_artifact::vision::IMAGE_PAD_ID)
        .count();
    assert_eq!(
        pads, vision_tokens,
        "the image's {vision_tokens} placeholders are in the prompt"
    );
    assert_eq!(response["usage"]["output_tokens"], 0);
}

// ── number, point and box: the primitives that generate (GitHub #242) ───
//
// The `Compute` seam's mock honours a permitted set and reports a
// probability inside it (`MockCompute`), so everything above the seam —
// the schedule, the forced prefix, the place-weighted uncertainty and the
// rescaling onto the submitted image — is testable here without a GPU (ADR
// 0006). What is *not* testable here is whether the model points at the
// right button, which is `classify_pointing_gpu.rs`.

/// The media stack the spatial tests need, at a size that leaves the
/// placeholder template room to work.
fn vision_limits() -> ignis_artifact::vision::ProcessorOptions {
    ignis_artifact::vision::ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 1 << 20,
        max_decoded_pixels: 1 << 20,
        max_raw_patches: 1 << 16,
        max_vision_tokens: 1 << 14,
    }
}

fn seeing_server(compute: Arc<MockCompute>) -> Server {
    use ignis_server::media::{MediaAcquirer, MediaPolicy};
    use support::media::processor;

    let limits = vision_limits();
    server(compute)
        .with_request_timeout(std::time::Duration::from_secs(10))
        .with_media(Arc::new(MediaAcquirer::new(
            Arc::new(processor(limits.clone())),
            limits,
            MediaPolicy::new(false, 1 << 20),
        )))
}

/// The place-weighted sum a trace implies, recomputed from the response's
/// own digits — the acceptance, restated as arithmetic a reader can check.
fn sigma_of(digits: &JsonValue) -> f64 {
    let digits = digits.as_array().expect("a digit trace is an array");
    let width = digits.len();
    digits
        .iter()
        .enumerate()
        .map(|(place, digit)| {
            let probability = digit["probability"].as_f64().expect("a probability");
            (1.0 - probability) * 10f64.powi((width - 1 - place) as i32)
        })
        .sum()
}

/// Acceptance 4: `uncertainty` is the trace's place-weighted sum, in units
/// of the value itself.
#[tokio::test]
async fn a_numbers_uncertainty_is_its_own_trace_summed_by_place() {
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"The queue is backing up.",
        "questions":{"depth":{"type":"number","instructions":"How many items are waiting?"}}}"#;
    let (status, response) = decide(&app(compute), body).await;

    assert_eq!(status, 200, "{response}");
    let answer = &response["answers"]["depth"];
    assert_eq!(answer["type"], "number", "{response}");
    let digits = &answer["digits"];
    assert_eq!(digits.as_array().expect("a trace").len(), 3, "the default width");

    // The number really is its digits, read left to right.
    let spelled: u64 = digits
        .as_array()
        .unwrap()
        .iter()
        .fold(0, |value, digit| value * 10 + digit["digit"].as_u64().expect("a digit"));
    assert_eq!(answer["number"].as_u64().expect("a number"), spelled);

    let reported = answer["uncertainty"].as_f64().expect("an uncertainty");
    let expected = sigma_of(digits);
    assert!(
        (reported - expected).abs() < 1e-6,
        "uncertainty {reported} must be the trace's place-weighted sum {expected}: an \
         unsure hundreds digit is worth a hundred times an unsure units one"
    );
    assert!(
        reported > 0.0,
        "and it is a real number: the mock draws at a real confidence, so a handler \
         that never read the trace could not pass by reporting zero"
    );
    // A constrained decode generates, and says so.
    assert_eq!(response["usage"]["output_tokens"], 3, "{response}");
}

/// Acceptance 5: a non-square image returns pixels consistent with **its
/// own** dimensions on both axes.
#[tokio::test]
async fn a_point_on_a_non_square_image_is_in_that_images_pixels() {
    use support::media::{data_uri, png};

    // 16:9, and inside the fixture processor's decoded-pixel budget.
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 360;

    let compute = Arc::new(MockCompute::new());
    let body = format!(
        r#"{{"state":[{{"type":"image_url","image_url":{{"url":"{}"}}}}],
            "questions":{{"where":{{"type":"point","instructions":"click the blue button"}}}}}}"#,
        data_uri(&png(WIDTH, HEIGHT))
    );
    let (status, response) = decide(&seeing_server(compute.clone()).app(), &body).await;

    assert_eq!(status, 200, "{response}");
    let answer = &response["answers"]["where"];
    assert_eq!(answer["type"], "point", "{response}");

    // The two axes are read on the same 0-999 scale and land on different
    // pixel scales, which is the whole reason the server does this rather
    // than the caller.
    for (axis, side) in [("x", WIDTH), ("y", HEIGHT)] {
        let normalized = answer["normalized"][axis].as_u64().expect("a normalized reading");
        let pixels = answer["pixels"][axis].as_i64().expect("a pixel reading");
        assert!(normalized <= 999, "{axis} is on the declared 0-999 scale: {normalized}");
        let expected = (normalized as f64 / 999.0 * f64::from(side)).round() as i64;
        assert_eq!(
            pixels, expected,
            "{axis}={normalized} of 999 on a {side}px axis is {expected} px, not {pixels}"
        );
        assert!(pixels >= 0 && pixels <= i64::from(side), "and inside the image");

        // The uncertainty travels with it, in the same units.
        let sigma = answer["uncertainty"][axis].as_f64().expect("an uncertainty");
        let native = sigma_of(&answer["digits"][axis]);
        assert!(
            (sigma - native * f64::from(side) / 999.0).abs() < 1e-6,
            "{axis}'s uncertainty is its trace's sum rescaled onto {side} px: {sigma}"
        );
    }

    // The forced prefix is in the prompt, and the prompt's MRoPE positions
    // grew with it. The leaf refuses a chunk whose positions and tokens
    // disagree, so a prefix appended without them is a 400 on the card and
    // nothing at all here — which is why this is asserted rather than left
    // to the GPU test to discover.
    let jobs: Vec<_> = compute.prefill_calls().into_iter().flatten().collect();
    let prompt_tokens: usize = jobs.iter().map(|job| job.tokens.len()).sum();
    let multimodal = jobs
        .iter()
        .find_map(|job| job.multimodal.clone())
        .expect("a point over an image carries one");
    assert_eq!(
        multimodal.prompt_tokens(),
        prompt_tokens,
        "three positions per token, forced prefix included"
    );
    assert!(
        jobs.iter().any(|job| job.permitted.is_some()),
        "and the prefill drew the run's first digit, which is where a constrained run starts"
    );

    // The two axes do not share a divisor. Asserted on a reading of this
    // run's own, unconditionally: the earlier version only checked this
    // inside `if x == y`, which the mock never satisfies — a branch that
    // never runs, carrying an assertion that was backwards anyway (equal
    // readings on a non-square image must give *different* pixels).
    let reading = answer["normalized"]["x"].as_u64().expect("a reading").max(1);
    let across = (reading as f64 / 999.0 * f64::from(WIDTH)).round() as i64;
    let down = (reading as f64 / 999.0 * f64::from(HEIGHT)).round() as i64;
    assert_ne!(
        across, down,
        "{reading} of 999 is {across} px across and {down} px down on a {WIDTH}x{HEIGHT} image; a caller handed one divisor would be right along x and wrong along y"
    );

    assert_eq!(
        response["usage"]["output_tokens"], 11,
        "three digits, the forced separator, three digits — and no prefix, which \
         rides in the prompt"
    );
}

/// The forced prefix is in the **prompt**, not the schedule: it costs
/// prefill, not a decode round each.
#[tokio::test]
async fn the_opening_literal_is_prefilled_and_the_separator_is_forced() {
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"The queue is backing up.",
        "questions":{"depth":{"type":"number","instructions":"How many items are waiting?",
        "digits":2}}}"#;
    let (status, response) = decide(&app(compute.clone()), body).await;
    assert_eq!(status, 200, "{response}");

    let jobs: Vec<_> = compute.prefill_calls().into_iter().flatten().collect();
    let prompt: Vec<u32> = jobs.iter().flat_map(|job| job.tokens.iter().copied()).collect();
    // `DecidingTemplate::encode_literal` is one token per character.
    let prefix: Vec<u32> = "{\"value\":".chars().map(|c| c as u32).collect();
    assert!(
        prompt.windows(prefix.len()).any(|window| window == prefix),
        "the opening literal is prefilled with the prompt"
    );
    assert_eq!(
        compute.decode_calls().len(),
        2,
        "and only the digits cost rounds: two digits, two rounds, the last of \
         which carries no set"
    );
    assert_eq!(response["answers"]["depth"]["digits"].as_array().unwrap().len(), 2);
    assert_eq!(response["usage"]["output_tokens"], 2);
}

/// A `box` is four numbers over one prefill, not four questions.
#[tokio::test]
async fn a_box_reads_four_axes_from_one_run() {
    use support::media::{data_uri, png};

    let compute = Arc::new(MockCompute::new());
    let body = format!(
        r#"{{"state":[{{"type":"image_url","image_url":{{"url":"{}"}}}}],
            "questions":{{"frame":{{"type":"box","instructions":"the blue button","digits":2}}}}}}"#,
        data_uri(&png(256, 128))
    );
    let (status, response) = decide(&seeing_server(compute.clone()).app(), &body).await;

    assert_eq!(status, 200, "{response}");
    let answer = &response["answers"]["frame"];
    assert_eq!(answer["type"], "box", "{response}");
    for axis in ["x0", "y0", "x1", "y1"] {
        assert!(answer["pixels"][axis].is_i64(), "{axis} has a pixel reading: {answer}");
        assert_eq!(answer["digits"][axis].as_array().expect("a trace").len(), 2);
    }
    // x scales by the width and y by the height, on the same reading.
    for (axis, side) in [("x0", 256u32), ("y0", 128), ("x1", 256), ("y1", 128)] {
        let normalized = answer["normalized"][axis].as_u64().unwrap();
        let expected = (normalized as f64 / 99.0 * f64::from(side)).round() as i64;
        assert_eq!(answer["pixels"][axis].as_i64().unwrap(), expected, "{axis}");
    }
    let requests: Vec<_> =
        compute.prefill_calls().iter().flatten().map(|job| job.request).collect();
    assert!(
        requests.windows(2).all(|pair| pair[0] == pair[1]),
        "one request, not four: the y digits are read after x has been forced, so \
         the model knows where it put x - {requests:?}"
    );
}

/// A `point` needs an image to be a position on, and says so rather than
/// answering in pixels of nothing.
#[tokio::test]
async fn a_point_over_a_text_state_is_an_error_not_a_guess() {
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"a paragraph of text",
        "questions":{"where":{"type":"point","instructions":"the button"}}}"#;
    let (status, response) = decide(&app(compute), body).await;

    assert_eq!(status, 200, "a per-question failure, not a refused request: {response}");
    assert_eq!(response["answers"]["where"]["type"], "error", "{response}");
    assert_eq!(response["answers"]["where"]["code"], "state_carries_no_image");
}

// ── the model the caller names is the model that answers ────────────────

#[tokio::test]
async fn a_model_this_engine_does_not_load_is_refused_not_echoed() {
    // Jev documents the response's `model` as "the model that **performed**
    // the evaluation". Echoing the caller's string back would make a local
    // Qwen claim to be `jev-latest`, and would quietly serve a request
    // addressed somewhere else — which `/v1/chat/completions` refuses.
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"s","model":"jev-latest","questions":{"q":{"type":"noul","instructions":"Urgent?"}}}"#;
    let (status, response) = decide(&app(compute.clone()), body).await;

    assert_eq!(status, 422, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .expect("a message")
            .contains("jev-latest"),
        "the refusal names the model that was asked for: {response}"
    );
    assert!(compute.prefill_calls().is_empty());
}

#[tokio::test]
async fn the_answer_reports_the_model_that_performed_it() {
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"s","questions":{"q":{"type":"noul","instructions":"Urgent?"}}}"#;
    let (status, response) = decide(&app(compute), body).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(response["model"], MODEL, "the loaded model, named by the engine");
}

#[tokio::test]
async fn a_lane_tag_on_the_model_reaches_the_decision() {
    // `model@lane` is ignis's second entry point for the **Lane tag**
    // (GitHub #120). A decision defaults to `Agent`, and a caller who says
    // otherwise is believed — so the suffix must be read, not echoed.
    let compute = Arc::new(MockCompute::new());
    let body = format!(
        r#"{{"state":"s","model":"{MODEL}@interactive","questions":{{"q":{{"type":"noul","instructions":"Urgent?"}}}}}}"#
    );
    let (status, response) = decide(&app(compute), &body).await;
    assert_eq!(status, 200, "the tag is read off the model, not sent to it: {response}");
    assert_eq!(
        response["model"], MODEL,
        "and the answer names the model, without the suffix"
    );
}

// ── everything a caller can get wrong is refused before the first submit ─

#[tokio::test]
async fn an_image_state_on_a_text_only_load_is_refused_not_answered_with_an_error() {
    // The request was never servable, so it is the caller's to fix — a 422,
    // not a 200 carrying an error in an answer slot.
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":[{"type":"image_url","image_url":{"url":"https://example.test/a.png"}}],
                   "questions":{"q":{"type":"choice","instructions":"Which?","criteria":{"a":null,"b":null}}}}"#;
    let (status, response) = decide(&app(compute.clone()), body).await;

    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "media_unsupported");
    assert!(response.get("answers").is_none(), "nothing was answered");
    assert!(compute.prefill_calls().is_empty());
}

#[tokio::test]
async fn one_unservable_question_refuses_the_whole_request_before_any_of_them_runs() {
    // The acceptance's real shape: the *second* question is the bad one, so
    // a handler that validated-then-asked question by question would have
    // spent the first question's prefill before finding out.
    let compute = Arc::new(MockCompute::new());
    let body = r#"{
      "state": "s",
      "questions": {
        "aaa_first": { "type": "noul", "instructions": "Urgent?" },
        "zzz_second": { "type": "choice", "instructions": "Which?", "criteria": { "": null, "x": null } }
      }
    }"#;
    let (status, response) = decide(&app(compute.clone()), body).await;

    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "unclean_option");
    assert!(
        compute.prefill_calls().is_empty(),
        "the first question's prefill was never spent"
    );
}

#[tokio::test]
async fn a_duplicate_option_is_refused_rather_than_silently_halved() {
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"s","questions":{"q":{"type":"choice","instructions":"Which?","criteria":{"a":"one","a":"two","b":null}}}}"#;
    let (status, response) = decide(&app(compute.clone()), body).await;
    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "duplicate_option");
    assert!(compute.prefill_calls().is_empty());
}

#[tokio::test]
async fn criteria_of_the_wrong_shape_refuses_in_its_own_vocabulary() {
    // Not serde's "data did not match any variant", which names neither the
    // question nor the field.
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"s","questions":{"mood":{"type":"score","instructions":"How?","criteria":"high"}}}"#;
    let (status, response) = decide(&app(compute), body).await;
    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "malformed_criteria");
    let message = response["error"]["message"].as_str().expect("a message");
    assert!(
        message.contains("mood") && message.contains("criteria"),
        "the refusal names the question and the field: {message}"
    );
}

// ── the fan-out: one state, N questions (GitHub #240, spec 04) ───────────

/// A body of `questions` questions over one `state` of `state_words`
/// distinct words.
///
/// The words are distinct so the evidence cannot be confused with anything
/// else in the prompt under [`SimpleTemplateProvider`]'s one-token-per-word
/// rendering, and the instructions differ per question so the questions are
/// not accidentally one prompt repeated.
fn fan_out(state_words: usize, questions: usize) -> String {
    let state = (0..state_words)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let asked = (0..questions)
        .map(|i| format!(r#""q{i}":{{"type":"noul","instructions":"Is q{i} urgent?"}}"#))
        .collect::<Vec<_>>()
        .join(",");
    format!(r#"{{"state":"{state}","model":"{MODEL}","questions":{{{asked}}}}}"#)
}

/// Every request id the compute was asked to prefill, and how many tokens
/// of it — summed over the chunks a prompt was cut into.
fn prefilled(compute: &MockCompute) -> std::collections::BTreeMap<u64, u32> {
    let mut totals = std::collections::BTreeMap::new();
    for job in compute.prefill_calls().into_iter().flatten() {
        *totals.entry(job.request).or_insert(0u32) += job.tokens.len() as u32;
    }
    totals
}

#[tokio::test]
async fn twenty_questions_over_one_state_prefill_it_once() {
    // Spec 04's acceptance 1, on the prefill token count rather than on wall
    // time. The mechanism is a **retained prefix** over the system block
    // (`decide::messages_for`): the first question pays for the state, the
    // other nineteen claim it and prefill only their own tail.
    const STATE: usize = 300;
    const QUESTIONS: usize = 20;
    let compute = Arc::new(MockCompute::new());
    let (status, response) = decide(&app(compute.clone()), &fan_out(STATE, QUESTIONS)).await;

    assert_eq!(status, 200, "{response}");
    assert_eq!(response["answers"].as_object().expect("a map").len(), QUESTIONS);

    let totals = prefilled(&compute);
    assert_eq!(totals.len(), QUESTIONS, "one request per question: {totals:?}");
    let first = *totals.values().next().expect("a first question");
    assert!(
        first > STATE as u32,
        "the first question prefills the whole state: {first} tokens"
    );
    for (request, tokens) in totals.iter().skip(1) {
        assert!(
            *tokens < STATE as u32,
            "question {request} prefilled {tokens} tokens — it cannot have \
             claimed the state and still run that many"
        );
    }
    let total: u32 = totals.values().sum();
    assert!(
        total < first + QUESTIONS as u32 * 32,
        "the state is paid for once, not twenty times: {total} tokens over \
         {QUESTIONS} questions, the first of them {first}"
    );
}

#[tokio::test]
async fn the_followers_of_a_fan_out_are_in_the_engine_together() {
    // The other half of the sequencing rule. The first question runs alone
    // so there is a prefix to claim; after that there is nothing left to
    // serialize, and answering the rest one at a time would cost N
    // `--request-timeout`s for no gain.
    //
    // Served in sequence only ever one request exists at a time, so a
    // prefill batch carrying two of them is proof the followers overlap.
    //
    // The batch is made wide on purpose rather than hoped for. `MockCompute`
    // answers instantly, so on a loaded runner the scheduler can drain the
    // followers one at a time as the handler submits them and every batch is
    // one request wide -- which is what this assertion read on a GitHub
    // Windows runner (`the widest prefill batch held 1 requests`) on a commit
    // that passed twice on the development machine. It proved the runner was
    // slow, not that the server serialized anything.
    //
    // So the compute holds the first batch that carries a follower until the
    // harness says go. While it is held the handler keeps being polled and
    // submits the rest, and the batch the scheduler forms when it comes back
    // out is all of them. A server that really did serialize its followers
    // would have nothing queued behind the held one and still fail.
    use std::sync::mpsc::sync_channel;
    let (entered_tx, entered_rx) = sync_channel(0);
    let (go_tx, go_rx) = sync_channel(0);
    let (released_tx, _released_rx) = sync_channel(64);
    let compute = Arc::new(HoldFollower {
        inner: MockCompute::new(),
        entered: entered_tx,
        go: std::sync::Mutex::new(go_rx),
        held: std::sync::atomic::AtomicBool::new(false),
        released: released_tx,
    });
    let app = server_over(compute.clone() as Arc<dyn ignis_core::Compute>).app();
    let body = fan_out(300, 8);
    let client = tokio::spawn(async move { decide(&app, &body).await });

    // Blocking, but off the runtime's thread -- the handler has to keep being
    // polled to submit the followers this waits for.
    tokio::task::spawn_blocking(move || entered_rx.recv())
        .await
        .expect("the waiter ran")
        .expect("a follower reached the compute");
    support::nudge().await;
    go_tx.send(()).expect("the held prefill is still waiting");

    let (status, _) = client.await.expect("the request ran");
    assert_eq!(status, 200);

    let batched = compute
        .inner
        .prefill_calls()
        .into_iter()
        .map(|call| {
            call.iter()
                .map(|job| job.request)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        })
        .max()
        .unwrap_or(0);
    assert!(
        batched > 1,
        "the followers reached the scheduler together — the widest prefill \
         batch held {batched} requests"
    );
}

/// Answers `victim`'s prefill with no readout, the way a leaf that ran the
/// forward pass but produced nothing usable would.
///
/// A per-question runtime failure has to be *per question*: the mock's own
/// `fail_prefill` fails the whole batch the way a leaf error does, and the
/// followers share a batch, so it would take all of them down together and
/// prove nothing about isolation. Dropping one job's readout leaves its
/// siblings' outcomes untouched, and a decision with no readout is exactly
/// what the scheduler finishes with `FinishReason::Error` (GitHub #238).
struct DropReadout {
    inner: MockCompute,
    victim: ignis_core::RequestId,
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

    fn release(&self, request: ignis_core::RequestId) {
        self.inner.release(request);
    }
}

#[tokio::test]
async fn one_question_failing_at_runtime_still_answers_the_others() {
    // Spec 04's acceptance 4. Everything a *caller* can get wrong is refused
    // before the first submit (`serve`); this is the other kind of failure —
    // the engine dropping one question after the others have been paid for.
    const QUESTIONS: usize = 8;
    // A follower, not the sequenced first question: the first one failing
    // would be indistinguishable from a fan-out that never started.
    let compute = Arc::new(DropReadout {
        inner: MockCompute::new(),
        victim: 3,
    });
    let app = server_over(compute.clone() as Arc<dyn ignis_core::Compute>).app();
    let (status, response) = decide(&app, &fan_out(300, QUESTIONS)).await;

    assert_eq!(status, 200, "one question's runtime failure is not the request's");
    let answers = response["answers"].as_object().expect("a map");
    assert_eq!(answers.len(), QUESTIONS, "every question keeps its slot");
    let failed: Vec<&String> = answers
        .iter()
        .filter(|(_, answer)| answer["type"] == "error")
        .map(|(id, _)| id)
        .collect();
    assert_eq!(failed.len(), 1, "exactly one slot carries the failure: {response}");
    for (id, answer) in answers {
        if failed.contains(&id) {
            assert!(answer["message"].is_string(), "the error says what happened: {answer}");
            continue;
        }
        assert_eq!(answer["type"], "noul", "question {id} keeps its paid-for answer");
        assert!(answer["noul"].is_number(), "{answer}");
    }
}

/// Holds the first `prefill_step` that carries a **follower**, and reports
/// every request the engine releases.
///
/// Gating `prefill_step` rather than `decode_step` (`GatedCompute`) is the
/// whole point: a decision never decodes, so the existing gate can never
/// hold one. Holding on a job whose request id is not the first one is what
/// makes "mid-fan-out" a fact rather than a hope — the followers are in the
/// engine at the instant the client goes away.
struct HoldFollower {
    inner: MockCompute,
    entered: std::sync::mpsc::SyncSender<()>,
    go: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
    held: std::sync::atomic::AtomicBool,
    released: std::sync::mpsc::SyncSender<ignis_core::RequestId>,
}

impl ignis_core::Compute for HoldFollower {
    fn prefill_step(
        &self,
        jobs: &[ignis_core::PrefillJob],
    ) -> Result<Vec<ignis_core::PrefillOutcome>, ignis_core::ComputeError> {
        // Request ids start at 0, so the sequenced first question is 0 and
        // every follower is 1 or above.
        let follower = jobs.iter().any(|job| job.request >= 1);
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

    fn release(&self, request: ignis_core::RequestId) {
        self.inner.release(request);
        let _ = self.released.send(request);
    }
}

#[tokio::test]
async fn a_client_disconnecting_mid_fan_out_leaves_no_request_running() {
    // Spec 04's acceptance 3. The fan-out is a unit of cancellation even
    // though it is not a unit of scheduling: the engine keeps working on a
    // request whose caller has gone until it is told otherwise, and twenty
    // of them is exactly the shape in which nineteen are forgotten.
    //
    // Nothing here sleeps (ADR 0006): the compute signals when a follower is
    // inside it, and reports every release on its way back out.
    use std::sync::mpsc::sync_channel;
    const QUESTIONS: usize = 8;

    let (entered_tx, entered_rx) = sync_channel(0);
    let (go_tx, go_rx) = sync_channel(0);
    let (released_tx, released_rx) = sync_channel(64);
    let compute = Arc::new(HoldFollower {
        inner: MockCompute::new(),
        entered: entered_tx,
        go: std::sync::Mutex::new(go_rx),
        held: std::sync::atomic::AtomicBool::new(false),
        released: released_tx,
    });
    let (app, _, cancelled) = recording_over(compute.clone() as Arc<dyn ignis_core::Compute>);

    let request = Request::builder()
        .method("POST")
        .uri("/v1/decide")
        .header("content-type", "application/json")
        .body(Body::from(fan_out(300, QUESTIONS).into_bytes()))
        .unwrap();
    let client = tokio::spawn(async move { app.oneshot(request).await });

    // Blocking, but off the runtime's thread — the handler has to keep being
    // polled to submit the followers this waits for.
    tokio::task::spawn_blocking(move || entered_rx.recv())
        .await
        .expect("the waiter ran")
        .expect("a follower reached the compute");

    // The client goes away. Dropping the handler's future drops every
    // in-flight `ask`, and each of those drops its `CancelOnDrop`.
    client.abort();
    support::nudge().await;
    go_tx.send(()).expect("the held prefill is still waiting");

    // Every one of the N requests releases — the ones that finished and the
    // ones that were cancelled alike — so the collector stops when it has
    // them all. The timeout is the failure path, not the exit: waiting it
    // out on every green run would be the sleep ADR 0006 forbids.
    let releases = tokio::task::spawn_blocking(move || {
        let mut seen = std::collections::BTreeSet::new();
        while seen.len() < QUESTIONS {
            match released_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(request) => {
                    seen.insert(request);
                }
                Err(_) => break,
            }
        }
        seen
    })
    .await
    .expect("the collector ran");

    let prefilled: std::collections::BTreeSet<u64> = compute
        .inner
        .prefill_calls()
        .into_iter()
        .flatten()
        .map(|job| job.request)
        .collect();
    assert!(
        prefilled.len() > 1,
        "a follower was in the engine when the client went away: {prefilled:?}"
    );
    assert!(
        prefilled.is_subset(&releases),
        "every request the engine started released its resources, none left \
         running: started {prefilled:?}, released {releases:?}"
    );

    // The assertion with teeth. Everything above is also true of a fan-out
    // that simply *finished* — a decision terminates in the tick its prefill
    // does, so the followers held here were a few microseconds from
    // answering nobody, and a `CancelOnDrop` deleted from `ask` leaves every
    // line above green (verified by mutation). What only a cancelled
    // request has is a cancel issued for it.
    //
    // Waited for rather than read: a release and its request's cancel are
    // two different journeys through the engine and the release wins the
    // race about half the time.
    //
    // One cancel, not one per follower. The compute is holding the whole
    // fan-out's prefill batch when `client.abort()` runs, so every cancel is
    // issued while the scheduler is blocked *inside* it; when it comes back
    // out it finishes that batch, and a decision terminates in the tick its
    // prefill does. A cancel that lands on a request the engine has already
    // terminated reaches nothing and is recorded nowhere, and which requests
    // lose that race is scheduling. Asserting one per follower failed about
    // one Windows run in ten with `followers {1, ..., 7}, cancelled {1, 2,
    // 3}` — a flaky assertion over a property the engine never promised.
    //
    // The teeth survive: a fan-out that merely finished issues *no* cancel
    // at all, which is exactly what deleting the `CancelOnDrop` from `ask`
    // produces. So: at least one, and every one of them a request this
    // fan-out started.
    let cancelled = tokio::task::spawn_blocking(move || {
        let mut seen = std::collections::BTreeSet::new();
        // The first one is waited for; the rest are whatever already
        // arrived, since the count is not the property under test.
        if let Ok(request) = cancelled.recv_timeout(std::time::Duration::from_secs(5)) {
            seen.insert(request);
            while let Ok(request) = cancelled.try_recv() {
                seen.insert(request);
            }
        }
        seen
    })
    .await
    .expect("the cancel collector ran");
    assert!(
        !cancelled.is_empty() && cancelled.is_subset(&prefilled),
        "the handler's future dropping cancels the requests it left behind: \
         started {prefilled:?}, cancelled {cancelled:?}"
    );
}

/// A [`Scheduler`] that records what the server submitted, and otherwise is
/// the real one.
///
/// The class a request carries reaches no response field and no metric, so
/// this is the seam it *can* be read at — the same one the engine drives.
struct RecordingScheduler {
    inner: ConcreteScheduler,
    submitted: Arc<std::sync::Mutex<Vec<(ignis_core::types::RequestInput, ignis_core::types::RequestClass)>>>,
    cancelled: std::sync::mpsc::SyncSender<ignis_core::RequestId>,
}

impl ignis_core::Scheduler for RecordingScheduler {
    fn submit(
        &mut self,
        input: ignis_core::types::RequestInput,
        class: ignis_core::types::RequestClass,
    ) -> Result<ignis_core::RequestId, ignis_core::SubmitError> {
        self.submitted.lock().unwrap().push((input.clone(), class));
        self.inner.submit(input, class)
    }

    fn cancel(&mut self, request: ignis_core::RequestId) -> bool {
        // Reported before delegating, and the return value is not the
        // property under test: the engine calls this for a request that has
        // already finished too, and `cancel` answers `false`. What a
        // disconnected client owes its siblings is that the cancel was
        // *issued*.
        //
        // A channel rather than a list because a test has to be able to
        // *wait* for this. The cancels are queued on the engine's command
        // channel and processed by the model thread after the tick already
        // in progress, so a test that reads a list right after the client
        // goes away reads it before the engine has seen a thing.
        let _ = self.cancelled.try_send(request);
        self.inner.cancel(request)
    }

    fn advance(&mut self) -> Vec<ignis_core::SchedEvent> {
        self.inner.advance()
    }

    fn is_idle(&self) -> bool {
        self.inner.is_idle()
    }

    fn model_id(&self) -> &str {
        ignis_core::Scheduler::model_id(&self.inner)
    }

    fn max_sequence_tokens(&self) -> u32 {
        ignis_core::Scheduler::max_sequence_tokens(&self.inner)
    }

    fn mode(&self) -> ignis_core::types::EngineMode {
        ignis_core::Scheduler::mode(&self.inner)
    }

    fn occupancy(&self) -> ignis_core::scheduler::Occupancy {
        ignis_core::Scheduler::occupancy(&self.inner)
    }
}

type Submitted = Arc<
    std::sync::Mutex<Vec<(ignis_core::types::RequestInput, ignis_core::types::RequestClass)>>,
>;
type Cancelled = std::sync::mpsc::Receiver<ignis_core::RequestId>;

fn recording() -> (axum::Router, Submitted) {
    let (app, submitted, _) = recording_over(Arc::new(MockCompute::new()));
    (app, submitted)
}

fn recording_over(compute: Arc<dyn ignis_core::Compute>) -> (axum::Router, Submitted, Cancelled) {
    let submitted: Submitted = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (cancelled, cancels) = std::sync::mpsc::sync_channel(64);
    let scheduler = RecordingScheduler {
        inner: scheduler_over(compute),
        submitted: submitted.clone(),
        cancelled,
    };
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(DecidingTemplate));
    (server.app(), submitted, cancels)
}

#[tokio::test]
async fn every_question_in_a_fan_out_carries_the_parents_class() {
    // Spec 04's acceptance 5. The class is read once, off the parent's Lane
    // tag, and handed to every internal request — a fan-out is twenty
    // requests but one caller, and nineteen of them silently defaulting to
    // Agent would be a lane the caller never asked for.
    use ignis_core::types::RequestClass;

    let (app, submitted) = recording();
    let (status, _) = decide(&app, &fan_out(300, 5)).await;
    assert_eq!(status, 200);
    let classes: Vec<RequestClass> = submitted.lock().unwrap().iter().map(|(_, c)| *c).collect();
    assert_eq!(classes.len(), 5);
    assert!(
        classes.iter().all(|class| *class == RequestClass::Agent),
        "a decision defaults to Agent, every one of them: {classes:?}"
    );

    let (app, submitted) = recording();
    let body = fan_out(300, 5).replace(
        &format!(r#""model":"{MODEL}""#),
        &format!(r#""model":"{MODEL}@interactive""#),
    );
    let (status, _) = decide(&app, &body).await;
    assert_eq!(status, 200);
    let classes: Vec<RequestClass> = submitted.lock().unwrap().iter().map(|(_, c)| *c).collect();
    assert_eq!(classes.len(), 5);
    assert!(
        classes.iter().all(|class| *class == RequestClass::Interactive),
        "and the caller's own Lane tag reaches all of them: {classes:?}"
    );
}

#[tokio::test]
async fn the_evidence_reaches_the_scheduler_inside_the_system_block() {
    // The mechanism, asserted where it is decided rather than inferred from
    // a token count. A retained prefix is cut at the page floor of
    // `system_block_tokens` (`Request::retained_prefix_point`) and nothing
    // outside the block is ever part of one — so evidence that is merely
    // *first* is re-prefilled per question however shared it looks.
    const STATE: usize = 300;
    let (app, submitted) = recording();
    let (status, _) = decide(&app, &fan_out(STATE, 3)).await;
    assert_eq!(status, 200);

    for (input, _) in submitted.lock().unwrap().iter() {
        let block = input
            .system_block_tokens
            .expect("a decision's prompt opens with a system block");
        assert!(
            block > STATE as u32,
            "the block holds the instruction and the whole {STATE}-token \
             state, not just the instruction: {block}"
        );
        assert!(
            block < input.tokens.len() as u32,
            "and stops before the question, which is this request's alone"
        );
    }
}

#[tokio::test]
async fn a_wave_wider_than_the_engine_is_retried_rather_than_failed() {
    // A wave is sized for an engine nobody else is using. When somebody is,
    // the engine answers `SubmitError::Full` — "not now" — and turning that
    // into an error in a question's slot would be a failure the one question
    // at a time loop before GitHub #240 never produced.
    //
    // An engine that admits two at a time stands in for a busy one: the
    // wave of seven followers cannot fit, and every question must still be
    // answered.
    const QUESTIONS: usize = 8;
    let compute = Arc::new(MockCompute::new());
    let scheduler = scheduler_admitting(compute.clone() as Arc<dyn ignis_core::Compute>, 2);
    let app = Server::new(Engine::new(Box::new(scheduler)), Box::new(DecidingTemplate)).app();

    let (status, response) = decide(&app, &fan_out(300, QUESTIONS)).await;
    assert_eq!(status, 200, "{response}");
    let answers = response["answers"].as_object().expect("a map");
    assert_eq!(answers.len(), QUESTIONS);
    for (id, answer) in answers {
        assert_eq!(
            answer["type"], "noul",
            "question {id} was answered rather than turned away: {answer}"
        );
    }
    // And the retries did not cost the sharing: the state is still prefilled
    // once, however many waves it took.
    let totals = prefilled(&compute);
    assert_eq!(totals.len(), QUESTIONS, "one request per question: {totals:?}");
    for (request, tokens) in totals.iter().skip(1) {
        assert!(
            *tokens < 300,
            "question {request} still claims the state: {tokens} tokens"
        );
    }
}

// ── decisions on the metrics listener (GitHub #241, spec 05) ────────────

/// What `MockCompute`'s readout puts inside the declared options:
/// `exp(-0.002)`, from the `OUTSIDE_THE_ANSWERS` nats it holds back
/// (`ignis_core::mock`). Every question gets the same one, which is what
/// makes the histogram's `_sum` an exact string rather than a range.
const MOCK_MASS: &str = "0.998002";

/// Scrapes until `done` holds, within a few seconds.
///
/// The telemetry consumer is asynchronous, so a series it feeds —
/// `ignis_decoded_tokens_total` among them — lands *after* the HTTP
/// response that caused it. Reading one straight after the response and
/// finding a zero says nothing at all: it would be zero for a readout that
/// decoded a hundred tokens too.
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

/// `GET /metrics` off the metrics listener's own app.
async fn scrape(metrics: &axum::Router) -> String {
    let request = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let response = metrics.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// The value of the sample named `name` with exactly `labels`.
fn sample(text: &str, name: &str, labels: &str) -> String {
    let key = match labels.is_empty() {
        true => format!("{name} "),
        false => format!("{name}{{{labels}}} "),
    };
    text.lines()
        .find_map(|line| line.strip_prefix(&key))
        .unwrap_or_else(|| panic!("no sample {name}{{{labels}}} in:\n{text}"))
        .to_owned()
}

#[tokio::test]
async fn a_served_decision_is_counted_under_its_type_and_its_mass_observed() {
    // Spec 05's acceptances 1, 2 and 3, through the two routers an operator
    // actually has: the decision goes to the API listener and the reading
    // comes off the metrics listener.
    let compute = Arc::new(MockCompute::new());
    let server = server(compute).with_metrics();
    let metrics = server.metrics_app().expect("--metrics is on");
    let app = server.app();

    // 3. Absent before any decision — a server that never sees one exports
    //    no decision series at all.
    let before = scrape(&metrics).await;
    assert!(!before.contains("ignis_decisions_total"), "{before}");
    assert!(!before.contains("ignis_decision_answer_mass"), "{before}");
    assert_eq!(sample(&before, "ignis_decoded_tokens_total", ""), "0");

    let (status, response) = decide(&app, JEV_CHOICE).await;
    assert_eq!(status, 200, "{response}");

    // 1. The counter moved, under this question's own primitive.
    let after = scrape(&metrics).await;
    assert_eq!(sample(&after, "ignis_decisions_total", "type=\"choice\""), "1");
    assert_eq!(sample(&after, "ignis_decisions_total", "type=\"noul\""), "0");
    assert_eq!(sample(&after, "ignis_decisions_total", "type=\"score\""), "0");

    // And the observation is the **answer mass**, not one of the other
    // numbers a readout carries. The mock holds back a fixed 0.002 nats
    // outside the declared options, so the reading is exactly `exp(-0.002)`
    // — pinned, because `_count == 1` and "it is a probability" would pass
    // just as well if the top option's probability, or the confidence, or a
    // constant had been observed instead.
    assert_eq!(sample(&after, "ignis_decision_answer_mass_count", ""), "1");
    assert_eq!(sample(&after, "ignis_decision_answer_mass_sum", ""), MOCK_MASS);
    assert_eq!(
        sample(&after, "ignis_decision_answer_mass_bucket", "le=\"0.999\""),
        "1",
        "and it lands in the bucket that reading belongs to"
    );
    assert_eq!(sample(&after, "ignis_decision_answer_mass_bucket", "le=\"0.995\""), "0");
    assert_eq!(
        sample(&after, "ignis_decision_answer_mass_bucket", "le=\"1\""),
        "1",
        "every reading was a probability"
    );

    // 2. A readout generates nothing, so neither token counter moved.
    //
    // Waited for rather than read: those two are fed by the asynchronous
    // telemetry consumer, so a scrape taken the instant the response
    // returns reports a zero whatever the request did. The settle is on a
    // series the decision *does* move through that same consumer.
    let settled = scrape_until(&metrics, |text| {
        text.contains("\nignis_requests_completed_total 1\n")
    })
    .await;
    assert_eq!(sample(&settled, "ignis_decoded_tokens_total", ""), "0");
    assert_eq!(sample(&settled, "ignis_generated_tokens_total", ""), "0");
    assert_eq!(response["usage"]["output_tokens"], 0);
}

#[tokio::test]
async fn every_question_of_a_fan_out_is_a_decision_of_its_own() {
    // Twenty questions over one `state` are twenty readouts, so they are
    // twenty decisions and twenty readings of answer mass. Counting the
    // HTTP request instead would hide the thing the histogram exists for:
    // one question's options collapsing while its siblings are fine.
    let compute = Arc::new(MockCompute::new());
    let server = server(compute).with_metrics();
    let metrics = server.metrics_app().expect("--metrics is on");
    let app = server.app();
    let body = r#"{
      "state": "Help! My payouts have been failing for 3 days.",
      "model": "test-model",
      "questions": {
        "a": { "type": "noul", "instructions": "Urgent?" },
        "b": { "type": "noul", "instructions": "Angry?" },
        "c": { "type": "choice", "instructions": "Which team?", "criteria": { "billing": null, "technical": null } },
        "d": { "type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Angry"] }
      }
    }"#;

    let (status, response) = decide(&app, body).await;
    assert_eq!(status, 200, "{response}");
    let text = scrape(&metrics).await;
    assert_eq!(sample(&text, "ignis_decisions_total", "type=\"noul\""), "2");
    assert_eq!(sample(&text, "ignis_decisions_total", "type=\"choice\""), "1");
    assert_eq!(sample(&text, "ignis_decisions_total", "type=\"score\""), "1");
    assert_eq!(sample(&text, "ignis_decision_answer_mass_count", ""), "4");
    // Four readings of the mock's one mass, summed — the histogram counts
    // questions, so a fan-out is four observations and not one.
    assert_eq!(sample(&text, "ignis_decision_answer_mass_sum", ""), "3.992008");
}

#[tokio::test]
async fn a_question_that_failed_is_not_counted_as_a_decision() {
    // A counter of decisions is a counter of readouts. A question whose
    // engine never answered has no mass to observe, and counting it would
    // put a hole in the histogram's own denominator — the rate of answered
    // questions would look right while the masses under it were one short.
    const QUESTIONS: usize = 8;
    let compute = Arc::new(DropReadout {
        inner: MockCompute::new(),
        victim: 3,
    });
    let server = server_over(compute as Arc<dyn ignis_core::Compute>).with_metrics();
    let metrics = server.metrics_app().expect("--metrics is on");
    let app = server.app();

    let (status, response) = decide(&app, &fan_out(300, QUESTIONS)).await;
    assert_eq!(status, 200, "{response}");
    let errors = response["answers"]
        .as_object()
        .expect("a map")
        .values()
        .filter(|answer| answer["type"] == "error")
        .count();
    assert_eq!(errors, 1, "{response}");

    let text = scrape(&metrics).await;
    assert_eq!(
        sample(&text, "ignis_decisions_total", "type=\"noul\""),
        (QUESTIONS - 1).to_string(),
        "the seven that were answered, not the eight that were asked"
    );
    assert_eq!(
        sample(&text, "ignis_decision_answer_mass_count", ""),
        (QUESTIONS - 1).to_string()
    );
}

#[tokio::test]
async fn a_server_without_metrics_serves_decisions_just_the_same() {
    // The projection is opt-in (ADR 0017) and the endpoint must not care:
    // `--metrics` off installs no `Metrics`, and the recording site is the
    // only thing that changes.
    let compute = Arc::new(MockCompute::new());
    let plain = server(compute);
    assert!(plain.metrics_app().is_none(), "metrics are off");
    let (status, response) = decide(&plain.app(), JEV_CHOICE).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(response["answers"]["department"]["type"], "choice");
}
