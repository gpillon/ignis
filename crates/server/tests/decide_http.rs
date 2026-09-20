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
struct DecidingTemplate;

impl TemplateProvider for DecidingTemplate {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
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
        SimpleTemplateProvider.prepare_multimodal(messages, options, tools, media)
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
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(DecidingTemplate))
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
async fn asking_for_thinking_inside_a_thinking_object_is_refused_too() {
    let compute = Arc::new(MockCompute::new());
    let body = r#"{"state":"s","thinking":{"enable_thinking":true},"questions":{"q":{"type":"noul","instructions":"Urgent?"}}}"#;
    let (status, response) = decide(&app(compute), body).await;
    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "thinking_unsupported");
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
    // same decision without the image is shorter by exactly the image.
    let prompt_tokens: usize = jobs.iter().map(|job| job.tokens.len()).sum();
    let text_only = Arc::new(MockCompute::new());
    let (_, _) = decide(
        &app(text_only.clone()),
        r#"{"state":"","questions":{"which":{"type":"choice","instructions":"Which number?","criteria":{"42":null,"47":null}}}}"#,
    )
    .await;
    let without: usize = text_only
        .prefill_calls()
        .into_iter()
        .flatten()
        .map(|job| job.tokens.len())
        .sum();
    assert_eq!(
        prompt_tokens - without,
        vision_tokens,
        "the image's {vision_tokens} placeholders are what the prompt grew by \
         ({prompt_tokens} with it, {without} without)"
    );
    assert_eq!(response["usage"]["output_tokens"], 0);
}
