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
    ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
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
    let compute = Arc::new(MockCompute::new());
    let (status, _) = decide(&app(compute.clone()), &fan_out(300, 8)).await;
    assert_eq!(status, 200);

    let batched = compute
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
        .body(Body::from(fan_out(300, 8).into_bytes()))
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

    let releases = tokio::task::spawn_blocking(move || {
        let mut seen = std::collections::BTreeSet::new();
        while let Ok(request) = released_rx.recv_timeout(std::time::Duration::from_secs(3)) {
            seen.insert(request);
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
    let cancelled: std::collections::BTreeSet<u64> =
        cancelled.lock().unwrap().iter().copied().collect();
    let followers: std::collections::BTreeSet<u64> =
        prefilled.iter().copied().filter(|&id| id >= 1).collect();
    assert!(
        followers.is_subset(&cancelled),
        "the handler's future dropping cancels every internal request still \
         alive: followers {followers:?}, cancelled {cancelled:?}"
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
    cancelled: Arc<std::sync::Mutex<Vec<ignis_core::RequestId>>>,
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
        // Recorded before delegating, and the return value is not the
        // property under test: the engine calls this for a request that has
        // already finished too, and `cancel` answers `false`. What a
        // disconnected client owes its siblings is that the cancel was
        // *issued*.
        self.cancelled.lock().unwrap().push(request);
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
type Cancelled = Arc<std::sync::Mutex<Vec<ignis_core::RequestId>>>;

fn recording() -> (axum::Router, Submitted) {
    let (app, submitted, _) = recording_over(Arc::new(MockCompute::new()));
    (app, submitted)
}

fn recording_over(compute: Arc<dyn ignis_core::Compute>) -> (axum::Router, Submitted, Cancelled) {
    let submitted: Submitted = Arc::new(std::sync::Mutex::new(Vec::new()));
    let cancelled: Cancelled = Arc::new(std::sync::Mutex::new(Vec::new()));
    let scheduler = RecordingScheduler {
        inner: scheduler_over(compute),
        submitted: submitted.clone(),
        cancelled: cancelled.clone(),
    };
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(DecidingTemplate));
    (server.app(), submitted, cancelled)
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
