//! The OpenAI-compatible HTTP surface (server-01): routes, request /
//! response schemas, handlers.
//!
//! Endpoints (`docs/design/ignis-v1.md` §2; open unless `--api-key` is set,
//! then each needs `Authorization: Bearer <key>` or answers `401`):
//! - `GET /v1/models` — the loaded model.
//! - `POST /v1/chat/completions` — chat completions, streaming (SSE) and
//!   non-streaming; routes into the core scheduler and streams tokens back
//!   as they are generated.
//! - `POST /v1/responses` — the OpenAI responses API (non-streaming in v1;
//!   a `stream: true` request is rejected with a 400).
//!
//! Error shape: OpenAI's `{"error": {message, type, code}}` body with the
//! matching status (400 bad request, 404 unknown model, 413 oversized
//! request, 503 engine full, 504 the engine did not finish the request in
//! the timeout).

use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{options, post};
use axum::{Json, Router};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tower_http::trace::{MakeSpan, TraceLayer};
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use ignis_core::{
    DecodeParams, FinishReason, RequestClass, RequestId, RequestInput, SchedEvent, SubmitError,
};

use crate::Server;
use crate::decoder::{Channel, OutputDecoder};
use crate::engine::{Completion, Engine, EventStream, RequestNotes, collect_completion};
use crate::media::{has_media, MediaRejection, MediaStats};
use crate::template::{
    check_content_parts, check_roles, ChatMessage, ContentRejection, RenderedPrompt, TemplateProvider,
    TemplateRejection,
};
use crate::thinking::{
    self, ThinkingDefaults, ThinkingError, ThinkingOptions, ThinkingRequestFields,
};
use crate::toolcall::{ToolCall as ScannedToolCall, ToolCallScanner, ToolEvent, ToolSchemas};

/// The request body limit of a `--vision` load, in bytes (the reference's
/// `--max-request-mib` default). It is this size because it is the media
/// budget's own: inline base64 is 4/3 of its decoded size, so the 256 MiB
/// `max_encoded_media_bytes` a request may spend needs 341 MiB of body to
/// arrive in. Anything lower would refuse, by byte count, media the budget
/// says is admissible -- and only for the inline path, since media fetched
/// by URL never crosses the body at all.
pub const MEDIA_REQUEST_BODY_LIMIT: usize = 384 << 20;

/// The request body limit of a text-only load, in bytes (GitHub #230). The
/// largest prompt the engine can accept at all is its attention envelope,
/// 1,048,576 tokens under hq-e8-2b, which is 3-4 MB of text; this is about
/// four times that, so an oversized prompt is refused by `--max-context`
/// with a 400 that names the context, never by a byte count that does not.
/// axum's own 2 MiB default sat *under* one max-context prompt.
pub const TEXT_REQUEST_BODY_LIMIT: usize = 16 << 20;

/// Build the OpenAI router for `server` (the axum state it serves behind).
///
/// `TraceLayer` (GitHub #81, ADR 0012) is the HTTP-ingress root span for
/// every request: `tower-http`'s well-tested span-per-request middleware,
/// not a hand-rolled equivalent. [`RootSpanMaker`] declares its
/// `request_id` field `Empty` at creation — the scheduler has not assigned
/// one yet at ingress — and the handler records it once
/// [`ignis_core::Scheduler::submit`] returns one, so every log record
/// emitted from inside the span (including this handler's own tail) gets
/// the request's real `trace_id` (`ignis_logging::trace_context`).
pub fn router(state: Arc<Server>) -> Router {
    let (v1, document) = v1_parts();
    let mut router = v1
        // Only the `/v1` routes above: the Playground's static pages, and
        // the API reference merged below, stay reachable without a key.
        .route_layer(middleware::from_fn_with_state(state.clone(), require_api_key));
    // The body cap, enforced before JSON parsing: wider with `--vision`,
    // which takes images inline as base64 data URIs (GitHub #179), than for
    // a text-only load, which only ever carries a prompt (GitHub #230).
    // Both are past what their own load can use, so the refusal an operator
    // meets is the one that names the real limit -- the context, or the
    // media budget -- rather than a byte count.
    router = router.layer(axum::extract::DefaultBodyLimit::max(if state.media.is_some() {
        MEDIA_REQUEST_BODY_LIMIT
    } else {
        TEXT_REQUEST_BODY_LIMIT
    }));
    // The Playground (GitHub #163): present unless `--no-ui` withheld the
    // assets -- served by default since ADR 0026's 2026-09-19 amendment.
    if let Some(assets) = state.playground {
        router = router.merge(crate::playground::router(assets));
    }
    // The Playground's copy of the Prometheus exposition (GitHub #89, ADR
    // 0017): only with the Playground and `--metrics`, and behind the same key
    // as `/v1` when one is set — an exposed server (ADR 0028) must not
    // publish its load to anyone. Prometheus itself scrapes the metrics
    // listener (`Server::metrics_app`); this listener has no `/metrics`.
    if let (Some(_), Some(metrics)) = (state.playground, &state.metrics) {
        router = router.merge(
            crate::metrics::router("/ui/metrics", Arc::clone(metrics))
                .route_layer(middleware::from_fn_with_state(state.clone(), require_api_key)),
        );
    }
    // The API reference (GitHub #251): `GET /v1` redirects to the Swagger
    // UI page at `/v1/docs/`, and the document it reads sits at
    // `/v1/openapi.json`. Merged *after* the key layer above, so a browser
    // pointed at `/v1` on a keyed server reaches the page rather than a
    // bare 401 it has no way to answer. Always served: the reference is
    // part of the `/v1` surface, not of the opt-in Playground.
    router = router.merge(crate::openapi::router(document));
    router
        .layer(middleware::from_fn(cors_headers))
        .layer(TraceLayer::new_for_http().make_span_with(RootSpanMaker))
        .with_state(state)
}

/// The `/v1` handler routes and the document they generated, built in the
/// one place (GitHub #251).
///
/// This is the seam that keeps the reference honest: `routes!` registers a
/// handler *and* its `#[utoipa::path]` entry in the same call, so a route
/// cannot exist without its documentation entry, nor an entry without its
/// route. `crates/server/tests/openapi_http.rs` asserts the resulting path
/// set, which is what fails when a new route is added by `.route()` alone.
///
/// `OPTIONS` is attached per path rather than through `routes!`: a CORS
/// preflight is a browser mechanism, not an operation a client calls, and
/// documenting it would put five meaningless entries in the document.
fn v1_parts() -> (Router<Arc<Server>>, utoipa::openapi::OpenApi) {
    OpenApiRouter::with_openapi(crate::openapi::ApiDoc::openapi())
        .routes(routes!(list_models))
        .routes(routes!(chat_completions))
        .routes(routes!(responses_api))
        // GitHub #239 — the decision endpoint, and the Jev name for it so an
        // unmodified Jev client reaches this server by changing the URL.
        // The alias is a `route`, not a `routes!`: it is the same handler
        // under a second name, and OpenAPI has no notion of an alias, so
        // listing it would duplicate every schema reference under it. The
        // `/v1/decide` description names it instead.
        .routes(routes!(crate::decide::decide))
        .route(
            "/v1/systemone",
            post(crate::decide::decide).options(cors_preflight),
        )
        .route("/v1/models", options(cors_preflight))
        .route("/v1/chat/completions", options(cors_preflight))
        .route("/v1/responses", options(cors_preflight))
        .route("/v1/decide", options(cors_preflight))
        .split_for_parts()
}

/// The OpenAPI document this build serves at `/v1/openapi.json`.
///
/// Public so the document's own assertions (the path set, the absence of
/// any monitoring path) can be made against the document itself rather
/// than through HTTP.
pub fn openapi() -> utoipa::openapi::OpenApi {
    v1_parts().1
}

/// The HTTP-ingress root span's shape: `request_id` is declared `Empty` —
/// unknown at ingress — and recorded once the request is admitted into the
/// scheduler (see `chat_completions`/`responses_api`). A request that never
/// reaches submission (a 400 before it, a CORS preflight, `GET
/// /v1/models`) simply never records it, so its span (and anything logged
/// under it) carries no `trace_id` — there is no request to correlate yet,
/// and this module never fabricates one (spec §19).
#[derive(Clone, Copy)]
struct RootSpanMaker;

impl<B> MakeSpan<B> for RootSpanMaker {
    fn make_span(&mut self, request: &axum::http::Request<B>) -> tracing::Span {
        tracing::info_span!(
            "ignis.http.request",
            method = %request.method(),
            path = %request.uri().path(),
            request_id = tracing::field::Empty,
        )
    }
}

/// With an API key configured, refuses a request that does not present it
/// as `Authorization: Bearer <key>` with OpenAI's `401 invalid_api_key`.
/// A CORS preflight passes untouched: browsers never attach credentials to
/// one, so gating it would break every cross-origin client.
async fn require_api_key(State(server): State<Arc<Server>>, req: Request, next: Next) -> Response {
    let Some(key) = &server.api_key else {
        return next.run(req).await;
    };
    if req.method() == axum::http::Method::OPTIONS {
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);
    if presented.is_some_and(|p| key.matches(p)) {
        return next.run(req).await;
    }
    let message = match presented {
        None => "missing API key: send `Authorization: Bearer <key>`",
        Some(_) => "incorrect API key provided",
    };
    let mut res = error_response(
        StatusCode::UNAUTHORIZED,
        "invalid_request_error",
        "invalid_api_key",
        message,
    );
    res.headers_mut()
        .insert(axum::http::header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    res
}

/// Answers a CORS preflight request with no body; `cors_headers` attaches
/// the `Access-Control-Allow-*` headers below.
async fn cors_preflight() -> StatusCode {
    StatusCode::OK
}

/// Adds permissive CORS headers to every response so the API is reachable
/// from browser-based clients on other origins.
async fn cors_headers(req: Request, next: Next) -> Response {
    let mut res = next.run(req).await;
    let headers = res.headers_mut();
    headers.insert("Access-Control-Allow-Origin", HeaderValue::from_static("*"));
    headers.insert(
        "Access-Control-Allow-Methods",
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        "Access-Control-Allow-Headers",
        HeaderValue::from_static("Content-Type, Authorization"),
    );
    res
}

/// The request's model, the templated prompt tokens, and the prompt-token
/// count (the usage figures) — one shared build path for both completion
/// endpoints.
fn build_request(
    server: &Server,
    model: Option<String>,
    messages: &[ChatMessage],
    params: DecodeParams,
    thinking: &ThinkingOptions,
    tools: &[JsonValue],
) -> Result<(RequestInput, String, u32), TemplateRejection> {
    // `model` is the model the request names; `None` (or a blank) falls
    // back to the loaded model. A model the engine does not load is
    // rejected at submit with a 404 (OpenAI's `model_not_found`).
    // The template seam: the artifact's frontend object set (artifact-02)
    // replaces this built-in provider through the same constructor
    // injection (v1 placeholder: deterministic word-hash tokens).
    let rendered = server.template.apply_chat_template(messages, thinking, tools)?;
    request_input(server, model, rendered, params, None)
}

/// The submitted request over already-templated `tokens`.
fn request_input(
    server: &Server,
    model: Option<String>,
    rendered: RenderedPrompt,
    params: DecodeParams,
    multimodal: Option<ignis_core::vision::Multimodal>,
) -> Result<(RequestInput, String, u32), TemplateRejection> {
    let model = model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| server.engine.model_id());
    let prompt_tokens = rendered.tokens.len() as u32;
    if prompt_tokens == 0 {
        return Err(TemplateRejection {
            code: "render_failed",
            message: "chat template rendered an empty prompt".to_owned(),
        });
    }
    // GitHub #193: a prompt carrying images reports its boundaries like any
    // other. Retained state is keyed by the images inside it as well as the
    // token ids (the #189 match key), so a request that sent another picture
    // never matches past the first placeholder they differ at.
    let input = RequestInput {
        decision: None,
        multimodal: multimodal.map(Arc::new),
        model: model.clone(),
        tokens: rendered.tokens,
        params,
        opener_tokens: rendered.opener_tokens,
        user_turn_tokens: rendered.user_turn_tokens,
        system_block_tokens: rendered.system_block_tokens,
        constrained: None,
};
    Ok((input, model, prompt_tokens))
}

/// [`build_request`] for any conversation: one carrying image parts on a
/// `--vision` load (GitHub #179) first has its media acquired and prepared
/// — a refusal is the error response, and dropping this future (a client
/// disconnect) stops that work before anything is submitted — then its
/// placeholders expanded, so `prompt_tokens` counts the image tokens. The
/// fourth value is the acquisition summary the request's events carry.
async fn prepare_request(
    server: &Server,
    model: Option<String>,
    messages: &[ChatMessage],
    params: DecodeParams,
    thinking: &ThinkingOptions,
    tools: &[JsonValue],
) -> Result<(RequestInput, String, u32, Option<MediaStats>), Response> {
    // GitHub #209: instruction messages are placed under the server's
    // policies before any template sees the conversation, on both paths.
    let messages = &server.instruction_policy.normalize(messages).map_err(template_rejection)?;
    let Some(acquirer) = server.media.as_ref().filter(|_| has_media(messages)) else {
        let (input, model, prompt_tokens) =
            build_request(server, model, messages, params, thinking, tools).map_err(template_rejection)?;
        return Ok((input, model, prompt_tokens, None));
    };
    let deadline = std::time::Instant::now() + server.request_timeout;
    let acquired = acquirer.acquire(messages, deadline).await.map_err(media_rejection)?;
    let (rendered, multimodal) = server
        .template
        .prepare_multimodal(messages, thinking, tools, acquired.media)
        .map_err(content_rejection)?;
    let (input, model, prompt_tokens) =
        request_input(server, model, rendered, params, Some(multimodal)).map_err(template_rejection)?;
    Ok((input, model, prompt_tokens, Some(acquired.stats)))
}

/// [`prepare_request`] for one question of a decision (GitHub #239): no
/// model override, no tools, thinking already forced off by the caller.
///
/// Shares the conversation path rather than duplicating it, so a decision's
/// prompt goes through the same instruction policy, the same media
/// acquisition and the same multimodal render as a chat turn — an image is
/// evidence here exactly as it is there. The error is a rendered response,
/// ready to stand in a question's slot.
pub(crate) async fn prepare_decision_request(
    server: &Server,
    model: Option<String>,
    messages: &[ChatMessage],
    params: DecodeParams,
    thinking: &ThinkingOptions,
) -> Result<(RequestInput, String, u32, Option<MediaStats>), (&'static str, String)> {
    prepare_request(server, model, messages, params, thinking, &[])
        .await
        .map_err(|response| {
            // The shared path answers with a rendered `Response`, which is
            // the wrong shape here: a decision's refusal is one of N, and
            // has to carry a code the decision's own 422 can name. What
            // survives the crossing is the status, which is enough to say
            // *which* of the two things went wrong.
            let code = if response.status() == StatusCode::BAD_REQUEST {
                "malformed_request"
            } else {
                "render_failed"
            };
            (code, format!("its prompt was refused ({})", response.status()))
        })
}

/// The response for refused media (GitHub #179): a 400 with the media
/// code, or the handler's own 504 when the request deadline passed.
fn media_rejection(rejection: MediaRejection) -> Response {
    let status = StatusCode::from_u16(rejection.status).unwrap_or(StatusCode::BAD_REQUEST);
    let type_ = if status == StatusCode::BAD_REQUEST { "invalid_request_error" } else { rejection.code };
    error_response(status, type_, rejection.code, rejection.message)
}

/// Split an optional `model` into its base id and any Lane tag (`CONTEXT.md`:
/// "the request's own statement of its class") named by an "@<lane>" suffix
/// (GitHub #120: the OpenAI `model` field has no room of its own for an
/// extension, so this is the second entry point for the `class` ignis
/// extension besides the `class` field itself, e.g. `"qwen2.5-7b@agent"`).
/// The suffix is always stripped from the returned model — including an
/// unrecognized one — since it is never part of the model id the scheduler
/// looks up; an empty base (`"@agent"`) is treated as having no suffix at
/// all, leaving the whole string as the model name.
pub(crate) fn split_model_lane(model: Option<String>) -> (Option<String>, Option<RequestClass>) {
    match model {
        None => (None, None),
        Some(m) => match m.rsplit_once('@') {
            Some((base, lane)) if !base.is_empty() => {
                (Some(base.to_string()), Some(RequestClass::from_extension(lane)))
            }
            _ => (Some(m), None),
        },
    }
}

/// Resolve the request's [`RequestClass`] from its two possible wire entry
/// points (GitHub #120): the `class` JSON extension field, and `from_model`
/// (any class already read off the model's "@<lane>" suffix by
/// [`split_model_lane`]). An explicit `class` field wins when both are set.
/// A `class` of the wrong JSON shape is refused — the same "honoured or
/// refused, never silently dropped" contract `top_k` uses (#101) — but an
/// absent field, a `null`, or a recognized-shape-but-unrecognized string all
/// fall through to [`RequestClass::from_extension`]'s own documented
/// default (`Interactive`), same as an untagged model.
fn resolve_class(
    class: Option<JsonValue>,
    from_model: Option<RequestClass>,
) -> Result<RequestClass, String> {
    match class {
        None | Some(JsonValue::Null) => Ok(from_model.unwrap_or_default()),
        Some(JsonValue::String(s)) => Ok(RequestClass::from_extension(&s)),
        Some(other) => Err(format!(
            "class is an ignis extension and must be a string (\"interactive\" or \"agent\"), got {other}"
        )),
    }
}

/// `model` and its Lane tag, resolved together (GitHub #120): strip any
/// "@<lane>" suffix ([`split_model_lane`]) and fold it with the explicit
/// `class` field ([`resolve_class`]) into the class to submit under. The one
/// path both completion endpoints share, so the two-entry-point resolution
/// lives in a single place rather than being repeated per handler.
fn resolve_model_and_class(
    model: Option<String>,
    class: Option<JsonValue>,
) -> Result<(Option<String>, RequestClass), String> {
    let (model, model_class) = split_model_lane(model);
    let class = resolve_class(class, model_class)?;
    Ok((model, class))
}

/// The sampling fields accepted by chat completions. Values stay as JSON at
/// the wire boundary so type, integer-width, and narrowing failures all use
/// the same OpenAI-shaped sampling error instead of Axum's generic rejection.
#[derive(Clone, Default, Deserialize, ToSchema)]
struct SamplingRequestFields {
    temperature: Option<JsonValue>,
    top_p: Option<JsonValue>,
    /// An ignis extension, not an OpenAI Chat Completions parameter.
    top_k: Option<JsonValue>,
    presence_penalty: Option<JsonValue>,
    frequency_penalty: Option<JsonValue>,
    seed: Option<JsonValue>,
}

impl SamplingRequestFields {
    /// `ignore_eos` is an ignis extension for bounded measurement streams.
    /// It is refused without a `max_tokens`, because the two together are
    /// what keeps such a request bounded: with neither an EOS nor a cap,
    /// a non-streaming request has nothing left to stop it, and only the
    /// streaming path cancels on client disconnect.
    fn resolve(self, max_tokens: Option<u32>, ignore_eos: bool) -> Result<DecodeParams, String> {
        let temperature = bounded_f32(
            "temperature",
            number("temperature", self.temperature, 0.0)?,
            0.0,
            2.0,
        )?;
        let top_p = bounded_f32("top_p", number("top_p", self.top_p, 1.0)?, 0.0, 1.0)?;
        let presence_penalty = bounded_f32(
            "presence_penalty",
            number("presence_penalty", self.presence_penalty, 0.0)?,
            -2.0,
            2.0,
        )?;
        let frequency_penalty = bounded_f32(
            "frequency_penalty",
            number("frequency_penalty", self.frequency_penalty, 0.0)?,
            -2.0,
            2.0,
        )?;
        let top_k = signed_integer(
            "top_k is an ignis extension and must be an integer",
            self.top_k,
            0,
        )?;
        if !(0..=20).contains(&top_k) {
            return Err(format!(
                "top_k is an ignis extension and must be between 0 and 20 inclusive; 0 selects ignis's 20-candidate sampler cap (got {top_k})"
            ));
        }

        if ignore_eos && max_tokens.is_none() {
            return Err(
                "ignore_eos is an ignis extension for bounded measurement streams and requires max_tokens"
                    .into(),
            );
        }

        if temperature == 0.0
            && (top_p != 1.0
                || (top_k != 0 && top_k != 20)
                || presence_penalty != 0.0
                || frequency_penalty != 0.0)
        {
            return Err(
                "temperature must be greater than 0 when top_p, the ignis top_k extension, presence_penalty, or frequency_penalty would otherwise be ignored by greedy sampling"
                    .into(),
            );
        }

        Ok(DecodeParams {
            max_tokens,
            temperature,
            top_p,
            top_k: top_k as i32,
            presence_penalty,
            frequency_penalty,
            // The leaf keys its counter-based RNG with all 64 bits. Casting
            // preserves the complete signed OpenAI seed domain bit-for-bit.
            seed: signed_integer("seed must be a signed 64-bit integer", self.seed, 0)? as u64,
            ignore_eos,
            thinking_budget: None,
        })
    }
}

fn number(name: &str, value: Option<JsonValue>, default: f64) -> Result<f64, String> {
    match value {
        None => Ok(default),
        Some(JsonValue::Number(value)) => value
            .as_f64()
            .ok_or_else(|| format!("{name} must be a finite number")),
        Some(value) => Err(format!("{name} must be a number (got {value})")),
    }
}

fn signed_integer(message: &str, value: Option<JsonValue>, default: i64) -> Result<i64, String> {
    match value {
        None => Ok(default),
        Some(JsonValue::Number(value)) => value.as_i64().ok_or_else(|| message.to_owned()),
        Some(_) => Err(message.to_owned()),
    }
}

fn bounded_f32(name: &str, value: f64, min: f64, max: f64) -> Result<f32, String> {
    if value.is_finite() && (min..=max).contains(&value) {
        let narrowed = value as f32;
        if value != 0.0 && narrowed == 0.0 {
            Err(format!(
                "{name} magnitude is too small to be represented by the engine (got {value})"
            ))
        } else {
            Ok(narrowed)
        }
    } else {
        Err(format!(
            "{name} must be between {min} and {max} inclusive (got {value})"
        ))
    }
}

fn invalid_sampling_parameter(message: impl Into<String>) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "invalid_sampling_parameter",
        message,
    )
}

/// Resolve one request's thinking controls, or the OpenAI-shaped 400 to
/// return instead. Validation errors and capability errors carry different
/// `code`s (the client's mistake vs. the loaded model's limitation).
fn resolve_thinking(
    server: &Server,
    fields: ThinkingRequestFields<'_>,
) -> Result<ThinkingOptions, Response> {
    let defaults = ThinkingDefaults {
        enable_thinking: server.default_enable_thinking,
        reasoning_effort: server.default_reasoning_effort,
    };
    let capabilities = server.template.thinking_capabilities();
    thinking::resolve(fields, &defaults, &capabilities).map_err(|err| match err {
        ThinkingError::Validation(message) => bad_request(&message),
        ThinkingError::Capability(message) => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "reasoning_effort_unsupported",
            message,
        ),
    })
}

/// The request's thinking budget (spec server/08) on its decode parameters,
/// and whether `max` dropped one — or the 400 for a malformed one. Set only
/// on a generation that starts inside the reasoning block: with thinking off
/// there is no block to close, and nothing for `max` to have dropped.
///
/// `effort` is the request's raw `reasoning_effort`: `max` is decided from
/// it (else from the server default), not from `thinking`, which carries
/// the effort the template takes.
fn with_thinking_budget(
    server: &Server,
    params: DecodeParams,
    value: Option<&JsonValue>,
    effort: Option<&JsonValue>,
    thinking: &ThinkingOptions,
) -> Result<(DecodeParams, bool), Response> {
    let max = thinking::runs_at_max(effort, server.default_reasoning_effort);
    let resolved = thinking::resolve_thinking_budget(value, server.default_thinking_budget, max)
        .map_err(|message| bad_request_param(&message, "thinking_budget"))?;
    let starts_in_reasoning = server.template.decoder_starts_in_reasoning(thinking);
    Ok((
        DecodeParams {
            thinking_budget: resolved.budget.filter(|_| starts_in_reasoning),
            ..params
        },
        resolved.dropped_by_max && starts_in_reasoning,
    ))
}

/// The two `tool_choice` values this template has a lever for (GitHub
/// #132). `"required"` and the named-function object form parse but are
/// rejected outright ([`parse_tool_choice`]) rather than represented here
/// — there is nothing a resolved value of this type could do with them,
/// since this text-instruction template has no way to *force* a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolChoice {
    /// The model decides (the default, and the only thing a text
    /// instruction template can actually offer beyond "don't call").
    Auto,
    /// The model is never told tools exist at all.
    None,
}

/// Parse `tool_choice`'s wire value: `"auto"` (default) or `"none"`.
/// `"required"` and the named-function object form are a 400 explaining
/// why, not a silently-ignored field (same posture as an unsupported
/// `reasoning_effort`, GitHub #68).
fn parse_tool_choice(tool_choice: Option<JsonValue>) -> Result<ToolChoice, Response> {
    match tool_choice {
        None => Ok(ToolChoice::Auto),
        Some(JsonValue::String(s)) if s == "auto" => Ok(ToolChoice::Auto),
        Some(JsonValue::String(s)) if s == "none" => Ok(ToolChoice::None),
        Some(JsonValue::String(s)) if s == "required" => Err(bad_request(
            "tool_choice: \"required\" is not supported — this template has no way to force a tool call; use \"auto\" and let the model decide, or \"none\"",
        )),
        Some(JsonValue::Object(_)) => Err(bad_request(
            "tool_choice naming a specific function is not supported — this template has no way to force a tool call; use \"auto\" and let the model decide, or \"none\"",
        )),
        Some(other) => Err(bad_request(&format!(
            "tool_choice must be \"auto\" or \"none\", got {other}"
        ))),
    }
}

/// Validate and resolve `tools` + `tool_choice` (GitHub #132) into the
/// tools slice that actually reaches the template.
///
/// Each `tools` entry must be `{"type": "function", "function": {"name":
/// <non-empty string>, ...}}` — anything else is a 400 naming the entry
/// and what is wrong with it. Extra fields (`description`, `parameters`,
/// …) ride through untouched; they are opaque JSON to ignis, meaningful
/// only to the template and the model — except the two defaults the
/// reference fills in ([`normalize_tool`]). [`ToolChoice::None`] discards
/// the validated tools — the template never sees them, so the model is
/// never told tools exist.
fn resolve_tools(
    tools: Option<Vec<JsonValue>>,
    tool_choice: Option<JsonValue>,
) -> Result<Vec<JsonValue>, Response> {
    let mut tools = tools.unwrap_or_default();
    tools.iter_mut().for_each(normalize_tool);
    for (index, tool) in tools.iter().enumerate() {
        let is_type_function = tool.get("type").and_then(JsonValue::as_str) == Some("function");
        let has_name = tool
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(JsonValue::as_str)
            .is_some_and(|name| !name.is_empty());
        if !is_type_function || !has_name {
            return Err(bad_request(&format!(
                "tools[{index}] must be {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}"
            )));
        }
    }
    match parse_tool_choice(tool_choice)? {
        ToolChoice::Auto => Ok(tools),
        ToolChoice::None => Ok(Vec::new()),
    }
}

/// The reference's tool normalization (GitHub #172, ninfer
/// `openai_schema.cpp` `parse_tools`), so the prompt carries the same tool
/// text: a function without `parameters` gets `{"type": "object",
/// "properties": {}}`, one without `strict` gets `false` (both also when
/// sent as `null`). Anything that is not a function object is left for
/// validation to reject.
fn normalize_tool(tool: &mut JsonValue) {
    let Some(function) = tool.get_mut("function").and_then(JsonValue::as_object_mut) else {
        return;
    };
    if function.get("parameters").is_none_or(JsonValue::is_null) {
        function.insert(
            "parameters".to_owned(),
            serde_json::json!({"type": "object", "properties": {}}),
        );
    }
    if function.get("strict").is_none_or(JsonValue::is_null) {
        function.insert("strict".to_owned(), JsonValue::Bool(false));
    }
}

/// Split a finished generation's tokens into `reasoning_content` /
/// `content` (GitHub #68): feeds the whole token list through the same
/// [`OutputDecoder`] the streaming path uses one token at a time, so the
/// two modes agree by construction (spec story 24).
fn split_reasoning(
    template: &dyn TemplateProvider,
    tokens: &[ignis_core::TokenId],
    thinking: &ThinkingOptions,
) -> (Option<String>, String) {
    let starts_in_reasoning = template.decoder_starts_in_reasoning(thinking);
    let mut decoder = OutputDecoder::new(template.token_decoder(), starts_in_reasoning);
    let mut deltas = decoder.push(tokens);
    deltas.extend(decoder.finish());
    let mut reasoning = String::new();
    let mut content = String::new();
    for delta in deltas {
        match delta.channel {
            Channel::Reasoning => reasoning.push_str(&delta.text),
            Channel::Content => content.push_str(&delta.text),
        }
    }
    (
        if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
        content,
    )
}

/// [`split_reasoning`] plus tool-call extraction (GitHub #121): the content
/// channel's text is fed through a fresh [`ToolCallScanner`] so the
/// non-streaming path parses tool calls with exactly the same rules as the
/// streaming path (a call left open by a truncated generation is dropped,
/// never returned half-written — acceptance criterion 3). Arguments are
/// typed by the request's `tools` (`schemas`).
fn split_reasoning_and_tools(
    template: &dyn TemplateProvider,
    tokens: &[ignis_core::TokenId],
    thinking: &ThinkingOptions,
    schemas: ToolSchemas,
) -> (Option<String>, String, Vec<ScannedToolCall>) {
    let (reasoning, content_text) = split_reasoning(template, tokens, thinking);
    let mut scanner = ToolCallScanner::with_schemas(schemas);
    let mut events = scanner.feed(&content_text);
    events.extend(scanner.finish());
    let mut content = String::new();
    let mut calls = Vec::new();
    for event in events {
        match event {
            ToolEvent::Content(text) => content.push_str(&text),
            ToolEvent::Call(call) => calls.push(call),
        }
    }
    (reasoning, content, calls)
}

/// Map a [`SubmitError`] from the engine's submit to the OpenAI-shaped
/// error response (404 unknown model, 400 context exceeded, 413 oversized,
/// 503 engine full). With metrics on, the rejection is counted here, on the
/// HTTP side, once the submit call has returned its error (ADR 0017): the
/// model thread sends no fact for it.
fn submit_error(server: &Server, err: SubmitError) -> Response {
    if let Some(metrics) = &server.metrics {
        metrics.record_rejected(crate::metrics::Rejection::of(&err));
    }
    match err {
        SubmitError::ContextExceeded { requested, limit } => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "context_length_exceeded",
            format!(
                "request needs {requested} tokens (prompt + max_tokens), over the model's {limit}-token context; lower max_tokens or shorten the prompt"
            ),
        ),
        SubmitError::UnknownModel(m) => error_response(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "model_not_found",
            format!("unknown model: {m} (loaded: {})", server.engine.model_id()),
        ),
        SubmitError::Oversized => error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_too_large",
            "request_too_large",
            "request exceeds the engine's KV pool (prompt + max_tokens exceeds the pool)",
        ),
        SubmitError::Full => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine_full",
            "engine_full",
            "the engine cannot admit the request right now (all lanes in use); retry",
        ),
    }
}

/// A 400 with the OpenAI error body.
fn bad_request(message: &str) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "invalid_request_error",
        message,
    )
}

/// The 400 for refused content parts (GitHub #175): the rejection's own
/// code, raised before the request reaches the engine.
fn content_rejection(rejection: ContentRejection) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        rejection.code,
        rejection.message,
    )
}

/// The 400 for a text-template/tokenizer/role refusal, before any request is
/// admitted to the engine (GitHub #208).
fn template_rejection(rejection: TemplateRejection) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        rejection.code,
        rejection.message,
    )
}

/// The OpenAI error body (`{"error": {message, type, code}}`).
fn error_response(
    status: StatusCode,
    type_: &str,
    code: &str,
    message: impl Into<String>,
) -> Response {
    error_response_naming(status, type_, code, message, None)
}

/// [`error_response`], naming the one request field at fault in
/// `error.param` when there is one.
fn error_response_naming(
    status: StatusCode,
    type_: &str,
    code: &str,
    message: impl Into<String>,
    param: Option<&str>,
) -> Response {
    (
        status,
        Json(ApiError {
            error: ErrorBody {
                message: message.into(),
                r#type: type_.into(),
                code: code.into(),
                param: param.map(str::to_owned),
            },
        }),
    )
        .into_response()
}

/// A 400 naming the one request field at fault in `error.param`.
fn bad_request_param(message: &str, param: &str) -> Response {
    error_response_naming(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "invalid_request_error",
        message,
        Some(param),
    )
}

/// The `504` body for a request `collect_completion` gave up on: names the
/// timeout that fired (GitHub #95) so an operator reading the error knows
/// what to raise with `--request-timeout`/`IGNIS_REQUEST_TIMEOUT`, rather
/// than suspecting a wedged engine when a healthy one just needed longer.
fn request_timeout_message(timeout: std::time::Duration) -> String {
    format!(
        "the request did not complete within the server's {}s timeout (the engine may be wedged) — raise it with --request-timeout/IGNIS_REQUEST_TIMEOUT",
        timeout.as_secs()
    )
}

/// Map the engine's [`FinishReason`] to the OpenAI `finish_reason` string
/// (GitHub #61 / P1-25): `stop` on the model's own EOS token, `length` on
/// `max_tokens` or the engine's reservation cap, `error` when the engine
/// gave up on the request (GitHub #166). `pub(crate)` since P3-06:
/// `telemetry.rs`'s `ignis.request.done` event reuses this exact mapping so
/// the request log and the HTTP response never disagree on the string.
pub(crate) fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Error => "error",
    }
}

/// The `500` a non-streaming request gets when the engine ended it with
/// [`FinishReason::Error`] (GitHub #166): there is no completion to return.
fn engine_error_response() -> Response {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "engine_error",
        "engine_error",
        "the engine could not run the request (its prefill failed repeatedly); see the server log",
    )
}

/// The OpenAI `finish_reason`, tool-calls aware (GitHub #121). A generation
/// that stopped cleanly (its own EOS) after emitting at least one complete
/// tool call reports `"tool_calls"`, matching what a real agent client
/// branches on to decide whether to execute a call rather than just render
/// text. A call left open when generation stopped (`mid_call`) never earns
/// that label — it was dropped whole by [`ToolCallScanner`], so reporting
/// `"tool_calls"` here would tell the client to run a call that was never
/// actually delivered (acceptance criterion 3); the plain `stop`/`length`
/// reason is reported instead, same as if no call had been attempted.
pub(crate) fn resolve_finish_reason(
    reason: FinishReason,
    any_calls: bool,
    mid_call: bool,
) -> &'static str {
    if !mid_call && any_calls && reason == FinishReason::Stop {
        "tool_calls"
    } else {
        finish_reason_str(reason)
    }
}

/// GitHub #70 via #121 acceptance criterion 5: a generation that reasoned
/// but never produced content or a tool call must not pass silently —
/// report it explicitly rather than let it look, on the wire, like an
/// ordinary short (or empty) answer. Shared by both response paths so the
/// message can't drift between them (each path computes `all_reasoning`
/// its own way — a whole-string check non-streaming, incremental flags
/// while streaming — since neither has the other's representation of the
/// generation to reuse).
fn report_if_all_reasoning_no_content(id: &str, all_reasoning: bool, finish_reason: &'static str) {
    if all_reasoning {
        tracing::warn!(
            id,
            finish_reason,
            "generation produced reasoning but no content or tool call \
             (token budget exhausted before an answer began)"
        );
    }
}

/// The Unix epoch seconds (OpenAI's `created` / `created_at` fields).
fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── GET /v1/models ────────────────────────────────────────────────────────

/// `GET /v1/models` — the loaded model (v1: a single model).
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "models",
    operation_id = "list_models",
    summary = "The loaded model",
    description = "One entry: the model this server loaded, with the context a single request may spend (`max_model_len`, prompt plus completion). A server serves one model.",
    responses(
        (status = 200, description = "The loaded model.", body = ModelList),
        (status = 401, description = "The server was started with `--api-key` and the request carried no matching bearer token.", body = ApiError),
    ),
)]
async fn list_models(State(server): State<Arc<Server>>) -> Json<ModelList> {
    let id = server.engine.model_id();
    Json(ModelList {
        object: "list",
        data: vec![ModelInfo {
            id,
            object: "model",
            owned_by: "ignis",
            max_model_len: server.engine.max_model_len(),
        }],
    })
}

/// The models list envelope.
#[derive(Serialize, ToSchema)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelInfo>,
}

/// One model entry.
#[derive(Serialize, ToSchema)]
struct ModelInfo {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    /// The context for one request, prompt plus `max_tokens`, in tokens
    /// (vLLM's name for it; the Playground's context bar reads it).
    max_model_len: u32,
}

// ── POST /v1/chat/completions ─────────────────────────────────────────────

/// A chat-completions request (OpenAI wire shape; unknown fields are
/// ignored — serde's default).
#[derive(Deserialize, ToSchema)]
struct ChatCompletionsRequest {
    model: Option<String>,
    /// The conversation (role + content messages). Must be non-empty.
    messages: Vec<ChatMessage>,
    /// `true` → stream the completion as SSE chunks.
    #[serde(default)]
    stream: bool,
    /// Streaming-only options. `include_usage: true` appends the trailing
    /// usage chunk (empty `choices`, populated `usage`) before `[DONE]`.
    stream_options: Option<StreamOptions>,
    max_tokens: Option<u32>,
    /// An ignis extension: keep decoding past the model's EOS token. Used
    /// by the G3 inter-token-latency lanes, which are ended by the
    /// measurement window rather than by the model. Requires `max_tokens`.
    #[serde(default)]
    ignore_eos: bool,
    #[serde(flatten)]
    sampling: SamplingRequestFields,
    /// The thinking controls (GitHub #68) — kept as raw JSON so the wire
    /// contract's tri-state semantics (absent / `null` / a bad type) and
    /// its own error messages are decided by `thinking::resolve`, not by
    /// serde's generic type-mismatch error.
    enable_thinking: Option<JsonValue>,
    reasoning_effort: Option<JsonValue>,
    preserve_thinking: Option<JsonValue>,
    chat_template_kwargs: Option<JsonValue>,
    /// An ignis extension (2026-09-24): the reasoning tokens this request
    /// may spend before the model's own close is forced. A whole number, at
    /// least 1; absent or `null` takes the server's `--thinking-budget`.
    thinking_budget: Option<JsonValue>,
    /// The tool definitions (GitHub #132) — opaque JSON, validated
    /// shallowly and passed to the template as-is (`resolve_tools`).
    tools: Option<Vec<JsonValue>>,
    /// `"auto"` (default) or `"none"`; anything else is a 400
    /// (`resolve_tools`) — this template has no lever to *force* a call.
    tool_choice: Option<JsonValue>,
    /// The Lane tag (GitHub #120, `CONTEXT.md`): an ignis extension, not an
    /// OpenAI parameter (the way `top_k` went in at #101). `"interactive"`
    /// (default) or `"agent"`, the admission class `admission.rs` schedules
    /// the request under. Also settable via an "@<lane>" suffix on `model`
    /// (`split_model_lane`); this field wins if both are set.
    class: Option<JsonValue>,
}

#[derive(Deserialize, ToSchema)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// `POST /v1/chat/completions` — non-streaming and streaming (SSE).
///
/// The request routes into the core scheduler (the request's event stream
/// carries its generated tokens) and the handler either collects them into
/// the single-response JSON (non-streaming) or wraps the stream in the
/// `chat.completion.chunk` SSE shape (streaming).
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "chat",
    operation_id = "chat_completions",
    summary = "A chat completion, streaming or not",
    description = "The OpenAI chat-completions contract, plus what ignis adds to it: `top_k`, `ignore_eos`, the thinking controls (`enable_thinking`, `reasoning_effort`, `preserve_thinking`, `chat_template_kwargs`, `thinking_budget`), and the `class` lane tag the scheduler admits under.

`stream: false` answers one JSON body. `stream: true` answers `text/event-stream`: one `data:` line per chunk in the `chat.completion.chunk` shape, a final chunk carrying `finish_reason` and an empty `delta`, then a literal `data: [DONE]` line. With `stream_options.include_usage: true` a usage-only chunk (empty `choices`) precedes it.

Tool calls come back whole -- one complete `tool_calls` delta per call, never a half-written fragment -- because they are parsed out of a closed block in the generated text.",
    request_body = ChatCompletionsRequest,
    responses(
        (status = 200, description = "The completion. `application/json` when `stream` is false or absent; `text/event-stream` when it is true.", content(
            (ChatCompletion = "application/json"),
            (Chunk = "text/event-stream"),
        )),
        (status = 400, description = "The request is malformed: an empty `messages`, an unknown role, a sampling parameter out of range, a tool definition this template cannot take.", body = ApiError),
        (status = 401, description = "The server was started with `--api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "The request named a model this server has not loaded.", body = ApiError),
        (status = 413, description = "The prompt is longer than this server's `--max-context`.", body = ApiError),
        (status = 503, description = "The engine is at capacity and the request was not admitted.", body = ApiError),
        (status = 504, description = "The engine did not finish the request within `--request-timeout`.", body = ApiError),
    ),
)]
async fn chat_completions(
    State(server): State<Arc<Server>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Response {
    if req.messages.is_empty() {
        return bad_request("messages must not be empty");
    }
    if let Err(rejection) = check_roles(&req.messages) {
        return template_rejection(rejection);
    }
    if let Err(rejection) = check_content_parts(&req.messages, server.media.is_some()) {
        return content_rejection(rejection);
    }
    let params = match req.sampling.resolve(req.max_tokens, req.ignore_eos) {
        Ok(params) => params,
        Err(message) => return invalid_sampling_parameter(message),
    };
    let thinking = match resolve_thinking(
        &server,
        ThinkingRequestFields {
            enable_thinking: req.enable_thinking.as_ref(),
            reasoning_effort: req.reasoning_effort.as_ref(),
            preserve_thinking: req.preserve_thinking.as_ref(),
            chat_template_kwargs: req.chat_template_kwargs.as_ref(),
        },
    ) {
        Ok(t) => t,
        Err(response) => return response,
    };
    let (params, budget_dropped) = match with_thinking_budget(
        &server,
        params,
        req.thinking_budget.as_ref(),
        req.reasoning_effort.as_ref(),
        &thinking,
    ) {
        Ok(resolved) => resolved,
        Err(response) => return response,
    };
    let tools = match resolve_tools(req.tools, req.tool_choice) {
        Ok(tools) => tools,
        Err(response) => return response,
    };
    let (model, class) = match resolve_model_and_class(req.model, req.class) {
        Ok(x) => x,
        Err(message) => return bad_request(&message),
    };
    let schemas = ToolSchemas::from_tools(&tools);
    let (input, model, prompt_tokens, media) =
        match prepare_request(&server, model, &req.messages, params, &thinking, &tools).await {
            Ok(prepared) => prepared,
            Err(response) => return response,
        };
    let notes = RequestNotes { media, thinking_budget_dropped: budget_dropped };
    let (request_id, mut stream) = match server.engine.submit_with_notes(input, class, notes).await {
        Ok(x) => x,
        Err(err) => return submit_error(&server, err),
    };
    // GitHub #81 / ADR 0012: the HTTP root span declared `request_id`
    // `Empty` at ingress (`RootSpanMaker`) since the scheduler had not
    // assigned one yet; record it now so every log record for the rest of
    // this span's life carries the request's real `trace_id`.
    tracing::Span::current().record("request_id", request_id);
    let id = format!("chatcmpl-{request_id}");
    let created = now();

    if req.stream {
        // The SSE response: the request's event stream wrapped in the
        // `chat.completion.chunk` shape (a `[DONE]` marker terminates).
        let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
        let starts_in_reasoning = server.template.decoder_starts_in_reasoning(&thinking);
        return Sse::new(ChunkStream::new(
            stream,
            CancelOnDrop::new(server.engine.clone(), request_id),
            id,
            created,
            model,
            OutputDecoder::new(server.template.token_decoder(), starts_in_reasoning),
            schemas,
            prompt_tokens,
            include_usage,
        ))
        .into_response();
    }
    // Non-streaming: collect the request's tokens to its completion (a
    // timeout guards a wedged engine from hanging the client).
    match collect_completion(&mut stream, server.request_timeout).await {
        Ok(Completion { reason: FinishReason::Error, .. }) => engine_error_response(),
        Ok(Completion { tokens, reason, thinking: budget }) => {
            let (reasoning_content, content, tool_calls) =
                split_reasoning_and_tools(server.template.as_ref(), &tokens, &thinking, schemas);
            let completion_tokens = tokens.len() as u32;
            let finish_reason = resolve_finish_reason(reason, !tool_calls.is_empty(), false);
            report_if_all_reasoning_no_content(
                &id,
                reasoning_content.as_deref().is_some_and(|r| !r.is_empty())
                    && content.is_empty()
                    && tool_calls.is_empty(),
                finish_reason,
            );
            let tool_calls = if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.into_iter().map(ToolCallOut::from).collect())
            };
            Json(ChatCompletion {
                id,
                object: "chat.completion",
                created,
                model,
                choices: vec![CompletionChoice {
                    index: 0,
                    message: AssistantMessage {
                        role: "assistant",
                        reasoning_content,
                        content,
                        tool_calls,
                    },
                    finish_reason,
                    thinking_budget_forced_at: budget.and_then(|b| b.forced_at),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens.saturating_add(completion_tokens),
                },
            })
            .into_response()
        }
        Err(_) => error_response(
            StatusCode::GATEWAY_TIMEOUT,
            "request_timeout",
            "request_timeout",
            request_timeout_message(server.request_timeout),
        ),
    }
}

/// The non-streaming completion response (OpenAI wire shape).
#[derive(Serialize, ToSchema)]
struct ChatCompletion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize, ToSchema)]
struct CompletionChoice {
    index: u8,
    message: AssistantMessage,
    finish_reason: &'static str,
    /// An ignis extension (spec server/08): the reasoning tokens emitted when
    /// the thinking budget forced the model's close. Absent — not `null` —
    /// when the close was not forced.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget_forced_at: Option<u32>,
}

#[derive(Serialize, ToSchema)]
struct AssistantMessage {
    role: &'static str,
    /// The model's thinking trace (GitHub #68) — omitted entirely (not
    /// serialized as an empty string) when thinking produced no reasoning,
    /// so a thinking-disabled response is structurally distinct from a
    /// thinking-enabled one that happened to reason not at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    content: String,
    /// Tool calls parsed out of the content channel (GitHub #121) — absent
    /// when the generation made none.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallOut>>,
}

/// One tool call, non-streaming wire shape (OpenAI: `message.tool_calls[]`).
#[derive(Serialize, ToSchema)]
struct ToolCallOut {
    id: String,
    r#type: &'static str,
    function: FunctionOut,
}

#[derive(Serialize, ToSchema)]
struct FunctionOut {
    name: String,
    /// A JSON-encoded object, e.g. `{"path":"a.txt"}` (never a
    /// half-written fragment — GitHub #121 acceptance criterion 3: an
    /// interrupted call is dropped entirely by `ToolCallScanner`, never
    /// surfaced here truncated).
    arguments: String,
}

impl From<ScannedToolCall> for ToolCallOut {
    fn from(call: ScannedToolCall) -> Self {
        ToolCallOut {
            id: call.id,
            r#type: "function",
            function: FunctionOut {
                name: call.name,
                arguments: call.arguments,
            },
        }
    }
}

/// The usage figures (the prompt's templated-token count + the generated
/// tokens).
#[derive(Serialize, ToSchema)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// One SSE chunk (the `chat.completion.chunk` shape): a token delta or the
/// final `finish_reason` chunk (an empty `delta`).
#[derive(Serialize, ToSchema)]
struct Chunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize, ToSchema)]
struct ChunkChoice {
    index: u8,
    delta: Delta,
    finish_reason: Option<&'static str>,
    /// See the matching field on `CompletionChoice`: only ever on the chunk
    /// that carries `finish_reason`.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget_forced_at: Option<u32>,
}

/// The token delta. An empty `content`, absent `reasoning_content` and
/// absent `tool_calls` serialize to `{}` (OpenAI's final chunk shape).
#[derive(Serialize, Default, ToSchema)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    content: String,
    /// A tool call's id + name + full arguments (GitHub #121) — delivered
    /// as one complete delta per call rather than character-by-character
    /// (the source is a closed XML block, not a token-streamed JSON
    /// fragment, so there is nothing partial left to stream by the time a
    /// call is known at all).
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

/// One streamed tool call delta (OpenAI: `delta.tool_calls[]`), keyed by
/// `index` — the field a real agent client's reassembly buffer groups on,
/// stable across chunks and never reused by two different calls (GitHub
/// #121 acceptance criterion 2).
#[derive(Serialize, ToSchema)]
struct ToolCallDelta {
    index: usize,
    id: String,
    r#type: &'static str,
    function: FunctionOut,
}

impl From<ScannedToolCall> for ToolCallDelta {
    fn from(call: ScannedToolCall) -> Self {
        ToolCallDelta {
            index: call.index,
            id: call.id,
            r#type: "function",
            function: FunctionOut {
                name: call.name,
                arguments: call.arguments,
            },
        }
    }
}

/// The streaming half of a chat completion: the request's event stream
/// (`EventStream`) wrapped in the `chat.completion.chunk` SSE shape.
///
/// The chunk shape (OpenAI): each generated token is one
/// `data: {"choices":[{"delta":{"content":...}}]}` line; the final chunk
/// carries `finish_reason: "stop"` and an empty `delta`; a terminal
/// `data: [DONE]` line ends the stream.
struct ChunkStream {
    /// The request's event stream (closes on the request's completion).
    stream: EventStream,
    /// Aborts the request if this body is dropped before it completed.
    cancel: CancelOnDrop,
    id: String,
    created: u64,
    model: String,
    /// The incremental reasoning/content decoder (GitHub #68) — owns the
    /// byte- and marker-splitting state across this request's whole token
    /// stream (`crate::decoder::OutputDecoder`).
    decoder: OutputDecoder,
    /// Scans the decoder's `Content` channel for `<tool_call>` blocks
    /// (GitHub #121) — reasoning text never passes through it.
    tool_scanner: ToolCallScanner,
    prompt_tokens: u32,
    /// `stream_options.include_usage` — gates the trailing usage chunk
    /// (OpenAI only sends it when the client opts in).
    include_usage: bool,
    /// Fully-formed events queued ahead of the next stream poll: a single
    /// generated token can yield zero, one, or two decoder deltas (a
    /// reasoning tail plus a content head, right at the `</think>`
    /// marker), and the `Done` event queues its flush deltas, the
    /// `finish_reason` chunk, and (opt-in) the usage chunk together, in
    /// that order, ahead of `[DONE]`.
    pending: VecDeque<Event>,
    /// The `[DONE]` marker has been emitted (exactly once, at the end).
    done_sent: bool,
    /// At least one non-empty `Reasoning` delta was ever queued (GitHub
    /// #70 via #121 acceptance criterion 5).
    emitted_reasoning: bool,
    /// At least one `Content` delta or tool call was ever queued.
    emitted_content_or_call: bool,
}

/// Cancels its request when dropped before the request has completed. An SSE
/// client that hangs up mid-generation would otherwise leave its lane and KV
/// reservation generating to `max_tokens` for nobody.
pub(crate) struct CancelOnDrop {
    engine: Engine,
    request: RequestId,
    completed: bool,
}

impl CancelOnDrop {
    pub(crate) fn new(engine: Engine, request: RequestId) -> Self {
        Self { engine, request, completed: false }
    }

    /// The request reached its own terminal event: dropping is now a
    /// clean end of stream, not a disconnect.
    pub(crate) fn completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.completed {
            self.engine.cancel(self.request);
        }
    }
}

impl ChunkStream {
    // The chunk envelope's fields are all per-request and all needed to
    // shape a `chat.completion.chunk`; grouping them would only rename the
    // argument list.
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: EventStream,
        cancel: CancelOnDrop,
        id: String,
        created: u64,
        model: String,
        decoder: OutputDecoder,
        schemas: ToolSchemas,
        prompt_tokens: u32,
        include_usage: bool,
    ) -> Self {
        Self {
            stream,
            cancel,
            id,
            created,
            model,
            decoder,
            tool_scanner: ToolCallScanner::with_schemas(schemas),
            prompt_tokens,
            include_usage,
            pending: VecDeque::new(),
            done_sent: false,
            emitted_reasoning: false,
            emitted_content_or_call: false,
        }
    }

    /// Routes one decoder delta to its SSE chunk(s): a `Reasoning` delta
    /// streams straight through; a `Content` delta is fed to the tool-call
    /// scanner first, since it may resolve to plain content, a complete
    /// tool call, or nothing yet (GitHub #121).
    fn queue_decoder_delta(&mut self, delta: crate::decoder::Delta) {
        match delta.channel {
            Channel::Reasoning => {
                if !delta.text.is_empty() {
                    self.emitted_reasoning = true;
                }
                let event = self.delta_chunk(delta);
                self.pending.push_back(event);
            }
            Channel::Content => {
                for tool_event in self.tool_scanner.feed(&delta.text) {
                    self.queue_tool_event(tool_event);
                }
            }
        }
    }

    /// Queues one tool-call-scanner event as its SSE chunk, tracking
    /// `emitted_content_or_call` along the way — both `ToolEvent`
    /// variants only ever carry non-empty payloads (`ToolCallScanner`
    /// never emits an empty one), so every call here is real content.
    fn queue_tool_event(&mut self, event: ToolEvent) {
        self.emitted_content_or_call = true;
        let chunk = match event {
            ToolEvent::Content(text) => self.delta_chunk(crate::decoder::Delta {
                channel: Channel::Content,
                text,
            }),
            ToolEvent::Call(call) => self.tool_call_chunk(call),
        };
        self.pending.push_back(chunk);
    }

    /// One chunk (a decoded delta or the final `finish_reason` chunk).
    fn chunk(&self, delta: Delta, finish_reason: Option<&'static str>) -> Event {
        self.chunk_with(delta, finish_reason, None)
    }

    /// [`ChunkStream::chunk`], carrying a forced close's reasoning-token
    /// count — which only the finish chunk does.
    fn chunk_with(&self, delta: Delta, finish_reason: Option<&'static str>, thinking_budget_forced_at: Option<u32>) -> Event {
        let chunk = Chunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
                thinking_budget_forced_at,
            }],
            usage: None,
        };
        // `serde_json` cannot fail on this (all serializable fields).
        Event::default().data(serde_json::to_string(&chunk).expect("chunk serializes"))
    }

    /// One decoder delta as its SSE chunk (`delta.reasoning_content` or
    /// `delta.content`, never both — GitHub #68).
    fn delta_chunk(&self, delta: crate::decoder::Delta) -> Event {
        let delta = match delta.channel {
            Channel::Reasoning => Delta {
                reasoning_content: Some(delta.text),
                content: String::new(),
                tool_calls: None,
            },
            Channel::Content => Delta {
                reasoning_content: None,
                content: delta.text,
                tool_calls: None,
            },
        };
        self.chunk(delta, None)
    }

    /// One complete tool call as its SSE chunk (GitHub #121): a `delta`
    /// carrying only `tool_calls`, mirroring `delta_chunk`'s shape for the
    /// reasoning/content channels.
    fn tool_call_chunk(&self, call: ScannedToolCall) -> Event {
        self.chunk(
            Delta {
                reasoning_content: None,
                content: String::new(),
                tool_calls: Some(vec![call.into()]),
            },
            None,
        )
    }

    /// The final summary chunk: empty `choices`, populated `usage` (OpenAI's
    /// trailing usage chunk, sent right before `[DONE]`).
    fn usage_chunk(&self, completion_tokens: u32) -> Event {
        let chunk = Chunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![],
            usage: Some(Usage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens,
                total_tokens: self.prompt_tokens.saturating_add(completion_tokens),
            }),
        };
        Event::default().data(serde_json::to_string(&chunk).expect("chunk serializes"))
    }
}

/// The stream adapter: the request's event stream into the SSE shape.
///
/// Pulls one event per poll (the driver routes events into the stream as
/// they are generated — tokens arrive in generation order); an `Evicted` /
/// `Restored` event for the request is skipped (it does not change the
/// generated-token sequence). A single scheduler event can queue more than
/// one SSE chunk (or none at all, while the decoder is holding back a
/// multi-byte character or a possible `</think>` prefix) — queued chunks
/// drain before the stream is polled again.
impl Stream for ChunkStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut ChunkStream>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `ChunkStream` is `Unpin` (all fields are `Unpin`), so we can get a
        // plain `&mut` back out of the pinned self.
        let this = self.get_mut();
        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            match this.stream.poll_recv(cx) {
                Poll::Ready(None) => {
                    // Stream closed (the request completed and the engine
                    // removed its route — or the engine stopped): the
                    // `[DONE]` marker ends the SSE stream (exactly once).
                    if !this.done_sent {
                        this.done_sent = true;
                        return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(event)) => match event {
                    SchedEvent::Token { token, .. } => {
                        // The incremental decoder: zero, one, or two deltas
                        // for this token (GitHub #68). Zero means the
                        // decoder is holding back — loop to poll the next
                        // scheduler event rather than returning nothing.
                        for delta in this.decoder.push(&[token]) {
                            this.queue_decoder_delta(delta);
                        }
                    }
                    // The request completed: flush whatever the decoder
                    // and the tool-call scanner held back, then the
                    // finish-reason chunk, then (opt-in) the usage chunk —
                    // all queued ahead of `[DONE]`.
                    SchedEvent::Done { reason, tokens, thinking, .. } => {
                        this.cancel.completed();
                        for delta in this.decoder.finish() {
                            this.queue_decoder_delta(delta);
                        }
                        // Read before `finish()` clears it (GitHub #121
                        // acceptance criterion 3): a stop/cancel landing
                        // mid-call must never be reported as a clean
                        // `tool_calls` finish.
                        let mid_call = this.tool_scanner.is_mid_call();
                        for tool_event in this.tool_scanner.finish() {
                            this.queue_tool_event(tool_event);
                        }
                        let any_calls = this.tool_scanner.any_calls();
                        let finish_reason = resolve_finish_reason(reason, any_calls, mid_call);
                        report_if_all_reasoning_no_content(
                            &this.id,
                            this.emitted_reasoning && !this.emitted_content_or_call,
                            finish_reason,
                        );
                        let forced_at = thinking.and_then(|b| b.forced_at);
                        this.pending
                            .push_back(this.chunk_with(Delta::default(), Some(finish_reason), forced_at));
                        if this.include_usage {
                            this.pending.push_back(this.usage_chunk(tokens));
                        }
                    }
                    // Other events for this request (admissions,
                    // evictions, restorations, requeues) do not change the
                    // generated-token sequence — keep draining.
                    _ => {}
                },
            }
        }
    }
}

// ── POST /v1/responses ────────────────────────────────────────────────────

/// A responses-API request (OpenAI wire shape). `input` is a plain string
/// (a single user turn) or a list of messages.
#[derive(Deserialize, ToSchema)]
struct ResponsesRequest {
    input: ResponsesInput,
    model: Option<String>,
    /// The responses API's name for the completion's token cap.
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    seed: Option<u64>,
    /// `true` is rejected in v1 (the responses API is non-streaming; a
    // streaming response is a later ticket).
    #[serde(default)]
    stream: bool,
    /// The thinking controls (GitHub #68) — same wire contract as chat
    /// completions, resolved through the same `thinking::resolve` path.
    enable_thinking: Option<JsonValue>,
    reasoning_effort: Option<JsonValue>,
    preserve_thinking: Option<JsonValue>,
    chat_template_kwargs: Option<JsonValue>,
    /// See the matching field on `ChatCompletionsRequest`.
    thinking_budget: Option<JsonValue>,
    /// An ignis extension, not an OpenAI parameter (GitHub #120) — see the
    /// matching field on `ChatCompletionsRequest`.
    class: Option<JsonValue>,
}

/// The responses API's `input` (a string or a message list).
#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
enum ResponsesInput {
    /// A plain string — treated as a single user turn.
    Text(String),
    /// A list of messages (role + content).
    Messages(Vec<ChatMessage>),
}

/// `POST /v1/responses` — the OpenAI responses API (non-streaming in v1).
///
/// The request routes into the core scheduler exactly like a chat
/// completion (the same submit / event-stream path); the response is the
/// responses API's `output`-message shape with the generated text in an
/// `output_text` content part.
#[utoipa::path(
    post,
    path = "/v1/responses",
    tag = "responses",
    operation_id = "responses",
    summary = "A response, the responses-API shape",
    description = "The same engine path as a chat completion, under the responses API's names: `input` (a string or a message list) instead of `messages`, `max_output_tokens` instead of `max_tokens`, and an `output` of messages carrying `output_text` content parts.

Non-streaming in v1: `stream: true` is refused with a 400 rather than answered partially.",
    request_body = ResponsesRequest,
    responses(
        (status = 200, description = "The response.", body = Responses),
        (status = 400, description = "The request is malformed, or asked for `stream: true`, which this endpoint does not serve in v1.", body = ApiError),
        (status = 401, description = "The server was started with `--api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "The request named a model this server has not loaded.", body = ApiError),
        (status = 413, description = "The prompt is longer than this server's `--max-context`.", body = ApiError),
        (status = 503, description = "The engine is at capacity and the request was not admitted.", body = ApiError),
        (status = 504, description = "The engine did not finish the request within `--request-timeout`.", body = ApiError),
    ),
)]
async fn responses_api(
    State(server): State<Arc<Server>>,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    if req.stream {
        return bad_request("streaming responses are not supported in v1 (non-streaming only)");
    }
    // `input` → messages: a string is a single user turn; a message list is
    // used as-is (an empty list is a 400).
    let messages = match req.input {
        ResponsesInput::Text(text) => vec![ChatMessage::text("user", text)],
        ResponsesInput::Messages(m) => m,
    };
    if messages.is_empty() {
        return bad_request("input must not be empty");
    }
    if let Err(rejection) = check_roles(&messages) {
        return template_rejection(rejection);
    }
    if let Err(rejection) = check_content_parts(&messages, server.media.is_some()) {
        return content_rejection(rejection);
    }
    let thinking = match resolve_thinking(
        &server,
        ThinkingRequestFields {
            enable_thinking: req.enable_thinking.as_ref(),
            reasoning_effort: req.reasoning_effort.as_ref(),
            preserve_thinking: req.preserve_thinking.as_ref(),
            chat_template_kwargs: req.chat_template_kwargs.as_ref(),
        },
    ) {
        Ok(t) => t,
        Err(response) => return response,
    };
    let params = DecodeParams {
        max_tokens: req.max_output_tokens,
        temperature: req.temperature.unwrap_or(0.0),
        seed: req.seed.unwrap_or(0),
        ..DecodeParams::default()
    };
    let (params, budget_dropped) = match with_thinking_budget(
        &server,
        params,
        req.thinking_budget.as_ref(),
        req.reasoning_effort.as_ref(),
        &thinking,
    ) {
        Ok(resolved) => resolved,
        Err(response) => return response,
    };
    let (model, class) = match resolve_model_and_class(req.model, req.class) {
        Ok(x) => x,
        Err(message) => return bad_request(&message),
    };
    let prepared = prepare_request(
        &server,
        model,
        &messages,
        params,
        &thinking,
        // The responses API carries no `tools` field (GitHub #132: only
        // `/v1/chat/completions` gets tool-calling support, matching the
        // scope this endpoint already keeps for reasoning/tool_calls).
        &[],
    )
    .await;
    let (input, model, prompt_tokens, media) = match prepared {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    let notes = RequestNotes { media, thinking_budget_dropped: budget_dropped };
    let (id, mut stream) = match server.engine.submit_with_notes(input, class, notes).await {
        Ok(x) => x,
        Err(err) => return submit_error(&server, err),
    };
    // GitHub #81 / ADR 0012: see the matching comment in `chat_completions`.
    tracing::Span::current().record("request_id", id);
    match collect_completion(&mut stream, server.request_timeout).await {
        Ok(Completion { reason: FinishReason::Error, .. }) => engine_error_response(),
        Ok(Completion { tokens, thinking: budget, .. }) => {
            // The responses API's v1 shape carries no `finish_reason`
            // field (only `status: "completed"`); the stop reason is not
            // surfaced here.
            // GitHub #68: `text` is the content channel only — the
            // reasoning trace is discarded (this endpoint has no field to
            // carry it, and leaking it into `text` is the bug being fixed).
            let (_reasoning_content, text) =
                split_reasoning(server.template.as_ref(), &tokens, &thinking);
            let output_tokens = tokens.len() as u32;
            Json(Responses {
                id: format!("resp_{id}"),
                object: "response",
                created_at: now(),
                model,
                output: vec![ResponseMessage {
                    r#type: "message",
                    id: format!("msg_{id}"),
                    role: "assistant",
                    status: "completed",
                    content: vec![ResponseContent {
                        r#type: "output_text",
                        text,
                        annotations: Vec::new(),
                    }],
                }],
                status: "completed",
                thinking_budget_forced_at: budget.and_then(|b| b.forced_at),
                usage: ResponsesUsage {
                    input_tokens: prompt_tokens,
                    output_tokens,
                    total_tokens: prompt_tokens.saturating_add(output_tokens),
                },
            })
            .into_response()
        }
        Err(_) => error_response(
            StatusCode::GATEWAY_TIMEOUT,
            "request_timeout",
            "request_timeout",
            request_timeout_message(server.request_timeout),
        ),
    }
}

/// The non-streaming responses response (the OpenAI responses API shape).
#[derive(Serialize, ToSchema)]
struct Responses {
    id: String,
    object: &'static str,
    created_at: u64,
    model: String,
    output: Vec<ResponseMessage>,
    status: &'static str,
    /// See the matching field on `CompletionChoice`; top-level here, where
    /// this shape keeps its per-response state.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget_forced_at: Option<u32>,
    usage: ResponsesUsage,
}

#[derive(Serialize, ToSchema)]
struct ResponseMessage {
    r#type: &'static str,
    id: String,
    role: &'static str,
    status: &'static str,
    content: Vec<ResponseContent>,
}

#[derive(Serialize, ToSchema)]
struct ResponseContent {
    r#type: &'static str,
    text: String,
    /// Always empty: this server annotates nothing (the field is the
    /// responses API's, kept so the shape matches).
    #[schema(value_type = Vec<Object>)]
    annotations: Vec<()>,
}

#[derive(Serialize, ToSchema)]
struct ResponsesUsage {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

// ── the error envelope ───────────────────────────────────────────────────

/// The OpenAI error body (`{"error": {...}}`). Every failure on this
/// surface answers in this shape, `/v1/decide` included.
#[derive(Serialize, ToSchema)]
pub(crate) struct ApiError {
    error: ErrorBody,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ErrorBody {
    message: String,
    r#type: String,
    code: String,
    /// The request field at fault, when one field is (OpenAI's `param`).
    /// Absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    param: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunk_serializes_in_the_openai_shape() {
        // Pin the chunk shape (a token delta, no finish_reason): a client
        // that parses the SSE stream must see these exact fields.
        let c = Chunk {
            id: "chatcmpl-7".into(),
            object: "chat.completion.chunk",
            created: 123,
            model: "test-model".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    reasoning_content: None,
                    content: "hello".into(),
                    tool_calls: None,
                },
                finish_reason: None,
                thinking_budget_forced_at: None,
            }],
            usage: None,
        };
        let json = serde_json::to_value(&c).expect("chunk serializes");
        assert_eq!(json["id"], "chatcmpl-7");
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["choices"][0]["delta"]["content"], "hello");
        // OpenAI's token-delta chunks carry `finish_reason: null` (the key
        // is present, the value null) — match that shape.
        assert!(json["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn the_final_chunk_has_a_finish_reason_and_an_empty_delta() {
        // The terminal chunk: finish_reason set, the delta empty (an empty
        // `content` is omitted — OpenAI's final-chunk shape).
        let c = Chunk {
            id: "chatcmpl-7".into(),
            object: "chat.completion.chunk",
            created: 123,
            model: "test-model".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some("stop"),
                thinking_budget_forced_at: None,
            }],
            usage: None,
        };
        let json = serde_json::to_value(&c).expect("chunk serializes");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        // No forced close: the ignis extension is absent, not `null`.
        assert!(json["choices"][0].get("thinking_budget_forced_at").is_none());
        // An empty delta serializes to `{}` (the content key is omitted).
        assert_eq!(json["choices"][0]["delta"], serde_json::json!({}));
    }

    #[test]
    fn the_usage_chunk_has_empty_choices_and_populated_usage() {
        // The trailing summary chunk (sent right before `[DONE]`): empty
        // `choices`, `usage` populated with the token counts.
        let c = Chunk {
            id: "chatcmpl-7".into(),
            object: "chat.completion.chunk",
            created: 123,
            model: "test-model".into(),
            choices: vec![],
            usage: Some(Usage {
                prompt_tokens: 58,
                completion_tokens: 1500,
                total_tokens: 1558,
            }),
        };
        let json = serde_json::to_value(&c).expect("chunk serializes");
        assert_eq!(json["choices"], serde_json::json!([]));
        assert_eq!(json["usage"]["prompt_tokens"], 58);
        assert_eq!(json["usage"]["completion_tokens"], 1500);
        assert_eq!(json["usage"]["total_tokens"], 1558);
    }

    // ── GitHub #120: tagged lanes (the `class` extension + "@<lane>") ────

    #[test]
    fn a_lane_suffix_on_model_is_stripped_and_parsed() {
        let (model, class) = split_model_lane(Some("qwen2.5-7b@agent".into()));
        assert_eq!(model.as_deref(), Some("qwen2.5-7b"));
        assert_eq!(class, Some(RequestClass::Agent));
    }

    #[test]
    fn an_unrecognized_lane_suffix_still_strips_and_defaults_to_interactive() {
        let (model, class) = split_model_lane(Some("qwen2.5-7b@classifier".into()));
        assert_eq!(model.as_deref(), Some("qwen2.5-7b"));
        assert_eq!(class, Some(RequestClass::Interactive));
    }

    #[test]
    fn a_model_with_no_at_sign_carries_no_lane() {
        let (model, class) = split_model_lane(Some("qwen2.5-7b".into()));
        assert_eq!(model.as_deref(), Some("qwen2.5-7b"));
        assert_eq!(class, None);
    }

    #[test]
    fn a_bare_at_prefix_with_no_model_id_is_left_untouched() {
        // No model name precedes "@" — nothing sensible to strip, so the
        // whole string stands as the model (and is rejected downstream as
        // an unknown model, same as today).
        let (model, class) = split_model_lane(Some("@agent".into()));
        assert_eq!(model.as_deref(), Some("@agent"));
        assert_eq!(class, None);
    }

    #[test]
    fn no_model_at_all_carries_no_lane() {
        assert_eq!(split_model_lane(None), (None, None));
    }

    #[test]
    fn an_explicit_class_field_wins_over_the_model_suffix() {
        let class = resolve_class(
            Some(serde_json::json!("interactive")),
            Some(RequestClass::Agent),
        )
        .expect("a string class resolves");
        assert_eq!(class, RequestClass::Interactive);
    }

    #[test]
    fn the_model_suffix_applies_when_no_explicit_class_is_set() {
        let class = resolve_class(None, Some(RequestClass::Agent)).expect("resolves");
        assert_eq!(class, RequestClass::Agent);
    }

    #[test]
    fn absent_class_and_untagged_model_default_to_interactive() {
        assert_eq!(resolve_class(None, None).unwrap(), RequestClass::Interactive);
        assert_eq!(
            resolve_class(Some(serde_json::Value::Null), None).unwrap(),
            RequestClass::Interactive
        );
    }

    #[test]
    fn an_unrecognized_class_string_defaults_to_interactive_rather_than_erroring() {
        assert_eq!(
            resolve_class(Some(serde_json::json!("classifier")), None).unwrap(),
            RequestClass::Interactive
        );
    }

    #[test]
    fn a_non_string_class_is_refused_like_top_k() {
        let err = resolve_class(Some(serde_json::json!(1)), None).unwrap_err();
        assert!(err.contains("ignis extension"), "message: {err}");
    }

    #[test]
    fn resolve_model_and_class_combines_the_lane_suffix_and_the_explicit_field() {
        // No explicit `class`: the "@<lane>" suffix decides, and is
        // stripped from the model handed back.
        let (model, class) =
            resolve_model_and_class(Some("qwen2.5-7b@agent".into()), None).expect("resolves");
        assert_eq!(model.as_deref(), Some("qwen2.5-7b"));
        assert_eq!(class, RequestClass::Agent);

        // An explicit `class` field overrides a conflicting suffix.
        let (model, class) = resolve_model_and_class(
            Some("qwen2.5-7b@agent".into()),
            Some(serde_json::json!("interactive")),
        )
        .expect("resolves");
        assert_eq!(model.as_deref(), Some("qwen2.5-7b"));
        assert_eq!(class, RequestClass::Interactive);

        // A badly-typed `class` is still refused even with a valid suffix.
        assert!(
            resolve_model_and_class(Some("qwen2.5-7b@agent".into()), Some(serde_json::json!(1)))
                .is_err()
        );
    }

    #[test]
    fn a_chat_completions_request_parses_the_class_extension_field() {
        let req: ChatCompletionsRequest = serde_json::from_value(serde_json::json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "class": "agent",
        }))
        .expect("class parses as a chat-completions field");
        assert_eq!(req.class, Some(serde_json::json!("agent")));
    }

    #[test]
    fn a_responses_request_parses_the_class_extension_field() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "input": "hi",
            "class": "agent",
        }))
        .expect("class parses as a responses field");
        assert_eq!(req.class, Some(serde_json::json!("agent")));
    }

    #[test]
    fn the_error_body_is_the_openai_shape() {
        let body = ApiError {
            error: ErrorBody {
                message: "nope".into(),
                r#type: "invalid_request_error".into(),
                code: "invalid_request_error".into(),
                param: None,
            },
        };
        let json = serde_json::to_value(&body).expect("error serializes");
        assert_eq!(json["error"]["message"], "nope");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "invalid_request_error");
        // No field at fault, no `param` key at all.
        assert!(json["error"].get("param").is_none(), "{json}");
    }

    #[test]
    fn a_responses_input_parses_as_a_string_or_a_message_list() {
        // `input: "hi"` → a single user turn.
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "input": "hi"
        }))
        .expect("string input parses");
        match req.input {
            ResponsesInput::Text(s) => assert_eq!(s, "hi"),
            _ => panic!("a string input must parse as Text"),
        }
        // `input: [...]` → a message list.
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({ "input": [
                { "role": "user", "content": "hi" }
            ] }))
        .expect("message input parses");
        match req.input {
            ResponsesInput::Messages(m) => {
                assert_eq!(m.len(), 1);
                assert_eq!(m[0].role, "user");
            }
            _ => panic!("a message-list input must parse as Messages"),
        }
    }

    // ── GitHub #121: tool-call and finish-reason hardening ──────────────

    #[test]
    fn a_clean_stop_with_a_complete_call_reports_tool_calls() {
        assert_eq!(
            resolve_finish_reason(FinishReason::Stop, true, false),
            "tool_calls"
        );
    }

    #[test]
    fn a_clean_stop_with_no_call_reports_stop() {
        assert_eq!(resolve_finish_reason(FinishReason::Stop, false, false), "stop");
    }

    #[test]
    fn a_mid_call_stop_never_reports_tool_calls_even_though_one_was_seen() {
        // Acceptance criterion 3: a call left open when generation stopped
        // was dropped whole by `ToolCallScanner` — reporting `tool_calls`
        // here would tell the client to run a call it never received.
        assert_eq!(resolve_finish_reason(FinishReason::Stop, true, true), "stop");
    }

    #[test]
    fn length_is_reported_regardless_of_calls_seen() {
        assert_eq!(resolve_finish_reason(FinishReason::Length, true, false), "length");
        assert_eq!(resolve_finish_reason(FinishReason::Length, false, false), "length");
    }

    #[test]
    fn an_engine_error_is_reported_as_error_regardless_of_calls_seen() {
        // GitHub #166: a request the engine gave up on is never a clean stop.
        assert_eq!(resolve_finish_reason(FinishReason::Error, true, false), "error");
        assert_eq!(resolve_finish_reason(FinishReason::Error, false, false), "error");
    }

    #[test]
    fn a_streamed_tool_call_delta_serializes_in_the_openai_shape() {
        let delta = Delta {
            reasoning_content: None,
            content: String::new(),
            tool_calls: Some(vec![ToolCallDelta {
                index: 0,
                id: "call_0".into(),
                r#type: "function",
                function: FunctionOut {
                    name: "read_file".into(),
                    arguments: r#"{"path":"a.txt"}"#.into(),
                },
            }]),
        };
        let json = serde_json::to_value(&delta).expect("delta serializes");
        assert_eq!(json["tool_calls"][0]["index"], 0);
        assert_eq!(json["tool_calls"][0]["id"], "call_0");
        assert_eq!(json["tool_calls"][0]["type"], "function");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "read_file");
        // A delta with tool_calls but no text carries no `content` key
        // (same "empty things are omitted" rule as `reasoning_content`).
        assert!(json.get("content").is_none());
    }

    #[test]
    fn an_assistant_message_with_no_tool_calls_omits_the_field_entirely() {
        let message = AssistantMessage {
            role: "assistant",
            reasoning_content: None,
            content: "hi".into(),
            tool_calls: None,
        };
        let json = serde_json::to_value(&message).expect("message serializes");
        assert!(json.as_object().unwrap().get("tool_calls").is_none());
    }

    // ── GitHub #132: tools / tool_choice validation ─────────────────────

    /// A well-formed tool already carrying the fields `normalize_tool`
    /// would fill in.
    fn tool(name: &str) -> JsonValue {
        serde_json::json!({"type": "function", "function": {
            "name": name, "parameters": {"type": "object"}, "strict": true
        }})
    }

    #[test]
    fn a_tool_gets_the_references_default_parameters_and_strict() {
        let sent = serde_json::json!({"type": "function", "function": {"name": "a", "parameters": null}});
        assert_eq!(
            resolve_tools(Some(vec![sent]), None).unwrap(),
            vec![serde_json::json!({"type": "function", "function": {
                "name": "a", "parameters": {"type": "object", "properties": {}}, "strict": false
            }})]
        );
    }

    #[test]
    fn no_tools_and_no_tool_choice_resolves_to_empty() {
        assert_eq!(resolve_tools(None, None).unwrap(), Vec::<JsonValue>::new());
    }

    #[test]
    fn well_formed_tools_pass_through_unchanged() {
        let tools = vec![tool("a"), tool("b")];
        assert_eq!(resolve_tools(Some(tools.clone()), None).unwrap(), tools);
    }

    #[test]
    fn a_tool_missing_type_is_rejected() {
        let bad = serde_json::json!({"function": {"name": "a"}});
        let err = resolve_tools(Some(vec![bad]), None).unwrap_err();
        assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn a_tool_with_a_non_string_name_is_rejected() {
        let bad = serde_json::json!({"type": "function", "function": {"name": 5}});
        assert!(resolve_tools(Some(vec![bad]), None).is_err());
    }

    #[test]
    fn tool_choice_none_drops_validated_tools() {
        let tools = vec![tool("a")];
        let resolved = resolve_tools(Some(tools), Some(serde_json::json!("none"))).unwrap();
        assert!(resolved.is_empty());
    }

    #[test]
    fn tool_choice_auto_keeps_tools() {
        let tools = vec![tool("a")];
        let resolved =
            resolve_tools(Some(tools.clone()), Some(serde_json::json!("auto"))).unwrap();
        assert_eq!(resolved, tools);
    }

    #[test]
    fn tool_choice_required_is_rejected() {
        assert!(resolve_tools(None, Some(serde_json::json!("required"))).is_err());
    }

    #[test]
    fn tool_choice_naming_a_function_object_is_rejected() {
        let choice = serde_json::json!({"type": "function", "function": {"name": "a"}});
        assert!(resolve_tools(None, Some(choice)).is_err());
    }

    #[test]
    fn an_unknown_tool_choice_string_is_rejected() {
        assert!(resolve_tools(None, Some(serde_json::json!("sometimes"))).is_err());
    }
}
