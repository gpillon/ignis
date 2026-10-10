//! `POST /v1/tokenize` and `POST /v1/detokenize` — the prompt counted without
//! being served (GitHub #285, spec server/10).
//!
//! An agent decides what to send by what it costs, and until now the only way
//! to learn a body's prompt-token count was to send it and read
//! `usage.prompt_tokens` off the answer: a prefill, a lane and a generation
//! nobody wanted. These two routes run the request's **own render path** —
//! [`crate::api::render_prompt`], the one `/v1/chat/completions` renders
//! through — and stop before submitting, so the count is the number the same
//! body would report when served: the same template, tool block, thinking
//! controls and system block, not an estimate of them.
//!
//! Nothing here touches the scheduler, the GPU or a KV page: no
//! `RequestInput` is submitted, no sequence allocated, no retained state read
//! or published, no media acquired. An image is **refused**
//! (`media_not_countable`), not counted: its cost is its grid after
//! preparation, which the server could only learn by fetching, decoding and
//! resizing the picture — a network fetch on a route whose promise is that it
//! is free — and a number that tracked a resize policy that has changed twice
//! is worse than none.
//!
//! The field names are ours, not vLLM's: nobody has checked its current
//! shape against the source, and a half-matching shape is worse than an
//! honestly different one.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use utoipa::ToSchema;

use ignis_core::DecodeParams;

use crate::api::{
    bad_request, bad_request_param, content_rejection, error_response, error_response_naming, render_prompt, resolve_model_and_class,
    forced_tool_call, resolve_thinking, resolve_tools, template_rejection, with_thinking_budget, ApiError, Structure,
};
use crate::media::has_media;
use crate::metrics::TokenizeRoute;
use crate::template::{check_roles, ChatMessage};
use crate::thinking::ThinkingRequestFields;
use crate::Server;

/// A tokenize request: **either** a chat body **or** a raw prompt, never
/// both. Unknown fields are ignored, which is how the sampling fields
/// (`temperature`, `max_tokens`, ...) are inert: they do not change a render.
#[derive(Deserialize, ToSchema)]
pub(crate) struct TokenizeRequest {
    /// The chat form: the conversation, rendered exactly as
    /// `/v1/chat/completions` would render it. Must be non-empty, and must
    /// not be sent with `prompt`.
    messages: Option<Vec<ChatMessage>>,
    /// The raw form: a string tokenized with no chat template applied at all.
    /// Must not be sent with `messages`.
    prompt: Option<String>,
    /// The model the body names, with its `@<lane>` suffix read as chat reads
    /// it. A server serves one model; naming another is a `404`. Chat form
    /// only.
    model: Option<String>,
    /// Include the token ids. Default `false`.
    #[serde(default)]
    return_token_ids: bool,
    /// Include the rendered prompt text. Default `false`.
    #[serde(default)]
    return_text: bool,
    /// Refused when `true`: there is nothing to stream.
    #[serde(default)]
    stream: bool,
    // The fields that change a render, as raw JSON for the reason
    // `ChatCompletionsRequest` keeps them so: the wire contract's errors are
    // decided by the same resolution code, not by serde's type mismatch.
    enable_thinking: Option<JsonValue>,
    reasoning_effort: Option<JsonValue>,
    preserve_thinking: Option<JsonValue>,
    chat_template_kwargs: Option<JsonValue>,
    thinking_budget: Option<JsonValue>,
    tools: Option<Vec<JsonValue>>,
    tool_choice: Option<JsonValue>,
}

/// What a body costs before it is served.
#[derive(Serialize, ToSchema)]
pub(crate) struct TokenizeResponse {
    /// The prompt's length in tokens: the number the same body reports as
    /// `usage.prompt_tokens` when served.
    count: u32,
    /// This server's `--model-max-context`: the most a request may spend, prompt
    /// plus completion, so a client decides with one call.
    max_model_len: u32,
    /// The prompt's token ids; only with `return_token_ids`.
    #[serde(skip_serializing_if = "Option::is_none")]
    token_ids: Option<Vec<u32>>,
    /// The rendered prompt text; only with `return_text`.
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

/// `POST /v1/tokenize` — a prompt's token count, without serving it.
#[utoipa::path(
    post,
    path = "/v1/tokenize",
    tag = "tokenize",
    operation_id = "tokenize",
    summary = "A prompt's token count, without serving it",
    description = "Renders a chat body exactly as `/v1/chat/completions` would -- the chat template, the tool block, the thinking controls, the system block -- and answers how many tokens it prefills, without submitting it: no lane, no GPU, no KV page, and the count is the `usage.prompt_tokens` the same body reports when served. `max_model_len` is the server's `--model-max-context`, so one call decides whether a body fits.

Send **either** `messages` (with `model`, `tools`, `tool_choice` and the thinking controls, which are validated as chat validates them) **or** a raw `prompt`, which is tokenized with no template applied. Sampling fields are ignored: they do not change a render. `stream` is refused.

`return_token_ids` adds the ids and `return_text` the rendered text.

A body carrying an `image_url` part is refused with `media_not_countable`: an image's cost is its grid after preparation, which the server can only learn by fetching and decoding it, and this route does neither. Send the request to learn it.",
    request_body = TokenizeRequest,
    responses(
        (status = 200, description = "The count, and the ceiling it is to be compared with.", body = TokenizeResponse),
        (status = 400, description = "Both `messages` and `prompt`, or neither; an empty `messages`; a body chat would refuse; `stream: true`; or an image part (`media_not_countable`).", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "The body named a model this server has not loaded.", body = ApiError),
        (status = 501, description = "The raw form on a load whose tokenizer cannot be asked for ids (`tokenizer_unavailable`).", body = ApiError),
    ),
)]
pub(crate) async fn tokenize(State(server): State<Arc<Server>>, Json(req): Json<TokenizeRequest>) -> Response {
    // One model for the whole request (spec model-switch/01).
    let server = Arc::new(server.pinned());
    if let Some(metrics) = &server.metrics {
        metrics.record_tokenize(TokenizeRoute::Tokenize);
    }
    if req.stream {
        return bad_request_param("stream is not supported here: there is nothing to stream", "stream");
    }
    let (return_ids, return_text) = (req.return_token_ids, req.return_text);
    let (ids, text) = match (req.messages.as_ref(), req.prompt.as_ref()) {
        (Some(_), Some(_)) => {
            return bad_request("send either `messages` or `prompt`, not both");
        }
        (None, None) => {
            return bad_request("send `messages` (a chat body) or `prompt` (a raw string) to count");
        }
        (None, Some(_)) => {
            let prompt = req.prompt.expect("matched Some");
            let text = return_text.then(|| prompt.clone());
            match raw(&server, prompt).await {
                Ok(ids) => (ids, text),
                Err(response) => return response,
            }
        }
        (Some(_), None) => match chat(&server, req).await {
            Ok(rendered) => rendered,
            Err(response) => return response,
        },
    };
    Json(TokenizeResponse {
        count: ids.len() as u32,
        max_model_len: server.active().engine.max_model_len(),
        text,
        token_ids: return_ids.then_some(ids),
    })
    .into_response()
}

/// The raw form: `prompt` through the tokenizer with no template around it.
async fn raw(server: &Arc<Server>, prompt: String) -> Result<Vec<u32>, Response> {
    let template = Arc::clone(&server.active().template);
    let ids = blocking(move || template.encode_literal(&prompt)).await?;
    ids.ok_or_else(|| {
        error_response(
            StatusCode::NOT_IMPLEMENTED,
            "invalid_request_error",
            "tokenizer_unavailable",
            "this server's template provider has no tokenizer to ask for ids; count a chat body with `messages` instead",
        )
    })
}

/// The chat form: the body validated by the code `/v1/chat/completions` uses,
/// then rendered by the path it renders through, and no further.
///
/// Returns the prompt's ids and, when the body asked for it, its text.
async fn chat(server: &Arc<Server>, req: TokenizeRequest) -> Result<(Vec<u32>, Option<String>), Response> {
    let messages: Vec<ChatMessage> = req.messages.unwrap_or_default();
    if messages.is_empty() {
        return Err(bad_request("messages must not be empty"));
    }
    check_roles(&messages).map_err(template_rejection)?;
    // A model with no vision tower refuses an image for that, naming itself
    // (spec flash-next/04), before this route's own reason below.
    if !server.active().family.takes_images() {
        server.check_content_parts(&messages).map_err(content_rejection)?;
    }
    // Before the content-part check, which on a load without `--vision-enabled`
    // would answer `vision_disabled`: the refusal that matters here is that
    // this route never counts an image, whatever the load can do with one.
    if has_media(&messages) {
        return Err(error_response_naming(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "media_not_countable",
            "an image's token cost is its grid after preparation, which this route would have to fetch, decode and resize the picture to learn; \
             it counts text only; send the request to /v1/chat/completions to learn what an image costs",
            Some("messages"),
        ));
    }
    server.check_content_parts(&messages).map_err(content_rejection)?;
    let thinking = resolve_thinking(
        server,
        ThinkingRequestFields {
            enable_thinking: req.enable_thinking.as_ref(),
            reasoning_effort: req.reasoning_effort.as_ref(),
            preserve_thinking: req.preserve_thinking.as_ref(),
            chat_template_kwargs: req.chat_template_kwargs.as_ref(),
        },
    )?;
    // The budget changes no render, but a body chat would refuse for it is
    // refused here for the same reason.
    with_thinking_budget(
        server,
        DecodeParams::default(),
        req.thinking_budget.as_ref(),
        req.reasoning_effort.as_ref(),
        &thinking,
    )?;
    let (tools, tool_choice) = resolve_tools(req.tools, req.tool_choice)?;
    // A forced call changes no render either, but one chat could not force
    // is refused here as it is there. The cap is a sampling field, inert.
    forced_tool_call(server, &tool_choice, &thinking, None)?;
    let (model, _) = resolve_model_and_class(req.model, None).map_err(|message| bad_request(&message))?;
    if let Some(model) = model.filter(|model| !model.is_empty() && *model != server.active().engine.model_id()) {
        return Err(error_response(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "model_not_found",
            format!("unknown model: {model} (loaded: {})", server.active().engine.model_id()),
        ));
    }
    let structure = if req.return_text { Structure::Text } else { Structure::Tokens };
    let server = Arc::clone(server);
    let prepared = blocking(move || render_prompt(&server, &messages, &thinking, &tools, structure)).await??;
    Ok((prepared.input.tokens, prepared.text.map(|text| text.text)))
}

/// Runs CPU work proportional to a prompt on the blocking pool, so a
/// 100K-token body cannot stall the async runtime.
async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<T, Response> {
    tokio::task::spawn_blocking(work).await.map_err(|err| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "internal_error",
            format!("the render did not finish: {err}"),
        )
    })
}

/// A detokenize request.
#[derive(Deserialize, ToSchema)]
pub(crate) struct DetokenizeRequest {
    /// Token ids of this server's vocabulary. An empty array answers an empty
    /// string.
    token_ids: Vec<u32>,
}

/// The text a run of token ids spells.
#[derive(Serialize, ToSchema)]
pub(crate) struct DetokenizeResponse {
    text: String,
}

/// `POST /v1/detokenize` — token ids back to text.
#[utoipa::path(
    post,
    path = "/v1/detokenize",
    tag = "tokenize",
    operation_id = "detokenize",
    summary = "Token ids back to text",
    description = "Decodes `token_ids` through the server's own tokenizer, the same one `/v1/tokenize` counts with: the ids `/v1/tokenize` returned come back as the text it rendered, byte for byte. An id outside the vocabulary is a `400` naming the first offending index; an empty array answers an empty string.",
    request_body = DetokenizeRequest,
    responses(
        (status = 200, description = "The decoded text.", body = DetokenizeResponse),
        (status = 400, description = "An id outside the vocabulary; the message names the first one's index.", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the request carried no matching bearer token.", body = ApiError),
    ),
)]
pub(crate) async fn detokenize(State(server): State<Arc<Server>>, Json(req): Json<DetokenizeRequest>) -> Response {
    // One model for the whole request (spec model-switch/01).
    let server = Arc::new(server.pinned());
    if let Some(metrics) = &server.metrics {
        metrics.record_tokenize(TokenizeRoute::Detokenize);
    }
    let template = Arc::clone(&server.active().template);
    let decoded = blocking(move || {
        if let Some(at) = req.token_ids.iter().position(|&id| !template.is_token(id)) {
            return Err(at_index(at, req.token_ids[at]));
        }
        Ok(template.render_tokens(&req.token_ids))
    })
    .await;
    match decoded {
        Ok(Ok(text)) => Json(DetokenizeResponse { text }).into_response(),
        Ok(Err(response)) | Err(response) => response,
    }
}

/// The `400` for the first id outside the vocabulary.
fn at_index(at: usize, id: u32) -> Response {
    bad_request_param(&format!("token_ids[{at}] is {id}, which is not a token of this server's vocabulary"), "token_ids")
}
