//! A Responses request, from its wire body to a submittable request (GitHub
//! #282, spec responses-api/01): the fields, the 400s, and the mapping of
//! input items onto the chat message path, so a Responses conversation
//! renders exactly the prompt the same conversation renders through chat
//! completions — and ADR 0029's content-matched reuse hits across both.

use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use utoipa::ToSchema;

use axum::response::Response;
use ignis_core::{RequestClass, RequestInput};

use crate::api::{self, SamplingRequestFields};
use crate::decoder::OutputDecoder;
use crate::engine::RequestNotes;
use crate::responses::events::{wire_number, ResponseEvents, ResponseObject};
use crate::template::{check_roles, ChatMessage, FunctionIn, MessageContent, ToolCallIn};
use crate::thinking::ThinkingRequestFields;
use crate::toolcall::ToolSchemas;
use crate::Server;

/// A Responses request: the `POST /v1/responses` body, and the body of a
/// socket's `response.create` beside its `type`, `stream_id` and `generate`.
/// Every other field OpenAI documents is inert or refused by value
/// (`crate::openai_fields`, spec server/09) — `include`, `prompt_cache_key`,
/// `client_metadata` and the rest that do not change the answer are
/// accepted — and a field outside that list is ignored.
#[derive(Clone, Default, Deserialize, ToSchema)]
pub(crate) struct CreateResponse {
    model: Option<String>,
    /// A string (one user message) or a list of input items: `message`
    /// (`role` `user`, `system`, `developer` or `assistant`; `content` a
    /// string or `input_text` / `input_image` / `output_text` parts),
    /// `function_call`, `function_call_output` and `reasoning`.
    #[schema(value_type = Object)]
    pub(crate) input: Option<JsonValue>,
    /// The system prompt, placed first under the server's instruction
    /// policies. Never inherited from a previous response.
    instructions: Option<JsonValue>,
    /// Function tools, in the Responses shape `{type: "function", name,
    /// description, parameters, strict}`. Any other tool type is a 400.
    #[schema(value_type = Vec<Object>)]
    tools: Option<Vec<JsonValue>>,
    /// `"auto"` (default), `"none"`, `"required"`, or `{"type": "function",
    /// "name": ...}` naming one of `tools`, forced as on chat completions.
    #[schema(value_type = Object)]
    tool_choice: Option<JsonValue>,
    /// Accepted; calls come back as the model writes them.
    parallel_tool_calls: Option<bool>,
    max_output_tokens: Option<u32>,
    /// An ignis extension: keep decoding past EOS. Requires
    /// `max_output_tokens`.
    #[serde(default)]
    ignore_eos: bool,
    #[serde(flatten)]
    sampling: SamplingRequestFields,
    /// `{effort, summary}`: `effort` resolves as `reasoning_effort` does;
    /// `summary` is accepted and produces nothing.
    #[schema(value_type = Object)]
    reasoning: Option<JsonValue>,
    /// `{format: {type: "text"}, verbosity}`: only the `text` format is
    /// served (structured output is not); `verbosity` is ignored.
    #[schema(value_type = Object)]
    text: Option<JsonValue>,
    /// `true` is a 400: ignis serves no background responses.
    #[schema(value_type = bool)]
    background: Option<JsonValue>,
    /// Over HTTP always a 400 `previous_response_not_found`: ignis stores no
    /// responses. The WebSocket mode's connection-local cache resolves it.
    #[schema(value_type = String)]
    pub(crate) previous_response_id: Option<JsonValue>,
    /// `true` answers `text/event-stream`. Ignored on the socket.
    #[schema(value_type = bool)]
    pub(crate) stream: Option<JsonValue>,
    /// The WebSocket mode's warm-up switch: `false` prefills the prompt and
    /// generates nothing. Over HTTP a 400.
    #[schema(value_type = bool)]
    pub(crate) generate: Option<JsonValue>,
    #[schema(value_type = Object)]
    metadata: Option<JsonValue>,
    #[schema(value_type = bool)]
    store: Option<JsonValue>,
    /// The thinking controls, the same ignis extensions as on chat
    /// completions.
    #[schema(value_type = Object)]
    enable_thinking: Option<JsonValue>,
    #[schema(value_type = Object)]
    reasoning_effort: Option<JsonValue>,
    #[schema(value_type = Object)]
    preserve_thinking: Option<JsonValue>,
    #[schema(value_type = Object)]
    chat_template_kwargs: Option<JsonValue>,
    #[schema(value_type = Object)]
    thinking_budget: Option<JsonValue>,
    /// The Lane tag, an ignis extension (see chat completions).
    #[schema(value_type = String)]
    class: Option<JsonValue>,
    /// Every field not named above — after `sampling` took its own — for
    /// `crate::openai_fields` to classify. The last flattened field, so it
    /// takes only what the others left.
    #[serde(flatten)]
    #[schema(ignore)]
    other: serde_json::Map<String, JsonValue>,
}

/// The user turn a system-only warm-up is rendered with, and cut from.
const WARM_UP_PLACEHOLDER: &str = ".";

/// A request ready to submit, and everything its response needs.
pub(crate) struct Prepared {
    pub input: RequestInput,
    pub class: RequestClass,
    pub notes: RequestNotes,
    /// The conversation's items, the previous response's history first:
    /// what a continuation of this response starts from, before its output.
    pub items: Vec<JsonValue>,
    pub start: ResponseStart,
}

/// A response before it has an id: what its [`ResponseEvents`] start from.
pub(crate) struct ResponseStart {
    response: ResponseObject,
    decoder: OutputDecoder,
    schemas: ToolSchemas,
    prompt_tokens: u32,
}

impl ResponseStart {
    /// The event producer for this response, now that it is `resp_<suffix>`.
    pub(crate) fn events(self, suffix: String, created_at: u64) -> ResponseEvents {
        let ResponseStart { mut response, decoder, schemas, prompt_tokens } = self;
        response.id = format!("resp_{suffix}");
        response.created_at = created_at;
        ResponseEvents::new(response, suffix, decoder, schemas, prompt_tokens)
    }
}

/// The 400s only HTTP answers: `previous_response_id` (ignis stores no
/// responses, so every id is unknown there — OpenAI's answer to an id it does
/// not have) and `generate` (a WebSocket-mode field; ignoring `false` would
/// generate).
pub(crate) fn http_refusal(req: &CreateResponse) -> Option<Response> {
    if let Some(id) = req.previous_response_id.as_ref().filter(|id| !id.is_null()) {
        return Some(previous_response_not_found(id));
    }
    if req.generate.as_ref().is_some_and(|g| !g.is_null()) {
        return Some(api::bad_request_param(
            "generate is a field of the WebSocket mode's response.create; over HTTP every response generates",
            "generate",
        ));
    }
    None
}

/// OpenAI's `previous_response_not_found`.
pub(crate) fn previous_response_not_found(id: &JsonValue) -> Response {
    let id = id.as_str().map(str::to_owned).unwrap_or_else(|| id.to_string());
    api::error_response_naming(
        axum::http::StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "previous_response_not_found",
        format!("Previous response with id '{id}' not found."),
        Some("previous_response_id"),
    )
}

/// Validate `req`, map its items (after `history`, a continued response's
/// conversation) onto chat messages, and render and prepare the request as
/// chat completions would. `warm_up` submits it as a warm-up request.
pub(crate) async fn prepare(
    server: &Server,
    req: CreateResponse,
    history: Vec<JsonValue>,
    warm_up: bool,
) -> Result<Prepared, Response> {
    crate::openai_fields::check(&req.other, crate::openai_fields::Surface::Responses)
        .map_err(api::refused_field)?;
    if let Some(text) = &req.text {
        let format = text.get("format").and_then(|f| f.get("type"));
        if format.is_some_and(|f| f != "text") {
            return Err(api::bad_request_param(
                &format!("text.format {} is not served: this server returns plain text only (no structured output)", format.unwrap()),
                "text.format",
            ));
        }
    }
    if req.background.as_ref().is_some_and(|b| b == &JsonValue::Bool(true)) {
        return Err(api::bad_request_param("background responses are not supported", "background"));
    }
    let instructions = api::optional_str(req.instructions.as_ref(), "instructions")
        .map_err(|message| api::bad_request_param(&message, "instructions"))?
        .map(str::to_owned);
    let tools = chat_tools(req.tools.as_deref().unwrap_or_default())?;
    let (tools, tool_choice) = api::resolve_tools(Some(tools), req.tool_choice.clone().map(chat_tool_choice))?;

    let mut items = history;
    items.extend(input_items(req.input.clone())?);
    let mut messages = messages(&items)?;
    if let Some(instructions) = &instructions {
        messages.insert(0, ChatMessage::text("system", instructions.clone()));
    }
    // A warm-up with no user message — Codex's prewarm, `instructions` and
    // `tools` alone — keeps the system block the next turn will share. A
    // chat template may refuse to render a conversation without a user query
    // (the Qwen template does), so it is rendered with a placeholder turn and
    // cut back to that block below; an ordinary request renders as sent.
    let system_only = warm_up && !messages.iter().any(|m| m.role == "user");
    if system_only {
        messages.push(ChatMessage::text("user", WARM_UP_PLACEHOLDER));
    }
    if messages.is_empty() {
        return Err(api::bad_request_param("input must not be empty", "input"));
    }
    check_roles(&messages).map_err(api::template_rejection)?;
    server.check_content_parts(&messages).map_err(api::content_rejection)?;

    let effort = req.reasoning_effort.clone().or_else(|| {
        req.reasoning.as_ref().and_then(|r| r.get("effort")).filter(|e| !e.is_null()).cloned()
    });
    let thinking = api::resolve_thinking(
        server,
        ThinkingRequestFields {
            enable_thinking: req.enable_thinking.as_ref(),
            reasoning_effort: effort.as_ref(),
            preserve_thinking: req.preserve_thinking.as_ref(),
            chat_template_kwargs: req.chat_template_kwargs.as_ref(),
        },
    )?;
    let params = req
        .sampling
        .clone()
        .resolve(req.max_output_tokens, req.ignore_eos, thinking.enable_thinking, server.seedless_seed)
        .map_err(api::invalid_sampling_parameter)?;
    let (params, budget_dropped) =
        api::with_thinking_budget(server, params, req.thinking_budget.as_ref(), effort.as_ref(), &thinking)?;
    let cap = req.max_output_tokens.map(|cap| (cap, "max_output_tokens"));
    let forced = api::forced_tool_call(server, &tool_choice, &thinking, cap)?;
    let (model, class) = api::resolve_model_and_class(req.model.clone(), req.class.clone())
        .map_err(|message| api::bad_request(&message))?;
    let schemas = ToolSchemas::from_tools(&tools);
    let (mut input, model, mut prompt_tokens, media) =
        api::prepare_request(server, model, &messages, params, &thinking, &tools).await?;
    input.warm_up = warm_up;
    // A warm-up generates nothing, so it has no call to force.
    input.forced_literal = forced.filter(|_| !warm_up);
    if system_only {
        let Some(block) = input.system_block_tokens.filter(|_| input.multimodal.is_none()) else {
            return Err(api::template_rejection(crate::template::TemplateRejection {
                code: "render_failed",
                message: "a warm-up with no user message keeps its system block, and this prompt has none to keep".into(),
            }));
        };
        // The prompt is the system block and nothing else: it publishes the
        // retained prefix there, and takes no checkpoint at an opener the
        // placeholder put in.
        input.tokens.truncate(block as usize);
        input.opener_tokens = None;
        input.user_turn_tokens = None;
        prompt_tokens = block;
    }

    let starts_in_reasoning = server.active().template.decoder_starts_in_reasoning(&thinking);
    let response = ResponseObject {
        id: String::new(),
        object: "response",
        created_at: 0,
        status: "in_progress",
        background: false,
        error: None,
        incomplete_details: None,
        instructions,
        max_output_tokens: req.max_output_tokens,
        model,
        output: Vec::new(),
        parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
        previous_response_id: req.previous_response_id.as_ref().and_then(|id| id.as_str()).map(str::to_owned),
        reasoning: json!({ "effort": effort, "summary": null }),
        store: req.store.as_ref().and_then(JsonValue::as_bool).unwrap_or(true),
        temperature: wire_number(params.temperature),
        text: json!({ "format": { "type": "text" } }),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| json!("auto")),
        tools: req.tools.clone().unwrap_or_default(),
        top_p: wire_number(params.top_p),
        truncation: "disabled",
        usage: None,
        metadata: req.metadata.clone().unwrap_or_else(|| json!({})),
        thinking_budget_forced_at: None,
    };
    Ok(Prepared {
        input,
        class,
        notes: RequestNotes { media, thinking_budget_dropped: budget_dropped, ..RequestNotes::default() },
        items,
        start: ResponseStart {
            response,
            decoder: OutputDecoder::new(server.active().template.token_decoder(), starts_in_reasoning),
            schemas,
            prompt_tokens,
        },
    })
}

/// The Responses `tools` in the shape chat completions takes (spec
/// server/07): `{type: "function", name, ...}` becomes `{type: "function",
/// function: {name, ...}}`. A tool already in the chat shape passes as it is.
/// Any other type is a hosted tool this server does not run: a 400 naming it.
fn chat_tools(tools: &[JsonValue]) -> Result<Vec<JsonValue>, Response> {
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let kind = tool.get("type").and_then(JsonValue::as_str);
            if kind != Some("function") {
                let kind = kind.map_or_else(|| "a tool without a type".to_owned(), |k| format!("'{k}'"));
                return Err(api::bad_request_param(
                    &format!("tools[{index}]: {kind} is not served; this server runs no hosted tools, only function tools the client executes"),
                    &format!("tools[{index}].type"),
                ));
            }
            if tool.get("function").is_some() {
                return Ok(tool.clone());
            }
            let mut function = tool.as_object().cloned().unwrap_or_default();
            function.remove("type");
            if !function.get("name").and_then(JsonValue::as_str).is_some_and(|n| !n.is_empty()) {
                return Err(api::bad_request_param(
                    &format!("tools[{index}] must name its function"),
                    &format!("tools[{index}].name"),
                ));
            }
            Ok(json!({ "type": "function", "function": function }))
        })
        .collect()
}

/// The Responses `tool_choice` in the shape chat completions takes: a named
/// function's `{"type": "function", "name": N}` becomes `{"type":
/// "function", "function": {"name": N}}` (GitHub #286). Every other value is
/// left for chat's validation, the chat shape included, as [`chat_tools`]
/// leaves a tool already in it.
fn chat_tool_choice(tool_choice: JsonValue) -> JsonValue {
    match tool_choice.get("name") {
        Some(name) if tool_choice.get("type") == Some(&json!("function")) && tool_choice.get("function").is_none() => {
            json!({ "type": "function", "function": { "name": name } })
        }
        _ => tool_choice,
    }
}

/// `input` as a list of items: a string is one user message.
fn input_items(input: Option<JsonValue>) -> Result<Vec<JsonValue>, Response> {
    match input {
        None | Some(JsonValue::Null) => Ok(Vec::new()),
        Some(JsonValue::String(text)) => Ok(vec![json!({ "type": "message", "role": "user", "content": text })]),
        Some(JsonValue::Array(items)) => Ok(items),
        Some(_) => Err(api::bad_request_param("input must be a string or a list of input items", "input")),
    }
}

/// The input items as chat messages.
///
/// The items of one assistant turn — its `reasoning`, its `message` and its
/// `function_call`s, in whatever order a client lists them — become one
/// assistant message, as the turn is one message on chat completions:
/// reasoning as its `reasoning_content`, text as its `content`, calls as its
/// `tool_calls`. A turn ends at any other item, and a second `reasoning`
/// starts the next one, as does a second assistant `message` — unless a call
/// came between them, which is how a response lists text written after a
/// call. `function_call_output` is a tool message.
pub(crate) fn messages(items: &[JsonValue]) -> Result<Vec<ChatMessage>, Response> {
    let mut messages = Vec::new();
    let mut turn: Option<Turn> = None;
    for (index, item) in items.iter().enumerate() {
        let at = |field: &str| format!("input[{index}]{field}");
        let Some(object) = item.as_object() else {
            return Err(api::bad_request_param("an input item must be an object", &at("")));
        };
        let kind = match object.get("type") {
            None => "message",
            Some(JsonValue::String(kind)) => kind.as_str(),
            Some(_) => return Err(api::bad_request_param("an input item's type must be a string", &at(".type"))),
        };
        match kind {
            "message" if object.get("role").and_then(JsonValue::as_str) == Some("assistant") => {
                let message = chat_message(item, index)?;
                let current = turn.get_or_insert_with(Turn::default);
                // Text the model wrote after a call is a second message item
                // of the same response, and the same turn's `content` on chat
                // completions: joined, as that content joins it.
                if current.has_message && current.message.tool_calls.is_some() {
                    let joined = current.message.content.text() + &message.content.text();
                    current.message.content = MessageContent::Text(joined);
                    continue;
                }
                if current.has_message {
                    messages.push(turn.take().expect("open").message);
                }
                let current = turn.get_or_insert_with(Turn::default);
                current.has_message = true;
                current.message.content = message.content;
                if message.reasoning_content.is_some() {
                    current.message.reasoning_content = message.reasoning_content;
                }
                if let Some(calls) = message.tool_calls {
                    current.message.tool_calls.get_or_insert_with(Vec::new).extend(calls);
                }
            }
            "message" => {
                if let Some(open) = turn.take() {
                    messages.push(open.message);
                }
                messages.push(chat_message(item, index)?);
            }
            "reasoning" => {
                let text = reasoning_text(item);
                if text.is_empty() {
                    // Only a summary or an encrypted blob: nothing this
                    // server produced, and nothing it can render.
                    continue;
                }
                if turn.as_ref().is_some_and(Turn::started) {
                    messages.push(turn.take().expect("open").message);
                }
                turn.get_or_insert_with(Turn::default).message.reasoning_content = Some(text);
            }
            "function_call" => {
                let field = |name: &str| {
                    object
                        .get(name)
                        .and_then(JsonValue::as_str)
                        .map(str::to_owned)
                        .ok_or_else(|| api::bad_request_param(&format!("a function_call item needs a string {name}"), &at(&format!(".{name}"))))
                };
                let call = ToolCallIn {
                    id: Some(field("call_id")?),
                    function: FunctionIn { name: field("name")?, arguments: field("arguments")? },
                };
                turn.get_or_insert_with(Turn::default).message.tool_calls.get_or_insert_with(Vec::new).push(call);
            }
            "function_call_output" => {
                if let Some(open) = turn.take() {
                    messages.push(open.message);
                }
                let call_id = object.get("call_id").and_then(JsonValue::as_str).ok_or_else(|| {
                    api::bad_request_param("a function_call_output item needs a string call_id", &at(".call_id"))
                })?;
                let output = match object.get("output") {
                    Some(JsonValue::String(text)) => json!(text),
                    Some(JsonValue::Array(parts)) => json!(parts.iter().map(chat_part).collect::<Vec<_>>()),
                    _ => {
                        return Err(api::bad_request_param(
                            "a function_call_output item needs an output string",
                            &at(".output"),
                        ))
                    }
                };
                let message = json!({ "role": "tool", "content": output, "tool_call_id": call_id });
                messages.push(serde_json::from_value(message).map_err(|e| {
                    api::bad_request_param(&format!("input[{index}]: {e}"), &at(""))
                })?);
            }
            other => {
                return Err(api::bad_request_param(
                    &format!("input[{index}]: '{other}' items are not served (message, function_call, function_call_output and reasoning are)"),
                    &at(".type"),
                ))
            }
        }
    }
    if let Some(open) = turn {
        messages.push(open.message);
    }
    Ok(messages)
}

/// The assistant turn being assembled from its items.
struct Turn {
    message: ChatMessage,
    has_message: bool,
}

impl Default for Turn {
    fn default() -> Self {
        Self { message: ChatMessage::text("assistant", ""), has_message: false }
    }
}

impl Turn {
    /// Whether anything but reasoning is already in it — past which a
    /// `reasoning` item belongs to the next turn.
    fn started(&self) -> bool {
        self.has_message || self.message.tool_calls.is_some() || self.message.reasoning_content.is_some()
    }
}

/// A `message` item as a chat message: its content parts in the chat shape,
/// every other field as chat completions reads it.
///
/// An assistant's `output_text` parts are its text, which chat completions
/// carries as a plain string: that is the form they take here, so the same
/// turn is the same message on both endpoints.
fn chat_message(item: &JsonValue, index: usize) -> Result<ChatMessage, Response> {
    let mut message = item.clone();
    let object = message.as_object_mut().expect("checked by the caller");
    object.remove("type");
    if let Some(JsonValue::Array(parts)) = object.get("content") {
        let parts: Vec<JsonValue> = parts.iter().map(chat_part).collect();
        object.insert("content".to_owned(), JsonValue::Array(parts));
    }
    let mut message: ChatMessage = serde_json::from_value(message).map_err(|e| {
        api::bad_request_param(&format!("input[{index}] is not a message: {e}"), &format!("input[{index}]"))
    })?;
    if let MessageContent::Parts(parts) = &message.content {
        let text_only = !parts.is_empty() && parts.iter().all(|p| p.kind.as_deref() == Some("text") && p.text.is_some());
        if message.role == "assistant" && text_only {
            message.content = MessageContent::Text(message.content.text());
        }
    }
    Ok(message)
}

/// One content part in the chat shape: `input_text` and `output_text` are
/// `text`, and `input_image`'s flat `image_url` string is the nested
/// `image_url` object chat completions takes. Anything else passes as it is,
/// for the chat path's own checks to accept or refuse by name.
fn chat_part(part: &JsonValue) -> JsonValue {
    match part.get("type").and_then(JsonValue::as_str) {
        Some("input_text" | "output_text") => json!({ "type": "text", "text": part.get("text") }),
        Some("input_image") => {
            let url = match part.get("image_url") {
                Some(JsonValue::Object(nested)) => nested.get("url").cloned(),
                other => other.cloned(),
            };
            json!({ "type": "image_url", "image_url": { "url": url } })
        }
        _ => part.clone(),
    }
}

/// A `reasoning` item's text: its `reasoning_text` parts, in order.
fn reasoning_text(item: &JsonValue) -> String {
    item.get("content")
        .and_then(JsonValue::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text").and_then(JsonValue::as_str))
        .collect()
}
