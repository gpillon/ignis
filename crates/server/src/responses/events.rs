//! The Responses API's event sequence for one response (GitHub #282, spec
//! responses-api/01): the one producer of every Responses event ignis sends.
//!
//! A submitted request's scheduler events go in — through the same
//! [`OutputDecoder`] (reasoning / content split) and [`ToolCallScanner`] the
//! chat completions path uses — and the ordered Responses events come out:
//! `response.created`, `response.in_progress`, then per output item
//! `response.output_item.added`, its content events, `response.output_item.done`,
//! and one terminal event (`response.completed`, `response.incomplete` or
//! `response.failed`). Every event carries a `sequence_number` counting from
//! 0 within the response.
//!
//! It has three consumers and no other producer: the SSE writer, the
//! WebSocket writer, and the non-streaming handler, whose body is the
//! terminal event's `response`. So the three transports cannot disagree on a
//! payload.
//!
//! **Output items**, in generation order: a `reasoning` item when the
//! thinking channel produced text, a `message` item when the content channel
//! did, and one `function_call` item per scanned tool call. The reasoning
//! item closes as soon as the content channel produces anything. The message
//! item stays open until the response ends, so text the model writes after a
//! tool call still lands in the one message, as it lands in the one
//! `content` of a chat completion; a function call found meanwhile is added,
//! and done, at the next output index.

use serde::Serialize;
use serde_json::{json, Value as JsonValue};
use utoipa::ToSchema;

use ignis_core::{FinishReason, SchedEvent};

use crate::decoder::{Channel, Delta, OutputDecoder};
use crate::toolcall::{ToolCall, ToolCallScanner, ToolEvent, ToolSchemas};

/// The response object (OpenAI's `Response`): what `response.created`, every
/// lifecycle event and the terminal event carry, and the non-streaming body.
#[derive(Clone, Debug, Serialize, ToSchema)]
pub(crate) struct ResponseObject {
    /// `resp_<suffix>`, the suffix being the request id (ADR 0012) — or, for
    /// a socket request that was queued before the scheduler gave it one, the
    /// queue ticket (`q<n>`).
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    /// `queued`, `in_progress`, `completed`, `incomplete`, `cancelled` or
    /// `failed`.
    pub status: &'static str,
    pub background: bool,
    /// Set on `failed`: `{code, message}`.
    pub error: Option<ResponseError>,
    /// Set on a length stop: `{reason: "max_output_tokens"}`.
    pub incomplete_details: Option<IncompleteDetails>,
    /// The request's `instructions`, as sent.
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub model: String,
    /// The output items so far, in `output_index` order.
    #[schema(value_type = Vec<OutputItem>)]
    pub output: Vec<JsonValue>,
    pub parallel_tool_calls: bool,
    pub previous_response_id: Option<String>,
    #[schema(value_type = Object)]
    pub reasoning: JsonValue,
    pub store: bool,
    pub temperature: f32,
    #[schema(value_type = Object)]
    pub text: JsonValue,
    #[schema(value_type = Object)]
    pub tool_choice: JsonValue,
    /// The request's `tools`, in the Responses shape it sent them in.
    #[schema(value_type = Vec<Object>)]
    pub tools: Vec<JsonValue>,
    pub top_p: f32,
    pub truncation: &'static str,
    /// Absent (`null`) until the response ends.
    pub usage: Option<ResponseUsage>,
    #[schema(value_type = Object)]
    pub metadata: JsonValue,
    /// An ignis extension (spec server/08): the reasoning tokens emitted when
    /// the thinking budget forced the model's close. Absent when it did not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_forced_at: Option<u32>,
}

/// A `failed` response's error.
#[derive(Clone, Debug, Serialize, ToSchema)]
pub(crate) struct ResponseError {
    pub code: String,
    pub message: String,
}

/// Why a response is `incomplete`.
#[derive(Clone, Debug, Serialize, ToSchema)]
pub(crate) struct IncompleteDetails {
    pub reason: &'static str,
}

/// The usage figures of a finished response.
#[derive(Clone, Debug, Serialize, ToSchema)]
pub(crate) struct ResponseUsage {
    pub input_tokens: u32,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: u32,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: u32,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub(crate) struct InputTokensDetails {
    /// The prompt tokens this request resumed from retained state or a
    /// shared prefix instead of prefilling (ADR 0029); 0 when none.
    pub cached_tokens: u32,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub(crate) struct OutputTokensDetails {
    /// The generated tokens on the thinking channel.
    pub reasoning_tokens: u32,
}

/// One output item, as the terminal response carries it (documentation of
/// the shapes this module writes; the items themselves are built as JSON).
#[allow(dead_code)]
#[derive(Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OutputItem {
    /// The thinking channel: one `reasoning_text` part, no summary, never
    /// `encrypted_content`.
    Reasoning {
        id: String,
        #[schema(value_type = Vec<Object>)]
        summary: Vec<JsonValue>,
        #[schema(value_type = Vec<Object>)]
        content: Vec<JsonValue>,
    },
    /// The content channel: one `output_text` part.
    Message {
        id: String,
        status: String,
        role: String,
        #[schema(value_type = Vec<Object>)]
        content: Vec<JsonValue>,
    },
    /// One tool call, `arguments` a JSON-encoded object.
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: String,
    },
}

/// An output item still receiving content: where it sits and what it holds.
struct OpenItem {
    index: usize,
    text: String,
}

/// How a response ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ending {
    Completed,
    Incomplete,
    Cancelled,
    Failed,
}

/// The event producer for one response. Feed it the request's scheduler
/// events ([`ResponseEvents::on_event`]) and it answers each with the
/// Responses events that event makes available, in order.
pub(crate) struct ResponseEvents {
    response: ResponseObject,
    /// What the item ids are derived from: `rs_<suffix>`, `msg_<suffix>`,
    /// `fc_<suffix>_<n>`, and `call_<suffix>_<n>` for a call's `call_id`, so
    /// no two responses in one conversation reuse a `call_id`.
    suffix: String,
    decoder: OutputDecoder,
    scanner: ToolCallScanner,
    sequence: u64,
    reasoning: Option<OpenItem>,
    message: Option<OpenItem>,
    prompt_tokens: u32,
    cached_tokens: u32,
    output_tokens: u32,
    reasoning_tokens: u32,
    ending: Option<Ending>,
}

impl ResponseEvents {
    /// A response whose object starts as `response` (its id, echo fields and
    /// model already set by the caller), decoding with `decoder` and scanning
    /// tool calls typed by `schemas`.
    pub(crate) fn new(
        response: ResponseObject,
        suffix: String,
        decoder: OutputDecoder,
        schemas: ToolSchemas,
        prompt_tokens: u32,
    ) -> Self {
        Self {
            response,
            suffix,
            decoder,
            scanner: ToolCallScanner::with_schemas(schemas),
            sequence: 0,
            reasoning: None,
            message: None,
            prompt_tokens,
            cached_tokens: 0,
            output_tokens: 0,
            reasoning_tokens: 0,
            ending: None,
        }
    }

    /// The response's id (`resp_...`).
    pub(crate) fn id(&self) -> &str {
        &self.response.id
    }

    /// How the response ended; `None` while it runs.
    pub(crate) fn ending(&self) -> Option<Ending> {
        self.ending
    }

    /// The response object as it stands (the terminal one once it ended).
    pub(crate) fn response(&self) -> &ResponseObject {
        &self.response
    }

    /// `response.created`: `status: "queued"` and then `response.queued` for
    /// a request that is waiting for admission, else `status: "in_progress"`.
    pub(crate) fn created(&mut self, queued: bool) -> Vec<JsonValue> {
        self.response.status = if queued { "queued" } else { "in_progress" };
        let mut events = vec![self.lifecycle("response.created")];
        if queued {
            events.push(self.lifecycle("response.queued"));
        }
        events
    }

    /// `response.in_progress`: the request was handed to the engine.
    pub(crate) fn in_progress(&mut self) -> JsonValue {
        self.response.status = "in_progress";
        self.lifecycle("response.in_progress")
    }

    /// The events one scheduler event makes available: token text, a reuse
    /// report (counted, never an event), or the whole ending on `Done`.
    pub(crate) fn on_event(&mut self, event: SchedEvent) -> Vec<JsonValue> {
        let mut out = Vec::new();
        match event {
            SchedEvent::Token { token, .. } => {
                self.output_tokens += 1;
                if self.decoder.channel() == Channel::Reasoning {
                    self.reasoning_tokens += 1;
                }
                for delta in self.decoder.push(&[token]) {
                    self.route(delta, &mut out);
                }
            }
            // The tokens this request did not prefill because it resumed
            // from retained state (a prompt checkpoint), or stood on a shared
            // prefix (retained or a live sibling's). A request claims one or
            // the other, never both, so the larger is the whole of it.
            SchedEvent::StateReused { tokens, .. } | SchedEvent::PrefixReused { tokens, .. } => {
                self.cached_tokens = self.cached_tokens.max(tokens);
            }
            SchedEvent::Done { reason, thinking, .. } => {
                self.response.thinking_budget_forced_at = thinking.and_then(|b| b.forced_at);
                let ending = match reason {
                    FinishReason::Stop => Ending::Completed,
                    FinishReason::Length => {
                        self.response.incomplete_details =
                            Some(IncompleteDetails { reason: "max_output_tokens" });
                        Ending::Incomplete
                    }
                    FinishReason::Error => {
                        self.response.error = Some(ResponseError {
                            code: "server_error".into(),
                            message: "the engine could not run the request (its prefill failed repeatedly); see the server log".into(),
                        });
                        Ending::Failed
                    }
                };
                self.finish(ending, &mut out);
            }
            _ => {}
        }
        out
    }

    /// The response was cancelled (`response.cancel`): what it produced so far
    /// is closed out, and it ends `incomplete` with `status: "cancelled"`.
    pub(crate) fn cancelled(&mut self) -> Vec<JsonValue> {
        let mut out = Vec::new();
        self.finish(Ending::Cancelled, &mut out);
        out
    }

    /// The response failed after it was created — the request timed out, the
    /// engine dropped it, or a queued request's admission was refused.
    pub(crate) fn failed(&mut self, code: &str, message: String) -> Vec<JsonValue> {
        let mut out = Vec::new();
        self.response.error = Some(ResponseError { code: code.into(), message });
        self.finish(Ending::Failed, &mut out);
        out
    }

    /// One lifecycle event carrying the response object as it stands.
    fn lifecycle(&mut self, kind: &str) -> JsonValue {
        let response = serde_json::to_value(&self.response).expect("the response serializes");
        self.event(json!({ "type": kind, "response": response }))
    }

    /// Stamp `event` with the next sequence number.
    fn event(&mut self, mut event: JsonValue) -> JsonValue {
        event["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        event
    }

    /// Route one decoder delta: reasoning text into the reasoning item,
    /// content text through the tool-call scanner.
    fn route(&mut self, delta: Delta, out: &mut Vec<JsonValue>) {
        match delta.channel {
            Channel::Reasoning => {
                if delta.text.is_empty() {
                    return;
                }
                self.open_reasoning(out);
                let item = self.reasoning.as_mut().expect("just opened");
                item.text.push_str(&delta.text);
                let (index, item_id) = (item.index, format!("rs_{}", self.suffix));
                out.push(self.event(json!({
                    "type": "response.reasoning_text.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "content_index": 0,
                    "delta": delta.text,
                })));
            }
            Channel::Content => {
                for event in self.scanner.feed(&delta.text) {
                    self.scanned(event, out);
                }
            }
        }
    }

    /// One tool-call scanner event: message text, or a whole call.
    fn scanned(&mut self, event: ToolEvent, out: &mut Vec<JsonValue>) {
        self.close_reasoning(out);
        match event {
            ToolEvent::Content(text) => {
                self.open_message(out);
                let item = self.message.as_mut().expect("just opened");
                item.text.push_str(&text);
                let (index, item_id) = (item.index, format!("msg_{}", self.suffix));
                out.push(self.event(json!({
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": index,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                })));
            }
            ToolEvent::Call(call) => self.function_call(call, out),
        }
    }

    fn open_reasoning(&mut self, out: &mut Vec<JsonValue>) {
        if self.reasoning.is_some() {
            return;
        }
        let index = self.response.output.len();
        let item = json!({ "id": format!("rs_{}", self.suffix), "type": "reasoning", "summary": [], "content": [] });
        self.response.output.push(item.clone());
        self.reasoning = Some(OpenItem { index, text: String::new() });
        out.push(self.event(json!({ "type": "response.output_item.added", "output_index": index, "item": item })));
        let item_id = format!("rs_{}", self.suffix);
        out.push(self.event(json!({
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": index,
            "content_index": 0,
            "part": { "type": "reasoning_text", "text": "" },
        })));
    }

    /// Close the reasoning item: its text is final once the content channel
    /// has produced anything, or the response ends.
    fn close_reasoning(&mut self, out: &mut Vec<JsonValue>) {
        // Taken: a second close is a no-op, and nothing reopens it, since the
        // decoder leaves the reasoning channel once and for good.
        let Some(OpenItem { index, text }) = self.reasoning.take() else {
            return;
        };
        let item_id = format!("rs_{}", self.suffix);
        let part = json!({ "type": "reasoning_text", "text": text });
        out.push(self.event(json!({
            "type": "response.reasoning_text.done",
            "item_id": item_id,
            "output_index": index,
            "content_index": 0,
            "text": text,
        })));
        out.push(self.event(json!({
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": index,
            "content_index": 0,
            "part": part,
        })));
        let item = json!({ "id": item_id, "type": "reasoning", "summary": [], "content": [part] });
        self.response.output[index] = item.clone();
        out.push(self.event(json!({ "type": "response.output_item.done", "output_index": index, "item": item })));
    }

    fn open_message(&mut self, out: &mut Vec<JsonValue>) {
        if self.message.is_some() {
            return;
        }
        let index = self.response.output.len();
        let item_id = format!("msg_{}", self.suffix);
        let item = json!({
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        });
        self.response.output.push(item.clone());
        self.message = Some(OpenItem { index, text: String::new() });
        out.push(self.event(json!({ "type": "response.output_item.added", "output_index": index, "item": item })));
        out.push(self.event(json!({
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": index,
            "content_index": 0,
            "part": { "type": "output_text", "text": "", "annotations": [] },
        })));
    }

    /// Close the message item with `status`.
    fn close_message(&mut self, status: &str, out: &mut Vec<JsonValue>) {
        let Some(OpenItem { index, text }) = self.message.take() else {
            return;
        };
        let item_id = format!("msg_{}", self.suffix);
        let part = json!({ "type": "output_text", "text": text, "annotations": [] });
        out.push(self.event(json!({
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": index,
            "content_index": 0,
            "text": text,
            "logprobs": [],
        })));
        out.push(self.event(json!({
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": index,
            "content_index": 0,
            "part": part,
        })));
        let item = json!({
            "id": item_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [part],
        });
        self.response.output[index] = item.clone();
        out.push(self.event(json!({ "type": "response.output_item.done", "output_index": index, "item": item })));
    }

    /// One whole tool call: its item is added, its arguments streamed in one
    /// delta (the call is parsed out of a closed block, so it is whole by the
    /// time it is known), and it is done.
    fn function_call(&mut self, call: ToolCall, out: &mut Vec<JsonValue>) {
        let index = self.response.output.len();
        let item_id = format!("fc_{}_{}", self.suffix, call.index);
        let call_id = format!("call_{}_{}", self.suffix, call.index);
        let added = json!({
            "id": item_id,
            "type": "function_call",
            "status": "in_progress",
            "call_id": call_id,
            "name": call.name,
            "arguments": "",
        });
        self.response.output.push(added.clone());
        out.push(self.event(json!({ "type": "response.output_item.added", "output_index": index, "item": added })));
        out.push(self.event(json!({
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "output_index": index,
            "delta": call.arguments,
        })));
        out.push(self.event(json!({
            "type": "response.function_call_arguments.done",
            "item_id": item_id,
            "output_index": index,
            "name": call.name,
            "arguments": call.arguments,
        })));
        let item = json!({
            "id": item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": call.name,
            "arguments": call.arguments,
        });
        self.response.output[index] = item.clone();
        out.push(self.event(json!({ "type": "response.output_item.done", "output_index": index, "item": item })));
    }

    /// End the response: everything still held back — the decoder's tail and
    /// the scanner's (a call left open is dropped whole, never delivered
    /// half-written) — then the open items closed, then the terminal event.
    fn finish(&mut self, ending: Ending, out: &mut Vec<JsonValue>) {
        for delta in self.decoder.finish() {
            self.route(delta, out);
        }
        for event in self.scanner.finish() {
            self.scanned(event, out);
        }
        self.close_reasoning(out);
        let status = match ending {
            Ending::Completed => "completed",
            Ending::Incomplete => "incomplete",
            Ending::Cancelled => "cancelled",
            Ending::Failed => "failed",
        };
        // GitHub #70: a generation that reasoned and produced nothing else is
        // reported, as on chat completions.
        let only_reasoning = !self.response.output.is_empty()
            && self.response.output.iter().all(|item| item["type"] == "reasoning");
        crate::api::report_if_all_reasoning_no_content(&self.response.id, only_reasoning, status);
        // A message cut short is `incomplete`, as OpenAI marks it.
        self.close_message(if ending == Ending::Completed { "completed" } else { "incomplete" }, out);
        out.push(self.end(ending, status));
    }

    /// The terminal event, with the usage figures.
    fn end(&mut self, ending: Ending, status: &'static str) -> JsonValue {
        self.ending = Some(ending);
        self.response.status = status;
        self.response.usage = Some(ResponseUsage {
            input_tokens: self.prompt_tokens,
            input_tokens_details: InputTokensDetails { cached_tokens: self.cached_tokens },
            output_tokens: self.output_tokens,
            output_tokens_details: OutputTokensDetails { reasoning_tokens: self.reasoning_tokens },
            total_tokens: self.prompt_tokens.saturating_add(self.output_tokens),
        });
        let kind = match ending {
            Ending::Completed => "response.completed",
            Ending::Incomplete | Ending::Cancelled => "response.incomplete",
            Ending::Failed => "response.failed",
        };
        self.lifecycle(kind)
    }
}
