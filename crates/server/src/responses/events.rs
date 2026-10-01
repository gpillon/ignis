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
//! **Output items**, in generation order and one after another — each
//! added, filled and done before the next is added: a `reasoning` item when
//! the thinking channel produced text, a `message` item when the content
//! channel did, and one `function_call` item per scanned tool call. The
//! reasoning item closes as soon as the content channel produces anything; a
//! call closes the message before it, and text the model writes after a call
//! opens a new message item.

use serde::Serialize;
use serde_json::{json, Value as JsonValue};
use utoipa::ToSchema;

use ignis_core::{FinishReason, SchedEvent, TokenId};

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

/// One output item, as its events and the response carry it: the one type
/// every item on the wire is serialized from.
#[derive(Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OutputItem {
    /// The thinking channel: one `reasoning_text` part, no summary, never
    /// `encrypted_content`.
    Reasoning {
        id: String,
        #[schema(value_type = Vec<Object>)]
        summary: Vec<JsonValue>,
        content: Vec<Part>,
    },
    /// The content channel: one `output_text` part.
    Message {
        id: String,
        status: &'static str,
        role: &'static str,
        content: Vec<Part>,
    },
    /// One tool call, `arguments` a JSON-encoded object.
    FunctionCall {
        id: String,
        status: &'static str,
        call_id: String,
        name: String,
        arguments: String,
    },
}

/// A content part of an output item.
#[derive(Clone, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Part {
    ReasoningText {
        text: String,
    },
    OutputText {
        text: String,
        /// Always empty: this server annotates nothing.
        #[schema(value_type = Vec<Object>)]
        annotations: Vec<JsonValue>,
    },
}

/// An output item still receiving content: where it sits, its id, and what
/// it holds.
struct OpenItem {
    index: usize,
    id: String,
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
    /// What the item ids are derived from: `rs_<suffix>`, `msg_<suffix>`
    /// (`msg_<suffix>_<n>` for text after a call), `fc_<suffix>_<n>`, and
    /// `call_<suffix>_<n>` for a call's `call_id`, so no two responses in one
    /// conversation reuse a `call_id`.
    suffix: String,
    decoder: OutputDecoder,
    scanner: ToolCallScanner,
    sequence: u64,
    reasoning: Option<OpenItem>,
    message: Option<OpenItem>,
    /// Message items opened so far.
    messages: usize,
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
            messages: 0,
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
        // The tokens this request did not prefill: a reuse report, counted,
        // never an event — the same quantity chat's `usage` reports.
        if let Some(reused) = crate::engine::reused_prompt_tokens(&event) {
            self.cached_tokens = self.cached_tokens.max(reused);
        }
        match event {
            SchedEvent::Token { token, .. } => return self.on_tokens(&[token]),
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
                            message: crate::api::ENGINE_ERROR_MESSAGE.into(),
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

    /// The events a run of tokens that arrived together makes available:
    /// one delta per channel the run touched, never one per token
    /// ([`crate::decoder::join`]). Each token is still counted on the
    /// channel it was generated on.
    pub(crate) fn on_tokens(&mut self, tokens: &[TokenId]) -> Vec<JsonValue> {
        let mut deltas = Vec::new();
        for &token in tokens {
            self.output_tokens += 1;
            if self.decoder.channel() == Channel::Reasoning {
                self.reasoning_tokens += 1;
            }
            deltas.extend(self.decoder.push(&[token]));
        }
        let mut out = Vec::new();
        for delta in crate::decoder::join(deltas) {
            self.route(delta, &mut out);
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
                let (index, id) = (item.index, item.id.clone());
                out.push(self.event(json!({
                    "type": "response.reasoning_text.delta",
                    "item_id": id,
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

    /// One tool-call scanner event: message text, or a whole call. Items
    /// follow one another: a call closes the message before it, and text
    /// after a call opens a new one.
    fn scanned(&mut self, event: ToolEvent, out: &mut Vec<JsonValue>) {
        self.close_reasoning(out);
        match event {
            ToolEvent::Content(text) => {
                self.open_message(out);
                let item = self.message.as_mut().expect("just opened");
                item.text.push_str(&text);
                let (index, id) = (item.index, item.id.clone());
                out.push(self.event(json!({
                    "type": "response.output_text.delta",
                    "item_id": id,
                    "output_index": index,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                })));
            }
            ToolEvent::Call(call) => {
                self.close_message("completed", out);
                self.function_call(call, out);
            }
        }
    }

    /// Add `item` at the next output index (`response.output_item.added`).
    fn add(&mut self, item: OutputItem, out: &mut Vec<JsonValue>) -> usize {
        let index = self.response.output.len();
        let item = serde_json::to_value(item).expect("an item serializes");
        self.response.output.push(item.clone());
        out.push(self.event(json!({ "type": "response.output_item.added", "output_index": index, "item": item })));
        index
    }

    /// The item at `index` is final (`response.output_item.done`).
    fn done(&mut self, index: usize, item: OutputItem, out: &mut Vec<JsonValue>) {
        let item = serde_json::to_value(item).expect("an item serializes");
        self.response.output[index] = item.clone();
        out.push(self.event(json!({ "type": "response.output_item.done", "output_index": index, "item": item })));
    }

    /// A content part's `added` or `done` event.
    fn part_event(&mut self, kind: &str, item: &OpenItem, part: Part) -> JsonValue {
        self.event(json!({
            "type": kind,
            "item_id": item.id,
            "output_index": item.index,
            "content_index": 0,
            "part": part,
        }))
    }

    fn open_reasoning(&mut self, out: &mut Vec<JsonValue>) {
        if self.reasoning.is_some() {
            return;
        }
        let id = format!("rs_{}", self.suffix);
        let index = self.add(OutputItem::Reasoning { id: id.clone(), summary: Vec::new(), content: Vec::new() }, out);
        let item = OpenItem { index, id, text: String::new() };
        out.push(self.part_event("response.content_part.added", &item, Part::ReasoningText { text: String::new() }));
        self.reasoning = Some(item);
    }

    /// Close the reasoning item: its text is final once the content channel
    /// has produced anything, or the response ends.
    fn close_reasoning(&mut self, out: &mut Vec<JsonValue>) {
        // Taken: a second close is a no-op, and nothing reopens it, since the
        // decoder leaves the reasoning channel once and for good.
        let Some(item) = self.reasoning.take() else {
            return;
        };
        let part = Part::ReasoningText { text: item.text.clone() };
        out.push(self.event(json!({
            "type": "response.reasoning_text.done",
            "item_id": item.id,
            "output_index": item.index,
            "content_index": 0,
            "text": item.text,
        })));
        out.push(self.part_event("response.content_part.done", &item, part.clone()));
        let done = OutputItem::Reasoning { id: item.id, summary: Vec::new(), content: vec![part] };
        self.done(item.index, done, out);
    }

    fn open_message(&mut self, out: &mut Vec<JsonValue>) {
        if self.message.is_some() {
            return;
        }
        // `msg_<suffix>`, and `msg_<suffix>_<n>` for text after the n-th call.
        let id = match self.messages {
            0 => format!("msg_{}", self.suffix),
            n => format!("msg_{}_{n}", self.suffix),
        };
        self.messages += 1;
        let added = OutputItem::Message { id: id.clone(), status: "in_progress", role: "assistant", content: Vec::new() };
        let index = self.add(added, out);
        let item = OpenItem { index, id, text: String::new() };
        let part = Part::OutputText { text: String::new(), annotations: Vec::new() };
        out.push(self.part_event("response.content_part.added", &item, part));
        self.message = Some(item);
    }

    /// Close the message item with `status`.
    fn close_message(&mut self, status: &'static str, out: &mut Vec<JsonValue>) {
        let Some(item) = self.message.take() else {
            return;
        };
        let part = Part::OutputText { text: item.text.clone(), annotations: Vec::new() };
        out.push(self.event(json!({
            "type": "response.output_text.done",
            "item_id": item.id,
            "output_index": item.index,
            "content_index": 0,
            "text": item.text,
            "logprobs": [],
        })));
        out.push(self.part_event("response.content_part.done", &item, part.clone()));
        let done = OutputItem::Message { id: item.id, status, role: "assistant", content: vec![part] };
        self.done(item.index, done, out);
    }

    /// One whole tool call: its item is added, its arguments streamed in one
    /// delta (the call is parsed out of a closed block, so it is whole by the
    /// time it is known), and it is done.
    fn function_call(&mut self, call: ToolCall, out: &mut Vec<JsonValue>) {
        let id = format!("fc_{}_{}", self.suffix, call.index);
        let call_id = format!("call_{}_{}", self.suffix, call.index);
        let added = OutputItem::FunctionCall {
            id: id.clone(),
            status: "in_progress",
            call_id: call_id.clone(),
            name: call.name.clone(),
            arguments: String::new(),
        };
        let index = self.add(added, out);
        out.push(self.event(json!({
            "type": "response.function_call_arguments.delta",
            "item_id": id,
            "output_index": index,
            "delta": call.arguments,
        })));
        out.push(self.event(json!({
            "type": "response.function_call_arguments.done",
            "item_id": id,
            "output_index": index,
            "name": call.name,
            "arguments": call.arguments,
        })));
        let done = OutputItem::FunctionCall { id, status: "completed", call_id, name: call.name, arguments: call.arguments };
        self.done(index, done, out);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::TokenDecoder;
    use crate::responses::{drive, Driven};

    /// Token `n` is `TEXTS[n]`.
    struct Texts(&'static [&'static str]);

    impl TokenDecoder for Texts {
        fn push(&mut self, token: TokenId) -> String {
            self.0[token as usize].to_owned()
        }
        fn finish(&mut self) -> String {
            String::new()
        }
    }

    fn events(texts: &'static [&'static str], thinking: bool) -> ResponseEvents {
        let response = ResponseObject {
            id: "resp_0".into(),
            object: "response",
            created_at: 0,
            status: "in_progress",
            background: false,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "m".into(),
            output: Vec::new(),
            parallel_tool_calls: true,
            previous_response_id: None,
            reasoning: JsonValue::Null,
            store: true,
            temperature: 0.0,
            text: JsonValue::Null,
            tool_choice: json!("auto"),
            tools: Vec::new(),
            top_p: 1.0,
            truncation: "disabled",
            usage: None,
            metadata: json!({}),
            thinking_budget_forced_at: None,
        };
        let decoder = OutputDecoder::new(Box::new(Texts(texts)), thinking);
        ResponseEvents::new(response, "0".into(), decoder, ToolSchemas::from_tools(&[]), 1)
    }

    fn done(tokens: u32) -> SchedEvent {
        SchedEvent::Done {
            request: 0,
            tokens,
            reason: FinishReason::Stop,
            spec: None,
            readout: None,
            attention: None,
            drawn: None,
            thinking: None,
        }
    }

    /// Every event for `tokens`, pushed one at a time or as one run.
    fn produced(texts: &'static [&'static str], thinking: bool, run: bool) -> Vec<JsonValue> {
        let mut events = events(texts, thinking);
        let tokens: Vec<TokenId> = (0..texts.len() as u32).collect();
        let mut out = Vec::new();
        if run {
            out.extend(events.on_tokens(&tokens));
        } else {
            for &token in &tokens {
                out.extend(events.on_event(SchedEvent::Token { request: 0, token }));
            }
        }
        out.extend(events.on_event(done(tokens.len() as u32)));
        out
    }

    fn of_type<'a>(events: &'a [JsonValue], kind: &str) -> Vec<&'a JsonValue> {
        events.iter().filter(|e| e["type"] == kind).collect()
    }

    fn joined(events: &[JsonValue], kind: &str) -> String {
        of_type(events, kind).iter().map(|e| e["delta"].as_str().unwrap()).collect()
    }

    #[test]
    fn a_run_across_the_thinking_close_is_one_reasoning_delta_then_one_content_delta() {
        const TEXTS: &[&str] = &["let me ", "think", "</thi", "nk>The ", "answer"];
        let run = produced(TEXTS, true, true);
        let deltas: Vec<(&str, &str)> = run
            .iter()
            .filter(|e| e["type"].as_str().is_some_and(|t| t.ends_with("text.delta")))
            .map(|e| (e["type"].as_str().unwrap(), e["delta"].as_str().unwrap()))
            .collect();
        assert_eq!(
            deltas,
            [("response.reasoning_text.delta", "let me think"), ("response.output_text.delta", "The answer")]
        );
        let usage = &run.last().unwrap()["response"]["usage"];
        assert_eq!(usage["output_tokens"], 5);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 4, "still counted token by token");
    }

    #[test]
    fn a_run_is_byte_for_byte_the_text_and_calls_token_by_token_gives() {
        const TEXTS: &[&str] = &[
            "plan\n", "</think>\n", "Reading ", "<tool_", "call>\n<function=read>\n<parameter=path>\na.txt\n</para",
            "meter>\n</function>\n</tool_call>", " then ", "more <", "tool_call>\n<function=b>\n</function>\n</tool_call>",
        ];
        let (one, run) = (produced(TEXTS, true, false), produced(TEXTS, true, true));
        for kind in ["response.reasoning_text.delta", "response.output_text.delta"] {
            assert_eq!(joined(&run, kind), joined(&one, kind), "{kind}");
        }
        let calls = |events: &[JsonValue]| -> Vec<JsonValue> {
            of_type(events, "response.function_call_arguments.done").iter().map(|e| e["arguments"].clone()).collect()
        };
        assert_eq!(calls(&run), calls(&one));
        assert_eq!(calls(&run).len(), 2);
        assert_eq!(run.last().unwrap()["response"]["output"], one.last().unwrap()["response"]["output"]);
        assert!(of_type(&run, "response.output_text.delta").len() < of_type(&one, "response.output_text.delta").len());
    }

    /// A round's tokens waiting together are one delta, and the ending still
    /// comes after it.
    #[tokio::test]
    async fn a_round_of_tokens_already_waiting_is_one_delta_and_the_ending_follows_it() {
        const TEXTS: &[&str] = &["one ", "two ", "three ", "four"];
        let (route, mut stream) = tokio::sync::mpsc::unbounded_channel();
        for token in 0..4 {
            route.send(SchedEvent::Token { request: 0, token }).unwrap();
        }
        route.send(done(4)).unwrap();
        let mut events = events(TEXTS, false);
        let mut emitted = Vec::new();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let driven = drive(&mut events, &mut stream, deadline, |e| emitted.push(e)).await;
        assert!(matches!(driven, Driven::Ended));
        let deltas = of_type(&emitted, "response.output_text.delta");
        assert_eq!(deltas.len(), 1, "{emitted:?}");
        assert_eq!(deltas[0]["delta"], "one two three four");
        let at = |kind: &str| emitted.iter().position(|e| e["type"] == kind).unwrap();
        assert!(at("response.output_text.delta") < at("response.output_text.done"));
        assert_eq!(emitted.last().unwrap()["type"], "response.completed");
    }
}
