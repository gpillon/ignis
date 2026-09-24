//! Tool-call XML parsing over the content channel (GitHub #121): the chat
//! template's tool-call dialect —
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=PARAM>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! (`.qwen/tmp/chat_template.jinja`; the same format the system prompt
//! instructs the model to reply in when tools are offered) — is scanned out
//! of the `Channel::Content` text [`crate::decoder::OutputDecoder`] already
//! produced, split from plain content, and reassembled into the OpenAI
//! `tool_calls` wire shape.
//!
//! **Boundary robustness.** The scanner buffers everything from the moment
//! it sees the opening `<tool_call>` tag until the matching `</tool_call>`,
//! so a tag split across arbitrary chunk boundaries can never be missed —
//! matching only needs to run against the *whole* buffered span, not track
//! partial-tag state token by token. A call is emitted as a single complete
//! event (id + name + JSON-encoded arguments) only once its closing tag is
//! seen and its body parses; several calls in one response each get the
//! next `index` in the order they closed, so indices are stable and never
//! merge two calls together.
//!
//! **An interrupted or malformed call is dropped whole, never leaked
//! half-written.** A stop/cancel landing before `</tool_call>` closes, or a
//! closed block whose body does not parse as `<function=...>`, is silently
//! discarded at the point it is discovered — never surfaced as partial XML
//! in `content` and never surfaced as a tool call with truncated or
//! garbled arguments (GitHub #121 acceptance criterion 3). The caller
//! learns generation stopped from `finish_reason` (`stop`/`length`); it is
//! never told a tool call happened when the call did not actually
//! complete.
//!
//! **Argument typing follows the tool's schema.** Each `<parameter>` body is
//! raw text, and the dialect carries no type of its own, so the type comes
//! from the request's `tools` ([`ToolSchemas`]): a parameter the schema
//! declares a `string` is that exact text even when it reads as JSON — a
//! `write_file` whose content is `123` or `{"a":1}` gets that text, not a
//! number or an object — and a declared `integer`, `number`, `boolean`,
//! `null`, `object` or `array` is parsed as that type, falling back to the
//! text when it does not parse (the same rule as vLLM's and SGLang's
//! `qwen3_coder` parsers). A parameter the schema does not describe (no
//! schemas, an unknown function, an undeclared or untyped parameter) keeps
//! the schema-free rule: valid JSON round-trips as its own type, anything
//! else is a string. A single leading and single trailing newline
//! immediately inside the `<parameter=...>` / `</parameter>` tags is
//! trimmed (the template's own formatting whitespace); interior newlines —
//! deliberately preserved by the system prompt's "can span multiple lines"
//! contract — are not touched.
//!
//! **An unclosed parameter ends where the next one starts.** The model
//! sometimes omits a `</parameter>`, most often on the last parameter
//! before `</function>`. Such a value runs to the next `<parameter=` or to
//! the end of the function body, whichever comes first, instead of being
//! dropped (the `qwen3_coder` parsers' same boundary rule).

use std::collections::HashMap;

use serde_json::Value as JsonValue;

const OPEN_TAG: &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";
const FUNCTION_PREFIX: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_PREFIX: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

/// One fully parsed, complete tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Position among the calls emitted by this scanner (0, 1, 2, ...) —
    /// the OpenAI streaming `tool_calls[].index` a client keys its
    /// reassembly buffer on.
    pub index: usize,
    pub id: String,
    pub name: String,
    /// A JSON-encoded object (`serde_json::to_string` of the parsed
    /// parameters), e.g. `{"path":"a.txt","recursive":true}`.
    pub arguments: String,
}

/// One event out of the scanner: ordinary content text (tool-call markup
/// already stripped out), or a complete tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolEvent {
    Content(String),
    Call(ToolCall),
}

enum State {
    /// Scanning for `<tool_call>`; text that cannot still become that tag
    /// streams straight out as `ToolEvent::Content`.
    Scanning,
    /// Between `<tool_call>` and its matching `</tool_call>` — buffering,
    /// nothing streams out until the block resolves one way or the other.
    Buffering,
}

/// The JSON types a request's `tools` declare for each function's
/// parameters — what [`ToolCallScanner`] types an argument by. Built once
/// per request from the validated `tools` array; an empty set (the
/// default) types every argument by the schema-free rule.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolSchemas {
    /// function name → parameter name → the declared type names
    /// (`"string"`, `"integer"`, ...). An empty list is an untyped
    /// parameter.
    functions: HashMap<String, HashMap<String, Vec<String>>>,
}

impl ToolSchemas {
    /// Reads `tools[].function.{name, parameters.properties}`. Entries
    /// without a name or without `properties` contribute nothing; a
    /// property's types come from its `type` (a name or a list of names),
    /// else from its `anyOf` / `oneOf` members' `type`, else from its
    /// `enum` values' JSON kinds.
    pub fn from_tools(tools: &[JsonValue]) -> Self {
        let mut functions = HashMap::new();
        for tool in tools {
            let Some(function) = tool.get("function") else { continue };
            let Some(name) = function.get("name").and_then(JsonValue::as_str) else { continue };
            let Some(properties) = function
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(JsonValue::as_object)
            else {
                continue;
            };
            let params = properties
                .iter()
                .map(|(param, schema)| (param.clone(), declared_types(schema)))
                .collect();
            functions.insert(name.to_string(), params);
        }
        Self { functions }
    }

    fn param_types(&self, function: &str, param: &str) -> &[String] {
        self.functions
            .get(function)
            .and_then(|params| params.get(param))
            .map_or(&[], Vec::as_slice)
    }
}

/// A property schema's declared type names (see [`ToolSchemas::from_tools`]).
fn declared_types(schema: &JsonValue) -> Vec<String> {
    fn names(ty: &JsonValue) -> Vec<String> {
        match ty {
            JsonValue::String(s) => vec![s.clone()],
            JsonValue::Array(list) => list.iter().filter_map(JsonValue::as_str).map(str::to_string).collect(),
            _ => Vec::new(),
        }
    }
    if let Some(ty) = schema.get("type") {
        return names(ty);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(members) = schema.get(key).and_then(JsonValue::as_array) {
            return members.iter().filter_map(|m| m.get("type")).flat_map(names).collect();
        }
    }
    if let Some(values) = schema.get("enum").and_then(JsonValue::as_array) {
        let mut kinds: Vec<String> = values.iter().map(|v| json_kind(v).to_string()).collect();
        kinds.dedup();
        return kinds;
    }
    Vec::new()
}

/// The JSON Schema type name of a JSON value.
fn json_kind(value: &JsonValue) -> &'static str {
    match value {
        JsonValue::Null => "null",
        JsonValue::Bool(_) => "boolean",
        JsonValue::Number(n) if n.is_i64() || n.is_u64() => "integer",
        JsonValue::Number(_) => "number",
        JsonValue::String(_) => "string",
        JsonValue::Array(_) => "array",
        JsonValue::Object(_) => "object",
    }
}

/// Scans a request's whole content-channel text for tool-call blocks,
/// across as many `feed` calls as the text arrives in.
pub struct ToolCallScanner {
    state: State,
    hold: String,
    next_index: usize,
    schemas: ToolSchemas,
}

impl Default for ToolCallScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallScanner {
    /// A scanner with no schemas: every argument is typed by the
    /// schema-free rule.
    pub fn new() -> Self {
        Self::with_schemas(ToolSchemas::default())
    }

    /// A scanner that types each argument by the request's `tools`.
    pub fn with_schemas(schemas: ToolSchemas) -> Self {
        Self {
            state: State::Scanning,
            hold: String::new(),
            next_index: 0,
            schemas,
        }
    }

    /// True while a `<tool_call>` has been opened but not yet closed —
    /// checked by the caller *before* [`finish`](Self::finish) to tell a
    /// clean end (last thing emitted was a complete call, or plain
    /// content) from a truncated one (acceptance criterion 3).
    pub fn is_mid_call(&self) -> bool {
        matches!(self.state, State::Buffering)
    }

    /// Whether this scanner has ever emitted a complete call.
    pub fn any_calls(&self) -> bool {
        self.next_index > 0
    }

    /// Feed newly available content text; returns the events it makes
    /// available (zero, one, or more).
    pub fn feed(&mut self, text: &str) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        self.hold.push_str(text);
        loop {
            match self.state {
                State::Scanning => {
                    if let Some(idx) = self.hold.find(OPEN_TAG) {
                        if idx > 0 {
                            out.push(ToolEvent::Content(self.hold[..idx].to_string()));
                        }
                        self.hold.drain(..idx + OPEN_TAG.len());
                        self.state = State::Buffering;
                        continue;
                    }
                    let holdback = tag_prefix_holdback(&self.hold, OPEN_TAG);
                    let safe_len = self.hold.len() - holdback;
                    if safe_len == 0 {
                        break;
                    }
                    let safe: String = self.hold.drain(..safe_len).collect();
                    out.push(ToolEvent::Content(safe));
                    break;
                }
                State::Buffering => {
                    let Some(idx) = self.hold.find(CLOSE_TAG) else {
                        break; // still incomplete; wait for more text
                    };
                    let body: String = self.hold.drain(..idx + CLOSE_TAG.len()).collect();
                    let inner = &body[..body.len() - CLOSE_TAG.len()];
                    if let Some((name, arguments)) = parse_function_block(inner, &self.schemas) {
                        let index = self.next_index;
                        self.next_index += 1;
                        out.push(ToolEvent::Call(ToolCall {
                            index,
                            id: format!("call_{index}"),
                            name,
                            arguments,
                        }));
                    }
                    // A closed block that does not parse as a well-formed
                    // `<function=...>` is dropped silently — same
                    // never-leak-garbage rule as an interrupted one.
                    self.state = State::Scanning;
                    continue;
                }
            }
        }
        out
    }

    /// Flush whatever is left at the end of generation.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        match self.state {
            State::Scanning => {
                let text = std::mem::take(&mut self.hold);
                if text.is_empty() {
                    vec![]
                } else {
                    vec![ToolEvent::Content(text)]
                }
            }
            State::Buffering => {
                // Truncated mid-call: drop the whole buffered span, raw
                // XML included — never leaked as content, never surfaced
                // as a call.
                self.hold.clear();
                vec![]
            }
        }
    }
}

/// The length of the trailing suffix of `text` that is a proper (non-empty,
/// non-full) prefix of `tag` — the same holdback shape as
/// `decoder::holdback_len`, generalized to any literal tag.
fn tag_prefix_holdback(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&len| text.ends_with(&tag[..len]))
        .unwrap_or(0)
}

/// Parses `<function=NAME>...<parameter=P>VALUE</parameter>...</function>`
/// (the text between `<tool_call>` and `</tool_call>`, both already
/// stripped) into `(name, json_arguments)`, typing each argument by
/// `schemas`. Returns `None` if the block does not open with `<function=`
/// or never closes with `</function>`.
fn parse_function_block(inner: &str, schemas: &ToolSchemas) -> Option<(String, String)> {
    let after_prefix = inner.trim_start().strip_prefix(FUNCTION_PREFIX)?;
    let name_end = after_prefix.find('>')?;
    let name = after_prefix[..name_end].to_string();
    let body = after_prefix[name_end + 1..].find(FUNCTION_CLOSE).map(|end| &after_prefix[name_end + 1..name_end + 1 + end])?;

    let mut args = serde_json::Map::new();
    let mut rest = body;
    while let Some(start) = rest.find(PARAMETER_PREFIX) {
        let after = &rest[start + PARAMETER_PREFIX.len()..];
        let Some(name_end) = after.find('>') else {
            break; // malformed trailing parameter tag: stop, keep what we have
        };
        let param_name = after[..name_end].to_string();
        let after_name = &after[name_end + 1..];
        // The value ends at its `</parameter>`, or — when the model left it
        // unclosed — at the next `<parameter=` or the end of the body.
        let close = after_name.find(PARAMETER_CLOSE);
        let next = after_name.find(PARAMETER_PREFIX);
        let (value_end, resume) = match (close, next) {
            (Some(c), Some(n)) if n < c => (n, n),
            (Some(c), _) => (c, c + PARAMETER_CLOSE.len()),
            (None, Some(n)) => (n, n),
            (None, None) => (after_name.len(), after_name.len()),
        };
        let value = trim_one_newline(&after_name[..value_end]);
        let types = schemas.param_types(&name, &param_name);
        args.insert(param_name, parse_argument_value(value, types));
        rest = &after_name[resume..];
    }
    let arguments = serde_json::to_string(&serde_json::Value::Object(args))
        .expect("a Map<String, Value> always serializes");
    Some((name, arguments))
}

/// Trims exactly one leading and one trailing `\n` (the template's own
/// formatting newlines around a parameter value), never more — interior
/// and any *extra* surrounding newlines are the model's own content.
fn trim_one_newline(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}

/// A parameter's JSON value, given the schema's declared `types` for it.
///
/// - No declared type: valid JSON (a number, bool, array, object, or an
///   explicitly quoted string) round-trips as that type; anything else —
///   ordinary text — becomes a JSON string of exactly that text.
/// - `null` declared and the text is `null`: JSON null.
/// - `string` declared: the text, whatever it looks like.
/// - Otherwise the first declared type the text parses as (an `integer`
///   must be integral, a `boolean` is `true`/`false` in any case, an
///   `object`/`array` must be JSON of that kind), else the text as a
///   string — a value never disappears because it disagrees with its
///   schema.
fn parse_argument_value(value: &str, types: &[String]) -> JsonValue {
    let text = || JsonValue::String(value.to_string());
    if types.is_empty() {
        return serde_json::from_str(value).unwrap_or_else(|_| text());
    }
    let declares = |name: &str| types.iter().any(|t| t == name);
    let trimmed = value.trim();
    if declares("null") && trimmed == "null" {
        return JsonValue::Null;
    }
    if declares("string") {
        return text();
    }
    for ty in types {
        let parsed = match ty.as_str() {
            "integer" => trimmed.parse::<i64>().ok().map(JsonValue::from),
            "number" => serde_json::from_str::<JsonValue>(trimmed).ok().filter(JsonValue::is_number),
            "boolean" => match trimmed.to_ascii_lowercase().as_str() {
                "true" => Some(JsonValue::Bool(true)),
                "false" => Some(JsonValue::Bool(false)),
                _ => None,
            },
            "object" => serde_json::from_str::<JsonValue>(trimmed).ok().filter(JsonValue::is_object),
            "array" => serde_json::from_str::<JsonValue>(trimmed).ok().filter(JsonValue::is_array),
            _ => None,
        };
        if let Some(parsed) = parsed {
            return parsed;
        }
    }
    text()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str]) -> Vec<ToolEvent> {
        let mut scanner = ToolCallScanner::new();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(scanner.feed(chunk));
        }
        events.extend(scanner.finish());
        events
    }

    fn calls(events: &[ToolEvent]) -> Vec<&ToolCall> {
        events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Call(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    fn content(events: &[ToolEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Content(s) => Some(s.as_str()),
                _ => None,
            })
            .collect()
    }

    fn args(call: &ToolCall) -> serde_json::Value {
        serde_json::from_str(&call.arguments).expect("valid JSON arguments")
    }

    #[test]
    fn plain_content_with_no_tool_call_passes_through_unchanged() {
        let events = run(&["hello ", "world"]);
        assert_eq!(content(&events), "hello world");
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn a_single_tool_call_in_one_chunk_is_parsed() {
        let events = run(&[
            "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>",
        ]);
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(args(calls[0]), serde_json::json!({"path": "a.txt"}));
    }

    #[test]
    fn text_before_the_call_is_preserved_as_content() {
        let events = run(&[
            "Let me check that.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>",
        ]);
        assert_eq!(content(&events), "Let me check that.\n\n");
        assert_eq!(calls(&events).len(), 1);
    }

    #[test]
    fn a_tool_call_split_across_many_chunk_boundaries_still_reassembles() {
        let events = run(&[
            "before <tool_",
            "call>\n<function=read",
            "_file>\n<param",
            "eter=path>\na.txt\n</parameter>\n</functio",
            "n>\n</tool_call> after",
        ]);
        assert_eq!(content(&events), "before  after");
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(args(calls[0]), serde_json::json!({"path": "a.txt"}));
    }

    #[test]
    fn the_open_tag_split_one_byte_at_a_time_is_still_found() {
        let chunks: Vec<String> = "<tool_call>\n<function=f>\n</function>\n</tool_call>"
            .chars()
            .map(|c| c.to_string())
            .collect();
        let refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        let events = run(&refs);
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
    }

    #[test]
    fn several_calls_in_one_response_get_distinct_stable_indices() {
        let events = run(&[
            "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>",
        ]);
        let calls = calls(&events);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].index, 1);
        assert_eq!(calls[1].name, "b");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn multiple_parameters_all_land_in_arguments() {
        let events = run(&[
            "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>",
        ]);
        let calls = calls(&events);
        assert_eq!(
            args(calls[0]),
            serde_json::json!({"path": "a.txt", "content": "hello"})
        );
    }

    #[test]
    fn a_multiline_parameter_value_keeps_its_interior_newlines() {
        let events = run(&[
            "<tool_call>\n<function=write_file>\n<parameter=content>\nline one\nline two\n</parameter>\n</function>\n</tool_call>",
        ]);
        let calls = calls(&events);
        assert_eq!(
            args(calls[0])["content"],
            serde_json::json!("line one\nline two")
        );
    }

    #[test]
    fn numeric_and_boolean_and_array_arguments_round_trip_typed() {
        let events = run(&[
            "<tool_call>\n<function=f>\n<parameter=n>\n42\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n<parameter=arr>\n[\"a\",\"b\"]\n</parameter>\n</function>\n</tool_call>",
        ]);
        let a = args(calls(&events)[0]);
        assert_eq!(a["n"], serde_json::json!(42));
        assert_eq!(a["b"], serde_json::json!(true));
        assert_eq!(a["arr"], serde_json::json!(["a", "b"]));
    }

    #[test]
    fn a_call_interrupted_before_it_closes_is_dropped_not_leaked() {
        // A stop/cancel landing mid-call (acceptance criterion 3): the
        // open tag and everything after it must never surface as content
        // or as a malformed call.
        let events = run(&["some text <tool_call>\n<function=read_file>\n<parameter=path>\na"]);
        assert_eq!(content(&events), "some text ");
        assert!(calls(&events).is_empty());
        for e in &events {
            if let ToolEvent::Content(text) = e {
                assert!(!text.contains('<'), "{text:?}");
            }
        }
    }

    #[test]
    fn a_malformed_block_that_never_opens_a_function_is_dropped() {
        let events = run(&["<tool_call>garbage, no function here</tool_call>after"]);
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "after");
    }

    #[test]
    fn is_mid_call_reports_true_only_while_a_call_is_open() {
        let mut scanner = ToolCallScanner::new();
        assert!(!scanner.is_mid_call());
        scanner.feed("<tool_call>\n<function=f>\n");
        assert!(scanner.is_mid_call());
        scanner.feed("</function>\n</tool_call>");
        assert!(!scanner.is_mid_call());
    }

    #[test]
    fn any_calls_is_false_until_the_first_call_closes() {
        let mut scanner = ToolCallScanner::new();
        assert!(!scanner.any_calls());
        scanner.feed("<tool_call>\n<function=f>\n</function>\n</tool_call>");
        assert!(scanner.any_calls());
    }

    #[test]
    fn streaming_and_whole_string_feeds_agree_by_construction() {
        let whole = "hi <tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call> bye";
        let streamed = run(&whole
            .as_bytes()
            .chunks(3)
            .map(|b| std::str::from_utf8(b).unwrap_or_default())
            .collect::<Vec<_>>());
        let batched = run(&[whole]);
        assert_eq!(content(&streamed), content(&batched));
        assert_eq!(calls(&streamed).len(), calls(&batched).len());
        assert_eq!(calls(&streamed)[0].arguments, calls(&batched)[0].arguments);
    }

    // ── schema-typed arguments and unclosed parameters ───────────────────

    fn run_with(tools: serde_json::Value, chunks: &[&str]) -> Vec<ToolEvent> {
        let tools: Vec<JsonValue> = serde_json::from_value(tools).unwrap();
        let mut scanner = ToolCallScanner::with_schemas(ToolSchemas::from_tools(&tools));
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(scanner.feed(chunk));
        }
        events.extend(scanner.finish());
        events
    }

    fn write_file_tool() -> serde_json::Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "line": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "force": {"type": "boolean"},
                        "tags": {"type": "array"},
                        "meta": {"type": "object"},
                        "note": {"type": ["string", "null"]},
                        "limit": {"type": ["integer", "null"]},
                        "mode": {"enum": ["a", "b"]},
                        "either": {"anyOf": [{"type": "integer"}, {"type": "boolean"}]},
                        "free": {}
                    }
                }
            }
        }])
    }

    fn one_param(param: &str, value: &str) -> String {
        format!("<tool_call>\n<function=write_file>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>")
    }

    fn typed(param: &str, value: &str) -> serde_json::Value {
        let events = run_with(write_file_tool(), &[&one_param(param, value)]);
        args(calls(&events)[0])[param].clone()
    }

    #[test]
    fn a_declared_string_keeps_json_looking_text_as_text() {
        assert_eq!(typed("content", "123"), serde_json::json!("123"));
        assert_eq!(typed("content", "{\"a\":1}"), serde_json::json!("{\"a\":1}"));
        assert_eq!(typed("content", "true"), serde_json::json!("true"));
        assert_eq!(typed("content", "[1, 2]"), serde_json::json!("[1, 2]"));
        assert_eq!(typed("content", "\"quoted\""), serde_json::json!("\"quoted\""));
        assert_eq!(typed("content", "null"), serde_json::json!("null"));
    }

    #[test]
    fn declared_scalar_types_are_parsed_as_that_type() {
        assert_eq!(typed("line", "42"), serde_json::json!(42));
        assert_eq!(typed("ratio", "0.5"), serde_json::json!(0.5));
        assert_eq!(typed("ratio", "3"), serde_json::json!(3));
        assert_eq!(typed("force", "true"), serde_json::json!(true));
        assert_eq!(typed("force", "False"), serde_json::json!(false));
        assert_eq!(typed("tags", "[\"a\",\"b\"]"), serde_json::json!(["a", "b"]));
        assert_eq!(typed("meta", "{\"k\": 1}"), serde_json::json!({"k": 1}));
    }

    #[test]
    fn a_value_that_disagrees_with_its_schema_is_kept_as_text() {
        assert_eq!(typed("line", "4.5"), serde_json::json!("4.5"));
        assert_eq!(typed("line", "forty"), serde_json::json!("forty"));
        assert_eq!(typed("force", "yes"), serde_json::json!("yes"));
        assert_eq!(typed("tags", "{\"not\": \"an array\"}"), serde_json::json!("{\"not\": \"an array\"}"));
        assert_eq!(typed("meta", "[1]"), serde_json::json!("[1]"));
    }

    #[test]
    fn a_nullable_type_reads_null_and_otherwise_its_other_type() {
        assert_eq!(typed("note", "null"), serde_json::Value::Null);
        assert_eq!(typed("note", "42"), serde_json::json!("42"));
        assert_eq!(typed("limit", "null"), serde_json::Value::Null);
        assert_eq!(typed("limit", "7"), serde_json::json!(7));
    }

    #[test]
    fn enum_and_any_of_members_supply_the_types() {
        assert_eq!(typed("mode", "a"), serde_json::json!("a"));
        assert_eq!(typed("either", "7"), serde_json::json!(7));
        assert_eq!(typed("either", "true"), serde_json::json!(true));
    }

    #[test]
    fn an_untyped_or_undeclared_parameter_keeps_the_schema_free_rule() {
        assert_eq!(typed("free", "42"), serde_json::json!(42));
        assert_eq!(typed("undeclared", "{\"x\":1}"), serde_json::json!({"x": 1}));
        assert_eq!(typed("undeclared", "plain text"), serde_json::json!("plain text"));
        // A function the tools do not name at all.
        let events = run_with(
            write_file_tool(),
            &["<tool_call>\n<function=other>\n<parameter=content>\n123\n</parameter>\n</function>\n</tool_call>"],
        );
        assert_eq!(args(calls(&events)[0])["content"], serde_json::json!(123));
    }

    #[test]
    fn schemas_ignore_tools_without_a_name_or_properties() {
        let tools: Vec<JsonValue> = serde_json::from_value(serde_json::json!([
            {"type": "function", "function": {"parameters": {"properties": {"x": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "f"}},
            {"type": "function", "function": {"name": "g", "parameters": {"type": "object"}}}
        ]))
        .unwrap();
        assert_eq!(ToolSchemas::from_tools(&tools), ToolSchemas::default());
    }

    #[test]
    fn an_unclosed_last_parameter_runs_to_the_end_of_the_function() {
        let events = run(&[
            "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nline one\nline two\n</function>\n</tool_call>",
        ]);
        assert_eq!(
            args(calls(&events)[0]),
            serde_json::json!({"path": "a.txt", "content": "line one\nline two"})
        );
    }

    #[test]
    fn an_unclosed_parameter_ends_where_the_next_one_starts() {
        let events = run(&[
            "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>",
        ]);
        assert_eq!(
            args(calls(&events)[0]),
            serde_json::json!({"path": "a.txt", "content": "hello"})
        );
    }

    #[test]
    fn an_unclosed_parameter_split_across_chunks_still_parses() {
        let events = run(&[
            "<tool_call>\n<function=f>\n<param",
            "eter=x>\n1\n</funct",
            "ion>\n</tool_call>",
        ]);
        assert_eq!(args(calls(&events)[0]), serde_json::json!({"x": 1}));
    }
}
