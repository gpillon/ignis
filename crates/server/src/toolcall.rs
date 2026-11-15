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
//! closed block whose body does not parse as `<function=...>`, is
//! discarded at the point it is discovered (the closed block is counted,
//! below) — never surfaced as partial XML
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
//!
//! **Every call block is checked and counted** (GitHub #608, ADR 0094) when
//! the scanner is given a [`CallCheck`] ([`ToolCallScanner::checking`]): a
//! call against its tool's `parameters` by the grammar crate's validator, a
//! call to a tool the request did not declare as invalid, and a closed block
//! that does not parse as dropped ([`CallVerdict`]). What the scanner emits
//! does not change.
//!
//! **Where a call, its body and its values end is one rule** (GitHub #610,
//! ADR 0094), [`Cuts`]: every tool keeps the first-occurrence cuts above,
//! and a **held tool** ([`ToolCallScanner::holding`]) the confirmed close
//! of `ignis_grammar::call`, the rule its grammar reads by: a value ends
//! only at a line that is a valid next step of the call, so a value may
//! hold the dialect's own tags. No tool is held before the grammar holds
//! its calls. With the check on, the values of a tool not held from which
//! the first-occurrence cuts drop text after a tag, text the confirmed close
//! would keep in the value, are counted ([`CallCheck`]).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use ignis_grammar::call;
use serde_json::Value as JsonValue;

/// The tool-call dialect's opening tag: one token on the Qwen tokenizers, which
/// a turn may write instead of a structured answer (GitHub #592).
pub(crate) const OPEN_TAG: &str = call::CALL_OPEN;
const CLOSE_TAG: &str = call::CALL_CLOSE;
const FUNCTION_PREFIX: &str = call::FUNCTION_OPEN;
const FUNCTION_CLOSE: &str = call::FUNCTION_CLOSE;
const PARAMETER_PREFIX: &str = call::PARAMETER_OPEN;
const PARAMETER_CLOSE: &str = call::PARAMETER_CLOSE;

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
    /// The turn's one call has closed
    /// ([`ToolCallScanner::ending_at_first_call`]): nothing more is read.
    Ended,
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
    /// The **held tools** (GitHub #611), by function name, with what the
    /// confirmed close knows of each, their parameters in the schema's
    /// order: set only beside the grammar that holds their calls
    /// ([`ToolSchemas::holding`]), so that every scanner of the response
    /// reads those calls as the grammar does.
    held: HashMap<String, call::Parameters>,
}

/// The call check of one response (GitHub #608, ADR 0094): every function
/// the request declares, with its `parameters`, and the metrics each call
/// block's [`CallVerdict`] is counted in. Made only with metrics on, so a
/// server without them copies and checks nothing.
#[derive(Debug)]
pub struct CallCheck {
    /// function name → its `parameters` as sent (`true` when it sent none),
    /// for every declared function, with properties or not.
    parameters: HashMap<String, JsonValue>,
    /// function name → what the confirmed close knows of it (#610), what a
    /// value it ends is weighed by.
    rules: HashMap<String, call::Parameters>,
    metrics: Arc<crate::metrics::Metrics>,
}

impl CallCheck {
    /// The check of a request whose resolved `tools` are these (none under
    /// `tool_choice: "none"`, where every call names an undeclared tool), or
    /// `None` without `metrics`.
    pub fn new(tools: &[JsonValue], metrics: Option<&Arc<crate::metrics::Metrics>>) -> Option<Self> {
        let metrics = Arc::clone(metrics?);
        let parameters = tools
            .iter()
            .filter_map(|tool| {
                let function = tool.get("function")?;
                let name = function.get("name").and_then(JsonValue::as_str)?;
                let parameters = function.get("parameters").filter(|p| !p.is_null()).cloned().unwrap_or(JsonValue::Bool(true));
                Some((name.to_owned(), parameters))
            })
            .collect::<HashMap<_, _>>();
        let rules = parameters.iter().map(|(name, schema)| (name.clone(), rule_parameters(schema))).collect();
        Some(Self { parameters, rules, metrics })
    }

    /// A call to `name` with `arguments` (the JSON the parser delivers)
    /// against the declared function's `parameters`, by
    /// `ignis_grammar::validate`.
    pub fn verdict(&self, name: &str, arguments: &str) -> CallVerdict {
        use ignis_grammar::validate::{violations, Kind};
        let Some(parameters) = self.parameters.get(name) else {
            return CallVerdict::Invalid(vec![CallViolation::UndeclaredTool]);
        };
        let json = ignis_grammar::json::Json::parse;
        let (Ok(schema), Ok(arguments)) = (json(&parameters.to_string()), json(arguments)) else {
            // Both are serde_json's own output: never reached.
            return CallVerdict::Valid;
        };
        let mut kinds: Vec<CallViolation> = violations(&schema, &arguments)
            .into_iter()
            .map(|violation| match violation.kind {
                Kind::Missing => CallViolation::Missing,
                Kind::Type => CallViolation::Type,
                Kind::Enum => CallViolation::Enum,
                Kind::Unknown => CallViolation::UnknownParameter,
                Kind::Constraint => CallViolation::Constraint,
            })
            .collect();
        kinds.sort_unstable();
        kinds.dedup();
        match kinds.is_empty() {
            true => CallVerdict::Valid,
            false => CallVerdict::Invalid(kinds),
        }
    }

    /// Count a closed block: the call it parsed as, or `None` when it did
    /// not parse.
    fn count(&self, parsed: Option<(&str, &str)>) {
        let verdict = match parsed {
            Some((name, arguments)) => self.verdict(name, arguments),
            None => CallVerdict::Dropped,
        };
        self.metrics.record_tool_call(&verdict);
    }

    /// Count the values of a call to a tool that is not held, read from
    /// `text` (the block, its `</tool_call>` included) by the
    /// first-occurrence cuts, from which those cuts dropped text after a tag
    /// inside them that the confirmed close would have kept in the value
    /// (GitHub #610): the measure of extending that rule to every tool. The
    /// dropped text is what the reading skipped (after a `</parameter>`, or
    /// between `</function>` and `</tool_call>`), not only whitespace, and
    /// lying before the value's end under the confirmed close. Only a value
    /// of a parameter the tool declares a string counts, the values the rule
    /// is for. Not counted: a value the confirmed close ends at no step of
    /// the block (a close with a required parameter missing, which that rule
    /// would not deliver at all, or an end past the block's `</tool_call>`);
    /// one whose text the cuts moved into another parameter, dropping
    /// nothing (an opener of a written or undeclared parameter, which may be
    /// the model's own envelope); one both rules end at the same place (the
    /// residue among them).
    fn count_split_values(&self, text: &str, block: &Block, schemas: &ToolSchemas) {
        let name = &text[block.name.clone()];
        let Some(parameters) = self.rules.get(name) else { return };
        let mut written = vec![false; parameters.declared().len()];
        let mut split = 0;
        for value in &block.values {
            let param = &text[value.name.clone()];
            if let Some(index) = parameters.index(param) {
                written[index] = true;
            }
            if !schemas.param_types(name, param).iter().any(|t| t == "string") {
                continue;
            }
            let start = value.text.start;
            let Some((end, _)) = call::value_end(&text.as_bytes()[start..], parameters, &written) else { continue };
            let kept = start..start + end;
            let dropped = block.skipped.iter().any(|gap| {
                let (from, to) = (gap.start.max(kept.start), gap.end.min(kept.end));
                from < to && text.as_bytes()[from..to].iter().any(|b| !b.is_ascii_whitespace())
            });
            if dropped {
                split += 1;
            }
        }
        if split > 0 {
            self.metrics.record_tool_call_values_split(split);
        }
    }
}

/// What the confirmed close knows of a tool whose `parameters` are these
/// (GitHub #610): the names its `properties` declare, and those its
/// `required` lists.
pub(crate) fn rule_parameters(parameters: &JsonValue) -> call::Parameters {
    let declared = parameters.get("properties").and_then(JsonValue::as_object).map(|p| p.keys().cloned().collect()).unwrap_or_else(Vec::new);
    let required = parameters.get("required").and_then(JsonValue::as_array).map(|r| r.iter().filter_map(JsonValue::as_str).collect()).unwrap_or_else(Vec::new);
    call::Parameters::new(declared, required)
}

/// What the call check found of one call block the scanner closed (GitHub
/// #608, ADR 0094).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallVerdict {
    /// The call names a declared tool, and its arguments validate against
    /// the tool's `parameters`.
    Valid,
    /// The call names a tool the request did not declare, or its arguments
    /// break the tool's `parameters`: each kind of violation once, in
    /// [`CallViolation::ALL`] order.
    Invalid(Vec<CallViolation>),
    /// A closed block that did not parse as a call, dropped as ever.
    Dropped,
}

/// The kinds of violation an invalid call is counted by: the validator's
/// kinds (`ignis_grammar::validate::Kind`) and the call's own, a tool the
/// request did not declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallViolation {
    /// A required parameter (or a required property inside one) is absent.
    Missing,
    /// A value of a type its schema does not admit.
    Type,
    /// A value outside its `enum`, or other than its `const`.
    Enum,
    /// A parameter (or a property inside one) the schema does not declare,
    /// where the schema closes the object (`additionalProperties: false`).
    /// An open object admits it: that call is valid.
    UnknownParameter,
    /// A value of the right type that breaks another keyword of its schema:
    /// `pattern`, `format`, a bound, a length, a count, `multipleOf`.
    Constraint,
    /// A call to a function the request's `tools` do not declare; its
    /// arguments are checked against nothing.
    UndeclaredTool,
}

impl CallViolation {
    pub const ALL: [CallViolation; 6] =
        [Self::Missing, Self::Type, Self::Enum, Self::UnknownParameter, Self::Constraint, Self::UndeclaredTool];

    /// The `kind` label of `ignis_tool_call_violations_total`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Type => "type",
            Self::Enum => "enum",
            Self::UnknownParameter => "unknown_parameter",
            Self::Constraint => "constraint",
            Self::UndeclaredTool => "undeclared_tool",
        }
    }
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
        Self { functions, held: HashMap::new() }
    }

    /// The same schemas, whose calls to these held tools (function name →
    /// what the confirmed close knows of it, its parameters declared in the
    /// schema's order) every scanner reads by the confirmed close
    /// ([`ToolCallScanner::holding`]), delivering their arguments in the
    /// schema's order (GitHub #611). For a request whose content the tool
    /// calls' grammar holds, and only then.
    pub fn holding(self, held: HashMap<String, call::Parameters>) -> Self {
        Self { held, ..self }
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
    /// Where each closed block is checked and counted: only a scanner that
    /// delivers a response's calls has one.
    check: Option<CallCheck>,
    /// The scan ends at its first call's close (`parallel_tool_calls:
    /// false`).
    ends_at_first_call: bool,
    /// The held tools, by function name, whose calls end by the confirmed
    /// close ([`ToolCallScanner::holding`]).
    held: HashMap<String, call::Parameters>,
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

    /// A scanner that types each argument by the request's `tools`, and
    /// reads the calls of the held tools they carry
    /// ([`ToolSchemas::holding`]) by the confirmed close.
    pub fn with_schemas(schemas: ToolSchemas) -> Self {
        Self {
            state: State::Scanning,
            hold: String::new(),
            next_index: 0,
            held: schemas.held.clone(),
            schemas,
            check: None,
            ends_at_first_call: false,
        }
    }

    /// Read the calls to these held tools (function name → what the
    /// confirmed close knows of it) by the confirmed close (GitHub #610,
    /// ADR 0094) instead of the first-occurrence cuts: such a call ends only
    /// at `</function>` and `</tool_call>` once every required parameter is
    /// written, and a value only at a line that is a valid next step, so a
    /// call that closes with a required parameter missing stays open. For a
    /// call the grammar holds to its tool's schema; no caller holds a tool
    /// before it does.
    pub fn holding(self, held: HashMap<String, call::Parameters>) -> Self {
        Self { held, ..self }
    }

    /// With `on`, the scan ends where its first call block closes, parsed
    /// or dropped (`parallel_tool_calls: false`, GitHub #609): that close
    /// ends the turn, and nothing fed after it is read, a second call
    /// already in the same text included. Where a block closes is this
    /// scanner's to say, so a `</tool_call>` its caller never feeds it (the
    /// reasoning's) ends nothing, and a change to where a call closes is a
    /// change to when the turn ends.
    pub fn ending_at_first_call(self, on: bool) -> Self {
        Self { ends_at_first_call: on, ..self }
    }

    /// Whether the scan ended at its first call's close
    /// ([`ToolCallScanner::ending_at_first_call`]): the caller stops the
    /// generation there.
    pub fn ended(&self) -> bool {
        matches!(self.state, State::Ended)
    }

    /// Check every call block this scanner closes by `check` (GitHub #608):
    /// what a response's own scanner does, never a copy of it (the
    /// first-token probe of #518) or a history's round trip.
    pub fn checking(mut self, check: Option<CallCheck>) -> Self {
        self.check = check;
        self
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
        if self.ended() {
            return out;
        }
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
                    let (len, block, held) = {
                        let cuts = self.cuts();
                        let Some((len, block)) = cuts.read_call(&self.hold) else {
                            break; // still incomplete; wait for more text
                        };
                        (len, block, matches!(cuts, Cuts::Confirmed(_)))
                    };
                    let body: String = self.hold.drain(..len).collect();
                    let parsed = block.as_ref().map(|block| {
                        let name = &body[block.name.clone()];
                        let order = self.held.get(name).filter(|_| held);
                        (name.to_string(), arguments(&body, block, &self.schemas, order))
                    });
                    if let Some(check) = &self.check {
                        check.count(parsed.as_ref().map(|(name, arguments)| (name.as_str(), arguments.as_str())));
                        if let (Some(block), false) = (&block, held) {
                            check.count_split_values(&body, block, &self.schemas);
                        }
                    }
                    if let Some((name, arguments)) = parsed {
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
                    // `<function=...>` is dropped — same never-leak-garbage
                    // rule as an interrupted one — and counted `dropped`.
                    self.state = State::Scanning;
                    if self.ends_at_first_call {
                        self.state = State::Ended;
                        self.hold.clear();
                        break;
                    }
                    continue;
                }
                State::Ended => break,
            }
        }
        out
    }

    /// Flush whatever is left at the end of generation.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        match self.state {
            State::Ended => vec![],
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

/// Where the call parser ends a call, its function body and each value
/// (GitHub #610, ADR 0094): one rule, chosen per call by the function it
/// names.
#[derive(Debug, Clone, Copy)]
enum Cuts<'a> {
    /// The rule of every tool that is not held: the call at its
    /// `</tool_call>`'s first occurrence, the body at its `</function>`'s, a
    /// value at its `</parameter>`'s, or at the next `<parameter=` when that
    /// comes first (a forgotten closer).
    FirstOccurrence,
    /// A held tool's: the confirmed close, `ignis_grammar::call::line`.
    Confirmed(&'a call::Parameters),
}

/// One call block as read: where its function's name and each value's
/// parameter name and raw text lie in the text it was read from, values in
/// the order written, a parameter written twice included, and the text the
/// reading skipped.
#[derive(Debug)]
struct Block {
    name: Range<usize>,
    values: Vec<Value>,
    /// What lies in no value and is no tag the reading took: before the
    /// first opener, after a `</parameter>` up to the next one, and between
    /// `</function>` and `</tool_call>`. The first-occurrence cuts drop it.
    skipped: Vec<Range<usize>>,
}

#[derive(Debug)]
struct Value {
    name: Range<usize>,
    /// Untrimmed: the newlines around it included.
    text: Range<usize>,
}

impl ToolCallScanner {
    /// The rule of the call the hold buffers: the confirmed close once it
    /// names a held tool (a `</tool_call>` always follows the name's `>`, so
    /// the name is read before the call can close).
    fn cuts(&self) -> Cuts<'_> {
        if self.held.is_empty() {
            return Cuts::FirstOccurrence;
        }
        let named = self.hold.trim_start().strip_prefix(FUNCTION_PREFIX).and_then(|rest| rest.find('>').map(|end| &rest[..end]));
        match named.and_then(|name| self.held.get(name)) {
            Some(parameters) => Cuts::Confirmed(parameters),
            None => Cuts::FirstOccurrence,
        }
    }
}

impl Cuts<'_> {
    /// The call in `text` (what follows its `<tool_call>`), once it has
    /// closed: the length of the block, its `</tool_call>` included, and
    /// the block, or `None` for a block that does not parse as a call.
    fn read_call(self, text: &str) -> Option<(usize, Option<Block>)> {
        match self {
            Cuts::FirstOccurrence => {
                let idx = text.find(CLOSE_TAG)?;
                Some((idx + CLOSE_TAG.len(), read_function_block(&text[..idx])))
            }
            // The close ends at a `</tool_call>`: no need to read before one.
            Cuts::Confirmed(parameters) if text.contains(CLOSE_TAG) => {
                read_held_call(text, parameters).map(|(len, block)| (len, Some(block)))
            }
            Cuts::Confirmed(_) => None,
        }
    }
}

/// Reads `<function=NAME>...<parameter=P>VALUE</parameter>...</function>`
/// (the text between `<tool_call>` and `</tool_call>`, both already
/// stripped) by the first-occurrence cuts. Returns `None` if the block does
/// not open with `<function=` or never closes with `</function>`.
fn read_function_block(inner: &str) -> Option<Block> {
    let trimmed = inner.trim_start();
    let after_prefix = trimmed.strip_prefix(FUNCTION_PREFIX)?;
    let base = inner.len() - trimmed.len() + FUNCTION_PREFIX.len();
    let name_end = after_prefix.find('>')?;
    let name = base..base + name_end;
    let body_start = name.end + 1;
    let body = inner[body_start..].find(FUNCTION_CLOSE).map(|end| &inner[body_start..body_start + end])?;

    let mut values = Vec::new();
    let mut skipped = Vec::new();
    let mut rest = 0;
    while let Some(start) = body[rest..].find(PARAMETER_PREFIX) {
        skipped.push(body_start + rest..body_start + rest + start);
        let after = rest + start + PARAMETER_PREFIX.len();
        let Some(name_end) = body[after..].find('>') else {
            rest += start;
            break; // malformed trailing parameter tag: stop, keep what we have
        };
        let value_start = after + name_end + 1;
        let after_name = &body[value_start..];
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
        values.push(Value {
            name: body_start + after..body_start + after + name_end,
            text: body_start + value_start..body_start + value_start + value_end,
        });
        rest = value_start + resume;
    }
    skipped.push(body_start + rest..body_start + body.len());
    skipped.push(body_start + body.len() + FUNCTION_CLOSE.len()..inner.len());
    Some(Block { name, values, skipped })
}

/// Reads a held tool's call from `text` (what follows its `<tool_call>`)
/// by the confirmed close: the length of the block, its `</tool_call>`
/// included, and the block, or `None` while the call has not closed. What
/// precedes the first parameter's opener is not read.
fn read_held_call(text: &str, parameters: &call::Parameters) -> Option<(usize, Block)> {
    let trimmed = text.trim_start();
    let base = text.len() - trimmed.len() + FUNCTION_PREFIX.len();
    let name_end = trimmed.strip_prefix(FUNCTION_PREFIX)?.find('>')?;
    let name = base..base + name_end;
    let mut written = vec![false; parameters.declared().len()];
    let mut values = Vec::new();
    // The value being read, by its parameter's name, and where it starts.
    let mut open: Option<Range<usize>> = None;
    let mut at = name.end + 1;
    loop {
        let (end, step) = call::value_end(&text.as_bytes()[at..], parameters, &written)?;
        if let Some(name) = open.take() {
            values.push(Value { name, text: at..at + end });
        }
        match step {
            call::Step::Parameter { index, len } => {
                written[index] = true;
                let after = at + end + len;
                open = Some(after - 1 - parameters.declared()[index].len()..after - 1);
                at = after;
            }
            call::Step::Close { len } => return Some((at + end + len, Block { name, values, skipped: Vec::new() })),
        }
    }
}

/// The JSON arguments of `block`, read from `text`: each value trimmed of
/// one newline either side and typed by `schemas`; a parameter written
/// twice keeps its last value; keys sorted. For a held tool's call
/// (`order`: its parameters, declared in the schema's order), in the
/// schema's order (GitHub #611), each value read as the grammar held it
/// (its `call::Reading`), and a parameter left out that is delivered `null`
/// when omitted (a strict tool's nullable one) as `null` (GitHub #612).
fn arguments(text: &str, block: &Block, schemas: &ToolSchemas, order: Option<&call::Parameters>) -> String {
    let name = &text[block.name.clone()];
    let Some(order) = order else {
        let mut args = serde_json::Map::new();
        for value in &block.values {
            let param = &text[value.name.clone()];
            let types = schemas.param_types(name, param);
            args.insert(param.to_string(), parse_argument_value(trim_one_newline(&text[value.text.clone()]), types));
        }
        return serde_json::to_string(&serde_json::Value::Object(args)).expect("a Map<String, Value> always serializes");
    };
    // Compact JSON, a key at a time: no map here keeps an order.
    let written = |key: &str| block.values.iter().rev().find(|value| &text[value.name.clone()] == key).map(|value| trim_one_newline(&text[value.text.clone()]));
    let fields: Vec<String> = order
        .declared()
        .iter()
        .enumerate()
        .filter_map(|(index, key)| {
            let value = match written(key) {
                Some(value) => order.reading_of(index).read(value),
                None if order.nulled_when_omitted(index) => "null".to_owned(),
                None => return None,
            };
            Some(format!("{}:{}", JsonValue::String(key.clone()), value))
        })
        .collect();
    format!("{{{}}}", fields.join(","))
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

    /// The tools of the call-check tests (#608): a closed `edit` with a
    /// required path, an integer line and an enum mode, and a `note` with no
    /// properties at all.
    fn checked_tools() -> Vec<JsonValue> {
        serde_json::from_value(serde_json::json!([
            {"type": "function", "function": {"name": "edit", "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "pattern": "^[^ ]+$"},
                    "line": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["replace", "append"]}
                },
                "required": ["path"],
                "additionalProperties": false
            }}},
            {"type": "function", "function": {"name": "note", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "ping"}}
        ]))
        .unwrap()
    }

    #[test]
    fn a_call_check_names_each_kind_of_violation() {
        let metrics = Arc::new(crate::metrics::Metrics::new());
        let check = CallCheck::new(&checked_tools(), Some(&metrics)).unwrap();
        let invalid = |kinds: &[CallViolation]| CallVerdict::Invalid(kinds.to_vec());
        assert_eq!(check.verdict("edit", r#"{"path":"a.txt","line":3,"mode":"append"}"#), CallVerdict::Valid);
        assert_eq!(check.verdict("edit", r#"{"line":3}"#), invalid(&[CallViolation::Missing]));
        assert_eq!(check.verdict("edit", r#"{"path":"a.txt","line":"three"}"#), invalid(&[CallViolation::Type]));
        assert_eq!(check.verdict("edit", r#"{"path":"a.txt","mode":"Replace"}"#), invalid(&[CallViolation::Enum]));
        assert_eq!(check.verdict("edit", r#"{"path":"a.txt","force":true}"#), invalid(&[CallViolation::UnknownParameter]));
        assert_eq!(check.verdict("edit", r#"{"path":"a b"}"#), invalid(&[CallViolation::Constraint]));
        assert_eq!(
            check.verdict("edit", r#"{"file":"a.txt","line":1.5}"#),
            invalid(&[CallViolation::Missing, CallViolation::Type, CallViolation::UnknownParameter])
        );
        assert_eq!(check.verdict("search", r#"{"path":"a.txt"}"#), invalid(&[CallViolation::UndeclaredTool]));
        // A tool without properties, or without parameters, is declared, and
        // its open object admits any parameter.
        assert_eq!(check.verdict("note", r#"{"text":"hi"}"#), CallVerdict::Valid);
        assert_eq!(check.verdict("ping", r#"{"anything":[1]}"#), CallVerdict::Valid);
        // `tool_choice: "none"` declares nothing.
        let none = CallCheck::new(&[], Some(&metrics)).unwrap();
        assert_eq!(none.verdict("edit", r#"{"path":"a.txt"}"#), invalid(&[CallViolation::UndeclaredTool]));
        assert!(CallCheck::new(&checked_tools(), None).is_none(), "no metrics, no check");
    }

    #[test]
    fn a_checking_scanner_counts_every_block_it_closes_and_emits_what_it_did() {
        let chunks = [
            "before <tool_call>\n<function=edit>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=edit>\n<parameter=line>\nten\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\nno function here\n</tool_call> between ",
            "<tool_call>\n<function=search>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=edit>\n<parameter=path>\nb.txt\n</paramet",
        ];
        let metrics = Arc::new(crate::metrics::Metrics::new());
        let tools = checked_tools();
        let mut checked = ToolCallScanner::with_schemas(ToolSchemas::from_tools(&tools)).checking(CallCheck::new(&tools, Some(&metrics)));
        let mut plain = ToolCallScanner::with_schemas(ToolSchemas::from_tools(&tools));
        let (mut with, mut without) = (Vec::new(), Vec::new());
        for chunk in chunks {
            with.extend(checked.feed(chunk));
            without.extend(plain.feed(chunk));
        }
        with.extend(checked.finish());
        without.extend(plain.finish());
        assert_eq!(with, without, "the check changes nothing the scanner emits");
        assert_eq!(calls(&with).len(), 3);
        let text = metrics.render();
        for line in [
            "ignis_tool_calls_total{outcome=\"valid\"} 1",
            "ignis_tool_calls_total{outcome=\"invalid\"} 2",
            // The block left open at the end is a cut, not a dropped block.
            "ignis_tool_calls_total{outcome=\"dropped\"} 1",
            "ignis_tool_call_violations_total{kind=\"missing\"} 1",
            "ignis_tool_call_violations_total{kind=\"type\"} 1",
            "ignis_tool_call_violations_total{kind=\"undeclared_tool\"} 1",
            "ignis_tool_call_violations_total{kind=\"enum\"} 0",
        ] {
            assert!(text.contains(&format!("\n{line}\n")), "no `{line}` in:\n{text}");
        }
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

    /// `chunks` through a scanner that ends at its first call's close
    /// (`parallel_tool_calls: false`, GitHub #609); whether it ended.
    fn run_to_first_call(chunks: &[&str]) -> (Vec<ToolEvent>, bool) {
        let mut scanner = ToolCallScanner::new().ending_at_first_call(true);
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(scanner.feed(chunk));
        }
        let ended = scanner.ended();
        assert!(!(ended && scanner.is_mid_call()), "an ended scan holds no open call");
        events.extend(scanner.finish());
        (events, ended)
    }

    const FIRST: &str = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";
    const SECOND: &str = "<tool_call>\n<function=b>\n</function>\n</tool_call>";

    #[test]
    fn a_scan_ending_at_its_first_call_reads_nothing_after_that_calls_close() {
        let one_chunk = format!("before {FIRST}\n{SECOND}\nafter");
        let (events, ended) = run_to_first_call(&[&one_chunk]);
        assert!(ended);
        assert_eq!(calls(&events).len(), 1, "{events:?}");
        assert_eq!(calls(&events)[0].name, "a");
        assert_eq!(content(&events), "before ", "nothing after the close is content");

        let (events, ended) = run_to_first_call(&["before <tool_", &FIRST[6..], "\n<tool_call>\n", SECOND, "after"]);
        assert!(ended);
        assert_eq!(calls(&events).iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["a"]);
        assert_eq!(content(&events), "before ");
    }

    #[test]
    fn a_scan_ending_at_its_first_call_ends_at_a_dropped_block_too() {
        // The first block closes and does not parse: the turn's call was
        // written, and the one after it is not read.
        let (events, ended) = run_to_first_call(&["<tool_call>\nno function here\n</tool_call>", SECOND]);
        assert!(ended);
        assert!(calls(&events).is_empty(), "{events:?}");
        assert_eq!(content(&events), "");
    }

    #[test]
    fn a_scan_ending_at_its_first_call_reads_text_with_no_call_to_its_end() {
        let (events, ended) = run_to_first_call(&["no ", "call <tool_", "call> left open"]);
        assert!(!ended);
        assert_eq!(content(&events), "no call ");
        let (events, ended) = run_to_first_call(&["just ", "text"]);
        assert!(!ended);
        assert_eq!(content(&events), "just text");
    }

    // ── where a call, its body and its values end (#610) ─────────────────

    /// `parse_function_block` as it read before #610, kept verbatim: what
    /// the first-occurrence cuts must keep reading.
    fn parse_function_block_before_610(inner: &str, schemas: &ToolSchemas) -> Option<(String, String)> {
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

    /// The scanner as it read before #610, over a whole text: its content
    /// and its calls (name, arguments).
    fn scan_before_610(text: &str, schemas: &ToolSchemas, first_call_only: bool) -> (String, Vec<(String, String)>) {
        let (mut content, mut calls, mut rest) = (String::new(), Vec::new(), text);
        while let Some(open) = rest.find(OPEN_TAG) {
            content.push_str(&rest[..open]);
            let after = &rest[open + OPEN_TAG.len()..];
            let Some(close) = after.find(CLOSE_TAG) else {
                return (content, calls);
            };
            calls.extend(parse_function_block_before_610(&after[..close], schemas));
            rest = &after[close + CLOSE_TAG.len()..];
            if first_call_only {
                return (content, calls);
            }
        }
        content.push_str(rest);
        (content, calls)
    }

    /// A small xorshift: the tag soups below are the same on every run.
    fn generator(mut seed: u64) -> impl FnMut() -> u64 {
        move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        }
    }

    /// Text made of the dialect's tags, names, values and stray brackets.
    fn tag_soup(next: &mut impl FnMut() -> u64) -> String {
        const PIECES: [&str; 24] = [
            "<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>", "edit", "path",
            "content", "line", "x", ">", "\n", "\n", " ", "<", "/", "=", "é", "42", "true", "{\"k\":1}", "\r\n", "a b",
        ];
        (0..next() % 48).map(|_| PIECES[next() as usize % PIECES.len()]).collect()
    }

    fn soup_tools() -> Vec<JsonValue> {
        serde_json::from_value(serde_json::json!([
            {"type": "function", "function": {"name": "edit", "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "line": {"type": "integer"}},
                "required": ["path"]
            }}},
            {"type": "function", "function": {"name": "x"}}
        ]))
        .unwrap()
    }

    /// `text` in random chunks, cut at character boundaries.
    fn chunked<'a>(text: &'a str, next: &mut impl FnMut() -> u64) -> Vec<&'a str> {
        let mut chunks = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut cut = (next() as usize % 7 + 1).min(rest.len());
            while !rest.is_char_boundary(cut) {
                cut += 1;
            }
            chunks.push(&rest[..cut]);
            rest = &rest[cut..];
        }
        chunks
    }

    /// Every tool keeps the first-occurrence cuts: on generated tag soup,
    /// fed in random chunks, checked or not, ending at the first call or
    /// not, the scanner emits what it emitted before #610.
    #[test]
    fn the_first_occurrence_cuts_read_every_text_as_before() {
        let tools = soup_tools();
        let schemas = ToolSchemas::from_tools(&tools);
        let metrics = Arc::new(crate::metrics::Metrics::new());
        let mut next = generator(0x9e37_79b9_7f4a_7c15);
        let mut calls_seen = 0;
        for round in 0..20_000 {
            let mut text = tag_soup(&mut next);
            if round % 2 == 0 {
                // Half the texts sit inside a whole call, so most of them parse.
                text = format!("<tool_call>\n<function=edit>\n<parameter=path>\n{text}\n</parameter>\n</function>\n</tool_call>");
            }
            let first_call_only = round % 3 == 0;
            let check = (round % 5 != 0).then(|| CallCheck::new(&tools, Some(&metrics)).unwrap());
            let mut scanner = ToolCallScanner::with_schemas(schemas.clone()).ending_at_first_call(first_call_only).checking(check);
            let mut events = Vec::new();
            for chunk in chunked(&text, &mut next) {
                events.extend(scanner.feed(chunk));
            }
            events.extend(scanner.finish());
            let got: Vec<(String, String)> = calls(&events).iter().map(|c| (c.name.clone(), c.arguments.clone())).collect();
            calls_seen += got.len();
            assert_eq!((content(&events), got), scan_before_610(&text, &schemas, first_call_only), "{text:?}");
        }
        assert!(calls_seen > 5_000, "the soups parse as calls often enough: {calls_seen}");
    }

    /// The tool of the confirmed-close tests: `path` and `content`
    /// required, `mode` optional, `line` an integer.
    fn write_file_tools() -> Vec<JsonValue> {
        serde_json::from_value(serde_json::json!([
            {"type": "function", "function": {"name": "write_file", "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string", "enum": ["create", "append"]},
                    "line": {"type": "integer"}
                },
                "required": ["path", "content"]
            }}},
            {"type": "function", "function": {"name": "read", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]
            }}}
        ]))
        .unwrap()
    }

    /// A scanner over [`write_file_tools`] holding `held` of them.
    fn holding(held: &[&str]) -> ToolCallScanner {
        let tools = write_file_tools();
        let held = tools
            .iter()
            .map(|tool| &tool["function"])
            .filter(|function| held.contains(&function["name"].as_str().unwrap()))
            .map(|function| (function["name"].as_str().unwrap().to_owned(), rule_parameters(&function["parameters"])))
            .collect();
        ToolCallScanner::with_schemas(ToolSchemas::from_tools(&tools)).holding(held)
    }

    fn scan(mut scanner: ToolCallScanner, chunks: &[&str]) -> Vec<ToolEvent> {
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(scanner.feed(chunk));
        }
        events.extend(scanner.finish());
        events
    }

    /// A file about tool calls: its content holds a `</parameter>` line
    /// followed by text, openers of a written and of an undeclared
    /// parameter, a `</function>` line not followed by `</tool_call>`, and
    /// a `</tool_call>` line.
    const FILE_ABOUT_CALLS: &str = "line one\n</parameter>\n<parameter=path>\nother.md\n<parameter=force>\n</function>\nnot yet\n</tool_call>\nlast line";

    fn write_call(content: &str) -> String {
        format!("before <tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n<parameter=content>\n{content}\n</parameter>\n</function>\n</tool_call> after")
    }

    #[test]
    fn a_held_calls_value_keeps_the_dialects_own_tags_however_the_text_arrives() {
        let text = write_call(FILE_ABOUT_CALLS);
        let expected = serde_json::json!({"path": "notes.md", "content": FILE_ABOUT_CALLS});
        let whole = scan(holding(&["write_file"]), &[&text]);
        assert_eq!(content(&whole), "before  after");
        assert_eq!(calls(&whole).len(), 1, "{whole:?}");
        assert_eq!(args(calls(&whole)[0]), expected);
        for cut in 1..text.len() {
            let events = scan(holding(&["write_file"]), &[&text[..cut], &text[cut..]]);
            assert_eq!(events.iter().filter(|e| matches!(e, ToolEvent::Call(_))).count(), 1, "cut at {cut}");
            assert_eq!(args(calls(&events)[0]), expected, "cut at {cut}");
            assert_eq!(content(&events), "before  after", "cut at {cut}");
        }
        let bytes: Vec<String> = text.chars().map(String::from).collect();
        let events = scan(holding(&["write_file"]), &bytes.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(args(calls(&events)[0]), expected);
        // The same text, not held: cut at the first `</tool_call>`, as ever.
        let free = scan(holding(&[]), &[&text]);
        assert_eq!(calls(&free).len(), 1);
        assert_eq!(args(calls(&free)[0])["content"], "line one");
        assert!(content(&free).contains("last line"), "{free:?}");
    }

    #[test]
    fn a_held_call_closing_with_a_required_parameter_missing_stays_open() {
        let early = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>";
        let mut scanner = holding(&["write_file"]);
        assert!(scanner.feed(early).is_empty());
        assert!(scanner.is_mid_call(), "not a close");
        assert!(scanner.finish().is_empty(), "left open at the end: dropped, never leaked");
        // What follows is still the value, up to the missing parameter's opener.
        let late = "\n<parameter=content>\nx\n</parameter>\n</function>\n</tool_call>";
        let events = scan(holding(&["write_file"]), &[early, late]);
        assert_eq!(
            args(calls(&events)[0]),
            serde_json::json!({"path": "a.txt\n</parameter>\n</function>\n</tool_call>", "content": "x"})
        );
    }

    #[test]
    fn a_held_calls_value_cannot_hold_its_own_confirmed_close() {
        let text = write_call("an example:\n</function>\n</tool_call>\nand the rest");
        let events = scan(holding(&["write_file"]), &[&text]);
        assert_eq!(args(calls(&events)[0]), serde_json::json!({"path": "notes.md", "content": "an example:"}));
        assert_eq!(content(&events), "before \nand the rest\n</parameter>\n</function>\n</tool_call> after");
    }

    #[test]
    fn a_held_call_tolerates_forgotten_closers_and_takes_its_parameters_in_any_order() {
        let text = "<tool_call>\n<function=write_file>\n<parameter=content>\nhi\n<parameter=mode>\nappend\n<parameter=path>\na.txt\n</function>\n</tool_call>";
        let events = scan(holding(&["write_file"]), &[text]);
        assert_eq!(args(calls(&events)[0]), serde_json::json!({"content": "hi", "mode": "append", "path": "a.txt"}));
    }

    #[test]
    fn only_the_held_tools_calls_read_by_the_confirmed_close() {
        // `read` is not held: its value is cut at the first `</parameter>`.
        let free = "<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\nb\n</parameter>\n</function>\n</tool_call>";
        let held = write_call("a\n</parameter>\nb");
        let events = scan(holding(&["write_file"]), &[free, &held]);
        let calls = calls(&events);
        assert_eq!(args(calls[0]), serde_json::json!({"path": "a"}));
        assert_eq!(args(calls[1])["content"], "a\n</parameter>\nb");
        assert_eq!((calls[0].index, calls[1].index), (0, 1));
    }

    #[test]
    fn a_scan_ending_at_its_first_call_ends_at_a_held_calls_confirmed_close() {
        let text = format!("{}\n{SECOND}", write_call(FILE_ABOUT_CALLS));
        let mut scanner = holding(&["write_file"]).ending_at_first_call(true);
        let events = scanner.feed(&text);
        assert!(scanner.ended());
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(args(calls(&events)[0])["content"], FILE_ABOUT_CALLS);
    }

    /// The counts of `chunks` through a checking scanner over
    /// [`write_file_tools`], none held: the values split, and the calls.
    fn split_values(chunks: &[&str]) -> (String, String) {
        let tools = write_file_tools();
        let metrics = Arc::new(crate::metrics::Metrics::new());
        let scanner = holding(&[]).checking(CallCheck::new(&tools, Some(&metrics)));
        let events = scan(scanner, chunks);
        let text = metrics.render();
        let value = |series: &str| {
            text.lines().find_map(|line| line.strip_prefix(series).map(|v| v.trim().to_owned())).unwrap_or_else(|| "absent".into())
        };
        (value("ignis_tool_call_values_split_total "), format!("{} calls", calls(&events).len()))
    }

    #[test]
    fn a_value_whose_text_the_cuts_dropped_is_counted_split() {
        let split = |content: &str| split_values(&[&write_call(content)]);
        // Text after a `</parameter>` line, after one inside a line, and
        // after a `</function>` line not followed by `</tool_call>`: dropped
        // by the cuts, kept by the confirmed close. One each.
        assert_eq!(split("a\n</parameter>\nb"), ("1".into(), "1 calls".into()));
        assert_eq!(split("a</parameter>b"), ("1".into(), "1 calls".into()));
        assert_eq!(split("a\n</function>\nb"), ("1".into(), "1 calls".into()));
        // Split across chunks, counted once.
        let text = write_call("a\n</parameter>\nb");
        assert_eq!(split_values(&text.as_bytes().chunks(5).map(|c| std::str::from_utf8(c).unwrap()).collect::<Vec<_>>()).0, "1");
    }

    #[test]
    fn a_value_the_cuts_dropped_nothing_of_or_the_confirmed_close_never_ends_is_not_counted() {
        let none = ("0".to_owned(), "1 calls".to_owned());
        // A plain call, and one with forgotten closers.
        assert_eq!(split_values(&[&write_call("plain")]), none);
        let forgotten = "<tool_call>\n<function=write_file>\n<parameter=path>\na\n<parameter=content>\nb\n</function>\n</tool_call>";
        assert_eq!(split_values(&[forgotten]), none);
        // Text the cuts moved into another parameter, dropping nothing: an
        // opener of a written parameter, one inside a line, an undeclared
        // parameter after a closed value, and values closed on their own
        // line. The confirmed close would keep the opener in the value, but
        // it may be the model's own envelope.
        assert_eq!(split_values(&[&write_call("a\n<parameter=path>\nb")]), none);
        assert_eq!(split_values(&[&write_call("see <parameter=mode> here")]), none);
        let invented = "<tool_call>\n<function=write_file>\n<parameter=path>\na\n</parameter>\n<parameter=force>\ntrue\n</parameter>\n<parameter=content>\nb\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(split_values(&[invented]), none);
        let inline = "<tool_call>\n<function=write_file>\n<parameter=path>a.txt</parameter>\n<parameter=content>b</parameter>\n</function>\n</tool_call>";
        assert_eq!(split_values(&[inline]), none);
        // The residue: the value holds its call's own confirmed close, where both rules end it.
        assert_eq!(split_values(&[&write_call("x\n</function>\n</tool_call>\ny")]).0, "0");
        // A close with a required parameter missing: the confirmed close would not close at all.
        let early = "<tool_call>\n<function=write_file>\n<parameter=content>\na\n</parameter>\nb\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(split_values(&[early]), none);
        // A value cut at a tag whose confirmed end lies past the block's `</tool_call>`: not seen.
        assert_eq!(split_values(&[&write_call("a\n</parameter>\nb\n</tool_call>\nc")]).0, "0");
        // Only declared strings count: an integer, an undeclared parameter, an undeclared tool.
        let typed = "<tool_call>\n<function=write_file>\n<parameter=line>\n1\n</parameter>\nx\n<parameter=path>\na\n</parameter>\n<parameter=content>\nb\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(split_values(&[typed]), none);
        let undeclared = "<tool_call>\n<function=search>\n<parameter=q>\na\n</parameter>\nb\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(split_values(&[undeclared]), none);
        // No call, no family.
        assert_eq!(split_values(&["just text"]), ("absent".into(), "0 calls".into()));
    }

    #[test]
    fn a_scan_that_does_not_end_at_its_first_call_reads_every_call() {
        let mut scanner = ToolCallScanner::new().ending_at_first_call(false);
        let events = scanner.feed(&format!("{FIRST}\n{SECOND}"));
        assert!(!scanner.ended());
        assert_eq!(calls(&events).len(), 2);
    }

    // ── #611: what the grammar holds is what the parser delivers ─────────

    /// The held tools as sent: `write_file` (two free strings and an enum)
    /// and `get_time` (a pattern and a format), strict; `search`, free.
    const HELD_TOOLS: &str = r#"[
        {"type":"function","function":{"name":"write_file","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string","enum":["w","a"]}},"required":["path","content","mode"],"additionalProperties":false}}},
        {"type":"function","function":{"name":"get_time","strict":true,"parameters":{"type":"object","properties":{"zone":{"type":"string","pattern":"^[A-Z][a-z]+/[A-Z][a-z]+$"},"day":{"type":"string","format":"date"}},"required":["zone","day"],"additionalProperties":false}}},
        {"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}
    ]"#;

    /// `edit`, strict and typed (#612): an object holding a list of edits,
    /// an integer with bounds and one without, a boolean, a nested array of
    /// objects, a recursive definition, and two nullable parameters, a
    /// string and a bounded integer; `search`, free, beside it.
    const TYPED_TOOLS: &str = r##"[
        {"type":"function","function":{"name":"edit","strict":true,"parameters":{"type":"object","properties":{"change":{"type":"object","properties":{"edits":{"type":"array","items":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},"required":["old","new"],"additionalProperties":false}}},"required":["edits"],"additionalProperties":false},"line":{"type":"integer","minimum":1,"maximum":1000},"id":{"type":"integer"},"dry_run":{"type":"boolean"},"rows":{"type":"array","items":{"type":"array","items":{"type":"object","properties":{"k":{"type":"string","enum":["a","b"]},"v":{"type":"number"}},"required":["k","v"],"additionalProperties":false}}},"tree":{"$ref":"#/$defs/node"},"note":{"type":["string","null"]},"limit":{"anyOf":[{"type":"integer","minimum":0,"maximum":9},{"type":"null"}]}},"required":["change","line","id","dry_run","rows","tree","note","limit"],"additionalProperties":false,"$defs":{"node":{"type":"object","properties":{"name":{"type":"string"},"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["name","kids"],"additionalProperties":false}}}}},
        {"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}
    ]"##;

    /// The tools' grammar over a vocabulary of single bytes, a scanner that
    /// reads their calls as a held request's do, and each strict tool's
    /// `parameters` for the validator.
    fn held_parts() -> (std::sync::Arc<ignis_grammar::ToolCallGrammar>, impl Fn() -> ToolCallScanner, HashMap<String, ignis_grammar::json::Json>) {
        held_parts_of(HELD_TOOLS)
    }

    /// [`held_parts`] of the tools `sent`.
    fn held_parts_of(sent: &str) -> (std::sync::Arc<ignis_grammar::ToolCallGrammar>, impl Fn() -> ToolCallScanner, HashMap<String, ignis_grammar::json::Json>) {
        use ignis_grammar::json::Json;
        let tools: Vec<JsonValue> = serde_json::from_str(sent).unwrap();
        let Json::Arr(sent) = Json::parse(sent).unwrap() else { unreachable!() };
        let declared: Vec<ignis_grammar::DeclaredTool> = sent
            .iter()
            .map(|tool| {
                let function = tool.get("function").unwrap();
                ignis_grammar::DeclaredTool {
                    name: function.get("name").unwrap().as_str().unwrap().to_owned(),
                    hold: match function.get("strict") == Some(&Json::Bool(true)) {
                        true => ignis_grammar::ToolHold::Strict(function.get("parameters").unwrap().compact()),
                        false => ignis_grammar::ToolHold::Free,
                    },
                }
            })
            .collect();
        let parameters = sent
            .iter()
            .filter_map(|tool| {
                let function = tool.get("function").unwrap();
                (function.get("strict") == Some(&Json::Bool(true)))
                    .then(|| (function.get("name").unwrap().as_str().unwrap().to_owned(), function.get("parameters").unwrap().clone()))
            })
            .collect();
        let vocab = ignis_grammar::Vocab::new((0..=255u8).map(|b| ignis_grammar::VocabEntry::Text(vec![b])).collect(), &[]);
        let grammar = ignis_grammar::GrammarCache::new(vocab, 1 << 26).tool_calls(&declared).unwrap();
        let held: HashMap<String, call::Parameters> = grammar.held().map(|(name, p)| (name.to_owned(), p.clone())).collect();
        let scanner = move || ToolCallScanner::with_schemas(ToolSchemas::from_tools(&tools).holding(held.clone()));
        (grammar, scanner, parameters)
    }

    /// Everything the grammar accepts, the parser delivers as calls to the
    /// declared tools whose held arguments the validator finds valid: texts
    /// grown from the dialect's pieces, each piece taken only where the
    /// grammar takes it, read back by the scanner in random chunks.
    #[test]
    fn what_the_grammar_accepts_the_parser_delivers_valid() {
        const PIECES: &[&str] = &[
            "<tool_call>", "\n", "<function=write_file>", "<function=get_time>", "<function=search>", "<parameter=path>",
            "<parameter=content>", "<parameter=mode>", "<parameter=zone>", "<parameter=day>", "<parameter=query>",
            "<parameter=x>", "</parameter>", "</function>", "</tool_call>", "a.txt", "w", "a", "x", " ", "\t", "<", "</",
            "Europe/Rome", "2026-11-15", "é", "hello\n", "</div>", "\r\n",
            // Whole steps, so that many texts finish their calls.
            "<parameter=path>\na.txt\n</parameter>\n", "<parameter=content>\nhi\n</parameter>\n", "<parameter=mode>\nw\n</parameter>\n",
            "<parameter=zone>\nEurope/Rome\n</parameter>\n", "<parameter=day>\n2026-11-15\n</parameter>\n", "</function>\n</tool_call>",
        ];
        let (texts, delivered, _) = grammar_to_parser(HELD_TOOLS, PIECES, false, 0x5DEE_CE66_D1CE_4E5B, 3_000, 400);
        assert!(delivered > 500, "{delivered} calls from {texts} texts");
    }

    /// The same of a typed tool (#612), half the texts' steps a printable
    /// byte the grammar takes there, so that its JSON grows past the pieces:
    /// an object holding a list of edits, a bounded integer and a long one, a
    /// boolean, a nested array of objects, a recursive definition, and the
    /// nullable parameters written or left out.
    #[test]
    fn what_the_grammar_accepts_of_typed_and_nullable_values_the_parser_delivers_valid() {
        const PIECES: &[&str] = &[
            "<tool_call>", "\n", "<function=edit>", "<function=search>", "<parameter=change>", "<parameter=line>",
            "<parameter=id>", "<parameter=dry_run>", "<parameter=rows>", "<parameter=tree>", "<parameter=note>",
            "<parameter=limit>", "<parameter=query>", "</parameter>", "</function>", "</tool_call>", "\"", "}", "]", ",",
            "null", "true", "x", " ", "</",
            // Whole steps, so that many texts finish their calls.
            "<parameter=change>\n{\"edits\": [{\"old\": \"a\", \"new\": \"b\"}]}\n</parameter>\n", "<parameter=line>\n7\n</parameter>\n",
            "<parameter=id>\n-12\n</parameter>\n", "<parameter=dry_run>\ntrue\n</parameter>\n", "<parameter=rows>\n[[]]\n</parameter>\n",
            "<parameter=tree>\n{\"name\": \"t\", \"kids\": []}\n</parameter>\n", "<parameter=note>\nnull\n</parameter>\n",
            "<parameter=limit>\n3\n</parameter>\n", "</function>\n</tool_call>",
        ];
        let (texts, delivered, held) = grammar_to_parser(TYPED_TOOLS, PIECES, true, 0x9E37_79B9_7F4A_7C15, 2_000, 1_200);
        assert!(held > 300, "{held} held calls, {delivered} in all, from {texts} texts");
    }

    /// Texts grown from `pieces` (and, with `bytes`, half the steps a
    /// printable byte or a line break, closers first), each step taken only where the
    /// grammar of the tools `sent` takes it, cut at their last point in free
    /// text and read back by a holding scanner in random chunks: every held
    /// call closed is delivered and valid. The texts, the calls delivered,
    /// the held ones among them.
    fn grammar_to_parser(sent: &str, pieces: &[&str], bytes: bool, seed: u64, count: usize, steps: u64) -> (usize, usize, usize) {
        let (grammar, scanner, parameters) = held_parts_of(sent);
        let mut next = generator(seed);
        let (mut texts, mut delivered, mut held_delivered) = (0, 0, 0);
        for _ in 0..count {
            let mut m = ignis_grammar::ToolCallMatcher::new(grammar.clone());
            let (mut text, mut free_at, mut opened_at) = (String::new(), 0, 0);
            // The calls closed, held and free, as of the last point in free text.
            let (mut closes, mut closes_at_free) = ((0, 0), (0, 0));
            for _ in 0..(next() % steps) {
                let byte;
                let piece = if bytes && next() % 2 == 0 {
                    // Closers weigh most, so that the JSON ends and the call goes on.
                    let weight = |b: u8| match b {
                        b'"' | b'}' | b']' => 24,
                        b'\n' => 4,
                        _ => 1,
                    };
                    let allowed: Vec<u8> = m.allowed_bytes().into_iter().filter(|&b| b == b'\n' || (0x20..0x7F).contains(&b)).flat_map(|b| vec![b; weight(b)]).collect();
                    let Some(&b) = allowed.get(next() as usize % allowed.len().max(1)) else { continue };
                    byte = [b];
                    std::str::from_utf8(&byte).unwrap()
                } else {
                    pieces[next() as usize % pieces.len()]
                };
                let was_in_call = m.in_call();
                if m.accept_bytes(piece.as_bytes()) {
                    if !was_in_call && m.in_call() {
                        opened_at = text.len();
                    }
                    text.push_str(piece);
                    if was_in_call && !m.in_call() {
                        let name = text[opened_at..].split_once(FUNCTION_PREFIX).and_then(|(_, rest)| rest.split_once('>')).map(|(name, _)| name);
                        match parameters.contains_key(name.unwrap_or_default()) {
                            true => closes.0 += 1,
                            false => closes.1 += 1,
                        }
                    }
                    if !m.in_call() {
                        (free_at, closes_at_free) = (text.len(), closes);
                    }
                }
            }
            // The text up to its last point in free text: every call it opened closed.
            let text = &text[..free_at];
            let events = scan(scanner(), &chunked(text, &mut next));
            let calls = calls(&events);
            // Every held call is delivered; a free one, whose body the grammar
            // leaves free, may be dropped as today, its `</function>` unwritten.
            let held = calls.iter().filter(|call| parameters.contains_key(&call.name)).count();
            assert_eq!(held, closes_at_free.0, "{text:?}");
            assert!(calls.len() - held <= closes_at_free.1, "{text:?}");
            for call in &calls {
                let arguments = ignis_grammar::json::Json::parse(&call.arguments).unwrap();
                match parameters.get(&call.name) {
                    Some(schema) => assert!(ignis_grammar::validate::is_valid(schema, &arguments), "{} {} from {text:?}", call.name, call.arguments),
                    None => assert_eq!(call.name, "search", "{text:?}"),
                }
            }
            // Free text may hold the call's other tags; an opener always opens a call.
            assert!(!content(&events).contains("<tool_call>"), "{text:?}");
            texts += 1;
            delivered += calls.len();
            held_delivered += held;
        }
        (texts, delivered, held_delivered)
    }

    /// Every valid call written in the template's form is accepted, the
    /// parser reading back what was written, except the ones the grammar
    /// leaves out by name: the residue (a value the parser itself ends
    /// early) and spike S1's two miss points (spec #582: while a required
    /// parameter is missing, a free value's line starting `</function`, or a
    /// `</parameter>` line followed by `</`).
    #[test]
    fn every_valid_call_in_the_templates_form_is_accepted() {
        const TEXT: &[&str] = &[
            "a", "x y", "\n", "<parameter=path>", "<parameter=mode>", "</parameter>", "</function>", "</tool_call>",
            "<tool_call>", "<div>", "</div>", " ", "\t", "é", "<", "</", "parameter", "function", "\r\n", "line\n",
        ];
        let (grammar, scanner, parameters) = held_parts();
        let mut next = generator(0x2545_F491_4F6C_DD1D);
        let mut free = || -> String { (0..next() % 6).map(|_| TEXT[next() as usize % TEXT.len()]).collect() };
        let mut next = generator(0x9E37_79B9_7F4A_7C15);
        let (mut accepted, mut residue, mut miss_points) = (0, 0, 0);
        for _ in 0..4_000 {
            let (name, mut args): (&str, Vec<(&str, String)>) = match next() % 3 {
                0 => ("get_time", vec![("zone", ["Europe/Rome", "America/Lima"][next() as usize % 2].into()), ("day", ["2026-11-15", "1999-02-28"][next() as usize % 2].into())]),
                _ => ("write_file", vec![("path", free()), ("content", free()), ("mode", ["w", "a"][next() as usize % 2].into())]),
            };
            for i in (1..args.len()).rev() {
                args.swap(i, next() as usize % (i + 1));
            }
            let mut text = format!("<tool_call>\n<function={name}>\n");
            for (k, v) in &args {
                text.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
            }
            text.push_str("</function>\n</tool_call>");
            // What the parser reads back, against what was written.
            let events = scan(scanner(), &[&text]);
            let read = calls(&events).first().map(|call| call.arguments.clone());
            let written: serde_json::Map<String, JsonValue> = args.iter().map(|(k, v)| (k.to_string(), JsonValue::String(v.clone()))).collect();
            let same = read.as_deref().and_then(|a| serde_json::from_str::<serde_json::Map<String, JsonValue>>(a).ok()) == Some(written);
            if !same {
                residue += 1;
                continue;
            }
            // While a later parameter is still to write, a free value may not
            // hold the two miss points, read with the closer line the
            // template writes after it: a line starting `</function`, and a
            // line starting `</parameter>` whose whitespace, across lines,
            // runs on to a `</`.
            let misses = args.iter().take(args.len() - 1).filter(|(k, _)| matches!(*k, "path" | "content")).any(|(_, v)| {
                let read = format!("{v}\n</parameter>");
                let starts = std::iter::once(0).chain(read.match_indices('\n').map(|(at, _)| at + 1));
                starts.into_iter().any(|at| {
                    let line = &read[at..];
                    line.starts_with("</function")
                        || line.strip_prefix("</parameter>").is_some_and(|rest| rest.trim_start_matches([' ', '\t', '\r', '\n']).starts_with("</"))
                })
            });
            let mut m = ignis_grammar::ToolCallMatcher::new(grammar.clone());
            let taken = m.accept_bytes(text.as_bytes()) && !m.in_call();
            if misses {
                miss_points += 1;
                continue;
            }
            assert!(taken, "{text:?}");
            assert!(ignis_grammar::validate::is_valid(&parameters[name], &ignis_grammar::json::Json::parse(&read.unwrap()).unwrap()));
            accepted += 1;
        }
        assert!(accepted > 2_000 && residue > 0 && miss_points > 0, "{accepted} accepted, {residue} residue, {miss_points} at the miss points");
    }

    /// `value` as the template writes a typed one: `tojson`'s spelling, a
    /// space after each `,` and `:` (`style` 0), compact (1), or
    /// pretty-printed (2).
    fn spelled(value: &ignis_grammar::json::Json, style: u64) -> String {
        use ignis_grammar::json::Json;
        match (style, value) {
            (1, _) => value.compact(),
            (2, _) => value.pretty(),
            (_, Json::Arr(items)) => format!("[{}]", items.iter().map(|item| spelled(item, 0)).collect::<Vec<_>>().join(", ")),
            (_, Json::Obj(entries)) => {
                let entries: Vec<String> = entries.iter().map(|(k, v)| format!("{}: {}", Json::Str(k.clone()).compact(), spelled(v, 0))).collect();
                format!("{{{}}}", entries.join(", "))
            }
            (_, leaf) => leaf.compact(),
        }
    }

    /// Every valid call of the typed tool written in the template's form is
    /// accepted and read back as written (#612): arguments its schema's own
    /// grammar writes (the grammar's subset: keys in the schema's order,
    /// literals in their canonical spelling), each typed value spelled as
    /// `tojson` spells it, compact or pretty-printed (within the whitespace
    /// bound), the parameters in random order, a nullable one left out half
    /// the times it is `null`. Carved out by name: a string-or-null value
    /// whose text reads `null`, which is `null` (the documented limit).
    #[test]
    fn every_valid_typed_call_in_the_templates_form_is_accepted_and_read_back() {
        use ignis_grammar::json::Json;
        let (grammar, scanner, parameters) = held_parts_of(TYPED_TOOLS);
        let schema = &parameters["edit"];
        let vocab = ignis_grammar::Vocab::new((0..=255u8).map(|b| ignis_grammar::VocabEntry::Text(vec![b])).collect(), &[]);
        let documents = ignis_grammar::GrammarCache::new(vocab, 1 << 26).json_schema(&schema.compact(), true).unwrap();
        let mut next = generator(0xD1B5_4A32_D192_ED03);
        let (mut accepted, mut long, mut omitted, mut limit) = (0, 0, 0, 0);
        for seed in 0..1_500 {
            let Some(bytes) = ignis_grammar::random_document(&documents, seed, 80, 6_000).unwrap() else { continue };
            let doc = Json::parse(std::str::from_utf8(&bytes).unwrap()).unwrap();
            if doc.get("note").and_then(Json::as_str).is_some_and(|note| note.trim() == "null") {
                limit += 1;
                continue;
            }
            let Json::Obj(mut entries) = doc.clone() else { unreachable!() };
            for i in (1..entries.len()).rev() {
                entries.swap(i, next() as usize % (i + 1));
            }
            let mut text = "<tool_call>\n<function=edit>\n".to_owned();
            for (k, v) in &entries {
                if *v == Json::Null && next() % 2 == 0 {
                    omitted += 1;
                    continue;
                }
                let value = match v {
                    Json::Str(note) => note.clone(),
                    typed => {
                        let pretty = spelled(typed, next() % 3);
                        let indent = pretty.split('\n').map(|line| line.len() - line.trim_start().len() + 1).max().unwrap_or(0);
                        if indent > 40 { spelled(typed, 0) } else { pretty }
                    }
                };
                text.push_str(&format!("<parameter={k}>\n{value}\n</parameter>\n"));
            }
            text.push_str("</function>\n</tool_call>");
            let mut m = ignis_grammar::ToolCallMatcher::new(grammar.clone());
            assert!(m.accept_bytes(text.as_bytes()) && !m.in_call(), "{text:?}");
            let events = scan(scanner(), &chunked(&text, &mut next));
            let calls = calls(&events);
            assert_eq!(calls.len(), 1, "{text:?}");
            assert_eq!(calls[0].arguments, doc.compact(), "{text:?}");
            assert!(ignis_grammar::validate::is_valid(schema, &Json::parse(&calls[0].arguments).unwrap()), "{}", calls[0].arguments);
            long += doc.get("id").and_then(Json::number_text).is_some_and(|id| id.trim_start_matches('-').len() > 19) as usize;
            accepted += 1;
        }
        assert!(accepted > 1_000 && long > 0 && omitted > 0, "{accepted} accepted, {long} past i64, {omitted} nulls left out, {limit} at the limit");
    }

    /// A call of `edit` with these values, the rest of them valid, its
    /// parameters in the reverse of the schema's order.
    fn edit_call(with: &[(&str, &str)], without: &[&str]) -> String {
        let values = [
            ("change", r#"{"edits": [{"old": "x", "new": "y"}]}"#),
            ("line", "12"),
            ("id", "123456789012345678901234567890"),
            ("dry_run", "false"),
            ("rows", r#"[[{"k": "a", "v": 1.50}], []]"#),
            ("tree", r#"{"name": "root", "kids": [{"name": "leaf", "kids": []}]}"#),
            ("note", "a note"),
            ("limit", "3"),
        ];
        let mut text = "<tool_call>\n<function=edit>\n".to_owned();
        for (k, v) in values.iter().rev().filter(|(k, _)| !without.contains(k)) {
            let v = with.iter().find(|(name, _)| name == k).map_or(*v, |(_, v)| *v);
            text.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
        }
        text + "</function>\n</tool_call>"
    }

    /// What a held `edit` call is delivered as.
    fn edit_arguments(with: &[(&str, &str)], without: &[&str]) -> serde_json::Value {
        let (_, scanner, _) = held_parts_of(TYPED_TOOLS);
        let events = scan(scanner(), &[&edit_call(with, without)]);
        serde_json::from_str(&calls(&events)[0].arguments).unwrap()
    }

    #[test]
    fn a_held_calls_typed_values_are_delivered_as_written_and_its_nulls_as_null() {
        let (_, scanner, _) = held_parts_of(TYPED_TOOLS);
        let events = scan(scanner(), &[&edit_call(&[], &[])]);
        // In the schema's order, the nested keys as written, every digit of
        // an integer past i64 kept, a number in its canonical spelling.
        assert_eq!(
            calls(&events)[0].arguments,
            r#"{"change":{"edits":[{"old":"x","new":"y"}]},"line":12,"id":123456789012345678901234567890,"dry_run":false,"rows":[[{"k":"a","v":1.5}],[]],"tree":{"name":"root","kids":[{"name":"leaf","kids":[]}]},"note":"a note","limit":3}"#
        );
        // A nullable parameter left out, or written `null`, is `null`.
        for (with, without) in [(&[][..], &["note", "limit"][..]), (&[("note", "null"), ("limit", "null")][..], &[][..])] {
            let arguments = edit_arguments(with, without);
            assert_eq!((&arguments["note"], &arguments["limit"]), (&JsonValue::Null, &JsonValue::Null), "{with:?} {without:?}");
        }
    }

    /// The documented limit: a string-or-null value whose text, trimmed,
    /// reads `null` is `null`, never the string `null`; any other text is
    /// itself.
    #[test]
    fn a_string_or_null_value_whose_text_reads_null_is_null() {
        for (text, note) in [("null", JsonValue::Null), (" null ", JsonValue::Null), ("\"null\"", JsonValue::from("\"null\"")), ("nullable", JsonValue::from("nullable"))] {
            assert_eq!(edit_arguments(&[("note", text)], &[])["note"], note, "{text:?}");
        }
    }
}
