//! The served close-out's corpus (GitHub #617, spec #582, ADR 0094): the
//! agent requests `held_tools_gpu.rs` serves on the 27B, each beside its
//! **free twin**, and the conversations held to one call per turn under
//! `parallel_tool_calls: false` on the 27B (`held_tools_gpu.rs`) and on
//! Flash-Next (`flash_next_parallel_tool_calls_gpu.rs`). `held_tools_http.rs`
//! sends every request of the corpus, and every twin, to the mock 27B first,
//! so a request the server would refuse is found on the CPU, not on the card.
//!
//! A request's free twin is the same request with every tool sent
//! `strict: false`: nothing held, the calls read as written. The template
//! renders each tool's `strict` into the prompt, so on chat completions a
//! twin's prompt is its request's but for that word; on the Responses API a
//! tool sent without `strict` (the default, normalized and held, #614) is
//! rendered `strict: false` too, so there the twin's prompt is the request's
//! own and, greedy, its delivery is what the model wrote before the first
//! miss and would have written past it.
//!
//! Included by `#[path]`; each binary uses a part of it.

#![allow(dead_code)]

use std::future::Future;

use ignis_grammar::json::Json;
use serde_json::{json, Value as JsonValue};

// ── the tools, in the chat shape ─────────────────────────────────────────
//
// A strict tool writes `"strict":true,` just before its `parameters`, so the
// Responses default sends it without `strict` by dropping those bytes. Every
// strict schema is in OpenAI's strict subset: each object closed, each
// property required, no length keyword.

/// A free string, a long one, and an enum.
pub const WRITE_FILE: &str = r#"{"type":"function","function":{"name":"write_file","description":"Write a text file, replacing it or appending to it.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string","description":"The file's path, relative to the project."},"content":{"type":"string","description":"The text to write."},"mode":{"type":"string","enum":["replace","append"],"description":"Whether to replace the file or append to it."}},"required":["path","content","mode"],"additionalProperties":false}}}"#;
pub const READ_FILE: &str = r#"{"type":"function","function":{"name":"read_file","description":"Read a text file.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string","description":"The file's path, relative to the project."}},"required":["path"],"additionalProperties":false}}}"#;
/// A pattern and a format.
pub const MEETING: &str = r#"{"type":"function","function":{"name":"book_meeting","description":"Book a meeting room for a day.","strict":true,"parameters":{"type":"object","properties":{"room":{"type":"string","pattern":"^R[0-9]{3}$","description":"The room, R and three digits."},"day":{"type":"string","format":"date","description":"The day, as YYYY-MM-DD."},"title":{"type":"string","description":"The meeting's title."}},"required":["room","day","title"],"additionalProperties":false}}}"#;
/// Not strict: its arguments stay free.
pub const SEARCH: &str = r#"{"type":"function","function":{"name":"search","description":"Search the web.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}"#;
/// Many parameters: enums, a bounded integer, an array of enums, two
/// nullable strings and a nested object.
pub const TICKET: &str = r#"{"type":"function","function":{"name":"create_ticket","description":"Open a ticket in the project's tracker.","strict":true,"parameters":{"type":"object","properties":{"title":{"type":"string","description":"A one-line summary."},"body":{"type":"string","description":"What happens and how to reproduce it, in Markdown."},"kind":{"type":"string","enum":["bug","feature","chore"],"description":"The kind of work."},"priority":{"type":"string","enum":["low","medium","high","urgent"],"description":"How soon it must be done."},"estimate_hours":{"type":"integer","minimum":1,"maximum":40,"description":"Hours of work, 1 to 40."},"labels":{"type":"array","items":{"type":"string","enum":["backend","frontend","docs","infra"]},"description":"The areas it touches."},"assignee":{"type":["string","null"],"description":"Who works on it, or null for nobody yet."},"due":{"type":["string","null"],"description":"The due day as YYYY-MM-DD, or null for none."},"reporter":{"type":"object","properties":{"name":{"type":"string"},"team":{"type":"string","enum":["core","web","ops"]}},"required":["name","team"],"additionalProperties":false,"description":"Who reported it."}},"required":["title","body","kind","priority","estimate_hours","labels","assignee","due","reporter"],"additionalProperties":false}}}"#;
/// A bounded integer, a boolean, an array of objects inside an object, a
/// nullable string.
pub const EDIT: &str = r#"{"type":"function","function":{"name":"edit_file","description":"Replace text in a file.","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string","description":"The file's path."},"line":{"type":"integer","minimum":1,"maximum":100000,"description":"The line the first edit starts on."},"dry_run":{"type":"boolean","description":"Show the change without writing it."},"change":{"type":"object","properties":{"edits":{"type":"array","items":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},"required":["old","new"],"additionalProperties":false}}},"required":["edits"],"additionalProperties":false},"note":{"type":["string","null"],"description":"Why, or null."}},"required":["path","line","dry_run","change","note"],"additionalProperties":false}}}"#;
/// An enum, a pattern, a bounded integer, and two parameters the schema
/// requires and the descriptions call optional: a model that leaves them
/// out closes its call early.
pub const DEPLOY: &str = r#"{"type":"function","function":{"name":"deploy","description":"Deploy a service.","strict":true,"parameters":{"type":"object","properties":{"service":{"type":"string","enum":["api","web","worker"]},"version":{"type":"string","pattern":"^v[0-9]+\\.[0-9]+\\.[0-9]+$","description":"The release tag, like v1.2.3."},"region":{"type":"string","enum":["eu-west","us-east","ap-south"]},"replicas":{"type":"integer","minimum":1,"maximum":10},"approved_by":{"type":"string","description":"Optional: who approved the deploy; leave it out when nobody did."},"notes":{"type":"string","description":"Optional free-form notes; omit them when there are none."}},"required":["service","version","region","replicas","approved_by","notes"],"additionalProperties":false}}}"#;
/// [`EDIT`] as a client writes it without `strict`: its objects open,
/// `dry_run` and `note` optional. Sent to the Responses API, normalized.
pub const EDIT_OPEN: &str = r#"{"type":"function","function":{"name":"edit_file","description":"Replace text in a file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"The file's path."},"line":{"type":"integer","minimum":1,"description":"The line the first edit starts on."},"dry_run":{"type":"boolean","description":"Show the change without writing it; false when left out."},"change":{"type":"object","properties":{"edits":{"type":"array","items":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},"required":["old","new"]}}},"required":["edits"]},"note":{"type":["string","null"],"description":"Why, if it is worth saying."}},"required":["path","line","change"]}}}"#;
/// [`WRITE_FILE`] as a client writes it without `strict`: open, `mode`
/// optional.
pub const WRITE_OPEN: &str = r#"{"type":"function","function":{"name":"write_file","description":"Write a text file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string","enum":["replace","append"],"description":"replace when left out."}},"required":["path","content"]}}}"#;

/// A directory copy as a client writes it without `strict`: an `options`
/// object it requires, whose two members are optional. Normalized, every
/// member of a nested object is required (spec #582, #614), so a held call
/// writes both where the model may have written one.
pub const COPY_OPEN: &str = r#"{"type":"function","function":{"name":"copy_dir","description":"Copy a directory.","parameters":{"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},"options":{"type":"object","properties":{"recursive":{"type":"boolean"},"depth":{"type":"integer","minimum":1}},"required":[],"description":"How to copy; each member optional."}},"required":["from","to","options"]}}}"#;

/// `tool_choice: "required"`.
pub const REQUIRED: &str = r#""tool_choice":"required""#;
/// `tool_choice` naming `write_file`.
pub const NAMED_WRITE: &str = r#""tool_choice":{"type":"function","function":{"name":"write_file"}}"#;
/// `response_format` asking whether the file was saved.
pub const OUTCOME: &str = r#""response_format":{"type":"json_schema","json_schema":{"name":"outcome","strict":true,"schema":{"type":"object","properties":{"saved":{"type":"boolean"},"path":{"type":"string"}},"required":["saved","path"],"additionalProperties":false}}}"#;

/// A case's label: the docs/dialect.md request of #611's GPU leg, whose
/// content holds an example of this same tool's call.
pub const DIALECT: &str = "a file about the call dialect";

pub const MAX_TOKENS: u32 = 4096;
/// A request with thinking on reasons first.
pub const MAX_TOKENS_THINKING: u32 = 8192;

// ── the corpus ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Chat,
    Responses,
}

impl Api {
    pub fn path(self) -> &'static str {
        match self {
            Api::Chat => "/v1/chat/completions",
            Api::Responses => "/v1/responses",
        }
    }
}

/// What a request must deliver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    /// One call or more.
    Call,
    /// One call or more, each to this function.
    Named(&'static str),
    /// A call, or an answer valid under the request's `response_format`.
    CallOrAnswer,
}

/// One agent request of the corpus.
#[derive(Clone, Debug)]
pub struct Case {
    pub label: &'static str,
    pub api: Api,
    /// The Responses default: each strict tool sent without `strict`.
    pub without_strict: bool,
    pub prompt: &'static str,
    /// The chat shape; the Responses API sends them flat.
    pub tools: &'static [&'static str],
    pub thinking: bool,
    /// Members sent beside `tools`, as JSON text; empty for none.
    pub field: &'static str,
    pub expect: Expect,
    /// What the prompt puts against the schema, for a case built to make
    /// the model miss; empty for none.
    pub bait: &'static str,
}

impl Case {
    fn new(api: Api, label: &'static str, prompt: &'static str, tools: &'static [&'static str]) -> Self {
        Case { label, api, without_strict: false, prompt, tools, thinking: false, field: "", expect: Expect::Call, bait: "" }
    }

    fn chat(label: &'static str, prompt: &'static str, tools: &'static [&'static str]) -> Self {
        Self::new(Api::Chat, label, prompt, tools)
    }

    fn responses(label: &'static str, prompt: &'static str, tools: &'static [&'static str]) -> Self {
        Self::new(Api::Responses, label, prompt, tools)
    }

    fn thinking(self) -> Self {
        Case { thinking: true, ..self }
    }

    fn field(self, field: &'static str) -> Self {
        Case { field, ..self }
    }

    fn expect(self, expect: Expect) -> Self {
        Case { expect, ..self }
    }

    fn without_strict(self) -> Self {
        Case { without_strict: true, ..self }
    }

    fn bait(self, bait: &'static str) -> Self {
        Case { bait, ..self }
    }

    pub fn max_tokens(&self) -> u32 {
        if self.thinking {
            MAX_TOKENS_THINKING
        } else {
            MAX_TOKENS
        }
    }

    /// `response_format`'s schema, when the case sends one.
    pub fn answer_schema(&self) -> Option<Json> {
        let fields = Json::parse(&format!("{{{}}}", self.field)).ok()?;
        Some(fields.get("response_format")?.get("json_schema")?.get("schema")?.clone())
    }

    /// The type a held request of this case counts under in
    /// `ignis_structured_*`: an answer's grammar beside the tools wins.
    pub fn counted_as(&self) -> &'static str {
        if self.field.contains("response_format") {
            "json_schema"
        } else {
            "tool"
        }
    }

    /// The tools as this case sends them, or (`held` false) its free twin's.
    pub fn tools_sent(&self, held: bool) -> Vec<String> {
        self.tools
            .iter()
            .map(|tool| {
                let tool = match self.api {
                    Api::Chat => (*tool).to_owned(),
                    Api::Responses => flat(tool),
                };
                match (held, self.without_strict) {
                    (true, false) => tool,
                    (true, true) => tool.replacen("\"strict\":true,", "", 1),
                    (false, _) => strict_false(&tool),
                }
            })
            .collect()
    }

    /// The request's body, or (`held` false) its free twin's; greedy,
    /// `max_tokens` at most.
    pub fn body(&self, model: &str, held: bool, max_tokens: u32) -> String {
        let tools = self.tools_sent(held).join(",");
        let prompt = JsonValue::String(self.prompt.into());
        let field = if self.field.is_empty() { String::new() } else { format!(",{}", self.field) };
        let thinking = self.thinking;
        match self.api {
            Api::Chat => format!(
                r#"{{"model":"{model}","messages":[{{"role":"user","content":{prompt}}}],"tools":[{tools}],"enable_thinking":{thinking},"temperature":0,"max_tokens":{max_tokens}{field}}}"#
            ),
            Api::Responses => format!(
                r#"{{"model":"{model}","input":{prompt},"tools":[{tools}],"enable_thinking":{thinking},"temperature":0,"max_output_tokens":{max_tokens}{field}}}"#
            ),
        }
    }
}

/// Cases built against the model's prior, so that it misses: each served on
/// chat completions with its tools strict, and on the Responses API with
/// them sent without `strict`, where its free twin's prompt is its own.
/// Every one forces a call, so a model that would ask a question instead
/// still writes one.
const BAITS: &[(&str, &str, &str, &[&str])] = &[
    (
        "enum",
        "a priority and a kind outside their enums",
        "Open a ticket with priority 'critical' and kind 'incident': the payment service is down. Estimate 3 hours, nobody assigned, no due date, labels backend, reported by Sam from the ops team.",
        &[TICKET],
    ),
    ("enum", "a mode outside its enum", "Write the text 'x' to log.txt in 'overwrite' mode.", &[WRITE_FILE]),
    (
        "bound",
        "an estimate above its bound",
        "Open a feature ticket for a full rewrite of the scheduler: medium priority, estimated at 120 hours, assigned to lee, due 2027-03-01, labels backend and infra, reported by Kim from the core team.",
        &[TICKET],
    ),
    ("bound", "replicas above their bound", "Deploy api v1.4.0 to eu-west with 25 replicas, approved by Mia, notes: routine.", &[DEPLOY]),
    (
        "word for a number",
        "a word for the estimate",
        "Open a high-priority bug ticket: the CSV export drops its last row. Write the estimate as 'TBD', we do not know it yet. Nobody assigned, no due date, labels backend, reported by Ana from the web team.",
        &[TICKET],
    ),
    (
        "word for a number",
        "a word for the replicas",
        "Deploy web v2.4.1 to us-east with replicas set to 'auto', approved by Mia, notes: canary.",
        &[DEPLOY],
    ),
    ("pattern", "a room outside its pattern", "Book room B12 on 2026-12-03 for the design review.", &[MEETING]),
    ("format", "a day that is not a date", "Book room R101 for next Tuesday's retro.", &[MEETING]),
    ("early close", "an approver and notes the prompt leaves out", "Deploy the api service v3.1.0 to eu-west with 2 replicas.", &[DEPLOY]),
    (
        "early close",
        "an approver the prompt says there is none",
        "Deploy worker v0.9.2 to ap-south with 4 replicas. No approval was needed and there are no notes.",
        &[DEPLOY],
    ),
    ("early close", "a ticket with only a title", "Open a ticket: the docs site's search box does not work.", &[TICKET]),
    (
        "invented parameter",
        "parameters the tool does not declare",
        "Write 'hi' to a.txt, replacing it, with file permissions 600 and UTF-8 encoding.",
        &[WRITE_FILE],
    ),
    ("invented parameter", "a line limit the tool does not take", "Read only the first 20 lines of README.md.", &[READ_FILE]),
];

/// Every request of the corpus.
pub fn corpus() -> Vec<Case> {
    let mut cases = vec![
        Case::chat("a checklist", "Save a file notes/todo.md with a checklist of three things to do this week.", &[WRITE_FILE, READ_FILE]),
        Case::chat(
            "a long script",
            "Write a Python script primes.py that prints the first twenty prime numbers, with a docstring and comments.",
            &[WRITE_FILE],
        ),
        Case::chat(
            DIALECT,
            "Write docs/dialect.md: a short explanation of the tool-call format you use, with a complete example call inside a fenced block.",
            &[WRITE_FILE],
        ),
        Case::chat("two reads", "Read both README.md and Cargo.toml.", &[READ_FILE, WRITE_FILE]),
        Case::chat("a meeting, a free tool beside", "Book room R204 on 2026-12-03 for the quarterly planning meeting.", &[MEETING, SEARCH]),
        Case::chat("an append, thinking on", "Append a line saying 'release cut' to CHANGES.txt.", &[WRITE_FILE, READ_FILE, SEARCH]).thinking(),
        Case::chat(
            "a ticket, every parameter given",
            "Open a ticket: the login page crashes on Safari when the password holds an emoji. It is a high-priority bug, about 6 hours of work, labels frontend, assign it to dana, due 2026-12-01, reported by Luca from the web team.",
            &[TICKET],
        ),
        Case::chat(
            "a ticket, two nulls",
            "Open a low-priority chore ticket to update the copyright year in the docs footer: one hour, labels docs, nobody assigned yet, no due date, reported by Ana from the ops team.",
            &[TICKET],
        ),
        Case::chat(
            "a typed edit",
            "In src/main.rs, starting on line 12, replace the call hello() with greet() and the constant 10 with 20. Make it a dry run, with no note.",
            &[EDIT],
        ),
        Case::chat(
            "ten edits, a long typed value",
            "In src/names.rs, starting on line 3, rename these ten identifiers, one edit each: a to alpha, b to bravo, c to charlie, d to delta, e to echo, f to foxtrot, g to golf, h to hotel, i to india, j to juliet. Not a dry run; note that it is the NATO rename.",
            &[EDIT],
        ),
        Case::chat("a read and a free search", "Read Cargo.toml and search the web for the Rust 2024 edition guide.", &[READ_FILE, SEARCH]),
        Case::chat("tool_choice required", "What does the README say?", &[READ_FILE, WRITE_FILE]).field(REQUIRED),
        Case::chat("tool_choice named", "The build passed.", &[WRITE_FILE, READ_FILE]).field(NAMED_WRITE).expect(Expect::Named("write_file")),
        // Thinking on, the forced opening joins the model's own run at the
        // reasoning's close, where it may yield to the grammar (#615): the
        // call still names the tool a named choice names.
        Case::chat("tool_choice required, thinking on", "What does the README say?", &[READ_FILE, WRITE_FILE]).field(REQUIRED).thinking(),
        Case::chat("tool_choice named, thinking on", "The build passed.", &[WRITE_FILE, READ_FILE])
            .field(NAMED_WRITE)
            .expect(Expect::Named("write_file"))
            .thinking(),
        Case::chat("response_format beside", "Save the word 'hello' to greeting.txt, then say whether it was saved.", &[WRITE_FILE])
            .field(OUTCOME)
            .expect(Expect::CallOrAnswer),
        Case::chat(
            "a long ticket, thinking on",
            "Open a detailed ticket for this bug: when two users save the same document within a second, the second save silently overwrites the first. Write a thorough body with steps to reproduce, the expected and the actual behaviour, and a suggested fix. High priority, 8 hours, labels backend, assigned to omar, due 2026-12-15, reported by Ivo from the core team.",
            &[TICKET],
        )
        .thinking(),
        Case::responses(
            "a strict ticket",
            "Open an urgent bug ticket: search returns stale results after an index rebuild. 2 hours of work, labels backend and infra, assign it to omar, due 2026-11-30, reported by Ivo from the core team.",
            &[TICKET],
        ),
        Case::responses("two strict reads", "Read both package.json and tsconfig.json.", &[READ_FILE, WRITE_FILE]),
        Case::responses("tool_choice required, thinking on", "What does the README say?", &[READ_FILE, WRITE_FILE]).field(REQUIRED).thinking(),
        Case::responses("tool_choice named, thinking on", "The build passed.", &[WRITE_FILE, READ_FILE])
            .field(NAMED_WRITE)
            .expect(Expect::Named("write_file"))
            .thinking(),
        Case::responses("the default: a nested object's optional members", "Copy the directory assets/ to backup/assets/, recursively.", &[COPY_OPEN]).without_strict(),
        Case::responses("the default: an open typed edit", "In lib/util.py, on line 40, replace the call old_name() with new_name().", &[EDIT_OPEN]).without_strict(),
        Case::responses(
            "the default: an open write and a parameter it lacks",
            "Write 'hello world' to out/hello.txt, replacing it, with permissions 644 and a trailing newline.",
            &[WRITE_OPEN],
        )
        .without_strict()
        .bait("invented parameter"),
    ];
    for &(bait, label, prompt, tools) in BAITS {
        cases.push(Case::chat(label, prompt, tools).field(REQUIRED).bait(bait));
        cases.push(Case::responses(label, prompt, tools).field(REQUIRED).without_strict().bait(bait));
    }
    cases
}

// ── tools as sent ────────────────────────────────────────────────────────

/// A chat tool in the Responses API's flat shape: its function's members
/// beside its `type`, in the order sent.
pub fn flat(tool: &str) -> String {
    let members = tool.strip_prefix(r#"{"type":"function","function":{"#).and_then(|rest| rest.strip_suffix("}}")).expect("a chat tool");
    format!(r#"{{"type":"function",{members}}}"#)
}

/// `tool` sent `strict: false`: in place of its `strict: true`, or beside
/// its name.
pub fn strict_false(tool: &str) -> String {
    match tool.contains("\"strict\":true") {
        true => tool.replacen("\"strict\":true", "\"strict\":false", 1),
        false => tool.replacen("\"name\":", "\"strict\":false,\"name\":", 1),
    }
}

/// How a tool's calls are served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    Free,
    /// Sent `strict: true`.
    Strict,
    /// Sent to the Responses API without `strict` (#614).
    Normalized,
}

/// A tool as sent: its function's name, its `parameters`, how it is held.
pub struct Declared {
    pub name: String,
    pub parameters: Json,
    pub hold: Hold,
}

/// The tools `sent` declares, read in the order written.
pub fn declared(api: Api, sent: &[String]) -> Vec<Declared> {
    sent.iter()
        .map(|tool| {
            let tool = Json::parse(tool).expect("a tool is JSON");
            let function = tool.get("function").unwrap_or(&tool);
            let hold = match (function.get("strict"), api) {
                (Some(Json::Bool(true)), _) => Hold::Strict,
                (None | Some(Json::Null), Api::Responses) => Hold::Normalized,
                _ => Hold::Free,
            };
            Declared {
                name: function.get("name").and_then(Json::as_str).expect("a tool's name").to_owned(),
                parameters: function.get("parameters").expect("a tool's parameters").clone(),
                hold,
            }
        })
        .collect()
}

// ── what a response delivered ────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// What a whole response delivered.
#[derive(Clone, Debug, Default)]
pub struct Delivered {
    /// `finish_reason` on chat completions, `status` on the Responses API.
    pub end: String,
    pub reasoning: String,
    pub text: String,
    pub calls: Vec<Call>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// The Responses API's `tools` echo: each tool's `strict`.
    pub echoed_strict: Vec<JsonValue>,
}

impl Delivered {
    pub fn read(api: Api, response: &JsonValue) -> Self {
        match api {
            Api::Chat => {
                let choice = &response["choices"][0];
                let message = &choice["message"];
                let calls = message["tool_calls"].as_array().map(Vec::as_slice).unwrap_or_default();
                Delivered {
                    end: choice["finish_reason"].as_str().unwrap_or_default().to_owned(),
                    reasoning: message["reasoning_content"].as_str().unwrap_or_default().to_owned(),
                    text: message["content"].as_str().unwrap_or_default().to_owned(),
                    calls: calls
                        .iter()
                        .map(|call| Call {
                            id: call["id"].as_str().unwrap_or_default().to_owned(),
                            name: call["function"]["name"].as_str().unwrap_or_default().to_owned(),
                            arguments: call["function"]["arguments"].as_str().unwrap_or_default().to_owned(),
                        })
                        .collect(),
                    prompt_tokens: response["usage"]["prompt_tokens"].as_u64().unwrap_or_default(),
                    completion_tokens: response["usage"]["completion_tokens"].as_u64().unwrap_or_default(),
                    echoed_strict: Vec::new(),
                }
            }
            Api::Responses => {
                let items = response["output"].as_array().map(Vec::as_slice).unwrap_or_default();
                let texts = |kind: &str| -> String {
                    items
                        .iter()
                        .filter(|item| item["type"] == kind)
                        .flat_map(|item| item["content"].as_array().map(Vec::as_slice).unwrap_or_default())
                        .filter_map(|part| part["text"].as_str())
                        .collect()
                };
                Delivered {
                    end: response["status"].as_str().unwrap_or_default().to_owned(),
                    reasoning: texts("reasoning"),
                    text: texts("message"),
                    calls: items
                        .iter()
                        .filter(|item| item["type"] == "function_call")
                        .map(|item| Call {
                            id: item["call_id"].as_str().unwrap_or_default().to_owned(),
                            name: item["name"].as_str().unwrap_or_default().to_owned(),
                            arguments: item["arguments"].as_str().unwrap_or_default().to_owned(),
                        })
                        .collect(),
                    prompt_tokens: response["usage"]["input_tokens"].as_u64().unwrap_or_default(),
                    completion_tokens: response["usage"]["output_tokens"].as_u64().unwrap_or_default(),
                    echoed_strict: response["tools"]
                        .as_array()
                        .map(|tools| tools.iter().map(|tool| tool.get("function").unwrap_or(tool)["strict"].clone()).collect())
                        .unwrap_or_default(),
                }
            }
        }
    }

    /// The text and the calls, one call a line: what two deliveries are
    /// compared by.
    pub fn delivery(&self) -> String {
        let calls: String = self.calls.iter().map(|call| format!("\n{} {}", call.name, call.arguments)).collect();
        format!("{}{calls}", self.text)
    }
}

// ── what is wrong with a call ────────────────────────────────────────────

/// One fault of a call's arguments: the validator's kind of violation
/// (`missing`, `type`, `enum`, `unknown`, `constraint`), or `undeclared`
/// for a key no schema declares in an open object; where, as a JSON pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fault {
    pub kind: String,
    pub at: String,
    pub keyword: &'static str,
}

/// Every fault of `arguments` under `parameters` as sent (#608's validator,
/// then the keys an open object does not declare); `None` when they are not
/// JSON.
pub fn faults(parameters: &Json, arguments: &str) -> Option<Vec<Fault>> {
    let value = Json::parse(arguments).ok()?;
    let mut out: Vec<Fault> = ignis_grammar::validate::violations(parameters, &value)
        .into_iter()
        .map(|v| Fault { kind: v.kind.label().to_owned(), at: v.at, keyword: v.keyword })
        .collect();
    let mut undeclared = Vec::new();
    undeclared_keys(parameters, &value, "", &mut undeclared);
    for at in undeclared {
        if !out.iter().any(|fault| fault.at == at) {
            out.push(Fault { kind: "undeclared".into(), at, keyword: "properties" });
        }
    }
    Some(out)
}

fn undeclared_keys(schema: &Json, value: &Json, at: &str, out: &mut Vec<String>) {
    match (value, schema.get("properties"), schema.get("items")) {
        (Json::Obj(members), Some(Json::Obj(properties)), _) => {
            for (key, member) in members {
                let path = format!("{at}/{}", ignis_grammar::json::pointer_token(key));
                match properties.iter().find(|(name, _)| name == key) {
                    Some((_, schema)) => undeclared_keys(schema, member, &path, out),
                    None => out.push(path),
                }
            }
        }
        (Json::Arr(items), _, Some(schema)) => {
            for (i, item) in items.iter().enumerate() {
                undeclared_keys(schema, item, &format!("{at}/{i}"), out);
            }
        }
        _ => {}
    }
}

/// The value at JSON pointer `at` in `value`.
pub fn pointed<'a>(value: &'a Json, at: &str) -> Option<&'a Json> {
    at.split('/').skip(1).try_fold(value, |value, token| {
        let token = ignis_grammar::json::unescape_pointer_token(token);
        match value {
            Json::Obj(members) => members.iter().find(|(key, _)| *key == token).map(|(_, member)| member),
            Json::Arr(items) => token.parse::<usize>().ok().and_then(|i| items.get(i)),
            _ => None,
        }
    })
}

/// The value at `at` in `arguments`, compact and cut at 80 characters, or
/// `(absent)`.
pub fn shown_at(arguments: &str, at: &str) -> String {
    let shown = Json::parse(arguments).ok().and_then(|value| pointed(&value, at).map(Json::compact));
    match shown {
        Some(text) if text.chars().count() > 80 => format!("{}...", text.chars().take(80).collect::<String>()),
        Some(text) => text,
        None => "(absent)".into(),
    }
}

/// Where two texts part, as a byte offset into both; `None` when equal.
pub fn parting(a: &str, b: &str) -> Option<usize> {
    let differ = a.char_indices().zip(b.chars()).find(|((_, x), y)| x != y).map(|((i, _), _)| i);
    differ.or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

/// Up to `n` characters of `text` from byte `from`, line breaks shown.
pub fn snippet(text: &str, from: usize, n: usize) -> String {
    text.get(from..).unwrap_or_default().chars().take(n).collect::<String>().replace('\n', "\\n")
}

// ── one call per turn ────────────────────────────────────────────────────
//
// `parallel_tool_calls: false` (#609, #615): a conversation of a few turns,
// each turn's calls answered, so a model that would make several calls at
// once makes them one turn at a time.

/// The tools the conversations send.
pub const PARALLEL_TOOLS: &[&str] = &[READ_FILE, WRITE_FILE];
/// A task a model does with several calls in one turn when it may.
pub const PARALLEL_PROMPT: &str = "Read these three files: README.md, Cargo.toml and LICENSE.";
/// A conversation's turns at most.
pub const MAX_TURNS: usize = 4;

/// One turn of a conversation.
#[derive(Clone, Debug)]
pub struct Turn {
    pub end: String,
    pub calls: Vec<Call>,
    pub text: String,
}

/// What a tool returned, as the conversation answers each call.
fn result(call: &Call) -> String {
    format!("{} returned: ok, 12 lines.", call.name)
}

/// A conversation on `api` over `tools` (their JSON as sent) from `prompt`,
/// with `parallel_tool_calls` sent as `parallel` (absent when `None`):
/// greedy, thinking off, until a turn makes no call or after
/// [`MAX_TURNS`]; a turn answered other than 200 is an error naming it.
/// `post` sends a body to a path and gives the status and the body back.
pub async fn conversation<F, Fut>(post: F, model: &str, api: Api, tools: &[String], prompt: &str, parallel: Option<bool>) -> Result<Vec<Turn>, String>
where
    F: Fn(&'static str, String) -> Fut,
    Fut: Future<Output = (u16, String)>,
{
    let parallel = parallel.map_or(String::new(), |on| format!(",\"parallel_tool_calls\":{on}"));
    let tools = tools.join(",");
    let mut history: Vec<JsonValue> = vec![json!({ "role": "user", "content": prompt })];
    let mut turns = Vec::new();
    for _ in 0..MAX_TURNS {
        let history_text = serde_json::to_string(&history).unwrap();
        let body = match api {
            Api::Chat => format!(
                r#"{{"model":"{model}","messages":{history_text},"tools":[{tools}],"enable_thinking":false,"temperature":0,"max_tokens":2048{parallel}}}"#
            ),
            Api::Responses => format!(
                r#"{{"model":"{model}","input":{history_text},"tools":[{tools}],"enable_thinking":false,"temperature":0,"max_output_tokens":2048{parallel}}}"#
            ),
        };
        let (status, text) = post(api.path(), body).await;
        if status != 200 {
            return Err(format!("turn {}: {status} {text}", turns.len() + 1));
        }
        let response: JsonValue = serde_json::from_str(&text).map_err(|e| format!("turn {}: {e}: {text}", turns.len() + 1))?;
        let delivered = Delivered::read(api, &response);
        let turn = Turn { end: delivered.end.clone(), calls: delivered.calls.clone(), text: delivered.text.clone() };
        let more = !turn.calls.is_empty();
        match api {
            Api::Chat if more => {
                let calls: Vec<JsonValue> = turn
                    .calls
                    .iter()
                    .map(|call| json!({ "id": call.id, "type": "function", "function": { "name": call.name, "arguments": call.arguments } }))
                    .collect();
                history.push(json!({ "role": "assistant", "content": turn.text, "tool_calls": calls }));
                for call in &turn.calls {
                    history.push(json!({ "role": "tool", "tool_call_id": call.id, "content": result(call) }));
                }
            }
            Api::Responses if more => {
                for call in &turn.calls {
                    history.push(json!({ "type": "function_call", "call_id": call.id, "name": call.name, "arguments": call.arguments }));
                }
                for call in &turn.calls {
                    history.push(json!({ "type": "function_call_output", "call_id": call.id, "output": result(call) }));
                }
            }
            _ => {}
        }
        turns.push(turn);
        if !more {
            break;
        }
    }
    Ok(turns)
}

/// Print a conversation's turns.
pub fn print_turns(what: &str, turns: &[Turn]) {
    eprintln!("{what}: {} turns", turns.len());
    for (i, turn) in turns.iter().enumerate() {
        eprintln!("  turn {}: {}, {} calls{}", i + 1, turn.end, turn.calls.len(), if turn.text.is_empty() { String::new() } else { format!(", text {:?}", snippet(&turn.text, 0, 80)) });
        for call in &turn.calls {
            eprintln!("    {} {}", call.name, call.arguments);
        }
    }
}

/// The control (no `parallel_tool_calls`) makes two calls or more in its
/// first turn, so the prompt shows what the limit changes; under `false`
/// every turn makes one call at most, and the first exactly one, the turn
/// ending as a call ends it (`tool_calls` on chat completions, `completed`
/// on the Responses API). Both conversations printed; what fails, said, so
/// a caller can go on to its next conversation and fail at the end.
pub fn one_call_per_turn(what: &str, api: Api, control: &Result<Vec<Turn>, String>, limited: &Result<Vec<Turn>, String>) -> Vec<String> {
    let mut failures = Vec::new();
    let (control, limited) = match (control, limited) {
        (Ok(control), Ok(limited)) => (control, limited),
        (control, limited) => {
            for (name, error) in [("the control", control.as_ref().err()), ("under false", limited.as_ref().err())] {
                if let Some(error) = error {
                    eprintln!("{what}, {name}: {error}");
                    failures.push(format!("{what}, {name}: {error}"));
                }
            }
            return failures;
        }
    };
    print_turns(&format!("{what}, parallel_tool_calls absent (the control)"), control);
    print_turns(&format!("{what}, parallel_tool_calls false"), limited);
    let first = control.first().map_or(0, |turn| turn.calls.len());
    if first < 2 {
        failures.push(format!("{what}: the control made {first} calls in its first turn, so this prompt does not show the limit"));
    }
    let ends = match api {
        Api::Chat => "tool_calls",
        Api::Responses => "completed",
    };
    match limited.first() {
        Some(opening) if (opening.calls.len(), opening.end.as_str()) == (1, ends) => {}
        opening => failures.push(format!("{what}: the first turn under false is not one call ending {ends}: {opening:?}")),
    }
    for (i, turn) in limited.iter().enumerate() {
        if turn.calls.len() > 1 {
            failures.push(format!("{what}: turn {} made {} calls under parallel_tool_calls false", i + 1, turn.calls.len()));
        }
    }
    failures
}
