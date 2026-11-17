//! `decide` (spec #656, ticket #659): the server tool declared on
//! `POST /v1/decide` (`crate::decide::decide`), whose call sends the route
//! its arguments as the harness wrote them.
//!
//! A decision reads its `state` and its `questions` in the order their keys
//! were written (`crate::decide::OrderedValue`): a `state` record in another
//! order is another prompt, and a `choice`'s options in another order sit
//! at other answer labels, so the same readout names another option. rmcp
//! hands a tool its arguments as a `serde_json::Map`, which sorts its keys
//! in this build (no `preserve_order`), so the generic call
//! ([`super::call`]) would ask the model a decision nobody wrote. This tool
//! reads them instead from the request's body as the harness sent it
//! ([`super::Written`], kept by `/v1/mcp` before rmcp parses it), and sends
//! the route the `arguments` object's text unchanged. Without that body,
//! the call is the tool's error, never the sorted map.

use axum::body::Body;
use axum::http::{header, HeaderValue, Request};
use axum::Router;
use rmcp::model::{CallToolResult, ContentBlock};
use serde::Deserialize;
use serde_json::value::RawValue;

use super::call;
use super::tools::ServerTool;
use super::Written;

/// The tool's name, as its route's extension declares it.
pub(crate) const TOOL: &str = "decide";

/// What `instructions` say of the tool: when to use it, and that it is the
/// 27B's (Flash-Next has no readouts).
pub(crate) const INSTRUCTIONS: &str = "decide classifies or judges content with typed questions (yes/no, choice, score) read off the model; the 27B only.";

/// The arguments a call may carry, the route's own: its `model`, which
/// would switch the server, and its thinking fields are not the tool's.
const ARGUMENTS: [&str; 2] = ["state", "questions"];

/// Call `decide` through `routes` with the arguments `written` holds,
/// presenting `authorization`, the harness's own header.
pub(crate) async fn decide(
    routes: &Router,
    tool: &ServerTool,
    written: Option<&Written>,
    authorization: Option<HeaderValue>,
) -> CallToolResult {
    let Some(written) = written else {
        return unread("the call's body as the harness wrote it was not kept, and without it the order of its keys would be lost");
    };
    let body = match arguments(&written.0) {
        Ok(Arguments::Taken(body)) => body,
        Ok(Arguments::Refused(name)) => {
            let message = format!("{TOOL} takes `state` and `questions` only, not `{name}`");
            return call::answer(crate::api::bad_request_param(&message, &name)).await;
        }
        Err(why) => return unread(&why),
    };
    let mut request = Request::builder()
        .method(tool.method.clone())
        .uri(&tool.path)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(authorization) = authorization {
        request = request.header(header::AUTHORIZATION, authorization);
    }
    match request.body(Body::from(body)) {
        Ok(request) => call::answer(call::send(routes, request).await).await,
        Err(err) => unread(&format!("the arguments make no request: {err}")),
    }
}

/// What a call's message holds for the route.
#[derive(Debug, PartialEq)]
enum Arguments {
    /// The `arguments` object's text, as written: the route's body.
    Taken(String),
    /// An argument the tool does not take, named.
    Refused(String),
}

/// The JSON-RPC message of a `tools/call`, read for its arguments' text.
#[derive(Deserialize)]
struct Message<'a> {
    #[serde(borrow)]
    params: Params<'a>,
}

#[derive(Deserialize)]
struct Params<'a> {
    name: String,
    #[serde(borrow, default)]
    arguments: Option<&'a RawValue>,
}

/// `message`'s arguments for the route: the object's text exactly as it
/// was written, keys, order, spacing and numbers' spelling, when each key
/// is one of [`ARGUMENTS`]. No arguments is an empty object, which the
/// route refuses naming what is missing. Every key is checked, a key
/// written twice too: the route refuses a duplicate itself.
fn arguments(message: &[u8]) -> Result<Arguments, String> {
    let message: Message = serde_json::from_slice(message).map_err(|err| format!("the call's message does not read: {err}"))?;
    if message.params.name != TOOL {
        return Err(format!("the call's message names `{}`, not `{TOOL}`", message.params.name));
    }
    let Some(arguments) = message.params.arguments else {
        return Ok(Arguments::Taken("{}".to_owned()));
    };
    let keys: crate::decide::Ordered<&RawValue> =
        serde_json::from_str(arguments.get()).map_err(|err| format!("the arguments are not one JSON object: {err}"))?;
    if let Some((name, _)) = keys.entries().iter().find(|(name, _)| !ARGUMENTS.contains(&name.as_str())) {
        return Ok(Arguments::Refused(name.clone()));
    }
    Ok(Arguments::Taken(arguments.get().to_owned()))
}

/// The tool's error when the arguments as written cannot be read.
fn unread(why: &str) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(format!("{TOOL}: {why}"))])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_of(name: &str, arguments: &str) -> Vec<u8> {
        format!(r#"{{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{{"name":"{name}","arguments":{arguments}}}}}"#).into_bytes()
    }

    /// Spec #656: the route is sent the `arguments` object byte for byte,
    /// its keys out of alphabetical order at every depth, its spacing and a
    /// number's spelling as written.
    #[test]
    fn the_route_is_sent_the_arguments_byte_for_byte() {
        let written = "{ \"state\": {\"zeta\": 149.0, \"alpha\": [2, 1]},\n  \"questions\": {\"z\": {\"type\": \"choice\", \"instructions\": \"Which?\", \"criteria\": {\"zulu\": \"z\", \"alpha\": \"a\"}}} }";
        assert_eq!(arguments(&call_of(TOOL, written)), Ok(Arguments::Taken(written.to_owned())));
        assert_eq!(arguments(&call_of(TOOL, r#"{"questions": {}, "state": "s"}"#)), Ok(Arguments::Taken(r#"{"questions": {}, "state": "s"}"#.to_owned())));
        let none = br#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"decide"}}"#;
        assert_eq!(arguments(none), Ok(Arguments::Taken("{}".to_owned())));
    }

    /// The route's `model` and thinking fields are refused by name, wherever
    /// they sit in the object.
    #[test]
    fn any_other_argument_is_refused_by_name() {
        for name in ["model", "enable_thinking", "reasoning_effort", "preserve_thinking", "chat_template_kwargs", "stateX"] {
            let written = format!(r#"{{"state": "s", "{name}": 1, "questions": {{}}}}"#);
            assert_eq!(arguments(&call_of(TOOL, &written)), Ok(Arguments::Refused(name.to_owned())), "{name}");
        }
    }

    /// A message that is not a call of this tool, or whose arguments are
    /// not an object, is no body to send: never the sorted map.
    #[test]
    fn a_message_that_is_no_call_of_decide_is_refused() {
        assert!(arguments(&call_of("read_catalog", r#"{"state": "s"}"#)).unwrap_err().contains("read_catalog"));
        assert!(arguments(&call_of(TOOL, "[1]")).is_err());
        assert!(arguments(b"[{}]").is_err());
    }

    /// Spec #656: `decide`'s line of `instructions` keeps within its share
    /// of the 1 KB, and says the tool is the 27B's.
    #[test]
    fn the_instructions_line_keeps_within_its_share() {
        assert!(INSTRUCTIONS.len() <= 120, "{} bytes", INSTRUCTIONS.len());
        assert!(INSTRUCTIONS.starts_with(TOOL) && INSTRUCTIONS.contains("27B"), "{INSTRUCTIONS}");
        assert_eq!(INSTRUCTIONS.lines().count(), 1);
    }

    /// The question types the tool's input lists are the route's own, as
    /// the document states them.
    #[test]
    fn the_question_types_the_tool_lists_are_the_routes() {
        let document = serde_json::to_value(crate::api::openapi()).unwrap();
        let tool = &document["paths"]["/v1/decide"]["post"]["x-ignis-tool"];
        let listed = &tool["input"]["properties"]["questions"]["additionalProperties"]["properties"]["type"]["enum"];
        assert_eq!(listed, &document["components"]["schemas"]["QuestionKind"]["enum"], "{tool}");
        assert!(listed.as_array().is_some_and(|kinds| kinds.len() == 8), "{listed}");
    }
}
