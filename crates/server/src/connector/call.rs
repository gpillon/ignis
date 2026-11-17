//! A server tool's call (spec #656): the request it makes of its route, sent
//! in process through the router the server serves, and what the harness
//! reads of the answer.
//!
//! Through the router, not to the handler: the request meets every layer a
//! client's would (the key, the serving gate, the count of model use, the
//! body cap), so a refusal is the route's own status, error JSON and
//! `Retry-After`, word for word. The caller's `Authorization` goes with it.
//! The answer is text in `content`, the one part of a result every harness
//! shows its model; a refusal is that text with `isError` set, and nothing
//! is retried: the agent reads the server's state and decides.

use axum::body::Body;
use axum::http::{header, HeaderValue, Method, Request};
use axum::response::Response;
use axum::Router;
use rmcp::model::{CallToolResult, ContentBlock, JsonObject};
use serde_json::Value;
use tower::ServiceExt;

use super::tools::ServerTool;

/// Call `tool` with `arguments` through `routes`, presenting
/// `authorization`, the harness's own header.
pub(crate) async fn call(
    routes: &Router,
    tool: &ServerTool,
    arguments: Option<JsonObject>,
    authorization: Option<HeaderValue>,
) -> CallToolResult {
    let request = match request(tool, arguments, authorization) {
        Ok(request) => request,
        Err(refused) => return CallToolResult::error(vec![ContentBlock::text(refused)]),
    };
    answer(send(routes, request).await).await
}

/// `request`, sent in process through `routes`: the route's answer.
pub(crate) async fn send(routes: &Router, request: Request<Body>) -> Response {
    routes.clone().oneshot(request).await.unwrap_or_else(|never| match never {})
}

/// The request a call of `tool` sends: the fixed query, then the arguments,
/// as the query of a `GET` or a `DELETE`, else as the JSON body. A tool
/// that declares no input sends no argument.
pub(crate) fn request(
    tool: &ServerTool,
    arguments: Option<JsonObject>,
    authorization: Option<HeaderValue>,
) -> Result<Request<Body>, String> {
    let in_query = matches!(tool.method, Method::GET | Method::DELETE);
    let mut query: Vec<(String, String)> = tool.query.iter().map(|(name, value)| (name.clone(), value.clone())).collect();
    let mut body = None;
    if tool.takes_arguments {
        let arguments = arguments.unwrap_or_default();
        let given = |name: &String| arguments.get(name).is_some_and(|value| !value.is_null());
        if !tool.require_any.is_empty() && !tool.require_any.iter().any(given) {
            let names: Vec<String> = tool.require_any.iter().map(|name| format!("`{name}`")).collect();
            return Err(format!("{}: give {}", tool.tool.name, names.join(" or ")));
        }
        if in_query {
            for (name, value) in arguments {
                if let Some(value) = query_value(&name, value)? {
                    query.push((name, value));
                }
            }
        } else {
            body = Some(Value::Object(arguments).to_string());
        }
    }
    let mut uri = tool.path.clone();
    if !query.is_empty() {
        uri.push('?');
        uri.push_str(&serde_urlencoded::to_string(&query).map_err(|err| format!("the arguments make no query: {err}"))?);
    }
    let request = with_key(Request::builder().method(tool.method.clone()).uri(uri), authorization);
    let request = match body {
        Some(body) => request.header(header::CONTENT_TYPE, "application/json").body(Body::from(body)),
        None => request.body(Body::empty()),
    };
    request.map_err(|err| format!("the arguments make no request: {err}"))
}

/// `request`, presenting `authorization`, the harness's own header, when
/// its call carried one: the route checks it as it would the harness's own
/// request.
pub(crate) fn with_key(request: axum::http::request::Builder, authorization: Option<HeaderValue>) -> axum::http::request::Builder {
    match authorization {
        Some(authorization) => request.header(header::AUTHORIZATION, authorization),
        None => request,
    }
}

/// An argument as a query value: a string as it is, a number or a boolean
/// as JSON writes it, a list of those joined by commas; `null` leaves it
/// out. An object has no query form.
fn query_value(name: &str, value: Value) -> Result<Option<String>, String> {
    let scalar = |value: Value| match value {
        Value::String(text) => Ok(text),
        Value::Number(_) | Value::Bool(_) => Ok(value.to_string()),
        _ => Err(format!("argument `{name}`: a value of a query is a string, a number, a boolean or a list of those")),
    };
    match value {
        Value::Null => Ok(None),
        Value::Array(items) => Ok(Some(items.into_iter().map(scalar).collect::<Result<Vec<_>, _>>()?.join(","))),
        value => scalar(value).map(Some),
    }
}

/// What the harness reads of the route's answer: its body as text; on a
/// refusal, `isError` with the server's error body exactly as it was
/// written, then `Retry-After: <seconds>` as a second text when the
/// answer carries that header.
pub(crate) async fn answer(response: Response) -> CallToolResult {
    let status = response.status();
    let retry_after = response.headers().get(header::RETRY_AFTER).and_then(|value| value.to_str().ok()).map(str::to_owned);
    let body = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
        Ok(body) => body,
        Err(err) => return CallToolResult::error(vec![ContentBlock::text(format!("the server's answer ({status}) could not be read: {err}"))]),
    };
    let text = String::from_utf8_lossy(&body).into_owned();
    if status.is_success() {
        return CallToolResult::success(vec![ContentBlock::text(text)]);
    }
    // An answer with no body (a layer's bare status) says its status.
    let text = if text.trim().is_empty() { format!("HTTP {status}") } else { text };
    let mut content = vec![ContentBlock::text(text)];
    if let Some(retry_after) = retry_after {
        content.push(ContentBlock::text(format!("Retry-After: {retry_after}")));
    }
    CallToolResult::error(content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn tool(method: Method, takes_arguments: bool, query: &[(&str, &str)]) -> ServerTool {
        ServerTool {
            tool: rmcp::model::Tool::new("t", "d", JsonObject::new()),
            method,
            path: "/v1/route".into(),
            query: query.iter().map(|(name, value)| (name.to_string(), value.to_string())).collect::<BTreeMap<_, _>>(),
            takes_arguments,
            changes: false,
            require_any: Vec::new(),
        }
    }

    fn arguments(value: Value) -> Option<JsonObject> {
        value.as_object().cloned()
    }

    async fn body_text(request: Request<Body>) -> String {
        String::from_utf8(axum::body::to_bytes(request.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap()
    }

    /// A `GET` sends the fixed query and its arguments in the query, a list
    /// joined by commas, `null` left out, each value encoded.
    #[tokio::test]
    async fn a_get_sends_its_arguments_in_the_query_after_the_fixed_one() {
        let get = tool(Method::GET, true, &[("format", "yaml")]);
        let request = request(&get, arguments(serde_json::json!({ "names": ["a.b", "c d"], "skip": null, "n": 2 })), None).unwrap();
        assert_eq!(request.uri(), "/v1/route?format=yaml&n=2&names=a.b%2Cc+d");
        assert!(body_text(request).await.is_empty());
        let refused = request_err(&get, serde_json::json!({ "o": { "a": 1 } }));
        assert!(refused.contains("argument `o`"), "{refused}");
    }

    fn request_err(tool: &ServerTool, value: Value) -> String {
        request(tool, arguments(value), None).err().expect("refused")
    }

    /// A `POST` with an input sends its arguments as the JSON body; one with
    /// none sends no body whatever it was given; the caller's key goes too.
    #[tokio::test]
    async fn a_post_sends_its_arguments_as_the_body_and_the_key_goes_with_it() {
        let key = HeaderValue::from_static("Bearer sk-x");
        let post = request(&tool(Method::POST, true, &[]), arguments(serde_json::json!({ "model": "m" })), Some(key.clone())).unwrap();
        assert_eq!((post.method(), post.uri().to_string()), (&Method::POST, "/v1/route".to_owned()));
        assert_eq!(post.headers()[header::AUTHORIZATION], key);
        assert_eq!(post.headers()[header::CONTENT_TYPE], "application/json");
        assert_eq!(body_text(post).await, r#"{"model":"m"}"#);
        let bare = request(&tool(Method::POST, false, &[]), arguments(serde_json::json!({ "ignored": 1 })), None).unwrap();
        assert!(bare.headers().get(header::CONTENT_TYPE).is_none());
        assert!(body_text(bare).await.is_empty());
    }
}
