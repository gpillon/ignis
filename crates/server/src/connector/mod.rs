//! The connector (spec #656): what ignis offers a coding agent's harness
//! beyond chat. This is its MCP server, `POST /v1/mcp`: MCP over Streamable
//! HTTP, built on `rmcp`, on the port and behind the key the rest of `/v1`
//! has.
//!
//! - **The tools** are the routes whose `#[utoipa::path]` carries an
//!   `x-ignis-tool` extension ([`tools`]), read once from the OpenAPI
//!   document the router is built with: `tools/list` is the same bytes
//!   whatever model is loaded, through a switch or a patch.
//! - **A call** goes to its route in process, through the router, with the
//!   caller's key ([`call`]): it meets every gate a client's request meets,
//!   and its refusal comes back as the tool's error, the server's own JSON.
//! - **`/v1/mcp` itself is never held** by the serving gate: `initialize` and
//!   `tools/list` answer while a model switches or warms up, and a tool's
//!   own call meets the gate on its route.
//! - **Stateless**: every request stands alone, no session id, so an ignis
//!   restart strands no harness. A request's answer is `application/json`,
//!   or `text/event-stream` when the call sends a notification (a progress
//!   token's) before it. A `GET` (the server-to-client stream) is `405`.
//! - **Any `Host`** is admitted, as on every other route (behind a tunnel or
//!   on a LAN the name is not a loopback one); rmcp's `Origin` list stays
//!   empty, its default, which checks none.
//!
//! `connector.enabled` off answers the route `404`; `connector.allow_changes`
//! lists the tools that change the server. Both are start-only.

pub mod call;
pub mod tools;

use std::sync::Arc;

use axum::extract::Request;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Router;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, Implementation, ListToolsResult, PaginatedRequestParams, ServerCapabilities,
    ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::never::NeverSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData, RoleServer, ServerHandler};
use tower::ServiceExt;

/// Where the MCP server is served.
pub const PATH: &str = "/v1/mcp";

/// What `initialize` tells the harness's model about ignis, the first line
/// what ignis is: pi shows the first line alone. Static, fixed at start,
/// and under 1 KB with every tool family's lines.
const ABOUT: &str = "\
ignis is the inference server this agent's model runs on; these tools reach the rest of it.
A failed call returns the server's error JSON and Retry-After; nothing is retried for you.";

/// The connector of one router (`api::router`): its MCP service, over the
/// tools its document declares and the router their calls go through.
#[derive(Clone)]
pub struct Connector {
    service: StreamableHttpService<Mcp, NeverSessionManager>,
}

impl Connector {
    /// The connector serving `tools`, each call sent through `routes`: the
    /// server's router without this connector, so it holds no reference to
    /// itself.
    pub fn new(tools: tools::Table, routes: Router) -> Self {
        let shared = Arc::new(Shared { instructions: instructions(&tools), tools, routes });
        let config = StreamableHttpServerConfig::default()
            .with_legacy_session_mode(false)
            .with_json_response(true)
            .disable_allowed_hosts();
        let service = StreamableHttpService::new(
            move || Ok(Mcp(Arc::clone(&shared))),
            Arc::new(NeverSessionManager::default()),
            config,
        );
        Self { service }
    }
}

/// `instructions`, from what `tools` lists.
fn instructions(_tools: &tools::Table) -> String {
    ABOUT.to_owned()
}

/// What every request's handler shares: built once per router.
struct Shared {
    tools: tools::Table,
    instructions: String,
    routes: Router,
}

/// The MCP handler of one request (rmcp builds one per request in
/// stateless mode): a handle on [`Shared`].
#[derive(Clone)]
pub struct Mcp(Arc<Shared>);

impl ServerHandler for Mcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("ignis", env!("CARGO_PKG_VERSION")))
            .with_instructions(self.0.instructions.clone())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(self.0.tools.listed()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.0.tools.named(name).map(|tool| tool.tool.clone())
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let Some(tool) = self.0.tools.named(&request.name) else {
            return Err(ErrorData::invalid_params(format!("unknown tool: {}", request.name), None));
        };
        // The harness's own key: the route checks it as it would the
        // harness's request.
        let authorization = context
            .extensions
            .get::<axum::http::request::Parts>()
            .and_then(|parts| parts.headers.get(header::AUTHORIZATION))
            .cloned();
        Ok(call::call(&self.0.routes, tool, request.arguments, authorization).await.into())
    }
}

/// `POST /v1/mcp` — the connector's MCP server.
#[utoipa::path(
    post,
    path = "/v1/mcp",
    tag = "connector",
    operation_id = "mcp",
    summary = "The connector's MCP server",
    description = "MCP over Streamable HTTP, for a coding agent's harness: one MCP server entry pointing at this URL, with the same bearer key as the rest of `/v1`. The body is one JSON-RPC message of MCP: `initialize`, `tools/list`, `tools/call`, `ping`, or a notification. Each request stands alone (no `Mcp-Session-Id`, so a restart of the server strands no client): it must carry `Content-Type: application/json` and an `Accept` naming both `application/json` and `text/event-stream`. A `GET` is `405`: there is no server-to-client stream.\n\n`initialize` answers ignis's `instructions`, fixed at start. `tools/list` answers the server tools: the routes whose operation in this document carries an `x-ignis-tool` extension (its `name`, the `description` the model reads, the `read_only` and `destructive` hints, and the narrower `input` it takes), sorted by name, the same bytes whatever model is loaded; the tools that change the server are listed only with `connector.allow_changes` on. `tools/call` calls the tool's route on this server with the caller's key, through every check a request of the client's own would meet, and answers its body as text; a refusal is `isError: true`, its text the route's error JSON, then `Retry-After: <seconds>` when the route sent that header. Nothing is retried.\n\nThis route is never held while a model switches or warms up; a tool's call is held, or refused, as its route is. Any `Host` header is admitted. With `connector.enabled` off it answers `404`.",
    request_body(content = Object, description = "One JSON-RPC message of MCP."),
    responses(
        (status = 200, description = "A request's JSON-RPC answer: `application/json`, or `text/event-stream` when the call sent a notification first, its last event the answer.", content(
            (Object = "application/json"),
            (String = "text/event-stream"),
        )),
        (status = 202, description = "A notification, accepted."),
        (status = 400, description = "Not one JSON-RPC message of MCP, or an `MCP-Protocol-Version` header this server does not speak."),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = crate::api::ApiError),
        (status = 404, description = "The connector is off (`connector_disabled`): `connector.enabled` is false.", body = crate::api::ApiError),
        (status = 406, description = "`Accept` does not name both `application/json` and `text/event-stream`."),
        (status = 415, description = "`Content-Type` is not `application/json`."),
    ),
)]
pub(crate) async fn mcp(request: Request) -> Response {
    // `api::router` puts the connector in every request's extensions when
    // `connector.enabled` is on; off, or in the router the connector's own
    // calls go through, there is none.
    let Some(connector) = request.extensions().get::<Connector>().cloned() else {
        return crate::api::error_response(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "connector_disabled",
            "the connector is off on this server (connector.enabled is false)",
        );
    };
    let response = connector.service.oneshot(request).await.unwrap_or_else(|never| match never {});
    response.map(axum::body::Body::new).into_response()
}

#[cfg(test)]
mod tests {
    /// Spec #656: the base of `instructions` keeps within its share of the
    /// 1 KB the whole text has, so each tool family's lines fit beside it.
    #[test]
    fn the_base_instructions_keep_within_their_share() {
        assert!(super::ABOUT.len() <= 200, "{} bytes", super::ABOUT.len());
        assert!(!super::ABOUT.lines().next().unwrap().is_empty());
    }
}
