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
//!   A tool whose call is more than its route's has a module of its own:
//!   `decide` ([`decide`]), which sends its arguments as the harness wrote
//!   them ([`Written`]), and `generate_image` ([`image`]), whose calls the
//!   server runs one at a time, whichever router they came by
//!   ([`crate::Server::image_calls`]).
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
//! - **The skill** is the connector's other route, `GET /v1/connector/skill`
//!   ([`skill`]): the `ignis` skill whose scripts handle files on the
//!   harness's side.
//!
//! `connector.enabled` off answers both routes `404`; `connector.allow_changes`
//! lists the tools that change the server. Both are start-only.

pub mod call;
pub mod decide;
pub mod image;
pub mod skill;
pub mod tools;

use std::sync::Arc;

use axum::body::{Body, Bytes};
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

/// The skill's install, a line per shell, in `instructions`: the skill's
/// scripts handle a file on the harness's side, which no tool can. Both
/// work with `IGNIS_BASE_URL` unset, and with `IGNIS_API_KEY` unset (a server
/// without a key ignores the header). The PowerShell form downloads to a file
/// first, `s`, since Windows PowerShell 5.1 pipes a native command's output
/// as text, which would corrupt the archive; it does so in the skills folder,
/// handing `tar` that relative name, which any `tar` reads (Git for
/// Windows's, first in some `PATH`s, reads `C:` as a host), deletes it, and
/// goes back where it was. At most 380 bytes, the ticket's share of the 1 KB;
/// the guide's forms are the longer, spelt-out ones.
pub(crate) const INSTALL: &str = r#"Skill (files) sh: d=~/.agents/skills;mkdir -p $d;curl -fsS -H "Authorization: Bearer $IGNIS_API_KEY" ${IGNIS_BASE_URL:-http://127.0.0.1:8000}/v1/connector/skill|tar xz -C $d
ps: $u=$env:IGNIS_BASE_URL;if(!$u){$u='http://127.0.0.1:8000'};md -f ~/.agents/skills|pushd;curl.exe -fsS -H "Authorization: Bearer $env:IGNIS_API_KEY" -o s $u/v1/connector/skill;tar xzf s;rm s;popd"#;

/// The change tools' line in `instructions`, only when they are listed
/// (`connector.allow_changes`): the harm a patch can do, at 120 bytes at
/// most.
pub(crate) const CHANGES: &str = "change_config, download_model, cancel_download change this server; a patch to a reload field reloads the model.";

/// The connector of one router (`api::router`): its MCP service, over the
/// tools its document declares and the router their calls go through.
#[derive(Clone)]
pub struct Connector {
    service: StreamableHttpService<Mcp, NeverSessionManager>,
    /// The server the router serves: whose body limit `/v1/mcp` names.
    server: Arc<crate::Server>,
}

impl Connector {
    /// The connector serving `tools`, each call sent through `routes`: the
    /// server's router without this connector, so it holds no reference to
    /// itself. `server` is the one `routes` serve: what `generate_image`
    /// reads the models of, and holds its calls on.
    pub fn new(tools: tools::Table, routes: Router, server: Arc<crate::Server>) -> Self {
        let shared = Arc::new(Shared { instructions: instructions(&tools), tools, routes, server: Arc::clone(&server) });
        let config = StreamableHttpServerConfig::default()
            .with_legacy_session_mode(false)
            .with_json_response(true)
            .disable_allowed_hosts()
            // One limit on a body: the server's own, which holds every `/v1`
            // body before this route reads it (`api::request_body_limit`).
            // rmcp's default of its own, 4 MiB, would be a second.
            .with_max_request_body_bytes(usize::MAX);
        let service = StreamableHttpService::new(
            move || Ok(Mcp(Arc::clone(&shared))),
            Arc::new(NeverSessionManager::default()),
            config,
        );
        Self { service, server }
    }
}

/// `instructions`, from what `tools` lists.
fn instructions(tools: &tools::Table) -> String {
    let mut text = ABOUT.to_owned();
    if tools.named(decide::TOOL).is_some() {
        text.push('\n');
        text.push_str(decide::INSTRUCTIONS);
    }
    if tools.named(image::TOOL).is_some() {
        text.push('\n');
        text.push_str(image::INSTRUCTIONS);
    }
    text.push('\n');
    text.push_str(INSTALL);
    if tools.lists_changes() {
        text.push('\n');
        text.push_str(CHANGES);
    }
    text
}

/// What every request's handler shares: built once per router.
struct Shared {
    tools: tools::Table,
    instructions: String,
    routes: Router,
    server: Arc<crate::Server>,
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
        // A tool whose call is more than its route's has its own module.
        let result = match tool.tool.name.as_ref() {
            decide::TOOL => decide::decide(&self.0.routes, tool, written(&context), authorization).await,
            image::TOOL => image::generate(&self.0, tool, request.arguments, authorization, &context).await,
            _ => call::call(&self.0.routes, tool, request.arguments, authorization).await,
        };
        Ok(result.into())
    }
}

/// The body of a `/v1/mcp` request as the harness wrote it, kept in the
/// request's extensions by [`mcp`] before rmcp parses it. rmcp hands a tool
/// its arguments as a `serde_json::Map`, whose keys this build sorts; a tool
/// whose arguments' key order is part of their meaning reads them here
/// (`decide`, [`decide`]).
#[derive(Clone)]
pub(crate) struct Written(pub(crate) Bytes);

/// The body `context`'s request was sent with, as written ([`Written`]):
/// rmcp puts the request's HTTP parts, its extensions among them, in the
/// context of the call.
fn written(context: &RequestContext<RoleServer>) -> Option<&Written> {
    context.extensions.get::<axum::http::request::Parts>()?.extensions.get::<Written>()
}

/// The `404` of a connector that is off, the one answer both of its routes
/// give.
fn disabled() -> Response {
    crate::api::error_response(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "connector_disabled",
        "the connector is off on this server (connector.enabled is false)",
    )
}

/// `POST /v1/mcp` — the connector's MCP server.
#[utoipa::path(
    post,
    path = "/v1/mcp",
    tag = "connector",
    operation_id = "mcp",
    summary = "The connector's MCP server",
    description = "MCP over Streamable HTTP, for a coding agent's harness: one MCP server entry pointing at this URL, with the same bearer key as the rest of `/v1`. The body is one JSON-RPC message of MCP: `initialize`, `tools/list`, `tools/call`, `ping`, or a notification. Each request stands alone (no `Mcp-Session-Id`, so a restart of the server strands no client): it must carry `Content-Type: application/json` and an `Accept` naming both `application/json` and `text/event-stream`. A `GET` is `405`: there is no server-to-client stream.\n\n`initialize` answers ignis's `instructions`, fixed at start. `tools/list` answers the server tools: the routes whose operation in this document carries an `x-ignis-tool` extension (its `name`, the `description` the model reads, the `read_only` and `destructive` hints, and the narrower `input` it takes), sorted by name, the same bytes whatever model is loaded; the tools that change the server are listed only with `connector.allow_changes` on. `tools/call` calls the tool's route on this server with the caller's key, through every check a request of the client's own would meet, and answers its body as text (`generate_image` answers a PNG preview of its image, at most 512 px on its longest edge, then a line of text; with vision off in the agent's model, a line of text naming the skill's script and no image); a refusal is `isError: true`, its text the route's error JSON, then `Retry-After: <seconds>` when the route sent that header. Nothing is retried.\n\nThis route is never held while a model switches or warms up; a tool's call is held, or refused, as its route is. Any `Host` header is admitted. With `connector.enabled` off it answers `404`.",
    request_body(content = Object, description = "One JSON-RPC message of MCP."),
    responses(
        (status = 200, description = "A request's JSON-RPC answer: `application/json`, or `text/event-stream` when the call sent a notification first, its last event the answer.", content(
            (Object = "application/json"),
            (String = "text/event-stream"),
        )),
        (status = 202, description = "A notification, accepted."),
        (status = 400, description = "Not one JSON-RPC message of MCP, or an `MCP-Protocol-Version` header this server does not speak; or a body that could not be read, with the error JSON (`invalid_request_error`).", body = crate::api::ApiError),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = crate::api::ApiError),
        (status = 404, description = "The connector is off (`connector_disabled`): `connector.enabled` is false.", body = crate::api::ApiError),
        (status = 406, description = "`Accept` does not name both `application/json` and `text/event-stream`."),
        (status = 413, description = "The body is over the server's limit on every `/v1` body, 16 MiB on a text-only load and 384 MiB with vision on (`request_body_too_large`).", body = crate::api::ApiError),
        (status = 415, description = "`Content-Type` is not `application/json`."),
    ),
)]
pub(crate) async fn mcp(request: Request) -> Response {
    // `api::router` puts the connector in every request's extensions when
    // `connector.enabled` is on; off, or in the router the connector's own
    // calls go through, there is none.
    let Some(connector) = request.extensions().get::<Connector>().cloned() else {
        return disabled();
    };
    // The body is read whole first and kept as written ([`Written`]), held
    // to the server's limit on every `/v1` body, which its layer set.
    let (mut parts, body) = request.into_parts();
    let body = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(body) => body,
        Err(err) if past_its_limit(&err) => {
            return crate::api::body_too_large(crate::api::request_body_limit(connector.server.active().media.is_some()));
        }
        Err(err) => return crate::api::bad_request(&format!("the request body could not be read: {err}")),
    };
    parts.extensions.insert(Written(body.clone()));
    let request = Request::from_parts(parts, Body::from(body));
    let response = connector.service.oneshot(request).await.unwrap_or_else(|never| match never {});
    response.map(axum::body::Body::new).into_response()
}

/// Whether reading a body failed on a length limit: the server's own for
/// every `/v1` body.
fn past_its_limit(err: &axum::Error) -> bool {
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(err) = cause {
        if err.is::<http_body_util::LengthLimitError>() {
            return true;
        }
        cause = err.source();
    }
    false
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

    /// Spec #656, ticket #660: the change tools' line keeps within its share.
    #[test]
    fn the_change_tools_line_keeps_within_its_share() {
        assert!(super::CHANGES.len() <= 120, "{} bytes", super::CHANGES.len());
        assert_eq!(super::CHANGES.lines().count(), 1);
    }

    /// Spec #656, ticket #662: the skill's two install forms keep within
    /// their share of the 1 KB.
    #[test]
    fn the_skill_install_keeps_within_its_share() {
        assert!(super::INSTALL.len() <= 380, "{} bytes", super::INSTALL.len());
    }

    /// What a PowerShell install form does wrong, if anything. It must work
    /// in the skills folder (`Push-Location`), download the archive there
    /// under a relative name, give `tar` that name alone, delete it, and
    /// leave the shell where it was: a `C:` path is a host name to Git for
    /// Windows's `tar`, which may come first in `PATH`.
    fn powershell_install_faults(form: &str) -> Vec<String> {
        let commands: Vec<&str> = form.split(['\n', ';']).map(str::trim).filter(|command| !command.is_empty()).collect();
        let at = |found: &dyn Fn(&str) -> bool| commands.iter().position(|command| found(command));
        let words = |command: &str| command.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        let mut faults = Vec::new();
        let into = at(&|command| words(command).iter().any(|word| word.ends_with("pushd") || word == "Push-Location"));
        let back = at(&|command| command == "popd" || command == "Pop-Location");
        let download = at(&|command| command.starts_with("curl.exe "));
        let archive = download.and_then(|i| words(commands[i]).into_iter().skip_while(|word| word != "-o").nth(1));
        let Some(archive) = archive else {
            return vec![format!("no `curl.exe … -o <file>`: {form}")];
        };
        if archive.contains(['/', '\\', '$', ':']) {
            faults.push(format!("the archive is downloaded to `{archive}`, not a name relative to the skills folder"));
        }
        if let Some(into) = into {
            // The folder named there, or a variable set to it.
            let set = |word: &str| {
                commands.iter().any(|command| command.replace(' ', "").starts_with(&format!("{word}=")) && command.ends_with(".agents/skills\""))
            };
            if !commands[into].contains(".agents/skills") && !words(commands[into]).iter().any(|word| set(word)) {
                faults.push(format!("`{}` enters no skills folder", commands[into]));
            }
        }
        let tar = at(&|command| command.starts_with("tar "));
        if let Some(tar) = tar.filter(|&tar| words(commands[tar]) != ["tar", "xzf", archive.as_str()]) {
            faults.push(format!("`{}` is not `tar xzf {archive}`", commands[tar]));
        }
        let removed = at(&|command| ["rm", "Remove-Item"].iter().any(|remove| words(command) == [*remove, archive.as_str()]));
        let order = [into, download, tar, removed, back];
        if order.iter().any(Option::is_none) || !order.windows(2).all(|pair| pair[0] < pair[1]) {
            faults.push(format!("not in the skills folder, download, `tar`, delete, back: {order:?}"));
        }
        faults
    }

    /// Spec #656, ticket #662: the PowerShell install in `instructions` and
    /// the guide's work in the skills folder with relative names and leave
    /// no file behind.
    #[test]
    fn the_powershell_installs_work_in_the_skills_folder_and_leave_nothing_behind() {
        let line = super::INSTALL.lines().find_map(|line| line.strip_prefix("ps: ")).expect("the PowerShell line");
        assert_eq!(powershell_install_faults(line), Vec::<String>::new(), "{line}");
        let guide = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/user/guides/connector.md")).unwrap();
        let block = guide.split("```powershell\n").nth(1).and_then(|rest| rest.split("```").next()).expect("the guide's PowerShell form");
        assert_eq!(powershell_install_faults(block), Vec::<String>::new(), "{block}");
        let bad = r#"$d="$HOME/.agents/skills";curl.exe -fsS -o $d/s http://h/v1/connector/skill;tar xzf $d/s -C $d"#;
        assert_eq!(powershell_install_faults(bad).len(), 3, "{:?}", powershell_install_faults(bad));
    }
}
