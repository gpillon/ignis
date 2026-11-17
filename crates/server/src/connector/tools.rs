//! The connector's tool table (spec #656): built once, at start, from the
//! OpenAPI document `/v1/openapi.json` serves, where each server tool is
//! declared on its own route's `#[utoipa::path]` by an `x-ignis-tool`
//! extension. A tool and its route cannot drift apart, and the list a
//! harness reads depends on the build and the start's `connector` group
//! alone: never on the model loaded, a switch or a patch.
//!
//! A route becomes a tool with one annotation:
//!
//! ```text
//! extensions(("x-ignis-tool" = json!({
//!     "name": "read_catalog",
//!     "description": "When to call it, for the model.",
//!     "read_only": true,
//!     "destructive": false,
//! })))
//! ```
//!
//! and, optionally, `"input"`: the tool's input, a JSON Schema object
//! narrower than the route's own (absent, the tool takes no arguments);
//! `"query"`: query parameters the tool always sends (`{"format": "yaml"}`);
//! `"changes": true`: the tool changes the server, and is listed only with
//! `connector.allow_changes` on; `"require_any": ["group", "names"]`: a call
//! must carry at least one of those arguments, else it is a tool error
//! before the route is asked. A call sends its arguments as the route's
//! query on a `GET` or a `DELETE`, as its JSON body on any other method
//! ([`super::call`]).

use std::collections::BTreeMap;

use axum::http::Method;
use rmcp::model::{JsonObject, Tool, ToolAnnotations};
use serde::Deserialize;

/// The extension's key on an operation of the document.
pub const EXTENSION: &str = "x-ignis-tool";

/// What an `x-ignis-tool` extension declares. Any other key is refused, so
/// a misspelt one fails the table's build rather than being ignored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    /// The tool's name, unique in the table.
    name: String,
    /// What the model reads: when to call it. The route's own description
    /// is for an API reader.
    description: String,
    /// `readOnlyHint`: the tool changes nothing.
    read_only: bool,
    /// `destructiveHint`: what the tool changes may not be undone by
    /// calling it again.
    destructive: bool,
    /// The tool's input, narrower than the route's; absent, none.
    #[serde(default)]
    input: Option<JsonObject>,
    /// The query the tool always sends.
    #[serde(default)]
    query: BTreeMap<String, String>,
    /// Listed only with `connector.allow_changes` on.
    #[serde(default)]
    changes: bool,
    /// Argument names of which a call must carry at least one.
    #[serde(default)]
    require_any: Vec<String>,
}

/// One server tool: what `tools/list` shows of it, and the route it calls.
#[derive(Debug, Clone)]
pub struct ServerTool {
    /// The tool as the harness lists it.
    pub tool: Tool,
    /// The route's method.
    pub method: Method,
    /// The route's path.
    pub path: String,
    /// The query the tool always sends.
    pub query: BTreeMap<String, String>,
    /// Whether the tool declares an input, which its call sends on.
    pub takes_arguments: bool,
    /// Whether it changes the server (`connector.allow_changes`).
    pub changes: bool,
    /// Argument names of which a call must carry at least one (none: no
    /// requirement).
    pub require_any: Vec<String>,
}

/// The tools a connector lists, sorted by name.
#[derive(Debug, Clone, Default)]
pub struct Table {
    tools: Vec<ServerTool>,
}

impl Table {
    /// The tools `document` declares, those that change the server only
    /// with `allow_changes`.
    ///
    /// # Panics
    ///
    /// On a malformed extension, or two tools of one name: the document is
    /// the build's own, so this is a bug its tests meet, never an operator's.
    pub fn from_document(document: &utoipa::openapi::OpenApi, allow_changes: bool) -> Self {
        let mut tools = Vec::new();
        for (path, item) in &document.paths.paths {
            let operations = [
                (Method::GET, &item.get),
                (Method::PUT, &item.put),
                (Method::POST, &item.post),
                (Method::DELETE, &item.delete),
                (Method::PATCH, &item.patch),
            ];
            for (method, operation) in operations {
                let Some(declared) = operation.as_ref().and_then(|operation| operation.extensions.as_ref()?.get(EXTENSION))
                else {
                    continue;
                };
                let declared: Declaration = serde_json::from_value(declared.clone())
                    .unwrap_or_else(|err| panic!("{method} {path}: its {EXTENSION} extension: {err}"));
                tools.push(ServerTool::declared(declared, method, path));
            }
        }
        tools.sort_by(|a, b| a.tool.name.cmp(&b.tool.name));
        for pair in tools.windows(2) {
            assert_ne!(pair[0].tool.name, pair[1].tool.name, "two routes declare the tool {}", pair[0].tool.name);
        }
        tools.retain(|tool| allow_changes || !tool.changes);
        Self { tools }
    }

    /// The tools as `tools/list` answers them, in name order.
    pub fn listed(&self) -> Vec<Tool> {
        self.tools.iter().map(|tool| tool.tool.clone()).collect()
    }

    /// The listed tool named `name`: a tool the table left out is not
    /// callable either.
    pub fn named(&self, name: &str) -> Option<&ServerTool> {
        self.tools.iter().find(|tool| tool.tool.name == name)
    }

    /// Whether a tool that changes the server is listed.
    pub fn lists_changes(&self) -> bool {
        self.tools.iter().any(|tool| tool.changes)
    }
}

impl ServerTool {
    fn declared(declared: Declaration, method: Method, path: &str) -> Self {
        let takes_arguments = declared.input.is_some();
        let input = declared.input.unwrap_or_else(|| {
            serde_json::json!({ "type": "object", "properties": {} }).as_object().cloned().expect("an object")
        });
        assert_eq!(input.get("type").and_then(|kind| kind.as_str()), Some("object"), "{}: an input is an object schema", declared.name);
        // Both hints, always: a hint left out reads to a client as MCP's
        // default, which for `destructiveHint` is true.
        let annotations = ToolAnnotations::new().read_only(declared.read_only).destructive(declared.destructive);
        Self {
            tool: Tool::new(declared.name, declared.description, input).with_annotations(annotations),
            method,
            path: path.to_owned(),
            query: declared.query,
            takes_arguments,
            changes: declared.changes,
            require_any: declared.require_any,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use utoipa::openapi::extensions::ExtensionsBuilder;
    use utoipa::openapi::path::{HttpMethod, OperationBuilder, PathItem, PathsBuilder};

    /// A document whose routes declare `tools`, each `(method, path, extension)`.
    fn document(tools: &[(HttpMethod, &str, serde_json::Value)]) -> utoipa::openapi::OpenApi {
        let mut paths = PathsBuilder::new();
        for (method, path, extension) in tools {
            let extensions = ExtensionsBuilder::new().add(EXTENSION, extension.clone()).build();
            let operation = OperationBuilder::new().extensions(Some(extensions)).build();
            paths = paths.path(*path, PathItem::new(method.clone(), operation));
        }
        utoipa::openapi::OpenApiBuilder::new().paths(paths.build()).build()
    }

    fn declared(name: &str, changes: bool) -> serde_json::Value {
        json!({ "name": name, "description": "d", "read_only": !changes, "destructive": changes, "changes": changes })
    }

    fn names(table: &Table) -> Vec<String> {
        table.listed().iter().map(|tool| tool.name.to_string()).collect()
    }

    /// The tools that change the server are listed only with
    /// `allow_changes`, and the list is in name order whatever the paths'.
    #[test]
    fn the_change_tools_are_listed_only_with_allow_changes_and_the_list_is_in_name_order() {
        let document = document(&[
            (HttpMethod::Get, "/v1/a", declared("zeta_read", false)),
            (HttpMethod::Post, "/v1/b", declared("alpha_change", true)),
            (HttpMethod::Get, "/v1/c", declared("mid_read", false)),
        ]);
        let off = Table::from_document(&document, false);
        assert_eq!(names(&off), ["mid_read", "zeta_read"]);
        assert!(!off.lists_changes());
        assert!(off.named("alpha_change").is_none(), "an unlisted tool is not callable");
        let on = Table::from_document(&document, true);
        assert_eq!(names(&on), ["alpha_change", "mid_read", "zeta_read"]);
        assert!(on.lists_changes());
        let change = on.named("alpha_change").unwrap();
        assert_eq!((change.method.clone(), change.path.as_str()), (Method::POST, "/v1/b"));
    }

    /// A tool with no input takes an empty object, and both hints are
    /// written whatever their value; a declared input and query are kept.
    #[test]
    fn a_tool_lists_both_hints_and_its_input() {
        let narrowed = json!({ "type": "object", "properties": { "group": { "type": "string" } }, "required": ["group"] });
        let document = document(&[
            (HttpMethod::Get, "/v1/a", declared("plain", false)),
            (
                HttpMethod::Get,
                "/v1/b",
                json!({ "name": "narrowed", "description": "d", "read_only": true, "destructive": false, "input": narrowed, "query": { "format": "yaml" } }),
            ),
        ]);
        let table = Table::from_document(&document, false);
        let listed = serde_json::to_value(table.listed()).unwrap();
        assert_eq!(listed[1]["name"], "plain");
        assert_eq!(listed[1]["inputSchema"], json!({ "type": "object", "properties": {} }));
        assert_eq!(listed[1]["annotations"], json!({ "readOnlyHint": true, "destructiveHint": false }));
        assert_eq!(listed[0]["inputSchema"], narrowed);
        let tool = table.named("narrowed").unwrap();
        assert!(tool.takes_arguments && !table.named("plain").unwrap().takes_arguments);
        assert_eq!(tool.query, BTreeMap::from([("format".to_owned(), "yaml".to_owned())]));
    }

    #[test]
    #[should_panic(expected = "two routes declare the tool same")]
    fn two_routes_may_not_declare_one_tool() {
        let document = document(&[(HttpMethod::Get, "/v1/a", declared("same", false)), (HttpMethod::Post, "/v1/b", declared("same", true))]);
        Table::from_document(&document, false);
    }

    #[test]
    #[should_panic(expected = "unknown field `read_onyl`")]
    fn a_misspelt_key_fails_the_build() {
        let document = document(&[(HttpMethod::Get, "/v1/a", json!({ "name": "n", "description": "d", "read_onyl": true, "destructive": false }))]);
        Table::from_document(&document, false);
    }
}
