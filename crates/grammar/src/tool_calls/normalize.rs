//! A tool's `parameters` normalized into strict form (GitHub #614, spec
//! #582, ADR 0094): what the Responses API does by default with a function
//! tool sent without `strict`, as OpenAI's does.
//!
//! - **Every object is closed**: `additionalProperties: false`, so no
//!   parameter the schema does not declare is admitted.
//! - **Every property is required** in the strict form, which is what the
//!   grammar writes of a nested object either way; the top-level parameters
//!   the client left optional are returned beside it, and a call may leave
//!   them out, its arguments then omitting them. The arguments stay valid
//!   under the client's own schema.
//! - **What closing would change is not normalized**: an object that admits
//!   other keys by name (`additionalProperties` `true` or a schema), a nested
//!   object that declares no property (a map, which closing would hold to
//!   `{}`), and a `required` naming a key the object does not declare. The
//!   strict form must then compile under `strict`
//!   ([`crate::GrammarCache::json_schema`]): a schema outside OpenAI's strict
//!   subset does not normalize either. Such a tool stays free.
//!
//! Only the keywords that hold schemas are walked (`properties`, `items`,
//! `anyOf`, `$defs`, `definitions`): `enum`, `const` and the annotations
//! hold data, never a schema.

use crate::json::Json;

/// A tool's `parameters` in strict form, and the parameters the client left
/// optional, in the schema's order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StrictForm {
    /// The strict form, compact, key order kept.
    pub schema: String,
    pub optional: Vec<String>,
}

/// `parameters` in strict form, or `None` where closing its objects would
/// change what the client's schema admits (above).
pub(crate) fn strict_form(parameters: &Json) -> Option<StrictForm> {
    let mut schema = parameters.clone();
    let optional = close(&mut schema, true)?;
    Some(StrictForm { schema: schema.compact(), optional })
}

/// Close `schema` and every subschema it holds, and return the properties
/// its own object leaves optional; `root` for the tool's `parameters`, whose
/// object may declare none (a tool without parameters).
fn close(schema: &mut Json, root: bool) -> Option<Vec<String>> {
    let Json::Obj(members) = schema else { return Some(Vec::new()) };
    let mut optional = Vec::new();
    if is_object(members) {
        if !matches!(member(members, "additionalProperties"), None | Some(Json::Bool(false))) {
            return None;
        }
        let declared: Vec<String> = match member(members, "properties") {
            Some(Json::Obj(properties)) => properties.iter().map(|(name, _)| name.clone()).collect(),
            None => Vec::new(),
            Some(_) => return None,
        };
        if declared.is_empty() && !root {
            return None;
        }
        let required: Vec<&str> = match member(members, "required") {
            Some(Json::Arr(names)) => names.iter().map(Json::as_str).collect::<Option<_>>()?,
            None => Vec::new(),
            Some(_) => return None,
        };
        if required.iter().any(|name| !declared.iter().any(|d| d == name)) {
            return None;
        }
        optional = declared.iter().filter(|name| !required.contains(&name.as_str())).cloned().collect();
        set(members, "additionalProperties", Json::Bool(false));
        set(members, "required", Json::Arr(declared.into_iter().map(Json::Str).collect()));
    }
    for (keyword, value) in members.iter_mut() {
        match (keyword.as_str(), value) {
            ("properties" | "$defs" | "definitions", Json::Obj(schemas)) => {
                for (_, schema) in schemas {
                    close(schema, false)?;
                }
            }
            ("anyOf", Json::Arr(schemas)) => {
                for schema in schemas {
                    close(schema, false)?;
                }
            }
            ("items", schema) => {
                close(schema, false)?;
            }
            _ => {}
        }
    }
    Some(optional)
}

/// Whether a schema's `type` is `object`, alone or beside others.
fn is_object(members: &[(String, Json)]) -> bool {
    match member(members, "type") {
        Some(Json::Str(kind)) => kind == "object",
        Some(Json::Arr(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("object")),
        _ => false,
    }
}

fn member<'a>(members: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    members.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// Set `key` to `value`, in its place when present, else last.
fn set(members: &mut Vec<(String, Json)>, key: &str, value: Json) {
    match members.iter_mut().find(|(k, _)| k == key) {
        Some((_, slot)) => *slot = value,
        None => members.push((key.to_owned(), value)),
    }
}
