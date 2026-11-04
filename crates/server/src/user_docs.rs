//! The user guide's generated reference (ADR 0066, GitHub #347): the pages the
//! code can write about itself — every config field, the built-in profiles,
//! the built-in catalog, the metrics a server exports, the HTTP routes and
//! the command line — rendered as Markdown for `docs/user/reference/`.
//!
//! Each page is read from the table the server itself runs on (the field
//! table, `profile::built_in`, `Catalog::built_in`, `Metrics::render`,
//! `api::openapi`, `config::help_text`), so it cannot describe a field,
//! route or metric the binary does not have. `tests/user_docs_reference.rs`
//! holds the committed pages equal to these renderings and rewrites them
//! under `IGNIS_BLESS_USER_DOCS=1` (`make docs-reference`).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::Value;

use crate::config::cli::default_text;
use crate::config::field::{FieldMeta, FAMILIES};
use crate::config::profile::{built_in, CardClass, AUTO_PROFILE, DEFAULT_PROFILE, LEGACY_16GB_PROFILE, NONE_PROFILE};
use crate::config::schema::GROUPS;
use crate::download::catalog::Catalog;
use crate::metrics::{LoadReservations, LocateFound, Metrics, Primitive, TokenizeRoute};
use ignis_core::flash_next_counters::FlashNextCounters;
use ignis_core::locate::{Compression, Kind as LocateKind, Method as LocateMethod};

/// One generated page: its file name under `docs/user/reference/` and its text.
pub struct Page {
    pub file: &'static str,
    pub text: String,
}

/// Every generated page, in the order the reference lists them.
pub fn pages() -> Vec<Page> {
    vec![
        Page { file: "configuration.md", text: configuration() },
        Page { file: "profiles.md", text: profiles() },
        Page { file: "models.md", text: models() },
        Page { file: "metrics.md", text: metrics() },
        Page { file: "http-routes.md", text: http_routes() },
        Page { file: "cli.md", text: cli() },
    ]
}

const GENERATED: &str = "Generated from the code by `make docs-reference` \
     (`crates/server/src/user_docs.rs`); an edit here is overwritten, and \
     `cargo test` fails while this page and the code disagree.";

fn front_matter(title: &str, description: &str, order: u32) -> String {
    format!(
        "---\ntitle: \"{title}\"\ndescription: \"{description}\"\nsidebar:\n  order: {order}\n  badge:\n    text: generated\n    variant: note\neditUrl: false\n---\n\n:::note\n{GENERATED}\n:::\n\n"
    )
}

/// A value inside a table cell: one line, its pipes escaped.
fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

/// A doc comment as Markdown: rustdoc's leading spaces trimmed, its
/// paragraphs kept.
fn paragraphs(description: &str) -> String {
    description
        .split("\n\n")
        .map(|p| p.lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" "))
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn field_entry(out: &mut String, meta: &FieldMeta) {
    let _ = writeln!(out, "### `{}`\n", meta.file_key());
    let _ = writeln!(out, "| | |\n|---|---|");
    let _ = writeln!(out, "| Flag | `{}`{} |", meta.flag(), if meta.switch { " (may stand bare: `true`)" } else { "" });
    let _ = writeln!(out, "| Env var | `{}` |", meta.env());
    let _ = writeln!(out, "| Kind | `{}`{} |", cell(meta.kind), if meta.repeatable { ", repeatable" } else { "" });
    let _ = writeln!(out, "| Default | `{}` |", cell(&default_text(meta)));
    let rule = meta.validator.describe();
    if !rule.is_empty() {
        let _ = writeln!(out, "| Rule | {} |", cell(&rule));
    }
    let _ = writeln!(out, "| Models | {} |", cell(&meta.applies.describe()));
    if meta.scoped {
        let scoped: Vec<String> = FAMILIES.iter().map(|family| format!("`{}`", meta.scoped_flag(*family))).collect();
        let _ = writeln!(out, "| Per model | {} |", scoped.join(", "));
    }
    let patch = match (meta.patchable, meta.reload_required) {
        (false, _) => "not patchable",
        (true, false) => "patchable live",
        (true, true) => "patchable, with a model reload",
    };
    let get = if meta.visible { "shown by `GET`" } else { "hidden from `GET`" };
    let _ = writeln!(out, "| `/v1/config` | {get}, {patch} |\n");
    let text = paragraphs(meta.description);
    if !text.is_empty() {
        let _ = writeln!(out, "{text}\n");
    }
}

/// Every field, by group, from the table `help --fields` prints.
pub fn configuration() -> String {
    let mut out = front_matter(
        "Every config field",
        "Every field ignis-server takes, by group: its flag, env var, default, rule, the models it applies to and whether it changes live.",
        1,
    );
    out.push_str(
        "Every setting is one field with the same name on every surface: the flag \
         `--<group>-<field>`, the env var `IGNIS_<GROUP>_<FIELD>` and the config-file key \
         `<group>.<field>`. Precedence, highest first: the per-model flag, the flag, the per-model \
         env var, the env var, the per-model file key, the file key, the profile, the default. \
         [Configuration](../configuration/index.md) explains the model; `ignis-server help --fields` \
         prints this same table from the binary you run.\n\n",
    );
    for (group, fields) in GROUPS {
        let _ = writeln!(out, "## {group}\n");
        for meta in *fields {
            field_entry(&mut out, meta);
        }
    }
    finish(out)
}

/// The built-in profiles, every value each one sets.
pub fn profiles() -> String {
    let mut out = front_matter(
        "Built-in profiles",
        "The hardware profiles built into ignis-server and every value each one sets, general and per model.",
        2,
    );
    let _ = writeln!(
        out,
        "A profile is a layer of defaults for one kind of card, between the config file and the \
         built-in defaults: any flag, env var or file value still wins over it. An unset `--profile` \
         is `{DEFAULT_PROFILE}`, and `{AUTO_PROFILE}` takes the built-in profile of the card's class, \
         read from its total memory; `--profile {NONE_PROFILE}` skips the layer. A config file's \
         `profiles:` section defines more, or shadows these \
         ([Profiles and defaults](../concepts/profiles.md)). `{LEGACY_16GB_PROFILE}` is the former \
         name of `{}` and is still accepted everywhere a profile is named, as the same profile.\n",
        CardClass::Gb16.profile()
    );
    let _ = writeln!(
        out,
        "A profile follows the card's memory, not its generation. The RTX 3090 and 3090 Ti are \
         supported, on the 27B's 16 GB entries; the RTX 4090 and the 16 GB RTX 40 cards are \
         expected to work, and reports are welcome; the RTX 50 and RTX PRO 6000 run every model. \
         The NVFP4 27B, Flash-Next and the image model load on Blackwell alone \
         ([What it runs on](../start/index.md#what-it-runs-on)).\n"
    );
    for (name, layer) in built_in() {
        let _ = writeln!(out, "## {name}\n");
        if let Some(class) = CardClass::ALL.into_iter().find(|class| class.profile() == name) {
            let _ = writeln!(out, "`--profile {AUTO_PROFILE}` picks it on a card of the {class} class: a total memory {}.\n", class.totals());
        }
        let _ = writeln!(out, "| Field | Model | Value |\n|---|---|---|");
        let mut rows: Vec<(String, String, String)> = layer
            .entries()
            .map(|(meta, family, candidate)| {
                let model = family.map_or("every model", |f| f.name());
                (meta.file_key(), model.to_owned(), candidate.raw.clone())
            })
            .collect();
        rows.sort();
        for (key, model, value) in rows {
            let _ = writeln!(out, "| [`{key}`](configuration.md#{}) | {model} | `{}` |", anchor(&key), cell(&value));
        }
        out.push('\n');
    }
    finish(out)
}

/// The anchor GitHub and the site give a field's heading (`` `server.bind` ``
/// is `serverbind`).
fn anchor(file_key: &str) -> String {
    file_key.chars().filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-').collect::<String>().to_lowercase()
}

/// The built-in catalog: every entry, every file it pins.
pub fn models() -> String {
    let mut out = front_matter(
        "Built-in models",
        "The catalog compiled into ignis-server: every model it can fetch, the repository and revision it is pinned at, and every file's size and SHA-256.",
        3,
    );
    out.push_str(
        "The models the server knows how to fetch without an operator catalog. Each is pinned \
         file by file, byte count and SHA-256, at a fixed revision of its Hugging Face repository, \
         and nothing outside the binary can change a pin. Each entry states the license its weights are \
         published under, linked to its text ([Get a model](../guides/models.md), \
         [an operator catalog](../guides/operator-catalog.md)).\n\n",
    );
    let catalog = Catalog::built_in();
    let _ = writeln!(out, "| Id | Repository | Size | License | Artifact |\n|---|---|---|---|---|");
    for entry in catalog.entries() {
        let _ = writeln!(
            out,
            "| [`{id}`](#{anchor}) | [`{repo}`](https://huggingface.co/{repo}) | {gib:.1} GiB | {license} | `{artifact}` |",
            id = entry.id,
            anchor = anchor(&entry.id),
            repo = entry.repo,
            gib = entry.gib(),
            license = entry.license.as_ref().map_or_else(|| "—".to_owned(), |license| format!("[`{}`]({})", license.id, license.url)),
            artifact = entry.artifact
        );
    }
    out.push('\n');
    for entry in catalog.entries() {
        let _ = writeln!(out, "## {}\n", entry.id);
        let _ = writeln!(
            out,
            "[`{repo}`](https://huggingface.co/{repo}) at revision `{rev}`, {bytes} bytes in {n} files.\n",
            repo = entry.repo,
            rev = entry.revision,
            bytes = thousands(entry.total_bytes()),
            n = entry.files.len()
        );
        let _ = writeln!(out, "| File | Bytes | SHA-256 |\n|---|---|---|");
        for file in entry.files_by_size() {
            let _ = writeln!(out, "| `{}` | {} | `{}` |", file.name, thousands(file.bytes), file.sha256);
        }
        out.push('\n');
    }
    finish(out)
}

fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// One family of metrics a server exports only once something happened, and
/// the step that makes a fresh [`Metrics`] export it.
struct Activation {
    when: &'static str,
    activate: fn(&Metrics),
}

/// Every family that is absent from a fresh server's exposition, with what
/// brings it in. A family added behind a new condition shows up here as a
/// metric with no row until it gets one: `every_metric_the_code_names_is_listed`
/// fails.
const ACTIVATIONS: &[Activation] = &[
    Activation { when: "after the first decision", activate: |m| m.record_decision(Primitive::Choice, Some(0.5)) },
    Activation {
        when: "after the first `locate`",
        activate: |m| m.record_locate(LocateKind::Prose, LocateMethod::Shortlist, Compression::None, LocateFound::False),
    },
    Activation { when: "once the thinking budget has closed a reasoning block", activate: |m| m.record_thinking_forced_close() },
    Activation { when: "once the loop guard has watched a request", activate: |m| m.record_loop_guard_watched() },
    Activation {
        when: "after the first request that ran speculative verify rounds",
        activate: |m| m.record_speculation(&ignis_core::types::SpecCounters::round(1, 1)),
    },
    Activation { when: "after the first call to a counting route", activate: |m| m.record_tokenize(TokenizeRoute::Tokenize) },
    Activation { when: "on a server that has seen an image job", activate: |m| m.set_image_queue(0, 0) },
    Activation { when: "on a server that has been asked for a transcription", activate: |m| m.set_speech_gauges(0, 0) },
    Activation {
        when: "on a load with the KV-disk tier",
        activate: |m| m.set_load_reservations(LoadReservations { kv_disk_bytes: Some(1), ..LoadReservations::none() }),
    },
    Activation { when: "on a Flash-Next load", activate: |m| m.set_flash_next(Some(&FlashNextCounters::default())) },
];

/// `name -> (type, help)` for every family an exposition declares.
fn declared(text: &str) -> BTreeMap<String, (String, String)> {
    let mut help: BTreeMap<String, String> = BTreeMap::new();
    let mut declared = BTreeMap::new();
    for line in text.lines() {
        if let Some((name, text)) = line.strip_prefix("# HELP ").and_then(|rest| rest.split_once(' ')) {
            help.insert(name.to_owned(), text.to_owned());
        } else if let Some((name, kind)) = line.strip_prefix("# TYPE ").and_then(|rest| rest.split_once(' ')) {
            declared.insert(name.to_owned(), (kind.to_owned(), help.remove(name).unwrap_or_default()));
        }
    }
    declared
}

/// Every metric family the listener can declare, from its own exposition:
/// a fresh server's, then each conditional family's with its condition.
pub fn metrics() -> String {
    let mut out = front_matter(
        "Metrics",
        "Every Prometheus metric ignis-server exports: its name, type, meaning, and when it appears.",
        4,
    );
    out.push_str(
        "The metrics listener (on by default, keyless, on loopback) serves these in the Prometheus \
         text format, and the Playground's Monitor reads the same set at `/ui/metrics` \
         ([Telemetry and metrics](../guides/telemetry.md)). A family marked *always* is in every \
         scrape from the start, zeros included; the others appear once what the last column names \
         has happened, and then stay.\n\n",
    );
    let mut rows: BTreeMap<String, (String, String, &'static str)> = BTreeMap::new();
    for (name, (kind, help)) in declared(&Metrics::new().render()) {
        rows.insert(name, (kind, help, "always"));
    }
    for activation in ACTIVATIONS {
        let metrics = Metrics::new();
        (activation.activate)(&metrics);
        for (name, (kind, help)) in declared(&metrics.render()) {
            rows.entry(name).or_insert((kind, help, activation.when));
        }
    }
    let _ = writeln!(out, "| Metric | Type | Meaning | Exported |\n|---|---|---|---|");
    for (name, (kind, help, when)) in &rows {
        let _ = writeln!(out, "| `{name}` | {kind} | {} | {when} |", cell(help));
    }
    finish(out)
}

/// Every HTTP route the OpenAPI document describes.
pub fn http_routes() -> String {
    let mut out = front_matter(
        "HTTP routes",
        "Every route of the HTTP surface, from the OpenAPI document the server builds from its handlers.",
        5,
    );
    out.push_str(
        "The routes as the server's own OpenAPI document lists them: a route cannot exist without \
         its entry ([ADR 0036](../../adr/0036-the-v1-surface-documents-itself.md)). A running server \
         serves the full document, schemas included, at `/v1/openapi.json` and a browsable page at \
         `/v1/docs/`, both without a key. [The API](../api/index.md) explains each route.\n\n",
    );
    let document = serde_json::to_value(crate::api::openapi()).expect("the OpenAPI document serializes");
    let mut rows: Vec<(String, String, String, String)> = Vec::new();
    if let Some(paths) = document["paths"].as_object() {
        for (path, operations) in paths {
            let Some(operations) = operations.as_object() else { continue };
            for (method, operation) in operations {
                let tag = operation["tags"][0].as_str().unwrap_or("").to_owned();
                let summary = summary_of(operation);
                rows.push((tag, path.clone(), method.to_uppercase(), summary));
            }
        }
    }
    rows.sort();
    let mut current: Option<String> = None;
    for (tag, path, method, summary) in rows {
        if current.as_deref() != Some(tag.as_str()) {
            if current.is_some() {
                out.push('\n');
            }
            let heading = if tag.is_empty() { "Other" } else { tag.as_str() };
            let _ = writeln!(out, "## {heading}\n\n| Method | Path | What it does |\n|---|---|---|");
            current = Some(tag.clone());
        }
        let _ = writeln!(out, "| {method} | `{path}` | {} |", cell(&summary));
    }
    finish(out)
}

fn summary_of(operation: &Value) -> String {
    let summary = operation["summary"].as_str().unwrap_or("").trim();
    if !summary.is_empty() {
        return summary.to_owned();
    }
    operation["description"].as_str().unwrap_or("").split("\n\n").next().unwrap_or("").trim().to_owned()
}

/// The command line: the usage block of `ignis-server --help`.
pub fn cli() -> String {
    let mut out = front_matter(
        "Command line",
        "ignis-server's commands and subcommands, as its --help prints them.",
        6,
    );
    out.push_str(
        "The commands as `ignis-server --help` prints them. Every field's flag is in \
         [every config field](configuration.md); `ignis-server help --fields --format json` gives \
         the same table to a tool.\n\n",
    );
    let help = crate::config::help_text();
    let first_group = format!("\n{}:\n", GROUPS[0].0.to_ascii_uppercase());
    let usage = help.split(&first_group).next().unwrap_or(&help).trim_end();
    let _ = writeln!(out, "```text\n{usage}\n```");
    finish(out)
}

/// One trailing newline, whatever the last section left.
fn finish(mut out: String) -> String {
    while out.ends_with("\n\n") {
        out.pop();
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_anchor_is_its_headings_slug() {
        assert_eq!(anchor("server.bind"), "serverbind");
        assert_eq!(anchor("reuse.kv_host_pool_bytes"), "reusekv_host_pool_bytes");
        assert_eq!(anchor("qwen3.8-27b"), "qwen38-27b");
    }

    #[test]
    fn thousands_groups_by_three() {
        assert_eq!(thousands(7), "7");
        assert_eq!(thousands(6981), "6,981");
        assert_eq!(thousands(19406942468), "19,406,942,468");
    }

    /// ADR 0078: the built-in models page names each entry's license, linked.
    #[test]
    fn the_built_in_models_page_links_each_entrys_license() {
        let page = models();
        assert!(page.contains("| License |"), "{page}");
        assert!(page.contains("[`qwen-community-1.0`](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis/blob/8d94db83cd8e58d6be31e796723e5e35b960638c/LICENSE)"));
        assert!(page.contains("[`openmdw-1.1`]"));
    }

    /// GitHub #480: the profiles page says, under each built-in profile, the
    /// card class `auto` picks it for and the totals that class covers.
    #[test]
    fn the_profiles_page_names_the_class_auto_picks_each_profile_for() {
        let page = profiles();
        assert!(page.contains("RTX 3090 and 3090 Ti are supported, on the 27B's 16 GB entries"), "{page}");
        assert!(page.contains("load on Blackwell alone"), "{page}");
        assert!(page.contains("## rtx-24gb\n\n`--profile auto` picks it on a card of the 24 GB class: a total memory from 20 to under 28 GiB.\n"), "{page}");
        assert!(page.contains("## rtx-16gb\n\n`--profile auto` picks it on a card of the 16 GB class: a total memory below 20 GiB.\n"), "{page}");
        assert!(page.contains("## rtx5090\n\n`--profile auto` picks it on a card of the 32 GB class: a total memory from 28 to 40 GiB.\n"), "{page}");
        assert!(page.contains("## rtxpro6000\n\n`--profile auto` picks it on a card of the 96 GB class: a total memory from 88 to 104 GiB.\n"), "{page}");
    }

    #[test]
    fn a_table_cell_stays_one_cell() {
        assert_eq!(cell("a | b\nc"), "a \\| b c");
    }

    /// Every `ignis_*` family name the metrics module spells out is a row of
    /// the page, so a family behind a new condition cannot go unlisted.
    #[test]
    fn every_metric_the_code_names_is_listed() {
        let page = metrics();
        let source = include_str!("metrics.rs");
        let mut missing = Vec::new();
        for (at, _) in source.match_indices("\"ignis_") {
            let name: String = source[at + 1..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
            // A prefix (`"ignis_kv_disk_"` in a `contains` check), and the
            // series a histogram family derives, are not families.
            if name.ends_with('_') || ["_bucket", "_count", "_sum"].iter().any(|s| name.ends_with(s)) {
                continue;
            }
            if !page.contains(&format!("| `{name}` |")) && !missing.contains(&name) {
                missing.push(name);
            }
        }
        assert!(missing.is_empty(), "metrics the page does not list: {missing:?}");
    }

    #[test]
    fn every_field_has_its_heading_and_every_page_its_front_matter() {
        let page = configuration();
        for meta in crate::config::schema::all_fields() {
            assert!(page.contains(&format!("### `{}`\n", meta.file_key())), "{}", meta.file_key());
        }
        for page in pages() {
            assert!(page.text.starts_with("---\ntitle: \""), "{}", page.file);
            assert!(page.text.ends_with('\n') && !page.text.ends_with("\n\n"), "{}", page.file);
        }
    }
}
