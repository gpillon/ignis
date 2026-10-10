//! The command line's verbs (spec config-v2/01 §The CLI surface,
//! config-v2/02 §The CLI, model-download/02): `help`, `help --fields`,
//! `config generate` / `print` / `patch`, and `model download` / `list`. A
//! bare invocation (`ignis-server --model-artifact …`) still serves; a verb is
//! only ever the first word.
//!
//! Every verb takes the same field flags a start does, resolved by the same
//! code ([`super::gather`]): only what is done with the result differs —
//! written fresh, printed, or merged into an existing file. Nothing here
//! writes: a verb that writes returns [`ConfigOutcome::Write`] and `main`
//! performs it, so resolution stays testable without a disk.

use std::path::PathBuf;

use serde_json::Value;

use super::field::{FieldMeta, FAMILIES};
use super::file::{self, Files, Format};
use super::schema::{all_fields, GROUPS};
use super::source;
use super::{gather, Config, ConfigError, ConfigOutcome, FileChoice};
use crate::download::{self, ListFormat, ModelCommand};

/// `help` and `help --fields [--format text|json]`.
pub(super) fn help(args: &[String]) -> Result<ConfigOutcome, ConfigError> {
    let mut args = args.to_vec();
    let fields = take_switch(&mut args, "--fields");
    let format = take_value(&mut args, "--format")?;
    if let Some(extra) = args.first() {
        return Err(ConfigError(format!("`help` takes `--fields` and `--format`, not `{extra}`")));
    }
    match (fields, format.as_deref()) {
        (false, None) => Ok(ConfigOutcome::Help(super::help_text())),
        (true, None | Some("text")) => Ok(ConfigOutcome::Print(fields_text())),
        (true, Some("json")) => Ok(ConfigOutcome::Print(fields_json())),
        (true, Some(other)) => Err(ConfigError(format!("`help --fields --format {other}`: the formats are text and json"))),
        (false, Some(_)) => Err(ConfigError("`--format` goes with `help --fields`".to_owned())),
    }
}

/// One field's default as help prints it: `unset` for none, a string bare.
pub(crate) fn default_text(meta: &FieldMeta) -> String {
    match (meta.default)() {
        Value::Null => "unset".to_owned(),
        Value::String(s) => s,
        other => other.to_string(),
    }
}

/// `help --fields`: every field, every attribute, from the field table
/// itself — the running binary as its own reference (spec config-v2/01 AC
/// 10).
pub fn fields_text() -> String {
    let mut text = String::from(
        "Every field ignis-server takes. Precedence, highest first:\n  \
         <family>-flag > flag > <family>-env > env > <family>-file > file > profile > default\n\
         Families: qwen38 (Qwen3.8-27B), qwen38flashnext (Qwen3.8-Flash-Next).\n",
    );
    for (group, fields) in GROUPS {
        text.push_str(&format!("\n[{group}]\n"));
        for meta in *fields {
            text.push_str(&format!("\n{}\n", meta.file_key()));
            text.push_str(&format!("    flag:      {}\n", meta.flag()));
            text.push_str(&format!("    env:       {}\n", meta.env()));
            text.push_str(&format!("    kind:      {}{}\n", meta.kind, if meta.switch { " (the flag may stand bare: true)" } else { "" }));
            text.push_str(&format!("    default:   {}\n", default_text(meta)));
            let rule = meta.validator.describe();
            if !rule.is_empty() {
                text.push_str(&format!("    rule:      {rule}\n"));
            }
            text.push_str(&format!("    families:  {}\n", meta.applies.describe()));
            if meta.scoped {
                let scoped: Vec<String> = FAMILIES.iter().map(|family| meta.scoped_flag(*family)).collect();
                text.push_str(&format!("    per family: {}\n", scoped.join(", ")));
            }
            let mut http = vec![if meta.visible { "shown by GET" } else { "hidden from GET" }];
            http.push(match (meta.patchable, meta.reload_required) {
                (false, _) => "not patchable",
                (true, false) => "patchable live",
                (true, true) => "patchable with a model reload",
            });
            text.push_str(&format!("    /v1/config: {}\n", http.join(", ")));
            for line in meta.description.lines() {
                text.push_str(&format!("    {}\n", line.trim()));
            }
        }
    }
    text
}

/// `help --fields --format json`: the same table, one object per field, for
/// a tool to read.
pub fn fields_json() -> String {
    let fields: Vec<Value> = all_fields()
        .map(|meta| {
            serde_json::json!({
                "group": meta.group,
                "name": meta.name,
                "flag": meta.flag(),
                "env": meta.env(),
                "file_key": meta.file_key(),
                "kind": meta.kind,
                "default": (meta.default)(),
                "rule": meta.validator.describe(),
                "families": meta.applies.describe(),
                "scoped_flags": if meta.scoped { FAMILIES.iter().map(|f| meta.scoped_flag(*f)).collect() } else { Vec::new() },
                "visible": meta.visible,
                "patchable": meta.patchable,
                "reload_required": meta.reload_required,
                "description": meta.description.lines().map(str::trim).collect::<Vec<_>>().join(" ").trim().to_owned(),
            })
        })
        .collect();
    let mut text = serde_json::to_string_pretty(&fields).expect("plain JSON");
    text.push('\n');
    text
}

/// `config generate|print|patch …`.
pub(super) fn config(args: &[String], env: &dyn Fn(&str) -> Option<String>, files: &dyn Files) -> Result<ConfigOutcome, ConfigError> {
    let Some((verb, rest)) = args.split_first() else {
        return Err(ConfigError("`config` takes a verb: generate, print or patch".to_owned()));
    };
    let mut rest = rest.to_vec();
    match verb.as_str() {
        "generate" => generate(&mut rest, env, files),
        "print" => print(&mut rest, env, files),
        "patch" => patch(&mut rest, env, files),
        other => Err(ConfigError(format!("`config {other}`: the verbs are generate, print and patch"))),
    }
}

/// `config generate [--format json|yaml] [--out <path>] [--force]
/// [--dry-run] [field flags]`: a fresh file from the flags, the environment,
/// the profile and the defaults — no existing file is read, ever, since this
/// is what makes one. Refuses to overwrite a file without `--force`;
/// `--dry-run` checks and prints, and writes nothing.
fn generate(args: &mut Vec<String>, env: &dyn Fn(&str) -> Option<String>, files: &dyn Files) -> Result<ConfigOutcome, ConfigError> {
    let format = take_value(args, "--format")?;
    let out = take_value(args, "--out")?.map(PathBuf::from);
    let force = take_switch(args, "--force");
    let dry_run = take_switch(args, "--dry-run");
    let parsed = source::parse_args(args)?;
    if parsed.config.is_some() {
        return Err(ConfigError("`config generate` reads no config file: it writes one (`--out`)".to_owned()));
    }
    let profile = parsed.profile.clone();
    let (sources, _) = gather(parsed, env, files, FileChoice::None)?;
    let config = Config::from_sources(sources)?;
    let format = match (format, &out) {
        (Some(name), _) => Format::parse(&name).ok_or_else(|| ConfigError(format!("`--format {name}`: the formats are json and yaml")))?,
        (None, Some(path)) => Format::of_path(path).ok_or_else(|| {
            ConfigError(format!("`--out {}`: its extension names no format; add `--format json|yaml`", path.display()))
        })?,
        (None, None) => return Err(ConfigError("`config generate` to stdout needs `--format json|yaml`".to_owned())),
    };
    let mut document = file::effective_document(&config);
    if let Some(name) = profile {
        document.as_object_mut().expect("a document is a map").insert("profile".to_owned(), Value::String(name));
    }
    let contents = format.write(&document);
    match out {
        Some(path) => {
            if files.exists(&path) && !force {
                return Err(ConfigError(format!("`{}` exists: `--force` overwrites it", path.display())));
            }
            Ok(if dry_run { ConfigOutcome::Print(contents) } else { ConfigOutcome::Write { path, contents } })
        }
        None => Ok(ConfigOutcome::Print(contents)),
    }
}

/// `config print [--file <path>] [--format json|yaml] [field flags]`: what
/// would actually run — the file if there is one (`--file`, else the first
/// place a start looks), the environment, and the flags given here. Never
/// writes. A field `GET /v1/config` hides is shown as set or not, never its
/// value: a terminal's scrollback is no place for a key.
fn print(args: &mut Vec<String>, env: &dyn Fn(&str) -> Option<String>, files: &dyn Files) -> Result<ConfigOutcome, ConfigError> {
    let path = take_value(args, "--file")?.map(PathBuf::from).unwrap_or_else(default_path);
    let format = match take_value(args, "--format")? {
        Some(name) => Format::parse(&name).ok_or_else(|| ConfigError(format!("`--format {name}`: the formats are json and yaml")))?,
        None => Format::Yaml,
    };
    let parsed = source::parse_args(args)?;
    if parsed.config.is_some() {
        return Err(ConfigError("`config print` names its file with `--file`".to_owned()));
    }
    let (sources, _) = gather(parsed, env, files, FileChoice::IfPresent(path))?;
    let config = Config::from_sources(sources)?;
    let mut document = file::effective_document(&config);
    hide_invisible(&mut document);
    Ok(ConfigOutcome::Print(format.write(&document)))
}

/// `config patch [--file <path>] [--out <path>] [field flags]`: the file
/// (which must exist — `config generate` makes it), with the environment
/// and the flags given here written into it, validated as a start would be,
/// and written back (or to `--out`). With no field flag it still re-resolves
/// and writes: the environment's values land in the file, and a stale
/// spelling in it is rewritten in canonical form. Every key it does not
/// change — other fields, family sections, `profiles:` — is kept.
fn patch(args: &mut Vec<String>, env: &dyn Fn(&str) -> Option<String>, files: &dyn Files) -> Result<ConfigOutcome, ConfigError> {
    let path = take_value(args, "--file")?.map(PathBuf::from).unwrap_or_else(default_path);
    let out = take_value(args, "--out")?.map(PathBuf::from);
    let parsed = source::parse_args(args)?;
    if parsed.config.is_some() {
        return Err(ConfigError("`config patch` names its file with `--file`".to_owned()));
    }
    let profile = parsed.profile.clone();
    let (sources, _) = gather(parsed, env, files, FileChoice::Required(path.clone()))?;
    Config::from_sources(sources.clone())?;
    let mut changes = file::changes_of(&sources.file)?;
    changes.extend(file::changes_of(&sources.env)?);
    changes.extend(file::changes_of(&sources.flags)?);
    let contents = file::rewrite(files, &path, changes, profile.as_deref())?;
    let target = out.unwrap_or_else(|| path.clone());
    // An `--out` of the other format gets the same document in its own.
    let contents = match (Format::of_path(&path).unwrap_or(Format::Yaml), Format::of_path(&target)) {
        (from, Some(to)) if from != to => to.write(&from.read(&contents).expect("a rewritten file reads back")),
        _ => contents,
    };
    Ok(ConfigOutcome::Write { contents, path: target })
}

/// `model download …` and `model list …` (spec model-download/02): the
/// configuration a start would read and the catalog it would load, turned
/// into what to do, for `main` to run. `model verify` and `model convert`
/// are reserved, so any other word is an unknown command.
pub(super) fn model(args: &[String], env: &dyn Fn(&str) -> Option<String>, files: &dyn Files) -> Result<ConfigOutcome, ConfigError> {
    let Some((verb, rest)) = args.split_first() else {
        return Err(ConfigError("`model` takes a command: download or list".to_owned()));
    };
    let mut rest = rest.to_vec();
    match verb.as_str() {
        "download" => model_download(&mut rest, env, files),
        "list" => model_list(&mut rest, env, files),
        other => Err(ConfigError(format!("unknown command `model {other}` (the model commands are download and list)"))),
    }
}

/// `model download [<id>…] [--all] [--out <dir>] [field flags]`: each named
/// entry, every entry with `--all`, or the configured `model.id` with
/// neither, into `--out` or `download.path`, from `download.endpoint` with
/// the token [`download::bearer_token`] picks. The ids may stand anywhere
/// among the flags. `download.enabled` is not read: it gates only the start's own
/// fetch, and this command is the operator's explicit yes.
fn model_download(args: &mut Vec<String>, env: &dyn Fn(&str) -> Option<String>, files: &dyn Files) -> Result<ConfigOutcome, ConfigError> {
    let all = take_switch(args, "--all");
    let out = take_value(args, "--out")?.map(PathBuf::from);
    let (named, flags) = super::source::split_positionals(args);
    *args = flags;
    let mut ids: Vec<String> = Vec::new();
    for id in named {
        if !ids.iter().any(|named| named.eq_ignore_ascii_case(&id)) {
            ids.push(id);
        }
    }
    let config = super::start_config(args, env, files)?;
    let catalog = download::catalog::load(files, config.download_catalog.as_deref()).map_err(|err| ConfigError(err.0))?;
    let lookup = |id: &str| {
        catalog.entry(id).cloned().ok_or_else(|| {
            ConfigError(format!("`{id}` is not in the catalog (the ids are {})", catalog.ids().join(", ")))
        })
    };
    let entries = match (all, ids.as_slice()) {
        (true, []) => catalog.entries().to_vec(),
        (true, _) => return Err(ConfigError("`model download --all` fetches every entry: name ids or `--all`, not both".to_owned())),
        (false, []) => vec![lookup(&config.model)?],
        (false, ids) => ids.iter().map(|id| lookup(id)).collect::<Result<_, _>>()?,
    };
    let token = download::bearer_token(&config.download_endpoint, config.download_token.as_ref(), env("HF_TOKEN").as_deref());
    Ok(ConfigOutcome::Model(ModelCommand::Download {
        entries,
        dir: out.unwrap_or_else(|| config.model_download_path.clone()),
        endpoint: config.download_endpoint.clone(),
        token,
    }))
}

/// `model list [--format text|json] [field flags]`: every entry of the
/// catalog a start would load, with its state under `download.path`.
fn model_list(args: &mut Vec<String>, env: &dyn Fn(&str) -> Option<String>, files: &dyn Files) -> Result<ConfigOutcome, ConfigError> {
    let format = match take_value(args, "--format")?.as_deref() {
        None | Some("text") => ListFormat::Text,
        Some("json") => ListFormat::Json,
        Some(other) => return Err(ConfigError(format!("`model list --format {other}`: the formats are text and json"))),
    };
    let config = super::start_config(args, env, files)?;
    let catalog = download::catalog::load(files, config.download_catalog.as_deref()).map_err(|err| ConfigError(err.0))?;
    Ok(ConfigOutcome::Model(ModelCommand::List { catalog, dir: config.model_download_path.clone(), format }))
}

/// Where `config print` and `config patch` look with no `--file`: the first
/// place a start looks.
fn default_path() -> PathBuf {
    file::discovery_candidates(&|_: &str| None).remove(0)
}

/// Replace every set value of a field `GET /v1/config` hides with a marker.
fn hide_invisible(document: &mut Value) {
    for meta in all_fields().filter(|meta| !meta.visible) {
        if let Some(value) = document.get_mut(meta.group).and_then(|group| group.get_mut(meta.name)) {
            if !value.is_null() {
                *value = Value::String("(set, not shown)".to_owned());
            }
        }
    }
}

/// Remove `--name <value>` from `args`, returning the value.
fn take_value(args: &mut Vec<String>, name: &str) -> Result<Option<String>, ConfigError> {
    let Some(at) = args.iter().position(|arg| arg == name) else {
        return Ok(None);
    };
    if at + 1 >= args.len() {
        return Err(ConfigError(format!("`{name}` requires a value")));
    }
    let value = args.remove(at + 1);
    args.remove(at);
    Ok(Some(value))
}

/// Remove a bare `--name` from `args`, saying whether it was there.
fn take_switch(args: &mut Vec<String>, name: &str) -> bool {
    match args.iter().position(|arg| arg == name) {
        Some(at) => {
            args.remove(at);
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::field::scope;
    use crate::config::file::testing::MemFiles;
    use crate::config::resolve_with;
    use ignis_core::compute::ModelFamily;

    fn run(argv: &[&str], env: &'static [(&'static str, &'static str)], files: &MemFiles) -> Result<ConfigOutcome, ConfigError> {
        let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
        resolve_with(&argv, move |key: &str| env.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string()), files)
    }

    fn printed(outcome: ConfigOutcome) -> String {
        match outcome {
            ConfigOutcome::Print(text) => text,
            other => panic!("expected Print, got {other:?}"),
        }
    }

    fn written(outcome: ConfigOutcome) -> (PathBuf, String) {
        match outcome {
            ConfigOutcome::Write { path, contents } => (path, contents),
            other => panic!("expected Write, got {other:?}"),
        }
    }

    fn started(argv: &[&str], files: &MemFiles) -> Config {
        match run(argv, &[], files).expect("starts") {
            ConfigOutcome::Config(config) => config,
            other => panic!("expected a start, got {other:?}"),
        }
    }

    fn yaml(text: &str) -> Value {
        Format::Yaml.read(text).expect("YAML")
    }

    /// Spec config-v2/01, Testing: every field in the table appears in `help
    /// --fields`, on every surface — the CLI side of the exhaustiveness test.
    #[test]
    fn help_fields_names_every_field_on_every_surface() {
        let text = printed(run(&["help", "--fields"], &[], &MemFiles::default()).unwrap());
        for meta in all_fields() {
            for name in [meta.file_key(), meta.flag(), meta.env()] {
                assert!(text.contains(&name), "{name} missing from help --fields");
            }
            if meta.scoped {
                assert!(text.contains(&meta.scoped_flag(ModelFamily::FlashNext)), "{}", meta.file_key());
            }
        }
        let json: Vec<Value> =
            serde_json::from_str(&printed(run(&["help", "--fields", "--format", "json"], &[], &MemFiles::default()).unwrap())).unwrap();
        assert_eq!(json.len(), all_fields().count());
        let api_key = json.iter().find(|f| f["file_key"] == "server.api_key").unwrap();
        assert_eq!((api_key["visible"].as_bool(), api_key["patchable"].as_bool()), (Some(false), Some(false)));
        assert!(run(&["help", "--nonsense"], &[], &MemFiles::default()).is_err());
    }

    #[test]
    fn the_bare_invocation_serves_and_only_known_words_are_verbs() {
        let files = MemFiles::default();
        assert!(matches!(run(&[], &[], &files).unwrap(), ConfigOutcome::Config(_)));
        assert!(matches!(run(&["--server-bind", "127.0.0.1:1"], &[], &files).unwrap(), ConfigOutcome::Config(_)));
        assert!(matches!(run(&["help"], &[], &files).unwrap(), ConfigOutcome::Help(_)));
        assert!(matches!(run(&["version"], &[], &files).unwrap(), ConfigOutcome::Version(_)));
        let err = run(&["serve"], &[], &files).unwrap_err().0;
        assert!(err.contains("unknown command `serve`"), "{err}");
        assert!(run(&["config"], &[], &files).unwrap_err().0.contains("generate, print or patch"));
        assert!(run(&["config", "save"], &[], &files).is_err(), "no separate save verb: patch is it");
    }

    /// Spec config-v2/01, Testing: `config generate` with no flags, read back
    /// by a start, in both formats — and either model starts on it.
    #[test]
    fn generate_writes_every_default_and_a_start_reads_it_back() {
        let defaults = started(&[], &MemFiles::default());
        for name in ["new.yaml", "new.json"] {
            let (path, contents) = written(run(&["config", "generate", "--out", name], &[], &MemFiles::default()).unwrap());
            assert_eq!(path, PathBuf::from(name));
            let files = MemFiles::with(&[(name, &contents)]);
            let read = started(&["--config", name], &files);
            assert_eq!(read.settings(), defaults.settings(), "{name}");
            for family in FAMILIES {
                read.for_family(family).unwrap_or_else(|e| panic!("{name}: {e}"));
            }
            let document = Format::of_path(&path).unwrap().read(&contents).unwrap();
            for meta in all_fields() {
                // A field only one family takes is never written at the
                // general scope (GitHub #311's family-section fix, now also
                // `effective_document`'s): it lives in that family's own
                // section instead, the one place a start of the other
                // family never reads.
                let written = match meta.home_family() {
                    Some(family) => document[meta.group].get(scope(family)).and_then(|s| s.get(meta.name)),
                    None => document[meta.group].get(meta.name),
                };
                assert!(written.is_some(), "{name} writes {}", meta.file_key());
            }
        }
    }

    #[test]
    fn generate_takes_the_usual_flags_and_writes_a_scoped_value_in_its_family_section() {
        let text = printed(
            run(
                &[
                    "config",
                    "generate",
                    "--format",
                    "yaml",
                    "--reuse-kv-host-pool-bytes",
                    "8G",
                    "--qwen38flashnext-reuse-kv-host-pool-bytes",
                    "1G",
                ],
                &[("IGNIS_SERVER_REQUEST_TIMEOUT", "45")],
                &MemFiles::default(),
            )
            .unwrap(),
        );
        let document = yaml(&text);
        assert_eq!(document["reuse"]["kv_host_pool_bytes"], "8G");
        assert_eq!(document["reuse"]["qwen38flashnext"]["kv_host_pool_bytes"], "1G");
        assert_eq!(document["server"]["request_timeout"], 45, "the environment is resolved as a start would");
        assert!(text.find("server:").unwrap() < text.find("model:").unwrap(), "groups in declaration order:\n{text}");
        let files = MemFiles::with(&[("g.yaml", &text)]);
        let read = started(&["--config", "g.yaml"], &files);
        assert_eq!(read.for_family(ModelFamily::FlashNext).unwrap().host_pool_bytes, 1 << 30);
        assert_eq!(read.for_family(ModelFamily::Qwen38_27b).unwrap().host_pool_bytes, 8 << 30);
    }

    #[test]
    fn generate_refuses_an_existing_file_without_force_and_dry_run_writes_nothing() {
        let files = MemFiles::with(&[("c.yaml", "server: {}\n")]);
        let err = run(&["config", "generate", "--out", "c.yaml"], &[], &files).unwrap_err().0;
        assert!(err.contains("c.yaml") && err.contains("--force"), "{err}");
        let (path, _) = written(run(&["config", "generate", "--out", "c.yaml", "--force"], &[], &files).unwrap());
        assert_eq!(path, PathBuf::from("c.yaml"));
        let dry = printed(run(&["config", "generate", "--out", "d.json", "--dry-run"], &[], &files).unwrap());
        assert!(serde_json::from_str::<Value>(&dry).is_ok(), "the format still comes from --out");
        assert!(files.written.lock().unwrap().is_empty(), "resolution never writes; main does");
    }

    #[test]
    fn generate_needs_a_format_for_stdout_and_format_wins_over_the_extension() {
        let files = MemFiles::default();
        assert!(run(&["config", "generate"], &[], &files).unwrap_err().0.contains("--format"));
        let (_, contents) = written(run(&["config", "generate", "--out", "x.yaml", "--format", "json"], &[], &files).unwrap());
        assert!(serde_json::from_str::<Value>(&contents).is_ok(), "{contents}");
        assert!(run(&["config", "generate", "--out", "x.toml"], &[], &files).unwrap_err().0.contains("--format"));
    }

    #[test]
    fn generate_validates_as_a_start_would_and_reads_no_file_ever() {
        let files = MemFiles::with(&[("bad.yaml", "nonsense: true\n")]);
        let err = run(&["config", "generate", "--format", "yaml", "--model-prefill-chunk", "1000"], &[], &files).unwrap_err().0;
        assert!(err.contains("128") && err.contains("1000"), "{err}");
        assert!(run(&["config", "generate", "--format", "yaml"], &[("IGNIS_CONFIG", "bad.yaml")], &files).is_ok());
        assert!(run(&["config", "generate", "--format", "yaml", "--config", "bad.yaml"], &[], &files).is_err());
    }

    /// Spec config-v2/02, Testing: `config print` never touches the disk.
    #[test]
    fn print_shows_the_file_the_environment_and_the_flags_and_writes_nothing() {
        let files = MemFiles::with(&[("ignis.config.yaml", "server:\n  request_timeout: 77\n")]);
        let text = printed(
            run(
                &["config", "print", "--reuse-prompt", "off", "--server-api-key", "sk-secret"],
                &[("IGNIS_MODEL_MAX_CONTEXT", "1024")],
                &files,
            )
            .unwrap(),
        );
        let document = yaml(&text);
        assert_eq!(document["server"]["request_timeout"], 77, "the default path's file");
        assert_eq!(document["model"]["max_context"], 1024);
        assert_eq!(document["reuse"]["prompt"], false);
        assert!(!text.contains("sk-secret") && text.contains("not shown"), "{text}");
        assert!(files.written.lock().unwrap().is_empty());
        let none = printed(run(&["config", "print", "--file", "absent.yaml"], &[], &files).unwrap());
        assert_eq!(yaml(&none)["server"]["request_timeout"], 30, "no file is not an error");
    }

    /// Spec config-v2/02, Testing: `config patch` refuses a file that is not
    /// there, naming `config generate`.
    #[test]
    fn patch_refuses_without_a_file_and_names_generate() {
        let err = run(&["config", "patch", "--file", "c.yaml"], &[], &MemFiles::default()).unwrap_err().0;
        assert!(err.contains("c.yaml") && err.contains("config generate"), "{err}");
    }

    #[test]
    fn patch_writes_the_flags_and_the_environment_and_keeps_everything_else() {
        let original = "profile: rtx5090\nserver:\n  request_timeout: 77\nreuse:\n  qwen38:\n    retained_host: 4\nprofiles:\n  mine:\n    vram:\n      headroom_bytes: 2G\n";
        let files = MemFiles::with(&[("c.yaml", original)]);
        let (path, contents) = written(
            run(&["config", "patch", "--file", "c.yaml", "--reuse-kv-host-pool-bytes", "8G"], &[("IGNIS_MODEL_MAX_CONTEXT", "65536")], &files)
                .unwrap(),
        );
        assert_eq!(path, PathBuf::from("c.yaml"));
        let document = yaml(&contents);
        assert_eq!(document["reuse"]["kv_host_pool_bytes"], "8G");
        assert_eq!(document["model"]["max_context"], 65536);
        assert_eq!(document["server"]["request_timeout"], 77);
        assert_eq!(document["reuse"]["qwen38"]["retained_host"], 4);
        assert_eq!(document["profiles"]["mine"]["vram"]["headroom_bytes"], "2G");
        assert_eq!(document["profile"], "rtx5090");
        assert!(document["vision"].is_null(), "a field nobody named is not added");
        let (other, _) = written(run(&["config", "patch", "--file", "c.yaml", "--out", "d.yaml"], &[], &files).unwrap());
        assert_eq!(other, PathBuf::from("d.yaml"));
    }

    /// Spec config-v2/02, Testing: with no field flag `config patch` still
    /// rewrites, and the content changes only where re-resolution differs.
    #[test]
    fn patch_with_no_field_flag_rewrites_only_what_resolution_changes() {
        let canonical = Format::Yaml.write(&yaml("server:\n  request_timeout: 77\n"));
        let files = MemFiles::with(&[("c.yaml", &canonical)]);
        let (_, same) = written(run(&["config", "patch", "--file", "c.yaml"], &[], &files).unwrap());
        assert_eq!(same, canonical, "nothing to change, nothing changed");
        let files = MemFiles::with(&[("s.yaml", "reuse:\n  kv_host_pool_bytes: 2147483648\n")]);
        let (_, normalized) = written(run(&["config", "patch", "--file", "s.yaml"], &[], &files).unwrap());
        assert_eq!(yaml(&normalized)["reuse"]["kv_host_pool_bytes"], "2G", "a stale spelling is rewritten");
        let (_, from_env) = written(run(&["config", "patch", "--file", "s.yaml"], &[("IGNIS_SERVER_UI", "false")], &files).unwrap());
        assert_eq!(yaml(&from_env)["server"]["ui"], false, "the environment lands in the file");
    }

    /// What `model download` resolved to: the ids, the directory, the
    /// endpoint and the token.
    fn download_of(outcome: ConfigOutcome) -> (Vec<String>, PathBuf, String, Option<String>) {
        match outcome {
            ConfigOutcome::Model(ModelCommand::Download { entries, dir, endpoint, token }) => {
                (entries.into_iter().map(|entry| entry.id).collect(), dir, endpoint, token.map(|token| token.as_str().to_owned()))
            }
            other => panic!("expected model download, got {other:?}"),
        }
    }

    const ALL_IDS: [&str; 3] = ["qwen3.8-27b", "qwen3.8-27b-abliterated", "qwen3.8-flash-next"];

    /// Spec model-download/02 AC 12, the resolution half: the ids named, or
    /// every entry, or the configured `model.id`; into `download.path` or
    /// `--out`; `download.enabled` never read, so off changes nothing.
    #[test]
    fn model_download_takes_ids_all_or_the_configured_model() {
        let files = MemFiles::default();
        let (ids, dir, endpoint, token) = download_of(run(&["model", "download"], &[], &files).unwrap());
        assert_eq!((ids, dir, endpoint, token), (vec!["qwen3.8-27b".to_owned()], PathBuf::from("./models"), "https://huggingface.co".to_owned(), None));
        let (ids, ..) = download_of(run(&["model", "download", "--model-id", "qwen3.8-flash-next"], &[], &files).unwrap());
        assert_eq!(ids, ["qwen3.8-flash-next"], "no id: the configured model");
        let (ids, ..) = download_of(run(&["model", "download", "qwen3.8-flash-next", "QWEN3.8-27B", "qwen3.8-27b"], &[], &files).unwrap());
        assert_eq!(ids, ["qwen3.8-flash-next", "qwen3.8-27b"], "in the order named, each once");
        let (ids, ..) = download_of(run(&["model", "download", "--all", "--download-enabled", "false"], &[], &files).unwrap());
        assert_eq!(ids, ALL_IDS);
        let (_, dir, ..) = download_of(run(&["model", "download", "qwen3.8-27b", "--download-path", "D:/m"], &[], &files).unwrap());
        assert_eq!(dir, PathBuf::from("D:/m"));
        let (ids, dir, ..) = download_of(run(&["model", "download", "--download-path", "D:/m", "qwen3.8-flash-next"], &[], &files).unwrap());
        assert_eq!((ids, dir), (vec!["qwen3.8-flash-next".to_owned()], PathBuf::from("D:/m")), "an id after a flag and its value");
        let (ids, dir, ..) = download_of(run(&["model", "download", "qwen3.8-27b", "--out", "E:/c", "qwen3.8-flash-next", "--download-enabled", "false"], &[], &files).unwrap());
        assert_eq!((ids, dir), (vec!["qwen3.8-27b".to_owned(), "qwen3.8-flash-next".to_owned()], PathBuf::from("E:/c")), "ids on both sides of a flag");
        let (_, dir, ..) = download_of(run(&["model", "download", "--out", "E:/carry", "--download-path", "D:/m"], &[("IGNIS_DOWNLOAD_PATH", "C:/x")], &files).unwrap());
        assert_eq!(dir, PathBuf::from("E:/carry"), "--out wins over download.path");
    }

    /// AC 12: an id outside the catalog fails naming the ones there are;
    /// `--all` beside ids, and any `model` command but the two, are refused.
    #[test]
    fn model_download_refuses_an_unknown_id_naming_the_known_ones() {
        let files = MemFiles::default();
        let err = run(&["model", "download", "qwen3.8-27b", "qwen9"], &[], &files).unwrap_err().0;
        assert!(err.contains("`qwen9`") && err.contains(&ALL_IDS.join(", ")), "{err}");
        let err = run(&["model", "download", "--model-id", "custom"], &[], &files).unwrap_err().0;
        assert!(err.contains("`custom`"), "{err}");
        assert!(run(&["model", "download", "qwen3.8-27b", "--all"], &[], &files).unwrap_err().0.contains("not both"));
        for verb in ["verify", "convert", "fetch"] {
            let err = run(&["model", verb], &[], &files).unwrap_err().0;
            assert!(err.contains(&format!("unknown command `model {verb}`")) && err.contains("download and list"), "{err}");
        }
        assert!(run(&["model"], &[], &files).unwrap_err().0.contains("download or list"));
    }

    /// AC 5, through the command line: the configured token wins, `HF_TOKEN`
    /// reaches Hugging Face only.
    #[test]
    fn model_download_carries_the_token_the_rule_picks() {
        let files = MemFiles::default();
        let hf: &'static [(&str, &str)] = &[("HF_TOKEN", "hf_personal")];
        assert_eq!(download_of(run(&["model", "download"], hf, &files).unwrap()).3.as_deref(), Some("hf_personal"));
        let mirror = ["model", "download", "--download-endpoint", "https://mirror.example.com"];
        assert_eq!(download_of(run(&mirror, hf, &files).unwrap()).3, None);
        let configured = ["model", "download", "--download-endpoint", "https://mirror.example.com", "--download-token", "tok"];
        let (_, _, endpoint, token) = download_of(run(&configured, hf, &files).unwrap());
        assert_eq!((endpoint.as_str(), token.as_deref()), ("https://mirror.example.com", Some("tok")));
    }

    /// AC 2, 3 and 13 through the command line: an operator catalog a config
    /// file names, beside it, joins the listing; one that cannot be read
    /// refuses the command.
    #[test]
    fn model_list_reads_the_catalog_the_config_file_names_beside_it() {
        let catalog = "models:\n  - id: acme-ft\n    repo: acme/ft\n    revision: v1\n    artifact: a.ninfer\n    files:\n      - { name: a.ninfer.graft.json, bytes: 1, sha256: 0000000000000000000000000000000000000000000000000000000000000001 }\n      - { name: a.ninfer, bytes: 2, sha256: 0000000000000000000000000000000000000000000000000000000000000002 }\n";
        let files = MemFiles::with(&[("conf/ignis.config.yaml", "download:\n  catalog: acme.yaml\n  path: /srv/models\n"), ("conf/acme.yaml", catalog)]);
        match run(&["model", "list", "--config", "conf/ignis.config.yaml", "--format", "json"], &[], &files).unwrap() {
            ConfigOutcome::Model(ModelCommand::List { catalog, dir, format }) => {
                assert_eq!(catalog.ids(), ["qwen3.8-27b", "qwen3.8-27b-abliterated", "qwen3.8-flash-next", "acme-ft"]);
                assert_eq!((dir, format), (PathBuf::from("/srv/models"), ListFormat::Json));
            }
            other => panic!("{other:?}"),
        }
        let (ids, ..) = download_of(run(&["model", "download", "acme-ft", "--config", "conf/ignis.config.yaml"], &[], &files).unwrap());
        assert_eq!(ids, ["acme-ft"]);
        let missing = MemFiles::with(&[("conf/ignis.config.yaml", "download:\n  catalog: nowhere.yaml\n")]);
        for argv in [&["model", "list", "--config", "conf/ignis.config.yaml"][..], &["model", "download", "--config", "conf/ignis.config.yaml"]] {
            let err = run(argv, &[], &missing).unwrap_err().0;
            assert!(err.contains("nowhere.yaml") && err.contains("download.catalog"), "{err}");
        }
        assert!(run(&["model", "list", "--format", "yaml"], &[], &files).unwrap_err().0.contains("text and json"));
    }

    /// Spec model-download/02 AC 7: `config print` shows the token as set,
    /// never its value, from a flag or from the environment.
    #[test]
    fn print_never_shows_the_download_token() {
        let files = MemFiles::default();
        let text = printed(run(&["config", "print", "--download-token", "hf_secret_flag"], &[("IGNIS_DOWNLOAD_ENDPOINT", "https://m.example.com")], &files).unwrap());
        assert!(!text.contains("hf_secret_flag") && text.contains("not shown"), "{text}");
        assert_eq!(yaml(&text)["download"]["endpoint"], "https://m.example.com", "the endpoint is shown");
        let text = printed(run(&["config", "print"], &[("IGNIS_DOWNLOAD_TOKEN", "hf_secret_env")], &files).unwrap());
        assert!(!text.contains("hf_secret_env"), "{text}");
    }

    #[test]
    fn patch_validates_the_whole_result_before_writing() {
        let files = MemFiles::with(&[("c.yaml", "server:\n  request_timeout: 77\n")]);
        let err = run(&["config", "patch", "--file", "c.yaml", "--model-prefill-chunk", "1000"], &[], &files).unwrap_err().0;
        assert!(err.contains("128"), "{err}");
        let files = MemFiles::with(&[("c.yaml", "profile: none\nvision:\n  max_tokens: 8192\n")]);
        assert!(run(&["config", "patch", "--file", "c.yaml"], &[], &files).unwrap_err().0.contains("--vision-enabled"));
    }
}
