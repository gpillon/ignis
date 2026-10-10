//! The command line's verbs (spec config-v2/01 §The CLI surface,
//! config-v2/02 §The CLI): `help`, `help --fields`, and `config generate` /
//! `print` / `patch`. A bare invocation (`ignis-server --model-artifact …`)
//! still serves; a verb is only ever the first word.
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
                assert!(document[meta.group].get(meta.name).is_some(), "{name} writes {}", meta.file_key());
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

    #[test]
    fn patch_validates_the_whole_result_before_writing() {
        let files = MemFiles::with(&[("c.yaml", "server:\n  request_timeout: 77\n")]);
        let err = run(&["config", "patch", "--file", "c.yaml", "--model-prefill-chunk", "1000"], &[], &files).unwrap_err().0;
        assert!(err.contains("128"), "{err}");
        let files = MemFiles::with(&[("c.yaml", "vision:\n  max_tokens: 8192\n")]);
        assert!(run(&["config", "patch", "--file", "c.yaml"], &[], &files).unwrap_err().0.contains("--vision-enabled"));
    }
}
