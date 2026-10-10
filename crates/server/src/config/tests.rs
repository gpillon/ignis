//! `config::resolve` end to end: every field by flag and by env var, the
//! rules that tie fields together, and the fit to a model family. The
//! behaviour each test pins is the one the flat flag set had before ADR 0046
//! regrouped it; only the spellings moved.

use super::*;
use crate::instruction::{DeveloperMessagePolicy, SystemMessagePolicy};
use crate::thinking::ReasoningEffort;
use ignis_core::compute::ModelFamily;
use ignis_core::speculation::{FLASH_NEXT_DEFAULT_DRAFT_TOKENS, FLASH_NEXT_VERIFY_ROWS};

fn no_env(_: &str) -> Option<String> {
    None
}

fn env_map(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |key| pairs.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string())
}

fn args(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn expect_config(outcome: ConfigOutcome) -> Config {
    match outcome {
        ConfigOutcome::Config(c) => c,
        other => panic!("expected Config, got {other:?}"),
    }
}

fn config(flags: &[&str]) -> Config {
    expect_config(resolve(&args(flags), no_env).expect("resolve"))
}

fn refused(flags: &[&str]) -> String {
    resolve(&args(flags), no_env).expect_err("must be refused").0
}

fn help() -> String {
    let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve") else {
        panic!("--help is help");
    };
    text
}

/// A field's two help lines (flag, env and default; then its summary).
fn help_block(flag: &str) -> String {
    let text = help();
    let lines: Vec<&str> = text.lines().collect();
    let at = lines
        .iter()
        .position(|line| line.trim_start().starts_with(&format!("{flag} ")))
        .unwrap_or_else(|| panic!("help documents {flag}:\n{text}"));
    format!("{}\n{}", lines[at], lines.get(at + 1).unwrap_or(&""))
}

// ── the config file (spec config-v2/01 §The config file) ─────────────────

use file::testing::MemFiles;
use file::Format;

fn with_files(flags: &[&str], env: impl Fn(&str) -> Option<String>, files: &MemFiles) -> Result<Config, ConfigError> {
    resolve_with(&args(flags), env, files).map(expect_config)
}

/// Spec config-v2/01, Testing: a file written with every default reads back
/// to the same config in both formats — and starts either model, though it
/// names fields only one of them has (at their defaults).
#[test]
fn a_file_of_every_default_reads_back_to_the_defaults_in_both_formats() {
    let defaults = config(&[]);
    let document = file::settings_document(defaults.settings());
    for (name, format) in [("c.yaml", Format::Yaml), ("c.json", Format::Json)] {
        let files = MemFiles::with(&[(name, &format.write(&document))]);
        let read = with_files(&["--config", name], no_env, &files).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(read.settings(), defaults.settings(), "{name}");
        assert_eq!(read.basis.sources().file_source, source::FileSource::Explicit(PathBuf::from(name)));
        for family in field::FAMILIES {
            read.for_family(family).unwrap_or_else(|e| panic!("{name} on {}: {e}", family.name()));
        }
    }
}

#[test]
fn a_file_value_sits_below_the_environment_and_the_flags() {
    let files = MemFiles::with(&[(
        "c.yaml",
        "reuse:\n  kv_host_pool_bytes: 3G\nmodel:\n  max_context: 65536\n  prefill_chunk: 512\n",
    )]);
    let env = env_map(&[("IGNIS_MODEL_MAX_CONTEXT", "32768")]);
    let config = with_files(&["--config", "c.yaml", "--model-prefill-chunk", "256"], env, &files).unwrap();
    assert_eq!(config.host_pool_bytes, 3 * GIB, "the file over the default");
    assert_eq!(config.max_context, 32_768, "the env over the file");
    assert_eq!(config.prefill_chunk, 256, "the flag over the file");
}

#[test]
fn a_file_is_named_by_the_flag_or_the_env_var_and_must_exist() {
    let files = MemFiles::with(&[("e.json", r#"{"server": {"request_timeout": 77}}"#)]);
    let env = env_map(&[("IGNIS_CONFIG", "e.json")]);
    assert_eq!(with_files(&[], env, &files).unwrap().request_timeout_secs, 77);
    let err = with_files(&["--config", "missing.yaml"], no_env, &files).unwrap_err().0;
    assert!(err.contains("missing.yaml"), "{err}");
    let err = with_files(&[], env_map(&[("IGNIS_CONFIG", "gone.yaml")]), &files).unwrap_err().0;
    assert!(err.contains("gone.yaml"), "an operator who named a file meant it to exist: {err}");
    assert!(resolve(&args(&["--config", "e.json"]), no_env).is_err(), "`resolve` reads no filesystem");
}

/// Spec config-v2/01, Testing: an unknown field or the wrong type is a
/// `ConfigError` naming the field.
#[test]
fn a_file_naming_an_unknown_field_or_a_bad_value_is_refused_by_name() {
    for (text, says) in [
        ("server:\n  bnd: x\n", "server.bnd"),
        ("model:\n  max_context: lots\n", "model.max_context"),
        ("model:\n  max_context: [1, 2]\n", "model.max_context"),
        ("vram:\n  headroom_bytes: true\n", "vram.headroom_bytes"),
        ("model: [\n", "not valid"),
    ] {
        let files = MemFiles::with(&[("c.yaml", text)]);
        let err = with_files(&["--config", "c.yaml"], no_env, &files).unwrap_err().0;
        assert!(err.contains(says) && err.contains("c.yaml"), "{text}: {err}");
    }
}

/// The family sections are the file's half of family scope.
#[test]
fn a_files_family_section_applies_to_that_family_alone() {
    let files = MemFiles::with(&[(
        "c.yaml",
        "reuse:\n  kv_host_pool_bytes: 8G\n  qwen38flashnext:\n    kv_host_pool_bytes: 1G\nngram:\n  hot_bytes: 4G\n",
    )]);
    let config = with_files(&["--config", "c.yaml"], no_env, &files).unwrap();
    assert_eq!(config.for_family(ModelFamily::FlashNext).unwrap().host_pool_bytes, GIB);
    assert_eq!(config.for_family(ModelFamily::Qwen38_27b).unwrap_err().0.contains("ngram.hot_bytes"), true);
    let (switched, dropped) = fit_to_family(&config, ModelFamily::Qwen38_27b).unwrap();
    assert_eq!((switched.host_pool_bytes, dropped), (8 * GIB, vec!["--ngram-hot-bytes".to_owned()]));
}

/// Every group struct and `Settings` derive their serde through the field
/// table: JSON and YAML round-trip, and a bad document is refused by name.
#[test]
fn the_settings_and_every_group_serialize_and_deserialize_in_both_formats() {
    let settings = config(&["--reuse-kv-host-pool-bytes", "3G", "--switch-known-models", "a=F:/a.ninfer", "--server-api-key", "k"])
        .settings()
        .clone();
    let json = serde_json::to_string(&settings).unwrap();
    assert_eq!(serde_json::from_str::<schema::Settings>(&json).unwrap(), settings);
    let yaml = serde_yaml::to_string(&settings).unwrap();
    assert_eq!(serde_yaml::from_str::<schema::Settings>(&yaml).unwrap(), settings);
    let reuse: schema::ReuseGroup = serde_json::from_str(r#"{"kv_host_pool_bytes": "3G"}"#).unwrap();
    assert_eq!(reuse, settings.reuse);
    let err = serde_json::from_str::<schema::ReuseGroup>(r#"{"kv_host_pool": 1}"#).unwrap_err().to_string();
    assert!(err.contains("reuse.kv_host_pool"), "{err}");
}

// ── config-file auto-discovery (spec config-v2/02) ───────────────────────

/// The per-user config directory's candidate, as discovery spells it here.
fn user_candidate() -> (&'static [(&'static str, &'static str)], &'static str) {
    if cfg!(windows) {
        (&[("APPDATA", "U")], "U/ignis/config.yaml")
    } else {
        (&[("XDG_CONFIG_HOME", "U")], "U/ignis/config.yaml")
    }
}

/// Spec config-v2/02, Testing: the working directory is found before the
/// user config directory when both hold a file; either is used when it is
/// the only one; none found is the no-file case, not an error.
#[test]
fn discovery_finds_the_working_directory_first_then_the_user_directory() {
    let (env, user_path) = user_candidate();
    let user = PathBuf::from("U").join("ignis").join("config.yaml");
    assert_eq!(user, PathBuf::from(user_path).components().collect::<PathBuf>());
    let both = MemFiles::with(&[("ignis.config.yaml", "server:\n  request_timeout: 11\n")]);
    both.files.borrow_mut().insert(user.clone(), "server:\n  request_timeout: 22\n".to_owned());
    let found = with_files(&[], env_map(env), &both).unwrap();
    assert_eq!(found.request_timeout_secs, 11);
    assert_eq!(found.basis.sources().file_source, source::FileSource::Discovered(PathBuf::from("ignis.config.yaml")));

    let json_only = MemFiles::with(&[("ignis.config.json", r#"{"server": {"request_timeout": 33}}"#)]);
    assert_eq!(with_files(&[], env_map(env), &json_only).unwrap().request_timeout_secs, 33);

    let user_only = MemFiles::default();
    user_only.files.borrow_mut().insert(user.clone(), "server:\n  request_timeout: 22\n".to_owned());
    let found = with_files(&[], env_map(env), &user_only).unwrap();
    assert_eq!((found.request_timeout_secs, found.basis.sources().file_source.kind()), (22, "discovered"));

    let none = with_files(&[], env_map(env), &MemFiles::default()).unwrap();
    assert_eq!(none.basis.sources().file_source, source::FileSource::None);
}

/// Spec config-v2/02, Testing: a file named by `--config` or `IGNIS_CONFIG`
/// short-circuits discovery entirely — no candidate is even looked at.
#[test]
fn a_named_file_short_circuits_discovery_without_looking() {
    let (env, _) = user_candidate();
    let files = MemFiles::with(&[("named.yaml", "server: {}\n"), ("ignis.config.yaml", "server:\n  request_timeout: 11\n")]);
    let named = with_files(&["--config", "named.yaml"], env_map(env), &files).unwrap();
    assert_eq!(named.request_timeout_secs, DEFAULT_REQUEST_TIMEOUT_SECS);
    assert_eq!(named.basis.sources().file_source.kind(), "explicit");
    assert!(files.looked_at.borrow().is_empty(), "{:?}", files.looked_at.borrow());
    let from_env = with_files(&[], env_map(&[("IGNIS_CONFIG", "named.yaml")]), &files).unwrap();
    assert_eq!(from_env.basis.sources().file_source, source::FileSource::Explicit(PathBuf::from("named.yaml")));
    assert!(files.looked_at.borrow().is_empty());
}

/// Spec config-v2/02, Testing: the startup line, for all three sources.
#[test]
fn the_startup_log_names_the_config_source_in_all_three_states() {
    use tracing_subscriber::layer::SubscriberExt;
    let (env, _) = user_candidate();
    let files = MemFiles::with(&[("named.yaml", "server: {}\n"), ("ignis.config.yaml", "server: {}\n")]);
    let cases = [
        (with_files(&["--config", "named.yaml"], env_map(env), &files).unwrap(), "explicit", "named.yaml"),
        (with_files(&[], env_map(env), &files).unwrap(), "discovered", "ignis.config.yaml"),
        (config(&[]), "none", "none"),
    ];
    for (config, source, path) in cases {
        let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, || log_source(&config));
        let lines = sink.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        let event: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(event["event_name"], "ignis.config.source", "{event}");
        let field = |name: &str| event.get(name).or_else(|| event["attributes"].get(name)).cloned();
        assert_eq!(field("source"), Some(source.into()), "{event}");
        assert_eq!(field("path"), Some(path.into()), "{event}");
        assert_eq!(field("profile"), Some("rtx5090".into()), "{event}");
    }
}

// ── hardware profiles (spec config-v2/02 §`--profile`) ───────────────────

/// The implicit default profile restates the hardcoded defaults: an unset
/// `--profile` changes nothing on the owner's machine.
#[test]
fn the_default_profile_changes_nothing() {
    let unset = config(&[]);
    let named = config(&["--profile", "rtx5090"]);
    assert_eq!(unset.settings(), named.settings());
    assert_eq!(unset.basis.sources().profile_name, "rtx5090");
    let headroom = unset.basis.resolution().origin("vram", "headroom_bytes").expect("the profile gave it");
    assert_eq!(headroom.source, source::Source::Profile);
}

/// Spec config-v2/02, Testing: a profile's value resolves exactly where no
/// flag, env var or file names the field, and a profile defined in the file
/// resolves by the same path as a built-in one.
#[test]
fn a_profile_value_applies_only_where_nothing_else_names_the_field() {
    let files = MemFiles::with(&[(
        "c.yaml",
        "profiles:\n  small-card:\n    vram:\n      headroom_bytes: 3G\n    reuse:\n      kv_host_pool_bytes: 1G\n      qwen38flashnext:\n        kv_host_pool_bytes: 512M\n",
    )]);
    let profiled = with_files(&["--config", "c.yaml", "--profile", "small-card"], no_env, &files).unwrap();
    assert_eq!(profiled.vram, VramMode::Derived { headroom_bytes: 3 * GIB });
    assert_eq!(profiled.host_pool_bytes, GIB);
    assert_eq!(profiled.for_family(ModelFamily::FlashNext).unwrap().host_pool_bytes, 512 << 20, "a profile has family sections too");
    let flag = with_files(&["--config", "c.yaml", "--profile", "small-card", "--vram-headroom-bytes", "2G"], no_env, &files).unwrap();
    assert_eq!(flag.vram, VramMode::Derived { headroom_bytes: 2 * GIB }, "a flag over the profile");
    let env = env_map(&[("IGNIS_VRAM_HEADROOM_BYTES", "4G")]);
    assert_eq!(with_files(&["--config", "c.yaml", "--profile", "small-card"], env, &files).unwrap().vram, VramMode::Derived { headroom_bytes: 4 * GIB });
    // A profile value is a default: an operator's budget wins over its
    // headroom without the two being refused as a pair.
    let budget = with_files(&["--config", "c.yaml", "--profile", "small-card", "--vram-budget-bytes", "20G"], no_env, &files).unwrap();
    assert_eq!(budget.vram, VramMode::Explicit { budget_bytes: 20 * GIB, allow_oversubscription: false });
}

#[test]
fn the_file_value_wins_over_its_own_profile_and_the_profile_is_named_by_flag_env_or_file() {
    let files = MemFiles::with(&[(
        "c.yaml",
        "profile: from-file\nvram:\n  headroom_bytes: 5G\nreuse:\n  kv_host_pool_bytes: 3G\nprofiles:\n  from-file:\n    reuse:\n      kv_host_pool_bytes: 1G\n      retained_device: 2\n  from-env:\n    reuse:\n      retained_device: 4\n",
    )]);
    let from_file = with_files(&["--config", "c.yaml"], no_env, &files).unwrap();
    assert_eq!(from_file.basis.sources().profile_name, "from-file");
    assert_eq!(from_file.host_pool_bytes, 3 * GIB, "the file over its profile");
    assert_eq!(from_file.retained_device_slots, 2);
    let env = env_map(&[("IGNIS_PROFILE", "from-env")]);
    assert_eq!(with_files(&["--config", "c.yaml"], env, &files).unwrap().retained_device_slots, 4, "the env var over the file's choice");
    let env = env_map(&[("IGNIS_PROFILE", "from-env")]);
    assert_eq!(with_files(&["--config", "c.yaml", "--profile", "from-file"], env, &files).unwrap().retained_device_slots, 2, "the flag over the env var");
    let err = with_files(&["--config", "c.yaml", "--profile", "h100"], no_env, &files).unwrap_err().0;
    assert!(err.contains("`h100`") && err.contains("from-env, from-file, rtx5090"), "{err}");
    assert!(refused(&["--profile", "nope"]).contains("rtx5090"));
}

/// A profile and the family scope compose (spec config-v2/02 AC 10): the
/// card's shape from the profile, each model's on top from the file.
#[test]
fn a_family_override_wins_over_the_profile() {
    let files = MemFiles::with(&[(
        "c.yaml",
        "profile: card\nreuse:\n  qwen38flashnext:\n    kv_host_pool_bytes: 1G\nprofiles:\n  card:\n    reuse:\n      kv_host_pool_bytes: 6G\n",
    )]);
    let config = with_files(&["--config", "c.yaml"], no_env, &files).unwrap();
    assert_eq!(config.for_family(ModelFamily::Qwen38_27b).unwrap().host_pool_bytes, 6 * GIB, "the card's");
    assert_eq!(config.for_family(ModelFamily::FlashNext).unwrap().host_pool_bytes, GIB, "the model's over the card's");
}

// ── the n-gram cache and KV-disk ─────────────────────────────────────────

/// The n-gram cache is on and beside the model by default; `model` says so
/// explicitly, `auto` is the OS cache directory, anything else a directory;
/// flags win over the environment; bad values are refused.
#[test]
fn ngram_cache_defaults_flags_env_and_invalid_values() {
    use ignis_core::ngram_cache::CacheLocation;
    let default = config(&[]);
    assert!(default.ngram_cache.enabled);
    assert_eq!(default.ngram_cache.location, CacheLocation::Model);

    let env = env_map(&[("IGNIS_NGRAM_PERSIST", "false"), ("IGNIS_NGRAM_PERSIST_PATH", "custom")]);
    let config_env = expect_config(resolve(&[], &env).unwrap());
    assert!(!config_env.ngram_cache.enabled);
    assert_eq!(config_env.ngram_cache.location, CacheLocation::Directory(PathBuf::from("custom")));

    let flags = args(&["--ngram-persist", "true", "--ngram-persist-path", "auto"]);
    let config_flags = expect_config(resolve(&flags, &env).unwrap());
    assert!(config_flags.ngram_cache.enabled);
    assert_eq!(config_flags.ngram_cache.location, CacheLocation::Auto);

    let config_model = expect_config(resolve(&args(&["--ngram-persist-path", "model"]), &env).unwrap());
    assert_eq!(config_model.ngram_cache.location, CacheLocation::Model);
    // A bool's flag may stand bare.
    assert!(config(&["--ngram-persist"]).ngram_cache.enabled);
    // An empty value is an unset one, as everywhere.
    assert_eq!(config(&["--ngram-persist-path", ""]).ngram_cache.location, CacheLocation::Model);

    for flags in [vec!["--ngram-persist", "yes"], vec!["--ngram-persist-path"]] {
        assert!(resolve(&args(&flags), no_env).is_err(), "{flags:?}");
    }
}

/// KV-disk (spec vram-budget/03): unnamed is the family's budget and beside
/// the model; a size or `0`, `model`, `auto` or a directory by flag or
/// environment, the flag winning; a bad size is refused.
#[test]
fn kv_disk_defaults_flags_env_and_invalid_values() {
    use ignis_core::ngram_cache::CacheLocation;
    let default = config(&[]);
    assert_eq!(default.kv_disk_bytes, None, "the family's");
    assert_eq!(default.kv_disk_location, CacheLocation::Model);

    let env = env_map(&[("IGNIS_KV_DISK_BYTES", "8G"), ("IGNIS_KV_DISK_PATH", "auto")]);
    let from_env = expect_config(resolve(&[], &env).unwrap());
    assert_eq!(from_env.kv_disk_bytes, Some(8 << 30));
    assert_eq!(from_env.kv_disk_location, CacheLocation::Auto);

    let flags = args(&["--kv-disk-bytes", "0", "--kv-disk-path", "D:/kv"]);
    let from_flags = expect_config(resolve(&flags, &env).unwrap());
    assert_eq!(from_flags.kv_disk_bytes, Some(0), "0 is off, and the flag wins");
    assert_eq!(from_flags.kv_disk_location, CacheLocation::Directory(PathBuf::from("D:/kv")));

    let model = expect_config(resolve(&args(&["--kv-disk-path", "model"]), &env).unwrap());
    assert_eq!(model.kv_disk_location, CacheLocation::Model);

    for flags in [vec!["--kv-disk-bytes", "lots"], vec!["--kv-disk-bytes"], vec!["--kv-disk-path"]] {
        assert!(resolve(&args(&flags), no_env).is_err(), "{flags:?}");
    }
    assert!(resolve(&[], &env_map(&[("IGNIS_KV_DISK_BYTES", "-1")])).is_err());
    assert!(help().contains("--kv-disk-bytes") && help().contains("--kv-disk-path"));
}

// ── precedence, aliases and the CLI's own flags ──────────────────────────

#[test]
fn no_args_no_env_falls_back_to_defaults() {
    let config = config(&[]);
    assert_eq!(config.model, DEFAULT_MODEL);
    assert!(!config.model_named);
    assert_eq!(config.bind, DEFAULT_BIND);
    assert_eq!(config.artifact, None);
    assert!(config.enable_thinking);
    assert_eq!(config.reasoning_effort, None);
    assert_eq!(config.prefill_chunk, DEFAULT_PREFILL_CHUNK);
    assert_eq!(config.max_context, DEFAULT_MAX_CONTEXT);
    assert_eq!(config.kv_format, KvFormat::HqE8_2b);
    assert_eq!(config.kv_pool, None, "the KV pool policy's size");
    assert!(!config.allow_expert_cache_below_floor);
    assert_eq!(config.vram, VramMode::Derived { headroom_bytes: DEFAULT_VRAM_HEADROOM_BYTES });
    assert_eq!(config.host_pool_bytes, DEFAULT_HOST_POOL_BYTES);
    assert_eq!(config.speculation, None);
    assert_eq!(config.request_timeout_secs, DEFAULT_REQUEST_TIMEOUT_SECS);
    assert_eq!(config.switch_drain_timeout_secs, DEFAULT_SWITCH_DRAIN_TIMEOUT_SECS);
    assert!(config.allow_model_switch);
    assert!(config.known_models.is_empty());
    assert_eq!(config.basis.family(), None);
}

#[test]
fn env_only_wins_over_defaults() {
    let env = env_map(&[
        ("IGNIS_MODEL_ID", "custom-model"),
        ("IGNIS_SERVER_BIND", "0.0.0.0:9000"),
        ("IGNIS_MODEL_ARTIFACT", "/path/to.ninfer"),
        ("IGNIS_MODEL_ENABLE_THINKING", "false"),
        ("IGNIS_MODEL_REASONING_EFFORT", "low"),
    ]);
    let config = expect_config(resolve(&[], env).expect("resolve"));
    assert_eq!(config.model, "custom-model");
    assert!(config.model_named);
    assert_eq!(config.bind, "0.0.0.0:9000");
    assert_eq!(config.artifact, Some(PathBuf::from("/path/to.ninfer")));
    assert!(!config.enable_thinking);
    assert_eq!(config.reasoning_effort, Some(ReasoningEffort::Low));
}

#[test]
fn flag_only_wins_over_defaults() {
    let config = config(&[
        "--model-id",
        "flag-model",
        "--server-bind",
        "0.0.0.0:1234",
        "--model-artifact",
        "/flag/artifact.ninfer",
        "--model-enable-thinking",
        "false",
        "--model-reasoning-effort",
        "high",
    ]);
    assert_eq!(config.model, "flag-model");
    assert_eq!(config.bind, "0.0.0.0:1234");
    assert_eq!(config.artifact, Some(PathBuf::from("/flag/artifact.ninfer")));
    assert!(!config.enable_thinking);
    assert_eq!(config.reasoning_effort, Some(ReasoningEffort::High));
}

/// ADR 0046 renamed every flag into its group, with one spelling per field:
/// the old short aliases went with the old names.
#[test]
fn the_old_short_aliases_and_flat_names_are_refused() {
    for flag in ["-m", "-b", "-a", "--model", "--bind", "--artifact", "--kv-host-pool-bytes", "--no-ui"] {
        let err = refused(&[flag, "x"]);
        assert!(err.contains(&format!("`{flag}`")), "{err}");
    }
    assert!(refused(&["--kv-host-pool-bytes", "8G"]).contains("`--reuse-kv-host-pool-bytes`"), "a hint at the new name");
}

#[test]
fn the_retired_telemetry_sink_flag_is_refused_and_its_env_var_ignored() {
    // ADR 0025: the interval counters are a log event now, so there is no
    // separate sink to point anywhere. A leftover `--telemetry` in a launch
    // script must fail loudly rather than be silently dropped.
    for flag in ["--telemetry", "-t"] {
        let err = refused(&[flag, "/tmp/telemetry.jsonl"]);
        assert!(err.contains(flag), "{err}");
    }
    let env = env_map(&[("IGNIS_TELEMETRY", "/tmp/telemetry.jsonl")]);
    assert_eq!(
        resolve(&[], env).expect("resolve"),
        resolve(&[], no_env).expect("resolve"),
        "IGNIS_TELEMETRY no longer changes the resolved config"
    );
    assert!(!help().contains("telemetry"), "help must not document it");
}

#[test]
fn a_flag_wins_over_a_matching_env_var_per_field_independently() {
    let env = env_map(&[("IGNIS_SERVER_BIND", "0.0.0.0:9000"), ("IGNIS_MODEL_ARTIFACT", "/env/artifact.ninfer")]);
    let config = expect_config(resolve(&args(&["--server-bind", "0.0.0.0:1234"]), env).expect("resolve"));
    assert_eq!(config.bind, "0.0.0.0:1234", "flag must win over env");
    assert_eq!(
        config.artifact,
        Some(PathBuf::from("/env/artifact.ninfer")),
        "env must still apply to a field the flag didn't touch"
    );
}

#[test]
fn an_unrecognized_flag_is_a_config_error() {
    assert!(refused(&["--nope"]).contains("--nope"));
}

#[test]
fn a_flag_missing_its_value_is_a_config_error() {
    assert!(refused(&["--server-bind"]).contains("--server-bind"));
}

#[test]
fn an_invalid_enable_thinking_is_refused_naming_where_it_came_from() {
    let flag_err = refused(&["--model-enable-thinking", "nope"]);
    assert!(flag_err.starts_with("`--model-enable-thinking` ") && flag_err.contains("`nope`"), "{flag_err}");
    let env_err = resolve(&[], env_map(&[("IGNIS_MODEL_ENABLE_THINKING", "nope")])).unwrap_err().0;
    assert_eq!(
        env_err.replacen("IGNIS_MODEL_ENABLE_THINKING", "--model-enable-thinking", 1),
        flag_err,
        "one reason, whichever way it arrived"
    );
}

#[test]
fn the_thinking_budget_ships_on_and_off_turns_it_off() {
    // Spec server/08: a measured default, so that a client that knows
    // nothing of the extension still gets an answer at `xhigh`.
    assert_eq!(config(&[]).thinking_budget, Some(DEFAULT_THINKING_BUDGET));
    assert_eq!(DEFAULT_THINKING_BUDGET, 32_768);
    // An empty env var is an unset one, as for the other defaults.
    let from_empty = expect_config(resolve(&[], env_map(&[("IGNIS_MODEL_THINKING_BUDGET", "")])).expect("resolve"));
    assert_eq!(from_empty.thinking_budget, Some(DEFAULT_THINKING_BUDGET));
    // `off`, by flag or by env: no default budget at all.
    assert_eq!(config(&["--model-thinking-budget", "off"]).thinking_budget, None);
    let off = expect_config(resolve(&[], env_map(&[("IGNIS_MODEL_THINKING_BUDGET", "off")])).expect("resolve"));
    assert_eq!(off.thinking_budget, None);
    // A number, and the flag over the env var.
    let env = env_map(&[("IGNIS_MODEL_THINKING_BUDGET", "off")]);
    let flag = expect_config(resolve(&args(&["--model-thinking-budget", "12288"]), env).expect("resolve"));
    assert_eq!(flag.thinking_budget, Some(12288));
    let six = expect_config(resolve(&[], env_map(&[("IGNIS_MODEL_THINKING_BUDGET", "6144")])).expect("resolve"));
    assert_eq!(six.thinking_budget, Some(6144));
}

#[test]
fn a_thinking_budget_that_is_neither_a_count_nor_off_refuses_the_start() {
    for bad in ["0", "-1", "lots", "8k", "OFF"] {
        let err = refused(&["--model-thinking-budget", bad]);
        assert!(err.contains("off") && err.contains(bad), "{bad}: the message names the way to say none: {err}");
    }
}

#[test]
fn the_help_names_the_thinking_budget_default_and_off() {
    let block = help_block("--model-thinking-budget");
    assert!(block.contains("IGNIS_MODEL_THINKING_BUDGET"), "{block}");
    assert!(block.contains(&format!("default: {DEFAULT_THINKING_BUDGET}")), "{block}");
    assert!(block.contains("off"), "{block}");
}

/// GitHub #307: the help names Flash-Next's MTP backend and its row budget,
/// so an operator finds them without the user docs.
#[test]
fn the_help_names_mtp_and_the_draft_row_budget() {
    let spec = help_block("--spec-backend");
    assert!(spec.contains("mtp") && spec.contains("off"), "{spec}");
    let rows = help_block("--spec-draft-rows");
    assert!(rows.contains("IGNIS_SPEC_DRAFT_ROWS"), "{rows}");
    assert!(rows.contains(&format!("0 (= {FLASH_NEXT_VERIFY_ROWS})")), "{rows}");
}

#[test]
fn an_invalid_reasoning_effort_is_refused_naming_every_value() {
    let err = refused(&["--model-reasoning-effort", "nonsense"]);
    assert!(err.contains("--model-reasoning-effort") && err.contains("nonsense") && err.contains("xhigh"), "{err}");
    let env_err = resolve(&[], env_map(&[("IGNIS_MODEL_REASONING_EFFORT", "nonsense")])).unwrap_err().0;
    assert!(env_err.starts_with("`IGNIS_MODEL_REASONING_EFFORT`"), "{env_err}");
}

#[test]
fn help_short_circuits_before_other_flags_are_validated() {
    assert!(matches!(resolve(&args(&["--help", "--nonsense"]), no_env).expect("resolve"), ConfigOutcome::Help(_)));
}

#[test]
fn help_short_circuits_even_where_it_would_otherwise_be_consumed_as_a_value() {
    // `--server-bind` normally requires a following value; `--help` still
    // wins rather than being swallowed as that value.
    assert!(matches!(resolve(&args(&["--server-bind", "--help"]), no_env).expect("resolve"), ConfigOutcome::Help(_)));
}

#[test]
fn help_alias_short_circuits_too() {
    assert!(matches!(resolve(&args(&["-h"]), no_env).expect("resolve"), ConfigOutcome::Help(_)));
}

#[test]
fn version_short_circuits_before_other_flags_are_validated() {
    assert!(matches!(resolve(&args(&["--version", "--nonsense"]), no_env).expect("resolve"), ConfigOutcome::Version(_)));
}

#[test]
fn version_alias_short_circuits_too() {
    assert!(matches!(resolve(&args(&["-V"]), no_env).expect("resolve"), ConfigOutcome::Version(_)));
}

// ── the engine-shape fields (GitHub #87) ─────────────────────────────────

#[test]
fn the_default_context_admits_a_32k_prompt_plus_a_generation_budget() {
    // G2's largest cell is a 32,768-token prompt; the default cap must admit
    // it *and* leave room to generate, without editing code.
    let config = config(&[]);
    assert!(config.max_context > 32_768, "{}", config.max_context);
    // The pool is the policy's, which the load's plan refuses when it cannot
    // hold one such sequence (GitHub #210, ADR 0045).
    assert_eq!(config.kv_pool, None);
}

#[test]
fn the_engine_shape_env_vars_win_over_the_defaults() {
    let env = env_map(&[("IGNIS_MODEL_PREFILL_CHUNK", "2048"), ("IGNIS_MODEL_MAX_CONTEXT", "16384")]);
    let config = expect_config(resolve(&[], env).expect("resolve"));
    assert_eq!(config.prefill_chunk, 2048);
    assert_eq!(config.max_context, 16_384);
}

#[test]
fn the_engine_shape_flags_win_over_their_env_vars() {
    let env = env_map(&[("IGNIS_MODEL_PREFILL_CHUNK", "2048"), ("IGNIS_MODEL_MAX_CONTEXT", "16384")]);
    let a = args(&["--model-prefill-chunk", "128", "--model-max-context", "8192"]);
    let config = expect_config(resolve(&a, env).expect("resolve"));
    assert_eq!(config.prefill_chunk, 128, "flag must win over env");
    assert_eq!(config.max_context, 8_192, "flag must win over env");
}

#[test]
fn the_decode_share_is_a_percent_below_100_and_unset_by_default() {
    // GitHub #306: unset, the model family's own (`EngineShape::for_family`).
    assert_eq!(config(&[]).decode_share_percent, None);
    assert_eq!(config(&["--model-decode-share", "50"]).decode_share_percent, Some(50));
    let zero = expect_config(resolve(&[], env_map(&[("IGNIS_MODEL_DECODE_SHARE", "0")])).expect("resolve"));
    assert_eq!(zero.decode_share_percent, Some(0), "0 is ADR 0018's one round per chunk");
    for bad in ["100", "150", "-1", "0.5", "half"] {
        let err = refused(&["--model-decode-share", bad]);
        assert!(err.contains("--model-decode-share") && err.contains(bad), "{bad}: {err}");
    }
}

#[test]
fn an_unaligned_prefill_chunk_is_a_usage_error() {
    // The alignment rule is the reference's own; an unaligned width is
    // rejected before any loader work, not at the first long prompt.
    let err = refused(&["--model-prefill-chunk", "1000"]);
    assert!(err.contains("128"), "the message must name the rule: {err}");
    assert!(err.contains("1000"), "the message must name the value: {err}");
}

#[test]
fn a_zero_prefill_chunk_is_a_usage_error() {
    assert!(refused(&["--model-prefill-chunk", "0"]).contains("nonzero"));
}

#[test]
fn a_non_numeric_prefill_chunk_is_a_usage_error() {
    assert!(refused(&["--model-prefill-chunk", "wide"]).contains("--model-prefill-chunk"));
}

#[test]
fn an_invalid_prefill_chunk_env_var_is_a_usage_error_too() {
    // Same rule whichever way the value arrived: the env var is not a back
    // door around the validation.
    let err = resolve(&[], env_map(&[("IGNIS_MODEL_PREFILL_CHUNK", "300")])).expect_err("must reject");
    assert!(err.0.contains("128") && err.0.contains("IGNIS_MODEL_PREFILL_CHUNK"), "{err}");
}

#[test]
fn a_zero_max_context_is_a_usage_error() {
    assert!(refused(&["--model-max-context", "0"]).contains("--model-max-context"));
}

// ── the default max_tokens (ADR 0045, GitHub #309) ───────────────────────

#[test]
fn the_default_max_tokens_is_38912_and_takes_a_value_from_every_spelling() {
    assert_eq!(config(&[]).default_max_tokens, 38_912);
    // AC 10: a non-default value, 8,192, from the flag and from the env var
    // in turn (make's knob: `mk/flags-selftest.sh`).
    assert_eq!(config(&["--model-default-max-tokens", "8192"]).default_max_tokens, 8_192);
    let env = env_map(&[("IGNIS_MODEL_DEFAULT_MAX_TOKENS", "8192")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("resolve")).default_max_tokens, 8_192);
    let flag = expect_config(resolve(&args(&["--model-default-max-tokens", "4096"]), &env).expect("resolve"));
    assert_eq!(flag.default_max_tokens, 4_096, "the flag wins over the env var");
    assert_eq!(config(&["--model-default-max-tokens", "0"]).default_max_tokens, 0, "0 is none");
    // Past the context it is accepted: the scheduler clamps it to what each
    // prompt leaves, so it acts as the context.
    let past = config(&["--model-max-context", "40960", "--model-default-max-tokens", "1000000"]);
    assert_eq!(past.default_max_tokens, 1_000_000);
}

#[test]
fn a_malformed_default_max_tokens_is_refused_naming_where_it_came_from() {
    for bad in ["eight", "-1", "8k", "1.5"] {
        assert!(refused(&["--model-default-max-tokens", bad]).contains("--model-default-max-tokens"), "{bad}");
    }
    let err = resolve(&[], env_map(&[("IGNIS_MODEL_DEFAULT_MAX_TOKENS", "lots")])).expect_err("env");
    assert!(err.0.contains("IGNIS_MODEL_DEFAULT_MAX_TOKENS"), "{err}");
}

#[test]
fn the_help_names_the_default_max_tokens_and_none() {
    let block = help_block("--model-default-max-tokens");
    assert!(block.contains("IGNIS_MODEL_DEFAULT_MAX_TOKENS"), "{block}");
    assert!(block.contains("default: 38912"), "{block}");
    assert!(block.contains("0 = none"), "{block}");
}

#[test]
fn an_unnamed_pool_is_left_to_the_vram_plan_at_any_context() {
    // No auto budget to outgrow any more (GitHub #210): the pool is what the
    // VRAM budget leaves, in either format and at any cap, and the plan —
    // not this config — refuses a start where that is short.
    for (format, context) in [("bf16", 200_000u32), ("hq-e8-2b", 600_000)] {
        let config = config(&["--model-kv-format", format, "--model-max-context", &context.to_string()]);
        assert_eq!(config.max_context, context);
        assert_eq!(config.kv_pool, None, "{format} at {context}");
    }
}

#[test]
fn help_lists_the_engine_shape_flags() {
    let text = help();
    for flag in ["--model-prefill-chunk", "--model-max-context", "--model-kv-format", "--vram-kv-pool-bytes"] {
        assert!(text.contains(flag), "help must document {flag}:\n{text}");
    }
    assert!(text.contains("hq-e8-2b"), "help must name both formats:\n{text}");
}

// ── the KV format and pool budget (GitHub #122) ──────────────────────────

#[test]
fn the_kv_format_flag_wins_over_the_env_var_and_the_default() {
    let env = env_map(&[("IGNIS_MODEL_KV_FORMAT", "bf16")]);
    let flag = expect_config(resolve(&args(&["--model-kv-format", "hq-e8-2b"]), env).expect("resolve"));
    assert_eq!(flag.kv_format, KvFormat::HqE8_2b);
    // Both halves name the format the default is *not*, so neither can pass
    // by agreeing with it (GitHub #123 made the default hq-e8-2b).
    let env = env_map(&[("IGNIS_MODEL_KV_FORMAT", "bf16")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).kv_format, KvFormat::Bf16);
}

#[test]
fn an_unknown_kv_format_is_a_usage_error() {
    let err = refused(&["--model-kv-format", "fp8"]);
    assert!(err.contains("--model-kv-format") && err.contains("fp8") && err.contains("bf16"), "{err}");
}

#[test]
fn the_same_named_budget_buys_more_tokens_under_hq() {
    // The format is a real option: one budget, two capacities. This is the
    // whole reason the pool is described in bytes.
    let geometry = ignis_core::KvGeometry::qwen38_27b();
    let named = |format| config(&["--model-kv-format", format, "--vram-kv-pool-bytes", "4G"]);
    let (bf16, hq) = (named("bf16"), named("hq-e8-2b"));
    assert_eq!(bf16.kv_pool, hq.kv_pool);
    let Some(KvPoolSize::Bytes(bytes)) = hq.kv_pool else {
        panic!("named in bytes: {:?}", hq.kv_pool);
    };
    let bf16_capacity = ignis_core::plan_kv_pool(bf16.kv_format, geometry, bytes).token_capacity;
    let hq_capacity = ignis_core::plan_kv_pool(hq.kv_format, geometry, bytes).token_capacity;
    assert!(hq_capacity > bf16_capacity * 7, "{hq_capacity} vs {bf16_capacity}");
    // And it clears the standard target profile: 8 lanes x 40,960.
    assert!(hq_capacity >= 8 * 40_960);
}

#[test]
fn a_named_pool_budget_resolves_from_the_flag_and_the_env() {
    assert_eq!(config(&["--vram-kv-pool-bytes", "8G"]).kv_pool, Some(KvPoolSize::Bytes(8 * 1024 * 1024 * 1024)));
    let env = env_map(&[("IGNIS_VRAM_KV_POOL_BYTES", "6144MiB")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).kv_pool, Some(KvPoolSize::Bytes(6144 * 1024 * 1024)));
    // A bare count is still a byte count.
    assert_eq!(config(&["--vram-kv-pool-bytes", "4294967296"]).kv_pool, Some(KvPoolSize::Bytes(4 << 30)));
}

#[test]
fn a_named_pool_takes_a_token_count_with_binary_multipliers() {
    // ADR 0045 (AC 5): the quantity the owner decides in, the same context
    // on either model and either format.
    for (raw, tokens) in [("512Ktok", 524_288), ("2Mtok", 2 * 1024 * 1024), ("1000tok", 1_000), ("512ktok", 524_288), (" 64Ktok ", 65_536)] {
        assert_eq!(config(&["--vram-kv-pool-bytes", raw]).kv_pool, Some(KvPoolSize::Tokens(tokens)), "{raw}");
    }
    // The env var takes the same spellings, and the flag wins over it.
    let env = env_map(&[("IGNIS_VRAM_KV_POOL_BYTES", "512Ktok")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("env")).kv_pool, Some(KvPoolSize::Tokens(524_288)));
    let flag = expect_config(resolve(&args(&["--vram-kv-pool-bytes", "4G"]), &env).expect("flag"));
    assert_eq!(flag.kv_pool, Some(KvPoolSize::Bytes(4 << 30)));
}

#[test]
fn the_config_only_parses_a_pool_and_the_plan_judges_it() {
    // ADR 0045: a byte count is a different context on each model, and only
    // the load knows which. The config accepts it in either format, and the
    // plan refuses it below one context and a page per retained slot
    // (`vram.rs`'s `a_named_pool_below_one_context_refuses_*`).
    for format in ["bf16", "hq-e8-2b"] {
        assert_eq!(config(&["--vram-kv-pool-bytes", "1M", "--model-kv-format", format]).kv_pool, Some(KvPoolSize::Bytes(1 << 20)));
    }
    // What the plan will make of 512 MiB on the 27B at the default context:
    // under hq it buys more than one context and the retained slots' pages,
    // under BF16 not -- the format still decides.
    let geometry = ignis_core::KvGeometry::qwen38_27b();
    let floor = DEFAULT_MAX_CONTEXT.div_ceil(64) + DEFAULT_RETAINED_HOST_SLOTS + DEFAULT_RETAINED_DEVICE_SLOTS;
    let pages = |format: KvFormat| KvPoolSize::Bytes(512 << 20).pages(format.page_bytes(geometry));
    assert!(pages(KvFormat::HqE8_2b) >= floor, "{} vs {floor}", pages(KvFormat::HqE8_2b));
    assert!(pages(KvFormat::Bf16) < floor, "{} vs {floor}", pages(KvFormat::Bf16));
}

#[test]
fn a_malformed_pool_budget_is_a_usage_error() {
    for raw in ["", "4 GiB please", "-1", "4TB", "G", "tok", "Ktok", "4Gtok", "1.5Ktok", "-1tok", "12 tok"] {
        match resolve(&args(&["--vram-kv-pool-bytes", raw]), no_env) {
            // An empty value is an unset one, the same as every other flag.
            Ok(config) if raw.is_empty() => assert_eq!(expect_config(config).kv_pool, None),
            Ok(_) => panic!("`{raw}` must not parse as a byte or token count"),
            Err(err) => assert!(err.0.contains("--vram-kv-pool-bytes"), "{}", err.0),
        }
    }
}

/// ADR 0045 (AC 6): `--vram-allow-expert-cache-below-floor` is Flash-Next's;
/// off by default, on from the flag or the environment, and refused on the
/// 27B, which has no expert cache.
#[test]
fn the_expert_cache_floor_opt_in_parses_and_refuses_the_27b() {
    let default = config(&[]);
    assert!(!default.allow_expert_cache_below_floor);
    let flag = config(&["--vram-allow-expert-cache-below-floor"]);
    assert!(flag.allow_expert_cache_below_floor);
    for (raw, on) in [("true", true), ("1", true), ("off", false)] {
        let env = move |key: &str| (key == "IGNIS_VRAM_ALLOW_EXPERT_CACHE_BELOW_FLOOR").then(|| raw.to_owned());
        assert_eq!(expect_config(resolve(&[], env).expect(raw)).allow_expert_cache_below_floor, on, "{raw}");
    }
    let err = resolve(&[], env_map(&[("IGNIS_VRAM_ALLOW_EXPERT_CACHE_BELOW_FLOOR", "maybe")])).expect_err("bad");
    assert!(err.0.contains("IGNIS_VRAM_ALLOW_EXPERT_CACHE_BELOW_FLOOR"), "{}", err.0);
    assert!(served_model_for(&flag, ModelFamily::FlashNext).is_ok());
    assert!(served_model_for(&default, ModelFamily::Qwen38_27b).is_ok());
    let err = served_model_for(&flag, ModelFamily::Qwen38_27b).expect_err("no expert cache");
    assert!(err.0.contains("--vram-allow-expert-cache-below-floor") && err.0.contains("27B"), "{}", err.0);
}

#[test]
fn the_help_says_lanes_share_the_pool_and_names_the_new_spellings() {
    // AC 40 (P1): a lane no longer holds its own whole context.
    let lanes = help_block("--spec-decode-lanes");
    assert!(!lanes.contains("whole context"), "{lanes}");
    assert!(lanes.contains("sharing the KV pool"), "{lanes}");
    assert!(lanes.contains("min(524,288 tokens, lanes x `max_context`)") && lanes.contains("retained slot"), "{lanes}");
    let pool = help_block("--vram-kv-pool-bytes");
    assert!(pool.contains("Ktok") && pool.contains("524,288"), "{pool}");
    let floor = help_block("--vram-allow-expert-cache-below-floor");
    assert!(floor.contains("IGNIS_VRAM_ALLOW_EXPERT_CACHE_BELOW_FLOOR") && floor.contains("12 GiB"), "{floor}");
}

// ── the VRAM budget (GitHub #210, ADR 0030) ──────────────────────────────

const GIB: u64 = 1024 * 1024 * 1024;

#[test]
fn no_memory_flag_derives_the_budget_with_a_one_and_a_half_gib_headroom() {
    assert_eq!(config(&[]).vram, VramMode::Derived { headroom_bytes: 1536 * 1024 * 1024 });
}

#[test]
fn help_names_the_vram_headroom_default() {
    let block = help_block("--vram-headroom-bytes");
    assert!(block.contains("default: 1536M"), "{block}");
}

#[test]
fn the_headroom_and_the_budget_resolve_from_flags_and_env() {
    assert_eq!(config(&["--vram-headroom-bytes", "2G"]).vram, VramMode::Derived { headroom_bytes: 2 * GIB });
    let env = env_map(&[("IGNIS_VRAM_HEADROOM_BYTES", "512M")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).vram, VramMode::Derived { headroom_bytes: 512 << 20 });
    assert_eq!(
        config(&["--vram-budget-bytes", "28G"]).vram,
        VramMode::Explicit { budget_bytes: 28 * GIB, allow_oversubscription: false }
    );
    let env = env_map(&[("IGNIS_VRAM_BUDGET_BYTES", "30G"), ("IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION", "true")]);
    assert_eq!(
        expect_config(resolve(&[], env).expect("resolve")).vram,
        VramMode::Explicit { budget_bytes: 30 * GIB, allow_oversubscription: true }
    );
    assert_eq!(
        config(&["--vram-budget-bytes", "30G", "--vram-allow-oversubscription"]).vram,
        VramMode::Explicit { budget_bytes: 30 * GIB, allow_oversubscription: true }
    );
}

#[test]
fn a_headroom_with_a_budget_is_refused_from_any_mix_of_flags_and_env() {
    let cases: [(Vec<String>, &'static [(&'static str, &'static str)]); 4] = [
        (args(&["--vram-headroom-bytes", "1G", "--vram-budget-bytes", "28G"]), &[]),
        (args(&["--vram-budget-bytes", "28G"]), &[("IGNIS_VRAM_HEADROOM_BYTES", "1G")]),
        (args(&["--vram-headroom-bytes", "1G"]), &[("IGNIS_VRAM_BUDGET_BYTES", "28G")]),
        (vec![], &[("IGNIS_VRAM_HEADROOM_BYTES", "1G"), ("IGNIS_VRAM_BUDGET_BYTES", "28G")]),
    ];
    for (a, env) in cases {
        let err = resolve(&a, env_map(env)).expect_err("mutually exclusive").0;
        assert!(err.contains("HEADROOM") || err.contains("headroom"), "{err}");
        assert!(err.contains("BUDGET") || err.contains("budget"), "{err}");
        assert!(err.contains("mutually exclusive"), "{err}");
    }
}

#[test]
fn oversubscription_without_a_budget_is_refused() {
    let cases: [(Vec<String>, &'static [(&'static str, &'static str)]); 3] = [
        (args(&["--vram-allow-oversubscription"]), &[]),
        (vec![], &[("IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION", "1")]),
        (args(&["--vram-allow-oversubscription", "--vram-headroom-bytes", "2G"]), &[]),
    ];
    for (a, env) in cases {
        let err = resolve(&a, env_map(env)).expect_err("needs a budget").0;
        assert!(err.contains("OVERSUBSCRIPTION") || err.contains("oversubscription"), "{err}");
        assert!(err.contains("--vram-budget-bytes"), "{err}");
    }
}

#[test]
fn malformed_vram_values_are_usage_errors() {
    for (flag, raw) in [("--vram-headroom-bytes", "lots"), ("--vram-budget-bytes", "28 GiB please"), ("--vram-budget-bytes", "0")] {
        assert!(refused(&[flag, raw]).contains(flag), "{flag} {raw}");
    }
    let env = env_map(&[("IGNIS_VRAM_BUDGET_BYTES", "28G"), ("IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION", "maybe")]);
    assert!(resolve(&[], env).expect_err("not a bool").0.contains("IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION"));
}

#[test]
fn help_documents_the_vram_flags() {
    let text = help();
    for name in [
        "--vram-headroom-bytes",
        "IGNIS_VRAM_HEADROOM_BYTES",
        "--vram-budget-bytes",
        "IGNIS_VRAM_BUDGET_BYTES",
        "--vram-allow-oversubscription",
        "IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION",
    ] {
        assert!(text.contains(name), "help must document {name}:\n{text}");
    }
}

// ── the KV-RAM host tier byte budget (P4-07, GitHub #125) ────────────────

#[test]
fn an_explicit_host_pool_budget_overrides_the_default() {
    assert_eq!(config(&["--reuse-kv-host-pool-bytes", "512M"]).host_pool_bytes, 512 << 20);
    let env = env_map(&[("IGNIS_REUSE_KV_HOST_POOL_BYTES", "1G")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).host_pool_bytes, GIB);
}

#[test]
fn the_host_pool_budget_flag_wins_over_its_env_var() {
    let env = env_map(&[("IGNIS_REUSE_KV_HOST_POOL_BYTES", "1G")]);
    let config = expect_config(resolve(&args(&["--reuse-kv-host-pool-bytes", "256M"]), env).expect("resolve"));
    assert_eq!(config.host_pool_bytes, 256 << 20, "flag must win over env");
}

#[test]
fn a_zero_host_pool_budget_is_accepted_and_disables_the_tier() {
    // Unlike an empty string (an unset value), `0` is an explicit, legal
    // operator choice: no host tier at all.
    assert_eq!(config(&["--reuse-kv-host-pool-bytes", "0"]).host_pool_bytes, 0);
}

#[test]
fn a_malformed_host_pool_budget_is_a_usage_error() {
    assert!(refused(&["--reuse-kv-host-pool-bytes", "not-a-size"]).contains("--reuse-kv-host-pool-bytes"));
}

#[test]
fn help_lists_the_host_pool_budget_flag() {
    assert!(help().contains("--reuse-kv-host-pool-bytes"));
}

// ── GitHub #186: cross-request state reuse (ADR 0029) ────────────────────

/// (device, host) retained slots.
fn retained_slots_of(config: &Config) -> (u32, u32) {
    (config.retained_device_slots, config.retained_host_slots)
}

#[test]
fn prompt_reuse_is_on_unless_the_operator_turns_it_off() {
    let on = config(&[]);
    assert!(on.prompt_reuse, "on by default (ADR 0029)");
    assert_eq!(
        retained_slots_of(&on),
        (0, 2 * ignis_core::N_DECODE_LANES as u32),
        "no device slot and two host slots per decode lane by default (GitHub #281)"
    );
    let off = config(&["--reuse-prompt", "off"]);
    assert!(!off.prompt_reuse);
    assert_eq!(retained_slots_of(&off), (0, 0), "reuse off reserves no slot of either kind");
    let env = env_map(&[("IGNIS_REUSE_PROMPT", "off")]);
    assert!(!expect_config(resolve(&[], env).expect("resolve")).prompt_reuse, "the env var turns it off too");
    let env = env_map(&[("IGNIS_REUSE_PROMPT", "off")]);
    assert!(expect_config(resolve(&args(&["--reuse-prompt", "on"]), env).expect("resolve")).prompt_reuse, "flag must win over env");
}

#[test]
fn a_malformed_prompt_reuse_value_is_a_usage_error() {
    let err = refused(&["--reuse-prompt", "maybe"]);
    assert!(err.contains("--reuse-prompt") && err.contains("maybe"), "{err}");
}

#[test]
fn retained_device_and_host_slots_are_configurable_by_flag_or_env() {
    // GitHub #281: each kind is sized on its own, its default kept when only
    // the other is named.
    let host_default = 2 * ignis_core::N_DECODE_LANES as u32;
    assert_eq!(retained_slots_of(&config(&["--reuse-retained-device", "3"])), (3, host_default));
    assert_eq!(retained_slots_of(&config(&["--reuse-retained-host", "5"])), (0, 5));
    // A card with VRAM to spare keeps every image on the device.
    assert_eq!(retained_slots_of(&config(&["--reuse-retained-device", "16", "--reuse-retained-host", "0"])), (16, 0));
    let env = env_map(&[("IGNIS_REUSE_RETAINED_DEVICE", "12"), ("IGNIS_REUSE_RETAINED_HOST", "7")]);
    assert_eq!(retained_slots_of(&expect_config(resolve(&[], env).expect("resolve"))), (12, 7));
    let env = env_map(&[("IGNIS_REUSE_RETAINED_DEVICE", "12"), ("IGNIS_REUSE_RETAINED_HOST", "7")]);
    let a = args(&["--reuse-retained-device", "1", "--reuse-retained-host", "2"]);
    assert_eq!(retained_slots_of(&expect_config(resolve(&a, env).expect("resolve"))), (1, 2), "the flag wins over its env var");
    for flag in ["--reuse-retained-device", "--reuse-retained-host"] {
        assert!(refused(&[flag, "8G"]).contains(flag), "{flag}");
    }
}

#[test]
fn the_config_records_whether_the_host_retained_count_was_named() {
    // Spec flash-next/05: an unnamed count is the 27B's default here and
    // Flash-Next's own once the artifact names its model, so the config
    // keeps which of the two it is.
    assert!(!config(&[]).retained_host_named);
    assert!(!config(&["--reuse-retained-device", "2"]).retained_host_named, "the device count names nothing of the host's");
    assert!(config(&["--reuse-retained-host", "16"]).retained_host_named, "named at the 27B's default value is still named");
    let env = env_map(&[("IGNIS_REUSE_RETAINED_HOST", "3")]);
    assert!(expect_config(resolve(&[], env).expect("resolve")).retained_host_named, "the env var names it too");
}

#[test]
fn the_retained_interactive_ttl_is_configurable_and_needs_reuse_on() {
    assert_eq!(config(&[]).retained_interactive_ttl_secs, 300, "five minutes by default");
    assert_eq!(config(&["--reuse-retained-interactive-ttl", "60"]).retained_interactive_ttl_secs, 60);
    let env = env_map(&[("IGNIS_REUSE_RETAINED_INTERACTIVE_TTL", "0")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).retained_interactive_ttl_secs, 0, "zero is a legal choice");
    assert!(refused(&["--reuse-retained-interactive-ttl", "5m"]).contains("--reuse-retained-interactive-ttl"));
    let err = refused(&["--reuse-prompt", "off", "--reuse-retained-interactive-ttl", "60"]);
    assert!(err.contains("--reuse-prompt"), "names what it needs: {err}");
}

#[test]
fn retained_slots_with_reuse_off_are_accepted_for_live_siblings() {
    // With reuse off nothing outlives a request, but live siblings still
    // share a head when slots are given for it — the engine before #186.
    let device = config(&["--reuse-prompt", "off", "--reuse-retained-device", "4"]);
    assert!(!device.prompt_reuse);
    assert_eq!(retained_slots_of(&device), (4, 0), "the kind not named stays at zero");
    assert_eq!(retained_slots_of(&config(&["--reuse-prompt", "off", "--reuse-retained-host", "3"])), (0, 3));
}

// ── GitHub #209: instruction-message policies ────────────────────────────

#[test]
fn instruction_policies_default_to_merge_and_inplace() {
    assert_eq!(
        config(&[]).instruction_policy,
        InstructionPolicy { system: SystemMessagePolicy::Merge, developer: DeveloperMessagePolicy::Inplace }
    );
}

#[test]
fn every_instruction_policy_value_is_named_by_flag_or_env_and_the_flag_wins() {
    for system in SystemMessagePolicy::ALL {
        assert_eq!(config(&["--server-system-message-policy", system.as_str()]).instruction_policy.system, system);
        let env = move |key: &str| (key == "IGNIS_SERVER_SYSTEM_MESSAGE_POLICY").then(|| system.as_str().to_owned());
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).instruction_policy.system, system);
    }
    for developer in DeveloperMessagePolicy::ALL {
        assert_eq!(config(&["--server-developer-message-policy", developer.as_str()]).instruction_policy.developer, developer);
        let env = move |key: &str| (key == "IGNIS_SERVER_DEVELOPER_MESSAGE_POLICY").then(|| developer.as_str().to_owned());
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).instruction_policy.developer, developer);
    }
    let env = env_map(&[("IGNIS_SERVER_SYSTEM_MESSAGE_POLICY", "strict"), ("IGNIS_SERVER_DEVELOPER_MESSAGE_POLICY", "reject")]);
    let a = args(&["--server-system-message-policy", "merge", "--server-developer-message-policy", "into-system"]);
    assert_eq!(
        expect_config(resolve(&a, env).expect("resolve")).instruction_policy,
        InstructionPolicy { system: SystemMessagePolicy::Merge, developer: DeveloperMessagePolicy::IntoSystem },
        "flags win over env"
    );
}

#[test]
fn an_unknown_instruction_policy_names_the_allowed_values() {
    let err = refused(&["--server-system-message-policy", "inplace"]);
    assert!(err.contains("--server-system-message-policy"), "{err}");
    assert!(err.contains("`inplace`"), "names the value: {err}");
    assert!(err.contains("merge, strict"), "names the allowed ones: {err}");

    let err = resolve(&[], env_map(&[("IGNIS_SERVER_DEVELOPER_MESSAGE_POLICY", "drop")])).expect_err("not a developer policy").0;
    assert!(err.contains("`drop`"), "names the value: {err}");
    assert!(err.contains("IGNIS_SERVER_DEVELOPER_MESSAGE_POLICY"), "names the env var it came from: {err}");
    assert!(err.contains("inplace, into-system, after-system, one-after-system, reject"), "names the allowed ones: {err}");

    let padded = config(&["--server-developer-message-policy", " One-After-System "]);
    assert_eq!(padded.instruction_policy.developer, DeveloperMessagePolicy::OneAfterSystem, "case and padding do not matter");
}

// ── the request timeout (GitHub #95) ─────────────────────────────────────

#[test]
fn the_request_timeout_env_var_wins_over_the_default() {
    let env = env_map(&[("IGNIS_SERVER_REQUEST_TIMEOUT", "90")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).request_timeout_secs, 90);
}

#[test]
fn the_request_timeout_flag_wins_over_its_env_var() {
    let env = env_map(&[("IGNIS_SERVER_REQUEST_TIMEOUT", "90")]);
    let config = expect_config(resolve(&args(&["--server-request-timeout", "45"]), env).expect("resolve"));
    assert_eq!(config.request_timeout_secs, 45, "flag must win over env");
}

#[test]
fn a_request_timeout_outside_one_to_the_ceiling_is_a_usage_error() {
    let err = refused(&["--server-request-timeout", "0"]);
    assert!(err.contains("1..=3600") && err.contains("got 0"), "{err}");
    assert!(refused(&["--server-request-timeout", "soon"]).contains("--server-request-timeout"));
    let err = refused(&["--server-request-timeout", "3601"]);
    assert!(err.contains("3600") && err.contains("3601"), "the ceiling and the value: {err}");
    let err = resolve(&[], env_map(&[("IGNIS_SERVER_REQUEST_TIMEOUT", "0")])).expect_err("must reject").0;
    assert!(err.contains("IGNIS_SERVER_REQUEST_TIMEOUT"), "{err}");
}

#[test]
fn help_lists_the_request_timeout_flag() {
    assert!(help().contains("--server-request-timeout"));
}

// ── the model switch (spec model-switch/01) ──────────────────────────────

#[test]
fn the_switch_drain_timeout_defaults_to_thirty_seconds_takes_zero_and_has_a_ceiling() {
    assert_eq!(config(&[]).switch_drain_timeout_secs, 30);
    let env = env_map(&[("IGNIS_SWITCH_DRAIN_TIMEOUT", "90")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).switch_drain_timeout_secs, 90);
    let env = env_map(&[("IGNIS_SWITCH_DRAIN_TIMEOUT", "90")]);
    let zero = expect_config(resolve(&args(&["--switch-drain-timeout", "0"]), env).expect("resolve"));
    assert_eq!(zero.switch_drain_timeout_secs, 0, "flag must win over env, and 0 cuts at once");
    let err = refused(&["--switch-drain-timeout", "3601"]);
    assert!(err.contains("3600") && err.contains("3601"), "{err}");
}

#[test]
fn a_model_field_may_switch_models_unless_the_operator_says_otherwise() {
    assert!(config(&[]).allow_model_switch);
    for (raw, on) in [("true", true), ("1", true), ("ON", true), ("false", false), ("0", false), ("off", false)] {
        assert_eq!(config(&["--switch-allow-implicit", raw]).allow_model_switch, on, "{raw}");
    }
    let env = env_map(&[("IGNIS_SWITCH_ALLOW_IMPLICIT", "off")]);
    assert!(!expect_config(resolve(&[], env).expect("resolve")).allow_model_switch);
    assert!(refused(&["--switch-allow-implicit", "maybe"]).contains("maybe"));
}

#[test]
fn known_models_come_from_repeated_flags_or_the_env_var_and_a_flag_replaces_the_env() {
    let a = args(&[
        "--switch-known-models",
        "qwen3.8-flash-next=F:/models/Qwen3.8-Flash-Next.ninfer",
        "--switch-known-models",
        " qwen3.8-27b = F:\\models\\Qwen3.8-27B.ninfer ",
    ]);
    let config = expect_config(resolve(&a, no_env).expect("resolve"));
    assert_eq!(
        config.known_models,
        BTreeMap::from([
            ("qwen3.8-flash-next".to_owned(), PathBuf::from("F:/models/Qwen3.8-Flash-Next.ninfer")),
            ("qwen3.8-27b".to_owned(), PathBuf::from("F:\\models\\Qwen3.8-27B.ninfer")),
        ])
    );
    let env = env_map(&[("IGNIS_SWITCH_KNOWN_MODELS", "qwen3.8-27b=F:/27b.ninfer;qwen3.8-flash-next=F:/fn.ninfer")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("resolve")).known_models.len(), 2);
    let flag = expect_config(resolve(&args(&["--switch-known-models", "other=F:/other.ninfer"]), &env).expect("resolve"));
    assert_eq!(flag.known_models.keys().collect::<Vec<_>>(), ["other"], "a flag overrides its env var whole");
    let err = refused(&["--switch-known-models", "m=F:/a.ninfer", "--switch-known-models", "m=F:/b.ninfer"]);
    assert!(err.contains("`m`") && err.contains("twice"), "{err}");
}

// ── speculation as a load option (P5-02, GitHub #150) ────────────────────

#[test]
fn spec_dflash2_with_a_window_in_range_parses() {
    for n in 1..=7u32 {
        let config = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", &n.to_string()]);
        assert_eq!(config.speculation, Some(Speculation::new(SpeculativeBackend::Dflash2, n).unwrap()));
    }
}

#[test]
fn a_draft_window_outside_1_to_7_is_refused_naming_the_flag_and_the_value() {
    for raw in ["0", "8", "15", "-1", "seven"] {
        let err = refused(&["--spec-backend", "dflash2", "--spec-draft-tokens", raw]);
        assert!(err.contains("--spec-draft-tokens") && err.contains(raw), "{err}");
    }
    assert!(refused(&["--spec-backend", "dflash2", "--spec-draft-tokens", "8"]).contains("1..=7"));
}

#[test]
fn an_unknown_speculative_backend_is_refused_naming_every_backend() {
    let err = refused(&["--spec-backend", "eagle", "--spec-draft-tokens", "3"]);
    assert!(err.contains("--spec-backend") && err.contains("eagle"), "{err}");
    assert!(err.contains("dflash2") && err.contains("mtp"), "{err}");
}

/// GitHub #307: Flash-Next's head takes a default window, a forced one, a
/// row budget, and `off`.
#[test]
fn spec_mtp_spec_off_and_draft_rows_parse() {
    assert_eq!(
        config(&["--spec-backend", "mtp"]).speculation,
        Some(Speculation::new(SpeculativeBackend::Mtp, FLASH_NEXT_DEFAULT_DRAFT_TOKENS).unwrap())
    );
    let forced = config(&["--spec-backend", "mtp", "--spec-draft-tokens", "3", "--spec-draft-rows", "6"]);
    assert_eq!(forced.speculation, Some(Speculation::new(SpeculativeBackend::Mtp, 3).unwrap()));
    assert_eq!(forced.draft_rows, Some(6));
    let off = config(&["--spec-backend", "off"]);
    assert!(off.speculation_off && off.speculation.is_none());
    let default = config(&[]);
    assert!(!default.speculation_off && default.draft_rows.is_none());
    for raw in ["1", "9", "rows"] {
        let err = refused(&["--spec-draft-rows", raw]);
        assert!(err.contains("--spec-draft-rows") && err.contains(raw), "{err}");
    }
}

#[test]
fn spec_without_a_draft_window_is_refused() {
    let err = refused(&["--spec-backend", "dflash2"]);
    assert!(err.contains("--spec-draft-tokens") && err.contains("1..=7"), "{err}");
}

#[test]
fn a_draft_window_without_spec_is_refused_rather_than_ignored() {
    assert!(refused(&["--spec-draft-tokens", "7"]).contains("--spec-backend"));
}

#[test]
fn the_draft_head_defaults_to_full_and_takes_shortlist_by_flag_or_env() {
    let full = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7"]);
    assert_eq!(full.speculation.map(|s| s.proposal_head()), Some(ProposalHead::Full));
    let short = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7", "--spec-draft-head", "shortlist"]);
    assert_eq!(short.speculation.map(|s| s.proposal_head()), Some(ProposalHead::Shortlist));
    assert_eq!(short.speculation.map(|s| s.draft_tokens()), Some(7));
    let env = env_map(&[("IGNIS_SPEC_BACKEND", "dflash2"), ("IGNIS_SPEC_DRAFT_TOKENS", "7"), ("IGNIS_SPEC_DRAFT_HEAD", "shortlist")]);
    let from_env = expect_config(resolve(&[], env).expect("resolve"));
    assert_eq!(from_env.speculation.map(|s| s.proposal_head()), Some(ProposalHead::Shortlist));
}

#[test]
fn a_draft_head_without_spec_or_with_an_unknown_name_is_refused() {
    let err = refused(&["--spec-draft-head", "shortlist"]);
    assert!(err.contains("--spec-draft-head") && err.contains("--spec-backend"), "{err}");
    let err = refused(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7", "--spec-draft-head", "tiny"]);
    assert!(err.contains("--spec-draft-head") && err.contains("tiny"), "{err}");
}

#[test]
fn mtp_takes_no_proposal_head_and_spec_off_takes_no_draft_options() {
    let err = refused(&["--spec-backend", "mtp", "--spec-draft-head", "shortlist"]);
    assert!(err.contains("--spec-draft-head") && err.contains("mtp"), "{err}");
    assert!(resolve(&args(&["--spec-backend", "mtp", "--spec-draft-tokens", "2", "--spec-draft-head", "shortlist"]), no_env).is_err());
    assert!(resolve(&args(&["--spec-backend", "mtp", "--spec-draft-head", "full"]), no_env).is_ok(), "naming the only head is harmless");

    let err = refused(&["--spec-backend", "off", "--spec-draft-tokens", "7"]);
    assert!(err.contains("--spec-draft-tokens") && err.contains("off"), "{err}");
    assert!(refused(&["--spec-backend", "off", "--spec-draft-head", "shortlist"]).contains("--spec-draft-head"));
    assert!(resolve(&[], env_map(&[("IGNIS_SPEC_BACKEND", "off"), ("IGNIS_SPEC_DRAFT_TOKENS", "3")])).is_err());
    assert!(config(&["--spec-backend", "off"]).speculation.is_none());
}

#[test]
fn the_speculation_env_vars_apply_and_the_flags_win_over_them() {
    let env = env_map(&[("IGNIS_SPEC_BACKEND", "dflash2"), ("IGNIS_SPEC_DRAFT_TOKENS", "3")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).speculation.map(|s| s.draft_tokens()), Some(3));
    let env = env_map(&[("IGNIS_SPEC_BACKEND", "dflash2"), ("IGNIS_SPEC_DRAFT_TOKENS", "3")]);
    let flag = expect_config(resolve(&args(&["--spec-draft-tokens", "7"]), env).expect("resolve"));
    assert_eq!(flag.speculation.map(|s| s.draft_tokens()), Some(7), "flag must win over env");
}

#[test]
fn help_lists_the_speculation_flags() {
    let text = help();
    assert!(text.contains("--spec-backend") && text.contains("--spec-draft-tokens"), "{text}");
}

// ── rope scaling as a load option (GitHub #227) ──────────────────────────

#[test]
fn rope_scaling_is_off_by_default() {
    let config = config(&[]);
    assert_eq!(config.rope_scaling, RopeScaling::NONE);
    assert!(!config.rope_scaling.is_yarn());
}

#[test]
fn the_rope_scaling_flag_and_env_carry_the_reference_grammar() {
    let yarn = config(&["--model-rope-scaling", "yarn:4,t=0.25"]);
    assert!(yarn.rope_scaling.is_yarn());
    assert_eq!(yarn.rope_scaling.factor(), 4.0);
    assert_eq!(yarn.rope_scaling.temperature(), 0.25);
    let env = env_map(&[("IGNIS_MODEL_ROPE_SCALING", "yarn:2")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).rope_scaling.factor(), 2.0);
    let env = env_map(&[("IGNIS_MODEL_ROPE_SCALING", "yarn:2")]);
    let none = expect_config(resolve(&args(&["--model-rope-scaling", "none"]), env).expect("resolve"));
    assert_eq!(none.rope_scaling, RopeScaling::NONE, "flag must win over env");
}

#[test]
fn a_bad_rope_scaling_is_a_startup_error_naming_the_flag() {
    assert!(refused(&["--model-rope-scaling", "yarn:0.5"]).contains("--model-rope-scaling"));
    assert!(refused(&["--model-rope-scaling", "linear"]).contains("--model-rope-scaling"));
}

// ── the loaded model's family (spec flash-next/04, GitHub #302) ──────────

#[test]
fn flash_next_refuses_speculation_and_vision_at_start_naming_itself() {
    let spec = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7"]);
    let err = served_model_for(&spec, ModelFamily::FlashNext).expect_err("the 27B's drafter").0;
    assert!(err.contains("--spec-backend dflash2") && err.contains("Qwen3.8-Flash-Next"), "{err}");
    assert!(err.contains("drafts with mtp"), "{err}");
    let mtp = config(&["--spec-backend", "mtp", "--spec-draft-rows", "6"]);
    assert!(served_model_for(&mtp, ModelFamily::FlashNext).is_ok(), "Flash-Next drafts with mtp");
    // `--spec-draft-rows` is Flash-Next's, so the 27B is refused for it
    // first, in declaration order; without it, for the backend.
    let mtp_alone = config(&["--spec-backend", "mtp"]);
    let err = served_model_for(&mtp_alone, ModelFamily::Qwen38_27b).expect_err("Flash-Next's head").0;
    assert!(err.contains("--spec-backend mtp") && err.contains("drafts with dflash2"), "{err}");
    let vision = config(&["--vision-enabled"]);
    let err = served_model_for(&vision, ModelFamily::FlashNext).expect_err("no vision").0;
    assert!(err.contains("--vision-enabled") && err.contains("Qwen3.8-Flash-Next"), "{err}");
}

/// GitHub #228: the 27B's GQA envelope bounds `--model-max-context` per KV
/// format; Flash-Next's QSA is not bound by it.
#[test]
fn max_context_past_the_gqa_envelope_is_refused_for_the_27b_only() {
    let cfg = |ctx: &str, fmt: &str| config(&["--model-max-context", ctx, "--model-kv-format", fmt, "--vram-kv-pool-bytes", "64G"]);
    assert!(served_model_for(&cfg("524288", "bf16"), ModelFamily::Qwen38_27b).is_ok());
    let err = served_model_for(&cfg("524289", "bf16"), ModelFamily::Qwen38_27b).expect_err("past linear").0;
    assert!(err.contains("--model-max-context") && err.contains("524288") && err.contains("bf16"), "{err}");
    assert!(served_model_for(&cfg("1048576", "hq-e8-2b"), ModelFamily::Qwen38_27b).is_ok());
    let err = served_model_for(&cfg("1048577", "hq-e8-2b"), ModelFamily::Qwen38_27b).expect_err("past absolute").0;
    assert!(err.contains("1048576"), "{err}");
    assert!(served_model_for(&cfg("1048577", "hq-e8-2b"), ModelFamily::FlashNext).is_ok());
}

/// GitHub #306: `--spec-decode-lanes` is Flash-Next's, 1 to the engine's 8.
#[test]
fn decode_lanes_parse_bound_and_refuse_the_27b() {
    assert_eq!(config(&[]).decode_lanes, None);
    let flag = config(&["--spec-decode-lanes", "1"]);
    assert_eq!(flag.decode_lanes, Some(1));
    let env = expect_config(resolve(&[], env_map(&[("IGNIS_SPEC_DECODE_LANES", "8")])).expect("resolve"));
    assert_eq!(env.decode_lanes, Some(8));
    for raw in ["0", "9", "-1", "many"] {
        let err = refused(&["--spec-decode-lanes", raw]);
        assert!(err.contains("--spec-decode-lanes") && err.contains(raw), "{err}");
    }
    assert!(served_model_for(&flag, ModelFamily::FlashNext).is_ok());
    let err = served_model_for(&flag, ModelFamily::Qwen38_27b).expect_err("a fixed lane count").0;
    assert!(err.contains("--spec-decode-lanes 1") && err.contains("Qwen3.8-27B does not take it"), "{err}");
}

/// `--ngram-hot-bytes` (GitHub #306): unnamed is `None`, the 1 GiB default
/// the load always had; a size or `auto`, from the flag or the environment,
/// the flag winning; anything else is refused, and so is any value on the
/// 27B, which has no n-gram table.
#[test]
fn ngram_hot_bytes_parse_and_refuse_the_27b() {
    use ignis_core::ngram_table::HotBudget;
    let default = config(&[]);
    assert_eq!(default.ngram_hot_bytes, None);
    for (raw, budget) in [("4G", HotBudget::Bytes(4 << 30)), ("auto", HotBudget::Auto), ("0", HotBudget::Bytes(0))] {
        assert_eq!(config(&["--ngram-hot-bytes", raw]).ngram_hot_bytes, Some(budget), "{raw}");
    }
    let env = env_map(&[("IGNIS_NGRAM_HOT_BYTES", "auto")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("resolve")).ngram_hot_bytes, Some(HotBudget::Auto));
    let flag = expect_config(resolve(&args(&["--ngram-hot-bytes", "512M"]), &env).expect("resolve"));
    assert_eq!(flag.ngram_hot_bytes, Some(HotBudget::Bytes(512 << 20)), "the flag wins over the env");
    for raw in ["lots", "-1", "auto2"] {
        let err = refused(&["--ngram-hot-bytes", raw]);
        assert!(err.contains("--ngram-hot-bytes") && err.contains("auto") && err.contains(raw), "{err}");
    }
    assert!(resolve(&args(&["--ngram-hot-bytes"]), no_env).is_err());
    assert!(served_model_for(&flag, ModelFamily::FlashNext).is_ok());
    assert!(served_model_for(&default, ModelFamily::Qwen38_27b).is_ok());
    let err = served_model_for(&flag, ModelFamily::Qwen38_27b).expect_err("no n-gram table").0;
    assert!(err.contains("--ngram-hot-bytes 512M") && err.contains("Qwen3.8-27B"), "{err}");
}

#[test]
fn help_lists_the_ngram_hot_bytes_flag_with_its_default_and_auto() {
    let block = help_block("--ngram-hot-bytes");
    for needle in ["IGNIS_NGRAM_HOT_BYTES", "1 GiB", "auto", "whole"] {
        assert!(block.contains(needle), "{needle} missing: {block}");
    }
}

#[test]
fn the_27b_takes_every_start_option_it_took_before() {
    let config = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "7", "--vision-enabled"]);
    assert_eq!(served_model_for(&config, ModelFamily::Qwen38_27b), Ok(DEFAULT_MODEL.to_owned()));
}

/// With no `--model-id`, a load is served under its own model's id: the
/// artifact's family decides, not the 27B's default.
#[test]
fn an_unnamed_served_id_is_the_loaded_models_own() {
    let plain = config(&[]);
    assert_eq!(served_model_for(&plain, ModelFamily::FlashNext), Ok("qwen3.8-flash-next".to_owned()));
    assert_eq!(served_model_for(&plain, ModelFamily::Qwen38_27b), Ok("qwen3.8-27b".to_owned()));
}

/// A served id that names the other model is refused at start, whether the
/// flag or the environment named it: the artifact decides the model, and a
/// client must never be told it talks to one while the other answers. Any
/// other id is the operator's to choose.
#[test]
fn a_served_id_naming_the_other_model_is_refused_at_start() {
    let named = |id: &'static str| config(&["--model-id", id]);
    let err = served_model_for(&named("qwen3.8-flash-next"), ModelFamily::Qwen38_27b).expect_err("27B artifact").0;
    assert!(err.contains("qwen3.8-flash-next") && err.contains("Qwen3.8-27B"), "{err}");
    let err = served_model_for(&named("qwen3.8-27b"), ModelFamily::FlashNext).expect_err("Flash-Next artifact").0;
    assert!(err.contains("qwen3.8-27b") && err.contains("Qwen3.8-Flash-Next"), "{err}");
    let from_env = expect_config(resolve(&[], env_map(&[("IGNIS_MODEL_ID", "qwen3.8-flash-next")])).expect("resolve"));
    assert!(served_model_for(&from_env, ModelFamily::Qwen38_27b).is_err(), "the env form too");
    assert_eq!(served_model_for(&named("my-id"), ModelFamily::FlashNext), Ok("my-id".to_owned()));
    assert_eq!(served_model_for(&named("qwen3.8-27b"), ModelFamily::Qwen38_27b), Ok("qwen3.8-27b".to_owned()));
}

// ── family scope (spec config-v2/01 §Family scope) ───────────────────────

/// The problem ADR 0046 was written for: the 27B's production arena and
/// Flash-Next's own, both named on one command line, each applied to its
/// own model at start and across a switch.
#[test]
fn each_family_keeps_its_own_resource_shape() {
    let config = config(&[
        "--reuse-kv-host-pool-bytes",
        "8G",
        "--qwen38flashnext-reuse-kv-host-pool-bytes",
        "1G",
        "--qwen38-model-max-context",
        "131072",
    ]);
    assert_eq!(config.host_pool_bytes, 8 * GIB, "before a family is known, the general value");
    let qwen = config.for_family(ModelFamily::Qwen38_27b).unwrap();
    assert_eq!((qwen.host_pool_bytes, qwen.max_context), (8 * GIB, 131_072));
    let flash = config.for_family(ModelFamily::FlashNext).unwrap();
    assert_eq!((flash.host_pool_bytes, flash.max_context), (GIB, DEFAULT_MAX_CONTEXT));
    let (switched, dropped) = fit_to_family(&qwen, ModelFamily::FlashNext).unwrap();
    assert_eq!(switched.host_pool_bytes, GIB, "a switch takes the target family's own");
    assert!(dropped.is_empty());
    let (back, _) = fit_to_family(&switched, ModelFamily::Qwen38_27b).unwrap();
    assert_eq!(back.host_pool_bytes, 8 * GIB, "and back again");
    assert_eq!(back.basis.family(), Some(ModelFamily::Qwen38_27b));
}

/// A family-scoped value is checked when it is written, not at the first
/// load of its family.
#[test]
fn a_bad_family_scoped_value_is_refused_at_start() {
    let err = refused(&["--qwen38flashnext-model-max-context", "lots"]);
    assert!(err.contains("--qwen38flashnext-model-max-context") && err.contains("lots"), "{err}");
    let env = env_map(&[("IGNIS_QWEN38_MODEL_PREFILL_CHUNK", "1000")]);
    assert!(resolve(&[], env).expect_err("unaligned").0.contains("IGNIS_QWEN38_MODEL_PREFILL_CHUNK"));
    let err = refused(&["--qwen38flashnext-spec-backend", "dflash2"]);
    assert!(err.contains("--spec-draft-tokens"), "the family's own rules apply to its scoped values: {err}");
    // A field only one family has carries no scope to give it.
    assert!(refused(&["--qwen38-ngram-hot-bytes", "1G"]).contains("unrecognized"));
}

/// Spec config-v2/01 AC 8 (and model-switch/01): a switch drops the values
/// named for the other model instead of refusing them, says which, and
/// leaves the rest — a shared value past the target's bound is still refused.
#[test]
fn a_switch_drops_the_values_only_the_other_model_takes_and_names_them() {
    let started = config(&["--vision-enabled", "--vision-max-tokens", "8192", "--spec-backend", "dflash2", "--spec-draft-tokens", "7"]);
    let (fitted, dropped) = fit_to_family(&started, ModelFamily::FlashNext).unwrap();
    assert_eq!(dropped, ["--vision-enabled", "--vision-max-tokens", "--spec-backend", "--spec-draft-tokens"]);
    assert!(fitted.vision.is_none() && fitted.speculation.is_none());
    assert_eq!(fitted.model, "qwen3.8-flash-next");

    let flash = config(&["--spec-decode-lanes", "1", "--ngram-hot-bytes", "4G", "--vram-allow-expert-cache-below-floor", "--spec-draft-rows", "6"]);
    let (fitted, dropped) = fit_to_family(&flash, ModelFamily::Qwen38_27b).unwrap();
    assert_eq!(dropped, ["--vram-allow-expert-cache-below-floor", "--spec-draft-rows", "--spec-decode-lanes", "--ngram-hot-bytes"]);
    assert!(fitted.decode_lanes.is_none() && fitted.ngram_hot_bytes.is_none() && !fitted.allow_expert_cache_below_floor);

    let (kept, dropped) = fit_to_family(&config(&["--vision-enabled"]), ModelFamily::Qwen38_27b).unwrap();
    assert!(kept.vision.is_some() && dropped.is_empty(), "nothing dropped for the model it was named for");

    let wide = config(&["--model-max-context", "600000", "--model-kv-format", "bf16"]);
    assert!(fit_to_family(&wide, ModelFamily::Qwen38_27b).is_err(), "a bound is not a drop");
    assert!(fit_to_family(&wide, ModelFamily::FlashNext).is_ok());
}

/// A patch wins over every start source, is validated as a start would be,
/// and keeps the family the config was fitted for.
#[test]
fn a_patch_wins_over_the_command_line_and_is_validated_whole() {
    let started = config(&["--server-request-timeout", "30", "--reuse-kv-host-pool-bytes", "8G"]).for_family(ModelFamily::FlashNext).unwrap();
    let mut patch = source::Layer::default();
    let timeout = schema::field("server", "request_timeout").unwrap();
    patch.set(timeout, None, source::Candidate { raw: "90".into(), spelling: "server.request_timeout".into() }).unwrap();
    let patched = started.with_patch(&patch).unwrap();
    assert_eq!(patched.request_timeout_secs, 90);
    assert_eq!(patched.host_pool_bytes, 8 * GIB, "the rest unchanged");
    assert_eq!(patched.basis.family(), Some(ModelFamily::FlashNext));
    assert_eq!(patched.model, "qwen3.8-flash-next");

    let lanes = schema::field("spec", "decode_lanes").unwrap();
    patch.set(lanes, None, source::Candidate { raw: "99".into(), spelling: "spec.decode_lanes".into() }).unwrap();
    let err = started.with_patch(&patch).unwrap_err().0;
    assert!(err.contains("spec.decode_lanes") && err.contains("99"), "{err}");
}

// ── vision as a load option (GitHub #177) ────────────────────────────────

#[test]
fn vision_is_off_by_default() {
    assert_eq!(config(&[]).vision, None);
}

#[test]
fn the_vision_flag_loads_the_default_envelope() {
    let config = config(&["--vision-enabled"]);
    assert_eq!(config.vision, Some(Vision::default()));
    assert_eq!(config.vision.unwrap().max_tokens(), DEFAULT_VISION_MAX_TOKENS);
}

#[test]
fn the_vision_envelope_can_be_lowered() {
    assert_eq!(config(&["--vision-enabled", "--vision-max-tokens", "8192"]).vision.map(|v| v.max_tokens()), Some(8192));
}

#[test]
fn a_vision_envelope_without_vision_is_refused_rather_than_ignored() {
    assert!(refused(&["--vision-max-tokens", "8192"]).contains("--vision-enabled"));
    assert!(resolve(&[], env_map(&[("IGNIS_VISION_MAX_TOKENS", "8192")])).is_err(), "the env form too");
}

#[test]
fn the_embedding_pool_defaults_to_one_envelope_wide_item() {
    // GitHub #243 must not move the VRAM plan for an operator who does not
    // ask: the default pool is the reservation vision always took.
    let vision = config(&["--vision-enabled"]).vision.expect("vision on");
    assert_eq!(vision.pool_bytes(262_144), vision.output_transient_bytes(262_144));
}

#[test]
fn the_embedding_pool_can_be_widened() {
    let vision = config(&["--vision-enabled", "--vision-embedding-pool-mib", "1280"]).vision.expect("vision on");
    assert_eq!(vision.pool_bytes(262_144), 1280 * 1024 * 1024);
    let env = env_map(&[("IGNIS_VISION_ENABLED", "1"), ("IGNIS_VISION_EMBEDDING_POOL_MIB", "640")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).vision.unwrap().pool_bytes(262_144), 640 * 1024 * 1024);
}

#[test]
fn an_embedding_pool_below_the_envelope_is_raised_not_refused() {
    // An operator lowering the pool should not have to recompute the
    // envelope's bytes to keep the load working: the floor is applied, and
    // the plan reports what was actually reserved.
    let vision = config(&["--vision-enabled", "--vision-embedding-pool-mib", "1"]).vision.expect("vision on");
    assert_eq!(vision.requested_pool_bytes(), 1024 * 1024);
    assert_eq!(vision.pool_bytes(262_144), vision.output_transient_bytes(262_144));
}

#[test]
fn an_embedding_pool_without_vision_is_refused_rather_than_ignored() {
    assert!(refused(&["--vision-embedding-pool-mib", "640"]).contains("--vision-enabled"));
    assert!(resolve(&[], env_map(&[("IGNIS_VISION_EMBEDDING_POOL_MIB", "640")])).is_err(), "the env form too");
}

#[test]
fn an_embedding_pool_outside_the_range_is_refused_naming_it() {
    for raw in ["0", "65537", "-1", "lots"] {
        let err = refused(&["--vision-enabled", "--vision-embedding-pool-mib", raw]);
        assert!(err.contains("--vision-embedding-pool-mib") && err.contains(raw), "{err}");
    }
}

#[test]
fn a_vision_envelope_outside_the_range_is_refused_naming_it() {
    for raw in ["0", "1048577", "-1", "lots"] {
        let err = refused(&["--vision-enabled", "--vision-max-tokens", raw]);
        assert!(err.contains("--vision-max-tokens") && err.contains(raw), "{err}");
    }
    assert!(refused(&["--vision-enabled", "--vision-max-tokens", "1048577"]).contains("1048576"));
}

#[test]
fn the_vision_env_vars_apply_and_the_flags_win_over_them() {
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_VISION_MAX_TOKENS", "4096")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).vision.map(|v| v.max_tokens()), Some(4096));
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_VISION_MAX_TOKENS", "4096")]);
    let flag = expect_config(resolve(&args(&["--vision-max-tokens", "2048"]), env).expect("resolve"));
    assert_eq!(flag.vision.map(|v| v.max_tokens()), Some(2048), "flag must win over env");
    assert_eq!(expect_config(resolve(&[], env_map(&[("IGNIS_VISION_ENABLED", "false")])).expect("resolve")).vision, None);
    let err = resolve(&[], env_map(&[("IGNIS_VISION_ENABLED", "maybe")])).expect_err("bad bool").0;
    assert!(err.contains("IGNIS_VISION_ENABLED") && err.contains("maybe"), "{err}");
}

#[test]
fn help_lists_the_vision_flags() {
    let text = help();
    assert!(text.contains("--vision-enabled ") && text.contains("--vision-max-tokens"), "{text}");
    // GitHub #248: the flag shrinks an image, and says so.
    assert!(text.contains("is shrunk to fit"), "{text}");
}

/// GitHub #195: the two are independent load options again.
#[test]
fn vision_and_dflash2_resolve_together_as_two_independent_load_options() {
    let both = config(&["--vision-enabled", "--spec-backend", "dflash2", "--spec-draft-tokens", "4"]);
    assert_eq!(both.vision, Some(Vision::default()));
    assert_eq!(both.speculation, Speculation::new(SpeculativeBackend::Dflash2, 4).ok());
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_SPEC_BACKEND", "dflash2"), ("IGNIS_SPEC_DRAFT_TOKENS", "4")]);
    let from_env = expect_config(resolve(&[], env).expect("the env form too"));
    assert_eq!((from_env.vision, from_env.speculation), (both.vision, both.speculation));
    let vision_only = config(&["--vision-enabled"]);
    assert_eq!((vision_only.vision, vision_only.speculation), (both.vision, None));
    let spec_only = config(&["--spec-backend", "dflash2", "--spec-draft-tokens", "4"]);
    assert_eq!((spec_only.vision, spec_only.speculation), (None, both.speculation));
}

// ── media acquisition (GitHub #179) ──────────────────────────────────────

#[test]
fn media_defaults_to_no_private_network_and_a_one_gib_cache() {
    assert_eq!(config(&["--vision-enabled"]).media, MediaOptions { allow_private_network: false, cache_bytes: 1024 << 20 });
    assert_eq!(config(&[]).media, MediaOptions::default());
}

#[test]
fn media_flags_set_the_private_network_opt_in_and_the_cache() {
    let flags = config(&["--vision-enabled", "--media-allow-private-network", "--media-cache-mib", "0"]);
    assert_eq!(flags.media, MediaOptions { allow_private_network: true, cache_bytes: 0 });
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK", "true"), ("IGNIS_MEDIA_CACHE_MIB", "64")]);
    let mixed = expect_config(resolve(&args(&["--media-cache-mib", "128"]), env).expect("resolve"));
    assert_eq!(mixed.media, MediaOptions { allow_private_network: true, cache_bytes: 128 << 20 });
}

#[test]
fn media_flags_without_vision_are_refused_rather_than_ignored() {
    for a in [&["--media-allow-private-network"][..], &["--media-cache-mib", "10"]] {
        let err = refused(a);
        assert!(err.contains(a[0]) && err.contains("--vision-enabled"), "{err}");
    }
}

#[test]
fn a_media_cache_outside_the_range_is_refused_naming_it() {
    for raw in ["65537", "-1", "lots"] {
        let err = refused(&["--vision-enabled", "--media-cache-mib", raw]);
        assert!(err.contains("--media-cache-mib") && err.contains(raw), "{err}");
    }
    let env = env_map(&[("IGNIS_VISION_ENABLED", "true"), ("IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK", "maybe")]);
    assert!(resolve(&[], env).expect_err("bad bool").0.contains("maybe"));
}

#[test]
fn help_lists_the_media_flags() {
    let text = help();
    assert!(text.contains("--media-allow-private-network") && text.contains("--media-cache-mib"), "{text}");
}

// ── the Playground (GitHub #163, ADR 0026) ───────────────────────────────

#[test]
fn the_playground_is_on_by_default_and_off_when_asked() {
    assert!(config(&[]).ui);
    assert!(config(&["--server-ui"]).ui);
    assert!(!config(&["--server-ui", "false"]).ui);
}

#[test]
fn a_bare_bool_flag_leaves_the_next_flag_alone() {
    // The next argument is parsed as a flag of its own, not as a value.
    for (argv, want) in [(&["--server-ui", "--server-bind", "b"][..], true), (&["--server-ui", "off", "--server-bind", "b"], false)] {
        let config = config(argv);
        assert_eq!(config.ui, want, "{argv:?}");
        assert_eq!(config.bind, "b", "{argv:?}");
    }
}

#[test]
fn the_ui_env_var_applies_and_the_flags_win_over_it() {
    assert!(!expect_config(resolve(&[], env_map(&[("IGNIS_SERVER_UI", "false")])).expect("resolve")).ui);
    assert!(expect_config(resolve(&args(&["--server-ui"]), env_map(&[("IGNIS_SERVER_UI", "off")])).expect("resolve")).ui);
    assert!(!expect_config(resolve(&args(&["--server-ui", "false"]), env_map(&[("IGNIS_SERVER_UI", "true")])).expect("resolve")).ui);
    let err = resolve(&[], env_map(&[("IGNIS_SERVER_UI", "maybe")])).expect_err("not a boolean").0;
    assert!(err.contains("IGNIS_SERVER_UI"), "{err}");
}

#[test]
fn help_lists_the_ui_flag() {
    assert!(help().contains("--server-ui"));
}

// ── fetching a missing model (GitHub #234) ───────────────────────────────

#[test]
fn model_downloads_are_on_by_default_and_land_in_models() {
    let config = config(&[]);
    assert!(config.model_download);
    assert_eq!(config.model_download_path, PathBuf::from(DEFAULT_MODEL_DOWNLOAD_PATH));
}

#[test]
fn the_model_download_flag_stands_bare_or_takes_a_value() {
    for (argv, want) in [(&["--download-enabled", "--server-bind", "b"][..], true), (&["--download-enabled", "false", "--server-bind", "b"], false)] {
        let config = config(argv);
        assert_eq!(config.model_download, want, "{argv:?}");
        assert_eq!(config.bind, "b", "{argv:?}");
    }
}

#[test]
fn the_model_download_env_vars_apply_and_the_flags_win_over_them() {
    let env = env_map(&[("IGNIS_DOWNLOAD_ENABLED", "false"), ("IGNIS_DOWNLOAD_PATH", "/srv/models")]);
    let from_env = expect_config(resolve(&[], env).expect("resolve"));
    assert!(!from_env.model_download);
    assert_eq!(from_env.model_download_path, PathBuf::from("/srv/models"));
    assert!(expect_config(resolve(&args(&["--download-enabled"]), env_map(&[("IGNIS_DOWNLOAD_ENABLED", "off")])).expect("resolve")).model_download);
    let env = env_map(&[("IGNIS_DOWNLOAD_PATH", "/srv/models")]);
    let flag = expect_config(resolve(&args(&["--download-path", "D:/weights"]), env).expect("resolve"));
    assert_eq!(flag.model_download_path, PathBuf::from("D:/weights"));
    let err = resolve(&[], env_map(&[("IGNIS_DOWNLOAD_ENABLED", "maybe")])).expect_err("not a boolean").0;
    assert!(err.contains("IGNIS_DOWNLOAD_ENABLED"), "{err}");
}

#[test]
fn an_empty_model_download_path_is_the_default_not_the_working_directory() {
    // An unset-looking env var (`IGNIS_DOWNLOAD_PATH=`) must not turn the
    // destination into `.`, where a 19 GB file would land wherever the
    // server happened to be started from.
    let config = expect_config(resolve(&[], env_map(&[("IGNIS_DOWNLOAD_PATH", "")])).expect("resolve"));
    assert_eq!(config.model_download_path, PathBuf::from(DEFAULT_MODEL_DOWNLOAD_PATH));
}

#[test]
fn help_lists_both_model_download_flags() {
    let text = help();
    assert!(text.contains("--download-enabled") && text.contains("--download-path"), "{text}");
}

// ── Prometheus metrics (GitHub #89, ADR 0017) ────────────────────────────

#[test]
fn metrics_are_off_by_default_and_on_their_own_listener_when_on() {
    assert_eq!(config(&[]).metrics, None);
    assert_eq!(config(&["--server-metrics"]).metrics, Some(DEFAULT_METRICS_BIND.to_owned()));
    assert_ne!(DEFAULT_METRICS_BIND, DEFAULT_BIND);
}

#[test]
fn metrics_bind_moves_the_metrics_listener_in_either_order() {
    for argv in [
        ["--server-metrics", "--server-metrics-bind", "127.0.0.1:9100"],
        ["--server-metrics-bind", "127.0.0.1:9100", "--server-metrics"],
    ] {
        assert_eq!(config(&argv).metrics.as_deref(), Some("127.0.0.1:9100"), "{argv:?}");
    }
}

#[test]
fn metrics_bind_without_metrics_or_value_or_on_the_api_address_is_refused() {
    assert!(refused(&["--server-metrics-bind", "127.0.0.1:9100"]).contains("--server-metrics"));
    assert!(resolve(&args(&["--server-metrics", "--server-metrics-bind"]), no_env).is_err());
    let err = refused(&["--server-metrics", "--server-bind", "127.0.0.1:7000", "--server-metrics-bind", "127.0.0.1:7000"]);
    assert!(err.contains("--server-bind"), "{err}");
}

/// ADR 0046: every field has its env var, metrics included — the drift
/// (`IGNIS_PROMETHEUS` with no flag, `--metrics` with no env var) the one
/// declaration per field exists to end. The old names stay without effect.
#[test]
fn metrics_has_its_env_var_now_and_no_alias() {
    let env = env_map(&[("IGNIS_SERVER_METRICS", "true"), ("IGNIS_SERVER_METRICS_BIND", "127.0.0.1:9100")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).metrics.as_deref(), Some("127.0.0.1:9100"));
    for name in ["IGNIS_METRICS", "IGNIS_PROMETHEUS", "IGNIS_METRICS_BIND"] {
        let env = move |key: &str| (key == name).then(|| "127.0.0.1:9100".to_owned());
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).metrics, None, "{name}");
    }
    for alias in ["-M", "--prometheus", "--server-metrics=true", "--metrics"] {
        assert!(resolve(&args(&[alias]), no_env).is_err(), "`{alias}` is not a metrics alias");
    }
}

#[test]
fn help_lists_the_metrics_flags() {
    let text = help();
    assert!(text.contains("--server-metrics ") && text.contains("--server-metrics-bind"), "{text}");
}

// ── the API key ──────────────────────────────────────────────────────────

#[test]
fn the_api_key_is_unset_by_default_and_an_empty_value_stays_unset() {
    assert_eq!(config(&[]).api_key, None);
    assert_eq!(expect_config(resolve(&[], env_map(&[("IGNIS_SERVER_API_KEY", "")])).expect("resolve")).api_key, None);
}

#[test]
fn the_api_key_resolves_flag_over_env() {
    let env = env_map(&[("IGNIS_SERVER_API_KEY", "from-env")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("resolve")).api_key, Some(ApiKeySetting::Fixed(ApiKey::new("from-env"))));
    let flag = expect_config(resolve(&args(&["--server-api-key", "from-flag"]), &env).expect("resolve"));
    assert_eq!(flag.api_key, Some(ApiKeySetting::Fixed(ApiKey::new("from-flag"))), "flag must win over env");
}

#[test]
fn auto_asks_for_a_generated_key_from_the_flag_or_the_env() {
    assert_eq!(config(&["--server-api-key", "auto"]).api_key, Some(ApiKeySetting::Generate));
    let env = env_map(&[("IGNIS_SERVER_API_KEY", "auto")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).api_key, Some(ApiKeySetting::Generate));
}

#[test]
fn a_generated_key_is_fresh_and_long() {
    let a = ApiKey::generate().expect("random source");
    let b = ApiKey::generate().expect("random source");
    assert_ne!(a, b);
    let hex = a.as_str().strip_prefix("sk-ignis-").expect("prefix");
    assert_eq!(hex.len(), 64);
    assert!(hex.chars().all(|c| c.is_ascii_hexdigit()), "{hex}");
}

#[test]
fn an_api_key_matches_only_itself_and_never_prints() {
    let key = ApiKey::new("sk-secret");
    assert!(key.matches("sk-secret"));
    for other in ["", "sk-secre", "sk-secret!", "sk-Secret"] {
        assert!(!key.matches(other), "{other:?}");
    }
    let config = config(&["--server-api-key", "sk-secret"]);
    assert!(!format!("{config:?}").contains("sk-secret"));
    assert!(!format!("{:?}", config.for_family(ModelFamily::Qwen38_27b).unwrap()).contains("sk-secret"));
}

#[test]
fn help_lists_the_api_key_flag() {
    let text = help();
    assert!(text.contains("--server-api-key") && text.contains("IGNIS_SERVER_API_KEY"), "{text}");
}

// ── exposure (ADR 0028) ──────────────────────────────────────────────────

#[test]
fn nothing_is_exposed_by_default() {
    let config = config(&[]);
    assert_eq!(config.expose, None);
    assert_eq!(config.api_key, None, "no exposure, no forced key");
}

#[test]
fn expose_resolves_flag_over_env() {
    assert_eq!(config(&["--server-expose", "cloudflare-quick"]).expose, Some(Expose::CloudflareQuick));
    let env = env_map(&[("IGNIS_SERVER_EXPOSE", "cloudflare-quick")]);
    assert_eq!(expect_config(resolve(&[], &env).expect("resolve")).expose, Some(Expose::CloudflareQuick));
    let err = resolve(&args(&["--server-expose", "nope"]), &env).expect_err("the flag wins, and is checked");
    assert!(err.0.contains("nope"), "{err}");
}

#[test]
fn an_unknown_expose_mode_is_a_usage_error() {
    let err = refused(&["--server-expose", "ngrok"]);
    assert!(err.contains("--server-expose") && err.contains("ngrok") && err.contains("cloudflare-quick"), "{err}");
}

#[test]
fn exposing_without_a_key_generates_one() {
    assert_eq!(config(&["--server-expose", "cloudflare-quick"]).api_key, Some(ApiKeySetting::Generate));
    // An empty key is no key.
    let env = env_map(&[("IGNIS_SERVER_API_KEY", ""), ("IGNIS_SERVER_EXPOSE", "cloudflare-quick")]);
    assert_eq!(expect_config(resolve(&[], env).expect("resolve")).api_key, Some(ApiKeySetting::Generate));
}

#[test]
fn exposing_keeps_the_key_the_operator_chose() {
    let mine = config(&["--server-expose", "cloudflare-quick", "--server-api-key", "sk-mine"]);
    assert_eq!(mine.api_key, Some(ApiKeySetting::Fixed(ApiKey::new("sk-mine"))));
    let env = env_map(&[("IGNIS_SERVER_API_KEY", "sk-env")]);
    let from_env = expect_config(resolve(&args(&["--server-expose", "cloudflare-quick"]), env).expect("resolve"));
    assert_eq!(from_env.api_key, Some(ApiKeySetting::Fixed(ApiKey::new("sk-env"))));
}

#[test]
fn help_lists_the_expose_flag() {
    let text = help();
    assert!(text.contains("--server-expose") && text.contains("cloudflare-quick"), "{text}");
}
