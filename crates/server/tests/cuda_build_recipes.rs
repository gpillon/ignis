//! The CUDA builds turn on the server's `cuda` feature in a form cargo applies
//! to the server (GitHub #408). ignis-server has a dev-dependency on itself
//! (its integration tests' `fake-speech`), and cargo reads
//! `--features ignis-server/cuda` as a feature of that dependency, which a
//! build of the binaries never compiles: v0.9.0 and v0.9.1 shipped a server
//! that served the CPU mock while reporting ready, and no step failed. The
//! rule held here is the general one: no build recipe names a feature as
//! `<member>/<feature>` for a workspace member that has a dependency of its
//! own name. The release's check of the shipped binary
//! (`mk/cuda-build-check.sh`, tested by its own selftest) looks for the
//! warning only a build without `cuda` carries, so the last test holds that
//! text to the server's.

use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &Path) -> String {
    // A checkout that turns LF into CRLF still holds the same recipe.
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).replace("\r\n", "\n")
}

/// The `name` of a manifest's `[package]` table.
fn package_name(manifest: &str) -> Option<String> {
    let mut in_package = false;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            in_package = line == "[package]";
        } else if in_package && line.starts_with("name") {
            let (_, value) = line.split_once('=')?;
            return Some(value.trim().trim_matches('"').to_string());
        }
    }
    None
}

/// Whether a manifest has a dependency named after its own package, in any
/// of TOML's spellings: `name = { .. }` or `name.path = ".."` inside a
/// dependencies table, or a `[..dependencies.name]` table of its own. Cargo
/// allows that only as a dev-dependency, and it is what captures
/// `--features <name>/<feature>`.
fn depends_on_itself(manifest: &str) -> bool {
    let Some(name) = package_name(manifest) else {
        return false;
    };
    let mut in_dependencies = false;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            let table = line.trim_matches(['[', ']']);
            if table.ends_with(&format!("dependencies.{name}")) {
                return true;
            }
            in_dependencies = table.ends_with("dependencies");
        } else if in_dependencies
            && line.strip_prefix(name.as_str()).is_some_and(|rest| rest.trim_start().starts_with(['=', '.']))
        {
            return true;
        }
    }
    false
}

/// Whether a shell token passes the features flag: `Some("")` when the value
/// is the next token, `Some(value)` for `--features=value`. The flag starts
/// the token, or follows a comma or a parenthesis, as in Make's
/// `$(if $(c),--features x)`.
fn features_flag(token: &str) -> Option<&str> {
    for flag in ["--features", "-F"] {
        let Some(at) = token.rfind(flag) else {
            continue;
        };
        if at > 0 && !token[..at].ends_with([',', '(']) {
            continue;
        }
        let rest = &token[at + flag.len()..];
        if rest.is_empty() {
            return Some("");
        }
        if let Some(value) = rest.strip_prefix('=') {
            return Some(value);
        }
    }
    None
}

/// Every feature a recipe's text passes to `--features` or `-F`, one per
/// name. Comment lines are skipped, a line continued with `\` or a backtick
/// is read with the next one, and a quoted list may be separated by commas or
/// spaces.
fn features_named(text: &str) -> Vec<String> {
    let code: Vec<&str> = text.lines().filter(|line| !line.trim_start().starts_with('#')).collect();
    let code = code.join("\n").replace("\\\n", " ").replace("`\n", " ");
    let tokens: Vec<&str> = code.split_whitespace().collect();
    let mut named = Vec::new();
    let mut at = 0;
    while at < tokens.len() {
        let Some(inline) = features_flag(tokens[at]) else {
            at += 1;
            continue;
        };
        at += 1;
        let first = if inline.is_empty() {
            let Some(next) = tokens.get(at) else {
                break;
            };
            at += 1;
            *next
        } else {
            inline
        };
        let mut value = first.to_string();
        // A quoted list runs to the token that closes the quote.
        if let Some(quote) = first.chars().next().filter(|c| *c == '"' || *c == '\'') {
            while (value.len() < 2 || !value.ends_with(quote)) && at < tokens.len() {
                value.push(' ');
                value.push_str(tokens[at]);
                at += 1;
            }
        }
        for feature in value.split([',', ' ']) {
            let feature = feature.trim_matches(|c| matches!(c, '"' | '\'' | '`' | ')'));
            if !feature.is_empty() {
                named.push(feature.to_string());
            }
        }
    }
    named
}

/// The files that build ignis or check a build: the Containerfile, the
/// Makefile and everything under `mk/`, the workflows. `scripts/` drives only
/// local runs, and the release workflow skips a push that touches nothing else.
fn recipes(repo: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let mut files = vec![repo.join("Containerfile"), repo.join("Makefile")];
    walk(&repo.join("mk"), &mut files);
    walk(&repo.join(".github/workflows"), &mut files);
    for path in &files[..2] {
        assert!(path.is_file(), "{} is missing", path.display());
    }
    files
}

#[test]
fn no_build_recipe_names_a_feature_through_a_member_that_depends_on_itself() {
    let repo = repo();
    let mut self_dependent = Vec::new();
    for entry in std::fs::read_dir(repo.join("crates")).unwrap().flatten() {
        let Ok(manifest) = std::fs::read_to_string(entry.path().join("Cargo.toml")) else {
            continue;
        };
        if depends_on_itself(&manifest) {
            self_dependent.push(package_name(&manifest).unwrap());
        }
    }
    // The member that broke v0.9.0 and v0.9.1; without it the rule below
    // would hold of nothing. If ignis-server stops depending on itself, this
    // assertion goes with it.
    assert!(
        self_dependent.iter().any(|name| name == "ignis-server"),
        "crates/server/Cargo.toml no longer reads as depending on itself; self-dependent members found: {self_dependent:?}"
    );
    let mut wrong = Vec::new();
    for path in recipes(&repo) {
        for feature in features_named(&read(&path)) {
            for member in &self_dependent {
                if feature.starts_with(&format!("{member}/")) {
                    wrong.push(format!("{}: --features {feature}", path.strip_prefix(&repo).unwrap_or(&path).display()));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "cargo applies these to a dev-dependency a build never compiles, not to the member: {}. \
         Name the member's own feature (`-p <member> --features <feature>`, or `--workspace --features <feature>`).",
        wrong.join("; ")
    );
}

#[test]
fn the_scan_finds_the_form_that_shipped_without_cuda() {
    // The spellings v0.9.0 and v0.9.1 were built with: the Containerfile's
    // build stage and the release workflow's Windows build.
    let containerfile = "RUN cargo build --release --locked --target x86_64-unknown-linux-gnu \\\n        --workspace --features ignis-server/cuda \\\n    && strip x\n";
    let windows = "        run: |\n          cargo build --release --locked --target x86_64-pc-windows-msvc `\n            --workspace --features ignis-server/cuda\n";
    assert_eq!(features_named(containerfile), ["ignis-server/cuda"]);
    assert_eq!(features_named(windows), ["ignis-server/cuda"]);
    // The Makefile's workspace features, the other forms cargo takes, and a
    // comment that names the spelling without passing it.
    assert_eq!(
        features_named("WORKSPACE_FEATURES := $(if $(filter 1,$(CUDA)),--features ignis-server/cuda)\n"),
        ["ignis-server/cuda"]
    );
    assert_eq!(features_named("cargo check --features=cuda,ignis-core/logits-capture"), ["cuda", "ignis-core/logits-capture"]);
    assert_eq!(features_named("cargo build -F \"cuda ignis-server/cuda\" --workspace"), ["cuda", "ignis-server/cuda"]);
    assert_eq!(features_named("cargo build --features \\\n  ignis-server/cuda"), ["ignis-server/cuda"]);
    assert_eq!(features_named("$(if $(filter 1,$(CUDA)),--features=ignis-server/cuda)"), ["ignis-server/cuda"]);
    assert!(features_named("cargo build --no-default-features --workspace").is_empty());
    assert!(features_named("# never `--features ignis-server/cuda` here\n").is_empty());

    let manifest = "[package]\nname = \"ignis-server\"\n\n[dev-dependencies]\nignis-server = { path = \".\", features = [\"fake-speech\"] }\n";
    assert_eq!(package_name(manifest).as_deref(), Some("ignis-server"));
    assert!(depends_on_itself(manifest));
    assert!(depends_on_itself("[package]\nname = \"ignis-server\"\n\n[dev-dependencies.ignis-server]\npath = \".\"\n"));
    assert!(depends_on_itself("[package]\nname = \"ignis-server\"\n\n[dev-dependencies]\nignis-server.path = \".\"\n"));
    assert!(!depends_on_itself("[package]\nname = \"ignis-server\"\n\n[dependencies]\nignis-server-x = \"1\"\n"));
    assert!(!depends_on_itself("[package]\nname = \"ignis-server\"\n\n[features]\nignis-server = []\n"));
}

#[test]
fn the_release_check_looks_for_the_warning_only_a_build_without_cuda_carries() {
    let repo = repo();
    let script = read(&repo.join("mk/cuda-build-check.sh"));
    let marker = script
        .lines()
        .find_map(|line| line.strip_prefix("marker='")?.strip_suffix('\''))
        .expect("mk/cuda-build-check.sh declares marker='...'");
    let load = read(&repo.join("crates/server/src/load.rs"));
    assert!(load.contains(marker), "crates/server/src/load.rs no longer says {marker:?}: the release check would pass any build");
    for recipe in ["Containerfile", ".github/workflows/release.yml"] {
        assert!(read(&repo.join(recipe)).contains("bash mk/cuda-build-check.sh "), "{recipe} no longer runs mk/cuda-build-check.sh");
    }
}
