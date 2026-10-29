//! The CUDA builds turn on the server's `cuda` feature in a form cargo applies
//! to the server (GitHub #408). ignis-server has a dev-dependency on itself
//! (its integration tests' `fake-speech`), and cargo reads
//! `--features ignis-server/cuda` as a feature of that dependency, which a
//! build of the binaries never compiles: v0.9.0 and v0.9.1 shipped a server
//! that served the CPU mock while reporting ready, and no step failed. The
//! rule held here is the general one: no build recipe names a feature as
//! `<member>/<feature>` for a workspace member that has a dependency of its
//! own name. The release's check of the shipped binary
//! (`mk/cuda-build-check.sh`) looks for the warning only a build without
//! `cuda` carries, so the last test holds that text to the server's.

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

/// Whether a manifest has a dependency keyed by its own package name: cargo
/// allows that only as a dev-dependency, and it is what captures
/// `--features <name>/<feature>`.
fn depends_on_itself(manifest: &str) -> bool {
    let Some(name) = package_name(manifest) else {
        return false;
    };
    manifest.lines().map(str::trim_start).any(|line| {
        line.strip_prefix(name.as_str()).is_some_and(|rest| rest.trim_start().starts_with('='))
    })
}

/// Every feature a recipe's text passes to `--features`, one per name.
fn features_named(text: &str) -> Vec<String> {
    let mut named = Vec::new();
    for (at, flag) in text.match_indices("--features") {
        let rest = &text[at + flag.len()..];
        let rest = rest.strip_prefix('=').unwrap_or(rest).trim_start_matches([' ', '\t']);
        let value: String = rest.chars().take_while(|c| !c.is_whitespace() && *c != ')').collect();
        for feature in value.split(',') {
            let feature = feature.trim_matches(|c| c == '"' || c == '\'' || c == '`');
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
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let mut files = vec![repo.join("Containerfile"), repo.join("Makefile"), repo.join("local.mk.example")];
    walk(&repo.join("mk"), &mut files);
    walk(&repo.join(".github/workflows"), &mut files);
    files.retain(|path| path.is_file());
    files
}

#[test]
fn no_build_recipe_names_a_feature_through_a_member_that_depends_on_itself() {
    let repo = repo();
    let mut captured = Vec::new();
    for entry in std::fs::read_dir(repo.join("crates")).unwrap().flatten() {
        let Ok(manifest) = std::fs::read_to_string(entry.path().join("Cargo.toml")) else {
            continue;
        };
        if depends_on_itself(&manifest) {
            captured.push(package_name(&manifest).unwrap());
        }
    }
    let recipes = recipes(&repo);
    assert!(recipes.iter().any(|path| path.ends_with("Containerfile")), "no recipe found under {}", repo.display());
    let mut wrong = Vec::new();
    for path in &recipes {
        for feature in features_named(&read(path)) {
            for member in &captured {
                if feature.starts_with(&format!("{member}/")) {
                    wrong.push(format!("{}: --features {feature}", path.strip_prefix(&repo).unwrap_or(path).display()));
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
    // The two spellings v0.9.0 and v0.9.1 were built with.
    let containerfile = "RUN cargo build --release --locked --target x86_64-unknown-linux-gnu \\\n        --workspace --features ignis-server/cuda \\\n    && strip x\n";
    let makefile = "WORKSPACE_FEATURES := $(if $(filter 1,$(CUDA)),--features ignis-server/cuda)\n";
    assert_eq!(features_named(containerfile), ["ignis-server/cuda"]);
    assert_eq!(features_named(makefile), ["ignis-server/cuda"]);
    assert_eq!(features_named("cargo check --features=cuda,ignis-core/logits-capture"), ["cuda", "ignis-core/logits-capture"]);

    let manifest = "[package]\nname = \"ignis-server\"\n\n[dev-dependencies]\nignis-server = { path = \".\", features = [\"fake-speech\"] }\n";
    assert_eq!(package_name(manifest).as_deref(), Some("ignis-server"));
    assert!(depends_on_itself(manifest));
    assert!(!depends_on_itself("[package]\nname = \"ignis-server\"\n\n[dependencies]\nignis-server-x = \"1\"\n"));
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
