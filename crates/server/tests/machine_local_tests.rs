//! The machine-local test targets (GitHub #709): the CPU-only, not
//! `#[ignore]`d tests that read a fixture only the owner's machine holds (an
//! artifact, a tokenizer or chat template under `F:/ai/...`, routing traces, a
//! Python venv). Where the fixture is absent they print a skip and return, and
//! a skip counts as green (`docs/agents/testing.md`, "The gate"), so on CI
//! they always skip. `make test-machine-local` runs them for real once per
//! release, from the one list `mk/machine-local-tests.txt` holds.
//!
//! The rule this test applies is mechanical: a test target under
//! `crates/*/tests/` that is not `cuda`-gated and prints a skip is either on
//! the list or exempt through a line that names its reason (a fixture CI
//! provides, a CPU CI runners do not have, tests that are all `#[ignore]`d),
//! and every line names a target that exists and still prints a skip.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const LIST: &str = "mk/machine-local-tests.txt";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A file, as read on any checkout: a Windows one turns LF into CRLF.
fn read(path: &str) -> String {
    let path = repo().join(path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).replace("\r\n", "\n")
}

/// Whether a target is compiled only with the `cuda` feature: an inner
/// `#![cfg(...)]` that names it, alone or inside `all(...)`. `cargo test`
/// never builds such a target. The feature is written escaped here, so this
/// file is no GPU change by the rule in `docs/agents/testing.md`, "Done
/// means", which reads it unescaped.
fn cuda_gated(source: &str) -> bool {
    source.lines().map(str::trim_start).any(|line| line.starts_with("#![cfg(") && line.contains("feature = \"cuda\""))
}

/// Whether a target prints a skip: a print macro (`print!`, `println!`,
/// `eprint!`, `eprintln!`) whose string starts with "skip", in any case
/// (`skip:`, `SKIP:`, `skipped:`), on the macro's line or the next.
fn prints_a_skip(source: &str) -> bool {
    source.match_indices("!(").any(|(at, _)| {
        let before = &source[..at];
        if !(before.ends_with("print") || before.ends_with("println")) {
            return false;
        }
        let argument = source[at + 2..].trim_start();
        argument.strip_prefix('"').is_some_and(|text| text.get(..4).is_some_and(|word| word.eq_ignore_ascii_case("skip")))
    })
}

/// Every test target under `crates/*/tests/`, by its path from the repo root.
fn targets() -> BTreeMap<String, String> {
    let mut targets = BTreeMap::new();
    let crates = repo().join("crates");
    for krate in std::fs::read_dir(&crates).expect("crates/") {
        let krate = krate.expect("a crate").file_name().into_string().expect("a UTF-8 crate name");
        let Ok(files) = std::fs::read_dir(crates.join(&krate).join("tests")) else { continue };
        for file in files {
            let name = file.expect("a test file").file_name().into_string().expect("a UTF-8 file name");
            if name.ends_with(".rs") {
                let path = format!("crates/{krate}/tests/{name}");
                targets.insert(path.clone(), read(&path));
            }
        }
    }
    targets
}

/// The list's lines: a target's path, or `exempt <path>: <reason>`; blank
/// lines and `#` comments aside.
struct List {
    listed: Vec<String>,
    exempt: Vec<(String, String)>,
}

fn parse(text: &str) -> List {
    let mut list = List { listed: Vec::new(), exempt: Vec::new() };
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty() && !line.starts_with('#')) {
        match line.strip_prefix("exempt ") {
            Some(rest) => {
                let (path, reason) = rest.split_once(':').unwrap_or_else(|| panic!("{LIST}: `{line}` names no reason after a colon"));
                list.exempt.push((path.trim().to_owned(), reason.trim().to_owned()));
            }
            None => list.listed.push(line.to_owned()),
        }
    }
    list
}

/// A target's name, as `cargo test --test <name>` selects it.
fn stem(path: &str) -> &str {
    path.rsplit('/').next().and_then(|file| file.strip_suffix(".rs")).expect("a .rs path")
}

#[test]
fn every_machine_local_target_is_listed_or_exempt_and_every_line_names_one() {
    let list = parse(&read(LIST));
    let targets = targets();
    let machine_local: BTreeSet<&str> =
        targets.iter().filter(|(_, source)| !cuda_gated(source) && prints_a_skip(source)).map(|(path, _)| path.as_str()).collect();

    let mut named = BTreeSet::new();
    for path in list.listed.iter().chain(list.exempt.iter().map(|(path, _)| path)) {
        assert!(named.insert(path.as_str()), "{LIST} names {path} twice");
        assert!(targets.contains_key(path), "{LIST} names {path}, which is not a test target any more");
    }
    for (path, reason) in &list.exempt {
        assert!(!reason.is_empty(), "{LIST} exempts {path} without a reason");
    }

    let unlisted: Vec<&str> = machine_local.iter().copied().filter(|path| !named.contains(path)).collect();
    assert!(
        unlisted.is_empty(),
        "these test targets print a skip and are not in {LIST}: add each one that skips on a fixture only the owner's machine \
         holds, and exempt with its reason one that skips for another cause:\n  {}",
        unlisted.join("\n  ")
    );
    let stale: Vec<&str> = named.iter().copied().filter(|path| !machine_local.contains(path)).collect();
    assert!(stale.is_empty(), "{LIST} names targets that print no skip, or are cuda-gated: take them off\n  {}", stale.join("\n  "));

    // `--test <name>` selects that name in every crate, so a listed target's
    // name must be its own.
    for path in &list.listed {
        let others: Vec<&String> = targets.keys().filter(|other| *other != path && stem(other) == stem(path)).collect();
        assert!(others.is_empty(), "`--test {}` runs {others:?} besides {path}", stem(path));
    }
}

/// The two readings the rule rests on, on sources written here.
#[test]
fn the_rule_reads_skips_and_cuda_gates_as_written() {
    assert!(prints_a_skip("eprintln!(\"skip: {ARTIFACT} does not exist\");"));
    assert!(prints_a_skip("eprintln!(\"SKIP: this CPU lacks AVX2\");"));
    assert!(prints_a_skip("eprintln!(\"skipped: no Flash-Next artifact at {}\", path.display());"));
    assert!(prints_a_skip("println!(\n        \"skip: the reference checkout {} is not on this machine\",\n        r.display()\n    );"));
    assert!(!prints_a_skip("Outcome { skip: RetainedSkip::PublishNoSlot, .. }"), "a field named skip is no print");
    assert!(!prints_a_skip("eprintln!(\"{name}: skip the warm-up\");"), "a skip later in the text is no skip");
    assert!(!prints_a_skip("gpu_profile::skip_or_fail(\"no artifact\");"), "the GPU profile's helper decides for itself");

    assert!(cuda_gated("//! A GPU test.\n#![cfg(feature = \"cuda\")]\nuse x;"));
    assert!(cuda_gated("#![cfg(all(feature = \"cuda\", feature = \"attn-tap\"))]"));
    assert!(!cuda_gated("#![cfg(unix)]"));
    assert!(!cuda_gated("#[cfg(feature = \"cuda\")]\nfn on_the_card() {}"), "an item gated alone leaves the target in cargo test");

    let list = parse("# comment\n\ncrates/a/tests/x.rs\nexempt crates/b/tests/y.rs: skips only without AVX2\n");
    assert_eq!(list.listed, ["crates/a/tests/x.rs"]);
    assert_eq!(list.exempt, [("crates/b/tests/y.rs".to_owned(), "skips only without AVX2".to_owned())]);
}
