//! GitHub #267 / #577, structurally: the acceptance replay's measurement
//! seams -- the teacher-forced verify round and its forced verdict
//! (`step::decode_program_verify_forced`, the decode options'
//! `forced_tokens`) and the proposal readout (`dflash2_readout`) -- are
//! test-only. No serving path names them: not the runtime that drives the
//! leaf, not the server, not the bench's client, and in the core crate only
//! the modules that define them. The leaf forces a round only when the decode
//! options carry `forced_tokens`, which only the forced round's Rust entry
//! sets; its verify graphs are captured without a recording and refuse one
//! (kernel/src/decode_graph.cu).

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every `.rs` file under `dir`, recursively.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    files
}

/// The seams' names, as code would call them.
const SEAMS: [&str; 5] = ["decode_program_verify_forced", "dflash2_readout", "forced_tokens", "propose_sampled", "verify_logits"];

/// The core crate's files that define the seams, and its root, which
/// declares the readout's module.
const DEFINING: [&str; 4] = ["dflash2_readout.rs", "step.rs", "step_abi.rs", "lib.rs"];

#[test]
fn no_serving_path_names_the_forced_round_or_the_readout() {
    let root = repo_root();
    let mut scanned = 0;
    let mut offenders = Vec::new();
    for dir in ["crates/core/src", "crates/runtime/src", "crates/server/src", "crates/bench/src"] {
        for path in rust_files(&root.join(dir)) {
            let defining = dir == "crates/core/src"
                && path.parent() == Some(&root.join(dir))
                && DEFINING.iter().any(|name| path.file_name().is_some_and(|f| f == *name));
            if defining {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            scanned += 1;
            for (n, line) in text.lines().enumerate() {
                // Code, not the doc comments that point at the seams.
                let code = line.trim_start();
                if code.starts_with("//") {
                    continue;
                }
                if let Some(seam) = SEAMS.iter().find(|seam| code.contains(*seam)) {
                    offenders.push(format!("{}:{}: {seam}: {}", path.display(), n + 1, code));
                }
            }
        }
    }
    assert!(scanned > 100, "scanned only {scanned} files -- wrong root?");
    assert!(offenders.is_empty(), "a serving path names a test-only replay seam:\n{}", offenders.join("\n"));
}
