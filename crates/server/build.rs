//! Embeds the Playground frontend build (`web/dist`) into `ignis-server`
//! when it exists (GitHub #163, ADR 0026). Never runs npm: without a build,
//! the generated table is empty and the server serves its fallback page.
//!
//! Writes `$OUT_DIR/playground_assets.rs`, an expression of type
//! `playground::Assets` (`&[(path, include_bytes!(..))]`).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let web = manifest.join("../../web");
    let dist = web.join("dist");

    // Watch `web/dist` whether or not a build is there. Cargo treats a
    // missing watched path as always changed (a server rebuild on every
    // cargo run), and watching `web/` instead would scan `node_modules`; so
    // an absent `web/dist` is created empty (gitignored), and the first
    // frontend build, a rebuild, or a delete then re-embeds.
    if web.is_dir() && !dist.exists() {
        let _ = fs::create_dir(&dist);
    }
    println!("cargo:rerun-if-changed={}", dist.display());

    let mut files = Vec::new();
    if dist.join("index.html").is_file() {
        collect(&dist, "", &mut files);
        files.sort();
    }

    let mut table = String::from("&[\n");
    for (name, path) in &files {
        table.push_str(&format!("    ({name:?}, include_bytes!({path:?})),\n"));
    }
    table.push(']');

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    fs::write(out.join("playground_assets.rs"), table).expect("write playground_assets.rs");
}

/// Every file under `dir`, keyed by its `/`-separated path below the build
/// root (`prefix` is the path of `dir` itself).
fn collect(dir: &Path, prefix: &str, files: &mut Vec<(String, String)>) {
    for entry in fs::read_dir(dir).expect("read web/dist") {
        let entry = entry.expect("read web/dist entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        let key = if prefix.is_empty() { name } else { format!("{prefix}/{name}") };
        let path = entry.path();
        if path.is_dir() {
            collect(&path, &key, files);
        } else {
            let absolute = fs::canonicalize(&path).expect("canonicalize web/dist file");
            files.push((key, absolute.to_string_lossy().into_owned()));
        }
    }
}
