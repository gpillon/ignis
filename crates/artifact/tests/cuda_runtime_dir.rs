//! The CUDA runtime directory a Linux `--features cuda` build links (GitHub
//! #498): `build.rs` picks it with `cudart_dir`, compiled here from the same
//! file, over a toolkit's layout (`lib64`) and an environment-installed
//! CUDA's (`targets/x86_64-linux/lib` with no `lib64`). Plain directories, so
//! it runs on every host.

#[path = "../build/cuda_runtime.rs"]
mod cuda_runtime;

use std::path::{Path, PathBuf};

use cuda_runtime::{cudart_dir, CUDART};

/// An empty toolkit root of its own.
fn root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("ignis-cudart-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// `relative` under `root`, a file with its folders.
fn place(root: &Path, relative: &str) -> PathBuf {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"").unwrap();
    path
}

#[test]
fn a_toolkits_lib64_holds_the_runtime() {
    let root = root("lib64");
    place(&root, &format!("lib64/{CUDART}"));
    assert_eq!(cudart_dir(&root, "x86_64"), Ok(root.join("lib64")));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_environment_installed_cuda_holds_it_under_targets() {
    let root = root("targets");
    place(&root, &format!("targets/x86_64-linux/lib/{CUDART}"));
    assert_eq!(cudart_dir(&root, "x86_64"), Ok(root.join("targets").join("x86_64-linux").join("lib")));
    // The target's own architecture names the directory.
    assert!(cudart_dir(&root, "aarch64").is_err());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn lib64_is_tried_first() {
    let root = root("both");
    place(&root, &format!("lib64/{CUDART}"));
    place(&root, &format!("targets/x86_64-linux/lib/{CUDART}"));
    assert_eq!(cudart_dir(&root, "x86_64"), Ok(root.join("lib64")));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn neither_refuses_naming_the_paths_it_tried() {
    // A lib64 with only the versioned runtime has nothing `-lcudart` links,
    // and a directory under the link name is no library.
    let root = root("neither");
    place(&root, &format!("lib64/{CUDART}.13"));
    std::fs::create_dir_all(root.join("targets/x86_64-linux/lib").join(CUDART)).unwrap();
    let err = cudart_dir(&root, "x86_64").expect_err("no runtime to link");
    for tried in [root.join("lib64"), root.join("targets").join("x86_64-linux").join("lib")] {
        assert!(err.contains(&tried.display().to_string()), "{err}");
    }
    assert!(err.contains("CUDA_HOME") && err.starts_with(&format!("no {CUDART} in ")), "{err}");
    let _ = std::fs::remove_dir_all(&root);
}
