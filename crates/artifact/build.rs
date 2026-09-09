//! Build script for the `ignis-artifact` crate.
//!
//! Pure-Rust by default: the `CpuDevice` mock is the ADR 0006 stand-in while
//! the RTX 5090 is held by the reference runner, so the default build links
//! nothing. When the `cuda` feature is enabled this links the kernel leaf's
//! static libraries (its own C ABI surface, which carries the flat C device
//! surface `kernel/src/device.cu`, plus the vendored reference substrate of
//! ADR 0010) and the CUDA import libs, building the leaf first.
//!
//! The leaf is built on *every* run of this script, and this script re-runs
//! whenever any kernel source below changes. Both halves matter: until
//! GitHub #94 the build was skipped whenever `kernel/build/*.lib` merely
//! *existed*, and the only `rerun-if-changed` named the `kernel` directory
//! itself -- which cargo does not track recursively. Together those meant an
//! edited `.cu` was silently linked from the previous archive, and a binary
//! could carry a kernel many commits old while every test around it passed.
//! That is what invalidated the first G2 gate run (GitHub #93): the server
//! under measurement still prefilled a span one token at a time, the
//! pre-#84 behaviour, and looked ~175x slower than the reference.
//!
//! `kernel/build.ps1` is cmake + ninja, so a build with nothing to do costs
//! a few seconds and a real change is compiled. The `cuda` feature already
//! requires that toolchain (MSVC, Ninja, nvcc) -- there is no longer a path
//! where a stale archive stands in for it.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The kernel source trees whose contents force a leaf rebuild, relative to
/// `kernel/`. A whitelist rather than a walk of `kernel/` itself: the build
/// directory lives there too, and asking cargo to watch a file cmake writes
/// would rebuild on every single invocation, forever.
const KERNEL_SOURCE_ROOTS: &[&str] = &["src", "include", "tests", "vendor/src", "vendor/include"];

/// Files directly under `kernel/` that decide how the leaf is built.
const KERNEL_BUILD_FILES: &[&str] = &["CMakeLists.txt", "build.ps1"];

/// Emit a `rerun-if-changed` for every file under `dir`, recursively. A
/// missing or unreadable directory is silently skipped: the whitelist above
/// is a superset (`tests` is optional), and a directory that does not exist
/// cannot be the thing that went stale.
fn watch_tree(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            watch_tree(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn main() {
    // The `cuda` feature gates all kernel linking; the default (CPU) build is
    // pure Rust and links nothing.
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let kernel_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernel");
    let build_dir = kernel_dir.join("build");
    // Two archives: the leaf's own C ABI surface and the vendored reference
    // substrate it is built on (ADR 0010, kernel/vendor/VENDOR.md).
    let libraries = ["ignis_kernel", "ignis_vendor"];

    // Declared before the build runs, so a build that fails still leaves
    // cargo watching the source that must change to fix it.
    for root in KERNEL_SOURCE_ROOTS {
        watch_tree(&kernel_dir.join(root));
    }
    for file in KERNEL_BUILD_FILES {
        println!("cargo:rerun-if-changed={}", kernel_dir.join(file).display());
    }

    let script = kernel_dir.join("build.ps1");
    let out = Command::new("powershell")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            script.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run kernel/build.ps1");
    if !out.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&out.stdout));
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        panic!(
            "kernel leaf build failed ({}) — check the toolchain \
             (NINFER_WINDOWS_BUILD_NOTES.md) and re-run",
            out.status
        );
    }
    // A build that reported success but produced no archive would otherwise
    // fail much later, as an opaque linker error.
    for name in libraries {
        let lib = build_dir.join(format!("{name}.lib"));
        if !lib.exists() {
            eprintln!("{}", String::from_utf8_lossy(&out.stdout));
            panic!("kernel leaf build reported success but {} is missing", lib.display());
        }
    }

    println!("cargo:rustc-link-search={}", build_dir.display());
    for name in libraries {
        println!("cargo:rustc-link-lib=static={name}");
    }

    // CUDA runtime (dynamic): the static library imports cudart symbols; the
    // import lib is resolved at link time, the DLL (cudart64_*.dll) at runtime
    // from the CUDA toolkit's bin/x64 directory.
    let cuda = env::var("CUDA_PATH")
        .unwrap_or_else(|_| r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1".into());
    let cuda_lib = PathBuf::from(&cuda).join("lib").join("x64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search={}", cuda_lib.display());
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
