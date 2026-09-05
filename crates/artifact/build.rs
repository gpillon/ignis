//! Build script for the `ignis-artifact` crate.
//!
//! Pure-Rust by default: the `CpuDevice` mock is the ADR 0006 stand-in while
//! the RTX 5090 is held by the reference runner, so the default build links
//! nothing. When the `cuda` feature is enabled this mirrors
//! `crates/core/build.rs`: it links the kernel leaf's static libraries (its
//! own C ABI surface, which carries the flat C device surface
//! `kernel/src/device.cu`, plus the vendored reference substrate of ADR 0010)
//! and the CUDA import libs, auto-building the leaf if either is missing.

use std::path::PathBuf;
use std::process::Command;
use std::env;

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

    if libraries
        .iter()
        .any(|name| !build_dir.join(format!("{name}.lib")).exists())
    {
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
    }

    println!("cargo:rustc-link-search={}", build_dir.display());
    for name in libraries {
        println!("cargo:rustc-link-lib=static={name}");
    }

    // CUDA runtime (dynamic): the static library imports cudart symbols; the
    // import lib is resolved at link time, the DLL (cudart64_*.dll) at runtime
    // from the CUDA toolkit's bin/x64 directory.
    let cuda = env::var("CUDA_PATH")
        .unwrap_or_else(|_| "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1".into());
    let cuda_lib = PathBuf::from(&cuda).join("lib").join("x64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search={}", cuda_lib.display());
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    println!("cargo:rerun-if-changed={}", kernel_dir.display());
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}