//! Build script: link the kernel leaf (ADR 0001).
//!
//! Expects `kernel/build/ignis_kernel.lib` and `kernel/build/ignis_vendor.lib`
//! (the vendored reference substrate, ADR 0010), both prebuilt by
//! `kernel/build.ps1`, the proven cmake + ninja + nvcc flow. If either library
//! is missing, the script is run automatically.

use std::path::PathBuf;
use std::process::Command;
use std::env;

fn main() {
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
    // import lib is resolved at link time, the DLL (cudart64_13.dll) at
    // runtime from the CUDA toolkit's bin/x64 directory.
    let cuda = env::var("CUDA_PATH")
        .unwrap_or_else(|_| "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1".into());
    let cuda_lib = PathBuf::from(&cuda).join("lib").join("x64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search={}", cuda_lib.display());
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // NOTE: no explicit MSVC-CRT dylibs here. link.exe auto-links the CRT
    // (rustc passes /DEFAULTLIB:msvcrt) and resolves the import libs from the
    // system paths; this machine's VC redist is a minimal install with no
    // atlsx64 import libs. Revisit if a future kernel module needs msvcp
    // (C++ STL) explicitly.

    println!("cargo:rerun-if-changed={}", kernel_dir.display());
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}