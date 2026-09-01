# 01 — kernel build skeleton

Status: resolved

CMake build of the kernel leaf: `nvcc` (CUDA 13.1) targeting SM120a, producing
the static library the Rust side links. Rust side: `build.rs` in `ignis-core`
that locates the prebuilt `.lib`, plus a build script that drives CMake.
Verification: a trivial "hello kernel" (empty entry point compiled + linked,
called from `ignis-core`) proving the FFI path end-to-end.

## Resolution (2026-09-02)

- `kernel/CMakeLists.txt` (Ninja + nvcc 13.1.80, `CMAKE_CUDA_ARCHITECTURES=120a`,
  `-Xcompiler -GS-` because lld cannot satisfy MSVC /GS security-cookie symbols)
- `kernel/build.ps1` (vcvars64 import + cmake + ninja; ninja fallback
  `F:\ai\q38\tools\ninja`; **nvcc path with forward slashes** — CMake 3.28
  mis-quotes `CMakeCUDACompiler.cmake` when the value has backslashes + spaces)
- `kernel/include/ignis_kernel.h` + `kernel/src/hello.cu` (hello + vector-sum
  smoke kernels)
- `crates/core/build.rs` (auto-builds the leaf when missing, links it + the
  CUDA import libs; no explicit MSVC-CRT dylibs — link.exe auto-links)
- `crates/core/src/ffi.rs` (unsafe extern "C" bindings + smoke tests)
- **`.cargo/config.toml` pins `x86_64-pc-windows-msvc`** (the machine's default
  rustup target is `x86_64-pc-windows-gnu`, which cannot consume MSVC COFF
  archives + CRT import libs; the MSVC target was added via `rustup target add`)
- Runtime: `cudart64_13.dll` resolves from `C:\Program Files\NVIDIA GPU
  Computing Toolkit\CUDA\v13.1\bin\x64` (CUDA 13 splits bin into 32/64-bit)
- Verified: `cargo test -p ignis-core` passes both smoke tests, including a real
  CUDA kernel launch on the RTX 5090.