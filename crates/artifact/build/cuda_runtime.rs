//! Where a Linux CUDA toolkit keeps the runtime the link names (GitHub
//! #498). Shared by `build.rs`, which links it, and
//! `tests/cuda_runtime_dir.rs`, which holds the choice to both layouts: plain
//! `std`, so the build script and the test compile this same file.

use std::path::{Path, PathBuf};

/// The file `-lcudart` resolves to.
pub const CUDART: &str = "libcudart.so";

/// The directory under the toolkit root `cuda` that holds [`CUDART`] on
/// Linux: the toolkit's own `lib64`, else `targets/<arch>-linux/lib`, where an
/// environment-installed CUDA keeps its libraries with no `lib64` beside them
/// (`arch` the target's, `x86_64`). When neither holds it, the refusal names
/// both paths and the variables that choose the root, in place of a link
/// error that names neither.
pub fn cudart_dir(cuda: &Path, arch: &str) -> Result<PathBuf, String> {
    let candidates = [cuda.join("lib64"), cuda.join("targets").join(format!("{arch}-linux")).join("lib")];
    if let Some(dir) = candidates.iter().find(|dir| dir.join(CUDART).is_file()) {
        return Ok(dir.clone());
    }
    Err(format!(
        "no {CUDART} in {} or {}: the CUDA runtime is looked for under CUDA_HOME, else CUDA_PATH, else /usr/local/cuda (here {}); point CUDA_HOME at a CUDA toolkit that holds it",
        candidates[0].display(),
        candidates[1].display(),
        cuda.display()
    ))
}
