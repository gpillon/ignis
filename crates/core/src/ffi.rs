//! FFI bindings for the kernel leaf (flat C ABI, ADR 0001).
//!
//! Mirror of `kernel/include/ignis_kernel.h` — keep 1:1 when the surface
//! grows (ticket 03+).

unsafe extern "C" {
    /// Ticket 01 smoke test: proves the FFI path end-to-end.
    pub fn ignis_kernel_hello() -> u32;

    /// `c[i] = a[i] + b[i]` for `i in [0, n)`.
    /// Returns 0 on success, -1 on CUDA error.
    pub fn ignis_kernel_vector_sum(a: *const f32, b: *const f32, c: *mut f32, n: usize) -> i32;

    // Ticket 03: decode step (NVFP4 GEMM + GQA attention). Flat C ABI (ADR 0001):
    // explicit pointers + sizes, a stream handle (null = stream 0), an int return
    // code (0 = ok, -1 = CUDA error / invalid argument). All buffer pointers are
    // host memory; the leaf does the H2D/D2H copies internally.

    /// NVFP4 decode GEMM (GEMV path, single token):
    /// `out[m] = bias[m] + sum_k x[k] * W[m,k]`. Weights are NVFP4-quantized
    /// (E2M1 codes, 2 per byte; E4M3 scale per 16-element group).
    /// `act`: bf16 [k]; `wt_codes`: E2M1 [m][k/2] bytes; `wt_scales`: E4M3 [m][k/16].
    /// `bias` (nullable) and `out`: bf16 [m]. `k` must be a multiple of 16.
    /// `stream`: null = stream 0. Returns 0 on success, -1 on error.
    pub fn ignis_nvfp4_gemm_decode(
        act: *const std::ffi::c_void,
        wt_codes: *const std::ffi::c_void,
        wt_scales: *const std::ffi::c_void,
        bias: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        m: i64,
        k: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// GQA attention decode (single token, paged bf16 KV cache).
    /// `q`: bf16 [num_q_heads][head_dim]. `kv_cache`: bf16, two paged planes (K
    /// then V), each [num_blocks][block_size][num_kv_heads][head_dim].
    /// `block_table`: i32 [num_blocks] (logical block -> physical page id).
    /// `out`: bf16 [num_q_heads][head_dim]. `seq_len` <= num_blocks*block_size.
    /// `stream`: null = stream 0. Returns 0 on success, -1 on error.
    pub fn ignis_gqa_attention_decode(
        q: *const std::ffi::c_void,
        kv_cache: *const std::ffi::c_void,
        block_table: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        num_q_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        seq_len: i64,
        block_size: i64,
        num_blocks: i64,
        softmax_scale: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_smoke() {
        assert_eq!(unsafe { ignis_kernel_hello() }, 42);
    }

    #[test]
    fn vector_sum_smoke() {
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
        let mut c = vec![0.0f32; n];
        let rc = unsafe { ignis_kernel_vector_sum(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n) };
        assert_eq!(rc, 0, "CUDA vector sum failed (is the GPU available?)");
        for i in 0..n {
            assert_eq!(c[i], (i as f32) * 3.0, "mismatch at {i}");
        }
    }
}