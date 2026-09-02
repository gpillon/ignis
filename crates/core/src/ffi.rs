//! FFI bindings for the kernel leaf (flat C ABI, ADR 0001).
//!
//! Mirror of `kernel/include/ignis_kernel.h` — keep 1:1 when the surface
//! grows (ticket 03+).

/// Opaque CUDA graph handle, mirroring `struct ignis_graph` in
/// `kernel/include/ignis_kernel.h` (one element, never dereferenced across
/// the boundary; captured by `ignis_graph_begin_capture` /
/// `ignis_graph_end_capture`).
#[repr(C)]
pub struct IgnisGraph {
    _private: [u8; 0],
}

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
    /// then V), each [num_blocks][num_kv_heads][block_size][head_dim]
    /// (kv_head-major within a page; head_dim fastest).
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

    // ------------------------------------------------------------------
    // kernel-abi (tickets 05/06/10): prefill + GDN, pointwise / output
    // path, and eager CUDA-graph capture. Mirrors
    // `kernel/include/ignis_kernel.h` 1:1.
    //
    // NOTE (ADR 0006 / 0007): these surface declarations + geometry are
    // CPU-verifiable. The ticket-05 kernels (GQA prefill + GDN step) are now
    // implemented (kernel/src/gqa_attention_prefill.cuh, gdn_step.cuh,
    // prefill_gdn_surface.cu) and GPU-verified (tests/kernel_abi01_gpu launches
    // them on the GPU even with the model loaded, ADR 0006 nuance). The
    // pointwise/output (06) and CUDA-graph (10) .cu, and the 99% performance
    // gate (ADR 0007) driven by ignis-bench, remain pending.
    // ------------------------------------------------------------------

    /// GQA prefill attention (batched, multi-token), the prefill path.
    /// `q`: bf16 [batch][seq_len][num_q_heads][head_dim]. `kv_cache`: bf16,
    /// two paged planes (K then V), each [batch][num_blocks][num_kv_heads]
    /// [block_size][head_dim] (kv_head-major within a page; head_dim fastest).
    /// `block_table`: i32 [batch][num_blocks].
    /// `out`: bf16 [batch][seq_len][num_q_heads][head_dim]. `seq_len` <=
    /// num_blocks*block_size. `softmax_scale` <= 0 selects 1/sqrt(head_dim).
    /// `stream`: null = stream 0. Returns 0 on success, -1 on error.
    pub fn ignis_gqa_attention_prefill(
        q: *const std::ffi::c_void,
        kv_cache: *const std::ffi::c_void,
        block_table: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        batch: i64,
        seq_len: i64,
        num_q_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        block_size: i64,
        num_blocks: i64,
        softmax_scale: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// GDN (linear-attention) recurrent step, batched. `x`: bf16
    /// [batch][state_dim]. `state_in`/`state_out`: bf16
    /// [batch][num_gdn_layers][state_rows][state_cols] (state_out receives
    /// the updated state; state_in may alias state_out). `stream`: null =
    /// stream 0. Returns 0 on success, -1 on error.
    pub fn ignis_gdn_step(
        x: *const std::ffi::c_void,
        state_in: *const std::ffi::c_void,
        state_out: *mut std::ffi::c_void,
        batch: i64,
        num_gdn_layers: i64,
        state_rows: i64,
        state_cols: i64,
        state_dim: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// RMSNorm (LayerNorm when `center` is non-null). `x`: bf16 [n].
    /// `weight` (nullable): bf16 [n]. `center` (nullable): bf16 [n].
    /// `out`: bf16 [n]. `eps` <= 0 selects 1e-6. Returns 0 on success,
    /// -1 on error.
    pub fn ignis_rmsnorm(
        x: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        center: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        n: i64,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Embedding lookup: `out[row] = table[id[row]]`. `table`: bf16
    /// [vocab][hidden]. `id`: i32 [batch]. `out`: bf16 [batch][hidden].
    /// Returns 0 on success, -1 on error.
    pub fn ignis_embedding(
        table: *const std::ffi::c_void,
        id: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        batch: i64,
        vocab: i64,
        hidden: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Greedy sampling: `out[i] = argmax(logits[i])`. `logits`: f32
    /// [batch][vocab]. `out`: i32 [batch]. Ties resolve to the lowest index.
    /// Returns 0 on success, -1 on error.
    pub fn ignis_greedy_sample(
        logits: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        batch: i64,
        vocab: i64,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Begin a CUDA graph capture on `stream` (null = stream 0). The caller
    /// issues the prefill/decode kernel launches while the capture is active,
    /// then calls `ignis_graph_end_capture`. Returns 0 on success, -1 on
    /// error.
    pub fn ignis_graph_begin_capture(stream: *mut std::ffi::c_void) -> i32;

    /// End the capture, materializing the graph into `*out` (a
    /// graph-executable). Returns 0 on success, -1 on error.
    pub fn ignis_graph_end_capture(stream: *mut std::ffi::c_void, out: *mut *mut IgnisGraph) -> i32;

    /// Launch a captured graph on `stream`. Returns 0 on success, -1 on
    /// error.
    pub fn ignis_graph_launch(graph: *mut IgnisGraph, stream: *mut std::ffi::c_void) -> i32;

    /// Destroy a captured graph. NULL is a no-op.
    pub fn ignis_graph_destroy(graph: *mut IgnisGraph);
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

    // -------------------------------------------------------------------------
    // CPU-verifiable geometry for the kernel-abi C-ABI surface (tickets
    // 05/06/10). Pure Rust, no FFI calls — these pin the expected output
    // sizes for the flat-C-ABI kernels so the contract is testable on CPU.
    // The ticket-05 .cu (prefill + GDN step) are now implemented and
    // GPU-verified (tests/kernel_abi01_gpu); the 99% performance gate
    // (ADR 0007) driven by ignis-bench remains pending.
    // -------------------------------------------------------------------------

    /// GQA prefill output element count: `[batch][seq_len][num_q_heads][head_dim]`.
    fn gqa_prefill_out_elems(batch: u64, seq_len: u64, num_q_heads: u64, head_dim: u64) -> u64 {
        batch * seq_len * num_q_heads * head_dim
    }

    /// GQA prefill output byte count (bf16 = 2 bytes/elem).
    fn gqa_prefill_out_bytes(batch: u64, seq_len: u64, num_q_heads: u64, head_dim: u64) -> u64 {
        gqa_prefill_out_elems(batch, seq_len, num_q_heads, head_dim) * 2
    }

    /// GDN state tensor element count:
    /// `[batch][num_gdn_layers][state_rows][state_cols]`.
    fn gdn_state_elems(batch: u64, num_gdn_layers: u64, state_rows: u64, state_cols: u64) -> u64 {
        batch * num_gdn_layers * state_rows * state_cols
    }

    /// Embedding output element count: `[batch][hidden]`.
    fn embedding_out_elems(batch: u64, hidden: u64) -> u64 {
        batch * hidden
    }

    #[test]
    fn gqa_prefill_geometry_pins_out_shape() {
        // Representative canary shape: 4-batch, 512-token prefill, 8 q-heads,
        // head_dim 128. Pins the ABI out-size contract to the values the
        // kernel computes (independent of the helper's formula: 4*512*8*128
        // = 2_097_152 elems, bf16 = 4_194_304 bytes).
        let elems = gqa_prefill_out_elems(4, 512, 8, 128);
        assert_eq!(elems, 2_097_152);
        assert_eq!(gqa_prefill_out_bytes(4, 512, 8, 128), 4_194_304);
    }

    #[test]
    fn gdn_state_geometry_pins_state_shape() {
        // 8-lane batch, 12 GDN layers, 64x64 state (8*12*64*64 = 393_216).
        assert_eq!(gdn_state_elems(8, 12, 64, 64), 393_216);
    }

    #[test]
    fn embedding_geometry_pins_out_shape() {
        // Qwen hidden = 5120, 64-batch decode step (64*5120 = 327_680).
        assert_eq!(embedding_out_elems(64, 5120), 327_680);
    }
}