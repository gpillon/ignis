// ignis kernel leaf - B2 (kernel-abi 09, GitHub #32): the CUDA-graph decode
// replay (the decode hot path) over persistent device staging buffers (ADR
// 0008).
//
// A CUDA graph is a DAG of kernel nodes bound to *fixed* device pointers/sizes
// as captured between ignis_graph_begin_capture / ignis_graph_end_capture;
// ignis_graph_launch replays that exact DAG, and the leaf has no per-node
// update primitive (ADR 0008). "The graph reads the latest activation each
// decode step" is therefore achieved by *persistent device staging buffers*
// (stable addresses, allocated once at construction, the lifetime of the
// backend): the graph reads/writes fixed device buffers; each step H2D's the
// new input into the fixed input buffer, launches the graph (every node
// operates on the fixed buffers), and D2H's the logits (ADR 0008).
//
// The representative decode sequence captured here (the decode step's kernel
// structure — the mechanism this ticket delivers; the *full* per-layer stack
// + the host pointwise glue as device kernels is the 99%-gate performance
// material, ADR 0005 / 0007, ticket 20):
//   embed (token id -> hidden state)
//   -> GQA attention decode (q + paged KV -> attn out)   [representative]
//   -> GDN step (x -> recurrent-state update)            [representative]
//   -> final RMSNorm (hidden state -> normalized)
//   -> lm_head GEMV (normalized -> logits)
// Every node reads/writes the *fixed* device buffers (the staging buffers +
// the device-resident weights + the paged KV / GDN state). The sequence is
// launched *identically* for the capture (into the graph's DAG) and for the
// eager reference run, so the replayed logits are bit-identical to the eager
// logits (the kernel-abi 03 "replay == eager" invariant, ADR 0007).
//
// Style follows the existing surfaces (ADR 0001): a flat C ABI, an opaque
// handle, a 0/-1 int return code. The capture re-uses the kernel-abi 03 graph
// primitives (graph_capture.cu — ignis_graph_begin_capture /
// ignis_graph_end_capture / ignis_graph_launch / ignis_graph_destroy); the
// per-step H2D/D2H (the token id in, the logits out) are synchronous copies
// *around* the graph launch (never inside the capture window — ADR 0008).
//
// NOTE (capture stream): a CUDA graph cannot be captured on the legacy
// default stream. The leaf creates a leaf-owned non-blocking capture stream
// (created by ignis_decode_graph_new, destroyed by ignis_decode_graph_free).
//
// NOTE (self-skip, ADR 0006): a no-GPU host, a busy GPU, or a VRAM shortfall
// (the staging does not fit alongside the model) leaves the decode graph None
// — the compute adapter falls back to the eager sequence (ADR 0003). The VRAM
// shortfall is checked *before* the allocation (cudaMemGetInfo), so a
// VRAM-constrained GPU self-skips without a fault.

#include "ignis_kernel.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace ignis {

// The device kernels the decode sequence composes (defined in the sibling
// surface .cu files — the kernel-abi 01/02/05/10 surfaces). Only the
// prototypes are needed here (the kernels are launched directly on the
// capture stream — no per-call malloc/sync, ADR 0008), so the signatures
// mirror the .cuh definitions 1:1 (the pattern graph_capture.cu uses).
__global__ void embed_gather_kernel(const std::int32_t* __restrict__ ids,
                                    const __nv_bfloat16* __restrict__ table,
                                    __nv_bfloat16* __restrict__ out, std::int32_t d,
                                    std::int32_t batch);

__global__ void gqa_attention_decode_kernel(const __nv_bfloat16* __restrict__ kv,
                                            const std::int32_t* __restrict__ block_table,
                                            const __nv_bfloat16* __restrict__ q,
                                            __nv_bfloat16* __restrict__ out, int num_q_heads,
                                            int num_kv_heads, int head_dim, int seq_len,
                                            int block_size, int num_blocks,
                                            float softmax_scale);

__global__ void gdn_step_kernel(const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ state_in,
                                __nv_bfloat16* __restrict__ state_out, int batch,
                                int num_layers, int state_rows, int state_cols,
                                int state_dim);

__global__ void rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ weight,
                               const __nv_bfloat16* __restrict__ center,
                               __nv_bfloat16* __restrict__ out, int n, float eps);

__global__ void bf16_gemm_kernel(const __nv_bfloat16* __restrict__ act,
                                 const __nv_bfloat16* __restrict__ wt,
                                 const __nv_bfloat16* __restrict__ bias,
                                 __nv_bfloat16* __restrict__ out, std::int32_t tokens,
                                 std::int32_t m, std::int32_t k);

}  // namespace ignis

namespace {

int report(const char* what, cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[ignis-kernel] decode-graph %s: %s\n", what,
               cudaGetErrorString(err));
  return -1;
}

// The decode graph's staging buffers (the fixed-address device memory, ADR
// 0008). The read-only weights (embedding / final norm / lm_head) are either
// H2D'd once (the host / synthetic case — the leaf owns the copies, freed by
// ignis_decode_graph_free) or the device-resident pointers are bound as-is
// (the artifact case — the leaf does NOT own them, the artifact's arena
// outlives the decode graph, ADR 0002). The mutable state (paged KV, the
// block table, the GDN recurrent state) + the per-step intermediates (hidden
// / normed / logits / q / attn out / x) are always leaf-allocated (zeroed
// where a decode step starts from a fresh state, the ADR 0003 eager
// geometry).
struct DecodeStaging {
  // Per-step input (H2D'd each step) + intermediates (the graph's DAG reads /
  // writes these fixed buffers).
  std::int32_t* d_token;          // i32 [1]
  __nv_bfloat16* d_hidden;        // bf16 [hidden]
  __nv_bfloat16* d_normed;       // bf16 [hidden]
  __nv_bfloat16* d_logits;       // bf16 [vocab]
  __nv_bfloat16* d_q;            // bf16 [num_q_heads][head_dim]
  __nv_bfloat16* d_attn_out;     // bf16 [num_q_heads][head_dim]
  __nv_bfloat16* d_x;            // bf16 [gdn_state_dim]
  // Mutable decode state (zeroed at construction — a fresh representative
  // request's state, the ADR 0003 eager geometry).
  __nv_bfloat16* d_kv_cache;     // bf16, two paged planes (K then V)
  std::int32_t* d_block_table;   // i32 [num_blocks]
  __nv_bfloat16* d_gdn_state;    // bf16 [gdn_num_layers][state_rows][state_cols]
  // Read-only weights (device — the H2D'd copies or the artifact's pointers).
  // Non-const so the cudaMalloc idiom (the existing surfaces' pattern) takes
  // &s.d_* cleanly; the kernels' const params accept them implicitly.
  __nv_bfloat16* d_embedding;         // bf16 [vocab][hidden]
  __nv_bfloat16* d_final_norm;        // bf16 [hidden]
  __nv_bfloat16* d_lm_head;           // bf16 [vocab][hidden]
  int weights_copied;                 // 1 = the leaf H2D'd the weights (frees them)
};

// The total device VRAM the decode graph needs (the staging buffers + the
// weight copies, the host case only — the device-resident case binds the
// artifact's pointers, no new VRAM). Used for the ADR 0006 self-skip
// pre-check (a VRAM shortfall leaves the graph None — the eager fallback).
std::size_t total_bytes(const ignis_decode_graph_geom& g, int weights_on_device) {
  const std::size_t bf16 = sizeof(__nv_bfloat16);
  const std::size_t gqa_width =
      static_cast<std::size_t>(g.num_q_heads) * static_cast<std::size_t>(g.head_dim);
  const std::size_t kv_plane_elems =
      static_cast<std::size_t>(g.num_blocks) * static_cast<std::size_t>(g.num_kv_heads) *
      static_cast<std::size_t>(g.block_size) * static_cast<std::size_t>(g.head_dim);
  const std::size_t gdn_state_elems =
      static_cast<std::size_t>(g.gdn_num_layers) *
      static_cast<std::size_t>(g.gdn_state_rows) *
      static_cast<std::size_t>(g.gdn_state_cols);
  std::size_t total = 0;
  total += sizeof(std::int32_t);  // d_token
  total += static_cast<std::size_t>(g.hidden) * bf16 * 2;  // d_hidden + d_normed
  total += static_cast<std::size_t>(g.vocab) * bf16;  // d_logits
  total += gqa_width * bf16 * 2;  // d_q + d_attn_out
  total += static_cast<std::size_t>(g.gdn_state_dim) * bf16;  // d_x
  total += 2 * kv_plane_elems * bf16;  // d_kv_cache (two planes)
  total += static_cast<std::size_t>(g.num_blocks) * sizeof(std::int32_t);  // d_block_table
  total += gdn_state_elems * bf16;  // d_gdn_state
  if (weights_on_device == 0) {
    // The host case: the weight copies (the embedding + lm_head + final norm).
    total += static_cast<std::size_t>(g.vocab) * static_cast<std::size_t>(g.hidden) *
             bf16 * 2;  // d_embedding + d_lm_head
    total += static_cast<std::size_t>(g.hidden) * bf16;  // d_final_norm
  }
  return total;
}

// Allocate the staging buffers (the fixed-address device memory, ADR 0008).
// The mutable state (the paged KV, the GDN state, the q / x / attn_out
// buffers) is zeroed — a fresh decode starts from the zero state (the paged
// KV is zero-filled so unfilled pages attend as zeros, mirroring the host
// tier's zero-init, A3 / #30). The block table is the identity mapping (the
// synthetic / dev convention, ADR 0001). Returns false on a VRAM shortfall
// (the ADR 0006 self-skip — the subset that allocated is freed, no leak).
bool alloc_staging(DecodeStaging& s, const ignis_decode_graph_geom& g) {
  // A one-step allocation helper (stops the chain on the first VRAM
  // shortfall, ADR 0006 — the subset that allocated is freed by the caller).
  // Generic over the member's pointer type (a T** address); the cudaMalloc
  // idiom (the existing surfaces' pattern) allocates into *p.
  auto ok = [&](auto* p, std::size_t bytes) {
    if (*p != nullptr) return true;  // already allocated (a re-entry guard)
    return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
  };
  const std::size_t gqa_width =
      static_cast<std::size_t>(g.num_q_heads) * static_cast<std::size_t>(g.head_dim);
  const std::size_t kv_plane_elems =
      static_cast<std::size_t>(g.num_blocks) * static_cast<std::size_t>(g.num_kv_heads) *
      static_cast<std::size_t>(g.block_size) * static_cast<std::size_t>(g.head_dim);
  const std::size_t gdn_state_elems =
      static_cast<std::size_t>(g.gdn_num_layers) *
      static_cast<std::size_t>(g.gdn_state_rows) *
      static_cast<std::size_t>(g.gdn_state_cols);

  // The allocation chain stops at the first failure (a VRAM shortfall, ADR
  // 0006 — the subset that allocated is freed below, no leak on a partial
  // OOM). Each buffer is a fixed-address device allocation (ADR 0008).
  if (!ok(&s.d_token, sizeof(std::int32_t))) return false;
  if (!ok(&s.d_hidden, static_cast<std::size_t>(g.hidden) * sizeof(__nv_bfloat16)))
    return false;
  if (!ok(&s.d_normed, static_cast<std::size_t>(g.hidden) * sizeof(__nv_bfloat16)))
    return false;
  if (!ok(&s.d_logits, static_cast<std::size_t>(g.vocab) * sizeof(__nv_bfloat16)))
    return false;
  if (!ok(&s.d_q, gqa_width * sizeof(__nv_bfloat16))) return false;
  if (!ok(&s.d_attn_out, gqa_width * sizeof(__nv_bfloat16))) return false;
  if (!ok(&s.d_x, static_cast<std::size_t>(g.gdn_state_dim) * sizeof(__nv_bfloat16)))
    return false;
  if (!ok(&s.d_kv_cache, 2 * kv_plane_elems * sizeof(__nv_bfloat16))) return false;
  if (!ok(&s.d_block_table, static_cast<std::size_t>(g.num_blocks) * sizeof(std::int32_t)))
    return false;
  if (!ok(&s.d_gdn_state, gdn_state_elems * sizeof(__nv_bfloat16))) return false;

  // Zero the mutable state (the fresh representative request's geometry) +
  // fill the block table (the identity page mapping, the synthetic / dev
  // convention, ADR 0001). A copy failure (a VRAM shortfall / a busy GPU,
  // ADR 0006) frees the subset that allocated (no leak) and self-skips.
  bool fine =
      cudaMemset(s.d_kv_cache, 0, 2 * kv_plane_elems * sizeof(__nv_bfloat16)) ==
          cudaSuccess &&
      cudaMemset(s.d_gdn_state, 0, gdn_state_elems * sizeof(__nv_bfloat16)) ==
          cudaSuccess &&
      cudaMemset(s.d_q, 0, gqa_width * sizeof(__nv_bfloat16)) == cudaSuccess &&
      cudaMemset(s.d_attn_out, 0, gqa_width * sizeof(__nv_bfloat16)) == cudaSuccess &&
      cudaMemset(s.d_x, 0, static_cast<std::size_t>(g.gdn_state_dim) *
                               sizeof(__nv_bfloat16)) ==
          cudaSuccess;
  std::vector<std::int32_t> table(static_cast<std::size_t>(g.num_blocks));
  for (std::size_t i = 0; i < table.size(); ++i) table[i] = static_cast<std::int32_t>(i);
  if (fine) {
    fine =
        cudaMemcpy(s.d_block_table, table.data(),
                   static_cast<std::size_t>(g.num_blocks) * sizeof(std::int32_t),
                   cudaMemcpyHostToDevice) == cudaSuccess;
  }
  if (!fine) {
    // A partial allocation / copy (a VRAM shortfall, ADR 0006): free the
    // subset that allocated (no leak on a partial OOM) and self-skip.
    if (s.d_token != nullptr) cudaFree(s.d_token);
    if (s.d_hidden != nullptr) cudaFree(s.d_hidden);
    if (s.d_normed != nullptr) cudaFree(s.d_normed);
    if (s.d_logits != nullptr) cudaFree(s.d_logits);
    if (s.d_q != nullptr) cudaFree(s.d_q);
    if (s.d_attn_out != nullptr) cudaFree(s.d_attn_out);
    if (s.d_x != nullptr) cudaFree(s.d_x);
    if (s.d_kv_cache != nullptr) cudaFree(s.d_kv_cache);
    if (s.d_block_table != nullptr) cudaFree(s.d_block_table);
    if (s.d_gdn_state != nullptr) cudaFree(s.d_gdn_state);
    std::memset(&s, 0, sizeof(s));
    return false;
  }
  return true;
}

// Load the read-only weights into the device buffers (the H2D-once host /
// synthetic case — the leaf owns the copies, freed by
// ignis_decode_graph_free) or bind the device-resident pointers (the artifact
// case — the artifact's arena, ADR 0002, no per-call H2D, the leaf does NOT
// own them). Returns false on a VRAM shortfall (the ADR 0006 self-skip — the
// subset that allocated is freed, no leak).
bool load_weights(DecodeStaging& s, const ignis_decode_graph_weights& w) {
  if (w.weights_on_device != 0) {
    // The device-resident case: bind the artifact's pointers (the leaf does
    // NOT own them — the artifact's arena outlives the decode graph, ADR
    // 0002). A null binding is a clean self-skip (the eager fallback).
    if (w.embedding == nullptr || w.final_norm == nullptr || w.lm_head == nullptr)
      return false;
    // The device-resident case: bind the artifact's device pointers (the
    // leaf does NOT own them — the artifact's arena outlives the decode
    // graph, ADR 0002). `const_cast<void*>` strips the `const void*`
    // qualifier (the artifact's read-only weight buffers — the leaf's
    // staging members are non-const so the kernels' const params accept
    // them, the kernel-abi 09 build fix, GitHub #32).
    s.d_embedding =
        reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(w.embedding));
    s.d_final_norm =
        reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(w.final_norm));
    s.d_lm_head =
        reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(w.lm_head));
    s.weights_copied = 0;
    return true;
  }
  // The host case: H2D the weights once (leaf-owned device buffers — freed
  // by ignis_decode_graph_free, the ADR 0002 "no per-call H2D" applied to
  // the one-shot construction).
  s.weights_copied = 1;
  cudaError_t e = cudaSuccess;
  // The embedding table (bf16 [vocab][hidden]).
  e = cudaMalloc(&s.d_embedding, static_cast<std::size_t>(w.embedding_bytes));
  if (e == cudaSuccess)
    e = cudaMemcpy(s.d_embedding, w.embedding, static_cast<std::size_t>(w.embedding_bytes),
                   cudaMemcpyHostToDevice);
  // The final-norm weight (bf16 [hidden]).
  if (e == cudaSuccess)
    e = cudaMalloc(&s.d_final_norm, static_cast<std::size_t>(w.final_norm_bytes));
  if (e == cudaSuccess)
    e = cudaMemcpy(s.d_final_norm, w.final_norm,
                   static_cast<std::size_t>(w.final_norm_bytes),
                   cudaMemcpyHostToDevice);
  // The lm_head weight (bf16 [vocab][hidden]).
  if (e == cudaSuccess)
    e = cudaMalloc(&s.d_lm_head, static_cast<std::size_t>(w.lm_head_bytes));
  if (e == cudaSuccess)
    e = cudaMemcpy(s.d_lm_head, w.lm_head, static_cast<std::size_t>(w.lm_head_bytes),
                   cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    // A partial weight load (a VRAM shortfall, ADR 0006): free the subset
    // that allocated (no leak on a partial OOM) and self-skip.
    if (s.d_embedding != nullptr) cudaFree(s.d_embedding);
    if (s.d_final_norm != nullptr)
      cudaFree(s.d_final_norm);
    if (s.d_lm_head != nullptr) cudaFree(s.d_lm_head);
    s.d_embedding = nullptr;
    s.d_final_norm = nullptr;
    s.d_lm_head = nullptr;
    s.weights_copied = 0;
    return false;
  }
  return true;
}

// The representative decode sequence (the graph's DAG, ADR 0008): embed ->
// GQA attention -> GDN step -> final RMSNorm -> lm_head GEMV, every kernel
// reading/writing the *fixed* staging buffers. This is the single source of
// the sequence — it is invoked *inside* the capture window (building the
// graph's DAG) and *eagerly* (the bit-identical reference, the kernel-abi 03
// invariant, ADR 0007). No malloc / sync / copy inside the sequence (the
// capture window's contract — CUDA graph capture forbids them).
cudaError_t launch_sequence(const DecodeStaging& s, const ignis_decode_graph_geom& g,
                            cudaStream_t stream) {
  // 1. embed (token id -> hidden state): out[0*hidden + k] =
  //    table[id[0]*hidden + k]. Grid-stride over the [1][hidden] output (the
  //    embed launcher's geometry, kernel-abi 02).
  {
    const unsigned grid =
        std::max(1u, static_cast<unsigned>((static_cast<std::size_t>(g.hidden) + 127) / 128));
    ignis::embed_gather_kernel<<<grid, 128, 0, stream>>>(
        s.d_token, s.d_embedding, s.d_hidden, static_cast<std::int32_t>(g.hidden), 1);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  }
  // 2. GQA attention decode (q + paged KV -> attn out): a representative
  //    side-branch (it does not feed the logits — the attention output is a
  //    separate buffer; the logits are GEMV(norm(embed)) — but it exercises
  //    the decode attention kernel's capture, the B2 mechanism). `seq_len`
  //    is the paged capacity (the compute-adapter's eager decode passes the
  //    full capacity too — the zero-filled KV makes the attention
  //    deterministic, the A3 / #30 convention).
  {
    const unsigned grid = static_cast<unsigned>(g.num_q_heads);
    const unsigned block = static_cast<unsigned>(g.head_dim);
    ignis::gqa_attention_decode_kernel<<<grid, block, 0, stream>>>(
        s.d_kv_cache, s.d_block_table, s.d_q, s.d_attn_out,
        static_cast<int>(g.num_q_heads), static_cast<int>(g.num_kv_heads),
        static_cast<int>(g.head_dim), static_cast<int>(g.num_blocks * g.block_size),
        static_cast<int>(g.block_size), static_cast<int>(g.num_blocks), 1.0f);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  }
  // 3. GDN step (x -> recurrent-state update): a representative side-branch
  //    (the GDN recurrent state is updated in-place; the GDN output
  //    projection is the 99%-gate material, #20). One block per (d_v row,
  //    gdn layer); one thread per d_k column; the block's dynamic smem holds
  //    the d_k partials (the gdn_step launcher's geometry, kernel-abi 01).
  {
    const dim3 grid(static_cast<unsigned>(g.gdn_state_rows),
                    static_cast<unsigned>(g.gdn_num_layers));
    const unsigned block = static_cast<unsigned>(g.gdn_state_cols);
    const std::size_t smem =
        static_cast<std::size_t>(g.gdn_state_cols) * sizeof(float);
    ignis::gdn_step_kernel<<<grid, block, static_cast<std::size_t>(smem), stream>>>(
        s.d_x, s.d_gdn_state, s.d_gdn_state, 1, static_cast<int>(g.gdn_num_layers),
        static_cast<int>(g.gdn_state_rows), static_cast<int>(g.gdn_state_cols),
        static_cast<int>(g.gdn_state_dim));
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  }
  // 4. final RMSNorm (hidden state -> normalized): one 1024-thread block,
  //    grid-stride over [hidden] (the rmsnorm launcher's geometry,
  //    kernel-abi 02). weight = the final-norm weight, center = null (the
  //    RMSNorm mode).
  {
    ignis::rmsnorm_kernel<<<1, 1024, 0, stream>>>(
        s.d_hidden, s.d_final_norm, nullptr, s.d_normed,
        static_cast<int>(g.hidden), 1e-6f);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  }
  // 5. lm_head GEMV (normalized -> logits): out[0][m] = sum_k normed[k] *
  //    W[m][k]. Rowsplit tiling, tokens == 1 (the decode GEMV special case,
  //    kernel-abi 10).
  {
    const dim3 grid(static_cast<unsigned>((g.vocab + 15) / 16), 1u);
    const dim3 block(16, 16);
    ignis::bf16_gemm_kernel<<<grid, block, 0, stream>>>(
        s.d_normed, s.d_lm_head, nullptr, s.d_logits, 1,
        static_cast<std::int32_t>(g.vocab), static_cast<std::int32_t>(g.hidden));
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  }
  return cudaSuccess;
}

// NOTE: the opaque decode-graph handle (`struct ignis_decode_graph`) is
// defined at *global scope* (after this anonymous namespace closes) so it
// completes the C struct tag (the header's `struct ignis_decode_graph;`)
// rather than creating a second, C++-linkage struct (which would be
// ambiguous — the kernel-abi 09 build fix, GitHub #32). It references the
// `DecodeStaging` below (visible at global scope — the anonymous namespace's
// members are injected into the enclosing global scope).

// Free the core staging buffers (the non-weight device memory) + null them
// (the no-leak partial-OOM cleanup, ADR 0006 — the weight copies are freed
// separately by ignis_decode_graph_free, honoring the weights_copied flag).
void free_core_staging(DecodeStaging& s) {
  if (s.d_token != nullptr) cudaFree(s.d_token);
  if (s.d_hidden != nullptr) cudaFree(s.d_hidden);
  if (s.d_normed != nullptr) cudaFree(s.d_normed);
  if (s.d_logits != nullptr) cudaFree(s.d_logits);
  if (s.d_q != nullptr) cudaFree(s.d_q);
  if (s.d_attn_out != nullptr) cudaFree(s.d_attn_out);
  if (s.d_x != nullptr) cudaFree(s.d_x);
  if (s.d_kv_cache != nullptr) cudaFree(s.d_kv_cache);
  if (s.d_block_table != nullptr) cudaFree(s.d_block_table);
  if (s.d_gdn_state != nullptr) cudaFree(s.d_gdn_state);
  s.d_token = nullptr;
  s.d_hidden = nullptr;
  s.d_normed = nullptr;
  s.d_logits = nullptr;
  s.d_q = nullptr;
  s.d_attn_out = nullptr;
  s.d_x = nullptr;
  s.d_kv_cache = nullptr;
  s.d_block_table = nullptr;
  s.d_gdn_state = nullptr;
}

// Free the weight copies (the host case — the leaf owns them; the device-
// resident case binds the artifact's pointers, which are NOT freed, ADR 0002)
// + null them (the no-leak cleanup, honoring the weights_copied flag).
void free_weight_staging(DecodeStaging& s) {
  if (s.weights_copied != 0) {
    if (s.d_embedding != nullptr) cudaFree(s.d_embedding);
    if (s.d_final_norm != nullptr) cudaFree(s.d_final_norm);
    if (s.d_lm_head != nullptr) cudaFree(s.d_lm_head);
  }
  s.d_embedding = nullptr;
  s.d_final_norm = nullptr;
  s.d_lm_head = nullptr;
  s.weights_copied = 0;
}

// Free the full staging (the core + the weight copies) + null it (the
// no-leak cleanup on a construction failure, ADR 0006 — the partial-OOM
// contract: the subset that allocated is freed, no leak).
void free_full_staging(DecodeStaging& s) {
  free_core_staging(s);
  free_weight_staging(s);
}

}  // namespace

// The opaque decode-graph handle (ADR 0008): the captured graph (the
// kernel-abi 03 ignis_graph) + the leaf-owned capture stream + the staging
// buffers. Defined at *global scope* (completing the C struct tag, the
// header's `struct ignis_decode_graph;` — not a second C++ struct, the
// kernel-abi 09 build fix, GitHub #32). Never dereferenced across the C
// boundary (the C side sees only the opaque tag); the leaf's internal
// layout (the fields below).
struct ignis_decode_graph {
  struct ignis_graph* graph;  // the captured decode DAG (null on a self-skip)
  cudaStream_t stream;        // the leaf-owned non-blocking capture stream
  DecodeStaging staging;      // the fixed-address device buffers
  ignis_decode_graph_geom geom;  // the representative decode geometry
};

// B2 (ADR 0008): construct the decode graph. Allocates the staging buffers
// (the fixed-address device memory, the representative decode geometry),
// loads the read-only weights (the H2D-once host case / the device-resident
// artifact case), creates the capture stream, captures the representative
// decode sequence on the fixed buffers (the graph's DAG), and instantiates
// the graph. Returns 0 on success, -1 on a CUDA error (no GPU / busy / OOM —
// the caller self-skips, ADR 0006; the VRAM shortfall is checked *before*
// the allocation, so a VRAM-constrained GPU self-skips without a fault).
extern "C" int ignis_decode_graph_new(const ignis_decode_graph_geom* geom,
                                      const ignis_decode_graph_weights* wts,
                                      struct ignis_decode_graph** out) {
  if (geom == nullptr || wts == nullptr || out == nullptr) return -1;
  *out = nullptr;

  // The ADR 0006 self-skip pre-check (a no-GPU host, a VRAM-constrained
  // GPU — the staging does not fit alongside the model). A no-GPU host
  // fails cudaMemGetInfo cleanly; the caller treats a -1 as "graph None,
  // the eager fallback" (never a fault).
  std::size_t free_bytes = 0;
  std::size_t total_vram = 0;
  if (cudaMemGetInfo(&free_bytes, &total_vram) != cudaSuccess)
    return -1;  // no GPU — the caller self-skips (ADR 0006)
  // The host case H2D's the weights (the device-resident case binds the
  // artifact's pointers — no new VRAM for the weights).
  const std::size_t needed =
      total_bytes(*geom, wts->weights_on_device) + (1u << 20);  // + a 1 MB margin
  if (free_bytes < needed)
    return -1;  // a VRAM shortfall (a busy / loaded GPU) — the eager fallback

  DecodeStaging staging;
  std::memset(&staging, 0, sizeof(staging));
  if (!alloc_staging(staging, *geom))
    return -1;  // a VRAM shortfall (ADR 0006 — the eager fallback)
  if (!load_weights(staging, *wts)) {
    // A partial weight load (a VRAM shortfall, ADR 0006): the core staging
    // is allocated (load_weights freed its partial weight subset); free the
    // core staging (no leak) and self-skip (the eager fallback).
    free_core_staging(staging);
    return -1;
  }

  // The leaf-owned non-blocking capture stream (a CUDA graph cannot be
  // captured on the legacy default stream — the kernel-abi 03 note).
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    return -1;  // no GPU — the caller self-skips (ADR 0006)

  // Capture the representative decode sequence on the capture stream (the
  // kernel-abi 03 begin/end pairing — the graph's DAG). The kernel launches
  // run on the stream the capture began on (the leaf's non-blocking stream
  // — the capture's contract).
  int begin_rc = ignis_graph_begin_capture(stream);
  if (begin_rc != 0) {
    cudaStreamDestroy(stream);
    return -1;  // a capture already in progress, or a CUDA error (ADR 0006)
  }
  cudaError_t seq_err = launch_sequence(staging, *geom, stream);
  struct ignis_graph* graph = nullptr;
  int end_rc = ignis_graph_end_capture(stream, &graph);
  if (seq_err != cudaSuccess || end_rc != 0) {
    // The capture did not materialize (a busy / absent GPU, a stream
    // mismatch) — the eager fallback (ADR 0003 / ADR 0006).
    if (graph != nullptr) ignis_graph_destroy(graph);
    cudaStreamDestroy(stream);
    free_full_staging(staging);  // no leak (the partial-OOM contract, ADR 0006)
    return -1;
  }

  ignis_decode_graph* g = new (std::nothrow) ignis_decode_graph();
  if (g == nullptr) {
    ignis_graph_destroy(graph);
    cudaStreamDestroy(stream);
    free_full_staging(staging);  // no leak (a host OOM, ADR 0006)
    return -1;
  }
  g->graph = graph;
  g->stream = stream;
  g->staging = staging;
  g->geom = *geom;
  *out = g;
  return 0;
}

// B2 (ADR 0008): replay the decode graph (the per-step hot path). H2D the
// token id (the per-step input) into the fixed input buffer, launch the
// graph (the whole decode DAG runs on the fixed buffers), D2H the logits.
// No per-step capture, no node update (ADR 0008). `stream`: null = the
// graph's capture stream (the leaf-owned non-blocking stream). Returns 0 on
// success, -1 on a CUDA error (a busy GPU self-skips, ADR 0006).
extern "C" int ignis_decode_graph_replay(struct ignis_decode_graph* g, int32_t token_id,
                                          void* out_logits_host, void* stream) {
  if (g == nullptr || g->graph == nullptr || out_logits_host == nullptr)
    return -1;  // a null handle / logits pointer is a clean -1 (before any CUDA call)
  const cudaStream_t s =
      (stream != nullptr) ? static_cast<cudaStream_t>(stream) : g->stream;
  // 1. H2D the per-step input (the token id) into the fixed input buffer.
  cudaError_t e =
      cudaMemcpy(g->staging.d_token, &token_id, sizeof(std::int32_t),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return report("replay H2D", e);
  // 2. Launch the graph (the whole decode DAG runs on the fixed buffers —
  //    the kernel-abi 03 replay, ADR 0008: one launch over the DAG).
  if (ignis_graph_launch(g->graph, s) != 0)
    return -1;  // a null graph / a CUDA error (a busy GPU self-skips, ADR 0006)
  // 3. Wait for the DAG to finish (the logits are D2H'd after the graph).
  e = cudaStreamSynchronize(s);
  if (e != cudaSuccess) return report("replay sync", e);
  // 4. D2H the logits (bf16 [vocab]).
  e = cudaMemcpy(out_logits_host, g->staging.d_logits,
                 static_cast<std::size_t>(g->geom.vocab) * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToHost);
  return report("replay D2H", e);
}

// B2 (the kernel-abi 03 "replay == eager" invariant, ADR 0007): run the
// representative decode sequence *eagerly* (no graph) over the same fixed
// buffers — the eager reference for the bit-exact check. Same inputs /
// buffers / kernels as ignis_decode_graph_replay, so the logits must be
// bit-identical (the graph replay's verification). Returns 0 on success, -1
// on a CUDA error (a busy GPU self-skips, ADR 0006).
extern "C" int ignis_decode_graph_eager(struct ignis_decode_graph* g, int32_t token_id,
                                        void* out_logits_host, void* stream) {
  if (g == nullptr || out_logits_host == nullptr) return -1;
  const cudaStream_t s =
      (stream != nullptr) ? static_cast<cudaStream_t>(stream) : g->stream;
  // 1. H2D the per-step input (the token id) into the fixed input buffer.
  cudaError_t e =
      cudaMemcpy(g->staging.d_token, &token_id, sizeof(std::int32_t),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return report("eager H2D", e);
  // 2. Run the decode sequence eagerly on the stream (the same kernels as the
  //    capture — the bit-identical reference, the kernel-abi 03 invariant).
  e = launch_sequence(g->staging, g->geom, s);
  if (e != cudaSuccess) return report("eager sequence", e);
  // 3. Wait for the sequence to finish.
  e = cudaStreamSynchronize(s);
  if (e != cudaSuccess) return report("eager sync", e);
  // 4. D2H the logits (bf16 [vocab]).
  e = cudaMemcpy(out_logits_host, g->staging.d_logits,
                 static_cast<std::size_t>(g->geom.vocab) * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToHost);
  return report("eager D2H", e);
}

// B2 (ADR 0008): free the decode graph (the captured graph, the leaf-owned
// capture stream, the staging buffers, and — the host case — the H2D'd
// weight copies). NULL is a no-op.
extern "C" void ignis_decode_graph_free(struct ignis_decode_graph* g) {
  if (g == nullptr) return;  // NULL is a no-op (no CUDA calls)
  // Destroy the captured graph (the kernel-abi 03 handle; NULL is a no-op —
  // the eager-fallback case, ADR 0006).
  if (g->graph != nullptr) ignis_graph_destroy(g->graph);
  // Free the staging buffers (the core + the weight copies, the host case —
  // honoring the weights_copied flag, ADR 0002: the device-resident case
  // binds the artifact's pointers, which are NOT freed here).
  free_full_staging(g->staging);
  if (g->stream != nullptr) cudaStreamDestroy(g->stream);
  delete g;
}