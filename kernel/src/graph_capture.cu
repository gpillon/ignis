// ignis kernel leaf - Ticket 10 (kernel-abi-03): eager CUDA-graph capture at
// startup.
//
// Implements the C ABI functions declared in include/ignis_kernel.h for
// kernel-abi-03: ignis_graph_begin_capture / ignis_graph_end_capture /
// ignis_graph_launch / ignis_graph_destroy, plus ignis_graph_startup_check
// (the startup verification: capture a representative prefill + decode
// kernel sequence into a CUDA graph, replay it, and confirm the replayed
// outputs match the eager outputs bit-exactly).
//
// Style follows the ticket-03/05/06 surfaces: explicit pointers + sizes, a
// stream handle (null = stream 0), a 0/-1 int return code, and an opaque
// handle across the boundary. The captured work is a sequence of raw
// kernel launches (the prefill + GDN + decode kernels in the sibling .cuh
// files, provenance in kernel/NOTICE); the device inputs are H2D'd before
// the capture window, so the graph holds only kernel nodes.
//
// NOTE (capture stream): a CUDA graph cannot be captured on the legacy
// default stream. When the caller passes a null stream, the leaf uses a
// leaf-owned non-blocking stream (created by begin, destroyed by
// ignis_graph_destroy along with the graph). A non-null stream is the
// caller's (it must not be the legacy default stream, which cannot be
// captured; the leaf does not own it).
//
// NOTE (v1, design §1/§2): eager capture at startup, one capture at a time
// (the engine captures once at startup; lazy / concurrent capture is a
// later optimization). The canary-suite 99% performance gate (ADR 0007) is
// driven by ignis-bench (ticket 20), not here.

#include "ignis_kernel.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ignis {

// Forward declarations of the three ported kernels (defined in the sibling
// surface .cu files: gqa_attention_prefill + gdn_step in
// prefill_gdn_surface.cu, gqa_attention_decode in decode_surface.cu).
// graph_capture.cu launches them directly (the captured graph holds kernel
// nodes), so it only needs the prototypes — not the .cuh definitions.
// Including the .cuh definitions here would duplicate each kernel in this
// translation unit (an LNK4006 at .lib link time, since the surface .cu
// files already define them). The signatures mirror the .cuh definitions 1:1.
__global__ void gqa_attention_prefill_kernel(const __nv_bfloat16* __restrict__ kv,
                                             const std::int32_t* __restrict__ block_table,
                                             const __nv_bfloat16* __restrict__ q,
                                             __nv_bfloat16* __restrict__ out, int batch,
                                             int seq_len, int num_q_heads, int num_kv_heads,
                                             int head_dim, int block_size, int num_blocks,
                                             float softmax_scale);

__global__ void gdn_step_kernel(const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ state_in,
                                __nv_bfloat16* __restrict__ state_out, int batch,
                                int num_layers, int state_rows, int state_cols,
                                int state_dim);

__global__ void gqa_attention_decode_kernel(const __nv_bfloat16* __restrict__ kv,
                                            const std::int32_t* __restrict__ block_table,
                                            const __nv_bfloat16* __restrict__ q,
                                            __nv_bfloat16* __restrict__ out, int num_q_heads,
                                            int num_kv_heads, int head_dim, int seq_len,
                                            int block_size, int num_blocks, float softmax_scale);

}  // namespace ignis

// The opaque graph handle (ADR 0001 — opaque across the C ABI boundary;
// the Rust side mirrors it as a zero-sized struct and only ever holds a
// pointer to it). The leaf-internal layout owns the captured graph, its
// instantiated executable, and (when the leaf created the capture stream)
// the stream itself.
struct ignis_graph {
  cudaGraph_t graph;         // the captured graph (retained for inspection)
  cudaGraphExec_t exec;      // the instantiated executable (what gets launched)
  cudaStream_t capture_stream;  // the non-blocking stream the capture ran on
  int owns_stream;           // 1 = the leaf created capture_stream (to free it)
};

namespace {

// --------------------------------------------------------------------------
// Leaf-internal capture pairing (single capture at a time — v1's eager
// startup capture is a single-shot on the driver thread; lazy / concurrent
// capture is a later optimization, design §1). `g_pending` holds the stream
// (and ownership) of the active capture, set by ignis_graph_begin_capture
// and consumed by ignis_graph_end_capture.
// --------------------------------------------------------------------------
struct PendingCapture {
  cudaStream_t stream = nullptr;
  int owns_stream = 0;  // 1 = the leaf created the stream (destroy it on end/destroy)
  int active = 0;
} g_pending;

int graph_report(cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[ignis-kernel] CUDA error: %s\n", cudaGetErrorString(err));
  return -1;
}

// --- Startup-check geometry: a representative prefill + decode step ------
// Small synthetic shapes (a few KB of VRAM — runs even with the model
// loaded, the ADR 0006 nuance). The prefill path is the ticket-05 batched
// GQA prefill attention + the GDN linear-attention step (the per-layer
// recurrent update); the decode path is the ticket-03 single-token GQA
// decode attention.
constexpr int kBatch = 1;
constexpr int kSeqLen = 8;
constexpr int kNumQHeads = 4;
constexpr int kNumKVHeads = 2;
constexpr int kHeadDim = 8;
constexpr int kBlockSize = 4;
constexpr int kNumBlocks = 8;
constexpr int kGdnLayers = 1;
constexpr int kStateRows = 4;
constexpr int kStateCols = 4;
constexpr int kStateDim = kStateCols + kStateRows + 2;  // the k / v / g / beta block

constexpr std::size_t kPrefillQ =
    static_cast<std::size_t>(kBatch) * kSeqLen * kNumQHeads * kHeadDim;  // 256
constexpr std::size_t kPrefillKV =
    2 * static_cast<std::size_t>(kBatch) * kNumBlocks * kBlockSize * kNumKVHeads * kHeadDim;  // 1024
constexpr std::size_t kPrefillTable = static_cast<std::size_t>(kBatch) * kNumBlocks;  // 8
constexpr std::size_t kGdnX = static_cast<std::size_t>(kBatch) * kStateDim;  // 10
constexpr std::size_t kGdnState =
    static_cast<std::size_t>(kBatch) * kGdnLayers * kStateRows * kStateCols;  // 16
constexpr std::size_t kDecodeQ = static_cast<std::size_t>(kNumQHeads) * kHeadDim;  // 32
constexpr std::size_t kDecodeKV =
    2 * static_cast<std::size_t>(kNumBlocks) * kBlockSize * kNumKVHeads * kHeadDim;  // 1024
constexpr std::size_t kDecodeTable = kNumBlocks;  // 8

// The startup check's device buffers (inputs are H2D'd before the capture
// window; the outputs are written by the kernels and read back per phase).
struct StepBuffers {
  __nv_bfloat16* pf_q;
  __nv_bfloat16* pf_kv;
  std::int32_t* pf_table;
  __nv_bfloat16* pf_out;
  __nv_bfloat16* gdn_x;
  __nv_bfloat16* gdn_state_in;
  __nv_bfloat16* gdn_state_out;
  __nv_bfloat16* dec_q;
  __nv_bfloat16* dec_kv;
  std::int32_t* dec_table;
  __nv_bfloat16* dec_out;
};

// Allocate the startup check's device buffers (a few KB total).
cudaError_t alloc_step(StepBuffers& b) {
  cudaError_t e;
  e = cudaMalloc(&b.pf_q, kPrefillQ * sizeof(__nv_bfloat16));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.pf_kv, kPrefillKV * sizeof(__nv_bfloat16));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.pf_table, kPrefillTable * sizeof(std::int32_t));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.pf_out, kPrefillQ * sizeof(__nv_bfloat16));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.gdn_x, kGdnX * sizeof(__nv_bfloat16));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.gdn_state_in, kGdnState * sizeof(__nv_bfloat16));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.gdn_state_out, kGdnState * sizeof(__nv_bfloat16));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.dec_q, kDecodeQ * sizeof(__nv_bfloat16));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.dec_kv, kDecodeKV * sizeof(__nv_bfloat16));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.dec_table, kDecodeTable * sizeof(std::int32_t));
  if (e != cudaSuccess) return e;
  e = cudaMalloc(&b.dec_out, kDecodeQ * sizeof(__nv_bfloat16));
  if (e != cudaSuccess) return e;
  return cudaSuccess;
}

// Free the startup check's device buffers (null-safe).
void free_step(const StepBuffers& b) {
  if (b.pf_q != nullptr) cudaFree(b.pf_q);
  if (b.pf_kv != nullptr) cudaFree(b.pf_kv);
  if (b.pf_table != nullptr) cudaFree(b.pf_table);
  if (b.pf_out != nullptr) cudaFree(b.pf_out);
  if (b.gdn_x != nullptr) cudaFree(b.gdn_x);
  if (b.gdn_state_in != nullptr) cudaFree(b.gdn_state_in);
  if (b.gdn_state_out != nullptr) cudaFree(b.gdn_state_out);
  if (b.dec_q != nullptr) cudaFree(b.dec_q);
  if (b.dec_kv != nullptr) cudaFree(b.dec_kv);
  if (b.dec_table != nullptr) cudaFree(b.dec_table);
  if (b.dec_out != nullptr) cudaFree(b.dec_out);
}

// H2D the startup check's inputs (the outputs are written by the kernels).
// `h` is a bundle of the host input buffers (see ignis_graph_startup_check).
struct StepHostInputs {
  std::vector<__nv_bfloat16> pf_q;
  std::vector<__nv_bfloat16> pf_kv;
  std::vector<std::int32_t> pf_table;
  std::vector<__nv_bfloat16> gdn_x;
  std::vector<__nv_bfloat16> gdn_state_in;
  std::vector<__nv_bfloat16> dec_q;
  std::vector<__nv_bfloat16> dec_kv;
  std::vector<std::int32_t> dec_table;
};

cudaError_t h2d_inputs(const StepBuffers& d, const StepHostInputs& h) {
  cudaError_t e;
  e = cudaMemcpy(d.pf_q, h.pf_q.data(), kPrefillQ * sizeof(__nv_bfloat16),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(d.pf_kv, h.pf_kv.data(), kPrefillKV * sizeof(__nv_bfloat16),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(d.pf_table, h.pf_table.data(), kPrefillTable * sizeof(std::int32_t),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(d.gdn_x, h.gdn_x.data(), kGdnX * sizeof(__nv_bfloat16),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(d.gdn_state_in, h.gdn_state_in.data(), kGdnState * sizeof(__nv_bfloat16),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(d.dec_q, h.dec_q.data(), kDecodeQ * sizeof(__nv_bfloat16),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(d.dec_kv, h.dec_kv.data(), kDecodeKV * sizeof(__nv_bfloat16),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(d.dec_table, h.dec_table.data(), kDecodeTable * sizeof(std::int32_t),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  return cudaSuccess;
}

// D2H the three outputs (prefill out, GDN state out, decode out).
cudaError_t d2h_outputs(const StepBuffers& d, std::vector<__nv_bfloat16>& out_prefill,
                        std::vector<__nv_bfloat16>& out_gdn, std::vector<__nv_bfloat16>& out_decode) {
  cudaError_t e;
  e = cudaMemcpy(out_prefill.data(), d.pf_out, kPrefillQ * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(out_gdn.data(), d.gdn_state_out, kGdnState * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) return e;
  e = cudaMemcpy(out_decode.data(), d.dec_out, kDecodeQ * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToHost);
  return e;
}

// Launch a representative "prefill + decode" step on the stream: GQA prefill
// attention (the prefill path) + GDN linear-attention step (the per-layer
// recurrent update) + GQA decode attention (the decode path). The launch
// configurations mirror the ticket-05/03 surfaces 1:1 (grid / block /
// dynamic-smem geometry and argument order, see prefill_gdn_surface.cu and
// decode_surface.cu). Used both for the eager run and inside the capture.
cudaError_t launch_step(cudaStream_t s, const StepBuffers& b) {
  cudaError_t e;
  // GQA prefill: one block per (q head, position, batch); one thread per
  // head_dim element. Dynamic smem: head_dim floats for the per-key block
  // reduce (see gqa_attention_prefill.cuh).
  {
    const dim3 grid(static_cast<unsigned>(kNumQHeads), static_cast<unsigned>(kSeqLen),
                    static_cast<unsigned>(kBatch));
    const unsigned threads = static_cast<unsigned>(kHeadDim);
    const unsigned smem = static_cast<unsigned>(kHeadDim * sizeof(float));
    ignis::gqa_attention_prefill_kernel<<<grid, threads, smem, s>>>(
        b.pf_kv, b.pf_table, b.pf_q, b.pf_out,
        kBatch, kSeqLen, kNumQHeads, kNumKVHeads, kHeadDim, kBlockSize, kNumBlocks, 1.0f);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
  }
  // GDN step: one block per (dv row, batch*layer); one thread per d_k column.
  // Dynamic smem: state_cols floats for the per-row block reduce (see
  // gdn_step.cuh).
  {
    const dim3 grid(static_cast<unsigned>(kStateRows),
                    static_cast<unsigned>(kBatch * kGdnLayers));
    const unsigned threads = static_cast<unsigned>(kStateCols);
    const unsigned smem = static_cast<unsigned>(kStateCols * sizeof(float));
    ignis::gdn_step_kernel<<<grid, threads, smem, s>>>(
        b.gdn_x, b.gdn_state_in, b.gdn_state_out,
        kBatch, kGdnLayers, kStateRows, kStateCols, kStateDim);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
  }
  // GQA decode: one block (head_dim threads) per q head (static shared in
  // the kernel, so dynamic smem = 0; see gqa_attention_decode.cuh).
  {
    ignis::gqa_attention_decode_kernel<<<static_cast<unsigned>(kNumQHeads),
                                          static_cast<unsigned>(kHeadDim), 0, s>>>(
        b.dec_kv, b.dec_table, b.dec_q, b.dec_out,
        kNumQHeads, kNumKVHeads, kHeadDim, kSeqLen, kBlockSize, kNumBlocks, 1.0f);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
  }
  return cudaSuccess;
}

// Bit-exact bf16 comparison of two host buffers (the eager and graph runs
// use the same kernels + inputs, so the outputs must be bit-identical — a
// divergence is a real failure, not a tolerance issue).
bool bf16_equal(const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    uint16_t va, vb;
    std::memcpy(&va, &a[i], sizeof(uint16_t));
    std::memcpy(&vb, &b[i], sizeof(uint16_t));
    if (va != vb) return false;
  }
  return true;
}

}  // namespace

// Ticket 10 (kernel-abi-03): begin a CUDA-graph capture on `stream` (null =
// a leaf-owned non-blocking stream — the legacy default stream cannot be
// captured). The caller issues the prefill/decode kernel launches on the
// same stream while the capture is active, then calls
// ignis_graph_end_capture to materialize the graph. Returns 0 on success,
// -1 on error (a capture already in progress, or a CUDA error).
extern "C" int ignis_graph_begin_capture(void* stream) {
  if (g_pending.active != 0) {
    return -1;  // a capture is already active (single capture at a time, v1)
  }

  cudaStream_t s;
  int owns;
  if (stream != nullptr) {
    s = static_cast<cudaStream_t>(stream);  // the caller's (non-blocking) stream
    owns = 0;
  } else {
    // Leaf-owned non-blocking stream (destroyed by ignis_graph_destroy when
    // the graph handle is destroyed, or by a failed end below).
    s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
      return -1;  // no GPU / CUDA error — the caller self-skips (ADR 0006)
    }
    owns = 1;
  }

  // Reject a double capture on this stream (even if the leaf's pairing state
  // did not see the begin — e.g. a stream the caller began elsewhere).
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaError_t err = cudaStreamGetCaptureInfo(s, &status);
  if (err != cudaSuccess) {
    if (owns != 0) cudaStreamDestroy(s);
    return graph_report(err);  // a real CUDA error (report for debuggability)
  }
  if (status != cudaStreamCaptureStatusNone) {
    if (owns != 0) cudaStreamDestroy(s);
    return -1;  // a capture is already active on this stream (a guard, not a CUDA error)
  }

  // Thread-local capture: the captured work is confined to the capturing
  // thread (other threads / streams are unaffected; the startup path is
  // single-threaded, v1).
  err = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    if (owns != 0) cudaStreamDestroy(s);
    return graph_report(err);
  }

  g_pending.stream = s;
  g_pending.owns_stream = owns;
  g_pending.active = 1;
  return 0;
}

// Ticket 10 (kernel-abi-03): end the capture, materializing the graph into
// *out (an executable graph). `stream` must match the stream passed to
// ignis_graph_begin_capture (null = the leaf-owned stream). Returns 0 on
// success, -1 on error (no active capture, a stream mismatch, or a CUDA
// error).
extern "C" int ignis_graph_end_capture(void* stream, struct ignis_graph** out) {
  if (out == nullptr) return -1;
  *out = nullptr;

  if (g_pending.active == 0) {
    return -1;  // no active capture (begin was not called, or already ended)
  }
  if (stream != nullptr && stream != static_cast<void*>(g_pending.stream)) {
    return -1;  // stream mismatch (the capture must end on its own stream)
  }

  const cudaStream_t s = g_pending.stream;
  g_pending.active = 0;  // release the pairing before the (slow) end call

  cudaGraph_t graph = nullptr;
  cudaError_t err = cudaStreamEndCapture(s, &graph);
  if (err != cudaSuccess) {
    // The capture was aborted (thread mode releases the stream on error);
    // a partial graph may still materialize — destroy it defensively.
    if (graph != nullptr) cudaGraphDestroy(graph);
    if (g_pending.owns_stream != 0) {
      cudaStreamDestroy(s);
      g_pending.owns_stream = 0;
    }
    return graph_report(err);
  }

  cudaGraphExec_t exec = nullptr;
  err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  if (err != cudaSuccess || exec == nullptr) {
    cudaGraphDestroy(graph);
    if (g_pending.owns_stream != 0) {
      cudaStreamDestroy(s);
      g_pending.owns_stream = 0;
    }
    return graph_report(err);
  }

  ignis_graph* g = static_cast<ignis_graph*>(std::malloc(sizeof(ignis_graph)));
  if (g == nullptr) {
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    if (g_pending.owns_stream != 0) {
      cudaStreamDestroy(s);
      g_pending.owns_stream = 0;
    }
    return -1;  // host OOM — not a CUDA error; report as a failure
  }
  g->graph = graph;
  g->exec = exec;
  g->capture_stream = s;
  g->owns_stream = g_pending.owns_stream;
  g_pending.owns_stream = 0;
  *out = g;
  return 0;
}

// Ticket 10 (kernel-abi-03): launch a captured graph on `stream` (null =
// the graph's own capture stream — the legacy default stream is avoided for
// graph launches). Returns 0 on success, -1 on error (a null graph handle
// is a clean -1, before any CUDA call).
extern "C" int ignis_graph_launch(struct ignis_graph* g, void* stream) {
  if (g == nullptr || g->exec == nullptr) {
    return -1;  // the null guard runs before any CUDA call (CPU-verifiable)
  }
  cudaStream_t s;
  if (stream != nullptr) {
    s = static_cast<cudaStream_t>(stream);
  } else {
    s = g->capture_stream;  // the graph's home (non-blocking) stream
  }
  return graph_report(cudaGraphLaunch(g->exec, s));
}

// Ticket 10 (kernel-abi-03): destroy a captured graph (and, when the leaf
// created the capture stream, the stream). NULL is a no-op.
extern "C" void ignis_graph_destroy(struct ignis_graph* g) {
  if (g == nullptr) return;  // NULL is a no-op (no CUDA calls)
  if (g->exec != nullptr) cudaGraphExecDestroy(g->exec);
  if (g->graph != nullptr) cudaGraphDestroy(g->graph);
  if (g->owns_stream != 0 && g->capture_stream != nullptr) {
    cudaStreamDestroy(g->capture_stream);
  }
  std::free(g);
}

// Ticket 10 (kernel-abi-03): the startup verification. Captures a
// representative prefill + decode kernel sequence into a CUDA graph, runs
// the same sequence eagerly and via graph replay, and confirms the replayed
// outputs match the eager outputs bit-exactly (the capture mechanism is
// verified; the canary-suite 99% performance gate is ADR 0007, driven by
// ignis-bench — ticket 20).
//
// stream: null = stream 0 for the eager phase (the capture itself runs on
// the leaf-owned non-blocking stream). Returns 0 if the capture verified and
// replay ≡ eager, -1 on a CUDA error (GPU unavailable / busy — the caller
// self-skips, ADR 0006), -2 if the capture succeeded but the replayed
// outputs diverged from the eager outputs (a real failure — the graph path
// is broken; not a skip condition).
extern "C" int ignis_graph_startup_check(void* stream) {
  const cudaStream_t eager_s = static_cast<cudaStream_t>(stream);  // null = stream 0

  // --- Deterministic synthetic inputs (bf16-exact, index-varying — the
  //     ticket-05 test style; multiples of 0.25 are exact in bf16). -------
  StepHostInputs h;
  h.pf_q.resize(kPrefillQ);
  h.pf_kv.resize(kPrefillKV);
  h.pf_table.resize(kPrefillTable);
  h.gdn_x.resize(kGdnX);
  h.gdn_state_in.resize(kGdnState);
  h.dec_q.resize(kDecodeQ);
  h.dec_kv.resize(kDecodeKV);
  h.dec_table.resize(kDecodeTable);
  for (std::size_t i = 0; i < h.pf_q.size(); ++i) {
    h.pf_q[i] = __float2bfloat16_rn(0.25f * static_cast<float>(i % 16) + 0.25f);
  }
  for (std::size_t i = 0; i < h.pf_kv.size(); ++i) {
    h.pf_kv[i] = __float2bfloat16_rn(0.125f * static_cast<float>(i % 16) + 0.25f);
  }
  for (std::size_t i = 0; i < h.pf_table.size(); ++i) {
    h.pf_table[i] = static_cast<std::int32_t>(i % kNumBlocks);  // valid page ids
  }
  // GDN x decomposes (k, v, g, beta) per batch (see gdn_step.cuh): keep the
  // gate pre-decay g <= 0 (a contraction) and beta positive.
  for (std::size_t i = 0; i < h.gdn_x.size(); ++i) {
    h.gdn_x[i] = __float2bfloat16_rn(0.25f * static_cast<float>(i % 4));
  }
  for (std::size_t bi = 0; bi < static_cast<std::size_t>(kBatch); ++bi) {
    const std::size_t base = bi * static_cast<std::size_t>(kStateDim);
    h.gdn_x[base + kStateCols + kStateRows] = __float2bfloat16_rn(-0.5f);  // g
    h.gdn_x[base + kStateCols + kStateRows + 1] = __float2bfloat16_rn(1.0f);  // beta
  }
  for (std::size_t i = 0; i < h.gdn_state_in.size(); ++i) {
    h.gdn_state_in[i] = __float2bfloat16_rn(0.25f * static_cast<float>(i % 4));
  }
  for (std::size_t i = 0; i < h.dec_q.size(); ++i) {
    h.dec_q[i] = __float2bfloat16_rn(0.25f * static_cast<float>(i % 8));
  }
  for (std::size_t i = 0; i < h.dec_kv.size(); ++i) {
    h.dec_kv[i] = __float2bfloat16_rn(0.125f * static_cast<float>(i % 8) + 0.125f);
  }
  for (std::size_t i = 0; i < h.dec_table.size(); ++i) {
    h.dec_table[i] = static_cast<std::int32_t>(i % kNumBlocks);  // valid page ids
  }

  // --- Device buffers (a few KB total — runs even with the model loaded,
  //     the ADR 0006 nuance). ---------------------------------------------
  StepBuffers d{};  // value-initialized (null pointers) so free_step is safe
  cudaError_t err = alloc_step(d);
  if (err != cudaSuccess) {
    free_step(d);  // release the subset of buffers that did allocate (no leak on partial OOM)
    return -1;  // OOM / device unavailable — the caller self-skips (ADR 0006)
  }
  err = h2d_inputs(d, h);  // the inputs are H2D'd BEFORE the capture window
  if (err != cudaSuccess) {
    free_step(d);
    return -1;
  }

  // --- Eager phase: the prefill + decode step on `eager_s` ---------------
  std::vector<__nv_bfloat16> eager_prefill(kPrefillQ);
  std::vector<__nv_bfloat16> eager_gdn(kGdnState);
  std::vector<__nv_bfloat16> eager_decode(kDecodeQ);
  err = launch_step(eager_s, d);
  if (err == cudaSuccess) err = cudaStreamSynchronize(eager_s);
  if (err == cudaSuccess) {
    err = d2h_outputs(d, eager_prefill, eager_gdn, eager_decode);
  }
  if (err != cudaSuccess) {
    free_step(d);
    return -1;
  }

  // --- Graph phase: capture the same step, replay it, read back ----------
  ignis_graph* g = nullptr;
  // Begin the capture on the leaf-owned non-blocking stream (null = the
  // leaf creates it; it is owned by the graph handle and destroyed by
  // ignis_graph_destroy).
  int rc = ignis_graph_begin_capture(nullptr);
  if (rc != 0) {
    free_step(d);
    return -1;
  }
  // The captured launches must run on the capture stream (the leaf-owned
  // stream recorded by begin — same-TU access to the leaf-internal state).
  const cudaError_t step_err = launch_step(g_pending.stream, d);
  // Always end the capture: a launch failure during the capture aborts it
  // (thread mode), and the end call releases the leaf pairing + the owned
  // stream on either path (its error path cleans up the leaf state).
  rc = ignis_graph_end_capture(nullptr, &g);  // null = the leaf-owned stream
  if (rc != 0 || g == nullptr || step_err != cudaSuccess) {
    ignis_graph_destroy(g);  // a no-op when g is null
    free_step(d);
    return -1;
  }

  // Replay the graph on its own capture stream (null = the graph's home
  // stream — a clean apples-to-apples comparison with the eager phase).
  const int replay_rc = ignis_graph_launch(g, nullptr);
  cudaError_t rep_err = cudaSuccess;
  if (replay_rc == 0) {
    rep_err = cudaStreamSynchronize(g->capture_stream);
  }
  std::vector<__nv_bfloat16> graph_prefill(kPrefillQ);
  std::vector<__nv_bfloat16> graph_gdn(kGdnState);
  std::vector<__nv_bfloat16> graph_decode(kDecodeQ);
  if (replay_rc == 0 && rep_err == cudaSuccess) {
    rep_err = d2h_outputs(d, graph_prefill, graph_gdn, graph_decode);
  }
  if (replay_rc != 0 || rep_err != cudaSuccess) {
    ignis_graph_destroy(g);
    free_step(d);
    return -1;  // a CUDA error on replay / D2H (device busy / OOM) — the
                 // caller self-skips (ADR 0006)
  }

  // --- Confirm replay ≡ eager (bit-exact: same kernels + same inputs) ---
  const bool matched = bf16_equal(eager_prefill, graph_prefill) &&
                       bf16_equal(eager_gdn, graph_gdn) &&
                       bf16_equal(eager_decode, graph_decode);
  ignis_graph_destroy(g);
  free_step(d);
  if (!matched) {
    std::fprintf(stderr,
                 "[ignis-kernel] graph startup check: the graph replay diverged "
                 "from the eager path (the capture mechanism is broken)\n");
    return -2;
  }
  return 0;
}