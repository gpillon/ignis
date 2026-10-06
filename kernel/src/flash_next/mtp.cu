// ignis kernel leaf -- the Flash-Next MTP head's step (spec flash-next/07 phase D, GitHub #307;
// OURS, ADR 0043). See mtp.h for the math. Kernels here: the head's RMSNorm (grouped or not) and
// the fp32 sum of its two projections; everything else is the trunk's ops.

#include "mtp.h"

#include "embed_head.h"
#include "hc.h"

#include "ninfer/ops/argmax.h"

#include <cuda_bf16.h>

#include <exception>
#include <string>

namespace ignis::flash_next::mtp {

namespace {

constexpr int32_t kThreads = 256;

std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

__device__ __forceinline__ float block_sum(float v, float *shared) {
  for (int offset = 16; offset > 0; offset /= 2) v += __shfl_xor_sync(0xffffffffU, v, offset);
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  float total = 0.0F;
  if (threadIdx.x == 0) {
    for (int w = 0; w < static_cast<int>(blockDim.x) / 32; ++w) total += shared[w];
    shared[0] = total;
  }
  __syncthreads();
  return shared[0];
}

// One CTA per (row, group) of `width` elements: y = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)),
// w indexed by the group's own columns. x rows `in_stride` apart, y rows `out_stride` apart.
__global__ void rms_kernel(const __nv_bfloat16 *__restrict__ x, int64_t in_stride, const __nv_bfloat16 *__restrict__ w,
                           float eps, int32_t groups, int32_t width, __nv_bfloat16 *__restrict__ y,
                           int64_t out_stride) {
  __shared__ float partial[kThreads / 32];
  const int32_t row = static_cast<int32_t>(blockIdx.x) / groups;
  const int32_t group = static_cast<int32_t>(blockIdx.x) % groups;
  const __nv_bfloat16 *in = x + row * in_stride + static_cast<int64_t>(group) * width;
  __nv_bfloat16 *out = y + row * out_stride + static_cast<int64_t>(group) * width;
  const __nv_bfloat16 *weight = w + static_cast<int64_t>(group) * width;
  float ss = 0.0F;
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < width; i += blockDim.x) {
    const float v = __bfloat162float(in[i]);
    ss = fmaf(v, v, ss);
  }
  const float r = rsqrtf(block_sum(ss, partial) / static_cast<float>(width) + eps);
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < width; i += blockDim.x) {
    out[i] = __float2bfloat16_rn(__bfloat162float(in[i]) * r * (1.0F + __bfloat162float(weight[i])));
  }
}

// X[r][s][i] = bf16(hidden[r][s][i] + embedding[r][i]), both BF16, summed in fp32.
__global__ void sum_kernel(const __nv_bfloat16 *__restrict__ hidden, const __nv_bfloat16 *__restrict__ embedding,
                           int32_t streams, int32_t width, int64_t n, __nv_bfloat16 *__restrict__ x) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / (static_cast<int64_t>(streams) * width);
  const int64_t col = i % width;
  x[i] = __float2bfloat16_rn(__bfloat162float(hidden[i]) + __bfloat162float(embedding[row * width + col]));
}

// One CTA per lane: its first draft and its chain stack, 16 bytes per thread.
__global__ void first_drafts_kernel(const int32_t *__restrict__ picks, const uint4 *__restrict__ residual,
                                    const int32_t *__restrict__ commit, int32_t columns, int32_t width,
                                    int32_t row_vectors, int32_t *__restrict__ drafts,
                                    int32_t *__restrict__ chain_tokens, uint4 *__restrict__ stack) {
  const int32_t lane = static_cast<int32_t>(blockIdx.x);
  const int32_t row = lane * columns + commit[lane] - 1;
  if (threadIdx.x == 0) {
    drafts[lane * width] = picks[row];
    chain_tokens[lane] = picks[row];
  }
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < row_vectors; i += blockDim.x) {
    stack[static_cast<int64_t>(lane) * row_vectors + i] = residual[static_cast<int64_t>(row) * row_vectors + i];
  }
}

__global__ void append_drafts_kernel(const int32_t *__restrict__ chain_tokens, int32_t lanes, int32_t width,
                                     int32_t step, int32_t *__restrict__ drafts) {
  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  if (lane < lanes) drafts[lane * width + step] = chain_tokens[lane];
}

int32_t launched(const char *what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return 0;
  fn_set_error(std::string("mtp ") + what + ": " + cudaGetErrorString(err));
  return -1;
}

}  // namespace

std::size_t combine_scratch_bytes(const Geometry &g, int32_t rows) {
  const auto r = static_cast<std::size_t>(rows);
  const std::size_t wide = aligned(r * g.residual_width() * sizeof(__nv_bfloat16));
  const std::size_t narrow = aligned(r * g.hidden * sizeof(__nv_bfloat16));
  // The embedded stacks, the normed embedding and its projection, the normed stacks and theirs
  // (fn_linear takes no scratch of its own).
  return wide + 2 * narrow + 2 * wide;
}

int32_t combine(const Geometry &g, const MtpWeights &w, const Linear &embed, const int32_t *tokens,
                const Batch &batch, const Buffers &b, ninfer::DeviceArena &scratch, cudaStream_t stream) {
  const int32_t rows = batch.rows();
  const auto r = static_cast<std::size_t>(rows);
  try {
    auto scope = scratch.scope();
    auto *embedded = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * g.residual_width() * 2).data);
    auto *e = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * g.hidden * 2).data);
    auto *e_proj = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * g.hidden * 2).data);
    auto *n = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * g.residual_width() * 2).data);
    auto *n_proj = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * g.residual_width() * 2).data);
    // The token embedding (fn_embed writes it into every stream; stream 0 is read).
    if (fn_embed(g, embed, tokens, rows, embedded, stream) != 0) return -1;
    rms_kernel<<<rows, kThreads, 0, stream>>>(embedded, g.residual_width(),
                                              static_cast<const __nv_bfloat16 *>(w.norm_embedding), g.rms_norm_eps, 1,
                                              g.hidden, e, g.hidden);
    if (launched("embedding norm") != 0) return -1;
    rms_kernel<<<rows * g.streams, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16 *>(b.residual),
                                                          g.residual_width(),
                                                          static_cast<const __nv_bfloat16 *>(w.norm_hidden),
                                                          g.rms_norm_eps, g.streams, g.hidden, n, g.residual_width());
    if (launched("stack norm") != 0) return -1;
    if (fn_linear(w.fc_embedding, e, rows, e_proj, false, scratch, stream) != 0) return -1;
    if (fn_linear(w.fc_hidden, n, rows * g.streams, n_proj, false, scratch, stream) != 0) return -1;
    const int64_t total = static_cast<int64_t>(rows) * g.residual_width();
    sum_kernel<<<static_cast<unsigned>((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        n_proj, e_proj, g.streams, g.hidden, total, static_cast<__nv_bfloat16 *>(b.residual));
    return launched("combine");
  } catch (const std::exception &ex) {
    fn_set_error(std::string("mtp combine: ") + ex.what());
    return -1;
  }
}

int32_t layer(const Context &ctx, int32_t attention_ordinal, const MtpWeights &w, const ignis_moe_slot *experts,
              const Batch &batch, const Buffers &b, ninfer::DeviceArena &scratch, cudaStream_t stream) {
  const Geometry &g = ctx.g;
  const LayerWeights &l = w.layer;
  const int32_t rows = batch.rows();
  const auto fail = [](const std::string &what, const char *detail) {
    fn_set_error("mtp layer: " + what + ": " + detail);
    return -1;
  };
  // The attention sublayer, on the head's own section.
  if (fn_hc_mix(g, l.attn_hc, b.residual, rows, b.x, b.inj, scratch, stream) != 0) {
    return fail("attention mix", fn_last_error());
  }
  {
    auto scope = scratch.scope();
    Selection selection;
    try {
      selection.tokens = static_cast<int32_t *>(
          scratch.alloc_bytes(static_cast<std::size_t>(rows) * g.selection_width() * sizeof(int32_t)).data);
      selection.counts = static_cast<int32_t *>(scratch.alloc_bytes(static_cast<std::size_t>(rows) * sizeof(int32_t)).data);
    } catch (const std::exception &e) {
      return fail("selection", e.what());
    }
    if (fn_indexer_select(ctx, attention_ordinal, l.qsa.indexer, batch, b.x, selection, scratch, stream) != 0) {
      return fail("indexer", fn_last_error());
    }
    if (fn_qsa_attention(ctx, attention_ordinal, l.qsa, batch, b.x, selection, b.y, scratch, stream) != 0) {
      return fail("attention", fn_last_error());
    }
  }
  if (fn_hc_inject(g, b.y, b.inj, rows, b.residual, stream) != 0) return fail("attention inject", fn_last_error());

  // The MoE sublayer: the router, every selected expert resident, the shared expert in line.
  if (fn_hc_mix(g, l.mlp_hc, b.residual, rows, b.x, b.inj, scratch, stream) != 0) {
    return fail("MoE mix", fn_last_error());
  }
  if (ignis_moe_shared_expert(l.moe.shared_gate, l.moe.shared_up, l.moe.shared_down, b.x, static_cast<uint32_t>(rows),
                              b.shared_h, b.shared_out, stream) != 0) {
    return fail("shared expert", ignis_moe_last_error());
  }
  if (ignis_moe_router(b.x, static_cast<uint32_t>(rows), l.moe.router, b.router_ids, b.router_weights, b.router_logits,
                       stream) != 0) {
    return fail("router", ignis_moe_last_error());
  }
  const bool decode = static_cast<uint32_t>(rows) <= b.workspace.decode_tokens;
  const int32_t rc = decode ? ignis_moe_experts_decode(b.x, static_cast<uint32_t>(rows), b.router_ids, b.router_weights,
                                                       experts, &b.workspace, b.acc, stream)
                            : ignis_moe_experts_prefill(b.x, static_cast<uint32_t>(rows), b.router_ids,
                                                        b.router_weights, experts, &b.workspace, b.acc, stream);
  // The routed accumulator must be zero on every op's entry (ignis_moe.h).
  const auto fail_moe = [&](const char *what) {
    const std::string detail = ignis_moe_last_error();
    (void)cudaMemsetAsync(b.acc, 0, static_cast<std::size_t>(rows) * g.hidden * sizeof(int64_t), stream);
    return fail(what, detail.c_str());
  };
  if (rc != 0) return fail_moe("experts");
  if (ignis_moe_combine(b.acc, b.shared_out, b.x, l.moe.shared_expert_gate, static_cast<uint32_t>(rows), b.y, stream) !=
      0) {
    return fail_moe("combine");
  }
  if (fn_hc_inject(g, b.y, b.inj, rows, b.residual, stream) != 0) return fail("MoE inject", fn_last_error());
  return 0;
}

int32_t argmax(const Geometry &g, const void *logits, int32_t rows, int32_t *out, cudaStream_t stream) {
  try {
    const ninfer::Tensor all(const_cast<void *>(logits), ninfer::DType::BF16, {g.vocab, rows, 1, 1});
    ninfer::Tensor picks(out, ninfer::DType::I32, {rows, 1, 1, 1});
    ninfer::ops::argmax(all, picks, g.vocab, stream);
  } catch (const std::exception &e) {
    fn_set_error(std::string("mtp argmax: ") + e.what());
    return -1;
  }
  return 0;
}

int32_t first_drafts(const Geometry &g, const int32_t *picks, const void *residual, const int32_t *commit,
                     int32_t lanes, int32_t columns, int32_t width, int32_t *drafts, int32_t *chain_tokens,
                     void *chain_stack, cudaStream_t stream) {
  const int32_t row_vectors = g.residual_width() * 2 / 16;
  first_drafts_kernel<<<lanes, kThreads, 0, stream>>>(picks, static_cast<const uint4 *>(residual), commit, columns,
                                                       width, row_vectors, drafts, chain_tokens,
                                                       static_cast<uint4 *>(chain_stack));
  return launched("first drafts");
}

int32_t append_drafts(const int32_t *chain_tokens, int32_t lanes, int32_t width, int32_t step, int32_t *drafts,
                      cudaStream_t stream) {
  append_drafts_kernel<<<1, 32, 0, stream>>>(chain_tokens, lanes, width, step, drafts);
  return launched("append drafts");
}

}  // namespace ignis::flash_next::mtp
