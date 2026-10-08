// ignis kernel leaf -- the Flash-Next QSA attention sublayer (spec flash-next/04, GitHub #302,
// slice S2): OURS (ADR 0043), no port claim. See qsa.h for the oracle and the call's order. This
// file: the norms + rope, the K/V append in both formats, the gate, and fn_qsa_attention; the
// dense attention kernel is qsa_dense.cu. Since GitHub #306's fusion (step 6, fusion.h's Qsa) a
// call of up to 8 rows takes q, k and v in one grouped GEMV launch and a split sparse call's
// combine takes the gate. The append stays a launch of its own: under hq-e8-2b it must follow the
// listed-row decode (it rewrites the residual-window slot of position - kGqaHqRecentKeys, which
// that decode may read: #258), and under BF16 precede the attention (which reads its page).
//
// The hq-e8-2b append encodes each row with the vendored codec device functions (hq_codec.cuh:
// the engine sign diagonal, hq_encode_row_warp with the (kv head, position, role) dither seed)
// and keeps the residual window exactly as the vendored fill does (gqa_attention_prefill_hq.cuh):
// sink keys, and the keys within kGqaHqRecentKeys of the lane's frontier after the call, stored in
// the rotated frame with their ring bits -- a chunk wider than the ring writes only its last
// kGqaHqRecentKeys keys there, so no two rows of a call race for one ring slot.

#include "qsa.h"

#include "fusion.h"

#include "ignis_fp8_linear.h"
#include "ignis_seq_internal.h"

#include "ninfer/ops/gqa_attention.h"
#include "ops/kernel/hq_codec.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <exception>
#include <string>

namespace ignis::flash_next {

namespace qsa {

namespace {

__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = 16; offset > 0; offset /= 2) v += __shfl_xor_sync(0xffffffffU, v, offset);
  return v;
}

__device__ __forceinline__ int32_t row_position(const int32_t *positions, int32_t tokens, int32_t row) {
  return positions[row / tokens] + row % tokens;
}

struct PrepareArgs {
  indexer::Rope rope;
  const __nv_bfloat16 *q_norm;
  const __nv_bfloat16 *k_norm;
  const int32_t *positions;
  int32_t tokens;
  float eps;
  const __nv_bfloat16 *qg;
  __nv_bfloat16 *q;
  __nv_bfloat16 *k;
};

// One warp per (row, head) of the 24 query heads and the 2 key heads; lane l holds elements
// l + 32 s, so rope pair (i, i + 32) is lane i's x[0], x[1].
__global__ void __launch_bounds__((kQHeads + kKvHeads) * 32) prepare_kernel(PrepareArgs a) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const bool is_q = warp < kQHeads;
  const int head = is_q ? warp : warp - kQHeads;
  const __nv_bfloat16 *src = is_q ? a.qg + (static_cast<int64_t>(row) * kQHeads + head) * 2 * kHeadDim
                                  : a.k + (static_cast<int64_t>(row) * kKvHeads + head) * kHeadDim;
  const __nv_bfloat16 *w = is_q ? a.q_norm : a.k_norm;
  float x[8];
  float ss = 0.0F;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    x[s] = __bfloat162float(src[s * 32 + lane]);
    ss = fmaf(x[s], x[s], ss);
  }
  const float r = 1.0F / sqrtf(warp_sum(ss) / static_cast<float>(kHeadDim) + a.eps);
#pragma unroll
  for (int s = 0; s < 8; ++s) x[s] = round_bf16((x[s] * r) * (1.0F + __bfloat162float(w[s * 32 + lane])));
  const float phi = static_cast<float>(row_position(a.positions, a.tokens, row)) * a.rope.inv_freq[lane];
  const float c = round_bf16(cosf(phi));
  const float sn = round_bf16(sinf(phi));
  const float x1 = x[0], x2 = x[1];
  x[0] = round_bf16(round_bf16(x1 * c) + round_bf16(-x2 * sn));
  x[1] = round_bf16(round_bf16(x2 * c) + round_bf16(x1 * sn));
  __nv_bfloat16 *dst = is_q ? a.q + (static_cast<int64_t>(row) * kQHeads + head) * kHeadDim
                            : a.k + (static_cast<int64_t>(row) * kKvHeads + head) * kHeadDim;
#pragma unroll
  for (int s = 0; s < 8; ++s) dst[s * 32 + lane] = __float2bfloat16_rn(x[s]);
}

struct AppendArgs {
  Kv kv;
  const int32_t *slots;
  const int32_t *positions;
  int32_t tokens;
  int32_t rows;
  const __nv_bfloat16 *k;
  const __nv_bfloat16 *v;
};

// The physical page of `position` in the lane's block-table row; a slot outside the table or an
// unmapped page traps rather than write another sequence's page or residual window.
__device__ __forceinline__ int32_t page_of(const Kv &kv, int32_t slot, int32_t position) {
  const int32_t logical = position >> 6;
  if (slot < 0 || slot >= kv.slots || position < 0 || logical >= kv.logical_pages) __trap();
  const int32_t page = kv.block_tables[static_cast<int64_t>(slot) * kv.logical_pages + logical];
  if (page < 0) __trap();
  return page;
}

// BF16: one thread per 16-byte chunk of every (row, role, head).
__global__ void append_bf16_kernel(AppendArgs a) {
  constexpr int kChunks = kHeadDim / 8;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(a.rows) * 2 * kKvHeads * kChunks) return;
  const int chunk = static_cast<int>(i % kChunks);
  const int head = static_cast<int>((i / kChunks) % kKvHeads);
  const bool role_v = (i / (kChunks * kKvHeads)) % 2 != 0;
  const int32_t row = static_cast<int32_t>(i / (kChunks * kKvHeads * 2));
  const int32_t lane = row / a.tokens;
  const int32_t position = a.positions[lane] + row % a.tokens;
  const int32_t page = page_of(a.kv, a.slots[lane], position);
  auto *plane = static_cast<__nv_bfloat16 *>(role_v ? a.kv.v : a.kv.k);
  const int64_t dst = ((static_cast<int64_t>(page) * kKvHeads + head) * 64 + (position & 63)) * kHeadDim + chunk * 8;
  const int64_t src = (static_cast<int64_t>(row) * kKvHeads + head) * kHeadDim + chunk * 8;
  *reinterpret_cast<uint4 *>(plane + dst) = *reinterpret_cast<const uint4 *>((role_v ? a.v : a.k) + src);
}

constexpr int kHqWarps = 8;
constexpr int32_t kGroupedRows = 8;  // the grouped projection's ceiling: the FP8 GEMV route's

// The vendored residual-row addressing at Flash-Next's KV head count.
struct HqGeometry {
  static constexpr int KVHeads = kKvHeads;
};

// hq-e8-2b: one warp per (row, head, role).
__global__ void __launch_bounds__(kHqWarps * 32) append_hq_kernel(AppendArgs a) {
  using namespace ninfer::ops;
  __shared__ float u_scaled[kHqWarps][kHqSmemFloatsPerRow];
  __shared__ std::uint32_t syms[kHqWarps][kHqSmemSymbolsPerRow];
  __shared__ std::int8_t signs[kHqHeadDim];
  hq_engine_signs_fill(signs);
  __syncthreads();
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int64_t unit = static_cast<int64_t>(blockIdx.x) * kHqWarps + warp;
  if (unit >= static_cast<int64_t>(a.rows) * kKvHeads * 2) return;
  const bool role_v = unit % 2 != 0;
  const int head = static_cast<int>((unit / 2) % kKvHeads);
  const int32_t row = static_cast<int32_t>(unit / (2 * kKvHeads));
  const int32_t seq = row / a.tokens;
  const int32_t slot = a.slots[seq];
  const int32_t position = a.positions[seq] + row % a.tokens;
  const int32_t frontier = a.positions[seq] + a.tokens;
  const int64_t row_at = (static_cast<int64_t>(page_of(a.kv, slot, position)) * kKvHeads + head) * 64 + (position & 63);
  const __nv_bfloat16 *src = (role_v ? a.v : a.k) + (static_cast<int64_t>(row) * kKvHeads + head) * kHeadDim;
  hq_encode_row_warp(src, signs, 0, u_scaled[warp], syms[warp],
                     static_cast<std::uint8_t *>(role_v ? a.kv.v : a.kv.k) + row_at * kHqRowBudgetBytes,
                     static_cast<std::uint8_t *>(role_v ? a.kv.v_meta : a.kv.k_meta) + row_at * kHqMetaBytes,
                     hq_dither_row_seed(head, position, role_v));
  if (a.kv.residual_k != nullptr && (position < static_cast<int32_t>(kGqaHqSinkKeys) ||
                                     position + static_cast<int32_t>(kGqaHqRecentKeys) >= frontier)) {
    hq_store_rotated_row_warp(
        src, signs, hq_residual_row<HqGeometry>(role_v ? a.kv.residual_v : a.kv.residual_k, slot, head, position));
    if (lane == 0 && !role_v && head == 0) {
      hq_ring_mark_valid(a.kv.ring + static_cast<int64_t>(slot) * (kGqaHqRecentKeys / 32), position);
    }
  }
}

// out *= bf16(sigmoid(gate)), rounded once more: torch's attn_output * torch.sigmoid(gate) in
// BF16. One thread per 8 elements.
__global__ void gate_kernel(const __nv_bfloat16 *__restrict__ qg, __nv_bfloat16 *__restrict__ out, int64_t chunks) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= chunks) return;
  const int64_t element = i * 8;
  const int64_t row_head = element / kHeadDim;
  const int64_t d = element % kHeadDim;
  const uint4 gv = *reinterpret_cast<const uint4 *>(qg + row_head * 2 * kHeadDim + kHeadDim + d);
  uint4 ov = *reinterpret_cast<const uint4 *>(out + element);
  const auto *gh = reinterpret_cast<const __nv_bfloat16 *>(&gv);
  auto *oh = reinterpret_cast<__nv_bfloat16 *>(&ov);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float s = round_bf16(1.0F / (1.0F + expf(-__bfloat162float(gh[j]))));
    oh[j] = __float2bfloat16_rn(__bfloat162float(oh[j]) * s);
  }
  *reinterpret_cast<uint4 *>(out + element) = ov;
}

Status launched(const char *what) { return cudaPeekAtLastError() == cudaSuccess ? nullptr : what; }

constexpr std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// The call's activations, in allocation order: qg, k, v, q, out.
std::size_t activation_bytes(int32_t rows) {
  const auto r = static_cast<std::size_t>(rows) * sizeof(__nv_bfloat16);
  return aligned(r * kQProjWidth) + 2 * aligned(r * kKvWidth) + 2 * aligned(r * kOutWidth);
}

bool shaped(const Linear &l, int32_t rows, int32_t cols) {
  return l.data != nullptr && l.rows == rows && l.cols == cols;
}

}  // namespace

Status check_geometry(const Geometry &g) {
  if (g.q_heads != kQHeads || g.kv_heads != kKvHeads || g.head_dim != kHeadDim || g.rotary_dim != kRotaryDim) {
    return "qsa: the attention is written for 24 query heads, 2 KV heads of 256, rotary 64";
  }
  if (g.hidden <= 0 || g.hidden % 64 != 0) return "qsa: hidden must be a positive multiple of 64";
  return nullptr;
}

Status prepare(const Geometry &g, const indexer::Rope &rope, const void *q_norm, const void *k_norm,
               const Batch &batch, const __nv_bfloat16 *qg, __nv_bfloat16 *q, __nv_bfloat16 *k,
               cudaStream_t stream) {
  if (batch.rows() <= 0 || batch.positions == nullptr) return "qsa prepare: an empty batch or no positions";
  PrepareArgs a{rope, static_cast<const __nv_bfloat16 *>(q_norm), static_cast<const __nv_bfloat16 *>(k_norm),
                batch.positions, batch.tokens, g.rms_norm_eps, qg, q, k};
  prepare_kernel<<<batch.rows(), (kQHeads + kKvHeads) * 32, 0, stream>>>(a);
  return launched("qsa prepare: launch failed");
}

Status append(const Geometry &g, const Kv &kv, const Batch &batch, const __nv_bfloat16 *k,
              const __nv_bfloat16 *v, cudaStream_t stream) {
  (void)g;
  if (batch.rows() <= 0 || batch.slots == nullptr || batch.positions == nullptr) {
    return "qsa append: an empty batch, or no slots or positions";
  }
  if (kv.block_tables == nullptr || kv.logical_pages <= 0 || kv.slots <= 0 || kv.k == nullptr ||
      kv.v == nullptr) {
    return "qsa append: no pages";
  }
  AppendArgs a{kv, batch.slots, batch.positions, batch.tokens, batch.rows(), k, v};
  if (kv.kv_format == IGNIS_KV_FORMAT_BF16) {
    const int64_t threads = static_cast<int64_t>(batch.rows()) * 2 * kKvHeads * (kHeadDim / 8);
    append_bf16_kernel<<<static_cast<unsigned>((threads + 255) / 256), 256, 0, stream>>>(a);
    return launched("qsa append (bf16): launch failed");
  }
  if (kv.kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) {
    if (kv.k_meta == nullptr || kv.v_meta == nullptr) return "qsa append (hq-e8-2b): no metadata planes";
    if ((kv.residual_k == nullptr) != (kv.ring == nullptr) || (kv.residual_k == nullptr) != (kv.residual_v == nullptr)) {
      return "qsa append (hq-e8-2b): the residual window needs both side planes and the ring";
    }
    const int64_t units = static_cast<int64_t>(batch.rows()) * kKvHeads * 2;
    append_hq_kernel<<<static_cast<unsigned>((units + kHqWarps - 1) / kHqWarps), kHqWarps * 32, 0, stream>>>(a);
    return launched("qsa append (hq-e8-2b): launch failed");
  }
  return "qsa append: unknown KV format";
}

Status gate(const Batch &batch, const __nv_bfloat16 *qg, __nv_bfloat16 *out, cudaStream_t stream) {
  const int64_t chunks = static_cast<int64_t>(batch.rows()) * kOutWidth / 8;
  if (chunks <= 0) return "qsa gate: an empty batch";
  gate_kernel<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0, stream>>>(qg, out, chunks);
  return launched("qsa gate: launch failed");
}

int32_t run(const Geometry &g, const Kv &kv, const indexer::Rope &rope, const QsaWeights &w, const Batch &batch,
            const void *x, const Selection &selection, void *y, ninfer::DeviceArena &scratch, cudaStream_t stream) {
  auto fail = [](const std::string &what) {
    fn_set_error("fn_qsa_attention: " + what);
    return -1;
  };
  if (const Status st = check_geometry(g)) return fail(st);
  if (const sparse::Status st = sparse::check_geometry(g)) return fail(st);
  const int32_t rows = batch.rows();
  if (rows <= 0 || x == nullptr || y == nullptr) return fail("an empty batch or null activations");
  if (!shaped(w.q_proj, kQProjWidth, g.hidden) || !shaped(w.k_proj, kKvWidth, g.hidden) ||
      !shaped(w.v_proj, kKvWidth, g.hidden) || !shaped(w.o_proj, g.hidden, kOutWidth) || w.q_norm == nullptr ||
      w.k_norm == nullptr) {
    return fail("a weight is missing or has the wrong shape");
  }
  if (selection.dense && (batch.lanes != 1 || batch.max_visible > g.dense_threshold())) {
    return fail("a dense call is one lane of at most dense_threshold() visible tokens");
  }
  if (batch.tokens == 1 && batch.lanes > kMaxDecodeLanes) {
    return fail("a decode call is at most " + std::to_string(kMaxDecodeLanes) + " lanes");
  }
  if (batch.verify != nullptr && rows > kMaxDecodeLanes) {
    return fail("a verify call is at most " + std::to_string(kMaxDecodeLanes) + " rows");
  }
  const bool hq = kv.kv_format == IGNIS_KV_FORMAT_HQ_E8_2B;

  try {
    auto scope = scratch.scope();
    const auto r = static_cast<std::size_t>(rows) * sizeof(__nv_bfloat16);
    auto *qg = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kQProjWidth).data);
    auto *k = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kKvWidth).data);
    auto *v = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kKvWidth).data);
    auto *q = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kOutWidth).data);
    auto *out = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kOutWidth).data);
    // GitHub #306, step 6: the three projections of x in one grouped GEMV launch when all three are
    // FP8 and the call is a GEMV's width; a split call's combine takes the output gate.
    const bool fusion = fused(Fusion::Qsa);
    const auto fp8 = [](const Linear &l) { return l.format == WeightFormat::Fp8RowScale; };
    if (fusion && rows <= kGroupedRows && fp8(w.q_proj) && fp8(w.k_proj) && fp8(w.v_proj)) {
      const ignis_fp8_segment segments[3] = {{w.q_proj.data, static_cast<uint32_t>(kQProjWidth), qg},
                                             {w.k_proj.data, static_cast<uint32_t>(kKvWidth), k},
                                             {w.v_proj.data, static_cast<uint32_t>(kKvWidth), v}};
      if (ignis_fp8_linear_grouped(segments, 3, static_cast<uint32_t>(g.hidden), x, static_cast<uint32_t>(rows), 0,
                                   stream) != 0) {
        return fail(std::string("the grouped projections: ") + ignis_fp8_linear_last_error());
      }
    } else if (fn_linear(w.q_proj, x, rows, qg, false, scratch, stream) != 0 ||
               fn_linear(w.k_proj, x, rows, k, false, scratch, stream) != 0 ||
               fn_linear(w.v_proj, x, rows, v, false, scratch, stream) != 0) {
      return -1;
    }
    if (const Status st = prepare(g, rope, w.q_norm, w.k_norm, batch, qg, q, k, stream)) return fail(st);

    // Where the attention reads K/V: the pages (BF16), or, under hq-e8-2b, the rows it reads
    // decoded into a plain-frame scratch BEFORE the append overwrites the ring.
    sparse::KvSource source;
    source.k = static_cast<const __nv_bfloat16 *>(kv.k);
    source.v = static_cast<const __nv_bfloat16 *>(kv.v);
    source.block_tables = kv.block_tables;
    source.logical_pages = kv.logical_pages;
    source.kv_heads = kKvHeads;
    source.mode = sparse::KvSource::Mode::Paged;
    if (hq) {
      sparse::HqSource src;
      src.k_codes = static_cast<const uint8_t *>(kv.k);
      src.k_meta = static_cast<const uint8_t *>(kv.k_meta);
      src.v_codes = static_cast<const uint8_t *>(kv.v);
      src.v_meta = static_cast<const uint8_t *>(kv.v_meta);
      src.block_tables = kv.block_tables;
      src.logical_pages = kv.logical_pages;
      src.kv_heads = kKvHeads;
      src.fresh_k = k;
      src.fresh_v = v;
      src.residual_k = kv.residual_k;
      src.residual_v = kv.residual_v;
      src.ring_valid = kv.ring;
      sparse::Status st = nullptr;
      // A decode or verify call decodes its rows' listed tokens; a prefill chunk its lane's visible
      // rows, once for all its rows.
      if (!selection.dense && (batch.tokens == 1 || batch.verify != nullptr)) {
        const std::size_t bytes = sparse::listed_hq_bytes(g, rows);
        auto *kd = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(bytes).data);
        auto *vd = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(bytes).data);
        st = sparse::decode_listed_hq(g, src, batch, selection, kd, vd, &source, stream);
      } else {
        const std::size_t bytes = sparse::visible_hq_bytes(g, batch.max_visible);
        auto *kd = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(bytes).data);
        auto *vd = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(bytes).data);
        st = sparse::decode_visible_hq(g, src, batch, kd, vd, &source, stream);
      }
      if (st != nullptr) return fail(st);
    }
    if (const Status st = append(g, kv, batch, k, v, stream)) return fail(st);

    bool gated = false;
    if (selection.dense) {
      if (const Status st = attend_dense(g, source, kv.slots, batch, q, out, stream)) return fail(st);
    } else {
      const std::size_t partials = sparse::partial_bytes(g, rows);
      void *partial = partials == 0 ? nullptr : scratch.alloc_bytes(partials).data;
      gated = fusion && sparse::splits_for(g, rows) > 1;
      if (const sparse::Status st =
              sparse::attend(g, source, batch, q, selection, out, partial, stream, gated ? qg : nullptr)) {
        return fail(st);
      }
    }
    if (!gated) {
      if (const Status st = gate(batch, qg, out, stream)) return fail(st);
    }
    return fn_linear(w.o_proj, out, rows, y, false, scratch, stream);
  } catch (const std::exception &e) {
    return fail(e.what());
  }
}

}  // namespace qsa

int32_t fn_qsa_attention(const Context &ctx, int32_t attn_ordinal, const QsaWeights &w, const Batch &batch,
                         const void *x, const Selection &selection, void *y, ninfer::DeviceArena &scratch,
                         cudaStream_t stream) {
  auto fail = [](const std::string &what) {
    fn_set_error("fn_qsa_attention: " + what);
    return -1;
  };
  ignis_seq_pool *pool = ctx.pool;
  if (pool == nullptr) return fail("no seq pool");
  if (ctx.rope.attention_factor != 1.0F) return fail("a rope attention factor other than 1 is not supported");
  if (pool->kv_format != ctx.kv_format || pool->kv_head_dim != qsa::kHeadDim ||
      pool->kv_num_kv_heads != qsa::kKvHeads) {
    return fail("the seq pool's KV planes are not this load's format and geometry");
  }
  qsa::Kv kv;
  try {
    const auto plane = [&](ignis_kv_plane_role role) {
      return pool->kv_pool.plane(ignis_kv_plane_index(ctx.kv_format, attn_ordinal, role)).data;
    };
    kv.kv_format = ctx.kv_format;
    kv.block_tables = static_cast<const int32_t *>(pool->kv_pool.block_tables().data);
    kv.logical_pages = static_cast<int32_t>(pool->kv_pool.logical_page_capacity());
    kv.slots = pool->kv_pool.table_row_count();
    kv.k = plane(IGNIS_KV_PLANE_K);
    kv.v = plane(IGNIS_KV_PLANE_V);
    if (ctx.kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) {
      kv.k_meta = plane(IGNIS_KV_PLANE_K_META);
      kv.v_meta = plane(IGNIS_KV_PLANE_V_META);
      if (pool->has_hq_residual()) {
        kv.residual_k = static_cast<__nv_bfloat16 *>(pool->hq_residual_plane(false, attn_ordinal, 0));
        kv.residual_v = static_cast<__nv_bfloat16 *>(pool->hq_residual_plane(true, attn_ordinal, 0));
        kv.ring = pool->hq_ring_words(0);
      }
    }
  } catch (const std::exception &e) {
    return fail("attention layer " + std::to_string(attn_ordinal) + ": " + e.what());
  }
  return qsa::run(ctx.g, kv, indexer::rope_from(ctx.rope), w, batch, x, selection, y, scratch, stream);
}

// The activations, the widest hq decode a dense or decode call makes (dense_threshold() visible
// rows, or the listed rows of at most kMaxDecodeLanes lanes), and the sparse route's partials. A
// sparse PREFILL call under hq-e8-2b also decodes every visible row (S3's decode_visible_hq,
// 2 * visible_hq_bytes(g, max_visible)): that scales with max_visible, so it is S3's plan line,
// not this one (coordinator, 2026-10-05).
std::size_t fn_qsa_attention_scratch_bytes(const Geometry &g, int32_t rows) {
  if (rows <= 0) return 0;
  const std::size_t decoded = std::max(sparse::listed_hq_bytes(g, std::min(rows, qsa::kMaxDecodeLanes)),
                                       sparse::visible_hq_bytes(g, g.dense_threshold()));
  return qsa::activation_bytes(rows) + 2 * qsa::aligned(decoded) + qsa::aligned(sparse::partial_bytes(g, rows));
}

}  // namespace ignis::flash_next
