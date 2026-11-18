// ignis kernel leaf: the FP8 row-scale GEMV's row sums at decode widths -- OURS (ADR 0043).
//
// fp8_linear.cu's paired GEMV (ignis_fp8_linear_grouped_paired: the 27B's grouped q | k | gate | v
// exception projection, GitHub #678) sums its rows here. A warp holds R output rows of row-scale
// payloads (flash-next-layout.md 6.1); lane l sums its 16-code chunks l, l + 32, ... (< cols / 16)
// of each row in that order, the 16 products of a chunk in order as fmaf of the fp32 code and the
// fp32 activation, then the xor butterfly 16, 8, 4, 2, 1: fp8_linear.cu's gemv_row order for every
// output, so the one-row GEMV's bits whatever R, the grid and the batches. Two things differ from
// gemv_row: the code vectors of the warp's rows are loaded a batch at a time, every load of a batch
// before the batch's first FMA (load_batch, then hold), and each activation vector is loaded once
// for the R rows. A kernel zeroes its sums (start), then per batch of batch_chunks chunks from
// chunk 0 up: load_batch, hold, accumulate; then finish. The batch sizes were measured on
// Flash-Next's decode GEMVs (GitHub #580): a whole row of up to 6 chunks a lane in one batch, 6 at
// a time past that (4 for four rows): one batch of 12 was 1.1-2.6 us slower on the 2560 x 6144 out
// projection.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>

namespace ignis::fp8_rows {

// The chunks a lane loads per batch, for R rows of `chunks` chunks.
__host__ __device__ constexpr int batch_chunks(int rows, int chunks) {
  return chunks <= 6 ? chunks : rows <= 2 ? 6 : 4;
}

__device__ __forceinline__ uint32_t word(const uint4 &v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

// Lane `lane`'s code vectors of chunks j0 .. j0 + U - 1 of each row (row r's `cols` codes at
// rows[r]), chunk j at vector lane + 32 j; a chunk past the row is not loaded (cols is a multiple
// of 64, not always of 512).
template <int R, int U>
__device__ __forceinline__ void load_batch(const uint8_t *const (&rows)[R], int cols, int lane, int j0,
                                           uint4 (&v)[R][U]) {
  const int vectors = cols / 16;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = lane + 32 * (j0 + u);
      v[r][u] = c < vectors ? __ldcs(reinterpret_cast<const uint4 *>(rows[r]) + c) : make_uint4(0u, 0u, 0u, 0u);
    }
}

// Every load above issues before anything below reads one. Left to itself ptxas sinks each code
// load to its first use, leaving one or two 16-byte loads a lane in flight (gemv_row's loop: one a
// trip) and the GEMV bound by latency well under the card's bandwidth. A full-warp __syncwarp is a
// NOP in the SASS of these converged warps, and ptxas keeps the loads above it.
__device__ __forceinline__ void hold() { __syncwarp(); }

// Two E4M3 codes (the low 16 bits) to fp32, as gemv_row converts them (exact).
__device__ __forceinline__ float2 e4m3x2_to_float2(uint32_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v & 0xFFFFu), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2 *>(&h));
}

template <int R, int TOK>
__device__ __forceinline__ void start(float (&acc)[R][TOK]) {
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < TOK; ++t) acc[r][t] = 0.0f;
}

// acc[r][t] += the products of the batch's chunks of row r with token t of x [TOK][cols], chunk by
// chunk in order. kReadOnly: the activations through the read-only cache (gemv_row's __ldg); off
// for a kernel that reads them right after its producer's programmatic wait.
template <bool kReadOnly, int R, int U, int TOK>
__device__ __forceinline__ void accumulate(const uint4 (&v)[R][U], const __nv_bfloat16 *x, int cols, int lane, int j0,
                                           float (&acc)[R][TOK]) {
  const int vectors = cols / 16;
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int c = lane + 32 * (j0 + u);
    if (c < vectors) {
      float2 w[R][8];
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t cw = word(v[r][u], i);
          w[r][2 * i] = e4m3x2_to_float2(cw);
          w[r][2 * i + 1] = e4m3x2_to_float2(cw >> 16);
        }
#pragma unroll
      for (int t = 0; t < TOK; ++t) {
        const uint4 *xr = reinterpret_cast<const uint4 *>(x + static_cast<std::size_t>(t) * cols) + 2 * c;
        uint4 x0, x1;
        if constexpr (kReadOnly) {
          x0 = __ldg(xr);
          x1 = __ldg(xr + 1);
        } else {
          x0 = xr[0];
          x1 = xr[1];
        }
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          const uint32_t xw = word(p < 4 ? x0 : x1, p & 3);
          const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&xw));
#pragma unroll
          for (int r = 0; r < R; ++r) {
            acc[r][t] = fmaf(w[r][p].x, a.x, acc[r][t]);
            acc[r][t] = fmaf(w[r][p].y, a.y, acc[r][t]);
          }
        }
      }
    }
  }
}

// The lanes' partial sums to the rows' sums, on every lane: gemv_row's butterfly.
template <int R, int TOK>
__device__ __forceinline__ void finish(float (&acc)[R][TOK]) {
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < TOK; ++t)
#pragma unroll
      for (int m = 16; m > 0; m >>= 1) acc[r][t] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][t], m);
}

// The sum lane `lane` stores: row lane / TOK, token lane % TOK (a lane past R * TOK stores none).
template <int R, int TOK>
__device__ __forceinline__ float pick(const float (&acc)[R][TOK], int lane) {
  float s = 0.0f;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < TOK; ++t) s = lane == r * TOK + t ? acc[r][t] : s;
  return s;
}

}  // namespace ignis::fp8_rows
