// ignis kernel leaf -- one Flash-Next rope pair's rotation (ADR 0059): OURS (ADR 0043), no port
// claim. Both of Flash-Next's rope sites rotate through it -- the attention's q and k heads
// (qsa.cu prepare_kernel) and the indexer's queries and pooled block keys (indexer.cu rope_row) --
// so the two round alike by construction. Device-only: indexer.h, which declares Rope, is also
// read by host-only tests. GitHub #558: also where the indexer's rows rotate (a batch's rotation
// source, as the attention's) and a block's AxisRecord in its key slot, which verify.cu restores.

#pragma once

#include "indexer.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ignis::flash_next::indexer {

// Rotates pair (x1, x2) -- dims (i, i + 32) of a head, i = `pair` -- to `position` in torch's BF16
// arithmetic: phi = fp32(position) * inv_freq in fp32 (ADR 0059 D2); cos and sin times the table's
// scale rounded to BF16 (D1: YaRN's attention factor on queries and keys alike, 1.0f without
// scaling); each product rounded to BF16, then their sum.
__device__ __forceinline__ void rope_pair(float &x1, float &x2, const Rope &rope, int32_t position, int pair) {
  const auto bf16 = [](float v) { return __bfloat162float(__float2bfloat16_rn(v)); };
  const float phi = static_cast<float>(position) * rope.inv_freq[pair];
  const float c = bf16(cosf(phi) * rope.scale);
  const float s = bf16(sinf(phi) * rope.scale);
  const float a = x1, b = x2;
  x1 = bf16(bf16(a * c) + bf16(-b * s));
  x2 = bf16(bf16(b * c) + bf16(a * s));
}

// GitHub #558: the axis rope pair `pair` rotates at, of a token's three (temporal, height, width):
// pair i at axis i % 3, the checkpoint's interleaved M-RoPE (qsa.cu's rotation_position). Three
// equal axes are one position for every pair.
__device__ __forceinline__ int32_t axis_of(const int32_t (&axes)[3], int pair) {
  const int axis = pair % 3;
  return axis == 0 ? axes[0] : axis == 1 ? axes[1] : axes[2];
}

// Where a batch's rows rotate (flash_next_internal.h's Batch, at most one of the two named): a
// span's three axes (axis-major [3][rows]), the row's position plus its lane's rope delta, or its
// position.
struct RotationSource {
  const int32_t *positions = nullptr;
  const int32_t *rope_positions = nullptr;
  const int32_t *rope_deltas = nullptr;
  int32_t tokens = 0;
  int32_t rows = 0;  // the batch's rows, the axes' stride
};

__device__ __forceinline__ void rotation_axes(const RotationSource &s, int32_t row, int32_t (&axes)[3]) {
  if (s.rope_positions != nullptr) {
#pragma unroll
    for (int a = 0; a < 3; ++a) axes[a] = s.rope_positions[static_cast<int64_t>(a) * s.rows + row];
    return;
  }
  const int32_t lane = row / s.tokens;
  int32_t position = s.positions[lane] + row % s.tokens;
  if (s.rope_deltas != nullptr) position += s.rope_deltas[lane];
  axes[0] = axes[1] = axes[2] = position;
}

// The key slot of `block` in a lane's section: block_keys of the physical page its block-table row
// names for the block's first position.
__device__ __forceinline__ __nv_bfloat16 *block_key_slot(__nv_bfloat16 *block_keys, const int32_t *block_tables,
                                                         int32_t logical_pages, int32_t slot, int32_t block) {
  const int32_t first = block * kCompress;
  const int32_t page = block_tables[static_cast<size_t>(slot) * logical_pages + first / kPageTokens];
  return block_keys + (static_cast<size_t>(page) * kBlocksPerPage + (first % kPageTokens) / kCompress) * kHeadDim;
}

// A block's AxisRecord (indexer.h) out of its key slot: its first token's three axes, or that
// token's KV position on all three where the slot holds no record of this block.
__device__ __forceinline__ void read_axis_record(const __nv_bfloat16 *key_slot, int32_t first, int32_t (&axes)[3]) {
  const auto *words = reinterpret_cast<const int32_t *>(key_slot);
  const bool held = words[0] == kAxisRecordMagic && words[1] == first;
#pragma unroll
  for (int a = 0; a < 3; ++a) axes[a] = held ? words[2 + a] : first;
}

__device__ __forceinline__ void write_axis_record(__nv_bfloat16 *key_slot, int32_t first, const int32_t (&axes)[3]) {
  auto *words = reinterpret_cast<int32_t *>(key_slot);
  words[0] = kAxisRecordMagic;
  words[1] = first;
#pragma unroll
  for (int a = 0; a < 3; ++a) words[2 + a] = axes[a];
}

}  // namespace ignis::flash_next::indexer
