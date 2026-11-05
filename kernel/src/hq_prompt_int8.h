// The integer band's passes (GitHub #510, spec #503), kernel/src/hq_prompt_int8.cu: what
// ignis_hq_prompt_attention (kernel/src/hq_prompt_attention.cu) runs under
// IgnisHqPromptPrecision::kInt8, for the residual window and for each band. Internal to the leaf.
#pragma once

#include "ignis_hq_prompt_attention.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ignis_hq_prompt_int8 {

// One pass, as hq_prompt_attention.cu's Queries, Columns, PassBuffers and Carry hold it.
struct Pass {
  const std::uint8_t *q8;  // INT8 Q [token][24][256], the codec's rotated frame
  const float *q_scale;    // [token][24]
  std::int32_t start;      // the chunk's first position
  std::int32_t width;      // its tokens
  float scale;
  // The pass's keys by column: column i < split is key first0 + i, a later one key
  // first1 + i - split; columns from `count` on hold none.
  std::int32_t first0, split, first1, count;
  std::uint8_t *k;      // INT8 K [kv_head][capacity][256], one scale per 32-key tile, stored
  std::uint8_t *v;      // s8 V^T [kv_head][capacity / 32][256 dims x 32 keys], stored by attend
  float *scales;        // [role][kv_head][capacity / 32]: K's, then V^T's
  std::int32_t capacity;  // columns per KV head, a multiple of 32
  std::int32_t splits;
  std::int32_t resume;  // 0 on a call's first pass
  __half *carry_o;      // the split states, as the other passes keep them
  float *carry_m;
  float *carry_l;
};

// The key splits the band runs when a call does not name them: 2 up to 16,384 visible keys, 3
// above.
std::int32_t splits(std::uint32_t visible_keys);

// The pass's V^T in s8 by 128-key group, then its attention, at the card's variant.
void attend(const IgnisHqPromptKeys &src, const Pass &pass, cudaStream_t stream);

}  // namespace ignis_hq_prompt_int8
