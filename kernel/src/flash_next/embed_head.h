// ignis kernel leaf -- Flash-Next's token embedding and output head (spec
// flash-next/04, GitHub #302; OURS, ADR 0043). S1-private.
//
// Embedding: embed_tokens' row of each token id, decoded as the converter's
// quantized reference decodes FP8 (bf16(e4m3(code) * scale), flash-next-layout.md 6.1)
// or copied when BF16, then repeated into every hyper-connection stream
// (transformers: hidden_states.repeat(1, 1, hc_count)).
// Head: the final hyper-connection mixer (which is also the model's final
// norm) and lm_head, BF16 logits as the BF16 module returns them.

#pragma once

#include "flash_next_internal.h"

namespace ignis::flash_next {

// residual (BF16 [rows][streams * hidden]) = every stream of each token's
// embedding row; `ids` DEVICE [rows], each in [0, vocab).
int32_t fn_embed(const Geometry &g, const Linear &embed, const int32_t *ids, int32_t rows,
                 void *residual, cudaStream_t stream);

// GitHub #557: a media item's vision rows over its placeholders' embeddings
// (transformers: inputs_embeds.masked_scatter(image_mask, image_embeds), before
// the repeat into the streams): column i of `columns` (BF16 [count][hidden],
// contiguous -- one run of an embedding pool page, in VRAM or mapped host
// memory, read in place) into every stream of residual row `rows[i]` (DEVICE
// [count], each in [0, residual_rows)). Every other row is left as it is.
int32_t fn_embed_columns(const Geometry &g, const void *columns, const int32_t *rows, int32_t count,
                         int32_t residual_rows, void *residual, cudaStream_t stream);

// GitHub #557: one run of a pool page's media columns, fn_embed_columns' operands:
// `count` contiguous BF16 [hidden] columns at `columns`, over the rows `rows`
// (DEVICE [count]) -- a prefill chunk's, or (GitHub #564) the MTP head's entries'.
struct MediaRun {
  const void *columns = nullptr;
  const int32_t *rows = nullptr;
  int32_t count = 0;
};

// logits (BF16 [rows][vocab]) = lm_head(final_mix(residual)) for `rows`
// residual rows (prefill passes each lane's last token only).
int32_t fn_head(const Geometry &g, const HcWeights &final_mixer, const Linear &head,
                const void *residual, int32_t rows, void *logits, ninfer::DeviceArena &scratch,
                cudaStream_t stream);
std::size_t fn_head_scratch_bytes(const Geometry &g, int32_t rows);

}  // namespace ignis::flash_next
