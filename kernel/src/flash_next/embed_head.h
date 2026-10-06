// ignis kernel leaf -- Flash-Next's token embedding and output head (spec
// flash-next/04, GitHub #302; OURS, ADR 0043). S1-private.
//
// Embedding: embed_tokens' row of each token id, decoded as the converter's
// quantized reference decodes FP8 (bf16(e4m3(code) * scale), layout.md 6.1)
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

// logits (BF16 [rows][vocab]) = lm_head(final_mix(residual)) for `rows`
// residual rows (prefill passes each lane's last token only).
int32_t fn_head(const Geometry &g, const HcWeights &final_mixer, const Linear &head,
                const void *residual, int32_t rows, void *logits, ninfer::DeviceArena &scratch,
                cudaStream_t stream);
std::size_t fn_head_scratch_bytes(const Geometry &g, int32_t rows);

}  // namespace ignis::flash_next
