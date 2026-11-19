// GitHub #685 (spec #684): the draft window's cap per drafter, and the
// vendored domains DFlash2's must stay inside -- OURS, not vendored.
//
// Each speculative backend takes its own window (IGNIS_DFLASH2_MAX_DRAFT_TOKENS,
// IGNIS_MTP_MAX_DRAFT_TOKENS, IGNIS_VERIFY_ONLY_MAX_DRAFT_TOKENS,
// kernel/include/ignis_model.h), refused past it at load naming the backend.
//
// A DFlash2 round of k drafts a lane runs k + 1 columns a lane through vendored
// ops whose registered domains end at 16 columns, or at 15 positions for the
// selector: the edge of every one of them is a window of 15. Their limits are
// local constants or runtime checks of the vendored sources, not symbols this
// leaf can name, so each is restated here beside the file that states it (as
// kIgnisGqaQueryVerifyCapWidth restates the attention's, ignis_gqa_workspace.h),
// except the sampler's, which is the vendored header's own symbol. The check is
// one-sided: it holds the cap to these restatements, so a vendor bump that
// moves an edge must restate it here.
//
// Include from a .cu translation unit only: the sampler's header declares
// device functions.
#pragma once

#include "ignis_model.h"
#include "ops/common/sampling_workspace.h"

#include <cstdint>

// The widest window `backend` accepts; 0 for no backend or an unknown one.
inline constexpr uint32_t ignis_max_draft_tokens(int32_t backend) {
  return backend == IGNIS_SPECULATIVE_DFLASH2       ? IGNIS_DFLASH2_MAX_DRAFT_TOKENS
         : backend == IGNIS_SPECULATIVE_VERIFY_ONLY ? IGNIS_VERIFY_ONLY_MAX_DRAFT_TOKENS
         : backend == IGNIS_SPECULATIVE_MTP         ? IGNIS_MTP_MAX_DRAFT_TOKENS
                                                    : 0;
}

// The backend's name in a refusal.
inline constexpr const char *ignis_speculative_backend_name(int32_t backend) {
  return backend == IGNIS_SPECULATIVE_DFLASH2       ? "DFLASH2"
         : backend == IGNIS_SPECULATIVE_VERIFY_ONLY ? "VERIFY_ONLY"
         : backend == IGNIS_SPECULATIVE_MTP         ? "MTP"
                                                    : "NONE";
}

// The widest window of any drafter: what sizes the trellis head's widest
// verify call (kernel/src/model.cu).
inline constexpr uint32_t kIgnisWidestDraftTokens =
    IGNIS_DFLASH2_MAX_DRAFT_TOKENS > IGNIS_MTP_MAX_DRAFT_TOKENS
        ? (IGNIS_DFLASH2_MAX_DRAFT_TOKENS > IGNIS_VERIFY_ONLY_MAX_DRAFT_TOKENS ? IGNIS_DFLASH2_MAX_DRAFT_TOKENS
                                                                               : IGNIS_VERIFY_ONLY_MAX_DRAFT_TOKENS)
        : (IGNIS_MTP_MAX_DRAFT_TOKENS > IGNIS_VERIFY_ONLY_MAX_DRAFT_TOKENS ? IGNIS_MTP_MAX_DRAFT_TOKENS
                                                                           : IGNIS_VERIFY_ONLY_MAX_DRAFT_TOKENS);

// The vendored domains, in columns a lane (k + 1) unless named positions (k).
// Each holds at two lanes or more where the op takes one lane wider; a round
// takes the narrower bound at every lane count.
//
// The masked block's W: 1..16 (kernel/vendor/include/ninfer/ops/prepare_masked_block.h,
// checked in src/ops/wrapper/prepare_masked_block.cpp).
inline constexpr uint32_t kIgnisMaskedBlockMaxColumns = 16;
// The drafter's attention, `swa`'s T: 1..16 (include/ninfer/ops/swa.h,
// src/ops/launcher/swa.cu `swa_resolve_plan`).
inline constexpr uint32_t kIgnisDrafterAttentionMaxColumns = 16;
// The selector's scores, predecessors and walk, P: 1..15 positions
// (include/ninfer/ops/dflash2_selector_{scores,predecessors,walk}.h, checked in
// src/ops/wrapper/dflash2_draft.cpp).
inline constexpr uint32_t kIgnisSelectorMaxPositions = 15;
// The verify round's attention at two lanes or more, `kMaximumVerifyTokens`
// (src/ops/wrapper/gqa_attention.cpp).
inline constexpr uint32_t kIgnisVerifyAttentionMaxColumns = 16;
// The GDN record and fold: the replay's width 2..16
// (src/ops/linear_attention/gated_delta_net/replay.cpp), and the snapshot forms
// of `gated_delta_net` and `causal_conv1d_silu`, W 1..16 at two lanes or more
// (`kMaximumWidth`, gated_delta_net.cpp and src/ops/wrapper/causal_conv1d_silu.cpp).
inline constexpr uint32_t kIgnisGdnRecordMaxColumns = 16;

// Whether a round of `k` drafts a lane runs inside every one of them, the
// sampler's columns (`kSamplerMaxColumns`: the sampled accept's k + 1) among
// them.
inline constexpr bool ignis_draft_window_in_vendored_domains(uint32_t k) {
  return k >= 1 && k <= kIgnisSelectorMaxPositions && k + 1 <= kIgnisMaskedBlockMaxColumns &&
         k + 1 <= kIgnisDrafterAttentionMaxColumns && k + 1 <= kIgnisVerifyAttentionMaxColumns &&
         k + 1 <= kIgnisGdnRecordMaxColumns && k + 1 <= static_cast<uint32_t>(ninfer::ops::kSamplerMaxColumns);
}

static_assert(ignis_draft_window_in_vendored_domains(15) && !ignis_draft_window_in_vendored_domains(16),
              "the vendored domains' edge is a window of 15 drafts (16 columns a lane)");
static_assert(ignis_draft_window_in_vendored_domains(IGNIS_DFLASH2_MAX_DRAFT_TOKENS),
              "DFlash2's draft cap must stay inside the vendored ops' domains: the masked block, the drafter's "
              "attention, the selector, the verify attention, the GDN record and fold, the sampler's columns");
