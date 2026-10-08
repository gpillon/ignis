// ignis kernel leaf -- the switches of GitHub #306's decode fusions (OURS, ADR 0043).
//
// Each fusion merges launches of the one-lane decode round without changing a bit of what they
// compute (docs/findings/2026-10-08-flash-next-decode-fusion.md, the plan's steps 1-7). Each is on
// by default; its own environment variable set to "0" turns it off at its first read, and
// IGNIS_FN_FUSION=0 turns every one off, so a regression can be isolated without bisecting. A
// switch is read when a call is launched: a decode graph keeps the route it was captured with.
// set_fused is for the tests and benches, and wins over the environment from then on.

#pragma once

#include <cstdint>

namespace ignis::flash_next {

enum class Fusion : int32_t {
  HcNorm,   // step 1: the HC norm inside the mix's down launch      IGNIS_FN_HC_FUSED
  Inject,   // step 2: the pending inject (and combine) in the next mix  IGNIS_FN_INJECT_FUSED
  Score,    // step 3: the indexer score over the visible blocks only    IGNIS_FN_SCORE_FUSED
  Gdn,      // step 4: one grouped GEMV for qkv, z, a, b; gating in conv IGNIS_FN_GDN_FUSED
  Route,    // step 5: router select and residency's demand resolve  IGNIS_FN_ROUTE_FUSED
  Qsa,      // step 6: one grouped GEMV for q, k, v; the gate in the combine  IGNIS_FN_QSA_FUSED
  Staging,  // step 7: a round's inputs staged in one pinned copy        IGNIS_FN_STAGING_FUSED
  kCount
};

bool fused(Fusion f);
void set_fused(Fusion f, bool on);

}  // namespace ignis::flash_next
