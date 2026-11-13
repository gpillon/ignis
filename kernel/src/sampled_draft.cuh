// The sampled draft's device arithmetic (GitHub #576, spec #574, ADR 0090):
// how a lane that samples draws its draft from the drafter's distribution q
// and how its accept takes that draft against the target's sampling
// distribution p. The host reference is `ignis_core::sampled_draft`
// (crates/core/src/sampled_draft.rs, GitHub #575): every function here is
// that file's function of the same name, in f32 as there, and a GPU test
// holds these to the same double-precision oracle over the same grid of
// uniforms (crates/core/tests/sampled_draft_gpu.rs).
//
// Ours, not vendored: the vendored accept (`speculative_accept_greedy_drafts`)
// takes a fixed draft with probability p(d), the rule for a one-hot q; this is
// the rule for any q. It reads the vendored sampler's device primitives
// (`ops/kernel/sampling_device.cuh`: its counter-based uniform and its
// support pick), which are inline device functions with no kernel of their
// own, so this header can sit in any translation unit.
#pragma once

#include "permitted_tokens.h"

#include "ninfer/ops/sampling.h"

#include <cmath>
#include <cstdint>

// Host and device: the round runs these on the device, and the leaf's own
// kernel test (kernel/tests/test_sampled_draft.cu) walks the same functions
// on the host to say what a round must have written.
#define IGNIS_SAMPLED_DRAFT_FN __host__ __device__ inline

// The selector's candidates per draft column: kDflash2SelectorTopK, which
// sizes every per-column array below.
inline constexpr int kIgnisSampledDraftCandidates = 16;

// The sampler's RNG purposes of the rule's three draws (ADR 0090). The
// vendored sampler keys its counter-based uniform by (seed, position,
// purpose) and names purposes 0..4 (`ninfer::ops::SamplePurpose`); #586 put a
// repair's draw at 16. Two draws on one purpose share their uniform, which
// biases the output with no error, so each of these has its own: the draft's
// draw from q, the accept's comparison, and the refusal's draw from what p
// has left over q. All three are keyed at the draft column's position, the
// one the vendored accept keys its own draws by; the bonus token after an
// accepted run keeps the vendored bonus purpose.
inline constexpr std::int32_t kIgnisSamplePurposeDraft          = 17;
inline constexpr std::int32_t kIgnisSamplePurposeSampledAccept  = 18;
inline constexpr std::int32_t kIgnisSamplePurposeSampledRefusal = 19;
static_assert(kIgnisSamplePurposeDraft > ninfer::ops::kSamplePurposeSpeculativeBonus &&
                  kIgnisSamplePurposeDraft != kIgnisSamplePurposeRepair &&
                  kIgnisSamplePurposeSampledAccept != kIgnisSamplePurposeRepair &&
                  kIgnisSamplePurposeSampledRefusal != kIgnisSamplePurposeRepair,
              "the sampled draft's purposes must be distinct from the vendored ones and the repair's");

// GitHub #577 (test-only): the teacher-forced round's coin under the sampled
// draft (`ignis_forced_sampled_verdict`), keyed like the three above. Only a
// forced round draws it, and no serving path forces one.
inline constexpr std::int32_t kIgnisSamplePurposeForcedCoin = 20;
static_assert(kIgnisSamplePurposeForcedCoin > kIgnisSamplePurposeSampledRefusal &&
                  kIgnisSamplePurposeForcedCoin != kIgnisSamplePurposeRepair,
              "the forced coin's purpose must be its own");

// The probability of `token` in a distribution of `n` (id, probability)
// entries: the sum over the entries that hold it, 0 when none does
// (`Distribution::prob`).
IGNIS_SAMPLED_DRAFT_FN float ignis_distribution_prob(const std::int32_t *ids, const float *probs, int n,
                                                std::int32_t token) {
  float p = 0.0f;
  for (int j = 0; j < n; ++j) {
    if (ids[j] == token) {
      p += probs[j];
    }
  }
  return p;
}

// The inverse CDF over `n` weights in their order, the weight of entry `j`
// being `weight(j)` floored at 0: the first index at which the running sum of
// the positive weights passes `u` times their total, or the last positive one
// when rounding leaves the goal at the total; -1 when none is positive
// (`inverse_cdf`).
template <typename Weight>
IGNIS_SAMPLED_DRAFT_FN int ignis_inverse_cdf(int n, Weight weight, float u) {
  float total = 0.0f;
  for (int j = 0; j < n; ++j) {
    total += fmaxf(weight(j), 0.0f);
  }
  if (!(total > 0.0f)) {
    return -1;
  }
  const float goal = u * total;
  float cum = 0.0f;
  int picked = -1;
  for (int j = 0; j < n; ++j) {
    const float w = fmaxf(weight(j), 0.0f);
    if (w > 0.0f) {
      cum += w;
      picked = j;
      if (goal < cum) {
        break;
      }
    }
  }
  return picked;
}

// The drafter's distribution q over a column's `n` (at most
// kIgnisSampledDraftCandidates) candidates (`drafter_distribution`): the
// softmax of `scores` at `temperature`, cut to the smallest prefix of the
// candidates in descending score (ties toward the smaller index) that holds
// `top_p` of the mass, the sampler's inclusive top-p, and renormalized over
// it. A `temperature` of 0 or below puts all the mass on the first highest
// score: the selector walk's draft.
IGNIS_SAMPLED_DRAFT_FN void ignis_drafter_distribution(const float *scores, int n, float temperature,
                                                  float top_p, float *q) {
  int order[kIgnisSampledDraftCandidates];
  for (int i = 0; i < n; ++i) {
    int at = i;
    while (at > 0 && scores[i] > scores[order[at - 1]]) {
      order[at] = order[at - 1];
      --at;
    }
    order[at] = i;
  }
  for (int i = 0; i < n; ++i) {
    q[i] = 0.0f;
  }
  if (!(temperature > 0.0f)) {
    q[order[0]] = 1.0f;
    return;
  }
  const float max = scores[order[0]];
  for (int i = 0; i < n; ++i) {
    q[i] = expf((scores[i] - max) / temperature);
  }
  int kept = n;
  if (top_p < 1.0f) {
    float total = 0.0f;
    for (int i = 0; i < n; ++i) {
      total += q[i];
    }
    const float goal = top_p * total;
    float cum = 0.0f;
    for (int at = 0; at < n; ++at) {
      cum += q[order[at]];
      if (cum >= goal) {
        kept = at + 1;
        break;
      }
    }
  }
  float mass = 0.0f;
  for (int at = 0; at < kept; ++at) {
    mass += q[order[at]];
  }
  for (int at = kept; at < n; ++at) {
    q[order[at]] = 0.0f;
  }
  for (int at = 0; at < kept; ++at) {
    q[order[at]] /= mass;
  }
}

// The draft drawn from `q` over `n` candidates with its own uniform `u`: the
// inverse CDF over the candidates in their order (`draw_draft`). A candidate
// with no mass is never drawn; q always holds some, and a q that held none
// would draw the first candidate.
IGNIS_SAMPLED_DRAFT_FN int ignis_draw_draft(const float *q, int n, float u) {
  const int picked = ignis_inverse_cdf(n, [q](int i) { return q[i]; }, u);
  return picked < 0 ? 0 : picked;
}

// The accept's verdict on `draft` at one column (`accept_verdict`): taken when
// `u_accept x q(draft) < p(draft)`, p the target's sampling distribution over
// its `n_target` support entries in the sampler's order and q the drafter's
// over its `n_drafter` candidates; otherwise refused, `*token` drawn by
// `u_refusal` from max(p - q, 0) over the target's support in its order, or
// from p itself when rounding leaves that no mass. Returns whether the draft
// was taken; `*token` is the draft then.
IGNIS_SAMPLED_DRAFT_FN bool ignis_sampled_verdict(const std::int32_t *target_ids, const float *target_probs,
                                             int n_target, const std::int32_t *drafter_ids,
                                             const float *drafter_probs, int n_drafter,
                                             std::int32_t draft, float u_accept, float u_refusal,
                                             std::int32_t *token) {
  const float qd = ignis_distribution_prob(drafter_ids, drafter_probs, n_drafter, draft);
  const float pd = ignis_distribution_prob(target_ids, target_probs, n_target, draft);
  if (u_accept * qd < pd) {
    *token = draft;
    return true;
  }
  int j = ignis_inverse_cdf(
      n_target,
      [=](int i) {
        return target_probs[i] - ignis_distribution_prob(drafter_ids, drafter_probs, n_drafter, target_ids[i]);
      },
      u_refusal);
  if (j < 0) {
    j = ignis_inverse_cdf(n_target, [=](int i) { return target_probs[i]; }, u_refusal);
  }
  *token = target_ids[j < 0 ? 0 : j];
  return false;
}

// GitHub #577 (test-only): the teacher-forced verdict under the sampled draft
// (`forced_verdict`, crates/core/src/sampled_draft.rs) at a column whose
// recorded token is `recorded` and whose draft is that token. `live_taken`
// and `live_token` are `ignis_sampled_verdict` on `live_draft`, the draft the
// lane's own draw proposes there, under the column's own uniforms. Where that
// live column emits the recorded token the verdict is the live one, so a
// recording of the lane's own rounds is forced back into them; elsewhere the
// coin `u_coin` takes the recorded token when `u_coin x p(x) < q(x)` (or
// `u_coin < q(x)` where p gives it nothing). Taken with probability
// min(p(x), q(x)) / p(x), a live round's.
IGNIS_SAMPLED_DRAFT_FN bool ignis_forced_sampled_verdict(const std::int32_t *target_ids,
                                                    const float *target_probs, int n_target,
                                                    const std::int32_t *drafter_ids,
                                                    const float *drafter_probs, int n_drafter,
                                                    std::int32_t recorded, std::int32_t live_draft,
                                                    bool live_taken, std::int32_t live_token,
                                                    float u_coin) {
  if (live_taken && live_draft == recorded) {
    return true;
  }
  if (!live_taken && live_token == recorded) {
    return false;
  }
  const float p = ignis_distribution_prob(target_ids, target_probs, n_target, recorded);
  const float q = ignis_distribution_prob(drafter_ids, drafter_probs, n_drafter, recorded);
  return p > 0.0f ? u_coin * p < q : u_coin < q;
}
