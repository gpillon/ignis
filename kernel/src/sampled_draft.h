// The sampled draft's two round steps (GitHub #576, spec #574, ADR 0090),
// leaf-internal: the draft's draw, which the DFlash2 drafter's forward runs
// after its selector walk (kernel/src/dflash2_drafter.cu), and the accept,
// which the verify pass runs in place of the vendored one on a load under
// the sampled draft (kernel/src/decode_graph.cu). The arithmetic is
// kernel/src/sampled_draft.cuh's. Both only enqueue work on `stream`, at
// addresses the caller owns, so the same calls run eagerly and record into a
// verify graph; both throw `std::exception` on an argument or kernel error.
#pragma once

#include "ninfer/ops/sampling.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Each sampling lane's drafts drawn from the drafter's distribution q, after
// the selector walk wrote every lane's fixed drafts into `drafts` (I32
// `[k, B]`). A lane whose config is greedy (temperature 0) keeps the walk's
// drafts; a lane at temperature T draws column `s`'s draft from the softmax of
// its transition-score row after the draft drawn at column `s - 1` (the
// anchor's at column 0), at T x `temperature_factor`, cut by `top_p`, with its
// own uniform at position `base_positions[lane] + s + 1`. `scores` is the
// selector's F32 `[top_k (succ), top_k (pred), k, B]`, `candidates` its I32
// `[top_k, k, B]` token ids. Each sampling lane's q is kept for the accept:
// `q_ids` and `q_probs` are `[top_k, k, B]`, the candidates' ids and their
// probabilities (a greedy lane's are left as they were).
//
// GitHub #577 (test-only): `forced`, device I32 `[k + 1, B]`, is a
// teacher-forced round's recorded tokens after each lane's anchor. A sampling
// lane then drafts each column's recorded token wherever it is one of the
// column's candidates (its draw elsewhere), and its next column's q follows
// that draft: the q a live round that drew the recording read. Null in every
// serving round and in every verify graph.
void ignis_sampled_draft_draw(const ninfer::Tensor &scores, const ninfer::Tensor &candidates,
                              const ninfer::ops::SamplingConfig *configs,
                              const std::int32_t *base_positions, float temperature_factor,
                              float top_p, ninfer::Tensor &drafts, std::int32_t *q_ids,
                              float *q_probs, cudaStream_t stream,
                              const std::int32_t *forced = nullptr);

// Whether the accept below serves a `token_domain`-wide vocabulary at window
// `k`: it builds each column's target distribution on the vendored sampler's
// multi-block route, which a vocabulary of one tile or less does not take.
bool ignis_sampled_accept_serves(std::int32_t token_domain, std::int32_t k);

// The bytes the accept takes from its workspace at window `k` across `batch`
// lanes: the vendored sampler's workspace layout of `k + 1` columns, once per
// lane -- the same bytes the vendored accept takes, so the verify round's
// `accept_workspace` serves either.
std::size_t ignis_sampled_accept_workspace_bytes(std::int32_t token_domain, std::int32_t k,
                                                 std::int32_t batch);

// The accept of a verify round under the sampled draft: the vendored accept's
// contract (`speculative_accept_greedy_drafts`, ninfer/ops/speculative_round.h)
// -- the same tensors, the same outputs, each produced token of a sampling
// lane counted in its config's `token_counts` -- under a different rule for
// the sampling lanes. A greedy lane takes the longest draft prefix the target
// argmax agrees with and the argmax after it, the vendored greedy branch's
// rule. A sampling lane builds column i's target distribution p as the
// sampler does (its penalties with the i drafts before it as the round-local
// overlay, top-k, temperature, inclusive top-p), takes draft i when
// u x q(d) < p(d), and on a refusal draws its token from max(p - q, 0); every
// draft accepted, it draws the bonus token from the last column as the
// vendored accept does. `q_ids`/`q_probs` are the draw's `[top_k, k, B]`
// (ignis_sampled_draft_draw); null, q is one-hot on each lane's draft (a
// verify-only load's drafts, which come per call).
//
// GitHub #577 (test-only): `forced` is the draw's (the same `[k + 1, B]`
// recorded tokens, drafted along the recording). A sampling lane then takes
// each recorded token by the teacher-forced verdict
// (`ignis_forced_sampled_verdict`: the live verdict of the lane's own draw
// where it emits the recorded token, else a coin of its own, taken with
// probability min(p, q) / p either way), emits the recorded token where it
// stops and the recorded one after a whole run in place of the bonus draw, so
// its licensed run is the recording's. A recorded id outside the vocabulary
// (the pad after a stop) is never counted. Greedy lanes are as without it.
// Null in every serving round and in every verify graph.
void ignis_sampled_accept(const ninfer::Tensor &target_tokens, const ninfer::Tensor &logits,
                          const ninfer::Tensor &drafts, const ninfer::Tensor &current_extents,
                          ninfer::Tensor &lengths, ninfer::Tensor &anchors,
                          ninfer::Tensor &licensed_tokens, ninfer::Tensor &licensed_counts,
                          ninfer::Tensor &accepted, std::int32_t token_domain,
                          const ninfer::ops::SamplingConfig *configs, const std::int32_t *q_ids,
                          const float *q_probs, ninfer::DeviceArena &workspace, cudaStream_t stream,
                          const std::int32_t *forced = nullptr);
