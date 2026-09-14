// ignis kernel leaf - P5-05 (GitHub #155): the DFlash2 drafter's program
// glue, shared by the prefill chunk (kernel/src/step.cu), the verify round's
// traversal (kernel/src/decode_graph.cu) and its commit (step.cu). Ours, not
// vendored (ADR 0010): every numerical step is a vendored op, and the op
// sequences are the reference's `dflash2_append_context_impl` and
// `dflash2_propose_batch_impl` (targets/qwen3_6/impl/runtime/dflash2_impl.h
// at the pinned commit). Leaf-internal, never across the C ABI.
//
// Each function only enqueues work on the model's stream and throws
// `std::exception` on a kernel or argument error; the caller owns the
// synchronization and the error channel.

#pragma once

#include "ignis_seq_internal.h"
#include "model_internal.h"

#include "ninfer/ops/kv_cache_append_prefix.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

// The drafter's context append over `[width, batch]` columns of feature taps
// (P5-03, GitHub #152, generalized to a batch here): `features` is BF16
// `[5 x hidden, width * batch]`, lane-major, the target's layer-5/19/33/47/61
// outputs concatenated per column; `positions` is device I32 `[width, batch]`
// (each column's absolute position); `counts` and `lanes` are device I32
// `[batch]` -- how many leading columns of each lane are appended, and the
// pool slot whose window lane receives them. The features are projected and
// normalized once, then every drafter layer's keys (normalized and roped at
// their absolute positions) and values land in ring slot `p mod 2048` of the
// lane. A lane with count 0 writes nothing. Allocations come from `arena`,
// under the caller's scope.
void ignis_dflash2_append_context(ignis_model *model, ignis_seq_pool *pool,
                                  ninfer::DeviceArena &arena, const ninfer::Tensor &features,
                                  const ninfer::Tensor &positions, const ninfer::Tensor &counts,
                                  const ninfer::Tensor &lanes,
                                  ninfer::ops::KVCacheAppendPrefixExecutionEnvelope envelope);

// The drafter's forward for a verify round at batch `width`: each lane's
// anchor and `k` mask tokens at positions `base + j`, five layers of two-tap
// dynamic conv and sliding-window attention over the lane's window, the
// target's output head over the `k` draft columns, the per-column top-k and
// the selector lattice walk -- writing the round's drafts straight into
// `model->verify->drafts` (`[k, width]`). Reads the round's staged anchors,
// base positions, valid columns and pool slots; writes nothing but the
// drafts and its own scratch, so a lane's window is read, never written.
// Every address is model- or pool-owned and the attention envelope is the
// load's whole context, so the same call runs eagerly and records into a
// verify graph.
void ignis_dflash2_propose(ignis_model *model, ignis_seq_pool *pool, std::uint32_t width);

// The verify round's half of the append: the committed columns' feature
// taps (`model->verify->features`, written by the traversal) at the round's
// positions, `model->verify->append_counts` columns per lane (staged by the
// caller after the cut), into each lane's window.
void ignis_dflash2_append_round(ignis_model *model, ignis_seq_pool *pool, std::uint32_t width);
