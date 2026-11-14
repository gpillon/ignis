// ignis kernel leaf: the 27B's decode and verify rounds read hq-e8-2b on ignis's warp-specialized
// attention route (kernel/src/hq_verify_ws.cuh) -- OURS (spec #570, GitHub #572), leaf-internal.
//
// The route takes the vendored small-T call's place in the batch body's attention
// (`run_gqa_layer_batch`, kernel/src/gqa_layer.cu: every decode round and every verify round,
// captured or eager); the vendored call stays every other call's: the eager per-chunk layer, the
// prefill, the BF16 format, Flash-Next. It is chosen once per load, by card:
//   - compute capability major 12 (Blackwell) takes it, unless the measurement switch
//     IGNIS_HQ_DECODE_ATTENTION=vendored names today's route;
//   - every other card keeps today's route whatever the switch says: the route was built and
//     measured for Blackwell alone (its 87,408 bytes of static shared memory per block, its
//     mbarrier ring);
//   - and only where this binary holds the route's sm_120a image for the device (a build whose
//     list has no entry for it, or a 12.x card such as sm_121 that the 120a code does not run on,
//     keeps today's route: nothing launches a kernel the device has no image of).
// The load reports the route (`ignis_model_hq_decode_attention`, kernel/include/ignis_model.h).
//
// What the route writes is the vendored route's, byte for byte (kernel/tests/test_hq_verify_exact.cu),
// from the vendored call's arguments, its split capacity, its workspace and its partial layout, and
// the vendored reducer runs after it unchanged; so the VRAM plan and a round's node count are the
// vendored route's too.
#pragma once

#include "process_environment.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ignis::hq_decode_attention {

// What the measurement switch IGNIS_HQ_DECODE_ATTENTION names at a load, in the one parse the
// leaf's route switches share (kernel/src/process_environment.h):
//   - Unset (or empty): the card decides;
//   - Ours, `ours`: the card's own default where the route runs; it never takes the route
//     anywhere else;
//   - Vendored, `vendored`: today's route, the vendored kernel's hq small-T route, on any card.
using Switch = ignis::RouteSwitch;

// The switch in the process environment (kernel/src/process_environment.h), read once per load.
// Throws std::invalid_argument, naming the variable, on any value but `ours` and `vendored`.
Switch switch_for_load();

// The cards the route runs on: compute capability major 12.
constexpr bool runs_on(int compute_capability_major) {
  return compute_capability_major == 12;
}

// The one rule (true: the route): an hq-e8-2b load, on a card it runs on, with its image in the
// binary for the device, and the switch not naming today's route.
constexpr bool takes_ours(Switch named, bool hq_kv, int compute_capability_major, bool image_here) {
  return named != Switch::Vendored && hq_kv && runs_on(compute_capability_major) && image_here;
}

// Whether this binary holds the route's kernel image for the current device
// (kernel/src/hq_decode_attention_ws.cu, the sm_120a set): false where the CUDA runtime finds
// none, the error it leaves consumed, and false for the stand-in a build compiles below compute
// capability 12.0. Called at load, outside any capture.
bool image_here();

// Whether a call is one the route serves: the 27B's 24 query heads over an hq-e8-2b cache, on the
// vendored small-T route (1..8 columns a lane), which is every call of the batch body.
bool serves(std::int32_t q_heads, std::int32_t width, std::int32_t batch, ninfer::DType cache_dtype,
            ninfer::ops::GqaExecutionEnvelope envelope);

// `ninfer::ops::gqa_attention`'s small-T route over a call `serves` admits, with its arguments:
// the partials on the route, then the vendored reducer, each a programmatic dependent of the
// launch before it. Throws std::invalid_argument on a call `serves` refuses or a tensor of the
// wrong shape; enqueues on `stream` and does not synchronize.
void attend(const ninfer::Tensor &q, const ninfer::Tensor &k, const ninfer::Tensor &v,
            const ninfer::Tensor &positions, const ninfer::Tensor &valid_columns,
            const ninfer::Tensor &kv_table_rows, const ninfer::Tensor &gate, float scale,
            const ninfer::PagedKVBatchLayerView &cache, ninfer::ops::GqaExecutionEnvelope envelope,
            ninfer::WorkspaceArena &workspace, ninfer::Tensor &out, cudaStream_t stream);

// The route's partial kernel's arguments, the vendored hq route's
// (kernel/vendor/src/ops/launcher/gqa_attention_decode_hq_routes.cuh), in their order.
struct PartialLaunch {
  dim3 grid;  // (KV heads, split capacity, lanes)
  const __nv_bfloat16 *q = nullptr;
  const __nv_bfloat16 *k = nullptr;  // the round's new rows, appended by the owning split
  const __nv_bfloat16 *v = nullptr;
  const std::int32_t *positions = nullptr;
  std::uint8_t *codes_k = nullptr;
  std::uint8_t *codes_v = nullptr;
  std::uint8_t *meta_k = nullptr;
  std::uint8_t *meta_v = nullptr;
  __nv_bfloat16 *residual_k = nullptr;
  __nv_bfloat16 *residual_v = nullptr;
  std::uint32_t *ring_valid = nullptr;
  const std::int32_t *block_tables = nullptr;
  const std::int32_t *valid_columns = nullptr;
  const std::int32_t *table_rows = nullptr;
  std::int32_t table_stride = 0;
  std::int32_t tokens = 0;
  std::int32_t full_width = 0;
  std::int32_t column_begin = 0;
  std::int32_t logical_capacity = 0;
  float scale = 0.0f;
  __nv_bfloat16 *partial_acc = nullptr;
  float *partial_m = nullptr;
  float *partial_l = nullptr;
};

// Launches the route's partial kernel (kernel/src/hq_decode_attention_ws.cu) as a programmatic
// dependent of the launch before it. Throws std::runtime_error on a launch error.
void launch_partials(const PartialLaunch &launch, cudaStream_t stream);

} // namespace ignis::hq_decode_attention
