// ignis kernel leaf -- the Flash-Next program: its model object, its plan
// lines, and the prefill and decode drivers behind the step ABI (spec
// flash-next/04, GitHub #302, slice S1; OURS, ADR 0043).
//
// A Flash-Next load is an ignis_model whose `flash_next` member is set: the
// handle keeps the step ABI's shared pieces (its stream, the prefill scratch
// arena, the decode round's arena, the sampling staging, the decode graphs)
// and FlashNextModel holds the rest -- the bound weights, the activations,
// the MoE block's buffers, the n-gram rows' staging and a borrowed pointer to
// the load's expert residency. The program ABI entry points (step.cu,
// decode_graph.cu) dispatch here on that member; the 27B's paths are
// untouched.
//
// What a chunk and a round run is the layer sequence of
// flash_next_internal.h, over a Batch: a prefill chunk is one lane of up to
// prefill_chunk_tokens tokens, run eagerly; a decode round is 1..decode_lanes
// lanes of one token, replayed from a graph captured per width.

#pragma once

#include "flash_next_internal.h"

#include "ignis_model.h"
#include "ignis_moe.h"
#include "ignis_residency.h"
#include "ignis_step.h"
#include "../model_internal.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ignis_seq;

namespace ignis::flash_next {

struct Weights;

// The decode lanes a load serves when its options name none (spec 04: three
// agents), and the most it may name: the MoE decode route's and the GDN
// per-lane forms' widest call, within the step ABI's own bound.
inline constexpr uint32_t kDefaultDecodeLanes = 3;
uint32_t max_decode_lanes();

// Every device reservation a Flash-Next load makes beside its weights (ADR
// 0030), computed once from the load's arguments and allocated exactly.
struct Sizes {
  std::size_t prefill_scratch = 0;  // one chunk's peak over the layer's ops
  std::size_t decode_scratch = 0;   // one round's, at decode_lanes
  std::size_t activations = 0;      // residual, x, y, injections, ids, slots, positions, n-gram rows
  std::size_t moe = 0;              // ignis_moe_plan's total and a second router line (the lookahead)
  std::size_t sampling_logits = 0;  // [IGNIS_DECODE_MAX_BATCH][vocab] BF16
  std::size_t sampling_workspace = 0;
};

struct FlashNextModel {
  Geometry g;
  std::unique_ptr<Weights> weights;
  int32_t kv_format = 0;
  uint32_t prefill_chunk_tokens = 0;
  uint32_t max_context_tokens = 0;
  uint32_t decode_lanes = 0;
  ninfer::ops::RopeFrequencies rope{};
  // Borrowed: the caller's residency outlives the model (ignis_model.h).
  ignis_residency *residency = nullptr;
  Sizes sizes;

  // The rows every activation buffer holds: a chunk's, or a round's lanes.
  int32_t rows() const;

  // Activations, token-major (flash_next_internal.h), each at a stable
  // address so a captured round reads and writes the same bytes on replay.
  std::unique_ptr<ninfer::DeviceBuffer> residual;    // BF16 [rows][streams * hidden]
  std::unique_ptr<ninfer::DeviceBuffer> x;           // BF16 [rows][hidden]: a sublayer's input
  std::unique_ptr<ninfer::DeviceBuffer> y;           // BF16 [rows][hidden]: its output
  std::unique_ptr<ninfer::DeviceBuffer> injections;  // fp32 [rows][streams]
  // A call's inputs, staged by the host before the forward: token ids
  // [rows], each lane's slot and first position [lanes], and the n-gram
  // table rows [rows][ngram_heads][ngram_row_bytes].
  std::unique_ptr<ninfer::DeviceBuffer> token_ids;
  std::unique_ptr<ninfer::DeviceBuffer> slots;
  std::unique_ptr<ninfer::DeviceBuffer> positions;
  std::unique_ptr<ninfer::DeviceBuffer> ngram_rows;
  // The MoE block (kern's ignis_moe.h): its workspace and routed
  // accumulator, this layer's router outputs and the next layer's (the
  // lookahead residency ranks), and the shared expert's buffers.
  std::unique_ptr<ninfer::DeviceBuffer> moe_workspace;
  std::unique_ptr<ninfer::DeviceBuffer> moe_acc;
  std::unique_ptr<ninfer::DeviceBuffer> router_ids;
  std::unique_ptr<ninfer::DeviceBuffer> router_weights;
  std::unique_ptr<ninfer::DeviceBuffer> router_logits;
  std::unique_ptr<ninfer::DeviceBuffer> lookahead_ids;
  std::unique_ptr<ninfer::DeviceBuffer> lookahead_weights;
  std::unique_ptr<ninfer::DeviceBuffer> lookahead_logits;
  std::unique_ptr<ninfer::DeviceBuffer> shared_h;
  std::unique_ptr<ninfer::DeviceBuffer> shared_out;
  // Recorded on residency's lookahead branch once the lookahead router has read `x`; the
  // layer's stream waits on it before the next sublayer's mix rewrites `x`.
  cudaEvent_t lookahead_read = nullptr;
  // The shared expert's branch: forked from the layer's stream after the MoE mix, joined before
  // the combine.
  cudaStream_t shared_stream = nullptr;
  cudaEvent_t shared_fork = nullptr;
  cudaEvent_t shared_done = nullptr;

  // The lane-state views of the pool the decode graphs were captured
  // against (a replay reads these addresses): one indexer section per
  // attention layer, and the n-gram conv state.
  const ignis_seq_pool *captured_pool = nullptr;

  FlashNextModel();
  ~FlashNextModel();
};

// Binds a Flash-Next load's weights and checks every option and geometry it
// runs, allocating nothing: the half of a load ignis_model_plan_reservations
// shares. Null and *error on any refusal.
std::unique_ptr<FlashNextModel> bind_model(const ignis_bound_tensor *tensors, uint64_t count,
                                           const ignis_topology &topology, uint32_t prefill_chunk_tokens,
                                           uint32_t max_context_tokens, int32_t kv_format,
                                           const ignis_model_load_options *options, std::string *error);

// The load's plan lines, from a bound model.
ignis_model_reservations reservations(const FlashNextModel &fn);
// What a loaded model holds, read off its buffers (ignis_model_stats).
ignis_model_reservations reserved(const ignis_model &model);

// Allocates every reservation and prepares the device (the MoE ops, the FP8
// linear). `model` carries the bound FlashNextModel. 0, or -1 and *error.
int32_t finish_load(ignis_model &model, std::string *error);

// The step ABI, for a model with `flash_next` set. Errors go to the step
// ABI's channel (ignis_step_last_error).
int32_t program_prefill(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq, const int32_t *token_ids,
                        uint64_t num_tokens, uint64_t start_position, const ignis_sampling_params *sampling,
                        const ignis_prefill_options *options, float *out_logits);
int32_t program_decode(ignis_model *model, ignis_seq_pool *pool, ignis_seq *const *sequences,
                       uint64_t batch_size, const ignis_sampling_params *sampling, int32_t *out_token_ids,
                       const ignis_decode_options *options);
// Captures a round's graph per width 1..decode_lanes; never fails a width
// for good (it stays eager). *error names the last width that failed.
int32_t capture_decode_graphs(ignis_model *model, ignis_seq_pool *pool, uint32_t *out_ready_mask,
                              std::string *error);

}  // namespace ignis::flash_next
