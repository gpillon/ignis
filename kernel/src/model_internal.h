// ignis kernel leaf - the `ignis_model` handle's real definition, shared
// between model.cu (P1-17, GitHub #53: load / stats / free) and step.cu
// (P1-18, GitHub #54: the degenerate embed -> norm -> head -> argmax
// program). Never exposed across the C ABI (kernel/include/ignis_model.h
// keeps `struct ignis_model;` opaque) -- this header is leaf-internal.

#pragma once

#include "ignis_model.h"
#include "ignis_step.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

// One decoder layer's leaf-crossing weights (the reference's per-layer
// weight struct shape, e.g. targets/qwen3_6 FullLayerW / GdnLayerW --
// written by us: ADR 0009, the program layer is ours, not vendored).
struct GqaLayerWeights {
  ninfer::Weight input_norm;
  ninfer::Weight query_key_gate_value;
  ninfer::Weight query_norm;
  ninfer::Weight key_norm;
  ninfer::Weight output;
  ninfer::Weight post_attention_norm;
  ninfer::Weight mlp_gate_up;
  ninfer::Weight mlp_down;
};

struct GdnLayerWeights {
  ninfer::Weight input_norm;
  ninfer::Weight a_log;
  ninfer::Weight dt_bias;
  ninfer::Weight convolution;
  ninfer::Weight a_b_projection;
  ninfer::Weight query_key_value_z;
  ninfer::Weight norm;
  ninfer::Weight output;
  ninfer::Weight post_attention_norm;
  ninfer::Weight mlp_gate_up;
  ninfer::Weight mlp_down;
};

struct LayerWeights {
  ignis_layer_kind kind = IGNIS_LAYER_GDN;
  GqaLayerWeights gqa{};
  GdnLayerWeights gdn{};
};

// The opaque loaded-model handle (never dereferenced across the boundary).
struct ignis_model {
  ninfer::Weight token_embedding;
  ninfer::Weight final_norm;
  ninfer::Weight output_head;
  std::vector<LayerWeights> layers;
  uint64_t vram_bytes = 0;
  uint64_t bound_tensor_count = 0;

  // Program-layer resources (ADR 0009, GitHub #54): the step ABI's stream
  // and scratch arena for degenerate-program intermediates (embedding /
  // norm / logits / argmax buffers). Owned by the model handle so Rust
  // never sees a stream (the spec: "streams are internal to the leaf").
  // `hidden` / `vocab` / `rms_norm_eps` are copied from the topology
  // descriptor at load time so the step ABI does not need it again.
  uint64_t hidden = 0;
  uint64_t vocab = 0;
  float rms_norm_eps = 0.0F;
  cudaStream_t stream = nullptr;
  std::unique_ptr<ninfer::DeviceArena> scratch;
  uint64_t last_step_micros = 0;
  uint64_t last_step_kernel_count = 0;

  // P2-02 (GitHub #84): the chunk width the scratch arena above was sized
  // for at load (P2-01, GitHub #83). The chunked prefill route cuts a span
  // into chunks of this width; the last chunk of a span may be narrower.
  uint32_t prefill_chunk_tokens = 0;

  // P3-03 (GitHub #99): device-side sampling's staging buffers. Separate
  // from `scratch` above (which resets every call) because a decode round's
  // configs/positions must sit at addresses a future decode CUDA graph
  // (P3-05/#102) can replay reading -- the host writes this round's values
  // into these same buffers every call, it never reallocates them.
  // `sampling_single_configs`/`sampling_single_positions` back the
  // one-sequence-at-a-time calls (`ignis_prefill`, `ignis_decode`'s
  // per-degenerate-step loop, `ignis_program_prefill`);
  // `sampling_decode_configs`/`sampling_decode_positions` are sized for
  // `IGNIS_DECODE_MAX_BATCH` lanes and back `ignis_program_decode`.
  // `sampling_workspace` is `ninfer::ops::sample`'s own transient scratch
  // (candidate selection, not the caller's inputs above), sized once for the
  // widest lane count and reset via its own Scope every call -- it carries
  // no cross-call state, so it does not need a stable address.
  // `*_out` are the device I32 destinations `ninfer::ops::sample` writes
  // picked ids into -- distinct buffers from `*_positions` above (the op's
  // contract forbids `out` aliasing `logical_positions`).
  std::unique_ptr<ninfer::DeviceBuffer> sampling_single_configs;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_single_positions;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_single_out;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_configs;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_positions;
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_out;
  // The decode round's batched logits: BF16 [vocab, IGNIS_DECODE_MAX_BATCH]
  // -- lane i's forward pass copies its own single-token logits into column
  // i (device-to-device, still inside its own scratch scope) so the round's
  // sampling is one `ninfer::ops::sample` call over every lane, not one call
  // per lane.
  std::unique_ptr<ninfer::DeviceBuffer> sampling_decode_logits;
  std::unique_ptr<ninfer::DeviceArena> sampling_workspace;

  // P3-05 (GitHub #102, ADR 0019): the decode CUDA graphs' own resources,
  // reserved once at load (scratch/staging) and captured once after the
  // sequence pool exists (`ignis_decode_graph_capture`). `max_context_tokens`
  // is copied from `ignis_model_load`'s argument so the graph's fixed,
  // conservative `GqaExecutionEnvelope` (every replay, regardless of a
  // lane's actual position) needs no second parameter threaded through
  // capture.
  uint32_t max_context_tokens = 0;
  // Separate from `scratch` above: a graph replays fixed device addresses,
  // and a prefill chunk (which only ever uses `scratch`) landing between two
  // replays must never alias what a replay rereads. Sized once for one
  // lane's per-layer peak at T=1 (mirrors `scratch`'s own sizing at
  // `prefill_chunk_tokens`, kernel/src/model.cu) and reused, via its own
  // `Scope`, sequentially across a graph's lanes -- the same reuse pattern
  // the eager per-lane decode loop already applies to `scratch`.
  std::unique_ptr<ninfer::DeviceArena> decode_graph_scratch;
  // This round's token id per lane (I32 x IGNIS_DECODE_MAX_BATCH),
  // refreshed by one H2D copy before a replay; a captured graph's embedding
  // step reads lane i's id directly from column i, no per-lane device copy.
  std::unique_ptr<ninfer::DeviceBuffer> decode_graph_token_ids;
  // This round's physical pool slot per lane (I32 x IGNIS_DECODE_MAX_BATCH),
  // refreshed the same way. The single value at column i serves both GQA's
  // `kv_table_rows` (selecting a row of the pool-wide block-table matrix)
  // and GDN's `initial_state_slots`/`snapshot_base_slots` (in place: same
  // buffer for both, since a physical slot is a physical slot) -- read by
  // the kernels at replay time, never baked at capture time (ADR 0019).
  std::unique_ptr<ninfer::DeviceBuffer> decode_graph_slots;
  // A constant zero I32 scalar: `offset_i32_positions`'s `source` argument
  // for RoPE's per-lane position (`positions[0] = 0 + sampling_decode_positions[lane]`),
  // graph-safe where `fill_i32_positions`'s host-scalar `start` is not.
  std::unique_ptr<ninfer::DeviceBuffer> decode_graph_zero;
  std::array<cudaGraphExec_t, IGNIS_DECODE_MAX_BATCH> decode_graph_exec{};
  std::array<bool, IGNIS_DECODE_MAX_BATCH> decode_graph_ready{};
  // Set by the most recent `ignis_program_decode` call: 1 if it replayed a
  // graph, 0 if it ran the eager loop (`ignis_program_stats`'s
  // `graph_launches`).
  uint64_t last_step_graph_launches = 0;
};
