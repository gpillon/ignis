// ignis kernel leaf - the `ignis_model` handle's real definition, shared
// between model.cu (P1-17, GitHub #53: load / stats / free) and step.cu
// (P1-18, GitHub #54: the degenerate embed -> norm -> head -> argmax
// program). Never exposed across the C ABI (kernel/include/ignis_model.h
// keeps `struct ignis_model;` opaque) -- this header is leaf-internal.

#pragma once

#include "ignis_model.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

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
};
