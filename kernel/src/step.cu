// ignis kernel leaf - P1-18 (GitHub #54): the degenerate program (embedding
// -> final RMSNorm -> W8G32 output head -> argmax) through the step ABI
// (ADR 0009). The program layer (this file) is ours, not vendored; it
// dispatches to the ADR 0010 vendored ops (ninfer::ops::embedding /
// rmsnorm / linear / argmax). Every decoder layer is skipped
// (`skip_layers`, test-only -- P1-21/P1-22 add the GQA/GDN layer bodies).
// The model handle owns the step's stream and scratch arena
// (kernel/src/model_internal.h) so no stream or host activation pointer
// crosses this ABI.

#include "ignis_step.h"

#include "ignis_gdn_layer.h"
#include "ignis_gqa_layer.h"
#include "ignis_seq_internal.h"
#include "model_internal.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The last error message on this thread (ignis_step_last_error) -- a
// separate channel from ignis_model_last_error (each ABI surface owns its
// own, model.cu's convention).
thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

// bf16 storage -> f32: bit-exact promotion (bf16 is fp32's top 16 bits, zero
// extended).
float bf16_to_f32(std::uint16_t bits) {
  const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

// Runs embedding -> final RMSNorm -> output head -> argmax for one token
// (the degenerate program, GitHub #54). `out_logits`, if non-null, receives
// `model->vocab` host floats. Returns 0 on success, -1 on error (message
// set via set_error).
int32_t run_degenerate_step(ignis_model *model, int32_t token_id, int32_t *out_token_id,
                             float *out_logits) {
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);

  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    ninfer::Tensor ids = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
    cudaError_t err = cudaMemcpyAsync(ids.data, &token_id, sizeof(token_id),
                                      cudaMemcpyHostToDevice, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_step: cudaMemcpyAsync(ids) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    ninfer::Tensor embed_out = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::embedding(ids, model->token_embedding, embed_out, model->stream);

    // The final norm's weight is a dense BF16 [hidden] tensor (not a
    // quantized `Weight`); `rmsnorm` takes a plain `Tensor` view of it.
    ninfer::Tensor norm_weight(const_cast<void *>(model->final_norm.qdata), ninfer::DType::BF16,
                               {hidden, 1, 1, 1});
    ninfer::Tensor norm_out = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(embed_out, norm_weight, model->rms_norm_eps, /*unit_offset=*/false,
                         norm_out, model->stream);

    ninfer::Tensor logits = model->scratch->alloc(ninfer::DType::BF16, {vocab, 1, 1, 1});
    ninfer::ops::linear(norm_out, model->output_head, logits, model->stream);

    ninfer::Tensor argmax_out = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
    ninfer::ops::argmax(logits, argmax_out, vocab, model->stream);

    int32_t host_token_id = -1;
    err = cudaMemcpyAsync(&host_token_id, argmax_out.data, sizeof(host_token_id),
                          cudaMemcpyDeviceToHost, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_step: cudaMemcpyAsync(argmax) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    std::vector<std::uint16_t> host_logits_bits;
    if (out_logits != nullptr) {
      host_logits_bits.resize(static_cast<std::size_t>(vocab));
      err = cudaMemcpyAsync(host_logits_bits.data(), logits.data,
                            host_logits_bits.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost, model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_step: cudaMemcpyAsync(logits) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }

    err = cudaStreamSynchronize(model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_step: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    if (out_logits != nullptr) {
      for (std::int32_t v = 0; v < vocab; ++v) {
        out_logits[v] = bf16_to_f32(host_logits_bits[static_cast<std::size_t>(v)]);
      }
    }
    *out_token_id = host_token_id;
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_step: ") + e.what());
    return -1;
  }
}

bool validate_common(const ignis_model *model, const int32_t *token_ids, uint64_t count,
                     const ignis_sampling_params *sampling, int32_t skip_layers) {
  if (model == nullptr || token_ids == nullptr || sampling == nullptr || count == 0) {
    set_error("ignis_step: null argument or empty batch");
    return false;
  }
  if (sampling->greedy == 0) {
    set_error("ignis_step: only greedy sampling is supported (G1)");
    return false;
  }
  if (skip_layers == 0) {
    set_error(
        "ignis_step: skip_layers=0 is not yet supported (no layer body -- GitHub #57/#58)");
    return false;
  }
  return true;
}

// Runs embedding -> all decoder layers -> final norm -> output head ->
// argmax for one token.  The two residual buffers stay in the outer scratch
// scope while every layer takes (and releases) its own nested scope, so the
// program never materializes an activation on the host.
int32_t run_program_token(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                          int32_t token_id, int32_t *out_token_id) {
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    ninfer::Tensor ids = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
    cudaError_t err = cudaMemcpyAsync(ids.data, &token_id, sizeof(token_id),
                                      cudaMemcpyHostToDevice, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program: cudaMemcpyAsync(ids) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    ninfer::Tensor left = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::Tensor right = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::embedding(ids, model->token_embedding, left, model->stream);

    uint64_t dispatches = 0;
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_step(model, pool, seq, layer, left.data, right.data, 1)
          : ignis_gdn_layer_step(model, pool, seq, layer, left.data, right.data, 1);
      if (rc != 0) {
        const char *detail = model->layers[layer].kind == IGNIS_LAYER_GQA
            ? ignis_gqa_layer_last_error()
            : ignis_gdn_layer_last_error();
        set_error("ignis_program: layer " + std::to_string(layer) + " failed: " + detail);
        return -1;
      }
      std::swap(left, right);
      ++dispatches;
    }

    ninfer::Tensor norm_weight(const_cast<void *>(model->final_norm.qdata), ninfer::DType::BF16,
                               {hidden, 1, 1, 1});
    ninfer::Tensor normalized = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(left, norm_weight, model->rms_norm_eps, /*unit_offset=*/false,
                         normalized, model->stream);
    ninfer::Tensor logits = model->scratch->alloc(ninfer::DType::BF16, {vocab, 1, 1, 1});
    ninfer::ops::linear(normalized, model->output_head, logits, model->stream);
    ninfer::Tensor argmax_out = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
    ninfer::ops::argmax(logits, argmax_out, vocab, model->stream);
    err = cudaMemcpyAsync(out_token_id, argmax_out.data, sizeof(*out_token_id),
                          cudaMemcpyDeviceToHost, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program: cudaMemcpyAsync(argmax) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    err = cudaStreamSynchronize(model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    model->last_step_kernel_count = dispatches;
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_program: ") + e.what());
    return -1;
  }
}

bool validate_program(const ignis_model *model, const ignis_seq_pool *pool,
                      const ignis_seq *seq, const int32_t *tokens, uint64_t count,
                      const ignis_sampling_params *sampling) {
  if (model == nullptr || pool == nullptr || seq == nullptr || tokens == nullptr ||
      sampling == nullptr || count == 0) {
    set_error("ignis_program: null argument or empty batch");
    return false;
  }
  if (sampling->greedy == 0) {
    set_error("ignis_program: only greedy sampling is supported (G1)");
    return false;
  }
  return true;
}

} // namespace

extern "C" int32_t ignis_prefill(struct ignis_model *model, const int32_t *token_ids,
                                 uint64_t num_tokens, uint64_t /*start_position*/,
                                 int32_t skip_layers, const struct ignis_sampling_params *sampling,
                                 int32_t *out_token_id, float *out_logits) {
  if (out_token_id == nullptr) {
    set_error("ignis_prefill: out_token_id is null");
    return -1;
  }
  if (!validate_common(model, token_ids, num_tokens, sampling, skip_layers)) {
    return -1;
  }
  // The degenerate program has no cross-token state: only the span's last
  // position feeds the output head (a real prefill's earlier positions only
  // exist to advance KV/GDN state, which `skip_layers` has none of).
  const int32_t last_token = token_ids[num_tokens - 1];
  return run_degenerate_step(model, last_token, out_token_id, out_logits);
}

extern "C" int32_t ignis_decode(struct ignis_model *model, const int32_t *token_ids,
                                uint64_t batch_size, int32_t skip_layers,
                                const struct ignis_sampling_params *sampling,
                                int32_t *out_token_ids, float *out_logits) {
  if (out_token_ids == nullptr) {
    set_error("ignis_decode: out_token_ids is null");
    return -1;
  }
  if (!validate_common(model, token_ids, batch_size, sampling, skip_layers)) {
    return -1;
  }
  for (uint64_t i = 0; i < batch_size; ++i) {
    float *slot_logits =
        out_logits == nullptr ? nullptr : out_logits + i * static_cast<uint64_t>(model->vocab);
    const int32_t rc = run_degenerate_step(model, token_ids[i], &out_token_ids[i], slot_logits);
    if (rc != 0) {
      return rc;
    }
  }
  return 0;
}

extern "C" const char *ignis_step_last_error(void) {
  return g_last_error.c_str();
}

extern "C" int32_t ignis_program_prefill(struct ignis_model *model,
                                           struct ignis_seq_pool *pool,
                                           struct ignis_seq *seq,
                                           const int32_t *token_ids,
                                           uint64_t num_tokens,
                                           uint64_t start_position,
                                           const struct ignis_sampling_params *sampling) {
  if (!validate_program(model, pool, seq, token_ids, num_tokens, sampling)) {
    return -1;
  }
  if (seq->position != start_position) {
    set_error("ignis_program_prefill: start_position does not match the sequence frontier");
    return -1;
  }
  if (num_tokens > seq->kv.mapped_token_capacity() - seq->position) {
    set_error("ignis_program_prefill: span exceeds the sequence KV capacity");
    return -1;
  }
  const auto began = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < num_tokens; ++i) {
    int32_t successor = -1;
    if (run_program_token(model, pool, seq, token_ids[i], &successor) != 0) {
      return -1;
    }
    seq->pending_token = successor;
    ++seq->position;
  }
  model->last_step_micros = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - began).count());
  return 0;
}

extern "C" int32_t ignis_program_decode(struct ignis_model *model,
                                          struct ignis_seq_pool *pool,
                                          struct ignis_seq *const *sequences,
                                          uint64_t batch_size,
                                          const struct ignis_sampling_params *sampling,
                                          int32_t *out_token_ids) {
  if (model == nullptr || pool == nullptr || sequences == nullptr || out_token_ids == nullptr ||
      sampling == nullptr || batch_size == 0 || sampling->greedy == 0) {
    set_error("ignis_program_decode: null argument, empty batch, or non-greedy sampling");
    return -1;
  }
  const auto began = std::chrono::steady_clock::now();
  uint64_t dispatches = 0;
  for (uint64_t i = 0; i < batch_size; ++i) {
    ignis_seq *seq = sequences[i];
    if (seq == nullptr || seq->pending_token < 0) {
      set_error("ignis_program_decode: sequence is null or was not prefilled");
      return -1;
    }
    if (seq->position >= seq->kv.mapped_token_capacity()) {
      set_error("ignis_program_decode: sequence reached its KV capacity");
      return -1;
    }
    const int32_t emitted = seq->pending_token;
    int32_t successor = -1;
    if (run_program_token(model, pool, seq, emitted, &successor) != 0) {
      return -1;
    }
    out_token_ids[i] = emitted;
    seq->pending_token = successor;
    ++seq->position;
    dispatches += model->last_step_kernel_count;
  }
  model->last_step_kernel_count = dispatches;
  model->last_step_micros = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - began).count());
  return 0;
}

extern "C" int32_t ignis_program_stats(const struct ignis_model *model,
                                         const struct ignis_seq_pool *pool,
                                         struct ignis_program_stats *out_stats) {
  if (model == nullptr || pool == nullptr || out_stats == nullptr) {
    return -1;
  }
  out_stats->vram_bytes = model->vram_bytes + model->scratch->capacity() +
                          pool->kv_arena.capacity() + pool->gdn_arena.capacity();
  out_stats->last_step_micros = model->last_step_micros;
  out_stats->kernel_count = model->last_step_kernel_count;
  return 0;
}
