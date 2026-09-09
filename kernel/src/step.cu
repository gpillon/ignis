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
#include "layer_internal.h"
#include "model_internal.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
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

// P3-03 (GitHub #99): `sampling->size` must match what this leaf compiled
// against (ADR 0016) -- checked wherever a caller-supplied
// ignis_sampling_params is read.
bool sampling_size_ok(const ignis_sampling_params &sampling) {
  return sampling.size == sizeof(ignis_sampling_params);
}

// The ABI struct -> the vendored op's own config (P3-03, GitHub #99).
// `greedy` nonzero forces the argmax branch regardless of `temperature`,
// matching the doc comment on `ignis_sampling_params`. `min_p` is not one of
// this ticket's six exposed parameters, so it stays disabled. `token_counts`
// is the caller's per-sequence penalty-count row, or null where the call
// site has no sequence state to penalize against (the degenerate G1 path
// never reaches this helper at all -- it stays pure argmax).
ninfer::ops::SamplingConfig to_sampling_config(const ignis_sampling_params &abi,
                                               std::int32_t *token_counts) {
  ninfer::ops::SamplingConfig cfg;
  cfg.temperature = (abi.greedy != 0) ? 0.0f : abi.temperature;
  cfg.top_k = abi.top_k;
  cfg.top_p = abi.top_p;
  cfg.min_p = 0.0f;
  cfg.presence_penalty = abi.presence_penalty;
  cfg.frequency_penalty = abi.frequency_penalty;
  cfg.seed = abi.seed;
  cfg.token_counts = token_counts;
  return cfg;
}

// Samples one sequence's single-row logits (already computed into the
// active scratch scope) through the vendored device-side sampler, using the
// model's stable "single" staging buffers (model_internal.h) so this call's
// config/position/output never alias another call's -- required by
// `ninfer::ops::sample`'s own no-alias contract, not just tidiness. `position`
// is the absolute logical position of the token this draw is the successor
// of (the caller has not yet advanced `seq->position` past it). Returns 0
// and fills `*out_token_id` on success, -1 (message set) on a kernel/copy
// error.
int32_t sample_single(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                      const ninfer::Tensor &logits, const ignis_sampling_params &sampling,
                      std::int32_t purpose, std::int32_t position, int32_t *out_token_id) {
  const ninfer::ops::SamplingConfig cfg =
      to_sampling_config(sampling, pool->token_counts_for(seq->slot));
  cudaError_t err = cudaMemcpyAsync(model->sampling_single_configs->p, &cfg, sizeof(cfg),
                                    cudaMemcpyHostToDevice, model->stream);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_program: cudaMemcpyAsync(sampling config) failed: ") +
              cudaGetErrorString(err));
    return -1;
  }
  err = cudaMemcpyAsync(model->sampling_single_positions->p, &position, sizeof(position),
                        cudaMemcpyHostToDevice, model->stream);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_program: cudaMemcpyAsync(sampling position) failed: ") +
              cudaGetErrorString(err));
    return -1;
  }
  const ninfer::Tensor positions_tensor(model->sampling_single_positions->p, ninfer::DType::I32,
                                        {1, 1, 1, 1});
  ninfer::Tensor out_tensor(model->sampling_single_out->p, ninfer::DType::I32, {1, 1, 1, 1});
  try {
    ninfer::DeviceArena::Scope workspace_scope = model->sampling_workspace->scope();
    ninfer::ops::sample(
        logits, out_tensor, static_cast<std::int32_t>(model->vocab),
        static_cast<const ninfer::ops::SamplingConfig *>(model->sampling_single_configs->p),
        positions_tensor, purpose, *model->sampling_workspace, model->stream);
  } catch (const std::exception &e) {
    set_error(std::string("ignis_program: sample() failed: ") + e.what());
    return -1;
  }
  err = cudaMemcpyAsync(out_token_id, model->sampling_single_out->p, sizeof(*out_token_id),
                        cudaMemcpyDeviceToHost, model->stream);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_program: cudaMemcpyAsync(sampled token) failed: ") +
              cudaGetErrorString(err));
    return -1;
  }
  return 0;
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
    ninfer::ops::rmsnorm(embed_out, norm_weight, model->rms_norm_eps, /*unit_offset=*/true,
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
  if (!sampling_size_ok(*sampling)) {
    set_error("ignis_step: unrecognized ignis_sampling_params size " +
              std::to_string(sampling->size));
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
// device-side sample for one token.  The two residual buffers stay in the
// outer scratch scope while every layer takes (and releases) its own nested
// scope, so the program never materializes an activation on the host.
// `out_logits`, if non-null, receives this token's full vocab-length logits
// (GitHub #72 debug path -- the copy-back mirrors run_degenerate_step's
// above). `mode` is the call's compute-policy mode (P2-03, GitHub #85): the
// layer steps dispatch every NVFP4 projection under the policy
// `ignis_policy_for` resolves for it. `sampling` (P3-03, GitHub #99) selects
// how the successor is drawn from this token's logits; the logical position
// fed to the sampler's RNG is `seq->position` (the caller advances it only
// after this call returns).
int32_t run_program_token(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                          int32_t token_id, const ignis_sampling_params &sampling,
                          int32_t *out_token_id, float *out_logits, LinearPolicyMode mode) {
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
      // P2-03 (GitHub #85): the mode-bearing step variants (the public ABI
      // entry points with their stream synchronization and position
      // advance) so this per-token route honors ADR 0016's `compute_policy`
      // override, not just the engine default.
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_step_mode(model, pool, seq, layer, left.data, right.data, 1, mode)
          : ignis_gdn_layer_step_mode(model, pool, seq, layer, left.data, right.data, 1, mode);
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
    ninfer::ops::rmsnorm(left, norm_weight, model->rms_norm_eps, /*unit_offset=*/true,
                         normalized, model->stream);
    ninfer::Tensor logits = model->scratch->alloc(ninfer::DType::BF16, {vocab, 1, 1, 1});
    ninfer::ops::linear(normalized, model->output_head, logits, model->stream);
    if (sample_single(model, pool, seq, logits, sampling, ninfer::ops::kSamplePurposePrefill,
                      static_cast<std::int32_t>(seq->position), out_token_id) != 0) {
      return -1;
    }

    std::vector<std::uint16_t> host_logits_bits;
    if (out_logits != nullptr) {
      host_logits_bits.resize(static_cast<std::size_t>(vocab));
      err = cudaMemcpyAsync(host_logits_bits.data(), logits.data,
                            host_logits_bits.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost, model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program: cudaMemcpyAsync(logits) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }

    err = cudaStreamSynchronize(model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    if (out_logits != nullptr) {
      for (std::int32_t v = 0; v < vocab; ++v) {
        out_logits[v] = bf16_to_f32(host_logits_bits[static_cast<std::size_t>(v)]);
      }
    }
    model->last_step_kernel_count = dispatches;
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_program: ") + e.what());
    return -1;
  }
}

// P3-03 (GitHub #99): the decode-round forward half of run_program_token
// above -- embedding -> all decoder layers -> final norm -> output head --
// stopping short of sampling. Instead of drawing a token, it copies this
// sequence's BF16 logits (device-to-device, before the scratch scope that
// owns them exits) into `logits_column`, one lane's column of the decode
// round's shared `[vocab, batch_size]` staging buffer
// (`model->sampling_decode_logits`), so `ignis_program_decode` can sample
// every lane in the round with one `ninfer::ops::sample` call instead of one
// call per lane. Returns 0 on success, -1 on a kernel/copy error.
int32_t run_program_token_forward_only(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                       int32_t token_id, LinearPolicyMode mode,
                                       void *logits_column) {
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    ninfer::Tensor ids = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
    cudaError_t err = cudaMemcpyAsync(ids.data, &token_id, sizeof(token_id),
                                      cudaMemcpyHostToDevice, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_decode: cudaMemcpyAsync(ids) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    ninfer::Tensor left = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::Tensor right = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::embedding(ids, model->token_embedding, left, model->stream);

    uint64_t dispatches = 0;
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_step_mode(model, pool, seq, layer, left.data, right.data, 1, mode)
          : ignis_gdn_layer_step_mode(model, pool, seq, layer, left.data, right.data, 1, mode);
      if (rc != 0) {
        const char *detail = model->layers[layer].kind == IGNIS_LAYER_GQA
            ? ignis_gqa_layer_last_error()
            : ignis_gdn_layer_last_error();
        set_error("ignis_program_decode: layer " + std::to_string(layer) + " failed: " + detail);
        return -1;
      }
      std::swap(left, right);
      ++dispatches;
    }

    ninfer::Tensor norm_weight(const_cast<void *>(model->final_norm.qdata), ninfer::DType::BF16,
                               {hidden, 1, 1, 1});
    ninfer::Tensor normalized = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(left, norm_weight, model->rms_norm_eps, /*unit_offset=*/true,
                         normalized, model->stream);
    ninfer::Tensor logits = model->scratch->alloc(ninfer::DType::BF16, {vocab, 1, 1, 1});
    ninfer::ops::linear(normalized, model->output_head, logits, model->stream);
    err = cudaMemcpyAsync(logits_column, logits.data,
                          static_cast<std::size_t>(vocab) * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToDevice, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_decode: cudaMemcpyAsync(logits column) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    model->last_step_kernel_count = dispatches;
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_program_decode: ") + e.what());
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
  if (!sampling_size_ok(*sampling)) {
    set_error("ignis_program: unrecognized ignis_sampling_params size " +
              std::to_string(sampling->size));
    return false;
  }
  return true;
}

// P2-02 (GitHub #84): runs one prefill chunk -- embedding for the whole
// chunk, every decoder layer's body dispatched once over the chunk's
// `num_tokens` tokens with no per-layer synchronization, then (only when
// `compute_output` is set -- the chunk containing the span's last
// position) the final norm/head/argmax enqueued for that last token's
// column alone. Everything above is enqueued on the model's stream before
// the one synchronization this function performs, so a chunk is exactly
// one pipelined unit of device work. Returns 0 on success (leaving
// `*out_token_id` and, if `compute_output`, `*out_logits` filled) or -1 on
// a kernel error (message set via set_error, naming `chunk_offset` and
// `seq`); the caller is responsible for not advancing `seq`'s position
// state when this returns -1.
int32_t run_program_chunk(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                          const int32_t *token_ids, uint64_t num_tokens, uint64_t chunk_offset,
                          bool compute_output, const ignis_sampling_params &sampling,
                          int32_t *out_token_id, float *out_logits, LinearPolicyMode mode) {
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  const auto T = static_cast<std::int32_t>(num_tokens);
  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    ninfer::Tensor ids = model->scratch->alloc(ninfer::DType::I32, {T, 1, 1, 1});
    cudaError_t err =
        cudaMemcpyAsync(ids.data, token_ids, static_cast<std::size_t>(T) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_prefill: cudaMemcpyAsync(ids) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    ninfer::Tensor left = model->scratch->alloc(ninfer::DType::BF16, {hidden, T, 1, 1});
    ninfer::Tensor right = model->scratch->alloc(ninfer::DType::BF16, {hidden, T, 1, 1});
    ninfer::ops::embedding(ids, model->token_embedding, left, model->stream);

    uint64_t dispatches = 0;
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_run_body(model, pool, seq, layer, left.data, right.data, num_tokens,
                                     mode)
          : ignis_gdn_layer_run_body(model, pool, seq, layer, left.data, right.data, num_tokens,
                                     mode);
      if (rc != 0) {
        const char *detail = model->layers[layer].kind == IGNIS_LAYER_GQA
            ? ignis_gqa_layer_last_error()
            : ignis_gdn_layer_last_error();
        set_error("ignis_program_prefill: chunk at span offset " + std::to_string(chunk_offset) +
                  " (" + std::to_string(num_tokens) + " tokens) for sequence slot " +
                  std::to_string(seq->slot) + " failed: layer " + std::to_string(layer) + ": " +
                  detail);
        return -1;
      }
      std::swap(left, right);
      ++dispatches;
    }

    // `left` and `right` were swapped once per layer, so after an even
    // layer count the final residual is back in `left`.
    ninfer::Tensor final_residual = left;
    std::vector<std::uint16_t> host_logits_bits;
    if (compute_output) {
      auto *last_token_hidden = static_cast<std::uint8_t *>(final_residual.data) +
                                static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(hidden) *
                                    sizeof(uint16_t);
      const ninfer::Tensor last_token(static_cast<void *>(last_token_hidden), ninfer::DType::BF16,
                                      {hidden, 1, 1, 1});
      const ninfer::Tensor norm_weight(const_cast<void *>(model->final_norm.qdata),
                                       ninfer::DType::BF16, {hidden, 1, 1, 1});
      ninfer::Tensor normalized = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
      ninfer::ops::rmsnorm(last_token, norm_weight, model->rms_norm_eps, /*unit_offset=*/true,
                           normalized, model->stream);
      ninfer::Tensor logits = model->scratch->alloc(ninfer::DType::BF16, {vocab, 1, 1, 1});
      ninfer::ops::linear(normalized, model->output_head, logits, model->stream);
      // The absolute logical position of the span's last token in this
      // chunk: `seq->position` is still the chunk's pre-advance frontier
      // here (the caller advances it only after this function returns 0).
      const auto last_position = static_cast<std::int32_t>(seq->position + T - 1);
      if (sample_single(model, pool, seq, logits, sampling, ninfer::ops::kSamplePurposePrefill,
                        last_position, out_token_id) != 0) {
        return -1;
      }
      if (out_logits != nullptr) {
        host_logits_bits.resize(static_cast<std::size_t>(vocab));
        err = cudaMemcpyAsync(host_logits_bits.data(), logits.data,
                              host_logits_bits.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost, model->stream);
        if (err != cudaSuccess) {
          set_error(std::string("ignis_program_prefill: cudaMemcpyAsync(logits) failed: ") +
                    cudaGetErrorString(err));
          return -1;
        }
      }
    }

    // One synchronization for the whole chunk (P2-02, GitHub #84): every
    // layer's body above only enqueues work, and (when present) so does the
    // output head, so this confirms the entire chunk -- not one layer --
    // completed before the caller advances `seq`'s position state.
    err = cudaStreamSynchronize(model->stream);
    if (err != cudaSuccess) {
      set_error("ignis_program_prefill: chunk at span offset " + std::to_string(chunk_offset) +
                " (" + std::to_string(num_tokens) + ") tokens for sequence slot " +
                std::to_string(seq->slot) +
                " failed: cudaStreamSynchronize: " + cudaGetErrorString(err));
      return -1;
    }

    if (compute_output && out_logits != nullptr) {
      for (std::int32_t v = 0; v < vocab; ++v) {
        out_logits[v] = bf16_to_f32(host_logits_bits[static_cast<std::size_t>(v)]);
      }
    }
    // Only advance every GQA layer's position counter once the synchronize
    // above confirms the whole chunk's device work actually completed
    // (mirrors `ignis_gqa_layer_step`'s own ordering) -- GDN has no
    // separate counter to advance, its state was already updated in place
    // by the enqueued work.
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      if (model->layers[layer].kind == IGNIS_LAYER_GQA) {
        seq->gqa_positions[ignis_gqa_relative_layer(layer)] +=
            static_cast<std::uint32_t>(num_tokens);
      }
    }
    model->last_step_kernel_count = dispatches;
    return 0;
  } catch (const std::exception &e) {
    set_error("ignis_program_prefill: chunk at span offset " + std::to_string(chunk_offset) +
              " (" + std::to_string(num_tokens) + " tokens) for sequence slot " +
              std::to_string(seq->slot) + " failed: " + e.what());
    return -1;
  }
}

// The default route (ADR 0016, P2-02, GitHub #84): cuts `num_tokens` into
// `model->prefill_chunk_tokens`-wide chunks (the last one possibly
// narrower) and runs each as one multi-token traversal of the 64 layers.
// Only the chunk holding the span's last position computes the output
// head, matching the per-token route's contract that only the span's last
// position's successor/logits are observable. A chunk that fails leaves
// `seq` at its pre-chunk position: every earlier chunk in this span already
// committed its position advance, and this loop stops before advancing for
// the failing one.
int32_t run_program_prefill_chunked(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                    const int32_t *token_ids, uint64_t num_tokens,
                                    const ignis_sampling_params &sampling, float *out_logits,
                                    LinearPolicyMode mode) {
  const uint64_t chunk_width = model->prefill_chunk_tokens;
  uint64_t offset = 0;
  while (offset < num_tokens) {
    const uint64_t chunk_len = std::min<uint64_t>(chunk_width, num_tokens - offset);
    const bool is_last_chunk = (offset + chunk_len == num_tokens);
    int32_t successor = -1;
    float *slot_logits = is_last_chunk ? out_logits : nullptr;
    if (run_program_chunk(model, pool, seq, token_ids + offset, chunk_len, offset, is_last_chunk,
                          sampling, &successor, slot_logits, mode) != 0) {
      return -1;
    }
    seq->position += chunk_len;
    if (is_last_chunk) {
      seq->pending_token = successor;
    }
    offset += chunk_len;
  }
  return 0;
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
                                           const struct ignis_sampling_params *sampling,
                                           const struct ignis_prefill_options *options,
                                           float *out_logits) {
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
  // ADR 0016 (P2-02, GitHub #84): NULL means the production defaults
  // (chunked route, engine default compute policy); a non-null options
  // pointer whose `size` this leaf does not recognize is rejected outright,
  // so a caller compiled against a wider future struct fails loudly instead
  // of silently reading past what it wrote. P2-03 (GitHub #85): the
  // `compute_policy` field now reaches a dispatch site -- every NVFP4
  // projection in the program takes the mode's policy (AllowA4 under the
  // engine default, the reference's text-model policy; A16Only under the
  // override, for tests that compare the routes on identical inputs).
  int32_t route = IGNIS_PREFILL_ROUTE_CHUNKED;
  LinearPolicyMode mode = LinearPolicyMode::kEngineDefault;
  if (options != nullptr) {
    if (options->size != sizeof(struct ignis_prefill_options)) {
      set_error("ignis_program_prefill: unrecognized ignis_prefill_options size " +
                std::to_string(options->size));
      return -1;
    }
    if (options->route != IGNIS_PREFILL_ROUTE_CHUNKED &&
        options->route != IGNIS_PREFILL_ROUTE_PER_TOKEN) {
      set_error("ignis_program_prefill: unrecognized prefill route " +
                std::to_string(options->route));
      return -1;
    }
    if (options->compute_policy != IGNIS_PREFILL_COMPUTE_POLICY_ENGINE_DEFAULT &&
        options->compute_policy != IGNIS_PREFILL_COMPUTE_POLICY_A16_ONLY) {
      set_error("ignis_program_prefill: unrecognized compute policy " +
                std::to_string(options->compute_policy));
      return -1;
    }
    route = options->route;
    mode = options->compute_policy == IGNIS_PREFILL_COMPUTE_POLICY_A16_ONLY
               ? LinearPolicyMode::kA16Only
               : LinearPolicyMode::kEngineDefault;
  }

  const auto began = std::chrono::steady_clock::now();
  int32_t rc = 0;
  if (route == IGNIS_PREFILL_ROUTE_PER_TOKEN) {
    // Test-only self-oracle route (ADR 0016): the per-token loop, one
    // traversal and one synchronization per layer per token, under the
    // call's compute-policy mode (P2-03, GitHub #85).
    for (uint64_t i = 0; i < num_tokens; ++i) {
      int32_t successor = -1;
      // Only the span's last position is the one whose logits GitHub #72
      // needs (it decides the successor ignis_program_decode emits first) --
      // every earlier position stays argmax-only.
      float *slot_logits = (i + 1 == num_tokens) ? out_logits : nullptr;
      if (run_program_token(model, pool, seq, token_ids[i], *sampling, &successor, slot_logits,
                            mode) != 0) {
        rc = -1;
        break;
      }
      seq->pending_token = successor;
      ++seq->position;
    }
  } else {
    rc = run_program_prefill_chunked(model, pool, seq, token_ids, num_tokens, *sampling,
                                     out_logits, mode);
  }
  if (rc != 0) {
    return rc;
  }
  model->last_step_micros = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - began).count());
  return 0;
}

// P3-03 (GitHub #99): one call over every decode-ready lane, each sampled
// with its own `sampling[i]`. Every lane's forward pass runs first (still
// one sequential per-sequence traversal each, `run_program_token_forward_only`
// above -- this ticket does not fuse the layer bodies across sequences),
// writing its logits into its own column of the round's shared
// `[vocab, batch_size]` staging buffer; then one `ninfer::ops::sample` call
// draws every lane's successor together. The round is atomic: no sequence's
// `pending_token`/`position` advances unless every lane's forward pass and
// the batched sample both succeed, so a mid-round failure never leaves one
// lane's state ahead of another's.
extern "C" int32_t ignis_program_decode(struct ignis_model *model,
                                          struct ignis_seq_pool *pool,
                                          struct ignis_seq *const *sequences,
                                          uint64_t batch_size,
                                          const struct ignis_sampling_params *sampling,
                                          int32_t *out_token_ids) {
  if (model == nullptr || pool == nullptr || sequences == nullptr || sampling == nullptr ||
      out_token_ids == nullptr || batch_size == 0) {
    set_error("ignis_program_decode: null argument or empty batch");
    return -1;
  }
  if (batch_size > IGNIS_DECODE_MAX_BATCH) {
    set_error("ignis_program_decode: batch_size " + std::to_string(batch_size) +
              " exceeds IGNIS_DECODE_MAX_BATCH (" + std::to_string(IGNIS_DECODE_MAX_BATCH) + ")");
    return -1;
  }
  for (uint64_t i = 0; i < batch_size; ++i) {
    if (!sampling_size_ok(sampling[i])) {
      set_error("ignis_program_decode: unrecognized ignis_sampling_params size " +
                std::to_string(sampling[i].size) + " at index " + std::to_string(i));
      return -1;
    }
  }

  const auto began = std::chrono::steady_clock::now();
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  const auto batch = static_cast<std::int32_t>(batch_size);
  std::vector<int32_t> emitted(batch_size, -1);
  std::vector<std::int32_t> positions(batch_size, 0);
  std::vector<ninfer::ops::SamplingConfig> configs(batch_size);

  try {
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
      emitted[i] = seq->pending_token;
      positions[i] = static_cast<std::int32_t>(seq->position);
      configs[i] = to_sampling_config(sampling[i], pool->token_counts_for(seq->slot));
    }

    // P3-05 (GitHub #102, ADR 0019): a decode graph is captured per exact
    // batch width, never padded -- a round at this exact width replays it
    // when one is ready, otherwise falls back to the eager per-lane loop
    // below unchanged. Both paths share the same sampling staging buffers
    // and the same round semantics (atomic: no sequence's
    // pending_token/position advances unless the whole round succeeds).
    const bool use_graph =
        batch_size >= 1 && batch_size <= IGNIS_DECODE_MAX_BATCH &&
        model->decode_graph_ready[batch_size - 1];

    cudaError_t err =
        cudaMemcpyAsync(model->sampling_decode_configs->p, configs.data(),
                        batch_size * sizeof(ninfer::ops::SamplingConfig), cudaMemcpyHostToDevice,
                        model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_decode: cudaMemcpyAsync(sampling configs) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    err = cudaMemcpyAsync(model->sampling_decode_positions->p, positions.data(),
                          batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                          model->stream);
    if (err != cudaSuccess) {
      set_error(
          std::string("ignis_program_decode: cudaMemcpyAsync(sampling positions) failed: ") +
          cudaGetErrorString(err));
      return -1;
    }

    uint64_t dispatches = 0;
    if (use_graph) {
      std::vector<std::int32_t> slots(batch_size, 0);
      for (uint64_t i = 0; i < batch_size; ++i) {
        slots[i] = sequences[i]->slot;
      }
      err = cudaMemcpyAsync(model->decode_graph_token_ids->p, emitted.data(),
                            batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                            model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaMemcpyAsync(graph token ids) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
      err = cudaMemcpyAsync(model->decode_graph_slots->p, slots.data(),
                            batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                            model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaMemcpyAsync(graph slots) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
      err = cudaGraphLaunch(model->decode_graph_exec[batch_size - 1], model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaGraphLaunch failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
      // A replayed graph's kernel_count is whatever it captured, which this
      // call does not separately track -- kernel_count stays meaningful for
      // the eager path; graph_launches (below) is how a caller tells the
      // two apart.
    } else {
      for (uint64_t i = 0; i < batch_size; ++i) {
        ignis_seq *seq = sequences[i];
        void *column = static_cast<std::uint8_t *>(model->sampling_decode_logits->p) +
            i * static_cast<std::size_t>(vocab) * sizeof(std::uint16_t);
        // P2-03 (GitHub #85): decode takes the engine's own compute-policy
        // mode (ADR 0016: the flat decode ABI has no options struct, so the
        // `A16_ONLY` override is reachable only through the prefill entry
        // point) -- every NVFP4 projection in the decode round runs under
        // AllowA4, the reference's text-model policy.
        if (run_program_token_forward_only(model, pool, seq, emitted[i],
                                           LinearPolicyMode::kEngineDefault, column) != 0) {
          return -1;
        }
        dispatches += model->last_step_kernel_count;
      }

      const ninfer::Tensor logits_tensor(model->sampling_decode_logits->p, ninfer::DType::BF16,
                                         {vocab, batch, 1, 1});
      ninfer::Tensor out_tensor(model->sampling_decode_out->p, ninfer::DType::I32, {batch, 1, 1, 1});
      const ninfer::Tensor positions_tensor(model->sampling_decode_positions->p, ninfer::DType::I32,
                                            {batch, 1, 1, 1});
      ninfer::DeviceArena::Scope workspace_scope = model->sampling_workspace->scope();
      ninfer::ops::sample(
          logits_tensor, out_tensor, vocab,
          static_cast<const ninfer::ops::SamplingConfig *>(model->sampling_decode_configs->p),
          positions_tensor, ninfer::ops::kSamplePurposeDecode, *model->sampling_workspace,
          model->stream);
    }

    std::vector<int32_t> successors(batch_size, -1);
    err = cudaMemcpyAsync(successors.data(), model->sampling_decode_out->p,
                          batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_decode: cudaMemcpyAsync(sampled tokens) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    err = cudaStreamSynchronize(model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_decode: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    for (uint64_t i = 0; i < batch_size; ++i) {
      out_token_ids[i] = emitted[i];
      sequences[i]->pending_token = successors[i];
      ++sequences[i]->position;
    }
    model->last_step_kernel_count = dispatches;
    model->last_step_graph_launches = use_graph ? 1 : 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_program_decode: ") + e.what());
    return -1;
  }

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
  // P3-03 (GitHub #99): the sampling staging buffers and per-slot penalty
  // counts are real device allocations too, small as they are next to the
  // weights and KV/GDN pools -- "VRAM reported" means all of it. P3-05
  // (GitHub #102, ADR 0019) adds the decode graphs' own scratch and staging
  // reservation -- separate from `scratch` above, so it is also separate
  // here.
  out_stats->vram_bytes =
      model->vram_bytes + model->scratch->capacity() + pool->kv_arena.capacity() +
      pool->gdn_arena.capacity() + pool->sampling_counts.bytes +
      model->sampling_single_configs->bytes + model->sampling_single_positions->bytes +
      model->sampling_single_out->bytes + model->sampling_decode_configs->bytes +
      model->sampling_decode_positions->bytes + model->sampling_decode_out->bytes +
      model->sampling_decode_logits->bytes + model->sampling_workspace->capacity() +
      model->decode_graph_scratch->capacity() + model->decode_graph_token_ids->bytes +
      model->decode_graph_slots->bytes + model->decode_graph_zero->bytes;
  out_stats->last_step_micros = model->last_step_micros;
  out_stats->kernel_count = model->last_step_kernel_count;
  out_stats->graph_launches = model->last_step_graph_launches;
  uint32_t ready_mask = 0;
  for (uint32_t width = 1; width <= IGNIS_DECODE_MAX_BATCH; ++width) {
    if (model->decode_graph_ready[width - 1]) {
      ready_mask |= (1u << (width - 1));
    }
  }
  out_stats->decode_graph_ready_mask = ready_mask;
  return 0;
}
