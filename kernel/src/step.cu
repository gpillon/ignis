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

#include "dflash2_drafter.h"
#include "ignis_gdn_layer.h"
#include "ignis_gqa_layer.h"
#include "ignis_seq_internal.h"
#include "layer_internal.h"
#include "model_internal.h"
#include "permitted_tokens.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
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

// P6-06 (GitHub #242): a lane's permitted set is refused wherever it cannot
// be honoured, rather than ignored -- a constraint silently dropped is a
// wrong answer that looks like a right one.
bool unconstrained(const ignis_sampling_params &sampling) {
  return sampling.permitted_count == 0;
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
                      std::int32_t purpose, std::int32_t position, int32_t *out_token_id,
                      float *out_permitted_prob) {
  // GitHub #242: the prefill's own draw is the first token of a constrained
  // run, so this path masks too -- staged through the decode buffers' lane
  // 0, which no decode round is using while a prefill holds the stream.
  const bool constrained = !unconstrained(sampling);
  if (constrained) {
    if (sampling.permitted_count > IGNIS_MAX_PERMITTED_TOKENS) {
      set_error("ignis_program: permitted_count " + std::to_string(sampling.permitted_count) +
                " exceeds IGNIS_MAX_PERMITTED_TOKENS (" +
                std::to_string(IGNIS_MAX_PERMITTED_TOKENS) + ")");
      return -1;
    }
    if (sampling.permitted_ids == nullptr) {
      set_error("ignis_program: permitted_ids is null with a nonzero count");
      return -1;
    }
    std::vector<std::int32_t> row(IGNIS_MAX_PERMITTED_TOKENS, -1);
    const auto vocab = static_cast<std::int32_t>(model->vocab);
    for (uint32_t k = 0; k < sampling.permitted_count; ++k) {
      const int32_t id = sampling.permitted_ids[k];
      if (id < 0 || id >= vocab) {
        set_error("ignis_program: permitted id " + std::to_string(id) +
                  " is outside the vocabulary");
        return -1;
      }
      row[k] = id;
    }
    const std::int32_t count = static_cast<std::int32_t>(sampling.permitted_count);
    cudaError_t staged =
        cudaMemcpyAsync(model->sampling_decode_permitted->p, row.data(),
                        row.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, model->stream);
    if (staged == cudaSuccess) {
      staged = cudaMemcpyAsync(model->sampling_decode_permitted_counts->p, &count, sizeof(count),
                               cudaMemcpyHostToDevice, model->stream);
    }
    if (staged != cudaSuccess) {
      set_error(std::string("ignis_program: cudaMemcpyAsync(permitted set) failed: ") +
                cudaGetErrorString(staged));
      return -1;
    }
    if (ignis_permit_mask(logits.data, vocab, 1,
                          static_cast<const int32_t *>(model->sampling_decode_permitted->p),
                          static_cast<const int32_t *>(model->sampling_decode_permitted_counts->p),
                          IGNIS_MAX_PERMITTED_TOKENS, model->stream) != 0) {
      set_error("ignis_program: permitted-set mask launch failed");
      return -1;
    }
  }
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
  // GitHub #242: the drawn token's share of its own set, while the logits
  // are still here. Read from lane 0's staging, the same row the mask used.
  if (constrained && out_permitted_prob != nullptr) {
    if (ignis_permit_probability(
            logits.data, static_cast<std::int32_t>(model->vocab), 1,
            static_cast<const int32_t *>(model->sampling_decode_permitted->p),
            static_cast<const int32_t *>(model->sampling_decode_permitted_counts->p),
            IGNIS_MAX_PERMITTED_TOKENS,
            static_cast<const int32_t *>(model->sampling_single_out->p),
            static_cast<float *>(model->sampling_decode_permitted_probs->p),
            model->stream) != 0) {
      set_error("ignis_program: permitted-set probability launch failed");
      return -1;
    }
    err = cudaMemcpyAsync(out_permitted_prob, model->sampling_decode_permitted_probs->p,
                          sizeof(float), cudaMemcpyDeviceToHost, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program: cudaMemcpyAsync(permitted probability) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
  } else if (out_permitted_prob != nullptr) {
    *out_permitted_prob = 0.0f;
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
                          int32_t *out_token_id, float *out_logits, LinearPolicyMode mode,
                          float *permitted_prob_out) {
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
                      static_cast<std::int32_t>(seq->position), out_token_id,
                      permitted_prob_out) != 0) {
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

// P5-03 (GitHub #152): the drafter's weights come with the model and its
// per-sequence window with the pool, so the two must have been built for the
// same speculative backend. A drafter-bearing model over a plain pool would
// have no window to write; a plain model over a drafter pool would leave
// every window unwritten while snapshots carry it as if it were state. Either
// is refused by name before any device work.
bool dflash2_matches_load(const ignis_model *model, const ignis_seq_pool *pool, const char *op) {
  if (pool->speculative_backend == model->speculative_backend) {
    return true;
  }
  set_error(std::string(op) + ": the sequence pool was built for speculative backend " +
            std::to_string(pool->speculative_backend) + ", this model was loaded with " +
            std::to_string(model->speculative_backend) +
            "; the drafter's weights and its per-sequence window come as a pair");
  return false;
}

// The first absolute position of a prefill span whose target features reach
// the drafter's window: the span's last kIgnisDflash2WindowTokens positions,
// or all of it when it is shorter. Anything earlier would be overwritten in
// the ring before the span ends, so it is never tapped or projected -- which
// is what bounds the drafter's TTFT cost by the window, not by the prompt.
std::uint64_t dflash2_tap_from(std::uint64_t span_start, std::uint64_t num_tokens) {
  return span_start + num_tokens -
         std::min<std::uint64_t>(num_tokens, kIgnisDflash2WindowTokens);
}

void copy_i32_to_device(ninfer::Tensor &dst, const std::int32_t *src, std::size_t count,
                        cudaStream_t stream) {
  const cudaError_t err = cudaMemcpyAsync(dst.data, src, count * sizeof(std::int32_t),
                                          cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync(i32) failed: ") +
                             cudaGetErrorString(err));
  }
}

// P5-03 (GitHub #152): the drafter's context append over one chunk's feature
// taps -- the reference's `dflash2_append_context`, for one lane. `features`
// is BF16 [5 x hidden, count]: the target's layer-5/19/33/47/61 outputs for
// positions `host_scalars[2..]`, concatenated per column; the append itself
// (kernel/src/dflash2_drafter.h, shared with the verify round since P5-05)
// lands them in the slot's lane of the window. `host_scalars` is [count,
// lane, positions...] and must outlive the caller's synchronize: every copy
// here is enqueued on the model's stream.
void append_dflash2_context(ignis_model *model, ignis_seq_pool *pool,
                            const ninfer::Tensor &features, const std::int32_t *host_scalars,
                            std::int32_t count) {
  const cudaStream_t stream = model->stream;
  ninfer::Tensor commit_count = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
  ninfer::Tensor lane         = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
  ninfer::Tensor positions    = model->scratch->alloc(ninfer::DType::I32, {count, 1, 1, 1});
  copy_i32_to_device(commit_count, host_scalars, 1, stream);
  copy_i32_to_device(lane, host_scalars + 1, 1, stream);
  copy_i32_to_device(positions, host_scalars + 2, static_cast<std::size_t>(count), stream);
  ignis_dflash2_append_context(
      model, pool, *model->scratch, features, positions, commit_count, lane,
      {static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(count)});
}

// GitHub #157: the anchor taps a verify round at extent 0 left on `seq`
// (`ignis_seq::dflash2_pending`), appended into its window at the anchor's
// position ahead of whatever continues the sequence. Returns whether it
// enqueued the append; the caller moves the frontier past the anchor, and
// drops the taps, only once its synchronize confirms it. `host_scalars` must
// outlive that synchronize. Allocates from `model->scratch` under the
// caller's scope.
bool append_dflash2_pending(ignis_model *model, ignis_seq_pool *pool, const ignis_seq &seq,
                            std::vector<std::int32_t> &host_scalars) {
  if (!seq.dflash2_pending) {
    return false;
  }
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  ninfer::Tensor features = model->scratch->alloc(
      ninfer::DType::BF16, {static_cast<std::int32_t>(kDflash2TapLayers.size()) * hidden, 1, 1, 1});
  if (seq.dflash2_pending_features.size() != features.bytes()) {
    throw std::runtime_error("the carried anchor taps are " +
                             std::to_string(seq.dflash2_pending_features.size()) +
                             " bytes, not one feature column");
  }
  const cudaError_t err =
      cudaMemcpyAsync(features.data, seq.dflash2_pending_features.data(), features.bytes(),
                      cudaMemcpyHostToDevice, model->stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync(carried anchor taps) failed: ") +
                             cudaGetErrorString(err));
  }
  host_scalars = {1, seq.slot, static_cast<std::int32_t>(seq.dflash2_position)};
  append_dflash2_context(model, pool, features, host_scalars.data(), 1);
  return true;
}

// ---------------------------------------------------------------------------
// GitHub #92, acceptance criterion 1: where per-chunk prefill wall time goes.
//
// Diagnostic scaffolding, not a production path. It is inert unless the
// environment names a file in IGNIS_CHUNK_PROFILE; with the variable unset
// (every production run and every other test) the cost is one cached boolean
// test per chunk and not one CUDA event is created or recorded.
//
// Per chunk it separates, into one JSONL record:
//   cpu_enqueue_ms  host wall issuing the chunk's launches (chunk entry up to
//                   just before cudaStreamSynchronize) -- the dispatch cost
//                   as the host pays it
//   sync_ms         host wall blocked inside cudaStreamSynchronize
//   entry_gap_ms    device idle between the PREVIOUS chunk's last op and this
//                   chunk's first -- the bubble the forced per-chunk
//                   synchronization actually opens on the device
//   gpu_span_ms     device wall from this chunk's first enqueued op to its
//                   last
//   embed_ms        device span of the id memcpy + embedding
//   layers_ms       sum of the 64 layer bodies' own device spans (compute)
//   head_ms         device span of the final norm / head / sample (last chunk
//                   of a span only; 0 elsewhere)
//   layer_gap_ms    gpu_span - embed - layers - head: device idle *between*
//                   layer bodies, i.e. launch latency the host failed to hide
//
// Events are recorded on the model's own stream, so they order with the work
// they bracket and add no synchronization of their own; every elapsed time is
// read after the chunk's existing synchronize, when all of them have
// completed.
class ChunkProfiler {
public:
  // One profiler per thread. The CUDA events below bracket one chunk's work,
  // so two threads prefilling at once must not share them. They cannot today
  // -- `model->scratch` is a bump allocator with no synchronization and the
  // model owns a single stream, so concurrent prefill on one model is already
  // excluded -- but roadmap phase 6 is exactly about lifting that, and this
  // should not be the thing that then has to be found. The destination file
  // stays process-wide, so every thread's records land in one place.
  static ChunkProfiler &instance() {
    thread_local ChunkProfiler profiler;
    return profiler;
  }

  bool enabled() const { return out_ != nullptr; }

  // A new prefill span: resets the chunk counter and drops the carried-over
  // end event, so the first chunk of a span reports no entry gap (the gap
  // before it is the caller's, not the chunk loop's).
  void begin_span(uint64_t num_tokens, uint32_t chunk_width) {
    if (!enabled()) { return; }
    ++span_index_;
    chunk_index_ = 0;
    have_prev_end_ = false;
    span_tokens_ = num_tokens;
    span_chunk_width_ = chunk_width;
  }

  // Grows the event pool to cover `layers` layer bodies. Called on the chunk
  // path before the first record of a chunk.
  void ensure_events(std::size_t layers) {
    if (!enabled()) { return; }
    if (begin_ == nullptr) {
      cudaEventCreate(&begin_);
      cudaEventCreate(&head_begin_);
      cudaEventCreate(&end_[0]);
      cudaEventCreate(&end_[1]);
    }
    while (layer_begin_.size() < layers) {
      cudaEvent_t b = nullptr;
      cudaEvent_t e = nullptr;
      cudaEventCreate(&b);
      cudaEventCreate(&e);
      layer_begin_.push_back(b);
      layer_end_.push_back(e);
    }
  }

  void record_begin(cudaStream_t stream) {
    if (enabled()) { cudaEventRecord(begin_, stream); }
  }
  void record_layer_begin(std::size_t layer, cudaStream_t stream) {
    if (enabled()) { cudaEventRecord(layer_begin_[layer], stream); }
  }
  void record_layer_end(std::size_t layer, cudaStream_t stream) {
    if (enabled()) { cudaEventRecord(layer_end_[layer], stream); }
  }
  void record_head_begin(cudaStream_t stream) {
    if (enabled()) { cudaEventRecord(head_begin_, stream); }
  }
  void record_end(cudaStream_t stream) {
    if (enabled()) { cudaEventRecord(end_[parity_], stream); }
  }

  // Called after the chunk's own cudaStreamSynchronize returned success, so
  // every event above has completed and is readable without blocking.
  void report(std::size_t layers, uint64_t chunk_offset, uint64_t chunk_tokens,
              bool compute_output, double cpu_enqueue_ms, double sync_ms) {
    if (!enabled()) { return; }
    const double gpu_span_ms = elapsed(begin_, end_[parity_]);
    const double embed_ms = layers > 0 ? elapsed(begin_, layer_begin_[0]) : 0.0;
    double layers_ms = 0.0;
    for (std::size_t layer = 0; layer < layers; ++layer) {
      layers_ms += elapsed(layer_begin_[layer], layer_end_[layer]);
    }
    const double head_ms = elapsed(head_begin_, end_[parity_]);
    const double layer_gap_ms = gpu_span_ms - embed_ms - layers_ms - head_ms;
    const double entry_gap_ms = have_prev_end_ ? elapsed(end_[1 - parity_], begin_) : 0.0;
    std::fprintf(out_,
                 "{\"thread\":%llu,\"span\":%llu,\"span_tokens\":%llu,"
                 "\"chunk_width\":%u,\"chunk\":%llu,"
                 "\"chunk_offset\":%llu,\"chunk_tokens\":%llu,\"last\":%d,\"layers\":%llu,"
                 "\"cpu_enqueue_ms\":%.4f,\"sync_ms\":%.4f,\"entry_gap_ms\":%.4f,"
                 "\"gpu_span_ms\":%.4f,\"embed_ms\":%.4f,\"layers_ms\":%.4f,\"head_ms\":%.4f,"
                 "\"layer_gap_ms\":%.4f}\n",
                 static_cast<unsigned long long>(thread_key_),
                 static_cast<unsigned long long>(span_index_),
                 static_cast<unsigned long long>(span_tokens_), span_chunk_width_,
                 static_cast<unsigned long long>(chunk_index_),
                 static_cast<unsigned long long>(chunk_offset),
                 static_cast<unsigned long long>(chunk_tokens), compute_output ? 1 : 0,
                 static_cast<unsigned long long>(layers), cpu_enqueue_ms, sync_ms, entry_gap_ms,
                 gpu_span_ms, embed_ms, layers_ms, head_ms, layer_gap_ms);
    std::fflush(out_);
    ++chunk_index_;
    have_prev_end_ = true;
    parity_ = 1 - parity_;
  }

  // Per-layer device spans for one chunk, emitted separately so the JSONL
  // above stays one line per chunk. Written only when the profile asked for
  // the per-layer detail (IGNIS_CHUNK_PROFILE_LAYERS set).
  void report_layers(std::size_t layers, uint64_t chunk_offset) {
    if (!enabled() || !per_layer_) { return; }
    for (std::size_t layer = 0; layer < layers; ++layer) {
      std::fprintf(out_,
                   "{\"thread\":%llu,\"span\":%llu,\"chunk_offset\":%llu,"
                   "\"layer\":%llu,\"layer_ms\":%.4f,"
                   "\"gap_before_ms\":%.4f}\n",
                   static_cast<unsigned long long>(thread_key_),
                   static_cast<unsigned long long>(span_index_),
                   static_cast<unsigned long long>(chunk_offset),
                   static_cast<unsigned long long>(layer),
                   elapsed(layer_begin_[layer], layer_end_[layer]),
                   layer == 0 ? elapsed(begin_, layer_begin_[0])
                              : elapsed(layer_end_[layer - 1], layer_begin_[layer]));
    }
    std::fflush(out_);
  }

private:
  // Opened once for the process, on whichever thread profiles first. A shared
  // `std::FILE *` needs no lock of ours: `std::fprintf` locks the stream
  // internally on both MSVC and POSIX, so records interleave whole rather
  // than tearing.
  static std::FILE *shared_sink() {
    static std::FILE *const sink = []() -> std::FILE * {
      const char *path = std::getenv("IGNIS_CHUNK_PROFILE");
      if (path == nullptr || path[0] == 0) { return nullptr; }
      return std::fopen(path, "ab");
    }();
    return sink;
  }

  static bool shared_per_layer() {
    static const bool on = std::getenv("IGNIS_CHUNK_PROFILE_LAYERS") != nullptr;
    return on;
  }

  // A small dense id per profiling thread, in first-chunk order, so records
  // from different threads stay separable: `span` and `chunk` below are
  // per-thread counters and would otherwise collide.
  static uint64_t next_thread_key() {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
  }

  ChunkProfiler() : out_(shared_sink()), per_layer_(shared_per_layer()) {}

  static double elapsed(cudaEvent_t from, cudaEvent_t to) {
    float ms = 0.0F;
    if (cudaEventElapsedTime(&ms, from, to) != cudaSuccess) { return -1.0; }
    return static_cast<double>(ms);
  }

  std::FILE *out_ = nullptr;
  bool per_layer_ = false;
  const uint64_t thread_key_ = next_thread_key();
  std::vector<cudaEvent_t> layer_begin_;
  std::vector<cudaEvent_t> layer_end_;
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t head_begin_ = nullptr;
  cudaEvent_t end_[2] = {nullptr, nullptr};
  int parity_ = 0;
  bool have_prev_end_ = false;
  uint64_t span_index_ = 0;
  uint64_t chunk_index_ = 0;
  uint64_t span_tokens_ = 0;
  uint32_t span_chunk_width_ = 0;
};

// GitHub #178: one prefill span's multimodal inputs, read off its options.
// The default value is a text span.
struct SpanMultimodal {
  const int32_t *positions = nullptr; // axis-major [3, span_tokens]
  uint64_t span_tokens = 0;
  int32_t rope_delta = 0;
  const ignis_media_embedding *media = nullptr;
  const int32_t *scatter = nullptr; // span-relative, strictly increasing
  uint32_t count = 0;
  uint32_t first_column = 0;
};

bool validate_span_multimodal(const ignis_model *model, int32_t route, const SpanMultimodal &span) {
  const auto refuse = [](const std::string &why) {
    set_error("ignis_program_prefill: " + why);
    return false;
  };
  if (span.positions == nullptr) {
    return (span.media == nullptr && span.count == 0) ||
           refuse("media columns need the span's multimodal positions");
  }
  if (model->vision_output == nullptr) {
    return refuse("a multimodal span on a model loaded without vision");
  }
  if (route != IGNIS_PREFILL_ROUTE_CHUNKED) {
    return refuse("the per-token route has no multimodal form");
  }
  if (span.media == nullptr) {
    return span.count == 0 || refuse("scatter indices without a media embedding");
  }
  if (span.media->model != model) {
    return refuse("the media embedding was encoded by another model");
  }
  if (span.count == 0 || span.scatter == nullptr) {
    return refuse("a media embedding needs at least one scatter index");
  }
  if (static_cast<uint64_t>(span.first_column) + span.count >
      static_cast<uint64_t>(span.media->columns)) {
    return refuse("columns " + std::to_string(span.first_column) + "+" +
                  std::to_string(span.count) + " exceed the embedding's " +
                  std::to_string(span.media->columns));
  }
  for (uint32_t i = 0; i < span.count; ++i) {
    const int32_t index = span.scatter[i];
    if (index < 0 || static_cast<uint64_t>(index) >= span.span_tokens ||
        (i > 0 && index <= span.scatter[i - 1])) {
      return refuse("scatter index " + std::to_string(i) + " (" + std::to_string(index) +
                    ") is out of order or outside the span");
    }
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
//
// P5-03 (GitHub #152): on a pool with the drafter, the chunk also taps the
// target's layer-5/19/33/47/61 outputs for its positions at or past
// `dflash2_tap_from` into chunk-scoped scratch and appends their projected
// context to the drafter's window before the same synchronization, so the
// window and the sequence's other sections complete together.
int32_t run_program_chunk(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                          const int32_t *token_ids, uint64_t num_tokens, uint64_t chunk_offset,
                          uint64_t dflash2_tap_from, bool compute_output,
                          const ignis_sampling_params &sampling, int32_t *out_token_id,
                          float *out_logits, LinearPolicyMode mode,
                          const SpanMultimodal &multimodal, float *permitted_prob_out) {
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  const auto T = static_cast<std::int32_t>(num_tokens);
  // The chunk's own columns the drafter taps: [tap_first, T).
  const std::uint64_t chunk_start = seq->position;
  const std::int32_t tap_first =
      dflash2_tap_from > chunk_start
          ? static_cast<std::int32_t>(std::min<std::uint64_t>(dflash2_tap_from - chunk_start, num_tokens))
          : 0;
  const std::int32_t tap_count = pool->has_dflash2() ? T - tap_first : 0;
  // GitHub #92 criterion 1: inert unless IGNIS_CHUNK_PROFILE is set.
  ChunkProfiler &profiler = ChunkProfiler::instance();
  profiler.ensure_events(model->layers.size());
  const auto cpu_chunk_start = std::chrono::steady_clock::now();
  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    profiler.record_begin(model->stream);
    // GitHub #157: a span continuing an extent-0 round writes that round's
    // anchor first, so a span wider than the ring still overwrites it in
    // order. Its own scope: none of it outlives the enqueued append.
    std::vector<std::int32_t> pending_scalars;
    bool pending_appended = false;
    if (chunk_offset == 0 && pool->has_dflash2()) {
      ninfer::DeviceArena::Scope pending_scope = model->scratch->scope();
      pending_appended = append_dflash2_pending(model, pool, *seq, pending_scalars);
    }
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

    // GitHub #178: a multimodal chunk rotates at its slice of the span's
    // three axes, and its placeholder rows take the media item's columns.
    // Both host staging vectors outlive the chunk's synchronize below.
    const void *rope_positions = nullptr;
    std::vector<std::int32_t> rope_positions_host;
    std::vector<std::int32_t> scatter_host;
    if (multimodal.positions != nullptr) {
      rope_positions_host.resize(3 * static_cast<std::size_t>(T));
      for (std::size_t axis = 0; axis < 3; ++axis) {
        std::copy_n(multimodal.positions + axis * multimodal.span_tokens + chunk_offset, T,
                    rope_positions_host.data() + axis * static_cast<std::size_t>(T));
      }
      ninfer::Tensor rotation = model->scratch->alloc(ninfer::DType::I32, {T, 3, 1, 1});
      err = cudaMemcpyAsync(rotation.data, rope_positions_host.data(),
                            rope_positions_host.size() * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice, model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_prefill: cudaMemcpyAsync(rope positions) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
      rope_positions = rotation.data;
    }
    if (multimodal.media != nullptr) {
      const int32_t *const scatter_end = multimodal.scatter + multimodal.count;
      const int32_t *const first = std::lower_bound(
          multimodal.scatter, scatter_end, static_cast<std::int32_t>(chunk_offset));
      const int32_t *const last =
          std::lower_bound(first, scatter_end, static_cast<std::int32_t>(chunk_offset + T));
      const auto count = static_cast<std::int32_t>(last - first);
      if (count > 0) {
        scatter_host.resize(static_cast<std::size_t>(count));
        for (std::int32_t i = 0; i < count; ++i) {
          scatter_host[static_cast<std::size_t>(i)] =
              first[i] - static_cast<std::int32_t>(chunk_offset);
        }
        ninfer::Tensor indices = model->scratch->alloc(ninfer::DType::I32, {count, 1, 1, 1});
        err = cudaMemcpyAsync(indices.data, scatter_host.data(),
                              scatter_host.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                              model->stream);
        if (err != cudaSuccess) {
          set_error(std::string("ignis_program_prefill: cudaMemcpyAsync(scatter indices) failed: ") +
                    cudaGetErrorString(err));
          return -1;
        }
        const std::size_t column =
            multimodal.first_column + static_cast<std::size_t>(first - multimodal.scatter);
        const ninfer::Tensor columns(static_cast<std::uint8_t *>(model->vision_output->p) +
                                         column * static_cast<std::size_t>(hidden) *
                                             sizeof(std::uint16_t),
                                     ninfer::DType::BF16, {hidden, count, 1, 1});
        ninfer::ops::scatter(columns, indices, left, model->stream);
      }
    }

    // P5-03 (GitHub #152): the feature taps, never persisted and never a
    // section -- the scratch scope above frees them with the chunk.
    ninfer::Tensor features;
    std::vector<std::int32_t> dflash2_scalars;
    if (tap_count > 0) {
      features = model->scratch->alloc(
          ninfer::DType::BF16,
          {static_cast<std::int32_t>(kDflash2TapLayers.size()) * hidden, tap_count, 1, 1});
      dflash2_scalars.reserve(static_cast<std::size_t>(tap_count) + 2);
      dflash2_scalars.push_back(tap_count);
      dflash2_scalars.push_back(seq->slot);
      for (std::int32_t i = 0; i < tap_count; ++i) {
        dflash2_scalars.push_back(static_cast<std::int32_t>(chunk_start + tap_first + i));
      }
    }

    uint64_t dispatches = 0;
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      profiler.record_layer_begin(layer, model->stream);
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_run_body(model, pool, seq, layer, left.data, right.data, num_tokens,
                                     mode, rope_positions)
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
      profiler.record_layer_end(layer, model->stream);
      std::swap(left, right);
      ++dispatches;
      if (tap_count > 0) {
        // `left` is this layer's output now. Tap j occupies rows
        // [j * hidden, (j + 1) * hidden) of every feature column.
        const auto tap = std::find(kDflash2TapLayers.begin(), kDflash2TapLayers.end(), layer);
        if (tap != kDflash2TapLayers.end()) {
          const auto j = static_cast<std::size_t>(tap - kDflash2TapLayers.begin());
          const std::size_t row_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
          err = cudaMemcpy2DAsync(
              static_cast<std::uint8_t *>(features.data) + j * row_bytes,
              static_cast<std::size_t>(features.nb[1]),
              static_cast<const std::uint8_t *>(left.data) +
                  static_cast<std::size_t>(tap_first) * row_bytes,
              static_cast<std::size_t>(left.nb[1]), row_bytes, static_cast<std::size_t>(tap_count),
              cudaMemcpyDeviceToDevice, model->stream);
          if (err != cudaSuccess) {
            set_error("ignis_program_prefill: feature tap at layer " + std::to_string(layer) +
                      " failed: " + cudaGetErrorString(err));
            return -1;
          }
        }
      }
    }
    if (tap_count > 0) {
      append_dflash2_context(model, pool, features, dflash2_scalars.data(), tap_count);
    }
    profiler.record_head_begin(model->stream);

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
                        last_position, out_token_id, permitted_prob_out) != 0) {
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
    profiler.record_end(model->stream);
    const auto cpu_enqueue_end = std::chrono::steady_clock::now();
    err = cudaStreamSynchronize(model->stream);
    if (err != cudaSuccess) {
      set_error("ignis_program_prefill: chunk at span offset " + std::to_string(chunk_offset) +
                " (" + std::to_string(num_tokens) + ") tokens for sequence slot " +
                std::to_string(seq->slot) +
                " failed: cudaStreamSynchronize: " + cudaGetErrorString(err));
      return -1;
    }
    if (profiler.enabled()) {
      const auto cpu_sync_end = std::chrono::steady_clock::now();
      const auto to_ms = [](std::chrono::steady_clock::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
      };
      profiler.report_layers(model->layers.size(), chunk_offset);
      profiler.report(model->layers.size(), chunk_offset, num_tokens, compute_output,
                      to_ms(cpu_enqueue_end - cpu_chunk_start),
                      to_ms(cpu_sync_end - cpu_enqueue_end));
    }

    if (compute_output && out_logits != nullptr) {
      for (std::int32_t v = 0; v < vocab; ++v) {
        out_logits[v] = bf16_to_f32(host_logits_bits[static_cast<std::size_t>(v)]);
      }
    }
    // Only advance every layer's position counter once the synchronize above
    // confirms the whole chunk's device work actually completed (mirrors
    // `ignis_gqa_layer_step`'s and `ignis_gdn_layer_step`'s own ordering).
    // Both kinds, because both are state a snapshot has to find consistent:
    // a GDN layer's own state is updated in place by the enqueued work, but
    // its counter is what lets `ignis_seq_at_chunk_boundary` say so (P4-06,
    // GitHub #124).
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      if (model->layers[layer].kind == IGNIS_LAYER_GQA) {
        seq->gqa_positions[ignis_gqa_relative_layer(layer)] +=
            static_cast<std::uint32_t>(num_tokens);
      } else {
        seq->gdn_positions[ignis_gdn_relative_layer(layer)] +=
            static_cast<std::uint32_t>(num_tokens);
      }
    }
    if (pending_appended) {
      seq->dflash2_position += 1;
      seq->dflash2_pending = false;
      seq->dflash2_pending_features.clear();
    }
    if (tap_count > 0) {
      seq->dflash2_position = chunk_start + num_tokens;
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
                                    LinearPolicyMode mode, const SpanMultimodal &multimodal,
                                    float *permitted_prob) {
  const uint64_t chunk_width = model->prefill_chunk_tokens;
  ChunkProfiler::instance().begin_span(num_tokens, model->prefill_chunk_tokens);
  const uint64_t tap_from = dflash2_tap_from(seq->position, num_tokens);
  uint64_t offset = 0;
  while (offset < num_tokens) {
    const uint64_t chunk_len = std::min<uint64_t>(chunk_width, num_tokens - offset);
    const bool is_last_chunk = (offset + chunk_len == num_tokens);
    int32_t successor = -1;
    float *slot_logits = is_last_chunk ? out_logits : nullptr;
    if (run_program_chunk(model, pool, seq, token_ids + offset, chunk_len, offset, tap_from,
                          is_last_chunk, sampling, &successor, slot_logits, mode, multimodal,
                          is_last_chunk ? permitted_prob : nullptr) != 0) {
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
  if (!dflash2_matches_load(model, pool, "ignis_program_prefill")) {
    return -1;
  }
  if (seq->position != start_position) {
    set_error("ignis_program_prefill: start_position does not match the sequence frontier");
    return -1;
  }
  if (num_tokens > ignis_seq_token_capacity(*seq) - seq->position) {
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
  SpanMultimodal multimodal;
  if (options != nullptr) {
    multimodal.positions = options->mrope_positions;
    multimodal.span_tokens = num_tokens;
    multimodal.rope_delta = options->rope_delta;
    multimodal.media = options->media;
    multimodal.scatter = options->media_scatter_indices;
    multimodal.count = options->media_column_count;
    multimodal.first_column = options->media_first_column;
  }
  if (!validate_span_multimodal(model, route, multimodal)) {
    return -1;
  }

  const auto began = std::chrono::steady_clock::now();
  // GitHub #242: where the span's own draw reports its restricted
  // probability, when the caller asked and declared a set.
  float *permitted_prob = (options != nullptr) ? options->out_permitted_prob : nullptr;
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
                            mode, (i + 1 == num_tokens) ? permitted_prob : nullptr) != 0) {
        rc = -1;
        break;
      }
      seq->pending_token = successor;
      ++seq->position;
    }
  } else {
    rc = run_program_prefill_chunked(model, pool, seq, token_ids, num_tokens, *sampling,
                                     out_logits, mode, multimodal, permitted_prob);
  }
  if (rc != 0) {
    return rc;
  }
  if (multimodal.positions != nullptr) {
    seq->rope_delta = multimodal.rope_delta;
  }
  // P5-05 (GitHub #155): the rewrite checkpoint is the window as the latest
  // prefill span leaves it. The reference saves it during prefill, at its chat
  // template's rewrite boundary, and restores it when a later turn rewrites
  // the text past that boundary; ignis has no such reuse path, so nothing
  // reads it yet. It is kept beside the window it was taken from, so a
  // snapshot, restore or clone never carries one that belongs to other text.
  if (pool->has_dflash2()) {
    try {
      pool->dflash2_checkpoint->copy_lane_from(*pool->dflash2_window, seq->slot, model->stream);
      const cudaError_t err = cudaStreamSynchronize(model->stream);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(err));
      }
    } catch (const std::exception &e) {
      set_error(std::string("ignis_program_prefill: rewrite checkpoint copy failed: ") + e.what());
      return -1;
    }
  }
  model->last_step_micros = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - began).count());
  return 0;
}

// P3-03 (GitHub #99) / GitHub #111: one call over every decode-ready
// lane, each sampled with its own `sampling[i]`. The round is one `B`-wide
// traversal of the model (`ignis_decode_graph_run_batch`,
// kernel/src/decode_graph.cu) leaving `[vocab, batch_size]` logits in the
// shared staging buffer, then one `ninfer::ops::sample` call drawing every
// lane's successor together -- so eight lanes stream the weights once, not
// eight times (requirement 17). #99's per-lane forward loop, which shared
// only the sampling, is gone: it made the round's cost scale with the batch
// (#111 measured B=4 at 4.81x its own B=1 round, against the reference's
// 1.07x). The round is atomic: no sequence's `pending_token`/`position`
// advances unless the traversal and the batched sample both succeed, so a
// mid-round failure never leaves one lane's state ahead of another's.
namespace {

// P5-04 (GitHub #153): advances every per-layer frontier of `seq` by
// `tokens` once a round's device work is confirmed complete -- the same
// bookkeeping the decode round does at one token (see the comment there),
// at a committed run's length.
void advance_frontiers(const ignis_model *model, ignis_seq *seq, std::uint32_t tokens) {
  seq->position += tokens;
  for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
    if (model->layers[layer].kind == IGNIS_LAYER_GQA) {
      seq->gqa_positions[ignis_gqa_relative_layer(layer)] += tokens;
    } else {
      seq->gdn_positions[ignis_gdn_relative_layer(layer)] += tokens;
    }
  }
}

// P5-04 (GitHub #153): one verify round over `batch_size` lanes at the
// load's window `k` (spec 05, "The verify round"). The device-side pass is
// `ignis_verify_graph_run_batch` (kernel/src/decode_graph.cu), replayed from
// the width's verify graph when one is ready and enqueued directly
// otherwise; this function stages its inputs, reads its outputs back, and
// does the host-side half of the commit -- the cut at the first stop id,
// the frontier advance, the ReplaySSM fold and the accepted-hidden
// selection.
//
// Per lane i, with `p` its frontier and `a` the anchor (its pending token):
//   extent   = min(k, draft_counts[i], remaining_tokens - 1, capacity - p - 1)
//   columns  = [a, d_1 .. d_extent, a, ...] at positions p, p+1, .., p+extent
//   accept   -> `n` accepted drafts (n <= extent), licensed = [d_1..d_n, t*]
//               where t* is the correction/bonus token
//   run      = [a, d_1 .. d_n], cut at the first stop id inclusive -> c tokens
//   commit   = the first c columns: KV frontier p -> p + c (the columns'
//              K/V were appended in place at p..p+c-1 and the rest are
//              overwritten by the next round), fold records[0..c) into the
//              GDN slot and conv taps, pending <- licensed[c-1] (the
//              successor of the run's last token: a draft the target agreed
//              with, or t*)
// so a lane whose run was cut stands exactly where a per-token decode of the
// same text would, and one whose run was not stands with t* pending -- what
// today's round leaves after one token, at every token of the run. The round
// is atomic: no lane's state moves unless every step succeeded.
int32_t run_verify_round(ignis_model *model, ignis_seq_pool *pool,
                         ignis_seq *const *sequences, uint64_t batch_size,
                         const ignis_sampling_params *sampling, const ignis_decode_options &options,
                         int32_t *out_token_ids) {
  IgnisVerifyRound &verify = *model->verify;
  const std::uint32_t k = verify.window;
  const std::uint32_t lane_columns = k + 1;
  const auto batch = static_cast<std::int32_t>(batch_size);
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  // P5-05 (GitHub #155): on a load with the drafter the leaf proposes every
  // lane's drafts itself, inside the verify pass, and appends the committed
  // columns' feature taps to each lane's window after the cut.
  const bool drafter = verify.drafter_scratch != nullptr;
  if (drafter && (options.drafts != nullptr || options.draft_counts != nullptr)) {
    set_error("ignis_program_decode: this model was loaded with the DFlash2 drafter, which "
              "proposes every lane's drafts; drafts and draft_counts must be NULL");
    return -1;
  }

  std::vector<std::int32_t> anchors(batch_size, 0);
  std::vector<std::int32_t> drafts(static_cast<std::size_t>(k) * batch_size, 0);
  std::vector<std::int32_t> base_positions(batch_size, 0);
  std::vector<std::int32_t> extents(batch_size, 0);
  std::vector<std::int32_t> valid_columns(batch_size, 1);
  std::vector<std::int32_t> slots(batch_size, 0);
  // GitHub #195: on a vision load, the columns' rotation positions -- the
  // same `base + min(j, extent)` the traversal derives, plus the lane's own
  // `rope_delta`. Empty (and unstaged) on every other load.
  std::vector<std::int32_t> rope_positions(
      verify.rope_positions == nullptr ? 0 : static_cast<std::size_t>(lane_columns) * batch_size, 0);
  std::vector<ninfer::ops::SamplingConfig> configs(batch_size);
  try {
    for (uint64_t i = 0; i < batch_size; ++i) {
      ignis_seq *seq = sequences[i];
      if (seq == nullptr || seq->pending_token < 0) {
        set_error("ignis_program_decode: sequence is null or was not prefilled");
        return -1;
      }
      const uint64_t capacity = ignis_seq_token_capacity(*seq);
      if (seq->position >= capacity) {
        set_error("ignis_program_decode: sequence reached its KV capacity");
        return -1;
      }
      // The lane's extent: never more drafts than proposed, than its budget
      // leaves after the anchor, or than its context leaves after the
      // anchor. 0 is the fallback step inside this same round.
      uint64_t extent = k;
      if (!drafter && options.drafts == nullptr) {
        extent = 0;
      } else if (!drafter && options.draft_counts != nullptr) {
        extent = std::min<uint64_t>(extent, options.draft_counts[i]);
      }
      if (sampling[i].remaining_tokens != 0) {
        extent = std::min<uint64_t>(extent, sampling[i].remaining_tokens - 1);
      }
      extent = std::min<uint64_t>(extent, capacity - seq->position - 1);
      if (sampling[i].stop_id_count != 0 && sampling[i].stop_ids == nullptr) {
        set_error("ignis_program_decode: stop_ids is null with a nonzero stop_id_count at index " +
                  std::to_string(i));
        return -1;
      }
      anchors[i] = seq->pending_token;
      base_positions[i] = static_cast<std::int32_t>(seq->position);
      extents[i] = static_cast<std::int32_t>(extent);
      valid_columns[i] = static_cast<std::int32_t>(extent + 1);
      slots[i] = seq->slot;
      configs[i] = to_sampling_config(sampling[i], pool->token_counts_for(seq->slot));
      for (uint64_t j = 0; j < extent && !drafter; ++j) {
        drafts[i * k + j] = options.drafts[i * k + j];
      }
      // The lane's rotation positions: the traversal's own column rule with
      // the sequence's delta added. A text sequence's delta is 0, so the two
      // matrices are equal and nothing about a text round changes.
      if (!rope_positions.empty()) {
        for (std::uint32_t j = 0; j < lane_columns; ++j) {
          rope_positions[i * lane_columns + j] =
              base_positions[i] + std::min<std::int32_t>(static_cast<std::int32_t>(j), extents[i]) +
              seq->rope_delta;
        }
      }
    }

    // GitHub #157: each lane's carried anchor taps, into its window before
    // the drafter reads it inside the pass. A failure below leaves the lane
    // unmoved, and the append it enqueued rewrites the same slot with the
    // same bytes when the lane runs again.
    std::vector<std::vector<std::int32_t>> pending_scalars(batch_size);
    std::vector<bool> pending_appended(batch_size, false);
    if (drafter) {
      ninfer::DeviceArena::Scope pending_scope = model->scratch->scope();
      for (uint64_t i = 0; i < batch_size; ++i) {
        pending_appended[i] = append_dflash2_pending(model, pool, *sequences[i], pending_scalars[i]);
      }
    }

    const auto stage = [&](ninfer::DeviceBuffer &buffer, const void *host, std::size_t bytes,
                           const char *what) -> bool {
      const cudaError_t err =
          cudaMemcpyAsync(buffer.p, host, bytes, cudaMemcpyHostToDevice, model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaMemcpyAsync(") + what + ") failed: " +
                  cudaGetErrorString(err));
        return false;
      }
      return true;
    };
    const std::size_t lane_bytes = batch_size * sizeof(std::int32_t);
    if (!stage(*verify.anchors, anchors.data(), lane_bytes, "verify anchors") ||
        // The drafter writes this buffer on the device, inside the pass.
        (!drafter &&
         !stage(*verify.drafts, drafts.data(), drafts.size() * sizeof(std::int32_t), "verify drafts")) ||
        !stage(*verify.base_positions, base_positions.data(), lane_bytes, "verify base positions") ||
        !stage(*verify.extents, extents.data(), lane_bytes, "verify extents") ||
        !stage(*verify.valid_columns, valid_columns.data(), lane_bytes, "verify valid columns") ||
        // The accept RNG's position base is the lane's frontier, the same
        // logical position today's sampler keys its draw by.
        !stage(*verify.lengths, base_positions.data(), lane_bytes, "verify lengths") ||
        // GitHub #195: a vision load's rotation positions, at the address its
        // verify graphs read.
        (!rope_positions.empty() &&
         !stage(*verify.rope_positions, rope_positions.data(),
                rope_positions.size() * sizeof(std::int32_t), "verify rope positions")) ||
        !stage(*model->decode_graph_slots, slots.data(), lane_bytes, "verify slots") ||
        !stage(*model->sampling_decode_configs, configs.data(),
               batch_size * sizeof(ninfer::ops::SamplingConfig), "verify sampling configs")) {
      return -1;
    }

    const bool use_graph = verify.graph_ready[batch_size - 1];
    if (use_graph) {
      const cudaError_t err = cudaGraphLaunch(verify.graph_exec[batch_size - 1], model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaGraphLaunch(verify) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    } else if (ignis_verify_graph_run_batch(model, pool, static_cast<uint32_t>(batch_size),
                                            LinearPolicyMode::kEngineDefault) != 0) {
      set_error(std::string("ignis_program_decode: ") + ignis_decode_graph_last_error());
      return -1;
    }

    std::vector<std::int32_t> licensed(static_cast<std::size_t>(lane_columns) * batch_size, 0);
    std::vector<std::int32_t> accepted(batch_size, 0);
    cudaError_t err = cudaMemcpyAsync(licensed.data(), verify.licensed_tokens->p,
                                      licensed.size() * sizeof(std::int32_t),
                                      cudaMemcpyDeviceToHost, model->stream);
    if (err == cudaSuccess) {
      err = cudaMemcpyAsync(accepted.data(), verify.accepted->p, lane_bytes,
                            cudaMemcpyDeviceToHost, model->stream);
    }
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_decode: cudaMemcpyAsync(accept results) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    err = cudaStreamSynchronize(model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_decode: cudaStreamSynchronize(verify) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    // The host half of the commit: the run, its cut, the fold rows.
    std::vector<std::int32_t> committed(batch_size, 0);
    std::vector<std::int32_t> next_pending(batch_size, -1);
    std::vector<ninfer::ops::GdnReplayFoldRow> fold_rows(batch_size);
    std::vector<std::int32_t> selectors(batch_size, 0);
    for (uint64_t i = 0; i < batch_size; ++i) {
      const std::int32_t n = accepted[i];
      if (n < 0 || n > extents[i]) {
        set_error("ignis_program_decode: the accept kernel reported " + std::to_string(n) +
                  " accepted drafts for an extent of " + std::to_string(extents[i]));
        return -1;
      }
      const std::int32_t *lane_licensed = licensed.data() + i * lane_columns;
      // run[j]: j == 0 the anchor, else licensed[j-1] (= the j-th draft,
      // accepted); run length n + 1 before the cut.
      const auto run_at = [&](std::int32_t j) {
        return j == 0 ? anchors[i] : lane_licensed[j - 1];
      };
      std::int32_t c = n + 1;
      for (std::int32_t j = 0; j < n + 1; ++j) {
        bool stop = false;
        for (uint32_t s = 0; s < sampling[i].stop_id_count && !stop; ++s) {
          stop = sampling[i].stop_ids[s] == run_at(j);
        }
        if (stop) {
          c = j + 1;
          break;
        }
      }
      committed[i] = c;
      // The successor of the run's last token: the draft after it when the
      // target agreed with that draft, else the correction/bonus token.
      next_pending[i] = lane_licensed[c - 1];
      for (std::int32_t j = 0; j < c; ++j) {
        out_token_ids[i * lane_columns + j] = run_at(j);
      }
      fold_rows[i] = ninfer::ops::GdnReplayFoldRow{.linear_state_slot = slots[i],
                                                   .commit_columns = c};
      selectors[i] = c - 1;
    }

    // The fold: every GDN layer's slot and conv taps rebuilt from the
    // committed prefix of this round's records, in one vendored call over
    // every lane. A zero-length commit is that op's own strict no-op; this
    // round never asks for one (c >= 1), the anchor is always committed.
    ninfer::ops::gdn_replay_fold(verify.records, pool->gdn_pool.all_layers_view(),
                                 std::span<const ninfer::ops::GdnReplayFoldRow>(fold_rows.data(),
                                                                                batch_size),
                                 model->stream);

    // The accepted hidden state per lane (the run's last column's final
    // residual), kept for the drafter's continuation input (P5-05).
    if (!stage(*verify.selectors, selectors.data(), lane_bytes, "verify selectors")) {
      return -1;
    }
    const ninfer::Tensor hidden_columns(verify.hidden->p, ninfer::DType::BF16,
                                        {hidden, static_cast<std::int32_t>(lane_columns), batch, 1});
    const ninfer::Tensor selector_tensor(verify.selectors->p, ninfer::DType::I32, {batch, 1, 1, 1});
    ninfer::Tensor selected(verify.selected_hidden->p, ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::ops::speculative_select_accepted_hidden(hidden_columns, selector_tensor, selected,
                                                    model->stream);

    // P5-05 (GitHub #155): the committed columns' feature taps into each
    // lane's window -- its whole committed run, or nothing for a lane at
    // extent 0, whose round read its window and leaves it as it was.
    std::vector<std::int32_t> append_counts(batch_size, 0);
    // GitHub #157: what that lane carries instead -- its anchor column's
    // taps, read back for the sequence's handle.
    std::vector<std::vector<std::uint8_t>> carried_taps(batch_size);
    if (drafter) {
      for (uint64_t i = 0; i < batch_size; ++i) {
        append_counts[i] = extents[i] == 0 ? 0 : committed[i];
      }
      if (!stage(*verify.append_counts, append_counts.data(), lane_bytes, "drafter append counts")) {
        return -1;
      }
      ignis_dflash2_append_round(model, pool, static_cast<std::uint32_t>(batch_size));
      const std::size_t column_bytes =
          kDflash2TapLayers.size() * static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
      for (uint64_t i = 0; i < batch_size; ++i) {
        if (extents[i] != 0) {
          continue;
        }
        carried_taps[i].resize(column_bytes);
        const cudaError_t copied = cudaMemcpyAsync(
            carried_taps[i].data(),
            static_cast<const std::uint8_t *>(verify.features->p) + i * lane_columns * column_bytes,
            column_bytes, cudaMemcpyDeviceToHost, model->stream);
        if (copied != cudaSuccess) {
          set_error(std::string("ignis_program_decode: cudaMemcpyAsync(anchor taps) failed: ") +
                    cudaGetErrorString(copied));
          return -1;
        }
      }
    }

    err = cudaStreamSynchronize(model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_decode: cudaStreamSynchronize(fold) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    // The penalty rows: at temperature > 0 the accept kernel counted every
    // licensed token -- the accepted drafts and the correction/bonus token --
    // in the lane's occurrence counts. A cut run keeps only
    // `licensed[0..c)` (its drafts plus the new pending token); the tokens
    // past it were never emitted, so their counts come back off, or the row
    // a snapshot or a prefix clone carries would run ahead of the text.
    // Greedy lanes count nothing (the kernel's own contract). Done after the
    // fold is confirmed, as the last device step before the commit, and in
    // one read, one adjustment and one write for every such count.
    std::vector<std::int32_t *> rollback;
    for (uint64_t i = 0; i < batch_size; ++i) {
      if (!(configs[i].temperature > 0.0f) || configs[i].token_counts == nullptr) {
        continue;
      }
      const std::int32_t produced = accepted[i] + 1;
      const std::int32_t *lane_licensed = licensed.data() + i * lane_columns;
      for (std::int32_t j = committed[i]; j < produced; ++j) {
        rollback.push_back(configs[i].token_counts + lane_licensed[j]);
      }
    }
    if (!rollback.empty()) {
      std::vector<std::int32_t> counts(rollback.size(), 0);
      for (std::size_t r = 0; r < rollback.size() && err == cudaSuccess; ++r) {
        err = cudaMemcpyAsync(&counts[r], rollback[r], sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, model->stream);
      }
      if (err == cudaSuccess) {
        err = cudaStreamSynchronize(model->stream);
      }
      // A token licensed twice in one run has two entries reading the same
      // count; each entry takes one occurrence off the value read.
      for (std::size_t r = 0; r < rollback.size(); ++r) {
        std::int32_t taken = 0;
        for (std::size_t q = 0; q <= r; ++q) {
          taken += rollback[q] == rollback[r] ? 1 : 0;
        }
        counts[r] = counts[r] > taken ? counts[r] - taken : 0;
      }
      for (std::size_t r = 0; r < rollback.size() && err == cudaSuccess; ++r) {
        err = cudaMemcpyAsync(rollback[r], &counts[r], sizeof(std::int32_t),
                              cudaMemcpyHostToDevice, model->stream);
      }
      if (err == cudaSuccess) {
        err = cudaStreamSynchronize(model->stream);
      }
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: penalty count rollback failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }

    // Only now, with the fold confirmed on the device, does any lane's
    // state move (the decode round's own ordering).
    for (uint64_t i = 0; i < batch_size; ++i) {
      sequences[i]->pending_token = next_pending[i];
      advance_frontiers(model, sequences[i], static_cast<std::uint32_t>(committed[i]));
      options.out_committed_counts[i] = committed[i];
      if (pending_appended[i]) {
        sequences[i]->dflash2_position += 1;
      }
      if (append_counts[i] > 0) {
        sequences[i]->dflash2_position = sequences[i]->position;
      }
      if (drafter) {
        // GitHub #157: a lane at extent 0 carries its anchor's taps, but
        // only while its window stands exactly at that anchor -- the
        // position the carried column is appended at.
        ignis_seq &seq = *sequences[i];
        seq.dflash2_pending = extents[i] == 0 &&
                              seq.dflash2_position == static_cast<std::uint64_t>(base_positions[i]);
        seq.dflash2_pending_features.swap(carried_taps[i]);
        if (!seq.dflash2_pending) {
          seq.dflash2_pending_features.clear();
        }
      }
      if (options.out_extents != nullptr) {
        options.out_extents[i] = static_cast<std::uint32_t>(extents[i]);
      }
    }
    model->last_step_kernel_count = model->layers.size();
    model->last_step_graph_launches = use_graph ? 1 : 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_program_decode: ") + e.what());
    return -1;
  }
  return 0;
}

} // namespace

extern "C" int32_t ignis_program_decode(struct ignis_model *model,
                                          struct ignis_seq_pool *pool,
                                          struct ignis_seq *const *sequences,
                                          uint64_t batch_size,
                                          const struct ignis_sampling_params *sampling,
                                          int32_t *out_token_ids,
                                          const struct ignis_decode_options *options) {
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
  if (!dflash2_matches_load(model, pool, "ignis_program_decode")) {
    return -1;
  }
  for (uint64_t i = 0; i < batch_size; ++i) {
    if (!sampling_size_ok(sampling[i])) {
      set_error("ignis_program_decode: unrecognized ignis_sampling_params size " +
                std::to_string(sampling[i].size) + " at index " + std::to_string(i));
      return -1;
    }
  }
  // P5-04 (GitHub #153, ADR 0016): NULL means today's round; a window must
  // be the load's own, never padded to it.
  if (options != nullptr) {
    if (options->size != sizeof(struct ignis_decode_options)) {
      set_error("ignis_program_decode: unrecognized ignis_decode_options size " +
                std::to_string(options->size));
      return -1;
    }
    if (options->speculative_window != 0) {
      const uint32_t loaded = model->verify == nullptr ? 0 : model->verify->window;
      if (options->speculative_window != loaded) {
        set_error("ignis_program_decode: speculative_window " +
                  std::to_string(options->speculative_window) + " is not the window this model was loaded with (" +
                  std::to_string(loaded) + ")");
        return -1;
      }
      if (options->out_committed_counts == nullptr) {
        set_error("ignis_program_decode: out_committed_counts is null for a verify round");
        return -1;
      }
      // GitHub #242: a drafted column is proposed by a second model and
      // accepted by a kernel that knows nothing of a permitted set, so a
      // verify round cannot honour one. Refused rather than applied to the
      // anchor alone, which would constrain one token of a run and leave
      // the drafts free.
      for (uint64_t i = 0; i < batch_size; ++i) {
        if (!unconstrained(sampling[i])) {
          set_error("ignis_program_decode: a permitted token set cannot ride a verify round (index " +
                    std::to_string(i) + ")");
          return -1;
        }
      }
      const auto began = std::chrono::steady_clock::now();
      const int32_t rc =
          run_verify_round(model, pool, sequences, batch_size, sampling, *options, out_token_ids);
      if (rc != 0) {
        return rc;
      }
      model->last_step_micros = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - began).count());
      return 0;
    }
  }

  const auto began = std::chrono::steady_clock::now();
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  const auto batch = static_cast<std::int32_t>(batch_size);
  std::vector<int32_t> emitted(batch_size, -1);
  std::vector<std::int32_t> positions(batch_size, 0);
  std::vector<std::int32_t> rope_positions(batch_size, 0);
  std::vector<ninfer::ops::SamplingConfig> configs(batch_size);
  // GitHub #242: this round's permitted sets, flattened into the fixed
  // per-lane row the staging buffer holds. `-1` is the filler for a lane's
  // unused entries -- never a vocabulary id, and never read, since the
  // device reads only the first `permitted_counts[lane]` of them.
  std::vector<std::int32_t> permitted(batch_size * IGNIS_MAX_PERMITTED_TOKENS, -1);
  std::vector<std::int32_t> permitted_counts(batch_size, 0);
  bool any_constrained = false;

  try {
    for (uint64_t i = 0; i < batch_size; ++i) {
      ignis_seq *seq = sequences[i];
      if (seq == nullptr || seq->pending_token < 0) {
        set_error("ignis_program_decode: sequence is null or was not prefilled");
        return -1;
      }
      if (seq->position >= ignis_seq_token_capacity(*seq)) {
        set_error("ignis_program_decode: sequence reached its KV capacity");
        return -1;
      }
      emitted[i] = seq->pending_token;
      positions[i] = static_cast<std::int32_t>(seq->position);
      rope_positions[i] = positions[i] + seq->rope_delta;
      configs[i] = to_sampling_config(sampling[i], pool->token_counts_for(seq->slot));

      const ignis_sampling_params &lane = sampling[i];
      if (lane.permitted_count == 0) {
        continue;
      }
      if (lane.permitted_count > IGNIS_MAX_PERMITTED_TOKENS) {
        set_error("ignis_program_decode: permitted_count " +
                  std::to_string(lane.permitted_count) + " at index " + std::to_string(i) +
                  " exceeds IGNIS_MAX_PERMITTED_TOKENS (" +
                  std::to_string(IGNIS_MAX_PERMITTED_TOKENS) + ")");
        return -1;
      }
      if (lane.permitted_ids == nullptr) {
        set_error("ignis_program_decode: permitted_ids is null with a nonzero count at index " +
                  std::to_string(i));
        return -1;
      }
      for (uint32_t k = 0; k < lane.permitted_count; ++k) {
        const int32_t id = lane.permitted_ids[k];
        if (id < 0 || id >= vocab) {
          set_error("ignis_program_decode: permitted id " + std::to_string(id) + " at index " +
                    std::to_string(i) + " is outside the vocabulary");
          return -1;
        }
        permitted[i * IGNIS_MAX_PERMITTED_TOKENS + k] = id;
      }
      permitted_counts[i] = static_cast<std::int32_t>(lane.permitted_count);
      any_constrained = true;
    }

    // P3-05 (GitHub #102, ADR 0019): a decode graph is captured per exact
    // batch width, never padded -- a round at this exact width replays it
    // when one is ready, otherwise falls back to the eager per-lane loop
    // below unchanged. Both paths share the same sampling staging buffers
    // and the same round semantics (atomic: no sequence's
    // pending_token/position advances unless the whole round succeeds).
    //
    // GitHub #242: a round carrying a constrained lane takes the eager path
    // whatever the graph offers. The mask has to run between the traversal
    // and the sampler, and adding a node to a graph captured at load -- for
    // a path that runs six rounds per number and never in a throughput
    // workload -- would put new failure modes on every decode in the
    // process to save a graph launch on almost none. The eager path is the
    // same op sequence; what it costs is the submission, measured at 2.2%
    // of a round (`docs/findings/2026-09-18-decode-round-anatomy.md`).
    const bool use_graph =
        batch_size >= 1 && batch_size <= IGNIS_DECODE_MAX_BATCH &&
        model->decode_graph_ready[batch_size - 1] && !any_constrained;

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
    // GitHub #242: staged only when some lane is constrained, so an
    // ordinary round issues exactly the copies it issued before.
    if (any_constrained) {
      err = cudaMemcpyAsync(model->sampling_decode_permitted->p, permitted.data(),
                            permitted.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                            model->stream);
      if (err == cudaSuccess) {
        err = cudaMemcpyAsync(model->sampling_decode_permitted_counts->p, permitted_counts.data(),
                              batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                              model->stream);
      }
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaMemcpyAsync(permitted sets) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }
    // GitHub #178: a vision load's rounds rotate at `position + rope_delta`,
    // staged at the address its graphs read.
    if (model->decode_rope_positions != nullptr) {
      err = cudaMemcpyAsync(model->decode_rope_positions->p, rope_positions.data(),
                            batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                            model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaMemcpyAsync(rope positions) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }

    // GitHub #111: the round's per-lane token ids and physical pool
    // slots are staged for *both* paths -- the B-wide traversal reads them
    // from device memory whether it is being replayed from a captured graph
    // or enqueued directly, so there is no host-indexed variant left.
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

    if (use_graph) {
      err = cudaGraphLaunch(model->decode_graph_exec[batch_size - 1], model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaGraphLaunch failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    } else {
      // The same op sequence the graph captured, enqueued directly. P2-03
      // (GitHub #85): decode takes the engine's own compute-policy mode
      // (ADR 0016: the flat decode ABI has no options struct, so the
      // `A16_ONLY` override is reachable only through the prefill entry
      // point) -- every NVFP4 projection in the decode round runs under
      // AllowA4, the reference's text-model policy.
      if (ignis_decode_graph_run_batch(model, pool, static_cast<uint32_t>(batch_size),
                                       LinearPolicyMode::kEngineDefault) != 0) {
        set_error(std::string("ignis_program_decode: ") + ignis_decode_graph_last_error());
        return -1;
      }
      // GitHub #242: the constraint is applied to the logits the sampler is
      // about to read, so the vendored op is unchanged (ADR 0010) and every
      // parameter above still means what it meant.
      if (any_constrained &&
          ignis_permit_mask(model->sampling_decode_logits->p, vocab,
                            static_cast<uint32_t>(batch_size),
                            static_cast<const int32_t *>(model->sampling_decode_permitted->p),
                            static_cast<const int32_t *>(
                                model->sampling_decode_permitted_counts->p),
                            IGNIS_MAX_PERMITTED_TOKENS, model->stream) != 0) {
        set_error("ignis_program_decode: permitted-set mask launch failed");
        return -1;
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
      // GitHub #242: and the committed token's share of the set it was drawn
      // from, one float per lane, while the logits are still here.
      if (any_constrained && options != nullptr && options->out_permitted_probs != nullptr &&
          ignis_permit_probability(
              model->sampling_decode_logits->p, vocab, static_cast<uint32_t>(batch_size),
              static_cast<const int32_t *>(model->sampling_decode_permitted->p),
              static_cast<const int32_t *>(model->sampling_decode_permitted_counts->p),
              IGNIS_MAX_PERMITTED_TOKENS,
              static_cast<const int32_t *>(model->sampling_decode_out->p),
              static_cast<float *>(model->sampling_decode_permitted_probs->p),
              model->stream) != 0) {
        set_error("ignis_program_decode: permitted-set probability launch failed");
        return -1;
      }
    }

    std::vector<int32_t> successors(batch_size, -1);
    err = cudaMemcpyAsync(successors.data(), model->sampling_decode_out->p,
                          batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_decode: cudaMemcpyAsync(sampled tokens) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    // GitHub #242: a caller that asked for the probabilities always gets an
    // answer for every lane -- zeros for a round nothing constrained, so the
    // field never carries the previous round's numbers.
    if (options != nullptr && options->out_permitted_probs != nullptr) {
      if (any_constrained) {
        err = cudaMemcpyAsync(options->out_permitted_probs,
                              model->sampling_decode_permitted_probs->p,
                              batch_size * sizeof(float), cudaMemcpyDeviceToHost, model->stream);
        if (err != cudaSuccess) {
          set_error(
              std::string("ignis_program_decode: cudaMemcpyAsync(permitted probabilities) failed: ") +
              cudaGetErrorString(err));
          return -1;
        }
      } else {
        for (uint64_t i = 0; i < batch_size; ++i) {
          options->out_permitted_probs[i] = 0.0f;
        }
      }
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
      // Only once the synchronize above confirms the round's device work
      // completed (mirrors `ignis_gqa_layer_step`'s own ordering, and the
      // chunked prefill's). GitHub #111: both decode paths now read
      // their RoPE/attention positions from `sampling_decode_positions`
      // (staged from `seq->position`) rather than from this counter, but the
      // counter still feeds the per-token prefill route and the layer
      // bodies' KV-capacity check, so a decode round must keep it truthful.
      // Before #111 the graph path left it behind by one per round, so a
      // round that fell back to eager after a replay read a stale position.
      // The GDN counters advance here for the same reason the chunk loop
      // advances them (P4-06, GitHub #124): nothing reads them on the
      // forward pass, but a snapshot's chunk-boundary check does.
      for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
        if (model->layers[layer].kind == IGNIS_LAYER_GQA) {
          ++sequences[i]->gqa_positions[ignis_gqa_relative_layer(layer)];
        } else {
          ++sequences[i]->gdn_positions[ignis_gdn_relative_layer(layer)];
        }
      }
    }
    // GitHub #111: one traversal of the model per round, whatever
    // the batch width and whichever path ran it -- the dispatch count is the
    // layer count, not the layer count times the width. This is the leaf
    // instrumentation the issue's acceptance asks for: at B>1 it stays equal
    // to the B=1 round's, where before it was B times it.
    model->last_step_kernel_count = model->layers.size();
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
      model->decode_graph_slots->bytes;
  if (model->decode_rope_positions != nullptr) {
    out_stats->vram_bytes += model->decode_rope_positions->bytes;
  }
  // P5-02 (GitHub #150) / P5-03 (GitHub #152): the drafter's window and its
  // checkpoint, one lane per slot of a pool built with the drafter -- its
  // weights are already in `model->vram_bytes`.
  if (pool->has_dflash2()) {
    out_stats->vram_bytes += pool->dflash2_arena->capacity();
  }
  // P5-04 (GitHub #153): the verify substrate's own buffers, records and
  // accept scratch, present only under a draft window (its traversal scratch
  // is `decode_graph_scratch` above, sized for it at load).
  if (model->verify != nullptr) {
    out_stats->vram_bytes += model->verify->device_bytes();
  }
  // GitHub #177: the vision output transient (its weights are already in
  // `model->vram_bytes`, and its encoder workspace in `scratch`, GitHub #212).
  if (model->vision_output != nullptr) {
    out_stats->vram_bytes += model->vision_output->bytes;
  }
  out_stats->last_step_micros = model->last_step_micros;
  out_stats->kernel_count = model->last_step_kernel_count;
  out_stats->graph_launches = model->last_step_graph_launches;
  uint32_t ready_mask = 0;
  uint32_t verify_ready_mask = 0;
  for (uint32_t width = 1; width <= IGNIS_DECODE_MAX_BATCH; ++width) {
    if (model->decode_graph_ready[width - 1]) {
      ready_mask |= (1u << (width - 1));
    }
    if (model->verify != nullptr && model->verify->graph_ready[width - 1]) {
      verify_ready_mask |= (1u << (width - 1));
    }
  }
  out_stats->decode_graph_ready_mask = ready_mask;
  out_stats->verify_graph_ready_mask = verify_ready_mask;
  return 0;
}
