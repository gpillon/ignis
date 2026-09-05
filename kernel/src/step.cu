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

#include "model_internal.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

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
