// ignis kernel leaf - P3-05 (GitHub #102, ADR 0019): captures one CUDA
// graph per exact decode batch width 1..IGNIS_DECODE_MAX_BATCH. The program
// layer is ours; every dispatched op is the ADR 0010 vendored reference
// implementation, called through the graph-safe layer bodies
// (kernel/src/gqa_layer.cu / gdn_layer.cu's `*_run_body_graph`).
//
// Capture only: replaying a captured graph (`cudaGraphLaunch`) is
// `ignis_program_decode`'s own concern (kernel/src/step.cu) -- it refreshes
// the round's staging buffers, launches, and reads results back, falling
// back to the eager per-lane loop for a width with no ready graph.

#include "ignis_step.h"

#include "ignis_gdn_layer.h"
#include "ignis_gqa_layer.h"
#include "layer_internal.h"
#include "model_internal.h"

#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

// One lane's forward pass for a captured decode graph: embedding (this
// round's token id read from `decode_graph_token_ids[lane]`) -> every
// decoder layer's graph-safe body -> final norm -> output head, writing
// this lane's BF16 logits into column `lane` of the shared
// `sampling_decode_logits` staging buffer (P3-03) -- the same buffer and
// column layout the eager loop (`run_program_token_forward_only`,
// kernel/src/step.cu) already uses, so one batched `ninfer::ops::sample`
// call below serves both paths identically.
int32_t run_decode_graph_lane(ignis_model *model, ignis_seq_pool *pool, uint32_t lane,
                              LinearPolicyMode mode) {
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  ninfer::DeviceArena::Scope scope = model->decode_graph_scratch->scope();
  try {
    const ninfer::Tensor ids(
        static_cast<std::uint8_t *>(model->decode_graph_token_ids->p) +
            static_cast<std::size_t>(lane) * sizeof(std::int32_t),
        ninfer::DType::I32, {1, 1, 1, 1});
    ninfer::Tensor left = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::Tensor right = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::embedding(ids, model->token_embedding, left, model->stream);

    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_run_body_graph(model, pool, layer, lane, left.data, right.data, mode)
          : ignis_gdn_layer_run_body_graph(model, pool, layer, lane, left.data, right.data, mode);
      if (rc != 0) {
        const char *detail = model->layers[layer].kind == IGNIS_LAYER_GQA
            ? ignis_gqa_layer_last_error()
            : ignis_gdn_layer_last_error();
        set_error("ignis_decode_graph: lane " + std::to_string(lane) + " layer " +
                  std::to_string(layer) + " failed: " + detail);
        return -1;
      }
      std::swap(left, right);
    }

    const ninfer::Tensor norm_weight(const_cast<void *>(model->final_norm.qdata), ninfer::DType::BF16,
                                     {hidden, 1, 1, 1});
    ninfer::Tensor normalized =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(left, norm_weight, model->rms_norm_eps, /*unit_offset=*/true, normalized,
                         model->stream);
    ninfer::Tensor logits = model->decode_graph_scratch->alloc(ninfer::DType::BF16, {vocab, 1, 1, 1});
    ninfer::ops::linear(normalized, model->output_head, logits, model->stream);

    void *column = static_cast<std::uint8_t *>(model->sampling_decode_logits->p) +
        static_cast<std::size_t>(lane) * static_cast<std::size_t>(vocab) * sizeof(std::uint16_t);
    const cudaError_t err = cudaMemcpyAsync(column, logits.data,
                                            static_cast<std::size_t>(vocab) * sizeof(std::uint16_t),
                                            cudaMemcpyDeviceToDevice, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_decode_graph: cudaMemcpyAsync(logits column) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    return 0;
  } catch (const std::exception &e) {
    set_error("ignis_decode_graph: lane " + std::to_string(lane) + ": " + e.what());
    return -1;
  }
}

// Records one width-W graph: W sequential lane forward passes (program
// order, capture-time-fixed `lane` indices 0..W-1) followed by one batched
// `ninfer::ops::sample` reading `model->sampling_decode_{configs,positions,
// logits,out}` at their stable addresses (P3-03) -- exactly the op sequence
// `ignis_program_decode`'s eager loop already runs, captured once. On any
// failure the partial graph is discarded and `*out_exec` stays null; the
// caller leaves this width on the eager fallback.
int32_t capture_decode_graph_width(ignis_model *model, ignis_seq_pool *pool, uint32_t width,
                                   cudaGraphExec_t *out_exec) {
  *out_exec = nullptr;
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  const auto batch = static_cast<std::int32_t>(width);

  cudaError_t err = cudaStreamBeginCapture(model->stream, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_decode_graph_capture: cudaStreamBeginCapture failed: ") +
              cudaGetErrorString(err));
    return -1;
  }

  bool ok = true;
  for (uint32_t lane = 0; ok && lane < width; ++lane) {
    ok = run_decode_graph_lane(model, pool, lane, LinearPolicyMode::kEngineDefault) == 0;
  }
  if (ok) {
    const ninfer::Tensor logits_tensor(model->sampling_decode_logits->p, ninfer::DType::BF16,
                                       {vocab, batch, 1, 1});
    ninfer::Tensor out_tensor(model->sampling_decode_out->p, ninfer::DType::I32, {batch, 1, 1, 1});
    const ninfer::Tensor positions_tensor(model->sampling_decode_positions->p, ninfer::DType::I32,
                                          {batch, 1, 1, 1});
    try {
      ninfer::DeviceArena::Scope workspace_scope = model->sampling_workspace->scope();
      ninfer::ops::sample(
          logits_tensor, out_tensor, vocab,
          static_cast<const ninfer::ops::SamplingConfig *>(model->sampling_decode_configs->p),
          positions_tensor, ninfer::ops::kSamplePurposeDecode, *model->sampling_workspace,
          model->stream);
    } catch (const std::exception &e) {
      set_error(std::string("ignis_decode_graph_capture: sample() failed: ") + e.what());
      ok = false;
    }
  }

  cudaGraph_t graph = nullptr;
  const cudaError_t end_err = cudaStreamEndCapture(model->stream, &graph);
  if (end_err != cudaSuccess) {
    set_error(std::string("ignis_decode_graph_capture: cudaStreamEndCapture failed: ") +
              cudaGetErrorString(end_err));
    return -1;
  }
  if (!ok) {
    cudaGraphDestroy(graph);
    return -1;
  }

  cudaGraphExec_t exec = nullptr;
  const cudaError_t inst_err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  if (inst_err != cudaSuccess) {
    set_error(std::string("ignis_decode_graph_capture: cudaGraphInstantiate failed: ") +
              cudaGetErrorString(inst_err));
    return -1;
  }
  *out_exec = exec;
  return 0;
}

} // namespace

extern "C" int32_t ignis_decode_graph_capture(struct ignis_model *model, struct ignis_seq_pool *pool,
                                              uint64_t *out_capture_micros,
                                              uint32_t *out_ready_mask) {
  if (model == nullptr || pool == nullptr) {
    set_error("ignis_decode_graph_capture: null argument");
    return -1;
  }
  const auto began = std::chrono::steady_clock::now();
  uint32_t ready_mask = 0;
  for (uint32_t width = 1; width <= IGNIS_DECODE_MAX_BATCH; ++width) {
    cudaGraphExec_t exec = nullptr;
    if (capture_decode_graph_width(model, pool, width, &exec) == 0) {
      model->decode_graph_exec[width - 1] = exec;
      model->decode_graph_ready[width - 1] = true;
      ready_mask |= (1u << (width - 1));
    } else {
      // g_last_error above already names the failing width; a capture
      // failure degrades performance (this width stays on the eager
      // fallback) and never refuses service (ADR 0019, GitHub #102's
      // acceptance) -- so this loop keeps going for every other width
      // rather than aborting.
      model->decode_graph_exec[width - 1] = nullptr;
      model->decode_graph_ready[width - 1] = false;
    }
  }
  if (out_capture_micros != nullptr) {
    *out_capture_micros = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - began)
            .count());
  }
  if (out_ready_mask != nullptr) {
    *out_ready_mask = ready_mask;
  }
  return 0;
}

extern "C" const char *ignis_decode_graph_last_error(void) {
  return g_last_error.c_str();
}
