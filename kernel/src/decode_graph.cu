// ignis kernel leaf - P3-05 (GitHub #102, ADR 0019) and GitHub #111 (ADR
// 0020): the decode round's forward pass, and one CUDA graph capturing
// it per exact batch width 1..IGNIS_DECODE_MAX_BATCH. The program layer is
// ours; every dispatched op is the ADR 0010 vendored reference
// implementation, called through the graph-safe layer bodies
// (kernel/src/gqa_layer.cu / gdn_layer.cu's `*_run_body_graph`).
//
// The round is one batch-wide traversal of the model
// (`ignis_decode_graph_run_batch`), and this file owns both it and the
// capture of it. Replay (`cudaGraphLaunch`) is `ignis_program_decode`'s own
// concern (kernel/src/step.cu) -- it refreshes the round's staging buffers,
// launches, and reads results back; at a width with no ready graph it calls
// the same traversal directly instead.

#include "ignis_step.h"

#include "dflash2_drafter.h"
#include "ignis_gdn_layer.h"
#include "ignis_gqa_layer.h"
#include "layer_internal.h"
#include "model_internal.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

} // namespace

// GitHub #111: one decode round's complete forward pass as a single
// `width`-wide traversal of the model -- embedding (every lane's token id
// read from `decode_graph_token_ids`, a contiguous I32 [width]) -> every
// decoder layer's graph-safe body once, with batch-shaped activations ->
// final norm -> output head, writing the round's `[vocab, width]` BF16
// logits straight into the shared `sampling_decode_logits` staging buffer
// (P3-03) for the batched `ninfer::ops::sample` the caller runs next.
//
// This replaces #102's per-lane loop, which streamed the whole model once
// per lane and shared only the sampling: the round's cost scaled with the
// batch (#111 measured 4.81x its own B=1 round at B=4, against the
// reference's 1.07x). Every per-sequence input the traversal needs -- token
// id, absolute position, physical pool slot -- is read from device staging
// indexed by row, so the lanes' KV pages, GDN slots and conv taps stay
// isolated inside the one call.
//
// Enqueue-only and free of host-visible branching on device data, so the
// same call records a capture (`capture_decode_graph_width` below) and runs
// a round eagerly at a width whose capture failed
// (`ignis_program_decode`, kernel/src/step.cu).
int32_t ignis_decode_graph_run_batch(ignis_model *model, ignis_seq_pool *pool, uint32_t width,
                                     LinearPolicyMode mode) {
  if (model == nullptr || pool == nullptr) {
    set_error("ignis_decode_graph: null argument");
    return -1;
  }
  if (width == 0 || width > IGNIS_DECODE_MAX_BATCH) {
    set_error("ignis_decode_graph: width " + std::to_string(width) +
              " is not in 1..IGNIS_DECODE_MAX_BATCH");
    return -1;
  }
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  const auto batch = static_cast<std::int32_t>(width);
  ninfer::DeviceArena::Scope scope = model->decode_graph_scratch->scope();
  try {
    const ninfer::Tensor ids(model->decode_graph_token_ids->p, ninfer::DType::I32,
                             {batch, 1, 1, 1});
    ninfer::Tensor left =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::Tensor right =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::ops::embedding(ids, model->token_embedding, left, model->stream);

    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_run_body_graph(model, pool, layer, width, left.data, right.data, mode)
          : ignis_gdn_layer_run_body_graph(model, pool, layer, width, left.data, right.data, mode);
      if (rc != 0) {
        const char *detail = model->layers[layer].kind == IGNIS_LAYER_GQA
            ? ignis_gqa_layer_last_error()
            : ignis_gdn_layer_last_error();
        set_error("ignis_decode_graph: width " + std::to_string(width) + " layer " +
                  std::to_string(layer) + " failed: " + detail);
        return -1;
      }
      std::swap(left, right);
    }

    const ninfer::Tensor norm_weight(const_cast<void *>(model->final_norm.qdata), ninfer::DType::BF16,
                                     {hidden, 1, 1, 1});
    ninfer::Tensor normalized =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, batch, 1, 1});
    ninfer::ops::rmsnorm(left, norm_weight, model->rms_norm_eps, /*unit_offset=*/true, normalized,
                         model->stream);
    // The output head writes the round's logits directly in the sampler's
    // own `[vocab, batch]` column layout, so no per-lane column copy is
    // needed at all any more.
    ninfer::Tensor logits(model->sampling_decode_logits->p, ninfer::DType::BF16,
                          {vocab, batch, 1, 1});
    ninfer::ops::linear(normalized, model->output_head, logits, model->stream);
    return 0;
  } catch (const std::exception &e) {
    set_error("ignis_decode_graph: width " + std::to_string(width) + ": " + e.what());
    return -1;
  }
}

// P5-04 (GitHub #153): the verify round's device-side pass at batch `width`
// (layer_internal.h). Column order is lane-major: lane b's `k+1` columns are
// contiguous, which is both what the vendored `speculative_*` ops mean by a
// `[K+1,B]` matrix and how the verify layer bodies view the residual, so no
// column shuffle exists anywhere in the round.
//
// The accept kernel reads the round's sampling configs from
// `sampling_decode_configs` (per lane, as the decode round stages them) and
// keys its stateless RNG by `lengths` (each lane's frontier) and the
// speculative purposes, so a lane's draws depend on its own seed and
// position alone -- G3's isolation property, kept by construction.
int32_t ignis_verify_graph_run_batch(ignis_model *model, ignis_seq_pool *pool, uint32_t width,
                                     LinearPolicyMode mode) {
  if (model == nullptr || pool == nullptr) {
    set_error("ignis_verify_graph: null argument");
    return -1;
  }
  if (model->verify == nullptr) {
    set_error("ignis_verify_graph: the model was loaded without a draft window");
    return -1;
  }
  if (width == 0 || width > IGNIS_DECODE_MAX_BATCH) {
    set_error("ignis_verify_graph: width " + std::to_string(width) +
              " is not in 1..IGNIS_DECODE_MAX_BATCH");
    return -1;
  }
  IgnisVerifyRound &verify = *model->verify;
  const auto k = static_cast<std::int32_t>(verify.window);
  const std::int32_t lane_columns = k + 1;
  const auto batch = static_cast<std::int32_t>(width);
  const std::int32_t columns = lane_columns * batch;
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  ninfer::DeviceArena::Scope scope = model->decode_graph_scratch->scope();
  try {
    // Column 0 of every lane is its anchor, columns 1..extent its drafts,
    // the tail the anchor again; positions `base + min(j, extent)`.
    const ninfer::Tensor anchors(verify.anchors->p, ninfer::DType::I32, {batch, 1, 1, 1});
    const ninfer::Tensor drafts(verify.drafts->p, ninfer::DType::I32, {k, batch, 1, 1});
    const ninfer::Tensor base_positions(verify.base_positions->p, ninfer::DType::I32,
                                        {batch, 1, 1, 1});
    const ninfer::Tensor extents(verify.extents->p, ninfer::DType::I32, {batch, 1, 1, 1});
    ninfer::Tensor verify_ids(verify.verify_ids->p, ninfer::DType::I32, {lane_columns, batch, 1, 1});
    ninfer::Tensor positions(verify.positions->p, ninfer::DType::I32, {lane_columns, batch, 1, 1});
    // P5-05 (GitHub #155): on a load with the drafter, its forward writes
    // this round's drafts from each lane's window before the verify inputs
    // read them -- one pass, inside the same graph.
    if (verify.drafter_scratch != nullptr) {
      ignis_dflash2_propose(model, pool, width);
    }
    ninfer::ops::speculative_prepare_verify_inputs(anchors, drafts, base_positions, extents,
                                                   verify_ids, positions, model->stream);

    const ninfer::Tensor ids(verify.verify_ids->p, ninfer::DType::I32, {columns, 1, 1, 1});
    ninfer::Tensor left =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::Tensor right =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::ops::embedding(ids, model->token_embedding, left, model->stream);

    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_run_body_verify(model, pool, layer, width, left.data, right.data, mode)
          : ignis_gdn_layer_run_body_verify(model, pool, layer, width, left.data, right.data, mode);
      if (rc != 0) {
        const char *detail = model->layers[layer].kind == IGNIS_LAYER_GQA
            ? ignis_gqa_layer_last_error()
            : ignis_gdn_layer_last_error();
        set_error("ignis_verify_graph: width " + std::to_string(width) + " layer " +
                  std::to_string(layer) + " failed: " + detail);
        return -1;
      }
      std::swap(left, right);
      // P5-05 (GitHub #155): `left` is this layer's output now. At a tapped
      // layer every column's output lands in rows [j * hidden, (j + 1) *
      // hidden) of `features` -- the prefill chunk's own tap, over the round's
      // columns.
      const auto tap = std::find(kDflash2TapLayers.begin(), kDflash2TapLayers.end(), layer);
      if (verify.features != nullptr && tap != kDflash2TapLayers.end()) {
        const auto j = static_cast<std::size_t>(tap - kDflash2TapLayers.begin());
        const std::size_t row_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
        const cudaError_t tap_err = cudaMemcpy2DAsync(
            static_cast<std::uint8_t *>(verify.features->p) + j * row_bytes,
            kDflash2TapLayers.size() * row_bytes, left.data, static_cast<std::size_t>(left.nb[1]),
            row_bytes, static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice, model->stream);
        if (tap_err != cudaSuccess) {
          set_error("ignis_verify_graph: feature tap at layer " + std::to_string(layer) +
                    " failed: " + cudaGetErrorString(tap_err));
          return -1;
        }
      }
    }

    // The final residual of every column, kept past the round for the
    // accepted-hidden selection (the drafter's continuation input, P5-05).
    cudaError_t err = cudaMemcpyAsync(verify.hidden->p, left.data,
                                      static_cast<std::size_t>(hidden) * columns * sizeof(uint16_t),
                                      cudaMemcpyDeviceToDevice, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_verify_graph: cudaMemcpyAsync(hidden) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }

    const ninfer::Tensor norm_weight(const_cast<void *>(model->final_norm.qdata), ninfer::DType::BF16,
                                     {hidden, 1, 1, 1});
    ninfer::Tensor normalized =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::ops::rmsnorm(left, norm_weight, model->rms_norm_eps, /*unit_offset=*/true, normalized,
                         model->stream);
    ninfer::Tensor logits(verify.logits->p, ninfer::DType::BF16, {vocab, columns, 1, 1});
    ninfer::ops::linear(normalized, model->output_head, logits, model->stream);
    ninfer::Tensor target_tokens(verify.target_tokens->p, ninfer::DType::I32, {columns, 1, 1, 1});
    ninfer::ops::argmax(logits, target_tokens, vocab, model->stream);

    // The vendored accept: the greedy branch keeps the longest draft prefix
    // the target argmax agrees with and takes the target argmax at the
    // divergence; the sampling branch (temperature > 0) is the
    // distribution-preserving rule over the same columns. Its outputs are
    // read back by the caller after the round.
    const ninfer::Tensor targets_rows(verify.target_tokens->p, ninfer::DType::I32,
                                      {lane_columns, batch, 1, 1});
    const ninfer::Tensor logits_rows(verify.logits->p, ninfer::DType::BF16,
                                     {vocab, lane_columns, batch, 1});
    ninfer::Tensor lengths(verify.lengths->p, ninfer::DType::I32, {batch, 1, 1, 1});
    ninfer::Tensor anchors_out(verify.anchors->p, ninfer::DType::I32, {batch, 1, 1, 1});
    ninfer::Tensor licensed_tokens(verify.licensed_tokens->p, ninfer::DType::I32,
                                   {lane_columns, batch, 1, 1});
    ninfer::Tensor licensed_counts(verify.licensed_counts->p, ninfer::DType::I32, {batch, 1, 1, 1});
    ninfer::Tensor accepted(verify.accepted->p, ninfer::DType::I32, {batch, 1, 1, 1});
    ninfer::DeviceArena::Scope accept_scope = verify.accept_workspace->scope();
    ninfer::ops::speculative_accept_greedy_drafts(
        targets_rows, logits_rows, drafts, extents, lengths, anchors_out, licensed_tokens,
        licensed_counts, accepted, vocab,
        static_cast<const ninfer::ops::SamplingConfig *>(model->sampling_decode_configs->p),
        *verify.accept_workspace, model->stream);
    return 0;
  } catch (const std::exception &e) {
    set_error("ignis_verify_graph: width " + std::to_string(width) + ": " + e.what());
    return -1;
  }
}

namespace {

// P5-04 (GitHub #153): records one width-W verify graph -- exactly
// `ignis_verify_graph_run_batch`, captured once. Same failure handling as
// the decode capture below.
int32_t capture_verify_graph_width(ignis_model *model, ignis_seq_pool *pool, uint32_t width,
                                   cudaGraphExec_t *out_exec) {
  *out_exec = nullptr;
  cudaError_t err = cudaStreamBeginCapture(model->stream, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_verify_graph_capture: cudaStreamBeginCapture failed: ") +
              cudaGetErrorString(err));
    return -1;
  }
  const bool ok =
      ignis_verify_graph_run_batch(model, pool, width, LinearPolicyMode::kEngineDefault) == 0;
  cudaGraph_t graph = nullptr;
  const cudaError_t end_err = cudaStreamEndCapture(model->stream, &graph);
  if (end_err != cudaSuccess) {
    set_error(std::string("ignis_verify_graph_capture: cudaStreamEndCapture failed: ") +
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
    set_error(std::string("ignis_verify_graph_capture: cudaGraphInstantiate failed: ") +
              cudaGetErrorString(inst_err));
    return -1;
  }
  *out_exec = exec;
  return 0;
}

// Records one width-W graph: one W-wide model traversal
// (`ignis_decode_graph_run_batch` above) followed by one batched
// `ninfer::ops::sample` reading `model->sampling_decode_{configs,positions,
// logits,out}` at their stable addresses (P3-03) -- exactly the op sequence
// `ignis_program_decode`'s eager path runs, captured once. On any failure
// the partial graph is discarded and `*out_exec` stays null; the caller
// leaves this width on the eager fallback.
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

  bool ok = ignis_decode_graph_run_batch(model, pool, width, LinearPolicyMode::kEngineDefault) == 0;
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
  // Cleared up front (not just overwritten per-width below): otherwise a
  // failure from an earlier call to this function survives a later call
  // that captures every width cleanly, breaking
  // ignis_decode_graph_last_error's "empty if every width captured"
  // contract (kernel/include/ignis_step.h).
  set_error("");
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
  // P5-04 (GitHub #153): the verify graphs, one per exact width at the
  // load's window, under the same never-refuse-service rule -- a width whose
  // verify capture failed runs the verify traversal eagerly.
  if (model->verify != nullptr) {
    for (uint32_t width = 1; width <= IGNIS_DECODE_MAX_BATCH; ++width) {
      cudaGraphExec_t exec = nullptr;
      if (capture_verify_graph_width(model, pool, width, &exec) == 0) {
        model->verify->graph_exec[width - 1] = exec;
        model->verify->graph_ready[width - 1] = true;
      } else {
        model->verify->graph_exec[width - 1] = nullptr;
        model->verify->graph_ready[width - 1] = false;
      }
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
