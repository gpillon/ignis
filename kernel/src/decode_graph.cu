// ignis kernel leaf - P3-05 (GitHub #102, ADR 0019) and GitHub #111 (ADR
// 0020): the decode round's forward pass, and one CUDA graph capturing
// it per exact batch width 1..decode_lanes, the load's (GitHub #336). The program layer is
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

#include "decode_graph_copies.h"
#include "deferred_verify.h"
#include "dflash2_drafter.h"
#include "extended_attention.h"
#include "gdn_gating_home.h"
#include "graph_nodes.h"
#include "ignis_gdn_layer.h"
#include "ignis_gqa_layer.h"
#include "layer_internal.h"
#include "mixed_graphs.h"
#include "model_internal.h"
#include "mtp_head.h"
#include "sampled_draft.h"
#include "tiled_argmax.h"
#include "trellis_body.h"
#include "flash_next/program.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"

#include "process_environment.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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
  // GitHub #336: the load's lanes, which every round buffer is sized for.
  if (width == 0 || width > model->decode_lanes) {
    set_error("ignis_decode_graph: width " + std::to_string(width) + " is not in 1.." +
              std::to_string(model->decode_lanes) + ", the load's decode lanes");
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

    // GitHub #677, #680 (spec #672, kernel/src/decode_graph_copies.h): at a
    // copy-free width every layer adds its output into its input residual in
    // place (`left` is both), as the prefill's GDN body does since GitHub
    // #507 -- the layer's input RMSNorm is that residual's only other reader,
    // and it runs before the first add -- so no layer copies the residual and
    // the two buffers never swap; and each GDN layer's recurrence reads q, k
    // and v where the conv wrote them: at one column as views of the conv
    // output plane, where they are contiguous already, past one column as the
    // planes the split conv writes by lane slot (kernel/src/gdn_layer.cu
    // `run_gdn_layer_graph`, kernel/src/gdn_conv_split.h). A layer change
    // keeps both: nothing may read the layer's input after its first add into
    // the residual, and nothing may write the conv's q, k and v between the
    // conv and the recurrence. Elsewhere (the copying round of earlier
    // releases) each layer copies its input into `right` and adds there.
    const bool in_place = ignis::decode_graph_copies::takes(model->decode_copy_free_widths, width);
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      void *out = in_place ? left.data : right.data;
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_run_body_graph(model, pool, layer, width, left.data, out, mode)
          : ignis_gdn_layer_run_body_graph(model, pool, layer, width, left.data, out, mode);
      if (rc != 0) {
        const char *detail = model->layers[layer].kind == IGNIS_LAYER_GQA
            ? ignis_gqa_layer_last_error()
            : ignis_gdn_layer_last_error();
        set_error("ignis_decode_graph: width " + std::to_string(width) + " layer " +
                  std::to_string(layer) + " failed: " + detail);
        return -1;
      }
      if (!in_place) {
        std::swap(left, right);
      }
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
    ignis::trellis_body::output_head(*model, normalized, logits, model->stream);
    // GitHub #620: on an MTP load each lane's head row at the position before
    // the one it consumed, and its continuation hidden moves to the round's
    // (kernel/src/mtp_head.h) -- graph-safe, so every captured width keeps
    // the head current as the eager round does.
    if (model->drafter == IgnisDrafter::kMtp) {
      std::string head_error;
      if (ignis::mtp::decode_round(*model, *pool, width, normalized, &head_error) != 0) {
        set_error("ignis_decode_graph: width " + std::to_string(width) + ": " + head_error);
        return -1;
      }
    }
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
//
// GitHub #686: at the round's shape `verify`, which every call below is handed.
//
// GitHub #713 (spec #711): one traversal for every layout. A mixed round (`verify` its base shape,
// whose buffers both shapes hold) runs the ops that see only columns once over its columns -- the
// embedding, every norm, linear and MLP, the taps, the output head and its argmax -- and the ops
// that see lanes once a group on its slice: the drafter's lane ops, the verify inputs, the GQA
// attention and KV append and the GDN conv record and recurrence (each group on the route a
// uniform round of its shape and lanes takes), the base group on the branch beside the extended
// one (`ignis_each_group`); and the accept once a group, one after the other, on the workspace the
// groups share. A uniform round makes the calls it always made.
int32_t ignis_verify_graph_run_batch(ignis_model *model, ignis_seq_pool *pool, IgnisVerifyRound &verify,
                                     const ignis::round_layout::Layout &layout, LinearPolicyMode mode) {
  if (model == nullptr || pool == nullptr) {
    set_error("ignis_verify_graph: null argument");
    return -1;
  }
  const uint32_t width = layout.lanes();
  // GitHub #687: the shape's own lanes, the load's decode lanes at its base
  // shape and no more than IGNIS_EXTENDED_SHAPE_MAX_LANES at an extended one.
  if (width == 0 || width > verify.lanes) {
    set_error("ignis_verify_graph: width " + std::to_string(width) + " is not in 1.." +
              std::to_string(verify.lanes) + ", the lanes of the round shape at " + std::to_string(verify.window) +
              " drafts");
    return -1;
  }
  if (!layout.uniform() && (model->drafter != IgnisDrafter::kDflash2 || model->extended_shape == nullptr ||
                            &verify != model->base_shape.get())) {
    set_error("ignis_verify_graph: a mixed round is a DFlash2 load's, on its base shape's buffers");
    return -1;
  }
  const auto columns = static_cast<std::int32_t>(layout.columns());
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  ninfer::DeviceArena::Scope scope = model->decode_graph_scratch->scope();
  try {
    // GitHub #619: the proposal seam, filled by the load's drafter before
    // the verify inputs read the drafts -- one pass, inside the same graph.
    // DFlash2's forward writes this round's drafts from each lane's window
    // (P5-05, GitHub #155); the MTP head's chain from each lane's
    // continuation hidden and anchor (GitHub #621, kernel/src/mtp_head.h);
    // the verify substrate alone's were staged by the round from its caller.
    switch (model->drafter) {
    case IgnisDrafter::kDflash2:
      ignis_dflash2_propose(model, pool, verify, layout);
      break;
    case IgnisDrafter::kMtp: {
      std::string head_error;
      if (ignis::mtp::propose(*model, *pool, verify, width, &head_error) != 0) {
        set_error("ignis_verify_graph: width " + std::to_string(width) + ": " + head_error);
        return -1;
      }
      break;
    }
    case IgnisDrafter::kHost:
    case IgnisDrafter::kNone:
      break;
    }
    // GitHub #692 (test-only): a teacher-forced round that names its drafts
    // verifies them in place of the proposal, written over it here -- an
    // eager round's alone; a capture refuses them (below).
    if (verify.forced_drafts != nullptr) {
      const cudaError_t err =
          cudaMemcpyAsync(verify.drafts->p, verify.forced_drafts,
                          static_cast<std::size_t>(layout.drafts()) * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                          model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_verify_graph: cudaMemcpyAsync(forced drafts) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }
    // Column 0 of every lane is its anchor, columns 1..extent its drafts,
    // the tail the anchor again; positions `base + min(j, extent)`. GitHub
    // #712: a group's lanes at a time, at its columns and drafts: its rows of
    // a per-lane buffer, of a column buffer and of the drafts.
    using Group = ignis::round_layout::Group;
    const auto lane_rows = [](const ninfer::DeviceBuffer &all, const Group &group) {
      return ninfer::Tensor(static_cast<std::int32_t *>(all.p) + group.first_lane, ninfer::DType::I32,
                            {static_cast<std::int32_t>(group.lanes), 1, 1, 1});
    };
    const auto column_rows = [](const ninfer::DeviceBuffer &all, const Group &group) {
      return ninfer::Tensor(static_cast<std::int32_t *>(all.p) + group.first_column, ninfer::DType::I32,
                            {static_cast<std::int32_t>(group.lane_columns()), static_cast<std::int32_t>(group.lanes), 1,
                             1});
    };
    const auto draft_rows = [](const ninfer::DeviceBuffer &all, const Group &group) {
      return ninfer::Tensor(static_cast<std::int32_t *>(all.p) + group.first_draft, ninfer::DType::I32,
                            {static_cast<std::int32_t>(group.window), static_cast<std::int32_t>(group.lanes), 1, 1});
    };
    ignis_each_group(*model, layout, [&](const Group &group, cudaStream_t on) {
      ninfer::Tensor group_ids = column_rows(*verify.verify_ids, group);
      ninfer::Tensor group_positions = column_rows(*verify.positions, group);
      ninfer::ops::speculative_prepare_verify_inputs(lane_rows(*verify.anchors, group), draft_rows(*verify.drafts, group),
                                                     lane_rows(*verify.base_positions, group),
                                                     lane_rows(*verify.extents, group), group_ids, group_positions, on);
    });

    const ninfer::Tensor ids(verify.verify_ids->p, ninfer::DType::I32, {columns, 1, 1, 1});
    ninfer::Tensor left =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::Tensor right =
        model->decode_graph_scratch->alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::ops::embedding(ids, model->token_embedding, left, model->stream);

    // GitHub #680 (spec #672, kernel/src/decode_graph_copies.h): copy-free,
    // every layer adds its output into its input residual in place, as the
    // decode round does (above), so no layer copies the residual and the
    // buffers never swap; `left` is each layer's output where the feature
    // taps below read it either way. The GDN record body writes q, k and v
    // apart already. Elsewhere (the copying round of earlier releases) each
    // layer copies its input into `right` and adds there.
    const bool in_place = model->verify_copy_free;
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      void *out = in_place ? left.data : right.data;
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_run_body_verify(model, pool, verify, layer, layout, left.data, out, mode)
          : ignis_gdn_layer_run_body_verify(model, pool, verify, layer, layout, left.data, out, mode);
      if (rc != 0) {
        const char *detail = model->layers[layer].kind == IGNIS_LAYER_GQA
            ? ignis_gqa_layer_last_error()
            : ignis_gdn_layer_last_error();
        set_error("ignis_verify_graph: width " + std::to_string(width) + " layer " +
                  std::to_string(layer) + " failed: " + detail);
        return -1;
      }
      if (!in_place) {
        std::swap(left, right);
      }
      // P5-05 (GitHub #155): `left` is this layer's output now. At a tapped
      // layer every column's output lands in rows [j * hidden, (j + 1) *
      // hidden) of `features` -- the prefill chunk's own tap, over the round's
      // columns. DFlash2's alone (GitHub #619).
      const auto tap = std::find(kDflash2TapLayers.begin(), kDflash2TapLayers.end(), layer);
      if (model->drafter == IgnisDrafter::kDflash2 && tap != kDflash2TapLayers.end()) {
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
    ignis::trellis_body::output_head(*model, normalized, logits, model->stream);
    ninfer::Tensor target_tokens(verify.target_tokens->p, ninfer::DType::I32, {columns, 1, 1, 1});
    // GitHub #718: a mixed round under the mixed graph's build zeroes its argmax's output by a kernel
    // where the vendored launcher's memset node would wait behind the output head, the same answer
    // (kernel/src/tiled_argmax.h); every other round, the vendored op.
    if (!layout.uniform() && model->mixed_graph_build) {
      ignis::tiled_argmax::argmax(logits, target_tokens, vocab, model->stream);
    } else {
      ninfer::ops::argmax(logits, target_tokens, vocab, model->stream);
    }

    // The vendored accept: the greedy branch keeps the longest draft prefix
    // the target argmax agrees with and takes the target argmax at the
    // divergence; the sampling branch (temperature > 0) is the
    // distribution-preserving rule over the same columns. Its outputs are
    // read back by the caller after the round. GitHub #713: a group's lanes
    // at a time, one group after the other on the workspace they share.
    const auto *configs = static_cast<const ninfer::ops::SamplingConfig *>(model->sampling_decode_configs->p);
    for (const Group &group : layout.groups) {
      if (group.lanes == 0) {
        continue;
      }
      const ninfer::Tensor targets_rows = column_rows(*verify.target_tokens, group);
      const ninfer::Tensor logits_rows(
          static_cast<std::uint8_t *>(verify.logits->p) +
              static_cast<std::size_t>(group.first_column) * vocab * sizeof(std::uint16_t),
          ninfer::DType::BF16,
          {vocab, static_cast<std::int32_t>(group.lane_columns()), static_cast<std::int32_t>(group.lanes), 1});
      const ninfer::Tensor drafts = draft_rows(*verify.drafts, group);
      const ninfer::Tensor extents = lane_rows(*verify.extents, group);
      ninfer::Tensor lengths = lane_rows(*verify.lengths, group);
      ninfer::Tensor anchors_out = lane_rows(*verify.anchors, group);
      ninfer::Tensor licensed_tokens = column_rows(*verify.licensed_tokens, group);
      ninfer::Tensor licensed_counts = lane_rows(*verify.licensed_counts, group);
      ninfer::Tensor accepted = lane_rows(*verify.accepted, group);
      ninfer::DeviceArena::Scope accept_scope = verify.accept_workspace->scope();
      if (verify.sampled_draft) {
        // GitHub #576 (ADR 0090): under the sampled draft the accept is ours
        // -- the sampling lanes take their drafts by the ratio of p to the q
        // their draw kept (one-hot on a verify-only load's drafts), the greedy
        // lanes by today's rule -- with the vendored accept's outputs.
        const std::size_t draft_words = static_cast<std::size_t>(group.first_draft) * kDflash2SelectorTopK;
        ignis_sampled_accept(targets_rows, logits_rows, drafts, extents, lengths, anchors_out, licensed_tokens,
                             licensed_counts, accepted, vocab, configs + group.first_lane,
                             verify.draft_candidates != nullptr
                                 ? static_cast<const std::int32_t *>(verify.draft_candidates->p) + draft_words
                                 : nullptr,
                             verify.draft_q != nullptr ? static_cast<const float *>(verify.draft_q->p) + draft_words
                                                       : nullptr,
                             *verify.accept_workspace, model->stream,
                             // GitHub #577 (test-only): a teacher-forced round's
                             // recording, set only while one runs eagerly.
                             verify.forced_tokens != nullptr ? verify.forced_tokens + group.first_column : nullptr);
      } else {
        ninfer::ops::speculative_accept_greedy_drafts(targets_rows, logits_rows, drafts, extents, lengths,
                                                      anchors_out, licensed_tokens, licensed_counts, accepted,
                                                      vocab, configs + group.first_lane, *verify.accept_workspace,
                                                      model->stream);
      }
    }
    return 0;
  } catch (const std::exception &e) {
    set_error("ignis_verify_graph: width " + std::to_string(width) + ": " + e.what());
    return -1;
  }
}

namespace {

// GitHub #572: a captured graph's node count (`ignis_program_stats`), what a route that takes
// another's place in a round must leave as it was; 0 if the runtime cannot say.
uint32_t node_count(cudaGraph_t graph) {
  std::size_t nodes = 0;
  if (cudaGraphGetNodes(graph, nullptr, &nodes) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return static_cast<uint32_t>(nodes);
}

// GitHub #677: how many of a captured graph's nodes are copies (`ignis_program_stats`); 0 if the
// runtime cannot say.
uint32_t copy_node_count(cudaGraph_t graph) {
  return ignis::graph_nodes::of_type(graph, cudaGraphNodeTypeMemcpy);
}

// P5-04 (GitHub #153): records one width-W verify graph -- exactly
// `ignis_verify_graph_run_batch`, captured once at shape `verify` (GitHub
// #686). Same failure handling as the decode capture below. GitHub #713: at
// `layout`, a uniform round of its width or a mixed round's mix. GitHub #718: and, where asked, its
// root nodes and memset nodes; a mixed round's under the mixed graph's build instantiated with its
// nodes' priorities (`ignis_mixed_priorities`).
int32_t capture_verify_graph(ignis_model *model, ignis_seq_pool *pool, IgnisVerifyRound &verify,
                             const ignis::round_layout::Layout &layout, cudaGraphExec_t *out_exec, uint32_t *out_nodes,
                             uint32_t *out_copies, uint32_t *out_roots = nullptr, uint32_t *out_memsets = nullptr) {
  *out_exec = nullptr;
  *out_nodes = 0;
  *out_copies = 0;
  // GitHub #577: a teacher-forced round's recording is set only while that
  // round runs eagerly; a graph that baked it in would force serving rounds.
  if (verify.forced_tokens != nullptr) {
    set_error("ignis_verify_graph_capture: a teacher-forced round's recording is set");
    return -1;
  }
  // GitHub #692: so are its forced drafts.
  if (verify.forced_drafts != nullptr) {
    set_error("ignis_verify_graph_capture: a teacher-forced round's forced drafts are set");
    return -1;
  }
  cudaError_t err = cudaStreamBeginCapture(model->stream, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_verify_graph_capture: cudaStreamBeginCapture failed: ") +
              cudaGetErrorString(err));
    return -1;
  }
  const bool ok = ignis_verify_graph_run_batch(model, pool, verify, layout, LinearPolicyMode::kEngineDefault) == 0;
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
  const uint32_t nodes = node_count(graph);
  const uint32_t copies = copy_node_count(graph);
  const uint32_t roots = out_roots != nullptr ? ignis::graph_nodes::roots(graph) : 0;
  const uint32_t memsets = out_memsets != nullptr ? ignis::graph_nodes::of_type(graph, cudaGraphNodeTypeMemset) : 0;
  const unsigned long long flags =
      ignis_mixed_priorities(*model, layout) ? static_cast<unsigned long long>(cudaGraphInstantiateFlagUseNodePriority) : 0ULL;
  cudaGraphExec_t exec = nullptr;
  const cudaError_t inst_err = cudaGraphInstantiate(&exec, graph, flags);
  cudaGraphDestroy(graph);
  if (inst_err != cudaSuccess) {
    set_error(std::string("ignis_verify_graph_capture: cudaGraphInstantiate failed: ") +
              cudaGetErrorString(inst_err));
    return -1;
  }
  *out_exec = exec;
  *out_nodes = nodes;
  *out_copies = copies;
  if (out_roots != nullptr) {
    *out_roots = roots;
  }
  if (out_memsets != nullptr) {
    *out_memsets = memsets;
  }
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
                                   cudaGraphExec_t *out_exec, uint32_t *out_nodes, uint32_t *out_copies) {
  *out_exec = nullptr;
  *out_nodes = 0;
  *out_copies = 0;
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

  const uint32_t nodes = node_count(graph);
  const uint32_t copies = copy_node_count(graph);
  cudaGraphExec_t exec = nullptr;
  const cudaError_t inst_err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  if (inst_err != cudaSuccess) {
    set_error(std::string("ignis_decode_graph_capture: cudaGraphInstantiate failed: ") +
              cudaGetErrorString(inst_err));
    return -1;
  }
  *out_exec = exec;
  *out_nodes = nodes;
  *out_copies = copies;
  return 0;
}

// A measurement switch read at each capture by `read`, which throws std::invalid_argument naming
// the variable on a value it does not know (kernel/src/process_environment.h `named_switch`), so
// that value refuses the capture by name. False (error set) on a refusal.
template <class Read, class Value> bool capture_switch(Read read, Value &named) {
  try {
    named = read();
  } catch (const std::invalid_argument &e) {
    set_error(std::string("ignis_decode_graph_capture: ") + e.what());
    return false;
  }
  return true;
}

// A width's graph from an earlier capture of the same load, gone before the width captures again
// (GitHub #674: a measurement switch read at capture, flipped in one process).
void release_graph(cudaGraphExec_t &exec, bool &ready, uint32_t &nodes) {
  if (exec != nullptr) {
    cudaGraphExecDestroy(exec);
    exec = nullptr;
  }
  ready = false;
  nodes = 0;
}

// GitHub #674: the round timing's switch (kernel/src/step.cu), read at each capture: its events
// are made the first time it is on, and every capture starts its blocks afresh. False (error set)
// on a value other than `0` or `1`, or when CUDA cannot make the events.
bool read_round_timing(ignis_model *model) {
  const auto read = [] { return ignis::named_switch("IGNIS_ROUND_TIMING", false, {{"0", false}, {"1", true}}); };
  if (!capture_switch(read, model->round_timing)) {
    return false;
  }
  for (RoundTiming *timing : {&model->plain_timing, &model->verify_timing}) {
    timing->recorded = 0;
    timing->collected = 0;
    timing->blocks = {};
    for (std::uint64_t i = 0; model->round_timing && i < RoundTiming::kRing; ++i) {
      for (cudaEvent_t *event : {&timing->pre[i], &timing->post[i]}) {
        const cudaError_t err = *event == nullptr ? cudaEventCreate(event) : cudaSuccess;
        if (err != cudaSuccess) {
          set_error(std::string("ignis_decode_graph_capture: IGNIS_ROUND_TIMING: cudaEventCreate failed: ") +
                    cudaGetErrorString(err));
          model->round_timing = false;
          return false;
        }
      }
    }
  }
  return true;
}

// The device's compute capability major, for a rule that follows the card; false (error set,
// naming `what`) when the device cannot say it.
bool device_generation(int *major, const char *what) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) err = cudaDeviceGetAttribute(major, cudaDevAttrComputeCapabilityMajor, device);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_decode_graph_capture: the GPU generation for ") + what + ": " +
              cudaGetErrorString(err));
    return false;
  }
  return true;
}

// GitHub #677, #680 (spec #672): the decode widths, and whether the verify round, run copy-free
// from this capture on (kernel/src/decode_graph_copies.h), from IGNIS_DECODE_GRAPH_COPIES and the
// card, read at each capture. False (error set) on a value other than `1` or `wide`, or when the
// device cannot say its generation.
bool read_decode_graph_copies(ignis_model *model) {
  ignis::decode_graph_copies::Switch named = ignis::decode_graph_copies::Switch::Unset;
  int major = 0;
  if (!capture_switch(ignis::decode_graph_copies::switch_at_capture, named) ||
      !device_generation(&major, "the copy-free round")) {
    return false;
  }
  model->decode_copy_free_widths = ignis::decode_graph_copies::copy_free_widths(named, major);
  model->verify_copy_free = ignis::decode_graph_copies::verify_copy_free(named, major);
  return true;
}

// GitHub #681 (spec #672): where each GDN layer's gating runs from this capture on
// (kernel/src/gdn_gating_home.h), from IGNIS_GDN_GATING and the card, read at each capture; the
// branch's stream and events are made by the first capture that takes it. False (error set) on a
// value other than `chain` or `branch`, when the device cannot say its generation, or when CUDA
// cannot make the branch.
bool read_gdn_gating(ignis_model *model) {
  ignis::gdn_gating::Home named = ignis::gdn_gating::kUnset;
  int major = 0;
  if (!capture_switch(ignis::gdn_gating::switch_at_capture, named) ||
      !device_generation(&major, "the GDN gating's home")) {
    return false;
  }
  const ignis::gdn_gating::Home home = ignis::gdn_gating::home_on(named, major);
  std::string error;
  if (home == ignis::gdn_gating::Home::Branch && model->gdn_gating_branch.stream == nullptr &&
      !model->gdn_gating_branch.create(&error)) {
    set_error("ignis_decode_graph_capture: the GDN gating's " + error);
    return false;
  }
  model->gdn_gating_home = home;
  return true;
}

// GitHub #718 (spec #711): IGNIS_MIXED_GRAPH_BUILD, read at each capture of a load with an extended
// shape (kernel/src/mixed_graphs.h `built`): the mixed rounds' build from here on, their graphs' and
// their eager rounds' alike (`ignis_model::mixed_graph_build`). The first capture that reads the
// build makes the branch the extended group of a mixed round runs on, at the device's greatest
// stream priority, kept for the load. False (error set) on a value other than `0` or `1`, and where
// CUDA cannot make that branch.
bool read_mixed_graph_build(ignis_model *model) {
  if (!capture_switch(ignis::mixed_graphs::built, model->mixed_graph_build)) {
    return false;
  }
  ignis::flash_next::SideBranch &branch = model->mixed_priority_branch;
  if (!model->mixed_graph_build || branch.stream != nullptr) {
    return true;
  }
  int least = 0;
  int greatest = 0;
  cudaError_t err = cudaDeviceGetStreamPriorityRange(&least, &greatest);
  if (err == cudaSuccess) {
    err = cudaStreamCreateWithPriority(&branch.stream, cudaStreamNonBlocking, greatest);
  }
  if (err == cudaSuccess) {
    err = cudaEventCreateWithFlags(&branch.fork, cudaEventDisableTiming);
  }
  if (err == cudaSuccess) {
    err = cudaEventCreateWithFlags(&branch.join, cudaEventDisableTiming);
  }
  if (err != cudaSuccess) {
    branch.destroy();
    set_error(std::string("ignis_decode_graph_capture: IGNIS_MIXED_GRAPH_BUILD: the extended group's branch: ") +
              cudaGetErrorString(err));
    return false;
  }
  return true;
}

// GitHub #713 (spec #711): IGNIS_MIXED_ROUNDS, read at each capture (a measurement switch, ADR 0093's
// kind, kernel/src/mixed_graphs.h `asked`): `1` captures a verify graph and, on a load whose verify
// rounds defer, the drafter's append after it for each mix the load holds (GitHub #714: every mix of
// kMixTable whose lanes fit the load's and whose base group's fold fits the pool's GDN slots,
// `ignis::mixed_graphs::planned`, the mixes the load's plan holds); `0` none; GitHub #715: unset,
// the mixes of a load whose shape policy is `mixed`, and none under `aggregate` or `all`. Every
// mixed graph of an earlier capture goes first, so a capture under `0` gives their memory back.
// A mix whose capture fails has no graph -- its verify graph is
// released where its append graph fails, so a mix is both or neither -- its rounds run at the
// uniform choice (kernel/src/step.cu), its bit is in `mixed_failed_mask` and the capture's error
// names it, and no capture is refused for it. Each captured mix's bytes are the device's free
// memory before its capture less after. False (error set) on a value other than `0` or `1`, and
// where CUDA cannot make the branch a mixed round runs its base group on.
bool capture_mixed_graphs(ignis_model *model, ignis_seq_pool *pool) {
  for (std::uint32_t i = 0; i < ignis::round_layout::kMixes; ++i) {
    ignis_release_mixed_graphs(*model, i, true);
  }
  model->mixed_failed_mask = 0;
  model->mixed_capture_asked = false;
  bool captures = false;
  // GitHub #715: unset, the shape policy asks, where the load runs `mixed`.
  const bool policy_mixed = model->shape_policy.policy == ignis::round_shape::Policy::kMixed;
  if (!capture_switch([policy_mixed] { return ignis::mixed_graphs::asked(policy_mixed); }, captures)) {
    return false;
  }
  IgnisVerifyRound *const base = model->base_shape.get();
  if (!captures || base == nullptr || model->extended_shape == nullptr) {
    return true;
  }
  model->mixed_capture_asked = true;
  const std::int64_t slots = pool->gdn_pool.slot_count();
  const std::uint32_t planned =
      ignis::mixed_graphs::planned(true, model->drafter == IgnisDrafter::kDflash2, model->extended_shape->window,
                                   model->decode_lanes, slots);
  // GitHub #714 (test-only): the mix the test seam fails, once.
  const int fault = model->mixed_capture_fault;
  model->mixed_capture_fault = -1;
  for (std::uint32_t at = 0; at < ignis::round_layout::kMixes; ++at) {
    if (((planned >> at) & 1u) == 0) {
      continue;
    }
    const auto &mix = ignis::round_layout::kMixTable[at];
    const ignis::round_layout::Layout layout =
        ignis::round_layout::layout_of(mix[0], model->extended_shape->window, mix[1], base->window);
    const std::string name = "the mixed round " + std::to_string(mix[0]) + "+" + std::to_string(mix[1]);
    const auto failed = [&](const std::string &why) {
      ignis_release_mixed_graphs(*model, at, false);
      model->mixed_failed_mask |= 1u << at;
      set_error("ignis_decode_graph_capture: " + name + ": " + why);
    };
    // The plan's rule and the round's own refusal name one set of mixes; a mix the rule holds and
    // the round refuses has no graph, as any failed capture.
    if (const char *refusal = ignis_mixed_round_refusal(*model, slots, layout)) {
      failed(std::string("the round refuses it: ") + refusal);
      continue;
    }
    if (fault == static_cast<int>(at)) {
      failed("its capture failed at the test seam's ask (ignis_decode_graph_fail_mixed_capture)");
      continue;
    }
    std::string branch_error;
    if (model->mixed_branch.stream == nullptr && !model->mixed_branch.create(&branch_error)) {
      set_error("ignis_decode_graph_capture: IGNIS_MIXED_ROUNDS: " + branch_error);
      return false;
    }
    std::size_t free_before = 0;
    std::size_t free_after = 0;
    std::size_t total = 0;
    const bool measured = cudaMemGetInfo(&free_before, &total) == cudaSuccess;
    uint32_t copies = 0;
    if (capture_verify_graph(model, pool, *base, layout, &model->mixed_verify_exec[at], &model->mixed_verify_nodes[at],
                             &copies, &model->mixed_verify_roots[at], &model->mixed_verify_memsets[at]) != 0) {
      model->mixed_verify_exec[at] = nullptr;
      failed("its verify graph: " + g_last_error);
      continue;
    }
    if (base->defer && base->staging != nullptr) {
      std::string error;
      if (ignis::deferred_verify::capture_append(model, pool, *base, layout, &model->mixed_append_exec[at],
                                                 &model->mixed_append_nodes[at], &error, &model->mixed_append_roots[at],
                                                 &model->mixed_append_memsets[at]) != 0) {
        model->mixed_append_exec[at] = nullptr;
        failed("the drafter's append after it: " + error);
        continue;
      }
    }
    if (measured && cudaMemGetInfo(&free_after, &total) == cudaSuccess && free_after < free_before) {
      model->mixed_graph_bytes[at] = free_before - free_after;
    }
  }
  return true;
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
  // GitHub #302: a Flash-Next load captures its own rounds, widths
  // 1..decode_lanes, under the same never-refuse-service rule.
  if (model->flash_next) {
    const auto began = std::chrono::steady_clock::now();
    uint32_t ready_mask = 0;
    std::string error;
    const int32_t rc = ignis::flash_next::capture_decode_graphs(model, pool, &ready_mask, &error);
    set_error(error);
    if (out_capture_micros != nullptr) {
      *out_capture_micros = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - began)
              .count());
    }
    if (out_ready_mask != nullptr) {
      *out_ready_mask = ready_mask;
    }
    return rc;
  }
  if (!read_round_timing(model) || !read_decode_graph_copies(model) || !read_gdn_gating(model)) {
    return -1;
  }
  // GitHub #686: the shapes whose verify and append graphs this capture
  // makes, the load's base shape and (GitHub #687) its extended shape where
  // it has one; none on a load with no window.
  IgnisVerifyRound *const shapes[] = {model->base_shape.get(), model->extended_shape.get()};
  IgnisVerifyRound *const shape = model->base_shape.get();
  // GitHub #676: IGNIS_DEFER_VERIFY, read at each capture like the round
  // timing; a load with no window has no verify round to defer. Both of a
  // load's shapes defer alike.
  if (shape != nullptr && !capture_switch(ignis::deferred_verify::read_switch, shape->defer)) {
    return -1;
  }
  if (model->extended_shape != nullptr) {
    model->extended_shape->defer = shape->defer;
  }
  // GitHub #688: IGNIS_EXTENDED_ATTENTION, read at each capture of a load
  // with an extended shape (kernel/src/extended_attention.h): the route of
  // every extended round's chunks from here on, its graphs' and its eager
  // rounds' alike. A load without one never reads it.
  if (model->extended_shape != nullptr &&
      !capture_switch(ignis::extended_attention::read_switch, model->extended_attention)) {
    return -1;
  }
  // GitHub #718: IGNIS_MIXED_GRAPH_BUILD, the mixed rounds' build, on the same loads.
  if (model->extended_shape != nullptr && !read_mixed_graph_build(model)) {
    return -1;
  }
  const auto began = std::chrono::steady_clock::now();
  uint32_t ready_mask = 0;
  // GitHub #674: a capture again on the same load (a measurement switch
  // read at capture, flipped) replaces the graphs it captured before, once
  // no round of theirs is still on the stream.
  const cudaError_t synced = cudaStreamSynchronize(model->stream);
  if (synced != cudaSuccess) {
    set_error(std::string("ignis_decode_graph_capture: cudaStreamSynchronize failed: ") +
              cudaGetErrorString(synced));
    return -1;
  }
  for (uint32_t width = 1; width <= IGNIS_DECODE_MAX_BATCH; ++width) {
    release_graph(model->decode_graph_exec[width - 1], model->decode_graph_ready[width - 1],
                  model->decode_graph_nodes[width - 1]);
    model->decode_graph_copy_nodes[width - 1] = 0;
    for (IgnisVerifyRound *const held : shapes) {
      if (held == nullptr) {
        continue;
      }
      release_graph(held->graph_exec[width - 1], held->graph_ready[width - 1], held->graph_nodes[width - 1]);
      held->graph_copy_nodes[width - 1] = 0;
      // GitHub #676: and the drafter's append graph.
      if (held->append_exec[width - 1] != nullptr) {
        cudaGraphExecDestroy(held->append_exec[width - 1]);
        held->append_exec[width - 1] = nullptr;
      }
      held->append_nodes[width - 1] = 0;
    }
  }
  // GitHub #336: the widths the load's lanes serve, and no wider: a round
  // past them is refused, so a graph for it would only hold memory.
  for (uint32_t width = 1; width <= model->decode_lanes; ++width) {
    cudaGraphExec_t exec = nullptr;
    if (capture_decode_graph_width(model, pool, width, &exec, &model->decode_graph_nodes[width - 1],
                                   &model->decode_graph_copy_nodes[width - 1]) == 0) {
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
  // verify capture failed runs the verify traversal eagerly. On an MTP load
  // the head's chain is in them (GitHub #621). GitHub #687: each of the
  // load's round shapes at its own widths, 1..its lanes; a failed width is in
  // the shape's stats (`ignis_program_stats`), which the runtime reports.
  for (IgnisVerifyRound *const held : shapes) {
    if (held == nullptr) {
      continue;
    }
    for (uint32_t width = 1; width <= held->lanes; ++width) {
      cudaGraphExec_t exec = nullptr;
      if (capture_verify_graph(model, pool, *held, ignis::round_layout::uniform(width, held->window, held->extended()),
                               &exec, &held->graph_nodes[width - 1], &held->graph_copy_nodes[width - 1]) == 0) {
        held->graph_exec[width - 1] = exec;
        held->graph_ready[width - 1] = true;
      } else {
        held->graph_exec[width - 1] = nullptr;
        held->graph_ready[width - 1] = false;
      }
    }
  }
  // GitHub #676: the DFlash2 drafter's append after a deferring round's
  // commit, one graph per width, under the same never-refuse-service rule: a
  // width whose capture failed appends eagerly. GitHub #687: each shape's.
  for (IgnisVerifyRound *const held : shapes) {
    if (held == nullptr || !held->defer || held->staging == nullptr) {
      continue;
    }
    for (uint32_t width = 1; width <= held->lanes; ++width) {
      std::string error;
      if (ignis::deferred_verify::capture_append(model, pool, *held,
                                                 ignis::round_layout::uniform(width, held->window, held->extended()),
                                                 &held->append_exec[width - 1], &held->append_nodes[width - 1],
                                                 &error) != 0) {
        set_error("ignis_decode_graph_capture: the drafter's append at " + std::to_string(held->window) +
                  " drafts, width " + std::to_string(width) + ": " + error);
      }
    }
  }
  // GitHub #713: and the mixed rounds' graphs where IGNIS_MIXED_ROUNDS asks for them.
  if (!capture_mixed_graphs(model, pool)) {
    return -1;
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

extern "C" int32_t ignis_decode_graph_fail_mixed_capture(struct ignis_model *model, uint32_t extended_lanes,
                                                         uint32_t base_lanes) {
  if (model == nullptr || model->flash_next) {
    set_error("ignis_decode_graph_fail_mixed_capture: a 27B model's mixed graphs");
    return -1;
  }
  model->mixed_capture_fault =
      ignis::round_layout::mix_index(ignis::round_layout::layout_of(extended_lanes, 0, base_lanes, 0));
  return 0;
}
