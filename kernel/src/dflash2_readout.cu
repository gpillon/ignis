// ignis kernel leaf - GitHub #267: the DFlash2 proposal readout, a
// test-only measurement seam -- see kernel/include/ignis_dflash2_readout.h
// for the contract.

#include "ignis_dflash2_readout.h"

#include "dflash2_drafter.h"
#include "ignis_seq_internal.h"
#include "ignis_step.h"
#include "model_internal.h"
#include "step_internal.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <string>

namespace {

// The last error message on this thread (ignis_dflash2_readout_last_error):
// this seam's own channel, like every other ABI surface of the leaf.
thread_local std::string g_last_error;

int32_t fail(std::string message) {
  g_last_error = std::move(message);
  return -1;
}

static_assert(IGNIS_DFLASH2_READOUT_TOP_K == kDflash2SelectorTopK,
              "the readout's candidate rows are the selector's top-k");

} // namespace

extern "C" int32_t ignis_dflash2_readout(struct ignis_model *model, struct ignis_seq_pool *pool,
                                         const struct ignis_seq *seq,
                                         const struct ignis_dflash2_readout *readout) {
  if (model == nullptr || pool == nullptr || seq == nullptr || readout == nullptr) {
    return fail("ignis_dflash2_readout: null argument");
  }
  if (readout->size != sizeof(struct ignis_dflash2_readout)) {
    return fail("ignis_dflash2_readout: unrecognized ignis_dflash2_readout size " +
                std::to_string(readout->size));
  }
  if (readout->out_drafts == nullptr || readout->out_candidate_ids == nullptr ||
      readout->out_candidate_logits == nullptr || readout->out_scores == nullptr) {
    return fail("ignis_dflash2_readout: the drafts, candidates, candidate logits and scores "
                "outputs are required");
  }
  if (model->verify == nullptr || model->verify->drafter_scratch == nullptr) {
    return fail("ignis_dflash2_readout: this model was loaded without the DFlash2 drafter");
  }
  if (pool->speculative_backend != model->speculative_backend || !pool->has_dflash2()) {
    return fail("ignis_dflash2_readout: the sequence pool was not built for this model's drafter");
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    return fail("ignis_dflash2_readout: the sequence was not drawn from this pool");
  }
  IgnisVerifyRound &verify = *model->verify;
  const std::uint32_t k = verify.window;
  // The caller sized every output by its window; the copies below are the
  // load's, so another window would over- or under-run them.
  if (readout->window != k) {
    return fail("ignis_dflash2_readout: window " + std::to_string(readout->window) +
                " is not the load's draft window " + std::to_string(k));
  }
  if (readout->extent > k) {
    return fail("ignis_dflash2_readout: extent " + std::to_string(readout->extent) +
                " is past the load's window of " + std::to_string(k));
  }
  const std::uint64_t capacity = ignis_seq_token_capacity(*seq);
  if (seq->position + readout->extent >= capacity) {
    return fail("ignis_dflash2_readout: extent " + std::to_string(readout->extent) +
                " runs past the sequence's capacity at position " + std::to_string(seq->position));
  }
  if (seq->dflash2_pending || seq->dflash2_position != seq->position) {
    return fail("ignis_dflash2_readout: the sequence's window stands at " +
                std::to_string(seq->dflash2_position) + ", not at its frontier " +
                std::to_string(seq->position) +
                (seq->dflash2_pending ? " (it carries an extent-0 round's anchor)" : ""));
  }
  if (readout->out_drafter_logits != nullptr && model->proposal_token_ids != nullptr) {
    return fail("ignis_dflash2_readout: the drafter logits are read over the full head; this "
                "load proposes through the shortlist head");
  }
  if (readout->sampling != nullptr && !ignis::step::sampling_size_ok(*readout->sampling)) {
    return fail("ignis_dflash2_readout: unrecognized ignis_sampling_params size " +
                std::to_string(readout->sampling->size));
  }

  // Lane 0 of a width-1 round, staged where the forward reads it. Host
  // values outlive the synchronize below.
  const std::int32_t anchor        = seq->pending_token;
  if (anchor < 0) {
    return fail("ignis_dflash2_readout: the sequence has no pending token (it was not prefilled)");
  }
  const auto base                  = static_cast<std::int32_t>(seq->position);
  const auto valid_columns         = static_cast<std::int32_t>(readout->extent + 1);
  const std::int32_t slot          = seq->slot;
  const cudaStream_t stream        = model->stream;
  const struct {
    ninfer::DeviceBuffer *buffer;
    const std::int32_t *value;
    const char *what;
  } staged[] = {
      {verify.anchors.get(), &anchor, "anchor"},
      {verify.base_positions.get(), &base, "base position"},
      {verify.valid_columns.get(), &valid_columns, "valid columns"},
      {model->decode_graph_slots.get(), &slot, "slot"},
  };
  for (const auto &stage : staged) {
    const cudaError_t err =
        cudaMemcpyAsync(stage.buffer->p, stage.value, sizeof(std::int32_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      return fail(std::string("ignis_dflash2_readout: cudaMemcpyAsync(") + stage.what +
                  ") failed: " + cudaGetErrorString(err));
    }
  }
  // GitHub #577: under the sampled draft the lane's draw reads its sampling
  // config -- its temperature and seed -- at lane 0, as the round stages it;
  // a NULL `sampling` is a greedy lane, which draws nothing. No penalty row:
  // the readout runs no accept.
  const bool draws = verify.sampled_draft && readout->sampling != nullptr &&
                     readout->sampling->greedy == 0 && readout->sampling->temperature > 0.0f;
  ignis_sampling_params greedy{};
  greedy.size = sizeof(greedy);
  greedy.greedy = 1;
  const ninfer::ops::SamplingConfig config =
      ignis::step::to_sampling_config(readout->sampling != nullptr ? *readout->sampling : greedy, nullptr);
  if (verify.sampled_draft) {
    const cudaError_t err = cudaMemcpyAsync(model->sampling_decode_configs->p, &config, sizeof(config),
                                            cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      return fail(std::string("ignis_dflash2_readout: cudaMemcpyAsync(sampling config) failed: ") +
                  cudaGetErrorString(err));
    }
  }

  try {
    const Dflash2ProposalReadout sink{readout->out_candidate_ids, readout->out_candidate_logits,
                                      readout->out_scores, readout->out_drafter_logits,
                                      draws ? readout->out_q : nullptr};
    ignis_dflash2_propose(model, pool, 1, &sink);
  } catch (const std::exception &e) {
    (void)cudaStreamSynchronize(stream);
    return fail(std::string("ignis_dflash2_readout: ") + e.what());
  }
  cudaError_t err = cudaMemcpyAsync(readout->out_drafts, verify.drafts->p, k * sizeof(std::int32_t),
                                    cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    (void)cudaStreamSynchronize(stream);
    return fail(std::string("ignis_dflash2_readout: cudaMemcpyAsync(drafts) failed: ") +
                cudaGetErrorString(err));
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return fail(std::string("ignis_dflash2_readout: cudaStreamSynchronize failed: ") +
                cudaGetErrorString(err));
  }
  // GitHub #577: where nothing was drawn, q is one-hot on each walk draft,
  // the fixed draft's distribution.
  if (readout->out_q != nullptr && !draws) {
    constexpr std::uint32_t top_k = IGNIS_DFLASH2_READOUT_TOP_K;
    for (std::uint32_t s = 0; s < k; ++s) {
      float *row = readout->out_q + s * top_k;
      for (std::uint32_t i = 0; i < top_k; ++i) {
        row[i] = readout->out_candidate_ids[s * top_k + i] == readout->out_drafts[s] ? 1.0f : 0.0f;
      }
    }
  }
  return 0;
}

extern "C" int32_t ignis_verify_readout_logits(struct ignis_model *model, uint32_t lanes, uint32_t window,
                                               uint16_t *out_logits) {
  if (model == nullptr || out_logits == nullptr) {
    return fail("ignis_verify_readout_logits: null argument");
  }
  if (model->verify == nullptr) {
    return fail("ignis_verify_readout_logits: this model was loaded without a draft window");
  }
  // The caller sized `out_logits` by its window; the copy below is the
  // load's `k + 1` rows a lane.
  if (window != model->verify->window) {
    return fail("ignis_verify_readout_logits: window " + std::to_string(window) +
                " is not the load's draft window " + std::to_string(model->verify->window));
  }
  // GitHub #336: the round's buffers, its logits included, are sized for the
  // load's decode lanes, not for IGNIS_DECODE_MAX_BATCH.
  if (lanes == 0 || lanes > model->decode_lanes) {
    return fail("ignis_verify_readout_logits: lanes " + std::to_string(lanes) +
                " is not in 1.." + std::to_string(model->decode_lanes) + ", the load's decode lanes");
  }
  // `[vocab, k+1, B]` with the vocabulary fastest: the first `lanes` lanes
  // are one contiguous prefix.
  const std::size_t bytes = static_cast<std::size_t>(model->vocab) * (model->verify->window + 1) *
                            lanes * sizeof(std::uint16_t);
  cudaError_t err = cudaMemcpyAsync(out_logits, model->verify->logits->p, bytes,
                                    cudaMemcpyDeviceToHost, model->stream);
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(model->stream);
  }
  if (err != cudaSuccess) {
    return fail(std::string("ignis_verify_readout_logits: cudaMemcpyAsync failed: ") +
                cudaGetErrorString(err));
  }
  return 0;
}

extern "C" const char *ignis_dflash2_readout_last_error(void) {
  return g_last_error.c_str();
}
