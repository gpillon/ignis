// ignis kernel leaf - GitHub #620 (spec #618): the MTP head's proposal readout,
// a test-only measurement seam (kernel/include/ignis_mtp_readout.h) -- OURS.

#include "ignis_mtp_readout.h"

#include "ignis_kv_capture.h"

#include "ignis_seq_internal.h"
#include "model_internal.h"
#include "mtp_head.h"
#include "trellis_body.h"

#include "ninfer/ops/argmax.h"
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

thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

float bf16_to_f32(std::uint16_t bits) {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0F;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

void check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
  }
}

// Why `model` serves no readout of the head, or nullptr.
const char *head_refusal(const ignis_model *model) {
  if (model == nullptr) {
    return "null model";
  }
  if (model->flash_next || model->drafter != IgnisDrafter::kMtp) {
    return "the model was not loaded with the 27B's MTP head";
  }
  return nullptr;
}

// `hidden` (device BF16 [hidden, 1]) through the load's output head into
// `scores` (host F32 [vocab]), and its argmax into `*draft` when asked.
void score(ignis_model &model, const ninfer::Tensor &hidden, float *scores, std::int32_t *draft) {
  const auto vocab = static_cast<std::int32_t>(model.vocab);
  ninfer::Tensor logits = model.scratch->alloc(ninfer::DType::BF16, {vocab, 1, 1, 1});
  ignis::trellis_body::output_head(model, hidden, logits, model.stream);
  ninfer::Tensor best = model.scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
  ninfer::ops::argmax(logits, best, vocab, model.stream);
  std::vector<std::uint16_t> bits(static_cast<std::size_t>(vocab));
  std::int32_t argmax = -1;
  check(cudaMemcpyAsync(bits.data(), logits.data, bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                        model.stream),
        "cudaMemcpyAsync(scores)");
  check(cudaMemcpyAsync(&argmax, best.data, sizeof(argmax), cudaMemcpyDeviceToHost, model.stream),
        "cudaMemcpyAsync(draft)");
  check(cudaStreamSynchronize(model.stream), "cudaStreamSynchronize");
  if (scores != nullptr) {
    for (std::size_t v = 0; v < bits.size(); ++v) {
      scores[v] = bf16_to_f32(bits[v]);
    }
  }
  if (draft != nullptr) {
    *draft = argmax;
  }
}

}  // namespace

extern "C" int32_t ignis_mtp_readout(struct ignis_model *model, struct ignis_seq_pool *pool, struct ignis_seq *seq,
                                     struct ignis_mtp_readout *readout) {
  if (pool == nullptr || seq == nullptr || readout == nullptr) {
    set_error("ignis_mtp_readout: null argument");
    return -1;
  }
  if (const char *refusal = head_refusal(model)) {
    set_error(std::string("ignis_mtp_readout: ") + refusal);
    return -1;
  }
  if (readout->size != sizeof(struct ignis_mtp_readout)) {
    set_error("ignis_mtp_readout: unrecognized ignis_mtp_readout size " + std::to_string(readout->size));
    return -1;
  }
  if (pool->speculative_backend != model->speculative_backend || !pool->has_mtp_head()) {
    set_error("ignis_mtp_readout: the sequence pool was not built for the MTP head");
    return -1;
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    set_error("ignis_mtp_readout: the sequence was not drawn from this pool");
    return -1;
  }
  if (const char *refusal = ignis_seq_restore_refusal(*seq)) {
    set_error(std::string("ignis_mtp_readout: ") + refusal);
    return -1;
  }
  if (seq->position == 0 || seq->pending_token < 0) {
    set_error("ignis_mtp_readout: the sequence has no frontier to draft at (no position or no pending token)");
    return -1;
  }
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const std::size_t column_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    if (readout->out_continuation != nullptr) {
      check(cudaMemcpyAsync(readout->out_continuation, pool->mtp_continuation_of(seq->slot), column_bytes,
                            cudaMemcpyDeviceToHost, model->stream),
            "cudaMemcpyAsync(continuation hidden)");
    }
    ninfer::Tensor head_hidden = model->scratch->alloc(ninfer::DType::BF16, {hidden, 1, 1, 1});
    std::string error;
    if (ignis::mtp::frontier_row(*model, *pool, *seq, seq->pending_token, *model->scratch, head_hidden, &error) !=
        0) {
      set_error("ignis_mtp_readout: " + error);
      return -1;
    }
    if (readout->out_head_hidden != nullptr) {
      check(cudaMemcpyAsync(readout->out_head_hidden, head_hidden.data, column_bytes, cudaMemcpyDeviceToHost,
                            model->stream),
            "cudaMemcpyAsync(head hidden)");
    }
    score(*model, head_hidden, readout->out_scores, &readout->out_draft);
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_mtp_readout: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_mtp_readout_scores(struct ignis_model *model, const uint16_t *hidden, float *out_scores) {
  if (hidden == nullptr || out_scores == nullptr) {
    set_error("ignis_mtp_readout_scores: null argument");
    return -1;
  }
  if (const char *refusal = head_refusal(model)) {
    set_error(std::string("ignis_mtp_readout_scores: ") + refusal);
    return -1;
  }
  const auto width = static_cast<std::int32_t>(model->hidden);
  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    ninfer::Tensor column = model->scratch->alloc(ninfer::DType::BF16, {width, 1, 1, 1});
    check(cudaMemcpyAsync(column.data, hidden, static_cast<std::size_t>(width) * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice, model->stream),
          "cudaMemcpyAsync(hidden)");
    score(*model, column, out_scores, nullptr);
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_mtp_readout_scores: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_mtp_readout_state(const struct ignis_seq_pool *pool, const struct ignis_seq *seq,
                                           int32_t first_position, int32_t count, uint16_t *out_keys,
                                           uint16_t *out_values) {
  if (pool == nullptr || seq == nullptr || out_keys == nullptr || out_values == nullptr) {
    set_error("ignis_mtp_readout_state: null argument");
    return -1;
  }
  if (!pool->has_mtp_head()) {
    set_error("ignis_mtp_readout_state: the sequence pool was not built for the MTP head");
    return -1;
  }
  if (count <= 0) {
    set_error("ignis_mtp_readout_state: count must be positive");
    return -1;
  }
  const std::int32_t heads = pool->kv_num_kv_heads;
  const std::int32_t dim = pool->kv_head_dim;
  std::vector<std::uint16_t> rows(static_cast<std::size_t>(count) * static_cast<std::size_t>(dim));
  for (std::int32_t role = 0; role < 2; ++role) {
    std::uint16_t *out = role == 0 ? out_keys : out_values;
    for (std::int32_t head = 0; head < heads; ++head) {
      if (ignis_kv_capture_rows(pool, seq, pool->mtp_head_layer(), role, head, first_position, count, dim,
                                rows.data()) != 0) {
        set_error(std::string("ignis_mtp_readout_state: ") + ignis_kv_capture_last_error());
        return -1;
      }
      for (std::int32_t i = 0; i < count; ++i) {
        std::memcpy(out + (static_cast<std::size_t>(i) * heads + head) * dim,
                    rows.data() + static_cast<std::size_t>(i) * dim, static_cast<std::size_t>(dim) * sizeof(std::uint16_t));
      }
    }
  }
  return 0;
}

static_assert(IGNIS_MTP_READOUT_RING_WORDS == kIgnisHqRingWords, "the readout's ring words are the pool's");

extern "C" int32_t ignis_mtp_readout_ring(const struct ignis_seq_pool *pool, const struct ignis_seq *seq,
                                          uint32_t *out_target_words, uint32_t *out_head_words) {
  if (pool == nullptr || seq == nullptr || out_target_words == nullptr || out_head_words == nullptr) {
    set_error("ignis_mtp_readout_ring: null argument");
    return -1;
  }
  if (!pool->has_mtp_head()) {
    set_error("ignis_mtp_readout_ring: the sequence pool was not built for the MTP head");
    return -1;
  }
  if (!pool->has_hq_residual()) {
    set_error("ignis_mtp_readout_ring: the sequence pool holds no hq-e8-2b residual window");
    return -1;
  }
  const std::size_t bytes = static_cast<std::size_t>(kIgnisHqRingWords) * sizeof(std::uint32_t);
  cudaError_t err = cudaMemcpy(out_target_words, pool->hq_ring_words(seq->slot), bytes, cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    err = cudaMemcpy(out_head_words, pool->hq_head_ring_words(seq->slot), bytes, cudaMemcpyDeviceToHost);
  }
  if (err != cudaSuccess) {
    set_error(std::string("ignis_mtp_readout_ring: cudaMemcpy failed: ") + cudaGetErrorString(err));
    return -1;
  }
  return 0;
}

extern "C" int32_t ignis_mtp_readout_round_hidden(struct ignis_model *model, uint32_t lanes, uint16_t *out_hidden) {
  if (out_hidden == nullptr) {
    set_error("ignis_mtp_readout_round_hidden: null argument");
    return -1;
  }
  if (const char *refusal = head_refusal(model)) {
    set_error(std::string("ignis_mtp_readout_round_hidden: ") + refusal);
    return -1;
  }
  if (model->verify == nullptr) {
    set_error("ignis_mtp_readout_round_hidden: the model was loaded without a draft window");
    return -1;
  }
  if (lanes == 0 || lanes > model->decode_lanes) {
    set_error("ignis_mtp_readout_round_hidden: lanes " + std::to_string(lanes) + " is not in 1.." +
              std::to_string(model->decode_lanes) + ", the load's decode lanes");
    return -1;
  }
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto columns = static_cast<std::int32_t>(lanes * (model->verify->window + 1));
  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    // The commit's own call (kernel/src/mtp_head.cu `commit_rows`): the
    // final norm over the round's kept residual, at the round's width.
    const ninfer::Tensor residual(model->verify->hidden->p, ninfer::DType::BF16, {hidden, columns, 1, 1});
    ninfer::Tensor normalized = model->scratch->alloc(ninfer::DType::BF16, {hidden, columns, 1, 1});
    const ninfer::Tensor gain(const_cast<void *>(model->final_norm.qdata), ninfer::DType::BF16, {hidden, 1, 1, 1});
    ninfer::ops::rmsnorm(residual, gain, model->rms_norm_eps, /*unit_offset=*/true, normalized, model->stream);
    check(cudaMemcpyAsync(out_hidden, normalized.data,
                          static_cast<std::size_t>(hidden) * static_cast<std::size_t>(columns) * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, model->stream),
          "cudaMemcpyAsync(round hidden)");
    check(cudaStreamSynchronize(model->stream), "cudaStreamSynchronize");
    return 0;
  } catch (const std::exception &e) {
    set_error(std::string("ignis_mtp_readout_round_hidden: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_mtp_readout_chain(struct ignis_model *model, struct ignis_seq_pool *pool,
                                           struct ignis_seq *seq, const struct ignis_mtp_chain_readout *readout) {
  if (pool == nullptr || seq == nullptr || readout == nullptr) {
    set_error("ignis_mtp_readout_chain: null argument");
    return -1;
  }
  if (const char *refusal = head_refusal(model)) {
    set_error(std::string("ignis_mtp_readout_chain: ") + refusal);
    return -1;
  }
  if (readout->size != sizeof(struct ignis_mtp_chain_readout)) {
    set_error("ignis_mtp_readout_chain: unrecognized ignis_mtp_chain_readout size " + std::to_string(readout->size));
    return -1;
  }
  if (readout->out_drafts == nullptr || readout->out_scores == nullptr) {
    set_error("ignis_mtp_readout_chain: the drafts and scores outputs are required");
    return -1;
  }
  if (model->verify == nullptr) {
    set_error("ignis_mtp_readout_chain: the model was loaded without a draft window");
    return -1;
  }
  if (pool->speculative_backend != model->speculative_backend || !pool->has_mtp_head()) {
    set_error("ignis_mtp_readout_chain: the sequence pool was not built for the MTP head");
    return -1;
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    set_error("ignis_mtp_readout_chain: the sequence was not drawn from this pool");
    return -1;
  }
  if (const char *refusal = ignis_seq_restore_refusal(*seq)) {
    set_error(std::string("ignis_mtp_readout_chain: ") + refusal);
    return -1;
  }
  IgnisVerifyRound &verify = *model->verify;
  const std::uint32_t k = verify.window;
  // The caller sized its buffers by its window and extent; the chain runs
  // the load's window.
  if (readout->window != k) {
    set_error("ignis_mtp_readout_chain: window " + std::to_string(readout->window) +
              " is not the load's draft window " + std::to_string(k));
    return -1;
  }
  if (readout->extent == 0 || readout->extent > k) {
    set_error("ignis_mtp_readout_chain: extent " + std::to_string(readout->extent) + " is not in 1.." +
              std::to_string(k) + ", the load's draft window");
    return -1;
  }
  if (seq->position == 0 || seq->pending_token < 0) {
    set_error("ignis_mtp_readout_chain: the sequence has no frontier to draft at (no position or no pending token)");
    return -1;
  }
  // The chain writes the head's rows up to position - 2 + extent; the round
  // at that extent writes the target's up to position + extent.
  if (seq->position + readout->extent >= ignis_seq_token_capacity(*seq)) {
    set_error("ignis_mtp_readout_chain: extent " + std::to_string(readout->extent) +
              " runs past the sequence's capacity at position " + std::to_string(seq->position));
    return -1;
  }

  const std::size_t vocab = model->vocab;
  const cudaStream_t stream = model->stream;
  // Lane 0 of a width-1 round, staged where the chain reads it. The host
  // values outlive the synchronize below.
  const std::int32_t anchor = seq->pending_token;
  const auto base = static_cast<std::int32_t>(seq->position);
  const auto extent = static_cast<std::int32_t>(readout->extent);
  const std::int32_t slot = seq->slot;
  ninfer::DeviceArena::Scope scope = model->scratch->scope();
  try {
    const struct {
      ninfer::DeviceBuffer *buffer;
      const std::int32_t *value;
      const char *what;
    } staged[] = {
        {verify.anchors.get(), &anchor, "anchor"},
        {verify.base_positions.get(), &base, "base position"},
        {verify.extents.get(), &extent, "extent"},
        {model->decode_graph_slots.get(), &slot, "slot"},
    };
    for (const auto &stage : staged) {
      check(cudaMemcpyAsync(stage.buffer->p, stage.value, sizeof(std::int32_t), cudaMemcpyHostToDevice, stream),
            (std::string("cudaMemcpyAsync(") + stage.what + ")").c_str());
    }
    // Every step's scores, the whole window's (the chain runs every step).
    const ninfer::DeviceSpan scores = model->scratch->alloc_bytes(k * vocab * sizeof(std::uint16_t));
    // Under hq-e8-2b, the slot's head residual window and ring words as the
    // next step would read them.
    const bool hq = pool->has_hq_residual();
    const std::size_t plane_bytes = hq ? pool->hq_residual_plane_bytes() : 0;
    const std::size_t ring_bytes = static_cast<std::size_t>(kIgnisHqRingWords) * sizeof(std::uint32_t);
    void *const head_k = hq ? pool->hq_residual_plane(false, pool->mtp_head_layer(), slot) : nullptr;
    void *const head_v = hq ? pool->hq_residual_plane(true, pool->mtp_head_layer(), slot) : nullptr;
    void *const head_ring = hq ? static_cast<void *>(pool->hq_head_ring_words(slot)) : nullptr;
    ninfer::DeviceSpan saved{};
    const auto saved_at = [&](std::size_t offset) { return static_cast<unsigned char *>(saved.data) + offset; };
    if (hq) {
      saved = model->scratch->alloc_bytes(2 * plane_bytes + ring_bytes);
      check(cudaMemcpyAsync(saved_at(0), head_k, plane_bytes, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync(save the head's residual keys)");
      check(cudaMemcpyAsync(saved_at(plane_bytes), head_v, plane_bytes, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync(save the head's residual values)");
      check(cudaMemcpyAsync(saved_at(2 * plane_bytes), head_ring, ring_bytes, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync(save the head's ring words)");
    }
    std::string error;
    const int32_t rc = ignis::mtp::propose(*model, *pool, 1, &error, static_cast<std::uint16_t *>(scores.data));
    // Put back even after a chain that failed part way: it may have appended.
    if (hq) {
      check(cudaMemcpyAsync(head_k, saved_at(0), plane_bytes, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync(restore the head's residual keys)");
      check(cudaMemcpyAsync(head_v, saved_at(plane_bytes), plane_bytes, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync(restore the head's residual values)");
      check(cudaMemcpyAsync(head_ring, saved_at(2 * plane_bytes), ring_bytes, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync(restore the head's ring words)");
    }
    if (rc != 0) {
      (void)cudaStreamSynchronize(stream);
      set_error("ignis_mtp_readout_chain: " + error);
      return -1;
    }
    // Lane 0's drafts lead `verify.drafts` (lane-major), its scores the
    // readout's blocks (step-major at width 1).
    check(cudaMemcpyAsync(readout->out_drafts, verify.drafts->p, readout->extent * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync(drafts)");
    check(cudaMemcpyAsync(readout->out_scores, scores.data, readout->extent * vocab * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync(scores)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return 0;
  } catch (const std::exception &e) {
    (void)cudaStreamSynchronize(stream);
    set_error(std::string("ignis_mtp_readout_chain: ") + e.what());
    return -1;
  }
}

extern "C" const char *ignis_mtp_readout_last_error(void) {
  return g_last_error.c_str();
}
