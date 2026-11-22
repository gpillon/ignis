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

#include "attention_readout.h"
#include "deferred_verify.h"
#include "dflash2_drafter.h"
#include "ignis_gdn_layer.h"
#include "ignis_gqa_layer.h"
#include "ignis_gqa_workspace.h"
#include "ignis_hq_ring.h"
#include "ignis_reasoning_redirect.h"
#include "ignis_seq_internal.h"
#include "layer_internal.h"
#include "model_internal.h"
#include "round_shape.h"
#include "mtp_head.h"
#include "permitted_tokens.h"
#include "step_internal.h"
#include "teacher_forcing.h"
#include "trellis_body.h"
#include "verify_return.h"
#include "flash_next/program.h"

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
#include <array>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// The last error message on this thread (ignis_step_last_error) -- a
// separate channel from ignis_model_last_error (each ABI surface owns its
// own, model.cu's convention).
thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

// GitHub #674: the round timing, the measurement switch IGNIS_ROUND_TIMING=1 (read at each graph
// capture, kernel/src/decode_graph.cu; off, each call below is one test of a bool). Around every
// replayed round's graph, an event recorded on the model's stream just before the launch and one
// just after it: the GPU's clock, with no tracer, which under Nsight Systems adds ~400 us to every
// graph launch on the RTX 5090's host. `before` is the launch's side; GitHub #689: `shape` the
// round's shape, 1 for an extended round (GitHub #713: `kMixedBlock` for a mixed one).
void round_timing_mark(ignis_model *model, RoundTiming &timing, bool before, std::uint8_t shape = 0) {
  if (!model->round_timing) {
    return;
  }
  const std::uint64_t at = timing.recorded % RoundTiming::kRing;
  if (cudaEventRecord(before ? timing.pre[at] : timing.post[at], model->stream) != cudaSuccess) {
    (void)cudaGetLastError();
  }
  if (before) {
    timing.shape[at] = shape;
  } else {
    ++timing.recorded;
  }
}

float round_timing_median(std::vector<float> samples) {
  std::sort(samples.begin(), samples.end());
  return samples.empty() ? 0.0f : samples[samples.size() / 2];
}

// Every timed round of `kind` whose events and the next timed round's launch have completed, read
// without waiting: its span (the event before its graph to the one after), its period (its event
// before to the next round's) and its gap (its event after to the next round's event before),
// unless the two began 50 ms or more apart (a request or a prefill between them), each in its round
// shape's block (kernel/src/round_timing.h). Each 256 periods of a shape, one line on stderr: their
// median, their mean, and the block's median span and gap, under `kind` for the base shape,
// `kind extended` for the extended one and (GitHub #713) `kind mixed` for the mixed rounds.
void round_timing_collect(ignis_model *model, RoundTiming &timing, const char *kind) {
  if (!model->round_timing) {
    return;
  }
  if (timing.recorded > timing.collected + RoundTiming::kRing) {
    timing.collected = timing.recorded - RoundTiming::kRing;  // those rounds' events were recorded over
  }
  while (timing.collected + 1 < timing.recorded) {
    const std::uint64_t round = timing.collected;
    const std::uint64_t at = round % RoundTiming::kRing;
    const std::uint64_t next = (round + 1) % RoundTiming::kRing;
    if (cudaEventQuery(timing.post[at]) != cudaSuccess || cudaEventQuery(timing.pre[next]) != cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    float span = 0.0f;
    float period = 0.0f;
    float gap = 0.0f;
    if (cudaEventElapsedTime(&span, timing.pre[at], timing.post[at]) == cudaSuccess &&
        cudaEventElapsedTime(&period, timing.pre[at], timing.pre[next]) == cudaSuccess &&
        cudaEventElapsedTime(&gap, timing.post[at], timing.pre[next]) == cudaSuccess) {
      ignis::round_timing::add(timing.blocks, timing.shape[at], span, period, gap);
    }
    (void)cudaGetLastError();
    ++timing.collected;
    for (std::size_t shape = 0; shape < timing.blocks.size(); ++shape) {
      ignis::round_timing::Block &block = timing.blocks[shape];
      if (block.period.size() < 256) {
        continue;
      }
      double mean = 0.0;
      for (const float sample : block.period) {
        mean += sample;
      }
      mean /= static_cast<double>(block.period.size());
      std::fprintf(stderr,
                   "ignis leaf: round timing %s%s n %zu: period median %.1f us mean %.1f, span median %.1f, gap "
                   "median %.1f\n",
                   kind,
                   shape == 0                                 ? ""
                   : shape == ignis::round_timing::kMixedBlock ? " mixed"
                                                               : " extended",
                   block.period.size(), round_timing_median(block.period), mean,
                   round_timing_median(block.span), round_timing_median(block.gap));
      std::fflush(stderr);
      block = {};
    }
  }
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

// GitHub #586 (spec #287): the device row of `column` in the scores a repair
// reads -- the model's decode logits, `[vocab, decode_lanes]` BF16.
void *kept_row(ignis_model *model, std::int32_t column) {
  return static_cast<std::uint8_t *>(model->sampling_decode_logits->p) +
         static_cast<std::size_t>(column) * static_cast<std::size_t>(model->vocab) *
             sizeof(std::uint16_t);
}

// GitHub #586: `column` has just been written; its new stamp.
std::uint64_t stamp_kept_column(ignis_model *model, std::int32_t column) {
  const std::uint64_t epoch = ++model->kept_epoch;
  model->kept_column_epochs[static_cast<std::size_t>(column)] = epoch;
  return epoch;
}

// GitHub #590: whether the scores `kept` names are still the ones its draw
// was made from -- a decode-logits column no later round or prefill draw has
// written, or the verify logits of the last verify round, the one that cut
// the lane's commit. GitHub #687: a load's two round shapes write one logits
// buffer (the base shape's, which the extended one holds too), so the last
// verify round is the latest of either shape's (`ignis_latest_verify_shape`);
// a record's column is the buffer's own, at the stride of the round that
// wrote it. A verify round's record carries its round's stamp, never 0.
bool kept_scores_current(const ignis_model *model, const ignis_kept_scores &kept) {
  if (kept.source == IgnisKeptSource::kVerifyLogits) {
    const IgnisVerifyRound *latest = ignis_latest_verify_shape(*model);
    if (latest == nullptr) {
      return false;
    }
    const std::uint64_t columns =
        latest->logits->bytes / (static_cast<std::uint64_t>(model->vocab) * sizeof(std::uint16_t));
    return static_cast<std::uint64_t>(kept.column) < columns && latest->kept_epoch == kept.epoch;
  }
  return static_cast<std::uint32_t>(kept.column) < model->decode_lanes &&
         model->kept_column_epochs[static_cast<std::size_t>(kept.column)] == kept.epoch;
}

// GitHub #590: the device row `kept` names, in whichever buffer holds it --
// the verify logits are `[vocab, k + 1, decode_lanes]` BF16, one row a column.
const void *kept_scores_row(ignis_model *model, const ignis_kept_scores &kept) {
  if (kept.source == IgnisKeptSource::kVerifyLogits) {
    return static_cast<const std::uint8_t *>(model->base_shape->logits->p) +
           static_cast<std::size_t>(kept.column) * static_cast<std::size_t>(model->vocab) *
               sizeof(std::uint16_t);
  }
  return kept_row(model, kept.column);
}

// GitHub #586: whether a draw under `sampling` added one to its token's
// penalty count -- the vendored sampler counts in its temperature branch
// only, and the program always hands it the sequence's row.
bool draw_counts(const ignis_sampling_params &sampling) {
  return to_sampling_config(sampling, nullptr).temperature > 0.0f;
}

// GitHub #586: a prefill's draw keeps the row it came from (`logits`, the
// chunk's scratch, about to go) for a repair, in the decode logits' column of
// the sequence's slot -- free between the rounds that use it as a lane column,
// and one 485 KiB device copy on the stream before the chunk's synchronize.
// Only for a request held to a grammar (`keeps_scores`, GitHub #287): every
// other prefill makes no copy. A
// slot past the load's lanes, which no serving pool has, keeps nothing, and a
// repair of its draw is refused. `*out_kept` says what was kept.
int32_t keep_draw_scores(ignis_model *model, const ignis_seq *seq, const void *logits,
                         std::int32_t position, const ignis_sampling_params &sampling,
                         ignis_kept_scores *out_kept) {
  *out_kept = ignis_kept_scores{};
  if (seq->slot < 0 || static_cast<std::uint32_t>(seq->slot) >= model->decode_lanes) {
    return 0;
  }
  const cudaError_t err =
      cudaMemcpyAsync(kept_row(model, seq->slot), logits,
                      static_cast<std::size_t>(model->vocab) * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToDevice, model->stream);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_program_prefill: cudaMemcpyAsync(kept scores) failed: ") +
              cudaGetErrorString(err));
    return -1;
  }
  out_kept->column = seq->slot;
  out_kept->epoch = stamp_kept_column(model, seq->slot);
  out_kept->position = position;
  out_kept->counted = draw_counts(sampling);
  return 0;
}

// GitHub #315 (ADR 0048): a lane whose reasoning redirect is armed reads its
// stop ids wherever its successor is assigned, not only in a verify round, so
// a count without its ids is refused there too.
bool redirect_inputs_ok(const ignis_sampling_params &sampling) {
  return sampling.reasoning_close_id < 0 || sampling.stop_id_count == 0 || sampling.stop_ids != nullptr;
}

// Occurrences moved in penalty-count rows: each entry a count's device address
// and what to add to it, clamped at 0, in one read and one write per distinct
// count, each side synchronized. A verify round takes back the tokens it
// licensed past its cut; a redirect (GitHub #315) moves the occurrence its
// sampler counted from the stop id it drew to the `</think>` that replaced it.
// Either way the row a snapshot or a prefix clone carries matches the text.
cudaError_t adjust_penalty_counts(const std::vector<std::pair<std::int32_t *, std::int32_t>> &moves,
                                  cudaStream_t stream) {
  std::vector<std::pair<std::int32_t *, std::int32_t>> merged;
  for (const auto &[count, delta] : moves) {
    const auto same = std::find_if(merged.begin(), merged.end(),
                                   [&](const auto &entry) { return entry.first == count; });
    if (same == merged.end()) {
      merged.emplace_back(count, delta);
    } else {
      same->second += delta;
    }
  }
  std::vector<std::int32_t> values(merged.size(), 0);
  cudaError_t err = cudaSuccess;
  for (std::size_t r = 0; r < merged.size() && err == cudaSuccess; ++r) {
    err = cudaMemcpyAsync(&values[r], merged[r].first, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                          stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  for (std::size_t r = 0; r < merged.size(); ++r) {
    values[r] = std::max(values[r] + merged[r].second, 0);
  }
  for (std::size_t r = 0; r < merged.size() && err == cudaSuccess; ++r) {
    err = cudaMemcpyAsync(merged[r].first, &values[r], sizeof(std::int32_t), cudaMemcpyHostToDevice,
                          stream);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  return err;
}

// GitHub #315: the occurrence a temperature draw counted for the stop id it
// drew, moved to the `</think>` that replaced it. A greedy draw counts
// nothing, so it moves nothing.
void redirect_penalty_move(const ninfer::ops::SamplingConfig &cfg, std::int32_t drawn,
                           std::int32_t close_id,
                           std::vector<std::pair<std::int32_t *, std::int32_t>> *moves) {
  if (!(cfg.temperature > 0.0f) || cfg.token_counts == nullptr) {
    return;
  }
  moves->emplace_back(cfg.token_counts + drawn, -1);
  moves->emplace_back(cfg.token_counts + close_id, 1);
}

// GitHub #315 (ADR 0048): a span's draw -- the sequence's pending token, once
// the span's device work is confirmed -- redirected when it is a stop id drawn
// with the lane's reasoning block open: the prefill's place in the one cut
// both leaves share, with nothing committed before the draw. Whether it was
// redirected goes to `*out_redirected`, or nowhere. Returns 0, or -1 with the
// error set.
int32_t redirect_span_draw(ignis_seq_pool *pool, ignis_seq *seq, const ignis_sampling_params &sampling,
                           cudaStream_t stream, int32_t *out_redirected) {
  const IgnisRoundCut cut =
      ignis_cut_and_redirect(nullptr, 0, seq->pending_token, sampling.stop_ids, sampling.stop_id_count,
                             sampling.reasoning_close_id);
  if (cut.redirected) {
    std::vector<std::pair<std::int32_t *, std::int32_t>> moves;
    redirect_penalty_move(to_sampling_config(sampling, pool->token_counts_for(seq->slot)),
                          seq->pending_token, cut.next_pending, &moves);
    const cudaError_t err = moves.empty() ? cudaSuccess : adjust_penalty_counts(moves, stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_prefill: moving the redirected draw's penalty count failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    seq->pending_token = cut.next_pending;
    // GitHub #586: the pending token is the close, not the draw the kept
    // scores were sampled for.
    seq->kept = ignis_kept_scores{};
  }
  if (out_redirected != nullptr) {
    *out_redirected = cut.redirected ? 1 : 0;
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
    ignis::trellis_body::output_head(*model, norm_out, logits, model->stream);

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
  // GitHub #315 (ADR 0048): the degenerate program assigns no successor a
  // later round emits, so a redirect it was asked for would be dropped.
  if (sampling->reasoning_close_id >= 0) {
    set_error("ignis_step: the degenerate program cannot redirect a reasoning block's end");
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
    ignis::trellis_body::output_head(*model, normalized, logits, model->stream);
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
  if (!redirect_inputs_ok(*sampling)) {
    set_error("ignis_program: stop_ids is null with a nonzero stop_id_count");
    return false;
  }
  return true;
}

// P5-03 (GitHub #152): the drafter's weights come with the model and its
// per-sequence window with the pool, so the two must have been built for the
// same speculative backend. A drafter-bearing model over a plain pool would
// have no window to write; a plain model over a drafter pool would leave
// every window unwritten while snapshots carry it as if it were state. Either
// is refused by name before any device work. So the pool's drafter is the
// model's, and a step past this check chooses by `model->drafter` alone
// (GitHub #619).
bool drafter_matches_load(const ignis_model *model, const ignis_seq_pool *pool, const char *op) {
  if (pool->speculative_backend != model->speculative_backend) {
    set_error(std::string(op) + ": the sequence pool was built for speculative backend " +
              std::to_string(pool->speculative_backend) + ", this model was loaded with " +
              std::to_string(model->speculative_backend) +
              "; the drafter's weights and its per-sequence window come as a pair");
    return false;
  }
  // GitHub #620: the MTP head's layer is the pool's last, past the model's
  // own GQA layers, and its continuation hidden is the model's width.
  if (model->drafter == IgnisDrafter::kMtp) {
    std::int32_t gqa_layers = 0;
    for (const auto &layer : model->layers) {
      gqa_layers += layer.kind == IGNIS_LAYER_GQA ? 1 : 0;
    }
    if (pool->kv_num_layers != gqa_layers + kIgnisMtpHeadLayers ||
        model->hidden != static_cast<uint64_t>(kIgnisMtpHidden)) {
      set_error(std::string(op) + ": the sequence pool stores " + std::to_string(pool->kv_num_layers) +
                " attention layers, not this model's " + std::to_string(gqa_layers) +
                " and the MTP head's one, or its continuation hidden is not the model's width");
      return false;
    }
  }
  return true;
}

// The first absolute position of a prefill span whose target features reach
// the drafter's window: the span's last kIgnisDflash2WindowTokens positions,
// or all of it when it is shorter. Anything earlier would be overwritten in
// the ring before the span ends, so it is never tapped or projected -- which
// is what bounds the drafter's TTFT cost by the window, not by the prompt.
//
// GitHub #506: and never below the caller's drafter floor
// (`ignis_prefill_options::drafter_floor`), the first position a later window
// reads. A span whose positions no retained image and no decode reads through
// the window taps nothing at all, so the drafter's cost is bounded by where
// the window is read, not by how many jobs the prompt was cut into.
std::uint64_t dflash2_tap_from(std::uint64_t span_start, std::uint64_t num_tokens,
                               std::uint64_t drafter_floor) {
  return std::max<std::uint64_t>(
      span_start + num_tokens - std::min<std::uint64_t>(num_tokens, kIgnisDflash2WindowTokens),
      drafter_floor);
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
// here is enqueued on the model's stream. `call` names the projections' route
// widths (GitHub #521): a prefill chunk's taps, or a carried anchor.
void append_dflash2_context(ignis_model *model, ignis_seq_pool *pool,
                            const ninfer::Tensor &features, const std::int32_t *host_scalars,
                            std::int32_t count, IgnisA16Call call) {
  const cudaStream_t stream = model->stream;
  ninfer::Tensor commit_count = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
  ninfer::Tensor lane         = model->scratch->alloc(ninfer::DType::I32, {1, 1, 1, 1});
  ninfer::Tensor positions    = model->scratch->alloc(ninfer::DType::I32, {count, 1, 1, 1});
  copy_i32_to_device(commit_count, host_scalars, 1, stream);
  copy_i32_to_device(lane, host_scalars + 1, 1, stream);
  copy_i32_to_device(positions, host_scalars + 2, static_cast<std::size_t>(count), stream);
  ignis_dflash2_append_context(
      model, pool, *model->scratch, features, positions, commit_count, lane,
      {static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(count)}, call);
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
  append_dflash2_context(model, pool, features, host_scalars.data(), 1, IgnisA16Call::General);
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
  // GitHub #624: the axes the position before the span rotated at, or null.
  const int32_t *previous = nullptr; // [3]
};

// GitHub #260 (ADR 0038): an attention readout armed by the entry point for
// the span's last chunk, read off its options.
struct ArmedAttentionReadout {
  int32_t gqa_ordinal = -1;
  int32_t query_head = 0;
  int64_t key_begin = 0;
  int64_t key_count = 0;
  float *host_scores = nullptr; // [key_count]
  int32_t *host_read = nullptr; // 1 read, 0 not
  // GitHub #263 (ADR 0039): the head set, parallel arrays of `set_count`
  // heads, the keys their argmax skips, and where each head's key goes.
  uint32_t set_count = 0;
  const int32_t *set_gqa_ordinals = nullptr;
  const int32_t *set_query_heads = nullptr;
  uint32_t excluded_count = 0;
  const int32_t *excluded = nullptr;
  int32_t *host_set_argmax = nullptr; // [set_count]
  // GitHub #264 (ADR 0040): each head's score at its own argmax, and the four
  // around it -- the image grid's columns say which of those exist.
  uint32_t grid_cols = 0;
  float *host_set_peak = nullptr;       // [set_count]
  float *host_set_neighbours = nullptr; // [4 * set_count], or null with rows
  // GitHub #275 (ADR 0041): every head's whole row over a text span, or null.
  float *host_set_rows = nullptr; // [set_count * key_count]
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
  if (!model->vision_pool.present()) {
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
//
// GitHub #586: with `compute_output`, `*out_kept` (when non-null) says where
// the draw's scores are kept for a repair -- the caller makes it the
// sequence's only where it makes the draw its pending token.
int32_t run_program_chunk(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                          const int32_t *token_ids, uint64_t num_tokens, uint64_t chunk_offset,
                          uint64_t dflash2_tap_from, bool compute_output,
                          const ignis_sampling_params &sampling, int32_t *out_token_id,
                          float *out_logits, LinearPolicyMode mode,
                          const SpanMultimodal &multimodal, float *permitted_prob_out,
                          const ArmedAttentionReadout *attention_readout,
                          std::uint16_t *span_logits_out, ignis_kept_scores *out_kept,
                          std::uint16_t *span_hidden_out) {
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  const auto T = static_cast<std::int32_t>(num_tokens);
  // GitHub #619: what the chunk does for the load's drafter, chosen here once
  // (the pool's drafter is the model's, `drafter_matches_load`). DFlash2
  // writes the anchor an extent-0 round carried, taps the chunk's own columns
  // [tap_first, T) into its window and stands the window's frontier at the
  // chunk's end; the verify substrate alone does nothing here.
  const bool dflash2 = model->drafter == IgnisDrafter::kDflash2;
  const std::uint64_t chunk_start = seq->position;
  const std::int32_t tap_first =
      dflash2_tap_from > chunk_start
          ? static_cast<std::int32_t>(std::min<std::uint64_t>(dflash2_tap_from - chunk_start, num_tokens))
          : 0;
  const std::int32_t tap_count = dflash2 ? T - tap_first : 0;
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
    if (chunk_offset == 0 && dflash2) {
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
    std::int32_t media_column = 0;
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
        const auto column =
            static_cast<std::int32_t>(multimodal.first_column) +
            static_cast<std::int32_t>(first - multimodal.scatter);
        media_column = column;
        // GitHub #243: the chunk's columns are contiguous in the embedding's
        // column space but sit in its pool pages, so the scatter runs once
        // per page run. Same `ops::scatter` the single output transient
        // took, over a narrower column range and the matching slice of the
        // indices -- no kernel here reads a page table.
        const VisionEmbeddingPool &pool = model->vision_pool;
        for (std::int32_t done = 0; done < count;) {
          const std::int32_t at = column + done;
          const std::int32_t page = at / pool.page_columns;
          const std::int32_t within = at % pool.page_columns;
          const std::int32_t take = std::min(count - done, pool.page_columns - within);
          const ninfer::Tensor columns(
              pool.page_ptr(multimodal.media->pages[static_cast<std::size_t>(page)]) +
                  static_cast<std::size_t>(within) * static_cast<std::size_t>(hidden) *
                      sizeof(std::uint16_t),
              ninfer::DType::BF16, {hidden, take, 1, 1});
          ninfer::ops::scatter(columns, indices.slice(0, done, take), left, model->stream);
          done += take;
        }
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

    // GitHub #260: the attention readout's scores, at this chunk's scope so
    // they outlive the layer that writes them and reach the copy below. Only
    // for the chunk that asks -- no allocation and no launch otherwise, and
    // the reservation for it is the prefill plan's (`plan_load_sizes`).
    //
    // GitHub #263: one target per armed GQA layer -- the pointing head's, and
    // every layer holding a head of the set -- each with its own read flag;
    // the readout is read only when every armed layer read. The set's packed
    // argmax lives beside the scores (8 bytes a head), zeroed once here,
    // before the first armed layer raises it.
    std::array<AttentionReadoutTarget, kReadoutGqaLayers> readout_targets{};
    std::array<bool, kReadoutGqaLayers> readout_armed{};
    std::array<bool, kReadoutGqaLayers> readout_layer_read{};
    bool readout_read = false;
    ninfer::Tensor readout_scores;
    ninfer::DeviceSpan readout_set_best{};
    ninfer::DeviceSpan readout_set_neighbours{};
    ninfer::DeviceSpan readout_set_rows{};
    if (attention_readout != nullptr) {
      readout_scores = model->scratch->alloc(
          ninfer::DType::FP32, {static_cast<std::int32_t>(attention_readout->key_count), 1, 1, 1});
      if (attention_readout->set_count > 0) {
        readout_set_best = model->scratch->alloc_bytes(
            static_cast<std::size_t>(attention_readout->set_count) * sizeof(unsigned long long));
        err = cudaMemsetAsync(readout_set_best.data, 0, readout_set_best.bytes, model->stream);
        if (err != cudaSuccess) {
          set_error(std::string("ignis_program_prefill: cudaMemsetAsync(head set) failed: ") +
                    cudaGetErrorString(err));
          return -1;
        }
        // GitHub #264: four scores a head, written by each armed layer's
        // gather. All-ones bytes are a quiet NaN, which is what "off the
        // grid" reads as, so a slot no gather reached never looks like a
        // score. A set read in rows over a text span (#275) has no grid and
        // may ask for none.
        if (attention_readout->host_set_neighbours != nullptr) {
          readout_set_neighbours = model->scratch->alloc_bytes(
              static_cast<std::size_t>(attention_readout->set_count) * 4 * sizeof(float));
          err = cudaMemsetAsync(readout_set_neighbours.data, 0xff, readout_set_neighbours.bytes,
                                model->stream);
          if (err != cudaSuccess) {
            set_error(std::string("ignis_program_prefill: cudaMemsetAsync(head set neighbours) "
                                  "failed: ") +
                      cudaGetErrorString(err));
            return -1;
          }
        }
        // GitHub #275 (ADR 0041): every head's row, one float per key, from
        // the room the load reserved beside the text score room. Every slot
        // is written by the fused launch of the head's layer, so no fill.
        if (attention_readout->host_set_rows != nullptr) {
          readout_set_rows = model->scratch->alloc_bytes(
              static_cast<std::size_t>(attention_readout->set_count) *
              static_cast<std::size_t>(attention_readout->key_count) * sizeof(float));
        }
      }
      const auto arm = [&](std::int32_t ordinal) -> AttentionReadoutTarget & {
        const auto at = static_cast<std::size_t>(ordinal);
        AttentionReadoutTarget &target = readout_targets[at];
        if (!readout_armed[at]) {
          readout_armed[at] = true;
          target.query_head = -1;
          target.key_begin = attention_readout->key_begin;
          target.key_count = attention_readout->key_count;
          target.device_set_best = static_cast<unsigned long long *>(readout_set_best.data);
          target.device_set_neighbours = static_cast<float *>(readout_set_neighbours.data);
          target.device_set_rows = static_cast<float *>(readout_set_rows.data);
          target.grid_cols = static_cast<std::int32_t>(attention_readout->grid_cols);
          target.excluded_count = static_cast<std::int32_t>(attention_readout->excluded_count);
          for (uint32_t e = 0; e < attention_readout->excluded_count; ++e) {
            target.excluded[e] = attention_readout->excluded[e];
          }
          target.read = &readout_layer_read[at];
        }
        return target;
      };
      AttentionReadoutTarget &pointing = arm(attention_readout->gqa_ordinal);
      pointing.query_head = attention_readout->query_head;
      pointing.device_scores = static_cast<float *>(readout_scores.data);
      for (uint32_t i = 0; i < attention_readout->set_count; ++i) {
        AttentionReadoutTarget &target = arm(attention_readout->set_gqa_ordinals[i]);
        target.set_query_head[target.set_heads] = attention_readout->set_query_heads[i];
        target.set_slot[target.set_heads] = static_cast<std::int32_t>(i);
        ++target.set_heads;
      }
    }

    // GitHub #507: a GDN layer runs in place on `left` (its body consumes its
    // input residual and adds into it, no copy of it); a GQA layer writes
    // `right` and the two swap. The test-only switch back to the pre-#507
    // prefill (kernel/src/layer_internal.h) swaps after every layer.
    const bool gdn_in_place = !ignis_gdn_layer_layout_copies();
    uint64_t dispatches = 0;
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      profiler.record_layer_begin(layer, model->stream);
      const bool reads_attention =
          attention_readout != nullptr && model->layers[layer].kind == IGNIS_LAYER_GQA &&
          readout_armed[ignis_gqa_relative_layer(layer)];
      const bool in_place = gdn_in_place && model->layers[layer].kind != IGNIS_LAYER_GQA;
      const int32_t rc = model->layers[layer].kind == IGNIS_LAYER_GQA
          ? ignis_gqa_layer_run_body(
                model, pool, seq, layer, left.data, right.data, num_tokens, mode, rope_positions,
                reads_attention ? &readout_targets[ignis_gqa_relative_layer(layer)] : nullptr)
          : ignis_gdn_layer_run_body(model, pool, seq, layer, left.data,
                                     in_place ? left.data : right.data, num_tokens, mode);
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
      if (!in_place) {
        std::swap(left, right);
      }
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
      append_dflash2_context(model, pool, features, dflash2_scalars.data(), tap_count,
                             IgnisA16Call::PrefillTaps);
    }
    profiler.record_head_begin(model->stream);

    // `left` holds every layer's output once the layer is done (a swap after
    // each layer that wrote `right`), so the final residual is in `left`.
    ninfer::Tensor final_residual = left;
    // GitHub #620: on an MTP load, the head's rows the chunk licenses and the
    // continuation hidden its last column leaves (kernel/src/mtp_head.h);
    // with the readout, the post-final-norm hidden of every column.
    if (model->drafter == IgnisDrafter::kMtp) {
      // GitHub #624: on a multimodal chunk, the axes its positions rotated
      // at from the one before it -- the span's column before the chunk, or
      // before the span the position the caller names (`mrope_previous`) --
      // and the media columns its embedding took. The host vector outlives
      // the chunk's synchronize below.
      ignis::mtp::ChunkMultimodal head_media;
      std::vector<std::int32_t> head_axes;
      if (multimodal.positions != nullptr) {
        const std::size_t entries = static_cast<std::size_t>(T) + 1;
        head_axes.resize(3 * entries);
        for (std::size_t axis = 0; axis < 3; ++axis) {
          std::int32_t *row = head_axes.data() + axis * entries;
          row[0] = chunk_offset > 0 ? multimodal.positions[axis * multimodal.span_tokens + chunk_offset - 1]
                   : multimodal.previous != nullptr ? multimodal.previous[axis]
                                                    : 0;
          std::copy_n(rope_positions_host.data() + axis * static_cast<std::size_t>(T), T, row + 1);
        }
        head_media.axes = head_axes.data();
        head_media.media = multimodal.media;
        head_media.scatter = scatter_host.data();
        head_media.count = static_cast<std::int32_t>(scatter_host.size());
        head_media.first_column = media_column;
      }
      std::string head_error;
      if (ignis::mtp::prefill_chunk(*model, *pool, *seq, ids, final_residual, T, head_media, span_hidden_out,
                                    &head_error) != 0) {
        set_error("ignis_program_prefill: chunk at span offset " + std::to_string(chunk_offset) + " (" +
                  std::to_string(num_tokens) + " tokens) for sequence slot " + std::to_string(seq->slot) +
                  " failed: " + head_error);
        return -1;
      }
    }
    // The measurement readout (`ignis_prefill_options::out_span_logits`):
    // the head over every column of the chunk, 32 columns at a time, each
    // block copied out before the next reuses its buffers (stream order).
    if (span_logits_out != nullptr) {
      constexpr std::int32_t kSpanBlock = kSpanLogitsBlockColumns;
      ninfer::DeviceArena::Scope span_scope = model->scratch->scope();
      const ninfer::Tensor norm_weight(const_cast<void *>(model->final_norm.qdata),
                                       ninfer::DType::BF16, {hidden, 1, 1, 1});
      ninfer::Tensor block_norm = model->scratch->alloc(ninfer::DType::BF16, {hidden, kSpanBlock, 1, 1});
      ninfer::Tensor block_logits = model->scratch->alloc(ninfer::DType::BF16, {vocab, kSpanBlock, 1, 1});
      for (std::int32_t c0 = 0; c0 < T; c0 += kSpanBlock) {
        const std::int32_t cols = std::min<std::int32_t>(kSpanBlock, T - c0);
        auto *block_hidden = static_cast<std::uint8_t *>(final_residual.data) +
                             static_cast<std::size_t>(c0) * static_cast<std::size_t>(hidden) *
                                 sizeof(uint16_t);
        const ninfer::Tensor columns(static_cast<void *>(block_hidden), ninfer::DType::BF16,
                                     {hidden, cols, 1, 1});
        ninfer::Tensor normalized(block_norm.data, ninfer::DType::BF16, {hidden, cols, 1, 1});
        ninfer::Tensor logits(block_logits.data, ninfer::DType::BF16, {vocab, cols, 1, 1});
        ninfer::ops::rmsnorm(columns, norm_weight, model->rms_norm_eps, /*unit_offset=*/true,
                             normalized, model->stream);
        ignis::trellis_body::output_head(*model, normalized, logits, model->stream);
        err = cudaMemcpyAsync(span_logits_out + static_cast<std::size_t>(c0) * static_cast<std::size_t>(vocab),
                              logits.data,
                              static_cast<std::size_t>(cols) * static_cast<std::size_t>(vocab) * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost, model->stream);
        if (err != cudaSuccess) {
          set_error(std::string("ignis_program_prefill: cudaMemcpyAsync(span logits) failed: ") +
                    cudaGetErrorString(err));
          return -1;
        }
      }
    }
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
      ignis::trellis_body::output_head(*model, normalized, logits, model->stream);
      // The absolute logical position of the span's last token in this
      // chunk: `seq->position` is still the chunk's pre-advance frontier
      // here (the caller advances it only after this function returns 0).
      const auto last_position = static_cast<std::int32_t>(seq->position + T - 1);
      if (sample_single(model, pool, seq, logits, sampling, ninfer::ops::kSamplePurposePrefill,
                        last_position, out_token_id, permitted_prob_out) != 0) {
        return -1;
      }
      // GitHub #586: the row this draw came from, kept for a repair.
      if (out_kept != nullptr && keep_draw_scores(model, seq, logits.data, last_position, sampling,
                                                  out_kept) != 0) {
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

    // GitHub #260: the scores the readout layer wrote, one float per key and
    // nothing else, copied out beside the logits row and confirmed by the
    // same synchronization -- and (GitHub #263) the set's packed argmax, 8
    // bytes a head, only when every armed layer read.
    if (attention_readout != nullptr) {
      readout_read = true;
      for (std::size_t ordinal = 0; ordinal < readout_armed.size(); ++ordinal) {
        readout_read = readout_read && (!readout_armed[ordinal] || readout_layer_read[ordinal]);
      }
    }
    std::vector<unsigned long long> host_set_best;
    if (readout_read) {
      err = cudaMemcpyAsync(attention_readout->host_scores, readout_scores.data,
                            static_cast<std::size_t>(attention_readout->key_count) * sizeof(float),
                            cudaMemcpyDeviceToHost, model->stream);
      if (err == cudaSuccess && attention_readout->set_count > 0) {
        host_set_best.resize(attention_readout->set_count);
        err = cudaMemcpyAsync(host_set_best.data(), readout_set_best.data,
                              host_set_best.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost, model->stream);
        if (err == cudaSuccess && attention_readout->host_set_neighbours != nullptr) {
          // GitHub #264: the four scores around each argmax, beside it.
          err = cudaMemcpyAsync(attention_readout->host_set_neighbours,
                                readout_set_neighbours.data, readout_set_neighbours.bytes,
                                cudaMemcpyDeviceToHost, model->stream);
        }
        if (err == cudaSuccess && attention_readout->host_set_rows != nullptr) {
          // GitHub #275: every head's row, behind the same synchronization.
          err = cudaMemcpyAsync(attention_readout->host_set_rows, readout_set_rows.data,
                                readout_set_rows.bytes, cudaMemcpyDeviceToHost, model->stream);
        }
      }
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_prefill: cudaMemcpyAsync(attention scores) failed: ") +
                  cudaGetErrorString(err));
        return -1;
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
    if (attention_readout != nullptr) {
      // A zero slot is a head no key was scored for: the set is not whole,
      // so none of it is reported read.
      for (std::size_t i = 0; readout_read && i < host_set_best.size(); ++i) {
        readout_read = host_set_best[i] != 0;
        attention_readout->host_set_argmax[i] =
            static_cast<int32_t>(static_cast<std::uint32_t>(host_set_best[i] & 0xffffffffu));
        // GitHub #264: the peak's own score is the packed value's high half,
        // order-preserved on the way in and undone here -- it costs the leaf
        // nothing to report.
        attention_readout->host_set_peak[i] =
            ignis_attention_unpack_score(static_cast<std::uint32_t>(host_set_best[i] >> 32));
      }
      *attention_readout->host_read = readout_read ? 1 : 0;
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
    // GitHub #506: over a chunk it tapped nothing of too. The positions below
    // the tap start are ones no later window reads, so the frontier stands at
    // the chunk's end exactly as it would have had they been tapped.
    if (dflash2) {
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
// the failing one. `drafter_floor` is the call's
// (`ignis_prefill_options::drafter_floor`, GitHub #506).
//
// GitHub #519: `draws` is false on a span with a next prompt token whose draw
// no retained image keeps (GitHub #520): no chunk computes the output head or
// samples, and the span leaves the sequence with no pending token (-1) rather
// than an earlier span's -- its draw is the prompt's last span's to make.
//
// GitHub #287: `keeps_scores` keeps the last chunk's draw's scores for a
// repair; without it the draw keeps none.
int32_t run_program_prefill_chunked(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                    const int32_t *token_ids, uint64_t num_tokens,
                                    const ignis_sampling_params &sampling, float *out_logits,
                                    LinearPolicyMode mode, const SpanMultimodal &multimodal,
                                    float *permitted_prob,
                                    const ArmedAttentionReadout *attention_readout,
                                    std::uint16_t *span_logits, uint64_t drafter_floor,
                                    bool draws, bool keeps_scores, std::uint16_t *span_hidden) {
  const uint64_t chunk_width = model->prefill_chunk_tokens;
  ChunkProfiler::instance().begin_span(num_tokens, model->prefill_chunk_tokens);
  const uint64_t tap_from = dflash2_tap_from(seq->position, num_tokens, drafter_floor);
  uint64_t offset = 0;
  while (offset < num_tokens) {
    const uint64_t chunk_len = std::min<uint64_t>(chunk_width, num_tokens - offset);
    const bool is_last_chunk = (offset + chunk_len == num_tokens);
    int32_t successor = -1;
    ignis_kept_scores kept;
    float *slot_logits = is_last_chunk ? out_logits : nullptr;
    if (run_program_chunk(model, pool, seq, token_ids + offset, chunk_len, offset, tap_from,
                          is_last_chunk && draws, sampling, &successor, slot_logits, mode, multimodal,
                          is_last_chunk ? permitted_prob : nullptr,
                          is_last_chunk ? attention_readout : nullptr,
                          span_logits == nullptr
                              ? nullptr
                              : span_logits + offset * static_cast<uint64_t>(model->vocab),
                          keeps_scores ? &kept : nullptr,
                          span_hidden == nullptr ? nullptr
                                                 : span_hidden + offset * static_cast<uint64_t>(model->hidden)) != 0) {
      return -1;
    }
    seq->position += chunk_len;
    if (is_last_chunk) {
      // `successor` stays -1 on a span that does not draw (GitHub #519), and
      // `kept` then keeps nothing (GitHub #586), as it does without
      // `keeps_scores` (GitHub #287).
      seq->pending_token = successor;
      seq->kept = kept;
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
  if (model != nullptr && model->flash_next) {
    set_error("ignis_prefill: the degenerate program is Qwen 3.8-27B's");
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
  if (model != nullptr && model->flash_next) {
    set_error("ignis_decode: the degenerate program is Qwen 3.8-27B's");
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
  // GitHub #302: a Flash-Next load runs its own program.
  if (model != nullptr && model->flash_next) {
    return ignis::flash_next::program_prefill(model, pool, seq, token_ids, num_tokens, start_position,
                                              sampling, options, out_logits);
  }
  if (!validate_program(model, pool, seq, token_ids, num_tokens, sampling)) {
    return -1;
  }
  if (const char *refusal = ignis_seq_restore_refusal(*seq)) {
    set_error(std::string("ignis_program_prefill: ") + refusal);
    return -1;
  }
  if (!drafter_matches_load(model, pool, "ignis_program_prefill")) {
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
    multimodal.previous = options->mrope_previous;
  }
  if (!validate_span_multimodal(model, route, multimodal)) {
    return -1;
  }
  // GitHub #624: the MTP head's row before a span continuing a sequence is
  // built from the span's first token, at a position no column of the span
  // holds, so its axes come with the span.
  if (model->drafter == IgnisDrafter::kMtp && multimodal.positions != nullptr && seq->position > 0 &&
      multimodal.previous == nullptr) {
    set_error("ignis_program_prefill: a multimodal span continuing a sequence on a load with the MTP head needs "
              "mrope_previous, the axes its position before the span rotated at");
    return -1;
  }
  std::uint16_t *span_logits = options != nullptr ? options->out_span_logits : nullptr;
  if (span_logits != nullptr && route != IGNIS_PREFILL_ROUTE_CHUNKED) {
    set_error("ignis_program_prefill: out_span_logits needs the chunked route");
    return -1;
  }
  // GitHub #620: the head's rows are written by the chunk loop alone; the
  // test-only per-token route would leave them unwritten. And the hiddens the
  // head reads are read out where it reads them, on an MTP load.
  if (model->drafter == IgnisDrafter::kMtp && route != IGNIS_PREFILL_ROUTE_CHUNKED) {
    set_error("ignis_program_prefill: an MTP load prefills on the chunked route, which writes the head's state");
    return -1;
  }
  std::uint16_t *span_hidden = options != nullptr ? options->out_span_hidden : nullptr;
  if (span_hidden != nullptr && model->drafter != IgnisDrafter::kMtp) {
    set_error("ignis_program_prefill: out_span_hidden reads the hiddens the MTP head reads, on an MTP load");
    return -1;
  }
  // GitHub #519: a span with a next prompt token draws nothing -- no output
  // head, no sample -- since only the prompt's last span's draw is read, and
  // (GitHub #520) a retained image's, which keeps it.
  if (const std::string why = ignis::step::next_prompt_token_refusal(
          static_cast<std::int32_t>(model->vocab), *sampling, options, out_logits);
      !why.empty()) {
    set_error("ignis_program_prefill: " + why);
    return -1;
  }
  const bool draws = ignis::step::span_draws(options);
  // GitHub #260 (ADR 0038): the attention readout. A negative ordinal or a
  // null score buffer asks for none. What is refused here is only what the
  // host can never mean -- a head or layer the model does not have, a span
  // wider than the load reserved room to score -- and keys that turn out not
  // to be there to read are the unread flag's business, not an error.
  ArmedAttentionReadout attention_readout;
  const bool arms_readout = options != nullptr && options->attention_gqa_ordinal >= 0 &&
                            options->out_attention_scores != nullptr;
  // GitHub #263: a head set rides the readout; named without it, it is a
  // caller that meant one and would silently get none.
  if (!arms_readout && options != nullptr &&
      (options->attention_set_count > 0 || options->attention_excluded_count > 0)) {
    set_error("ignis_program_prefill: a head set or excluded keys without an attention readout "
              "armed to read them beside");
    return -1;
  }
  if (arms_readout) {
    // The room `plan_load_sizes` (kernel/src/model.cu) reserved for the
    // scores: one F32 per token of one vision item at the load's item bound,
    // or (GitHub #275) per key of the text span the load was told a readout
    // reads, whichever is wider, capped by the context -- the same terms, or
    // a readout could outgrow its arena.
    const std::uint64_t text_keys =
        std::min<std::uint64_t>(model->attention_text_max_keys, model->max_context_tokens);
    const std::uint64_t capacity = std::max<std::uint64_t>(
        std::min<std::uint64_t>(model->vision_item_max_tokens, model->max_context_tokens),
        text_keys);
    // The GQA ordinals this model has, for the readout's layer and every
    // head of its set.
    std::array<bool, kReadoutGqaLayers> gqa_ordinal{};
    for (uint32_t layer = 0; layer < model->layers.size(); ++layer) {
      const std::uint32_t ordinal = ignis_gqa_relative_layer(layer);
      if (model->layers[layer].kind == IGNIS_LAYER_GQA && ordinal < gqa_ordinal.size()) {
        gqa_ordinal[ordinal] = true;
      }
    }
    const auto is_gqa_ordinal = [&](int32_t ordinal) {
      return ordinal >= 0 && ordinal < kReadoutGqaLayers &&
             gqa_ordinal[static_cast<std::size_t>(ordinal)];
    };
    const bool has_layer = is_gqa_ordinal(options->attention_gqa_ordinal);
    // GitHub #263: a head set names heads the model has, each once, at most
    // the reserved count, with its result buffer; its excluded keys sit
    // inside the span, at most what a launch carries.
    bool set_ok = options->attention_set_count <= static_cast<uint32_t>(kReadoutMaxSetHeads) &&
                  options->attention_excluded_count <= static_cast<uint32_t>(kReadoutMaxExcluded) &&
                  (options->attention_excluded_count == 0 ||
                   (options->attention_set_count > 0 && options->attention_excluded != nullptr));
    // GitHub #275: rows are a set's, at most the heads the load reserved
    // rows for, over at most the text span it reserved them over.
    const bool rows = options->out_attention_set_rows != nullptr;
    set_ok = set_ok &&
             (!rows || (options->attention_set_count > 0 &&
                        options->attention_set_count <= static_cast<uint32_t>(kReadoutMaxRowHeads) &&
                        static_cast<std::uint64_t>(options->attention_key_count) <= text_keys));
    if (set_ok && options->attention_set_count > 0) {
      set_ok = options->attention_set_gqa_ordinals != nullptr &&
               options->attention_set_query_heads != nullptr &&
               options->out_attention_set_argmax != nullptr &&
               // GitHub #264: the peak and its neighbours ride with the set,
               // and the image grid says which neighbours exist. A set
               // without them would answer with no way to read inside a cell
               // -- unless it reads rows (#275): a text span is no grid.
               options->out_attention_set_peak != nullptr &&
               ((options->out_attention_set_neighbours != nullptr &&
                 options->attention_grid_cols > 0 &&
                 static_cast<std::int64_t>(options->attention_grid_cols) <=
                     options->attention_key_count) ||
                (rows && options->out_attention_set_neighbours == nullptr &&
                 options->attention_grid_cols == 0)) &&
               (options->attention_excluded_count == 0 || options->attention_excluded != nullptr);
      std::vector<bool> seen(static_cast<std::size_t>(kReadoutGqaLayers) * kIgnisGqaQHeads, false);
      for (uint32_t i = 0; set_ok && i < options->attention_set_count; ++i) {
        const int32_t ordinal = options->attention_set_gqa_ordinals[i];
        const int32_t head = options->attention_set_query_heads[i];
        set_ok = is_gqa_ordinal(ordinal) && head >= 0 && head < kIgnisGqaQHeads;
        if (set_ok) {
          const std::size_t at = static_cast<std::size_t>(ordinal) * kIgnisGqaQHeads + head;
          set_ok = !seen[at];
          seen[at] = true;
        }
      }
      for (uint32_t e = 0; set_ok && e < options->attention_excluded_count; ++e) {
        set_ok = options->attention_excluded[e] >= 0 &&
                 options->attention_excluded[e] < options->attention_key_count;
      }
    }
    if (route != IGNIS_PREFILL_ROUTE_CHUNKED || options->out_attention_read == nullptr ||
        !has_layer || options->attention_query_head < 0 ||
        options->attention_query_head >= kIgnisGqaQHeads || options->attention_key_begin < 0 ||
        options->attention_key_count <= 0 ||
        static_cast<std::uint64_t>(options->attention_key_count) > capacity || !set_ok) {
      set_error("ignis_program_prefill: an attention readout of GQA layer " +
                std::to_string(options->attention_gqa_ordinal) + ", head " +
                std::to_string(options->attention_query_head) + ", over " +
                std::to_string(options->attention_key_count) +
                " keys with a head set of " + std::to_string(options->attention_set_count) +
                (rows ? " read in rows" : "") +
                " cannot be armed on this load (the chunked route, GQA layers and query heads "
                "the model has, each head once and at most " +
                std::to_string(kReadoutMaxSetHeads) + " of them, at most " +
                std::to_string(kReadoutMaxExcluded) + " excluded keys inside the span, at most " +
                std::to_string(capacity) +
                " keys -- the vision item or text span the load reserved scores for -- and, with "
                "a set, its peak and neighbour buffers and an image grid of 1.." +
                std::to_string(options->attention_key_count) +
                " columns, or its rows: at most " + std::to_string(kReadoutMaxRowHeads) +
                " heads over at most " + std::to_string(text_keys) + " keys)");
      return -1;
    }
    attention_readout.gqa_ordinal = options->attention_gqa_ordinal;
    attention_readout.query_head = options->attention_query_head;
    attention_readout.key_begin = options->attention_key_begin;
    attention_readout.key_count = options->attention_key_count;
    attention_readout.host_scores = options->out_attention_scores;
    attention_readout.host_read = options->out_attention_read;
    attention_readout.set_count = options->attention_set_count;
    attention_readout.set_gqa_ordinals = options->attention_set_gqa_ordinals;
    attention_readout.set_query_heads = options->attention_set_query_heads;
    attention_readout.excluded_count = options->attention_excluded_count;
    attention_readout.excluded = options->attention_excluded;
    attention_readout.host_set_argmax = options->out_attention_set_argmax;
    attention_readout.grid_cols = options->attention_grid_cols;
    attention_readout.host_set_peak = options->out_attention_set_peak;
    attention_readout.host_set_neighbours = options->out_attention_set_neighbours;
    attention_readout.host_set_rows = options->out_attention_set_rows;
    *attention_readout.host_read = 0;
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
      // GitHub #586: the test-only route keeps no scores for a repair.
      seq->kept = ignis_kept_scores{};
      ++seq->position;
    }
  } else {
    rc = run_program_prefill_chunked(model, pool, seq, token_ids, num_tokens, *sampling,
                                     out_logits, mode, multimodal, permitted_prob,
                                     arms_readout ? &attention_readout : nullptr, span_logits,
                                     options != nullptr ? options->drafter_floor : 0, draws,
                                     options != nullptr && options->keeps_scores != 0, span_hidden);
  }
  if (rc != 0) {
    return rc;
  }
  // GitHub #315 (ADR 0048): the span's draw is the token the first round
  // emits; a stop id drawn with the reasoning block open becomes `</think>`.
  // GitHub #519: a span that drew nothing has nothing to redirect.
  int32_t *redirected = options != nullptr ? options->out_reasoning_redirected : nullptr;
  if (!draws) {
    if (redirected != nullptr) {
      *redirected = 0;
    }
  } else if (redirect_span_draw(pool, seq, *sampling, model->stream, redirected) != 0) {
    return -1;
  }
  if (multimodal.positions != nullptr) {
    seq->rope_delta = multimodal.rope_delta;
  }
  // No rewrite checkpoint of the drafter window is taken here. The reference
  // saves one at its chat template's rewrite boundary and restores it when a
  // later turn rewrites the text past that boundary; ignis claims a prompt
  // checkpoint taken at the opener instead (ADR 0029), whose image carries
  // the window as it stood there. The copy it once made here, and the
  // synchronize after it, were read by nothing.
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

// GitHub #257 (spec runtime/06): the ring slots of each lane's appended but
// uncommitted columns, cleared when the round does not reach its commit.
//
// A verify pass appends every lane's `extent + 1` columns, and under
// hq-e8-2b each append also writes the column's ring slot of the residual
// window and sets its bit. A round that fails after the pass leaves every
// lane at its old frontier with those columns written past it; a retry at a
// smaller extent then reads the stale slots as the exact rows of older keys
// still inside its window. Armed once the pass is enqueued and disarmed at
// the commit, which clears only the rejected columns' bits itself. Best
// effort by construction -- a destructor has nowhere to report to -- and
// always safe: a cleared bit falls back to the codec.
struct UncommittedRingColumns {
  ignis_seq_pool *pool = nullptr;
  cudaStream_t stream  = nullptr;
  const std::vector<std::int32_t> *base_positions = nullptr;
  const std::vector<std::int32_t> *extents        = nullptr;
  const std::vector<std::int32_t> *slots          = nullptr;
  bool armed = false;

  ~UncommittedRingColumns() {
    if (!armed || !pool->has_hq_residual()) {
      return;
    }
    for (std::size_t i = 0; i < slots->size(); ++i) {
      const auto first = static_cast<std::uint64_t>((*base_positions)[i]);
      const auto extent = static_cast<std::uint64_t>((*extents)[i]);
      ninfer::apply_kv_ring_valid_words(
          pool->hq_ring_words((*slots)[i]),
          ignis_hq_ring_invalidate_mask(first, first + extent + 1),
          ninfer::KvRingWords{}, kIgnisHqRingWords, stream);
      // GitHub #621: and the MTP head's own words over every row its chain
      // and its commit may have written, [P - 1, P + extent)
      // (kernel/src/mtp_head.h `round_rows`).
      if (pool->has_mtp_head()) {
        ninfer::apply_kv_ring_valid_words(pool->hq_head_ring_words((*slots)[i]),
                                          ignis_hq_ring_invalidate_mask(first - 1, first + extent),
                                          ninfer::KvRingWords{}, kIgnisHqRingWords, stream);
      }
    }
    (void)cudaStreamSynchronize(stream);
  }
};

// P5-04 (GitHub #153): one verify round over `batch_size` lanes at the
// window `k` of its shape `verify` (GitHub #686: the caller's choice, handed
// to everything the round calls; spec 05, "The verify round"), its lanes laid
// out in the round's buffers by `layout` (GitHub #712, kernel/src/round_layout.h:
// each lane's window, column base and draft base, which every site below that
// reads a lane's columns or drafts takes). The device-side pass is
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
//              (GitHub #590: on a lane whose cut the host decides, first
//              shortened to the prefix the host keeps, its pending token the
//              draft after that prefix)
//   commit   = the first c columns: KV frontier p -> p + c (the columns'
//              K/V were appended in place at p..p+c-1 and the rest are
//              overwritten by the next round), fold records[0..c) into the
//              GDN slot and conv taps, pending <- licensed[c-1] (the
//              successor of the run's last token: a draft the target agreed
//              with, or t*)
// so a lane whose run was cut stands exactly where a per-token decode of the
// same text would, and one whose run was not stands with t* pending -- what
// today's round leaves after one token, at every token of the run. The round
// is atomic: no lane's state moves unless every step succeeded. GitHub #676:
// a round that leaves its fold in flight (kernel/src/verify_return.h) moves
// every lane's state once the fold is queued, not confirmed; a fault in it
// surfaces at the next leaf call that waits on the device.
int32_t run_verify_round(ignis_model *model, ignis_seq_pool *pool, IgnisVerifyRound &verify,
                         const ignis::round_layout::Layout &layout, ignis_seq *const *sequences, uint64_t batch_size,
                         const ignis_sampling_params *sampling, const ignis_decode_options &options,
                         int32_t *out_token_ids) {
  const std::uint32_t k = verify.window;
  // GitHub #689: whether the round is at its load's extended shape.
  const bool extended = verify.extended();
  // GitHub #712: a round of one shape, every lane at its window. GitHub #713:
  // or a mixed round, on the base shape's buffers (which both shapes hold),
  // each group at its shape's window, of a mix the load holds.
  const bool mixed = !layout.uniform();
  const bool laid_out =
      mixed ? &verify == model->base_shape.get() && model->extended_shape != nullptr &&
                  layout.groups[0].window == model->extended_shape->window && layout.groups[1].window == k
            : layout.window_of(0) == k && layout.extended(0) == extended;
  if (layout.lanes() != batch_size || !laid_out) {
    set_error("ignis_program_decode: a verify round's layout is not one of its " + std::to_string(batch_size) +
              " lanes at its round shape's " + std::to_string(k) + " drafts");
    return -1;
  }
  if (mixed) {
    if (const char *refusal = ignis_mixed_round_refusal(*model, pool->gdn_pool.slot_count(), layout)) {
      set_error(std::string("ignis_program_decode: a mixed round of this load: ") + refusal);
      return -1;
    }
    if (options.host_cut != nullptr) {
      set_error("ignis_program_decode: a mixed round takes no host cut");
      return -1;
    }
  }
  // GitHub #687: the caller's arrays are strided by the load's window
  // (`speculative_window`), the round's buffers by its shape's: equal on a
  // load of one shape, and on a load of two at its extended shape; a base
  // round there reads the first 8 of each lane's 16 and writes as many.
  const std::uint32_t caller_k = options.speculative_window;
  const std::uint32_t caller_columns = caller_k + 1;
  const auto hidden = static_cast<std::int32_t>(model->hidden);
  // GitHub #619: what fills the round's proposal follows the load's drafter.
  // One in the leaf proposes every lane's drafts itself, inside the verify
  // pass (P5-05, GitHub #155); the verify substrate alone takes them from
  // the caller. Around the pass, DFlash2 alone appends: the carried anchor's
  // taps before it and the committed columns' feature taps after the cut.
  const bool in_leaf = ignis_drafter_proposes_in_the_leaf(model->drafter);
  const bool dflash2 = model->drafter == IgnisDrafter::kDflash2;
  // GitHub #621: the MTP head's chain proposes inside the pass, and after the
  // cut the head's rows and continuation hidden follow the committed run
  // alone (kernel/src/mtp_head.h, "the head in a verify round").
  const bool mtp = model->drafter == IgnisDrafter::kMtp;
  if (in_leaf && (options.drafts != nullptr || options.draft_counts != nullptr)) {
    set_error(std::string("ignis_program_decode: this model was loaded with ") +
              ignis_drafter_name(model->drafter) +
              ", which proposes every lane's drafts; drafts and draft_counts must be NULL");
    return -1;
  }
  // GitHub #267 (test-only): a teacher-forced round licenses the caller's
  // recorded text, so it draws nothing -- and a lane at temperature > 0
  // would have had the accept kernel count its own draw in the lane's
  // penalty row, which the forced text then contradicts.
  //
  // GitHub #577 (test-only): under the sampled draft a sampling lane is
  // forced on the device instead. Its draw drafts along the recording and its
  // accept takes each recorded token by the teacher-forced verdict -- with
  // probability min(p, q) / p, a live round's, and as the live round itself
  // on a recording of the lane's own rounds -- counting in the penalty row
  // the recorded tokens it licenses. Such a round runs eagerly, the
  // recording at `verify.forced_tokens` while it runs.
  const auto samples = [&](uint64_t i) { return sampling[i].greedy == 0 && sampling[i].temperature > 0.0f; };
  bool forced_on_device = false;
  if (options.forced_tokens != nullptr) {
    for (uint64_t i = 0; i < batch_size; ++i) {
      if (samples(i) && !verify.sampled_draft) {
        set_error("ignis_program_decode: a teacher-forced round is greedy; lane " +
                  std::to_string(i) + " samples at temperature > 0");
        return -1;
      }
      forced_on_device = forced_on_device || samples(i);
    }
  }
  // GitHub #692 (test-only): a teacher-forced round that names its drafts
  // verifies them in place of the drafter's -- written over the proposal
  // inside the pass, at the round's stride, each lane's first `k` of the
  // caller's row -- and its verdict takes their leading agreement with the
  // recording. A DFlash2 load's, every lane greedy (a sampling lane's drafts
  // are its draws, which another set of drafts would leave without their q);
  // the round runs eagerly, the copy at `verify.forced_drafts` while it runs.
  std::vector<std::int32_t> forced_draft_rows;
  if (options.forced_drafts != nullptr) {
    if (!dflash2) {
      set_error("ignis_program_decode: forced_drafts replace the DFlash2 drafter's proposals; this model was "
                "loaded with " + std::string(ignis_drafter_name(model->drafter)));
      return -1;
    }
    forced_draft_rows.resize(layout.drafts());
    for (uint64_t i = 0; i < batch_size; ++i) {
      if (samples(i)) {
        set_error("ignis_program_decode: forced_drafts ride greedy lanes; lane " + std::to_string(i) +
                  " samples at temperature > 0");
        return -1;
      }
      const auto lane = static_cast<std::uint32_t>(i);
      std::copy_n(options.forced_drafts + i * caller_k, layout.window_of(lane),
                  forced_draft_rows.data() + layout.draft_of(lane));
    }
  }

  std::vector<std::int32_t> anchors(batch_size, 0);
  std::vector<std::int32_t> drafts(layout.drafts(), 0);
  std::vector<std::int32_t> base_positions(batch_size, 0);
  std::vector<std::int32_t> extents(batch_size, 0);
  std::vector<std::int32_t> valid_columns(batch_size, 1);
  std::vector<std::int32_t> slots(batch_size, 0);
  // GitHub #195: on a vision load, the columns' rotation positions -- the
  // same `base + min(j, extent)` the traversal derives, plus the lane's own
  // `rope_delta`. Empty (and unstaged) on every other load.
  std::vector<std::int32_t> rope_positions(verify.rope_positions == nullptr ? 0 : layout.columns(), 0);
  std::vector<ninfer::ops::SamplingConfig> configs(batch_size);
  try {
    for (uint64_t i = 0; i < batch_size; ++i) {
      ignis_seq *seq = sequences[i];
      if (seq == nullptr || seq->pending_token < 0) {
        set_error("ignis_program_decode: sequence is null or was not prefilled");
        return -1;
      }
      if (const char *refusal = ignis_seq_restore_refusal(*seq)) {
        set_error(std::string("ignis_program_decode: ") + refusal);
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
      const auto lane = static_cast<std::uint32_t>(i);
      uint64_t extent = layout.window_of(lane);
      if (!in_leaf && options.drafts == nullptr) {
        extent = 0;
      } else if (!in_leaf && options.draft_counts != nullptr) {
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
      for (uint64_t j = 0; j < extent && !in_leaf; ++j) {
        drafts[layout.draft_of(lane) + j] = options.drafts[i * caller_k + j];
      }
      // The lane's rotation positions: the traversal's own column rule with
      // the sequence's delta added. A text sequence's delta is 0, so the two
      // matrices are equal and nothing about a text round changes.
      if (!rope_positions.empty()) {
        for (std::uint32_t j = 0; j < layout.window_of(lane) + 1; ++j) {
          rope_positions[layout.column_of(lane) + j] =
              base_positions[i] + std::min<std::int32_t>(static_cast<std::int32_t>(j), extents[i]) +
              seq->rope_delta;
        }
      }
      // GitHub #577 (test-only): a sampling lane forced on the device may stop
      // at any column, so every successor up to its extent is checked now,
      // before anything is enqueued (kernel/src/teacher_forcing.h) -- every
      // lane's in such a round, so that no refusal after the pass leaves a
      // penalty row the device already counted in.
      if (forced_on_device) {
        const std::int32_t *forced = options.forced_tokens + i * caller_columns;
        const std::int32_t bad = ignis_teacher_forced_first_bad_successor(
            anchors[i], forced, extents[i], static_cast<std::int32_t>(model->vocab), sampling[i].stop_ids,
            sampling[i].stop_id_count);
        if (bad >= 0) {
          set_error("ignis_program_decode: lane " + std::to_string(i) + "'s forced token " +
                    std::to_string(forced[bad]) + " after its first " + std::to_string(bad) +
                    " recorded tokens is outside the vocabulary (the pad -1 ends a recording only after a "
                    "stop id)");
          return -1;
        }
      }
    }

    // GitHub #157: each lane's carried anchor taps, into its window before
    // the drafter reads it inside the pass. A failure below leaves the lane
    // unmoved, and the append it enqueued rewrites the same slot with the
    // same bytes when the lane runs again.
    std::vector<std::vector<std::int32_t>> pending_scalars(batch_size);
    std::vector<bool> pending_appended(batch_size, false);
    if (dflash2) {
      ninfer::DeviceArena::Scope pending_scope = model->scratch->scope();
      for (uint64_t i = 0; i < batch_size; ++i) {
        pending_appended[i] = append_dflash2_pending(model, pool, *sequences[i], pending_scalars[i]);
      }
    }

    // GitHub #676: a DFlash2 load's live round stages its inputs through
    // host-mapped memory and waits on a ticket for its accept
    // (kernel/src/verify_return.h); every other round copies and synchronizes
    // as earlier releases do. Its slot is the next round's to wait for once
    // everything it enqueued has run, whether it commits or fails.
    VerifyStaging *const staging =
        ignis_verify_round_maps(verify.defer, dflash2, options.forced_tokens != nullptr) ? verify.staging.get()
                                                                                       : nullptr;
    if (staging != nullptr) {
      const cudaError_t err = ignis::deferred_verify::begin(*staging);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: waiting for the verify staging's slot failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }
    struct SlotRelease {
      VerifyStaging *staging;
      cudaStream_t stream;
      cudaError_t release() {
        VerifyStaging *const releasing = std::exchange(staging, nullptr);
        return releasing == nullptr ? cudaSuccess : ignis::deferred_verify::end(*releasing, stream);
      }
      ~SlotRelease() { (void)release(); }
    } slot_release{staging, model->stream};
    const auto stage = [&](ninfer::DeviceBuffer &buffer, const void *host, std::size_t bytes,
                           const char *what) -> bool {
      if (staging != nullptr && ignis::deferred_verify::stage(*staging, buffer.p, host, bytes)) {
        return true;
      }
      const cudaError_t err =
          cudaMemcpyAsync(buffer.p, host, bytes, cudaMemcpyHostToDevice, model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaMemcpyAsync(") + what + ") failed: " +
                  cudaGetErrorString(err));
        return false;
      }
      return true;
    };
    // GitHub #676: what `stage` put in the slot, copied by one kernel.
    const auto flush = [&](const char *what) -> bool {
      const cudaError_t err =
          staging == nullptr ? cudaSuccess : ignis::deferred_verify::flush(*staging, model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: staging (") + what + ") failed: " + cudaGetErrorString(err));
        return false;
      }
      return true;
    };
    const std::size_t lane_bytes = batch_size * sizeof(std::int32_t);
    if (!stage(*verify.anchors, anchors.data(), lane_bytes, "verify anchors") ||
        // The drafter writes this buffer on the device, inside the pass.
        (!in_leaf &&
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
               batch_size * sizeof(ninfer::ops::SamplingConfig), "verify sampling configs") ||
        !flush("verify inputs")) {
      return -1;
    }

    // GitHub #577 (test-only): the recording the device forces, at
    // `verify.forced_tokens` for this round only. The verify graphs were
    // captured without it, so the round runs eagerly.
    std::unique_ptr<ninfer::DeviceBuffer> forced_device;
    struct ClearForcedTokens {
      IgnisVerifyRound *verify = nullptr;
      ~ClearForcedTokens() {
        if (verify != nullptr) {
          verify->forced_tokens = nullptr;
        }
      }
    } clear_forced_tokens;
    // GitHub #687: at the round's stride, each lane's first `k + 1` of the
    // caller's row; a pageable copy staged before the call returns.
    std::vector<std::int32_t> forced_rows;
    if (forced_on_device) {
      forced_rows.resize(layout.columns());
      for (uint64_t i = 0; i < batch_size; ++i) {
        const auto lane = static_cast<std::uint32_t>(i);
        std::copy_n(options.forced_tokens + i * caller_columns, layout.window_of(lane) + 1,
                    forced_rows.data() + layout.column_of(lane));
      }
      const std::size_t forced_bytes = forced_rows.size() * sizeof(std::int32_t);
      forced_device = std::make_unique<ninfer::DeviceBuffer>(forced_bytes);
      if (!stage(*forced_device, forced_rows.data(), forced_bytes, "forced tokens")) {
        return -1;
      }
      verify.forced_tokens = static_cast<const std::int32_t *>(forced_device->p);
      clear_forced_tokens.verify = &verify;
    }

    // GitHub #692 (test-only): the forced drafts, for this round only, read
    // by the eager pass below.
    struct ClearForcedDrafts {
      IgnisVerifyRound *verify = nullptr;
      ~ClearForcedDrafts() {
        if (verify != nullptr) {
          verify->forced_drafts = nullptr;
        }
      }
    } clear_forced_drafts;
    if (!forced_draft_rows.empty()) {
      verify.forced_drafts = forced_draft_rows.data();
      clear_forced_drafts.verify = &verify;
    }

    UncommittedRingColumns uncommitted{pool, model->stream, &base_positions, &extents, &slots};
    // GitHub #713: a mixed round's graph is its mix's, where the latest capture made one.
    const int mix = ignis::round_layout::mix_index(layout);
    const cudaGraphExec_t graph =
        mixed ? (mix >= 0 ? model->mixed_verify_exec[static_cast<std::size_t>(mix)] : nullptr)
              : (verify.graph_ready[batch_size - 1] ? verify.graph_exec[batch_size - 1] : nullptr);
    const bool use_graph = !forced_on_device && verify.forced_drafts == nullptr && graph != nullptr;
    // GitHub #590: the pass writes the verify logits, which a repair reads for
    // a lane whose commit the host cuts. Stamped before the work is enqueued,
    // so a round that fails part way leaves no earlier record believing them
    // its own.
    verify.kept_epoch = ++model->kept_epoch;
    // GitHub #713: a mixed round's stamp, whose logits are at its groups' strides.
    if (mixed) {
      model->mixed_kept_epoch = verify.kept_epoch;
    }
    // Armed before the launch: a pass that fails part way may still have
    // appended some of its columns.
    uncommitted.armed = true;
    if (use_graph) {
      round_timing_mark(model, model->verify_timing, true,
                        mixed ? static_cast<std::uint8_t>(ignis::round_timing::kMixedBlock) : extended ? 1 : 0);
      const cudaError_t err = cudaGraphLaunch(graph, model->stream);
      round_timing_mark(model, model->verify_timing, false);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaGraphLaunch(verify) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    } else if (ignis_verify_graph_run_batch(model, pool, verify, layout, LinearPolicyMode::kEngineDefault) != 0) {
      set_error(std::string("ignis_program_decode: ") + ignis_decode_graph_last_error());
      return -1;
    }

    std::vector<std::int32_t> licensed(layout.columns(), 0);
    std::vector<std::int32_t> accepted(batch_size, 0);
    cudaError_t err = cudaSuccess;
    // GitHub #676: through the slot, behind the ticket.
    const ignis::deferred_verify::Readback accept_reads[] = {
        {licensed.data(), verify.licensed_tokens->p, licensed.size() * sizeof(std::int32_t)},
        {accepted.data(), verify.accepted->p, lane_bytes}};
    if (staging != nullptr && ignis::deferred_verify::hand_back(*staging, accept_reads, model->stream, &err)) {
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: waiting for the verify round's accept failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    } else {
      err = cudaMemcpyAsync(licensed.data(), verify.licensed_tokens->p,
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
    }

    // GitHub #267 (test-only): teacher forcing. The pass ran exactly as a
    // round does -- the drafter's proposals, every column's K/V, taps and
    // ReplaySSM records -- and only the accept's verdict is replaced: the
    // lane accepts the leading drafts its recorded text agrees with and
    // takes the recorded token after them as its correction. Every column
    // the commit keeps is then one the pass computed on the recorded text,
    // so the commit below lands the sequence exactly where a round that
    // sampled that text would have. The token after the agreement is the
    // run's successor, so it must be a vocabulary id -- or the pad after a
    // recording that ends at a stop id (kernel/src/teacher_forcing.h).
    if (options.forced_tokens != nullptr) {
      if (in_leaf) {
        err = cudaMemcpyAsync(drafts.data(), verify.drafts->p, drafts.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, model->stream);
        if (err == cudaSuccess) {
          err = cudaStreamSynchronize(model->stream);
        }
        if (err != cudaSuccess) {
          set_error(std::string("ignis_program_decode: reading the drafter's proposals failed: ") +
                    cudaGetErrorString(err));
          return -1;
        }
      }
      for (uint64_t i = 0; i < batch_size; ++i) {
        const std::int32_t *forced = options.forced_tokens + i * caller_columns;
        // GitHub #577 (test-only): a sampling lane forced on the device has
        // its verdict already, the recording's run by the forced coin; one
        // that is not the recording's is the device's fault, and refused.
        const auto lane = static_cast<std::uint32_t>(i);
        if (forced_on_device && samples(i)) {
          if (!ignis_teacher_forced_run_is_recorded(licensed.data() + layout.column_of(lane), forced, accepted[i],
                                                   extents[i])) {
            set_error("ignis_program_decode: lane " + std::to_string(i) + "'s forced verdict on the device "
                      "licensed a run that is not its recording (" + std::to_string(accepted[i]) +
                      " taken of an extent of " + std::to_string(extents[i]) + ")");
            return -1;
          }
          continue;
        }
        const IgnisForcedVerdict verdict = ignis_teacher_forced_verdict(
            anchors[i], drafts.data() + layout.draft_of(lane), forced, extents[i],
            static_cast<std::int32_t>(model->vocab), sampling[i].stop_ids, sampling[i].stop_id_count,
            licensed.data() + layout.column_of(lane));
        if (verdict.refused) {
          set_error("ignis_program_decode: lane " + std::to_string(i) + "'s forced token " +
                    std::to_string(forced[verdict.accepted]) + " after its " +
                    std::to_string(verdict.accepted) +
                    " agreed drafts is outside the vocabulary (the pad -1 ends a recording only "
                    "after a stop id)");
          return -1;
        }
        accepted[i] = verdict.accepted;
      }
    }

    // The host half of the commit: the run, its cut, the fold rows.
    std::vector<std::int32_t> committed(batch_size, 0);
    std::vector<std::int32_t> next_pending(batch_size, -1);
    std::vector<bool> redirected(batch_size, false);
    // GitHub #590: the lanes whose cut the host decides (`host_cut`), read
    // once; NULL on every round no lane of which is held to a grammar.
    const std::uint8_t *host_cut_lanes = options.host_cut != nullptr ? options.host_cut_lanes : nullptr;
    std::vector<ninfer::ops::GdnReplayFoldRow> fold_rows(batch_size);
    std::vector<std::int32_t> selectors(batch_size, 0);
    std::vector<std::int32_t> run(layout.widest_lane_columns(), 0);
    for (uint64_t i = 0; i < batch_size; ++i) {
      const std::int32_t n = accepted[i];
      if (n < 0 || n > extents[i]) {
        set_error("ignis_program_decode: the accept kernel reported " + std::to_string(n) +
                  " accepted drafts for an extent of " + std::to_string(extents[i]));
        return -1;
      }
      const std::int32_t *lane_licensed = licensed.data() + layout.column_of(static_cast<std::uint32_t>(i));
      // run[j]: j == 0 the anchor, else licensed[j-1] (= the j-th draft,
      // accepted); run length n + 1 before the cut, and licensed[n] -- the
      // correction/bonus token -- the successor of its last.
      run[0] = anchors[i];
      for (std::int32_t j = 0; j < n; ++j) {
        run[static_cast<std::size_t>(j) + 1] = lane_licensed[j];
      }
      // GitHub #590: a lane whose cut the host decides keeps the prefix of
      // its run the host returns -- a grammar's, up to its first miss -- and
      // the draft after that prefix waits as its pending token. Every other
      // lane keeps its run whole, which is the cut below as it always was.
      std::int32_t keep = n + 1;
      if (host_cut_lanes != nullptr && host_cut_lanes[i] != 0) {
        keep = options.host_cut(options.host_cut_context, i, run.data(), n + 1);
        if (keep < 1 || keep > n + 1) {
          set_error("ignis_program_decode: the host's cut of lane " + std::to_string(i) + " kept " +
                    std::to_string(keep) + " tokens of a run of " + std::to_string(n + 1) +
                    " (1.." + std::to_string(n + 1) + ", or the host failed)");
          return -1;
        }
      }
      // The cut at the first stop id inclusive, and the successor of the
      // run's last committed token: the draft after it when the target agreed
      // with that draft, else the correction/bonus token. GitHub #315 (ADR
      // 0048): with the lane's reasoning block open, a stop id is cut before
      // instead and `</think>` is pending in its place.
      const IgnisRoundCut cut =
          ignis_cut_host_kept_run(run.data(), lane_licensed, keep, sampling[i].stop_ids,
                                  sampling[i].stop_id_count, sampling[i].reasoning_close_id);
      const std::int32_t c = cut.committed;
      committed[i] = c;
      next_pending[i] = cut.next_pending;
      redirected[i] = cut.redirected;
      for (std::int32_t j = 0; j < c; ++j) {
        out_token_ids[i * caller_columns + j] = run[static_cast<std::size_t>(j)];
      }
      fold_rows[i] = ninfer::ops::GdnReplayFoldRow{.linear_state_slot = slots[i],
                                                   .commit_columns = c};
      selectors[i] = c - 1;
    }

    // GitHub #257 (spec runtime/06): the columns past each lane's commit
    // wrote ring slots that older keys inside the window still name, so their
    // bits come off and those keys fall back to the codec until their
    // positions are appended again. The pass appends all `extent + 1`
    // columns whatever it accepts, so that is the range, not the licensed
    // run. Host masks on the round's stream, applied by the vendored kernel
    // (core/kv_ring_bits.h) outside the verify graph -- they change every
    // round -- and confirmed by the fold's synchronize below, or (GitHub #676)
    // left in flight with the fold.
    // The reference does the same at its commit (program_impl.h:~960).
    if (pool->has_hq_residual()) {
      for (uint64_t i = 0; i < batch_size; ++i) {
        const auto kept    = static_cast<std::uint64_t>(base_positions[i] + committed[i]);
        const auto written = static_cast<std::uint64_t>(base_positions[i] + extents[i] + 1);
        if (kept >= written) {
          continue;
        }
        ninfer::apply_kv_ring_valid_words(pool->hq_ring_words(slots[i]),
                                          ignis_hq_ring_invalidate_mask(kept, written),
                                          ninfer::KvRingWords{}, kIgnisHqRingWords, model->stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          set_error(std::string("ignis_program_decode: hq ring invalidation failed: ") +
                    cudaGetErrorString(err));
          return -1;
        }
      }
    }
    // GitHub #621: so are the MTP head's rows its chain wrote past the
    // commit, in the head's own words: the chain's [P - 1, P - 1 + extent)
    // less the commit's [P - 1, P - 1 + c).
    if (mtp && pool->has_hq_residual()) {
      for (uint64_t i = 0; i < batch_size; ++i) {
        const ignis::mtp::RoundRows rows =
            ignis::mtp::round_rows(static_cast<std::uint64_t>(base_positions[i]), static_cast<std::uint32_t>(extents[i]),
                                   static_cast<std::uint32_t>(committed[i]));
        if (rows.committed_end >= rows.chain_end) {
          continue;
        }
        ninfer::apply_kv_ring_valid_words(pool->hq_head_ring_words(slots[i]),
                                          ignis_hq_ring_invalidate_mask(rows.committed_end, rows.chain_end),
                                          ninfer::KvRingWords{}, kIgnisHqRingWords, model->stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          set_error(std::string("ignis_program_decode: the MTP head's hq ring invalidation failed: ") +
                    cudaGetErrorString(err));
          return -1;
        }
      }
    }

    // The fold: every GDN layer's slot and conv taps rebuilt from the
    // committed prefix of this round's records, in one vendored call over
    // every lane of a group (GitHub #712), at its shape's records: a uniform
    // round's one group, at the round's. A zero-length commit is that op's
    // own strict no-op; no lane asks for one (c >= 1), the anchor is always
    // committed. GitHub #713: a mixed round's base group folds from the mixed
    // base records, its rows behind `first_record_row` leading rows that
    // commit nothing, on slots none of its lanes holds (the op as it is).
    for (const ignis::round_layout::Group &group : layout.groups) {
      if (group.lanes == 0) {
        continue;
      }
      const IgnisVerifyRound *const shape = ignis_round_shape_at(*model, group.window);
      if (shape == nullptr) {
        set_error("ignis_program_decode: no round shape of this load drafts " + std::to_string(group.window));
        return -1;
      }
      // The records hold eight rows at most (`records_hold` keeps a group's within them).
      std::array<ninfer::ops::GdnReplayFoldRow, IGNIS_DECODE_MAX_BATCH> group_rows{};
      std::array<std::int32_t, IGNIS_DECODE_MAX_BATCH> leading{};
      const std::uint32_t rows = group.first_record_row + group.lanes;
      if (rows > group_rows.size() ||
          !ignis::round_layout::leading_fold_slots(slots.data() + group.first_lane, group.lanes,
                                                   group.first_record_row, pool->gdn_pool.slot_count(),
                                                   leading.data())) {
        set_error("ignis_program_decode: the pool has too few slots for the base group's leading fold rows");
        return -1;
      }
      for (std::uint32_t r = 0; r < group.first_record_row; ++r) {
        group_rows[r] = ninfer::ops::GdnReplayFoldRow{.linear_state_slot = leading[r], .commit_columns = 0};
      }
      std::copy_n(fold_rows.data() + group.first_lane, group.lanes, group_rows.data() + group.first_record_row);
      ninfer::ops::gdn_replay_fold(mixed && !group.extended ? model->mixed_base_records : shape->records,
                                   pool->gdn_pool.all_layers_view(),
                                   std::span<const ninfer::ops::GdnReplayFoldRow>(group_rows.data(), rows),
                                   model->stream);
    }

    // GitHub #621: the MTP head's rows the run licenses, [P - 1, P - 1 + c)
    // in each lane, built again from the continuation hidden, the round's
    // committed columns and their tokens: every row the chain built from the
    // head's own hidden below the new frontier is written over. The host
    // arrays live until the synchronize below.
    std::vector<std::int32_t> head_rows;
    // GitHub #624: on a vision load each row rotates at its position plus
    // the lane's rope delta, as the chain's and the target's columns did.
    std::vector<std::int32_t> head_rotation;
    if (mtp) {
      head_rows.resize(layout.columns());
      if (!rope_positions.empty()) {
        head_rotation.resize(head_rows.size());
      }
      for (uint64_t i = 0; i < batch_size; ++i) {
        const auto lane = static_cast<std::uint32_t>(i);
        for (std::uint32_t j = 0; j < layout.window_of(lane) + 1; ++j) {
          const std::size_t column = layout.column_of(lane) + j;
          head_rows[column] =
              base_positions[i] - 1 + std::min<std::int32_t>(static_cast<std::int32_t>(j), committed[i] - 1);
          if (!head_rotation.empty()) {
            head_rotation[column] = head_rows[column] + sequences[i]->rope_delta;
          }
        }
      }
      std::string head_error;
      if (ignis::mtp::commit_rows(*model, *pool, verify, static_cast<std::uint32_t>(batch_size), head_rows.data(),
                                  head_rotation.empty() ? nullptr : head_rotation.data(), committed.data(),
                                  &head_error) != 0) {
        set_error("ignis_program_decode: " + head_error);
        return -1;
      }
    }

    // The accepted hidden state per lane (the run's last column's final
    // residual), kept for the drafter's continuation input (P5-05).
    if (!stage(*verify.selectors, selectors.data(), lane_bytes, "verify selectors") || !flush("verify selectors")) {
      return -1;
    }
    // GitHub #712: a group's lanes at a time, at its columns.
    const std::size_t hidden_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    for (const ignis::round_layout::Group &group : layout.groups) {
      if (group.lanes == 0) {
        continue;
      }
      const auto lanes = static_cast<std::int32_t>(group.lanes);
      const ninfer::Tensor hidden_columns(
          static_cast<std::uint8_t *>(verify.hidden->p) + group.first_column * hidden_bytes, ninfer::DType::BF16,
          {hidden, static_cast<std::int32_t>(group.lane_columns()), lanes, 1});
      const ninfer::Tensor selector_tensor(static_cast<std::int32_t *>(verify.selectors->p) + group.first_lane,
                                           ninfer::DType::I32, {lanes, 1, 1, 1});
      ninfer::Tensor selected(static_cast<std::uint8_t *>(verify.selected_hidden->p) + group.first_lane * hidden_bytes,
                              ninfer::DType::BF16, {hidden, lanes, 1, 1});
      ninfer::ops::speculative_select_accepted_hidden(hidden_columns, selector_tensor, selected,
                                                      model->stream);
    }

    // P5-05 (GitHub #155): the committed columns' feature taps into each
    // lane's window -- its whole committed run, or nothing for a lane at
    // extent 0, whose round read its window and leaves it as it was.
    std::vector<std::int32_t> append_counts(batch_size, 0);
    // GitHub #157: what that lane carries instead -- its anchor column's
    // taps, read back for the sequence's handle.
    std::vector<std::vector<std::uint8_t>> carried_taps(batch_size);
    if (dflash2) {
      for (uint64_t i = 0; i < batch_size; ++i) {
        append_counts[i] = extents[i] == 0 ? 0 : committed[i];
      }
      if (!stage(*verify.append_counts, append_counts.data(), lane_bytes, "drafter append counts") ||
          !flush("drafter append counts")) {
        return -1;
      }
      // GitHub #676: from its graph, on a round that stages.
      if (staging != nullptr) {
        ignis::deferred_verify::append_round(model, pool, verify, layout);
      } else {
        ignis_dflash2_append_round(model, pool, verify, layout);
      }
      const std::size_t column_bytes =
          kDflash2TapLayers.size() * static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
      for (uint64_t i = 0; i < batch_size; ++i) {
        if (extents[i] != 0) {
          continue;
        }
        carried_taps[i].resize(column_bytes);
        const cudaError_t copied = cudaMemcpyAsync(
            carried_taps[i].data(),
            static_cast<const std::uint8_t *>(verify.features->p) +
                layout.column_of(static_cast<std::uint32_t>(i)) * column_bytes,
            column_bytes, cudaMemcpyDeviceToHost, model->stream);
        if (copied != cudaSuccess) {
          set_error(std::string("ignis_program_decode: cudaMemcpyAsync(anchor taps) failed: ") +
                    cudaGetErrorString(copied));
          return -1;
        }
      }
    }

    // The penalty rows: at temperature > 0 the accept kernel counted every
    // licensed token -- the accepted drafts and the correction/bonus token --
    // in the lane's occurrence counts. A cut run keeps only
    // `licensed[0..c)` (its drafts plus the new pending token); the tokens
    // past it were never emitted, so their counts come back off, or the row
    // a snapshot or a prefix clone carries would run ahead of the text. A
    // redirected lane's `licensed[c-1]` is the stop id `</think>` replaced
    // (GitHub #315), so that count moves to `</think>`. Greedy lanes count
    // nothing (the kernel's own contract). Done after the fold is confirmed,
    // as the last device step before the commit, and in one read, one
    // adjustment and one write for every such count (a token licensed twice
    // in one run is one count, taken back twice).
    std::vector<std::pair<std::int32_t *, std::int32_t>> rollback;
    for (uint64_t i = 0; i < batch_size; ++i) {
      if (!(configs[i].temperature > 0.0f) || configs[i].token_counts == nullptr) {
        continue;
      }
      const std::int32_t produced = accepted[i] + 1;
      const std::int32_t *lane_licensed = licensed.data() + layout.column_of(static_cast<std::uint32_t>(i));
      for (std::int32_t j = committed[i]; j < produced; ++j) {
        // GitHub #577: a forced lane's pad after a stop (-1) was never
        // counted; every other licensed token is a vocabulary id.
        if (lane_licensed[j] < 0 || lane_licensed[j] >= static_cast<std::int32_t>(model->vocab)) {
          continue;
        }
        rollback.emplace_back(configs[i].token_counts + lane_licensed[j], -1);
      }
      if (redirected[i]) {
        redirect_penalty_move(configs[i], lane_licensed[committed[i] - 1], next_pending[i], &rollback);
      }
    }

    // GitHub #676: a round that staged returns with its fold in flight when
    // nothing below reads the device on the host (kernel/src/verify_return.h):
    // every lane's state is committed from the host's numbers, every later
    // round and leaf call reaches the device behind the fold on this stream
    // (or the default one), and the pool's windows wait for the event first.
    // Anything else is confirmed here, as earlier releases did.
    const bool lane_at_extent_0 =
        std::any_of(extents.begin(), extents.end(), [](std::int32_t extent) { return extent == 0; });
    if (ignis_verify_fold_stays_in_flight(staging != nullptr, lane_at_extent_0, !rollback.empty())) {
      err = slot_release.release();
      if (err == cudaSuccess) {
        err = cudaEventRecord(pool->transfer.verify_fold, model->stream);
      }
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: leaving the verify round's fold in flight failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
      // The queued work goes to the device now, not at the driver's next
      // submission (Windows batches a stream's launches).
      if (cudaStreamQuery(model->stream) == cudaErrorNotReady) {
        (void)cudaGetLastError();
      }
      ++verify.folds_in_flight;
    } else {
      err = cudaStreamSynchronize(model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaStreamSynchronize(fold) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }
    if (!rollback.empty()) {
      err = adjust_penalty_counts(rollback, model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: penalty count rollback failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }

    // GitHub #621: the MTP head's continuation hidden, sequence state the
    // next round's chain starts from, moves last: a round that fails before
    // here leaves it the hidden of the lane's unmoved frontier.
    if (mtp) {
      std::string head_error;
      if (ignis::mtp::commit_continuation(*model, *pool, verify, static_cast<std::uint32_t>(batch_size),
                                          &head_error) != 0) {
        set_error("ignis_program_decode: " + head_error);
        return -1;
      }
      err = cudaStreamSynchronize(model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: cudaStreamSynchronize(the MTP head's continuation) failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }

    // Only now, with the fold confirmed on the device -- or, GitHub #676,
    // queued behind everything before it and ahead of everything after -- does
    // any lane's state move (the decode round's own ordering).
    uncommitted.armed = false;
    for (uint64_t i = 0; i < batch_size; ++i) {
      sequences[i]->pending_token = next_pending[i];
      // GitHub #586: a verify round's successor keeps no scores a repair
      // reads -- but (#590) a lane whose cut the host decided keeps its pending
      // token's: the verify logits' column the cut ended at, `committed - 1`,
      // whose draw it is (none for a redirected `</think>`). Its penalty count
      // is the accept kernel's, kept by the rollback above.
      sequences[i]->kept = ignis_kept_scores{};
      if (host_cut_lanes != nullptr && host_cut_lanes[i] != 0 && !redirected[i]) {
        ignis_kept_scores &kept = sequences[i]->kept;
        kept.source = IgnisKeptSource::kVerifyLogits;
        kept.column = static_cast<std::int32_t>(layout.column_of(static_cast<std::uint32_t>(i))) + committed[i] - 1;
        kept.epoch = verify.kept_epoch;
        kept.position = base_positions[i] + committed[i] - 1;
        kept.counted = configs[i].temperature > 0.0f && configs[i].token_counts != nullptr;
      }
      advance_frontiers(model, sequences[i], static_cast<std::uint32_t>(committed[i]));
      options.out_committed_counts[i] = committed[i];
      if (options.out_reasoning_redirected != nullptr) {
        options.out_reasoning_redirected[i] = redirected[i] ? 1 : 0;
      }
      if (pending_appended[i]) {
        sequences[i]->dflash2_position += 1;
      }
      if (append_counts[i] > 0) {
        sequences[i]->dflash2_position = sequences[i]->position;
      }
      if (dflash2) {
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
      // GitHub #689: on a load of two round shapes, what the lane asks of its
      // next round (kernel/src/round_shape.h), from this round's shape, its
      // extent and the drafts the target accepted -- not what a stop or the
      // host's cut kept: the drafts' quality, not the grammar's.
      // GitHub #712: at the lane's own shape.
      const bool lane_extended = layout.extended(static_cast<std::uint32_t>(i));
      if (model->extended_shape != nullptr) {
        ignis_seq &seq = *sequences[i];
        seq.asks_extended_shape = ignis::round_shape::asks_extended(model->shape_rule, seq.asks_extended_shape,
                                                                    lane_extended, extents[i], accepted[i]);
        // GitHub #690: and the round, for the shape policy's estimate of the lane.
        ignis::round_shape::record_round(seq.shape_rounds, seq.shape_rounds_recorded, lane_extended, extents[i],
                                         accepted[i]);
      }
      // GitHub #712: and each lane's shape, for the core.
      if (options.out_lane_shapes != nullptr) {
        options.out_lane_shapes[i] = lane_extended ? 1 : 0;
      }
    }
    model->last_step_kernel_count = model->layers.size();
    model->last_step_graph_launches = use_graph ? 1 : 0;
    // GitHub #687: the round in its shape's stats; GitHub #713: a mixed round is
    // neither shape's.
    if (!mixed) {
      ++verify.rounds;
      verify.eager_rounds += use_graph ? 0 : 1;
    }
    round_timing_collect(model, model->verify_timing, "verify");
  } catch (const std::exception &e) {
    set_error(std::string("ignis_program_decode: ") + e.what());
    return -1;
  }
  return 0;
}

// GitHub #690: whether the round of `sequences` runs at the load's extended shape under the shape
// rule and `policy`, from its lanes' asks and their recent rounds, as IGNIS_ROUND_SHAPE's `named`
// allows (kernel/src/round_shape.h `runs_extended`). A load of one shape never does.
bool round_runs_extended(const ignis_model &model, ignis_seq *const *sequences, uint64_t batch_size,
                         ignis::round_shape::Named named, const ignis::round_shape::ShapePolicy &policy) {
  if (model.extended_shape == nullptr) {
    return false;
  }
  std::array<std::uint8_t, IGNIS_DECODE_MAX_BATCH> asks{};
  std::array<ignis::round_shape::LaneEstimate, IGNIS_DECODE_MAX_BATCH> estimates{};
  for (uint64_t i = 0; i < batch_size; ++i) {
    asks[i] = sequences[i] != nullptr && sequences[i]->asks_extended_shape ? 1 : 0;
    if (sequences[i] != nullptr) {
      estimates[i] = ignis::round_shape::estimate_of(sequences[i]->shape_rounds, sequences[i]->shape_rounds_recorded,
                                                     policy.estimate_rounds);
    }
  }
  return ignis::round_shape::runs_extended(named, asks.data(), batch_size, model.extended_shape->lanes, policy,
                                           estimates.data());
}

// GitHub #714 (spec #711): why a round at `layout`'s mix runs at the uniform choice instead
// (IGNIS_MIXED_FALLBACK_*), or -1 where it runs as the mix: where the latest capture asked for the
// mixed graphs and the mix has no graph, its capture having failed or made none of it. A mix
// without a graph never runs eagerly once its graphs were asked; where no capture asked for them,
// a forced mix runs eagerly (the test seam's reference).
int mixed_fallback_reason(const ignis_model &model, const ignis::round_layout::Layout &layout) {
  const int mix = ignis::round_layout::mix_index(layout);
  if (mix < 0 || !model.mixed_capture_asked || model.mixed_verify_exec[static_cast<std::size_t>(mix)] != nullptr) {
    return -1;
  }
  return ((model.mixed_failed_mask >> mix) & 1u) != 0 ? IGNIS_MIXED_FALLBACK_CAPTURE_FAILED
                                                       : IGNIS_MIXED_FALLBACK_NOT_CAPTURED;
}

// GitHub #714: a round of `sequences` whose mix has no graph, at the uniform round shape the shape
// policy `aggregate` would choose for them under the shape rule (IGNIS_ROUND_SHAPE does not reach a
// mix's round), counted under `reason` in the program's stats.
int32_t run_mixed_fallback(ignis_model *model, ignis_seq_pool *pool, ignis_seq *const *sequences, uint64_t batch_size,
                           const ignis_sampling_params *sampling, const ignis_decode_options &options,
                           int32_t *out_token_ids, int reason) {
  ignis::round_shape::ShapePolicy aggregate = model->shape_policy;
  aggregate.policy = ignis::round_shape::Policy::kAggregate;
  IgnisVerifyRound &shape =
      round_runs_extended(*model, sequences, batch_size, ignis::round_shape::Named::kRule, aggregate)
          ? *model->extended_shape
          : *model->base_shape;
  const int32_t rc =
      run_verify_round(model, pool, shape,
                       ignis::round_layout::uniform(static_cast<std::uint32_t>(batch_size), shape.window, shape.extended()),
                       sequences, batch_size, sampling, options, out_token_ids);
  if (rc == 0) {
    ++model->mixed_fallback_rounds[static_cast<std::size_t>(reason)];
  }
  return rc;
}

// GitHub #713 (spec #711, test-only): a verify round at the shapes `options.forced_lane_shapes` names,
// lane i at the extended shape where its entry is nonzero and at the base shape otherwise. Every lane
// at one shape is that uniform round. Lanes of both are a mixed round, its extended lanes first in
// the round's rows and its base lanes after them, each group in the caller's order: every per-lane
// input is read and every output written back at the caller's index, and nothing moves on the
// device (a lane's KV pages, GDN slot and drafter window follow its pool slot). It runs from its
// mix's graph where the latest capture made one and eagerly where no capture asked for the mixed
// graphs, on the branch its base group runs on, made here before its first round where no capture
// made it; GitHub #714: where the latest capture asked for them and made none of this mix, at the
// uniform choice (`run_mixed_fallback`).
int32_t run_forced_mix(ignis_model *model, ignis_seq_pool *pool, ignis_seq *const *sequences, uint64_t batch_size,
                       const ignis_sampling_params *sampling, const ignis_decode_options &options,
                       int32_t *out_token_ids) {
  if (options.forced_shape_window != 0) {
    set_error("ignis_program_decode: forced_lane_shapes and forced_shape_window both name the round's shape");
    return -1;
  }
  if (model->extended_shape == nullptr) {
    set_error("ignis_program_decode: forced_lane_shapes name a round shape of a DFlash2 load of two");
    return -1;
  }
  IgnisVerifyRound &base = *model->base_shape;
  IgnisVerifyRound &extended = *model->extended_shape;
  // The round's rows: row r is the caller's lane rows[r].
  std::array<std::uint32_t, IGNIS_DECODE_MAX_BATCH> rows{};
  std::uint32_t extended_lanes = 0;
  for (std::uint32_t i = 0; i < batch_size; ++i) {
    if (options.forced_lane_shapes[i] != 0) {
      rows[extended_lanes++] = i;
    }
  }
  const auto lanes = static_cast<std::uint32_t>(batch_size);
  if (extended_lanes == 0 || extended_lanes == lanes) {
    IgnisVerifyRound &shape = extended_lanes == 0 ? base : extended;
    if (lanes > shape.lanes) {
      set_error("ignis_program_decode: a round of " + std::to_string(lanes) + " lanes at the round shape of " +
                std::to_string(shape.window) + " drafts, which runs " + std::to_string(shape.lanes) + " at most");
      return -1;
    }
    return run_verify_round(model, pool, shape, ignis::round_layout::uniform(lanes, shape.window, shape.extended()),
                            sequences, batch_size, sampling, options, out_token_ids);
  }
  std::uint32_t next = extended_lanes;
  for (std::uint32_t i = 0; i < batch_size; ++i) {
    if (options.forced_lane_shapes[i] == 0) {
      rows[next++] = i;
    }
  }
  const ignis::round_layout::Layout layout =
      ignis::round_layout::layout_of(extended_lanes, extended.window, lanes - extended_lanes, base.window);
  if (const char *refusal = ignis_mixed_round_refusal(*model, pool->gdn_pool.slot_count(), layout)) {
    set_error(std::string("ignis_program_decode: a mixed round of this load: ") + refusal);
    return -1;
  }
  if (const int reason = mixed_fallback_reason(*model, layout); reason >= 0) {
    return run_mixed_fallback(model, pool, sequences, batch_size, sampling, options, out_token_ids, reason);
  }
  std::string branch_error;
  if (model->mixed_branch.stream == nullptr && !model->mixed_branch.create(&branch_error)) {
    set_error("ignis_program_decode: a mixed round: " + branch_error);
    return -1;
  }
  // Each lane's inputs and outputs in the round's rows.
  const std::size_t caller_k = options.speculative_window;
  const std::size_t caller_columns = caller_k + 1;
  std::array<ignis_seq *, IGNIS_DECODE_MAX_BATCH> row_sequences{};
  std::array<ignis_sampling_params, IGNIS_DECODE_MAX_BATCH> row_sampling{};
  std::vector<std::int32_t> row_tokens(caller_columns * batch_size, -1);
  std::vector<std::int32_t> forced_tokens(options.forced_tokens != nullptr ? caller_columns * batch_size : 0);
  std::vector<std::int32_t> forced_drafts(options.forced_drafts != nullptr ? caller_k * batch_size : 0);
  for (std::uint32_t r = 0; r < lanes; ++r) {
    const std::uint32_t i = rows[r];
    row_sequences[r] = sequences[i];
    row_sampling[r] = sampling[i];
    std::copy_n(out_token_ids + i * caller_columns, caller_columns, row_tokens.data() + r * caller_columns);
    if (!forced_tokens.empty()) {
      std::copy_n(options.forced_tokens + i * caller_columns, caller_columns, forced_tokens.data() + r * caller_columns);
    }
    if (!forced_drafts.empty()) {
      std::copy_n(options.forced_drafts + i * caller_k, caller_k, forced_drafts.data() + r * caller_k);
    }
  }
  std::array<std::int32_t, IGNIS_DECODE_MAX_BATCH> row_committed{};
  std::array<std::uint32_t, IGNIS_DECODE_MAX_BATCH> row_extents{};
  std::array<std::int32_t, IGNIS_DECODE_MAX_BATCH> row_redirected{};
  std::array<std::uint32_t, IGNIS_DECODE_MAX_BATCH> row_shapes{};
  ignis_decode_options row_options = options;
  row_options.out_committed_counts = row_committed.data();
  row_options.out_extents = options.out_extents != nullptr ? row_extents.data() : nullptr;
  row_options.out_reasoning_redirected = options.out_reasoning_redirected != nullptr ? row_redirected.data() : nullptr;
  row_options.out_lane_shapes = options.out_lane_shapes != nullptr ? row_shapes.data() : nullptr;
  row_options.forced_tokens = forced_tokens.empty() ? nullptr : forced_tokens.data();
  row_options.forced_drafts = forced_drafts.empty() ? nullptr : forced_drafts.data();
  row_options.forced_lane_shapes = nullptr;
  const int32_t rc = run_verify_round(model, pool, base, layout, row_sequences.data(), batch_size,
                                      row_sampling.data(), row_options, row_tokens.data());
  if (rc != 0) {
    return rc;
  }
  for (std::uint32_t r = 0; r < lanes; ++r) {
    const std::uint32_t i = rows[r];
    std::copy_n(row_tokens.data() + r * caller_columns, caller_columns, out_token_ids + i * caller_columns);
    options.out_committed_counts[i] = row_committed[r];
    if (options.out_extents != nullptr) {
      options.out_extents[i] = row_extents[r];
    }
    if (options.out_reasoning_redirected != nullptr) {
      options.out_reasoning_redirected[i] = row_redirected[r];
    }
    if (options.out_lane_shapes != nullptr) {
      options.out_lane_shapes[i] = row_shapes[r];
    }
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
  // GitHub #302: a Flash-Next load runs its own program.
  if (model != nullptr && model->flash_next) {
    // GitHub #267: teacher forcing is the 27B verify round's; Flash-Next's
    // program has none, and a forced round run there would commit what its
    // target samples. Refused before the dispatch -- read only from options
    // of this leaf's size, a shorter one is the program's own to refuse.
    if (options != nullptr && options->size == sizeof(struct ignis_decode_options) &&
        options->forced_tokens != nullptr) {
      set_error("ignis_program_decode: forced_tokens is the 27B verify round's; a Flash-Next load has none");
      return -1;
    }
    // GitHub #692: and so are the forced drafts and the forced shape (GitHub
    // #713: and each lane's).
    if (options != nullptr && options->size == sizeof(struct ignis_decode_options) &&
        (options->forced_drafts != nullptr || options->forced_shape_window != 0 ||
         options->forced_lane_shapes != nullptr)) {
      set_error("ignis_program_decode: forced_drafts and forced_shape_window are the 27B verify round's; a "
                "Flash-Next load has none");
      return -1;
    }
    // GitHub #590: so is a host cut -- Flash-Next keeps no scores a repair
    // reads (#581), so a cut there would leave a miss nobody can repair.
    if (options != nullptr && options->size == sizeof(struct ignis_decode_options) &&
        (options->host_cut != nullptr || options->host_cut_lanes != nullptr)) {
      set_error("ignis_program_decode: a host cut is the 27B verify round's; a Flash-Next load has none "
                "(GitHub #581)");
      return -1;
    }
    return ignis::flash_next::program_decode(model, pool, sequences, batch_size, sampling, out_token_ids,
                                             options);
  }
  if (model == nullptr || pool == nullptr || sequences == nullptr || sampling == nullptr ||
      out_token_ids == nullptr || batch_size == 0) {
    set_error("ignis_program_decode: null argument or empty batch");
    return -1;
  }
  // GitHub #336: a round as wide as the load's lanes at most -- every
  // round buffer, the verify round's included, is sized for them.
  if (batch_size > model->decode_lanes) {
    set_error("ignis_program_decode: batch_size " + std::to_string(batch_size) +
              " exceeds the load's decode lanes (" + std::to_string(model->decode_lanes) + ")");
    return -1;
  }
  if (!drafter_matches_load(model, pool, "ignis_program_decode")) {
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
    // GitHub #329 item 5: the round's logits copy is a Flash-Next round's; refused here rather than
    // left unwritten.
    if (options->out_logits != nullptr) {
      set_error("ignis_program_decode: out_logits is read on a Flash-Next load only");
      return -1;
    }
    // GitHub #267: teacher forcing replaces a verify round's accept; a plain
    // round has none to replace.
    if (options->speculative_window == 0 && options->forced_tokens != nullptr) {
      set_error("ignis_program_decode: forced_tokens needs a verify round (speculative_window > 0)");
      return -1;
    }
    // GitHub #713: nor forced lane shapes, which name a verify round's.
    if (options->speculative_window == 0 && options->forced_lane_shapes != nullptr) {
      set_error("ignis_program_decode: forced_lane_shapes need a verify round (speculative_window > 0)");
      return -1;
    }
    // GitHub #590: a host cut names its lanes and its callback together, cuts
    // a verify round's run (a plain round's run is its anchor alone), and
    // never a teacher-forced one, whose run is the recording's.
    if ((options->host_cut == nullptr) != (options->host_cut_lanes == nullptr)) {
      set_error("ignis_program_decode: host_cut and host_cut_lanes are set together or not at all");
      return -1;
    }
    if (options->host_cut != nullptr && options->speculative_window == 0) {
      set_error("ignis_program_decode: a host cut needs a verify round (speculative_window > 0)");
      return -1;
    }
    if (options->host_cut != nullptr && options->forced_tokens != nullptr) {
      set_error("ignis_program_decode: a teacher-forced round commits its recording; it takes no host cut");
      return -1;
    }
    // GitHub #692 (test-only): the forced drafts and the forced shape name a
    // teacher-forced round's drafts and shape, so they ride one.
    if ((options->forced_drafts != nullptr || options->forced_shape_window != 0) &&
        options->forced_tokens == nullptr) {
      set_error("ignis_program_decode: forced_drafts and forced_shape_window ride a teacher-forced round "
                "(forced_tokens)");
      return -1;
    }
    if (options->speculative_window != 0) {
      // GitHub #686: the round's shape, chosen here and handed to the round.
      // GitHub #687: the load's window is its widest, its extended shape's on
      // a load of two.
      IgnisVerifyRound *shape = model->base_shape.get();
      const uint32_t loaded = shape == nullptr ? 0 : model->draft_tokens;
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
      // GitHub #690: how many of the round's lanes ask the extended shape going
      // into it, before its outcome moves their asks.
      if (options->out_shape_asks != nullptr) {
        std::uint32_t asking = 0;
        for (uint64_t i = 0; model->extended_shape != nullptr && i < batch_size; ++i) {
          asking += sequences[i] != nullptr && sequences[i]->asks_extended_shape ? 1 : 0;
        }
        *options->out_shape_asks = asking;
      }
      // GitHub #713 (test-only): a round whose lanes' shapes are forced runs
      // at them, a mixed round where they are both, and nothing below chooses.
      if (options->forced_lane_shapes != nullptr) {
        const auto began = std::chrono::steady_clock::now();
        const int32_t rc = run_forced_mix(model, pool, sequences, batch_size, sampling, *options, out_token_ids);
        if (rc != 0) {
          return rc;
        }
        model->last_step_micros = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - began).count());
        return 0;
      }
      // GitHub #692 (test-only): a teacher-forced round that names its shape
      // runs at it, and nothing below chooses one for it.
      // GitHub #687, #689, #690, #691: on a load of two round shapes, the
      // round runs at the extended shape when the shape policy gives it from
      // its lanes' asks and recent rounds, a lane the host cuts among them,
      // and it has no more lanes than that shape runs (kernel/src/round_shape.h);
      // IGNIS_ROUND_SHAPE, read here at each round, forces either shape over
      // the rule and the policy.
      if (options->forced_shape_window != 0) {
        shape = ignis_round_shape_at(*model, options->forced_shape_window);
        if (shape == nullptr) {
          set_error("ignis_program_decode: forced_shape_window " + std::to_string(options->forced_shape_window) +
                    " is no round shape of this load (its windows: " + std::to_string(model->base_shape->window) +
                    (model->extended_shape != nullptr ? " and " + std::to_string(model->extended_shape->window)
                                                      : std::string()) +
                    ")");
          return -1;
        }
        if (batch_size > shape->lanes) {
          set_error("ignis_program_decode: a round of " + std::to_string(batch_size) +
                    " lanes at the round shape of " + std::to_string(shape->window) + " drafts, which runs " +
                    std::to_string(shape->lanes) + " at most");
          return -1;
        }
      } else if (model->extended_shape != nullptr) {
        ignis::round_shape::Named named = ignis::round_shape::Named::kRule;
        try {
          named = ignis::round_shape::read_switch();
        } catch (const std::invalid_argument &e) {
          set_error(std::string("ignis_program_decode: ") + e.what());
          return -1;
        }
        // GitHub #690: from each lane's ask and what its recent rounds say, which `aggregate` reads.
        if (round_runs_extended(*model, sequences, batch_size, named, model->shape_policy)) {
          shape = model->extended_shape.get();
        }
      }
      const auto began = std::chrono::steady_clock::now();
      // GitHub #712: every lane at the round's shape.
      const ignis::round_layout::Layout layout =
          ignis::round_layout::uniform(static_cast<std::uint32_t>(batch_size), shape->window, shape->extended());
      const int32_t rc =
          run_verify_round(model, pool, *shape, layout, sequences, batch_size, sampling, *options, out_token_ids);
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
      if (const char *refusal = ignis_seq_restore_refusal(*seq)) {
        set_error(std::string("ignis_program_decode: ") + refusal);
        return -1;
      }
      if (seq->position >= ignis_seq_token_capacity(*seq)) {
        set_error("ignis_program_decode: sequence reached its KV capacity");
        return -1;
      }
      if (!redirect_inputs_ok(sampling[i])) {
        set_error("ignis_program_decode: stop_ids is null with a nonzero stop_id_count at index " +
                  std::to_string(i));
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

    // GitHub #586: the round writes its lanes' rows into the decode logits'
    // first `batch_size` columns, which a repair reads as the scores each
    // lane's draw came from. Stamped before the work is enqueued, so a round
    // that fails after writing them still leaves no sequence believing the
    // scores it kept there are its own.
    std::array<std::uint64_t, IGNIS_DECODE_MAX_BATCH> kept_epochs{};
    for (uint64_t i = 0; i < batch_size; ++i) {
      kept_epochs[i] = stamp_kept_column(model, static_cast<std::int32_t>(i));
    }

    if (use_graph) {
      round_timing_mark(model, model->plain_timing, true);
      err = cudaGraphLaunch(model->decode_graph_exec[batch_size - 1], model->stream);
      round_timing_mark(model, model->plain_timing, false);
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
    round_timing_collect(model, model->plain_timing, "plain");

    // GitHub #315 (ADR 0048): each lane's draw, after the anchor this round
    // fed -- a stop id drawn with the lane's reasoning block open becomes
    // `</think>`, and its counted occurrence moves with it.
    std::vector<IgnisRoundCut> cuts(batch_size);
    std::vector<std::pair<std::int32_t *, std::int32_t>> moves;
    for (uint64_t i = 0; i < batch_size; ++i) {
      cuts[i] = ignis_cut_and_redirect(&emitted[i], 1, successors[i], sampling[i].stop_ids,
                                       sampling[i].stop_id_count, sampling[i].reasoning_close_id);
      if (cuts[i].redirected) {
        redirect_penalty_move(configs[i], successors[i], cuts[i].next_pending, &moves);
      }
    }
    if (!moves.empty()) {
      err = adjust_penalty_counts(moves, model->stream);
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_decode: moving a redirected draw's penalty count failed: ") +
                  cudaGetErrorString(err));
        return -1;
      }
    }
    if (options != nullptr && options->out_reasoning_redirected != nullptr) {
      for (uint64_t i = 0; i < batch_size; ++i) {
        options->out_reasoning_redirected[i] = cuts[i].redirected ? 1 : 0;
      }
    }

    for (uint64_t i = 0; i < batch_size; ++i) {
      out_token_ids[i] = emitted[i];
      sequences[i]->pending_token = cuts[i].next_pending;
      // GitHub #586: where this draw's scores are kept -- none for a draw the
      // redirect replaced, whose pending token is the close.
      sequences[i]->kept = ignis_kept_scores{};
      if (!cuts[i].redirected) {
        sequences[i]->kept.column = static_cast<std::int32_t>(i);
        sequences[i]->kept.epoch = kept_epochs[i];
        sequences[i]->kept.position = positions[i];
        sequences[i]->kept.counted = draw_counts(sampling[i]);
      }
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

// GitHub #586 (spec #287): a **repair** of a sequence's pending token -- the
// contract is in ignis_step.h. Fail-closed everywhere: a repair it cannot make
// exactly is refused with the sequence as it was, never approximated.
extern "C" int32_t ignis_program_repair(struct ignis_model *model, struct ignis_seq_pool *pool,
                                          struct ignis_seq *seq,
                                          const struct ignis_sampling_params *sampling,
                                          const uint64_t *permitted_mask,
                                          uint64_t permitted_mask_words, int32_t *out_token_id) {
  if (model == nullptr || pool == nullptr || seq == nullptr || sampling == nullptr ||
      permitted_mask == nullptr || out_token_id == nullptr) {
    set_error("ignis_program_repair: null argument");
    return -1;
  }
  if (model->flash_next) {
    set_error("ignis_program_repair: a Flash-Next load keeps no scores for a repair (GitHub #581)");
    return -1;
  }
  if (!sampling_size_ok(*sampling)) {
    set_error("ignis_program_repair: unrecognized ignis_sampling_params size " +
              std::to_string(sampling->size));
    return -1;
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    set_error("ignis_program_repair: the sequence was not drawn from this pool");
    return -1;
  }
  if (const char *refusal = ignis_seq_restore_refusal(*seq)) {
    set_error(std::string("ignis_program_repair: ") + refusal);
    return -1;
  }
  // The short form rides the call that draws (a prefill or a round); a
  // repair's permitted set is its mask, and two sets for one draw would leave
  // the caller to guess which one held.
  if (!unconstrained(*sampling)) {
    set_error("ignis_program_repair: a repair's permitted set is its mask; permitted_ids rides a "
              "prefill or a round");
    return -1;
  }
  // A repair draws inside an answer, after the reasoning block: its token is
  // the mask's, and nothing turns it into a `</think>`.
  if (sampling->reasoning_close_id >= 0) {
    set_error("ignis_program_repair: a repair redirects nothing (reasoning_close_id must be -1)");
    return -1;
  }
  if (seq->pending_token < 0) {
    set_error("ignis_program_repair: the sequence has no pending token to repair");
    return -1;
  }
  const auto vocab = static_cast<std::int32_t>(model->vocab);
  const uint64_t words = (static_cast<uint64_t>(vocab) + 63) / 64;
  if (permitted_mask_words != words) {
    set_error("ignis_program_repair: a mask of " + std::to_string(permitted_mask_words) +
              " words for a vocabulary of " + std::to_string(vocab) + " columns, which takes " +
              std::to_string(words));
    return -1;
  }
  const auto tail_bits = static_cast<std::uint32_t>(vocab % 64);
  if (tail_bits != 0 && (permitted_mask[words - 1] >> tail_bits) != 0) {
    set_error("ignis_program_repair: the mask permits a column past the vocabulary");
    return -1;
  }
  bool permits_any = false;
  for (uint64_t w = 0; w < words && !permits_any; ++w) {
    permits_any = permitted_mask[w] != 0;
  }
  if (!permits_any) {
    set_error("ignis_program_repair: the mask permits no token");
    return -1;
  }
  const ignis_kept_scores kept = seq->kept;
  if (kept.column < 0) {
    set_error("ignis_program_repair: the pending token kept no scores (a verify round's the host "
              "did not cut, a redirected one, or a sequence restored or claimed since its draw)");
    return -1;
  }
  if (!kept_scores_current(model, kept)) {
    set_error("ignis_program_repair: the scores the pending token came from were written over by a "
              "later round or prefill; a repair follows the call that drew");
    return -1;
  }

  std::int32_t *counts = pool->token_counts_for(seq->slot);
  const std::int32_t drawn = seq->pending_token;
  // The miss's count is taken back first, so the repair reads the counts the
  // draw read; the sampler then counts the repaired token itself, in its
  // temperature branch, and the position counts once.
  if (kept.counted) {
    const cudaError_t err = adjust_penalty_counts({{counts + drawn, -1}}, model->stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_program_repair: taking back the draw's penalty count failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
  }
  std::int32_t repaired = -1;
  bool sampled = false;
  {
    ninfer::DeviceArena::Scope scope = model->scratch->scope();
    try {
      // The kept row is copied rather than masked in place, so a second
      // repair of the same position reads the scores the draw read.
      ninfer::Tensor row = model->scratch->alloc(ninfer::DType::BF16, {vocab, 1, 1, 1});
      ninfer::Tensor mask = model->scratch->alloc(ninfer::DType::I64,
                                                  {static_cast<std::int32_t>(words), 1, 1, 1});
      cudaError_t err = cudaMemcpyAsync(row.data, kept_scores_row(model, kept),
                                        static_cast<std::size_t>(vocab) * sizeof(std::uint16_t),
                                        cudaMemcpyDeviceToDevice, model->stream);
      if (err == cudaSuccess) {
        err = cudaMemcpyAsync(mask.data, permitted_mask, words * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice, model->stream);
      }
      if (err != cudaSuccess) {
        set_error(std::string("ignis_program_repair: cudaMemcpyAsync(kept scores, mask) failed: ") +
                  cudaGetErrorString(err));
      } else if (ignis_permit_mask_bits(row.data, vocab, static_cast<const std::uint64_t *>(mask.data),
                                        model->stream) != 0) {
        set_error("ignis_program_repair: mask launch failed");
      } else if (sample_single(model, pool, seq, row, *sampling, kIgnisSamplePurposeRepair,
                               kept.position, &repaired, nullptr) == 0) {
        err = cudaStreamSynchronize(model->stream);
        if (err != cudaSuccess) {
          set_error(std::string("ignis_program_repair: cudaStreamSynchronize failed: ") +
                    cudaGetErrorString(err));
        } else {
          sampled = true;
        }
      }
    } catch (const std::exception &e) {
      set_error(std::string("ignis_program_repair: ") + e.what());
    }
  }
  const bool in_vocab = repaired >= 0 && repaired < vocab;
  const bool permitted =
      sampled && in_vocab &&
      ((permitted_mask[static_cast<std::size_t>(repaired) / 64] >> (repaired % 64)) & 1ull) != 0;
  if (!permitted) {
    // The sequence goes back to what the draw left: its count, and nothing of
    // the repaired token's (the sampler counted it if it drew at all).
    std::vector<std::pair<std::int32_t *, std::int32_t>> moves;
    if (kept.counted) {
      moves.emplace_back(counts + drawn, 1);
    }
    if (sampled && in_vocab && draw_counts(*sampling)) {
      moves.emplace_back(counts + repaired, -1);
    }
    const cudaError_t err = moves.empty() ? cudaSuccess : adjust_penalty_counts(moves, model->stream);
    if (sampled) {
      // Only a row with no permitted column in reach draws outside its mask:
      // the short form the drawing call carried and this mask do not meet.
      set_error("ignis_program_repair: the repair drew " + std::to_string(repaired) +
                ", which its mask does not permit -- the scores it was kept from leave no "
                "permitted column in reach (a short permitted set the draw carried and this "
                "mask do not meet)");
    }
    if (err != cudaSuccess) {
      set_error(g_last_error + "; restoring the penalty counts failed: " + cudaGetErrorString(err));
    }
    return -1;
  }
  seq->pending_token = repaired;
  seq->kept.counted = draw_counts(*sampling);
  *out_token_id = repaired;
  return 0;
}

namespace {

// GitHub #686 (spec #684): one round shape's verify round, as
// `ignis_program_stats` reports it. GitHub #687: `shared_with` the shape
// whose buffers it holds too, none for a load's base shape.
ignis_round_shape_stats round_shape_stats(const IgnisVerifyRound &shape, const IgnisVerifyRound *shared_with) {
  ignis_round_shape_stats out{};
  out.window = shape.window;
  out.lanes = shape.lanes;
  out.vram_bytes = shape.device_bytes(shared_with);
  for (uint32_t width = 1; width <= IGNIS_DECODE_MAX_BATCH; ++width) {
    if (shape.graph_ready[width - 1]) {
      out.verify_graph_ready_mask |= 1u << (width - 1);
      out.verify_graph_nodes[width - 1] = shape.graph_nodes[width - 1];
      out.verify_graph_copy_nodes[width - 1] = shape.graph_copy_nodes[width - 1];
    }
    if (shape.append_exec[width - 1] != nullptr) {
      out.drafter_append_graph_ready_mask |= 1u << (width - 1);
      out.drafter_append_graph_nodes[width - 1] = shape.append_nodes[width - 1];
    }
  }
  out.verify_folds_in_flight = shape.folds_in_flight;
  out.verify_rounds = shape.rounds;
  out.eager_verify_rounds = shape.eager_rounds;
  return out;
}

} // namespace

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
      model->decode_graph_scratch->capacity();
  // GitHub #302: a Flash-Next model stages its decode graphs' token ids and
  // slots in its own buffers (`FlashNextModel::token_ids`, `slots`, counted
  // in its activations below), so these two are the 27B's alone.
  if (model->decode_graph_token_ids != nullptr) {
    out_stats->vram_bytes += model->decode_graph_token_ids->bytes;
  }
  if (model->decode_graph_slots != nullptr) {
    out_stats->vram_bytes += model->decode_graph_slots->bytes;
  }
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
  // is `decode_graph_scratch` above, sized for it at load). GitHub #687: a
  // load's extended shape holds them too, and nothing of its own.
  if (model->base_shape != nullptr) {
    out_stats->vram_bytes += model->base_shape->device_bytes();
  }
  // GitHub #177: the media embedding pool (its weights are already in
  // `model->vram_bytes`, and its encoder workspace in `scratch`, GitHub #212)
  // -- on the device only: a host-mapped one holds no VRAM (GitHub #415).
  out_stats->vram_bytes += model->vision_pool.device_bytes();
  // GitHub #302: Flash-Next's activations and MoE buffers, and its pool's
  // indexer and n-gram sections.
  if (model->flash_next) {
    out_stats->vram_bytes += ignis::flash_next::reserved(*model).activation_bytes;
  }
  if (pool->has_indexer()) {
    out_stats->vram_bytes += pool->indexer_state->bytes;
  }
  if (pool->has_ngram_conv()) {
    out_stats->vram_bytes += pool->ngram_conv->bytes;
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
    if (model->base_shape != nullptr && model->base_shape->graph_ready[width - 1]) {
      verify_ready_mask |= (1u << (width - 1));
    }
  }
  // GitHub #307: Flash-Next keeps its own verify round (kernel/src/flash_next/verify.h).
  if (model->flash_next) {
    verify_ready_mask = ignis::flash_next::verify_ready_mask(*model);
  }
  out_stats->decode_graph_ready_mask = ready_mask;
  out_stats->verify_graph_ready_mask = verify_ready_mask;
  // GitHub #572: the 27B's capture alone counts nodes (kernel/src/decode_graph.cu); Flash-Next's
  // leaves them at 0.
  for (uint32_t width = 1; width <= IGNIS_DECODE_MAX_BATCH; ++width) {
    out_stats->decode_graph_nodes[width - 1] =
        model->decode_graph_ready[width - 1] ? model->decode_graph_nodes[width - 1] : 0;
    out_stats->decode_graph_copy_nodes[width - 1] =
        model->decode_graph_ready[width - 1] ? model->decode_graph_copy_nodes[width - 1] : 0;
    out_stats->verify_graph_nodes[width - 1] =
        model->base_shape != nullptr && model->base_shape->graph_ready[width - 1]
            ? model->base_shape->graph_nodes[width - 1]
            : 0;
    out_stats->verify_graph_copy_nodes[width - 1] =
        model->base_shape != nullptr && model->base_shape->graph_ready[width - 1]
            ? model->base_shape->graph_copy_nodes[width - 1]
            : 0;
  }
  // GitHub #676: the verify rounds that returned with their fold in flight, and the drafter's append
  // graphs; a Flash-Next load has no `verify`.
  out_stats->verify_folds_in_flight = model->base_shape != nullptr ? model->base_shape->folds_in_flight : 0;
  out_stats->drafter_append_graph_ready_mask = 0;
  for (uint32_t width = 1; model->base_shape != nullptr && width <= IGNIS_DECODE_MAX_BATCH; ++width) {
    if (model->base_shape->append_exec[width - 1] != nullptr) {
      out_stats->drafter_append_graph_ready_mask |= (1u << (width - 1));
    }
  }
  // GitHub #681: the gating's home; Flash-Next's load never captures the 27B's rounds.
  out_stats->gdn_gating = static_cast<uint32_t>(model->gdn_gating_home);
  // GitHub #686: each round shape's own, the base shape first, then (GitHub
  // #687) the extended one where the load has it; a Flash-Next load has none.
  out_stats->round_shapes = 0;
  for (ignis_round_shape_stats &shape : out_stats->shapes) {
    shape = ignis_round_shape_stats{};
  }
  if (model->base_shape != nullptr) {
    out_stats->shapes[out_stats->round_shapes++] = round_shape_stats(*model->base_shape, nullptr);
  }
  if (model->extended_shape != nullptr) {
    out_stats->shapes[out_stats->round_shapes++] =
        round_shape_stats(*model->extended_shape, model->base_shape.get());
  }
  // GitHub #713: each mix's graphs.
  static_assert(IGNIS_ROUND_MIXES == ignis::round_layout::kMixes, "the ABI's mixes are the layout's");
  for (std::uint32_t i = 0; i < IGNIS_ROUND_MIXES; ++i) {
    out_stats->mixed_verify_graph_nodes[i] = model->mixed_verify_nodes[i];
    out_stats->mixed_append_graph_nodes[i] = model->mixed_append_nodes[i];
    // GitHub #714: and their bytes.
    out_stats->mixed_graph_bytes[i] = model->mixed_graph_bytes[i];
  }
  // GitHub #714: what the latest capture asked and could not capture, and the rounds that ran at
  // the uniform choice for a mix with no graph.
  out_stats->mixed_graphs_asked = model->mixed_capture_asked ? 1 : 0;
  out_stats->mixed_graph_failed_mask = model->mixed_failed_mask;
  for (std::uint32_t reason = 0; reason < IGNIS_MIXED_FALLBACK_REASONS; ++reason) {
    out_stats->mixed_fallback_rounds[reason] = model->mixed_fallback_rounds[reason];
  }
  return 0;
}

// GitHub #302: the program pieces Flash-Next's program shares
// (step_internal.h), the 27B's own above.
namespace ignis::step {

void set_error(std::string message) {
  ::set_error(std::move(message));
}

bool sampling_size_ok(const ignis_sampling_params &sampling) {
  return ::sampling_size_ok(sampling);
}

bool unconstrained(const ignis_sampling_params &sampling) {
  return ::unconstrained(sampling);
}

ninfer::ops::SamplingConfig to_sampling_config(const ignis_sampling_params &abi,
                                               std::int32_t *token_counts) {
  return ::to_sampling_config(abi, token_counts);
}

int32_t sample_single(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                      const ninfer::Tensor &logits, const ignis_sampling_params &sampling,
                      std::int32_t purpose, std::int32_t position, int32_t *out_token_id,
                      float *out_permitted_prob) {
  return ::sample_single(model, pool, seq, logits, sampling, purpose, position, out_token_id,
                         out_permitted_prob);
}

bool redirect_inputs_ok(const ignis_sampling_params &sampling) {
  return ::redirect_inputs_ok(sampling);
}

cudaError_t adjust_penalty_counts(const std::vector<std::pair<std::int32_t *, std::int32_t>> &moves,
                                  cudaStream_t stream) {
  return ::adjust_penalty_counts(moves, stream);
}

void redirect_penalty_move(const ninfer::ops::SamplingConfig &cfg, std::int32_t drawn,
                           std::int32_t close_id,
                           std::vector<std::pair<std::int32_t *, std::int32_t>> *moves) {
  ::redirect_penalty_move(cfg, drawn, close_id, moves);
}

int32_t redirect_span_draw(ignis_seq_pool *pool, ignis_seq *seq, const ignis_sampling_params &sampling,
                           cudaStream_t stream, int32_t *out_redirected) {
  return ::redirect_span_draw(pool, seq, sampling, stream, out_redirected);
}

int32_t next_prompt_token(const ignis_prefill_options *options) {
  return options != nullptr ? options->next_prompt_token : -1;
}

bool span_draws(const ignis_prefill_options *options) {
  return next_prompt_token(options) < 0 || options->keeps_draw != 0;
}

std::string next_prompt_token_refusal(std::int32_t vocab, const ignis_sampling_params &sampling,
                                      const ignis_prefill_options *options, const float *out_logits) {
  const std::int32_t next = next_prompt_token(options);
  if (next == -1) {
    return {};
  }
  const std::string named = "a span with a next prompt token (" + std::to_string(next) + ") ";
  if (next < 0 || next >= vocab) {
    return "next prompt token " + std::to_string(next) + " is neither -1 nor a vocabulary id below " +
           std::to_string(vocab);
  }
  if (options->route != IGNIS_PREFILL_ROUTE_CHUNKED) {
    return named + "runs the chunked route";
  }
  if (out_logits != nullptr || options->out_span_logits != nullptr) {
    return named + "draws nothing, so it has no logits to read; only the prompt's last span does";
  }
  if (!::unconstrained(sampling) || sampling.reasoning_close_id >= 0) {
    return named + "draws nothing, so a permitted set or a reasoning redirect has no draw to act on";
  }
  if ((options->attention_gqa_ordinal >= 0 && options->out_attention_scores != nullptr) ||
      options->attention_set_count != 0 || options->out_attention_set_rows != nullptr) {
    return named + "is not the prompt's last, whose last position an attention readout reads";
  }
  return {};
}

}  // namespace ignis::step
