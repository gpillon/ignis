// ignis kernel leaf -- the Flash-Next verify round's host side and its graphs (spec flash-next/07
// phase C, GitHub #307; OURS, ADR 0043): ignis_program_decode with a speculative window on a
// Flash-Next load. verify.h describes the round; this file stages it, runs its two passes (from
// the width's graphs when captured, eagerly otherwise), cuts each lane's run, and commits it.
//
// Per lane i, with p its frontier, a its pending token (the anchor) and k the width's window:
//   extent  e = min(k, draft_counts[i], remaining_tokens - 1); no drafts proposes nothing (e = 0)
//   columns [a, d_1 .. d_e, a ..] at positions p .. p + k (columns past e are no transition: the
//           GDN records stop at e + 1, and nothing they write is read before it is rewritten)
//   accept  n drafts (n <= e), licensed = [d_1 .. d_n, t*] (t* the correction or bonus token)
//   run     [a, d_1 .. d_n], cut at its first stop id inclusive: c tokens
//   commit  c columns: the frontier moves p -> p + c, pending <- licensed[c - 1]
// so a lane stands exactly where c one-token rounds over the same text leave it. The round is
// atomic: no lane's frontier moves unless both passes succeeded.

#include "program.h"

#include "bind.h"
#include "embed_head.h"
#include "verify.h"

#include "ignis_seq_internal.h"
#include "../step_internal.h"

#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

namespace ignis::flash_next {

namespace {

bool check_cuda(cudaError_t err, const std::string &what, std::string *error) {
  if (err == cudaSuccess) return true;
  *error = what + " failed: " + cudaGetErrorString(err);
  return false;
}

int32_t gdn_layers_of(const FlashNextModel &fn) {
  int32_t gdn = 0;
  for (const LayerWeights &layer : fn.weights->layers) gdn += layer.attention ? 0 : 1;
  return gdn;
}

// The pass of `width` lanes at its window: save, the forward over k + 1 columns per lane, the head
// on every column, the accept. What a pass graph holds.
int32_t run_pass(ignis_model *model, FlashNextModel &fn, const Context &ctx, const verify::Sections &sections,
                 uint32_t width, std::string *error) {
  verify::State &state = *fn.verify;
  const uint32_t window = state.window(width);
  cudaStream_t stream = model->stream;
  ninfer::DeviceArena &scratch = *model->decode_graph_scratch;
  auto scope = scratch.scope();
  const auto *slots = static_cast<const int32_t *>(fn.slots->p);
  const auto *positions = static_cast<const int32_t *>(fn.positions->p);
  if (verify::save(state, sections, fn.g, width, window, slots, positions, stream, error) != 0) return -1;
  Batch batch;
  batch.lanes = static_cast<int32_t>(width);
  batch.tokens = static_cast<int32_t>(window + 1);
  batch.slots = slots;
  batch.positions = positions;
  batch.max_visible = decode_max_visible(fn);
  batch.verify = &state.records;
  if (forward(fn, ctx, batch, IGNIS_RESIDENCY_DECODE, scratch, stream, error) != 0) return -1;
  if (fn_head(fn.g, fn.weights->final_mixer, fn.weights->head, fn.residual->p, batch.rows(),
              model->sampling_decode_logits->p, scratch, stream) != 0) {
    *error = std::string("the head: ") + fn_last_error();
    return -1;
  }
  return verify::accept(state, fn.g, width, window, model->sampling_decode_logits->p, model->sampling_decode_configs->p,
                        stream, error);
}

// The MTP head's drafting between the fold and the restore (spec flash-next/07 phase D): the
// alignment over the pass's stacks and licensed tokens -- entry p + j from the stack at p + j and
// t[p + j + 1] = licensed[j], the committed ones kept -- then each lane's first draft from its last
// committed column's S', the head's section back at the new frontier, then k - 1 chained steps,
// each from the previous step's S' and draft at the next position past the frontier.
int32_t run_drafts(ignis_model *model, FlashNextModel &fn, const Context &ctx, const verify::Sections &sections,
                   uint32_t width, std::string *error) {
  verify::State &state = *fn.verify;
  const uint32_t window = state.window(width);
  const auto lanes = static_cast<int32_t>(width);
  const auto columns = static_cast<int32_t>(window + 1);
  const auto k = static_cast<int32_t>(window);
  cudaStream_t stream = model->stream;
  ninfer::DeviceArena &scratch = *model->decode_graph_scratch;
  auto scope = scratch.scope();
  Batch batch;
  batch.lanes = lanes;
  batch.tokens = columns;
  batch.slots = static_cast<const int32_t *>(fn.slots->p);
  batch.positions = static_cast<const int32_t *>(fn.positions->p);
  batch.max_visible = decode_max_visible(fn);
  batch.verify = &state.records;
  auto *picks = static_cast<int32_t *>(state.picks->p);
  auto *chain_tokens = static_cast<int32_t *>(state.chain_tokens->p);
  auto *drafts = static_cast<int32_t *>(state.drafts_out->p);
  void *logits = model->sampling_decode_logits->p;
  const auto head = [&](int32_t rows, int32_t *out) {
    if (fn_head(fn.g, fn.mtp->mixer, fn.weights->head, fn.residual->p, rows, logits, scratch, stream) != 0) {
      *error = std::string("the MTP head's logits: ") + fn_last_error();
      return -1;
    }
    if (mtp::argmax(fn.g, logits, rows, out, stream) != 0) {
      *error = fn_last_error();
      return -1;
    }
    return 0;
  };
  if (mtp_entries(fn, ctx, batch, static_cast<const int32_t *>(state.licensed->p), scratch, stream, error) != 0 ||
      head(lanes * columns, picks) != 0) {
    return -1;
  }
  if (mtp::first_drafts(fn.g, picks, fn.residual->p, static_cast<const int32_t *>(state.commit->p), lanes, columns, k,
                        drafts, chain_tokens, state.chain_stack->p, stream) != 0) {
    *error = fn_last_error();
    return -1;
  }
  if (k > 1 && verify::restore_head(state, sections, fn.g, width, window, batch.slots, batch.positions, stream, error) != 0) {
    return -1;
  }
  const std::size_t stack_bytes = static_cast<std::size_t>(lanes) * fn.g.residual_width() * 2;
  for (int32_t step = 1; step < k; ++step) {
    Batch chain;
    chain.lanes = lanes;
    chain.tokens = 1;
    chain.slots = batch.slots;
    chain.positions = static_cast<const int32_t *>(state.chain_positions->p) + (step - 1) * lanes;
    chain.max_visible = batch.max_visible;
    if (cudaMemcpyAsync(fn.residual->p, state.chain_stack->p, stack_bytes, cudaMemcpyDeviceToDevice, stream) !=
        cudaSuccess) {
      *error = std::string("the MTP chain's stacks: ") + cudaGetErrorString(cudaGetLastError());
      return -1;
    }
    if (mtp_entries(fn, ctx, chain, chain_tokens, scratch, stream, error) != 0 || head(lanes, chain_tokens) != 0) {
      return -1;
    }
    if (step + 1 < k && cudaMemcpyAsync(state.chain_stack->p, fn.residual->p, stack_bytes, cudaMemcpyDeviceToDevice,
                                        stream) != cudaSuccess) {
      *error = std::string("the MTP chain's stacks: ") + cudaGetErrorString(cudaGetLastError());
      return -1;
    }
    if (mtp::append_drafts(chain_tokens, lanes, k, step, drafts, stream) != 0) {
      *error = fn_last_error();
      return -1;
    }
  }
  return 0;
}

int32_t run_commit(ignis_model *model, FlashNextModel &fn, const Context &ctx, const verify::Sections &sections,
                   uint32_t width, std::string *error) {
  const verify::State &state = *fn.verify;
  const uint32_t window = state.window(width);
  const auto *slots = static_cast<const int32_t *>(fn.slots->p);
  if (verify::fold(state, sections, width, window, slots, model->stream, error) != 0) return -1;
  if (fn.mtp != nullptr && run_drafts(model, fn, ctx, sections, width, error) != 0) return -1;
  return verify::restore(state, sections, fn.g, width, window, slots, static_cast<const int32_t *>(fn.positions->p),
                         model->stream, error);
}

// Begins a capture, runs `body`, ends it and instantiates. A failure leaves *exec null.
template <class Body>
bool capture(cudaStream_t stream, FlashNextModel &fn, cudaGraphExec_t *exec, std::string *failure, Body body) {
  cudaGraph_t graph = nullptr;
  bool ok = check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture",
                       failure);
  if (ok) {
    ok = body() == 0;
    // A pass that stopped after a lookahead step left residency's prefetch forked (ignis_residency.h
    // rule (b)): it joins before the capture ends.
    if (!ok) (void)ignis_residency_join(fn.residency, stream);
    const cudaError_t end = cudaStreamEndCapture(stream, &graph);
    if (end != cudaSuccess && ok) ok = check_cuda(end, "cudaStreamEndCapture", failure);
  }
  if (ok) ok = check_cuda(cudaGraphInstantiate(exec, graph, 0), "cudaGraphInstantiate", failure);
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (!ok) {
    *exec = nullptr;
    (void)cudaGetLastError();
    (void)ignis_residency_join(fn.residency, stream);
    (void)cudaGetLastError();
  }
  return ok;
}

}  // namespace

int32_t capture_verify_graphs(ignis_model *model, ignis_seq_pool *pool, std::string *error) {
  FlashNextModel &fn = *model->flash_next;
  if (fn.verify == nullptr) return 0;
  verify::State &state = *fn.verify;
  Views views = views_of(fn, pool);
  verify::Sections sections;
  if (!verify::sections_of(*pool, fn.g, gdn_layers_of(fn), &sections, error)) return -1;
  cudaStream_t stream = model->stream;
  for (uint32_t width = 1; width <= fn.decode_lanes; ++width) {
    const std::size_t at = width - 1;
    for (cudaGraphExec_t *exec : {&state.pass_exec[at], &state.commit_exec[at]}) {
      if (*exec != nullptr) cudaGraphExecDestroy(*exec);
      *exec = nullptr;
    }
    state.ready[at] = false;
    if (state.window(width) == 0) continue;
    std::string failure;
    const bool ok =
        capture(stream, fn, &state.pass_exec[at], &failure,
                [&] { return run_pass(model, fn, views.ctx, sections, width, &failure); }) &&
        capture(stream, fn, &state.commit_exec[at], &failure,
                [&] { return run_commit(model, fn, views.ctx, sections, width, &failure); });
    state.ready[at] = ok;
    if (!ok) *error = "width " + std::to_string(width) + " verify: " + failure;
  }
  return 0;
}

int32_t program_verify(ignis_model *model, ignis_seq_pool *pool, ignis_seq *const *sequences, uint64_t batch_size,
                       const ignis_sampling_params *sampling, int32_t *out_token_ids,
                       const ignis_decode_options *options) {
  const auto refuse = [](const std::string &why) {
    step::set_error("ignis_program_decode: " + why);
    return -1;
  };
  FlashNextModel &fn = *model->flash_next;
  const Geometry &g = fn.g;
  if (fn.verify == nullptr) return refuse("this Flash-Next load has no speculative decoding");
  verify::State &state = *fn.verify;
  const auto width = static_cast<uint32_t>(batch_size);
  const uint32_t k = state.window(width);
  if (options->speculative_window != k) {
    return refuse("speculative_window " + std::to_string(options->speculative_window) + " is not the window a round of " +
                  std::to_string(width) + " lanes runs on this load (" + std::to_string(k) + ")");
  }
  if (options->out_committed_counts == nullptr) return refuse("out_committed_counts is null for a verify round");
  // On an MTP load the head drafts each lane's next round at the end of this one (out_drafts);
  // the caller hands them back as this round's drafts, having hashed their n-gram rows.
  const bool mtp = fn.mtp != nullptr;
  const uint32_t columns = k + 1;
  const auto rows = static_cast<std::size_t>(width) * columns;

  std::vector<ninfer::ops::SamplingConfig> configs(batch_size);
  std::vector<int32_t> ids(rows, 0), slots(batch_size), positions(batch_size), extents(batch_size),
      valid(batch_size), anchors(batch_size), drafts(static_cast<std::size_t>(width) * k, 0);
  for (uint64_t i = 0; i < batch_size; ++i) {
    ignis_seq *seq = sequences[i];
    const std::string at = " at index " + std::to_string(i);
    if (seq == nullptr || seq->pending_token < 0) return refuse("sequence is null or was not prefilled" + at);
    if (!ignis_seq_belongs_to(*pool, *seq)) return refuse("the sequence was not drawn from this pool" + at);
    // Every column writes its position, drafted or not, and the head's chain past them: the whole
    // window must fit.
    const uint64_t room = std::min<uint64_t>(ignis_seq_token_capacity(*seq), fn.max_context_tokens);
    const uint32_t written = verify::written_positions(k, mtp);
    if (seq->position + written > room) {
      return refuse("the lane has no room for the " + std::to_string(written) + " positions a verify round writes" + at +
                    "; run it in a one-token round");
    }
    const ignis_sampling_params &lane = sampling[i];
    if (!step::sampling_size_ok(lane)) return refuse("unrecognized ignis_sampling_params size" + at);
    if (!step::unconstrained(lane)) return refuse("a permitted token set cannot ride a verify round" + at);
    if (lane.stop_id_count != 0 && lane.stop_ids == nullptr) return refuse("stop_ids is null with a count" + at);
    uint32_t extent = options->drafts == nullptr ? 0 : k;
    if (options->draft_counts != nullptr) extent = std::min(extent, options->draft_counts[i]);
    if (lane.remaining_tokens != 0) extent = std::min(extent, lane.remaining_tokens - 1);
    configs[i] = step::to_sampling_config(lane, pool->token_counts_for(seq->slot));
    anchors[i] = seq->pending_token;
    slots[i] = seq->slot;
    positions[i] = static_cast<int32_t>(seq->position);
    extents[i] = static_cast<int32_t>(extent);
    valid[i] = static_cast<int32_t>(extent + 1);
    for (uint32_t j = 0; j < columns; ++j) {
      int32_t id = anchors[i];
      if (j >= 1 && j <= extent) {
        id = options->drafts[i * k + (j - 1)];
        if (id < 0 || id >= g.vocab) return refuse("draft id " + std::to_string(id) + at + " is outside the vocabulary");
        drafts[i * k + (j - 1)] = id;
      }
      ids[i * columns + j] = id;
    }
  }

  const auto began = std::chrono::steady_clock::now();
  cudaStream_t stream = model->stream;
  const std::size_t row_bytes = static_cast<std::size_t>(g.ngram_heads) * g.ngram_row_bytes();
  const std::size_t lane_bytes = batch_size * sizeof(int32_t);
  std::string error;
  const auto stage = [&](void *dst, const void *src, std::size_t bytes, const char *what) {
    return check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), std::string("staging ") + what,
                      &error);
  };
  bool ok = stage(model->sampling_decode_configs->p, configs.data(), batch_size * sizeof(configs[0]), "the configs") &&
            stage(fn.token_ids->p, ids.data(), rows * sizeof(int32_t), "the ids") &&
            stage(fn.slots->p, slots.data(), lane_bytes, "the slots") &&
            stage(fn.positions->p, positions.data(), lane_bytes, "the positions") &&
            stage(fn.ngram_rows->p, options->ngram_rows, rows * row_bytes, "the n-gram rows") &&
            stage(state.valid_columns->p, valid.data(), lane_bytes, "the valid columns") &&
            stage(state.extents->p, extents.data(), lane_bytes, "the extents") &&
            // The accept RNG's position base is the frontier, as a one-token round keys its draw.
            stage(state.lengths->p, positions.data(), lane_bytes, "the accept lengths") &&
            stage(state.anchors->p, anchors.data(), lane_bytes, "the anchors") &&
            (k == 0 || stage(state.drafts->p, drafts.data(), drafts.size() * sizeof(int32_t), "the drafts"));
  if (!ok) return refuse(error);

  verify::Sections sections;
  if (!verify::sections_of(*pool, g, gdn_layers_of(fn), &sections, &error)) return refuse(error);
  const std::size_t at = width - 1;
  const bool use_graph = state.ready[at] && fn.captured_pool == pool;
  Views views = views_of(fn, pool);
  const auto fail = [&](const std::string &what) {
    (void)ignis_residency_join(fn.residency, stream);
    (void)cudaStreamSynchronize(stream);
    return refuse("the verify round failed: " + what + "; its lanes are not usable past it, release them");
  };

  // The pass.
  if (use_graph) {
    ok = ignis_residency_join(fn.residency, stream) == 0 &&
         check_cuda(cudaGraphLaunch(state.pass_exec[at], stream), "cudaGraphLaunch(verify pass)", &error);
    if (!ok && error.empty()) error = std::string("ignis_residency_join: ") + ignis_residency_last_error();
  } else {
    ok = run_pass(model, fn, views.ctx, sections, width, &error) == 0;
  }
  std::vector<int32_t> licensed(rows, 0), accepted(batch_size, 0);
  ok = ok &&
       check_cuda(cudaMemcpyAsync(licensed.data(), state.licensed->p, rows * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                  stream),
                  "reading the licensed tokens", &error) &&
       check_cuda(cudaMemcpyAsync(accepted.data(), state.accepted->p, lane_bytes, cudaMemcpyDeviceToHost, stream),
                  "reading the accepted counts", &error) &&
       check_cuda(cudaStreamSynchronize(stream), "the pass's synchronize", &error);
  if (!ok) return fail(error);

  // The host half: each lane's run and its cut.
  std::vector<int32_t> committed(batch_size, 0), pending(batch_size, -1);
  for (uint64_t i = 0; i < batch_size; ++i) {
    const int32_t n = accepted[i];
    if (n < 0 || n > extents[i]) {
      return fail("the accept reported " + std::to_string(n) + " drafts for an extent of " + std::to_string(extents[i]));
    }
    const int32_t *lane_licensed = licensed.data() + i * columns;
    const auto run_at = [&](int32_t j) { return j == 0 ? anchors[i] : lane_licensed[j - 1]; };
    int32_t c = n + 1;
    for (int32_t j = 0; j < n + 1 && c == n + 1; ++j) {
      for (uint32_t s = 0; s < sampling[i].stop_id_count; ++s) {
        if (sampling[i].stop_ids[s] == run_at(j)) {
          c = j + 1;
          break;
        }
      }
    }
    committed[i] = c;
    pending[i] = lane_licensed[c - 1];
    for (int32_t j = 0; j < c; ++j) out_token_ids[i * columns + j] = run_at(j);
  }

  // The commit, and on an MTP load the next round's drafts: the head's chain runs at the positions
  // past each lane's new frontier.
  ok = stage(state.commit->p, committed.data(), lane_bytes, "the committed counts");
  std::vector<int32_t> next_drafts(mtp ? static_cast<std::size_t>(width) * k : 0, -1);
  if (ok && mtp && k > 1) {
    std::vector<int32_t> chain(static_cast<std::size_t>(k - 1) * width);
    for (uint32_t step = 1; step < k; ++step) {
      for (uint64_t i = 0; i < batch_size; ++i) chain[(step - 1) * width + i] = positions[i] + committed[i] - 1 + step;
    }
    ok = stage(state.chain_positions->p, chain.data(), chain.size() * sizeof(int32_t), "the chain's positions");
  }
  if (ok && use_graph) {
    ok = check_cuda(cudaGraphLaunch(state.commit_exec[at], stream), "cudaGraphLaunch(verify commit)", &error);
  } else if (ok) {
    ok = run_commit(model, fn, views.ctx, sections, width, &error) == 0;
  }
  ok = ok && (!mtp || check_cuda(cudaMemcpyAsync(next_drafts.data(), state.drafts_out->p,
                                                 next_drafts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
                                 "reading the drafts", &error));
  ok = ok && check_cuda(cudaStreamSynchronize(stream), "the commit's synchronize", &error);
  if (!ok) return fail(error);

  // The penalty rows: at temperature > 0 the accept counted every licensed token; the ones past a
  // cut run were never emitted, so their counts come back off (the 27B's rule, step.cu).
  std::vector<int32_t *> rollback;
  for (uint64_t i = 0; i < batch_size; ++i) {
    if (!(configs[i].temperature > 0.0F) || configs[i].token_counts == nullptr) continue;
    for (int32_t j = committed[i]; j < accepted[i] + 1; ++j) {
      rollback.push_back(configs[i].token_counts + licensed[i * columns + j]);
    }
  }
  if (!rollback.empty()) {
    std::vector<int32_t> counts(rollback.size(), 0);
    for (std::size_t r = 0; r < rollback.size() && ok; ++r) {
      ok = check_cuda(cudaMemcpyAsync(&counts[r], rollback[r], sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
                      "reading a penalty count", &error);
    }
    ok = ok && check_cuda(cudaStreamSynchronize(stream), "the penalty read", &error);
    for (std::size_t r = 0; r < rollback.size() && ok; ++r) {
      int32_t taken = 0;
      for (std::size_t q = 0; q <= r; ++q) taken += rollback[q] == rollback[r] ? 1 : 0;
      counts[r] = std::max(counts[r] - taken, 0);
    }
    for (std::size_t r = 0; r < rollback.size() && ok; ++r) {
      ok = check_cuda(cudaMemcpyAsync(rollback[r], &counts[r], sizeof(int32_t), cudaMemcpyHostToDevice, stream),
                      "writing a penalty count", &error);
    }
    ok = ok && check_cuda(cudaStreamSynchronize(stream), "the penalty write", &error);
    if (!ok) return fail(error);
  }

  for (uint64_t i = 0; i < batch_size; ++i) {
    sequences[i]->pending_token = pending[i];
    advance_frontiers(sequences[i], static_cast<uint32_t>(committed[i]));
    if (options->out_drafts != nullptr) {
      for (uint32_t j = 0; j < k; ++j) options->out_drafts[i * k + j] = mtp ? next_drafts[i * k + j] : -1;
    }
    options->out_committed_counts[i] = committed[i];
    if (options->out_extents != nullptr) options->out_extents[i] = static_cast<uint32_t>(extents[i]);
    if (options->out_permitted_probs != nullptr) options->out_permitted_probs[i] = 0.0F;
  }
  model->last_step_micros = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - began).count());
  model->last_step_kernel_count = static_cast<uint64_t>(g.layers);
  model->last_step_graph_launches = use_graph ? 2 : 0;
  return 0;
}

}  // namespace ignis::flash_next
