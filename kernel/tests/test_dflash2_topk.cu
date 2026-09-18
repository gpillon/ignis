// The DFlash2 drafter's per-column top-k, ours against the vendored op --
// OURS, not vendored (kernel/include/ignis_dflash2_topk.h).
//
// ADR 0010 is why this test has this shape. The vendored op is not a
// tolerance oracle here, it is *the* oracle: `ninfer/ops/dflash2_topk.h`
// specifies an exact, deterministic selection -- the iterative largest-first
// removal of the BF16-ordered values, ties to the smaller row id, `values`
// carrying the logits entries bit-exactly -- so a replacement that is right
// agrees with it bit for bit, on every shape, with no bound to calibrate. A
// tolerance here would only be a place for a wrong answer to hide.
//
// What the arms are for:
//
//   production   [248046, 7] and [248046, 56]: the shapes the drafter calls
//                at batch 1 and at IGNIS_DECODE_MAX_BATCH, which is the only
//                thing the engine actually runs.
//   boundaries   row counts either side of the 2,048-row split -- one split,
//                exactly two, a ragged last split, and the narrowest column
//                the op admits -- because the split is ours and the vendored
//                op has no such seam to get wrong. `rows < k` is outside the
//                op's stated domain (its wrapper refuses it), so the narrow
//                end stops at rows == k.
//   ties         a column drawn from sixteen distinct values, so most of the
//                vocabulary ties and the smaller-row rule decides nearly
//                every slot. The arm asserts the ties are really there, or
//                it would be a second copy of the random arm.
//   extremes     +/-inf and both zeroes present, since -0.0 == 0.0 compares
//                equal and the row has to break it.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU fails this test rather than skipping it.

#include "ignis_dflash2_topk.h"

#include "ninfer/ops/dflash2_topk.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
    ++g_failed;
  }
}

#define CUDA_FATAL(expr)                                                                           \
  do {                                                                                             \
    const cudaError_t err_ = (expr);                                                               \
    if (err_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, cudaGetErrorString(err_));                    \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

struct DeviceBytes {
  void *p = nullptr;
  explicit DeviceBytes(std::size_t bytes) {
    if (bytes > 0) { CUDA_FATAL(cudaMalloc(&p, bytes)); }
  }
  ~DeviceBytes() {
    if (p != nullptr) { cudaFree(p); }
  }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
};

// The BF16 bit pattern of `value`, truncating toward zero. Any pattern is
// legal input, so truncation is only about being deterministic across hosts.
std::uint16_t to_bf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16);
}

float from_bf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

std::uint32_t next_random(std::uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

enum class Fill { Random, Ties, Extremes };

std::vector<std::uint16_t> make_logits(std::int32_t rows, std::int32_t columns, Fill fill,
                                       std::uint32_t seed) {
  std::vector<std::uint16_t> logits(static_cast<std::size_t>(rows) * columns);
  std::uint32_t state = seed | 1U;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const std::uint32_t r = next_random(state);
    switch (fill) {
    case Fill::Random:
      logits[i] = to_bf16(static_cast<float>(static_cast<std::int32_t>(r % 65536U) - 32768) / 4096.0F);
      break;
    case Fill::Ties:
      // Sixteen distinct values over the whole column: with 248k rows every
      // value repeats thousands of times and the row id decides the order.
      logits[i] = to_bf16(static_cast<float>(r % 16U));
      break;
    case Fill::Extremes: {
      const std::uint32_t pick = r % 8U;
      if (pick == 0) {
        logits[i] = to_bf16(INFINITY);
      } else if (pick == 1) {
        logits[i] = to_bf16(-INFINITY);
      } else if (pick == 2) {
        logits[i] = to_bf16(0.0F);
      } else if (pick == 3) {
        logits[i] = to_bf16(-0.0F);
      } else {
        logits[i] = to_bf16(static_cast<float>(static_cast<std::int32_t>(r % 512U) - 256));
      }
      break;
    }
    }
  }
  return logits;
}

const char *fill_name(Fill fill) {
  switch (fill) {
  case Fill::Random: return "random";
  case Fill::Ties: return "ties";
  case Fill::Extremes: return "extremes";
  }
  return "?";
}

// Runs both implementations over the same device input and compares.
void run_case(std::int32_t rows, std::int32_t columns, std::int32_t k, Fill fill,
              std::uint32_t seed) {
  const std::string label = std::string(fill_name(fill)) + " rows=" + std::to_string(rows) +
                            " columns=" + std::to_string(columns) + " k=" + std::to_string(k);
  const std::vector<std::uint16_t> host_logits = make_logits(rows, columns, fill, seed);
  const std::size_t out_count = static_cast<std::size_t>(k) * columns;

  DeviceBytes logits_device(host_logits.size() * sizeof(std::uint16_t));
  CUDA_FATAL(cudaMemcpy(logits_device.p, host_logits.data(),
                        host_logits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
  DeviceBytes ref_ids(out_count * sizeof(std::int32_t));
  DeviceBytes ref_values(out_count * sizeof(std::uint16_t));
  DeviceBytes our_ids(out_count * sizeof(std::int32_t));
  DeviceBytes our_values(out_count * sizeof(std::uint16_t));
  // Poisoned, so a slot neither implementation writes shows up as a
  // difference rather than as two matching zeroes.
  CUDA_FATAL(cudaMemset(ref_ids.p, 0x5A, out_count * sizeof(std::int32_t)));
  CUDA_FATAL(cudaMemset(ref_values.p, 0x5A, out_count * sizeof(std::uint16_t)));
  CUDA_FATAL(cudaMemset(our_ids.p, 0xA5, out_count * sizeof(std::int32_t)));
  CUDA_FATAL(cudaMemset(our_values.p, 0xA5, out_count * sizeof(std::uint16_t)));

  const ninfer::Tensor logits(logits_device.p, ninfer::DType::BF16, {rows, columns, 1, 1});
  ninfer::Tensor reference_ids(ref_ids.p, ninfer::DType::I32, {k, columns, 1, 1});
  ninfer::Tensor reference_values(ref_values.p, ninfer::DType::BF16, {k, columns, 1, 1});
  ninfer::Tensor mine_ids(our_ids.p, ninfer::DType::I32, {k, columns, 1, 1});
  ninfer::Tensor mine_values(our_values.p, ninfer::DType::BF16, {k, columns, 1, 1});

  ninfer::ops::dflash2_topk(logits, k, reference_ids, reference_values, /*stream=*/nullptr);

  const std::size_t workspace_bytes = ignis_dflash2_topk_workspace_bytes(rows, columns, k);
  DeviceBytes workspace(workspace_bytes);
  ignis_dflash2_topk(logits, k, mine_ids, mine_values, workspace.p, workspace_bytes,
                     /*stream=*/nullptr);
  CUDA_FATAL(cudaStreamSynchronize(nullptr));
  CUDA_FATAL(cudaGetLastError());

  std::vector<std::int32_t> ref_id_host(out_count);
  std::vector<std::int32_t> our_id_host(out_count);
  std::vector<std::uint16_t> ref_value_host(out_count);
  std::vector<std::uint16_t> our_value_host(out_count);
  CUDA_FATAL(cudaMemcpy(ref_id_host.data(), ref_ids.p, out_count * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_FATAL(cudaMemcpy(our_id_host.data(), our_ids.p, out_count * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_FATAL(cudaMemcpy(ref_value_host.data(), ref_values.p, out_count * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost));
  CUDA_FATAL(cudaMemcpy(our_value_host.data(), our_values.p, out_count * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost));

  std::size_t id_diff = 0;
  std::size_t value_diff = 0;
  std::size_t first_diff = out_count;
  for (std::size_t i = 0; i < out_count; ++i) {
    if (ref_id_host[i] != our_id_host[i]) {
      ++id_diff;
      first_diff = std::min(first_diff, i);
    }
    if (ref_value_host[i] != our_value_host[i]) {
      ++value_diff;
      first_diff = std::min(first_diff, i);
    }
  }

  // Guards against an arm that would pass while proving nothing.
  const std::int32_t expected = k;
  std::size_t written = 0;
  for (std::size_t i = 0; i < out_count; ++i) {
    if (ref_id_host[i] != 0x5A5A5A5A) { ++written; }
  }
  check(written == static_cast<std::size_t>(expected) * columns,
       label + ": the vendored op wrote " + std::to_string(written) + " of the " +
           std::to_string(static_cast<std::size_t>(expected) * columns) +
           " slots this shape has, so the comparison covers less than the shape");
  // Column 0's ids must be distinct: a run that returned one row k times
  // would match trivially if both were broken the same way, but only one of
  // the two is under test here.
  for (std::int32_t a = 0; a < expected; ++a) {
    for (std::int32_t b = a + 1; b < expected; ++b) {
      check(our_id_host[a] != our_id_host[b],
           label + ": our column 0 returned row " + std::to_string(our_id_host[a]) +
               " in both slot " + std::to_string(a) + " and slot " + std::to_string(b));
    }
  }
  if (fill == Fill::Ties && rows > 4096) {
    // The point of this arm: the winners must be the lowest-numbered rows
    // holding the top value, which only happens if ties break by row.
    const float top = from_bf16(our_value_host[0]);
    bool tie_decided = true;
    for (std::int32_t slot = 1; slot < expected; ++slot) {
      if (from_bf16(our_value_host[slot]) == top && our_id_host[slot] <= our_id_host[slot - 1]) {
        tie_decided = false;
      }
    }
    check(tie_decided, label + ": tied rows did not come back in increasing row order");
    check(from_bf16(our_value_host[expected - 1]) == top,
         label + ": this arm meant every slot to tie at the top value, so it is not exercising "
                 "the tie rule");
  }

  std::printf("  %-44s ids differ %zu, values differ %zu (of %zu), workspace %zu B\n",
             label.c_str(), id_diff, value_diff, out_count, workspace_bytes);
  check(id_diff == 0 && value_diff == 0,
       label + ": our top-k disagrees with the vendored op at flat slot " +
           std::to_string(first_diff) + " -- the selection is exact and deterministic, so any "
                                        "difference is a defect in one of them");
}

} // namespace

int run_all();

// A domain violation or a CUDA failure inside the vendored op arrives as an
// exception; without this it would surface as a bare abort with no message.
int main() {
  try {
    return run_all();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FATAL: %s\n", e.what());
    return 1;
  }
}

int run_all() {
  int device_count = 0;
  const cudaError_t count_err = cudaGetDeviceCount(&device_count);
  if (count_err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device (%s)\n",
                count_err == cudaSuccess ? "count 0" : cudaGetErrorString(count_err));
    return 1;
  }

  // The 27B vocabulary, which is what makes the vendored op's one-warp-per-
  // column shape cost what it costs.
  constexpr std::int32_t kVocab = 248046;
  constexpr std::int32_t kTopK = 16;

  std::printf("production shapes:\n");
  run_case(kVocab, 7, kTopK, Fill::Random, 0x1234u);
  run_case(kVocab, 56, kTopK, Fill::Random, 0x5678u);
  run_case(kVocab, 7, kTopK, Fill::Ties, 0x9abcu);
  run_case(kVocab, 7, kTopK, Fill::Extremes, 0xdef0u);

  std::printf("split boundaries:\n");
  for (const std::int32_t rows : {16, 17, 2047, 2048, 2049, 4096, 4097, 5000}) {
    run_case(rows, 3, kTopK, Fill::Random, static_cast<std::uint32_t>(rows) * 2654435761u);
  }

  std::printf("column counts:\n");
  run_case(9973, 1, kTopK, Fill::Random, 0x0f0fu);
  run_case(9973, 64, kTopK, Fill::Random, 0xf0f0u);

  // A k this implementation does not specialize must still be answered, by
  // forwarding: the engine's behaviour cannot depend on which one ran.
  std::printf("forwarded k:\n");
  run_case(9973, 4, 8, Fill::Random, 0x2468u);
  run_case(9973, 4, 32, Fill::Random, 0x1357u);

  if (g_failed != 0) {
    std::fprintf(stderr, "dflash2 top-k test: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("dflash2 top-k test: ok\n");
  return 0;
}
