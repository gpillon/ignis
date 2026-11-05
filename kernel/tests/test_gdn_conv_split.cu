// GitHub #507: the GDN prefill's split conv (kernel/src/gdn_conv_split.h, an adapted op) against
// the vendored op it replaces in the layer body -- OURS, not vendored.
//
// The oracle is what the prefill body ran before #507: `ninfer::ops::causal_conv1d_silu` into
// one [C, T] plane, then the three strided device-to-device copies that split it into the
// query, key and value planes. The conv is an exact elementwise expression, so the bound is bit
// equality on the three planes and on the taps the call leaves behind, with nothing to
// calibrate (ADR 0031: the vendored op is the oracle).
//
// What the arms are for:
//
//   widths     T across every form the vendored wrapper switches between -- 1 (decode), 2..16
//              (small-T), 17..64 (sequence), 65 and up (prefill) -- and the prefill chunks the
//              layer runs (1024, 2048, 4096), because the split conv is one form for all of
//              them and has to agree with each. T = 1, 2 and 3 also make the new taps mix the
//              old taps with the chunk's columns.
//   geometry   the 27B's [2048 | 2048 | 6144] channels, from a state of nonzero taps, so the
//              first three columns read the taps and every channel range lands in its plane.
//   refusals   a split whose widths do not add up to the input's channels, planes of another T,
//              and an odd width (the pair form), each refused by name before any launch.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or busy GPU fails this
// test rather than skipping it.

#include "gdn_conv_split.h"
#include "moe_fixture.h"

#include "ninfer/ops/causal_conv1d_silu.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

// The 27B's GDN conv channels (crates/core/src/compute.rs `ModelConfig::qwen38_27b`).
constexpr int32_t kQuery = 2048, kKey = 2048, kValue = 6144;
constexpr int32_t kChannels = kQuery + kKey + kValue;

struct Planes {
  std::vector<uint16_t> query, key, value, state;
};

// BF16 values in [-amplitude, amplitude), with exact zeros of both signs sprinkled in.
std::vector<uint16_t> bf16_values(uint32_t stream, std::size_t n, float amplitude) {
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = f32_to_bf16(hash_uniform(stream, i, amplitude));
    if (i % 61 == 0) v[i] = 0x0000u;
    if (i % 67 == 0) v[i] = 0x8000u;
  }
  return v;
}

ninfer::Tensor bf16_plane(const DeviceBytes &d, int32_t rows, int32_t cols) {
  return ninfer::Tensor(d.p, ninfer::DType::BF16, {rows, cols, 1, 1});
}

struct Inputs {
  std::vector<uint16_t> x, weight, state;
  explicit Inputs(int32_t T)
      : x(bf16_values(100 + static_cast<uint32_t>(T), static_cast<std::size_t>(kChannels) * T, 2.0f)),
        weight(bf16_values(7, static_cast<std::size_t>(kChannels) * 4, 1.0f)),
        state(bf16_values(9, static_cast<std::size_t>(kChannels) * 3, 2.0f)) {}
};

// What the prefill body ran before #507: the vendored conv into one plane, then the split.
Planes vendored_then_copies(const Inputs &in, int32_t T, cudaStream_t stream) {
  DeviceBytes x(in.x.size() * 2), weight(in.weight.size() * 2), state(in.state.size() * 2);
  DeviceBytes conv(in.x.size() * 2), query(static_cast<std::size_t>(kQuery) * T * 2),
      key(static_cast<std::size_t>(kKey) * T * 2), value(static_cast<std::size_t>(kValue) * T * 2);
  upload(x, in.x);
  upload(weight, in.weight);
  upload(state, in.state);
  const ninfer::Tensor tx = bf16_plane(x, kChannels, T);
  const ninfer::Tensor tw = bf16_plane(weight, kChannels, 4);
  ninfer::Tensor ts = bf16_plane(state, kChannels, 3);
  ninfer::Tensor tconv = bf16_plane(conv, kChannels, T);
  ninfer::ops::causal_conv1d_silu(tx, tw, ts, tconv, stream);
  const auto split = [&](int32_t offset, int32_t width, DeviceBytes &dst) {
    MOE_CUDA(cudaMemcpy2DAsync(dst.p, static_cast<std::size_t>(width) * 2,
                               static_cast<const uint8_t *>(conv.p) + static_cast<std::size_t>(offset) * 2,
                               static_cast<std::size_t>(kChannels) * 2, static_cast<std::size_t>(width) * 2,
                               static_cast<std::size_t>(T), cudaMemcpyDeviceToDevice, stream));
  };
  split(0, kQuery, query);
  split(kQuery, kKey, key);
  split(kQuery + kKey, kValue, value);
  MOE_CUDA(cudaStreamSynchronize(stream));
  return {download<uint16_t>(query.p, static_cast<std::size_t>(kQuery) * T),
          download<uint16_t>(key.p, static_cast<std::size_t>(kKey) * T),
          download<uint16_t>(value.p, static_cast<std::size_t>(kValue) * T),
          download<uint16_t>(state.p, in.state.size())};
}

Planes split_conv(const Inputs &in, int32_t T, cudaStream_t stream) {
  DeviceBytes x(in.x.size() * 2), weight(in.weight.size() * 2), state(in.state.size() * 2);
  DeviceBytes query(static_cast<std::size_t>(kQuery) * T * 2), key(static_cast<std::size_t>(kKey) * T * 2),
      value(static_cast<std::size_t>(kValue) * T * 2);
  upload(x, in.x);
  upload(weight, in.weight);
  upload(state, in.state);
  const ninfer::Tensor tx = bf16_plane(x, kChannels, T);
  const ninfer::Tensor tw = bf16_plane(weight, kChannels, 4);
  ninfer::Tensor ts = bf16_plane(state, kChannels, 3);
  ninfer::Tensor tq = bf16_plane(query, kQuery, T);
  ninfer::Tensor tk = bf16_plane(key, kKey, T);
  ninfer::Tensor tv = bf16_plane(value, kValue, T);
  ignis::gdn_conv::split_conv(tx, tw, ts, tq, tk, tv, stream);
  MOE_CUDA(cudaStreamSynchronize(stream));
  return {download<uint16_t>(query.p, static_cast<std::size_t>(kQuery) * T),
          download<uint16_t>(key.p, static_cast<std::size_t>(kKey) * T),
          download<uint16_t>(value.p, static_cast<std::size_t>(kValue) * T),
          download<uint16_t>(state.p, in.state.size())};
}

void same_bits(const std::vector<uint16_t> &got, const std::vector<uint16_t> &want, const std::string &label) {
  std::size_t wrong = 0, first = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      if (wrong == 0) first = i;
      ++wrong;
    }
  }
  check(got.size() == want.size() && wrong == 0,
        label + ": " + std::to_string(wrong) + " of " + std::to_string(want.size()) + " elements differ" +
            (wrong == 0 ? std::string() : ", the first at " + std::to_string(first)));
}

void split_conv_is_the_vendored_conv_then_the_copies(cudaStream_t stream) {
  for (const int32_t T : {1, 2, 3, 4, 7, 16, 17, 63, 64, 65, 1000, 1024, 2048, 4096}) {
    const Inputs in(T);
    const Planes want = vendored_then_copies(in, T, stream);
    const Planes got = split_conv(in, T, stream);
    const std::string label = "T=" + std::to_string(T);
    same_bits(got.query, want.query, label + " query");
    same_bits(got.key, want.key, label + " key");
    same_bits(got.value, want.value, label + " value");
    same_bits(got.state, want.state, label + " taps after the chunk");
    check(want.state != in.state, label + ": the call moves the taps");
  }
}

bool refused(const std::string &label, int32_t q, int32_t k, int32_t v, int32_t T, int32_t value_T,
             const std::string &needle) {
  DeviceBytes x(static_cast<std::size_t>(kChannels) * T * 2), weight(static_cast<std::size_t>(kChannels) * 8),
      state(static_cast<std::size_t>(kChannels) * 6), query(static_cast<std::size_t>(q) * T * 2),
      key(static_cast<std::size_t>(k) * T * 2), value(static_cast<std::size_t>(v) * value_T * 2);
  const ninfer::Tensor tx = bf16_plane(x, kChannels, T);
  const ninfer::Tensor tw = bf16_plane(weight, kChannels, 4);
  ninfer::Tensor ts = bf16_plane(state, kChannels, 3);
  ninfer::Tensor tq = bf16_plane(query, q, T);
  ninfer::Tensor tk = bf16_plane(key, k, T);
  ninfer::Tensor tv = bf16_plane(value, v, value_T);
  try {
    ignis::gdn_conv::split_conv(tx, tw, ts, tq, tk, tv, nullptr);
  } catch (const std::invalid_argument &e) {
    const bool named = std::string(e.what()).find(needle) != std::string::npos;
    check(named, label + ": refused, but as \"" + e.what() + "\"");
    return named;
  }
  check(false, label + ": not refused");
  return false;
}

void a_split_that_does_not_fit_is_refused() {
  refused("widths short of the channels", kQuery, kKey, kValue - 2, 8, 8, "channels");
  refused("a value plane of another T", kQuery, kKey, kValue, 8, 9, "T");
  refused("an odd width", kQuery - 1, kKey + 1, kValue, 8, 8, "even");
}

} // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "FAIL: no CUDA device\n");
    return 1;
  }
  cudaStream_t stream = nullptr;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  split_conv_is_the_vendored_conv_then_the_copies(stream);
  a_split_that_does_not_fit_is_refused();
  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
