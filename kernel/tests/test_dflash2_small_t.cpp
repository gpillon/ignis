// The DFlash2 drafter's launches of ours of the vendored NVFP4 small-T kernel (GitHub #694,
// kernel/src/dflash2_small_t.h) -- OURS. Each lever held bit for bit to today's path at the
// drafter's real shapes, its outputs and today's poisoned apart first, on activations spread over
// 2^-12..2^12 so that the sums round, and on weights whose codes and scales are drawn at random
// (the vendored harness's pattern repeats every 16 rows, so another row tile's weights would read
// the same):
//
//   key_value  the append's key and value rows alone, [2048, T] from the q | k | v weight
//              [6144, 5120] (`key_value_linear`), against rows 4096..6143 of today's whole product
//              (`ignis_linear_a16`, a verify round's call) at every width the lever takes, 5 to 32
//              columns: the base block, 8 columns a lane, at 1 to 4 lanes, and the extended block,
//              16, at 1 and 2 among them. A wrong offset into the code or the scale plane reads
//              another row's weights and fails here: rows 0..2047 and 2048..4095 of the product
//              must differ from them, or the comparison could not tell.
//   fill       `forward_linear` on the conv projection [1280, 5120] and the q | k | v weight against
//              `ops::linear` at every width it takes on 4 warps a CTA, 5 to 16 columns: the base
//              block at 1 and 2 lanes, the extended at 1, a window of 10's 11 among them, one launch
//              of ours a call by the launch counter.
//   routes     which route a call takes, as the measurement switches name it at each call: unset or
//              `1` ours at those widths, `0` today's at every width; never at 1, 4, 33 or 64
//              columns, for a prefill chunk's taps, or for another of the drafter's weights; any
//              other value refused by name.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or busy GPU fails this test.

#include "dflash2_small_t.h"

#include "ignis_nvfp4_a16_mma.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/linear_test_common.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using namespace ninfer;
namespace small_t = ignis::dflash2_small_t;

constexpr const char *kKeyValueSwitch = "IGNIS_DFLASH2_APPEND_KV_ONLY";
constexpr const char *kFillSwitch = "IGNIS_DFLASH2_SMALL_T_FILL";
constexpr std::int32_t kHidden = 5120, kQkvRows = 6144, kConvRows = 1280, kQueryRows = 4096, kKeyValueRows = 2048;

int g_failed = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failed;
  }
}

void cuda_ok(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

struct DeviceBytes {
  void *p = nullptr;
  explicit DeviceBytes(std::size_t bytes) { cuda_ok(cudaMalloc(&p, bytes), "cudaMalloc"); }
  ~DeviceBytes() { cudaFree(p); }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
};

void set_environment(const char *name, const char *value) {
#ifdef _WIN32
  SetEnvironmentVariableA(name, value);
#else
  if (value != nullptr) {
    setenv(name, value, 1);
  } else {
    unsetenv(name);
  }
#endif
}

// A drafter weight of `n` x 5120 on the device: the vendored harness's layout, every code and every
// group scale drawn at random (an E4M3 scale between 2^-3 and 1.875).
struct DeviceWeight {
  test::quantized_weight::PackedWeight host;
  DeviceBytes bytes;
  Weight weight;
  DeviceWeight(std::int32_t n, std::uint32_t seed)
      : host(test::linear::make_nvfp4_weight(n, kHidden, seed)), bytes(host.payload.size()),
        weight(host.device_weight(bytes.p)) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, 255), scale(0x20, 0x3f);
    for (std::uint64_t i = 0; i < host.code_plane_bytes; ++i) {
      host.payload[i] = static_cast<std::uint8_t>(code(rng));
    }
    for (std::uint64_t i = 0; i < host.scale_plane_bytes; ++i) {
      host.payload[host.scale_plane_offset + i] = static_cast<std::uint8_t>(scale(rng));
    }
    cuda_ok(cudaMemcpy(bytes.p, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice), "copy weight");
  }
};

// x[5120, T] on the device: a normal value times 2^-12..2^12, so that a sum of 5,120 products rounds.
struct Activation {
  DeviceBytes bytes;
  Tensor x;
  Activation(std::int32_t tokens, std::uint32_t seed)
      : bytes(static_cast<std::size_t>(kHidden) * tokens * 2), x(bytes.p, DType::BF16, {kHidden, tokens}) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(kHidden) * tokens);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_int_distribution<int> exponent(-12, 12);
    for (auto &bits : host) {
      const __nv_bfloat16 value = __float2bfloat16_rn(std::ldexp(normal(rng), exponent(rng)));
      std::memcpy(&bits, &value, sizeof(bits));
    }
    cuda_ok(cudaMemcpy(bytes.p, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "copy activation");
  }
};

// A BF16 [rows, T] output on the device, every byte `poison` first.
struct Output {
  DeviceBytes bytes;
  Tensor out;
  std::size_t elements;
  Output(std::int32_t rows, std::int32_t tokens, int poison)
      : bytes(static_cast<std::size_t>(rows) * tokens * 2), out(bytes.p, DType::BF16, {rows, tokens}),
        elements(static_cast<std::size_t>(rows) * tokens) {
    cuda_ok(cudaMemset(bytes.p, poison, elements * 2), "poison output");
  }
  std::vector<std::uint16_t> read() const {
    std::vector<std::uint16_t> host(elements);
    cuda_ok(cudaMemcpy(host.data(), bytes.p, elements * 2, cudaMemcpyDeviceToHost), "read output");
    return host;
  }
};

void key_value_bits(const DeviceWeight &qkv) {
  for (std::int32_t tokens = 5; tokens <= 32; ++tokens) {
    const Activation a(tokens, 9400U + static_cast<std::uint32_t>(tokens));
    Output whole(kQkvRows, tokens, 0xA5);
    Output kv(kKeyValueRows, tokens, 0x5A);
    ignis_linear_a16(a.x, qkv.weight, whole.out, IgnisA16Call::General, nullptr);
    const std::uint64_t before = small_t::launches();
    small_t::key_value_linear(a.x, qkv.weight, kv.out, nullptr);
    cuda_ok(cudaDeviceSynchronize(), "run the key and value rows");
    const std::uint64_t launches = small_t::launches() - before;
    const std::vector<std::uint16_t> today = whole.read();
    const std::vector<std::uint16_t> ours = kv.read();
    // How many of the key and value rows differ from the product's rows from `first_row` on.
    const auto differing = [&](std::int32_t first_row) {
      std::size_t count = 0;
      for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < kKeyValueRows; ++row) {
          const std::size_t at = static_cast<std::size_t>(token) * kKeyValueRows + row;
          const std::size_t from = static_cast<std::size_t>(token) * kQkvRows + first_row + row;
          count += ours[at] != today[from] ? 1 : 0;
        }
      }
      return count;
    };
    const std::size_t moved = differing(kQueryRows);
    const std::size_t apart = std::min(differing(0), differing(kKeyValueRows));
    const std::string what = "key_value [2048, 5120] x " + std::to_string(tokens) + " columns";
    std::printf("  %s: %s rows 4096..6143 of the whole product (%zu differ; %zu or more differ from rows 0 or "
                "2048 on), %llu launch(es)\n",
                what.c_str(), moved == 0 ? "bit-identical to" : "DIFFERS from", moved, apart,
                static_cast<unsigned long long>(launches));
    check(moved == 0, what + ": the whole product's key and value rows");
    check(apart > 0, what + ": the query rows' outputs differ from the key and value rows', so the comparison "
                            "tells the rows apart");
    check(launches == 1, what + ": one launch of ours");
  }
}

void fill_bits(const DeviceWeight &weight, const char *name) {
  set_environment(kFillSwitch, nullptr);
  for (std::int32_t tokens = 5; tokens <= 16; ++tokens) {
    const Activation a(tokens, 9500U + static_cast<std::uint32_t>(tokens));
    Output today(weight.weight.n, tokens, 0xA5);
    Output ours(weight.weight.n, tokens, 0x5A);
    ops::linear(a.x, weight.weight, today.out, nullptr);
    const std::uint64_t before = small_t::launches();
    small_t::forward_linear(a.x, weight.weight, ours.out, nullptr);
    cuda_ok(cudaDeviceSynchronize(), "run the products");
    const std::uint64_t launches = small_t::launches() - before;
    const bool same = ours.read() == today.read();
    const std::string what = std::string("fill ") + name + " x " + std::to_string(tokens) + " columns";
    std::printf("  %s: %s ops::linear, %llu launch(es) of ours\n", what.c_str(),
                same ? "bit-identical to" : "DIFFERS from", static_cast<unsigned long long>(launches));
    check(same, what + ": ops::linear's bytes");
    check(launches == 1, what + ": one launch of ours");
  }
}

// Whether `read` throws std::invalid_argument naming `name` under `name`=2.
template <class Read> bool refused_by_name(const char *name, Read read) {
  set_environment(name, "2");
  bool refused = false;
  try {
    (void)read();
  } catch (const std::invalid_argument &e) {
    refused = std::string(e.what()).find(name) != std::string::npos;
  }
  set_environment(name, nullptr);
  return refused;
}

void routes(const DeviceWeight &qkv, const DeviceWeight &conv) {
  Weight selector = conv.weight;  // the selector projection's shape: no lever takes it
  selector.n = 256;
  Weight attention_out = conv.weight;  // the attention's output projection [5120, 4096]
  attention_out.n = 5120;
  attention_out.k = 4096;
  const char *const values[3] = {nullptr, "1", "0"};
  for (const char *value : values) {
    const bool on = value == nullptr || std::strcmp(value, "1") == 0;
    set_environment(kKeyValueSwitch, value);
    set_environment(kFillSwitch, value);
    const std::string named = std::string("switches ") + (value == nullptr ? "unset" : std::string("=") + value);
    for (std::int32_t tokens : {1, 4, 5, 8, 11, 16, 17, 24, 32, 33, 48, 64}) {
      const bool kv_want = on && tokens >= 5 && tokens <= 32;
      check(small_t::projects_key_value_only(qkv.weight, tokens, IgnisA16Call::General) == kv_want,
            named + ": the append at " + std::to_string(tokens) + " columns " +
                (kv_want ? "projects the key and value rows" : "projects the whole product"));
      check(!small_t::projects_key_value_only(qkv.weight, tokens, IgnisA16Call::PrefillTaps),
            named + ": a prefill chunk's taps at " + std::to_string(tokens) + " columns keep the whole product");
      check(!small_t::projects_key_value_only(conv.weight, tokens, IgnisA16Call::General),
            named + ": no other weight's rows at " + std::to_string(tokens) + " columns");
      const bool fill_want = on && tokens >= 5 && tokens <= 16;
      check(small_t::fills(qkv.weight, tokens) == fill_want && small_t::fills(conv.weight, tokens) == fill_want,
            named + ": q | k | v and the conv projection at " + std::to_string(tokens) + " columns " +
                (fill_want ? "fill the card" : "keep the vendored launch"));
      check(!small_t::fills(selector, tokens) && !small_t::fills(attention_out, tokens),
            named + ": the selector and the attention output keep the vendored launch at " + std::to_string(tokens) +
                " columns");
    }
    // What a call does, by the launch counter.
    for (std::int32_t tokens : {8, 16, 17}) {
      const Activation a(tokens, 9600U + static_cast<std::uint32_t>(tokens));
      Output out(kQkvRows, tokens, 0);
      const std::uint64_t before = small_t::launches();
      small_t::forward_linear(a.x, qkv.weight, out.out, nullptr);
      cuda_ok(cudaDeviceSynchronize(), "run forward_linear");
      const std::uint64_t launches = small_t::launches() - before;
      const std::uint64_t want = on && tokens <= 16 ? 1 : 0;
      std::printf("  routes %s: q | k | v x %d columns, %llu launch(es) of ours\n", named.c_str(), tokens,
                  static_cast<unsigned long long>(launches));
      check(launches == want, named + ": q | k | v x " + std::to_string(tokens) + " columns takes " +
                                  (want == 1 ? "the 4-warp launch" : "ops::linear"));
    }
  }
  set_environment(kKeyValueSwitch, nullptr);
  set_environment(kFillSwitch, nullptr);
  check(refused_by_name(kKeyValueSwitch,
                        [&] { return small_t::projects_key_value_only(qkv.weight, 16, IgnisA16Call::General); }),
        std::string(kKeyValueSwitch) + "=2 refused by name");
  check(refused_by_name(kFillSwitch, [&] { return small_t::fills(qkv.weight, 16); }),
        std::string(kFillSwitch) + "=2 refused by name");
}

} // namespace

int main() {
  try {
    int devices = 0;
    cuda_ok(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    set_environment(kKeyValueSwitch, nullptr);
    set_environment(kFillSwitch, nullptr);
    std::printf("The DFlash2 drafter's small-T launches of ours (GitHub #694)\n");
    const DeviceWeight qkv(kQkvRows, 9401U);
    const DeviceWeight conv(kConvRows, 9402U);
    key_value_bits(qkv);
    fill_bits(conv, "conv projection [1280, 5120]");
    fill_bits(qkv, "q | k | v [6144, 5120]");
    routes(qkv, conv);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FATAL: %s\n", e.what());
    return 1;
  }
  if (g_failed != 0) {
    std::fprintf(stderr, "test_dflash2_small_t: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_dflash2_small_t: OK\n");
  return 0;
}
