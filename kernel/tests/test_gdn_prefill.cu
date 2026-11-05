// GitHub #514: the GDN prefill's chunked delta rule on our own kernels (kernel/src/gdn_prefill.h)
// against the vendored chunked path it replaces, and both against an fp64 recurrence --
// OURS, not vendored.
//
// The oracle is the vendored op the layer body calls (ADR 0031): `ninfer::ops::gated_delta_net`
// with `normalize_qk`, the state in place. Its chunked kernels round on the way (q and k
// normalized to bf16, W, U, the new values and every chunk's state stored as bf16, TF32
// operands), so it is not exact either, and the truth both are held to is an fp64 recurrence of
// the same inputs, token by token, on five value heads spread over the 48 (two q/k groups whole
// or in part, the first and the last head).
//
// The bound and its argument. Per output row (a token's 128 values of one head) the error is
// |o - o64| / |o64| (Euclidean), the norm floored at a quarter of the head's RMS row norm (a
// smaller row is a cancellation, where a path's error is its head's size times its rounding);
// per carried state (one head's 128 x 128) the same in the Frobenius norm. One bf16 rounding is
// at most 2^-9 relative and about 2^-9 / sqrt(3) RMS. The vendored path stores five bf16
// intermediates between the inputs and a row (q and k normalized, W or U, the new values, the
// chunk's state) and the output itself, and its rows land at 2.6 x 2^-9 RMS (measured, printed),
// their worst over 20,480 rows at about 4.6 times that, the tail of the same spread. So:
//   - kRmsBound = 4 x 2^-9 on each path's RMS row error against fp64, and kWorstBound = 16 x 2^-9
//     on its worst row; kRmsBound on each carried state;
//   - ours no farther from fp64 than the vendored path, worst row and RMS each, wherever the
//     vendored op runs its chunked kernels (64 tokens and more; below, it runs its fp32
//     recurrent kernel, and so does the layer body). Ours rounds to f16, 2^-11 per operand, q and
//     k included, and accumulates in f16 over spans of 32 products that stay below 2^15, each
//     folded into fp32; only its bf16 output is a bf16 rounding. Measured on the RTX 5090 its
//     rows land at 0.93 x 2^-9 RMS (the output's own rounding) and its states about six times
//     closer to fp64 than the vendored path's;
//   - ours against the vendored path on all 48 heads: every row within the triangle
//     inequality's sum of the two paths' bounds, 2 x kWorstBound (RMS 2 x kRmsBound), and every
//     carried state within 2 x kRmsBound.
//
// The cases: one call at each of the 27B's prefill chunks, 1024, 2048 and 4096, from a nonzero
// state; three calls in a row, 4096, 4096 and 777 (a ragged remainder chunk), the state carried in
// place between them as the layer body does; and short ragged calls (64, 100, 1000, 40), so the
// chunk's masked tail and a prefill of one 32-token chunk plus a remainder are covered too. The
// inputs stress the solve: consecutive keys strongly correlated (a shared direction plus noise),
// beta up to 0.999, decays from 0.002 to 3 per token, values of different sizes per head.
//
// The route per GPU generation and its measurement switch, IGNIS_GDN_PREFILL, are checked on the
// host: the generation's default, both names, and another value refused by name.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or busy GPU fails this
// test rather than skipping it.

#include "gdn_prefill.h"
#include "moe_fixture.h"

#include "ninfer/ops/gated_delta_net.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
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

using namespace moe_test;

namespace {

// The 27B's GDN geometry (crates/core/src/compute.rs `ModelConfig::qwen38_27b`).
constexpr int32_t kQkHeads = 16, kValueHeads = 48, kDim = 128;
constexpr int32_t kGroup = kValueHeads / kQkHeads;
constexpr int32_t kStateFloats = kDim * kDim * kValueHeads;
const float kScale = 1.0f / std::sqrt(static_cast<float>(kDim));
// The value heads the fp64 recurrence runs on.
constexpr int32_t kRefHeads[] = {0, 1, 2, 25, 47};
constexpr int32_t kRefCount = sizeof(kRefHeads) / sizeof(kRefHeads[0]);

// The bounds the header argues.
constexpr double kRmsBound = 4.0 * 0x1p-9;
constexpr double kWorstBound = 16.0 * 0x1p-9;

struct Inputs {
  std::vector<uint16_t> q, k, v;  // BF16 [T][heads][128]
  std::vector<float> g, beta;     // FP32 [T][48]
};

// Keys and queries: a per-head direction shared by every token plus noise, so consecutive keys
// are strongly correlated (the triangular solve's hard case), at token amplitudes from 0.25 to 4
// (the normalization removes them).
Inputs make_inputs(uint32_t stream, int32_t T) {
  Inputs in;
  const std::size_t qk = static_cast<std::size_t>(T) * kQkHeads * kDim;
  const std::size_t vv = static_cast<std::size_t>(T) * kValueHeads * kDim;
  in.q.resize(qk);
  in.k.resize(qk);
  in.v.resize(vv);
  for (int32_t t = 0; t < T; ++t) {
    for (int32_t h = 0; h < kQkHeads; ++h) {
      const float amp = 0.25f + 3.75f * (0.5f + 0.5f * hash_uniform(stream + 11, static_cast<uint64_t>(t) * kQkHeads + h, 1.0f));
      for (int32_t i = 0; i < kDim; ++i) {
        const std::size_t at = (static_cast<std::size_t>(t) * kQkHeads + h) * kDim + i;
        const float kd = hash_uniform(5, static_cast<uint64_t>(h) * kDim + i, 1.0f);
        const float qd = hash_uniform(6, static_cast<uint64_t>(h) * kDim + i, 1.0f);
        in.k[at] = f32_to_bf16(amp * (kd + 0.6f * hash_uniform(stream + 1, at, 1.0f)));
        in.q[at] = f32_to_bf16(amp * (qd + 0.9f * hash_uniform(stream + 2, at, 1.0f)));
      }
    }
  }
  for (std::size_t i = 0; i < vv; ++i) {
    const int32_t h = static_cast<int32_t>((i / kDim) % kValueHeads);
    in.v[i] = f32_to_bf16(hash_uniform(stream + 3, i, 0.5f + static_cast<float>(h % 7)));
  }
  in.g.resize(static_cast<std::size_t>(T) * kValueHeads);
  in.beta.resize(in.g.size());
  for (std::size_t i = 0; i < in.g.size(); ++i) {
    const float u = 0.5f + 0.5f * hash_uniform(stream + 4, i, 1.0f);
    in.g[i] = (i % 97 == 13) ? -3.0f : -(0.002f + 0.3f * u * u);
    const float b = 0.5f + 0.5f * hash_uniform(stream + 5, i, 1.0f);
    in.beta[i] = (i % 89 == 7) ? 0.999f : 0.02f + 0.97f * b;
  }
  return in;
}

std::vector<float> initial_state() {
  std::vector<float> s(kStateFloats);
  for (std::size_t i = 0; i < s.size(); ++i) s[i] = hash_uniform(77, i, 0.5f);
  return s;
}

// --- the fp64 recurrence, one value head --------------------------------------------------------

void unit(const uint16_t *x, double *out) {
  double sum = 0.0;
  for (int i = 0; i < kDim; ++i) sum += static_cast<double>(bf16_to_f32(x[i])) * bf16_to_f32(x[i]);
  const double inv = 1.0 / std::sqrt(sum + 1e-6);
  for (int i = 0; i < kDim; ++i) out[i] = bf16_to_f32(x[i]) * inv;
}

// S [value][key] in fp64, carried across calls; `out` [T][128].
void reference_head(const Inputs &in, int32_t T, int32_t head, std::vector<double> &S, std::vector<double> &out) {
  const int32_t qk = head / kGroup;
  out.assign(static_cast<std::size_t>(T) * kDim, 0.0);
  std::vector<double> qn(kDim), kn(kDim), pred(kDim);
  for (int32_t t = 0; t < T; ++t) {
    unit(&in.q[(static_cast<std::size_t>(t) * kQkHeads + qk) * kDim], qn.data());
    unit(&in.k[(static_cast<std::size_t>(t) * kQkHeads + qk) * kDim], kn.data());
    const double a = std::exp(static_cast<double>(in.g[static_cast<std::size_t>(t) * kValueHeads + head]));
    const double b = in.beta[static_cast<std::size_t>(t) * kValueHeads + head];
    const uint16_t *v = &in.v[(static_cast<std::size_t>(t) * kValueHeads + head) * kDim];
    for (int d = 0; d < kDim; ++d) {
      double p = 0.0;
      const double *row = &S[static_cast<std::size_t>(d) * kDim];
      for (int i = 0; i < kDim; ++i) p += row[i] * kn[i];
      pred[d] = p;
    }
    for (int d = 0; d < kDim; ++d) {
      const double u = b * (static_cast<double>(bf16_to_f32(v[d])) - a * pred[d]);
      double *row = &S[static_cast<std::size_t>(d) * kDim];
      double o = 0.0;
      for (int i = 0; i < kDim; ++i) {
        row[i] = a * row[i] + u * kn[i];
        o += row[i] * qn[i];
      }
      out[static_cast<std::size_t>(t) * kDim + d] = kScale * o;
    }
  }
}

// Sets (or, with null, removes) a variable in the operating system's environment, as a host
// process's own runtime does (Rust's std::env::set_var).
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

// The route per generation and the measurement switch (kernel/src/gdn_prefill.h `route_for_load`):
// ours by default on 8.x and on Blackwell, either one when named, anything else refused naming the
// variable.
void check_route() {
  constexpr const char *kSwitch = "IGNIS_GDN_PREFILL";
  set_environment(kSwitch, nullptr);
  check(ignis::gdn_prefill::route_for_load(8), "8.x runs ours by default");
  check(ignis::gdn_prefill::route_for_load(12), "Blackwell runs ours by default");
  set_environment(kSwitch, "vendored");
  check(!ignis::gdn_prefill::route_for_load(8), "IGNIS_GDN_PREFILL=vendored runs the vendored kernels on 8.x");
  check(!ignis::gdn_prefill::route_for_load(12), "IGNIS_GDN_PREFILL=vendored runs the vendored kernels on Blackwell");
  set_environment(kSwitch, "ours");
  check(ignis::gdn_prefill::route_for_load(12), "IGNIS_GDN_PREFILL=ours runs ours");
  set_environment(kSwitch, "fp16");
  bool refused = false;
  try {
    (void)ignis::gdn_prefill::route_for_load(8);
  } catch (const std::invalid_argument &e) {
    refused = std::string(e.what()).find(kSwitch) != std::string::npos;
  }
  check(refused, "another value of IGNIS_GDN_PREFILL is refused by name");
  set_environment(kSwitch, nullptr);
}

// --- the two device paths ---------------------------------------------------------------------

struct Device {
  DeviceBytes q, k, v, g, beta, out, state;
};

Device upload_call(const Inputs &in, int32_t T, const std::vector<float> &state) {
  Device d{DeviceBytes(in.q.size() * 2), DeviceBytes(in.k.size() * 2), DeviceBytes(in.v.size() * 2),
           DeviceBytes(in.g.size() * 4), DeviceBytes(in.beta.size() * 4),
           DeviceBytes(static_cast<std::size_t>(T) * kValueHeads * kDim * 2), DeviceBytes(state.size() * 4)};
  upload(d.q, in.q);
  upload(d.k, in.k);
  upload(d.v, in.v);
  upload(d.g, in.g);
  upload(d.beta, in.beta);
  upload(d.state, state);
  MOE_CUDA(cudaMemset(d.out.p, 0xFF, d.out.bytes));  // NaN everywhere a path leaves unwritten
  return d;
}

struct Tensors {
  ninfer::Tensor q, k, v, g, beta, out, state;
};

Tensors tensors(const Device &d, int32_t T) {
  return {ninfer::Tensor(d.q.p, ninfer::DType::BF16, {kDim, kQkHeads, T, 1}),
          ninfer::Tensor(d.k.p, ninfer::DType::BF16, {kDim, kQkHeads, T, 1}),
          ninfer::Tensor(d.v.p, ninfer::DType::BF16, {kDim, kValueHeads, T, 1}),
          ninfer::Tensor(d.g.p, ninfer::DType::FP32, {kValueHeads, T, 1, 1}),
          ninfer::Tensor(d.beta.p, ninfer::DType::FP32, {kValueHeads, T, 1, 1}),
          ninfer::Tensor(d.out.p, ninfer::DType::BF16, {kDim, kValueHeads, T, 1}),
          ninfer::Tensor(d.state.p, ninfer::DType::FP32, {kDim, kDim, kValueHeads, 1})};
}

struct Result {
  std::vector<uint16_t> out;  // BF16 [T][48][128]
  std::vector<float> state;   // FP32 [48][128][128]
};

Result download_call(const Device &d, int32_t T) {
  MOE_CUDA(cudaDeviceSynchronize());
  return {download<uint16_t>(d.out.p, static_cast<std::size_t>(T) * kValueHeads * kDim),
          download<float>(d.state.p, kStateFloats)};
}

// The layer body's call (kernel/src/gdn_layer.cu): normalize_qk, the state in place.
Result run_vendored(const Inputs &in, int32_t T, const std::vector<float> &state, ninfer::DeviceArena &arena,
                    cudaStream_t stream) {
  Device d = upload_call(in, T, state);
  Tensors t = tensors(d, T);
  ninfer::ops::gated_delta_net(t.q, t.k, t.v, t.g, t.beta, kScale, /*normalize_qk=*/true, arena, t.state, t.state,
                               t.out, stream);
  return download_call(d, T);
}

// Ours, the same call: the state in place, the workspace the op asks for.
Result run_ours(const Inputs &in, int32_t T, const std::vector<float> &state, const DeviceBytes &workspace,
                cudaStream_t stream) {
  Device d = upload_call(in, T, state);
  Tensors t = tensors(d, T);
  ignis::gdn_prefill::chunked(t.q, t.k, t.v, t.g, t.beta, kScale, t.state, t.state, t.out, workspace.p,
                              workspace.bytes, stream);
  return download_call(d, T);
}

// --- comparisons --------------------------------------------------------------------------------

struct Distance {
  double worst = 0.0, sum_sq = 0.0;
  std::size_t rows = 0;
  void add(double num_sq, double den_sq) {
    const double r = std::sqrt(num_sq / std::max(den_sq, 1e-30));
    worst = std::max(worst, r);
    sum_sq += r * r;
    ++rows;
  }
  double rms() const { return rows == 0 ? 0.0 : std::sqrt(sum_sq / static_cast<double>(rows)); }
};

// Output rows of the reference heads against fp64, each relative to its own norm floored at
// kRowFloor of the head's RMS row norm over the call: a row much smaller than its head's is a
// cancellation in S q, where the error a path makes is the head's size times its rounding, not
// the row's.
constexpr double kRowFloor = 0.25;

double head_rms_row(const std::vector<double> &ref, int32_t T) {
  double sum = 0.0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(T) * kDim; ++i) sum += ref[i] * ref[i];
  return std::sqrt(sum / static_cast<double>(T));
}

Distance rows_vs_reference(const std::vector<uint16_t> &out, int32_t T, const std::vector<std::vector<double>> &ref) {
  Distance dist;
  for (int r = 0; r < kRefCount; ++r) {
    const double floor_sq = std::pow(kRowFloor * head_rms_row(ref[r], T), 2);
    for (int32_t t = 0; t < T; ++t) {
      double num = 0.0, den = 0.0;
      for (int d = 0; d < kDim; ++d) {
        const double want = ref[r][static_cast<std::size_t>(t) * kDim + d];
        const double got = bf16_to_f32(out[(static_cast<std::size_t>(t) * kValueHeads + kRefHeads[r]) * kDim + d]);
        num += (got - want) * (got - want);
        den += want * want;
      }
      dist.add(std::isfinite(num) ? num : 1e300, std::max(den, floor_sq));
    }
  }
  return dist;
}

// Our output rows against the vendored path's on every head, floored at each head's RMS row.
Distance rows_vs_vendored(const std::vector<uint16_t> &ours, const std::vector<uint16_t> &vendored, int32_t T) {
  Distance dist;
  for (int32_t h = 0; h < kValueHeads; ++h) {
    double head = 0.0;
    for (int32_t t = 0; t < T; ++t) {
      for (int d = 0; d < kDim; ++d) {
        const double v = bf16_to_f32(vendored[(static_cast<std::size_t>(t) * kValueHeads + h) * kDim + d]);
        head += v * v;
      }
    }
    const double floor_sq = kRowFloor * kRowFloor * head / static_cast<double>(T);
    for (int32_t t = 0; t < T; ++t) {
      double num = 0.0, den = 0.0;
      for (int d = 0; d < kDim; ++d) {
        const std::size_t at = (static_cast<std::size_t>(t) * kValueHeads + h) * kDim + d;
        const double want = bf16_to_f32(vendored[at]), got = bf16_to_f32(ours[at]);
        num += (got - want) * (got - want);
        den += want * want;
      }
      dist.add(std::isfinite(num) ? num : 1e300, std::max(den, floor_sq));
    }
  }
  return dist;
}

Distance states_vs_vendored(const std::vector<float> &ours, const std::vector<float> &vendored) {
  Distance dist;
  for (int32_t h = 0; h < kValueHeads; ++h) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < kDim * kDim; ++i) {
      const std::size_t at = static_cast<std::size_t>(h) * kDim * kDim + i;
      const double diff = static_cast<double>(ours[at]) - vendored[at];
      num += diff * diff;
      den += static_cast<double>(vendored[at]) * vendored[at];
    }
    dist.add(std::isfinite(num) ? num : 1e300, den);
  }
  return dist;
}

// The carried state of the reference heads against fp64.
Distance states_vs_reference(const std::vector<float> &state, const std::vector<std::vector<double>> &ref) {
  Distance dist;
  for (int r = 0; r < kRefCount; ++r) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < kDim * kDim; ++i) {
      const double want = ref[r][i];
      const double got = state[static_cast<std::size_t>(kRefHeads[r]) * kDim * kDim + i];
      num += (got - want) * (got - want);
      den += want * want;
    }
    dist.add(std::isfinite(num) ? num : 1e300, den);
  }
  return dist;
}

void check_distance(const std::string &what, const Distance &d, double worst_bound, double rms_bound) {
  std::printf("    %-44s worst %.3e (%.2f of bound)  rms %.3e (%.2f)  over %zu\n", what.c_str(), d.worst,
              d.worst / worst_bound, d.rms(), d.rms() / rms_bound, d.rows);
  check(d.worst <= worst_bound, what + ": worst " + std::to_string(d.worst) + " > " + std::to_string(worst_bound));
  check(d.rms() <= rms_bound, what + ": rms " + std::to_string(d.rms()) + " > " + std::to_string(rms_bound));
}

// One case: consecutive calls from one state, the state carried between them.
void run_case(const char *name, const std::vector<int32_t> &calls, uint32_t stream, ninfer::DeviceArena &arena,
              const DeviceBytes &workspace, cudaStream_t cuda) {
  std::printf("  %s\n", name);
  std::vector<float> s_vendored = initial_state();
  std::vector<float> s_ours = s_vendored;
  std::vector<std::vector<double>> s_ref(kRefCount);
  for (int r = 0; r < kRefCount; ++r) {
    s_ref[r].assign(s_vendored.begin() + static_cast<std::ptrdiff_t>(kRefHeads[r]) * kDim * kDim,
                    s_vendored.begin() + static_cast<std::ptrdiff_t>(kRefHeads[r] + 1) * kDim * kDim);
  }
  for (std::size_t c = 0; c < calls.size(); ++c) {
    const int32_t T = calls[c];
    const Inputs in = make_inputs(stream + 101 * static_cast<uint32_t>(c), T);
    std::vector<std::vector<double>> o_ref(kRefCount);
    std::vector<std::thread> workers;
    for (int r = 0; r < kRefCount; ++r) {
      workers.emplace_back([&, r] { reference_head(in, T, kRefHeads[r], s_ref[r], o_ref[r]); });
    }
    const Result vendored = run_vendored(in, T, s_vendored, arena, cuda);
    const Result ours = run_ours(in, T, s_ours, workspace, cuda);
    for (auto &w : workers) w.join();
    const std::string at = "call " + std::to_string(c + 1) + " (T=" + std::to_string(T) + ")";
    const Distance v_rows = rows_vs_reference(vendored.out, T, o_ref);
    const Distance v_state = states_vs_reference(vendored.state, s_ref);
    const Distance o_rows = rows_vs_reference(ours.out, T, o_ref);
    const Distance o_state = states_vs_reference(ours.state, s_ref);
    check_distance(at + " vendored rows vs fp64", v_rows, kWorstBound, kRmsBound);
    check_distance(at + " vendored state vs fp64", v_state, kRmsBound, kRmsBound);
    check_distance(at + " ours rows vs fp64", o_rows, kWorstBound, kRmsBound);
    check_distance(at + " ours state vs fp64", o_state, kRmsBound, kRmsBound);
    // Below kMinTokens the vendored op runs its fp32 recurrent kernel alone, and so does the layer
    // body (ours is not called there): only the chunked kernels it replaces are the yardstick.
    if (T >= ignis::gdn_prefill::kMinTokens) {
      check(o_rows.worst <= v_rows.worst && o_rows.rms() <= v_rows.rms(), at + ": our rows are farther from fp64");
      check(o_state.worst <= v_state.worst && o_state.rms() <= v_state.rms(), at + ": our state is farther from fp64");
    }
    check_distance(at + " ours vs vendored rows, 48 heads", rows_vs_vendored(ours.out, vendored.out, T),
                   2 * kWorstBound, 2 * kRmsBound);
    check_distance(at + " ours vs vendored state, 48 heads", states_vs_vendored(ours.state, vendored.state),
                   2 * kRmsBound, 2 * kRmsBound);
    s_vendored = vendored.state;
    s_ours = ours.state;
  }
}

} // namespace

int main() {
  int device = 0;
  MOE_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  MOE_CUDA(cudaGetDeviceProperties(&prop, device));
  std::printf("gdn_prefill: sm_%d%d\n", prop.major, prop.minor);
  cudaStream_t stream = nullptr;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  ninfer::DeviceArena arena(
      ninfer::ops::gated_delta_net_workspace_capacity_bytes(kQkHeads, kValueHeads, /*normalize_qk=*/true, 1, 4096));

  DeviceBytes workspace(ignis::gdn_prefill::workspace_bytes(kQkHeads, kValueHeads, 4096));
  std::printf("gdn_prefill: the recurrence runs %d warps per CTA here\n", ignis::gdn_prefill::recurrence_warps());

  // The load reserves our workspace at the prefill chunk in place of the vendored op's
  // (kernel/src/model.cu's gdn_layer_scratch_bytes): at every width the body sends here it must not
  // shrink as the call widens, so the chunk's holds every narrower call, and it is no larger than
  // the vendored op's, so the route never grows the plan.
  for (int32_t T = ignis::gdn_prefill::kMinTokens; T <= 16384; ++T) {
    const std::size_t ours = ignis::gdn_prefill::workspace_bytes(kQkHeads, kValueHeads, T);
    const std::size_t vendored =
        ninfer::ops::gated_delta_net_workspace_capacity_bytes(kQkHeads, kValueHeads, /*normalize_qk=*/true, 1, T);
    if (ours > vendored) {
      check(false, "our workspace at T=" + std::to_string(T) + " exceeds the vendored reservation");
      break;
    }
    if (ours < ignis::gdn_prefill::workspace_bytes(kQkHeads, kValueHeads, T - 1)) {
      check(false, "our workspace at T=" + std::to_string(T) + " is smaller than at one token fewer");
      break;
    }
  }

  run_case("chunk 1024", {1024}, 1000, arena, workspace, stream);
  run_case("chunk 2048", {2048}, 2000, arena, workspace, stream);
  run_case("chunk 4096", {4096}, 3000, arena, workspace, stream);
  run_case("chunks 4096, 4096, 777 (the state carried in place)", {4096, 4096, 777}, 4000, arena, workspace, stream);
  run_case("ragged 64, 100, 1000, 40", {64, 100, 1000, 40}, 5000, arena, workspace, stream);

  check_route();

  // Refusals, each by name before any launch: value heads that are not a multiple of the q/k
  // heads, a v at another T, and a workspace too small.
  {
    const Inputs in = make_inputs(9, 64);
    Device d = upload_call(in, 64, initial_state());
    Tensors t = tensors(d, 64);
    const auto refuses = [&](const char *what, auto &&call) {
      bool refused = false;
      try {
        call();
      } catch (const std::invalid_argument &) {
        refused = true;
      }
      check(refused, std::string("refuses ") + what);
    };
    refuses("47 value heads over 16 q/k heads", [&] {
      ninfer::Tensor v47(d.v.p, ninfer::DType::BF16, {kDim, 47, 64, 1});
      ninfer::Tensor out47(d.out.p, ninfer::DType::BF16, {kDim, 47, 64, 1});
      ignis::gdn_prefill::chunked(t.q, t.k, v47, t.g, t.beta, kScale, t.state, t.state, out47, workspace.p,
                                  workspace.bytes, stream);
    });
    refuses("v at another T", [&] {
      ninfer::Tensor v63(d.v.p, ninfer::DType::BF16, {kDim, kValueHeads, 63, 1});
      ignis::gdn_prefill::chunked(t.q, t.k, v63, t.g, t.beta, kScale, t.state, t.state, t.out, workspace.p,
                                  workspace.bytes, stream);
    });
    refuses("a short workspace", [&] {
      ignis::gdn_prefill::chunked(t.q, t.k, t.v, t.g, t.beta, kScale, t.state, t.state, t.out, workspace.p,
                                  ignis::gdn_prefill::workspace_bytes(kQkHeads, kValueHeads, 64) - 256, stream);
    });
    MOE_CUDA(cudaDeviceSynchronize());
  }

  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "gdn_prefill: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("gdn_prefill: all checks passed\n");
  return EXIT_SUCCESS;
}
