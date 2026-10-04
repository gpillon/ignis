// The Flash-Next MoE ops' microbenchmark -- OURS (spec flash-next/02 Acceptance 6, GitHub #300).
//
// Achieved bandwidth (or throughput) per op against the device roofline, written as a report.
// It is a pass/fail CTest for one figure only, the spec's decode floor: the routed-expert decode
// launch for one token must read its bytes at >= 50% of the DRAM roofline at the study's K mix.
//
// The roofline is the device's theoretical DRAM bandwidth (memory clock x bus width x 2, from
// the device attributes); a sustained read measured by a plain streaming kernel is printed beside
// it. Expert weights must come from DRAM, as they do in a real decode (each layer selects other
// experts): the bench holds a pool of 320 distinct experts (~480 MB, several times the L2) with
// the study's K mix -- run 8, layer 1: gate/up 221/114/165/12 and down 233/106/159/14 of 512 at
// K = 2/2.5/3/4 -- and every timed call routes to ten experts the previous 31 calls did not use.
//
// ADR 0006 / docs/agents/testing.md: run on a free GPU, alone; a contended card reads slow.

#include "ignis_moe.h"
#include "moe_experts_common.h"
#include "moe_fixture.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

constexpr int kPool = 320;
constexpr double kDecodeFloor = 0.50;

double theoretical_dram_gbs() {
  int device = 0, clock_khz = 0, bus_bits = 0;
  MOE_CUDA(cudaGetDevice(&device));
  MOE_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device));
  MOE_CUDA(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device));
  return 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

__global__ void stream_read_kernel(const uint4 *p, std::size_t n, uint4 *sink) {
  uint4 acc{0, 0, 0, 0};
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = __ldcs(p + i);
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if ((acc.x & acc.y & acc.z & acc.w) == 0xFFFFFFFFu) sink[0] = acc;
}

double measured_read_gbs() {
  const std::size_t bytes = std::size_t{1} << 30;
  DeviceBytes buf(bytes), sink(16);
  MOE_CUDA(cudaMemset(buf.p, 1, bytes));
  cudaEvent_t a, b;
  MOE_CUDA(cudaEventCreate(&a));
  MOE_CUDA(cudaEventCreate(&b));
  stream_read_kernel<<<4096, 256>>>(buf.as<uint4>(), bytes / 16, sink.as<uint4>());
  MOE_CUDA(cudaEventRecord(a));
  const int reps = 10;
  for (int i = 0; i < reps; ++i) stream_read_kernel<<<4096, 256>>>(buf.as<uint4>(), bytes / 16, sink.as<uint4>());
  MOE_CUDA(cudaEventRecord(b));
  MOE_CUDA(cudaEventSynchronize(b));
  float ms = 0.0f;
  MOE_CUDA(cudaEventElapsedTime(&ms, a, b));
  return static_cast<double>(bytes) * reps / (ms * 1e-3) / 1e9;
}

// K class (k2) of pool expert i at the study's mix, interleaved so any window of the pool sees it.
uint32_t mix_k2(int i, const int (&counts)[4]) {
  const uint32_t k2[] = {4, 5, 6, 8};
  const int slot = (i * 389) % 512;  // distinct for i < 512: a spread sample of the 512 experts
  int edge = 0;
  for (int c = 0; c < 4; ++c) {
    edge += counts[c];
    if (slot < edge) return k2[c];
  }
  return 8;
}

struct Pool {
  DeviceBytes buffer;
  std::vector<ignis_moe_slot> slots;
  std::vector<uint64_t> data_bytes;  // per pool expert: gate/up + down record data, no padding
  explicit Pool(std::size_t total) : buffer(total) {}
};

std::unique_ptr<Pool> make_pool() {
  const int gu_counts[4] = {221, 114, 165, 12};
  const int dn_counts[4] = {233, 106, 159, 14};
  std::vector<Record> gu, dn;
  std::size_t total = 0;
  std::vector<std::vector<uint8_t>> blobs;
  for (int i = 0; i < kPool; ++i) {
    Record g = make_record(70000 + 4 * i, kH, 2 * kI, mix_k2(i, gu_counts), 0.03f, 0.3f);
    Record d = make_record(90000 + 4 * i, kI, kH, mix_k2(i * 7 + 1, dn_counts), 0.05f, 0.1f);
    blobs.push_back(g.bytes());
    blobs.push_back(d.bytes());
    total += blobs[blobs.size() - 2].size() + blobs.back().size();
    gu.push_back(std::move(g));
    dn.push_back(std::move(d));
  }
  auto pool = std::make_unique<Pool>(total);
  pool->slots.assign(static_cast<std::size_t>(kE) * 2, ignis_moe_slot{nullptr, 0, 0});
  std::size_t at = 0;
  for (int i = 0; i < kPool; ++i) {
    for (int p = 0; p < 2; ++p) {
      const auto &blob = blobs[2 * i + p];
      MOE_CUDA(cudaMemcpy(static_cast<char *>(pool->buffer.p) + at, blob.data(), blob.size(), cudaMemcpyHostToDevice));
      const Record &r = p == 0 ? gu[i] : dn[i];
      pool->slots[i * 2 + p] = {static_cast<char *>(pool->buffer.p) + at, r.k2, 0};
      at += blob.size();
    }
    pool->data_bytes.push_back(gu[i].words.size() * 2 + (gu[i].suh.size() + gu[i].svh.size()) * 2 +
                               dn[i].words.size() * 2 + (dn[i].suh.size() + dn[i].svh.size()) * 2);
  }
  return pool;
}

// `calls` routings of `tokens` tokens, each call's 10 * tokens experts distinct and taken from a
// window of the pool that the next several calls (>= 300 MB of weights) do not touch.
std::vector<int32_t> make_calls(int calls, int tokens) {
  const int width = kTop * tokens;
  const int windows = kPool / width;
  std::vector<int32_t> ids(static_cast<std::size_t>(calls) * width);
  for (int c = 0; c < calls; ++c) {
    for (int i = 0; i < width; ++i) ids[static_cast<std::size_t>(c) * width + i] = (c % windows) * width + i;
  }
  return ids;
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  cudaDeviceProp prop{};
  MOE_CUDA(cudaGetDeviceProperties(&prop, 0));
  const double roof = theoretical_dram_gbs();
  const double sustained = measured_read_gbs();
  std::printf("MoE microbenchmark on %s: DRAM roofline %.0f GB/s (theoretical), streaming read %.0f GB/s (%.0f%%), L2 %d MB\n",
              prop.name, roof, sustained, 100.0 * sustained / roof, prop.l2CacheSize >> 20);

  const auto pool = make_pool();
  {
    const int gu_counts[4] = {221, 114, 165, 12};
    const int dn_counts[4] = {233, 106, 159, 14};
    double gk = 0.0, dk = 0.0;
    for (int i = 0; i < kPool; ++i) {
      gk += mix_k2(i, gu_counts) / 2.0;
      dk += mix_k2(i * 7 + 1, dn_counts) / 2.0;
    }
    std::printf("  pool: %d experts, mean K gate/up %.3f, down %.3f (study: 2.48 / 2.47)\n", kPool, gk / kPool, dk / kPool);
  }
  DeviceBytes d_slots(pool->slots.size() * sizeof(ignis_moe_slot));
  upload(d_slots, pool->slots);
  DeviceBytes workspace(ignis_moe_workspace_bytes(64)), acc(64 * kH * 8);
  MOE_RC(ignis_moe_workspace_init(workspace.p, 64, acc.as<int64_t>(), nullptr));
  const std::vector<uint16_t> x = make_tokens(8, 31, 2.0f);
  DeviceBytes dx(x.size() * 2);
  upload(dx, x);

  bool floor_ok = true;
  for (int tokens : {1, 2, 3}) {
    const int calls = 320;
    const std::vector<int32_t> ids = make_calls(calls, tokens);
    std::vector<float> w(ids.size(), 0.1f);
    DeviceBytes dids(ids.size() * 4), dw(w.size() * 4);
    upload(dids, ids);
    upload(dw, w);
    // Bytes one call reads: each distinct expert's two records (data, no padding) and x.
    double bytes = 0.0;
    for (int c = 0; c < calls; ++c) {
      std::vector<int32_t> u(ids.begin() + static_cast<std::ptrdiff_t>(c) * tokens * kTop,
                             ids.begin() + static_cast<std::ptrdiff_t>(c + 1) * tokens * kTop);
      std::sort(u.begin(), u.end());
      u.erase(std::unique(u.begin(), u.end()), u.end());
      for (int32_t e : u) bytes += static_cast<double>(pool->data_bytes[e]);
      bytes += tokens * kH * 2.0;
    }
    bytes /= calls;
    auto launch = [&](int c) {
      MOE_RC(ignis_moe_experts_decode(dx.p, tokens, dids.as<int32_t>() + static_cast<std::size_t>(c) * tokens * kTop,
                                      dw.as<float>() + static_cast<std::size_t>(c) * tokens * kTop,
                                      d_slots.as<ignis_moe_slot>(), workspace.p, acc.as<int64_t>(), nullptr));
    };
    for (int c = 0; c < 64; ++c) launch(c);  // warm-up, also cycles the L2
    cudaEvent_t a, b;
    MOE_CUDA(cudaEventCreate(&a));
    MOE_CUDA(cudaEventCreate(&b));
    MOE_CUDA(cudaEventRecord(a));
    for (int c = 0; c < calls; ++c) launch(c);
    MOE_CUDA(cudaEventRecord(b));
    MOE_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    MOE_CUDA(cudaEventElapsedTime(&ms, a, b));
    const double us = 1e3 * ms / calls;
    const double gbs = bytes / (us * 1e-6) / 1e9;
    std::printf("  routed decode, %d token(s): %.1f us per layer, %.2f MB read, %.0f GB/s = %.1f%% of roofline (%.1f%% of streaming read)\n",
                tokens, us, bytes / 1e6, gbs, 100.0 * gbs / roof, 100.0 * gbs / sustained);
    if (tokens == 1 && gbs < kDecodeFloor * roof) floor_ok = false;
  }
  if (!floor_ok) {
    std::fprintf(stderr, "bench_moe: the 1-token decode is below %.0f%% of the DRAM roofline\n", 100.0 * kDecodeFloor);
    return 1;
  }
  std::printf("bench_moe: decode floor (>= %.0f%% of roofline at 1 token) met\n", 100.0 * kDecodeFloor);
  return 0;
}
