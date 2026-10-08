// The Flash-Next MoE ops' microbenchmark -- OURS (spec flash-next/02 Acceptance 6, GitHub #300).
//
// Achieved bandwidth (or throughput) per op against the device roofline, written as a report.
// One figure is a pass/fail, opt-in with --check-floor: the spec's decode floor, the
// routed-expert decode launch for one token reading its bytes at >= 50% of the DRAM roofline at
// the study's K mix. It is a tool, not a CTest, while that floor is an open criterion (2026-10-05:
// 33.5%), so the leaf's CTest stage stays usable; run it by hand on a free card:
//
//   kernel/build/tests/ignis_kernel_moe_bench.exe [--check-floor] [--decode-only] [--trace]
//
// --decode-only measures the routed decode routes alone; --trace adds each route's phase trace
// (moe_trace.h): the 1- and 3-token launch from DRAM and from the L2, stamped per unit.
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

#include "../src/moe_trace.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
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
  int size = 0;
  DeviceBytes buffer;
  std::vector<ignis_moe_slot> slots;
  std::vector<uint64_t> data_bytes;  // per pool expert: gate/up + down record data, no padding
  explicit Pool(std::size_t total) : buffer(total) {}
};

// A pool of `size` experts at the study's K mix, or every projection at `uniform_k2`.
std::unique_ptr<Pool> make_pool(int size, uint32_t uniform_k2) {
  const int gu_counts[4] = {221, 114, 165, 12};
  const int dn_counts[4] = {233, 106, 159, 14};
  std::vector<Record> gu, dn;
  std::size_t total = 0;
  std::vector<std::vector<uint8_t>> blobs;
  for (int i = 0; i < size; ++i) {
    Record g = make_record(70000 + 4 * i, kH, 2 * kI, uniform_k2 ? uniform_k2 : mix_k2(i, gu_counts), 0.03f, 0.3f);
    Record d = make_record(90000 + 4 * i, kI, kH, uniform_k2 ? uniform_k2 : mix_k2(i * 7 + 1, dn_counts), 0.05f, 0.1f);
    blobs.push_back(g.bytes());
    blobs.push_back(d.bytes());
    total += blobs[blobs.size() - 2].size() + blobs.back().size();
    gu.push_back(std::move(g));
    dn.push_back(std::move(d));
  }
  auto pool = std::make_unique<Pool>(total);
  pool->size = size;
  pool->slots.assign(static_cast<std::size_t>(kE) * 2, ignis_moe_slot{nullptr, 0, 0});
  std::size_t at = 0;
  for (int i = 0; i < size; ++i) {
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
// window of the pool that the next several calls (well past the L2) do not touch; `windows` = 1
// routes every call to the same experts, which then live in the L2.
std::vector<int32_t> make_calls(int calls, int tokens, int pool_size, int windows = 0) {
  const int width = kTop * tokens;
  if (windows == 0) windows = pool_size / width;
  std::vector<int32_t> ids(static_cast<std::size_t>(calls) * width);
  for (int c = 0; c < calls; ++c) {
    for (int i = 0; i < width; ++i) ids[static_cast<std::size_t>(c) * width + i] = (c % windows) * width + i;
  }
  return ids;
}

// One traced call's records (moe_trace.h), reduced: the launch's span from the first CTA's entry,
// and per unit kind the phase lengths and when the units ran. Words 2-5 are the multiplying
// warps' (begin, ready, multiplied, end) under both kernels; `staged` reads the staged kernel's
// aux stamps (words 6, 8-11), the register kernel's word 6 is a down unit's h wait.
struct CallTrace {
  bool staged = false;
  double span = 0.0;           // first CTA entry to last CTA exit, us
  double setup = 0.0;          // median CTA: entry to its first ticket, us
  double entry_skew = 0.0;     // first to last CTA entry, us
  int units[2] = {0, 0};
  double ready[2] = {0, 0}, mma[2] = {0, 0}, post[2] = {0, 0};  // medians, us
  double prep[2] = {0, 0}, reduce[2] = {0, 0};                   // staged: the aux warps' medians, us
  double first_begin[2] = {0, 0}, last_end[2] = {0, 0};          // from the first entry, us
  double swiglu_post = 0.0;    // register kernel: median mma-to-end of the arrivals that ran the SwiGLU
  double wait_median = 0.0, wait_max = 0.0;  // register kernel: down units' h wait; staged: the aux warps' stage wait
  double mma_busy = 0.0;       // sum of units' multiply phases / (CTAs x span)
  double pre_share = 0.0;      // sum of units' begin-to-ready / (CTAs x span)
  double wait_share = 0.0;     // register kernel: h waits / (CTAs x span); staged: aux busy / (CTAs x span)
  // Per distinct expert (index u): first gate/up begin, last gate/up end, h complete (the last
  // gate/up end, the route's own arrival making h whole), first down begin, last down end.
  std::vector<std::array<double, 5>> chain;
  // 1-us bins from the first entry: mean units in each phase (0 pre, 1 multiply, 2 post, 3 the
  // register kernel's h wait or the staged kernel's aux work) and CTAs alive.
  std::vector<std::array<double, 5>> bins;
};

double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

CallTrace reduce_trace(const std::vector<unsigned long long> &r, bool staged) {
  constexpr int W = IGNIS_MOE_TRACE_WORDS;
  CallTrace c;
  c.staged = staged;
  unsigned long long t0 = ~0ull, t1 = 0, last_entry = 0;
  std::vector<double> setups;
  int ctas = 0;
  for (int i = 0; i < IGNIS_MOE_TRACE_CTAS; ++i) {
    const unsigned long long *e = &r[static_cast<std::size_t>(IGNIS_MOE_TRACE_UNITS + i) * W];
    if (e[0] == 0) continue;
    ++ctas;
    t0 = std::min(t0, e[0]);
    last_entry = std::max(last_entry, e[0]);
    t1 = std::max(t1, e[2]);
    setups.push_back((e[1] - e[0]) * 1e-3);
  }
  c.span = (t1 - t0) * 1e-3;
  c.setup = median_of(setups);
  c.entry_skew = (last_entry - t0) * 1e-3;
  const int nbins = static_cast<int>(c.span) + 1;
  c.bins.assign(nbins, {0, 0, 0, 0, 0});
  auto spread = [&](double a, double b, int col) {  // add interval [a, b) us to the bins
    for (int bin = std::max(0, static_cast<int>(a)); bin < nbins && bin < b; ++bin) {
      const double lo = std::max(a, static_cast<double>(bin)), hi = std::min(b, bin + 1.0);
      if (hi > lo) c.bins[bin][col] += hi - lo;
    }
  };
  for (int i = 0; i < IGNIS_MOE_TRACE_CTAS; ++i) {
    const unsigned long long *e = &r[static_cast<std::size_t>(IGNIS_MOE_TRACE_UNITS + i) * W];
    if (e[0] != 0) spread((e[0] - t0) * 1e-3, (e[2] - t0) * 1e-3, 4);
  }
  auto at = [&](unsigned long long v) { return (static_cast<double>(v) - static_cast<double>(t0)) * 1e-3; };
  std::vector<double> ready[2], mma[2], post[2], prep[2], reduce[2], swiglu, wait;
  double busy = 0.0, pre = 0.0, waited = 0.0;
  c.first_begin[0] = c.first_begin[1] = 1e30;
  for (int u = 0; u < IGNIS_MOE_TRACE_UNITS; ++u) {
    const unsigned long long *e = &r[static_cast<std::size_t>(u) * W];
    if (e[2] == 0) continue;
    const int kind = static_cast<int>(e[1] & 0xFF);
    const int expert = static_cast<int>(e[1] >> 8 & 0xFF);
    const double b = at(e[2]), rd = at(e[3]), m = at(e[4]), en = at(e[5]);
    ++c.units[kind];
    ready[kind].push_back(rd - b);
    mma[kind].push_back(m - rd);
    post[kind].push_back(en - m);
    busy += m - rd;
    pre += rd - b;
    if (static_cast<int>(c.chain.size()) <= expert) c.chain.resize(expert + 1, {1e30, 0.0, 0.0, 1e30, 0.0});
    std::array<double, 5> &ch = c.chain[expert];
    if (kind == 0) {
      ch[0] = std::min(ch[0], b);
      ch[1] = std::max(ch[1], en);
      ch[2] = std::max(ch[2], en);
    } else {
      ch[3] = std::min(ch[3], b);
      ch[4] = std::max(ch[4], en);
    }
    c.first_begin[kind] = std::min(c.first_begin[kind], b);
    c.last_end[kind] = std::max(c.last_end[kind], en);
    spread(b, rd, 0);
    spread(rd, m, 1);
    spread(m, en, 2);
    if (staged) {
      wait.push_back(e[6] * 1e-3);
      if (e[8] != 0) {
        prep[kind].push_back(at(e[9]) - at(e[8]));
        spread(at(e[8]), at(e[9]), 3);
        waited += at(e[9]) - at(e[8]);
      }
      if (e[10] != 0) {
        reduce[kind].push_back(at(e[11]) - at(e[10]));
        spread(at(e[10]), at(e[11]), 3);
        waited += at(e[11]) - at(e[10]);
      }
    } else {
      if (kind == 0 && e[6] == 1) swiglu.push_back(en - m);
      if (kind == 1) {
        const double w = e[6] * 1e-3;
        waited += w;
        wait.push_back(w);
        spread(b, b + w, 3);
      }
    }
  }
  for (int k = 0; k < 2; ++k) {
    c.ready[k] = median_of(ready[k]);
    c.mma[k] = median_of(mma[k]);
    c.post[k] = median_of(post[k]);
    c.prep[k] = median_of(prep[k]);
    c.reduce[k] = median_of(reduce[k]);
  }
  c.swiglu_post = median_of(swiglu);
  c.wait_median = median_of(wait);
  c.wait_max = wait.empty() ? 0.0 : *std::max_element(wait.begin(), wait.end());
  c.mma_busy = ctas > 0 ? busy / (ctas * c.span) : 0.0;
  c.pre_share = ctas > 0 ? pre / (ctas * c.span) : 0.0;
  c.wait_share = ctas > 0 ? waited / (ctas * c.span) : 0.0;
  return c;
}

void print_trace(const char *what, int tokens, std::vector<CallTrace> calls) {
  std::sort(calls.begin(), calls.end(), [](const CallTrace &a, const CallTrace &b) { return a.span < b.span; });
  const CallTrace &m = calls[calls.size() / 2];
  std::printf("  trace, %d token(s), %s: median call of %zu: span %.1f us (spans %.1f..%.1f); CTA entry skew %.1f, "
              "entry to first ticket %.2f us\n",
              tokens, what, calls.size(), m.span, calls.front().span, calls.back().span, m.entry_skew, m.setup);
  const char *pre = m.staged ? "waiting for the operand" : "begin to ready";
  for (int k = 0; k < 2; ++k) {
    std::printf("    %s %d units: %s %.2f, multiply %.2f, to end %.2f us", k == 0 ? "gate/up" : "down", m.units[k], pre,
                m.ready[k], m.mma[k], m.post[k]);
    if (m.staged) std::printf("; aux prepare %.2f, reduce %.2f us", m.prep[k], m.reduce[k]);
    if (!m.staged && k == 0) std::printf(", SwiGLU arrivals to end %.2f", m.swiglu_post);
    std::printf(" (medians); first begins %.1f, last ends %.1f us\n", m.first_begin[k], m.last_end[k]);
  }
  if (m.staged) {
    std::printf("    aux warps' stage wait: median %.2f, max %.2f us\n", m.wait_median, m.wait_max);
    std::printf("    CTA time: mma warps multiplying %.0f%%, waiting for an operand %.0f%%; aux warps busy %.0f%%\n",
                100.0 * m.mma_busy, 100.0 * m.pre_share, 100.0 * m.wait_share);
  } else {
    std::printf("    waiting (down units for h): median %.2f, max %.2f us\n", m.wait_median, m.wait_max);
    std::printf("    CTA time: multiplying %.0f%%, begin to ready %.0f%%, waiting for h %.0f%%\n", 100.0 * m.mma_busy,
                100.0 * m.pre_share, 100.0 * m.wait_share);
  }
  std::printf("    per expert (us from the first entry): gate/up first begin .. last end | down first begin .. last end\n");
  for (std::size_t u = 0; u < m.chain.size(); ++u) {
    std::printf("      u%-2zu %5.1f .. %5.1f | %5.1f .. %5.1f\n", u, m.chain[u][0], m.chain[u][1], m.chain[u][3], m.chain[u][4]);
  }
  std::printf("    us  | units: pre  multiply  post  %s | CTAs alive\n", m.staged ? "aux" : "h-wait");
  for (std::size_t b = 0; b < m.bins.size(); ++b) {
    std::printf("    %3zu | %10.1f %9.1f %5.1f %7.1f | %5.0f\n", b, m.bins[b][0], m.bins[b][1], m.bins[b][2], m.bins[b][3],
                m.bins[b][4]);
  }
}

}  // namespace

int main(int argc, char **argv) {
  bool check_floor = false, decode_only = false, traced = false;
  std::string only_route;  // --route registers|tickets|clusters: that decode route alone
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    check_floor = check_floor || a == "--check-floor";
    decode_only = decode_only || a == "--decode-only";
    traced = traced || a == "--trace";
    if (a == "--route" && i + 1 < argc) only_route = argv[++i];
  }
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // every line out as it is printed, even if a launch dies
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  cudaDeviceProp prop{};
  MOE_CUDA(cudaGetDeviceProperties(&prop, 0));
  const double roof = theoretical_dram_gbs();
  const double sustained = measured_read_gbs();
  std::printf("MoE microbenchmark on %s: DRAM roofline %.0f GB/s (theoretical), streaming read %.0f GB/s (%.0f%%), L2 %d MB\n",
              prop.name, roof, sustained, 100.0 * sustained / roof, prop.l2CacheSize >> 20);

  const auto pool = make_pool(kPool, 0);
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
  DeviceBytes workspace(ignis_moe_workspace_bytes(IGNIS_MOE_DECODE_MAX_TOKENS, 64)), acc(64 * kH * 8);
  const ignis_moe_workspace ws{workspace.p, IGNIS_MOE_DECODE_MAX_TOKENS, 64};
  MOE_RC(ignis_moe_workspace_init(&ws, acc.as<int64_t>(), nullptr));
  const std::vector<uint16_t> x = make_tokens(8, 31, 2.0f);
  DeviceBytes dx(x.size() * 2);
  upload(dx, x);

  // Device time per call: `calls` launches captured in one CUDA graph and replayed, so WDDM's
  // per-launch submission cost (several us) is not measured -- the engine runs decode as graphs.
  cudaStream_t bench_stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&bench_stream, cudaStreamNonBlocking));
  auto time_us = [&](int calls, const std::function<void(int, cudaStream_t)> &launch) {
    for (int c = 0; c < 4; ++c) launch(c, bench_stream);  // first-call setup outside the capture
    MOE_CUDA(cudaStreamSynchronize(bench_stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    MOE_CUDA(cudaStreamBeginCapture(bench_stream, cudaStreamCaptureModeGlobal));
    for (int c = 0; c < calls; ++c) launch(c, bench_stream);
    MOE_CUDA(cudaStreamEndCapture(bench_stream, &graph));
    MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    MOE_CUDA(cudaGraphLaunch(exec, bench_stream));  // warm-up, also cycles the L2
    cudaEvent_t a, b;
    MOE_CUDA(cudaEventCreate(&a));
    MOE_CUDA(cudaEventCreate(&b));
    const int replays = 3;
    MOE_CUDA(cudaEventRecord(a, bench_stream));
    for (int r = 0; r < replays; ++r) MOE_CUDA(cudaGraphLaunch(exec, bench_stream));
    MOE_CUDA(cudaEventRecord(b, bench_stream));
    MOE_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    MOE_CUDA(cudaEventElapsedTime(&ms, a, b));
    MOE_CUDA(cudaGraphExecDestroy(exec));
    MOE_CUDA(cudaGraphDestroy(graph));
    return 1e3 * ms / (calls * replays);
  };

  // Every decode route (the workspace's decode_route), in one process so they compare; the floor
  // counts as met if one meets it.
  bool floor_ok = false;
  std::vector<uint32_t> routes = {IGNIS_MOE_DECODE_REGISTERS, IGNIS_MOE_DECODE_TICKETS};
  if (ignis_moe_decode_cluster_size() > 0) {
    routes.push_back(IGNIS_MOE_DECODE_CLUSTERS);
  } else {
    std::printf("  (this device runs no decode cluster: the clusters route is not measured)\n");
  }
  for (const uint32_t route : routes) {
    const char *names[] = {"tickets", "clusters", "registers"};
    if (!only_route.empty() && only_route != names[route]) continue;
    ignis_moe_workspace wsr = ws;
    wsr.decode_route = route;
    std::printf("  decode route: %s\n", decode_route_name(route).c_str());
    for (int tokens : {1, 2, 3}) {
      const int calls = 320;
      const std::vector<int32_t> ids = make_calls(calls, tokens, kPool);
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
      const double us = time_us(calls, [&](int c, cudaStream_t st) {
        MOE_RC(ignis_moe_experts_decode(dx.p, tokens, dids.as<int32_t>() + static_cast<std::size_t>(c) * tokens * kTop,
                                        dw.as<float>() + static_cast<std::size_t>(c) * tokens * kTop,
                                        d_slots.as<ignis_moe_slot>(), &wsr, acc.as<int64_t>(), st));
      });
      const double gbs = bytes / (us * 1e-6) / 1e9;
      std::printf("  routed decode, %d token(s): %.1f us per layer, %.2f MB read, %.0f GB/s = %.1f%% of roofline (%.1f%% of streaming read)\n",
                  tokens, us, bytes / 1e6, gbs, 100.0 * gbs / roof, 100.0 * gbs / sustained);
      if (tokens == 1 && gbs >= kDecodeFloor * roof) floor_ok = true;
    }

    // Diagnostics for the 1-token decode: the same experts every call (weights in the L2, so the
    // DRAM is out of the picture), and pools of one K class (all K = 2 against all K = 4: twice the
    // bytes for the same weights -- time that follows the bytes is memory-bound, time that does not
    // is compute- or latency-bound).
    auto decode_us = [&](const Pool &pl, int windows, double *mb) {
      DeviceBytes slots(pl.slots.size() * sizeof(ignis_moe_slot));
      upload(slots, pl.slots);
      const int calls = 256;
      const std::vector<int32_t> ids = make_calls(calls, 1, pl.size, windows);
      std::vector<float> w(ids.size(), 0.1f);
      DeviceBytes dids(ids.size() * 4), dw(w.size() * 4);
      upload(dids, ids);
      upload(dw, w);
      double bytes = 0.0;
      for (int c = 0; c < calls; ++c) {
        for (int r = 0; r < kTop; ++r) bytes += static_cast<double>(pl.data_bytes[ids[static_cast<std::size_t>(c) * kTop + r]]);
      }
      *mb = bytes / calls / 1e6;
      return time_us(calls, [&](int c, cudaStream_t st) {
        MOE_RC(ignis_moe_experts_decode(dx.p, 1, dids.as<int32_t>() + static_cast<std::size_t>(c) * kTop,
                                        dw.as<float>() + static_cast<std::size_t>(c) * kTop, slots.as<ignis_moe_slot>(),
                                        &wsr, acc.as<int64_t>(), st));
      });
    };
    {
      double mb = 0.0;
      const double l2 = decode_us(*pool, 1, &mb);
      std::printf("  diagnostic: 1-token decode with its experts L2-resident: %.1f us (%.2f MB)\n", l2, mb);
      for (uint32_t k2 : {4u, 8u}) {
        const auto uniform = make_pool(128, k2);
        const double us = decode_us(*uniform, 0, &mb);
        std::printf("  diagnostic: 1-token decode, every projection at K = %g: %.1f us for %.2f MB = %.0f GB/s\n", k2 / 2.0, us, mb,
                    mb * 1e6 / (us * 1e-6) / 1e9);
      }
    }

    // The phase trace: a graph of 64 calls as the timing above runs them, eight of them traced
    // (each into its own buffer), replayed once to warm up and once more to read.
    if (traced && route != IGNIS_MOE_DECODE_CLUSTERS) {
      for (int tokens : {1, 3}) {
        for (int windows : {0, 1}) {
          const int calls = 64, first = 40, n_traced = 8;
          const std::vector<int32_t> ids = make_calls(calls, tokens, kPool, windows);
          std::vector<float> w(ids.size(), 0.1f);
          DeviceBytes dids(ids.size() * 4), dw(w.size() * 4), tr(static_cast<std::size_t>(n_traced) * IGNIS_MOE_TRACE_BYTES);
          upload(dids, ids);
          upload(dw, w);
          auto launch = [&](int c, cudaStream_t st) {
            const int32_t *i = dids.as<int32_t>() + static_cast<std::size_t>(c) * tokens * kTop;
            const float *wt = dw.as<float>() + static_cast<std::size_t>(c) * tokens * kTop;
            if (c >= first && c < first + n_traced) {
              auto *buf = reinterpret_cast<unsigned long long *>(static_cast<char *>(tr.p) +
                                                                 static_cast<std::size_t>(c - first) * IGNIS_MOE_TRACE_BYTES);
              MOE_CUDA(cudaMemsetAsync(buf, 0, IGNIS_MOE_TRACE_BYTES, st));
              MOE_RC(ignis_moe_experts_decode_trace(dx.p, tokens, i, wt, d_slots.as<ignis_moe_slot>(), &wsr, acc.as<int64_t>(),
                                                    buf, st));
            } else {
              MOE_RC(ignis_moe_experts_decode(dx.p, tokens, i, wt, d_slots.as<ignis_moe_slot>(), &wsr, acc.as<int64_t>(), st));
            }
          };
          for (int c = 0; c < 4; ++c) launch(c, bench_stream);
          MOE_CUDA(cudaStreamSynchronize(bench_stream));
          cudaGraph_t graph;
          cudaGraphExec_t exec;
          MOE_CUDA(cudaStreamBeginCapture(bench_stream, cudaStreamCaptureModeGlobal));
          for (int c = 0; c < calls; ++c) launch(c, bench_stream);
          MOE_CUDA(cudaStreamEndCapture(bench_stream, &graph));
          MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
          MOE_CUDA(cudaGraphLaunch(exec, bench_stream));
          MOE_CUDA(cudaGraphLaunch(exec, bench_stream));
          MOE_CUDA(cudaStreamSynchronize(bench_stream));
          std::vector<CallTrace> reduced;
          for (int t = 0; t < n_traced; ++t) {
            reduced.push_back(reduce_trace(download<unsigned long long>(
                static_cast<char *>(tr.p) + static_cast<std::size_t>(t) * IGNIS_MOE_TRACE_BYTES, IGNIS_MOE_TRACE_BYTES / 8),
                route == IGNIS_MOE_DECODE_TICKETS && tokens <= 4));
          }
          print_trace(windows == 1 ? "experts in the L2" : "experts from DRAM", tokens, reduced);
          MOE_CUDA(cudaGraphExecDestroy(exec));
          MOE_CUDA(cudaGraphDestroy(graph));
        }
      }
    }
  }

  // The other ops, each over copies that together exceed the L2 so the weights come from DRAM.
  if (!decode_only) {
    // Router at decode: 32 router weights (2.6 MB each).
    const int copies = 32;
    const std::size_t wbytes = static_cast<std::size_t>(kE) * kH * 2;
    DeviceBytes rw(wbytes * copies), ids(8 * kTop * 4), w(8 * kTop * 4), logits(8 * kE * 4);
    MOE_CUDA(cudaMemset(rw.p, 0x3c, wbytes * copies));
    for (int tokens : {1, 3}) {
      const double us = time_us(256, [&](int c, cudaStream_t st) {
        MOE_RC(ignis_moe_router(dx.p, tokens, static_cast<char *>(rw.p) + (c % copies) * wbytes, ids.as<int32_t>(),
                                w.as<float>(), logits.as<float>(), st));
      });
      const double gbs = wbytes / (us * 1e-6) / 1e9;
      std::printf("  router, %d token(s): %.1f us, %.2f MB read, %.0f GB/s = %.1f%% of roofline\n", tokens, us, wbytes / 1e6, gbs,
                  100.0 * gbs / roof);
    }
  }
  if (!decode_only) {
    // Shared expert at decode: 32 copies of its three FP8 matrices (4.9 MB a set).
    const int copies = 32;
    const std::size_t gu_bytes = static_cast<std::size_t>(kI) * kH + 256 + kI * 2;  // codes, pad, scales
    const std::size_t dn_bytes = static_cast<std::size_t>(kH) * kI + 256 + kH * 2;
    const std::size_t set = (2 * gu_bytes + dn_bytes + 255) / 256 * 256;
    DeviceBytes weights(set * copies), h(8 * kI * 2), shared(8 * kH * 4);
    MOE_CUDA(cudaMemset(weights.p, 0x22, set * copies));
    const double bytes = 2.0 * kI * kH + 1.0 * kH * kI;
    for (int tokens : {1, 3}) {
      const double us = time_us(256, [&](int c, cudaStream_t st) {
        const char *base = static_cast<const char *>(weights.p) + (c % copies) * set;
        MOE_RC(ignis_moe_shared_expert(base, base + (gu_bytes + 15) / 16 * 16, base + 2 * ((gu_bytes + 15) / 16 * 16), dx.p, tokens,
                                       h.p, shared.as<float>(), st));
      });
      const double gbs = bytes / (us * 1e-6) / 1e9;
      std::printf("  shared expert (FP8), %d token(s): %.1f us, %.2f MB read, %.0f GB/s = %.1f%% of roofline\n", tokens, us,
                  bytes / 1e6, gbs, 100.0 * gbs / roof);
    }
    // Combine at decode.
    DeviceBytes wg(kH * 2), out(8 * kH * 2);
    MOE_CUDA(cudaMemset(wg.p, 0, kH * 2));
    const double us = time_us(256, [&](int, cudaStream_t st) {
      MOE_RC(ignis_moe_combine(acc.as<int64_t>(), shared.as<float>(), dx.p, wg.p, 1, out.p, st));
    });
    std::printf("  combine, 1 token: %.1f us\n", us);
  }
  if (!decode_only) {
    // Prefill, 2048 tokens: routed experts over the pool (uniform routing, ~64 rows per expert),
    // and the shared expert's FP8 linears; TFLOP/s against the BF16/FP16 dense tensor peak.
    const int tokens = 2048;
    const double peak_tflops = 209.5;  // RTX 5090, dense FP16/BF16 tensor (NVIDIA's figure)
    DeviceBytes pws(ignis_moe_workspace_bytes(1, tokens)), pacc(static_cast<std::size_t>(tokens) * kH * 8);
    const ignis_moe_workspace pdesc{pws.p, 1, static_cast<uint32_t>(tokens)};
    MOE_RC(ignis_moe_workspace_init(&pdesc, pacc.as<int64_t>(), nullptr));
    const std::vector<uint16_t> px = make_tokens(tokens, 32, 2.0f);
    std::vector<int32_t> ids(static_cast<std::size_t>(tokens) * kTop);
    for (int t = 0; t < tokens; ++t) {
      for (int r = 0; r < kTop; ++r) ids[static_cast<std::size_t>(t) * kTop + r] = (t * 37 + r * 32) % kPool;
    }
    std::vector<float> w(ids.size(), 0.1f);
    DeviceBytes dpx(px.size() * 2), dids(ids.size() * 4), dw(w.size() * 4);
    upload(dpx, px);
    upload(dids, ids);
    upload(dw, w);
    const double us = time_us(8, [&](int, cudaStream_t st) {
      MOE_RC(ignis_moe_experts_prefill(dpx.p, tokens, dids.as<int32_t>(), dw.as<float>(), d_slots.as<ignis_moe_slot>(), &pdesc,
                                       pacc.as<int64_t>(), st));
    });
    const double flops = 2.0 * tokens * kTop * (static_cast<double>(kH) * 2 * kI + static_cast<double>(kI) * kH);
    std::printf("  routed prefill, %d tokens: %.0f us per layer, %.1f TFLOP/s = %.1f%% of the %.1f dense tensor peak\n", tokens, us,
                flops / (us * 1e-6) / 1e12, 100.0 * flops / (us * 1e-6) / 1e12 / peak_tflops, peak_tflops);
    const std::size_t gu_bytes = static_cast<std::size_t>(kI) * kH + 256 + kI * 2;
    DeviceBytes gw(gu_bytes), uw(gu_bytes), dn(static_cast<std::size_t>(kH) * kI + 256 + kH * 2), h(static_cast<std::size_t>(tokens) * kI * 2),
        shared(static_cast<std::size_t>(tokens) * kH * 4);
    MOE_CUDA(cudaMemset(gw.p, 0x22, gu_bytes));
    MOE_CUDA(cudaMemset(uw.p, 0x22, gu_bytes));
    MOE_CUDA(cudaMemset(dn.p, 0x22, dn.bytes));
    const double sus = time_us(8, [&](int, cudaStream_t st) {
      MOE_RC(ignis_moe_shared_expert(gw.p, uw.p, dn.p, dpx.p, tokens, h.p, shared.as<float>(), st));
    });
    const double sflops = 2.0 * tokens * 3.0 * kH * kI;
    std::printf("  shared expert prefill, %d tokens: %.0f us, %.1f TFLOP/s = %.1f%% of peak\n", tokens, sus,
                sflops / (sus * 1e-6) / 1e12, 100.0 * sflops / (sus * 1e-6) / 1e12 / peak_tflops);
  }

  std::printf("bench_moe: decode floor (>= %.0f%% of roofline at 1 token, either route) %s\n", 100.0 * kDecodeFloor,
              floor_ok ? "met" : "NOT met");
  return check_floor && !floor_ok ? 1 : 0;
}
