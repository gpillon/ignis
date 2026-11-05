// The dense prefill against the fused prefill, per body shape and per chunk -- OURS (GitHub #509,
// spec #503; kernel/include/ignis_dense_prefill.h). A tool, NOT a CTest: timing wants the card to
// itself (make gpu-status first).
//
// For each chunk width T (--tokens, default 2048 and 4096) and each body shape at each K the
// public quant carries (bench_trellis_linear.cu's mix: 20 (matrix, K) entries, their layer
// counts), it times on one stream, best of --reps rounds of --inner calls, with the trellis words
// cycling through enough copies that each read misses the L2:
//   decode   ignis_dense_prefill_decode of one matrix;
//   GEMM     ignis_dense_prefill_gemm at the plan the op picks (BM, stages, raster printed), and
//            with --sweep at every tile height, depth and raster, the fastest named;
//   dense    the engine's sequence for one matrix, decode then GEMM;
//   fused    ignis_trellis_linear (ignis_trellis_linear_swiglu for gate and up), rotation included;
//   A8       ignis_dense_prefill_quantize of each shared input: qkv and z, q k and v, gate and up,
//            the output projections (in 6144) and down (in 17408);
//   fused A8 (GitHub #511) A8 fused into the RMSnorm that produces its input: each norm group's
//            fused RMSnorm less the vendored rmsnorm it replaces.
// Then the 64-layer body per chunk: dense = sum of each entry's dense x its calls + each group's
// A8 x its layers, against fused = sum of each entry's fused x its calls; and the chunk's A8
// before the fusion (standalone, every group) and after it (the norms' fused, down's and the
// output projections' still standalone).
//
//   ignis_kernel_dense_prefill_bench [--tokens 2048,4096] [--reps 5] [--inner 32] [--sweep] [--a8]
//
// --a8 times the A8 alone (the standalone groups and the fused RMSnorms), not the body.

#include "dense_prefill_internal.cuh"
#include "trellis_test_common.h"

#include "ninfer/ops/rmsnorm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace trellis_test;

namespace {

constexpr uint64_t kPoolBytes = 512ull << 20;  // more than five times a 96 MB L2

struct Shape {
  const char *name;
  uint32_t in, out, k2;
  int swiglu;  // gate and up, two matrices of this shape
};

struct MixEntry {
  Shape shape;
  int calls;
};

// The public quant's K mix of the 27B's body (bench_trellis_linear.cu's kMix).
const MixEntry kMix[] = {
    {{"gdn qkv", 5120, 10240, 4, 0}, 4},  {{"gdn qkv", 5120, 10240, 6, 0}, 42}, {{"gdn qkv", 5120, 10240, 8, 0}, 2},
    {{"gdn z", 5120, 6144, 4, 0}, 3},     {{"gdn z", 5120, 6144, 6, 0}, 40},    {{"gdn z", 5120, 6144, 8, 0}, 5},
    {{"gdn out", 6144, 5120, 6, 0}, 40},  {{"gdn out", 6144, 5120, 8, 0}, 8},
    {{"gqa q", 5120, 12288, 4, 0}, 6},    {{"gqa q", 5120, 12288, 6, 0}, 10},
    {{"gqa k, v", 5120, 1024, 8, 0}, 28}, {{"gqa k, v", 5120, 1024, 10, 0}, 4},
    {{"gqa o", 6144, 5120, 6, 0}, 14},    {{"gqa o", 6144, 5120, 8, 0}, 2},
    {{"mlp gate+up", 5120, 17408, 4, 1}, 9}, {{"mlp gate+up", 5120, 17408, 6, 1}, 49}, {{"mlp gate+up", 5120, 17408, 8, 1}, 6},
    {{"mlp down", 17408, 5120, 4, 0}, 3}, {{"mlp down", 17408, 5120, 6, 0}, 53}, {{"mlp down", 17408, 5120, 8, 0}, 8},
};

// The shared inputs' A8 launches of the body: (in, matrices, layers).
struct A8Group {
  const char *name;
  uint32_t in, matrices;
  int layers;
};
const A8Group kA8[] = {
    {"gdn qkv, z", 5120, 2, 48}, {"gqa q, k, v", 5120, 3, 16}, {"mlp gate, up", 5120, 2, 64},
    {"output (gdn, gqa)", 6144, 1, 64}, {"mlp down", 17408, 1, 64},
};

int g_reps = 5;
int g_inner = 32;  // every copy of a pool once per round, so small matrices miss the L2 too

template <typename F> double time_us(F &&launch, cudaStream_t s) {
  for (int i = 0; i < 2; ++i) launch(i);
  MOE_CUDA(cudaStreamSynchronize(s));
  cudaEvent_t a, b;
  MOE_CUDA(cudaEventCreate(&a));
  MOE_CUDA(cudaEventCreate(&b));
  double best = 1e30;
  for (int r = 0; r < g_reps; ++r) {
    MOE_CUDA(cudaEventRecord(a, s));
    for (int i = 0; i < g_inner; ++i) launch(i);
    MOE_CUDA(cudaEventRecord(b, s));
    MOE_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    MOE_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, 1000.0 * ms / g_inner);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return best;
}

// Copies of one shape's trellis matrix, cycled so that each read misses the L2.
struct Pool {
  std::vector<DeviceBytes> mem;
  std::vector<ignis_trellis_weight> w;
};

void make_pool(Pool &pool, const Shape &s) {
  const uint64_t words = static_cast<uint64_t>(s.in / 16) * (s.out / 16) * tile_words16(s.k2);
  const uint64_t one = words * 2 + 2ull * s.in + 2ull * s.out;
  const int copies = static_cast<int>(std::max<uint64_t>(2, std::min<uint64_t>(32, kPoolBytes / one + 1)));
  std::vector<uint16_t> scales(std::max(s.in, s.out));
  for (size_t i = 0; i < scales.size(); ++i) scales[i] = moe_test::f32_to_f16(i % 2 != 0 ? -0.05f : 0.05f);
  for (int c = 0; c < copies; ++c) {
    pool.mem.emplace_back(words * 2 + 256);
    device_hash_words(pool.mem.back().p, words, 900 + c);
    pool.mem.emplace_back(2ull * s.in);
    const void *suh = pool.mem.back().p;
    MOE_CUDA(cudaMemcpy(pool.mem.back().p, scales.data(), 2ull * s.in, cudaMemcpyHostToDevice));
    pool.mem.emplace_back(2ull * s.out);
    const void *svh = pool.mem.back().p;
    MOE_CUDA(cudaMemcpy(pool.mem.back().p, scales.data(), 2ull * s.out, cudaMemcpyHostToDevice));
    pool.w.push_back(ignis_trellis_weight{pool.mem[pool.mem.size() - 3].p, suh, svh, s.k2, s.in, s.out, 0});
  }
}

std::string plan_name(const ignis_dense::Plan &p) {
  return "BM " + std::to_string(p.bm) + ", " + std::to_string(p.stages) + " stages, " +
         (p.raster == ignis_dense::kTokensFastest ? "tokens" : "outputs") + " fastest";
}

struct Times {
  double decode = 0, gemm = 0, dense = 0, fused = 0;
};

Times time_entry(const Shape &s, uint32_t tokens, bool sweep, const ignis_dense::State &st, cudaStream_t stream) {
  Pool pool;
  make_pool(pool, s);
  const int copies = static_cast<int>(pool.w.size());
  DeviceBytes x(2ull * tokens * s.in), y(2ull * tokens * s.out), dense(static_cast<uint64_t>(s.in) * s.out);
  std::vector<uint16_t> xh(static_cast<size_t>(tokens) * s.in);
  for (size_t i = 0; i < xh.size(); ++i) xh[i] = moe_test::f32_to_bf16(moe_test::hash_uniform(5, i, 1.5f));
  upload(x, xh);
  DeviceBytes q(static_cast<size_t>(tokens) * s.in), scale(4ull * tokens), sum(4ull * tokens);
  const ignis_dense_prefill_activations a{q.p, scale.as<float>(), sum.as<int32_t>()};
  const ignis_trellis_weight *w0 = &pool.w[0];
  TL_RC(ignis_dense_prefill_quantize(x.p, tokens, &w0, 1, &a, stream));
  const ignis_dense_prefill_output o{y.p, s.out, IGNIS_TRELLIS_STORE, 0, 0, nullptr};
  Times t;
  t.decode = time_us([&](int i) { TL_RC(ignis_dense_prefill_decode(&pool.w[i % copies], dense.p, dense.bytes, stream)); }, stream);
  TL_RC(ignis_dense_prefill_decode(&pool.w[0], dense.p, dense.bytes, stream));
  const ignis_dense::Plan own = ignis_dense::plan(st, tokens, s.in, s.out);
  t.gemm = time_us([&](int) { TL_RC(ignis_dense_prefill_gemm(&pool.w[0], dense.p, &a, tokens, &o, stream)); }, stream);
  std::string best_name = plan_name(own);
  double best = t.gemm;
  if (sweep) {
    for (const int bm : ignis_dense::kTileHeights)
      for (const int stages : ignis_dense::kDepths)
        for (const int raster : {ignis_dense::kTokensFastest, ignis_dense::kOutputsFastest}) {
          ignis_dense::Plan p;
          p.bm = bm;
          p.stages = stages;
          p.raster = raster;
          const double us = time_us([&](int) {
            TL_RC(ignis_dense::gemm_at("bench", p, &pool.w[0], dense.p, &a, tokens, &o, nullptr, 0, stream));
          }, stream);
          std::printf("      sweep %-36s %9.1f us%s\n", plan_name(p).c_str(), us,
                      p.bm == own.bm && p.stages == own.stages && p.raster == own.raster ? "  (the plan)" : "");
          if (us < best) {
            best = us;
            best_name = plan_name(p);
          }
        }
  }
  t.dense = time_us([&](int i) {
    TL_RC(ignis_dense_prefill_decode(&pool.w[i % copies], dense.p, dense.bytes, stream));
    TL_RC(ignis_dense_prefill_gemm(&pool.w[i % copies], dense.p, &a, tokens, &o, stream));
  }, stream);
  // The fused prefill, at the same width.
  const uint32_t weights = s.swiglu ? 2 : 1;
  const uint64_t ws_bytes = ignis_trellis_workspace_bytes(s.in, s.out, weights, tokens);
  DeviceBytes ws_mem(ws_bytes);
  const ignis_trellis_workspace ws{ws_mem.p, ws_bytes};
  TL_RC(ignis_trellis_workspace_init(&ws, stream));
  if (s.swiglu) {
    t.fused = time_us([&](int i) {
      TL_RC(ignis_trellis_linear_swiglu(&pool.w[(2 * i) % copies], &pool.w[(2 * i + 1) % copies], x.p, tokens, y.p, s.out, &ws,
                                        stream));
    }, stream);
  } else {
    t.fused = time_us([&](int i) {
      TL_RC(ignis_trellis_linear(&pool.w[i % copies], x.p, tokens, y.p, s.out, IGNIS_TRELLIS_STORE, &ws, stream));
    }, stream);
  }
  const int mats = s.swiglu ? 2 : 1;
  const double flop = 2.0 * tokens * s.in * s.out * mats;
  std::printf("  %-12s %5u %5u K %-3.1f | decode %7.1f  GEMM %8.1f (%5.0f TOPS, %s)%s%s | dense %8.1f | fused %8.1f | f/d %.2f\n",
              s.name, s.in, s.out, s.k2 / 2.0, t.decode * mats, t.gemm * mats, flop / (t.gemm * mats * 1e-6) / 1e12,
              plan_name(own).c_str(), sweep ? ", fastest " : "", sweep ? best_name.c_str() : "", t.dense * mats, t.fused,
              t.fused / (t.dense * mats));
  t.decode *= mats;
  t.gemm *= mats;
  t.dense *= mats;
  return t;
}

double time_a8(const A8Group &g, uint32_t tokens, cudaStream_t stream) {
  std::vector<DeviceBytes> mem;
  std::vector<ignis_trellis_weight> w(g.matrices);
  std::vector<ignis_dense_prefill_activations> a(g.matrices);
  std::vector<uint16_t> suh(g.in);
  for (uint32_t k = 0; k < g.in; ++k) suh[k] = moe_test::f32_to_f16(k % 3 == 0 ? -0.04f : 0.05f);
  for (uint32_t i = 0; i < g.matrices; ++i) {
    mem.emplace_back(2ull * g.in);
    MOE_CUDA(cudaMemcpy(mem.back().p, suh.data(), 2ull * g.in, cudaMemcpyHostToDevice));
    w[i] = ignis_trellis_weight{mem.back().p, mem.back().p, mem.back().p, 6, g.in, 128, 0};  // only suh and in are read
    mem.emplace_back(static_cast<size_t>(tokens) * g.in);
    void *q = mem.back().p;
    mem.emplace_back(8ull * tokens);
    a[i] = ignis_dense_prefill_activations{q, static_cast<float *>(mem.back().p),
                                           reinterpret_cast<int32_t *>(static_cast<char *>(mem.back().p) + 4ull * tokens)};
  }
  DeviceBytes x(2ull * tokens * g.in);
  std::vector<uint16_t> xh(static_cast<size_t>(tokens) * g.in);
  for (size_t i = 0; i < xh.size(); ++i) xh[i] = moe_test::f32_to_bf16(moe_test::hash_uniform(6, i, 1.5f));
  upload(x, xh);
  std::vector<const ignis_trellis_weight *> wl(g.matrices);
  for (uint32_t i = 0; i < g.matrices; ++i) wl[i] = &w[i];
  return time_us([&](int) { TL_RC(ignis_dense_prefill_quantize(x.p, tokens, wl.data(), g.matrices, a.data(), stream)); }, stream);
}

// GitHub #511: A8 fused into the RMSnorm that produces its input, per chunk: a norm group's A8
// after the fusion is the fused RMSnorm less the vendored rmsnorm it replaces (h written for the
// GDN group alone, whose gating projection reads it), in us per chunk over the 64 layers. Down's
// A8 (in 17408) and the output projections' (in 6144) stay standalone.
double time_fused_norms(uint32_t tokens, cudaStream_t stream) {
  constexpr uint32_t kHidden = 5120;
  double norms = 0;
  std::vector<uint16_t> suh(kHidden), xh(static_cast<size_t>(tokens) * kHidden), norm(kHidden);
  for (uint32_t k = 0; k < kHidden; ++k) suh[k] = moe_test::f32_to_f16(k % 3 == 0 ? -0.04f : 0.05f);
  for (size_t i = 0; i < xh.size(); ++i) xh[i] = moe_test::f32_to_bf16(moe_test::hash_uniform(7, i, 1.5f));
  for (uint32_t k = 0; k < kHidden; ++k) norm[k] = moe_test::f32_to_bf16(moe_test::hash_uniform(8, k, 0.5f));
  DeviceBytes d_suh(2ull * kHidden), x(2ull * xh.size()), h(2ull * xh.size()), d_norm(2ull * kHidden);
  upload(d_suh, suh);
  upload(x, xh);
  upload(d_norm, norm);
  struct Norm {
    const char *name;
    uint32_t matrices;
    bool h;
    int layers;
  };
  const Norm groups[] = {{"gdn qkv, z", 2, true, 48}, {"gqa q, k, v", 3, false, 16}, {"mlp gate, up", 2, false, 64}};
  std::vector<DeviceBytes> mem;
  // Only suh and the shape of a matrix are read by A8.
  const ignis_trellis_weight w{d_suh.p, d_suh.p, d_suh.p, 6, kHidden, 128, 0};
  std::vector<ignis_dense_prefill_activations> a(3);
  for (int i = 0; i < 3; ++i) {
    mem.emplace_back(static_cast<size_t>(tokens) * kHidden);
    void *q = mem.back().p;
    mem.emplace_back(8ull * tokens);
    a[i] = ignis_dense_prefill_activations{q, static_cast<float *>(mem.back().p),
                                           reinterpret_cast<int32_t *>(static_cast<char *>(mem.back().p) + 4ull * tokens)};
  }
  const ignis_trellis_weight *wl[3] = {&w, &w, &w};
  const ninfer::Tensor xt(x.p, ninfer::DType::BF16, {static_cast<int32_t>(kHidden), static_cast<int32_t>(tokens), 1, 1});
  const ninfer::Tensor nt(d_norm.p, ninfer::DType::BF16, {static_cast<int32_t>(kHidden), 1, 1, 1});
  ninfer::Tensor ht(h.p, ninfer::DType::BF16, {static_cast<int32_t>(kHidden), static_cast<int32_t>(tokens), 1, 1});
  const double rms = time_us([&](int) { ninfer::ops::rmsnorm(xt, nt, 1e-6f, true, ht, stream); }, stream);
  for (const Norm &g : groups) {
    const double fused = time_us([&](int) {
      TL_RC(ignis_dense_prefill_rmsnorm_quantize(x.p, d_norm.p, 1e-6f, tokens, g.h ? h.p : nullptr, wl, g.matrices, a.data(),
                                                 stream));
    }, stream);
    std::printf("  fused A8 %-15s RMSnorm with A8 %7.1f us - RMSnorm %7.1f = A8 %7.1f us x %d layers\n", g.name, fused, rms,
                fused - rms, g.layers);
    norms += (fused - rms) * g.layers;
  }
  return norms;
}

std::vector<uint32_t> parse_list(const char *s) {
  std::vector<uint32_t> v;
  for (const char *p = s; *p != '\0';) {
    char *end = nullptr;
    v.push_back(static_cast<uint32_t>(std::strtoul(p, &end, 10)));
    p = *end == ',' ? end + 1 : end;
  }
  return v;
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<uint32_t> widths = {2048, 4096};
  bool sweep = false, a8_only = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) widths = parse_list(argv[++i]);
    else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) g_reps = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--inner") == 0 && i + 1 < argc) g_inner = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--sweep") == 0) sweep = true;
    else if (std::strcmp(argv[i], "--a8") == 0) a8_only = true;
    else {
      std::fprintf(stderr, "usage: %s [--tokens 2048,4096] [--reps 5] [--inner 32] [--sweep] [--a8]\n", argv[0]);
      return 2;
    }
  }
  int major = 0, minor = 0, sms = 0;
  MOE_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
  MOE_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
  MOE_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  TL_RC(ignis_trellis_prepare());
  TL_RC(ignis_dense_prefill_prepare());
  ignis_dense::State st;
  TL_RC(ignis_dense::state("bench", &st));
  std::printf("dense prefill bench: compute capability %d.%d, %d SMs; GEMM resident CTAs per SM: BM 256 %d/%d, BM 128 %d/%d "
              "(3/4 stages); best of %d rounds of %d calls\n",
              major, minor, sms, st.resident[0][0], st.resident[0][1], st.resident[1][0], st.resident[1][1], g_reps, g_inner);
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const uint32_t tokens : widths) {
    std::printf("== chunk of %u tokens (us per call; gate+up is the pair)\n", tokens);
    double decode = 0, gemm = 0, dense = 0, fused = 0, a8 = 0, a8_output = 0, a8_down = 0;
    if (!a8_only) {
      for (const MixEntry &e : kMix) {
        const Times t = time_entry(e.shape, tokens, sweep, st, stream);
        decode += t.decode * e.calls;
        gemm += t.gemm * e.calls;
        dense += t.dense * e.calls;
        fused += t.fused * e.calls;
      }
    }
    for (const A8Group &g : kA8) {
      const double us = time_a8(g, tokens, stream);
      std::printf("  A8 %-18s in %5u, %u matrix(es): %7.1f us x %d layers\n", g.name, g.in, g.matrices, us, g.layers);
      a8 += us * g.layers;
      if (g.in == 6144) a8_output += us * g.layers;
      if (g.in == 17408) a8_down += us * g.layers;
    }
    const double norms = time_fused_norms(tokens, stream);
    const double a8_fused = norms + a8_down + a8_output;
    if (!a8_only) {
      std::printf("chunk of %u tokens, the 64-layer body at the quant's K mix (ms): decode %.2f + GEMM %.2f (alone) | "
                  "dense (decode then GEMM) %.2f + A8 %.2f = %.2f | fused %.2f | fused / dense %.3f\n",
                  tokens, decode / 1e3, gemm / 1e3, dense / 1e3, a8 / 1e3, (dense + a8) / 1e3, fused / 1e3, fused / (dense + a8));
    }
    std::printf("chunk of %u tokens, the 64-layer body's A8 (ms): before the fusion %.3f (standalone, every group) | after "
                "%.3f = the norms' %.3f (fused) + down's %.3f + the output projections' %.3f (standalone)\n",
                tokens, a8 / 1e3, a8_fused / 1e3, norms / 1e3, a8_down / 1e3, a8_output / 1e3);
  }
  MOE_CUDA(cudaStreamDestroy(stream));
  return 0;
}
