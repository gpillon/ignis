// Flash-Next's routed experts at real geometry (fused gate/up 2560 x 1280, down 640 x 2560, ten
// of 512 experts per token, K per expert) -- OURS (spec flash-next/02 Acceptance 3 and 5,
// GitHub #300).
//
//   decode      1, 2, 3 and 8 tokens through ignis_moe_experts_decode, one launch for every
//               expert and K, against the fp64 product of the exactly decoded weights. The
//               routing covers all four K classes for both projections in one launch and
//               shares experts between consecutive tokens.
//   prefill    1 to 4096 tokens through ignis_moe_experts_prefill (device grouping, one launch
//               per projection family), routed with skew: expert 7 in every token (a group of the
//               whole chunk), expert 300 in token 0 only (a group of one), half the other picks
//               from 32 warm experts, expert 0 never. Checked against fp64 on a sample of tokens
//               that always includes the first and the last.
//   determinism the same call twice, the records placed in a different order in device memory
//               (other slot addresses), and the call captured in a CUDA graph and replayed
//               twice: every accumulator bit equal, for both routes.
//
// Tolerance. The kernels round each rotated activation to fp16 for the tensor cores (scaled by a
// power of two so it is a normal number: relative error <= 2^-11) twice, at the gate/up input
// and at the down input; everything else accumulates in fp32 or exactly. For a sum of products
// with independent rounding errors that is ~2^-11 / sqrt(3) ~ 3e-4 relative per output on each
// stage. Asserted per token: relative L2 error <= 2e-3 and max error <= 1e-2 of the token's
// largest output; the measured figures are printed.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_moe.h"
#include "moe_experts_common.h"
#include "moe_fixture.h"

#include <cstdint>
#include <cstdio>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

constexpr double kRelL2 = 2e-3;
constexpr double kRelMax = 1e-2;
constexpr uint32_t kMaxTokens = 4096;

struct Buffers {
  DeviceBytes workspace{ignis_moe_workspace_bytes(IGNIS_MOE_DECODE_MAX_TOKENS, kMaxTokens)};
  DeviceBytes acc{static_cast<std::size_t>(kMaxTokens) * kH * 8};
  ignis_moe_workspace ws{workspace.p, IGNIS_MOE_DECODE_MAX_TOKENS, kMaxTokens};
  Buffers() {
    MOE_RC(ignis_moe_workspace_init(&ws, acc.as<int64_t>(), nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  }
  void clear_acc(int tokens) {
    MOE_CUDA(cudaMemset(acc.p, 0, static_cast<std::size_t>(tokens) * kH * 8));
  }
};

// Experts that between them use every K class of both projections.
std::vector<int32_t> covering_experts() {
  std::set<uint32_t> gu, dn;
  std::vector<int32_t> picks;
  for (int e = 0; e < kE && (gu.size() < 4 || dn.size() < 4); ++e) {
    const uint32_t g = kRecordK2[ExpertSet::gu_record(e)];
    const uint32_t d = kRecordK2[(ExpertSet::dn_record(e) + 5) % kRecords];
    if (!gu.count(g) || !dn.count(d)) {
      picks.push_back(e);
      gu.insert(g);
      dn.insert(d);
    }
  }
  return picks;
}

// Prefill routing with skew: expert 7 in every token, 300 in token 0 only, 0 never, half the
// rest from 32 warm experts.
void make_skewed_routing(int tokens, uint32_t stream, std::vector<int32_t> &ids, std::vector<float> &weights) {
  std::vector<float> unused;
  make_routing(tokens, stream, 0, ids, weights);  // for the weights
  uint64_t draw = 0;
  for (int t = 0; t < tokens; ++t) {
    int32_t *row = &ids[static_cast<std::size_t>(t) * kTop];
    int n = 0;
    row[n++] = 7;
    if (t == 0) row[n++] = 300;
    while (n < kTop) {
      const uint32_t h = hash_u32(stream + 7, draw++);
      const int32_t e = (h & 1u) ? static_cast<int32_t>(16 + (h >> 1) % 32) : static_cast<int32_t>((h >> 1) % kE);
      if (e == 0 || e == 7 || e == 300 || std::find(row, row + n, e) != row + n) continue;
      row[n++] = e;
    }
  }
}

struct Call {
  std::vector<uint16_t> x;
  std::vector<int32_t> ids;
  std::vector<float> weights;
  DeviceBytes dx, dids, dw;
  Call(int tokens, uint32_t stream, bool skewed = false)
      : x(make_tokens(tokens, stream, 2.0f)), dx(x.size() * 2), dids(static_cast<std::size_t>(tokens) * kTop * 4),
        dw(static_cast<std::size_t>(tokens) * kTop * 4) {
    if (skewed) {
      make_skewed_routing(tokens, stream + 1, ids, weights);
    } else {
      make_routing(tokens, stream + 1, 3, ids, weights);
    }
    const std::vector<int32_t> cover = covering_experts();
    for (std::size_t i = 0; !skewed && i < cover.size() && i < static_cast<std::size_t>(kTop); ++i) {
      // Token 0 takes the covering experts first (keeping its ids distinct).
      for (int r = 0; r < kTop; ++r) {
        if (ids[r] == cover[i]) ids[r] = ids[i];
      }
      ids[i] = cover[i];
    }
    upload(dx, x);
    upload(dids, ids);
    upload(dw, weights);
  }
};

std::vector<int64_t> run_decode(Buffers &b, const ExpertSet &set, const Call &c, int tokens) {
  b.clear_acc(tokens);
  MOE_RC(ignis_moe_experts_decode(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(),
                                  set.d_slots.as<ignis_moe_slot>(), &b.ws, b.acc.as<int64_t>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  return download<int64_t>(b.acc.p, static_cast<std::size_t>(tokens) * kH);
}

void check_against_f64(const char *route, const ExpertSet &set, const Call &c, const std::vector<int64_t> &acc,
                       const std::vector<int> &token_list) {
  double worst_l2 = 0.0, worst_max = 0.0;
  for (int t : token_list) {
    const std::vector<double> ref = routed_f64(set, &c.ids[static_cast<std::size_t>(t) * kTop],
                                               &c.weights[static_cast<std::size_t>(t) * kTop], token_f64(c.x, t));
    double l2 = 0.0, mx = 0.0;
    rel_errors(fixed_to_f64(acc, t), ref, &l2, &mx);
    worst_l2 = std::max(worst_l2, l2);
    worst_max = std::max(worst_max, mx);
    check(l2 <= kRelL2, std::string(route) + ": token " + std::to_string(t) + " relative L2 error " + std::to_string(l2));
    check(mx <= kRelMax, std::string(route) + ": token " + std::to_string(t) + " relative max error " + std::to_string(mx));
  }
  std::printf("  %-28s %zu tokens checked: worst relative L2 %.2e, worst relative max %.2e\n", route,
              token_list.size(), worst_l2, worst_max);
}

void decode_arm(Buffers &b, const ExpertSet &set) {
  for (int tokens : {1, 2, 3, 8}) {
    const Call c(tokens, 600 + 10 * tokens);
    const std::vector<int64_t> acc = run_decode(b, set, c, tokens);
    std::vector<int> all(tokens);
    std::iota(all.begin(), all.end(), 0);
    const std::string name = "decode, " + std::to_string(tokens) + " token(s)";
    check_against_f64(name.c_str(), set, c, acc, all);
  }
}

// The accumulator is the caller's state and the ops add into it: two calls with no combine (and
// no zeroing) between leave exactly twice one call, bit for bit, on both routes.
void accumulator_arm(Buffers &b, const ExpertSet &set) {
  for (bool prefill : {false, true}) {
    const int tokens = prefill ? 40 : 3;
    const Call c(tokens, prefill ? 1301 : 1300, prefill);
    auto call = [&]() {
      MOE_RC(prefill ? ignis_moe_experts_prefill(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(),
                                                 set.d_slots.as<ignis_moe_slot>(), &b.ws, b.acc.as<int64_t>(), nullptr)
                     : ignis_moe_experts_decode(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(),
                                                set.d_slots.as<ignis_moe_slot>(), &b.ws, b.acc.as<int64_t>(), nullptr));
      MOE_CUDA(cudaDeviceSynchronize());
      return download<int64_t>(b.acc.p, static_cast<std::size_t>(tokens) * kH);
    };
    b.clear_acc(tokens);
    const std::vector<int64_t> once = call();
    std::vector<int64_t> twice = once;
    for (int64_t &v : twice) v *= 2;
    check(call() == twice,
          std::string(prefill ? "prefill" : "decode") + ": a second call without combine adds exactly the first again");
  }
  std::printf("  accumulator: a second call without combine adds exactly the first (decode and prefill)\n");
}

std::vector<int64_t> run_prefill(Buffers &b, const ExpertSet &set, const Call &c, int tokens) {
  b.clear_acc(tokens);
  MOE_RC(ignis_moe_experts_prefill(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(),
                                   &b.ws, b.acc.as<int64_t>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  return download<int64_t>(b.acc.p, static_cast<std::size_t>(tokens) * kH);
}

void prefill_arm(Buffers &b, const ExpertSet &set) {
  for (int tokens : {1, 3, 64, 257, 2048, 4096}) {
    const Call c(tokens, 1200 + tokens, true);
    const std::vector<int64_t> acc = run_prefill(b, set, c, tokens);
    std::vector<int> sample;
    const int step = tokens > 24 ? tokens / 23 : 1;
    for (int t = 0; t < tokens; t += step) sample.push_back(t);
    if (sample.back() != tokens - 1) sample.push_back(tokens - 1);
    const std::string name = "prefill, " + std::to_string(tokens) + " token(s)";
    check_against_f64(name.c_str(), set, c, acc, sample);
    if (tokens == 2048) {
      check(run_prefill(b, set, c, tokens) == acc, "prefill: a second run agrees bit for bit");
    }
  }
}

void determinism_arm(Buffers &b, ExpertSet &set) {
  struct Route {
    const char *name;
    int tokens;
    bool prefill;
  };
  for (const Route &route : {Route{"decode", 3, false}, Route{"prefill", 257, true}}) {
    const Call c(route.tokens, route.prefill ? 901 : 900, route.prefill);
    const int tokens = route.tokens;
    auto enqueue = [&](cudaStream_t stream) {
      MOE_CUDA(cudaMemsetAsync(b.acc.p, 0, static_cast<std::size_t>(tokens) * kH * 8, stream));
      if (route.prefill) {
        MOE_RC(ignis_moe_experts_prefill(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(),
                                         &b.ws, b.acc.as<int64_t>(), stream));
      } else {
        MOE_RC(ignis_moe_experts_decode(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(),
                                        &b.ws, b.acc.as<int64_t>(), stream));
      }
    };
    auto result = [&]() { return download<int64_t>(b.acc.p, static_cast<std::size_t>(tokens) * kH); };
    enqueue(nullptr);
    MOE_CUDA(cudaDeviceSynchronize());
    const std::vector<int64_t> first = result();
    enqueue(nullptr);
    MOE_CUDA(cudaDeviceSynchronize());
    check(result() == first, std::string(route.name) + ": a second run agrees bit for bit");

    // Other slot addresses: the records in reverse order.
    std::vector<int> order(2 * kRecords);
    std::iota(order.rbegin(), order.rend(), 0);
    set.place(order);
    enqueue(nullptr);
    MOE_CUDA(cudaDeviceSynchronize());
    check(result() == first, std::string(route.name) + ": records in other slots agree bit for bit");

    // Graph capture and two replays.
    cudaStream_t stream;
    MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    enqueue(stream);
    MOE_CUDA(cudaStreamEndCapture(stream, &graph));
    MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    for (int replay = 0; replay < 2; ++replay) {
      MOE_CUDA(cudaGraphLaunch(exec, stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      check(result() == first, std::string(route.name) + ": graph replay " + std::to_string(replay + 1) + " agrees bit for bit");
    }
    MOE_CUDA(cudaGraphExecDestroy(exec));
    MOE_CUDA(cudaGraphDestroy(graph));
    MOE_CUDA(cudaStreamDestroy(stream));
    std::vector<int> identity(2 * kRecords);
    std::iota(identity.begin(), identity.end(), 0);
    set.place(identity);
    std::printf("  determinism (%s, %d tokens): rerun, other slots and two graph replays compared bit for bit\n", route.name, tokens);
  }
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::printf("routed experts at 2560 / 640 / 512 x top-10, K in {2, 2.5, 3, 4}\n");
  ExpertSet set;
  std::vector<int> identity(2 * kRecords);
  std::iota(identity.begin(), identity.end(), 0);
  set.place(identity);
  Buffers buffers;
  decode_arm(buffers, set);
  accumulator_arm(buffers, set);
  prefill_arm(buffers, set);
  determinism_arm(buffers, set);
  if (g_failed != 0) {
    std::fprintf(stderr, "test_moe_experts: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_moe_experts: OK\n");
  return 0;
}
