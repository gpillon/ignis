// Flash-Next's routed experts at real geometry (fused gate/up 2560 x 1280, down 640 x 2560, ten
// of 512 experts per token, K per expert) -- OURS (spec flash-next/02 Acceptance 3 and 5,
// GitHub #300).
//
//   decode      1, 2, 3 and 8 tokens through ignis_moe_experts_decode, one launch for every
//               expert and K, against the fp64 product of the exactly decoded weights. The
//               routing covers all four K classes for both projections in one launch and
//               shares experts between consecutive tokens.
//   determinism the same call twice, the records placed in a different order in device memory
//               (other slot addresses), and the call captured in a CUDA graph and replayed
//               twice: every accumulator bit equal.
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
  DeviceBytes workspace{ignis_moe_workspace_bytes(kMaxTokens)};
  DeviceBytes acc{static_cast<std::size_t>(kMaxTokens) * kH * 8};
  Buffers() {
    MOE_RC(ignis_moe_workspace_init(workspace.p, kMaxTokens, acc.as<int64_t>(), nullptr));
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

struct Call {
  std::vector<uint16_t> x;
  std::vector<int32_t> ids;
  std::vector<float> weights;
  DeviceBytes dx, dids, dw;
  Call(int tokens, uint32_t stream)
      : x(make_tokens(tokens, stream, 2.0f)), dx(x.size() * 2), dids(static_cast<std::size_t>(tokens) * kTop * 4),
        dw(static_cast<std::size_t>(tokens) * kTop * 4) {
    make_routing(tokens, stream + 1, 3, ids, weights);
    const std::vector<int32_t> cover = covering_experts();
    for (std::size_t i = 0; i < cover.size() && i < static_cast<std::size_t>(kTop); ++i) {
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
                                  set.d_slots.as<ignis_moe_slot>(), b.workspace.p, b.acc.as<int64_t>(), nullptr));
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

void determinism_arm(Buffers &b, ExpertSet &set) {
  const int tokens = 3;
  const Call c(tokens, 900);
  const std::vector<int64_t> first = run_decode(b, set, c, tokens);
  check(run_decode(b, set, c, tokens) == first, "decode: a second run agrees bit for bit");

  // Other slot addresses: the records in reverse order.
  std::vector<int> order(2 * kRecords);
  std::iota(order.rbegin(), order.rend(), 0);
  set.place(order);
  check(run_decode(b, set, c, tokens) == first, "decode: records in other slots agree bit for bit");

  // Graph capture and two replays.
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  MOE_CUDA(cudaMemsetAsync(b.acc.p, 0, static_cast<std::size_t>(tokens) * kH * 8, stream));
  MOE_RC(ignis_moe_experts_decode(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(),
                                  set.d_slots.as<ignis_moe_slot>(), b.workspace.p, b.acc.as<int64_t>(), stream));
  MOE_CUDA(cudaStreamEndCapture(stream, &graph));
  MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  for (int replay = 0; replay < 2; ++replay) {
    MOE_CUDA(cudaGraphLaunch(exec, stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    check(download<int64_t>(b.acc.p, static_cast<std::size_t>(tokens) * kH) == first,
          "decode: graph replay " + std::to_string(replay + 1) + " agrees bit for bit");
  }
  MOE_CUDA(cudaGraphExecDestroy(exec));
  MOE_CUDA(cudaGraphDestroy(graph));
  MOE_CUDA(cudaStreamDestroy(stream));
  std::vector<int> identity(2 * kRecords);
  std::iota(identity.begin(), identity.end(), 0);
  set.place(identity);
  std::printf("  determinism: rerun, other slots and two graph replays compared bit for bit\n");
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
  determinism_arm(buffers, set);
  if (g_failed != 0) {
    std::fprintf(stderr, "test_moe_experts: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_moe_experts: OK\n");
  return 0;
}
