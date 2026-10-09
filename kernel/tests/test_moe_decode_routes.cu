// The tickets route's staged kernel against the register kernel it replaced -- OURS (GitHub #306,
// the decode fusion roadmap's step 8; kernel/src/moe_decode_staged.cu).
//
// Both kernels compute the same contract (ignis_moe_experts_decode); they differ only in where
// partial sums are rounded. The staged kernel's gate/up item sums 256 inputs in its fp32 MMA chain
// before the exact fixed-point conversion (the register kernel's unit sums 640), and its down item
// sums one 128-input block of h, the output Hadamard taken per block and the five blocks added in
// fixed point (the register kernel runs one fp32 chain over all 640). Each fp16 operand scale is a
// power of two under either kernel, only the range it is taken over changes, so the gate/up
// operands are the same bits. Two differences remain: the fp32 partial sums of both projections
// round differently, ~1e-6 of an output; and the gate/up sums' ~1e-7 difference flips the fp16
// rounding of a few of h's 640 entries (down's operand) by one ulp (2^-11), each flip moving the
// token's output by about 2^-11 / sqrt(640) ~ 2e-5 of its norm. So the kernels agree to a few
// 1e-5 -- in discrete steps: an expert with no flipped entry agrees to ~1e-6 -- an order below
// the fp64 bound both are held to (test_moe_experts: 2e-3).
//
//   per expert  ten experts covering every K class of both projections, one token, each alone
//               (a one-hot routing weight): staged against registers, and both against fp64;
//   per call    1, 2, 3 and 4 tokens sharing experts: staged against registers per token;
//   wide        5 and 8 tokens on the tickets route take the register kernel: bit for bit;
//   determinism the staged call twice, with the records in other slots, and replayed twice from a
//               CUDA graph: every accumulator bit equal. Every call above runs on one workspace
//               in sequence, so each also proves the previous one left it as it found it;
//   trace       the traced launch (ignis_moe_experts_decode_trace) of either kernel computes the
//               untraced launch's bits and stamps every work unit once;
//   alignment   the staged kernel refuses an x that is not 16-byte aligned; the register kernel
//               takes it.
//
// Asserted: relative L2 <= 2e-4 and max <= 1e-3 of the token's largest output between the
// kernels (ten flipped entries at the worst); the measured figures are printed. ADR 0006 /
// docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_moe.h"
#include "moe_experts_common.h"
#include "moe_fixture.h"

#include "../src/moe_trace.h"

#include <cstdint>
#include <cstdio>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

constexpr double kRouteRelL2 = 2e-4;
constexpr double kRouteRelMax = 1e-3;
constexpr double kF64RelL2 = 2e-3;
constexpr uint32_t kPrefillTokens = 64;

struct Workspace {
  DeviceBytes base{ignis_moe_workspace_bytes(IGNIS_MOE_DECODE_MAX_TOKENS, kPrefillTokens)};
  DeviceBytes acc{static_cast<std::size_t>(kPrefillTokens) * kH * 8};  // max(decode, prefill) rows
  ignis_moe_workspace ws{base.p, IGNIS_MOE_DECODE_MAX_TOKENS, kPrefillTokens};
  Workspace() {
    MOE_RC(ignis_moe_workspace_init(&ws, acc.as<int64_t>(), nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  }
};

struct Call {
  std::vector<uint16_t> x;
  std::vector<int32_t> ids;
  std::vector<float> weights;
  DeviceBytes dx, dids, dw;
  Call(int tokens, uint32_t stream, const std::vector<int32_t> &first_ids = {})
      : x(make_tokens(tokens, stream, 2.0f)), dx(x.size() * 2), dids(static_cast<std::size_t>(tokens) * kTop * 4),
        dw(static_cast<std::size_t>(tokens) * kTop * 4) {
    make_routing(tokens, stream + 1, 3, ids, weights);
    for (std::size_t i = 0; i < first_ids.size(); ++i) ids[i] = first_ids[i];
    upload(dx, x);
    upload(dids, ids);
    upload(dw, weights);
  }
  void set_weights(const std::vector<float> &w) {
    weights = w;
    upload(dw, weights);
  }
};

std::vector<int64_t> run(Workspace &w, const ExpertSet &set, const Call &c, int tokens, uint32_t route) {
  MOE_CUDA(cudaMemset(w.acc.p, 0, static_cast<std::size_t>(tokens) * kH * 8));
  ignis_moe_workspace ws = w.ws;
  ws.decode_route = route;
  MOE_RC(ignis_moe_experts_decode(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(), &ws,
                                  w.acc.as<int64_t>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  return download<int64_t>(w.acc.p, static_cast<std::size_t>(tokens) * kH);
}

struct Worst {
  double l2 = 0.0, max = 0.0;
};

// Staged against registers for every token of the call: asserted, the worst kept.
void compare_routes(const std::string &what, const std::vector<int64_t> &staged, const std::vector<int64_t> &registers,
                    int tokens, Worst *worst) {
  for (int t = 0; t < tokens; ++t) {
    double l2 = 0.0, mx = 0.0;
    rel_errors(fixed_to_f64(staged, t), fixed_to_f64(registers, t), &l2, &mx);
    worst->l2 = std::max(worst->l2, l2);
    worst->max = std::max(worst->max, mx);
    check(l2 <= kRouteRelL2, what + ", token " + std::to_string(t) + ": staged against registers, relative L2 " + std::to_string(l2));
    check(mx <= kRouteRelMax, what + ", token " + std::to_string(t) + ": staged against registers, relative max " + std::to_string(mx));
  }
}

// Experts that between them use every K class of both projections, then others up to ten.
std::vector<int32_t> covering_ten() {
  std::set<uint32_t> gu, dn;
  std::vector<int32_t> picks;
  for (int e = 0; e < kE && picks.size() < static_cast<std::size_t>(kTop); ++e) {
    const uint32_t g = kRecordK2[ExpertSet::gu_record(e)];
    const uint32_t d = kRecordK2[(ExpertSet::dn_record(e) + 5) % kRecords];
    if (!gu.count(g) || !dn.count(d) || (gu.size() == 4 && dn.size() == 4)) {
      picks.push_back(e);
      gu.insert(g);
      dn.insert(d);
    }
  }
  return picks;
}

void per_expert_arm(Workspace &w, const ExpertSet &set) {
  const std::vector<int32_t> ten = covering_ten();
  check(ten.size() == static_cast<std::size_t>(kTop), "ten covering experts");
  Call c(1, 700, ten);
  Worst routes;
  double f64_worst[2] = {0.0, 0.0};  // staged, registers
  std::vector<double> each;
  for (int r = 0; r < kTop; ++r) {
    std::vector<float> one_hot(kTop, 0.0f);
    one_hot[r] = 1.0f;
    c.set_weights(one_hot);
    const std::vector<int64_t> staged = run(w, set, c, 1, IGNIS_MOE_DECODE_TICKETS);
    const std::vector<int64_t> registers = run(w, set, c, 1, IGNIS_MOE_DECODE_REGISTERS);
    const std::string what = "expert " + std::to_string(c.ids[r]) + " (gate/up k2 " +
                             std::to_string(kRecordK2[ExpertSet::gu_record(c.ids[r])]) + ", down k2 " +
                             std::to_string(kRecordK2[(ExpertSet::dn_record(c.ids[r]) + 5) % kRecords]) + ")";
    Worst one;
    compare_routes(what, staged, registers, 1, &one);
    each.push_back(one.l2);
    routes.l2 = std::max(routes.l2, one.l2);
    routes.max = std::max(routes.max, one.max);
    const std::vector<double> ref = expert_f64(set, c.ids[r], token_f64(c.x, 0));
    for (int k = 0; k < 2; ++k) {
      double l2 = 0.0, mx = 0.0;
      rel_errors(fixed_to_f64(k == 0 ? staged : registers, 0), ref, &l2, &mx);
      f64_worst[k] = std::max(f64_worst[k], l2);
      check(l2 <= kF64RelL2, what + (k == 0 ? ", staged" : ", registers") + " against fp64, relative L2 " + std::to_string(l2));
    }
  }
  std::sort(each.begin(), each.end());
  std::printf("  per expert, 10 experts alone: staged against registers relative L2");
  for (double v : each) std::printf(" %.1e", v);
  std::printf(" (worst max %.2e); against fp64 worst relative L2 %.2e staged, %.2e registers\n", routes.max, f64_worst[0],
              f64_worst[1]);
}

void per_call_arm(Workspace &w, const ExpertSet &set) {
  for (int tokens : {1, 2, 3, 4}) {
    Worst worst;
    for (uint32_t stream : {800u, 820u, 840u}) {
      const Call c(tokens, stream + 10 * tokens);
      compare_routes("call of " + std::to_string(tokens) + " token(s)", run(w, set, c, tokens, IGNIS_MOE_DECODE_TICKETS),
                     run(w, set, c, tokens, IGNIS_MOE_DECODE_REGISTERS), tokens, &worst);
    }
    std::printf("  per call, %d token(s), 3 routings: staged against registers worst relative L2 %.2e, max %.2e\n", tokens,
                worst.l2, worst.max);
  }
  for (int tokens : {5, 8}) {
    const Call c(tokens, 900 + tokens);
    check(run(w, set, c, tokens, IGNIS_MOE_DECODE_TICKETS) == run(w, set, c, tokens, IGNIS_MOE_DECODE_REGISTERS),
          std::to_string(tokens) + " tokens on the tickets route take the register kernel, bit for bit");
  }
  std::printf("  5 and 8 tokens on the tickets route: the register kernel's bits\n");
}

void determinism_arm(Workspace &w, ExpertSet &set) {
  for (int tokens : {1, 3}) {
    const Call c(tokens, 950 + tokens);
    const std::vector<int64_t> first = run(w, set, c, tokens, IGNIS_MOE_DECODE_TICKETS);
    check(run(w, set, c, tokens, IGNIS_MOE_DECODE_TICKETS) == first, "staged: a second run agrees bit for bit");
    std::vector<int> order(2 * kRecords);
    std::iota(order.rbegin(), order.rend(), 0);
    set.place(order);
    check(run(w, set, c, tokens, IGNIS_MOE_DECODE_TICKETS) == first, "staged: records in other slots agree bit for bit");
    std::vector<int> identity(2 * kRecords);
    std::iota(identity.begin(), identity.end(), 0);
    set.place(identity);
    cudaStream_t stream;
    MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    MOE_CUDA(cudaMemsetAsync(w.acc.p, 0, static_cast<std::size_t>(tokens) * kH * 8, stream));
    MOE_RC(ignis_moe_experts_decode(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(),
                                    &w.ws, w.acc.as<int64_t>(), stream));
    MOE_CUDA(cudaStreamEndCapture(stream, &graph));
    MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    for (int replay = 0; replay < 2; ++replay) {
      MOE_CUDA(cudaGraphLaunch(exec, stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      check(download<int64_t>(w.acc.p, static_cast<std::size_t>(tokens) * kH) == first,
            "staged: graph replay " + std::to_string(replay + 1) + " agrees bit for bit");
    }
    MOE_CUDA(cudaGraphExecDestroy(exec));
    MOE_CUDA(cudaGraphDestroy(graph));
    MOE_CUDA(cudaStreamDestroy(stream));
  }
  std::printf("  determinism (1 and 3 tokens): rerun, other slots and two graph replays bit for bit\n");
}

// The diagnostic launch: on both kernels the traced call adds the untraced call's bits, and every
// work unit leaves exactly one record (75 items per distinct expert staged, 60 units registers).
void trace_arm(Workspace &w, const ExpertSet &set) {
  DeviceBytes trace(IGNIS_MOE_TRACE_BYTES);
  for (uint32_t route : {IGNIS_MOE_DECODE_TICKETS, IGNIS_MOE_DECODE_REGISTERS}) {
    for (int tokens : {1, 3}) {
      const Call c(tokens, 980 + tokens);
      const std::set<int32_t> distinct(c.ids.begin(), c.ids.end());
      const std::vector<int64_t> plain = run(w, set, c, tokens, route);
      ignis_moe_workspace ws = w.ws;
      ws.decode_route = route;
      MOE_CUDA(cudaMemset(w.acc.p, 0, static_cast<std::size_t>(tokens) * kH * 8));
      MOE_CUDA(cudaMemset(trace.p, 0, trace.bytes));
      MOE_RC(ignis_moe_experts_decode_trace(c.dx.p, tokens, c.dids.as<int32_t>(), c.dw.as<float>(),
                                            set.d_slots.as<ignis_moe_slot>(), &ws, w.acc.as<int64_t>(),
                                            trace.as<unsigned long long>(), nullptr));
      MOE_CUDA(cudaDeviceSynchronize());
      const std::string what = std::string(route == IGNIS_MOE_DECODE_TICKETS ? "staged" : "registers") + ", " +
                               std::to_string(tokens) + " token(s)";
      check(download<int64_t>(w.acc.p, static_cast<std::size_t>(tokens) * kH) == plain,
            what + ": the traced launch adds the untraced launch's bits");
      const std::vector<unsigned long long> r = download<unsigned long long>(trace.p, trace.bytes / 8);
      std::size_t units = 0;
      for (int u = 0; u < IGNIS_MOE_TRACE_UNITS; ++u) units += r[static_cast<std::size_t>(u) * IGNIS_MOE_TRACE_WORDS + 2] != 0;
      const std::size_t want = distinct.size() * (route == IGNIS_MOE_DECODE_TICKETS ? 75 : 60);
      check(units == want, what + ": " + std::to_string(units) + " unit records, " + std::to_string(want) + " work units");
    }
  }
  std::printf("  trace: both kernels' traced launches equal their untraced bits and record every unit once\n");
}

// The staged kernel copies x 16 bytes at a time and refuses an x off that alignment; the register
// kernel reads it an element at a time and takes it.
void alignment_arm(Workspace &w, const ExpertSet &set) {
  const Call c(2, 990);
  const char *shifted = static_cast<const char *>(c.dx.p) + 2;  // token 0 from its second element
  ignis_moe_workspace ws = w.ws;
  check(ignis_moe_experts_decode(shifted, 1, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(), &ws,
                                 w.acc.as<int64_t>(), nullptr) == -1 &&
            std::string(ignis_moe_last_error()).find("16-byte aligned") != std::string::npos,
        "staged: an x off 16-byte alignment is refused");
  ws.decode_route = IGNIS_MOE_DECODE_REGISTERS;
  MOE_CUDA(cudaMemset(w.acc.p, 0, kH * 8));
  check(ignis_moe_experts_decode(shifted, 1, c.dids.as<int32_t>(), c.dw.as<float>(), set.d_slots.as<ignis_moe_slot>(), &ws,
                                 w.acc.as<int64_t>(), nullptr) == 0,
        "registers: an x off 16-byte alignment is taken");
  MOE_CUDA(cudaDeviceSynchronize());
  MOE_CUDA(cudaMemset(w.acc.p, 0, kH * 8));
  std::printf("  alignment: the staged kernel refuses an x off 16 bytes, the register kernel takes it\n");
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::printf("tickets route: the staged kernel against the register kernel\n");
  ExpertSet set;
  std::vector<int> identity(2 * kRecords);
  std::iota(identity.begin(), identity.end(), 0);
  set.place(identity);
  Workspace w;
  per_expert_arm(w, set);
  per_call_arm(w, set);
  determinism_arm(w, set);
  trace_arm(w, set);
  alignment_arm(w, set);
  if (g_failed != 0) {
    std::fprintf(stderr, "test_moe_decode_routes: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_moe_decode_routes: OK\n");
  return 0;
}
