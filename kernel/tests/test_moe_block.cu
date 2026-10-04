// Flash-Next's whole MoE block composed from its ops at real geometry, on synthetic weights --
// OURS (spec flash-next/02 Acceptance 4's composition, GitHub #300).
//
// router -> routed experts (decode route at 1 and 3 tokens, prefill route at 64 and 300) ->
// shared expert -> combine, against an fp64 restatement of the checkpoint's block
// (Qwen4ExpTextSparseMoeBlock: sum of the routed experts weighted by the router, plus
// sigmoid(x . w_gate) times the shared expert) over the same decoded weights and the kernel's own
// selection. The selection itself is held to fp64 by test_moe_router; here it is only counted.
// The recorded-real-activation version of this test needs the converter's references and waits
// for the artifact (spec 02, machine-local).
//
// The whole block is also captured in one CUDA graph and replayed twice against the eager run,
// bit for bit: every op is capturable and leaves its workspace ready for the next replay.
//
// Tolerance: per token, relative L2 error <= 5e-3: the routed path's fp16 operand rounding
// (~4e-4, test_moe_experts), the shared expert's BF16 h (~1e-3) and the output's BF16 rounding
// (2^-9 / sqrt(3) ~ 1.1e-3) are the terms. Measured figures are printed.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "fp8_test_common.h"
#include "ignis_moe.h"
#include "moe_experts_common.h"
#include "moe_fixture.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

constexpr double kRelL2 = 5e-3;
constexpr uint32_t kMaxTokens = 512;

struct Block {
  ExpertSet experts;
  std::vector<uint16_t> router_w;  // BF16 [512][2560]
  Fp8Matrix gate, up, down;
  std::vector<uint16_t> w_gate;  // BF16 [2560]
  DeviceBytes d_router, d_gate, d_up, d_down, d_wgate;
  DeviceBytes workspace{ignis_moe_workspace_bytes(kMaxTokens)};
  DeviceBytes acc{static_cast<std::size_t>(kMaxTokens) * kH * 8};

  Block()
      : router_w(static_cast<std::size_t>(kE) * kH), gate(make_fp8(6001, kI, kH, 0.003f)), up(make_fp8(6002, kI, kH, 0.003f)),
        down(make_fp8(6003, kH, kI, 0.004f)), w_gate(kH), d_router(router_w.size() * 2), d_gate(gate.payload.size()),
        d_up(up.payload.size()), d_down(down.payload.size()), d_wgate(w_gate.size() * 2) {
    for (std::size_t i = 0; i < router_w.size(); ++i) router_w[i] = f32_to_bf16(hash_uniform(6004, i, 0.0346f));
    for (int k = 0; k < kH; ++k) w_gate[k] = f32_to_bf16(hash_uniform(6005, k, 0.02f));
    upload(d_router, router_w);
    upload(d_gate, gate.payload);
    upload(d_up, up.payload);
    upload(d_down, down.payload);
    upload(d_wgate, w_gate);
    std::vector<int> identity(2 * kRecords);
    std::iota(identity.begin(), identity.end(), 0);
    experts.place(identity);
    MOE_RC(ignis_moe_workspace_init(workspace.p, kMaxTokens, acc.as<int64_t>(), nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  }
};

void block_arm(Block &b, int tokens) {
  const std::vector<uint16_t> x = make_tokens(tokens, 6100 + tokens, 1.7f);
  DeviceBytes dx(x.size() * 2), ids(static_cast<std::size_t>(tokens) * kTop * 4), w(static_cast<std::size_t>(tokens) * kTop * 4),
      logits(static_cast<std::size_t>(tokens) * kE * 4), h(static_cast<std::size_t>(tokens) * kI * 2),
      shared(static_cast<std::size_t>(tokens) * kH * 4), out(static_cast<std::size_t>(tokens) * kH * 2);
  upload(dx, x);
  auto enqueue = [&](cudaStream_t stream) {
    MOE_RC(ignis_moe_router(dx.p, tokens, b.d_router.p, ids.as<int32_t>(), w.as<float>(), logits.as<float>(), stream));
    if (tokens <= IGNIS_MOE_DECODE_MAX_TOKENS) {
      MOE_RC(ignis_moe_experts_decode(dx.p, tokens, ids.as<int32_t>(), w.as<float>(), b.experts.d_slots.as<ignis_moe_slot>(),
                                      b.workspace.p, b.acc.as<int64_t>(), stream));
    } else {
      MOE_RC(ignis_moe_experts_prefill(dx.p, tokens, ids.as<int32_t>(), w.as<float>(), b.experts.d_slots.as<ignis_moe_slot>(),
                                       b.workspace.p, kMaxTokens, b.acc.as<int64_t>(), stream));
    }
    MOE_RC(ignis_moe_shared_expert(b.d_gate.p, b.d_up.p, b.d_down.p, dx.p, tokens, h.p, shared.as<float>(), stream));
    MOE_RC(ignis_moe_combine(b.acc.as<int64_t>(), shared.as<float>(), dx.p, b.d_wgate.p, tokens, out.p, stream));
  };
  enqueue(nullptr);
  MOE_CUDA(cudaDeviceSynchronize());
  // The whole block, every op of it, captured in one CUDA graph and replayed twice: the same
  // bits as the eager run (combine leaves the accumulator zeroed for the next replay).
  {
    const auto eager = download<uint16_t>(out.p, static_cast<std::size_t>(tokens) * kH);
    cudaStream_t stream;
    MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    enqueue(stream);
    MOE_CUDA(cudaStreamEndCapture(stream, &graph));
    MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    for (int replay = 0; replay < 2; ++replay) {
      MOE_CUDA(cudaMemset(out.p, 0, out.bytes));
      MOE_CUDA(cudaGraphLaunch(exec, stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      check(download<uint16_t>(out.p, static_cast<std::size_t>(tokens) * kH) == eager,
            "block x " + std::to_string(tokens) + ": graph replay " + std::to_string(replay + 1) + " equals the eager run");
    }
    MOE_CUDA(cudaGraphExecDestroy(exec));
    MOE_CUDA(cudaGraphDestroy(graph));
    MOE_CUDA(cudaStreamDestroy(stream));
  }
  const auto got_ids = download<int32_t>(ids.p, static_cast<std::size_t>(tokens) * kTop);
  const auto got_w = download<float>(w.p, static_cast<std::size_t>(tokens) * kTop);
  const auto got = download<uint16_t>(out.p, static_cast<std::size_t>(tokens) * kH);

  double worst = 0.0;
  const int step = tokens > 16 ? tokens / 11 : 1;
  int checked = 0;
  for (int t = 0; t < tokens; t += step, ++checked) {
    const std::vector<double> xt = token_f64(x, t);
    std::vector<double> ref = routed_f64(b.experts, &got_ids[static_cast<std::size_t>(t) * kTop],
                                         &got_w[static_cast<std::size_t>(t) * kTop], xt);
    std::vector<double> h64(kI);
    for (int r = 0; r < kI; ++r) {
      double g, ga, u, ua;
      fp8_row_f64(b.gate, r, xt.data(), &g, &ga);
      fp8_row_f64(b.up, r, xt.data(), &u, &ua);
      h64[r] = silu(g) * u;
    }
    double dot = 0.0;
    for (int k = 0; k < kH; ++k) dot += xt[k] * bf16_to_f32(b.w_gate[k]);
    const double sg = 1.0 / (1.0 + std::exp(-dot));
    for (int r = 0; r < kH; ++r) {
      double s, sa;
      fp8_row_f64(b.down, r, h64.data(), &s, &sa);
      ref[r] += sg * s;
    }
    std::vector<double> o(kH);
    for (int k = 0; k < kH; ++k) o[k] = bf16_to_f32(got[static_cast<std::size_t>(t) * kH + k]);
    double l2 = 0.0, mx = 0.0;
    rel_errors(o, ref, &l2, &mx);
    worst = std::max(worst, l2);
    check(l2 <= kRelL2, "block x " + std::to_string(tokens) + ": token " + std::to_string(t) + " relative L2 " + std::to_string(l2));
  }
  std::printf("  block, %3d token(s) (%s route): %d tokens checked, worst relative L2 %.2e\n", tokens,
              tokens <= IGNIS_MOE_DECODE_MAX_TOKENS ? "decode" : "prefill", checked, worst);
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::printf("whole MoE block: router -> routed experts -> shared expert -> combine\n");
  Block block;
  for (int tokens : {1, 3, 64, 300}) block_arm(block, tokens);
  if (g_failed != 0) {
    std::fprintf(stderr, "test_moe_block: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_moe_block: OK\n");
  return 0;
}
