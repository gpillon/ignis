// The MoE ops configure nothing lazily -- OURS (spec flash-next/02 Acceptance 5, GitHub #300).
//
// Kernel attributes and the decode launch's grid are set by an explicit per-device preparation
// at load (ignis_moe_prepare, ignis_fp8_linear_prepare, or ignis_moe_workspace_init), never on an
// op's first call, so that call may be inside a CUDA graph capture like any other:
//
//   unprepared   every op refuses on a device nobody prepared, naming the call to make, and
//                enqueues nothing;
//   first call   after preparation, the very first router, FP8 linear and routed-decode calls of
//                the process are made inside a stream capture; the graph replays to the same
//                bits as the same calls made eagerly afterwards.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "fp8_test_common.h"
#include "ignis_fp8_linear.h"
#include "ignis_moe.h"
#include "moe_experts_common.h"
#include "moe_fixture.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

bool refused(int32_t rc, const char *what) {
  const std::string msg = ignis_moe_last_error();
  const bool ok = rc != 0 && msg.find("not prepared") != std::string::npos;
  check(ok, std::string(what) + " refuses on an unprepared device (got rc " + std::to_string(rc) + ": " + msg + ")");
  return ok;
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  const int tokens = 3;
  const uint32_t max_tokens = 16;
  std::vector<uint16_t> x = make_tokens(tokens, 4242, 2.0f);
  std::vector<uint16_t> router_w(static_cast<std::size_t>(kE) * kH);
  for (std::size_t i = 0; i < router_w.size(); ++i) router_w[i] = f32_to_bf16(hash_uniform(4243, i, 0.0346f));
  const Fp8Matrix fp8 = make_fp8(4244, kI, kH, 0.003f);
  std::vector<ignis_moe_slot> slots(static_cast<std::size_t>(kE) * 2, ignis_moe_slot{nullptr, 0, 0});
  std::vector<std::unique_ptr<DeviceBytes>> records;
  // Four physical record pairs, one per K class, shared by all 512 experts through the table.
  const uint32_t k2s[4] = {4, 5, 6, 8};
  for (int i = 0; i < 4; ++i) {
    for (int p = 0; p < 2; ++p) {
      const Record r = p == 0 ? make_record(82000 + 4 * i, kH, 2 * kI, k2s[i], 0.03f, 0.3f)
                              : make_record(83000 + 4 * i, kI, kH, k2s[(i + 1) % 4], 0.05f, 0.1f);
      const std::vector<uint8_t> bytes = r.bytes();
      records.push_back(std::make_unique<DeviceBytes>(bytes.size()));
      upload(*records.back(), bytes);
      for (int e = i; e < kE; e += 4) slots[e * 2 + p] = {records.back()->p, r.k2, 0};
    }
  }
  DeviceBytes dx(x.size() * 2), drouter(router_w.size() * 2), dfp8(fp8.payload.size()), dslots(slots.size() * sizeof(ignis_moe_slot)),
      ids(tokens * kTop * 4), w(tokens * kTop * 4), logits(tokens * kE * 4), y(tokens * kI * 4), shared(tokens * kH * 4),
      h(tokens * kI * 2), out(tokens * kH * 2), wsb(ignis_moe_workspace_bytes(IGNIS_MOE_DECODE_MAX_TOKENS, max_tokens)),
      acc(max_tokens * kH * 8);
  upload(dx, x);
  upload(drouter, router_w);
  upload(dfp8, fp8.payload);
  upload(dslots, slots);
  const ignis_moe_workspace ws{wsb.p, IGNIS_MOE_DECODE_MAX_TOKENS, max_tokens};

  std::printf("MoE ops before and after the device is prepared\n");
  refused(ignis_moe_router(dx.p, tokens, drouter.p, ids.as<int32_t>(), w.as<float>(), logits.as<float>(), nullptr), "router");
  refused(ignis_moe_experts_decode(dx.p, tokens, ids.as<int32_t>(), w.as<float>(), dslots.as<ignis_moe_slot>(), &ws,
                                   acc.as<int64_t>(), nullptr),
          "routed decode");
  refused(ignis_moe_experts_prefill(dx.p, tokens, ids.as<int32_t>(), w.as<float>(), dslots.as<ignis_moe_slot>(), &ws,
                                    acc.as<int64_t>(), nullptr),
          "routed prefill");
  refused(ignis_moe_combine(acc.as<int64_t>(), shared.as<float>(), dx.p, dx.p, tokens, out.p, nullptr), "combine");
  refused(ignis_fp8_linear(dfp8.p, kI, kH, dx.p, tokens, y.p, 1, nullptr), "fp8 linear");
  refused(ignis_fp8_linear_swiglu(dfp8.p, dfp8.p, kI, kH, dx.p, tokens, h.p, nullptr), "fp8 swiglu");
  MOE_CUDA(cudaDeviceSynchronize());

  // Prepare (through the workspace's init, as a load does), then make every first call in a capture.
  MOE_RC(ignis_moe_workspace_init(&ws, acc.as<int64_t>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto enqueue = [&](cudaStream_t s) {
    MOE_RC(ignis_moe_router(dx.p, tokens, drouter.p, ids.as<int32_t>(), w.as<float>(), logits.as<float>(), s));
    MOE_RC(ignis_fp8_linear(dfp8.p, kI, kH, dx.p, tokens, y.p, 1, s));
    MOE_CUDA(cudaMemsetAsync(acc.p, 0, static_cast<std::size_t>(tokens) * kH * 8, s));
    MOE_RC(ignis_moe_experts_decode(dx.p, tokens, ids.as<int32_t>(), w.as<float>(), dslots.as<ignis_moe_slot>(), &ws,
                                    acc.as<int64_t>(), s));
  };
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  enqueue(stream);
  MOE_CUDA(cudaStreamEndCapture(stream, &graph));
  MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  MOE_CUDA(cudaGraphLaunch(exec, stream));
  MOE_CUDA(cudaStreamSynchronize(stream));
  const auto g_ids = download<int32_t>(ids.p, tokens * kTop);
  const auto g_y = download<float>(y.p, tokens * kI);
  const auto g_acc = download<int64_t>(acc.p, static_cast<std::size_t>(tokens) * kH);
  enqueue(nullptr);
  MOE_CUDA(cudaDeviceSynchronize());
  check(download<int32_t>(ids.p, tokens * kTop) == g_ids, "router: first call in a capture equals the eager call");
  check(download<float>(y.p, tokens * kI) == g_y, "fp8 linear: first call in a capture equals the eager call");
  check(download<int64_t>(acc.p, static_cast<std::size_t>(tokens) * kH) == g_acc,
        "routed decode: first call in a capture equals the eager call");
  MOE_CUDA(cudaGraphExecDestroy(exec));
  MOE_CUDA(cudaGraphDestroy(graph));
  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "test_moe_prepare: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_moe_prepare: OK\n");
  return 0;
}
