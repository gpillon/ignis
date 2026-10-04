// A selected expert projection that is not resident traps the expert kernels -- OURS (spec
// flash-next/02, the slot table, GitHub #300).
//
// Residency guarantees every projection the router selected is resident before an expert op
// runs; an entry that is not is a programming error, and the kernel must stop rather than read
// it as weights. This drives one route (argv[1]: decode or prefill) with a slot table whose
// entries are all absent and passes only if the launch fails. A trap leaves the CUDA context
// unusable, so each route is its own process (two CTest entries).
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails (it would also
// make the launch fail, so the device is checked first).

#include "ignis_moe.h"
#include "moe_fixture.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace moe_test;

int main(int argc, char **argv) {
  const bool prefill = argc > 1 && std::strcmp(argv[1], "prefill") == 0;
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  MOE_CUDA(cudaFree(nullptr));  // a working context before the deliberate failure
  const int tokens = prefill ? 32 : 2;
  const uint32_t max_tokens = 64;
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * IGNIS_MOE_HIDDEN, 0x3F80);  // 1.0
  std::vector<int32_t> ids(static_cast<std::size_t>(tokens) * IGNIS_MOE_TOP_K);
  std::vector<float> w(ids.size(), 0.1f);
  for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>((i % IGNIS_MOE_TOP_K) * 7 + i / IGNIS_MOE_TOP_K);
  std::vector<ignis_moe_slot> slots(static_cast<std::size_t>(IGNIS_MOE_EXPERTS) * 2, ignis_moe_slot{nullptr, 0, 0});
  DeviceBytes dx(x.size() * 2), dids(ids.size() * 4), dw(w.size() * 4), dslots(slots.size() * sizeof(ignis_moe_slot)),
      ws(ignis_moe_workspace_bytes(max_tokens)), acc(static_cast<std::size_t>(max_tokens) * IGNIS_MOE_HIDDEN * 8);
  upload(dx, x);
  upload(dids, ids);
  upload(dw, w);
  upload(dslots, slots);
  MOE_RC(ignis_moe_workspace_init(ws.p, max_tokens, acc.as<int64_t>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());

  const int32_t rc = prefill ? ignis_moe_experts_prefill(dx.p, tokens, dids.as<int32_t>(), dw.as<float>(),
                                                         dslots.as<ignis_moe_slot>(), ws.p, max_tokens, acc.as<int64_t>(), nullptr)
                             : ignis_moe_experts_decode(dx.p, tokens, dids.as<int32_t>(), dw.as<float>(),
                                                        dslots.as<ignis_moe_slot>(), ws.p, acc.as<int64_t>(), nullptr);
  const cudaError_t err = cudaDeviceSynchronize();
  std::printf("slot trap (%s route): launch rc %d, synchronize: %s\n", prefill ? "prefill" : "decode", rc, cudaGetErrorString(err));
  if (rc == 0 && err == cudaSuccess) {
    std::fprintf(stderr, "test_moe_slot_trap: an absent selected slot did not stop the kernel\n");
    return 1;
  }
  std::printf("test_moe_slot_trap: OK\n");
  // The context is gone after the trap; leave without touching it again.
  std::fflush(stdout);
  std::_Exit(0);
}
