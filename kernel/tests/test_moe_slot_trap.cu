// What the expert kernels refuse to read as weights or turn into a finite answer -- OURS (spec
// flash-next/02, the slot table and the routed accumulator, GitHub #300).
//
// Each mode (argv[1]) drives one route into one condition and passes only if the kernel stops
// with a trap -- cudaErrorLaunchFailure, which is what __trap() reports; an out-of-bounds access
// would be cudaErrorIllegalAddress and fails the test:
//
//   decode, prefill        a selected expert whose slot holds no record (residency did not make
//                          it resident: a programming error);
//   decode-id, prefill-id  an expert id outside [0, 512), which would index past the slot table
//                          and the grouping buffers;
//   decode-nan             a token whose activations are NaN, routed to resident experts: its
//                          contributions are not finite, and the fixed-point accumulator traps
//                          rather than storing a finite wrong number.
//
// A trap leaves the CUDA context unusable, so every mode is its own process (one CTest entry
// each). ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_moe.h"
#include "moe_experts_common.h"
#include "moe_fixture.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace moe_test;

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "decode";
  const bool prefill = mode.rfind("prefill", 0) == 0;
  const bool bad_id = mode.find("-id") != std::string::npos;
  const bool nan = mode == "decode-nan";
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  MOE_CUDA(cudaFree(nullptr));  // a working context before the deliberate failure
  const int tokens = prefill ? 32 : 2;
  const uint32_t max_tokens = 64;
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * kH, f32_to_bf16(0.5f));
  if (nan) {
    for (int k = 0; k < kH; ++k) x[k] = 0x7FC0;  // token 0: quiet NaN
  }
  std::vector<int32_t> ids(static_cast<std::size_t>(tokens) * kTop);
  for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>(i % kTop);  // experts 0..9
  if (bad_id) ids[3] = 600;
  std::vector<float> w(ids.size(), 0.1f);

  // Resident records for experts 0..9 unless the mode is the missing slot.
  std::vector<ignis_moe_slot> slots(static_cast<std::size_t>(kE) * 2, ignis_moe_slot{nullptr, 0, 0});
  std::vector<std::unique_ptr<DeviceBytes>> records;
  if (bad_id || nan) {
    for (int e = 0; e < kTop; ++e) {
      for (int p = 0; p < 2; ++p) {
        const Record r = p == 0 ? make_record(80000 + 4 * e, kH, 2 * kI, 5, 0.03f, 0.3f)
                                : make_record(81000 + 4 * e, kI, kH, 5, 0.05f, 0.1f);
        const std::vector<uint8_t> bytes = r.bytes();
        records.push_back(std::make_unique<DeviceBytes>(bytes.size()));
        upload(*records.back(), bytes);
        slots[e * 2 + p] = {records.back()->p, r.k2, 0};
      }
    }
  }
  DeviceBytes dx(x.size() * 2), dids(ids.size() * 4), dw(w.size() * 4), dslots(slots.size() * sizeof(ignis_moe_slot)),
      wsb(ignis_moe_workspace_bytes(IGNIS_MOE_DECODE_MAX_TOKENS, max_tokens)), acc(static_cast<std::size_t>(max_tokens) * kH * 8);
  upload(dx, x);
  upload(dids, ids);
  upload(dw, w);
  upload(dslots, slots);
  const ignis_moe_workspace ws{wsb.p, IGNIS_MOE_DECODE_MAX_TOKENS, max_tokens};
  MOE_RC(ignis_moe_workspace_init(&ws, acc.as<int64_t>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());

  const int32_t rc = prefill ? ignis_moe_experts_prefill(dx.p, tokens, dids.as<int32_t>(), dw.as<float>(),
                                                         dslots.as<ignis_moe_slot>(), &ws, acc.as<int64_t>(), nullptr)
                             : ignis_moe_experts_decode(dx.p, tokens, dids.as<int32_t>(), dw.as<float>(),
                                                        dslots.as<ignis_moe_slot>(), &ws, acc.as<int64_t>(), nullptr);
  const cudaError_t err = cudaDeviceSynchronize();
  std::printf("trap (%s): launch rc %d, synchronize: %s (%d)\n", mode.c_str(), rc, cudaGetErrorString(err), static_cast<int>(err));
  if (rc != 0 || err != cudaErrorLaunchFailure) {
    std::fprintf(stderr, "test_moe_slot_trap (%s): expected the launch to trap (cudaErrorLaunchFailure)\n", mode.c_str());
    std::fflush(stderr);
    std::_Exit(1);
  }
  std::printf("test_moe_slot_trap (%s): OK\n", mode.c_str());
  // The context is gone after the trap; leave without touching it again.
  std::fflush(stdout);
  std::_Exit(0);
}
