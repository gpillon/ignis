// ignis kernel leaf: where a cost lever measured on the RTX 5090 alone runs -- OURS (spec #684,
// GitHub #693, #694), leaf-internal.
//
// Three levers make the 27B's rounds past 8 columns a lane cheaper and write the same bytes as
// today's path: GitHub #693's 16-token W8A8 tile (IGNIS_FP8_W8A8_TILE16, kernel/src/fp8_exception.h)
// and GitHub #694's two drafter launches (IGNIS_DFLASH2_APPEND_KV_ONLY, IGNIS_DFLASH2_SMALL_T_FILL,
// kernel/src/dflash2_small_t.h). Each was measured faster on the RTX 5090 alone, and the spec keeps
// every other card as earlier releases ran it, so each lever's measurement switch reads:
//   - unset (or empty): the lever on a card of compute capability major 12, today's path on every
//     other, and on a device that cannot say its generation;
//   - `1`: the lever on any card, a measurement off the RTX 5090;
//   - `0`: today's path on every card.
// Any other value is refused, naming the variable. Each lever reads its switch at each call, so a
// graph keeps what its capture read, and once at load (kernel/src/model.cu), so a value it does
// not know refuses the load by name.
#pragma once

#include "process_environment.h"

#include <cuda_runtime.h>

namespace ignis::card_lever {

// What a lever's measurement switch names: unset (or empty), `1` or `0`.
enum class Switch { Unset, On, Off };

// The switch `name` in the operating system's environment (kernel/src/process_environment.h),
// read at each call. Throws std::invalid_argument, naming the variable, on any value but `1` or
// `0`.
inline Switch named(const char *name) {
  return ignis::named_switch(name, Switch::Unset, {{"1", Switch::On}, {"0", Switch::Off}});
}

// The cards a lever runs on with its switch unset: compute capability major 12, where it was
// measured.
constexpr bool runs_by_default(int compute_capability_major) { return compute_capability_major == 12; }

// The one rule: whether the lever runs, its switch naming `named`, on a card of compute capability
// major `compute_capability_major` (0 when the device cannot say).
constexpr bool takes(Switch named, int compute_capability_major) {
  return named == Switch::On || (named == Switch::Unset && runs_by_default(compute_capability_major));
}

// The current device's compute capability major, read once per process (the leaf drives one card);
// 0 when the device cannot say.
inline int device_major() {
  static const int major = [] {
    int device = 0;
    int value = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
      return 0;
    }
    return value;
  }();
  return major;
}

// Whether the lever behind the switch `name` runs at this call on this card (`takes`). Throws as
// `named` does.
inline bool takes_here(const char *name) { return takes(named(name), device_major()); }

} // namespace ignis::card_lever
