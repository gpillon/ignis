// GitHub #572 (spec #570): the attention route of the 27B's decode and verify rounds over
// hq-e8-2b, as a load picks it (kernel/src/hq_decode_attention.h) -- OURS.
//
// The rule, with the card's generation and the image injected: a Blackwell card (compute
// capability major 12) takes ignis's warp-specialized route unless the measurement switch
// IGNIS_HQ_DECODE_ATTENTION names today's route; every other card, a BF16 load, and a binary with
// no image of the route for the device keep today's route whatever the switch says; any value but
// `ours` and `vendored` is refused naming the variable. This is the check that an RTX 30 or 40
// card keeps today's route, on a card that is not one.
//
// Then the probe on the card at hand: the binary holds the route's image exactly when its build
// names the card's own arch-specific Blackwell target (the route is in the sm_120a set,
// kernel/CMakeLists.txt), and a probe that finds none leaves no error behind for the next launch
// check to read as its own. Not sm120a-only: on an older card the probe's answer is "no image".
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing GPU fails this test.

#include "hq_decode_attention.h"

#include "ignis_device.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using ignis::hq_decode_attention::Switch;

int g_failed = 0;

void check(bool ok, const std::string &what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failed;
}

// Sets (or, with null, removes) a variable in the operating system's environment, as a host
// process's own runtime does (Rust's std::env::set_var).
void set_environment(const char *name, const char *value) {
#ifdef _WIN32
  SetEnvironmentVariableA(name, value);
#else
  if (value != nullptr) {
    setenv(name, value, 1);
  } else {
    unsetenv(name);
  }
#endif
}

void check_rule() {
  using ignis::hq_decode_attention::takes_ours;
  check(takes_ours(Switch::Unset, true, 12, true), "Blackwell takes the route by default");
  check(takes_ours(Switch::Ours, true, 12, true), "Blackwell takes the route when ours is named");
  check(!takes_ours(Switch::Vendored, true, 12, true), "vendored keeps today's route on Blackwell");
  for (const int major : {7, 8, 9, 10, 11, 13}) {
    for (const Switch named : {Switch::Unset, Switch::Ours, Switch::Vendored}) {
      check(!takes_ours(named, true, major, true),
            "a card of major " + std::to_string(major) + " keeps today's route whatever the switch says");
    }
  }
  for (const Switch named : {Switch::Unset, Switch::Ours}) {
    check(!takes_ours(named, true, 12, false), "Blackwell with no image of the route keeps today's route");
    check(!takes_ours(named, false, 12, true), "a BF16 load keeps today's route");
  }
}

void check_switch() {
  constexpr const char *kSwitch = "IGNIS_HQ_DECODE_ATTENTION";
  using ignis::hq_decode_attention::switch_for_load;
  set_environment(kSwitch, nullptr);
  check(switch_for_load() == Switch::Unset, "unset, the card decides");
  set_environment(kSwitch, "ours");
  check(switch_for_load() == Switch::Ours, "IGNIS_HQ_DECODE_ATTENTION=ours");
  set_environment(kSwitch, "vendored");
  check(switch_for_load() == Switch::Vendored, "IGNIS_HQ_DECODE_ATTENTION=vendored");
  for (const char *bad : {"Ours", "ws12", "1", "throughput"}) {
    set_environment(kSwitch, bad);
    bool refused = false;
    try {
      (void)switch_for_load();
    } catch (const std::invalid_argument &e) {
      refused = std::string(e.what()).find(kSwitch) != std::string::npos;
    }
    check(refused, std::string("IGNIS_HQ_DECODE_ATTENTION=") + bad + " is refused by name");
  }
  set_environment(kSwitch, nullptr);
}

// The build's list names `<major><minor>a`.
bool built_for(int major, int minor) {
  const std::string list = std::string(";") + ignis_build_cuda_architectures() + ";";
  return list.find(";" + std::to_string(major) + std::to_string(minor) + "a;") != std::string::npos;
}

void check_probe() {
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    check(false, "a CUDA device");
    return;
  }
  const bool expected = built_for(prop.major, prop.minor);
  const bool image = ignis::hq_decode_attention::image_here();
  std::printf("hq_decode_attention: sm_%d%d, build %s, image %s\n", prop.major, prop.minor,
              ignis_build_cuda_architectures(), image ? "here" : "absent");
  check(image == expected, "the route's image is here exactly when the build names this card's "
                           "arch-specific target");
  check(cudaGetLastError() == cudaSuccess, "the probe leaves no error behind");
}

}  // namespace

int main() {
  check_rule();
  check_switch();
  check_probe();
  if (g_failed != 0) {
    std::fprintf(stderr, "hq_decode_attention_route: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("hq_decode_attention_route: all checks passed\n");
  return EXIT_SUCCESS;
}
