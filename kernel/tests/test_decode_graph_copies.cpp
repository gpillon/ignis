// GitHub #677 (spec #672): which of the 27B's decode rounds run copy-free, as each capture of the
// decode graphs decides it (kernel/src/decode_graph_copies.h) -- OURS.
//
// The rule, with the card's generation injected: on a card of compute capability 12 (the RTX 5090,
// the RTX PRO 6000) the width-1 decode round reads its residual and the GDN q, k and v in place,
// unless the measurement switch IGNIS_DECODE_GRAPH_COPIES=1 names main's copying round; an RTX 30
// or 40 card (Ampere 8.6, Ada 8.9) and every other generation keep main's whatever the switch says,
// and widths 2-8 keep main's everywhere. Then the switch as a capture reads it from the operating
// system's environment: unset or `1`, any other value refused naming the variable.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE -- but nothing here needs a GPU.

#include "decode_graph_copies.h"

#include "ignis_step.h"

#include <cstdint>
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

using ignis::decode_graph_copies::Switch;

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
  using ignis::decode_graph_copies::copy_free;
  using ignis::decode_graph_copies::copy_free_widths;
  check(copy_free(Switch::Unset, 12, 1), "compute capability 12 runs the width-1 round copy-free by default");
  check(!copy_free(Switch::Copies, 12, 1), "IGNIS_DECODE_GRAPH_COPIES=1 keeps main's width-1 round on Blackwell");
  check(copy_free_widths(Switch::Unset, 12) == 0x1u, "on Blackwell, width 1 alone is copy-free");
  check(copy_free_widths(Switch::Copies, 12) == 0u, "under the switch no width is copy-free");
  for (const int major : {7, 8, 9, 10, 11, 13}) {
    for (const Switch named : {Switch::Unset, Switch::Copies}) {
      check(copy_free_widths(named, major) == 0u,
            "a card of compute capability " + std::to_string(major) + ".x keeps main's round at every width");
    }
  }
  for (std::uint32_t width = 2; width <= IGNIS_DECODE_MAX_BATCH; ++width) {
    for (const Switch named : {Switch::Unset, Switch::Copies}) {
      check(!copy_free(named, 12, width), "width " + std::to_string(width) + " keeps main's round on Blackwell");
    }
  }
}

void check_switch() {
  constexpr const char *kSwitch = "IGNIS_DECODE_GRAPH_COPIES";
  using ignis::decode_graph_copies::switch_at_capture;
  set_environment(kSwitch, nullptr);
  check(switch_at_capture() == Switch::Unset, "unset, the card decides");
  set_environment(kSwitch, "");
  check(switch_at_capture() == Switch::Unset, "empty, the card decides");
  set_environment(kSwitch, "1");
  check(switch_at_capture() == Switch::Copies, "IGNIS_DECODE_GRAPH_COPIES=1, main's round");
  for (const char *bad : {"0", "true", "copies"}) {
    set_environment(kSwitch, bad);
    bool refused = false;
    try {
      (void)switch_at_capture();
    } catch (const std::invalid_argument &e) {
      refused = std::string(e.what()).find(kSwitch) != std::string::npos;
    }
    check(refused, std::string("IGNIS_DECODE_GRAPH_COPIES=") + bad + " is refused by name");
  }
  set_environment(kSwitch, nullptr);
}

}  // namespace

int main() {
  check_rule();
  check_switch();
  if (g_failed != 0) {
    std::fprintf(stderr, "decode_graph_copies: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("decode_graph_copies: all checks passed\n");
  return EXIT_SUCCESS;
}
