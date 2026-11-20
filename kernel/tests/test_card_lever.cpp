// Spec #684 (GitHub #693, #694): where a cost lever measured on the RTX 5090 alone runs, as each
// lever's measurement switch and the card decide it (kernel/src/card_lever.h) -- OURS.
//
// The rule, with the card's generation injected: unset, the lever on a card of compute capability
// 12 (the RTX 5090, where the levers were measured) and today's path on an RTX 30 or 40 card
// (Ampere 8.6, Ada 8.9), on every other generation and on a device that cannot say (0); `1` the
// lever on every card; `0` today's path on every card. Then each lever's switch as a call reads it
// from the operating system's environment: unset, empty, `1` or `0`, any other value refused
// naming the variable.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE -- but nothing here needs a GPU.

#include "card_lever.h"

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

using ignis::card_lever::Switch;

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
  using ignis::card_lever::takes;
  struct Row {
    Switch named;
    int major;
    bool takes;
    const char *what;
  };
  const Row rows[] = {
      {Switch::Unset, 12, true, "unset, compute capability 12 runs the lever"},
      {Switch::Unset, 8, false, "unset, an RTX 30 or 40 card (8.x) keeps today's path"},
      {Switch::Unset, 7, false, "unset, 7.x keeps today's path"},
      {Switch::Unset, 9, false, "unset, 9.x keeps today's path"},
      {Switch::Unset, 10, false, "unset, 10.x keeps today's path"},
      {Switch::Unset, 11, false, "unset, 11.x keeps today's path"},
      {Switch::Unset, 13, false, "unset, 13.x keeps today's path"},
      {Switch::Unset, 0, false, "unset, a device that cannot say its generation keeps today's path"},
      {Switch::On, 12, true, "`1` runs the lever on compute capability 12"},
      {Switch::On, 8, true, "`1` runs the lever on an RTX 30 or 40 card, a measurement"},
      {Switch::On, 0, true, "`1` runs the lever whatever the device says"},
      {Switch::Off, 12, false, "`0` keeps today's path on compute capability 12"},
      {Switch::Off, 8, false, "`0` keeps today's path on an RTX 30 or 40 card"},
  };
  for (const Row &row : rows) {
    check(takes(row.named, row.major) == row.takes, row.what);
  }
}

void check_switch(const char *name) {
  using ignis::card_lever::named;
  set_environment(name, nullptr);
  check(named(name) == Switch::Unset, std::string(name) + " unset: the card decides");
  set_environment(name, "");
  check(named(name) == Switch::Unset, std::string(name) + " empty: the card decides");
  set_environment(name, "1");
  check(named(name) == Switch::On, std::string(name) + "=1: the lever on any card");
  set_environment(name, "0");
  check(named(name) == Switch::Off, std::string(name) + "=0: today's path on every card");
  for (const char *bad : {"2", "true", "on", " 1"}) {
    set_environment(name, bad);
    bool refused = false;
    try {
      (void)named(name);
    } catch (const std::invalid_argument &e) {
      refused = std::string(e.what()).find(name) != std::string::npos;
    }
    check(refused, std::string(name) + "=" + bad + " is refused by name");
  }
  set_environment(name, nullptr);
}

}  // namespace

int main() {
  check_rule();
  for (const char *name : {"IGNIS_FP8_W8A8_TILE16", "IGNIS_DFLASH2_APPEND_KV_ONLY", "IGNIS_DFLASH2_SMALL_T_FILL"}) {
    check_switch(name);
  }
  if (g_failed != 0) {
    std::fprintf(stderr, "card_lever: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("card_lever: all checks passed\n");
  return EXIT_SUCCESS;
}
