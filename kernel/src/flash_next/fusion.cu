// ignis kernel leaf -- the switches of GitHub #306's decode fusions (OURS, ADR 0043). See fusion.h.

#include "fusion.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace ignis::flash_next {

namespace {

constexpr int32_t kCount = static_cast<int32_t>(Fusion::kCount);

constexpr const char *kEnv[] = {"IGNIS_FN_HC_FUSED",  "IGNIS_FN_INJECT_FUSED", "IGNIS_FN_SCORE_FUSED",
                                "IGNIS_FN_GDN_FUSED", "IGNIS_FN_ROUTE_FUSED",  "IGNIS_FN_QSA_FUSED",
                                "IGNIS_FN_STAGING_FUSED"};
static_assert(sizeof(kEnv) / sizeof(kEnv[0]) == kCount, "every fusion names its environment variable");

// kUnread (zero-initialized, so a fusion added later starts there too) until the first read
// seeds it from the environment.
constexpr int kUnread = 0, kOff = 1, kOn = 2;
std::atomic<int> g_on[kCount] = {};

bool off(const char *name) {
  const char *env = std::getenv(name);
  return env != nullptr && std::strcmp(env, "0") == 0;
}

}  // namespace

bool fused(Fusion f) {
  const auto i = static_cast<int32_t>(f);
  int on = g_on[i].load(std::memory_order_relaxed);
  if (on == kUnread) {
    const int seeded = off(kEnv[i]) || off("IGNIS_FN_FUSION") ? kOff : kOn;
    // A switch set meanwhile (set_fused) wins over the environment.
    on = g_on[i].compare_exchange_strong(on, seeded, std::memory_order_relaxed) ? seeded : on;
  }
  return on == kOn;
}

void set_fused(Fusion f, bool on) { g_on[static_cast<int32_t>(f)].store(on ? kOn : kOff, std::memory_order_relaxed); }

}  // namespace ignis::flash_next
