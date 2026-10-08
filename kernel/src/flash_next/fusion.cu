// ignis kernel leaf -- the switches of GitHub #306's decode fusions (OURS, ADR 0043). See fusion.h.

#include "fusion.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace ignis::flash_next {

namespace {

constexpr int32_t kCount = static_cast<int32_t>(Fusion::kCount);

constexpr const char *kEnv[kCount] = {"IGNIS_FN_HC_FUSED",  "IGNIS_FN_INJECT_FUSED", "IGNIS_FN_SCORE_FUSED",
                                      "IGNIS_FN_GDN_FUSED", "IGNIS_FN_ROUTE_FUSED",  "IGNIS_FN_QSA_FUSED",
                                      "IGNIS_FN_STAGING_FUSED"};

// -1 until the first read seeds it from the environment.
std::atomic<int> g_on[kCount] = {-1, -1, -1, -1, -1, -1, -1};

bool off(const char *name) {
  const char *env = std::getenv(name);
  return env != nullptr && std::strcmp(env, "0") == 0;
}

}  // namespace

bool fused(Fusion f) {
  const auto i = static_cast<int32_t>(f);
  int on = g_on[i].load(std::memory_order_relaxed);
  if (on < 0) {
    const int seeded = off(kEnv[i]) || off("IGNIS_FN_FUSION") ? 0 : 1;
    // A switch set meanwhile (set_fused) wins over the environment.
    on = g_on[i].compare_exchange_strong(on, seeded, std::memory_order_relaxed) ? seeded : on;
  }
  return on == 1;
}

void set_fused(Fusion f, bool on) { g_on[static_cast<int32_t>(f)].store(on ? 1 : 0, std::memory_order_relaxed); }

}  // namespace ignis::flash_next
