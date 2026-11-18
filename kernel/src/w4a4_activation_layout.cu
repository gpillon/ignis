// ignis kernel leaf: the layout the 27B's W4A4 prefill products read their activations in -- OURS
// (GitHub #652, spec #646). See w4a4_activation_layout.h.
#include "w4a4_activation_layout.h"

#include "process_environment.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ignis::w4a4_activation_layout {
namespace {

constexpr const char *kSwitch = "IGNIS_W4A4_ACTIVATION_LAYOUT";

// The layout each load names for its stream. Never destroyed: a launch may run while the process
// exits.
struct Registry {
  std::mutex mutex;
  std::unordered_map<cudaStream_t, Layout> streams;
};

Registry &registry() {
  static Registry *r = new Registry();
  return *r;
}

} // namespace

Layout layout_for_load() {
  return ignis::named_switch(kSwitch, Layout::KTile, {{"ktile", Layout::KTile}, {"token", Layout::Token}});
}

ForStream::ForStream(cudaStream_t stream, Layout layout) : stream_(stream) {
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (!r.streams.emplace(stream, layout).second) {
    throw std::invalid_argument("a W4A4 activation layout is already named for this stream");
  }
}

ForStream::~ForStream() {
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  r.streams.erase(stream_);
}

bool k_tile_major(cudaStream_t stream) {
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  const auto found = r.streams.find(stream);
  return found == r.streams.end() || found->second == Layout::KTile;
}

} // namespace ignis::w4a4_activation_layout
