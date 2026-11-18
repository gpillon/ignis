// ignis kernel leaf: the W4A4 TMA products' descriptor blocks kept on the device -- OURS
// (GitHub #650, spec #646). See tma_descriptor_cache.h.
#include "tma_descriptor_cache.h"

#include "process_environment.h"
#include "w4a4_tma_route.h"

#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ignis::tma_descriptors {
namespace {

constexpr const char *kSwitch = "IGNIS_TMA_DESCRIPTOR_CACHE";

// Which load's cache serves a stream. Never destroyed: a launch may run while the process exits.
struct Registry {
  std::mutex mutex;
  std::unordered_map<cudaStream_t, Cache *> caches;
};

Registry &registry() {
  static Registry *r = new Registry();
  return *r;
}

} // namespace

bool enabled_for_load() {
  return ignis::named_switch(kSwitch, true, {{"0", false}, {"1", true}});
}

std::size_t planned_slots(std::size_t products, std::uint32_t prefill_chunk_tokens) {
  return products * ignis::w4a4_tma::widths_up_to(prefill_chunk_tokens);
}

Cache::Cache(std::size_t slots, cudaStream_t stream)
    : buffer_(slots * kSlotBytes), stream_(stream), slots_(slots), held_(slots) {
  if (slots == 0) {
    throw std::invalid_argument("a TMA descriptor cache needs at least one slot");
  }
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (!r.caches.emplace(stream, this).second) {
    throw std::invalid_argument("a TMA descriptor cache already serves this stream");
  }
}

Cache::~Cache() {
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  const auto found = r.caches.find(stream_);
  if (found != r.caches.end() && found->second == this) {
    r.caches.erase(found);
  }
}

Cache::Counts Cache::counts() const {
  std::lock_guard<std::mutex> lock(registry().mutex);
  return Counts{where_.size(), uploads_};
}

const void *Cache::slot_for(const void *block, std::size_t size) {
  std::string key(static_cast<const char *>(block), size);
  const auto found = where_.find(key);
  if (found != where_.end()) {
    recency_.splice(recency_.begin(), recency_, found->second);
    return static_cast<const unsigned char *>(buffer_.p) + *found->second * kSlotBytes;
  }
  // A slot never used, else the one least recently read: its copy below is enqueued on the
  // stream every kernel that read it was, so it runs after them.
  std::size_t slot = 0;
  if (next_unused_ < slots_) {
    slot = next_unused_++;
    recency_.push_front(slot);
  } else {
    slot = recency_.back();
    where_.erase(held_[slot]);
    held_[slot].clear();
    recency_.splice(recency_.begin(), recency_, std::prev(recency_.end()));
  }
  void *device = static_cast<unsigned char *>(buffer_.p) + slot * kSlotBytes;
  // From the slot's own copy of the bytes, which outlives the call: pageable memory is staged
  // before the call returns, so the launcher's block may go out of scope at once.
  held_[slot] = std::move(key);
  if (cudaMemcpyAsync(device, held_[slot].data(), size, cudaMemcpyHostToDevice, stream_) !=
      cudaSuccess) {
    (void)cudaGetLastError();
    held_[slot].clear();
    // The slot holds nothing readable now: least recently read, so the next miss takes it.
    recency_.splice(recency_.end(), recency_, recency_.begin());
    return nullptr;
  }
  ++uploads_;
  where_.emplace(held_[slot], recency_.begin());
  return device;
}

const void *device_copy(const void *block, std::size_t size, cudaStream_t stream) {
  if (block == nullptr || size == 0 || size > kSlotBytes) return nullptr;
  Registry &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  const auto found = r.caches.find(stream);
  if (found == r.caches.end()) return nullptr;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) {
    (void)cudaGetLastError();
    return nullptr;
  }
  if (capture != cudaStreamCaptureStatusNone) return nullptr;
  return found->second->slot_for(block, size);
}

} // namespace ignis::tma_descriptors
