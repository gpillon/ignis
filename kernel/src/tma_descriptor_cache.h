// ignis kernel leaf: the W4A4 TMA products' descriptor blocks kept on the device, on Windows --
// OURS (GitHub #650, spec #646), leaf-internal.
//
// MSVC cannot pass the products' alignas(128) block of four tensor maps by value (C2719), so on
// Windows the vendored launchers (kernel/vendor/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu, whose
// `launch_tma` every W4A4 TMA product shares, and .../linear_swiglu/nvfp4/
// nvfp4_linear_swiglu_w4a4_tma.cu) hand the kernel a pointer to a device copy of it. Upstream
// allocates that copy from the device's memory pool, copies the 512 bytes into it from pageable
// memory and frees it, before every product: 247 times a chunk of the 27B's prefill, a stall of
// about 1 ms at every chunk boundary while the pool trims what they left, and about 110 ms for
// the pool's first allocation, paid by the first request after a load. Their recorded patch
// (ADR 0031) asks `device_copy` first and keeps upstream's upload where it answers null. Since
// GitHub #651 the 27B's MLP gate and up product runs ignis's fused op (kernel/src/prefill_swiglu.cu)
// at those widths, whose launcher asks the same way: one block a call, as the product it stands in
// for built.
//
// A load's cache is a block of slots it allocates at load -- the VRAM plan's `tma_descriptors`
// line -- serving the launches on the load's stream. A tensor map names only addresses, extents
// and strides, so the same bytes are the same descriptor: a block is copied into a slot the first
// time a launch builds it, ordered on the stream before that launch, and every later launch that
// builds the same bytes reads the slot. When every slot holds a block, the one least recently
// read is rewritten. A slot is only ever rewritten by a copy enqueued on the stream its readers
// launch on, so the copy runs after every kernel already enqueued there that reads it: the rule
// upstream's own pool block relies on (its allocation and its free both ordered on the consuming
// stream), since a map rewritten under a kernel still reading it hangs that kernel's TMA wait.
//
// The plan sizes the slots for a traversal's products at every width the TMA route takes
// (`planned_slots`): a text chunk's blocks are all resident at once, and stay so from one chunk
// to the next. A chunk whose activations sit elsewhere in the scratch (a multimodal chunk's
// scatter indices, the drafter's taps from a later column, an attention readout's room, all
// allocated before the layers) builds blocks of its own, which take the least recently read
// slots and are copied again when read again: a copy, never an allocation.
//
// `device_copy` answers null, and the launch keeps upstream's per-launch upload, on a stream no
// load's cache serves, inside a graph capture (a capture bakes its copy's source; the cache's
// lazy copy would be no node of the graph), and for a block wider than a slot. Linux passes the
// descriptors by value: its launchers never ask, and no load there plans a slot
// (`kThroughPointer`).
#pragma once

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace ignis::tma_descriptors {

// One slot: a W4A4 TMA product's four 128-byte tensor maps (`Nvfp4W4a4TmaDescriptors`), at the
// 128-byte alignment the block declares (slots sit at multiples of 512 from a 256-byte-aligned
// allocation).
inline constexpr std::size_t kSlotBytes = 512;

// Whether this platform's launchers take their descriptors through a pointer, the only case a
// load plans slots for: Windows.
#ifdef _WIN32
inline constexpr bool kThroughPointer = true;
#else
inline constexpr bool kThroughPointer = false;
#endif

// The measurement switch IGNIS_TMA_DESCRIPTOR_CACHE in the process environment
// (kernel/src/process_environment.h), read once per load: unset, empty or `1`, the cache; `0`,
// upstream's per-launch upload. Throws std::invalid_argument, naming the variable, on any other
// value.
bool enabled_for_load();

// The slots a load plans for `products` W4A4 TMA products a prefill traversal runs, with chunks
// up to `prefill_chunk_tokens` wide: one per product at every width the vendored dispatch sends
// to the TMA route (a multiple of 256 from 1,024 tokens), so the blocks of one chunk of every
// width are resident together. 0 below 1,024 tokens.
std::size_t planned_slots(std::size_t products, std::uint32_t prefill_chunk_tokens);

// A load's slots, serving the launches on its stream from construction to destruction.
class Cache {
public:
  // `slots` slots in device memory, serving `stream`. Throws when the allocation fails, when
  // `slots` is 0, or when another cache already serves `stream`.
  Cache(std::size_t slots, cudaStream_t stream);
  ~Cache();
  Cache(const Cache &) = delete;
  Cache &operator=(const Cache &) = delete;

  // The device bytes the slots hold: the plan line.
  std::size_t bytes() const { return buffer_.bytes; }

  struct Counts {
    // Slots holding a block.
    std::size_t resident = 0;
    // Copies into a slot since construction: one per block read while not resident.
    std::uint64_t uploads = 0;
  };
  Counts counts() const;

  // The device copy of `block` for a launch on this cache's stream, copied into a slot first when
  // it is not resident; null when the copy cannot be enqueued (the error consumed). Called with
  // the registry's lock held (`device_copy`).
  const void *slot_for(const void *block, std::size_t size);

private:
  ninfer::DeviceBuffer buffer_;
  cudaStream_t stream_ = nullptr;
  std::size_t slots_ = 0;
  // Each resident block's bytes and its place in `recency`, most recently read first.
  std::unordered_map<std::string, std::list<std::size_t>::iterator> where_;
  std::list<std::size_t> recency_;
  // What each slot holds, so a rewritten slot's old block leaves `where_`.
  std::vector<std::string> held_;
  std::size_t next_unused_ = 0;
  std::uint64_t uploads_ = 0;
};

// What the vendored launchers ask before upstream's per-launch upload: the device copy of the
// `size` bytes at `block` for a launch on `stream`, or null -- no cache serves `stream`, the
// stream is capturing a graph, `size` is past a slot, or the copy failed.
const void *device_copy(const void *block, std::size_t size, cudaStream_t stream);

} // namespace ignis::tma_descriptors
