// ignis kernel leaf: which NVML handle answers for a CUDA device on a MIG-enabled card -- OURS,
// leaf-internal (kernel/src/device.cu).
//
// The leaf asks NVML for a device's memory, name and processes without creating a CUDA context
// (GitHub #210), through the handle of the device's PCI bus id. On a card with MIG enabled (A100,
// A30, H100 and later data-center cards) that bus id names the physical card: its memory is every
// instance's, not the one the process runs on, and an unprivileged process may not read it at all.
// The instance's own handle answers instead. CUDA reports the instance's UUID for the device it
// enumerates, and NVML spells a MIG instance's UUID as `MIG-` and the same 16 bytes, so:
//   - the instance whose UUID is the device's;
//   - else, the only instance the process can see (a container is given one: a CUDA runtime that
//     reports the physical card's UUID still lands on it);
//   - else none, and every query reads the physical card as on a card without MIG.
// Host-only and free of NVML's types, so kernel/tests/test_nvml_mig.cpp runs it with no driver.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace ignis::nvml_mig {

// NVML's spelling of a UUID: `prefix` ("GPU-" or "MIG-") and 8-4-4-4-12 lowercase hex digits of
// `bytes`, written to `out` (truncated, always terminated, to `out_bytes`).
inline void format_uuid(const char *prefix, const unsigned char bytes[16], char *out, std::size_t out_bytes) {
  if (out == nullptr || out_bytes == 0) return;
  std::snprintf(out, out_bytes,
                "%s%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", prefix,
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8],
                bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
}

// One MIG instance slot as NVML enumerates them: whether the process can open it, and its UUID
// (empty when it cannot be read).
struct Slot {
  bool open;
  const char *uuid;
};

// The slot that answers for the device whose NVML UUID is `wanted` (null or empty when the CUDA
// runtime could not say), among `count` slots: the open one with that UUID, else the only open
// one; -1 for none.
inline int choose(const char *wanted, const Slot *slots, int count) {
  int only = -1;
  int open = 0;
  for (int i = 0; i < count; ++i) {
    if (!slots[i].open) continue;
    if (wanted != nullptr && wanted[0] != '\0' && slots[i].uuid != nullptr && std::strcmp(slots[i].uuid, wanted) == 0) {
      return i;
    }
    only = i;
    ++open;
  }
  return open == 1 ? only : -1;
}

}  // namespace ignis::nvml_mig
