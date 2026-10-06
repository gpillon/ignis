// ignis kernel leaf: Flash-Next residual-stack tap -- test-only diagnostic seam (spec
// flash-next/07 phase A). See kernel/include/ignis_fn_residual_tap.h for what it captures and
// why compiling into ignis_kernel does not make it ship.
//
// Style follows kv_capture.cu and attn_tap.cu: explicit pointers and sizes, int32 return codes
// (0 ok, -1 error via ignis_fn_residual_tap_last_error), validate before touching the GPU, and
// name the value that failed.

#include "ignis_fn_residual_tap.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>

namespace {

thread_local std::string g_last_error;

void set_error(std::string message) { g_last_error = std::move(message); }

struct Tap {
  std::atomic<bool> armed{false};
  std::uint16_t *out = nullptr;
  std::int64_t first = 0;
  std::int64_t max_rows = 0;
  std::int32_t row_elems = 0;
  std::int64_t written = 0;
};

Tap g_tap;

}  // namespace

extern "C" int32_t ignis_fn_residual_tap_arm(uint16_t *out_rows, int64_t first_position, int64_t max_rows,
                                             int32_t row_elems) {
  if (out_rows == nullptr) {
    set_error("ignis_fn_residual_tap_arm: null out_rows");
    return -1;
  }
  if (first_position < 0 || max_rows <= 0 || row_elems <= 0) {
    set_error("ignis_fn_residual_tap_arm: first_position " + std::to_string(first_position) + ", max_rows " +
              std::to_string(max_rows) + ", row_elems " + std::to_string(row_elems) +
              ": the first must be >= 0 and the others > 0");
    return -1;
  }
  if (g_tap.armed.load()) {
    set_error("ignis_fn_residual_tap_arm: already armed (one arm per process)");
    return -1;
  }
  g_tap.out = out_rows;
  g_tap.first = first_position;
  g_tap.max_rows = max_rows;
  g_tap.row_elems = row_elems;
  g_tap.written = 0;
  g_tap.armed.store(true);
  return 0;
}

extern "C" int32_t ignis_fn_residual_tap_disarm(int64_t *rows_written) {
  if (rows_written != nullptr) *rows_written = g_tap.armed.load() ? g_tap.written : 0;
  g_tap.armed.store(false);
  g_tap.out = nullptr;
  g_tap.written = 0;
  return 0;
}

extern "C" const char *ignis_fn_residual_tap_last_error(void) { return g_last_error.c_str(); }

int32_t ignis_fn_residual_tap_record(int64_t position, int32_t rows, int32_t row_elems, const void *device_rows,
                                     cudaStream_t stream) {
  if (!g_tap.armed.load()) return 0;
  if (row_elems != g_tap.row_elems) {
    set_error("the residual tap was armed for rows of " + std::to_string(g_tap.row_elems) +
              " elements, the chunk's rows have " + std::to_string(row_elems));
    return -1;
  }
  const std::int64_t begin = std::max<std::int64_t>(position, g_tap.first);
  const std::int64_t end = std::min<std::int64_t>(position + rows, g_tap.first + g_tap.max_rows);
  if (begin >= end) return 0;
  const auto row_bytes = static_cast<std::size_t>(row_elems) * sizeof(std::uint16_t);
  const cudaError_t err = cudaMemcpyAsync(
      g_tap.out + (begin - g_tap.first) * row_elems,
      static_cast<const unsigned char *>(device_rows) + static_cast<std::size_t>(begin - position) * row_bytes,
      static_cast<std::size_t>(end - begin) * row_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    set_error(std::string("the residual tap's copy: ") + cudaGetErrorString(err));
    return -1;
  }
  g_tap.written += end - begin;
  return 0;
}
