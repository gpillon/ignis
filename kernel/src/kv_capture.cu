// ignis kernel leaf: paged-KV row readback -- test-only diagnostic seam
// (GitHub #119). See kernel/include/ignis_kv_capture.h for what this is and
// is not, why it exists, and why it compiling into ignis_kernel.lib (this
// file matches kernel/CMakeLists.txt's `src/*.cu` glob) does not make it
// ship: nothing references its symbols unless crates/core's `kv-capture`
// cargo feature is on, and that feature is never enabled in production.
//
// Style follows seq.cu: explicit pointers + sizes, int32 return codes (0 =
// ok, -1 = error via ignis_kv_capture_last_error), no C++ types across the
// boundary, validate everything before touching the GPU. Each validation
// failure below names the specific thing that did not hold and the value
// found -- a shared "geometry mismatch" wording across unrelated failures
// costs a caller time exactly when precision is worth the most.

#include "ignis_kv_capture.h"
#include "ignis_seq_internal.h"

#include <cuda_runtime.h>

#include <string>

namespace {

thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

} // namespace

extern "C" int32_t ignis_kv_capture_rows(const struct ignis_seq_pool *pool,
                                         const struct ignis_seq *seq, int32_t gqa_layer_ordinal,
                                         int32_t role, int32_t kv_head, int32_t first_position,
                                         int32_t row_count, int32_t head_dim,
                                         uint16_t *out_rows) {
  if (pool == nullptr || seq == nullptr || out_rows == nullptr) {
    set_error("ignis_kv_capture_rows: null argument");
    return -1;
  }
  if (gqa_layer_ordinal < 0 || gqa_layer_ordinal >= kIgnisGqaLayerCount) {
    set_error("ignis_kv_capture_rows: gqa_layer_ordinal out of range [0, kIgnisGqaLayerCount)");
    return -1;
  }
  if (role != 0 && role != 1) {
    set_error("ignis_kv_capture_rows: role must be 0 (K) or 1 (V)");
    return -1;
  }
  if (row_count <= 0) {
    set_error("ignis_kv_capture_rows: row_count must be positive");
    return -1;
  }
  if (first_position < 0) {
    set_error("ignis_kv_capture_rows: first_position must be non-negative");
    return -1;
  }
  if (head_dim <= 0) {
    set_error("ignis_kv_capture_rows: head_dim must be positive");
    return -1;
  }

  // This seam reads BF16 value rows, which only a BF16 pool holds: an hq
  // pool's planes are the codec's code and metadata bytes, and the row
  // shape below does not describe them. Refused by name here rather than
  // left to the `dtype != BF16` check further down, which would report a
  // plane-level surprise instead of the actual cause (GitHub #122).
  if (pool->kv_format != IGNIS_KV_FORMAT_BF16) {
    set_error("ignis_kv_capture_rows: this pool stores KV in format " +
             std::to_string(pool->kv_format) +
             ", which holds no BF16 value rows -- this seam reads BF16 rows only");
    return -1;
  }
  // Plane index comes from the pool's own layout helper
  // (kernel/include/ignis_seq_internal.h), never re-derived here.
  const std::size_t plane_index = ignis_kv_plane_index(
      pool->kv_format, gqa_layer_ordinal, role == 0 ? IGNIS_KV_PLANE_K : IGNIS_KV_PLANE_V);
  const std::size_t plane_count = pool->kv_pool.plane_count();
  if (plane_index >= plane_count) {
    set_error("ignis_kv_capture_rows: plane_index " + std::to_string(plane_index) +
             " (from gqa_layer_ordinal " + std::to_string(gqa_layer_ordinal) + ", role " +
             std::to_string(role) + ") is out of range -- the pool has only " +
             std::to_string(plane_count) + " planes");
    return -1;
  }
  const ninfer::Tensor &plane = pool->kv_pool.plane(plane_index);
  // PageMajor plane shape is {leading_extent, kPagedKVPageSize, head_extent,
  // physical_pages} (plan_paged_kv_pool, kernel/vendor/src/core/
  // paged_kv_cache.cpp): ne[0] is head_dim, ne[1] is the *page* extent (64),
  // NOT the KV head count, ne[2] is the real KV head count. Every one of
  // these is validated explicitly rather than assumed -- a plane that is not
  // laid out this way (e.g. HeadMajor) must fail loudly here, not read from
  // the wrong offset silently. Each failure names the specific thing that
  // did not hold and the value found, rather than sharing one vague
  // "geometry mismatch" message across all of them.
  if (plane.dtype != ninfer::DType::BF16) {
    set_error("ignis_kv_capture_rows: plane " + std::to_string(plane_index) + " has dtype " +
             std::to_string(static_cast<int>(plane.dtype)) +
             " (ninfer::DType), expected BF16 (0) -- this pool was not built with BF16 KV "
             "planes at this index");
    return -1;
  }
  if (plane.ne[0] != head_dim) {
    set_error("ignis_kv_capture_rows: head_dim argument " + std::to_string(head_dim) +
             " does not match plane " + std::to_string(plane_index) + "'s actual ne[0]=" +
             std::to_string(plane.ne[0]));
    return -1;
  }
  if (plane.ne[1] != ninfer::kPagedKVPageSize) {
    set_error("ignis_kv_capture_rows: plane " + std::to_string(plane_index) + "'s ne[1]=" +
             std::to_string(plane.ne[1]) + " is not kPagedKVPageSize (" +
             std::to_string(ninfer::kPagedKVPageSize) +
             ") -- this plane is not PageMajor, the stride-based addressing below does not "
             "apply to it");
    return -1;
  }
  const std::int32_t num_kv_heads = plane.ne[2];
  if (kv_head < 0 || kv_head >= num_kv_heads) {
    set_error("ignis_kv_capture_rows: kv_head " + std::to_string(kv_head) +
             " is out of range [0, " + std::to_string(num_kv_heads) +
             ") -- plane " + std::to_string(plane_index) + "'s actual KV head count (ne[2])");
    return -1;
  }

  // Rows must already be real: never read past what this sequence has
  // actually written for this GQA layer (kernel/src/gqa_layer.cu advances
  // gqa_positions[gqa_layer_ordinal] past every token it appends).
  const std::uint64_t written = seq->gqa_positions[static_cast<std::size_t>(gqa_layer_ordinal)];
  const std::uint64_t last_position = static_cast<std::uint64_t>(first_position) +
                                      static_cast<std::uint64_t>(row_count);
  if (last_position > written) {
    set_error(
        "ignis_kv_capture_rows: [first_position, first_position + row_count) exceeds what this "
        "sequence has written for this GQA layer");
    return -1;
  }

  const auto page_ids = seq->kv.page_ids();
  const auto *byte_base = static_cast<const unsigned char *>(plane.data);
  for (int32_t i = 0; i < row_count; ++i) {
    const std::int32_t position = first_position + i;
    const std::size_t logical_page =
        static_cast<std::size_t>(position) / static_cast<std::size_t>(ninfer::kPagedKVPageSize);
    if (logical_page >= page_ids.size()) {
      set_error("ignis_kv_capture_rows: position maps past this sequence's mapped pages");
      return -1;
    }
    const std::int32_t physical_page = page_ids[logical_page];
    const std::int32_t page_offset = position % ninfer::kPagedKVPageSize;
    // The row's byte offset comes from the plane tensor's own strides, not
    // re-derived shape arithmetic: nb[1]/nb[2]/nb[3] are the byte strides of
    // the page-offset, kv_head and physical-page dimensions respectively
    // (ninfer::set_contiguous_strides -- standard row-major over the
    // PageMajor {head_dim, page, kv_head, physical_page} shape validated
    // above). This cannot drift if the pool's plane order or extents ever
    // change, because it never assumes a stride, only the dimension each one
    // belongs to.
    const std::int64_t byte_offset = static_cast<std::int64_t>(page_offset) * plane.nb[1] +
                                     static_cast<std::int64_t>(kv_head) * plane.nb[2] +
                                     static_cast<std::int64_t>(physical_page) * plane.nb[3];
    const auto *row_ptr = reinterpret_cast<const uint16_t *>(byte_base + byte_offset);

    const cudaError_t err =
        cudaMemcpy(out_rows + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_dim),
                  row_ptr, static_cast<std::size_t>(head_dim) * sizeof(uint16_t),
                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_kv_capture_rows: cudaMemcpy failed: ") +
               cudaGetErrorString(err));
      return -1;
    }
  }
  return 0;
}

extern "C" const char *ignis_kv_capture_last_error(void) {
  return g_last_error.c_str();
}
