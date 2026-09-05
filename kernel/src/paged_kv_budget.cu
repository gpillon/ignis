// ignis kernel leaf: paged KV page-budget query (see ignis_paged_kv_budget.h).
//
// Host-only: routes through the vendored reference's own plan_paged_kv_pool
// (core/paged_kv_cache.h, ADR 0010) rather than re-deriving the per-page byte
// formula, so a change to the reference's layout math is caught here instead
// of silently drifting from what the pool itself will allocate.

#include "ignis_paged_kv_budget.h"

#include "core/dtype.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace {

bool decode_dtype(int32_t code, ninfer::DType *out) {
    switch (code) {
        case 0: *out = ninfer::DType::BF16; return true;
        case 1: *out = ninfer::DType::FP32; return true;
        case 2: *out = ninfer::DType::I32; return true;
        case 3: *out = ninfer::DType::U8; return true;
        case 4: *out = ninfer::DType::I64; return true;
        case 5: *out = ninfer::DType::I8; return true;
        case 6: *out = ninfer::DType::FP16; return true;
        case 7: *out = ninfer::DType::FP8_E4M3FN; return true;
        default: return false;
    }
}

} // namespace

extern "C" int32_t ignis_paged_kv_page_budget(const struct ignis_paged_kv_plane *planes,
                                               int32_t plane_count, uint64_t vram_budget_bytes,
                                               uint32_t *out_page_count,
                                               uint64_t *out_page_bytes) {
    if (planes == nullptr || out_page_count == nullptr || out_page_bytes == nullptr ||
        plane_count <= 0) {
        return -1;
    }

    ninfer::PagedKVPoolSpec spec;
    spec.page_group_count      = 1;
    spec.logical_page_capacity = 1;
    spec.table_rows            = 1;
    spec.plane_order           = ninfer::PagedKVPlaneOrder::PageMajor;
    spec.planes.reserve(static_cast<std::size_t>(plane_count));
    for (int32_t i = 0; i < plane_count; ++i) {
        ninfer::DType dtype{};
        if (!decode_dtype(planes[i].dtype, &dtype)) { return -1; }
        if (planes[i].leading_extent <= 0 || planes[i].head_extent <= 0) { return -1; }
        ninfer::PagedKVPlaneSpec plane;
        plane.dtype          = dtype;
        plane.leading_extent = planes[i].leading_extent;
        plane.head_extent    = planes[i].head_extent;
        spec.planes.push_back(plane);
    }

    ninfer::LayoutBuilder builder;
    ninfer::PagedKVPoolLayout layout;
    try {
        layout = ninfer::plan_paged_kv_pool(builder, spec);
    } catch (const std::exception &) { return -1; }

    // plan_paged_kv_pool's per-plane storage is exactly
    // dtype_size * leading_extent * kPagedKVPageSize * head_extent *
    // page_group_count (LayoutRegion::bytes carries no page-count-dependent
    // padding), so one page group's payload scales linearly: no per-count
    // re-planning is needed to size a budget.
    const std::uint64_t unit_bytes = layout.payload_bytes();
    if (unit_bytes == 0) {
        *out_page_count = 0;
        *out_page_bytes = 0;
        return 0;
    }

    std::uint64_t page_count = vram_budget_bytes / unit_bytes;
    const std::uint64_t max_pages =
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    if (page_count > max_pages) { page_count = max_pages; }

    *out_page_count = static_cast<std::uint32_t>(page_count);
    *out_page_bytes = page_count * unit_bytes;
    return 0;
}
