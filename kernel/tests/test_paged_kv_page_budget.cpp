// Leaf-level page-budget query test (P1-16, GitHub #52).
//
// Ours, not vendored: exercises ignis_paged_kv_page_budget against the
// vendored core/paged_kv_cache.h geometry it wraps. Pure host arithmetic —
// no device needed, so (unlike the GPU op tests) a missing/busy GPU is not a
// concern here and this test always runs.

#include "ignis_paged_kv_budget.h"

#include "core/dtype.h"

#include <cstdio>
#include <cstdint>

namespace {

int failures = 0;

void expect(bool ok, const char *label) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

} // namespace

int main() {
    // One BF16 plane: head_dim=128, num_kv_heads=8 -> 128*64*8*2 bytes/page
    // (kPagedKVPageSize is 64, fixed by the vendored header).
    const ignis_paged_kv_plane bf16_plane{
        /*dtype=*/0, /*leading_extent=*/128, /*head_extent=*/8};
    const std::uint64_t expected_unit_bytes =
        static_cast<std::uint64_t>(ninfer::dtype_size(ninfer::DType::BF16)) * 128 * 64 * 8;

    std::uint32_t page_count = 0;
    std::uint64_t page_bytes = 0;
    int32_t rc = ignis_paged_kv_page_budget(&bf16_plane, 1, expected_unit_bytes * 10, &page_count,
                                            &page_bytes);
    expect(rc == 0, "single plane: return code");
    expect(page_count == 10, "single plane: page count");
    expect(page_bytes == expected_unit_bytes * 10, "single plane: page bytes");

    // A budget short of one page reports zero, not a negative or a fraction.
    rc = ignis_paged_kv_page_budget(&bf16_plane, 1, expected_unit_bytes - 1, &page_count,
                                    &page_bytes);
    expect(rc == 0, "short budget: return code");
    expect(page_count == 0, "short budget: page count");
    expect(page_bytes == 0, "short budget: page bytes");

    // Two planes (K and V, say): the budget is split across both in one page.
    const ignis_paged_kv_plane two_planes[2] = {
        {/*dtype=*/0, /*leading_extent=*/128, /*head_extent=*/8},
        {/*dtype=*/0, /*leading_extent=*/128, /*head_extent=*/8},
    };
    rc = ignis_paged_kv_page_budget(two_planes, 2, expected_unit_bytes * 20, &page_count,
                                    &page_bytes);
    expect(rc == 0, "two planes: return code");
    expect(page_count == 10, "two planes: page count");
    expect(page_bytes == expected_unit_bytes * 20, "two planes: page bytes");

    // Bad arguments are rejected, not silently clamped.
    rc = ignis_paged_kv_page_budget(&bf16_plane, 0, expected_unit_bytes, &page_count, &page_bytes);
    expect(rc == -1, "zero plane count is rejected");

    const ignis_paged_kv_plane bad_extent{/*dtype=*/0, /*leading_extent=*/0, /*head_extent=*/8};
    rc = ignis_paged_kv_page_budget(&bad_extent, 1, expected_unit_bytes, &page_count, &page_bytes);
    expect(rc == -1, "non-positive extent is rejected");

    const ignis_paged_kv_plane bad_dtype{/*dtype=*/99, /*leading_extent=*/128, /*head_extent=*/8};
    rc = ignis_paged_kv_page_budget(&bad_dtype, 1, expected_unit_bytes, &page_count, &page_bytes);
    expect(rc == -1, "unrecognized dtype is rejected");

    rc = ignis_paged_kv_page_budget(nullptr, 1, expected_unit_bytes, &page_count, &page_bytes);
    expect(rc == -1, "null planes is rejected");

    if (failures != 0) {
        std::fprintf(stderr, "paged kv page budget: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("paged kv page budget: ok\n");
    return 0;
}
