#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include "core/device.h"
#include "ninfer/ops/rmsnorm.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

inline constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Bf16GdnGatingScheduleId schedule;
};

constexpr std::array<RouteSpec, 6> k27Routes{{
    {{1, 1}, Bf16GdnGatingScheduleId::GemvPairedRows},
    {{2, 8}, Bf16GdnGatingScheduleId::SmallTSplit10},
    // As token tiles double, halve SplitK. This keeps the cooperative grid near 192 CTAs instead
    // of making T a launch limit. Once the unsplit grid has enough independent work, it also
    // removes the cooperative-residency constraint.
    {{9, 1024}, Bf16GdnGatingScheduleId::MmaCooperativeSplit8},
    {{1025, 2048}, Bf16GdnGatingScheduleId::MmaCooperativeSplit4},
    {{2049, 4096}, Bf16GdnGatingScheduleId::MmaCooperativeSplit2},
    {{4097, kAnyCols}, Bf16GdnGatingScheduleId::MmaUnsplit},
}};

constexpr std::array<RouteSpec, 5> k35Routes{{
    // The same progression keeps the long-range cooperative routes near 256 CTAs.
    {{1, 127}, Bf16GdnGatingScheduleId::MmaCooperativeSplit16},
    {{128, 1024}, Bf16GdnGatingScheduleId::MmaCooperativeSplit8},
    {{1025, 2048}, Bf16GdnGatingScheduleId::MmaCooperativeSplit4},
    {{2049, 4096}, Bf16GdnGatingScheduleId::MmaCooperativeSplit2},
    {{4097, kAnyCols}, Bf16GdnGatingScheduleId::MmaUnsplit},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes,
                                 std::int32_t last) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == last && expected == static_cast<std::int64_t>(last) + 1;
}

static_assert(catalog_is_closed(k27Routes, kAnyCols));
static_assert(catalog_is_closed(k35Routes, kAnyCols));

// Ignis portability patch (ADR 0077, GitHub #602): each model's cooperative split-K schedules,
// largest first. Where a card cannot hold a route's own split resident, the planner takes the
// first smaller one it holds, and the unsplit route only when none fits.
constexpr std::array<Bf16GdnGatingScheduleId, 3> k27Splits{{
    Bf16GdnGatingScheduleId::MmaCooperativeSplit8,
    Bf16GdnGatingScheduleId::MmaCooperativeSplit4,
    Bf16GdnGatingScheduleId::MmaCooperativeSplit2,
}};

constexpr std::array<Bf16GdnGatingScheduleId, 4> k35Splits{{
    Bf16GdnGatingScheduleId::MmaCooperativeSplit16,
    Bf16GdnGatingScheduleId::MmaCooperativeSplit8,
    Bf16GdnGatingScheduleId::MmaCooperativeSplit4,
    Bf16GdnGatingScheduleId::MmaCooperativeSplit2,
}};

// Ignis portability patch (ADR 0077, GitHub #602): the route table read once, for the planner
// and for bf16_gdn_gating_route_schedule.
template <std::size_t N>
Bf16GdnGatingScheduleId routed_schedule(const std::array<RouteSpec, N>& routes,
                                        std::int32_t cols) {
    for (const RouteSpec& route : routes) {
        if (route.cols.contains(cols)) { return route.schedule; }
    }
    throw std::logic_error("BF16 GDN gating: admitted problem has no covering route");
}

bool is_27(const Bf16GdnGatingProblem& problem) noexcept {
    return problem.heads == 48 && problem.input_rows == 5120;
}

bool is_35(const Bf16GdnGatingProblem& problem) noexcept {
    return problem.heads == 32 && problem.input_rows == 2048;
}

bool schedule_uses_mma(Bf16GdnGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return true;
    case Bf16GdnGatingScheduleId::GemvPairedRows:
    case Bf16GdnGatingScheduleId::SmallTSplit10:
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        return false;
    }
    return false;
}

// Ignis portability patch (ADR 0077, GitHub #602): the test-only cap on the SM count, 0 for none
// (bf16_gdn_gating_set_test_multiprocessor_cap).
std::atomic<int> test_multiprocessor_cap{0};

// Ignis portability patch (ADR 0077, GitHub #478): a cooperative launch fails unless the device
// holds its whole grid at once, the kernel's CTAs/SM times the SMs. Both come from the current
// device, read once per device and kernel, since the planner runs on every call.
//
// Ignis portability patch (ADR 0077, GitHub #602): the two factors are kept apart and the
// test-only cap applies to the SMs as they are read, so one process can plan under several caps.
std::int64_t device_resident_ctas(Bf16GdnGatingCooperativeKernel kernel) {
    struct Budget {
        int ctas_per_sm;
        int multiprocessors;
    };
    static std::mutex mutex;
    static std::map<std::pair<int, Bf16GdnGatingCooperativeKernel>, Budget> budgets;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(device, kernel);
    auto found     = budgets.find(key);
    if (found == budgets.end()) {
        int multiprocessors = 0;
        CUDA_CHECK(
            cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
        if (multiprocessors <= 0) {
            throw std::runtime_error("BF16 GDN gating: device reports no multiprocessors");
        }
        found = budgets
                    .emplace(key, Budget{bf16_gdn_gating_cooperative_ctas_per_sm(kernel),
                                         multiprocessors})
                    .first;
    }
    const int cap = test_multiprocessor_cap.load(std::memory_order_relaxed);
    const int multiprocessors =
        cap > 0 ? std::min(cap, found->second.multiprocessors) : found->second.multiprocessors;
    return static_cast<std::int64_t>(found->second.ctas_per_sm) * multiprocessors;
}

std::int32_t mma_tile_cols(const Bf16GdnGatingProblem& problem) noexcept {
    return is_35(problem) ? 64 : 128;
}

std::int32_t schedule_split_k(Bf16GdnGatingScheduleId schedule) {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        return 80;
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        return 10;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        return 32;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
        return 16;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        return 8;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        return 4;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return 2;
    case Bf16GdnGatingScheduleId::GemvPairedRows:
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return 1;
    }
    throw std::logic_error("BF16 GDN gating: unknown schedule");
}

// Ignis portability patch (ADR 0077, GitHub #478): the launched grid's split dimension. Split40
// launches forty splits of two K tiles each; the 80 of schedule_split_k sizes its workspace.
std::int32_t schedule_grid_splits(Bf16GdnGatingScheduleId schedule) {
    return schedule == Bf16GdnGatingScheduleId::MmaCooperativeSplit40 ? 40
                                                                       : schedule_split_k(schedule);
}

bool cooperative_grid_is_resident(Bf16GdnGatingScheduleId schedule, std::int32_t cols,
                                  std::int32_t tile_cols, std::int32_t row_tiles,
                                  std::int64_t resident_ctas) noexcept {
    const std::int64_t column_tiles = (static_cast<std::int64_t>(cols) + tile_cols - 1) / tile_cols;
    const std::int64_t grid_ctas =
        column_tiles * row_tiles * static_cast<std::int64_t>(schedule_grid_splits(schedule));
    return grid_ctas <= resident_ctas;
}

bool cooperative_27_grid_is_resident(Bf16GdnGatingScheduleId schedule, std::int32_t cols) {
    // BN128 uses 40 KiB of dynamic shared memory. Split8 and the fused split40 run 256 threads,
    // split4/2 512. The SM count is device-specific: a desktop RTX 5090 and its Laptop GPU do not
    // have the same cooperative-launch budget.
    //
    // Ignis portability patch (ADR 0077, GitHub #478): the CTAs/SM come from the occupancy query
    // on the kernel. On the registered CUDA 13.1/sm_120a build it gives the budget this planner
    // hardcoded: 2 for split8/4/2 and 1 for split40 (2 against its 80 workspace splits, the same
    // grid limit). An sm_86 build's split4/2 use 72 registers and hold 1. The gating test checks
    // the sm_120a values on a compute capability 12.0 device.
    Bf16GdnGatingCooperativeKernel kernel = Bf16GdnGatingCooperativeKernel::Mma27Split8;
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        kernel = Bf16GdnGatingCooperativeKernel::Norm27Split40;
        break;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        kernel = Bf16GdnGatingCooperativeKernel::Mma27Split8;
        break;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        kernel = Bf16GdnGatingCooperativeKernel::Mma27Split4;
        break;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        kernel = Bf16GdnGatingCooperativeKernel::Mma27Split2;
        break;
    default:
        throw std::logic_error("BF16 GDN gating: not a 27B cooperative schedule");
    }
    return cooperative_grid_is_resident(schedule, cols, 128, 3, device_resident_ctas(kernel));
}

bool cooperative_35_grid_is_resident(Bf16GdnGatingScheduleId schedule, std::int32_t cols) {
    // BN64 uses 24 KiB of dynamic shared memory and two 16-row tiles.
    //
    // Ignis portability patch (ADR 0077, GitHub #478): the CTAs/SM come from the occupancy query
    // on the kernel. On the registered CUDA 13.1/sm_120a build it gives the budget this planner
    // hardcoded: 2 for split32 (91/93 registers per thread) and 4 for split16/8/4/2 (at most 62).
    // An sm_86 build's split8/4/2 use 72 registers and hold 3. Split32 is both the plain schedule
    // and the control of the fused norm/control route, so it takes the smaller of the two
    // kernels' budgets. The gating test checks the sm_120a values on a compute capability 12.0
    // device.
    std::int64_t resident_ctas = 0;
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        resident_ctas =
            std::min(device_resident_ctas(Bf16GdnGatingCooperativeKernel::Mma35Split32),
                     device_resident_ctas(Bf16GdnGatingCooperativeKernel::Norm35Split32));
        break;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
        resident_ctas = device_resident_ctas(Bf16GdnGatingCooperativeKernel::Mma35Split16);
        break;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        resident_ctas = device_resident_ctas(Bf16GdnGatingCooperativeKernel::Mma35Split8);
        break;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        resident_ctas = device_resident_ctas(Bf16GdnGatingCooperativeKernel::Mma35Split4);
        break;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        resident_ctas = device_resident_ctas(Bf16GdnGatingCooperativeKernel::Mma35Split2);
        break;
    default:
        throw std::logic_error("BF16 GDN gating: not a 35B cooperative schedule");
    }
    return cooperative_grid_is_resident(schedule, cols, 64, 2, resident_ctas);
}

bool candidate_is_legal(Bf16GdnGatingScheduleId schedule,
                        const Bf16GdnGatingProblem& problem) {
    if (!bf16_gdn_gating_admits(problem)) { return false; }
    if (is_27(problem)) {
        switch (schedule) {
        case Bf16GdnGatingScheduleId::GemvPairedRows:
            return problem.cols == 1;
        case Bf16GdnGatingScheduleId::SmallTSplit10:
            return problem.cols >= 2 && problem.cols <= 8;
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
            return cooperative_27_grid_is_resident(schedule, problem.cols);
        case Bf16GdnGatingScheduleId::MmaUnsplit:
            return true;
        case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
            return false;
        }
    }

    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        return false;
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        return problem.cols <= 4 * 65'535;
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        return problem.cols <= 8 * 65'535;
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return true;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return cooperative_35_grid_is_resident(schedule, problem.cols);
    case Bf16GdnGatingScheduleId::GemvPairedRows:
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        return false;
    }
    return false;
}

// Ignis portability patch (ADR 0077, GitHub #602): a split-K schedule, whose grid the card must
// hold resident.
bool schedule_is_cooperative(Bf16GdnGatingScheduleId schedule) noexcept {
    return schedule_uses_mma(schedule) && schedule != Bf16GdnGatingScheduleId::MmaUnsplit;
}

// Ignis portability patch (ADR 0077, GitHub #602): the fallback of a cooperative route whose own
// schedule the card cannot hold resident, never a split larger than the route's own.
template <std::size_t N>
Bf16GdnGatingScheduleId resident_fallback(const std::array<Bf16GdnGatingScheduleId, N>& splits,
                                          Bf16GdnGatingScheduleId routed,
                                          const Bf16GdnGatingProblem& problem) {
    const std::int32_t own_split_k = schedule_split_k(routed);
    for (const Bf16GdnGatingScheduleId split : splits) {
        if (schedule_split_k(split) < own_split_k && candidate_is_legal(split, problem)) {
            return split;
        }
    }
    return Bf16GdnGatingScheduleId::MmaUnsplit;
}

std::size_t checked_partial_bytes(std::int32_t heads, std::int32_t split_k, std::int32_t cols) {
    const std::size_t logical_rows = static_cast<std::size_t>(2 * heads);
    const std::size_t split        = static_cast<std::size_t>(split_k);
    const std::size_t tokens       = static_cast<std::size_t>(cols);
    if (tokens > std::numeric_limits<std::size_t>::max() / logical_rows ||
        split > std::numeric_limits<std::size_t>::max() / (tokens * logical_rows)) {
        throw std::overflow_error("BF16 GDN gating workspace element count overflows size_t");
    }
    const std::size_t elements = split * tokens * logical_rows;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("BF16 GDN gating workspace byte count overflows size_t");
    }
    return elements * sizeof(float);
}

void execute_resolved(const Bf16GdnGatingPlan& plan, const Bf16GdnGatingProblem& problem,
                      const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                      const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                      Tensor& beta, cudaStream_t stream) {
    auto scratch_scope = ws.scope();
    DeviceSpan scratch{};
    if (plan.workspace_bytes != 0) { scratch = ws.alloc_bytes(plan.workspace_bytes); }

    switch (plan.schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        // Split80 exists only as the fused norm/control schedule; the plain control op never
        // resolves to it through k27Routes.
        throw std::logic_error("BF16 GDN gating: split40 is the fused norm-only schedule");
    case Bf16GdnGatingScheduleId::GemvPairedRows:
        bf16_gdn_gating_proj_gemv_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        bf16_gdn_gating_proj_small_t_split10_launch(x, a_weight, b_weight, A_log, dt_bias,
                                                    scratch.data, scratch.bytes, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        bf16_gdn_gating_proj_35_simt_c4_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta,
                                               stream);
        return;
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        bf16_gdn_gating_proj_35_simt_c8_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta,
                                               stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        bf16_gdn_gating_proj_35_mma_split32_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
        bf16_gdn_gating_proj_35_mma_split16_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        if (is_35(problem)) {
            bf16_gdn_gating_proj_35_mma_split8_launch(plan.token_variant, x, a_weight, b_weight,
                                                      A_log, dt_bias, scratch.data, g, beta,
                                                      stream);
        } else {
            bf16_gdn_gating_proj_mma_split8_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        }
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        if (is_35(problem)) {
            bf16_gdn_gating_proj_35_mma_split4_launch(plan.token_variant, x, a_weight, b_weight,
                                                      A_log, dt_bias, scratch.data, g, beta,
                                                      stream);
        } else {
            bf16_gdn_gating_proj_mma_split4_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        }
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        if (is_35(problem)) {
            bf16_gdn_gating_proj_35_mma_split2_launch(plan.token_variant, x, a_weight, b_weight,
                                                      A_log, dt_bias, scratch.data, g, beta,
                                                      stream);
        } else {
            bf16_gdn_gating_proj_mma_split2_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                   dt_bias, scratch.data, g, beta, stream);
        }
        return;
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        if (is_35(problem)) {
            bf16_gdn_gating_proj_35_mma_unsplit_launch(plan.token_variant, x, a_weight, b_weight,
                                                       A_log, dt_bias, g, beta, stream);
        } else {
            bf16_gdn_gating_proj_mma_unsplit_launch(plan.token_variant, x, a_weight, b_weight,
                                                    A_log, dt_bias, g, beta, stream);
        }
        return;
    }
    throw std::logic_error("BF16 GDN gating: unknown schedule");
}

// Ignis portability patch (ADR 0077, GitHub #602): a cooperative route is also asked at every
// multiple of its column tile inside the interval. Residency changes only where the column-tile
// count does, so inside a route that falls back on a smaller card each run of one schedule ends
// at such a multiple or at the route's last width, and a fixed schedule's workspace grows with
// the width: these widths hold every peak. On the RTX 5090 no route falls back, and each answer
// is the one at the route's last width.
template <std::size_t N>
std::size_t route_capacity(const std::array<RouteSpec, N>& routes, const Bf16GdnGatingProblem& base,
                           std::int32_t min_cols, std::int32_t max_cols) {
    std::size_t maximum = 0;
    const auto workspace = [&](std::int32_t cols) {
        return bf16_gdn_gating_resolve_plan({base.heads, base.input_rows, cols}).workspace_bytes;
    };
    const std::int64_t tile_cols = mma_tile_cols(base);
    for (const RouteSpec& route : routes) {
        if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum                     = std::max(maximum, workspace(endpoint));
        if (!schedule_is_cooperative(route.schedule)) { continue; }
        const std::int64_t first = std::max(route.cols.first, min_cols);
        for (std::int64_t cols = (first + tile_cols - 1) / tile_cols * tile_cols; cols < endpoint;
             cols += tile_cols) {
            maximum = std::max(maximum, workspace(static_cast<std::int32_t>(cols)));
        }
    }
    return maximum;
}

} // namespace

const char* bf16_gdn_gating_schedule_name(Bf16GdnGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::GemvPairedRows:
        return "gdn_gating_proj.bf16.gemv.paired_rows";
    case Bf16GdnGatingScheduleId::SmallTSplit10:
        return "gdn_gating_proj.bf16.small_t.split10";
    case Bf16GdnGatingScheduleId::SimtWarpRowC4:
        return "gdn_gating_proj.bf16.simt.warp_row.c4";
    case Bf16GdnGatingScheduleId::SimtWarpRowC8:
        return "gdn_gating_proj.bf16.simt.warp_row.c8";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        return "gdn_gating_proj.bf16.mma.cooperative_split40";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit32:
        return "gdn_gating_proj.bf16.mma.cooperative_split32";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit16:
        return "gdn_gating_proj.bf16.mma.cooperative_split16";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        return "gdn_gating_proj.bf16.mma.cooperative_split8";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        return "gdn_gating_proj.bf16.mma.cooperative_split4";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return "gdn_gating_proj.bf16.mma.cooperative_split2";
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return "gdn_gating_proj.bf16.mma.unsplit";
    }
    return "gdn_gating_proj.bf16.unknown";
}

const char* bf16_gdn_norm_gating_schedule_name(Bf16GdnNormGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnNormGatingScheduleId::Composed:
        return "gdn_norm_gating_proj.bf16.composed";
    case Bf16GdnNormGatingScheduleId::MmaCooperativeSplit40:
        return "gdn_norm_gating_proj.bf16.mma.cooperative_split40";
    case Bf16GdnNormGatingScheduleId::MmaCooperativeSplit32:
        return "gdn_norm_gating_proj.bf16.mma.cooperative_split32";
    }
    return "gdn_norm_gating_proj.bf16.unknown";
}

bool bf16_gdn_gating_admits(const Bf16GdnGatingProblem& problem) noexcept {
    if (problem.cols < 1) { return false; }
    return is_27(problem) || is_35(problem);
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId schedule,
                                                    const Bf16GdnGatingProblem& problem) {
    if (!candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("BF16 GDN gating: candidate is not legal for exact problem");
    }
    const bool mma                          = schedule_uses_mma(schedule);
    const Bf16GdnGatingTokenVariant variant = !mma ? Bf16GdnGatingTokenVariant::None
                                                   : ((problem.cols % mma_tile_cols(problem)) == 0
                                                          ? Bf16GdnGatingTokenVariant::Full
                                                          : Bf16GdnGatingTokenVariant::Predicated);
    const std::int32_t split_k              = schedule_split_k(schedule);
    const std::size_t workspace =
        split_k > 1 ? checked_partial_bytes(problem.heads, split_k, problem.cols) : 0;
    return {schedule, variant, workspace};
}

Bf16GdnGatingScheduleId bf16_gdn_gating_route_schedule(const Bf16GdnGatingProblem& problem) {
    if (!bf16_gdn_gating_admits(problem)) {
        throw std::invalid_argument(
            "BF16 GDN gating: exact problem or column count is not admitted");
    }
    return is_27(problem) ? routed_schedule(k27Routes, problem.cols)
                          : routed_schedule(k35Routes, problem.cols);
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_plan(const Bf16GdnGatingProblem& problem) {
    const Bf16GdnGatingScheduleId routed = bf16_gdn_gating_route_schedule(problem);
    if (candidate_is_legal(routed, problem)) {
        return bf16_gdn_gating_resolve_candidate(routed, problem);
    }
    if (schedule_is_cooperative(routed)) {
        return bf16_gdn_gating_resolve_candidate(
            is_27(problem) ? resident_fallback(k27Splits, routed, problem)
                           : resident_fallback(k35Splits, routed, problem),
            problem);
    }
    throw std::logic_error("BF16 GDN gating: admitted problem has no covering route");
}

std::size_t bf16_gdn_gating_capacity_workspace_bytes(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t min_cols, std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("BF16 GDN gating: invalid column interval");
    }
    const Bf16GdnGatingProblem base{heads, input_rows, 1};
    (void)bf16_gdn_gating_resolve_plan({heads, input_rows, min_cols});
    (void)bf16_gdn_gating_resolve_plan({heads, input_rows, max_cols});
    return is_27(base) ? route_capacity(k27Routes, base, min_cols, max_cols)
                       : route_capacity(k35Routes, base, min_cols, max_cols);
}

Bf16GdnNormGatingPlan bf16_gdn_norm_gating_resolve_plan(const Bf16GdnGatingProblem& problem) {
    Bf16GdnGatingPlan control            = bf16_gdn_gating_resolve_plan(problem);
    Bf16GdnNormGatingScheduleId schedule = Bf16GdnNormGatingScheduleId::Composed;
    std::int32_t norm_splits             = 0;
    if (is_35(problem) && problem.cols <= 16 &&
        candidate_is_legal(Bf16GdnGatingScheduleId::MmaCooperativeSplit32, problem)) {
        control  = bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId::MmaCooperativeSplit32,
                                                     problem);
        schedule = Bf16GdnNormGatingScheduleId::MmaCooperativeSplit32;
        norm_splits = 32;
    } else if (is_27(problem) && problem.cols <= 8 &&
               // Ignis portability patch (ADR 0077, GitHub #602): split40 only where the card
               // holds its grid resident (120 CTAs, one per SM on the 120a build), as split32
               // above; the composed route otherwise.
               candidate_is_legal(Bf16GdnGatingScheduleId::MmaCooperativeSplit40, problem)) {
        control  = bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId::MmaCooperativeSplit40,
                                                     problem);
        schedule = Bf16GdnNormGatingScheduleId::MmaCooperativeSplit40;
        norm_splits = 80;
    }
    const std::size_t norm_partial_bytes =
        static_cast<std::size_t>(norm_splits) * problem.cols * sizeof(float);
    return {schedule, control, control.workspace_bytes + norm_partial_bytes};
}

std::size_t bf16_gdn_norm_gating_capacity_workspace_bytes(std::int32_t heads,
                                                          std::int32_t input_rows,
                                                          std::int32_t min_cols,
                                                          std::int32_t max_cols) {
    std::size_t maximum =
        bf16_gdn_gating_capacity_workspace_bytes(heads, input_rows, min_cols, max_cols);
    if (heads == 48 && input_rows == 5120 && min_cols <= 8) {
        const std::int32_t fused_cols = std::min<std::int32_t>(max_cols, 8);
        maximum                       = std::max(
            maximum,
            bf16_gdn_norm_gating_resolve_plan({heads, input_rows, fused_cols}).workspace_bytes);
    }
    if (heads == 32 && input_rows == 2048 && min_cols <= 16) {
        const std::int32_t fused_cols = std::min<std::int32_t>(max_cols, 16);
        maximum                       = std::max(
            maximum,
            bf16_gdn_norm_gating_resolve_plan({heads, input_rows, fused_cols}).workspace_bytes);
    }
    return maximum;
}

void bf16_gdn_gating_set_test_multiprocessor_cap(int multiprocessors) noexcept {
    test_multiprocessor_cap.store(std::max(multiprocessors, 0), std::memory_order_relaxed);
}

void bf16_gdn_gating_execute_plan(const Bf16GdnGatingPlan& plan, const Tensor& x,
                                  const Weight& a_weight, const Weight& b_weight,
                                  const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                  Tensor& g, Tensor& beta, cudaStream_t stream) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnGatingPlan resolved = bf16_gdn_gating_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.token_variant != plan.token_variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("BF16 GDN gating: plan does not match the exact problem");
    }
    execute_resolved(plan, problem, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void bf16_gdn_gating_execute_candidate(Bf16GdnGatingScheduleId schedule, const Tensor& x,
                                       const Weight& a_weight, const Weight& b_weight,
                                       const Tensor& A_log, const Tensor& dt_bias,
                                       WorkspaceArena& ws, Tensor& g, Tensor& beta,
                                       cudaStream_t stream) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnGatingPlan plan = bf16_gdn_gating_resolve_candidate(schedule, problem);
    execute_resolved(plan, problem, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void bf16_gdn_gating_dispatch(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                              Tensor& g, Tensor& beta, cudaStream_t stream) {
    const Bf16GdnGatingPlan plan = bf16_gdn_gating_resolve_plan({g.ne[0], x.ne[0], x.ne[1]});
    bf16_gdn_gating_execute_plan(plan, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void bf16_gdn_norm_gating_dispatch(const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h,
                                   const Weight& a_weight, const Weight& b_weight,
                                   const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                   Tensor& g, Tensor& beta, cudaStream_t stream) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnNormGatingPlan plan = bf16_gdn_norm_gating_resolve_plan(problem);
    if (plan.schedule == Bf16GdnNormGatingScheduleId::Composed) {
        rmsnorm(x, norm_weight, eps, true, h, stream);
        execute_resolved(plan.control, problem, h, a_weight, b_weight, A_log, dt_bias, ws, g, beta,
                         stream);
        return;
    }

    auto scratch_scope = ws.scope();
    DeviceSpan scratch{};
    if (plan.workspace_bytes != 0) { scratch = ws.alloc_bytes(plan.workspace_bytes); }
    if (plan.schedule == Bf16GdnNormGatingScheduleId::MmaCooperativeSplit40) {
        bf16_gdn_norm_gating_proj_27_mma_split40_launch(plan.control.token_variant, x, norm_weight,
                                                        eps, h, a_weight, b_weight, A_log, dt_bias,
                                                        scratch.data, g, beta, stream);
        return;
    }
    bf16_gdn_norm_gating_proj_35_mma_split32_launch(plan.control.token_variant, x, norm_weight, eps,
                                                    h, a_weight, b_weight, A_log, dt_bias,
                                                    scratch.data, g, beta, stream);
}

} // namespace ninfer::ops::detail
