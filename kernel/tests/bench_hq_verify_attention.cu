// GitHub #268 (spec runtime/09): the hq-e8-2b verify attention kernel, alone,
// at the engine's real launch -- OURS, not vendored, and NOT a CTest test (a
// timing is a finding, not a pass/fail; docs/agents/testing.md).
//
// It launches exactly what the verify round launches for one GQA layer:
// gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry, 8, 4, true,
// true, CacheInput, GqaTcKVHq> on grid (KV heads, 85 splits, lanes) x 128
// threads, width 8 (draft 7 + bonus), every lane its own block-table row and
// a window of --ctx keys. The code/meta planes are filled by the vendored
// encoder from the committed real K/V rows (kernel/tests/fixtures), each row
// under its true (kv_head, position, role) dither seed, so the Rice streams
// have the real rows' bit density. L2 is flushed before every launch: in a
// real round the history is cold behind ~15 GB of weight traffic.
//
// Output:
//   - the median kernel time per launch, and per GQA layer per 1K keys per
//     lane (the unit of docs/findings/2026-09-24-hq-attention-at-long-context.md);
//   - an FNV-1a64 hash of every partial the launch wrote. The inputs are
//     deterministic, so two builds of the kernel print the same hash exactly
//     when their partials are bit-identical: the kernel-level check that a
//     decode rewrite changed no bit.
//
//   ignis_hq_verify_bench --lanes 8 --ctx 30720 [--iters 20] [--residual] [--decode-only]
//                         [--reference]
//
// --reference launches the kernel's reference hq route (IgnisThroughput =
// false: main's code path) instead of the throughput route, a same-binary A/B
// that prints the same hash.
//
// --decode-only times the kernel's tile decode alone (decode_only_kernel),
// once with the reference group decode and once with the throughput decoder.
//
// --residual turns the hq residual window on (exact sink + recent-ring rows
// from the side planes, as the engine serves them); off, every key is a
// codec row (the worst case).

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ops/kernel/gqa_attention_decode_hq.cuh"

using namespace ninfer;
using namespace ninfer::ops;

#ifndef IGNIS_HQ_KV_FIXTURE_PATH
#error "IGNIS_HQ_KV_FIXTURE_PATH must be defined by kernel/tests/CMakeLists.txt"
#endif

namespace {

#define CUDA_CHECK(expr)                                                                        \
    do {                                                                                        \
        const cudaError_t _err = (expr);                                                        \
        if (_err != cudaSuccess) {                                                              \
            std::fprintf(stderr, "FATAL: %s failed: %s\n", #expr, cudaGetErrorString(_err));    \
            std::exit(EXIT_FAILURE);                                                            \
        }                                                                                       \
    } while (0)

using Geometry = Gqa27Geometry;
constexpr int kTokens = 8;                 // verify width: 7 drafts + bonus
constexpr int kSplits = Geometry::DecodeSplits;  // the launcher's capacity at serving envelopes

std::uint64_t fnv1a64(const void* p, std::size_t n, std::uint64_t h = 0xcbf29ce484222325ull) {
    const auto* data = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// The fixture's row payload (validated by test_hq_codec_kv_rows.cu; here only
// the header fields that locate it are read).
struct Rows {
    int kv_heads = 0, roles = 0, layers = 0, per_block = 0;
    std::vector<std::uint16_t> bits;
};

Rows load_rows(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "FATAL: cannot open fixture %s\n", path);
        std::exit(EXIT_FAILURE);
    }
    std::vector<std::uint8_t> data;
    std::fseek(f, 0, SEEK_END);
    data.resize(static_cast<std::size_t>(std::ftell(f)));
    std::fseek(f, 0, SEEK_SET);
    if (std::fread(data.data(), 1, data.size(), f) != data.size()) {
        std::fprintf(stderr, "FATAL: short read on %s\n", path);
        std::exit(EXIT_FAILURE);
    }
    std::fclose(f);
    auto u32 = [&](std::size_t off) {
        std::uint32_t v;
        std::memcpy(&v, data.data() + off, 4);
        return static_cast<int>(v);
    };
    if (data.size() < 32 || std::memcmp(data.data(), "IGNHQKV1", 8) != 0 || u32(12) != kHqHeadDim) {
        std::fprintf(stderr, "FATAL: %s is not an IGNHQKV1 fixture at head_dim 256\n", path);
        std::exit(EXIT_FAILURE);
    }
    Rows r;
    r.kv_heads = u32(16);
    r.roles    = u32(20);
    r.layers   = u32(24);
    std::size_t off = 28 + 4 * static_cast<std::size_t>(r.layers);
    r.per_block      = u32(off + 4);
    const int total  = u32(off + 8);
    off += 12 + 8;
    r.bits.resize(static_cast<std::size_t>(total) * kHqHeadDim);
    std::memcpy(r.bits.data(), data.data() + off, r.bits.size() * 2);
    return r;
}

// One warp encodes one (lane, position, kv_head, role) row of the history into
// that lane's pages, from the fixture row of the same kv_head and role.
__global__ void fill_history_kernel(const __nv_bfloat16* rows, int kv_heads, int layers,
                                    int per_block, const std::int32_t* block_tables,
                                    int table_stride, int ctx, int lanes, GqaTcKVHq kv) {
    extern __shared__ float smem[];
    __shared__ std::int8_t signs[kHqHeadDim];
    hq_engine_signs_fill(signs);
    __syncthreads();
    const long long unit = static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const long long units = static_cast<long long>(lanes) * ctx * Geometry::KVHeads * 2;
    if (unit >= units) { return; }
    const bool role_v = (unit & 1) != 0;
    const int head    = static_cast<int>((unit >> 1) % Geometry::KVHeads);
    const long long lp = (unit >> 1) / Geometry::KVHeads;
    const int pos     = static_cast<int>(lp % ctx);
    const int lane    = static_cast<int>(lp / ctx);
    const int layer   = (pos / per_block + lane) % layers;
    const int src_row = ((layer * 2 + (role_v ? 1 : 0)) * kv_heads + head) * per_block +
                        (pos + 37 * lane) % per_block;
    float* u            = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
    const std::int32_t* table = block_tables + static_cast<long long>(lane) * table_stride;
    hq_encode_row_warp(rows + static_cast<long long>(src_row) * kHqHeadDim, signs, 0, u, syms,
                       hq_row_codes_mut<Geometry>(role_v ? kv.codes_v : kv.codes_k, table, head, pos),
                       hq_row_meta_mut<Geometry>(role_v ? kv.meta_v : kv.meta_k, table, head, pos),
                       hq_dither_row_seed(head, pos, role_v));
    if (kv.residual_k != nullptr &&
        (pos < static_cast<int>(kGqaHqSinkKeys) || pos >= ctx - static_cast<int>(kGqaHqRecentKeys))) {
        hq_store_rotated_row_warp(rows + static_cast<long long>(src_row) * kHqHeadDim, signs,
                                  hq_residual_row<Geometry>(role_v ? kv.residual_v : kv.residual_k,
                                                            lane, head, pos));
    }
}

__global__ void fill_q_kernel(__nv_bfloat16* q, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9u ^ 0x85EBCA6Bu;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    q[i] = __float2bfloat16((static_cast<float>(x & 0xFFFFu) / 65535.0f - 0.5f) * 4.0f);
}

// The verify kernel's tile decode alone (--decode-only): same grid, same split
// ranges, same 32-key tiles decoded into a swizzled K/V tile from code rows
// staged one tile ahead by cp.async, the same shared-memory footprint (two
// blocks per SM), no MMA. R = 0 is the reference group decode; R >= 1 the
// throughput decoder with each 8-lane group decoding R rows at once.
template <int R>
__launch_bounds__(128, 2) __global__ void decode_only_kernel(GqaTcKVHq kv,
                                                             const std::int32_t* block_tables,
                                                             int table_stride, int window,
                                                             std::uint32_t* sink) {
    constexpr int Bc = 32, D = kGqaHeadDim, Threads = 128;
    constexpr int kCodes = Bc * kHqRowBudgetBytes, kMeta = Bc * kHqMetaBytes;
    constexpr int kStage = 2 * (kCodes + kMeta);
    constexpr int kRows  = R == 0 ? 1 : R;
    __shared__ __align__(16) __nv_bfloat16 qkv_s[2 * Bc * D];
    __shared__ __align__(16) std::uint8_t stage_s[2 * kStage];
    __shared__ __align__(16) __nv_bfloat16 pad_s[Threads * 16];  // the kernel's P tile + ids
    const int kv_head = static_cast<int>(blockIdx.x);
    const int split   = static_cast<int>(blockIdx.y);
    const int lane_b  = static_cast<int>(blockIdx.z);
    const int tid     = static_cast<int>(threadIdx.x);
    const std::int32_t* block_table = block_tables + static_cast<long long>(lane_b) * table_stride;
    const int active = gqa_small_t_active_splits<Gqa27Geometry, false>(window, gridDim.y, 8);
    if (split >= active) { return; }
    const int tiles       = div_up(window, Bc);
    const int per_split   = div_up(tiles, active);
    const int split_start = split * per_split * Bc;
    const int split_end   = min(split_start + per_split * Bc, window);
    auto stage_tile = [&](int k0, int buf) {
        const int page = block_table[k0 >> kPagedKVPageShift];
        const long long co = paged_kv_element_offset<kHqCodePlaneExtent, Gqa27Geometry::KVHeads>(
            page, kv_head, k0 & kPagedKVPageMask, 0);
        const long long mo = paged_kv_element_offset<kHqMetaPlaneExtent, Gqa27Geometry::KVHeads>(
            page, kv_head, k0 & kPagedKVPageMask, 0);
        for (int chunk = tid; chunk < kStage / 16; chunk += Threads) {
            const int b = chunk * 16;
            const std::uint8_t* src = b < kCodes       ? kv.codes_k + co + b
                                      : b < 2 * kCodes ? kv.codes_v + co + (b - kCodes)
                                      : b < 2 * kCodes + kMeta ? kv.meta_k + mo + (b - 2 * kCodes)
                                                               : kv.meta_v + mo + (b - 2 * kCodes - kMeta);
            ninfer::ops::cp_async<16, Cache::cg>(stage_s + buf * kStage + b, src);
        }
        ninfer::ops::cp_commit();
    };
    std::uint32_t acc = 0;
    pad_s[tid] = __float2bfloat16(0.0f);
    stage_tile(split_start, 0);
    ninfer::ops::cp_wait<0>();
    __syncthreads();
    for (int k0 = split_start, kb = 0; k0 < split_end; k0 += Bc, ++kb) {
        if (k0 + Bc < split_end) { stage_tile(k0 + Bc, (kb + 1) & 1); }
        const std::uint8_t* stage = stage_s + (kb & 1) * kStage;
        const int group = tid >> 3, lane8 = tid & 7;
#pragma unroll 1
        for (int wave = 0; wave < 2 * Bc / (16 * kRows); ++wave) {
            HqGroupRow rows[kRows];
            bool live = true;
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const int row     = (wave * 16 + group) * kRows + r;
                const bool role_v = row >= Bc;
                const int key_l   = row & (Bc - 1);
                const int key     = k0 + key_l;
                live = live && key < split_end;
                rows[r] = HqGroupRow{stage + (role_v ? kCodes : 0) + key_l * kHqRowBudgetBytes,
                                     stage + 2 * kCodes + (role_v ? kMeta : 0) + key_l * kHqMetaBytes,
                                     qkv_s + (role_v ? Bc * D : 0) + key_l * D, key_l & 7,
                                     hq_dither_row_seed(kv_head, key, role_v), nullptr};
            }
            if (!live) { continue; }
            if constexpr (R == 0) {
                hq_decode_row_group(rows[0].codes, rows[0].meta, rows[0].out, lane8,
                                    rows[0].xor_chunk, rows[0].dither_seed);
            } else {
                hq_decode_rows_group_fast<R>(rows, lane8);
            }
        }
        ninfer::ops::cp_wait<0>();
        __syncthreads();
        acc += reinterpret_cast<const std::uint32_t*>(qkv_s)[(tid * 37 + k0) & (Bc * D - 1)];
        __syncthreads();
    }
    if (acc == 0x12345678u) { sink[0] = acc + static_cast<std::uint32_t>(__bfloat16_as_ushort(pad_s[tid])); }
}

} // namespace

int main(int argc, char** argv) {
    int lanes = 1, ctx = 30720, iters = 20;
    bool residual = false, decode_only = false, reference = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--lanes" && i + 1 < argc) { lanes = std::atoi(argv[++i]); }
        else if (a == "--ctx" && i + 1 < argc) { ctx = std::atoi(argv[++i]); }
        else if (a == "--iters" && i + 1 < argc) { iters = std::atoi(argv[++i]); }
        else if (a == "--residual") { residual = true; }
        else if (a == "--decode-only") { decode_only = true; }
        else if (a == "--reference") { reference = true; }
        else {
            std::fprintf(stderr, "usage: %s --lanes N --ctx KEYS [--iters N] [--residual] [--decode-only] [--reference]\n", argv[0]);
            return 2;
        }
    }
    if (lanes < 1 || ctx < kTokens || iters < 1) { return 2; }

    const Rows rows = load_rows(IGNIS_HQ_KV_FIXTURE_PATH);
    const int pages_per_lane = (ctx + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const long long pages    = static_cast<long long>(pages_per_lane) * lanes;
    const long long plane_rows = pages * Geometry::KVHeads * kPagedKVPageSize;

    // Lane b owns physical pages [b * pages_per_lane, (b+1) * pages_per_lane),
    // listed in reverse so the page walk is not a linear address sweep.
    std::vector<std::int32_t> table(static_cast<std::size_t>(pages));
    for (int b = 0; b < lanes; ++b) {
        for (int p = 0; p < pages_per_lane; ++p) {
            table[static_cast<std::size_t>(b) * pages_per_lane + p] =
                b * pages_per_lane + (pages_per_lane - 1 - p);
        }
    }
    std::vector<std::int32_t> table_rows(lanes), pos(static_cast<std::size_t>(lanes) * kTokens);
    for (int b = 0; b < lanes; ++b) {
        table_rows[b] = b;
        for (int t = 0; t < kTokens; ++t) { pos[b * kTokens + t] = ctx - kTokens + t; }
    }

    std::int32_t *d_table, *d_table_rows, *d_pos;
    CUDA_CHECK(cudaMalloc(&d_table, table.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_table_rows, table_rows.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_pos, pos.size() * 4));
    CUDA_CHECK(cudaMemcpy(d_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table_rows, table_rows.data(), table_rows.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));

    GqaTcKVHq kv{};
    CUDA_CHECK(cudaMalloc(&kv.codes_k, plane_rows * kHqCodePlaneExtent));
    CUDA_CHECK(cudaMalloc(&kv.codes_v, plane_rows * kHqCodePlaneExtent));
    CUDA_CHECK(cudaMalloc(&kv.meta_k, plane_rows * kHqMetaPlaneExtent));
    CUDA_CHECK(cudaMalloc(&kv.meta_v, plane_rows * kHqMetaPlaneExtent));
    if (residual) {
        const std::size_t side = static_cast<std::size_t>(lanes) *
                                 (kGqaHqSinkKeys + kGqaHqRecentKeys) * Geometry::KVHeads *
                                 kHqHeadDim * 2;
        CUDA_CHECK(cudaMalloc(&kv.residual_k, side));
        CUDA_CHECK(cudaMalloc(&kv.residual_v, side));
        // ring_valid stays null: every ring slot is valid (pre-filled planes).
    }

    __nv_bfloat16* d_rows;
    CUDA_CHECK(cudaMalloc(&d_rows, rows.bits.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_rows, rows.bits.data(), rows.bits.size() * 2, cudaMemcpyHostToDevice));
    {
        constexpr int kWarps = 8;
        const long long units = static_cast<long long>(lanes) * ctx * Geometry::KVHeads * 2;
        const std::size_t smem = kWarps * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * 4;
        fill_history_kernel<<<static_cast<unsigned>((units + kWarps - 1) / kWarps), kWarps * 32, smem>>>(
            d_rows, rows.kv_heads, rows.layers, rows.per_block, d_table, pages_per_lane, ctx, lanes, kv);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const long long q_elems = static_cast<long long>(lanes) * kTokens * Geometry::QHeads * kGqaHeadDim;
    __nv_bfloat16* d_q;
    CUDA_CHECK(cudaMalloc(&d_q, q_elems * 2));
    fill_q_kernel<<<static_cast<unsigned>((q_elems + 255) / 256), 256>>>(d_q, q_elems);
    CUDA_CHECK(cudaGetLastError());

    const long long acc_elems = q_elems * kSplits;
    const long long stat_elems = static_cast<long long>(lanes) * kTokens * Geometry::QHeads * kSplits;
    __nv_bfloat16* d_acc;
    float *d_m, *d_l;
    CUDA_CHECK(cudaMalloc(&d_acc, acc_elems * 2));
    CUDA_CHECK(cudaMalloc(&d_m, stat_elems * 4));
    CUDA_CHECK(cudaMalloc(&d_l, stat_elems * 4));

    // 256 MiB scrubbed before every launch: larger than the 5090's L2.
    constexpr std::size_t kFlushBytes = 256ull << 20;
    void* d_flush;
    CUDA_CHECK(cudaMalloc(&d_flush, kFlushBytes));

    const dim3 grid(Geometry::KVHeads, kSplits, lanes);
    const float scale = 1.0f / 16.0f;
    auto launch = [&]() {
        if (reference) {
            gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, 8, 4, true, true, GqaCachedInput,
                                                         GqaTcKVHq, false>
                <<<grid, kGqaHqDecodeThreads>>>(d_q, GqaCachedInput{}, d_pos, kv, d_table, nullptr,
                                                d_table_rows, pages_per_lane, kTokens, kTokens, 0,
                                                262144, scale, d_acc, d_m, d_l);
        } else {
            gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, 8, 4, true, true, GqaCachedInput,
                                                         GqaTcKVHq>
                <<<grid, kGqaHqDecodeThreads>>>(d_q, GqaCachedInput{}, d_pos, kv, d_table, nullptr,
                                                d_table_rows, pages_per_lane, kTokens, kTokens, 0,
                                                262144, scale, d_acc, d_m, d_l);
        }
    };

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    // Median, min and max GPU time of one launch, after two warm-up launches.
    auto time_launches = [&](auto&& fn, float& lo, float& hi) {
        std::vector<float> ms;
        for (int it = 0; it < iters + 2; ++it) {
            CUDA_CHECK(cudaMemsetAsync(d_flush, it & 0xFF, kFlushBytes));
            CUDA_CHECK(cudaEventRecord(e0));
            fn();
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(e1));
            float t = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&t, e0, e1));
            if (it >= 2) { ms.push_back(t * 1000.0f); }
        }
        std::sort(ms.begin(), ms.end());
        lo = ms.front();
        hi = ms.back();
        return ms[ms.size() / 2];
    };

    if (decode_only) {
        std::uint32_t* d_sink;
        CUDA_CHECK(cudaMalloc(&d_sink, 4));
        float lo = 0.0f, hi = 0.0f;
        const float ref_us = time_launches([&]() {
            decode_only_kernel<0><<<grid, 128>>>(kv, d_table, pages_per_lane, ctx, d_sink);
        }, lo, hi);
        const float r1_us = time_launches([&]() {
            decode_only_kernel<1><<<grid, 128>>>(kv, d_table, pages_per_lane, ctx, d_sink);
        }, lo, hi);
        const float r2_us = time_launches([&]() {
            decode_only_kernel<2><<<grid, 128>>>(kv, d_table, pages_per_lane, ctx, d_sink);
        }, lo, hi);
        const float r4_us = time_launches([&]() {
            decode_only_kernel<4><<<grid, 128>>>(kv, d_table, pages_per_lane, ctx, d_sink);
        }, lo, hi);
        std::printf("hq tile decode alone: lanes %d ctx %d: reference group decode %.1f us; "
                    "throughput decode, rows per group 1: %.1f us, 2: %.1f us, 4: %.1f us\n",
                    lanes, ctx, ref_us, r1_us, r2_us, r4_us);
        return 0;
    }

    float lo = 0.0f, hi = 0.0f;
    const float median_us = time_launches(launch, lo, hi);

    std::vector<std::uint8_t> host(static_cast<std::size_t>(acc_elems) * 2);
    CUDA_CHECK(cudaMemcpy(host.data(), d_acc, host.size(), cudaMemcpyDeviceToHost));
    std::uint64_t h = fnv1a64(host.data(), host.size());
    host.resize(static_cast<std::size_t>(stat_elems) * 4);
    CUDA_CHECK(cudaMemcpy(host.data(), d_m, host.size(), cudaMemcpyDeviceToHost));
    h = fnv1a64(host.data(), host.size(), h);
    CUDA_CHECK(cudaMemcpy(host.data(), d_l, host.size(), cudaMemcpyDeviceToHost));
    h = fnv1a64(host.data(), host.size(), h);

    std::printf("hq verify attention (%s route): lanes %d ctx %d residual %s: median %.1f us (min "
                "%.1f, max %.1f, %d launches) = %.3f us per layer per 1K keys per lane; partials "
                "fnv1a64 %016llx\n",
                reference ? "reference" : "throughput", lanes, ctx, residual ? "on" : "off",
                median_us, lo, hi, iters, median_us / (lanes * (ctx / 1024.0f)),
                static_cast<unsigned long long>(h));
    return 0;
}
