// GitHub #726: the 27B's hq-e8-2b small-T attention past 696,320 visible keys, against an FP64
// reference -- OURS, not vendored.
//
// With the 27B's 85 splits a split spans 128 pages of 64 keys at a window of 696,320 keys, and
// more past it: 129 at 696,321, 169 at 917,504, 193 at the envelope's 1,048,576. The hq source
// of gqa_attention_small_t_tc_partial_bf16_kernel and ignis's hq_verify_ws_kernel once staged a
// split's physical page ids in a 128-entry shared-memory table, with no bound on how many pages
// the split spans, so past 696,320 keys the copy ran over the RHT signs declared after it. Both
// now take each tile's page from the block table (ADR 0037: the vendored kernel's correctness
// patch; kernel/vendor/VENDOR.md, Ignis-patched behaviours).
//
// The vendored reference form (IgnisThroughput = false) cannot be this test's oracle: its
// source carried the same fill (no read of the table survives in its build, so it never showed
// the fault). So every route is held to the attention computed here in FP64 from the rows it
// should read:
//   - each key's row decoded on its own (hq_decode_row_thread, the one-thread decoder the
//     group decoders are bit-identical to) from the address the block table gives for that
//     key, under that key's dither seed, or its exact side row where the residual window
//     serves it (a sink, or a recent key whose ring slot is valid);
//   - q rotated into the codec frame, the softmax over the split's keys at or before each
//     column's position, and the output rotated back, all in FP64;
// over five splits of every case (the first two, the middle one, the last two: the sinks, the
// recent ring and the call's own columns among them). Every partial of every split must also
// stay bit-identical across the three routes (ADR 0093), and with the fused append every cache
// byte must be what the codec's own encode of the new rows writes at their positions: their
// code and meta rows, their side rows and their ring marks.
//
// Memory stays small: the block table maps the 16,384 logical pages onto a pool of 256
// physical pages (logical mod 256, so no two pages of one split share a physical page and a
// wrong page id changes the output), and the pages that hold the call's own columns onto
// pages of their own, so one split's append never lands on a page another split reads.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU, a kernel error or a
// missing fixture fails this test.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "ops/kernel/gqa_attention_decode_hq.cuh"

#include "hq_verify_ws.cuh"

using namespace ninfer;
using namespace ninfer::ops;

#ifndef IGNIS_HQ_KV_FIXTURE_PATH
#error "IGNIS_HQ_KV_FIXTURE_PATH must be defined by kernel/tests/CMakeLists.txt"
#endif

namespace {

int g_failed = 0;

void check(bool ok, const std::string& msg) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL: %s\n", msg.c_str());
        ++g_failed;
    }
}

#define CUDA_CHECK(expr)                                                                        \
    do {                                                                                        \
        const cudaError_t _err = (expr);                                                        \
        if (_err != cudaSuccess) {                                                              \
            std::fprintf(stderr, "FATAL: %s failed: %s\n", #expr, cudaGetErrorString(_err));    \
            std::exit(EXIT_FAILURE);                                                            \
        }                                                                                       \
    } while (0)

using Geometry = Gqa27Geometry;
constexpr int kTokenTile    = 8;
constexpr int kSplits       = Geometry::DecodeSplits;  // 85: every call past 40,320 keys
constexpr int kHeads        = Geometry::KVHeads;
constexpr int kGroup        = Geometry::GroupSize;
constexpr int kD            = kHqHeadDim;
constexpr int kRingWords    = static_cast<int>(kGqaHqRecentKeys) / 32;
constexpr int kCapacity     = static_cast<int>(kGqaAttentionMaximumVisibleKeys);  // 1,048,576
constexpr int kLogicalPages = kCapacity / kPagedKVPageSize;                       // 16,384
constexpr int kPool         = 256;  // > the 193 pages one split spans at the envelope
constexpr int kPhysical     = kPool + 2;  // + the pages of the call's own columns
constexpr float kScale      = 1.0f / 16.0f;
// The routes compute in BF16 (q rotated, the probabilities, the stored partials); a misread
// sign vector or page is off by the size of the output itself.
constexpr double kTolerance = 2e-2;

// The layouts of q and of the partials (gqa_q_index, gqa_partial_acc_index,
// gqa_partial_stat_index), on the host.
std::size_t gqa_q_index_host(int q_head, int token) {
    return (static_cast<std::size_t>(token) * Geometry::QHeads + q_head) * kD;
}
std::size_t gqa_acc_index_host(int q_head, int d, int token, int split, int tokens) {
    return d + static_cast<std::size_t>(kD) *
                   (q_head + static_cast<std::size_t>(Geometry::QHeads) *
                                 (token + static_cast<std::size_t>(tokens) * split));
}
std::size_t gqa_stat_index_host(int q_head, int token, int split, int tokens) {
    return q_head + static_cast<std::size_t>(Geometry::QHeads) *
                        (token + static_cast<std::size_t>(tokens) * split);
}

// The fixture's row payload, as test_hq_verify_exact.cu reads it (its full validation is
// test_hq_codec_kv_rows.cu's).
struct Rows {
    int layers = 0, per_block = 0;
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
    const bool read_ok = std::fread(data.data(), 1, data.size(), f) == data.size();
    std::fclose(f);
    auto u32 = [&](std::size_t off) {
        std::uint32_t v;
        std::memcpy(&v, data.data() + off, 4);
        return static_cast<int>(v);
    };
    if (!read_ok || data.size() < 32 || std::memcmp(data.data(), "IGNHQKV1", 8) != 0 ||
        u32(12) != kD || u32(16) != kHeads) {
        std::fprintf(stderr, "FATAL: %s is not the 27B IGNHQKV1 fixture\n", path);
        std::exit(EXIT_FAILURE);
    }
    Rows r;
    r.layers        = u32(24);
    std::size_t off = 28 + 4 * static_cast<std::size_t>(r.layers);
    r.per_block     = u32(off + 4);
    const int total = u32(off + 8);
    off += 20;
    if (data.size() != off + static_cast<std::size_t>(total) * kD * 2) {
        std::fprintf(stderr, "FATAL: %s has a truncated payload\n", path);
        std::exit(EXIT_FAILURE);
    }
    r.bits.resize(static_cast<std::size_t>(total) * kD);
    std::memcpy(r.bits.data(), data.data() + off, r.bits.size() * 2);
    return r;
}

__device__ const __nv_bfloat16* fixture_row(const __nv_bfloat16* rows, int layers, int per_block,
                                             long long key, int head, bool role_v) {
    const int layer = static_cast<int>((key / per_block) % layers);
    const int src   = ((layer * 2 + (role_v ? 1 : 0)) * kHeads + head) * per_block +
                    static_cast<int>((key * 7 + 13) % per_block);
    return rows + static_cast<long long>(src) * kD;
}

// One warp encodes one row of a physical page. A pool page holds rows that every logical page
// mapped onto it reads (each under its own key's dither seed); a page of the call's own columns
// holds its one logical page's rows, below fill_limit (the keys the fused append writes) and
// the window.
__global__ void fill_pages_kernel(const __nv_bfloat16* rows, int layers, int per_block,
                                  int own_logical0, int own_logical1, int fill_limit,
                                  GqaTcKVHq kv) {
    extern __shared__ float smem[];
    __shared__ std::int8_t signs[kD];
    hq_engine_signs_fill(signs);
    __syncthreads();
    const long long unit = static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (unit >= static_cast<long long>(kPhysical) * kPagedKVPageSize * kHeads * 2) { return; }
    const bool role_v = (unit & 1) != 0;
    const int head    = static_cast<int>((unit >> 1) % kHeads);
    const int offset  = static_cast<int>(((unit >> 1) / kHeads) % kPagedKVPageSize);
    const int page    = static_cast<int>(((unit >> 1) / kHeads) / kPagedKVPageSize);
    long long key     = static_cast<long long>(page) * kPagedKVPageSize + offset;
    if (page >= kPool) {
        const int logical = page == kPool ? own_logical0 : own_logical1;
        if (logical < 0) { return; }
        key = static_cast<long long>(logical) * kPagedKVPageSize + offset;
        if (key >= fill_limit) { return; }
    }
    float* u            = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
    hq_encode_row_warp(fixture_row(rows, layers, per_block, key, head, role_v), signs, 0, u, syms,
                       (role_v ? kv.codes_v : kv.codes_k) +
                           paged_kv_element_offset<kHqCodePlaneExtent, kHeads>(page, head, offset, 0),
                       (role_v ? kv.meta_v : kv.meta_k) +
                           paged_kv_element_offset<kHqMetaPlaneExtent, kHeads>(page, head, offset, 0),
                       hq_dither_row_seed(head, key, role_v));
}

// One warp stores the exact side row of one sink or recent key below fill_limit: a row of its
// own, not the one its code row holds, so a route that reads the wrong one differs.
__global__ void fill_side_kernel(const __nv_bfloat16* rows, int layers, int per_block, int window,
                                 int fill_limit, GqaTcKVHq kv) {
    __shared__ std::int8_t signs[kD];
    hq_engine_signs_fill(signs);
    __syncthreads();
    constexpr int kKeys = static_cast<int>(kGqaHqSinkKeys + kGqaHqRecentKeys);
    const int unit = static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    if (unit >= kKeys * kHeads * 2) { return; }
    const bool role_v = (unit & 1) != 0;
    const int head    = (unit >> 1) % kHeads;
    const int i       = (unit >> 1) / kHeads;
    const int key     = i < static_cast<int>(kGqaHqSinkKeys)
                            ? i
                            : window - static_cast<int>(kGqaHqRecentKeys) + (i - static_cast<int>(kGqaHqSinkKeys));
    if (key >= fill_limit) { return; }
    hq_store_rotated_row_warp(fixture_row(rows, layers, per_block, key + 0x5000, head, role_v), signs,
                              hq_residual_row<Geometry>(role_v ? kv.residual_v : kv.residual_k, 0, head, key));
}

// The fused append's body, outside any attention kernel: one warp encodes one (column, head,
// role) row of the new K/V where the block table puts its position, and with the residual
// window on stores its side row and marks its ring slot.
__global__ void reference_append_kernel(const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
                                        const std::int32_t* pos, int tokens,
                                        const std::int32_t* table, GqaTcKVHq kv) {
    extern __shared__ float smem[];
    __shared__ std::int8_t signs[kD];
    hq_engine_signs_fill(signs);
    __syncthreads();
    const int unit = static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    if (unit >= tokens * kHeads * 2) { return; }
    const bool role_v = (unit & 1) != 0;
    const int head    = (unit >> 1) % kHeads;
    const int t       = (unit >> 1) / kHeads;
    const int p       = pos[t];
    float* u            = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
    const __nv_bfloat16* src = (role_v ? v_new : k_new) + gqa_kv_new_index<Geometry>(head, 0, t);
    hq_encode_row_warp(src, signs, 0, u, syms,
                       hq_row_codes_mut<Geometry>(role_v ? kv.codes_v : kv.codes_k, table, head, p),
                       hq_row_meta_mut<Geometry>(role_v ? kv.meta_v : kv.meta_k, table, head, p),
                       hq_dither_row_seed(head, p, role_v));
    if (kv.residual_k != nullptr) {
        hq_store_rotated_row_warp(src, signs,
                                  hq_residual_row<Geometry>(role_v ? kv.residual_v : kv.residual_k, 0, head, p));
        hq_ring_mark_valid(kv.ring_valid, p);
    }
}

// One thread writes the rotated-frame row a route must read for one (role, head, key) of
// [key_begin, key_begin + keys): the side row where the residual window serves the key, else
// the code row the block table addresses, decoded under the key's own seed.
// out: [role][head][key][256].
__global__ void reference_rows_kernel(GqaTcKVHq kv, const std::int32_t* table, int key_begin,
                                      int keys, int window, __nv_bfloat16* out) {
    const long long unit = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (unit >= static_cast<long long>(keys) * kHeads * 2) { return; }
    const int key     = key_begin + static_cast<int>(unit % keys);
    const int head    = static_cast<int>((unit / keys) % kHeads);
    const bool role_v = (unit / keys) / kHeads != 0;
    __nv_bfloat16* dst = out + unit * kD;
    const bool side = kv.residual_k != nullptr &&
                      (key < static_cast<int>(kGqaHqSinkKeys) ||
                       (key >= window - static_cast<int>(kGqaHqRecentKeys) &&
                        hq_ring_slot_valid(kv.ring_valid, key)));
    if (side) {
        const __nv_bfloat16* row =
            hq_residual_row<Geometry>(role_v ? kv.residual_v : kv.residual_k, 0, head, key);
        for (int d = 0; d < kD; ++d) { dst[d] = row[d]; }
        return;
    }
    hq_decode_row_thread(hq_row_codes<Geometry>(role_v ? kv.codes_v : kv.codes_k, table, head, key),
                         hq_row_meta<Geometry>(role_v ? kv.meta_v : kv.meta_k, table, head, key),
                         dst, hq_dither_row_seed(head, key, role_v));
}

__global__ void fill_bf16_kernel(__nv_bfloat16* out, long long n, std::uint32_t salt, float amp) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9u ^ salt;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    out[i] = __float2bfloat16((static_cast<float>(x & 0xFFFFu) / 65535.0f - 0.5f) * amp);
}

double bf16_to_double(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return static_cast<double>(f);
}

// hq_engine_sign, on the host.
double engine_sign(int d) {
    std::uint32_t x = 0x5EED01u ^ (static_cast<std::uint32_t>(d) * 0x9E3779B9u);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    return (x & 1u) ? 1.0 : -1.0;
}

// The codec frame in FP64: forward = Hadamard(signs * x) / 16, inverse = signs * Hadamard(x) / 16
// (hq_fwht256_sign, hq_ifwht256_sign).
void hadamard(double* x) {
    for (int len = 1; len < kD; len <<= 1) {
        for (int i = 0; i < kD; ++i) {
            if ((i & len) == 0) {
                const double a = x[i], b = x[i + len];
                x[i]       = a + b;
                x[i + len] = a - b;
            }
        }
    }
}

void rotate(double* x, const double* signs) {
    for (int d = 0; d < kD; ++d) { x[d] *= signs[d]; }
    hadamard(x);
    for (int d = 0; d < kD; ++d) { x[d] /= 16.0; }
}

void unrotate(double* x, const double* signs) {
    hadamard(x);
    for (int d = 0; d < kD; ++d) { x[d] *= signs[d] / 16.0; }
}

struct Case {
    const char* name;
    int window;  // keys visible to the last column
    int tokens;  // 1..8, the call's width
    bool append;
    bool residual;
};

enum class Route { reference, throughput, ws12 };
const char* route_name(Route r) {
    return r == Route::reference ? "reference" : r == Route::throughput ? "throughput" : "ws12";
}

bool g_ws12_card = false;

template <typename CacheInput>
void launch(Route route, const Case& c, const __nv_bfloat16* q, CacheInput input,
            const std::int32_t* pos, GqaTcKVHq kv, const std::int32_t* table,
            const std::int32_t* table_rows, __nv_bfloat16* acc, float* m, float* l) {
    const dim3 grid(kHeads, kSplits, 1);
    if (route == Route::ws12) {
        ignis_hq_verify::hq_verify_ws_kernel<Geometry, CacheInput>
            <<<grid, ignis_hq_verify::kHqVerifyWsThreads>>>(
                q, input, pos, kv, table, nullptr, table_rows, kLogicalPages, c.tokens, c.tokens, 0,
                kCapacity, kScale, acc, m, l);
    } else if (route == Route::throughput) {
        gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, kTokenTile, 4, true, true, CacheInput,
                                                     GqaTcKVHq, true>
            <<<grid, kGqaHqDecodeThreads>>>(q, input, pos, kv, table, nullptr, table_rows,
                                            kLogicalPages, c.tokens, c.tokens, 0, kCapacity, kScale,
                                            acc, m, l);
    } else {
        gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, kTokenTile, 4, true, true, CacheInput,
                                                     GqaTcKVHq, false>
            <<<grid, kGqaHqDecodeThreads>>>(q, input, pos, kv, table, nullptr, table_rows,
                                            kLogicalPages, c.tokens, c.tokens, 0, kCapacity, kScale,
                                            acc, m, l);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
std::vector<std::uint8_t> to_host(const T* d, std::size_t bytes) {
    std::vector<std::uint8_t> h(bytes);
    CUDA_CHECK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
    return h;
}

void to_device(void* d, const std::vector<std::uint8_t>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
}

std::size_t differing(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { n += a[i] != b[i] ? 1 : 0; }
    return n;
}

// A split's key range, as the kernels split a window over the 85 active splits.
struct Range {
    int begin, end;
};
Range split_range(int window, int split) {
    const int tiles = (window + 31) / 32;
    const int units = (tiles + kSplits - 1) / kSplits;
    const int begin = split * units * 32;
    return {begin, std::min(begin + units * 32, window)};
}

void run_case(const Case& c, const Rows& rows, const __nv_bfloat16* d_rows) {
    const int first_pos = c.window - c.tokens;
    const int fill_limit = c.append ? first_pos : c.window;
    const int own0 = first_pos / kPagedKVPageSize, own1 = (c.window - 1) / kPagedKVPageSize;
    std::vector<std::int32_t> table(kLogicalPages);
    for (int p = 0; p < kLogicalPages; ++p) { table[p] = p % kPool; }
    table[own0] = kPool;
    if (own1 != own0) { table[own1] = kPool + 1; }
    std::vector<std::int32_t> pos(c.tokens);
    for (int j = 0; j < c.tokens; ++j) { pos[j] = first_pos + j; }

    std::int32_t *d_table, *d_pos, *d_table_rows;
    CUDA_CHECK(cudaMalloc(&d_table, table.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_pos, pos.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_table_rows, 4));
    CUDA_CHECK(cudaMemcpy(d_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_table_rows, 0, 4));

    GqaTcKVHq kv{};
    const std::size_t plane_rows = static_cast<std::size_t>(kPhysical) * kHeads * kPagedKVPageSize;
    const std::size_t code_bytes = plane_rows * kHqCodePlaneExtent;
    const std::size_t meta_bytes = plane_rows * kHqMetaPlaneExtent;
    const std::size_t side_bytes = static_cast<std::size_t>(kGqaHqSinkKeys + kGqaHqRecentKeys) * kHeads * kD * 2;
    const std::size_t ring_bytes = kRingWords * 4;
    CUDA_CHECK(cudaMalloc(&kv.codes_k, code_bytes));
    CUDA_CHECK(cudaMalloc(&kv.codes_v, code_bytes));
    CUDA_CHECK(cudaMalloc(&kv.meta_k, meta_bytes));
    CUDA_CHECK(cudaMalloc(&kv.meta_v, meta_bytes));
    CUDA_CHECK(cudaMemset(kv.codes_k, 0, code_bytes));
    CUDA_CHECK(cudaMemset(kv.codes_v, 0, code_bytes));
    CUDA_CHECK(cudaMemset(kv.meta_k, 0, meta_bytes));
    CUDA_CHECK(cudaMemset(kv.meta_v, 0, meta_bytes));
    constexpr int kWarps = 8;
    const std::size_t encode_smem = kWarps * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * 4;
    {
        const long long units = static_cast<long long>(kPhysical) * kPagedKVPageSize * kHeads * 2;
        fill_pages_kernel<<<static_cast<unsigned>((units + kWarps - 1) / kWarps), kWarps * 32, encode_smem>>>(
            d_rows, rows.layers, rows.per_block, own0, own1 != own0 ? own1 : -1, fill_limit, kv);
        CUDA_CHECK(cudaGetLastError());
    }
    if (c.residual) {
        CUDA_CHECK(cudaMalloc(&kv.residual_k, side_bytes));
        CUDA_CHECK(cudaMalloc(&kv.residual_v, side_bytes));
        CUDA_CHECK(cudaMemset(kv.residual_k, 0, side_bytes));
        CUDA_CHECK(cudaMemset(kv.residual_v, 0, side_bytes));
        constexpr int kKeys = static_cast<int>(kGqaHqSinkKeys + kGqaHqRecentKeys);
        fill_side_kernel<<<(kKeys * kHeads * 2 + kWarps - 1) / kWarps, kWarps * 32>>>(
            d_rows, rows.layers, rows.per_block, c.window, fill_limit, kv);
        CUDA_CHECK(cudaGetLastError());
        // Every ring slot valid but the call's own, cleared as a rollback leaves them before
        // the append marks them again.
        std::vector<std::uint32_t> ring(kRingWords, 0xFFFFFFFFu);
        if (c.append) {
            for (int p : pos) {
                const int r = p & (static_cast<int>(kGqaHqRecentKeys) - 1);
                ring[r >> 5] &= ~(1u << (r & 31));
            }
        }
        CUDA_CHECK(cudaMalloc(&kv.ring_valid, ring_bytes));
        CUDA_CHECK(cudaMemcpy(kv.ring_valid, ring.data(), ring_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const long long q_elems  = static_cast<long long>(c.tokens) * Geometry::QHeads * kD;
    const long long kv_elems = static_cast<long long>(c.tokens) * kHeads * kD;
    __nv_bfloat16 *d_q, *d_k_new, *d_v_new;
    CUDA_CHECK(cudaMalloc(&d_q, q_elems * 2));
    CUDA_CHECK(cudaMalloc(&d_k_new, kv_elems * 2));
    CUDA_CHECK(cudaMalloc(&d_v_new, kv_elems * 2));
    fill_bf16_kernel<<<static_cast<unsigned>((q_elems + 255) / 256), 256>>>(d_q, q_elems, 0x85EBCA6Bu, 4.0f);
    fill_bf16_kernel<<<static_cast<unsigned>((kv_elems + 255) / 256), 256>>>(d_k_new, kv_elems, 0xC2B2AE35u, 8.0f);
    fill_bf16_kernel<<<static_cast<unsigned>((kv_elems + 255) / 256), 256>>>(d_v_new, kv_elems, 0x27D4EB2Fu, 2.0f);
    CUDA_CHECK(cudaGetLastError());

    struct State {
        std::vector<std::uint8_t> codes_k, codes_v, meta_k, meta_v, side_k, side_v, ring;
    };
    auto snapshot = [&]() {
        State s;
        s.codes_k = to_host(kv.codes_k, code_bytes);
        s.codes_v = to_host(kv.codes_v, code_bytes);
        s.meta_k  = to_host(kv.meta_k, meta_bytes);
        s.meta_v  = to_host(kv.meta_v, meta_bytes);
        if (c.residual) {
            s.side_k = to_host(kv.residual_k, side_bytes);
            s.side_v = to_host(kv.residual_v, side_bytes);
            s.ring   = to_host(kv.ring_valid, ring_bytes);
        }
        return s;
    };
    auto restore = [&](const State& s) {
        to_device(kv.codes_k, s.codes_k);
        to_device(kv.codes_v, s.codes_v);
        to_device(kv.meta_k, s.meta_k);
        to_device(kv.meta_v, s.meta_v);
        if (c.residual) {
            to_device(kv.residual_k, s.side_k);
            to_device(kv.residual_v, s.side_v);
            to_device(kv.ring_valid, s.ring);
        }
    };
    auto state_differing = [&](const State& a, const State& b) {
        return differing(a.codes_k, b.codes_k) + differing(a.codes_v, b.codes_v) +
               differing(a.meta_k, b.meta_k) + differing(a.meta_v, b.meta_v) +
               differing(a.side_k, b.side_k) + differing(a.side_v, b.side_v) + differing(a.ring, b.ring);
    };

    // The cache a correct call leaves: the history, plus the codec's own encode of the new
    // rows at their positions.
    const State before = snapshot();
    State expected     = before;
    if (c.append) {
        const int units = c.tokens * kHeads * 2;
        reference_append_kernel<<<(units + kWarps - 1) / kWarps, kWarps * 32, encode_smem>>>(
            d_k_new, d_v_new, d_pos, c.tokens, d_table, kv);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        expected = snapshot();
        check(state_differing(before, expected) > 0, std::string(c.name) + ": the reference append wrote nothing");
    }

    // The FP64 reference over five splits, from the expected cache.
    std::vector<double> signs(kD);
    for (int d = 0; d < kD; ++d) { signs[d] = engine_sign(d); }
    const auto q_bits = to_host(d_q, static_cast<std::size_t>(q_elems) * 2);
    std::vector<double> q_rot(static_cast<std::size_t>(q_elems));
    for (int t = 0; t < c.tokens; ++t) {
        for (int h = 0; h < Geometry::QHeads; ++h) {
            double* row = &q_rot[gqa_q_index_host(h, t)];
            for (int d = 0; d < kD; ++d) {
                std::uint16_t b;
                std::memcpy(&b, q_bits.data() + 2 * (gqa_q_index_host(h, t) + d), 2);
                row[d] = bf16_to_double(b);
            }
            rotate(row, signs.data());
        }
    }
    const int checked[] = {0, 1, kSplits / 2, kSplits - 2, kSplits - 1};
    struct SplitRef {
        int split;
        Range range;
        std::vector<double> o;   // [token][q head][256], original frame
        std::vector<char> mass;  // [token][q head]: the split holds a key the column sees
    };
    std::vector<SplitRef> refs;
    for (int split : checked) {
        SplitRef ref{split, split_range(c.window, split), {}, {}};
        const int keys = ref.range.end - ref.range.begin;
        const std::size_t row_elems = static_cast<std::size_t>(keys) * kHeads * 2 * kD;
        __nv_bfloat16* d_ref_rows;
        CUDA_CHECK(cudaMalloc(&d_ref_rows, row_elems * 2));
        const long long threads = static_cast<long long>(keys) * kHeads * 2;
        reference_rows_kernel<<<static_cast<unsigned>((threads + 127) / 128), 128>>>(
            kv, d_table, ref.range.begin, keys, c.window, d_ref_rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint16_t> ref_rows(row_elems);
        CUDA_CHECK(cudaMemcpy(ref_rows.data(), d_ref_rows, row_elems * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_ref_rows));
        ref.o.assign(static_cast<std::size_t>(c.tokens) * Geometry::QHeads * kD, 0.0);
        ref.mass.assign(static_cast<std::size_t>(c.tokens) * Geometry::QHeads, 0);
        auto row_at = [&](bool role_v, int head, int key_l) {
            return &ref_rows[((static_cast<std::size_t>(role_v ? 1 : 0) * kHeads + head) * keys + key_l) * kD];
        };
        std::vector<std::thread> workers;
        for (int head = 0; head < kHeads; ++head) {
            workers.emplace_back([&, head]() {
                std::vector<double> score(keys), o(kD);
                for (int t = 0; t < c.tokens; ++t) {
                    const int seen = std::min(ref.range.end, pos[t] + 1) - ref.range.begin;
                    for (int qh = head * kGroup; qh < (head + 1) * kGroup; ++qh) {
                        if (seen <= 0) { continue; }
                        const double* qr = &q_rot[gqa_q_index_host(qh, t)];
                        double top = -std::numeric_limits<double>::infinity();
                        for (int k = 0; k < seen; ++k) {
                            const std::uint16_t* kr = row_at(false, head, k);
                            double s = 0.0;
                            for (int d = 0; d < kD; ++d) { s += qr[d] * bf16_to_double(kr[d]); }
                            score[k] = s * kScale;
                            top      = std::max(top, score[k]);
                        }
                        std::fill(o.begin(), o.end(), 0.0);
                        double sum = 0.0;
                        for (int k = 0; k < seen; ++k) {
                            const double p = std::exp(score[k] - top);
                            sum += p;
                            const std::uint16_t* vr = row_at(true, head, k);
                            for (int d = 0; d < kD; ++d) { o[d] += p * bf16_to_double(vr[d]); }
                        }
                        for (int d = 0; d < kD; ++d) { o[d] /= sum; }
                        unrotate(o.data(), signs.data());
                        std::copy(o.begin(), o.end(), &ref.o[gqa_q_index_host(qh, t)]);
                        ref.mass[static_cast<std::size_t>(t) * Geometry::QHeads + qh] = 1;
                    }
                }
            });
        }
        for (std::thread& w : workers) { w.join(); }
        refs.push_back(std::move(ref));
    }

    const std::size_t acc_bytes  = static_cast<std::size_t>(kSplits) * c.tokens * Geometry::QHeads * kD * 2;
    const std::size_t stat_bytes = static_cast<std::size_t>(kSplits) * c.tokens * Geometry::QHeads * 4;
    __nv_bfloat16* d_acc;
    float *d_m, *d_l;
    CUDA_CHECK(cudaMalloc(&d_acc, acc_bytes));
    CUDA_CHECK(cudaMalloc(&d_m, stat_bytes));
    CUDA_CHECK(cudaMalloc(&d_l, stat_bytes));

    struct Out {
        std::vector<std::uint8_t> acc, m, l;
        State state;
    };
    auto run = [&](Route route) {
        restore(before);
        CUDA_CHECK(cudaMemset(d_acc, 0xAB, acc_bytes));
        CUDA_CHECK(cudaMemset(d_m, 0xAB, stat_bytes));
        CUDA_CHECK(cudaMemset(d_l, 0xAB, stat_bytes));
        if (c.append) {
            launch(route, c, d_q, GqaAppendInput{d_k_new, d_v_new}, d_pos, kv, d_table, d_table_rows, d_acc, d_m, d_l);
        } else {
            launch(route, c, d_q, GqaCachedInput{}, d_pos, kv, d_table, d_table_rows, d_acc, d_m, d_l);
        }
        Out o{to_host(d_acc, acc_bytes), to_host(d_m, stat_bytes), to_host(d_l, stat_bytes), snapshot()};
        return o;
    };

    std::vector<Route> routes = {Route::reference, Route::throughput};
    if (g_ws12_card) { routes.push_back(Route::ws12); }
    std::vector<Out> outs;
    for (Route route : routes) {
        Out out = run(route);
        double worst = 0.0;
        int worst_split = -1;
        std::size_t mass_mismatch = 0;
        for (const SplitRef& ref : refs) {
            for (int t = 0; t < c.tokens; ++t) {
                for (int qh = 0; qh < Geometry::QHeads; ++qh) {
                    float l;
                    std::memcpy(&l, out.l.data() + 4 * gqa_stat_index_host(qh, t, ref.split, c.tokens), 4);
                    const bool mass = ref.mass[static_cast<std::size_t>(t) * Geometry::QHeads + qh] != 0;
                    if (!mass || !(l > 0.0f) || !std::isfinite(l)) {
                        mass_mismatch += (mass != (l > 0.0f && std::isfinite(l))) ? 1 : 0;
                        continue;
                    }
                    double err = 0.0, norm = 0.0;
                    for (int d = 0; d < kD; ++d) {
                        std::uint16_t b;
                        std::memcpy(&b, out.acc.data() + 2 * gqa_acc_index_host(qh, d, t, ref.split, c.tokens), 2);
                        const double got  = bf16_to_double(b) / l;
                        const double want = ref.o[gqa_q_index_host(qh, t) + d];
                        err += (got - want) * (got - want);
                        norm += want * want;
                    }
                    const double rel = std::sqrt(err) / std::max(std::sqrt(norm), 1e-30);
                    if (!(rel <= worst)) {
                        worst       = std::isnan(rel) ? std::numeric_limits<double>::infinity() : rel;
                        worst_split = ref.split;
                    }
                }
            }
        }
        const std::size_t cache_diff = state_differing(out.state, expected);
        std::printf("%-58s %-10s worst relative L2 vs FP64 %.3e (split %d), columns with mass "
                    "mismatched %zu, cache bytes differing from the codec's append %zu\n",
                    c.name, route_name(route), worst, worst_split, mass_mismatch, cache_diff);
        check(worst <= kTolerance && mass_mismatch == 0,
              std::string(c.name) + ": the " + route_name(route) + " route's partials are off the FP64 reference");
        check(cache_diff == 0, std::string(c.name) + ": the " + route_name(route) +
                                   " route left cache bytes the codec's append does not write");
        outs.push_back(std::move(out));
    }
    for (std::size_t i = 1; i < outs.size(); ++i) {
        const std::size_t d = differing(outs[0].acc, outs[i].acc) + differing(outs[0].m, outs[i].m) +
                              differing(outs[0].l, outs[i].l);
        std::printf("%-58s %-10s partial bytes differing from the reference route: %zu of %zu\n", c.name,
                    route_name(routes[i]), d, outs[0].acc.size() + 2 * outs[0].m.size());
        check(d == 0, std::string(c.name) + ": the " + route_name(routes[i]) +
                          " route's partials differ from the reference route's (ADR 0093)");
    }

    CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_table_rows));
    CUDA_CHECK(cudaFree(kv.codes_k));
    CUDA_CHECK(cudaFree(kv.codes_v));
    CUDA_CHECK(cudaFree(kv.meta_k));
    CUDA_CHECK(cudaFree(kv.meta_v));
    if (c.residual) {
        CUDA_CHECK(cudaFree(kv.residual_k));
        CUDA_CHECK(cudaFree(kv.residual_v));
        CUDA_CHECK(cudaFree(kv.ring_valid));
    }
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k_new));
    CUDA_CHECK(cudaFree(d_v_new));
    CUDA_CHECK(cudaFree(d_acc));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_l));
}

} // namespace

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    g_ws12_card = prop.major >= 12;
    if (!g_ws12_card) {
        std::printf("compute capability %d.%d: the ws12 route is Blackwell's and does not run on "
                    "this card; the vendored routes only\n",
                    prop.major, prop.minor);
    }
    const Rows rows = load_rows(IGNIS_HQ_KV_FIXTURE_PATH);
    __nv_bfloat16* d_rows;
    CUDA_CHECK(cudaMalloc(&d_rows, rows.bits.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_rows, rows.bits.data(), rows.bits.size() * 2, cudaMemcpyHostToDevice));

    const std::vector<Case> cases = {
        // The controls: 128 pages a split, the most the old table held.
        {"696,320 keys (128 pages a split), width 8, append, residual", 696320, 8, true, true},
        {"696,320 keys (128 pages a split), width 1", 696320, 1, false, false},
        {"696,321 keys (129 pages a split), width 8, append", 696321, 8, true, false},
        {"696,321 keys (129 pages a split), width 1, residual", 696321, 1, false, true},
        {"917,504 keys (169 pages a split), width 8, append, residual", 917504, 8, true, true},
        {"917,504 keys (169 pages a split), width 1, append", 917504, 1, true, false},
        {"1,048,576 keys (193 pages a split), width 8, append", 1048576, 8, true, false},
        {"1,048,576 keys (193 pages a split), width 8, residual", 1048576, 8, false, true},
        {"1,048,576 keys (193 pages a split), width 1, append, residual", 1048576, 1, true, true},
    };
    for (const Case& c : cases) { run_case(c, rows, d_rows); }
    CUDA_CHECK(cudaFree(d_rows));

    if (g_failed != 0) {
        std::fprintf(stderr, "test_hq_long_window: %d failures\n", g_failed);
        return EXIT_FAILURE;
    }
    std::printf("test_hq_long_window: ALL PASSED (%zu shapes, routes held to the FP64 reference: %s)\n",
                cases.size(), g_ws12_card ? "reference, throughput, ws12" : "reference, throughput");
    return 0;
}
