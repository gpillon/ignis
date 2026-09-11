// P4-03 (GitHub #119): the hq-e8-2b codec's measured-tolerance test at this
// model's real 27B geometry -- OURS, not vendored. The vendored oracle
// (kernel/vendor/tools/test_kv/test_hq_codec.cu, built as
// ignis_hq_codec_test) already proves the codec's internal correctness
// against an FP64 brute-force nearest-point search on a synthetic Gaussian
// corpus with random signs and a fixed (kv_head=0, role=K) seed; this test
// does not repeat any of that. It answers the part of the ticket the
// vendored oracle cannot: what the codec's round-trip error actually IS on
// real captured K/V rows at this model's 4-kv-head / 256-head-dim geometry,
// broken down by depth (GQA layer ordinal) and role (K vs V), and it derives
// its enforced tolerance FROM that measurement rather than copying one from
// elsewhere -- GitHub #96 is the project's own record of what copying a
// tolerance costs.
//
// Fixture: kernel/tests/fixtures/hq_kv_rows_27b.bin (+ its
// .provenance.json sidecar), captured once from a real prefill of the real
// qwen3.8-27b-nvfp4full-v2 artifact by
// crates/core/tests/hq_kv_fixture_capture_gpu.rs. A missing, truncated or
// corrupt fixture is a HARD FAILURE (ADR 0006 / docs/agents/testing.md:
// compute tests never skip), naming that capture test as how to regenerate
// it. The path is resolved from IGNIS_HQ_KV_FIXTURE_PATH, a compile
// definition kernel/tests/CMakeLists.txt sets from
// CMAKE_CURRENT_SOURCE_DIR -- never a relative path, which would depend on
// whatever directory CTest happens to run from.
//
// Uses the engine's REAL rotation (hq_engine_sign / hq_engine_signs_fill,
// not the vendored oracle's random per-run sign vector) and REAL per-row
// dither seeds (hq_dither_row_seed(kv_head, position, role_v), each row
// seeded with the true (kv_head, position, role) triple the fixture
// recorded it under, reconstructed from the fixture's documented row
// order) -- a measurement that sets a shipped tolerance must be taken
// through the exact transform and seeding the production engine applies,
// not a stand-in.
//
// Follows the vendored oracle's check()/g_failed convention so the two read
// alike; no SKIP_RETURN_CODE anywhere this file is registered
// (kernel/tests/CMakeLists.txt).

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ops/kernel/hq_codec.cuh"

using namespace ninfer::ops;

#ifndef IGNIS_HQ_KV_FIXTURE_PATH
#error "IGNIS_HQ_KV_FIXTURE_PATH must be defined by kernel/tests/CMakeLists.txt"
#endif

namespace {

int g_failed = 0;

void check(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++g_failed;
    }
}

double bf16_bits_to_float(std::uint16_t bits) {
    __nv_bfloat16 v;
    std::memcpy(&v, &bits, sizeof(bits));
    return __bfloat162float(v);
}

std::uint16_t float_to_bf16_bits(float f) {
    const __nv_bfloat16 v = __float2bfloat16(f);
    std::uint16_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

#define CUDA_CHECK(expr)                                                                        \
    do {                                                                                        \
        const cudaError_t _err = (expr);                                                        \
        if (_err != cudaSuccess) {                                                              \
            std::fprintf(stderr, "FATAL: %s failed: %s\n", #expr, cudaGetErrorString(_err));    \
            std::exit(EXIT_FAILURE);                                                            \
        }                                                                                        \
    } while (0)

// ---- fixture ----------------------------------------------------------------

// Corruption/version detection, not cryptography -- mirrors
// crates/core/tests/hq_kv_fixture_capture_gpu.rs's fnv1a64 exactly.
std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t n) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

struct Fixture {
    std::uint32_t head_dim       = 0;
    std::uint32_t kv_heads       = 0;
    std::uint32_t role_count     = 0;
    std::vector<std::uint32_t> layer_ordinals;
    std::uint32_t first_position = 0;
    std::uint32_t rows_per_block = 0;
    std::uint32_t total_rows     = 0;
    // total_rows * head_dim raw bf16 bit patterns, row order:
    // for layer in layer_ordinals { for role in [K,V] { for kv_head in
    // 0..kv_heads { for position in first_position..+rows_per_block {
    // head_dim x u16 } } } } -- documented in the fixture's own
    // .provenance.json "row_order" field.
    std::vector<std::uint16_t> rows;
};

[[noreturn]] void fatal_fixture(const char* path, const char* why) {
    std::fprintf(stderr,
                "FATAL: fixture %s is invalid: %s\n"
                "Regenerate it: stop ninfer, then `scripts/gpu-profile.ps1` (or, scoped, "
                "`cargo test -p ignis-core --features cuda -- --ignored "
                "hq_kv_fixture_capture_gpu --test-threads=1` after a `scripts/gpu-preflight.ps1` "
                "pass) -- see crates/core/tests/hq_kv_fixture_capture_gpu.rs.\n",
                path, why);
    std::exit(EXIT_FAILURE);
}

// Reads and fully validates the fixture: magic, format version, head_dim,
// the declared row count's internal consistency, the file size against the
// header's own accounting, and the FNV-1a64 payload checksum. Every failure
// path here is fatal (ADR 0006): a missing or corrupt fixture proves
// nothing about the codec, so it must never read as a skip.
Fixture load_fixture(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::fprintf(stderr,
                    "FATAL: cannot open fixture %s (%s)\n"
                    "Regenerate it: stop ninfer, then `scripts/gpu-profile.ps1` -- see "
                    "crates/core/tests/hq_kv_fixture_capture_gpu.rs.\n",
                    path, std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    std::vector<std::uint8_t> data;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size < 0) {
        std::fclose(f);
        fatal_fixture(path, "cannot determine file size");
    }
    data.resize(static_cast<std::size_t>(size));
    const std::size_t got = data.empty() ? 0 : std::fread(data.data(), 1, data.size(), f);
    std::fclose(f);
    if (got != data.size()) { fatal_fixture(path, "short read"); }

    constexpr std::size_t kFixedHeaderBytes = 8 /*magic*/ + 4 * 6 /*version..layer_count*/;
    if (data.size() < kFixedHeaderBytes) { fatal_fixture(path, "truncated header"); }
    if (std::memcmp(data.data(), "IGNHQKV1", 8) != 0) { fatal_fixture(path, "bad magic"); }

    auto read_u32 = [&](std::size_t off) {
        std::uint32_t v;
        std::memcpy(&v, data.data() + off, 4);
        return v;
    };

    std::size_t off              = 8;
    const std::uint32_t version  = read_u32(off);
    off += 4;
    if (version != 1) { fatal_fixture(path, "unsupported format_version (expected 1)"); }

    Fixture fx;
    fx.head_dim   = read_u32(off); off += 4;
    fx.kv_heads   = read_u32(off); off += 4;
    fx.role_count = read_u32(off); off += 4;
    const std::uint32_t layer_count = read_u32(off); off += 4;

    const std::size_t tail_bytes = 4u * layer_count + 4 * 3 + 8;
    if (data.size() < off + tail_bytes) { fatal_fixture(path, "truncated header (layers/tail)"); }

    fx.layer_ordinals.resize(layer_count);
    for (std::uint32_t i = 0; i < layer_count; ++i) {
        fx.layer_ordinals[i] = read_u32(off);
        off += 4;
    }
    fx.first_position = read_u32(off); off += 4;
    fx.rows_per_block = read_u32(off); off += 4;
    fx.total_rows     = read_u32(off); off += 4;
    std::uint64_t checksum;
    std::memcpy(&checksum, data.data() + off, 8);
    off += 8;

    if (fx.head_dim != static_cast<std::uint32_t>(kHqHeadDim)) {
        fatal_fixture(path, "head_dim does not match kHqHeadDim (fixture was captured for a "
                            "different geometry)");
    }
    const std::uint64_t expected_rows = static_cast<std::uint64_t>(layer_count) * fx.role_count *
                                        fx.kv_heads * fx.rows_per_block;
    if (expected_rows != fx.total_rows) {
        fatal_fixture(path, "total_rows does not match layer_count * role_count * kv_heads * "
                            "rows_per_block");
    }
    const std::size_t payload_bytes = static_cast<std::size_t>(fx.total_rows) * fx.head_dim * 2;
    if (data.size() != off + payload_bytes) {
        fatal_fixture(path, "file size does not match the header's declared payload size");
    }
    const std::uint64_t actual_checksum = fnv1a64(data.data() + off, payload_bytes);
    if (actual_checksum != checksum) {
        fatal_fixture(path, "FNV-1a64 checksum mismatch over the row payload (corrupted)");
    }

    fx.rows.resize(static_cast<std::size_t>(fx.total_rows) * fx.head_dim);
    std::memcpy(fx.rows.data(), data.data() + off, payload_bytes);
    return fx;
}

// Row index -> (layer_index, role, kv_head, position). ONE canonical
// definition, used identically on host (for grouping the measurement) and
// on device (for seeding); the fixture's row order guarantees this matches
// what was actually captured.
struct RowId {
    int layer_index;
    int role; // 0 = K, 1 = V
    int kv_head;
    int position;
};

__host__ __device__ inline RowId decompose_row(int r, int role_count, int kv_heads,
                                               int rows_per_block, int first_position) {
    const int block_size = kv_heads * rows_per_block;
    const int per_layer  = role_count * block_size;
    const int layer_index = r / per_layer;
    const int rem1        = r % per_layer;
    const int role         = rem1 / block_size;
    const int rem2         = rem1 % block_size;
    const int kv_head       = rem2 / rows_per_block;
    const int pos_off       = rem2 % rows_per_block;
    return RowId{layer_index, role, kv_head, first_position + pos_off};
}

// ---- device kernels -----------------------------------------------------------

__global__ void fill_engine_signs_kernel(std::int8_t* signs) { hq_engine_signs_fill(signs); }

__global__ void encode_kv_rows_kernel(const __nv_bfloat16* rows, const std::int8_t* signs,
                                      std::uint8_t* codes, std::uint8_t* meta, int n_rows,
                                      int role_count, int kv_heads, int rows_per_block,
                                      int first_position) {
    extern __shared__ float smem[];
    const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp >= n_rows) { return; }
    const RowId id = decompose_row(warp, role_count, kv_heads, rows_per_block, first_position);
    float* u             = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms  = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
    hq_encode_row_warp(rows + static_cast<std::size_t>(warp) * kHqHeadDim, signs, 0, u, syms,
                       codes + static_cast<std::size_t>(warp) * kHqRowBudgetBytes,
                       meta + static_cast<std::size_t>(warp) * kHqMetaBytes,
                       hq_dither_row_seed(id.kv_head, id.position, id.role == 1));
}

__global__ void decode_kv_rows_kernel(const std::uint8_t* codes, const std::uint8_t* meta,
                                      __nv_bfloat16* out, int n_rows, int role_count,
                                      int kv_heads, int rows_per_block, int first_position) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) { return; }
    const RowId id = decompose_row(i, role_count, kv_heads, rows_per_block, first_position);
    hq_decode_row_thread(codes + static_cast<std::size_t>(i) * kHqRowBudgetBytes,
                         meta + static_cast<std::size_t>(i) * kHqMetaBytes,
                         out + static_cast<std::size_t>(i) * kHqHeadDim,
                         hq_dither_row_seed(id.kv_head, id.position, id.role == 1));
}

// Un-rotate a decoded (rotated-frame) row back to the original frame, the
// same inverse transform a real consumer applies once per output row
// (hq_codec.cuh's own docs) -- reuses the vendored hq_ifwht256_sign exactly,
// no reimplementation.
__global__ void unrotate_rows_kernel(const __nv_bfloat16* rotated, const std::int8_t* signs,
                                     __nv_bfloat16* original, int n_rows) {
    const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp >= n_rows) { return; }
    const int lane = static_cast<int>(threadIdx.x & 31u);
    float reg[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        reg[s] = __bfloat162float(
            rotated[static_cast<std::size_t>(warp) * kHqHeadDim + s * 32 + lane]);
    }
    hq_ifwht256_sign(reg, signs, 0, lane);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        original[static_cast<std::size_t>(warp) * kHqHeadDim + s * 32 + lane] =
            __float2bfloat16(reg[s]);
    }
}

// Same as encode_kv_rows_kernel, but for the deliberately heavy-tailed arm
// below: that corpus is a small, separately indexed buffer, each row
// carrying the true (kv_head, position, role) triple of the real row it was
// copied and amplified from (arrays, not a formula).
__global__ void encode_rows_with_seeds_kernel(const __nv_bfloat16* rows, const std::int8_t* signs,
                                              std::uint8_t* codes, std::uint8_t* meta, int n_rows,
                                              const int* kv_heads_in, const int* positions_in,
                                              const std::uint8_t* roles_in) {
    extern __shared__ float smem[];
    const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp >= n_rows) { return; }
    float* u            = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
    hq_encode_row_warp(rows + static_cast<std::size_t>(warp) * kHqHeadDim, signs, 0, u, syms,
                       codes + static_cast<std::size_t>(warp) * kHqRowBudgetBytes,
                       meta + static_cast<std::size_t>(warp) * kHqMetaBytes,
                       hq_dither_row_seed(kv_heads_in[warp], positions_in[warp],
                                          roles_in[warp] != 0));
}

// ---- byte-occupancy check -----------------------------------------------------
//
// Every row's code plane must be exactly kHqRowBudgetBytes (64) and its meta
// plane exactly kHqMetaBytes (8) -- guaranteed by the fixed-stride buffers
// themselves, so the property actually worth checking is the documented
// stored-stream invariant (hq_codec.cuh's header comment): the tail of the
// row past its used bits is zero. That is a genuine per-row occupancy
// property distinct from the bit-count check the vendored oracle already
// does ([3] budget invariant there checks only that used_bits <= 512).
//
// Checked at 32-bit WORD granularity, not raw byte granularity: the stream
// is word-addressed everywhere it is produced or consumed (HqBitReader,
// hq_decode_row_group's two-word lane windows, the packer's
// `atomicOr(&syms[pos >> 5], ...)`), and each word's 32 bits are packed
// MSB-first while the codes buffer is a little-endian byte view of those
// same uint32 values -- so byte index and bitstream position are NOT the
// same sequence (byte 0 of a word holds that word's LAST 8 stream bits, not
// its first). A raw byte-index boundary check silently compares the wrong
// bytes; comparing whole words to zero sidesteps that reversal entirely
// (endianness cannot matter to an equality-with-zero test) while still
// checking real content, not just the bit count.
bool row_byte_occupancy_ok(const std::uint8_t* meta_row, const std::uint8_t* codes_row) {
    const unsigned used_bits =
        static_cast<unsigned>(meta_row[3]) | (static_cast<unsigned>(meta_row[4] & 3) << 8);
    if (used_bits == 0 || used_bits > 512u) { return false; }
    const unsigned tail_start_word = (used_bits + 31) / 32;
    const auto* words              = reinterpret_cast<const std::uint32_t*>(codes_row);
    for (unsigned w = tail_start_word; w < static_cast<unsigned>(kHqRowBudgetBytes) / 4; ++w) {
        if (words[w] != 0u) { return false; }
    }
    return true;
}

bool row_is_terminal_fallback(const std::uint8_t* codes_row) {
    for (int b = 0; b < 32; ++b) {
        if (codes_row[b] != 0xFFu) { return false; }
    }
    return true;
}

double median_of(std::vector<double> v) {
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct GroupStats {
    int layer_ordinal = 0;
    int role          = 0;
    int rows          = 0;
    int escalated     = 0;
    int fallback      = 0;
    double sig        = 0.0;
    double noise      = 0.0;
    double cos_xy     = 0.0;
    double cos_xx     = 0.0;
    double cos_yy     = 0.0;
    std::vector<double> rel_l2;
};

} // namespace

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    const Fixture fx = load_fixture(IGNIS_HQ_KV_FIXTURE_PATH);
    const int layer_count = static_cast<int>(fx.layer_ordinals.size());
    const int role_count  = static_cast<int>(fx.role_count);
    const int kv_heads    = static_cast<int>(fx.kv_heads);
    const int rows_per_block = static_cast<int>(fx.rows_per_block);
    const int first_position = static_cast<int>(fx.first_position);
    const int n_rows          = static_cast<int>(fx.total_rows);
    std::printf("fixture: %s -- %d rows (%d layers x %d roles x %d kv_heads x %d positions), "
                "head_dim %u\n",
                IGNIS_HQ_KV_FIXTURE_PATH, n_rows, layer_count, role_count, kv_heads,
                rows_per_block, fx.head_dim);

    // ---- engine signs, real rows on device --------------------------------
    std::int8_t* d_signs;
    CUDA_CHECK(cudaMalloc(&d_signs, kHqHeadDim));
    fill_engine_signs_kernel<<<1, 256>>>(d_signs);
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* d_rows;
    CUDA_CHECK(cudaMalloc(&d_rows, fx.rows.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_rows, fx.rows.data(), fx.rows.size() * 2, cudaMemcpyHostToDevice));

    std::uint8_t* d_codes;
    std::uint8_t* d_meta;
    CUDA_CHECK(cudaMalloc(&d_codes, static_cast<std::size_t>(n_rows) * kHqRowBudgetBytes));
    CUDA_CHECK(cudaMalloc(&d_meta, static_cast<std::size_t>(n_rows) * kHqMetaBytes));

    constexpr int kWarpsPerBlock = 8;
    const std::size_t smem_bytes =
        kWarpsPerBlock * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * 4;
    encode_kv_rows_kernel<<<(n_rows + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32,
                           smem_bytes>>>(d_rows, d_signs, d_codes, d_meta, n_rows, role_count,
                                        kv_heads, rows_per_block, first_position);
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* d_decoded_rotated;
    CUDA_CHECK(cudaMalloc(&d_decoded_rotated, fx.rows.size() * 2));
    decode_kv_rows_kernel<<<(n_rows + 255) / 256, 256>>>(d_codes, d_meta, d_decoded_rotated,
                                                        n_rows, role_count, kv_heads,
                                                        rows_per_block, first_position);
    CUDA_CHECK(cudaDeviceSynchronize());

    __nv_bfloat16* d_reconstructed;
    CUDA_CHECK(cudaMalloc(&d_reconstructed, fx.rows.size() * 2));
    unrotate_rows_kernel<<<(n_rows + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32>>>(
        d_decoded_rotated, d_signs, d_reconstructed, n_rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::uint8_t> hmeta(static_cast<std::size_t>(n_rows) * kHqMetaBytes);
    std::vector<std::uint8_t> hcodes(static_cast<std::size_t>(n_rows) * kHqRowBudgetBytes);
    std::vector<std::uint16_t> hreconstructed(fx.rows.size());
    CUDA_CHECK(cudaMemcpy(hmeta.data(), d_meta, hmeta.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hcodes.data(), d_codes, hcodes.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hreconstructed.data(), d_reconstructed, hreconstructed.size() * 2,
                         cudaMemcpyDeviceToHost));

    // ---- byte-occupancy check (every row, escalated included) -------------
    int byte_bad = 0, byte_bad_escalated = 0, escalated_checked = 0;
    for (int r = 0; r < n_rows; ++r) {
        const std::uint8_t* meta_row  = &hmeta[static_cast<std::size_t>(r) * kHqMetaBytes];
        const std::uint8_t* codes_row = &hcodes[static_cast<std::size_t>(r) * kHqRowBudgetBytes];
        const bool escalated          = ((meta_row[2] >> 4) & 3u) != 0;
        if (escalated) { ++escalated_checked; }
        if (!row_byte_occupancy_ok(meta_row, codes_row)) {
            ++byte_bad;
            if (escalated) { ++byte_bad_escalated; }
        }
    }
    std::printf("[budget] byte-occupancy check: %d/%d rows exactly %d+%d bytes with correct "
                "tail-zero padding (%d of them escalated; %d escalated rows failed)\n",
                n_rows - byte_bad, n_rows, kHqRowBudgetBytes, kHqMetaBytes, escalated_checked,
                byte_bad_escalated);
    check(byte_bad == 0, "a row's code plane is not exactly the fixed budget with zero tail "
                        "padding past ceil(used_bits/8)");

    // ---- per-(layer, role) round-trip measurement --------------------------
    std::vector<GroupStats> groups(static_cast<std::size_t>(layer_count) * role_count);
    for (int li = 0; li < layer_count; ++li) {
        for (int ro = 0; ro < role_count; ++ro) {
            GroupStats& g   = groups[static_cast<std::size_t>(li) * role_count + ro];
            g.layer_ordinal = static_cast<int>(fx.layer_ordinals[li]);
            g.role          = ro;
        }
    }
    int fallback_total = 0;
    for (int r = 0; r < n_rows; ++r) {
        const RowId id = decompose_row(r, role_count, kv_heads, rows_per_block, first_position);
        GroupStats& g  = groups[static_cast<std::size_t>(id.layer_index) * role_count + id.role];
        ++g.rows;
        const std::uint8_t* meta_row  = &hmeta[static_cast<std::size_t>(r) * kHqMetaBytes];
        const std::uint8_t* codes_row = &hcodes[static_cast<std::size_t>(r) * kHqRowBudgetBytes];
        if (((meta_row[2] >> 4) & 3u) != 0) { ++g.escalated; }
        if (row_is_terminal_fallback(codes_row)) {
            ++g.fallback;
            ++fallback_total;
        }

        double err2 = 0.0, sig2 = 0.0;
        for (int d = 0; d < kHqHeadDim; ++d) {
            const double orig  = bf16_bits_to_float(fx.rows[static_cast<std::size_t>(r) * kHqHeadDim + d]);
            const double recon = bf16_bits_to_float(hreconstructed[static_cast<std::size_t>(r) * kHqHeadDim + d]);
            const double e     = recon - orig;
            err2 += e * e;
            sig2 += orig * orig;
            g.cos_xy += orig * recon;
            g.cos_xx += orig * orig;
            g.cos_yy += recon * recon;
        }
        g.sig += sig2;
        g.noise += err2;
        g.rel_l2.push_back(sig2 > 0.0 ? std::sqrt(err2 / sig2) : 0.0);
    }
    check(fallback_total == 0, "a real captured row hit the terminal fallback (silently zeroed "
                              "K/V) instead of the alpha-halving rescue");

    std::printf("\n[per-layer/per-role round-trip error, real captured rows]\n");
    std::printf("%-6s %-4s %6s %10s %10s %10s %10s %6s %6s\n", "layer", "role", "rows",
               "med_relL2", "max_relL2", "cosine", "snr_dB", "esc", "fbk");
    double overall_sig = 0.0, overall_noise = 0.0, overall_cos_xy = 0.0, overall_cos_xx = 0.0,
          overall_cos_yy = 0.0;
    double worst_group_median = 0.0, worst_group_max = 0.0;
    double min_group_cosine = 1.0, min_group_snr = 1e300;
    for (auto& g : groups) {
        const double med   = median_of(g.rel_l2);
        const double worst = g.rel_l2.empty() ? 0.0 : *std::max_element(g.rel_l2.begin(), g.rel_l2.end());
        const double cosn  = g.cos_xy / (std::sqrt(g.cos_xx) * std::sqrt(g.cos_yy) + 1e-300);
        const double snr   = 10.0 * std::log10(g.sig / g.noise);
        std::printf("%-6d %-4s %6d %10.6f %10.6f %10.6f %10.2f %6d %6d\n", g.layer_ordinal,
                   g.role == 0 ? "K" : "V", g.rows, med, worst, cosn, snr, g.escalated,
                   g.fallback);
        worst_group_median = std::max(worst_group_median, med);
        worst_group_max     = std::max(worst_group_max, worst);
        min_group_cosine     = std::min(min_group_cosine, cosn);
        min_group_snr         = std::min(min_group_snr, snr);
        overall_sig += g.sig;
        overall_noise += g.noise;
        overall_cos_xy += g.cos_xy;
        overall_cos_xx += g.cos_xx;
        overall_cos_yy += g.cos_yy;
    }
    const double overall_cosine = overall_cos_xy / (std::sqrt(overall_cos_xx) * std::sqrt(overall_cos_yy) + 1e-300);
    const double overall_snr    = 10.0 * std::log10(overall_sig / overall_noise);
    std::printf("%-6s %-4s %6d %10s %10s %10.6f %10.2f\n", "ALL", "-", n_rows, "-", "-",
               overall_cosine, overall_snr);
    std::printf("[summary] worst per-group median relL2 %.6f, worst per-group max relL2 %.6f, "
                "min per-group cosine %.6f, min per-group SNR %.2f dB\n",
                worst_group_median, worst_group_max, min_group_cosine, min_group_snr);

    // ---- derived tolerance --------------------------------------------------
    //
    // Measured 2026-09-12 on kernel/tests/fixtures/hq_kv_rows_27b.bin
    // (provenance: capture_date_utc 2026-09-11T21:53:42Z, bin_sha256
    // 7513c00634557473ebcb76b95e40e733a7f213776d0b15d43bacbec6f1f764f3), 8
    // groups (GQA layer ordinals 0/5/10/15 x K/V), 1024 real captured rows
    // per group:
    //   worst per-group median relative L2 error: 0.369634 (layer 10, K)
    //   worst per-group max relative L2 error:    0.773022 (layer 0, V)
    //   min per-group original-frame cosine:      0.934583 (layer 0, K)
    //   min per-group SNR:                        8.40 dB  (layer 0, K)
    // Margin: ~22% headroom on the median bound, ~16% on the max bound
    // (rounded outward to a clean 2nd decimal -- max relative error is
    // already close to "no correlation" territory, so it gets a tighter
    // relative margin than the median), ~0.02 absolute off the cosine
    // floor, ~1.9 dB off the SNR floor. Generous enough to absorb ordinary
    // run-to-run FP32/GPU nondeterminism, tight enough to trip on an actual
    // regression. This is a tripwire on what this codec measurably does to
    // THIS model's real KV rows through THIS fixture -- never a constant
    // copied from the vendored synthetic oracle (whose 0.93 cosine floor is
    // a different corpus, different signs, different seeding) or from any
    // other test (GitHub #96 is the project's own record of what copying a
    // tolerance costs).
    constexpr double kMaxGroupMedianRelL2 = 0.45;
    constexpr double kMaxGroupMaxRelL2    = 0.90;
    constexpr double kMinGroupCosine      = 0.91;
    constexpr double kMinGroupSnrDb       = 6.5;
    check(worst_group_median <= kMaxGroupMedianRelL2,
         "worst per-group median relative L2 error exceeded the measured tolerance");
    check(worst_group_max <= kMaxGroupMaxRelL2,
         "worst per-group max relative L2 error exceeded the measured tolerance");
    check(min_group_cosine >= kMinGroupCosine,
         "a group's original-frame cosine fell below the measured tolerance");
    check(min_group_snr >= kMinGroupSnrDb,
         "a group's SNR fell below the measured tolerance");

    // ---- deliberate heavy-tailed arm: escalation must fire, budget must ----
    // ---- still hold, terminal fallback must not ----------------------------
    //
    // Real rows may or may not escalate on their own (the census above is
    // honest either way); this arm does not rely on that. It takes a
    // spread sample of REAL rows and amplifies a handful of their raw
    // channels before the same real encode path, so the row keeps its real
    // identity (kv_head/position/role, hence its real dither seed) but its
    // energy is now heavy-tailed enough that escalation must engage.
    // Amplifying raw channels (not rotated-frame words) is deliberate: the
    // Hadamard rotation is global, so a few huge raw inputs raise every
    // rotated coordinate's typical magnitude, which is what actually
    // stresses the row-wide bit budget rather than one lattice word.
    {
        constexpr int kHeavySamples    = 64;
        // Additive, row-norm-relative injection rather than a multiplicative
        // amplification of whatever value already sits at a fixed channel:
        // a real row's value at any single fixed index can be tiny (or
        // near-zero), making a pure multiplier a no-op regardless of its
        // size. Adding +/- kHeavyFactor * (this row's own L2 norm) to a
        // handful of channels instead guarantees a substantial, controlled
        // energy injection scaled to what "substantial" means for THIS row,
        // whether it is a layer-0 row (norm ~tens) or a layer-15 V row
        // (norm ~thousands, per the owner's own decode of the fixture).
        constexpr double kHeavyFactor  = 3.0;
        constexpr int kHeavyChannels[8] = {0, 32, 64, 96, 128, 160, 192, 224};
        const int stride                = std::max(1, n_rows / kHeavySamples);

        std::vector<std::uint16_t> heavy_rows(static_cast<std::size_t>(kHeavySamples) * kHqHeadDim);
        std::vector<int> heavy_kv_heads(kHeavySamples), heavy_positions(kHeavySamples);
        std::vector<std::uint8_t> heavy_roles(kHeavySamples);
        for (int i = 0; i < kHeavySamples; ++i) {
            const int r    = std::min(n_rows - 1, i * stride);
            const RowId id = decompose_row(r, role_count, kv_heads, rows_per_block, first_position);
            heavy_kv_heads[i]  = id.kv_head;
            heavy_positions[i] = id.position;
            heavy_roles[i]     = static_cast<std::uint8_t>(id.role);
            double row_norm = 0.0;
            for (int d = 0; d < kHqHeadDim; ++d) {
                const std::uint16_t bits = fx.rows[static_cast<std::size_t>(r) * kHqHeadDim + d];
                heavy_rows[static_cast<std::size_t>(i) * kHqHeadDim + d] = bits;
                const double v = bf16_bits_to_float(bits);
                row_norm += v * v;
            }
            row_norm = std::sqrt(row_norm);
            int sign = 1;
            for (int c : kHeavyChannels) {
                const double v = bf16_bits_to_float(heavy_rows[static_cast<std::size_t>(i) * kHqHeadDim + c]);
                heavy_rows[static_cast<std::size_t>(i) * kHqHeadDim + c] =
                    float_to_bf16_bits(static_cast<float>(v + sign * kHeavyFactor * row_norm));
                sign = -sign;
            }
        }

        __nv_bfloat16* d_heavy_rows;
        int* d_heavy_kv_heads;
        int* d_heavy_positions;
        std::uint8_t* d_heavy_roles;
        std::uint8_t* d_heavy_codes;
        std::uint8_t* d_heavy_meta;
        CUDA_CHECK(cudaMalloc(&d_heavy_rows, heavy_rows.size() * 2));
        CUDA_CHECK(cudaMalloc(&d_heavy_kv_heads, kHeavySamples * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_heavy_positions, kHeavySamples * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_heavy_roles, kHeavySamples));
        CUDA_CHECK(cudaMalloc(&d_heavy_codes, static_cast<std::size_t>(kHeavySamples) * kHqRowBudgetBytes));
        CUDA_CHECK(cudaMalloc(&d_heavy_meta, static_cast<std::size_t>(kHeavySamples) * kHqMetaBytes));
        CUDA_CHECK(cudaMemcpy(d_heavy_rows, heavy_rows.data(), heavy_rows.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_heavy_kv_heads, heavy_kv_heads.data(), kHeavySamples * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_heavy_positions, heavy_positions.data(), kHeavySamples * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_heavy_roles, heavy_roles.data(), kHeavySamples, cudaMemcpyHostToDevice));

        encode_rows_with_seeds_kernel<<<(kHeavySamples + kWarpsPerBlock - 1) / kWarpsPerBlock,
                                       kWarpsPerBlock * 32, smem_bytes>>>(
            d_heavy_rows, d_signs, d_heavy_codes, d_heavy_meta, kHeavySamples, d_heavy_kv_heads,
            d_heavy_positions, d_heavy_roles);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<std::uint8_t> heavy_hmeta(static_cast<std::size_t>(kHeavySamples) * kHqMetaBytes);
        std::vector<std::uint8_t> heavy_hcodes(static_cast<std::size_t>(kHeavySamples) * kHqRowBudgetBytes);
        CUDA_CHECK(cudaMemcpy(heavy_hmeta.data(), d_heavy_meta, heavy_hmeta.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(heavy_hcodes.data(), d_heavy_codes, heavy_hcodes.size(), cudaMemcpyDeviceToHost));

        int heavy_escalated = 0, heavy_fallback = 0, heavy_byte_bad = 0;
        for (int i = 0; i < kHeavySamples; ++i) {
            const std::uint8_t* meta_row  = &heavy_hmeta[static_cast<std::size_t>(i) * kHqMetaBytes];
            const std::uint8_t* codes_row = &heavy_hcodes[static_cast<std::size_t>(i) * kHqRowBudgetBytes];
            if (((meta_row[2] >> 4) & 3u) != 0) { ++heavy_escalated; }
            if (row_is_terminal_fallback(codes_row)) { ++heavy_fallback; }
            if (!row_byte_occupancy_ok(meta_row, codes_row)) { ++heavy_byte_bad; }
        }
        std::printf("\n[heavy-tailed arm] %d rows, +/-%.1fx-row-norm injected on channels "
                    "{0,32,...,224}: escalated %d/%d, terminal fallback %d/%d, byte-occupancy "
                    "bad %d/%d\n",
                    kHeavySamples, kHeavyFactor, heavy_escalated, kHeavySamples, heavy_fallback,
                    kHeavySamples, heavy_byte_bad, kHeavySamples);
        check(heavy_escalated == kHeavySamples,
             "the deliberately amplified heavy-tailed rows did not all report the escalation "
             "flag (meta byte 2 bits 4..5) nonzero");
        check(heavy_fallback == 0,
             "a deliberately amplified row hit the terminal fallback instead of the "
             "alpha-halving rescue -- the amplification factor is too extreme");
        check(heavy_byte_bad == 0,
             "a deliberately amplified row's code plane is not exactly the fixed budget with "
             "zero tail padding");

        CUDA_CHECK(cudaFree(d_heavy_rows));
        CUDA_CHECK(cudaFree(d_heavy_kv_heads));
        CUDA_CHECK(cudaFree(d_heavy_positions));
        CUDA_CHECK(cudaFree(d_heavy_roles));
        CUDA_CHECK(cudaFree(d_heavy_codes));
        CUDA_CHECK(cudaFree(d_heavy_meta));
    }

    CUDA_CHECK(cudaFree(d_signs));
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_meta));
    CUDA_CHECK(cudaFree(d_decoded_rotated));
    CUDA_CHECK(cudaFree(d_reconstructed));

    if (g_failed != 0) {
        std::fprintf(stderr, "test_hq_codec_kv_rows: %d failures\n", g_failed);
        return EXIT_FAILURE;
    }
    std::printf("test_hq_codec_kv_rows: ALL PASSED\n");
    return 0;
}
