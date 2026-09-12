// P4-04 (GitHub #122): the KV append path under both storage formats --
// OURS, not vendored.
//
// The ticket's claim is that the hq-e8-2b append writes fixed-budget rows
// "through the same page addressing BF16 uses". This test is what makes that
// checkable. It builds two sequence pools that differ in exactly one field
// (`ignis_seq_pool_spec::kv_format`), appends the SAME real K/V rows to a
// sequence drawn from each through the SAME vendored op (A2,
// `ninfer::ops::gqa_kv_append`), and reads both back through one row-address
// helper that knows only the sequence's block table and the plane's own
// strides. If the two formats disagreed about where a (position, kv_head)
// row lives, one of the two readbacks would come back zero.
//
// It is a CTest rather than a Rust GPU test on purpose. A2 (`gqa_kv_append`)
// is no longer on the serving path at all -- P2-04 (GitHub #86) fused the
// append into A1 -- so no Rust caller reaches it, and reaching it from Rust
// would mean widening the step ABI for a test. Here the internal pool
// definitions (ignis_seq_internal.h) are already in scope, the same way
// test_seq_alloc.cpp uses them, and the whole thing runs inside the GPU
// profile's `kernel/build.ps1 -Test` leg with no cargo feature gate. The
// serving path's own version of this claim is
// kernel/tests/test_hq_route_agreement.cu (GitHub #123), which drives A1.
//
// Rows are the committed real-activation fixture from P4-03
// (kernel/tests/fixtures/hq_kv_rows_27b.bin, captured from a real prefill of
// the 27B artifact): a synthetic corpus would prove the addressing but say
// nothing about the codec meeting its budget on rows this engine will
// actually store. IGNIS_HQ_KV_FIXTURE_PATH is the absolute path
// kernel/tests/CMakeLists.txt bakes in.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU, a kernel error, or a missing fixture fails this test.

#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include "ninfer/ops/gqa_attention.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef IGNIS_HQ_KV_FIXTURE_PATH
#error "IGNIS_HQ_KV_FIXTURE_PATH must be defined by kernel/tests/CMakeLists.txt"
#endif

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

void expect_rc(int32_t rc, int32_t want, const char *label) {
  if (rc != want) {
    std::fprintf(stderr, "  FAIL: %s (rc=%d, want %d)\n", label, rc, want);
    ++g_failed;
  }
}

#define CUDA_FATAL(expr)                                                                        \
  do {                                                                                          \
    const cudaError_t _err = (expr);                                                            \
    if (_err != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "FATAL: %s failed: %s\n", #expr, cudaGetErrorString(_err));          \
      std::exit(EXIT_FAILURE);                                                                  \
    }                                                                                           \
  } while (0)

float bf16_bits_to_float(std::uint16_t bits) {
  __nv_bfloat16 v;
  std::memcpy(&v, &bits, sizeof(bits));
  return __bfloat162float(v);
}

float fp16_bits_to_float(std::uint16_t bits) {
  __half v;
  std::memcpy(&v, &bits, sizeof(bits));
  return __half2float(v);
}

// ---- the committed real-row fixture ----------------------------------------
//
// Header layout and row order are the fixture's own
// (kernel/tests/fixtures/hq_kv_rows_27b.provenance.json, written by
// crates/core/tests/hq_kv_fixture_capture_gpu.rs); the integrity checks this
// loader repeats are the ones test_hq_codec_kv_rows.cu makes, because a
// truncated or corrupt fixture proves nothing and must never read as a pass.

std::uint64_t fnv1a64(const std::uint8_t *data, std::size_t n) {
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
  std::uint32_t layer_count    = 0;
  std::uint32_t rows_per_block = 0;
  std::vector<std::uint16_t> rows;

  // The fixture's documented row order:
  //   layer -> role -> kv_head -> position -> head_dim.
  const std::uint16_t *row(std::uint32_t layer, std::uint32_t role, std::uint32_t kv_head,
                           std::uint32_t position) const {
    const std::size_t block = ((static_cast<std::size_t>(layer) * role_count + role) * kv_heads +
                               kv_head) *
                              rows_per_block;
    return rows.data() + (block + position) * head_dim;
  }
};

[[noreturn]] void fatal_fixture(const char *why) {
  std::fprintf(stderr,
              "FATAL: fixture %s is invalid: %s\n"
              "Regenerate it: stop ninfer, then `scripts/gpu-profile.ps1` -- see "
              "crates/core/tests/hq_kv_fixture_capture_gpu.rs.\n",
              IGNIS_HQ_KV_FIXTURE_PATH, why);
  std::exit(EXIT_FAILURE);
}

Fixture load_fixture() {
  std::FILE *f = std::fopen(IGNIS_HQ_KV_FIXTURE_PATH, "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "FATAL: cannot open fixture %s (%s)\n", IGNIS_HQ_KV_FIXTURE_PATH,
                std::strerror(errno));
    std::exit(EXIT_FAILURE);
  }
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 0) {
    std::fclose(f);
    fatal_fixture("cannot determine file size");
  }
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  const std::size_t got = data.empty() ? 0 : std::fread(data.data(), 1, data.size(), f);
  std::fclose(f);
  if (got != data.size()) { fatal_fixture("short read"); }

  if (data.size() < 8 + 4 * 6) { fatal_fixture("truncated header"); }
  if (std::memcmp(data.data(), "IGNHQKV1", 8) != 0) { fatal_fixture("bad magic"); }
  auto read_u32 = [&](std::size_t off) {
    std::uint32_t v;
    std::memcpy(&v, data.data() + off, 4);
    return v;
  };
  std::size_t off = 8;
  if (read_u32(off) != 1) { fatal_fixture("unsupported format_version (expected 1)"); }
  off += 4;

  Fixture fx;
  fx.head_dim    = read_u32(off); off += 4;
  fx.kv_heads    = read_u32(off); off += 4;
  fx.role_count  = read_u32(off); off += 4;
  fx.layer_count = read_u32(off); off += 4;
  const std::size_t tail_bytes = 4u * fx.layer_count + 4 * 3 + 8;
  if (data.size() < off + tail_bytes) { fatal_fixture("truncated header (layers/tail)"); }
  off += 4u * fx.layer_count;    // layer ordinals: this test does not need them
  off += 4;                      // first_position
  fx.rows_per_block = read_u32(off); off += 4;
  const std::uint32_t total_rows = read_u32(off); off += 4;
  std::uint64_t checksum;
  std::memcpy(&checksum, data.data() + off, 8);
  off += 8;

  const std::uint64_t expected_rows = static_cast<std::uint64_t>(fx.layer_count) * fx.role_count *
                                      fx.kv_heads * fx.rows_per_block;
  if (expected_rows != total_rows) { fatal_fixture("total_rows is internally inconsistent"); }
  const std::size_t payload_bytes = static_cast<std::size_t>(total_rows) * fx.head_dim * 2;
  if (data.size() != off + payload_bytes) { fatal_fixture("file size does not match the header"); }
  if (fnv1a64(data.data() + off, payload_bytes) != checksum) {
    fatal_fixture("FNV-1a64 checksum mismatch over the row payload (corrupted)");
  }
  fx.rows.resize(static_cast<std::size_t>(total_rows) * fx.head_dim);
  std::memcpy(fx.rows.data(), data.data() + off, payload_bytes);
  return fx;
}

// ---- geometry --------------------------------------------------------------

constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kKvHeads     = 4;
constexpr std::int32_t kGqaOrdinal  = 0;
// Deliberately not a multiple of the 64-token page: the appended span must
// cross three whole pages and land partway into a fourth, so a fill that
// ignored the block table (or that assumed page-contiguous storage) writes
// somewhere the readback does not look.
constexpr std::int32_t kTokens      = 200;
constexpr std::uint32_t kMaxContext = 256;
// One page group per sequence's four pages, twice over: the decoy sequence
// below takes the first four, so the sequence under test is mapped to
// physical pages that are NOT its logical ones.
constexpr std::uint32_t kPoolPages  = 8;

ignis_seq_pool_spec pool_spec(int32_t kv_format) {
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads        = kKvHeads;
  spec.head_dim            = kHeadDim;
  spec.kv_format           = kv_format;
  spec.kv_page_group_count = kPoolPages;
  spec.max_context_tokens  = kMaxContext;
  spec.slot_count          = 2;
  // A small GDN/vocab geometry: this test never steps a layer, it only needs
  // the pool to build.
  spec.gdn_num_layers    = 2;
  spec.gdn_conv_channels = 6;
  spec.gdn_value_heads   = 2;
  spec.gdn_head_dim      = 4;
  spec.vocab             = 32;
  return spec;
}

// The one row-address helper both formats are read through: the sequence's
// own block table picks the physical page, the plane's own strides do the
// rest. Nothing here knows which format it is addressing -- that is the
// property under test.
const std::uint8_t *row_address(const ninfer::Tensor &plane, const ignis_seq *seq,
                                std::int32_t position, std::int32_t kv_head) {
  const auto page_ids = seq->kv.page_ids();
  const std::size_t logical_page =
      static_cast<std::size_t>(position) / static_cast<std::size_t>(ninfer::kPagedKVPageSize);
  if (logical_page >= page_ids.size()) {
    std::fprintf(stderr, "FATAL: position %d maps past the sequence's mapped pages\n", position);
    std::exit(EXIT_FAILURE);
  }
  const std::int64_t byte_offset =
      static_cast<std::int64_t>(position % ninfer::kPagedKVPageSize) * plane.nb[1] +
      static_cast<std::int64_t>(kv_head) * plane.nb[2] +
      static_cast<std::int64_t>(page_ids[logical_page]) * plane.nb[3];
  return static_cast<const std::uint8_t *>(plane.data) + byte_offset;
}

void read_row(const ninfer::Tensor &plane, const ignis_seq *seq, std::int32_t position,
              std::int32_t kv_head, void *out, std::size_t bytes) {
  CUDA_FATAL(cudaMemcpy(out, row_address(plane, seq, position, kv_head), bytes,
                        cudaMemcpyDeviceToHost));
}

// A device staging buffer for the append's inputs, freed by its destructor.
struct DeviceBytes {
  void *p = nullptr;
  explicit DeviceBytes(std::size_t bytes) { CUDA_FATAL(cudaMalloc(&p, bytes)); }
  ~DeviceBytes() { cudaFree(p); }
  DeviceBytes(const DeviceBytes &)            = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
};

} // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_err = cudaGetDeviceCount(&device_count);
  if (count_err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device (%s)\n",
                count_err == cudaSuccess ? "count 0" : cudaGetErrorString(count_err));
    return 1;
  }

  const Fixture fx = load_fixture();
  if (fx.head_dim != static_cast<std::uint32_t>(kHeadDim) ||
      fx.kv_heads != static_cast<std::uint32_t>(kKvHeads) || fx.role_count != 2 ||
      fx.rows_per_block < static_cast<std::uint32_t>(kTokens)) {
    fatal_fixture("geometry does not match this test's 27B head shape / token span");
  }

  // The append op's own contiguous [head_dim, kv_heads, T] order, from the
  // fixture's [kv_head][position][head_dim] blocks.
  std::vector<std::uint16_t> host_k(static_cast<std::size_t>(kHeadDim) * kKvHeads * kTokens);
  std::vector<std::uint16_t> host_v(host_k.size());
  for (std::int32_t t = 0; t < kTokens; ++t) {
    for (std::int32_t h = 0; h < kKvHeads; ++h) {
      const std::size_t dst =
          (static_cast<std::size_t>(t) * kKvHeads + static_cast<std::size_t>(h)) * kHeadDim;
      std::memcpy(host_k.data() + dst, fx.row(0, 0, static_cast<std::uint32_t>(h),
                                             static_cast<std::uint32_t>(t)),
                 static_cast<std::size_t>(kHeadDim) * 2);
      std::memcpy(host_v.data() + dst, fx.row(0, 1, static_cast<std::uint32_t>(h),
                                             static_cast<std::uint32_t>(t)),
                 static_cast<std::size_t>(kHeadDim) * 2);
    }
  }
  std::vector<std::int32_t> host_positions(kTokens);
  for (std::int32_t t = 0; t < kTokens; ++t) { host_positions[t] = t; }

  const std::size_t kv_bytes = host_k.size() * sizeof(std::uint16_t);
  DeviceBytes k_device(kv_bytes);
  DeviceBytes v_device(kv_bytes);
  DeviceBytes positions_device(host_positions.size() * sizeof(std::int32_t));
  CUDA_FATAL(cudaMemcpy(k_device.p, host_k.data(), kv_bytes, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(v_device.p, host_v.data(), kv_bytes, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(positions_device.p, host_positions.data(),
                        host_positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));

  const ninfer::Tensor k(k_device.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, kTokens, 1});
  const ninfer::Tensor v(v_device.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, kTokens, 1});
  const ninfer::Tensor positions(positions_device.p, ninfer::DType::I32, {kTokens, 1, 1, 1});

  // ---- BF16: the append is a copy, and it lands where the block table says -
  {
    const ignis_seq_pool_spec spec = pool_spec(IGNIS_KV_FORMAT_BF16);
    ignis_seq_pool *pool           = nullptr;
    expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "bf16 pool create");
    ignis_seq *decoy = nullptr;
    expect_rc(ignis_seq_alloc(pool, kMaxContext, &decoy), 0, "bf16 decoy alloc");
    ignis_seq *seq = nullptr;
    expect_rc(ignis_seq_alloc(pool, kMaxContext, &seq), 0, "bf16 alloc");
    check(seq->kv.page_ids()[0] != 0,
         "bf16: the sequence under test is mapped to a non-zero physical page (the decoy holds "
         "the first page group)");

    // The production view builder (kernel/include/ignis_seq_internal.h),
    // not a copy of it: a wrong plane or a missing quant_group there has to
    // turn this test red.
    ninfer::ops::gqa_kv_append(k, v, positions, ignis_kv_layer_view(pool, seq, kGqaOrdinal),
                               /*stream=*/nullptr);
    CUDA_FATAL(cudaStreamSynchronize(nullptr));

    const ninfer::Tensor &k_plane =
        pool->kv_pool.plane(ignis_kv_plane_index(pool->kv_format, kGqaOrdinal, IGNIS_KV_PLANE_K));
    const ninfer::Tensor &v_plane =
        pool->kv_pool.plane(ignis_kv_plane_index(pool->kv_format, kGqaOrdinal, IGNIS_KV_PLANE_V));
    std::vector<std::uint16_t> row(kHeadDim);
    int mismatches = 0;
    for (std::int32_t t = 0; t < kTokens; ++t) {
      for (std::int32_t h = 0; h < kKvHeads; ++h) {
        const std::size_t src =
            (static_cast<std::size_t>(t) * kKvHeads + static_cast<std::size_t>(h)) * kHeadDim;
        read_row(k_plane, seq, t, h, row.data(), row.size() * 2);
        if (std::memcmp(row.data(), host_k.data() + src, row.size() * 2) != 0) { ++mismatches; }
        read_row(v_plane, seq, t, h, row.data(), row.size() * 2);
        if (std::memcmp(row.data(), host_v.data() + src, row.size() * 2) != 0) { ++mismatches; }
      }
    }
    check(mismatches == 0,
         "bf16: every appended (position, kv_head) row reads back bit-identical at the address "
         "the block table gives (" + std::to_string(mismatches) + " mismatched)");

    // Nothing was written past the appended span, and nothing landed in the
    // decoy's pages -- both would be addressing bugs a row-by-row comparison
    // of the written span alone cannot see.
    read_row(k_plane, seq, kTokens, 0, row.data(), row.size() * 2);
    bool tail_zero = true;
    for (std::uint16_t bits : row) { tail_zero = tail_zero && bits == 0; }
    check(tail_zero, "bf16: the first position past the appended span is still zero");
    bool decoy_zero = true;
    for (std::int32_t t = 0; t < kTokens; t += 37) {
      read_row(k_plane, decoy, t, 0, row.data(), row.size() * 2);
      for (std::uint16_t bits : row) { decoy_zero = decoy_zero && bits == 0; }
    }
    check(decoy_zero, "bf16: the other sequence's pages were not touched");

    ignis_seq_release(pool, seq);
    ignis_seq_release(pool, decoy);
    ignis_seq_pool_free(pool);
  }

  // ---- hq-e8-2b: fixed-budget rows, same addressing ------------------------
  {
    const ignis_seq_pool_spec spec = pool_spec(IGNIS_KV_FORMAT_HQ_E8_2B);
    ignis_seq_pool *pool           = nullptr;
    expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "hq pool create");
    ignis_seq *decoy = nullptr;
    expect_rc(ignis_seq_alloc(pool, kMaxContext, &decoy), 0, "hq decoy alloc");
    ignis_seq *seq = nullptr;
    expect_rc(ignis_seq_alloc(pool, kMaxContext, &seq), 0, "hq alloc");

    // The production view builder (kernel/include/ignis_seq_internal.h),
    // not a copy of it: a wrong plane or a missing quant_group there has to
    // turn this test red.
    ninfer::ops::gqa_kv_append(k, v, positions, ignis_kv_layer_view(pool, seq, kGqaOrdinal),
                               /*stream=*/nullptr);
    CUDA_FATAL(cudaStreamSynchronize(nullptr));

    const ninfer::Tensor &code_plane =
        pool->kv_pool.plane(ignis_kv_plane_index(pool->kv_format, kGqaOrdinal, IGNIS_KV_PLANE_K));
    const ninfer::Tensor &meta_plane = pool->kv_pool.plane(
        ignis_kv_plane_index(pool->kv_format, kGqaOrdinal, IGNIS_KV_PLANE_K_META));

    std::uint8_t codes[kIgnisHqCodeRowBytes];
    std::uint8_t meta[kIgnisHqMetaRowBytes];
    int norm_mismatches  = 0;
    int budget_overruns  = 0;
    int tail_violations  = 0;
    int nonzero_rice_k   = 0;
    double worst_norm_error = 0.0;
    for (std::int32_t t = 0; t < kTokens; ++t) {
      for (std::int32_t h = 0; h < kKvHeads; ++h) {
        read_row(code_plane, seq, t, h, codes, sizeof(codes));
        read_row(meta_plane, seq, t, h, meta, sizeof(meta));

        // The stored row norm identifies WHICH source row was encoded here,
        // which is what turns this from "something was written" into "the
        // right row went to the right address". It is an FP16 round of the
        // row's own L2 norm (hq_codec.cuh's meta[0..1]).
        const std::size_t src =
            (static_cast<std::size_t>(t) * kKvHeads + static_cast<std::size_t>(h)) * kHeadDim;
        double sumsq = 0.0;
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
          const double x = bf16_bits_to_float(host_k[src + static_cast<std::size_t>(d)]);
          sumsq += x * x;
        }
        const double want = std::sqrt(sumsq);
        const std::uint16_t norm_bits =
            static_cast<std::uint16_t>(meta[0]) | (static_cast<std::uint16_t>(meta[1]) << 8);
        const double got = fp16_bits_to_float(norm_bits);
        const double relative_error = std::abs(got - want) / (want > 0.0 ? want : 1.0);
        worst_norm_error = std::max(worst_norm_error, relative_error);
        // FP16 carries ~11 significant bits, so 1e-3 is the representation
        // floor; 5e-3 leaves margin without admitting a different row.
        if (!(relative_error < 5e-3)) { ++norm_mismatches; }

        if ((meta[2] & 0x0Fu) != 0) { ++nonzero_rice_k; }
        const std::uint32_t used_bits =
            static_cast<std::uint32_t>(meta[3]) |
            ((static_cast<std::uint32_t>(meta[4]) & 0x03u) << 8);
        if (used_bits == 0 || used_bits > static_cast<std::uint32_t>(kIgnisHqCodeRowBytes) * 8) {
          ++budget_overruns;
        }
        // The stored-stream invariant the group decoder relies on: every
        // 32-bit code word at or past ceil(used_bits/32) is zero. Compared at
        // word granularity, not byte -- the packed stream is MSB-first within
        // each little-endian-stored word.
        std::uint32_t words[kIgnisHqCodeRowBytes / 4];
        std::memcpy(words, codes, sizeof(words));
        for (std::size_t w = (used_bits + 31) / 32; w < sizeof(words) / sizeof(words[0]); ++w) {
          if (words[w] != 0) { ++tail_violations; }
        }
      }
    }
    check(norm_mismatches == 0,
         "hq: every appended row's stored norm identifies its own source row (" +
             std::to_string(norm_mismatches) + " mismatched, worst relative error " +
             std::to_string(worst_norm_error) + ")");
    check(budget_overruns == 0,
         "hq: every row's used bits fit the fixed 64-byte code budget (" +
             std::to_string(budget_overruns) + " outside)");
    check(tail_violations == 0,
         "hq: every code word past the used bits is zero (" + std::to_string(tail_violations) +
             " violations)");
    check(nonzero_rice_k == 0, "hq: the packer's k=0 invariant holds on real rows (" +
                                   std::to_string(nonzero_rice_k) + " rows with k != 0)");

    // Same two negatives as the BF16 arm, read through the same helper: a
    // never-written row's metadata stays zeroed (ignis_seq_alloc zeroes the
    // pages), and the other sequence's pages are untouched.
    read_row(meta_plane, seq, kTokens, 0, meta, sizeof(meta));
    bool past_span_zero = true;
    for (std::uint8_t b : meta) { past_span_zero = past_span_zero && b == 0; }
    check(past_span_zero, "hq: the first position past the appended span is still zero");
    bool decoy_zero = true;
    for (std::int32_t t = 0; t < kTokens; t += 37) {
      read_row(meta_plane, decoy, t, 0, meta, sizeof(meta));
      for (std::uint8_t b : meta) { decoy_zero = decoy_zero && b == 0; }
    }
    check(decoy_zero, "hq: the other sequence's pages were not touched");

    // The V role is stored in its own plane pair, not aliased onto K's.
    const ninfer::Tensor &v_meta_plane = pool->kv_pool.plane(
        ignis_kv_plane_index(pool->kv_format, kGqaOrdinal, IGNIS_KV_PLANE_V_META));
    std::uint8_t v_meta[kIgnisHqMetaRowBytes];
    read_row(meta_plane, seq, 0, 0, meta, sizeof(meta));
    read_row(v_meta_plane, seq, 0, 0, v_meta, sizeof(v_meta));
    check(std::memcmp(meta, v_meta, sizeof(meta)) != 0,
         "hq: the K and V metadata planes hold different rows (the roles are not aliased)");

    ignis_seq_release(pool, seq);
    ignis_seq_release(pool, decoy);
    ignis_seq_pool_free(pool);
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "kv append format test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("kv append format test: ok\n");
  return 0;
}
