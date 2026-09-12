// P4-05 (GitHub #123): the hq-e8-2b attention routes against ignis's own
// BF16 route, on identical keys and values -- OURS, not vendored.
//
// ADR 0022 is the reason this test exists and the reason it has this shape.
// hq is lossy, and the project refuses to define "correct" as agreeing with
// another engine's output (ADR 0007). So the hq route earns its acceptance
// in-house: the SAME q / k / v / gate / positions go through A1
// (`ninfer::ops::gqa_attention`) twice, against two sequence pools that
// differ in exactly one field (`ignis_seq_pool_spec::kv_format`), and the two
// outputs are compared. Everything except the cache's storage format is
// bit-identical between the arms -- which is what makes the difference
// attributable to the codec rather than to the call.
//
// The two arms are the two routes the GQA layer actually dispatches
// (kernel/src/gqa_layer.cu):
//
//   prefill  B=1, W=200 -> the resolver's Prompt route. Under hq that is the
//            one-shot kernel that materializes the visible history into
//            rotated-frame BF16 scratch; under BF16 it is the FA2 prompt
//            kernel. The chunk is appended and attended in one call, as in a
//            real prefill.
//   decode   B=1..8, W=1 over a 200-token history -> the SmallT route, the
//            batch-wide decode round's own shape (ADR 0020). Each lane gets
//            its own sequence, its own block-table row and its own query, so
//            a row-to-slot mix-up shows up as a changed output rather than
//            passing by symmetry.
//
// Rows are the committed real-activation fixture from P4-03
// (kernel/tests/fixtures/hq_kv_rows_27b.bin, captured from a real prefill of
// the 27B artifact): the tolerance below is derived from the codec error
// measured on THESE rows by test_hq_codec_kv_rows.cu, so the two
// measurements have to be over the same corpus to compose at all.
//
// The cache views come from the production builders
// (`ignis_kv_layer_view` / `ignis_kv_batch_layer_view`,
// kernel/include/ignis_seq_internal.h) rather than from copies here: the U8 +
// quant_group-32 declaration those make IS the act of routing A1 to the hq
// kernels, so a bug there has to turn this test red.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU, a kernel error, or a missing fixture fails this test.

#include "ignis_gqa_workspace.h"
#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_bf16.h>
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

double bf16_bits_to_double(std::uint16_t bits) {
  __nv_bfloat16 v;
  std::memcpy(&v, &bits, sizeof(bits));
  return static_cast<double>(__bfloat162float(v));
}

std::uint16_t double_to_bf16_bits(double x) {
  const __nv_bfloat16 v = __float2bfloat16(static_cast<float>(x));
  std::uint16_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

// ---- the committed real-row fixture ----------------------------------------
//
// Header layout and row order are the fixture's own
// (kernel/tests/fixtures/hq_kv_rows_27b.provenance.json). The integrity checks
// repeated here are the ones test_hq_codec_kv_rows.cu makes: a truncated or
// corrupt fixture proves nothing and must never read as a pass.

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
    const std::size_t block =
        ((static_cast<std::size_t>(layer) * role_count + role) * kv_heads + kv_head) *
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

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQHeads     = 24;
constexpr std::int32_t kKvHeads    = 4;
constexpr std::int32_t kGqaOrdinal = 0;
// The attention scale A1 requires at this head_dim (1/sqrt(256)); the wrapper
// rejects anything else, so it is written as the reciprocal square root the
// layer computes rather than as a bare literal.
const float kScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
// The prefill chunk: wide enough to resolve to the Prompt route under both
// formats (the small-T tile is 6 tokens under BF16 and 8 under hq), and not a
// multiple of the 64-token page, so the span crosses three whole pages and
// lands partway into a fourth.
constexpr std::int32_t kPrefillTokens = 200;
constexpr std::int32_t kMaxLanes      = 8;
constexpr std::uint32_t kMaxContext   = 512;
// One sequence's 512-token context is 8 pages; the decode arm allocates
// kMaxLanes of them at once.
constexpr std::uint32_t kPoolPages = 8 * kMaxLanes;

ignis_seq_pool_spec pool_spec(int32_t kv_format) {
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads        = kKvHeads;
  spec.head_dim            = kHeadDim;
  spec.kv_format           = kv_format;
  spec.kv_page_group_count = kPoolPages;
  spec.max_context_tokens  = kMaxContext;
  spec.slot_count          = kMaxLanes;
  // A small GDN/vocab geometry: this test never steps a layer, it only needs
  // the pool to build.
  spec.gdn_num_layers    = 2;
  spec.gdn_conv_channels = 6;
  spec.gdn_value_heads   = 2;
  spec.gdn_head_dim      = 4;
  spec.vocab             = 32;
  return spec;
}

// A device staging buffer, freed by its destructor.
struct DeviceBytes {
  void *p = nullptr;
  explicit DeviceBytes(std::size_t bytes) { CUDA_FATAL(cudaMalloc(&p, bytes)); }
  ~DeviceBytes() { cudaFree(p); }
  DeviceBytes(const DeviceBytes &)            = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
};

// ---- the shared inputs -----------------------------------------------------
//
// Built once on the host and uploaded once; both format arms read the SAME
// device buffers, so nothing but the cache view differs between the two A1
// calls.
struct Inputs {
  // [head_dim, kv_heads, T] BF16, A1's own contiguous order.
  std::vector<std::uint16_t> k;
  std::vector<std::uint16_t> v;
  // [head_dim, q_heads, W, B] BF16.
  std::vector<std::uint16_t> q;
  std::vector<std::uint16_t> gate;
};

// K and V for `tokens` fresh rows, straight from the fixture's GQA layer 0.
void fill_kv(const Fixture &fx, std::int32_t tokens, std::int32_t first_position,
             std::vector<std::uint16_t> &k, std::vector<std::uint16_t> &v) {
  k.assign(static_cast<std::size_t>(kHeadDim) * kKvHeads * tokens, 0);
  v.assign(k.size(), 0);
  for (std::int32_t t = 0; t < tokens; ++t) {
    for (std::int32_t h = 0; h < kKvHeads; ++h) {
      const std::size_t dst =
          (static_cast<std::size_t>(t) * kKvHeads + static_cast<std::size_t>(h)) * kHeadDim;
      const auto position = static_cast<std::uint32_t>(first_position + t);
      std::memcpy(k.data() + dst, fx.row(0, 0, static_cast<std::uint32_t>(h), position),
                 static_cast<std::size_t>(kHeadDim) * 2);
      std::memcpy(v.data() + dst, fx.row(0, 1, static_cast<std::uint32_t>(h), position),
                 static_cast<std::size_t>(kHeadDim) * 2);
    }
  }
}

// Queries for `width` tokens across `batch` rows. Real captured rows again,
// and K rows specifically: post-RoPE Q and K come out of the same fused
// qk_norm_rope with the same per-head normalization, so a K row is a faithful
// stand-in for a query, while a V row's scale would sharpen the softmax in a
// way no real query does. The (layer, kv_head) pair walks with the q-head
// index so the 24 heads see 16 distinct source rows rather than 4, and the
// batch row shifts the source position so no two lanes ask the same question.
void fill_q(const Fixture &fx, std::int32_t width, std::int32_t batch,
            std::vector<std::uint16_t> &q) {
  q.assign(static_cast<std::size_t>(kHeadDim) * kQHeads * width * batch, 0);
  for (std::int32_t b = 0; b < batch; ++b) {
    for (std::int32_t w = 0; w < width; ++w) {
      for (std::int32_t h = 0; h < kQHeads; ++h) {
        const std::size_t dst =
            ((static_cast<std::size_t>(b) * width + w) * kQHeads + h) * kHeadDim;
        const auto layer = static_cast<std::uint32_t>((h / kKvHeads) % fx.layer_count);
        const auto head  = static_cast<std::uint32_t>(h % kKvHeads);
        const auto pos   = static_cast<std::uint32_t>((w + b * 31 + 7) % fx.rows_per_block);
        std::memcpy(q.data() + dst, fx.row(layer, 0, head, pos),
                   static_cast<std::size_t>(kHeadDim) * 2);
      }
    }
  }
}

// The sigmoid output gate, in `out`'s own flat element layout. A smooth,
// deterministic spread rather than a constant: a constant gate would scale
// both arms identically and hide a route that dropped the gate on one side.
void fill_gate(std::size_t elements, std::vector<std::uint16_t> &gate) {
  gate.resize(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    gate[i] = double_to_bf16_bits(std::sin(static_cast<double>(i) * 0.37) * 1.5);
  }
}

// ---- agreement statistics --------------------------------------------------
//
// One entry per (batch row, token, q head) output row of 256 elements: the
// relative L2 of the hq row against the BF16 row, and the pieces of an
// aggregate cosine / SNR. Reported the way test_hq_codec_kv_rows.cu reports
// the codec's own error, so the two numbers read on the same scale.
struct Agreement {
  std::vector<double> rel_l2;
  std::vector<double> row_norm;
  double signal = 0.0;
  double noise  = 0.0;
  double cos_xy = 0.0;
  double cos_xx = 0.0;
  double cos_yy = 0.0;

  void add_row(const std::uint16_t *reference, const std::uint16_t *candidate) {
    double err2 = 0.0;
    double sig2 = 0.0;
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
      const double x = bf16_bits_to_double(reference[d]);
      const double y = bf16_bits_to_double(candidate[d]);
      const double e = y - x;
      err2 += e * e;
      sig2 += x * x;
      cos_xy += x * y;
      cos_xx += x * x;
      cos_yy += y * y;
    }
    signal += sig2;
    noise += err2;
    rel_l2.push_back(sig2 > 0.0 ? std::sqrt(err2 / sig2) : 0.0);
    row_norm.push_back(std::sqrt(sig2));
  }

  // Rows whose relative error is past `bound`, as a fraction of all rows.
  double fraction_above(double bound) const {
    if (rel_l2.empty()) { return 0.0; }
    std::size_t over = 0;
    for (double e : rel_l2) {
      if (e > bound) { ++over; }
    }
    return static_cast<double>(over) / static_cast<double>(rel_l2.size());
  }

  std::size_t worst_index() const {
    return rel_l2.empty()
               ? 0
               : static_cast<std::size_t>(
                     std::max_element(rel_l2.begin(), rel_l2.end()) - rel_l2.begin());
  }

  double median() const {
    if (rel_l2.empty()) { return 0.0; }
    std::vector<double> v = rel_l2;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }

  double worst() const {
    return rel_l2.empty() ? 0.0 : *std::max_element(rel_l2.begin(), rel_l2.end());
  }

  double cosine() const { return cos_xy / (std::sqrt(cos_xx) * std::sqrt(cos_yy) + 1e-300); }

  double snr_db() const { return 10.0 * std::log10(signal / (noise + 1e-300)); }
};

// ---- derived tolerance -----------------------------------------------------
//
// Derived from the codec error test_hq_codec_kv_rows.cu measured on THIS
// fixture (its own "derived tolerance" comment, measured 2026-09-12) --
// which is why both tests read the same file, and why this one is worth
// nothing if that one is re-measured and these are not:
//
//   worst per-group median relative L2 0.369634, enforced at 0.45
//   worst per-group max relative L2    0.773022, enforced at 0.90
//   min per-group original-frame cosine 0.934583, enforced at 0.91
//   min per-group SNR                   8.40 dB,  enforced at 6.5 dB
//
// Carried through attention rather than copied from it. Attention's output is
// `sum_x p_x V_x`, a convex combination of cache rows, so the V path can
// contribute at most one row's own relative error (the degenerate case where
// the softmax is a point mass) and any spread averages independent per-row
// errors down. The K path does not perturb the output directly -- it perturbs
// the scores -- so it has no bound of that shape at all, which is exactly why
// the ceiling here is the codec's own row bound and not something tighter
// derived in the abstract. Attention cannot be held to a standard its inputs
// do not meet, and this test's job is to show the two routes agree to the
// codec's own accuracy, not to re-measure the codec.
//
// What the routes actually deliver, measured 2026-09-12 on this machine's
// RTX 5090 at the 27B geometry (the report below prints all of it every run,
// pass or fail -- 80x of unused slack sat unnoticed in GitHub #96 precisely
// because nobody printed the headroom):
//
//   prefill W=200 B=1  median relL2 0.227955  cosine 0.959193  SNR 10.73 dB
//                      0.229% of 4800 rows past the 0.90 codec row bound
//   prefill W=9..16    median relL2 0.3772..0.3611  cosine 0.9252..0.9285
//                      SNR 7.95..8.17 dB
//   decode  W=1 B=1..8 median relL2 0.174..0.190  cosine 0.9686..0.9775
//                      SNR 11.85..13.27 dB  0% of rows past 0.90
//
// The narrow prefill widths are the tightest arm, at 84% of the median
// ceiling, and that is the derivation working rather than failing. Agreement
// improves with the size of the visible history because the output is a
// convex combination that averages independent per-row errors down; at W=9
// the first queries see one or two keys, so there is nothing to average and
// the output error converges on the per-row codec error itself (0.369634
// worst per-group median, which 0.3772 sits alongside). A future change that
// makes this arm fail is a change to the codec or the route, not a bound that
// needs loosening -- loosening it past the codec's own measured error would
// make the whole comparison meaningless.
//
// Everywhere else there is room: the median runs at about half its ceiling,
// the cosine sits 0.05 above its floor, the SNR 4 dB above its own, and the
// over-the-row-bound fraction has roughly four times the room its limit
// allows. Generous enough to absorb ordinary reduction-order nondeterminism,
// tight enough that a route change which doubled the disagreement would trip
// it.
constexpr double kCeilingMedianRelL2 = 0.45;
constexpr double kCeilingRowRelL2    = 0.90;
constexpr double kCeilingCosine      = 0.91;
constexpr double kCeilingSnrDb       = 6.5;

// The per-row MAXIMUM is deliberately not one of the enforced bounds, and
// that is a finding rather than a concession. Over the 200-token prefill arm
// a handful of output rows come back past 1.0, which no bound derived from a
// per-row codec error could ever cover -- and the reason is visible in the
// numbers this test prints. Attention's output is `sum_x p_x V_x`; when two
// keys are nearly tied in score, a perturbation the size of the codec's own
// row error is enough to swap which one the softmax concentrates on, and the
// two routes then return two different V rows. Near-orthogonal rows differ by
// about sqrt(2) relative, which is what those outliers measure. That is a
// property of serving attention from a lossy cache at all, not of these
// kernels, and it is the reason ADR 0022 keeps BF16 as the oracle rather than
// asking hq to match it row for row.
//
// What IS bounded: the bulk (median), the aggregate direction and energy
// (cosine, SNR), and the FRACTION of rows allowed past the codec's own
// worst-row error. That last one is the check that would catch a route which
// started flipping the softmax often rather than rarely.
//
// That fraction is the one bound with no counterpart in the codec
// measurement, so it is set the way GitHub #96 says a bound must be: from
// what was measured, with a stated margin, not from a round number that felt
// safe. Measured 2026-09-12 at 0.229% of the prefill arm's 4,800 rows -- 11
// rows -- and bit-reproducible across consecutive runs (identical to six
// decimals, this being deterministic arithmetic over a committed fixture).
// Enforced at 0.5%, which is 24 of those rows: more than double the observed
// count, because at counts this small a single additional near-tied key
// swinging the other way on a different driver moves the percentage by 0.02
// points and must not turn the suite red. A route that started flipping the
// softmax as a matter of course would be far past 24.
constexpr double kCeilingFractionOverRow = 0.005;

void report(const char *arm, const Agreement &a);

void check_agreement(const char *arm, const Agreement &a) {
  report(arm, a);
  const std::string prefix = std::string(arm) + ": ";
  // Two ways this test could rot into proving nothing, both checked before
  // any bound is:
  //
  // A zeroed BF16 reference makes every relative error 0 and every bound
  // below vacuously true.
  check(a.signal > 0.0,
       prefix + "the BF16 route wrote an all-zero output, so there is nothing to compare against");
  // And if the view builder handed BOTH arms the BF16 planes and dtype --
  // the single most likely bug in the thing this test exists to check --
  // the two outputs would be bit-identical and every bound would pass with
  // the hq route never having run. A lossy cache must move the answer.
  check(a.noise > 0.0,
       prefix + "the two routes produced bit-identical output, so the hq cache view never "
                "selected the hq kernels and every bound below passed vacuously");
  check(a.median() <= kCeilingMedianRelL2,
       prefix + "median relative L2 between the hq and BF16 routes exceeded the codec's own "
                "measured median row error");
  check(a.cosine() >= kCeilingCosine,
       prefix + "cosine between the hq and BF16 routes fell below the codec's own measured "
                "row cosine");
  check(a.snr_db() >= kCeilingSnrDb,
       prefix + "SNR between the hq and BF16 routes fell below the codec's own measured row SNR");
  check(a.fraction_above(kCeilingRowRelL2) <= kCeilingFractionOverRow,
       prefix + "too many output rows are past the codec's own worst-row error -- the softmax is "
                "flipping between near-tied keys far more often than a lossy cache alone "
                "explains");
}

// ---- one A1 call against one format ---------------------------------------

struct Call {
  std::int32_t width = 0;
  std::int32_t batch = 1;
  // The absolute position of each lane's first fresh token.
  std::int32_t first_position = 0;
  // Rows appended into every lane's cache before the measured call (the
  // decode arm's history). Zero for the prefill arm, whose own call is the
  // append.
  std::int32_t history = 0;
};

// Runs `call` against a freshly built pool in `kv_format` and returns the
// BF16 output rows, host-side. Everything device-side is torn down before
// returning, so the two arms never hold two pools at once.
std::vector<std::uint16_t> run_route(const Fixture &fx, int32_t kv_format, const Call &call,
                                     const Inputs &in, const char *label) {
  const ignis_seq_pool_spec spec = pool_spec(kv_format);
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, (std::string(label) + " pool create").c_str());
  if (pool == nullptr) { std::exit(EXIT_FAILURE); }

  std::vector<ignis_seq *> lanes(static_cast<std::size_t>(call.batch), nullptr);
  for (std::int32_t b = 0; b < call.batch; ++b) {
    expect_rc(ignis_seq_alloc(pool, kMaxContext, &lanes[static_cast<std::size_t>(b)]), 0,
             (std::string(label) + " seq alloc").c_str());
  }

  // The history every lane shares, appended through A2 -- the same op and the
  // same production view builder both formats' real prefill path uses.
  if (call.history > 0) {
    std::vector<std::uint16_t> hk;
    std::vector<std::uint16_t> hv;
    fill_kv(fx, call.history, 0, hk, hv);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(call.history));
    for (std::int32_t t = 0; t < call.history; ++t) { positions[static_cast<std::size_t>(t)] = t; }

    DeviceBytes hk_device(hk.size() * 2);
    DeviceBytes hv_device(hv.size() * 2);
    DeviceBytes pos_device(positions.size() * sizeof(std::int32_t));
    CUDA_FATAL(cudaMemcpy(hk_device.p, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
    CUDA_FATAL(cudaMemcpy(hv_device.p, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));
    CUDA_FATAL(cudaMemcpy(pos_device.p, positions.data(), positions.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    const ninfer::Tensor hk_t(hk_device.p, ninfer::DType::BF16,
                             {kHeadDim, kKvHeads, call.history, 1});
    const ninfer::Tensor hv_t(hv_device.p, ninfer::DType::BF16,
                             {kHeadDim, kKvHeads, call.history, 1});
    const ninfer::Tensor pos_t(pos_device.p, ninfer::DType::I32, {call.history, 1, 1, 1});
    for (ignis_seq *seq : lanes) {
      ninfer::ops::gqa_kv_append(hk_t, hv_t, pos_t, ignis_kv_layer_view(pool, seq, kGqaOrdinal),
                                 /*stream=*/nullptr);
    }
    CUDA_FATAL(cudaStreamSynchronize(nullptr));
  }

  // The measured call's own inputs.
  const std::size_t out_elements = static_cast<std::size_t>(kHeadDim) * kQHeads * call.width *
                                   call.batch;
  std::vector<std::int32_t> positions(static_cast<std::size_t>(call.width) * call.batch);
  for (std::int32_t b = 0; b < call.batch; ++b) {
    for (std::int32_t w = 0; w < call.width; ++w) {
      positions[static_cast<std::size_t>(b) * call.width + w] = call.first_position + w;
    }
  }
  std::vector<std::int32_t> table_rows(static_cast<std::size_t>(call.batch));
  for (std::int32_t b = 0; b < call.batch; ++b) {
    table_rows[static_cast<std::size_t>(b)] = lanes[static_cast<std::size_t>(b)]->kv.bound_row();
  }

  DeviceBytes k_device(in.k.size() * 2);
  DeviceBytes v_device(in.v.size() * 2);
  DeviceBytes q_device(in.q.size() * 2);
  DeviceBytes gate_device(in.gate.size() * 2);
  DeviceBytes out_device(out_elements * 2);
  DeviceBytes pos_device(positions.size() * sizeof(std::int32_t));
  DeviceBytes rows_device(table_rows.size() * sizeof(std::int32_t));
  CUDA_FATAL(cudaMemcpy(k_device.p, in.k.data(), in.k.size() * 2, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(v_device.p, in.v.data(), in.v.size() * 2, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(q_device.p, in.q.data(), in.q.size() * 2, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(gate_device.p, in.gate.data(), in.gate.size() * 2, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemset(out_device.p, 0, out_elements * 2));
  CUDA_FATAL(cudaMemcpy(pos_device.p, positions.data(), positions.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(rows_device.p, table_rows.data(), table_rows.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice));

  const ninfer::Tensor q(q_device.p, ninfer::DType::BF16,
                        {kHeadDim, kQHeads, call.width, call.batch});
  const ninfer::Tensor k(k_device.p, ninfer::DType::BF16,
                        {kHeadDim, kKvHeads, call.width, call.batch});
  const ninfer::Tensor v(v_device.p, ninfer::DType::BF16,
                        {kHeadDim, kKvHeads, call.width, call.batch});
  const ninfer::Tensor pos(pos_device.p, ninfer::DType::I32, {call.width, call.batch, 1, 1});
  const ninfer::Tensor rows(rows_device.p, ninfer::DType::I32, {call.batch, 1, 1, 1});
  const ninfer::Tensor gate(gate_device.p, ninfer::DType::BF16,
                           {kHeadDim, kQHeads, call.width, call.batch});
  ninfer::Tensor out(out_device.p, ninfer::DType::BF16,
                    {kHeadDim, kQHeads, call.width, call.batch});

  // The production batched view: the pool-wide block-table matrix, with each
  // lane's own row selected by `rows` above -- exactly what
  // kernel/src/gqa_layer.cu's decode round hands A1.
  const ninfer::PagedKVBatchLayerView cache = ignis_kv_batch_layer_view(pool, kGqaOrdinal);
  // The envelope the layer would declare: every key this call can see.
  const ninfer::ops::GqaExecutionEnvelope envelope{
      /*min_visible_keys=*/1,
      /*max_visible_keys=*/static_cast<std::uint32_t>(call.first_position + call.width)};

  std::vector<std::uint16_t> host_out(out_elements, 0);
  try {
    // Sized through the engine's own function (kernel/include/
    // ignis_gqa_workspace.h), never a second copy of the arithmetic: the
    // narrow-Prompt-width arm below exists precisely to catch an
    // under-reservation there, and a test that computed its own answer could
    // not. At B > 1 the resolver takes the small-T route, which that function
    // documents as needing no correction, so the vendored query is asked
    // directly for the batch axis.
    const std::size_t workspace_bytes =
        call.batch == 1
            ? ignis_gqa_attention_workspace_bytes(cache.dtype, envelope, call.width)
            : ninfer::ops::gqa_attention_workspace_capacity_bytes(
                  kQHeads, cache.dtype, envelope, call.batch, call.width, call.width);
    ninfer::DeviceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
    CUDA_FATAL(cudaMemset(workspace.base(), 0, workspace.capacity()));
    ninfer::ops::gqa_attention(q, k, v, pos, /*valid_columns=*/ninfer::Tensor{}, rows, gate, kScale,
                               cache, envelope, workspace, out, /*stream=*/nullptr);
    CUDA_FATAL(cudaStreamSynchronize(nullptr));
    CUDA_FATAL(cudaMemcpy(host_out.data(), out_device.p, out_elements * 2, cudaMemcpyDeviceToHost));
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FATAL: %s: gqa_attention threw: %s\n", label, e.what());
    std::exit(EXIT_FAILURE);
  }

  for (ignis_seq *seq : lanes) { ignis_seq_release(pool, seq); }
  ignis_seq_pool_free(pool);
  return host_out;
}

// Printed on every arm, pass or fail: this bound is calibrated from a
// measurement, so how much room each arm actually has is what decides whether
// a later numerics change is safe (the lesson of GitHub #96). The worst row's
// own BF16 norm is printed beside the median one because that is what tells
// an outlier apart from a regression: a near-cancelling output row, or one
// whose softmax flipped, is small and rare; a broken route is neither.
void report(const char *arm, const Agreement &a) {
  const std::size_t worst = a.worst_index();
  std::vector<double> norms = a.row_norm;
  std::sort(norms.begin(), norms.end());
  const double median_norm = norms.empty() ? 0.0 : norms[norms.size() / 2];
  std::printf("[%s] rows %zu  median relL2 %.6f (%.0f%% of the %.2f bound)  cosine %.6f  "
              "SNR %.2f dB\n",
             arm, a.rel_l2.size(), a.median(), 100.0 * a.median() / kCeilingMedianRelL2,
             kCeilingMedianRelL2, a.cosine(), a.snr_db());
  std::printf("       max relL2 %.6f at row %zu (its BF16 norm %.4g vs median %.4g); %.3f%% of "
              "rows past the %.2f codec row bound (limit %.1f%%)\n",
             a.worst(), worst, a.row_norm.empty() ? 0.0 : a.row_norm[worst], median_norm,
             100.0 * a.fraction_above(kCeilingRowRelL2), kCeilingRowRelL2,
             100.0 * kCeilingFractionOverRow);
}

// Every 256-element output row of two runs, compared.
Agreement compare(const std::vector<std::uint16_t> &reference,
                  const std::vector<std::uint16_t> &candidate) {
  Agreement a;
  const std::size_t rows = reference.size() / kHeadDim;
  for (std::size_t r = 0; r < rows; ++r) {
    a.add_row(reference.data() + r * kHeadDim, candidate.data() + r * kHeadDim);
  }
  return a;
}

// Two lanes of the same run must not have produced the same output: each lane
// asks its own question (fill_q shifts the source row by the batch index), so
// identical rows would mean the batch axis never reached the kernel and the
// agreement above was measured over one lane repeated `batch` times.
bool lanes_differ(const std::vector<std::uint16_t> &out, std::int32_t width, std::int32_t batch) {
  if (batch < 2) { return true; }
  const std::size_t lane_elements = static_cast<std::size_t>(kHeadDim) * kQHeads * width;
  for (std::int32_t b = 1; b < batch; ++b) {
    if (std::memcmp(out.data(), out.data() + static_cast<std::size_t>(b) * lane_elements,
                    lane_elements * 2) != 0) {
      return true;
    }
  }
  return false;
}

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
      fx.rows_per_block < static_cast<std::uint32_t>(kPrefillTokens)) {
    fatal_fixture("geometry does not match this test's 27B head shape / token span");
  }
  std::printf("fixture: %s -- %u layers x %u roles x %u kv_heads x %u positions, head_dim %u\n",
             IGNIS_HQ_KV_FIXTURE_PATH, fx.layer_count, fx.role_count, fx.kv_heads,
             fx.rows_per_block, fx.head_dim);

  // ---- prefill: the Prompt route, one chunk appended and attended ----------
  {
    Inputs in;
    fill_kv(fx, kPrefillTokens, 0, in.k, in.v);
    fill_q(fx, kPrefillTokens, 1, in.q);
    fill_gate(in.q.size(), in.gate);
    const Call call{/*width=*/kPrefillTokens, /*batch=*/1, /*first_position=*/0, /*history=*/0};

    const std::vector<std::uint16_t> bf16 =
        run_route(fx, IGNIS_KV_FORMAT_BF16, call, in, "prefill bf16");
    const std::vector<std::uint16_t> hq =
        run_route(fx, IGNIS_KV_FORMAT_HQ_E8_2B, call, in, "prefill hq");
    check(bf16.size() == hq.size(), "prefill: both routes wrote the same output extent");
    check_agreement("prefill W=200 B=1", compare(bf16, hq));
  }

  // ---- prefill at the narrow Prompt widths ---------------------------------
  //
  // Widths 9..16 are the Prompt route (the hq small-T tile is 8) but sit at or
  // below the vendored capacity query's own 16-token verify cap, which is
  // exactly where that query stops summing an hq prompt call's two transient
  // riders and starts combining them with `max`. A real 12-token prompt is
  // what found it: the op overran the arena the layer had sized from that
  // answer and threw `bad allocation` mid-layer. This arm runs the whole
  // window through the engine's own sizing function
  // (`ignis_gqa_attention_workspace_bytes`, kernel/include/
  // ignis_gqa_workspace.h -- `run_route` above calls it, rather than
  // computing a second answer of its own), so the correction has a test
  // rather than a story and any future change to it has to keep every width
  // in this window running.
  for (std::int32_t width = 9; width <= 16; ++width) {
    Inputs in;
    fill_kv(fx, width, 0, in.k, in.v);
    fill_q(fx, width, 1, in.q);
    fill_gate(in.q.size(), in.gate);
    const Call call{width, /*batch=*/1, /*first_position=*/0, /*history=*/0};

    const std::string bf16_label = "prefill bf16 W=" + std::to_string(width);
    const std::string hq_label   = "prefill hq W=" + std::to_string(width);
    const std::vector<std::uint16_t> bf16 =
        run_route(fx, IGNIS_KV_FORMAT_BF16, call, in, bf16_label.c_str());
    const std::vector<std::uint16_t> hq =
        run_route(fx, IGNIS_KV_FORMAT_HQ_E8_2B, call, in, hq_label.c_str());
    check_agreement(("prefill W=" + std::to_string(width) + " B=1").c_str(), compare(bf16, hq));
  }

  // ---- decode: the SmallT route at every exact batch width 1..8 ------------
  //
  // One token per lane over a shared 200-token history, which is the decode
  // round's own shape (ADR 0020). Each lane owns a sequence, so `kv_table_rows`
  // really does select different pages per row.
  for (std::int32_t batch = 1; batch <= kMaxLanes; ++batch) {
    Inputs in;
    fill_kv(fx, /*tokens=*/1 * batch, /*first_position=*/kPrefillTokens, in.k, in.v);
    fill_q(fx, /*width=*/1, batch, in.q);
    fill_gate(in.q.size(), in.gate);
    const Call call{/*width=*/1, batch, /*first_position=*/kPrefillTokens,
                    /*history=*/kPrefillTokens};

    const std::string bf16_label = "decode bf16 B=" + std::to_string(batch);
    const std::string hq_label   = "decode hq B=" + std::to_string(batch);
    const std::vector<std::uint16_t> bf16 =
        run_route(fx, IGNIS_KV_FORMAT_BF16, call, in, bf16_label.c_str());
    const std::vector<std::uint16_t> hq =
        run_route(fx, IGNIS_KV_FORMAT_HQ_E8_2B, call, in, hq_label.c_str());

    check(lanes_differ(bf16, call.width, batch),
         "decode B=" + std::to_string(batch) +
             ": every lane produced the identical output, so the batch axis never reached the "
             "kernel and the agreement below is one lane measured " + std::to_string(batch) +
             " times");
    check(lanes_differ(hq, call.width, batch),
         "decode hq B=" + std::to_string(batch) + ": every lane produced the identical output");
    check_agreement(("decode W=1 B=" + std::to_string(batch)).c_str(), compare(bf16, hq));
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "hq route agreement test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("hq route agreement test: ok\n");
  return 0;
}
