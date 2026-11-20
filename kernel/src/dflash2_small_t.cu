// ignis kernel leaf: the DFlash2 drafter's launches of the vendored NVFP4 A16 small-T kernel
// (GitHub #694; kernel/src/dflash2_small_t.h) -- an ADAPTED op (ADR 0051 tier 2): the launch is
// restated from the vendored launcher, the levers are ours.
//
// Upstream: the reference, gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1 (the
// commit kernel/vendor/manifest.json pins, ADR 0010), Apache-2.0 (kernel/vendor/LICENSE), restated:
// src/ops/linear/nvfp4/nvfp4_small_t.cu `launch_exact` (the kernel's launch at a width: its
// schedule, its grid of the weight's rows over the schedule's rows a CTA, the contiguous output,
// the inverse weight divisor and the identity epilogue, then the launch's error check), as `launch`;
// `make_launchers` and `launchers` (the table of those launches by width from the first, indexed
// by the call's columns), as `make_launches` and `launch_at`. Called and not copied: the kernel
// nvfp4_small_t_kernel (ops/linear/nvfp4/nvfp4_small_t.cuh), the schedules
// Nvfp4LinearSmallTProductionSchedule and Nvfp4SmallTSchedule and kNvfp4LastSmallT
// (ops/linear/nvfp4/nvfp4_config.h), the output and epilogue (ops/linear/nvfp4/nvfp4_output.cuh)
// and validate_nvfp4_weight (ops/linear/nvfp4/nvfp4_format.h). Changed here: 4 warps a CTA in
// place of the production schedule's (`FourWarps`); one token tile a launch, every column in one
// CTA; a range of the weight's rows (the drafter's key and value rows, from row 4,096 of its code
// and scale planes) launched as a weight of their own; widths 5..16 and 5..32 only; a failed
// launch throws naming the op, and each launch is counted (`launches`). `require_operands`
// restates ignis's own operand checks of `ninfer::ops::linear` (kernel/src/linear.cu
// `validate_linear_semantics`), not upstream code (kernel/NOTICE).

#include "dflash2_small_t.h"

#include "process_environment.h"

#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <cuda_bf16.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace ignis::dflash2_small_t {

namespace {

using ninfer::ops::detail::Nvfp4ContiguousOutput;
using ninfer::ops::detail::Nvfp4DFlash2ConvProjGeometry;
using ninfer::ops::detail::Nvfp4DFlash2QkvGeometry;
using ninfer::ops::detail::Nvfp4GemvGeometry;
using ninfer::ops::detail::Nvfp4IdentityEpilogue;

// The warps a CTA of ours holds: the vendored schedule's own choice past 16 columns.
constexpr int kWarps = 4;

// The vendored production schedule at `Tokens` columns with 4 warps a CTA, every other field its
// own: a warp still owns its rows whole (one warp a row), sums the same values a lane in the same
// chains and reads the same activation tile, so a row's sum is the vendored launch's. Past 16
// columns it is the production schedule itself.
template <class Geometry, int Tokens> struct FourWarps {
  using Production = typename ninfer::ops::detail::Nvfp4LinearSmallTProductionSchedule<Geometry, Tokens>::Type;
  static_assert(Production::kWarpsPerRow == 1, "a row is one warp's");
  using Type = ninfer::ops::detail::Nvfp4SmallTSchedule<
      kWarps, Production::kWarpsPerRow, Production::kRowsPerWarp, Production::kValuesPerLane, Production::kTokenTile,
      Production::kAccumulatorChains, Production::kActivationAccess, Production::kScaleAccess,
      Production::kCodeCache, Production::kPhaseUnroll, Production::kBlockOrder, Production::kMinBlocksPerSm>;
};

// The widths each lever takes (dflash2_small_t.h).
constexpr std::int32_t kFirstTokens = 5;
constexpr std::int32_t kLastFillTokens = 16;
constexpr std::int32_t kLastKeyValueTokens = ninfer::ops::detail::kNvfp4LastSmallT;
static_assert(kLastKeyValueTokens == 32, "one small-T launch covers up to 32 columns");

// The key and value rows of the drafter's q | k | v weight as a weight of their own: 2,048 rows from
// row 4,096, both multiples of the scale plane's 128-row tile.
constexpr std::int32_t kQueryRows = 4096;
constexpr std::int32_t kKeyValueRows = 2048;
using KeyValueGeometry = Nvfp4GemvGeometry<kKeyValueRows, Nvfp4DFlash2QkvGeometry::kInputRows>;
static_assert(kQueryRows + kKeyValueRows == Nvfp4DFlash2QkvGeometry::kOutputRows);
static_assert(kQueryRows % 128 == 0, "the row range starts at a scale tile");

std::atomic<std::uint64_t> launch_count{0};

template <class Geometry> bool shaped(const ninfer::Weight &weight) {
  return weight.qtype == ninfer::QType::NVFP4 && weight.n == Geometry::kOutputRows &&
         weight.k == Geometry::kInputRows;
}

bool aligned16(const void *pointer) {
  return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & 15) == 0;
}

// `ninfer::ops::linear`'s checks of its operands, for [K, T] -> [rows, T].
void require_operands(const ninfer::Tensor &x, const ninfer::Tensor &out, std::int32_t rows, std::int32_t inputs,
                      const char *what) {
  const std::int32_t tokens = x.ne[1];
  const bool ok = x.dtype == ninfer::DType::BF16 && out.dtype == ninfer::DType::BF16 && x.ne[0] == inputs &&
                  x.ne[2] == 1 && x.ne[3] == 1 && out.ne[0] == rows && out.ne[1] == tokens && out.ne[2] == 1 &&
                  out.ne[3] == 1 && x.is_contiguous() && out.is_contiguous() && aligned16(x.data) &&
                  aligned16(out.data);
  if (!ok) {
    throw std::invalid_argument(std::string(what) + ": expected contiguous, 16-byte aligned BF16 [" +
                                std::to_string(inputs) + ", T] -> [" + std::to_string(rows) + ", T]");
  }
}

// One launch of the vendored kernel over `Geometry`'s rows at `Tokens` columns on 4 warps a CTA:
// `codes` and `scales` are the planes of a weight of exactly those rows.
template <class Geometry, int Tokens>
void launch(const ninfer::Tensor &x, const std::uint8_t *codes, const std::uint8_t *scales, float weight_scale_divisor,
            ninfer::Tensor &out, cudaStream_t stream) {
  using Schedule = typename FourWarps<Geometry, Tokens>::Type;
  static_assert(Schedule::kTokenTile == Tokens, "one CTA owns every column");
  constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
  const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16 *>(out.data), Geometry::kOutputRows};
  const float inverse_weight_divisor = 1.0F / weight_scale_divisor;
  ninfer::ops::detail::nvfp4_small_t_kernel<Geometry, Tokens, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(x.data), codes, scales, inverse_weight_divisor, Nvfp4IdentityEpilogue{},
      output);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("dflash2_small_t: the launch: ") + cudaGetErrorString(err));
  }
  launch_count.fetch_add(1, std::memory_order_relaxed);
}

using Launch = void (*)(const ninfer::Tensor &, const std::uint8_t *, const std::uint8_t *, float, ninfer::Tensor &,
                        cudaStream_t);

// `launch<Geometry, T>` for T = kFirstTokens + index.
template <class Geometry, std::size_t... Offsets> constexpr auto make_launches(std::index_sequence<Offsets...>) {
  return std::array<Launch, sizeof...(Offsets)>{&launch<Geometry, kFirstTokens + static_cast<int>(Offsets)>...};
}

template <class Geometry, std::int32_t LastTokens> const Launch &launch_at(std::int32_t tokens) {
  static constexpr auto kLaunches =
      make_launches<Geometry>(std::make_index_sequence<static_cast<std::size_t>(LastTokens - kFirstTokens + 1)>{});
  return kLaunches[static_cast<std::size_t>(tokens - kFirstTokens)];
}

// A whole weight of `Geometry`'s rows at the call's width.
template <class Geometry>
void launch_whole(const ninfer::Tensor &x, const ninfer::Weight &weight, ninfer::Tensor &out, cudaStream_t stream,
                  const char *what) {
  (void)ninfer::ops::detail::validate_nvfp4_weight(weight, what);
  require_operands(x, out, Geometry::kOutputRows, Geometry::kInputRows, what);
  launch_at<Geometry, kLastFillTokens>(x.ne[1])(x, static_cast<const std::uint8_t *>(weight.qdata),
                                                static_cast<const std::uint8_t *>(weight.scales),
                                                weight.weight_scale_divisor, out, stream);
}

// Whether today's A16 product of `tokens` columns of a weight of `rows` x `inputs` is one launch of
// the small-T kernel: not the tensor-core route, and no more columns than one launch covers.
bool one_small_t_launch(std::int32_t rows, std::int32_t inputs, std::int32_t tokens, IgnisA16Call call,
                        std::int32_t last_tokens) {
  return tokens >= kFirstTokens && tokens <= last_tokens &&
         !ignis_nvfp4_a16_mma_applies(rows, inputs, tokens, call);
}

} // namespace

bool append_kv_only() {
  return ignis::named_switch("IGNIS_DFLASH2_APPEND_KV_ONLY", true, {{"0", false}, {"1", true}});
}

bool small_t_fill() {
  return ignis::named_switch("IGNIS_DFLASH2_SMALL_T_FILL", true, {{"0", false}, {"1", true}});
}

bool projects_key_value_only(const ninfer::Weight &qkv, std::int32_t tokens, IgnisA16Call call) {
  return call == IgnisA16Call::General && shaped<Nvfp4DFlash2QkvGeometry>(qkv) &&
         one_small_t_launch(qkv.n, qkv.k, tokens, call, kLastKeyValueTokens) && append_kv_only();
}

void key_value_linear(const ninfer::Tensor &x, const ninfer::Weight &qkv, ninfer::Tensor &kv,
                      cudaStream_t stream) {
  const char *what = "the drafter's key and value rows";
  const std::int32_t tokens = x.ne[1];
  if (!shaped<Nvfp4DFlash2QkvGeometry>(qkv) || tokens < kFirstTokens || tokens > kLastKeyValueTokens) {
    throw std::invalid_argument(std::string(what) + ": the drafter's q | k | v weight at 5 to 32 columns");
  }
  (void)ninfer::ops::detail::validate_nvfp4_weight(qkv, what);
  require_operands(x, kv, kKeyValueRows, KeyValueGeometry::kInputRows, what);
  // The code plane is row-major, K / 2 bytes a row; the scale plane is 512 bytes for each 128-row
  // tile and each 64 inputs, the row tile outermost (ops/linear/nvfp4/nvfp4_codec.cuh
  // `nvfp4_scale_offset`).
  const std::size_t code_offset = static_cast<std::size_t>(kQueryRows) * KeyValueGeometry::kCodeBytesPerRow;
  const std::size_t scale_offset =
      static_cast<std::size_t>(kQueryRows / 128) * KeyValueGeometry::kScaleTilesPerRow * 512;
  launch_at<KeyValueGeometry, kLastKeyValueTokens>(tokens)(
      x, static_cast<const std::uint8_t *>(qkv.qdata) + code_offset,
      static_cast<const std::uint8_t *>(qkv.scales) + scale_offset, qkv.weight_scale_divisor, kv, stream);
}

bool fills(const ninfer::Weight &weight, std::int32_t tokens) {
  const bool drafter_product = shaped<Nvfp4DFlash2ConvProjGeometry>(weight) || shaped<Nvfp4DFlash2QkvGeometry>(weight);
  return drafter_product && one_small_t_launch(weight.n, weight.k, tokens, IgnisA16Call::General, kLastFillTokens) &&
         small_t_fill();
}

void forward_linear(const ninfer::Tensor &x, const ninfer::Weight &weight, ninfer::Tensor &out,
                    cudaStream_t stream) {
  if (!fills(weight, x.ne[1])) {
    ninfer::ops::linear(x, weight, out, stream);
  } else if (shaped<Nvfp4DFlash2ConvProjGeometry>(weight)) {
    launch_whole<Nvfp4DFlash2ConvProjGeometry>(x, weight, out, stream, "the drafter's conv projection");
  } else {
    launch_whole<Nvfp4DFlash2QkvGeometry>(x, weight, out, stream, "the drafter's q | k | v product");
  }
}

std::uint64_t launches() { return launch_count.load(std::memory_order_relaxed); }

} // namespace ignis::dflash2_small_t
