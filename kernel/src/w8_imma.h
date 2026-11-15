// ignis kernel leaf: the 27B MTP head's W8G32 matrices on an RTX 30 or RTX 40 card, in the forms
// those cards run at full rate -- OURS (GitHub #627, spec #618), leaf-internal.
//
// The head's five matrices (kernel/src/mtp_head.h) are W8G32 on every 27B artifact: int8 codes,
// one FP16 scale per row and 32 inputs. On Blackwell they run on the vendored `ops::linear`,
// whose routes at the head's geometries are all BF16 m16n8k16 MMA with an fp32 accumulator: its
// small-T kernel (up to 40 or 48 tokens), which only the sm_120a set holds, and the MMA kernels
// above it, the form a GeForce Ampere or Ada card runs at half rate (the full-rate rule,
// mk/full-rate-kernels.txt). On those cards a call runs on one of two routes, by its width:
//
//   - up to kSimtMaxTokens tokens (a decode round's and a chain step's lanes, the readout's one
//     row): the vendored W8 SIMT route, `launch_w8_simt_r8_c4` / `_c8`, called and not copied,
//     the route the vision tower already runs on these cards: the CUDA cores alone, each weight
//     dequantized to fp32 and every product an fp32 FMA on the BF16 activation, the matrix read
//     once; below that width a call is a read of the matrix, which no MMA form shortens;
//   - wider (a round's commit over several lanes, a prefill call of up to 128 rows): ours, the
//     integer MMA. The activation is split, per token and per 32 inputs (the weight's group), into
//     a 16-bit fixed point q = round(x 32512 / amax) written as two INT8 planes, q = 256 hi + lo,
//     hi in [-127, 127], lo in [-128, 127], with the scale amax / 32512 (`detail::split`). The
//     GEMM runs one mma.sync m16n8k32 s8 x s8 -> s32 per plane and group, the group's exact
//     integer sums to fp32 (exact below 2^22, which a group's 32 products of |code| <= 128 by
//     |hi| <= 127 or |lo| <= 128 stay under), times the weight's and the activation's scales, an
//     fp32 FMA per group: out[n, t] = sum_g ws[n, g] a[t, g] (256 sum w hi + sum w lo).
//
// The fixed point holds each activation within amax / 65024 of its group's largest, a 2^-16
// relative step that sits far below the BF16 rounding of every output (2^-8), so the head on
// these cards computes what it computes on Blackwell within the roundings the head's f64 checks
// already allow (crates/core/tests/mtp_head_gpu.rs). Plain A8, one INT8 plane, would add an
// activation step of 2^-8 at every matrix. The GEMM is deterministic, and the same at every
// width: the groups' order and the K split over the CTA's warps do not change with the tokens, so
// a column's bits do not depend on the columns beside it.
//
// The activation planes take `split_bytes(k, tokens)` bytes the caller hands over (`room`): the
// head gives each call a buffer of its own call that is dead or not yet written at that point
// (kernel/src/mtp_head.cu), so the VRAM plan is the same on every card. The SIMT route uses none.

#pragma once

#include "core/tensor.h"
#include "process_environment.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ignis::w8_imma {

// The widest call the vendored SIMT route takes (`linear`): its column tile is 4 or 8, so up to 8
// tokens it reads the matrix once.
inline constexpr std::int32_t kSimtMaxTokens = 8;

// The split activation's fixed point: |q| <= kSplitRange, so that q = 256 hi + lo with hi in
// [-127, 127] and lo in [-128, 127].
inline constexpr std::int32_t kSplitRange = 127 * 256;

// The route per GPU generation: ours on every card below compute capability 12 (Ampere and Ada,
// where the vendored routes are the half-rate BF16 MMA or, at the narrow widths, no kernel image
// at all), the vendored `ops::linear` on Blackwell unless the measurement switch names ours,
// which runs this route on a Blackwell card for a comparison; `vendored` changes nothing below 12.
constexpr bool takes_ours(RouteSwitch named, int compute_capability_major) {
  return compute_capability_major < 12 || named == RouteSwitch::Ours;
}

// The route a load takes (true: ours): `takes_ours` with the process environment's
// IGNIS_MTP_HEAD_LINEAR (`ours` or `vendored`, a measurement switch, not an operator setting),
// read at each call (kernel/src/model.cu calls it once per load). Throws std::invalid_argument,
// naming the variable, on any other value.
bool route_for_load(int compute_capability_major);

// Whether `w` is a matrix this route takes: W8G32 in the row-split layout, its rows a multiple of
// 16 and its inputs of 256 (the K split over the GEMM's 8 warps), as each of the head's five is.
bool serves(const ninfer::Weight &w);

// The bytes of the activation planes of a [k, tokens] input: hi and lo INT8 [tokens][k], then the
// scales fp32 [k / 32][tokens], each part 256-byte aligned.
std::size_t split_bytes(std::int32_t k, std::int32_t tokens);

// out = w x: x BF16 [k, T], out BF16 [n, T], both contiguous and 16-byte aligned, `w` one `serves`
// takes. T <= kSimtMaxTokens runs the vendored SIMT route and reads no room; a wider call writes
// the activation planes into `room` (256-byte aligned, at least `split_bytes(k, T)` bytes, none of
// which x or out may overlap) and runs the integer MMA. Enqueues on `stream`, allocates nothing
// and never synchronizes the host: capturable in a CUDA graph. Throws std::invalid_argument on
// anything else.
void linear(const ninfer::Tensor &x, const ninfer::Weight &w, ninfer::Tensor &out, void *room,
            std::size_t room_bytes, cudaStream_t stream);

// The two steps of the wide route, at any width, for its op test (kernel/tests/test_w8_imma.cu).
namespace detail {

struct Planes {
  std::int8_t *hi = nullptr;  // [tokens][k]
  std::int8_t *lo = nullptr;  // [tokens][k]
  float *scale = nullptr;     // [k / 32][tokens]
};

// Where the planes of a [k, tokens] input sit in `room`.
Planes planes(void *room, std::int32_t k, std::int32_t tokens);

// The activation's planes (above): per token and group, amax the largest |x|, q = clamp(rn(x
// (kSplitRange / amax)), +-kSplitRange) in fp32 (ties to even), hi = (q + 128) >> 8, lo = q - 256
// hi, scale = amax / kSplitRange; a group whose amax is 0 or not finite splits to zeros with scale
// 0. x as `linear` takes it, k a multiple of 256.
void split(const ninfer::Tensor &x, void *room, std::size_t room_bytes, cudaStream_t stream);

// The wide route, `split` then the integer MMA, at any width.
void imma(const ninfer::Tensor &x, const ninfer::Weight &w, ninfer::Tensor &out, void *room,
          std::size_t room_bytes, cudaStream_t stream);

}  // namespace detail

}  // namespace ignis::w8_imma
