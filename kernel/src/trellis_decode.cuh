// ignis kernel leaf: the trellis decode of an expert projection -- OURS (spec flash-next/02,
// ADR 0044), written from the format's definition (kernel/tests/fixtures/flash_next/record.py,
// `decode_inner`), not from any engine's source.
//
// The format, as the expert ops read it. A projection's inner weight W (in x out, fp16) is
// stored as 16 x 16 tiles, tile (tk, tn) at word (tk * out/16 + tn) * 16 K of an int16 tensor.
// A tile is a bitstream of 256 trellis steps; step p adds k_p bits (K, or for K = 2.5 two bits
// at even p and three at odd p). Read as little-endian u32 words, the stream is big-endian: stream
// bit b is bit 31 - b % 32 of u32 word b / 32. Step p's state is the 16 stream bits ending at
// end_p = k_0 + ... + k_p, most significant first, wrapping around the tile (tail-biting). Its
// value is the mul1 codebook: prod = state * 0x83DCD12D (mod 2^32), bsum = the sum of prod's
// bytes, value = fp16((1024 + bsum) * fp16(0x1EEE) + fp16(0xC931)) with one rounding -- one
// IMAD, one DP4A (whose accumulator input 0x6400 makes the sum the fp16 bits of 1024 + bsum)
// and half an HFMA2.
//
// Step p of a tile is element (r, c), r * 16 + c = perm[p], perm being exllamav3's tensor-core
// order: lane t of a warp owns steps 8t .. 8t+7, which are exactly its m16n8k16 B-operand
// registers (k = r, n = c) for both n8 halves of the tile. So a warp decodes a tile straight
// into MMA fragments: lane t needs only the 16 + 8K - K stream bits ending its last step, which
// for every K in {2, 2.5, 3, 4} lie inside two consecutive u32 words of the tile.
#ifndef IGNIS_TRELLIS_DECODE_CUH
#define IGNIS_TRELLIS_DECODE_CUH

#include <cuda_fp16.h>

#include <cstdint>

namespace ignis_trellis {

constexpr uint32_t kMul1 = 0x83DCD12Du;

// Exclusive end bit of step p in a tile of class k2 = 2 K.
__host__ __device__ constexpr int step_end(int k2, int p) {
  return k2 == 5 ? 3 * ((p + 1) / 2) + 2 * (p / 2 + 1) : (k2 / 2) * (p + 1);
}

// u32 words per tile: 256 * K / 32.
__host__ __device__ constexpr int tile_words(int k2) { return 4 * k2; }

// Which two u32 words of a tile lane `lane` reads, and how far its first window bit sits below
// the top of the 64-bit window (word w0 high, word w1 low). Shifting the window left by `off`
// puts every lane's first bit at bit 63, after which the eight states sit at the same offsets
// in every lane: a lane's steps start at an even step, so even K = 2.5's 2/3-bit pattern is
// lane-independent from there.
struct LanePlan {
  int w0;
  int w1;
  uint32_t off;
};

__host__ __device__ inline LanePlan lane_plan(int k2, int lane) {
  LanePlan plan{};
  const int words = tile_words(k2);
  const int first = step_end(k2, 8 * lane) - 16;  // first window bit; negative only for lane 0
  const int w0 = first >= 0 ? first / 32 : -1;
  plan.w0 = w0 < 0 ? words - 1 : w0;
  plan.w1 = (w0 + 1) % words;  // read but unused when the window ends inside w0
  plan.off = static_cast<uint32_t>(first - 32 * w0);
  return plan;
}

// The right shift that brings state j of a lane (0..7) to the bottom of the normalized window:
// state j ends 16 + (end of step j - end of step 0) bits below the window's top.
__host__ __device__ constexpr int state_shift(int k2, int j) {
  return 64 - (16 + step_end(k2, j) - step_end(k2, 0));
}

// Lane `plan`'s eight weights of one tile of class K2, from that tile's u32 words `word0`
// (= tile[w0]) and `word1` (= tile[w1]), as four half2 registers in fragment order: frag[0] = B
// rows (k0, k0 + 1) of n8 half 0, frag[1] = rows (k0 + 8, k0 + 9) of half 0, frag[2] and frag[3]
// the same for half 1.
template <int K2>
__device__ __forceinline__ void decode_fragment(uint32_t word0, uint32_t word1, const LanePlan &plan,
                                                uint32_t (&frag)[4]) {
  const uint64_t window = ((static_cast<uint64_t>(word0) << 32) | word1) << plan.off;
  const uint32_t hi = static_cast<uint32_t>(window >> 32);
  const uint32_t lo = static_cast<uint32_t>(window);
  uint32_t bits[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int sh = state_shift(K2, j);  // a constant once the loop is unrolled
    const uint32_t raw = sh >= 32 ? hi >> (sh - 32) : __funnelshift_r(lo, hi, sh);
    bits[j] = __dp4a((raw & 0xFFFFu) * kMul1, 0x01010101u, 0x6400u);
  }
  const __half2 k_inv = __halves2half2(__ushort_as_half(0x1EEE), __ushort_as_half(0x1EEE));
  const __half2 k_bias = __halves2half2(__ushort_as_half(0xC931), __ushort_as_half(0xC931));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t packed = __byte_perm(bits[2 * i], bits[2 * i + 1], 0x5410);
    const __half2 v = __hfma2(*reinterpret_cast<const __half2 *>(&packed), k_inv, k_bias);
    frag[i] = *reinterpret_cast<const uint32_t *>(&v);
  }
}

}  // namespace ignis_trellis

#endif  // IGNIS_TRELLIS_DECODE_CUH
