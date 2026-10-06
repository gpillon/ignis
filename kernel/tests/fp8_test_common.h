// FP8 row-scale weights (FP8_E4M3FN_ROW_BF16S / row-scale-v1, layout.md §6.1) for the Flash-Next
// op tests -- OURS (spec flash-next/02, GitHub #300): a payload builder and the fp64 product.
#ifndef IGNIS_FP8_TEST_COMMON_H
#define IGNIS_FP8_TEST_COMMON_H

#include "moe_fixture.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace moe_test {

struct Fp8Matrix {
  int rows = 0, cols = 0;
  std::vector<uint8_t> payload;  // codes, zero padding to 256, BF16 scales
  std::vector<double> code_value;  // e4m3 of every code, row-major
  std::vector<double> scale;
};

inline Fp8Matrix make_fp8(uint32_t stream, int rows, int cols, float scale_mag) {
  Fp8Matrix m;
  m.rows = rows;
  m.cols = cols;
  const std::size_t codes = static_cast<std::size_t>(rows) * cols;
  const std::size_t scale_at = (codes + 255) / 256 * 256;
  m.payload.assign(scale_at + static_cast<std::size_t>(rows) * 2, 0);
  m.code_value.resize(codes);
  for (std::size_t i = 0; i < codes; ++i) {
    uint8_t c = static_cast<uint8_t>(hash_u32(stream, i) & 0xFFu);
    if ((c & 0x7F) == 0x7F) c ^= 1u;  // E4M3FN's NaN codes are not weights
    m.payload[i] = c;
    m.code_value[i] = e4m3_to_f32(c);
  }
  m.scale.resize(rows);
  for (int r = 0; r < rows; ++r) {
    const uint16_t s = f32_to_bf16(scale_mag * (0.5f + std::fabs(hash_uniform(stream + 1, r, 1.0f))));
    std::memcpy(&m.payload[scale_at + 2 * static_cast<std::size_t>(r)], &s, 2);
    m.scale[r] = bf16_to_f32(s);
  }
  return m;
}

// y = scale[r] * sum_c code[r][c] * x[c] in fp64, and the bound's sum of |terms| * |scale|.
inline void fp8_row_f64(const Fp8Matrix &m, int r, const double *x, double *y, double *abs_sum) {
  double s = 0.0, a = 0.0;
  const double *row = &m.code_value[static_cast<std::size_t>(r) * m.cols];
  for (int c = 0; c < m.cols; ++c) {
    s += row[c] * x[c];
    a += std::fabs(row[c] * x[c]);
  }
  *y = s * m.scale[r];
  *abs_sum = a * std::fabs(m.scale[r]);
}

// The fp32 accumulation bound of both routes: products are exact, so error <= n u sum|terms| over
// the longest chain; the GEMV's lane chains are cols / 32 + 5 long and the MMA route's at most
// cols / 16 instructions plus the instruction's own 16-term reduction. Stated with margin as
// (cols / 8 + 64) u.
inline double fp8_sum_bound(int cols, double abs_sum) {
  return std::ldexp(static_cast<double>(cols / 8 + 64), -24) * abs_sum;
}

}  // namespace moe_test

#endif  // IGNIS_FP8_TEST_COMMON_H
