#!/bin/sh
# PROTOTYPE (sm8x dense prefill, phase 2d; GitHub #474), throwaway: sass-check.sh held to the i8
# band's kernels in ignis_sm8x_attn_bench, and only to them -- on sm_120a the same binary carries the
# 8-bit band (FP8 forms), the window and the FP16 band (HMMA), which the rule does not cover:
#   fa_i8_kernel<...>              every phase 2d variant and --diag copy (the last argument is the
#                                  diagnostic; every variant, diagnostic 0, must hold IMMA);
#   fa_i8_pair_kernel<...>         the variants with the head dim split over a warp pair, likewise;
#   fa_kernel<1,0,1>              the i8 band as 5ea54bc6 has it (i8ref), which must hold IMMA;
#   tile_kernel<1,0,1>             the i8 band's tiles (K rows in INT8, V^T in s8), no MMA.
#
#   tools/sm8x-dense-proto/sass-check-2d.sh BINARY [ARCH]
#
# NEGATIVE=window or NEGATIVE=fp8 checks the window's (BF16, f32 accumulate) or the 8-bit band's
# FA kernel instead, which must FAIL.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
case "${NEGATIVE:-}" in
  window)
    SASS_CHECK_PATTERN='fa_kernelILb0ELb0ELi0E' SASS_CHECK_KINDS='window=^fa_kernel<0,0,0>$' SASS_CHECK_IMMA='^$' \
      exec "$HERE/sass-check.sh" "$@" ;;
  fp8)
    SASS_CHECK_PATTERN='fa_kernelILb1ELb0ELi0E' SASS_CHECK_KINDS='fp8=^fa_kernel<1,0,0>$' SASS_CHECK_IMMA='^$' \
      exec "$HERE/sass-check.sh" "$@" ;;
esac
SASS_CHECK_PATTERN='fa_i8_kernel|fa_i8_pair_kernel|fa_kernelILb1ELb0ELi1E|tile_kernelILb1ELb0ELi1E' \
SASS_CHECK_KINDS='fa_i8=^fa_i8_kernel< pair=^fa_i8_pair_kernel< i8ref=^fa_kernel<1,0,1>$ tiles=^tile_kernel<1,0,1>$' \
SASS_CHECK_IMMA='^fa_i8_(pair_)?kernel<.*,0>$|^fa_kernel<1,0,1>$' \
  exec "$HERE/sass-check.sh" "$@"
