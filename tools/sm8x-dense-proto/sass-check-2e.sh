#!/bin/sh
# PROTOTYPE (sm8x dense prefill, phase 2e; GitHub #474), throwaway: sass-check.sh held to the
# kernels phase 2e runs in ignis_sm8x_attn_bench, and only to them -- on sm_120a the same binary
# carries the 8-bit band (FP8 forms) and the window as served (BF16, f32 accumulate), which the
# rule does not cover:
#   fa_i8f_kernel<...>             the fold group, the lazy rescale, the buffers, Q in shared memory
#                                  (the last argument is the diagnostic; every one with 0 must hold
#                                  IMMA), integer MMA only;
#   fa_i8qk2_kernel<0>             the control (INT8 Q K^T, FP16 P V): IMMA, and HMMA with an F16
#                                  accumulator only;
#   fa_kernel<0,1,0>               the FP16 band's kernel, which the W16 window runs: HMMA.F16 only;
#   fa_kernel<1,0,2>               5ea54bc6's i8qk, likewise;
#   tile_vgroup_kernel<...>        V^T with a scale per fold group, no MMA;
#   tile_kernel<1,0,2>             i8qk's tiles (INT8 K, FP16 V rows), no MMA;
#   and 2d's: fa_i8_kernel, fa_i8_pair_kernel, fa_kernel<1,0,1> (i8ref), tile_kernel<1,0,1>.
#
#   tools/sm8x-dense-proto/sass-check-2e.sh BINARY [ARCH]
#
# NEGATIVE=window or NEGATIVE=fp8 checks the window as served (BF16, f32 accumulate) or the 8-bit
# band's FA kernel instead, which must FAIL.
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
SASS_CHECK_PATTERN='fa_i8f_kernel|fa_i8qk2_kernel|tile_vgroup_kernel|fa_i8_kernel|fa_i8_pair_kernel|fa_kernelILb1ELb0ELi1E|fa_kernelILb1ELb0ELi2E|fa_kernelILb0ELb1ELi0E|tile_kernelILb1ELb0ELi1E|tile_kernelILb1ELb0ELi2E' \
SASS_CHECK_KINDS='i8f=^fa_i8f_kernel< qk2=^fa_i8qk2_kernel< vgroup=^tile_vgroup_kernel< w16=^fa_kernel<0,1,0>$ i8qk=^fa_kernel<1,0,2>$ fa_i8=^fa_i8_kernel< pair=^fa_i8_pair_kernel< i8ref=^fa_kernel<1,0,1>$ tiles=^tile_kernel<1,0,1>$ qktiles=^tile_kernel<1,0,2>$' \
SASS_CHECK_IMMA='^fa_i8f_kernel<.*,0>$|^fa_i8qk2_kernel<0>$|^fa_i8_(pair_)?kernel<.*,0>$|^fa_kernel<1,0,[12]>$' \
  exec "$HERE/sass-check.sh" "$@"
