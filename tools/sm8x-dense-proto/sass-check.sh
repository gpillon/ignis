#!/bin/sh
# PROTOTYPE (sm8x dense prefill, phases 2a, 2c and 2d; GitHub #474), throwaway: the seed of the spec's
# SASS check. Dumps the SASS of one architecture's cubins in a bench and holds its new kernels to
# the MMA forms a GeForce card runs at full rate:
#   - FAIL on any HMMA with an F32 accumulator, any BF16, TF32 or FP8/FP6/FP4 form, and any
#     QMMA, OMMA or warpgroup/tensor-memory MMA;
#   - FAIL unless every kernel the IMMA pattern names holds IMMA, and unless every required kind
#     is in the dump;
#   - an HMMA with an F16 accumulator is allowed by the rule but reported (none is expected).
# Prints each new kernel's MMA opcodes with their counts.
#
#   tools/sm8x-dense-proto/sass-check.sh BINARY [ARCH]
#
# ARCH defaults to sm_86 (also sm_120a). cuobjdump comes from PATH, else $CUDA_PATH/bin.
# The defaults hold the dense bench's kernels (every function whose name holds sm8x_dense_: the
# decode, the A8 quantization and phase 2c's shared A8, the GEMM and phase 2c's persistent GEMM);
# its meters (the register-loop peaks, the reference) and today's fused kernel are other functions.
# Three variables point it at another bench:
#   SASS_CHECK_PATTERN  the mangled names held to the rule (default sm8x_dense_); a pattern naming
#                       another kernel is the check's negative control (peak_kernel, the dense
#                       bench's f32-accumulate meter, must FAIL);
#   SASS_CHECK_KINDS    space-separated LABEL=REGEX pairs over the printed names, each of which must
#                       match a kernel in the dump (default: the dense bench's five kinds);
#   SASS_CHECK_IMMA     a regex over the printed names of the kernels that must hold IMMA (default
#                       ^sm8x_dense_gemm).
# A printed name is the kernel's name and its integer and bool template arguments, in order:
# fa_i8_kernel<2,0,0,0,0,0,0,0,0>.
set -eu
if [ $# -lt 1 ]; then
  echo "usage: $0 BINARY [ARCH]" >&2
  exit 2
fi
BIN=$1
ARCH=${2:-sm_86}
CUOBJDUMP=$(command -v cuobjdump 2>/dev/null || true)
if [ -z "$CUOBJDUMP" ] && [ -n "${CUDA_PATH:-}" ]; then
  CUOBJDUMP="$CUDA_PATH/bin/cuobjdump"
fi
if [ -z "$CUOBJDUMP" ]; then
  echo "sass-check: no cuobjdump on PATH or under \$CUDA_PATH/bin" >&2
  exit 2
fi
KINDS=${SASS_CHECK_KINDS:-"decode=^sm8x_dense_decode a8=^sm8x_dense_a8 a8_shared=^sm8x_dense_a8_shared gemm=^sm8x_dense_gemm gemm_persistent=^sm8x_dense_gemm_persistent"}
"$CUOBJDUMP" -sass -arch "$ARCH" "$BIN" | awk -v arch="$ARCH" -v pat="${SASS_CHECK_PATTERN:-sm8x_dense_}" \
  -v kinds_spec="$KINDS" -v imma_pat="${SASS_CHECK_IMMA:-^sm8x_dense_gemm}" '
function pretty(fn,   name, args, rest, n) {
  name = fn
  if (match(name, /sm8x_dense_[a-z0-9_]+/)) {
    rest = substr(name, RSTART + RLENGTH)
    name = substr(name, RSTART, RLENGTH)
  } else if (name ~ /^_Z/) {
    # The last length-prefixed name of _Z[N] <len><name>...: past any (anonymous) namespace.
    rest = substr(name, substr(name, 1, 3) == "_ZN" ? 4 : 3)
    name = ""
    while (match(rest, /^[0-9]+/)) {
      n = substr(rest, 1, RLENGTH) + 0
      name = substr(rest, RLENGTH + 1, n)
      rest = substr(rest, RLENGTH + 1 + n)
    }
  } else {
    rest = ""
  }
  args = ""
  while (match(rest, /L[ib][0-9]+E/)) {
    args = args (args == "" ? "" : ",") substr(rest, RSTART + 2, RLENGTH - 3)
    rest = substr(rest, RSTART + RLENGTH)
  }
  return args == "" ? name : name "<" args ">"
}
/Function : / { fn = $3; if (fn ~ pat) { seen[fn] = 1; nfn++ } next }
fn ~ pat {
  line = $0
  while (match(line, /[A-Z]*MMA[.A-Z0-9_]*/)) {
    op = substr(line, RSTART, RLENGTH)
    line = substr(line, RSTART + RLENGTH)
    key = fn SUBSEP op
    if (!(key in count)) { list[fn] = list[fn] " " op }
    count[key]++
  }
}
END {
  fail = 0
  nk = split(kinds_spec, kspec, " ")
  for (i = 1; i <= nk; i++) {
    eq = index(kspec[i], "=")
    klabel[i] = substr(kspec[i], 1, eq - 1)
    kre[i] = substr(kspec[i], eq + 1)
    kcount[i] = 0
  }
  for (f in seen) {
    p = pretty(f)
    for (i = 1; i <= nk; i++) if (p ~ kre[i]) kcount[i]++
    n = split(list[f], ops, " ")
    shown = ""; imma = 0
    for (j = 1; j <= n; j++) {
      op = ops[j]
      shown = shown sprintf(" %s x%d", op, count[f SUBSEP op])
      if (op ~ /^IMMA/) imma++
      if (op ~ /^(QMMA|OMMA|HGMMA|IGMMA|QGMMA|UTC)/ || (op ~ /^HMMA/ && op ~ /(F32|BF16|TF32|E4M3|E5M2|E2M1|E3M2|E2M3)/)) {
        printf("FAIL %s: %s is not a full-rate GeForce form\n", p, op); fail = 1
      } else if (op ~ /^HMMA/) {
        printf("note %s: %s (F16 accumulate) present\n", p, op)
      }
    }
    if (p ~ imma_pat && imma == 0) { printf("FAIL %s: no IMMA\n", p); fail = 1 }
    printf("%s %-36s%s\n", arch, p, shown == "" ? " (no MMA)" : shown)
  }
  for (i = 1; i <= nk; i++) if (kcount[i] == 0) { printf("FAIL: no %s kernel (%s) in the %s dump\n", klabel[i], kre[i], arch); fail = 1 }
  if (fail) { print "sass-check " arch ": FAIL"; exit 1 }
  printf("sass-check %s: PASS (%d new kernels, full-rate GeForce forms only: IMMA, and HMMA.F16 where noted)\n", arch, nfn)
}'
