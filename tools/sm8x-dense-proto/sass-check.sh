#!/bin/sh
# PROTOTYPE (sm8x dense prefill, phase 2a; GitHub #474), throwaway: the seed of the spec's SASS
# check. Dumps the SASS of one architecture's cubins in the dense bench and holds its new kernels
# (every function whose name holds sm8x_dense_: the decode, the A8 quantization, the GEMM) to the MMA
# forms a GeForce card runs at full rate:
#   - FAIL on any HMMA with an F32 accumulator, any BF16, TF32 or FP8/FP6/FP4 form, and any
#     QMMA, OMMA or warpgroup/tensor-memory MMA;
#   - FAIL unless every GEMM instantiation holds IMMA, and unless the decode, A8 and GEMM kernels
#     are all in the dump;
#   - an HMMA with an F16 accumulator is allowed by the rule but reported (none is expected).
# Prints each new kernel's MMA opcodes with their counts. The bench's meters (the register-loop
# peaks, the reference) and today's fused kernel are other functions and are not held to it.
#
#   tools/sm8x-dense-proto/sass-check.sh BINARY [ARCH]
#
# ARCH defaults to sm_86 (also sm_120a). cuobjdump comes from PATH, else $CUDA_PATH/bin.
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
"$CUOBJDUMP" -sass -arch "$ARCH" "$BIN" | awk -v arch="$ARCH" '
function pretty(fn,   name, args, rest) {
  name = fn
  sub(/^.*sm8x_dense_/, "sm8x_dense_", name)
  match(name, /^sm8x_dense_[a-z0-9]+/)
  rest = substr(name, RLENGTH + 1)
  name = substr(name, 1, RLENGTH)
  args = ""
  while (match(rest, /Li[0-9]+E/)) {
    args = args (args == "" ? "" : ",") substr(rest, RSTART + 2, RLENGTH - 3)
    rest = substr(rest, RSTART + RLENGTH)
  }
  return args == "" ? name : name "<" args ">"
}
/Function : / { fn = $3; if (fn ~ /sm8x_dense_/) { seen[fn] = 1; nfn++ } next }
fn ~ /sm8x_dense_/ {
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
  kinds["decode"] = 0; kinds["a8"] = 0; kinds["gemm"] = 0
  for (f in seen) {
    p = pretty(f)
    for (k in kinds) if (p ~ ("^sm8x_dense_" k)) kinds[k]++
    n = split(list[f], ops, " ")
    shown = ""; imma = 0
    for (i = 1; i <= n; i++) {
      op = ops[i]
      shown = shown sprintf(" %s x%d", op, count[f SUBSEP op])
      if (op ~ /^IMMA/) imma++
      if (op ~ /^(QMMA|OMMA|HGMMA|IGMMA|QGMMA|UTC)/ || (op ~ /^HMMA/ && op ~ /(F32|BF16|TF32|E4M3|E5M2|E2M1|E3M2|E2M3)/)) {
        printf("FAIL %s: %s is not a full-rate GeForce form\n", p, op); fail = 1
      } else if (op ~ /^HMMA/) {
        printf("note %s: %s (F16 accumulate) present\n", p, op)
      }
    }
    if (p ~ /^sm8x_dense_gemm/ && imma == 0) { printf("FAIL %s: no IMMA\n", p); fail = 1 }
    printf("%s %-28s%s\n", arch, p, shown == "" ? " (no MMA)" : shown)
  }
  for (k in kinds) if (kinds[k] == 0) { printf("FAIL: no sm8x_dense_%s kernel in the %s dump\n", k, arch); fail = 1 }
  if (fail) { print "sass-check " arch ": FAIL"; exit 1 }
  printf("sass-check %s: PASS (%d new kernels, integer MMA only)\n", arch, nfn)
}'
