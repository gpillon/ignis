#!/bin/sh
# PROTOTYPE (sm8x dense prefill, phase 2c; GitHub #474), throwaway: the phase-2c ablation of the dense
# INT8 prefill on an sm_86 card, in one command, into one plain-text report. Everything run-2b.sh
# reports, plus each of the four levers alone and their combinations.
#
#   ./run-2c.sh ARTIFACT [OUT_DIR]
#
# What it needs, and nothing else:
#   - beside this script, the bench: ignis_sm8x_dense_bench (linux/amd64) or, from Git Bash on
#     Windows, ignis_sm8x_dense_bench.exe. Both are built with CUDA 13.1 for sm_86 and sm_120a and
#     carry the CUDA runtime statically (the Linux one also libstdc++ and libgcc; it needs glibc
#     2.34 or newer). The trellis kernels it compares against are compiled into it.
#   - ARTIFACT: the base 16 GB trellis artifact, qwen3_8_27b_trellis_sc30-v2.ninfer. The bench reads
#     five of its matrices and the input rotations of their siblings (about 250 MB).
#   - an NVIDIA driver that supports CUDA 13.1 or newer (the "CUDA Version" nvidia-smi prints); the
#     bench refuses an older one by name.
#   - about 2.5 GB of free device memory, and the card to itself: the bench times kernels. The
#     script stops if another process holds the card, unless FORCE=1.
#   - a POSIX shell. nvidia-smi is optional (it adds the card's limits to the report).
#
# The levers, each behind a switch of the bench (--ablation runs them all):
#   A  one A8 launch per shared input (each matrix with its own input rotation)
#   B  a persistent tile scheduler, one CTA per SM, with a stream-K tail, exact in int32
#   C  the decode in L2-sized output slices, each multiplied while it is still in the L2
#   D  the GEMM main loop with its fragments double-buffered across the stage boundary, over 4 or
#      3 cp.async stages
#
# What it writes: OUT_DIR/report-2c.txt (default ./out-2c), the run's setting (date, OS, the card's
# compute capability, memory and clock limits, the driver) followed by the bench's output in full:
#   kernels   each new kernel's registers, spills, shared memory and CTAs per SM
#   L2        the L2-resident read bandwidth (a working set of at most half the L2)
#   peaks     register-loop rates: s8 m16n8k32, f16 m16n8k16 with f32 and with f16 accumulate
#   check     on the artifact's matrices, under every configuration of the ablation: the W8 codes
#             and the A8 values bit for bit, the GEMM's int32 accumulators against an exact
#             integer reference, the outputs against fp64 and against phase 1's V2 emulation
#             (PASS / FAIL per line, one summary line per configuration)
#   diag      the GEMM without its main loop's global loads, and without its ldmatrix, for each
#             main loop; the decode into the whole W against one slice buffer; lever C taken
#             apart; lever A's fold into the producers, estimated
#   speed     per configuration and per body shape at T = 2048 and 4096: decode, A8, GEMM (TOPS,
#             % of the s8 peak), their total, today's fused trellis prefill kernel, the ratio,
#             and the per-chunk totals
#   B and D   per shape, the GEMM under the grid launch, the persistent one, and each with lever D
#   ablation  per T, one per-chunk row per configuration, against phase 2a's route
#   overlap   the next matrix's (or slice's) decode on a second stream, for phase 2a's route and
#             for the lever combinations
#   project   the body's prefill GEMMs per 8K and per 64K tokens, per configuration
# and OUT_DIR/stderr-2c.txt (the bench's own mode lines). The exit status is the bench's: 0, or 1
# when a check failed. The run takes much longer than run-2b.sh's: expect half an hour to an hour.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
if [ $# -lt 1 ]; then
  echo "usage: $0 ARTIFACT [OUT_DIR]" >&2
  exit 2
fi
ART=$1
OUT=${2:-$HERE/out-2c}
BIN=$HERE/ignis_sm8x_dense_bench
# A copy that lost its executable bit on the way gets it back.
if [ -f "$BIN" ] && [ ! -x "$BIN" ]; then chmod +x "$BIN" 2>/dev/null || true; fi
[ -x "$BIN" ] || BIN=$HERE/ignis_sm8x_dense_bench.exe
if [ ! -x "$BIN" ]; then
  echo "run-2c: no ignis_sm8x_dense_bench beside this script" >&2
  exit 2
fi
if [ ! -r "$ART" ]; then
  echo "run-2c: cannot read the artifact $ART" >&2
  exit 2
fi
mkdir -p "$OUT"
REPORT=$OUT/report-2c.txt

if command -v nvidia-smi >/dev/null 2>&1; then
  # Busy only by nvidia-smi's own answer: a failing nvidia-smi prints its error on stdout.
  BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null) || BUSY=
  if [ -n "$BUSY" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "run-2c: another process holds the card (pid, memory):" >&2
    echo "$BUSY" >&2
    echo "run-2c: wait for it, or run with FORCE=1" >&2
    exit 3
  fi
fi

{
  echo "sm8x dense prefill, phase 2c (prototype, GitHub #474): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "os: $(uname -sm)"
  echo "bench: $(basename "$BIN")"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "card (nvidia-smi):"
    nvidia-smi --query-gpu=compute_cap,memory.total,memory.used,clocks.max.sm,clocks.max.memory,power.limit,pcie.link.gen.max,pcie.link.width.max,driver_version --format=csv 2>&1 | sed 's/^/  /'
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$CC" ] && [ "$CC" != "8.6" ]; then
      echo "note: this run is meant for an sm_86 card; this card is compute capability $CC"
    fi
  else
    echo "card: no nvidia-smi"
  fi
  echo
} > "$REPORT"

START=$(date +%s)
"$BIN" --artifact "$ART" --tokens 2048,4096 --reps 10 --diag --ablation >> "$REPORT" 2> "$OUT/stderr-2c.txt"
STATUS=$?
END=$(date +%s)
{
  echo
  echo "run-2c: bench exit status $STATUS, $((END - START)) s"
} >> "$REPORT"
echo "run-2c: wrote $REPORT (bench exit status $STATUS, $((END - START)) s)"
grep -E "check\(s\) FAILED|^check \[|^== ablation at|^[A-D+]+ +\||^phase 2a +\|" "$REPORT" || true
exit $STATUS
