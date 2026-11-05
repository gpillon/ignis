#!/bin/sh
# PROTOTYPE (sm8x dense prefill, phase 2d; GitHub #474), throwaway: the 27B's prompt attention on an
# sm_86 card, in one command, into one plain-text report: a whole causal prompt of 8K and of 64K
# tokens at chunk 4096, for the bands such a card can run, the i8 band held byte for byte to the
# kernel phase 1 measured, and the i8 kernel's stages and levers.
#
#   ./run-2d.sh [OUT_DIR]
#
# What it needs, and nothing else:
#   - beside this script, the bench: ignis_sm8x_attn_bench (linux/amd64) or, from Git Bash on
#     Windows, ignis_sm8x_attn_bench.exe. Both are built with CUDA 13.1 for sm_86 and sm_120a and
#     carry the CUDA runtime statically (the Linux one also libstdc++ and libgcc; it needs glibc
#     2.34 or newer). The attention's kernels and the hq-e8-2b codec's encoder are compiled into it:
#     it builds its own cache of Gaussian keys, so it reads no model.
#   - an NVIDIA driver that supports CUDA 13.1 or newer (the "CUDA Version" nvidia-smi prints); the
#     bench refuses an older one.
#   - about 1 GB of free device memory, and the card to itself: the bench times kernels. The script
#     stops if another process holds the card, unless FORCE=1 (under Windows nvidia-smi cannot tell
#     a compute process from the desktop, so there the script cannot check).
#   - a POSIX shell. nvidia-smi is optional (it adds the card's limits to the report).
#
# The bands (the bench's own header says more):
#   fp16   the FP16 band, what the card runs today (FP16 MMA with FP16 accumulate);
#   fp8    today's 8-bit band, compute capability 12.0 only (skipped below it);
#   i8ref  the i8 band as phase 1 measured it: INT8 Q K^T, P V on the u8 x s8 MMA;
#   i8     the same band on phase 2d's kernel, which must match i8ref byte for byte.
#
# What it writes: OUT_DIR/report-2d.txt (default ./out-2d), the run's setting (date, OS, the card's
# compute capability, memory and clock limits, the driver) followed by the bench's output in full:
#   kernels   each kernel's registers, spills, shared memory and CTAs per SM
#   variants  the i8 kernel's 24 lever combinations, by number
#   peaks     register-loop MMA rates: s8, f16 with f16 accumulate, f16 and bf16 with f32
#             accumulate (and block-scaled e4m3 on 12.0); the efficiencies are against these
#   check     every chunk of both prompts, every i8 variant, at the op's key splits and at 1 and 3
#             splits: the output and the state carried between passes, compared word for word with
#             i8ref (PASS / FAIL per line), and each other band's distance from i8ref (a broken
#             setup fails here)
#   speed     per prompt: per chunk and pass for each band (Q prep, window tiles, window FA, band
#             tiles, band FA, merge), then per prompt over the 16 GQA layers, with each band's MMA
#             floor and efficiency; every i8 variant; --diag: the i8 kernel with P V, the softmax or
#             Q K^T taken out, and every key split from 1 to 5 for i8 and fp16
# and OUT_DIR/stderr-2d.txt. The exit status is the bench's: 0, or 1 when a check failed. Expect
# about 10 to 20 minutes on an sm_86 card (the 5090 takes about a minute).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/out-2d}
BIN=$HERE/ignis_sm8x_attn_bench
# A copy that lost its executable bit on the way gets it back.
if [ -f "$BIN" ] && [ ! -x "$BIN" ]; then chmod +x "$BIN" 2>/dev/null || true; fi
[ -x "$BIN" ] || BIN=$HERE/ignis_sm8x_attn_bench.exe
if [ ! -x "$BIN" ]; then
  echo "run-2d: no ignis_sm8x_attn_bench beside this script" >&2
  exit 2
fi
mkdir -p "$OUT"
REPORT=$OUT/report-2d.txt

if command -v nvidia-smi >/dev/null 2>&1; then
  # Busy only by nvidia-smi's own answer: a failing nvidia-smi prints its error on stdout. Under
  # Windows' WDDM every process on the card, the desktop's too, shows [N/A] memory, so those rows
  # say nothing and are skipped: there, keep the card to yourself.
  BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -v '\[N/A\]') || BUSY=
  if [ -n "$BUSY" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "run-2d: another process holds the card (pid, memory):" >&2
    echo "$BUSY" >&2
    echo "run-2d: wait for it, or run with FORCE=1" >&2
    exit 3
  fi
fi

{
  echo "sm8x prompt attention, phase 2d (prototype, GitHub #474): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
"$BIN" --prompts 8192,65536 --chunk 4096 --rounds 5 --variants --diag >> "$REPORT" 2> "$OUT/stderr-2d.txt"
STATUS=$?
END=$(date +%s)
{
  echo
  echo "run-2d: bench exit status $STATUS, $((END - START)) s"
} >> "$REPORT"
echo "run-2d: wrote $REPORT (bench exit status $STATUS, $((END - START)) s)"
grep -E "^check: |^check \[FAIL|^== prompt|^  (fp16|fp8|i8ref|i8) +[0-9]" "$REPORT" || true
exit $STATUS
