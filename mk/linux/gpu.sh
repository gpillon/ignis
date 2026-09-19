#!/usr/bin/env bash
# GPU status and guard for the Makefile (the Linux GPU_* hooks,
# mk/os/linux.mk). The counterpart of mk/windows/gpu.ps1: the 5090 fits one
# engine or GPU test at a time, and the loser of a race dies with no
# diagnostic (docs/agents/testing.md), so:
#
#   status   VRAM in use, nvidia-smi's compute apps, and every process that
#            may hold the card (ninfer*, ignis-server, ignis-bench, *_gpu-*)
#   guard    exit 1 while the card is held: an ignis GPU process anywhere on
#            the machine (any worktree counts), a standing profile marker, or
#            more than IGNIS_GPU_THRESHOLD_MIB already resident.
#
# Unlike the Windows side there is no scripts/gpu-preflight.ps1 to delegate
# the VRAM half to -- under Linux nvidia-smi reports real per-process VRAM, so
# the threshold check is a query here rather than a separate script.

set -uo pipefail

action="${1:-status}"
threshold="${2:-8192}"

# `pgrep -f` on the process name only: a full command line would match this
# script itself, and `make` invoking it.
holders() {
    pgrep -a -f '(^|/)(ninfer[^/ ]*|ignis-server|ignis-bench|[^ ]*_gpu-[^ ]*)( |$)' 2>/dev/null |
        grep -v "$$" || true
}

ignis_holders() {
    holders | grep -Ev '(^|/)ninfer' || true
}

used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
        head -1 | tr -d ' '
}

case "$action" in
status)
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
        echo "compute apps:"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    else
        echo "nvidia-smi not found on PATH"
    fi
    found="$(holders)"
    if [ -n "$found" ]; then
        echo "processes that may hold the card:"
        echo "$found" | sed 's/^/  pid /'
    else
        echo "no ninfer / ignis-server / ignis-bench / *_gpu-* process running"
    fi
    exit 0
    ;;

guard)
    found="$(ignis_holders)"
    if [ -n "$found" ]; then
        echo "GPU guard: refused -- ignis GPU work is already running (another worktree or session counts):"
        echo "$found" | sed 's/^/  pid /'
        echo "  make stop (FORCE=1 for one make did not start), or wait; GPU_CHECK=0 skips this guard"
        exit 1
    fi

    # A standing pass means scripts/gpu-profile is (or was until a kill)
    # mid-run: its tests read this marker, so it is never touched here.
    # `ignis_core::gpu_profile` resolves it through std::env::temp_dir().
    marker="${TMPDIR:-/tmp}/ignis-gpu-preflight.ok"
    if [ -f "$marker" ]; then
        age_min=$(( ( $(date +%s) - $(stat -c %Y "$marker") ) / 60 ))
        if [ "$age_min" -lt 30 ]; then
            echo "GPU guard: refused -- a GPU profile pass is on record ($marker): a profile run looks in progress"
            echo "  wait for it; a pass older than 30 minutes is ignored. GPU_CHECK=0 skips this guard"
            exit 1
        fi
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU guard: refused -- nvidia-smi not found, so the card's state is unknown"
        echo "  GPU_CHECK=0 skips this guard (only if you know the card is free)"
        exit 1
    fi
    used="$(used_mib)"
    if [ -z "$used" ]; then
        echo "GPU guard: refused -- nvidia-smi reported no memory figure"
        exit 1
    fi
    if [ "$used" -ge "$threshold" ]; then
        echo "GPU guard: refused -- $used MiB already resident (threshold $threshold MiB):"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
        echo "  stop whatever holds the card, or raise GPU_THRESHOLD_MIB; GPU_CHECK=0 skips this guard"
        exit 1
    fi
    echo "GPU guard: card is free ($used MiB resident)"
    exit 0
    ;;

*)
    echo "usage: gpu.sh status|guard [threshold-mib]" >&2
    exit 2
    ;;
esac
