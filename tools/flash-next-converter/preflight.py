"""Before the first byte: a free GPU (held by our lock), a VRAM cap, enough disk.

The card fits one run at a time and the loser dies with no diagnostic, so the converter
refuses to start unless the shared GPU lock is held by the owner it was launched as, no
known GPU workload is running and little VRAM is in use (the Makefile's gpu-guard rules).
"""
import os
import shutil
import subprocess

LOCK_DIR = "F:/ai/opencode/.inference-qwen-worktrees/.swarm/gpu.lock"
GB = 1e9


class Refused(RuntimeError):
    pass


def lock_owner(lock_dir=LOCK_DIR):
    try:
        for line in open(os.path.join(lock_dir, "owner")):
            if line.startswith("owner="):
                return line.strip().split("=", 1)[1]
    except OSError:
        return None
    return None


HOLDER_PATTERNS = ("ninfer*", "ignis-server*", "ignis-bench*", "*_gpu-*")
VRAM_THRESHOLD_MIB = 8192   # mk/windows/gpu.ps1's guard: the desktop alone holds ~3 GB


def gpu_holders(names):
    """Process names that hold the card by convention (the Makefile's gpu-guard list).
    Under WDDM nvidia-smi lists every desktop app as a compute app, so names it is."""
    import fnmatch
    return sorted({n for n in names if any(fnmatch.fnmatch(n.lower(), p) for p in HOLDER_PATTERNS)})


def vram_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise Refused(f"nvidia-smi failed: {out.stderr.strip()}")
    return int(out.stdout.split()[0])


def process_names():
    import psutil
    return [p.info["name"] or "" for p in psutil.process_iter(["name"])]


def check_gpu(expected_owner, lock_dir=LOCK_DIR, names=process_names, vram_used=vram_used_mib):
    owner = lock_owner(lock_dir)
    if owner != expected_owner:
        raise Refused(f"the GPU lock is held by {owner!r}, not {expected_owner!r}: take it first "
                      f"(bash .swarm/gpu-lock.sh try {expected_owner} ...)")
    held = gpu_holders(names())
    if held:
        raise Refused(f"GPU workloads are running: {held}")
    used = vram_used()
    if used >= VRAM_THRESHOLD_MIB:
        raise Refused(f"{used} MiB of VRAM already in use (guard threshold {VRAM_THRESHOLD_MIB} MiB)")
    return used


def cap_vram(gb):
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, gb * GB / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    return frac, total


def free_bytes(path):
    """Free bytes on the drive of `path`, which need not exist yet (its drive must)."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            raise Refused(f"{path}: no such drive")
        p = parent
    return shutil.disk_usage(p).free


def check_disk(path, need_bytes, margin_bytes, what):
    free = free_bytes(path)
    if free < need_bytes + margin_bytes:
        raise Refused(f"{what}: {free / GB:.1f} GB free on {path}, need {need_bytes / GB:.1f} GB "
                      f"+ {margin_bytes / GB:.0f} GB margin")
    return free
