"""Before the first byte: a free GPU (held by our lock), a VRAM cap, enough disk.

The card fits one run at a time and the loser dies with no diagnostic, so the converter
refuses to start unless the shared GPU lock is held by the owner it was launched as and
no other process has a compute context on the card.
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


def foreign_compute_processes(own_pid):
    """[(pid, name, MiB)] of compute contexts on the GPU other than this process."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise Refused(f"nvidia-smi failed: {out.stderr.strip()}")
    procs = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) != own_pid:
            procs.append((int(parts[0]), parts[1], parts[2] if len(parts) > 2 else "?"))
    return procs


def check_gpu(expected_owner, lock_dir=LOCK_DIR):
    owner = lock_owner(lock_dir)
    if owner != expected_owner:
        raise Refused(f"the GPU lock is held by {owner!r}, not {expected_owner!r}: take it first "
                      f"(bash .swarm/gpu-lock.sh try {expected_owner} ...)")
    procs = foreign_compute_processes(os.getpid())
    if procs:
        raise Refused(f"another process is on the GPU: {procs}")


def cap_vram(gb):
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, gb * GB / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    return frac, total


def free_bytes(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free


def check_disk(path, need_bytes, margin_bytes, what):
    free = free_bytes(path)
    if free < need_bytes + margin_bytes:
        raise Refused(f"{what}: {free / GB:.1f} GB free on {path}, need {need_bytes / GB:.1f} GB "
                      f"+ {margin_bytes / GB:.0f} GB margin")
    return free
