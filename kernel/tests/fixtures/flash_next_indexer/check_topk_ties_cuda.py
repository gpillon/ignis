"""Check torch CUDA `topk`'s tie order against the indexer fixture's documented rule (GPU, ~30 s).

The converter's quantized and BF16 references ran the checkpoint's indexer on CUDA, so the tie
rule ignis must match is torch CUDA `topk`'s: the k largest, and among scores equal to the k-th
the lowest indices ("choose the first seen set", aten/src/ATen/native/cuda/TensorTopK.cu). The
fixture (record_indexer.py) states that rule explicitly; this script confirms torch CUDA follows
it on the fixture's own score rows (exact-zero ties by the hundred) and on synthetic tie vectors
at the sizes the indexer meets (up to 65536 blocks). Writes nothing.

It runs on the GPU, so it refuses to start unless the caller holds the swarm's GPU lock under the
name it passes (AGENTS.md, Testing): take the lock first, then

    F:/ai/ngram-venv/Scripts/python.exe kernel/tests/fixtures/flash_next_indexer/check_topk_ties_cuda.py --lock-owner <name>
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
K = 512
GPU_LOCK_OWNER = "F:/ai/opencode/.inference-qwen-worktrees/.swarm/gpu.lock/owner"


def require_gpu_lock(argv):
    """Exit (status 1) unless the swarm's GPU lock is held by the name given with --lock-owner."""
    name = argv[argv.index("--lock-owner") + 1] if "--lock-owner" in argv[:-1] else None
    try:
        with open(GPU_LOCK_OWNER, encoding="utf-8") as f:
            owner = dict(line.split("=", 1) for line in f.read().splitlines() if "=" in line).get("owner")
    except OSError:
        owner = None
    if name is None or owner != name:
        sys.exit(f"refused: the GPU lock is held by {owner!r}, not by --lock-owner {name!r}; "
                 "take it first (.swarm/gpu-lock.sh try <name> ...)")


def read_fixture(path):
    out = {}
    with open(path, "rb") as f:
        assert f.read(8) == b"IGNFX001"
        while True:
            head = f.read(4)
            if not head:
                break
            name = f.read(struct.unpack("<I", head)[0]).decode()
            code, ndim = struct.unpack("<II", f.read(8))
            dims = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
            nbytes = struct.unpack("<Q", f.read(8))[0]
            out[name] = (code, dims, f.read(nbytes))
    return out


def rule(s: torch.Tensor, k: int) -> torch.Tensor:
    order = torch.sort(s.cpu(), descending=True, stable=True).indices
    return torch.sort(order[:k]).values


def main():
    require_gpu_lock(sys.argv)
    dev = torch.device("cuda:0")
    fx = read_fixture(os.path.join(HERE, "indexer_ref.bin"))
    failures = 0
    for name, (code, dims, payload) in sorted(fx.items()):
        if not name.endswith(".scores"):
            continue
        s = torch.frombuffer(bytearray(payload), dtype=torch.float32)
        k = min(K, s.numel())
        got = torch.sort(s.to(dev).topk(k).indices.cpu()).values
        want = torch.from_numpy(np.frombuffer(fx[name.replace(".scores", ".rule_select")][2], dtype="<i4").copy()).long()
        ok = torch.equal(got, want)
        failures += not ok
        print(f"{name}: n={s.numel()} equal_to_rule={ok}")
    g = torch.Generator().manual_seed(1)
    for n in (513, 600, 2048, 8192, 32768, 65536):
        for frac in (1.0, 0.95, 0.7):
            s = torch.rand(n, generator=g)
            s[torch.rand(n, generator=g) < frac] = 0.0
            got = torch.sort(s.to(dev).topk(K).indices.cpu()).values
            ok = torch.equal(got, rule(s, K))
            failures += not ok
            print(f"synthetic n={n} zero_fraction={frac}: equal_to_rule={ok}")
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
    print("PASS" if failures == 0 else f"FAIL: {failures} cases differ from the rule")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
