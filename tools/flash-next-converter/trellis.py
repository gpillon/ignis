"""exllamav3's trellis quantizer and decoder, as run 8 drove them (real/e2e8.py).

Only this module imports exllamav3 (pinned 1.5.3, MIT): its batch quantizer encodes an
expert projection at one K with our Hessian, and `decode` reconstructs the weight a record
stands for (layout.md §3). Never set TORCH_CUDA_ARCH_LIST before importing it: the
extension rebuilds for minutes.
"""
import os

import numpy as np
import torch

import layout

EXLLAMAV3_VERSION = "1.5.3"


def _ext():
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3_lib import quantize as xq
    return ext, xq


def check_version():
    import importlib.metadata
    v = importlib.metadata.version("exllamav3")
    if v != EXLLAMAV3_VERSION:
        raise RuntimeError(f"exllamav3 {v} is installed; the converter is pinned to {EXLLAMAV3_VERSION}")
    return v


def shrink(H, target, s=0.05):
    """H + s * target * tr(H)/tr(target), per batch entry (run 8's metric)."""
    tr = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)
    trt = torch.diagonal(target, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-30)
    return H + s * target * (tr / trt)[:, None, None]


def k_value(k2):
    """The K exllamav3 takes: an int, or a float for the half step."""
    return k2 // 2 if k2 % 2 == 0 else k2 / 2


def hessian_data(Hs, keys, device):
    """exllamav3's H_data dicts, finalized by the first quantize call and reused for every K
    (run 8 did the same: the finalization's su draw is seeded, so it is the same for each K)."""
    return [{"H": Hs[b].clone(), "count": 1, "finalized": False, "device": torch.device(device), "L": None,
             "first_key": keys[b]} for b in range(Hs.shape[0])]


def quantize_batch(W, hds, k2, seeds, debug_dir, device):
    """W (B, out, in) float32, `hessian_data` dicts -> one record dict (NumPy
    trellis/suh/svh) per matrix, as exllamav3 returns them."""
    _, xq = _ext()
    K = k_value(k2)
    qas = [{"K": K, "devices": [torch.device(device).index or 0], "mul1": True, "apply_out_scales": True,
            "seed": s, "debug_dir": debug_dir} for s in seeds]
    res = xq.quantize_exl3_batch([W[b].T.contiguous() for b in range(W.shape[0])], hds, qas)
    out = []
    for _, t in res:
        out.append({"trellis": t["trellis"].cpu().numpy(), "suh": t["suh"].cpu().numpy(),
                    "svh": t["svh"].cpu().numpy()})
    return out


def decode(rec, k2, proj, device):
    """(out, in) float32 weight a record stands for (layout.md §3's oracle)."""
    ext, xq = _ext()
    i, o = layout.SHAPES[proj]
    w = torch.empty((i, o), dtype=torch.half, device=device)
    ext.reconstruct(w, torch.from_numpy(np.ascontiguousarray(rec["trellis"])).to(device), k_value(k2), False, True)
    w = xq.preapply_had_l(w.float(), 128)
    w *= torch.from_numpy(np.ascontiguousarray(rec["suh"])).to(device).float()[:, None]
    w = xq.preapply_had_r(w, 128)
    w *= torch.from_numpy(np.ascontiguousarray(rec["svh"])).to(device).float()[None, :]
    return w.T


def debug_dir(work):
    d = os.path.join(work, "state", "exl3_debug")
    os.makedirs(d, exist_ok=True)
    return d
