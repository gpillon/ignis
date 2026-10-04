"""The n-gram table in INT4 groups of 32: 90-byte rows, the shard files, the hot-row list
(layout.md §7).

A row is 160 values: bytes [0, 80) hold the 4-bit codes (byte j: value 2j in the low
nibble, 2j+1 in the high one), bytes [80, 90) the five fp16 group scales. A value decodes
to (nibble - 8) * scale. The scale is amax/7 rounded to the nearest fp16 and the codes are
computed against that fp16 scale, so quantizing a gathered row reproduces the stored row.
"""
import numpy as np
import torch

DIM, GROUP, ROW_BYTES = 160, 32, 90
GROUPS = DIM // GROUP


def _scales_and_codes(x):
    """x (n, 160) any float -> fp16 scales (n, 5) as float32 values, int codes (n, 5, 32)."""
    xs = x.float().reshape(-1, GROUPS, GROUP)
    s = (xs.abs().amax(-1) / 7.0).to(torch.float16).float()
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.round(xs / safe[..., None]).clamp_(-7, 7)
    q = torch.where(s[..., None] > 0, q, torch.zeros_like(q))
    return s, q


def encode_rows(x):
    """(n, 160) rows -> (n, 90) uint8."""
    n = x.shape[0]
    s, q = _scales_and_codes(x)
    nib = (q.reshape(n, DIM) + 8).to(torch.uint8)
    out = np.empty((n, ROW_BYTES), dtype=np.uint8)
    out[:, :80] = (nib[:, 0::2] | (nib[:, 1::2] << 4)).numpy()
    out[:, 80:] = s.to(torch.float16).numpy().view(np.uint8).reshape(n, 10)
    return out


def decode_rows(raw):
    """(n, 90) uint8 -> (n, 160) float32 values."""
    raw = np.ascontiguousarray(raw)
    n = raw.shape[0]
    b = torch.from_numpy(raw[:, :80].astype(np.int16))
    q = torch.empty(n, DIM)
    q[:, 0::2] = (b & 0xF) - 8
    q[:, 1::2] = (b >> 4) - 8
    s = torch.from_numpy(raw[:, 80:].copy().view(np.float16).astype(np.float32))
    return (q.view(n, GROUPS, GROUP) * s[..., None]).view(n, DIM)


def fake_quant(emb):
    """Gathered embeddings (..., 16 * 160) as the INT4 table stores them, in bf16."""
    shp = emb.shape
    s, q = _scales_and_codes(emb.reshape(-1, DIM))
    return (q * s[..., None]).reshape(shp).to(torch.bfloat16)


def hot_rows(ids, cap_rows):
    """Row ids ranked by lookup count (descending), ties by row id (ascending), capped."""
    u, c = np.unique(np.asarray(ids).ravel(), return_counts=True)
    order = np.lexsort((u, -c))
    return u[order][:cap_rows].astype(np.uint32)
