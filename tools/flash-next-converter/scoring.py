"""Readouts and scorers: top-64 with deterministic ties, exact and top-64 KLD, McNemar.

The top-64 scorer is the one spec 04 runs on the engine (layout.md §10): the
reference's 64 entries plus its tail mass, against the candidate's full distribution.
"""
import math

import torch

TOP = 64


def top_k(v, k):
    """(ids, values) of the k largest per row, descending; equal values in ascending id order.
    bf16 logits tie often, so the stored ids must not depend on the kernel's tie order."""
    n = v.shape[-1]
    t = torch.topk(v, k, dim=-1).values[..., -1:]
    key = (v > t).long() * 2 + (v == t).long()
    idx = torch.arange(n, device=v.device)
    score = key * (n + 1) - idx                     # strictly above the k-th first, then ties by lowest id
    ids = torch.topk(score, k, dim=-1).indices
    ids = torch.sort(ids, dim=-1).values            # ascending id, so the stable sort below keeps it on ties
    vals = v.gather(-1, ids)
    order = torch.sort(vals, dim=-1, descending=True, stable=True).indices
    return ids.gather(-1, order), vals.gather(-1, order)


def argmax(v):
    """First maximal index (torch.argmax's documented tie rule)."""
    return torch.argmax(v, dim=-1)


def kl_exact(lr, lc):
    """KL(ref || cand) per row from full log-softmaxes, nats."""
    return (lr.exp() * (lr - lc)).sum(-1)


def kl_top64(ref_ids, ref_lp, lc, eps=1e-12):
    """KL(ref || cand) from the reference's top-k log-probs and tail mass (layout.md §10)."""
    p = ref_lp.exp()
    c = lc.gather(-1, ref_ids)
    head = (p * (ref_lp - c)).sum(-1)
    R = (1 - p.sum(-1)).clamp(min=0)
    C = (1 - c.exp().sum(-1)).clamp(min=eps)
    tail = torch.where(R < eps, torch.zeros_like(R), R * (torch.log(R.clamp(min=eps)) - torch.log(C)))
    return head + tail


def mcnemar_p(b, c):
    """Exact two-sided binomial test on the discordant pairs (review/run4_paired.py)."""
    m, k = b + c, min(b, c)
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def paired(ref_ok, var_ok):
    """(lost, gained, p): questions the reference got right and the variant wrong, and back."""
    b = sum(1 for r, v in zip(ref_ok, var_ok) if r and not v)
    c = sum(1 for r, v in zip(ref_ok, var_ok) if v and not r)
    return b, c, mcnemar_p(b, c)


def db(rel):
    return 10 * math.log10(max(rel, 1e-12))
