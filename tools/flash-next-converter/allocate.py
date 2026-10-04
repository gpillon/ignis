"""Lagrangian K allocation over the sweep's distortion curves (run 8's rule).

Each expert projection was encoded at every K of the set; `Dv[e, j]` is its relative
proxy distortion at the j-th K and `en[e]` the energy that turns it into an absolute
cost. A multiplier mu prices the rate; the bisection finds the smallest price whose
picks keep the mean rate, scales included, at or under the budget. Gate/up and down
are allocated separately by the caller.
"""
import math

import torch

import layout


def allocate(en, Dv, proj, budget, iters=200):
    """Returns one k2 per expert (list of ints) with mean `layout.rate` <= budget."""
    R = torch.tensor([layout.rate(proj, k2) for k2 in layout.K2_SET], dtype=torch.float64,
                     device=Dv.device).expand(Dv.shape[0], -1)
    cost = en.double()[:, None] * Dv.double()
    lo, hi = 1e-20, 1e20
    for _ in range(iters):
        mu = math.sqrt(lo * hi)
        pick = torch.argmin(cost + mu * R, 1)
        lo, hi = (mu, hi) if R.gather(1, pick[:, None]).mean() > budget else (lo, mu)
    pick = torch.argmin(cost + hi * R, 1)
    out = [layout.K2_SET[j] for j in pick.tolist()]
    mean = sum(layout.rate(proj, k) for k in out) / len(out)
    if mean > budget + 1e-12:
        raise RuntimeError(f"{proj}: allocation mean {mean} over budget {budget}")
    return out
