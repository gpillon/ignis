"""An adaptive extent for the 27B's MTP head against the best fixed draft window, offline (GitHub #629).

python tools/mtp-replay/extent.py MANIFEST W7_DIR TABLE [--subset W=DIR ...] [--variant NAME=TABLE | linear ...]
       [--out FIGURES.json]

MANIFEST is #267's (tools/dflash2-replay), W7_DIR #628's records of every turn at window 7 (records.py),
TABLE #628's cost table (round milliseconds by window, lanes and context: `mtp_ms` the head's round,
`verify_only_ms` the verify substrate's alone, `chain_step_ms` the slope of their difference). `--subset
W=DIR` names a direct replay at window W of some jobs, from which the window-7 derivation is corrected as
#628 corrected it; `--variant` prices the same policies on another table (one launch, a linear fit).

Every round of the window-7 walk keeps its anchor. A policy gives each lane-round its chain steps `s` and
its extent `e`; the round commits min(accepted, e) + 1, as #628 derived windows 1 to 6 (it reads a policy
that cuts accepted drafts low: the walk it would have made starts rounds inside those runs). How a round
pays (`PRICINGS`):

- shapes: a graph per window and lane count; the round runs its lanes' widest extent, every lane at it:
  V(e) + H(s), V the substrate's round, H the head's part (the MTP round less the substrate), C(w) when
  s = e = w. A confidence stop learns its extent inside the chain: the width after it is a decision in the
  middle of the round, whose cost is not in the table.
- chain: today's graph per lane count at the load's window k; the chain skips its steps past the lanes'
  widest extent, each lane verifies its own: C(k) - (k - s) * chain_step.
- ragged: each lane at its own width in one round: V at the lanes' mean extent (linear between windows) +
  H(widest s). No leaf has such a round; an estimate.

At one lane `shapes` and `ragged` are the same price. Policies (`policies`): the fixed window, the oracle
(the extent the round's accepted run needed: a ceiling), the head's confidence (a lane's chain stops at
the first draft whose probability, or the chain's product of them, falls under a threshold) and the lane's
acceptance history (per lane and draft position, a decayed count of how often a draft that the run reached
was accepted, under a prior; each round the extent with the most expected tokens over its price). Both
read the lane's own signals and no clock: the same records give the same extents, and the figures carry the
SHA-256 of every config's extents, lane-round by lane-round.

Committed tokens per second pool the traffic cells as #628 pooled tokens a round: per cell, the tokens
over the time of its lane-rounds (a lane-round takes its whole round's time), the cells by #267's
traffic weights; the unweighted figure is all tokens over all time. Lanes: the cell's lanes walk the
rounds from as many job starts, every round once on every lane.

Every config also names the load's draft window, 4 to 7. Per lanes option (`shapes`, `chain`, or `ragged`
at several lanes and `shapes` at one), the tool picks for each policy the one config with the most
geometric-mean gain over the table's cells, each cell's gain against its best fixed window as #628 rated
it (windows 4 to 6 corrected by the subsets; the policy's own figure derived, uncorrected, the corrected
one beside it), reads the config tuned on one half of the jobs on the other half, and gives the overhead
every round could carry before the gain falls to the spec's 5%.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import records  # noqa: E402

K = records.K
# #267's traffic weights: each source x kind cell's output tokens in the whole dataset
# (tools/dflash2-replay/analyze.py, draft_rule.py).
WEIGHTS = {"A-content": 28586, "A-think": 58665, "B-content": 1524816, "B-think": 8088074,
           "C-content": 2085240, "C-think": 6471440}
CELLS = sorted(WEIGHTS)
PRICINGS = ("shapes", "chain", "ragged")
GATE = 1.05
# The confidence threshold's grid: 0.00 to 0.99 by 0.01, then 0.991 to 0.999.
THRESHOLDS = [round(i / 100, 2) for i in range(100)] + [round(0.99 + i / 1000, 3) for i in range(1, 10)]
# The history policy's grid: decay per observation, prior acceptance, the prior's weight in observations.
DECAYS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995)
PRIORS = (0.5, 0.6, 0.7, 0.8, 0.9)
STRENGTHS = (1.0, 4.0)


class Rounds:
    """A walk's rounds in record order (job, then anchor): what a policy reads and what a round commits."""

    def __init__(self, accepted, committed, probability, tested, cell, turn_start, job_start, job):
        self.accepted = np.asarray(accepted, np.int64)
        self.committed = np.asarray(committed, np.int64)
        self.probability = np.asarray(probability, np.float64)
        self.tested = np.asarray(tested, bool)
        self.cell = np.asarray(cell, np.int64)
        self.turn_start = np.asarray(turn_start, bool)
        self.job_start = np.asarray(job_start, bool)
        self.job = np.asarray(job, np.int64)

    def __len__(self):
        return len(self.accepted)

    @classmethod
    def from_records(cls, rows, manifest):
        meta = {(j, t["turn"]): f"{t['source']}-{t['kind']}" for j, job in enumerate(manifest["jobs"])
                for t in job["turns"]}
        cell = [CELLS.index(meta[(int(r["job"]), int(r["turn"]))]) for r in rows]
        key = rows["job"].astype(np.int64) * (1 << 32) + rows["turn"].astype(np.int64)
        turn_start = np.concatenate([[True], key[1:] != key[:-1]])
        job_start = np.concatenate([[True], rows["job"][1:] != rows["job"][:-1]])
        return cls(rows["accepted"], rows["committed"], rows["probability"], records.tested_at(rows, manifest),
                   cell, turn_start, job_start, rows["job"])

    def take(self, mask):
        out = Rounds(self.accepted[mask], self.committed[mask], self.probability[mask], self.tested[mask],
                     self.cell[mask], self.turn_start[mask], self.job_start[mask], self.job[mask])
        out.turn_start[:1] = True
        out.job_start[:1] = True
        return out


class Costs:
    """One cell of the cost table: round milliseconds by window (0 the plain round) at its lanes and context."""

    def __init__(self, cell):
        self.lanes, self.context = int(cell["lanes"]), int(cell["context"])
        windows = [cell["by_window"][str(w)] for w in range(1, K + 1)]
        self.C = np.array([cell["plain_ms"]] + [w["mtp_ms"] for w in windows])
        self.V = np.array([cell["plain_ms"]] + [w["verify_only_ms"] for w in windows])
        self.H = self.C - self.V
        self.step = float(cell["chain_step_ms"])

    def linear(self):
        """The same cell with the head's round and the substrate's each a least-squares line over windows 1
        to 7 (the plain round kept): the table without its kinks."""
        out = Costs.__new__(Costs)
        out.lanes, out.context = self.lanes, self.context
        w = np.arange(1, K + 1)
        cm, cb = np.polyfit(w, self.C[1:], 1)
        vm, vb = np.polyfit(w, self.V[1:], 1)
        out.C = np.concatenate([[self.C[0]], cb + cm * w])
        out.V = np.concatenate([[self.V[0]], vb + vm * w])
        out.H = out.C - out.V
        out.step = float(cm - vm)
        return out

    def round_ms(self, pricing, steps, widths, window):
        """The round's milliseconds: `steps` its chain steps, `widths` its extents ([rounds, lanes])."""
        widest = widths.max(1)
        if pricing == "shapes":
            return self.V[widest] + self.H[steps]
        if pricing == "chain":
            return self.C[window] - (window - steps) * self.step
        if pricing == "ragged":
            mean = widths.mean(1)
            lo = np.floor(mean).astype(np.int64)
            hi = np.minimum(lo + 1, K)
            return self.V[lo] + (mean - lo) * (self.V[hi] - self.V[lo]) + self.H[steps]
        raise ValueError(pricing)

    def lane_price(self, pricing, window):
        """What a lane's choice of extent e = 1..window costs it, as the history policy weighs it."""
        e = np.arange(1, window + 1)
        return self.C[window] - (window - e) * self.step if pricing == "chain" else self.C[e]


_SCHEDULES = {}


def schedule(rounds, lanes):
    """[steps, lanes]: the round each lane runs at each step. Lane i starts at the first job at or after
    i/lanes of the walk and runs every round once, wrapping, so every round runs once on every lane."""
    key = (id(rounds), lanes)
    if key not in _SCHEDULES:
        _SCHEDULES[key] = (rounds, _schedule(rounds, lanes))
    return _SCHEDULES[key][1]


def _schedule(rounds, lanes):
    n = len(rounds)
    starts = np.flatnonzero(rounds.job_start)
    first = []
    for i in range(lanes):
        j = np.searchsorted(starts, i * n // lanes)
        first.append(int(starts[j]) if j < len(starts) else 0)
    return (np.array(first)[None, :] + np.arange(n)[:, None]) % n


def confidence(rounds, signal, threshold, drop, window=K):
    """The confidence stop: per round (chain steps, extent). The chain stops at the first draft whose
    signal (`draft`: its probability; `chain`: the product along the chain) is under the threshold; the
    round keeps that draft, or with `drop` verifies only the drafts before it."""
    p = rounds.probability[:, :window]
    q = p if signal == "draft" else np.cumprod(p, axis=1)
    low = q < threshold
    stopped = low.any(1)
    first = low.argmax(1) + 1
    steps = np.where(stopped, first, window)
    extents = np.where(stopped & drop, first - 1, steps)
    return steps, extents


def fixed(rounds, w):
    return np.full(len(rounds), w), np.full(len(rounds), w)


def oracle(rounds, window=K):
    a = np.minimum(rounds.accepted, window)
    return a, a


def lay_out(rounds, costs, pricing, window, steps, extents):
    """A per-round proposal laid on the lanes: (lane extents [steps, lanes], round ms [steps])."""
    at = schedule(rounds, costs.lanes)
    s, e = steps[at], extents[at]
    ms = costs.round_ms(pricing, s.max(1), e, window)
    if pricing == "shapes":
        e = np.repeat(e.max(1, keepdims=True), costs.lanes, axis=1)
    return at, e, ms


def history(rounds, costs, pricing, window, configs):
    """The acceptance history, for each config (decay, prior, strength) at once: [configs, steps, lanes]
    extents and [configs, steps] round ms. Per lane and draft position j, a decayed count of the rounds
    whose run reached j - 1 and tested j, and of those that accepted j; the estimate of accepting j is
    (accepted + strength * prior) / (reached + strength); the chance the run reaches e is their product.
    A lane forgets at the start of a turn. Under `shapes` the round takes the one width with the most
    expected tokens of all its lanes over its price; under `chain` and `ragged` each lane takes its own."""
    at = schedule(rounds, costs.lanes)
    steps, lanes = at.shape
    g = len(configs)
    decay = np.array([c[0] for c in configs])[:, None, None]
    prior = np.array([c[1] for c in configs])[:, None, None]
    strength = np.array([c[2] for c in configs])[:, None, None]
    price = costs.lane_price(pricing, window)
    reached = np.zeros((g, lanes, K))
    took = np.zeros((g, lanes, K))
    out_e = np.empty((g, steps, lanes), np.int64)
    out_ms = np.empty((g, steps))
    j = np.arange(1, K + 1)
    for t in range(steps):
        r = at[t]
        fresh = rounds.turn_start[r]
        if fresh.any():
            reached[:, fresh] = 0.0
            took[:, fresh] = 0.0
        est = (took + strength * prior) / (reached + strength)
        expected = 1.0 + np.cumsum(np.cumprod(est[:, :, :window], axis=2), axis=2)  # [g, lanes, window]
        if pricing == "shapes":
            width = np.argmax(expected.sum(1) / price, axis=1) + 1  # [g]
            e = np.repeat(width[:, None], lanes, axis=1)
        else:
            e = np.argmax(expected / price, axis=2) + 1  # [g, lanes]
        out_e[:, t] = e
        out_ms[:, t] = costs.round_ms(pricing, e.max(1), e, window)
        a = rounds.accepted[r][None, :, None]
        seen = (j[None, None, :] <= e[:, :, None]) & rounds.tested[r][None, :, :] & (a >= j - 1)
        reached = np.where(seen, decay * reached + 1.0, reached)
        took = np.where(seen, decay * took + (a >= j), took)
    return at, out_e, out_ms


def history_configs():
    return [(d, p, s) for d in DECAYS for p in PRIORS for s in STRENGTHS]


class Outcome:
    """A policy's lane-rounds summed per traffic cell and per half of the jobs (even, odd)."""

    def __init__(self, rounds, at, extents, ms):
        a = rounds.accepted[at]
        tokens = np.minimum(np.minimum(a, extents) + 1, rounds.committed[at])
        group = rounds.cell[at] * 2 + (rounds.job[at] % 2)
        time = np.broadcast_to(ms[:, None], at.shape)
        n = len(CELLS) * 2
        self.tokens = np.bincount(group.ravel(), tokens.ravel(), n)
        self.time = np.bincount(group.ravel(), time.ravel(), n)
        self.lane_rounds = np.bincount(group.ravel(), minlength=n).astype(np.float64)
        self.cut = np.bincount(group.ravel(), (a > extents).ravel(), n)
        self.extent = np.bincount(group.ravel(), extents.ravel(), n)
        self.sha256 = hashlib.sha256(np.ascontiguousarray(extents, np.uint8).tobytes()).hexdigest()

    def _sel(self, half):
        idx = np.arange(len(CELLS) * 2)
        return idx if half is None else idx[idx % 2 == half]

    def rate(self, half=None, weighted=True, overhead_ms=0.0):
        """Committed tokens per second a lane; `overhead_ms` adds that much to every round."""
        sel = self._sel(half)
        tok, time, n = self.tokens[sel], self.time[sel] + overhead_ms * self.lane_rounds[sel], self.lane_rounds[sel]
        if not weighted:
            return 1000.0 * tok.sum() / time.sum()
        num = den = 0.0
        for c, name in enumerate(CELLS):
            t, s = tok[sel // 2 == c].sum(), time[sel // 2 == c].sum()
            if t > 0:
                num += WEIGHTS[name]
                den += WEIGHTS[name] * s / t
        return 1000.0 * num / den

    def cut_fraction(self, half=None):
        sel = self._sel(half)
        return float(self.cut[sel].sum() / self.lane_rounds[sel].sum())

    def mean_extent(self):
        return float(self.extent.sum() / self.lane_rounds.sum())

    def break_even_ms(self, target):
        """The overhead a round could carry before its traffic-weighted rate falls to `target`."""
        lo, hi = 0.0, 100.0
        if self.rate() <= target:
            return 0.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.rate(overhead_ms=mid) > target else (lo, mid)
        return lo


def pooled_tokens(tokens, cells):
    """Tokens a round, traffic-weighted as #628 pooled them (analyze_replay's `pooled`)."""
    num = den = 0.0
    for c, name in enumerate(CELLS):
        s = cells == c
        if s.any():
            num += WEIGHTS[name]
            den += WEIGHTS[name] / tokens[s].mean()
    return num / den


def correction(w7, subsets):
    """#628's correction of the window-7 derivation: per subset window w, the direct replay's tokens a
    round over the derivation from the window-7 rounds of the same jobs (traffic-weighted, unweighted),
    and kappa, the least-squares slope through 0 of (ratio - 1) against the fraction of rounds window w
    cuts (accepted > w): a policy's derived rate times 1 + kappa * its cut fraction is the model's."""
    out = {"windows": {}}
    for w, direct in sorted(subsets.items()):
        same = w7.take(np.isin(w7.job, np.unique(direct.job)))
        derived = np.minimum(same.accepted, w) + 1
        out["windows"][w] = {
            "ratio_weighted": pooled_tokens(direct.committed, direct.cell) / pooled_tokens(derived, same.cell),
            "ratio_unweighted": float(direct.committed.mean() / derived.mean()),
            "cut_fraction": float((same.accepted > w).mean()),
            "rounds_direct": len(direct)}
    for kind in ("weighted", "unweighted"):
        f = np.array([v["cut_fraction"] for v in out["windows"].values()])
        r = np.array([v["ratio_" + kind] for v in out["windows"].values()])
        out["kappa_" + kind] = float((f * (r - 1)).sum() / (f * f).sum()) if len(f) else 0.0
    return out


def fixed_rates(w7, costs, corr):
    """The fixed windows 0 to 7 as #628 rated them: derived, windows with a direct subset corrected."""
    out = {}
    for w in range(K + 1):
        o = Outcome(w7, *lay_out(w7, costs, "shapes", w, *fixed(w7, w)))
        ratio = corr["windows"].get(w, {})
        out[w] = {"outcome": o,
                  "weighted": o.rate() * ratio.get("ratio_weighted", 1.0),
                  "unweighted": o.rate(weighted=False) * ratio.get("ratio_unweighted", 1.0),
                  "halves": [o.rate(h) * ratio.get("ratio_weighted", 1.0) for h in (0, 1)]}
    return out


def summarize(o, corr, best):
    return {"weighted": o.rate(), "unweighted": o.rate(weighted=False),
            "weighted_corrected": o.rate() * (1 + corr["kappa_weighted"] * o.cut_fraction()),
            "unweighted_corrected": o.rate(weighted=False) * (1 + corr["kappa_unweighted"] * o.cut_fraction()),
            "halves": [o.rate(h) for h in (0, 1)],
            "gain": o.rate() / best["weighted"],
            "gain_unweighted": o.rate(weighted=False) / best["unweighted"],
            "mean_extent": o.mean_extent(), "cut_fraction": o.cut_fraction(),
            "break_even_ms": o.break_even_ms(GATE * best["weighted"]), "extents_sha256": o.sha256,
            "sums": {"tokens": o.tokens.tolist(), "time": o.time.tolist(), "lane_rounds": o.lane_rounds.tolist()}}


def rate_from_sums(sums, overhead_ms):
    """A summary's traffic-weighted rate with `overhead_ms` added to every round."""
    o = Outcome.__new__(Outcome)
    o.tokens, o.time, o.lane_rounds = (np.array(sums[k]) for k in ("tokens", "time", "lane_rounds"))
    return o.rate(overhead_ms=overhead_ms)


def leg_break_even(per_cell, cells):
    """The overhead every round of every cell could carry before the leg's geometric-mean gain falls to
    the gate."""
    def gain(x):
        return geo([rate_from_sums(p["sums"], x) / c["best_fixed"]["weighted"] for p, c in zip(per_cell, cells)])
    if gain(0.0) <= GATE:
        return 0.0
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gain(mid) > GATE else (lo, mid)
    return lo


# The load's draft window a policy adapts under, part of its config: under `chain` the round verifies the
# window's columns, under `shapes` and `ragged` the window caps the extent.
WINDOWS = (4, 5, 6, K)


def policies(w7, costs, pricing, corr, best):
    """Every policy config on one cell under one pricing: {family: [(config, summary)]}."""
    out = {"oracle": [], "confidence": [], "history": []}
    for window in WINDOWS:
        o = Outcome(w7, *lay_out(w7, costs, pricing, window, *oracle(w7, window)))
        out["oracle"].append(({"window": window}, summarize(o, corr, best)))
        for signal in ("draft", "chain"):
            for drop in ((False, True) if pricing != "chain" else (False,)):
                for tau in THRESHOLDS:
                    s, e = confidence(w7, signal, tau, drop, window)
                    o = Outcome(w7, *lay_out(w7, costs, pricing, window, s, e))
                    out["confidence"].append(({"window": window, "signal": signal, "drop": drop, "threshold": tau},
                                              summarize(o, corr, best)))
        configs = history_configs()
        at, e, ms = history(w7, costs, pricing, window, configs)
        out["history"] += [({"window": window, "decay": c[0], "prior": c[1], "strength": c[2]},
                            summarize(Outcome(w7, at, e[i], ms[i]), corr, best)) for i, c in enumerate(configs)]
    return out


def walk_check(w7, subsets, costs, family, config):
    """The chosen config at one lane, capped at each subset's window w, on the direct walk at w and on the
    window-7 walk of the same jobs: how much the walk's anchors move a policy's derived rate, beside the
    fixed window's own direct over derived."""
    out = {}
    for w, direct in sorted(subsets.items()):
        same = w7.take(np.isin(w7.job, np.unique(direct.job)))
        rates = []
        for walk in (direct, same):
            if family == "confidence":
                s, e = confidence(walk, config["signal"], config["threshold"], config["drop"], w)
                o = Outcome(walk, *lay_out(walk, costs, "shapes", w, s, e))
            else:
                at, e, ms = history(walk, costs, "shapes", w, [(config["decay"], config["prior"], config["strength"])])
                o = Outcome(walk, at, e[0], ms[0])
            rates.append(o.rate())
        fixed_w = [Outcome(walk, *lay_out(walk, costs, "shapes", w, *fixed(walk, w))).rate() for walk in (direct, same)]
        out[w] = {"direct_walk": rates[0], "window7_walk": rates[1], "ratio": rates[0] / rates[1],
                  "fixed_ratio": fixed_w[0] / fixed_w[1]}
    return out


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def choose(cells, family, pricing_of_cell):
    """One config for the whole leg: the most geometric-mean gain over its cells (traffic-weighted, the
    derived policy against the corrected fixed window), and the same chosen on one half of the jobs and
    read on the other. `pricing_of_cell` names each cell's pricing for this lanes option."""
    runs = [c["pricings"][pricing_of_cell(c)][family] for c in cells]
    n = len(runs[0])
    gains = [geo([run[i][1]["gain"] for run in runs]) for i in range(n)]
    i = int(np.argmax(gains))
    held = []
    for tune, read in ((0, 1), (1, 0)):
        g = [geo([run[k][1]["halves"][tune] / c["best_fixed_halves"][tune] for run, c in zip(runs, cells)])
             for k in range(n)]
        k = int(np.argmax(g))
        held.append({"tuned_on": tune, "config": runs[0][k][0],
                     "gain_read_on_the_other": geo([run[k][1]["halves"][read] / c["best_fixed_halves"][read]
                                                    for run, c in zip(runs, cells)])})
    configs = [run[i][0] for run in runs]
    return {"config": configs[0], "index": i,
            "gain": gains[i],
            "gain_unweighted": geo([run[i][1]["gain_unweighted"] for run in runs]),
            "gain_corrected": geo([run[i][1]["weighted_corrected"] / c["best_fixed"]["weighted"]
                                   for run, c in zip(runs, cells)]),
            "per_cell": [run[i][1] for run in runs],
            "break_even_ms": leg_break_even([run[i][1] for run in runs], cells),
            "per_cell_ceiling": [max(r[1]["gain"] for r in run) for run in runs],
            "held_out": held}


def analyze(manifest, w7_dir, table, subset_dirs, variants=()):
    rows = records.load(w7_dir, manifest)
    w7 = Rounds.from_records(rows, manifest)
    subsets = {w: Rounds.from_records(records.load(d, manifest), manifest) for w, d in subset_dirs.items()}
    corr = correction(w7, subsets)
    tables = [("table", table)] + list(variants)
    result = {"rounds": len(w7), "correction": corr, "tables": {}}
    for name, tab in tables:
        cells = []
        for cell in tab["cells"]:
            costs = Costs(cell) if name != "linear" else Costs(cell).linear()
            fx = fixed_rates(w7, costs, corr)
            best_w = max(fx, key=lambda w: fx[w]["weighted"])
            best = {"window": best_w, "weighted": fx[best_w]["weighted"],
                    "unweighted": max(v["unweighted"] for v in fx.values())}
            c = {"context": costs.context, "lanes": costs.lanes,
                 "fixed": {w: {k: v for k, v in f.items() if k != "outcome"} for w, f in fx.items()},
                 "best_fixed": best,
                 "best_fixed_halves": [max(f["halves"][h] for f in fx.values()) for h in (0, 1)],
                 "pricings": {}}
            for pricing in PRICINGS:
                if costs.lanes == 1 and pricing == "ragged":
                    continue
                c["pricings"][pricing] = policies(w7, costs, pricing, corr, best)
            cells.append(c)
        options = {
            "shapes": lambda c: "shapes",
            "chain": lambda c: "chain",
            "ragged": lambda c: "ragged" if c["lanes"] > 1 else "shapes"}
        verdicts = {}
        for option, of_cell in options.items():
            verdicts[option] = {f: choose(cells, f, of_cell) for f in ("confidence", "history", "oracle")}
        result["tables"][name] = {"cells": cells, "options": verdicts}
        if name == "table" and subsets:
            one = Costs(next(c for c in tab["cells"] if c["lanes"] == 1))
            result["walk_check"] = {f: walk_check(w7, subsets, one, f, verdicts["shapes"][f]["config"])
                                    for f in ("confidence", "history")}
            result["walk_check_cell"] = {"lanes": one.lanes, "context": one.context}
    return result


def print_report(result, out=sys.stdout):
    corr = result["correction"]
    print(f"rounds {result['rounds']}; correction kappa {corr['kappa_weighted']:.4f} (weighted), "
          f"{corr['kappa_unweighted']:.4f} (unweighted)", file=out)
    for w, v in corr["windows"].items():
        print(f"  subset w{w}: direct/derived {v['ratio_weighted']:.4f} (unweighted {v['ratio_unweighted']:.4f}), "
              f"cut fraction {v['cut_fraction']:.3f}, model {1 + corr['kappa_weighted'] * v['cut_fraction']:.4f}",
              file=out)
    for name, tab in result["tables"].items():
        print(f"== {name}", file=out)
        for c in tab["cells"]:
            b = c["best_fixed"]
            print(f"  {c['lanes']} lanes {c['context']:>6}: fixed " + " ".join(
                f"w{w}={f['weighted']:.1f}" for w, f in c["fixed"].items()) +
                f"  best w{b['window']} {b['weighted']:.1f} (unweighted best {b['unweighted']:.1f})", file=out)
        for option, fam in tab["options"].items():
            for f, v in fam.items():
                cells = " ".join(f"{p['gain']:.3f}" for p in v["per_cell"])
                held = " ".join(f"{h['gain_read_on_the_other']:.3f}" for h in v["held_out"])
                print(f"  {option:7} {f:10} {json.dumps(v['config'])}: gain {v['gain']:.4f}, leg break-even "
                      f"{v['break_even_ms']:.3f} ms (unweighted "
                      f"{v['gain_unweighted']:.4f}, corrected {v['gain_corrected']:.4f}; cells {cells}; held out "
                      f"{held}; cell ceilings " + " ".join(f"{x:.3f}" for x in v["per_cell_ceiling"]) + ")", file=out)
                print("  " + " " * 18 + "per cell: rate " + " ".join(f"{p['weighted']:.1f}" for p in v["per_cell"]) +
                      "; mean extent " + " ".join(f"{p['mean_extent']:.2f}" for p in v["per_cell"]) +
                      "; cut " + " ".join(f"{p['cut_fraction']:.3f}" for p in v["per_cell"]) +
                      "; break-even ms " + " ".join(f"{p['break_even_ms']:.3f}" for p in v["per_cell"]), file=out)
    for f, rows in result.get("walk_check", {}).items():
        print(f"walk check ({f}, {result['walk_check_cell']}): " + "; ".join(
            f"w{w} direct walk {v['direct_walk']:.1f} / window-7 walk {v['window7_walk']:.1f} = {v['ratio']:.4f} "
            f"(fixed w{w} {v['fixed_ratio']:.4f})" for w, v in rows.items()), file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("w7_dir")
    ap.add_argument("table")
    ap.add_argument("--subset", action="append", default=[], help="W=DIR: a direct replay at window W")
    ap.add_argument("--variant", action="append", default=[], help="NAME=TABLE, or `linear` for the fitted table")
    ap.add_argument("--out", help="the figures, JSON")
    args = ap.parse_args(argv)
    manifest = records.read_manifest(args.manifest)
    table = json.load(open(args.table, encoding="utf-8"))
    subsets = {int(s.split("=", 1)[0]): s.split("=", 1)[1] for s in args.subset}
    variants = []
    for v in args.variant:
        if v == "linear":
            variants.append(("linear", table))
        else:
            name, path = v.split("=", 1)
            variants.append((name, json.load(open(path, encoding="utf-8"))))
    result = analyze(manifest, args.w7_dir, table, subsets, variants)
    print_report(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
