"""Tokens per round under a draft rule, from the replay's records (GitHub #577, spec #574).

python draft_rule.py MANIFEST OUT_DIR [OUT_DIR ...] [--reference OUT_DIR_267]

Each OUT_DIR is one replay of the recorded turns (crates/runtime/examples/dflash2_replay.rs)
under one draft rule, named by its `rule.json` (none: today's fixed draft, as #267's run).
For each, in the form the study of #574 printed (`report-1-coupled-draft.md`):

- tokens per round, traffic-weighted over the source x kind cells as analyze.py pools them:
  `realized`, what the forced rounds committed, and `expected`, the Rao-Blackwellized
  expectation at the same rounds' anchors,

      E[committed] = 1 + sum_i prod_{j<i} a_j * m_i,

  with m_i the probability a live round under the rule takes column i's draft and a_j the
  probability it took the recorded token at column j given that the target sampled it.
  Under the fixed draft d: m_i = p_i(d_i), a_j = 1[r_j == d_j]. Under the sampled draft:
  m_i = sum_x min(p_i(x), q_i(x)), a_j = min(p_j, q_j)(r_j) / p_j(r_j) (q_j(r_j) where the
  target gives r_j nothing), q_i the drafter's distribution after the recorded token
  before it -- the forced round's coin;
- the survival curve per draft column: P(accepted >= i), realized and expected, and per
  column the probability a reached column is taken (the study's match probability);
- the split per kind of output: that probability per kind of the column's recorded token.

Given a today's-rule directory, it also prices the sampled draft offline at that
directory's anchors (the study's estimate, 3.738 against 3.461 on #267's rounds), so the
card's figure reads beside it. `--reference` names a run of today's rule to compare turn
by turn (#267's: `.scratch/dflash2-replay-267/replay/out`): under today's rule a replay's
rounds are the recording's, so its rounds per turn are the reference's.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import C, K, TOPK, TT, load, prob_of, read_manifest, sampling_probs  # noqa: E402

CATS = ["think", "prose", "tool-syntax", "tool-args", "edit/write"]
# Output-token share of each source x kind cell in the whole dataset (analyze.py's).
WEIGHTS = {"A-content": 28586, "A-think": 58665, "B-content": 1524816, "B-think": 8088074,
           "C-content": 2085240, "C-think": 6471440}
FIXED = {"draft_rule": "fixed"}


def read_rule(out_dir):
    """The directory's rule.json, or today's rule where there is none (#267's run)."""
    path = os.path.join(out_dir, "rule.json")
    return json.load(open(path)) if os.path.exists(path) else dict(FIXED)


def drafter_q(scores, temperature, top_p):
    """q over the candidates, the leaf's `ignis_drafter_distribution` and the host reference's
    `drafter_distribution`: softmax(scores / T), cut to the smallest prefix in descending
    score (ties toward the smaller index) holding top_p of the mass, renormalized; one-hot on
    the first maximum at T <= 0."""
    order = np.argsort(-scores, -1, kind="stable")
    if temperature <= 0.0:
        q = np.zeros_like(scores, dtype=np.float64)
        np.put_along_axis(q, order[..., :1], 1.0, -1)
        return q
    x = scores.astype(np.float64) / temperature
    q = np.exp(x - x.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)
    if top_p < 1.0:
        qs = np.take_along_axis(q, order, -1)
        cum = np.cumsum(qs, -1)
        kept = np.argmax(cum >= top_p, -1) + 1
        qs = np.where(np.arange(scores.shape[-1]) < kept[..., None], qs, 0.0)
        qs /= qs.sum(-1, keepdims=True)
        q = np.zeros_like(q)
        np.put_along_axis(q, order, qs, -1)
    return q


def rb(a, m):
    """sum_i prod_{j<i} a_j * m_i over the last axis."""
    before = np.cumprod(np.concatenate([np.ones_like(a[..., :1]), a[..., :-1]], -1), -1)
    return (before * m).sum(-1)


def survival(a, m):
    """Expected P(accepted >= i + 1), i = 0..K-1, per round: prod_{j<i} a_j * m_i."""
    before = np.cumprod(np.concatenate([np.ones_like(a[..., :1]), a[..., :-1]], -1), -1)
    return before * m


def sampled_terms(p_ids, p, rec, cand, scores, temperature, top_p):
    """(m, a) of the sampled draft along the recorded chain: q after the recorded token
    before (candidate 0's row at column 0, every row of which follows the anchor), mapped
    onto the target's support slots."""
    n = rec.shape[0]
    rec_in = cand == rec[..., None]                                   # [N, K, 16]
    rc = np.where(rec_in.any(-1), rec_in.argmax(-1), -1)             # recorded token's candidate
    pred = np.concatenate([np.zeros((n, 1), np.int64), np.maximum(rc[:, :-1], 0)], 1)
    rows = np.take_along_axis(scores.astype(np.float64), pred[:, :, None, None], 2)[:, :, 0, :]
    q = drafter_q(rows, temperature, top_p)                          # [N, K, 16]
    eq = p_ids[:, :, :, None] == cand[:, :, None, :]                 # [N, K, TT, 16]
    cm = np.where(eq.any(-1), eq.argmax(-1), -1)                     # candidate of each target slot
    qm = np.where(cm >= 0, np.take_along_axis(q, np.maximum(cm, 0), -1), 0.0)
    m = np.minimum(p, qm).sum(-1)
    p_r = prob_of(p_ids, p, rec)
    q_r = np.where(rc >= 0, np.take_along_axis(q, np.maximum(rc, 0)[..., None], -1)[..., 0], 0.0)
    a = np.where(p_r > 0, np.minimum(p_r, q_r) / np.where(p_r > 0, p_r, 1.0), q_r)
    return m, np.clip(a, 0.0, 1.0)


class Replay:
    """One directory's rounds, on the recorded chain."""

    def __init__(self, out_dir, manifest):
        self.out_dir = out_dir
        self.rule = read_rule(out_dir)
        R, cats = load(out_dir, manifest)
        self.R = R
        n = self.n = len(R)
        turn_meta = {}
        for j, job in enumerate(manifest["jobs"]):
            if j in cats:
                for t in job["turns"]:
                    turn_meta[(j, t["turn"])] = t
        keys = list(zip(R["job"].tolist(), R["turn"].tolist()))
        self.cells = np.array([f"{turn_meta[k]['source']}-{turn_meta[k]['kind']}" for k in keys])
        self.turns = [(manifest["jobs"][k[0]]["job"], k[1]) for k in keys]
        # The target's sampling distribution at every position of the recorded chain, from
        # the columns each round committed.
        probs = sampling_probs(R["top_logits"])
        offsets, total = {}, 0
        for key, t in sorted(turn_meta.items()):
            offsets[key] = total
            total += t["end"] - t["start"]
        P_ids = np.full((max(total, 1), TT), -1, np.int32)
        P_probs = np.zeros((max(total, 1), TT))
        P_cat = np.full(max(total, 1), -1, np.int8)
        base = np.array([offsets[k] - turn_meta[k]["start"] for k in keys], np.int64)
        ends = np.array([turn_meta[k]["end"] for k in keys], np.int64)
        for i in range(C):
            on_chain = i < R["committed"]
            idx = (base + R["anchor"] + i)[on_chain]
            P_ids[idx] = R["top_ids"][on_chain, i]
            P_probs[idx] = probs[on_chain, i]
        for key, off in offsets.items():
            t = turn_meta[key]
            c = cats[key[0]]
            span = np.full(t["end"] - t["start"], -1, np.int8)
            nxt = c[t["start"] + 1:t["end"]]
            span[:len(nxt)] = nxt
            P_cat[off:off + len(span)] = span
        col = np.arange(K)
        anchor = R["anchor"][:, None].astype(np.int64)
        self.gvalid = (anchor + col[None, :]) < ends[:, None]
        gidx = np.where(self.gvalid, base[:, None] + anchor + col[None, :], 0)
        self.p_ids = np.where(self.gvalid[..., None], P_ids[gidx], -1)
        self.p = np.where(self.gvalid[..., None], P_probs[gidx], 0.0)
        self.kind = np.where(self.gvalid, P_cat[gidx], -1)
        self.rec = R["forced"][:, :K]
        # The study's reached columns: on the recorded chain, every draft before taken.
        self.reached = (col[None, :] < R["committed"][:, None]) & self.gvalid
        # Of them, the draft columns (a round's last column past its extent verifies none).
        self.drafted = self.reached & (col[None, :] < R["extent"][:, None])
        self.accepted = R["committed"].astype(np.int64) - 1

    def terms(self, rule):
        """(m, a) of `rule` at these rounds' anchors."""
        if rule["draft_rule"] == "fixed":
            d = self.R["drafts"]
            return prob_of(self.p_ids, self.p, d), (self.rec == d).astype(float)
        temperature = rule.get("temperature", 1.0) * rule.get("drafter_temperature_factor", 0.9)
        return sampled_terms(self.p_ids, self.p, self.rec, self.R["cand_ids"], self.R["scores"], temperature,
                             rule.get("drafter_top_p", 0.95))

    def pooled(self, per_round, mask=None):
        """Traffic-weighted tok/round: total tokens over total rounds, per cell's token share."""
        num = den = 0.0
        for cell, w in WEIGHTS.items():
            s = self.cells == cell
            if mask is not None:
                s = s & mask
            if s.any():
                num += w
                den += w / per_round[s].mean()
        return num / den if den else float("nan")


def summary(replay, rule):
    """Per-round and per-column figures of `rule` at `replay`'s anchors."""
    m, a = replay.terms(rule)
    expected = 1 + rb(a * replay.gvalid, m * replay.gvalid)
    surv = survival(a * replay.gvalid, m * replay.gvalid)
    return {"m": m, "a": a, "expected": expected, "survival": surv}


def print_replay(replay, offline=None):
    rule = replay.rule
    s = summary(replay, rule)
    realized = replay.R["committed"].astype(float)
    name = rule["draft_rule"]
    print(f"\n== {replay.out_dir}: the {name} draft"
          + (f" (drafter T factor {rule['drafter_temperature_factor']}, top-p {rule['drafter_top_p']})"
             if name == "sampled" else ""))
    print(f"rounds {replay.n}, turns {len(set(replay.turns))}, readout disagreements "
          f"{int((replay.R['agreed'] == 0).sum())}")
    print(f"tok/round, traffic-weighted: realized {replay.pooled(realized):.3f}, expected {replay.pooled(s['expected']):.3f}"
          f"   (unweighted: realized {realized.mean():.3f}, expected {s['expected'].mean():.3f})")
    if offline is not None:
        print(f"offline price of the sampled draft at these anchors (the study's estimate): "
              f"{replay.pooled(offline['expected']):.3f}")
    print("by cell             rounds   realized   expected")
    for cell in WEIGHTS:
        sel = replay.cells == cell
        if sel.any():
            print(f"  {cell:16} {int(sel.sum()):7} {realized[sel].mean():10.3f} {s['expected'][sel].mean():10.3f}")
    print("survival per draft column: P(accepted >= i), realized | expected; and P(column taken | reached)")
    print("  col  reached  P(acc>=i) real   exp   taken|reached real   match prob   p(argmax)"
          + ("   offline sampled" if offline is not None else ""))
    for j in range(K):
        reach = replay.reached[:, j]
        drafted = replay.drafted[:, j]
        took = (replay.accepted[drafted] > j).mean() if drafted.any() else float("nan")
        line = (f"  d{j + 1}  {int(reach.sum()):7}  {(replay.accepted > j).mean():14.3f} {s['survival'][:, j].mean():6.3f}"
                f"   {took:18.3f}   {s['m'][reach, j].mean() if reach.any() else float('nan'):10.3f}"
                f"   {replay.p[reach, j, 0].mean() if reach.any() else float('nan'):9.3f}")
        if offline is not None:
            line += f"   {offline['m'][reach, j].mean() if reach.any() else float('nan'):15.3f}"
        print(line)
    print("per kind of output (reached columns pooled): taken|reached realized, match prob"
          + (", offline sampled" if offline is not None else ""))
    for ci, kind in enumerate(CATS):
        sel = replay.reached & (replay.kind == ci)
        if not sel.any():
            continue
        cols = np.nonzero(replay.drafted & (replay.kind == ci))
        took = (replay.accepted[cols[0]] > cols[1]).mean() if len(cols[0]) else float("nan")
        line = f"  {kind:11} {int(sel.sum()):7}  {took:.3f}  {s['m'][sel].mean():.3f}"
        if offline is not None:
            line += f"  {offline['m'][sel].mean():.3f}"
        print(line)
    return s


def compare_turns(replay, reference_dir, manifest):
    """Rounds per turn against a reference run of today's rule."""
    lines = [json.loads(l) for l in open(os.path.join(reference_dir, "turns.jsonl"))]
    ref = {(l["job"], l["turn"]): l for l in lines}
    counts = {}
    for key in replay.turns:
        counts[key] = counts.get(key, 0) + 1
    print(f"\nturns against {reference_dir}: rounds (tok/round) here | there")
    same = 0
    for (job, turn), rounds in sorted(counts.items()):
        there = ref.get((job, turn))
        if there is None:
            print(f"  {job} turn {turn}: {rounds} rounds, not in the reference")
            continue
        out = there["output"]
        same += rounds == there["rounds"]
        print(f"  {job} turn {turn} ({there['source']}-{there['kind']}, {out} tokens): {rounds} ({out / rounds:.3f}) | "
              f"{there['rounds']} ({there['tok_per_round']:.3f})")
    print(f"  {same} of {len(counts)} turns walk the reference's rounds")


def main(argv):
    reference = None
    if "--reference" in argv:
        at = argv.index("--reference")
        reference = argv[at + 1]
        argv = argv[:at] + argv[at + 2:]
    manifest = read_manifest(argv[0])
    replays = [Replay(d, manifest) for d in argv[1:]]
    results = {}
    for replay in replays:
        offline = None
        if replay.rule["draft_rule"] == "fixed":
            offline = summary(replay, {"draft_rule": "sampled", "temperature": 1.0,
                                       "drafter_temperature_factor": 0.9, "drafter_top_p": 0.95})
        print_replay(replay, offline)
        realized = replay.R["committed"].astype(float)
        results[replay.rule["draft_rule"]] = (replay.pooled(realized), replay.pooled(summary(replay, replay.rule)["expected"]))
        if reference is not None and replay.rule["draft_rule"] == "fixed":
            compare_turns(replay, reference, manifest)
    if "fixed" in results and "sampled" in results:
        (fr, fe), (sr, se) = results["fixed"], results["sampled"]
        print(f"\nthe sampled draft against today's, traffic-weighted tok/round: realized {sr:.3f} vs {fr:.3f} "
              f"({100 * (sr / fr - 1):+.1f}%), expected {se:.3f} vs {fe:.3f} ({100 * (se / fe - 1):+.1f}%); "
              f"offline on #267's rounds 3.738 vs 3.461 (+8.0%)")


if __name__ == "__main__":
    main(sys.argv[1:])
