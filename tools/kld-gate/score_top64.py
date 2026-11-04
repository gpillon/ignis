#!/usr/bin/env python3
"""Score a dense span_logits dump against the ref27b-top64 reference set.

Needs only python3 and numpy. Reads the set (manifest.json and its files)
and a dump dir (`wNNN.bin`, one per window, `[length][vocab]` little-endian
BF16, as `span_logits ARTIFACT KV windows.json OUT_DIR CHUNK` writes them),
streaming each window in slices of rows, so memory stays near
`--rows x vocab x 12` bytes whatever the dump's size.

Per position it measures, against the set's BF16 stream:
  - KL(BF16 || engine) with bench flash_next/kld.rs `kl_top64`'s formula: the
    BF16 top-64 log-probs r, the engine's log-probs c at the same ids
    (BF16 logits normalised by their own f64 log-sum-exp), the BF16
    remaining mass R one bucket against the engine's mass outside those ids:
      sum_i e^{r_i}(r_i - c_i) + R (log R - log(1 - sum_i e^{c_i})),
    the tail term dropped when R < 1e-12. A lower bound of the exact KL;
  - top-1 agreement with BF16 (engine argmax, ties by the lower id);
  - top-1 agreement with the set's 5090 trellis stream (`q_argmax`);
  - the NLL of the actual next token (a window's last position excluded);
and reports them per kind and over all positions (every position weighs the
same). The set's own 5090 trellis figure (from `q_lp_at_bf16_ids.f32`) and
the full-vocabulary KLD it was measured beside are printed for comparison.

Usage: python3 score_top64.py SET_DIR DUMP_DIR [--rows N] [--json OUT] [--per-position DIR]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

TAIL_EPS = 1e-12  # kld.rs TAIL_EPS


def kl_top64(ref_lp, cand_lp):
    """kld.rs kl_top64 per row: ref_lp (n, k) f32, cand_lp (n, k) f64."""
    r = ref_lp.astype(np.float64)
    c = cand_lp.astype(np.float64)
    p = np.exp(r)
    head = (p * (r - c)).sum(axis=1)
    ref_tail = np.maximum(1.0 - p.sum(axis=1), 0.0)
    cand_tail = np.maximum(1.0 - np.exp(c).sum(axis=1), TAIL_EPS)
    tail = ref_tail * (np.log(np.maximum(ref_tail, TAIL_EPS)) - np.log(cand_tail))
    return head + np.where(ref_tail < TAIL_EPS, 0.0, tail)


def read_words(set_dir, name, dtype, count):
    path = os.path.join(set_dir, name)
    size = os.path.getsize(path)
    want = count * np.dtype(dtype).itemsize
    if size != want:
        sys.exit("%s: %d bytes, the manifest implies %d" % (path, size, want))
    return np.fromfile(path, dtype=dtype)


class ReferenceSet(object):
    """The set read whole (about 90 MB), checked as references.rs checks it."""

    def __init__(self, set_dir):
        m = json.load(open(os.path.join(set_dir, "manifest.json")))
        self.manifest = m
        self.windows = m["windows"]
        self.top = int(m["top"])
        self.vocab = int(m["vocab"])
        positions = tokens = 0
        self.token_start = []
        for i, w in enumerate(self.windows):
            if w["index"] != i or w["first_position"] != positions or not (0 < w["valid"] <= w["length"]):
                sys.exit("manifest: window %d is out of order" % i)
            self.token_start.append(tokens)
            tokens += w["length"]
            positions += w["valid"]
        if positions != m["positions"]:
            sys.exit("manifest: the windows hold %d positions, the manifest says %d" % (positions, m["positions"]))
        P, T = positions, self.top
        self.positions = P
        self.tokens = read_words(set_dir, "tokens.u32", "<u4", tokens)
        self.bf16_ids = read_words(set_dir, "bf16_top64_ids.i32", "<i4", P * T).reshape(P, T)
        self.bf16_lp = read_words(set_dir, "bf16_top64_lp.f32", "<f4", P * T).reshape(P, T)
        self.q_argmax = read_words(set_dir, "q_argmax.i32", "<i4", P)
        q_at = os.path.join(set_dir, "q_lp_at_bf16_ids.f32")
        self.q_at_bf16 = read_words(set_dir, "q_lp_at_bf16_ids.f32", "<f4", P * T).reshape(P, T) \
            if os.path.exists(q_at) else None
        nl = os.path.join(set_dir, "bf16_next_lp.f32")
        self.bf16_next_lp = read_words(set_dir, "bf16_next_lp.f32", "<f4", P) if os.path.exists(nl) else None
        if (self.bf16_ids < 0).any() or (self.bf16_ids >= self.vocab).any():
            sys.exit("a stored BF16 token id is outside the vocabulary")
        full = os.path.join(set_dir, "full_kld_5090.json")
        self.full = json.load(open(full)) if os.path.exists(full) else None


class Tally(object):
    def __init__(self):
        self.kl = []
        self.top1 = []
        self.top1_q = []
        self.nll = []
        self.ref_nll = []
        self.q_kl = []

    def figures(self):
        kl = np.concatenate(self.kl)
        out = {"positions": int(kl.size), "kld_top64": float(kl.mean()), "median": float(np.median(kl)),
               "p99": float(np.percentile(kl, 99)), "max": float(kl.max()),
               "top1_bf16": float(np.concatenate(self.top1).mean()),
               "top1_q5090": float(np.concatenate(self.top1_q).mean())}
        nll = np.concatenate(self.nll)
        out["nll"] = float(nll.mean())
        out["ppl"] = float(math.exp(out["nll"]))
        if self.ref_nll:
            rn = np.concatenate(self.ref_nll)
            out["ref_nll"] = float(rn.mean())
            out["ref_ppl"] = float(math.exp(out["ref_nll"]))
        if self.q_kl:
            out["q5090_kld_top64"] = float(np.concatenate(self.q_kl).mean())
        return out


def score_window(ref, w, dump_dir, rows_per_slice):
    """Per-position arrays for one window, streamed from its dump file."""
    V, T = ref.vocab, ref.top
    path = os.path.join(dump_dir, "w%03d.bin" % w["index"])
    if not os.path.exists(path):
        sys.exit("%s: missing" % path)
    size = os.path.getsize(path)
    if size != w["length"] * V * 2:
        sys.exit("%s: %d bytes, expected %d (length %d x vocab %d x 2)" % (path, size, w["length"] * V * 2, w["length"], V))
    n = w["valid"]
    p0 = w["first_position"]
    tok = ref.tokens[ref.token_start[w["index"]]:ref.token_start[w["index"]] + w["length"]].astype(np.int64)
    kl = np.empty(n)
    top1 = np.empty(n, dtype=bool)
    top1_q = np.empty(n, dtype=bool)
    nll = np.full(n, np.nan)
    with open(path, "rb") as f:
        for s in range(0, n, rows_per_slice):
            e = min(n, s + rows_per_slice)
            f.seek(s * V * 2)
            raw = np.frombuffer(f.read((e - s) * V * 2), dtype="<u2").reshape(e - s, V)
            x = (raw.astype(np.uint32) << 16).view(np.float32)
            del raw
            bad = ~np.isfinite(x).all(axis=1)
            if bad.any():
                sys.exit("%s: row %d holds a NaN or an infinity" % (path, s + int(np.argmax(bad))))
            mx = x.max(axis=1)
            ex = np.exp(x - mx[:, None])
            lse = mx.astype(np.float64) + np.log(ex.sum(axis=1, dtype=np.float64))
            del ex
            rows = np.arange(e - s)[:, None]
            ids = ref.bf16_ids[p0 + s:p0 + e]
            cand = x[rows, ids].astype(np.float64) - lse[:, None]
            kl[s:e] = kl_top64(ref.bf16_lp[p0 + s:p0 + e], cand)
            am = x.argmax(axis=1)  # the first maximum: the lowest id on ties
            top1[s:e] = am == ids[:, 0]
            top1_q[s:e] = am == ref.q_argmax[p0 + s:p0 + e]
            last = min(e, w["length"] - 1)  # positions with a next token in the window
            if last > s:
                nxt = tok[s + 1:last + 1]
                nll[s:last] = -(x[np.arange(last - s), nxt].astype(np.float64) - lse[:last - s])
            del x
    return kl, top1, top1_q, nll


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("set_dir")
    ap.add_argument("dump_dir")
    ap.add_argument("--rows", type=int, default=256, help="rows per slice (memory ~ rows x vocab x 12 bytes)")
    ap.add_argument("--json", help="write the figures here")
    ap.add_argument("--per-position", help="write kl_top64.f64, top1_bf16.u8, top1_q5090.u8, nll.f64 here")
    ap.add_argument("--windows", type=int, default=0, help="score only the first N windows")
    a = ap.parse_args(argv)

    ref = ReferenceSet(a.set_dir)
    windows = ref.windows[:a.windows] if a.windows else ref.windows
    tallies = {"all": Tally()}
    per_pos = {"kl": [], "top1": [], "top1_q": [], "nll": []}
    t0 = time.time()
    for w in windows:
        t1 = time.time()
        kl, top1, top1_q, nll = score_window(ref, w, a.dump_dir, a.rows)
        p0, n = w["first_position"], w["valid"]
        has_next = ~np.isnan(nll)
        q_kl = kl_top64(ref.bf16_lp[p0:p0 + n], ref.q_at_bf16[p0:p0 + n]) if ref.q_at_bf16 is not None else None
        for key in ("all", w["kind"]):
            t = tallies.setdefault(key, Tally())
            t.kl.append(kl)
            t.top1.append(top1)
            t.top1_q.append(top1_q)
            t.nll.append(nll[has_next])
            if ref.bf16_next_lp is not None:
                t.ref_nll.append(-ref.bf16_next_lp[p0:p0 + n][has_next].astype(np.float64))
            if q_kl is not None:
                t.q_kl.append(q_kl)
        for k, v in (("kl", kl), ("top1", top1), ("top1_q", top1_q), ("nll", nll)):
            per_pos[k].append(v)
        print("w%02d %-5s KLD(top64) %.5f  top1 %.2f%%  top1 vs 5090 %.2f%%  ppl %.3f  (%.1fs)" % (
            w["index"], w["kind"], kl.mean(), top1.mean() * 100, top1_q.mean() * 100,
            math.exp(nll[has_next].mean()), time.time() - t1), flush=True)

    figures = {k: t.figures() for k, t in tallies.items()}
    order = ["all"] + sorted(k for k in figures if k != "all")
    full = ref.full["summary"] if ref.full and not a.windows else {}  # whole-set figures only
    print()
    print("%-6s %8s %10s %8s %8s %9s %8s %8s %10s %10s" % (
        "kind", "pos", "KLD top64", "median", "p99", "top-1", "vs 5090", "ppl", "5090 top64", "5090 full"))
    for k in order:
        f = figures[k]
        print("%-6s %8d %10.5f %8.4f %8.3f %8.2f%% %7.2f%% %8.3f %10s %10s" % (
            k, f["positions"], f["kld_top64"], f["median"], f["p99"], f["top1_bf16"] * 100, f["top1_q5090"] * 100,
            f["ppl"], "%.5f" % f["q5090_kld_top64"] if "q5090_kld_top64" in f else "-",
            "%.5f" % full[k]["mean"] if k in full else "-"))
    if "ref_ppl" in figures["all"]:
        print("BF16 reference ppl: " + ", ".join("%s %.3f" % (k, figures[k]["ref_ppl"]) for k in order))
    print("scored in %.0fs" % (time.time() - t0))

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"set": os.path.abspath(a.set_dir), "dump": os.path.abspath(a.dump_dir),
                       "figures": figures}, f, indent=1)
    if a.per_position:
        if not os.path.isdir(a.per_position):
            os.makedirs(a.per_position)
        for name, key, dt in (("kl_top64.f64", "kl", "<f8"), ("top1_bf16.u8", "top1", "u1"),
                              ("top1_q5090.u8", "top1_q", "u1"), ("nll.f64", "nll", "<f8")):
            np.concatenate(per_pos[key]).astype(dt).tofile(os.path.join(a.per_position, name))


if __name__ == "__main__":
    main()
