"""Fase 0 — is there anything in these hidden states to retrieve?

`02-storage.md` §7 defines two tests with different jobs.

Test 0a, the go/no-go: are the end-of-definition hidden states *spread out* or
collapsed onto one direction?  A causal model trains the last token of a body
(`}`) to predict `\n\n`, so they might all be nearly the same vector, and then
there is nothing to store.

The plan asks for mean pairwise cosine and the variance explained by the first
principal component.  Raw cosine in a residual stream is not readable on its
own: a handful of massive-activation dimensions give *any* two positions a
cosine around 0.8-0.9, so a high number means nothing without a reference.
Two things fix that, and both are reported: cosines are also computed after
mean-centering the set, and a control set of positions that are *not*
definition ends is measured the same way.  0a's verdict is the comparison
between the two, not the absolute number.

Test 0b, which tests the gate rather than the idea: for each symbol, is the
hidden at a *use* site closer to its own definition vector than to other
symbols' vectors?  The key match is already exact, so a low recall@1 does not
kill the idea — it says a cosine threshold cannot be the selectivity filter.
"""

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hiddens import hiddens_for_spans
from modeladapter import load_model, num_layers, set_deterministic
from symbols import iter_corpus, parse_corpus


def _cosine_stats(mat):
    """Mean/median pairwise cosine and first-PC variance share for a set of
    row vectors, raw and after removing the set mean."""
    out = {}
    for label, m in (("raw", mat), ("centered", mat - mat.mean(dim=0, keepdim=True))):
        norm = m / m.norm(dim=1, keepdim=True).clamp_min(1e-9)
        gram = norm @ norm.T
        n = gram.shape[0]
        off = gram[~torch.eye(n, dtype=torch.bool)]
        # first principal component's share of the total variance
        centred = m - m.mean(dim=0, keepdim=True)
        sv = torch.linalg.svdvals(centred.double())
        var = sv.pow(2)
        out[label] = {
            "mean_cosine": float(off.mean()),
            "median_cosine": float(off.median()),
            "p95_cosine": float(off.quantile(0.95)),
            "pc1_variance_share": float(var[0] / var.sum()),
            "effective_rank": float(
                torch.exp(-(var / var.sum() * (var / var.sum()).log()).sum())),
            "n": n,
        }
    return out


def _recall_at_1(queries, values, gold):
    """Fraction of query vectors whose nearest value (cosine) is the right one."""
    q = queries / queries.norm(dim=1, keepdim=True).clamp_min(1e-9)
    v = values / values.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sim = q @ v.T
    best = sim.argmax(dim=1)
    gold_sim = sim.gather(1, gold.unsqueeze(1)).squeeze(1)
    rank = (sim > gold_sim.unsqueeze(1)).sum(dim=1)
    return {
        "recall_at_1": float((best == gold).float().mean()),
        "mean_rank": float(rank.float().mean()) + 1.0,
        "mean_gold_cosine": float(gold_sim.mean()),
        "mean_other_cosine": float(
            (sim.sum(dim=1) - gold_sim).mean() / (sim.shape[1] - 1)),
        "n_queries": int(q.shape[0]),
        "n_values": int(v.shape[0]),
    }


def pick_symbols(root, n, rng, min_body_chars=120, max_file_chars=None,
                 per_file_cap=4):
    """Symbols with a body big enough to summarise, one per distinct name.

    Two constraints beyond "is a definition".  A file longer than the token
    budget gets truncated, and every symbol past the cut would silently come
    back empty, so oversized files are dropped outright rather than half read.
    And a cap per file keeps the sample spread across the corpus instead of
    being one enormous module's worth of functions, which would make the
    cosine spread a property of that module.
    """
    paths = iter_corpus(root)
    if max_file_chars is not None:
        paths = [p for p in paths
                 if os.path.getsize(os.path.join(root, p)) <= max_file_chars]
    syms, _, _ = parse_corpus(root, paths)
    by_name = {}
    for s in syms:
        if s.depth != 0 or s.kind not in ("fn", "function", "struct", "class"):
            continue
        if s.body_span[1] - s.body_span[0] < min_body_chars:
            continue
        by_name.setdefault(s.name, []).append(s)
    names = sorted(k for k, v in by_name.items() if len(v) == 1)
    rng.shuffle(names)
    chosen, per_file = [], {}
    for name in names:
        s = by_name[name][0]
        if per_file.get(s.path, 0) >= per_file_cap:
            continue
        per_file[s.path] = per_file.get(s.path, 0) + 1
        chosen.append(s)
        if len(chosen) >= n:
            break
    chosen.sort(key=lambda s: (s.path, s.body_span[0]))
    return chosen


def control_spans(text, offsets_count, rng, n, width):
    """Character spans of the same size as a definition but at arbitrary
    positions: the reference 0a is read against."""
    out = []
    if len(text) <= width + 1:
        return out
    for _ in range(n):
        a = rng.randrange(0, len(text) - width)
        out.append((a, a + width))
    return out


def run(args):
    set_deterministic(args.seed)
    rng = random.Random(args.seed)

    model, tok = load_model(args.model, device_map=args.device,
                            quant=args.quant)
    n_layers = num_layers(model)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else sorted({1, n_layers // 3, 2 * n_layers // 3, n_layers - 1}))
    print("layers under test:", layers, "of", n_layers, flush=True)

    # A file is dropped when it cannot fit the token budget whole: 3.487 bytes
    # per token is this corpus's measured rate (results/corpus-stats.json), and
    # the margin keeps a denser-than-average file from being cut mid-way.
    chosen = pick_symbols(args.root, args.n_symbols, rng,
                          max_file_chars=int(args.max_tokens * 3.487 * 0.85))
    print("symbols:", len(chosen), flush=True)

    by_file = {}
    for s in chosen:
        by_file.setdefault(s.path, []).append(s)

    defs = {k: [] for k in ((l, p) for l in layers for p in ("last", "mean"))}
    ctrl = {k: [] for k in defs}
    uses = {(l, "last"): [] for l in layers}
    use_gold = []
    kept_names = []

    import time as _time
    _t0 = _time.time()
    _files = sorted(by_file.items())
    for _i, (path, syms) in enumerate(_files):
        if _i % 10 == 0:
            print("  0a %d/%d files, %.1f min" % (_i, len(_files),
                                                  (_time.time() - _t0) / 60),
                  flush=True)
        with open(os.path.join(args.root, path), "r", encoding="utf-8",
                  errors="replace") as fh:
            text = fh.read()
        spans = [s.body_span for s in syms]
        widths = [b - a for a, b in spans]
        ctl = control_spans(text, 0, rng, len(spans),
                            max(1, sum(widths) // max(1, len(widths))))
        got, per_span = hiddens_for_spans(
            model, tok, text, layers, spans + ctl, ("last", "mean"),
            max_tokens=args.max_tokens)
        if not got:
            continue
        ok = [i for i in range(len(spans)) if per_span[i]]
        for key, rows in got.items():
            defs[key].append(rows[:len(spans)][ok])
            ctrl[key].append(rows[len(spans):])
        kept_names.extend(syms[i].name for i in ok)

    report = {"model": args.model, "quant": args.quant, "layers": layers,
              "n_symbols_requested": args.n_symbols,
              "n_symbols_measured": len(kept_names), "seed": args.seed,
              "test_0a": {}, "test_0b": {}}

    for key in sorted(defs, key=lambda k: (k[0], k[1])):
        d = torch.cat([x for x in defs[key] if len(x)], dim=0)
        c = torch.cat([x for x in ctrl[key] if len(x)], dim=0)
        d = d[~d.isnan().any(dim=1)]
        c = c[~c.isnan().any(dim=1)]
        report["test_0a"]["L%d/%s" % key] = {
            "definitions": _cosine_stats(d),
            "control": _cosine_stats(c),
        }

    # 0b needs use sites, which live in files the definition is not in.
    if args.uses:
        report["test_0b"] = _test_0b(args, model, tok, layers, chosen,
                                     defs, kept_names, rng)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report["test_0a"], indent=2, sort_keys=True))
    return report


def _test_0b(args, model, tok, layers, chosen, defs, kept_names, rng):
    """Cosine between the hidden at a use site and the definition vector."""
    import re as _re
    name_set = {s.name for s in chosen}
    def_file = {s.name: s.path for s in chosen}
    # Whole-identifier, like every other match in this study: `kv` found inside
    # `kv_pages` would put the query hidden at a position that has nothing to
    # do with the symbol, and quietly lower the recall it is meant to measure.
    pats = {n: _re.compile(r"(?<![A-Za-z0-9_$])%s(?![A-Za-z0-9_$])"
                           % _re.escape(n)) for n in name_set}
    paths = iter_corpus(args.root)
    sites = {}
    for path in paths:
        with open(os.path.join(args.root, path), "r", encoding="utf-8",
                  errors="replace") as fh:
            text = fh.read()
        for name in name_set:
            if def_file[name] == path or name in sites:
                continue
            m = pats[name].search(text)
            if m:
                sites[name] = (path, m.span())
    out = {}
    by_file = {}
    for name, (path, span) in sites.items():
        by_file.setdefault(path, []).append((name, span))

    vectors = {l: {} for l in layers}
    import time as _time
    _t0 = _time.time()
    _items = sorted(by_file.items())
    for _i, (path, items) in enumerate(_items):
        if _i % 10 == 0:
            print("  0b %d/%d files, %.1f min" % (_i, len(_items),
                                                  (_time.time() - _t0) / 60),
                  flush=True)
        with open(os.path.join(args.root, path), "r", encoding="utf-8",
                  errors="replace") as fh:
            text = fh.read()
        got, per_span = hiddens_for_spans(
            model, tok, text, layers, [s for _, s in items], ("last",),
            max_tokens=args.max_tokens)
        for (layer, _how), rows in got.items():
            for i, (name, _) in enumerate(items):
                if not rows[i].isnan().any():
                    vectors[layer][name] = rows[i]

    index = {n: i for i, n in enumerate(kept_names)}
    for layer in layers:
        common = [n for n in vectors[layer] if n in index]
        if len(common) < 8:
            out["L%d" % layer] = {"skipped": "only %d use sites" % len(common)}
            continue
        values = torch.cat([x for x in defs[(layer, "last")] if len(x)], dim=0)
        q = torch.stack([vectors[layer][n] for n in common])
        gold = torch.tensor([index[n] for n in common])
        out["L%d" % layer] = _recall_at_1(q, values, gold)
    return out


def determinism_check(args):
    """Two identical forwards must give identical logits, or the alpha=0 control
    in Fase 3 cannot be read (`02-storage.md` §7)."""
    set_deterministic(args.seed)
    model, tok = load_model(args.model, device_map=args.device, quant=args.quant)
    text = open(os.path.join(args.root, args.det_file), encoding="utf-8",
                errors="replace").read()[:4000]
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(next(model.parameters()).device)
    with torch.no_grad():
        a = model(input_ids=ids, use_cache=False).logits.float().cpu()
        b = model(input_ids=ids, use_cache=False).logits.float().cpu()
    same = bool(torch.equal(a, b))
    report = {"bitwise_identical": same,
              "max_abs_diff": float((a - b).abs().max()),
              "tokens": int(ids.shape[1]), "model": args.model}
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quant", default="nf4", choices=("nf4", "none"))
    ap.add_argument("--layers", default=None, help="comma separated")
    ap.add_argument("--n-symbols", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uses", action="store_true",
                    help="also run test 0b (needs a second pass over the corpus)")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--det-file", default="crates/core/src/prefix.rs")
    args = ap.parse_args()

    if args.determinism:
        rep = determinism_check(args)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rep, fh, indent=2, sort_keys=True)
        return
    run(args)


if __name__ == "__main__":
    main()
