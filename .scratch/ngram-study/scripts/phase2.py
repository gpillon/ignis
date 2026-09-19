"""Fasi 2-4 — inject, measure, sweep, and check the damage out of domain.

One index (fixed layer and pooling, built by `index_build.py`) is swept over
alpha and the cosine threshold, against the held-out files recorded in the
index manifest.  The criteria in `02-storage.md` §7 read the *primary* metric:
the NLL over the tokens following a match, not the whole-file number.

Fase 4 runs the same measurement with the same index over a corpus from
another project.  The plan is explicit that the out-of-domain corpus must be
code, not prose: on prose the match rate is about zero and the delta comes out
at zero by construction, which is a false all-clear.  What actually costs
something is a symbol name that is also a common word of code — `step`,
`host`, `next` — matching constantly on foreign code and injecting hidden
states that belong to this repo.  So the match rate is reported next to every
out-of-domain number.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from index_build import load_index, name_pattern
from inject import cosine_probe
from modeladapter import load_model, set_deterministic
from nll import bootstrap_ci, build_matches, run_file, window_mean
from symbols import iter_corpus

WINDOWS = (8, 32)


def rare_keys(keys, tokenizer):
    """§7 Fase 4's operational form of "rare and distinctive": at least two BPE
    tokens, and contained in no other key."""
    lens = {k: len(tokenizer(k, add_special_tokens=False)["input_ids"])
            for k in keys}
    ordered = sorted(keys, key=len)
    out = []
    for k in keys:
        if lens[k] < 2:
            continue
        if any(other != k and k in other for other in ordered
               if len(other) > len(k)):
            continue
        out.append(k)
    return out


def eval_corpus(root, files, model, tok, keys, key_rows, vectors, layer,
                alphas, thresholds, max_tokens, chunk, label, log=print,
                center=None):
    """Run every (alpha, threshold) over `files`, against the alpha=0 baseline."""
    texts = {}
    for path in files:
        with open(os.path.join(root, path), "r", encoding="utf-8",
                  errors="replace") as fh:
            texts[path] = fh.read()

    # Baseline, and the cosine distribution the thresholds are read off.
    base, cos_all = {}, []
    for path in files:
        t0 = time.time()
        r = run_file(model, tok, texts[path], keys, key_rows, vectors, layer,
                     alpha=0.0, max_tokens=max_tokens, chunk=chunk,
                     center=center)
        base[path] = r
        ids, offsets, pos, rows, _names = build_matches(
            tok, texts[path], keys, key_rows, max_tokens=max_tokens)
        if pos:
            device = next(model.parameters()).device
            c = cosine_probe(
                model, layer,
                torch.tensor(pos, dtype=torch.long, device=device),
                vectors.index_select(0, torch.tensor(rows)).to(device),
                torch.tensor([ids], dtype=torch.long, device=device),
                center=center)
            cos_all.extend(c.tolist())
        log("  [%s] baseline %-52s %5d tok %4d matches %.1fs"
            % (label, path[-52:], r["tokens"], len(r["positions"]),
               time.time() - t0))

    cos_sorted = sorted(cos_all)
    tau = {"none": None}
    if cos_sorted:
        tau["median"] = cos_sorted[len(cos_sorted) // 2]
        tau["high"] = cos_sorted[int(0.9 * (len(cos_sorted) - 1))]

    results = {}
    for alpha in alphas:
        for tname in (["none"] if alpha == 0.0 else list(tau)):
            key = "alpha=%g/tau=%s" % (alpha, tname)
            per_file, whole, rates, injected, identical = {}, [], [], [], True
            for path in files:
                b = base[path]
                r = run_file(model, tok, texts[path], keys, key_rows, vectors,
                             layer, alpha=alpha, cos_threshold=tau[tname],
                             max_tokens=max_tokens, chunk=chunk,
                             force_hook=(alpha == 0.0), center=center)
                if alpha == 0.0:
                    identical = identical and bool(
                        torch.equal(r["nll"], b["nll"]))
                for w in WINDOWS:
                    bm, n = window_mean(b["nll"], b["positions"], w)
                    rm, _ = window_mean(r["nll"], r["positions"], w)
                    if bm is None or bm == 0:
                        continue
                    per_file.setdefault(w, []).append(100.0 * (rm - bm) / bm)
                whole.append(100.0 * (float(r["nll"].mean()) -
                                      float(b["nll"].mean())) /
                             float(b["nll"].mean()))
                rates.append(100.0 * len(r["positions"]) / max(1, r["tokens"]))
                injected.append(100.0 * r["injected"] / max(1, r["tokens"]))
            results[key] = {
                "alpha": alpha, "tau_name": tname, "tau": tau[tname],
                "primary": {str(w): bootstrap_ci(per_file.get(w, []))
                            for w in WINDOWS},
                "whole_file": bootstrap_ci(whole),
                "match_rate_pct": sum(rates) / max(1, len(rates)),
                "injected_rate_pct": sum(injected) / max(1, len(injected)),
                "alpha0_reproduces_baseline": identical if alpha == 0.0 else None,
            }
            log("  [%s] %-22s primary8 %s  match %.2f%%  inj %.2f%%"
                % (label, key,
                   _fmt(results[key]["primary"]["8"]),
                   results[key]["match_rate_pct"],
                   results[key]["injected_rate_pct"]))
    return results, tau, cos_sorted


def _fmt(ci):
    if not ci:
        return "n/a"
    return "%+.3f%% [%+.3f, %+.3f]" % (ci["mean"], ci["lo"], ci["hi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--index", required=True, help="prefix without extension")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quant", default="nf4", choices=("nf4", "none"))
    ap.add_argument("--alphas", default="0,0.1,0.3,1.0")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--rare-only", action="store_true",
                    help="restrict keys to the rare/distinctive set (Fase 4)")
    ap.add_argument("--ood-root", default=None,
                    help="a second corpus, from another project (Fase 4)")
    ap.add_argument("--ood-files", type=int, default=20)
    ap.add_argument("--center", action="store_true",
                    help="subtract the index mean from every row before "
                         "normalising: alpha then scales the symbol-specific "
                         "component instead of the layer's shared direction")
    ap.add_argument("--shuffle-values", action="store_true",
                    help="permute the key->row mapping: the negative control "
                         "that separates 'the matched row carries information' "
                         "from 'any vector of that norm does this'")
    args = ap.parse_args()

    set_deterministic(0)
    man, vectors = load_index(args.index)
    keys = man["keys"]
    key_rows = {k: i for i, k in enumerate(keys)}
    model, tok = load_model(args.model, device_map=args.device, quant=args.quant)

    if args.shuffle_values:
        # Same keys, same match positions, same number of injections, same
        # vector norms — only the pairing is wrong.  If the delta is the same
        # as with the correct pairing, the key match carries no information
        # and what is being measured is the perturbation, not the retrieval.
        import random as _random
        rows = list(key_rows.values())
        _random.Random(1234).shuffle(rows)
        key_rows = dict(zip(key_rows.keys(), rows))

    if args.rare_only:
        keep = rare_keys(keys, tok)
        key_rows = {k: key_rows[k] for k in keep}
        keys = keep
    print("index rows %d, keys in use %d, layer %d, pooling %s"
          % (man["rows"], len(keys), man["layer"], man["pooling"]), flush=True)

    alphas = [float(x) for x in args.alphas.split(",")]
    report = {"index": args.index, "layer": man["layer"],
              "pooling": man["pooling"], "model": args.model,
              "quant": args.quant, "rare_only": args.rare_only,
              "shuffle_values": args.shuffle_values,
              "keys_in_use": len(keys), "alphas": alphas}

    center = vectors.float().mean(dim=0) if args.center else None
    report["center"] = bool(args.center)

    t0 = time.time()
    in_dom, tau, cos = eval_corpus(
        args.root, man["held_out_files"], model, tok, keys, key_rows, vectors,
        man["layer"], alphas, None, args.max_tokens, args.chunk, "in-domain",
        center=center)
    report["in_domain"] = in_dom
    report["thresholds"] = tau
    report["cosine_percentiles"] = {
        p: (cos[int(float(p) / 100 * (len(cos) - 1))] if cos else None)
        for p in ("5", "25", "50", "75", "95")
    }

    if args.ood_root:
        ood_files = iter_corpus(args.ood_root)
        cap = int(args.max_tokens * 3.487 * 0.85)
        ood_files = [p for p in ood_files
                     if os.path.getsize(os.path.join(args.ood_root, p)) <= cap]
        # Evenly spaced through the sorted list rather than the first N: taking
        # the head would sample one directory (`apps/`, `bench/`) and measure
        # that subsystem's vocabulary overlap instead of the project's.
        if len(ood_files) > args.ood_files:
            step = len(ood_files) / args.ood_files
            ood_files = [ood_files[int(i * step)] for i in range(args.ood_files)]
        ood, _, _ = eval_corpus(
            args.ood_root, ood_files, model, tok, keys, key_rows, vectors,
            man["layer"], alphas, None, args.max_tokens, args.chunk, "ood",
            center=center)
        report["ood"] = ood
        report["ood_files"] = ood_files
        report["ood_root"] = args.ood_root

    report["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print("wrote", args.out, "in %.1f min" % (report["wall_seconds"] / 60))


if __name__ == "__main__":
    main()
