"""Fase 1 — build the symbol index.

One row per symbol: the key is the symbol's name, the value is a hidden state
of the model at layer L with pooling P, taken at the *definition* site
(`02-storage.md` §5.2).  The rows go into a flat `.bin`; the key table, the
inverse file map and the manifest go into a `.json` next to it, so a file that
changes invalidates exactly its own rows (§5.4).

Held-out files are chosen, not sampled: the case under test is "the model
writes code that calls things it does not have in context", so the evaluation
files are the ones that *use* the most symbols defined elsewhere.  They are
excluded from the index and their names recorded in the manifest.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hiddens import hiddens_for_spans
from modeladapter import hidden_size, load_model, set_deterministic
from symbols import iter_corpus, parse_corpus

ROW_DTYPE = np.float16          # the injection renormalises, so 16 bits is ample


def name_pattern(name):
    """A whole-identifier match: `kv` must not fire inside `kv_pages`."""
    return re.compile(r"(?<![A-Za-z0-9_$])%s(?![A-Za-z0-9_$])" % re.escape(name))


def choose_held_out(root, paths, syms, n, max_chars):
    """Files that use the most symbols defined in *other* files."""
    defined = {}
    for s in syms:
        defined.setdefault(s.name, set()).add(s.path)
    names = sorted(defined, key=len, reverse=True)
    patterns = {n_: name_pattern(n_) for n_ in names}

    scored = []
    for path in paths:
        if os.path.getsize(os.path.join(root, path)) > max_chars:
            continue
        with open(os.path.join(root, path), "r", encoding="utf-8",
                  errors="replace") as fh:
            text = fh.read()
        foreign = sum(1 for n_ in names
                      if path not in defined[n_] and patterns[n_].search(text))
        scored.append((foreign, path))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [p for _, p in scored[:n]], dict((p, c) for c, p in scored[:n])


def build(args):
    set_deterministic(args.seed)
    paths = iter_corpus(args.root)
    syms, problems, _ = parse_corpus(args.root, paths)
    max_chars = int(args.max_tokens * 3.487 * 0.85)

    held_out, held_scores = choose_held_out(
        args.root, paths, syms, args.held_out, max_chars)
    held_set = set(held_out)

    # One row per distinct name.  A name defined in two places is ambiguous as
    # a key, so it is dropped rather than resolved arbitrarily; `results` keeps
    # the count so the drop is visible.
    by_name = {}
    for s in syms:
        if s.path in held_set:
            continue
        by_name.setdefault(s.name, []).append(s)
    ambiguous = sorted(k for k, v in by_name.items() if len(v) > 1)
    usable = {k: v[0] for k, v in by_name.items() if len(v) == 1}

    model, tok = load_model(args.model, device_map=args.device, quant=args.quant)
    hsize = hidden_size(model)

    per_file = {}
    for name, s in usable.items():
        per_file.setdefault(s.path, []).append(s)

    # Every (layer, pooling) the sweep needs comes out of ONE pass over the
    # corpus: the forward is the expensive part and it does not depend on which
    # tap is read, so building the six indices separately would run the same
    # 700 forwards six times.  The key set is shared by construction, which is
    # also what makes the sweep's axes comparable.
    layers = [int(x) for x in args.layers.split(",")]
    poolings = tuple(args.poolings.split(","))
    variants = [(l, p) for l in layers for p in poolings]

    keys, file_rows, skipped = [], {}, []
    rows = {v: [] for v in variants}
    files = sorted(per_file)
    t0 = time.time()
    for i, path in enumerate(files):
        full = os.path.join(args.root, path)
        if os.path.getsize(full) > max_chars:
            skipped.extend(s.name for s in per_file[path])
            continue
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        group = sorted(per_file[path], key=lambda s: s.body_span[0])
        got, per_span = hiddens_for_spans(
            model, tok, text, layers, [s.body_span for s in group],
            poolings, max_tokens=args.max_tokens)
        if not got:
            skipped.extend(s.name for s in group)
            continue
        ids = []
        for j, s in enumerate(group):
            if not per_span[j] or any(bool(got[v][j].isnan().any())
                                      for v in variants):
                skipped.append(s.name)
                continue
            ids.append(len(keys))
            keys.append(s.name)
            for v in variants:
                rows[v].append(got[v][j].numpy().astype(ROW_DTYPE))
        file_rows[path] = ids
        if (i + 1) % 25 == 0:
            print("  %d/%d files, %d rows, %.1f min"
                  % (i + 1, len(files), len(keys), (time.time() - t0) / 60),
                  flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    manifests = {}
    for layer, pooling in variants:
        block = (np.stack(rows[(layer, pooling)]) if rows[(layer, pooling)]
                 else np.zeros((0, hsize), ROW_DTYPE))
        prefix = "%s-L%d-%s" % (args.out, layer, pooling)
        with open(prefix + ".bin", "wb") as fh:
            fh.write(block.tobytes())
        manifest = {
            "layer": layer, "pooling": pooling,
            "layer_convention": "output of decoder layer index N, 0-based, "
                                "as registered by a forward hook",
            "model": args.model, "quant": args.quant,
            "max_tokens": args.max_tokens, "seed": args.seed,
            "hidden_size": int(block.shape[1]) if len(block) else hsize,
            "rows": int(block.shape[0]), "dtype": str(np.dtype(ROW_DTYPE)),
            "sha256": hashlib.sha256(block.tobytes()).hexdigest(),
            "held_out_files": held_out,
            "held_out_foreign_symbol_counts": held_scores,
            "ambiguous_names_dropped": len(ambiguous),
            "ambiguous_sample": ambiguous[:20],
            "skipped_names": len(skipped),
            "parse_problems": len(problems),
            "build_seconds": round(time.time() - t0, 1),
            "keys": keys,
            "file_rows": file_rows,
        }
        with open(prefix + ".json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        manifests[prefix] = manifest
        print("%s: %d rows, sha %s"
              % (prefix, manifest["rows"], manifest["sha256"][:16]), flush=True)

    print(json.dumps({"files": len(files), "rows": len(keys),
                      "held_out": held_out,
                      "ambiguous_dropped": len(ambiguous),
                      "skipped": len(skipped),
                      "minutes": round((time.time() - t0) / 60, 1)},
                     indent=2, sort_keys=True), flush=True)
    return manifests


def load_index(prefix):
    with open(prefix + ".json", encoding="utf-8") as fh:
        man = json.load(fh)
    rows = np.fromfile(prefix + ".bin", dtype=np.dtype(man["dtype"]))
    rows = rows.reshape(man["rows"], man["hidden_size"])
    return man, torch.from_numpy(rows.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True,
                    help="path prefix; -L<layer>-<pooling> is appended")
    ap.add_argument("--layers", required=True,
                    help="comma separated decoder layer indices")
    ap.add_argument("--poolings", default="last,mean")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quant", default="nf4", choices=("nf4", "none"))
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--held-out", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
