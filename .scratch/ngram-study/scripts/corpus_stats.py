"""Corpus and BPE statistics for `03-risultati.md`.

`02-storage.md` §4 counted n-grams word-level and flagged the gap to BPE as a
caveat worth ±2x.  The tokenizer is a download away, so this replaces the
caveat with the real numbers, and it also produces the operational form of
"rare and distinctive keys" (§7 Fase 4): a key qualifies when its name is at
least 2 BPE tokens *and* no other indexed symbol name contains it.

Usage:
    python corpus_stats.py --root <repo> --tokenizer <path> --out <json>
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symbols import iter_corpus, parse_corpus


def bpe_stats(tokenizer, texts):
    """Distinct BPE 2-grams and 3-grams over the corpus, plus token count."""
    total = 0
    two, three = set(), set()
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        total += len(ids)
        for i in range(len(ids) - 1):
            two.add((ids[i], ids[i + 1]))
        for i in range(len(ids) - 2):
            three.add((ids[i], ids[i + 1], ids[i + 2]))
    return total, len(two), len(three)


def selectivity(names, tokenizer):
    """Split symbol names into 'rare' and 'common' by the §7 Fase 4 rule."""
    lens = {}
    for name in names:
        lens[name] = len(tokenizer(name, add_special_tokens=False)["input_ids"])
    unique = {}
    ordered = sorted(names, key=len)
    for name in names:
        # "unique" = no *other* indexed name contains this one as a substring
        unique[name] = not any(
            other != name and name in other for other in ordered
            if len(other) > len(name)
        )
    rare = [n for n in names if lens[n] >= 2 and unique[n]]
    return lens, unique, rare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="path or hub id; omit to skip the BPE half")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = iter_corpus(args.root)
    syms, problems, per_file = parse_corpus(args.root, paths)

    total_bytes = 0
    texts = []
    for rel in paths:
        with open(os.path.join(args.root, rel), "r", encoding="utf-8",
                  errors="replace") as fh:
            t = fh.read()
        texts.append(t)
        total_bytes += len(t.encode("utf-8"))

    by_lang = collections.Counter(s.lang for s in syms)
    by_kind = collections.Counter("%s/%s" % (s.lang, s.kind) for s in syms)
    names = sorted({s.name for s in syms})

    report = {
        "files": len(paths),
        "bytes": total_bytes,
        "symbols": len(syms),
        "distinct_names": len(names),
        "symbols_by_lang": dict(by_lang),
        "symbols_by_kind": dict(sorted(by_kind.items())),
        "parse_problems": len(problems),
        "parse_problem_sample": problems[:25],
        "files_with_no_symbol": [p for p in paths if per_file.get(p, 0) == 0],
    }

    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        n_tok, n2, n3 = bpe_stats(tok, texts)
        lens, unique, rare = selectivity(names, tok)
        hist = collections.Counter(min(lens[n], 8) for n in names)
        report["bpe"] = {
            "tokenizer": args.tokenizer,
            "tokens": n_tok,
            "bytes_per_token": round(total_bytes / n_tok, 3),
            "distinct_2grams": n2,
            "distinct_3grams": n3,
            "sliding_rows_2plus3": n2 + n3,
            "name_token_length_hist": {str(k): v for k, v in sorted(hist.items())},
            "names_ge_2_tokens": sum(1 for n in names if lens[n] >= 2),
            "names_unique_substring": sum(1 for n in names if unique[n]),
            "rare_keys": len(rare),
            "rare_key_sample": rare[:20],
            "common_key_sample": sorted(
                n for n in names if lens[n] == 1 or not unique[n])[:20],
        }

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("parse_problem_sample",
                                   "files_with_no_symbol")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
