"""Write one `locate` question set (spec 18 § Phase A, The sets).

    python tools/locate-sets/generate.py --seed 20261010 --out .scratch/locate/A

240 questions -- 80 logs, 80 records, 80 prose -- into `<out>/manifest.json`,
deterministic per `--seed` (and per the sets named by `--exclude`, whose
HotpotQA questions this one does not reuse). Each question row carries the
`state` exactly as a caller would send it, the `instruction`, the `targets`
(segment indices; empty when absent), its `family`, `split` and `absent`
flag, and its segment count. `crates/server/tests/attention_head_locate_gpu.rs`
reads the manifest; `score.py` scores what that harness dumps.
"""

import argparse
import json
import os
import random

import logs
import prose
import records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=80)
    ap.add_argument("--cache", default=os.path.join(".scratch", "locate", "hotpot"),
                    help="where the HotpotQA dev file is downloaded")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="set directories whose HotpotQA questions this set must not reuse")
    args = ap.parse_args()

    exclude = set()
    for other in args.exclude:
        with open(os.path.join(other, "manifest.json"), encoding="utf-8") as f:
            exclude |= {q["event"] for q in json.load(f)["questions"] if q["family"] == "prose"}

    # One generator per family, each seeded from the set's seed, so adding a
    # family or changing one's draws never moves another's questions.
    n = args.per_family
    questions = (logs.generate(random.Random(f"{args.seed}/logs"), n)
                 + records.generate(random.Random(f"{args.seed}/records"), n)
                 + prose.generate(random.Random(f"{args.seed}/prose"), n, args.cache, exclude))
    os.makedirs(args.out, exist_ok=True)
    manifest = {"seed": args.seed, "per_family": n, "exclude": sorted(exclude),
                "questions": questions}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    summary = {}
    for q in questions:
        key = (q["family"], q["split"], q["absent"])
        summary[key] = summary.get(key, 0) + 1
    for key in sorted(summary):
        print(f"  {key[0]:8} {key[1]:10} {'absent' if key[2] else 'present':7} {summary[key]}")
    print(f"wrote {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
