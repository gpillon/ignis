"""Pool TTFT records from independent launches of one engine (ADR 0021).

    python scripts/ttft-pool.py --out pooled.json launch1.json launch2.json

Each input is an `ignis-bench ttft` record of the same engine, label and
session, one per process launch. The output is one record of the same shape:
every cell's samples from every launch, re-indexed, and the median taken over
all of them, so `ignis-bench g2` judges the pooled result rather than one
launch pair. A cell must be present, measured and cold in every launch.
"""

import argparse
import json
import statistics
import sys


def key(cell):
    image = cell.get("image")
    return (cell["prompt_tokens"], image and image["sha256"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("records", nargs="+")
    args = parser.parse_args()
    records = [json.load(open(path, encoding="utf-8")) for path in args.records]
    first = records[0]
    for path, record in zip(args.records, records):
        for field in ("session", "label", "engine", "artifact"):
            if record[field] != first[field]:
                sys.exit(f"{path}: {field} {record[field]!r} is not {first[field]!r}")
        if [key(c) for c in record["cells"]] != [key(c) for c in first["cells"]]:
            sys.exit(f"{path}: its cells are not the first record's")
    pooled_cells = []
    for index, cell in enumerate(first["cells"]):
        samples = []
        for path, record in zip(args.records, records):
            launch_cell = record["cells"][index]
            if launch_cell.get("error"):
                sys.exit(f"{path}: cell {launch_cell['prompt_tokens']} failed: {launch_cell['error']}")
            samples.extend(launch_cell["samples"])
        samples = [dict(sample, index=i) for i, sample in enumerate(samples)]
        pooled = dict(cell, samples=samples, median_ttft_ms=statistics.median(s["ttft_ms"] for s in samples))
        pooled_cells.append(pooled)
    out = dict(first, cells=pooled_cells, profile=f"{first['profile']} (pooled over {len(records)} launches)")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for cell in pooled_cells:
        print(f"  {cell['prompt_tokens']:>7} tokens  pooled median {cell['median_ttft_ms']:9.1f} ms over {len(cell['samples'])} samples")


if __name__ == "__main__":
    main()
