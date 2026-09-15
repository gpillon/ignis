"""GitHub #90 / ADR 0017: the verdict over ab.sh's legs.

Inference critical path (zero repeatable regression): g3's C=1 and C=4
aggregate decode tok/s and ITL p50/p95 (step timings as HTTP observes them),
plus g4's per-class delivered tok/s. A metric regresses *repeatably* when
every enabled launch is worse than every disabled launch -- the ranges do not
overlap -- whatever the size.

HTTP/telemetry plane (the 1% budget, s15 only): pooled per-class median TTFT
<= 1.01x the disabled baseline, pooled per-class delivered tok/s >= 0.99x.

s1 is a stress diagnostic: reported, never a verdict.

usage: python verdict.py [DIR]   (default: this script's directory)
"""

import glob
import json
import os
import statistics
import sys

HIGHER, LOWER = "higher", "lower"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def legs(root, kind, suffix):
    return [load(p) for p in sorted(glob.glob(os.path.join(root, f"{kind}-[0-9]-{suffix}.json")))]


def g3_values(record):
    return {
        "C=1 tok/s": (record["c1"]["aggregate_tok_s"], HIGHER),
        "C=4 tok/s": (record["c4"]["aggregate_tok_s"], HIGHER),
        "ITL p50 ms": (record["itl"].get("p50_ms"), LOWER),
        "ITL p95 ms": (record["itl"].get("p95_ms"), LOWER),
    }


def class_requests(record, cls):
    return [m for m in record["run"]["metrics"] if m["class"] == cls and m["ok"] and m["n_tokens"] > 0]


def delivered_tok_s(requests):
    """Total tokens over total decode time: bench's throughput-weighted rule."""
    decode_ms = sum(max(m["total_ms"] - m["ttft_ms"], 0.0) for m in requests)
    return sum(m["n_tokens"] for m in requests) * 1000.0 / decode_ms if decode_ms else 0.0


def g4_values(record):
    values = {}
    for cls in sorted({m["class"] for m in record["run"]["metrics"]}):
        requests = class_requests(record, cls)
        values[f"g4 {cls} tok/s"] = (delivered_tok_s(requests), HIGHER)
    return values


def repeatable_regression(base, candidate, direction):
    if not base or not candidate or None in base or None in candidate:
        return None
    if direction == HIGHER:
        return max(candidate) < min(base)
    return min(candidate) > max(base)


def critical_path(root, kind):
    rows, failed = [], False
    for suffix, extract in (("g3", g3_values), ("g4", g4_values)):
        base, cand = legs(root, "off", suffix), legs(root, kind, suffix)
        if not base or not cand:
            rows.append(f"  {suffix}: missing legs (off={len(base)}, {kind}={len(cand)})")
            failed = True
            continue
        for metric in extract(base[0]):
            direction = extract(base[0])[metric][1]
            b = [extract(r)[metric][0] for r in base]
            c = [extract(r).get(metric, (None, direction))[0] for r in cand]
            verdict = repeatable_regression(b, c, direction)
            failed |= verdict is not False
            ratio = statistics.mean(c) / statistics.mean(b) if None not in b + c and statistics.mean(b) else float("nan")
            label = {True: "REPEATABLE REGRESSION", False: "ok", None: "unreadable"}[verdict]
            rows.append(f"  {metric:<16} off={fmt(b)} {kind}={fmt(c)} mean ratio {ratio:.4f}  {label}")
    return rows, failed


def http_plane(root, kind):
    base, cand = legs(root, "off", "g4"), legs(root, kind, "g4")
    rows, failed = [], False
    if not base or not cand:
        return [f"  missing g4 legs (off={len(base)}, {kind}={len(cand)})"], True
    for cls in sorted({m["class"] for r in base for m in r["run"]["metrics"]}):
        b = [m for r in base for m in class_requests(r, cls)]
        c = [m for r in cand for m in class_requests(r, cls)]
        ttft = statistics.median(m["ttft_ms"] for m in c) / statistics.median(m["ttft_ms"] for m in b)
        tok = delivered_tok_s(c) / delivered_tok_s(b)
        ok_ttft, ok_tok = ttft <= 1.01, tok >= 0.99
        failed |= not (ok_ttft and ok_tok)
        rows.append(
            f"  {cls:<5} TTFT median ratio {ttft:.4f} ({'PASS' if ok_ttft else 'FAIL'} <= 1.01)"
            f"   delivered tok/s ratio {tok:.4f} ({'PASS' if ok_tok else 'FAIL'} >= 0.99)"
        )
    return rows, failed


def fmt(values):
    return "[" + ", ".join("-" if v is None else f"{v:.2f}" for v in values) + "]"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    overall = False
    for kind in ("on", "s15"):
        rows, failed = critical_path(root, kind)
        overall |= failed
        print(f"critical path, off vs {kind}: {'FAIL' if failed else 'PASS'}")
        print("\n".join(rows))
    rows, failed = http_plane(root, "s15")
    overall |= failed
    print(f"HTTP/telemetry plane, off vs s15 (1% budget): {'FAIL' if failed else 'PASS'}")
    print("\n".join(rows))
    if legs(root, "s1", "g3"):
        print("stress diagnostic, off vs s1 (no verdict):")
        print("\n".join(critical_path(root, "s1")[0]))
        print("\n".join(http_plane(root, "s1")[0]))
    print(f"verdict: {'FAIL' if overall else 'PASS'}")
    return 1 if overall else 0


if __name__ == "__main__":
    sys.exit(main())
