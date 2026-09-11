"""GitHub #92 criterion 1: turn the two passes' output into the decomposition.

  python .scratch/issue-92/analyze.py

Reads .scratch/issue-92/pass-a-clean.txt (the uninstrumented chunk-width
sweep) and .scratch/issue-92/chunks.jsonl (the leaf's per-chunk event
records) and prints:

  * the least-squares fit of total span wall time on chunk count -- the fixed
    per-chunk-boundary cost, read off a run with no instrumentation in it;
  * per chunk width, the mean split of a chunk's wall time into host enqueue,
    synchronize stall, boundary device idle, layer compute and inter-layer
    device idle.

Warm-up spans (the first span measured at each width) are dropped.
"""
import codecs
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read_text(path):
    """The log as text. PowerShell 5.1's Tee-Object writes UTF-16 LE."""
    raw = open(path, "rb").read()
    if raw[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return raw.decode("utf-16")
    return raw.decode("utf-8", "replace")


def read_sweep(path):
    """(width, chunks, mean_ms, min_ms) from the test's printed table."""
    rows = []
    if not os.path.exists(path):
        return rows
    for line in read_text(path).splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))))
    return rows


def fit(rows, which):
    """wall = fixed * chunks + constant, least squares. `which`: 2 mean, 3 min."""
    n = len(rows)
    if n < 2:
        return None
    xs = [float(r[1]) for r in rows]
    ys = [float(r[which]) for r in rows]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return slope, intercept, r2


def read_chunks(path):
    chunks, layers = [], []
    if not os.path.exists(path):
        return chunks, layers
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        (layers if "layer" in rec else chunks).append(rec)
    return chunks, layers


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    sweep_path = os.path.join(HERE, "pass-a-clean.txt")
    rows = read_sweep(sweep_path)
    print("== pass A, uninstrumented chunk-width sweep (8,192-token span) ==")
    if not rows:
        print("  (no rows parsed from %s)" % sweep_path)
    for width, chunks, m, lo in rows:
        print("  width %5d  chunks %3d  wall mean %8.1f ms  min %8.1f ms  -> %7.2f ms/chunk"
              % (width, chunks, m, lo, m / chunks))
    # Two fits. The all-widths one is reported because it is the obvious
    # thing to compute and it is *wrong*: at 256 and 512 tokens the chunk is
    # too narrow for the GEMM shapes, so per-token device compute itself
    # rises (see layers/token in pass B) and the regression charges that
    # compute to the chunk boundary. The plateau fit uses only the widths
    # whose per-token cost has flattened, where the remaining slope really is
    # a per-boundary cost.
    plateau = [r for r in rows if r[0] >= 1024]
    for label, which, subset, note in (
        ("all widths, mean", 2, rows, "contaminated by narrow-chunk compute loss"),
        ("all widths, min", 3, rows, "contaminated by narrow-chunk compute loss"),
        ("width >= 1024, min", 3, plateau, "per-token cost has plateaued here"),
    ):
        f = fit(subset, which)
        if f:
            slope, intercept, r2 = f
            print("  fit (%s): wall_ms = %.3f * chunks + %.1f   (R^2 = %.4f) -- %s"
                  % (label, slope, intercept, r2, note))
            print("        -> fixed cost per chunk boundary: %.3f ms"
                  " => %.1f ms over the default route's 8 chunks" % (slope, slope * 8))

    chunk_recs, layer_recs = read_chunks(os.path.join(HERE, "chunks.jsonl"))
    print()
    print("== pass B, leaf event decomposition (%d chunk records) ==" % len(chunk_recs))
    if not chunk_recs:
        print("  (no records -- was IGNIS_CHUNK_PROFILE set?)")
        return 0

    # Keyed by (thread, width): `span` and `chunk` are per-thread counters in
    # the leaf, so records from two profiling threads must not be pooled.
    by_width = {}
    for rec in chunk_recs:
        by_width.setdefault((rec.get("thread", 0), rec["chunk_width"]), []).append(rec)

    header = ("  width  chunk_ms  enqueue  sync_stall  entry_gap  embed   layers"
              "  head    layer_gap   gpu_span  layers/token  n")
    print(header)
    for key in sorted(by_width):
        width = key[1]
        recs = by_width[key]
        warm_up_span = min(r["span"] for r in recs)
        recs = [r for r in recs if r["span"] != warm_up_span]
        if not recs:
            continue
        wall = [r["cpu_enqueue_ms"] + r["sync_ms"] for r in recs]
        per_token = mean([r["layers_ms"] / r["chunk_tokens"] for r in recs])
        print("  %5d  %8.2f  %7.2f  %10.2f  %9.2f  %6.2f  %7.2f  %6.2f  %9.2f  %9.2f  %12.4f  %d"
              % (width, mean(wall), mean([r["cpu_enqueue_ms"] for r in recs]),
                 mean([r["sync_ms"] for r in recs]), mean([r["entry_gap_ms"] for r in recs]),
                 mean([r["embed_ms"] for r in recs]), mean([r["layers_ms"] for r in recs]),
                 mean([r["head_ms"] for r in recs]), mean([r["layer_gap_ms"] for r in recs]),
                 mean([r["gpu_span_ms"] for r in recs]), per_token, len(recs)))

    print()
    print("  Columns: chunk_ms = host wall for the chunk (enqueue + sync_stall).")
    print("           entry_gap = device idle between the previous chunk's last op and this")
    print("                       chunk's first -- the bubble the forced sync opens.")
    print("           layers = sum of the 64 layer bodies' own device spans (compute).")
    print("           layer_gap = gpu_span - embed - layers - head: device idle between")
    print("                       layer bodies, i.e. launch latency the host did not hide.")
    print("           layers/token = device compute per token. Flat above 1,024 tokens; the")
    print("                       rise at 256/512 is lost GEMM efficiency, not synchronization.")
    print("           Idle *inside* a layer body is invisible here -- it lands in `layers`.")
    print("           nsys_gap_report.py measures that residue directly.")

    if layer_recs:
        print()
        print("== per-layer device spans (pass B, one representative chunk) ==")
        widths = {}
        for rec in layer_recs:
            widths.setdefault((rec["span"], rec["chunk_offset"]), []).append(rec)
        key = sorted(widths)[len(widths) // 2]
        sample = sorted(widths[key], key=lambda r: r["layer"])
        busy = sum(r["layer_ms"] for r in sample)
        gaps = sum(r["gap_before_ms"] for r in sample)
        print("  span %d, chunk offset %d: %d layers, %.2f ms busy, %.3f ms in gaps before layers"
              % (key[0], key[1], len(sample), busy, gaps))
        slowest = sorted(sample, key=lambda r: -r["layer_ms"])[:5]
        print("  slowest layers: " + ", ".join("L%d %.2fms" % (r["layer"], r["layer_ms"])
                                               for r in slowest))
        print("  widest gap before a layer: %.3f ms" % max(r["gap_before_ms"] for r in sample))
    return 0


if __name__ == "__main__":
    sys.exit(main())
