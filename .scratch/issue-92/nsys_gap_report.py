"""GitHub #92: the independent cross-check on the leaf's own event records.

Reads the Nsight Systems capture of a width-1024 run and, over the prefill
region only, reports how much of the device timeline is inside a kernel and
how much is idle -- including the idle *inside* a layer body, which the
leaf's per-layer events cannot see.

  python .scratch/issue-92/nsys_gap_report.py .scratch/issue-92/nsys-w1024.sqlite
"""
import sqlite3
import sys

MS = 1e6  # nsys stores nanoseconds


def main(path):
    db = sqlite3.connect(path)
    rows = db.execute(
        "select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    if not rows:
        print("no kernels in the capture")
        return 1

    # The prefill region: the run's last four spans are the width's warm-up
    # plus three timed reps, and they are the only multi-second stretch of
    # back-to-back kernels in the process. Cut the region at the widest idle
    # gap that precedes them (model load / materialize sits before it).
    gaps = [(rows[i + 1][0] - rows[i][1], i) for i in range(len(rows) - 1)]
    widest, at = max(gaps)
    region = rows[at + 1:]
    span_ns = region[-1][1] - region[0][0]
    busy_ns = sum(end - start for start, end in region)
    # Idle inside the region: total minus the union of kernel intervals
    # (a single stream, so intervals do not overlap -- assert that).
    overlaps = sum(1 for i in range(len(region) - 1) if region[i + 1][0] < region[i][1])
    idle_ns = span_ns - busy_ns
    inter = [region[i + 1][0] - region[i][1] for i in range(len(region) - 1)]
    inter = [g for g in inter if g > 0]
    inter.sort(reverse=True)

    print("prefill region: %d kernels, %.1f ms wall" % (len(region), span_ns / MS))
    print("  device busy : %9.1f ms  (%.2f%%)" % (busy_ns / MS, 100.0 * busy_ns / span_ns))
    print("  device idle : %9.1f ms  (%.2f%%)" % (idle_ns / MS, 100.0 * idle_ns / span_ns))
    print("  overlapping kernel pairs: %d (expect 0 -- one stream)" % overlaps)
    print("  mean gap between consecutive kernels: %.1f us over %d gaps"
          % ((sum(inter) / len(inter)) / 1e3, len(inter)))
    # Gaps wider than 100 us are not launch latency: they are the boundaries
    # between the run's four spans (sequence allocation, the host-side BF16
    # logits promotion). Separating them keeps the launch-latency figure from
    # absorbing them.
    small = [g for g in inter if g <= 100_000]
    large = [g for g in inter if g > 100_000]
    print("  idle in gaps <= 100 us: %8.1f ms over %d gaps (mean %.1f us) -- launch latency"
          % (sum(small) / MS, len(small), (sum(small) / len(small)) / 1e3))
    print("  idle in gaps >  100 us: %8.1f ms over %d gaps -- span boundaries, not per-kernel"
          % (sum(large) / MS, len(large)))
    print("  widest 10 gaps (us): " + ", ".join("%.1f" % (g / 1e3) for g in inter[:10]))
    print("  region cut at a %.1f ms idle gap (model load sits before it)" % (widest / MS))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ".scratch/issue-92/nsys-w1024.sqlite"))
