"""GitHub #92: condense an `ncu --csv` dump into one row per kernel geometry.

  python .scratch/issue-92/ncu_summary.py .scratch/issue-92/ncu-t1024.csv

Answers the question the wall-clock decomposition cannot: a kernel that is
executing still may not be filling the machine. `launch__grid_size` against
the card's 170 SMs says whether the GEMM even has a block for every SM;
`sm__throughput` and the tensor-pipe utilization say how hard the ones it has
are working.
"""
import collections
import csv
import io
import re
import sys

SMS = 170  # RTX 5090 (GB202)


def geometry(name):
    m = re.search(r"(nvfp4_w4a4_\w+?_kernel)<\w*GemvGeometry<\s*(\d+),\s*(\d+)", name)
    if m:
        return "%s<%s,%s>" % (m.group(1), m.group(2), m.group(3))
    return name.split("(")[0][:60]


def main(path):
    rows = list(csv.reader(io.open(path, encoding="utf-8", errors="replace")))
    header = None
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if not r or len(r) < 3:
            continue
        if header is None:
            if "Metric Name" in r:
                header = r
            continue
        rec = dict(zip(header, r))
        name = rec.get("Kernel Name") or ""
        metric = rec.get("Metric Name") or ""
        value = (rec.get("Metric Value") or "").replace(",", "")
        if not name or not metric:
            continue
        try:
            data[geometry(name)][metric].append(float(value))
        except ValueError:
            pass

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("%-44s %7s %7s %7s %7s %9s" % ("kernel", "grid", "waves", "SM%", "tensor%", "us"))
    order = sorted(data, key=lambda k: -sum(data[k].get("gpu__time_duration.sum", [0])))
    for key in order:
        m = data[key]
        grid = mean(m.get("launch__grid_size", [float("nan")]))
        print("%-44s %7.0f %7.2f %7.1f %7.1f %9.1f" % (
            key, grid, grid / SMS,
            mean(m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", [float("nan")])),
            mean(m.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                       [float("nan")])),
            mean(m.get("gpu__time_duration.sum", [float("nan")])) / 1e3,
        ))
    print("\nwaves = grid blocks / %d SMs. Below 1.00 the kernel cannot occupy the card." % SMS)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
