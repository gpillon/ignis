"""Summarize an `nvidia-smi --query-gpu ... -l 1` capture over a time window.

The KV-format A/B (docs/agents/testing.md) samples GPU telemetry for a whole
`ignis-bench g3` run and then asks what the card was doing during one cell of
it. This reduces such a capture to one line.

The capture is expected to have been produced by:

    nvidia-smi --format=csv,noheader -l 1 \
      --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu

`nvidia-smi` stamps local time, so `--from` / `--to` are local `HH:MM:SS` too.
`--last <seconds>` selects the tail instead, which is how the ITL cell is
usually picked out: it is the last cell a `g3` run measures.

Usage:
    python scripts/gpu-telemetry-summary.py <capture.csv> [--label NAME]
        [--from HH:MM:SS --to HH:MM:SS | --last SECONDS]
"""

import argparse
import csv
import sys
from datetime import datetime


FIELDS = ("util %", "mem-util %", "power W", "sm MHz", "mem MHz", "temp C")


def read(path):
    """Every parseable sample as (datetime, util, mem_util, watts, sm, mem, temp)."""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 7:
                continue
            try:
                stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            except ValueError:
                # nvidia-smi writes warnings and a header-ish first line on some
                # drivers; a row that is not a sample is simply not one.
                continue
            try:
                values = [
                    int(row[1].strip().rstrip(" %")),
                    int(row[2].strip().rstrip(" %")),
                    float(row[3].strip().rstrip(" W")),
                    int(row[4].strip().rstrip(" MHz")),
                    int(row[5].strip().rstrip(" MHz")),
                    int(row[6].strip()),
                ]
            except ValueError:
                continue
            rows.append((stamp, *values))
    return rows


def select(rows, start, end, last):
    if last is not None:
        cutoff = rows[-1][0].timestamp() - last
        return [r for r in rows if r[0].timestamp() >= cutoff]
    if start or end:
        lo = start or "00:00:00"
        hi = end or "23:59:59"
        return [r for r in rows if lo <= r[0].strftime("%H:%M:%S") <= hi]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--label", default=None, help="name for the summary line")
    parser.add_argument("--from", dest="start", default=None, metavar="HH:MM:SS")
    parser.add_argument("--to", dest="end", default=None, metavar="HH:MM:SS")
    parser.add_argument("--last", type=float, default=None, metavar="SECONDS")
    args = parser.parse_args()

    if args.last is not None and (args.start or args.end):
        parser.error("--last and --from/--to select the window two different ways")

    rows = read(args.capture)
    if not rows:
        sys.exit(f"{args.capture}: no samples parsed")
    sel = select(rows, args.start, args.end, args.last)
    if not sel:
        sys.exit(
            f"{args.capture}: no samples in the requested window "
            f"(capture spans {rows[0][0]:%H:%M:%S} to {rows[-1][0]:%H:%M:%S})"
        )

    label = args.label or args.capture
    column = lambda i: [r[i] for r in sel]  # noqa: E731 - one expression, read once
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    print(
        f"{label:<34} n={len(sel):>3}  "
        f"{sel[0][0]:%H:%M:%S}-{sel[-1][0]:%H:%M:%S}  "
        f"util {mean(column(1)):3.0f}%  "
        f"mem-util {mean(column(2)):3.0f}%  "
        f"power {mean(column(3)):6.1f} W (max {max(column(3)):6.1f})  "
        f"sm {mean(column(4)):5.0f} MHz  "
        f"temp {mean(column(6)):2.0f} C"
    )


if __name__ == "__main__":
    main()
