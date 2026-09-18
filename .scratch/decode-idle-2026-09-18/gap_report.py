"""Where a decode round's device timeline is idle, and what it is idle between.

  python .scratch/decode-idle-2026-09-18/gap_report.py <capture.sqlite> [...]

Each capture is an Nsight Systems run of the server exported to sqlite:

  nsys profile --trace=cuda --sample=none --cpuctxsw=none --delay 60 --duration 8 \
       -o decode-B1lane <server ...>
  nsys export --type sqlite -o decode-B1lane.sqlite decode-B1lane.nsys-rep

The device timeline is one stream, so activities do not overlap and idle is
simply the sum of the holes between them. Every hole is attributed to the pair
it sits between, which is what turns "the GPU is 5% idle" into a call site.

With `--cuda-graph-trace=node` a replay is dissolved into its nodes, which is
how to ask whether a memcpy or memset node costs more than its own duration;
without it a replay is one activity, which is the lower-overhead way to size
the gaps *around* rounds. Node tracing roughly doubles the measured idle, so
the round-boundary numbers come from the graph-level capture.
"""
import sqlite3
import sys

BIG_GAP_NS = 20_000  # 20 us: above this is a host-side stall, below it launch latency


def activities(path):
    db = sqlite3.connect(path)
    names = {r[0]: r[1] for r in db.execute("select id, value from StringIds")}

    def table(query, shape):
        try:
            return [shape(row) for row in db.execute(query)]
        except sqlite3.OperationalError:
            return []

    return sorted(
        table("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL",
              lambda r: (r[0], r[1], names.get(r[2], "?")))
        + table("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY",
                lambda r: (r[0], r[1], "copy %d B" % (r[2] or 0)))
        + table("select start, end from CUPTI_ACTIVITY_KIND_MEMSET",
                lambda r: (r[0], r[1], "memset"))
        + table("select start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
                lambda r: (r[0], r[1], "GRAPH REPLAY")))


def report(path):
    events = activities(path)
    if not events:
        print("%s: no CUDA activity" % path)
        return
    span = events[-1][1] - events[0][0]
    idle = 0
    pairs = {}
    for i in range(1, len(events)):
        previous_end, gap_start = events[i - 1][1], events[i][0]
        if gap_start <= previous_end:
            continue
        gap = gap_start - previous_end
        idle += gap
        if gap >= BIG_GAP_NS:
            slot = pairs.setdefault((events[i - 1][2][:40], events[i][2][:40]), [0, 0])
            slot[0] += 1
            slot[1] += gap
    rounds = sum(1 for _, _, name in events if name == "GRAPH REPLAY")
    graph = sum(e - s for s, e, name in events if name == "GRAPH REPLAY")
    print("== %s ==" % path)
    print("  window %.3f s   idle %.1f ms (%.2f%%)" % (span / 1e9, idle / 1e6, 100 * idle / span))
    if rounds:
        print("  rounds %d   replay mean %.2f ms   idle per round %.0f us"
              % (rounds, graph / rounds / 1e6, idle / rounds / 1e3))
    print("  %-40s %-40s %6s %9s %8s" % ("after", "before", "count", "total ms", "mean us"))
    for (after, before), (count, total) in sorted(pairs.items(), key=lambda kv: -kv[1][1])[:8]:
        print("  %-40s %-40s %6d %9.1f %8.1f"
              % (after, before, count, total / 1e6, total / count / 1e3))
    print()


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        report(argument)
