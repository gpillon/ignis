import { histogramIncrease, histogramSum } from "./quantile.ts";
import { counterValues, type Histogram, type Snapshot } from "./snapshot.ts";

// The rolling in-memory history of scrapes (GitHub #165) and what is derived
// from it: counter rates, gains over a window, and window histograms. When a
// scrape finds the counters gone down, ignis restarted: every counter's gain
// across that pair then counts from zero, as Prometheus's `increase` does, so
// no rate is ever negative and none carries the old process's totals.

/** One scrape: when it ended, what it read, and how long the request took. */
export type Point = { at: number; snap: Snapshot; scrapeMs: number };

/** How far back the history reaches. */
export const HISTORY_MS = 15 * 60_000;

/** Reads one counter out of a scrape. */
export type CounterPick = (s: Snapshot) => number | null;

/** `points` with `point` added, anything older than `windowMs` before it dropped. */
export function append(points: Point[], point: Point, windowMs = HISTORY_MS): Point[] {
  const from = point.at - windowMs;
  return [...points.filter((p) => p.at >= from && p.at < point.at), point];
}

/** A counter's gain between two readings; after a reset, the later reading. */
export function increase(earlier: number | null, later: number | null): number | null {
  if (earlier === null || later === null) return null;
  return later < earlier ? later : later - earlier;
}

/** Whether any counter went down between two scrapes: the server restarted. */
export function isRestart(earlier: Snapshot, later: Snapshot): boolean {
  const before = counterValues(earlier);
  return counterValues(later).some((v, i) => v !== null && before[i] !== null && v < before[i]!);
}

/** A counter's gain from one scrape to the next, counting from zero across a restart. */
function gain(a: Point, b: Point, pick: CounterPick): number | null {
  return isRestart(a.snap, b.snap) ? pick(b.snap) : increase(pick(a.snap), pick(b.snap));
}

/** Per-second rates, one per point; the first point has none. */
export function rateSeries(points: Point[], pick: CounterPick): (number | null)[] {
  return points.map((p, i) => {
    if (i === 0) return null;
    const g = gain(points[i - 1], p, pick);
    const seconds = (p.at - points[i - 1].at) / 1000;
    return g === null || seconds <= 0 ? null : g / seconds;
  });
}

/**
 * Per-second rates, one per point, each over the scrapes in the `spanMs`
 * before it — smoother than scrape-to-scrape, for counters that move in steps.
 */
export function rollingRate(points: Point[], pick: CounterPick, spanMs: number): (number | null)[] {
  // Cumulative gains, so each span's gain is one subtraction.
  const cumulative: number[] = [0];
  for (let i = 1; i < points.length; i++) cumulative.push(cumulative[i - 1] + (gain(points[i - 1], points[i], pick) ?? 0));
  let start = 0;
  return points.map((p, i) => {
    while (points[start].at < p.at - spanMs) start++;
    const elapsed = p.at - points[start].at;
    if (start === i || elapsed <= 0 || pick(p.snap) === null) return null;
    return ((cumulative[i] - cumulative[start]) * 1000) / elapsed;
  });
}

/** The pairs of successive points whose earlier point is at or after `since`. */
function pairsSince(points: Point[], since: number): [Point, Point][] {
  const pairs: [Point, Point][] = [];
  for (let i = 1; i < points.length; i++) if (points[i - 1].at >= since) pairs.push([points[i - 1], points[i]]);
  return pairs;
}

/** A counter's total gain over the scrapes since `since`, or null with nothing to compare. */
export function increaseOver(points: Point[], pick: CounterPick, since: number): number | null {
  let total: number | null = null;
  for (const [a, b] of pairsSince(points, since)) {
    const g = gain(a, b, pick);
    if (g !== null) total = (total ?? 0) + g;
  }
  return total;
}

/** The observations a histogram gained over the scrapes since `since`. */
export function windowHistogram(points: Point[], pick: (s: Snapshot) => Histogram | null, since: number): Histogram | null {
  let total: Histogram | null = null;
  for (const [a, b] of pairsSince(points, since)) {
    const ha = pick(a.snap);
    const hb = pick(b.snap);
    if (!ha || !hb) continue;
    const g = isRestart(a.snap, b.snap) ? hb : histogramIncrease(ha, hb);
    total = total ? histogramSum(total, g) : g;
  }
  return total;
}
