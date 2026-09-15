import type { Histogram } from "./snapshot.ts";

// Latency from a fixed-bucket histogram, the way Prometheus's
// `histogram_quantile` reads one: the rank is placed in its bucket and
// interpolated linearly between the bucket's bounds. These are
// approximations whose resolution is the bucket width.

/** The `q` quantile in seconds, or null without observations. */
export function histogramQuantile(q: number, h: Histogram): number | null {
  const total = h.cumulative.at(-1) ?? 0;
  if (total <= 0) return null;
  const rank = q * total;
  const bucket = h.cumulative.findIndex((c) => c >= rank);
  if (bucket === -1) return null;
  const upper = h.bounds[bucket];
  if (upper === Infinity) {
    // Beyond the last finite bound all we know is "more than that".
    return h.bounds.length > 1 ? h.bounds[h.bounds.length - 2] : null;
  }
  const lower = bucket === 0 ? 0 : h.bounds[bucket - 1];
  const below = bucket === 0 ? 0 : h.cumulative[bucket - 1];
  const inBucket = h.cumulative[bucket] - below;
  if (inBucket <= 0) return upper;
  return lower + ((upper - lower) * (rank - below)) / inBucket;
}

/** Observations per bucket rather than cumulative. */
export function bucketCounts(h: Histogram): number[] {
  return h.cumulative.map((c, i) => c - (i === 0 ? 0 : h.cumulative[i - 1]));
}

/** The mean observation in seconds, or null without observations. */
export function histogramMean(h: Histogram): number | null {
  return h.count > 0 ? h.sum / h.count : null;
}

/** What `later` observed since `earlier`. A count that went down means the server restarted: then `later` is all of it. */
export function histogramIncrease(earlier: Histogram, later: Histogram): Histogram {
  const sameBuckets = earlier.bounds.length === later.bounds.length && earlier.bounds.every((b, i) => b === later.bounds[i]);
  if (!sameBuckets || later.count < earlier.count) return later;
  return {
    bounds: later.bounds,
    cumulative: later.cumulative.map((c, i) => Math.max(0, c - earlier.cumulative[i])),
    sum: Math.max(0, later.sum - earlier.sum),
    count: later.count - earlier.count,
  };
}

/** Two histograms over the same buckets, added. */
export function histogramSum(a: Histogram, b: Histogram): Histogram {
  return {
    bounds: b.bounds,
    cumulative: b.cumulative.map((c, i) => c + (a.cumulative[i] ?? 0)),
    sum: a.sum + b.sum,
    count: a.count + b.count,
  };
}
