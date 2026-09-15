// The geometry the Monitor's charts share (charts.tsx), kept pure so it is
// testable without a DOM: stacking, where a line breaks, which scrape the
// pointer is nearest, where a latency falls along histogram buckets, and
// where a tooltip opens.

export type Values = (number | null | undefined)[];

/** Each series' top at each index: its own value, or with `stacked` the sum up to it — unknown where every series is. */
export function stackTops(series: Values[], length: number, stacked: boolean): (number | null)[][] {
  const tops = series.map(() => new Array<number | null>(length).fill(null));
  for (let i = 0; i < length; i++) {
    const known = series.some((s) => s[i] !== null && s[i] !== undefined);
    let acc = 0;
    series.forEach((s, k) => {
      const v = s[i] ?? null;
      acc += v ?? 0;
      tops[k][i] = stacked ? (known ? acc : null) : v;
    });
  }
  return tops;
}

/** The runs of consecutive known indices from `start`: a line breaks where a value is unknown. */
export function knownRuns(values: Values, start = 0): number[][] {
  const runs: number[][] = [];
  let run: number[] = [];
  for (let i = start; i < values.length; i++) {
    if (values[i] === null || values[i] === undefined) {
      if (run.length) runs.push(run);
      run = [];
    } else run.push(i);
  }
  if (run.length) runs.push(run);
  return runs;
}

/** The index among `candidates` whose time is closest to `t`, or null without candidates. */
export function nearestIndex(times: number[], candidates: number[], t: number): number | null {
  let best: number | null = null;
  for (const i of candidates) if (best === null || Math.abs(times[i] - t) < Math.abs(times[best] - t)) best = i;
  return best;
}

/**
 * Where `value` falls along a histogram's buckets, in bucket widths from the
 * left: 5.75 is three quarters into the sixth. Inside +Inf it sits mid-bucket;
 * beyond every bound, at the right end.
 */
export function bucketOffset(bounds: number[], value: number): number {
  const i = bounds.findIndex((b) => value <= b);
  if (i === -1) return bounds.length;
  const lower = i === 0 ? 0 : bounds[i - 1];
  const upper = bounds[i];
  const frac = upper === Infinity ? 0.5 : (value - lower) / (upper - lower || 1);
  return i + Math.min(1, Math.max(0, frac));
}

/** A tooltip's left edge: `gap` right of `anchor`, or left of it when it would overflow `width`; never before 0. */
export function tipLeft(anchor: number, width: number, tipWidth: number, gap = 14): number {
  const left = anchor + gap + tipWidth > width ? anchor - gap - tipWidth : anchor + gap;
  return Math.max(0, left);
}
