import { formatCount, formatSeconds, formatShare, formatWindow } from "./format.ts";
import { type CounterPick, increaseOver, type Point, rollingRate, windowHistogram } from "./history.ts";
import { histogramMean, histogramQuantile } from "./quantile.ts";
import { type Histogram, REJECT_REASONS, type RejectReason, type Snapshot } from "./snapshot.ts";

// Everything the Monitor shows, derived from the scrape history for one
// window: counter gains and rates, scheduler load, latency quantiles and
// their trend, and a plain-words verdict on how the server is doing.

/** Rates are averaged over this span, since counters move in steps. */
export const RATE_SPAN_MS = 30_000;
/** Latency trends are quantiles over this sliding span. */
export const TREND_SPAN_MS = 60_000;

export type Values = (number | null)[];

export type Counter = {
  /** Since the server started. */
  total: number | null;
  /** Gained over the window. */
  window: number | null;
  /** The window's gain per minute. */
  perMin: number | null;
  /** Per minute at each scrape, over RATE_SPAN_MS. */
  perMinSeries: Values;
};

export type Latency = {
  /** The observations made over the window. */
  window: Histogram | null;
  count: number;
  p50: number | null;
  p95: number | null;
  p99: number | null;
  mean: number | null;
  lifetimeP95: number | null;
  lifetimeCount: number | null;
  trendP50: Values;
  trendP95: Values;
};

export type HealthLevel = "idle" | "healthy" | "busy" | "saturated";
/** The verdict: its level, what it rests on in a phrase, and the facts behind it. */
export type Health = { level: HealthLevel; summary: string; notes: string[] };

export type Dashboard = {
  now: number;
  from: number;
  windowMs: number;
  /** The first scrape inside the window: series drawn from here. */
  visibleStart: number;
  times: number[];
  version: string | null;
  running: number | null;
  waiting: number | null;
  runningSeries: Values;
  waitingSeries: Values;
  runningPeak: number | null;
  waitingPeak: number | null;
  tokens: Counter & { perSec: number | null; perSecSeries: Values; perRequest: number | null };
  accepted: Counter;
  completed: Counter;
  cancelled: Counter;
  rejected: Counter & { byReason: Record<RejectReason, { total: number | null; window: number | null }> };
  prefix: Counter & { perSec: number | null; perSecSeries: Values };
  evictions: Counter;
  ttft: Latency;
  duration: Latency;
  scrapeMsSeries: Values;
  health: Health;
};

const totalRejected: CounterPick = (s) => {
  const known = REJECT_REASONS.map((r) => s.rejected[r]).filter((v): v is number => v !== null);
  return known.length ? known.reduce((a, b) => a + b, 0) : null;
};

export function deriveDashboard(points: Point[], windowMs: number): Dashboard | null {
  const lastPoint = points.at(-1);
  if (!lastPoint) return null;
  const now = lastPoint.at;
  const from = now - windowMs;
  const visibleStart = Math.max(0, points.findIndex((p) => p.at >= from));
  const coveredMs = now - points[visibleStart].at;
  const last = lastPoint.snap;

  const counter = (pick: CounterPick): Counter => {
    const window = increaseOver(points, pick, from);
    return {
      total: pick(last),
      window,
      perMin: window !== null && coveredMs > 0 ? (window * 60_000) / coveredMs : null,
      perMinSeries: rollingRate(points, pick, RATE_SPAN_MS).map((r) => (r === null ? null : r * 60)),
    };
  };

  const latency = (pick: (s: Snapshot) => Histogram | null): Latency => {
    const window = windowHistogram(points, pick, from);
    const lifetime = pick(last);
    const trendP50: Values = [];
    const trendP95: Values = [];
    let start = 0;
    for (let i = 0; i < points.length; i++) {
      while (points[start].at < points[i].at - TREND_SPAN_MS) start++;
      const h = i > start ? windowHistogram(points.slice(start, i + 1), pick, -Infinity) : null;
      const has = h !== null && h.count > 0;
      trendP50.push(has ? histogramQuantile(0.5, h) : null);
      trendP95.push(has ? histogramQuantile(0.95, h) : null);
    }
    const has = window !== null && window.count > 0;
    return {
      window,
      count: window?.count ?? 0,
      p50: has ? histogramQuantile(0.5, window) : null,
      p95: has ? histogramQuantile(0.95, window) : null,
      p99: has ? histogramQuantile(0.99, window) : null,
      mean: has ? histogramMean(window) : null,
      lifetimeP95: lifetime ? histogramQuantile(0.95, lifetime) : null,
      lifetimeCount: lifetime?.count ?? null,
      trendP50,
      trendP95,
    };
  };

  const peak = (values: Values) => {
    const seen = values.slice(visibleStart).filter((v): v is number => v !== null);
    return seen.length ? Math.max(...seen) : null;
  };

  const tokenRate = rollingRate(points, (s) => s.generatedTokens, RATE_SPAN_MS);
  const prefixRate = rollingRate(points, (s) => s.prefixReusedTokens, RATE_SPAN_MS);
  const completed = counter((s) => s.completed);
  const tokensCounter = counter((s) => s.generatedTokens);
  const runningSeries = points.map((p) => p.snap.running);
  const waitingSeries = points.map((p) => p.snap.waiting);
  const byReason = Object.fromEntries(
    REJECT_REASONS.map((r) => [r, { total: last.rejected[r], window: increaseOver(points, (s) => s.rejected[r], from) }]),
  ) as Dashboard["rejected"]["byReason"];

  const dash: Dashboard = {
    now,
    from,
    windowMs,
    visibleStart,
    times: points.map((p) => p.at),
    version: last.version,
    running: last.running,
    waiting: last.waiting,
    runningSeries,
    waitingSeries,
    runningPeak: peak(runningSeries),
    waitingPeak: peak(waitingSeries),
    tokens: {
      ...tokensCounter,
      perSec: tokenRate.at(-1) ?? null,
      perSecSeries: tokenRate,
      perRequest: tokensCounter.window !== null && completed.window ? tokensCounter.window / completed.window : null,
    },
    accepted: counter((s) => s.accepted),
    completed,
    cancelled: counter((s) => s.cancelled),
    rejected: { ...counter(totalRejected), byReason },
    prefix: { ...counter((s) => s.prefixReusedTokens), perSec: prefixRate.at(-1) ?? null, perSecSeries: prefixRate },
    evictions: counter((s) => s.kvEvictions),
    ttft: latency((s) => s.ttft),
    duration: latency((s) => s.duration),
    scrapeMsSeries: points.map((p) => p.scrapeMs),
    health: { level: "idle", summary: "", notes: [] },
  };
  dash.health = assessHealth({
    windowMs,
    running: dash.running,
    waiting: dash.waiting,
    accepted: dash.accepted.window,
    completed: dash.completed.window,
    cancelled: dash.cancelled.window,
    rejected: { full: byReason.full.window, unknown_model: byReason.unknown_model.window, oversized: byReason.oversized.window },
    evictions: dash.evictions.window,
    ttftP95: dash.ttft.p95,
  });
  return dash;
}

export type HealthInput = {
  windowMs: number;
  running: number | null;
  waiting: number | null;
  accepted: number | null;
  completed: number | null;
  cancelled: number | null;
  rejected: Record<RejectReason, number | null>;
  evictions: number | null;
  ttftP95: number | null;
};

/** A first-token p95 above this reads as slow. */
const SLOW_TTFT_S = 5;

const plural = (n: number, one: string, many = `${one}s`) => `${formatCount(n)} ${n === 1 ? one : many}`;

/** How the server is doing, in words, from the latest gauges and the window's counters. */
export function assessHealth(h: HealthInput): Health {
  const n = (v: number | null) => v ?? 0;
  const win = formatWindow(h.windowMs);
  const full = n(h.rejected.full);
  const finished = n(h.completed) + n(h.cancelled);
  const slow = h.ttftP95 !== null && h.ttftP95 > SLOW_TTFT_S;

  const notes: string[] = [];
  if (full) notes.push(`${plural(full, "request")} turned away: engine full`);
  if (n(h.rejected.oversized)) notes.push(`${plural(n(h.rejected.oversized), "request")} too long to ever fit`);
  if (n(h.rejected.unknown_model)) notes.push(`${plural(n(h.rejected.unknown_model), "request")} for an unknown model`);
  if (n(h.waiting)) notes.push(`${formatCount(n(h.waiting))} waiting for a lane`);
  if (n(h.evictions)) notes.push(`${plural(n(h.evictions), "KV eviction")} to host RAM`);
  if (n(h.cancelled) >= 2 && n(h.cancelled) / finished >= 0.2) notes.push(`${formatShare(n(h.cancelled), finished)} of finished requests cancelled`);
  if (slow) notes.push(`p95 time to first token ${formatSeconds(h.ttftP95)}`);

  if (full) return { level: "saturated", summary: `${plural(full, "request")} turned away in the last ${win}`, notes };
  if (n(h.waiting) || n(h.evictions) || slow) {
    const summary = n(h.waiting)
      ? `${formatCount(n(h.waiting))} waiting, ${formatCount(n(h.running))} running`
      : n(h.evictions)
        ? `${formatCount(n(h.running))} running under KV-cache pressure`
        : `First tokens take ${formatSeconds(h.ttftP95)} at p95`;
    return { level: "busy", summary, notes };
  }
  if (n(h.running)) return { level: "healthy", summary: `${plural(n(h.running), "request")} running, nothing queued`, notes };
  if (n(h.accepted) || finished) return { level: "healthy", summary: `${plural(n(h.completed), "request")} completed in the last ${win}, quiet now`, notes };
  return { level: "idle", summary: `No requests in the last ${win}`, notes };
}

/** The header's live figures: the latest gauges and the generated-token rate. */
export function headerPulse(points: Point[]): { running: number | null; waiting: number | null; tokensPerSec: number | null } {
  const last = points.at(-1)?.snap;
  const recent = points.filter((p) => p.at >= (points.at(-1)?.at ?? 0) - RATE_SPAN_MS);
  return {
    running: last?.running ?? null,
    waiting: last?.waiting ?? null,
    tokensPerSec: rollingRate(recent, (s) => s.generatedTokens, RATE_SPAN_MS).at(-1) ?? null,
  };
}
