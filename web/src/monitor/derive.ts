import { formatCount, formatSeconds, formatShare, formatWindow } from "./format.ts";
import { type CounterPick, increaseOver, type Point, rollingRate, windowHistogram } from "./history.ts";
import { histogramMean, histogramQuantile } from "./quantile.ts";
import {
  emptySnapshot,
  type Histogram,
  REJECT_REASONS,
  type RejectReason,
  RETAINED_FAMILY_KEYS,
  RETAINED_KINDS,
  RETAINED_TIERS,
  type RetainedFamily,
  type RetainedKind,
  type RetainedTier,
  SLOT_SKIP_REASONS,
  type SlotSkipReason,
  type Snapshot,
  VRAM_LINES,
  type VramLine,
} from "./snapshot.ts";

// Everything the Monitor shows, derived from the scrape history for one
// window: counter gains and rates, scheduler load, latency quantiles and
// their trend, and a plain-words verdict on how the server is doing.

/** Rates are averaged over this span, since counters move in steps. */
export const RATE_SPAN_MS = 30_000;
/** Decoded tokens move with every token, so their rate needs only this short span. */
export const TOKEN_RATE_SPAN_MS = 10_000;
/** Latency trends are quantiles over this sliding span. */
export const TREND_SPAN_MS = 60_000;
/**
 * The tokens one KV page holds. Fixed in the kernel
 * (`kernel/vendor/src/core/paged_kv_cache.h`, `kPagedKVPageSize`) and passed
 * on by the CUDA leaf as `kv_page_tokens: 64`; no scrape carries it, so the
 * pool's tokens are its pages multiplied here rather than read from a gauge.
 */
export const TOKENS_PER_KV_PAGE = 64;

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


/**
 * A live figure against the constant that bounds it (ADR 0030
 * §Observability). The share is computed here, from two terms that are both
 * on the page, because no percentage is exported: a reader can always see
 * which of the two moved.
 */
export type Meter = { used: number | null; capacity: number | null; share: number | null; series: Values };


/** A counter read twice over: where it stands, and what it gained over the window. */
export type Tally = { total: number | null; window: number | null };

/** What the load reserved, and what is occupied of it. */
export type Memory = {
  /** Whether this load laid a plan out at all — the placeholder load has none. */
  planned: boolean;
  budgetBytes: number | null;
  /** The plan's eleven lines, in plan order. */
  lines: { line: VramLine; bytes: number | null }[];
  /**
   * The eleven lines added up, or null when the scrape does not carry all
   * eleven. A partial sum would understate the plan and overstate the room
   * left beside it, and the server writes the eleven together or not at all.
   */
  linesBytes: number | null;
  /** The budget less the lines: the room the plan left the KV pool. Negative on an oversubscribed load. */
  kvRoomBytes: number | null;
  /** The pool the room bought: its pages, one page, the two multiplied, and the tokens those pages hold. */
  kvPool: { pages: number | null; pageBytes: number | null; bytes: number | null; tokens: number | null };
  /** The budget beyond the lines and the pool's pages: page-rounding slack, and the pool's own tables. */
  spareBytes: number | null;
  /** Whether the plan overran its budget, which only --allow-vram-oversubscription allows. */
  oversubscribed: boolean;
  pagesInUse: Meter;
  arenaInUse: Meter;
  slotsInUse: Meter;
  skips: Tally & { byReason: Record<SlotSkipReason, Tally> };
  retained: Record<RetainedFamily, Record<RetainedTier, Record<RetainedKind, Tally>>>;
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
  /**
   * Token throughput. `live` when the server counts tokens as they are decoded;
   * otherwise it only has tokens on completed requests, which move in steps.
   */
  tokens: Counter & { live: boolean; spanMs: number; perSec: number | null; perSecSeries: Values; perRequest: number | null };
  accepted: Counter;
  completed: Counter;
  cancelled: Counter;
  rejected: Counter & { byReason: Record<RejectReason, Tally> };
  prefix: Counter & { perSec: number | null; perSecSeries: Values };
  evictions: Counter;
  ttft: Latency;
  duration: Latency;
  scrapeMsSeries: Values;
  memory: Memory;
  health: Health;
};

/** What is known of a set of series, added up; null when none of them is. */
function sumKnown(values: (number | null)[]): number | null {
  const known = values.filter((v): v is number => v !== null);
  return known.length ? known.reduce((a, b) => a + b, 0) : null;
}

const totalRejected: CounterPick = (s) => sumKnown(REJECT_REASONS.map((r) => s.rejected[r]));

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

  const { live, pick: tokenPick, spanMs: tokenSpanMs } = tokenSource(last);
  const tokenRate = rollingRate(points, tokenPick, tokenSpanMs);
  const prefixRate = rollingRate(points, (s) => s.prefixReusedTokens, RATE_SPAN_MS);
  const completed = counter((s) => s.completed);
  const tokensCounter = counter(tokenPick);
  // Per request is completed tokens over completed requests, whichever counter drives the rate.
  const completedTokens = increaseOver(points, (s) => s.generatedTokens, from);
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
      live,
      spanMs: tokenSpanMs,
      perSec: tokenRate.at(-1) ?? null,
      perSecSeries: tokenRate,
      perRequest: completedTokens !== null && completed.window ? completedTokens / completed.window : null,
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
    memory: deriveMemory(points, from),
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


/**
 * The memory panel's figures (GitHub #217, ADR 0030 §Observability): the
 * plan the load reserved, and what is occupied of each shape it bounds.
 *
 * The retained-state counters are read as plain totals and window gains
 * rather than rates: they move a handful of times a minute at most, and a
 * per-minute rate over a figure that small says less than the figure.
 */
export function deriveMemory(points: Point[], since: number): Memory {
  const last = points.at(-1)?.snap ?? emptySnapshot();
  const mem = last.memory;
  // The series a meter sparks is the window's own, so the window pill governs
  // these figures as it governs every other chart on the board.
  const inWindow = points.filter((p) => p.at >= since);
  const meter = (used: number | null, capacity: number | null, pick: CounterPick): Meter => ({
    used,
    capacity,
    share: used !== null && capacity !== null && capacity > 0 ? used / capacity : null,
    series: inWindow.map((p) => pick(p.snap)),
  });
  const tally = (pick: CounterPick): Tally => ({ total: pick(last), window: increaseOver(points, pick, since) });

  const lines = VRAM_LINES.map((line) => ({ line, bytes: mem.reserved[line] }));
  const linesBytes = lines.every((l) => l.bytes !== null) ? sumKnown(lines.map((l) => l.bytes)) : null;
  const kvPoolBytes = mem.kvPoolPages !== null && mem.kvPageBytes !== null ? mem.kvPoolPages * mem.kvPageBytes : null;
  const kvPoolTokens = mem.kvPoolPages === null ? null : mem.kvPoolPages * TOKENS_PER_KV_PAGE;
  const byReason = Object.fromEntries(
    SLOT_SKIP_REASONS.map((reason) => [reason, tally((s) => s.memory.slotSkips[reason])]),
  ) as Record<SlotSkipReason, Tally>;

  return {
    planned: mem.budgetBytes !== null && mem.budgetBytes > 0 && linesBytes !== null,
    budgetBytes: mem.budgetBytes,
    lines,
    linesBytes,
    kvRoomBytes: mem.budgetBytes !== null && linesBytes !== null ? mem.budgetBytes - linesBytes : null,
    kvPool: { pages: mem.kvPoolPages, pageBytes: mem.kvPageBytes, bytes: kvPoolBytes, tokens: kvPoolTokens },
    spareBytes: mem.budgetBytes === null || linesBytes === null ? null : mem.budgetBytes - linesBytes - (kvPoolBytes ?? 0),
    oversubscribed: mem.budgetBytes !== null && linesBytes !== null && linesBytes + (kvPoolBytes ?? 0) > mem.budgetBytes,
    pagesInUse: meter(mem.kvPoolUsedPages, mem.kvPoolPages, (s) => s.memory.kvPoolUsedPages),
    arenaInUse: meter(mem.kvRamArena.used, mem.kvRamArena.capacity, (s) => s.memory.kvRamArena.used),
    slotsInUse: meter(mem.retainedSlots.inUse, mem.retainedSlots.capacity, (s) => s.memory.retainedSlots.inUse),
    skips: {
      total: sumKnown(SLOT_SKIP_REASONS.map((reason) => byReason[reason].total)),
      window: sumKnown(SLOT_SKIP_REASONS.map((reason) => byReason[reason].window)),
      byReason,
    },
    retained: Object.fromEntries(
      RETAINED_FAMILY_KEYS.map((family) => [
        family,
        Object.fromEntries(
          RETAINED_TIERS.map((tier) => [
            tier,
            Object.fromEntries(RETAINED_KINDS.map((kind) => [kind, tally((s) => s.retained[family][tier][kind])])),
          ]),
        ),
      ]),
    ) as Memory["retained"],
  };
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

/** Which token counter drives throughput: decoded tokens when the server has them, else tokens on completed requests. */
function tokenSource(last: Snapshot): { live: boolean; pick: CounterPick; spanMs: number } {
  return last.decodedTokens !== null
    ? { live: true, pick: (s) => s.decodedTokens, spanMs: TOKEN_RATE_SPAN_MS }
    : { live: false, pick: (s) => s.generatedTokens, spanMs: RATE_SPAN_MS };
}

/** The header's live figures: the latest gauges and the token rate. */
export function headerPulse(points: Point[]): { running: number | null; waiting: number | null; tokensPerSec: number | null; live: boolean } {
  const lastPoint = points.at(-1);
  if (!lastPoint) return { running: null, waiting: null, tokensPerSec: null, live: false };
  const { live, pick, spanMs } = tokenSource(lastPoint.snap);
  const recent = points.filter((p) => p.at >= lastPoint.at - spanMs);
  return {
    running: lastPoint.snap.running,
    waiting: lastPoint.snap.waiting,
    tokensPerSec: rollingRate(recent, pick, spanMs).at(-1) ?? null,
    live,
  };
}
