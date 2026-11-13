import { formatCount, formatSeconds, formatShare, formatWindow } from "./format.ts";
import { type CounterPick, increaseOver, type Point, rollingRate, windowHistogram } from "./history.ts";
import { histogramMean, histogramQuantile } from "./quantile.ts";
import {
  DRAFT_POSITIONS,
  EXPERT_CLASSES,
  EXPERT_PHASES,
  type ExpertClass,
  type ExpertPhase,
  emptySnapshot,
  type Histogram,
  IMAGE_OUTCOMES,
  SPEECH_EXECUTORS,
  SPEECH_OUTCOMES,
  type SpeechOutcome,
  IMAGE_PHASES,
  type ImageOutcome,
  type ImagePhase,
  KV_DISK_SPILL_SOURCES,
  type KvDiskSpillSource,
  LOOP_OUTCOMES,
  type LoopOutcome,
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
  STRUCTURED_TYPES,
  VRAM_LINES,
  type VramLine,
} from "./snapshot.ts";

/**
 * The plan lines a server may predate, counted as 0 when absent: the 27B
 * trellis body's workspace (GitHub #339) and its dense prefill's buffer
 * (GitHub #509), Flash-Next's two (GitHub #306) and the image model's
 * condition budget (GitHub #324).
 */
const LATER_LINES: readonly VramLine[] = ["trellis_workspace", "trellis_dense", "residency", "expert_cache", "condition_budget"];

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
  /** The plan's twelve lines in plan order, then Flash-Next's residency and expert cache, then the image model's condition budget. */
  lines: { line: VramLine; bytes: number | null }[];
  /**
   * The twelve lines added up (on an image load: its weights and arena beside
   * the zeroed rest), or null when the scrape does not carry all twelve. A partial sum would understate the plan and overstate the room
   * left beside it, and the server writes the twelve together or not at all.
   */
  linesBytes: number | null;
  /** The expert cache's bytes: 0 on a load without one (the 27B), where the room left is all the KV pool's. */
  expertCacheBytes: number;
  /** The image model's condition budget (GitHub #324): 0 on a text load, which has a KV pool instead. */
  conditionBudgetBytes: number;
  /** The budget less the lines: the room the plan left the KV pool (on an image load, its condition budget). Negative on an oversubscribed load. */
  kvRoomBytes: number | null;
  /** The pool the room bought: its pages, one page, the two multiplied, and the tokens those pages hold. */
  kvPool: { pages: number | null; pageBytes: number | null; bytes: number | null; tokens: number | null };
  /** The budget beyond the lines and the pool's pages: page-rounding slack, and the pool's own tables. */
  spareBytes: number | null;
  /** Whether the plan overran its budget, which only --vram-allow-oversubscription allows. */
  oversubscribed: boolean;
  pagesInUse: Meter;
  arenaInUse: Meter;
  /** The KV-disk tier's bytes against its budget (spec vram-budget/03); null on a load without the tier. */
  diskInUse: Meter | null;
  /** The condition prefixes the image model's jobs hold, against its condition budget (GitHub #324); null on a load without one. */
  conditionInUse: Meter | null;
  slotsInUse: Meter;
  /**
   * Where the retained slots live (GitHub #281): how many in the pinned host
   * block and its bytes, and the rest, in VRAM. Null where the scrape does
   * not carry the host gauges -- a server from before them.
   */
  slotHomes: { hostSlots: number | null; hostBytes: number | null; deviceSlots: number | null };
  skips: Tally & { byReason: Record<SlotSkipReason, Tally> };
  retained: Record<RetainedFamily, Record<RetainedTier, Record<RetainedKind, Tally>>>;
};

/**
 * Flash-Next's expert residency over the window (GitHub #301, spec
 * flash-next/03 story 7): what tells a slow turn caused by misses from one
 * caused by compute. Shares and rates are computed here from the counters,
 * since the server exports no ratio.
 */
export type Experts = {
  /** Decode hits over decode hits plus misses, gained over the window; null without a selection. */
  decodeHitShare: number | null;
  prefillHitShare: number | null;
  /** Decode misses over the tokens decoded in the window. */
  missesPerToken: number | null;
  /** Host-to-device bytes per second, both phases, RATE_SPAN_MS rolling. */
  bytesPerSec: number | null;
  bytesPerSecSeries: Values;
  bytesMoved: Tally;
  prefetch: { issued: Tally; used: Tally };
  /** Decode seconds waited on demand copies over the tokens decoded in the window (GitHub #306). */
  stallPerToken: number | null;
  /** Seconds waited on demand copies, both phases, per second of wall time, RATE_SPAN_MS rolling. */
  stallShare: number | null;
  /** Seconds waited on demand copies, both phases. */
  stall: Tally;
  /** Each K class: its slots, and its hits and misses over both phases. */
  classes: { cls: ExpertClass; slots: Meter; hits: Tally; misses: Tally }[];
};

/** Flash-Next's n-gram rows over the window (GitHub #302). */
export type Ngram = {
  rows: Tally;
  /** Rows the RAM hot-row cache held, over all rows staged in the window. */
  hotShare: number | null;
  reads: Tally;
  readBytes: Tally;
  /** Bytes read from the artifact file per second, RATE_SPAN_MS rolling. */
  readBytesPerSec: number | null;
  readBytesPerSecSeries: Values;
};

/** Speculative decoding over the window (GitHub #307). */
export type Speculation = {
  rounds: Tally;
  drafted: Tally;
  accepted: Tally;
  /** Drafts committed over drafts verified, in the window. */
  acceptance: number | null;
  /** Tokens a verify round committed on average in the window: its anchor plus its accepted drafts. */
  tokensPerRound: number | null;
  /** Per draft position (1 first): rounds that committed it over rounds that proposed it, in the window. */
  byPosition: (number | null)[];
};

/**
 * The image jobs over the window (GitHub #324): how many ended and how, the
 * images made, how long the completed ones took and where the time went.
 * Shares are computed here from the phase counters, since the server
 * exports none.
 */
export type Images = {
  /** Jobs ended, all outcomes; `byOutcome` splits them. */
  jobs: Counter & { byOutcome: Record<ImageOutcome, Tally> };
  images: Counter;
  duration: Latency;
  waiting: number | null;
  running: number | null;
  /** Seconds in each phase over the window, and its share of all of them. */
  phases: { phase: ImagePhase; seconds: Tally; share: number | null }[];
};

/**
 * The speech model's transcriptions over the window (spec speech/02): how
 * many ended and how (both executors together), the seconds of audio they
 * covered, how long the completed ones waited from commit to transcript, and
 * the sessions and queue now.
 */
export type Speech = {
  /** Transcriptions ended, all outcomes; `byOutcome` splits them. */
  transcriptions: Counter & { byOutcome: Record<SpeechOutcome, Tally> };
  audioSeconds: Counter;
  wait: Latency;
  sessionsOpen: number | null;
  queueLength: number | null;
};

/**
 * The Structured output card's figures (#593): the requests held to a grammar,
 * the draws it rejected and the repairs, over the window, and how long a repair
 * took. Misses per request is a plain ratio, which a sloppy prompt can push
 * above one.
 */
export type StructuredOutputCard = {
  requests: Tally;
  misses: Tally;
  repairs: Tally;
  /** Misses over requests in the window; null with no request in it. */
  missesPerRequest: number | null;
  /** Repair times over the window, in seconds. */
  repairTime: Histogram | null;
  repairP50: number | null;
  repairP95: number | null;
};

/**
 * The reasoning's two ways of being cut short over the window (GitHub #330):
 * the loop guard's rewinds and the thinking budget's forced closes. Shares are
 * computed here from the counters, since the server exports none. The guard's
 * figures are null where the guard watched nothing (`loop_guard: off`); the
 * budget's close is its own counter, not the guard's `closed`.
 */
export type Reasoning = {
  /** Whether the guard's family is on the scrape at all. */
  guard: boolean;
  watched: Tally;
  looped: Tally;
  /** Requests that looped over requests watched, in the window. */
  loopedShare: number | null;
  /** Loops by what the guard did about them. */
  outcomes: Record<LoopOutcome, Tally>;
  /**
   * The guard's cancels in the window: a rewind's (a nudge or a close), each also one admission on the request
   * counters, and one whose continuation was refused.
   */
  guardCancels: number | null;
  rewoundTokens: Tally;
  /** Tokens taken back over the tokens decoded in the window. */
  rewoundShare: number | null;
  /** The lowest loop score each request reached, observed over the window. */
  minRatio: Histogram | null;
  /** Reasonings the thinking budget closed. */
  forcedCloses: Tally;
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
  evictions: Evictions;
  ttft: Latency;
  duration: Latency;
  scrapeMsSeries: Values;
  memory: Memory;
  /** Null on a load without an expert cache: a 27B load. */
  experts: Experts | null;
  /** Null on a load without an n-gram table: a 27B load. */
  ngram: Ngram | null;
  /** Null until a verify round ran: a load without speculation. */
  speculation: Speculation | null;
  /** Null until the image model admitted a job: a text-only server. */
  images: Images | null;
  /** Null until the speech model was asked for a transcription. */
  speech: Speech | null;
  /** Null until the loop guard watched a request or the thinking budget closed one. */
  reasoning: Reasoning | null;
  /** Null until a request held to a grammar ended. */
  structured: StructuredOutputCard | null;
  health: Health;
};

/** The residency tiers the eviction panel reports, fastest first (GitHub #224). */
export const EVICTION_TIERS = ["vram", "ram", "disk"] as const;
export type EvictionTierName = (typeof EVICTION_TIERS)[number];

/**
 * What one tier gave up (GitHub #224). An eviction is a *departure from a
 * tier*, and the three columns cost different things, so they are never
 * summed into one number:
 *
 * - `live` — a running or half-prefilled request lost residency here. Out of
 *   VRAM its work is snapshotted and survives; out of RAM it is destroyed and
 *   the request re-prefills from zero. This is the column that costs.
 * - `retained` — cached state (a prompt checkpoint or a shared prefix) left
 *   the tier for nowhere. A future request pays a miss; nothing in flight
 *   does. This column mixes pressure with ordinary turnover: a conversation's
 *   superseded checkpoints are discarded on the next capture (GitHub #187),
 *   which is housekeeping, not memory pressure. Watch `live` for pressure.
 * - `demoted` — state that moved down a tier and survived. Not a loss, but it
 *   is how the tier below fills up.
 *
 * A column is `null` when the tier has no such departure to report — and every
 * column is `null` on a tier that does not exist (`implemented: false`: the
 * disk row on a load without the KV-disk tier, spec vram-budget/03).
 */
export type TierEvictions = {
  tier: EvictionTierName;
  /** False when no metric feeds this row because this load has no such tier. */
  implemented: boolean;
  live: Counter | null;
  retained: Counter | null;
  demoted: Counter | null;
};

export type Evictions = Record<EvictionTierName, TierEvictions>;

/** What is known of a set of series, added up; null when none of them is. */
function sumKnown(values: (number | null)[]): number | null {
  const known = values.filter((v): v is number => v !== null);
  return known.length ? known.reduce((a, b) => a + b, 0) : null;
}

const totalRejected: CounterPick = (s) => sumKnown(REJECT_REASONS.map((r) => s.rejected[r]));

/** The live sequences written into the disk tier, from the device and from KV-RAM (spec vram-budget/03). */
const diskSpills = (s: Snapshot, source?: KvDiskSpillSource): number | null =>
  s.kvDisk ? sumKnown(KV_DISK_SPILL_SOURCES.filter((from) => !source || from === source).map((from) => s.kvDisk!.spills[from])) : null;

/** One retained-state family's figure for a tier, over both kinds (GitHub #224). */
function retainedAt(s: Snapshot, family: RetainedFamily, tier: RetainedTier): number | null {
  return sumKnown(RETAINED_KINDS.map((kind) => s.retained[family][tier][kind]));
}

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
    evictions: {
      // Out of VRAM: a live sequence snapshotted to RAM, retained state
      // dropped outright, and retained state spilled down to RAM — a spill is
      // recorded *into* `kv_ram`, and arriving there is the same act as
      // leaving the device. With the KV-disk tier, a live sequence written
      // straight to disk left the device too (spec vram-budget/03).
      vram: {
        tier: "vram",
        implemented: true,
        live: counter((s) => (s.kvDisk ? sumKnown([s.kvEvictions, diskSpills(s, "device")]) : s.kvEvictions)),
        retained: counter((s) => retainedAt(s, "discards", "device")),
        demoted: counter((s) => retainedAt(s, "spills", "kv_ram")),
      },
      // Out of RAM: a live snapshot dropped to make room (the request loses
      // every prefilled token), and retained blobs forgotten. With the
      // KV-disk tier below it, retained state spilled into the disk and live
      // snapshots demoted to it left RAM and survived; without, nothing is
      // demoted because there is no tier below RAM.
      ram: {
        tier: "ram",
        implemented: true,
        live: counter((s) => s.kvRamEvictions),
        retained: counter((s) => retainedAt(s, "discards", "kv_ram")),
        demoted: last.kvDisk ? counter((s) => sumKnown([retainedAt(s, "spills", "disk"), diskSpills(s, "kv_ram")])) : null,
      },
      // The disk has nothing below it, so its row reads arrivals into the tier
      // where the rows above read departures: `live` is the live sequences
      // spilled into it (either source), `demoted` the retained state spilled
      // into it, and `retained` the retained state it discarded (spec
      // vram-budget/03). On a load without the tier the series are absent and
      // the row stays inert, so the panel does not show a zero for a tier
      // that is off (ADR 0017).
      disk: last.kvDisk
        ? {
            tier: "disk",
            implemented: true,
            live: counter((s) => diskSpills(s)),
            retained: counter((s) => retainedAt(s, "discards", "disk")),
            demoted: counter((s) => retainedAt(s, "spills", "disk")),
          }
        : { tier: "disk", implemented: false, live: null, retained: null, demoted: null },
    },
    ttft: latency((s) => s.ttft),
    duration: latency((s) => s.duration),
    scrapeMsSeries: points.map((p) => p.scrapeMs),
    memory: deriveMemory(points, from),
    experts: last.experts ? deriveExperts(points, from) : null,
    ngram: last.ngram ? deriveNgram(points, from) : null,
    speculation: last.speculation ? deriveSpeculation(points, from) : null,
    images: last.images
      ? {
          jobs: {
            ...counter((s) => sumKnown(IMAGE_OUTCOMES.map((outcome) => s.images?.jobs[outcome] ?? null))),
            byOutcome: Object.fromEntries(
              IMAGE_OUTCOMES.map((outcome) => [outcome, { total: last.images!.jobs[outcome], window: increaseOver(points, (s) => s.images?.jobs[outcome] ?? null, from) }]),
            ) as Record<ImageOutcome, Tally>,
          },
          images: counter((s) => s.images?.images ?? null),
          duration: latency((s) => s.images?.duration ?? null),
          waiting: last.images.waiting,
          running: last.images.running,
          phases: deriveImagePhases(points, from),
        }
      : null,
    speech: last.speech
      ? {
          transcriptions: {
            ...counter((s) => sumKnown(SPEECH_EXECUTORS.flatMap((executor) => SPEECH_OUTCOMES.map((outcome) => s.speech?.transcriptions[executor][outcome] ?? null)))),
            byOutcome: Object.fromEntries(
              SPEECH_OUTCOMES.map((outcome) => {
                const pick = (s: Snapshot) => sumKnown(SPEECH_EXECUTORS.map((executor) => s.speech?.transcriptions[executor][outcome] ?? null));
                return [outcome, { total: pick(last), window: increaseOver(points, pick, from) }];
              }),
            ) as Record<SpeechOutcome, Tally>,
          },
          audioSeconds: counter((s) => s.speech?.audioSeconds ?? null),
          wait: latency((s) => s.speech?.wait ?? null),
          sessionsOpen: last.speech.sessionsOpen,
          queueLength: last.speech.queueLength,
        }
      : null,
    reasoning: last.loopGuard || last.thinkingForcedCloses !== null ? deriveReasoning(points, from) : null,
    structured: last.structured ? deriveStructured(points, from) : null,
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
    evictions: dash.evictions.vram.live?.window ?? null,
    ramDrops: dash.evictions.ram.live?.window ?? null,
    diskReadFailures: increaseOver(points, (s) => s.kvDisk?.failures.read ?? null, from),
    ttftP95: dash.ttft.p95,
  });
  return dash;
}


/** Each image phase's seconds, and its share of the window's image time (GitHub #324). */
function deriveImagePhases(points: Point[], since: number): Images["phases"] {
  const last = points.at(-1)?.snap ?? emptySnapshot();
  const seconds = IMAGE_PHASES.map((phase) => ({
    phase,
    seconds: { total: last.images?.phaseSeconds[phase] ?? null, window: increaseOver(points, (s) => s.images?.phaseSeconds[phase] ?? null, since) },
  }));
  const all = sumKnown(seconds.map((p) => p.seconds.window));
  return seconds.map((p) => ({ ...p, share: shareOf(p.seconds.window, all) }));
}

/** A part over its whole, null for an empty or unknown whole. */
const shareOf = (part: number | null, whole: number | null): number | null => (part !== null && whole !== null && whole > 0 ? part / whole : null);

/** One expert-cache family summed over the classes, for one phase or both. */
function expertSum(s: Snapshot, family: "hits" | "misses", phases: readonly ExpertPhase[]): number | null {
  const experts = s.experts;
  return experts ? sumKnown(EXPERT_CLASSES.flatMap((cls) => phases.map((phase) => experts[family][cls][phase]))) : null;
}

/** The residency panel's figures (GitHub #301): the window's gains, read off the counters. */
export function deriveExperts(points: Point[], since: number): Experts {
  const last = points.at(-1)?.snap ?? emptySnapshot();
  const inWindow = points.filter((p) => p.at >= since);
  const tally = (pick: CounterPick): Tally => ({ total: pick(last), window: increaseOver(points, pick, since) });
  const share = (phase: ExpertPhase) => {
    const hits = increaseOver(points, (s) => expertSum(s, "hits", [phase]), since);
    const misses = increaseOver(points, (s) => expertSum(s, "misses", [phase]), since);
    return shareOf(hits, hits !== null && misses !== null ? hits + misses : null);
  };
  const bytes: CounterPick = (s) => (s.experts ? sumKnown(EXPERT_PHASES.map((phase) => s.experts!.bytesMoved[phase])) : null);
  const bytesRate = rollingRate(points, bytes, RATE_SPAN_MS);
  const decodeMisses = increaseOver(points, (s) => expertSum(s, "misses", ["decode"]), since);
  const decodedInWindow = increaseOver(points, (s) => s.decodedTokens, since);
  const stall: CounterPick = (s) => (s.experts ? sumKnown(EXPERT_PHASES.map((phase) => s.experts!.stall[phase])) : null);
  return {
    decodeHitShare: share("decode"),
    prefillHitShare: share("prefill"),
    missesPerToken: shareOf(decodeMisses, decodedInWindow),
    bytesPerSec: bytesRate.at(-1) ?? null,
    bytesPerSecSeries: bytesRate,
    bytesMoved: tally(bytes),
    prefetch: { issued: tally((s) => s.experts?.prefetchIssued ?? null), used: tally((s) => s.experts?.prefetchUsed ?? null) },
    stallPerToken: shareOf(
      increaseOver(points, (s) => s.experts?.stall.decode ?? null, since),
      decodedInWindow,
    ),
    stallShare: rollingRate(points, stall, RATE_SPAN_MS).at(-1) ?? null,
    stall: tally(stall),
    classes: EXPERT_CLASSES.map((cls) => {
      const slots = last.experts?.slots[cls] ?? { capacity: null, inUse: null };
      const both = (family: "hits" | "misses"): CounterPick => (s) => (s.experts ? sumKnown(EXPERT_PHASES.map((phase) => s.experts![family][cls][phase])) : null);
      return {
        cls,
        slots: {
          used: slots.inUse,
          capacity: slots.capacity,
          share: shareOf(slots.inUse, slots.capacity),
          series: inWindow.map((p) => p.snap.experts?.slots[cls].inUse ?? null),
        },
        hits: tally(both("hits")),
        misses: tally(both("misses")),
      };
    }),
  };
}

/** The n-gram panel's figures (GitHub #302). */
export function deriveNgram(points: Point[], since: number): Ngram {
  const last = points.at(-1)?.snap ?? emptySnapshot();
  const tally = (pick: CounterPick): Tally => ({ total: pick(last), window: increaseOver(points, pick, since) });
  const rows = tally((s) => (s.ngram ? sumKnown([s.ngram.rows.hot, s.ngram.rows.file]) : null));
  const hot = increaseOver(points, (s) => s.ngram?.rows.hot ?? null, since);
  const readBytesRate = rollingRate(points, (s) => s.ngram?.readBytes ?? null, RATE_SPAN_MS);
  return {
    rows,
    hotShare: shareOf(hot, rows.window),
    reads: tally((s) => s.ngram?.reads ?? null),
    readBytes: tally((s) => s.ngram?.readBytes ?? null),
    readBytesPerSec: readBytesRate.at(-1) ?? null,
    readBytesPerSecSeries: readBytesRate,
  };
}

/** The Structured output card's figures (#593). */
export function deriveStructured(points: Point[], since: number): StructuredOutputCard {
  const last = points.at(-1)?.snap ?? emptySnapshot();
  const tally = (pick: CounterPick): Tally => ({ total: pick(last), window: increaseOver(points, pick, since) });
  const sum = (family: "requests" | "misses" | "repairs"): CounterPick => (s) =>
    s.structured ? sumKnown(STRUCTURED_TYPES.map((type) => s.structured![family][type])) : null;
  const requests = tally(sum("requests"));
  const misses = tally(sum("misses"));
  const repairTime = windowHistogram(points, (s) => s.structured?.repairTime ?? null, since);
  const has = repairTime !== null && repairTime.count > 0;
  return {
    requests,
    misses,
    repairs: tally(sum("repairs")),
    missesPerRequest: misses.window !== null && requests.window ? misses.window / requests.window : null,
    repairTime,
    repairP50: has ? histogramQuantile(0.5, repairTime) : null,
    repairP95: has ? histogramQuantile(0.95, repairTime) : null,
  };
}

/** The Reasoning card's figures (GitHub #330): the guard's loops and rewinds and the budget's closes, over the window. */
export function deriveReasoning(points: Point[], since: number): Reasoning {
  const last = points.at(-1)?.snap ?? emptySnapshot();
  const tally = (pick: CounterPick): Tally => ({ total: pick(last), window: increaseOver(points, pick, since) });
  const watched = tally((s) => s.loopGuard?.watched ?? null);
  const looped = tally((s) => s.loopGuard?.looped ?? null);
  const outcomes = Object.fromEntries(LOOP_OUTCOMES.map((outcome) => [outcome, tally((s) => s.loopGuard?.loops[outcome] ?? null)])) as Record<LoopOutcome, Tally>;
  const rewoundTokens = tally((s) => s.loopGuard?.rewoundTokens ?? null);
  const cancelling = [outcomes.rewound.window, outcomes.closed.window, outcomes.submit_refused.window];
  const guardCancels = cancelling.every((n) => n === null) ? null : cancelling.reduce<number>((sum, n) => sum + (n ?? 0), 0);
  return {
    guard: last.loopGuard !== null,
    watched,
    looped,
    loopedShare: shareOf(looped.window, watched.window),
    outcomes,
    guardCancels,
    rewoundTokens,
    rewoundShare: shareOf(rewoundTokens.window, increaseOver(points, (s) => s.decodedTokens, since)),
    minRatio: windowHistogram(points, (s) => s.loopGuard?.minRatio ?? null, since),
    forcedCloses: tally((s) => s.thinkingForcedCloses),
  };
}

/** The speculation panel's figures (GitHub #307): acceptance over the window, overall and per draft position. */
export function deriveSpeculation(points: Point[], since: number): Speculation {
  const last = points.at(-1)?.snap ?? emptySnapshot();
  const tally = (pick: CounterPick): Tally => ({ total: pick(last), window: increaseOver(points, pick, since) });
  const rounds = tally((s) => s.speculation?.rounds ?? null);
  const drafted = tally((s) => s.speculation?.drafted ?? null);
  const accepted = tally((s) => s.speculation?.accepted ?? null);
  const byPosition = Array.from({ length: DRAFT_POSITIONS }, (_, j) =>
    shareOf(
      increaseOver(points, (s) => s.speculation?.positionAccepted[j] ?? null, since),
      increaseOver(points, (s) => s.speculation?.positionDrafted[j] ?? null, since),
    ),
  );
  return {
    rounds,
    drafted,
    accepted,
    acceptance: shareOf(accepted.window, drafted.window),
    tokensPerRound:
      rounds.window !== null && accepted.window !== null && rounds.window > 0 ? (rounds.window + accepted.window) / rounds.window : null,
    byPosition,
  };
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
  // The twelve plan lines come together or not at all; the later ones (GitHub
  // #306, #324, #339) are absent from a server that predates them, and count as 0 then.
  const linesBytes = lines.filter((l) => !LATER_LINES.includes(l.line)).every((l) => l.bytes !== null) ? sumKnown(lines.map((l) => l.bytes)) : null;
  const conditionBudgetBytes = mem.reserved.condition_budget ?? 0;
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
    expertCacheBytes: mem.reserved.expert_cache ?? 0,
    conditionBudgetBytes,
    kvRoomBytes: mem.budgetBytes !== null && linesBytes !== null ? mem.budgetBytes - linesBytes : null,
    kvPool: { pages: mem.kvPoolPages, pageBytes: mem.kvPageBytes, bytes: kvPoolBytes, tokens: kvPoolTokens },
    spareBytes: mem.budgetBytes === null || linesBytes === null ? null : mem.budgetBytes - linesBytes - (kvPoolBytes ?? 0),
    oversubscribed: mem.budgetBytes !== null && linesBytes !== null && linesBytes + (kvPoolBytes ?? 0) > mem.budgetBytes,
    pagesInUse: meter(mem.kvPoolUsedPages, mem.kvPoolPages, (s) => s.memory.kvPoolUsedPages),
    arenaInUse: meter(mem.kvRamArena.used, mem.kvRamArena.capacity, (s) => s.memory.kvRamArena.used),
    diskInUse: last.kvDisk ? meter(last.kvDisk.used, last.kvDisk.capacity, (s) => s.kvDisk?.used ?? null) : null,
    conditionInUse: conditionBudgetBytes > 0 ? meter(mem.conditionBudgetUsedBytes, conditionBudgetBytes, (s) => s.memory.conditionBudgetUsedBytes) : null,
    slotsInUse: meter(mem.retainedSlots.inUse, mem.retainedSlots.capacity, (s) => s.memory.retainedSlots.inUse),
    slotHomes: {
      hostSlots: mem.retainedHost.slots,
      hostBytes: mem.retainedHost.bytes,
      deviceSlots:
        mem.retainedHost.slots !== null && mem.retainedSlots.capacity !== null ? mem.retainedSlots.capacity - mem.retainedHost.slots : null,
    },
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
  /** Live sequences evicted out of VRAM in the window: their work survives in RAM. */
  evictions: number | null;
  /** Live snapshots dropped out of RAM in the window: their work is gone (GitHub #224). */
  ramDrops: number | null;
  /** Disk files refused at restore in the window: torn, corrupt or foreign, so a live request re-prefilled (spec vram-budget/03). */
  diskReadFailures: number | null;
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
  // A RAM drop outranks a VRAM eviction: the evicted request keeps its work in
  // host RAM, the dropped one re-prefills from zero (GitHub #224).
  if (n(h.ramDrops)) notes.push(`${plural(n(h.ramDrops), "snapshot")} dropped from host RAM: re-prefilled from the start`);
  // A disk read failure costs what a RAM drop does: the request re-prefills.
  if (n(h.diskReadFailures)) notes.push(`${plural(n(h.diskReadFailures), "KV-disk file")} refused at restore: re-prefilled from the start`);
  if (n(h.cancelled) >= 2 && n(h.cancelled) / finished >= 0.2) notes.push(`${formatShare(n(h.cancelled), finished)} of finished requests cancelled`);
  if (slow) notes.push(`p95 time to first token ${formatSeconds(h.ttftP95)}`);

  if (full) return { level: "saturated", summary: `${plural(full, "request")} turned away in the last ${win}`, notes };
  // A dropped snapshot is lost work, so it saturates where an eviction only
  // makes the server busy (GitHub #224); so is a disk file refused at restore.
  if (n(h.ramDrops)) {
    return {
      level: "saturated",
      summary: `${plural(n(h.ramDrops), "snapshot")} dropped from host RAM in the last ${win}`,
      notes,
    };
  }
  if (n(h.diskReadFailures)) {
    return {
      level: "saturated",
      summary: `${plural(n(h.diskReadFailures), "KV-disk file")} refused at restore in the last ${win}`,
      notes,
    };
  }
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
