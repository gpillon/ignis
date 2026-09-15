import type { Exposition, Sample } from "./exposition.ts";

// One scrape read against ADR 0017's contract table: the series the panel
// knows how to draw, by name. Anything else ignis serves is kept in `unknown`
// and listed raw, never dropped silently.

export const REJECT_REASONS = ["full", "unknown_model", "oversized"] as const;
export type RejectReason = (typeof REJECT_REASONS)[number];

/** A histogram: upper bounds in seconds ending in +Inf, their cumulative counts, the sum and the count. */
export type Histogram = { bounds: number[]; cumulative: number[]; sum: number; count: number };

export type Snapshot = {
  version: string | null;
  waiting: number | null;
  running: number | null;
  accepted: number | null;
  completed: number | null;
  cancelled: number | null;
  rejected: Record<RejectReason, number | null>;
  /** Tokens on completed requests: moves only when a request ends. */
  generatedTokens: number | null;
  /** Tokens as each is decoded (GitHub #165): moves while requests run. Null from a server without it. */
  decodedTokens: number | null;
  kvEvictions: number | null;
  prefixReusedTokens: number | null;
  ttft: Histogram | null;
  duration: Histogram | null;
  unknown: Sample[];
};

const COUNTERS = {
  accepted: "ignis_requests_accepted_total",
  completed: "ignis_requests_completed_total",
  cancelled: "ignis_requests_cancelled_total",
  generatedTokens: "ignis_generated_tokens_total",
  decodedTokens: "ignis_decoded_tokens_total",
  kvEvictions: "ignis_kv_cache_evictions_total",
  prefixReusedTokens: "ignis_prefix_reused_tokens_total",
} as const;

const KNOWN = new Set<string>([
  "ignis_build_info",
  "ignis_scheduler_requests",
  "ignis_requests_rejected_total",
  "ignis_request_ttft_seconds",
  "ignis_request_duration_seconds",
  ...Object.values(COUNTERS),
]);

export function emptySnapshot(): Snapshot {
  return {
    version: null,
    waiting: null,
    running: null,
    accepted: null,
    completed: null,
    cancelled: null,
    rejected: { full: null, unknown_model: null, oversized: null },
    generatedTokens: null,
    decodedTokens: null,
    kvEvictions: null,
    prefixReusedTokens: null,
    ttft: null,
    duration: null,
    unknown: [],
  };
}

export function readSnapshot({ families }: Exposition): Snapshot {
  const samples = (name: string) => families.get(name)?.samples ?? [];
  const valueOf = (name: string, label?: [string, string]) =>
    samples(name).find((s) => (label ? s.labels[label[0]] === label[1] : Object.keys(s.labels).length === 0))?.value ?? null;

  const snap = emptySnapshot();
  snap.version = samples("ignis_build_info")[0]?.labels.version ?? null;
  snap.waiting = valueOf("ignis_scheduler_requests", ["state", "waiting"]);
  snap.running = valueOf("ignis_scheduler_requests", ["state", "running"]);
  for (const [key, name] of Object.entries(COUNTERS) as [keyof typeof COUNTERS, string][]) snap[key] = valueOf(name);
  for (const reason of REJECT_REASONS) snap.rejected[reason] = valueOf("ignis_requests_rejected_total", ["reason", reason]);
  snap.ttft = readHistogram(samples("ignis_request_ttft_seconds"), "ignis_request_ttft_seconds");
  snap.duration = readHistogram(samples("ignis_request_duration_seconds"), "ignis_request_duration_seconds");
  snap.unknown = [...families.values()].flatMap((f) => f.samples.filter((s) => !inContract(f.name, s)));
  return snap;
}

/** Whether a sample is one ADR 0017's table defines — its metric and, where labelled, a label value it lists. */
function inContract(family: string, s: Sample): boolean {
  if (!KNOWN.has(family)) return false;
  switch (family) {
    case "ignis_build_info":
      return true;
    case "ignis_scheduler_requests":
      return s.labels.state === "waiting" || s.labels.state === "running";
    case "ignis_requests_rejected_total":
      return (REJECT_REASONS as readonly string[]).includes(s.labels.reason);
    case "ignis_request_ttft_seconds":
    case "ignis_request_duration_seconds":
      return true;
    default:
      return Object.keys(s.labels).length === 0;
  }
}

/** Every counter in a scrape, in a fixed order — what a restart check compares. */
export function counterValues(s: Snapshot): (number | null)[] {
  return [
    ...Object.keys(COUNTERS).map((key) => s[key as keyof typeof COUNTERS]),
    ...REJECT_REASONS.map((r) => s.rejected[r]),
    s.ttft?.count ?? null,
    s.duration?.count ?? null,
  ];
}

function readHistogram(samples: Sample[], name: string): Histogram | null {
  const buckets = samples
    .filter((s) => s.name === `${name}_bucket` && s.labels.le !== undefined)
    .map((s) => ({ le: s.labels.le === "+Inf" ? Infinity : Number(s.labels.le), count: s.value }))
    .filter((b) => !Number.isNaN(b.le))
    .sort((a, b) => a.le - b.le);
  if (buckets.length === 0 || buckets.at(-1)?.le !== Infinity) return null;
  const count = samples.find((s) => s.name === `${name}_count`)?.value ?? buckets.at(-1)!.count;
  return {
    bounds: buckets.map((b) => b.le),
    cumulative: buckets.map((b) => b.count),
    sum: samples.find((s) => s.name === `${name}_sum`)?.value ?? 0,
    count,
  };
}
