import type { Exposition, Sample } from "./exposition.ts";

// One scrape read against ADR 0017's contract table, widened by ADR 0030
// §Observability: the series the panel knows how to draw, by name. Anything
// else ignis serves is kept in `unknown` and listed raw, never dropped
// silently.
//
// The decision family (#241) is in the contract but has no panel yet: it is
// recognised here so the page does not badge it "not in ADR 0017" — which
// would be a false statement about a series the ADR lists — and it is
// otherwise not read into the snapshot. Drawing it is the Monitor's own
// work. Note it is absent from a load that has served no decision, so a
// panel must treat it as missing rather than zero.

export const REJECT_REASONS = ["full", "unknown_model", "oversized"] as const;
export type RejectReason = (typeof REJECT_REASONS)[number];

/**
 * The plan's twelve lines, in the order `VramLines::entries()` lays them out
 * — which is the order the load itself reserves them in, so the panel reads
 * down the plan rather than down an alphabet.
 */
export const VRAM_LINES = [
  "weights",
  "cuda_context",
  "workspace",
  "media_embedding",
  "sampling",
  "decode_graph",
  "verify_round",
  "drafter_round",
  "lane_state",
  "retained_slots",
  "hq_residual_window",
  "residual",
] as const;
export type VramLine = (typeof VRAM_LINES)[number];

/** Where retained state lives: on the device, or spilled to the pinned host arena. */
export const RETAINED_TIERS = ["device", "kv_ram"] as const;
export type RetainedTier = (typeof RETAINED_TIERS)[number];

/** What was retained: a prompt checkpoint, or a shared prefix (GitHub #216). */
export const RETAINED_KINDS = ["checkpoint", "prefix"] as const;
export type RetainedKind = (typeof RETAINED_KINDS)[number];

/** The six retained-state families, keyed by the word the panel calls them by. */
export const RETAINED_FAMILIES = {
  reusedTokens: "ignis_retained_reused_tokens_total",
  hits: "ignis_retained_state_hits_total",
  misses: "ignis_retained_state_misses_total",
  spills: "ignis_retained_state_spills_total",
  discards: "ignis_retained_state_discards_total",
  restores: "ignis_retained_state_restores_total",
} as const;
export type RetainedFamily = keyof typeof RETAINED_FAMILIES;
export const RETAINED_FAMILY_KEYS = Object.keys(RETAINED_FAMILIES) as RetainedFamily[];

/** Why a publish or a capture left no reuse behind (ADR 0030 §Observability). */
export const SLOT_SKIP_REASONS = ["publish_skipped_no_slot", "capture_skipped_no_slot", "capture_skipped_no_page"] as const;
export type SlotSkipReason = (typeof SLOT_SKIP_REASONS)[number];

/** One retained-state family: a count per tier and kind. */
export type RetainedMatrix = Record<RetainedTier, Record<RetainedKind, number | null>>;

/**
 * What this load reserved and what is occupied of it. The constants are read
 * once at load and never move; the live figures move with the load, and each
 * is only meaningful against the constant beside it.
 */
export type Memory = {
  /** The plan's lines, in plan order. */
  reserved: Record<VramLine, number | null>;
  budgetBytes: number | null;
  kvPoolPages: number | null;
  kvPageBytes: number | null;
  kvPoolUsedPages: number | null;
  kvRamArena: { capacity: number | null; used: number | null };
  retainedSlots: { capacity: number | null; inUse: number | null };
  /** Of the retained slots, those in the pinned host block, and the block (GitHub #281). */
  retainedHost: { slots: number | null; bytes: number | null };
  slotSkips: Record<SlotSkipReason, number | null>;
};

/**
 * Flash-Next's eight K classes of expert projection (GitHub #301), in the
 * order the server writes them: the `class` label's whole vocabulary.
 */
export const EXPERT_CLASSES = ["gate_up_k2", "gate_up_k2_5", "gate_up_k3", "gate_up_k4", "down_k2", "down_k2_5", "down_k3", "down_k4"] as const;
export type ExpertClass = (typeof EXPERT_CLASSES)[number];

/** The step an expert projection was selected by. */
export const EXPERT_PHASES = ["decode", "prefill"] as const;
export type ExpertPhase = (typeof EXPERT_PHASES)[number];

/** Where an n-gram row came from: the RAM hot-row cache, or the artifact file (GitHub #302). */
export const NGRAM_SOURCES = ["hot", "file"] as const;
export type NgramSource = (typeof NGRAM_SOURCES)[number];

/** One expert-cache family: a count per K class and phase. */
export type ClassPhaseMatrix = Record<ExpertClass, Record<ExpertPhase, number | null>>;

/**
 * Flash-Next's expert residency (GitHub #301, ADR 0017's amendment of
 * 2026-10-05): what the VRAM expert cache served, what it copied in, and
 * its slots. Only a Flash-Next load exports it.
 */
export type ExpertResidency = {
  hits: ClassPhaseMatrix;
  misses: ClassPhaseMatrix;
  prefetchIssued: number | null;
  prefetchUsed: number | null;
  bytesMoved: Record<ExpertPhase, number | null>;
  /** Slots per class: reserved at load, and holding a projection now (null from a server without occupancy). */
  slots: Record<ExpertClass, { capacity: number | null; inUse: number | null }>;
};

/** Flash-Next's n-gram rows (GitHub #302): staged by source, and the file reads behind them. */
export type Ngram = {
  rows: Record<NgramSource, number | null>;
  reads: number | null;
  readBytes: number | null;
};

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
  /** Live snapshots dropped **out of** KV-RAM (GitHub #224). Null from a server without it. */
  kvRamEvictions: number | null;
  prefixReusedTokens: number | null;
  /** The six retained-state families, each split by tier and kind (GitHub #190, #216). */
  retained: Record<RetainedFamily, RetainedMatrix>;
  memory: Memory;
  /** Null on a load without an expert cache: a 27B load. */
  experts: ExpertResidency | null;
  /** Null on a load without an n-gram table: a 27B load. */
  ngram: Ngram | null;
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
  kvRamEvictions: "ignis_kv_ram_evictions_total",
  prefixReusedTokens: "ignis_prefix_reused_tokens_total",
} as const;

/** The host retained slots and their pinned block (GitHub #281): no label, read once at load. */
const RETAINED_HOST = {
  slots: "ignis_retained_host_slots",
  bytes: "ignis_retained_host_bytes",
} as const;

/** The memory gauges that carry no label. */
const MEMORY_GAUGES = {
  budgetBytes: "ignis_vram_budget_bytes",
  kvPoolPages: "ignis_kv_pool_pages",
  kvPageBytes: "ignis_kv_page_bytes",
  kvPoolUsedPages: "ignis_kv_pool_used_pages",
} as const;

/** The expert residency families (GitHub #301). */
const EXPERT_FAMILIES = {
  hits: "ignis_expert_cache_hits_total",
  misses: "ignis_expert_cache_misses_total",
  prefetchIssued: "ignis_expert_prefetches_issued_total",
  prefetchUsed: "ignis_expert_prefetches_used_total",
  bytesMoved: "ignis_expert_bytes_moved_total",
  slots: "ignis_expert_cache_slots",
} as const;

/** The n-gram families (GitHub #302). */
const NGRAM_FAMILIES = {
  rows: "ignis_ngram_rows_total",
  reads: "ignis_ngram_reads_total",
  readBytes: "ignis_ngram_read_bytes_total",
} as const;

/** The typed primitives a decision's question asks for (GitHub #241, ADR 0034). */
export const DECISION_TYPES = ["noul", "choice", "score"] as const;
export type DecisionType = (typeof DECISION_TYPES)[number];

const KNOWN = new Set<string>([
  "ignis_build_info",
  "ignis_decisions_total",
  "ignis_decision_answer_mass",
  "ignis_scheduler_requests",
  "ignis_requests_rejected_total",
  "ignis_request_ttft_seconds",
  "ignis_request_duration_seconds",
  "ignis_vram_reserved_bytes",
  "ignis_kv_ram_arena_bytes",
  "ignis_retained_slots",
  "ignis_retained_slot_skips_total",
  ...Object.values(COUNTERS),
  ...Object.values(MEMORY_GAUGES),
  ...Object.values(RETAINED_HOST),
  ...Object.values(RETAINED_FAMILIES),
  ...Object.values(EXPERT_FAMILIES),
  ...Object.values(NGRAM_FAMILIES),
]);

const emptyMatrix = (): RetainedMatrix =>
  Object.fromEntries(RETAINED_TIERS.map((tier) => [tier, Object.fromEntries(RETAINED_KINDS.map((kind) => [kind, null]))])) as RetainedMatrix;

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
    kvRamEvictions: null,
    prefixReusedTokens: null,
    retained: Object.fromEntries(RETAINED_FAMILY_KEYS.map((key) => [key, emptyMatrix()])) as Snapshot["retained"],
    memory: {
      reserved: Object.fromEntries(VRAM_LINES.map((line) => [line, null])) as Record<VramLine, number | null>,
      budgetBytes: null,
      kvPoolPages: null,
      kvPageBytes: null,
      kvPoolUsedPages: null,
      kvRamArena: { capacity: null, used: null },
      retainedSlots: { capacity: null, inUse: null },
      retainedHost: { slots: null, bytes: null },
      slotSkips: Object.fromEntries(SLOT_SKIP_REASONS.map((reason) => [reason, null])) as Record<SlotSkipReason, number | null>,
    },
    experts: null,
    ngram: null,
    ttft: null,
    duration: null,
    unknown: [],
  };
}

const classPhaseMatrix = (read: (cls: ExpertClass, phase: ExpertPhase) => number | null): ClassPhaseMatrix =>
  Object.fromEntries(EXPERT_CLASSES.map((cls) => [cls, Object.fromEntries(EXPERT_PHASES.map((phase) => [phase, read(cls, phase)]))])) as ClassPhaseMatrix;

export function readSnapshot({ families }: Exposition): Snapshot {
  const samples = (name: string) => families.get(name)?.samples ?? [];
  const valueOf = (name: string, label?: [string, string]) =>
    samples(name).find((s) => (label ? s.labels[label[0]] === label[1] : Object.keys(s.labels).length === 0))?.value ?? null;
  const tierKind = (name: string, tier: RetainedTier, kind: RetainedKind) =>
    samples(name).find((s) => s.labels.tier === tier && s.labels.kind === kind)?.value ?? null;

  const snap = emptySnapshot();
  snap.version = samples("ignis_build_info")[0]?.labels.version ?? null;
  snap.waiting = valueOf("ignis_scheduler_requests", ["state", "waiting"]);
  snap.running = valueOf("ignis_scheduler_requests", ["state", "running"]);
  for (const [key, name] of Object.entries(COUNTERS) as [keyof typeof COUNTERS, string][]) snap[key] = valueOf(name);
  for (const reason of REJECT_REASONS) snap.rejected[reason] = valueOf("ignis_requests_rejected_total", ["reason", reason]);

  for (const key of RETAINED_FAMILY_KEYS) {
    const name = RETAINED_FAMILIES[key];
    for (const tier of RETAINED_TIERS) for (const kind of RETAINED_KINDS) snap.retained[key][tier][kind] = tierKind(name, tier, kind);
  }

  const mem = snap.memory;
  for (const line of VRAM_LINES) mem.reserved[line] = valueOf("ignis_vram_reserved_bytes", ["line", line]);
  for (const [key, name] of Object.entries(MEMORY_GAUGES) as [keyof typeof MEMORY_GAUGES, string][]) mem[key] = valueOf(name);
  mem.kvRamArena = {
    capacity: valueOf("ignis_kv_ram_arena_bytes", ["state", "capacity"]),
    used: valueOf("ignis_kv_ram_arena_bytes", ["state", "used"]),
  };
  mem.retainedSlots = {
    capacity: valueOf("ignis_retained_slots", ["state", "capacity"]),
    inUse: valueOf("ignis_retained_slots", ["state", "in_use"]),
  };
  mem.retainedHost = { slots: valueOf(RETAINED_HOST.slots), bytes: valueOf(RETAINED_HOST.bytes) };
  for (const reason of SLOT_SKIP_REASONS) mem.slotSkips[reason] = valueOf("ignis_retained_slot_skips_total", ["reason", reason]);

  // Flash-Next's families, all or nothing per table: a 27B load renders none.
  if (families.has(EXPERT_FAMILIES.hits)) {
    const byLabels = (name: string, labels: Record<string, string>) =>
      samples(name).find((s) => Object.entries(labels).every(([k, v]) => s.labels[k] === v))?.value ?? null;
    snap.experts = {
      hits: classPhaseMatrix((cls, phase) => byLabels(EXPERT_FAMILIES.hits, { class: cls, phase })),
      misses: classPhaseMatrix((cls, phase) => byLabels(EXPERT_FAMILIES.misses, { class: cls, phase })),
      prefetchIssued: valueOf(EXPERT_FAMILIES.prefetchIssued),
      prefetchUsed: valueOf(EXPERT_FAMILIES.prefetchUsed),
      bytesMoved: { decode: valueOf(EXPERT_FAMILIES.bytesMoved, ["phase", "decode"]), prefill: valueOf(EXPERT_FAMILIES.bytesMoved, ["phase", "prefill"]) },
      slots: Object.fromEntries(
        EXPERT_CLASSES.map((cls) => [
          cls,
          { capacity: byLabels(EXPERT_FAMILIES.slots, { class: cls, state: "capacity" }), inUse: byLabels(EXPERT_FAMILIES.slots, { class: cls, state: "in_use" }) },
        ]),
      ) as ExpertResidency["slots"],
    };
  }
  if (families.has(NGRAM_FAMILIES.rows)) {
    snap.ngram = {
      rows: { hot: valueOf(NGRAM_FAMILIES.rows, ["source", "hot"]), file: valueOf(NGRAM_FAMILIES.rows, ["source", "file"]) },
      reads: valueOf(NGRAM_FAMILIES.reads),
      readBytes: valueOf(NGRAM_FAMILIES.readBytes),
    };
  }

  snap.ttft = readHistogram(samples("ignis_request_ttft_seconds"), "ignis_request_ttft_seconds");
  snap.duration = readHistogram(samples("ignis_request_duration_seconds"), "ignis_request_duration_seconds");
  snap.unknown = [...families.values()].flatMap((f) => f.samples.filter((s) => !inContract(f.name, s)));
  return snap;
}

/** Whether a label value is one of a fixed set the contract names. */
const oneOf = <T extends string>(set: readonly T[], value: string | undefined): boolean => (set as readonly string[]).includes(value ?? "");

/** Whether a sample is one the contract defines — its metric and, where labelled, a label value it lists. */
function inContract(family: string, s: Sample): boolean {
  if (!KNOWN.has(family)) return false;
  if ((Object.values(RETAINED_FAMILIES) as string[]).includes(family)) {
    return oneOf(RETAINED_TIERS, s.labels.tier) && oneOf(RETAINED_KINDS, s.labels.kind);
  }
  switch (family) {
    case "ignis_build_info":
      return true;
    case "ignis_scheduler_requests":
      return s.labels.state === "waiting" || s.labels.state === "running";
    case "ignis_requests_rejected_total":
      return oneOf(REJECT_REASONS, s.labels.reason);
    case "ignis_request_ttft_seconds":
    case "ignis_request_duration_seconds":
    case "ignis_decision_answer_mass":
      return true;
    case "ignis_decisions_total":
      return oneOf(DECISION_TYPES, s.labels.type);
    case "ignis_vram_reserved_bytes":
      return oneOf(VRAM_LINES, s.labels.line);
    case "ignis_kv_ram_arena_bytes":
      return s.labels.state === "capacity" || s.labels.state === "used";
    case "ignis_retained_slots":
      return s.labels.state === "capacity" || s.labels.state === "in_use";
    case "ignis_retained_slot_skips_total":
      return oneOf(SLOT_SKIP_REASONS, s.labels.reason);
    case EXPERT_FAMILIES.hits:
    case EXPERT_FAMILIES.misses:
      return oneOf(EXPERT_CLASSES, s.labels.class) && oneOf(EXPERT_PHASES, s.labels.phase) && Object.keys(s.labels).length === 2;
    case EXPERT_FAMILIES.bytesMoved:
      return oneOf(EXPERT_PHASES, s.labels.phase) && Object.keys(s.labels).length === 1;
    case EXPERT_FAMILIES.slots:
      return oneOf(EXPERT_CLASSES, s.labels.class) && (s.labels.state === "capacity" || s.labels.state === "in_use") && Object.keys(s.labels).length === 2;
    case NGRAM_FAMILIES.rows:
      return oneOf(NGRAM_SOURCES, s.labels.source) && Object.keys(s.labels).length === 1;
    default:
      return Object.keys(s.labels).length === 0;
  }
}

/**
 * Every counter in a scrape, in a fixed order — what a restart check
 * compares. The memory gauges are left out on purpose: a gauge falls in
 * ordinary service, and only a fall in something that never falls is
 * evidence the server started again.
 */
export function counterValues(s: Snapshot): (number | null)[] {
  return [
    ...Object.keys(COUNTERS).map((key) => s[key as keyof typeof COUNTERS]),
    ...REJECT_REASONS.map((r) => s.rejected[r]),
    ...RETAINED_FAMILY_KEYS.flatMap((key) => RETAINED_TIERS.flatMap((tier) => RETAINED_KINDS.map((kind) => s.retained[key][tier][kind]))),
    ...SLOT_SKIP_REASONS.map((reason) => s.memory.slotSkips[reason]),
    ...EXPERT_CLASSES.flatMap((cls) => EXPERT_PHASES.flatMap((phase) => [s.experts?.hits[cls][phase] ?? null, s.experts?.misses[cls][phase] ?? null])),
    s.experts?.prefetchIssued ?? null,
    s.experts?.prefetchUsed ?? null,
    ...EXPERT_PHASES.map((phase) => s.experts?.bytesMoved[phase] ?? null),
    ...NGRAM_SOURCES.map((source) => s.ngram?.rows[source] ?? null),
    s.ngram?.reads ?? null,
    s.ngram?.readBytes ?? null,
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
