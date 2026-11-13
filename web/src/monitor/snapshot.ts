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
 * The plan's fourteen lines (GitHub #339: `trellis_workspace`, the 27B
 * trellis body's, 0 on the served artifact and absent from a server that
 * predates it; GitHub #509: `trellis_dense`, the dense prefill's one buffer,
 * 0 unless the load runs it and absent from a server that predates it), then Flash-Next's two (GitHub #306: `residency`,
 * the expert residency's fixed bytes, and `expert_cache`, the VRAM cache of
 * experts; 0 on the 27B, absent from a server that predates them), in the order `VramLines::entries()` lays them out
 * — which is the order the load itself reserves them in, so the panel reads
 * down the plan rather than down an alphabet — then the image model's
 * `condition_budget` (GitHub #324: 0 on a text load, absent from a server
 * that predates it). The image model's weights are `weights` and its arena
 * `workspace`.
 */
export const VRAM_LINES = [
  "weights",
  "cuda_context",
  "workspace",
  "trellis_workspace",
  "trellis_dense",
  "media_embedding",
  "sampling",
  "decode_graph",
  "verify_round",
  "drafter_round",
  "lane_state",
  "retained_slots",
  "hq_residual_window",
  "residual",
  "residency",
  "expert_cache",
  "condition_budget",
] as const;
export type VramLine = (typeof VRAM_LINES)[number];

/**
 * Where retained state lives: on the device, spilled to the pinned host arena,
 * or written to disk. The disk rows exist only on a load with the KV-disk tier
 * (spec vram-budget/03): a scrape without them leaves that tier `null`.
 */
export const RETAINED_TIERS = ["device", "kv_ram", "disk"] as const;
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

/** How a live sequence reached the disk tier: straight from the device, or demoted from KV-RAM. */
export const KV_DISK_SPILL_SOURCES = ["device", "kv_ram"] as const;
export type KvDiskSpillSource = (typeof KV_DISK_SPILL_SOURCES)[number];

/** The KV-disk operations that can be refused: a write (margin or IO error), or a restore read (torn, corrupt, foreign file). */
export const KV_DISK_OPS = ["write", "read"] as const;
export type KvDiskOp = (typeof KV_DISK_OPS)[number];

/**
 * The KV-disk tier (spec vram-budget/03, ADR 0017): its budget and what the
 * committed files hold, the live sequences spilled into it and the operations
 * it refused. Null on a load without the tier, where the server renders none
 * of these series.
 */
export type KvDisk = {
  capacity: number | null;
  used: number | null;
  spills: Record<KvDiskSpillSource, number | null>;
  failures: Record<KvDiskOp, number | null>;
};

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
  /** The bytes of the condition prefixes the image model's jobs hold now, against the `condition_budget` line (GitHub #324). */
  conditionBudgetUsedBytes: number | null;
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

/** The draft positions a verify round can carry (MAX_DRAFT_TOKENS): the per-position families' labels "1".."7". */
export const DRAFT_POSITIONS = 7;

/**
 * Speculative decoding (GitHub #307): verify rounds, drafts verified and
 * committed, and per draft position how many rounds proposed it and committed
 * it. Absent until a request that ran verify rounds ended.
 */
export type Speculation = {
  rounds: number | null;
  drafted: number | null;
  accepted: number | null;
  positionDrafted: (number | null)[];
  positionAccepted: (number | null)[];
};

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
  /** Seconds the expert kernels waited on their demand copies (GitHub #306; null from a server that does not time them). */
  stall: Record<ExpertPhase, number | null>;
  /** The LRU's slots per class: reserved at load, a pinned hot set's not among them (ADR 0055), and holding a projection now (null from a server without occupancy). */
  slots: Record<ExpertClass, { capacity: number | null; inUse: number | null }>;
};

/** Flash-Next's n-gram rows (GitHub #302): staged by source, and the file reads behind them. */
export type Ngram = {
  rows: Record<NgramSource, number | null>;
  reads: number | null;
  readBytes: number | null;
};

/** How an image job ended (GitHub #324): `ignis_image_jobs_total`'s `outcome` label. */
export const IMAGE_OUTCOMES = ["completed", "cancelled", "stopped", "refused", "failed", "timed_out"] as const;
export type ImageOutcome = (typeof IMAGE_OUTCOMES)[number];

/** Where an image job's time goes (GitHub #324): `ignis_image_job_phase_seconds_total`'s `phase` label. */
export const IMAGE_PHASES = ["queued", "encoding", "denoising", "decoding"] as const;
export type ImagePhase = (typeof IMAGE_PHASES)[number];

/**
 * The image jobs (GitHub #324, ADR 0017's amendment of 2026-10-15): by
 * outcome, the images made, the seconds each phase took, the duration of the
 * completed jobs and the queue. Absent until the image model has admitted a
 * job, so null on a text-only server.
 */
export type ImageJobs = {
  jobs: Record<ImageOutcome, number | null>;
  images: number | null;
  phaseSeconds: Record<ImagePhase, number | null>;
  duration: Histogram | null;
  waiting: number | null;
  running: number | null;
};

/** How a transcription ended (spec speech/02, ADR 0017's amendment of 2026-10-24): `ignis_speech_transcriptions_total`'s `outcome` label. */
export const SPEECH_OUTCOMES = ["completed", "cancelled", "failed", "timed_out"] as const;
export type SpeechOutcome = (typeof SPEECH_OUTCOMES)[number];

/** Where a transcription ran: `ignis_speech_transcriptions_total`'s `executor` label. */
export const SPEECH_EXECUTORS = ["cpu", "gpu_loan"] as const;
export type SpeechExecutor = (typeof SPEECH_EXECUTORS)[number];

/**
 * The speech model's transcriptions (spec speech/02): by executor and
 * outcome, the seconds of audio covered, the wait from commit to transcript
 * of the completed ones, and the sessions and queue as the worker last set
 * them. Absent until the speech model was first asked, so null on a server
 * that never transcribed.
 */
export type SpeechTranscriptions = {
  transcriptions: Record<SpeechExecutor, Record<SpeechOutcome, number | null>>;
  audioSeconds: number | null;
  wait: Histogram | null;
  sessionsOpen: number | null;
  queueLength: number | null;
};

/** What the loop guard did about a loop (GitHub #330): `ignis_loop_guard_loops_total`'s `outcome` label, in the order the card reads them. */
export const LOOP_OUTCOMES = ["rewound", "closed", "detected", "too_little_kept", "no_room", "no_tokenizer", "submit_refused"] as const;
export type LoopOutcome = (typeof LOOP_OUTCOMES)[number];

/**
 * The loop guard's tallies (GitHub #330, ADR 0017's amendment): the requests
 * it watched and those that looped, every loop by what it did about it, the
 * tokens its rewinds took back, and the lowest loop score each watched request
 * reached. The server writes the whole family at the first watched request,
 * zeros included, and none of it with the guard off, so null on such a load.
 */
export type LoopGuard = {
  watched: number | null;
  looped: number | null;
  loops: Record<LoopOutcome, number | null>;
  rewoundTokens: number | null;
  /** Null until the family has a histogram to read. */
  minRatio: Histogram | null;
};

/** The types a `response_format` can ask for (#593): `ignis_structured_*`'s `type` label. */
export const STRUCTURED_TYPES = ["json_object", "json_schema"] as const;
export type StructuredType = (typeof STRUCTURED_TYPES)[number];

/**
 * Structured output (#593, ADR 0017's amendment of 2026-11-13): the completed
 * requests held to a grammar, the draws it rejected and the repairs that
 * answered them, by type, and how long a repair took. Absent until the first
 * such request ended, so null on a server that never served one.
 */
export type StructuredOutput = {
  requests: Record<StructuredType, number | null>;
  misses: Record<StructuredType, number | null>;
  repairs: Record<StructuredType, number | null>;
  repairTime: Histogram | null;
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
  /** The KV-disk tier (spec vram-budget/03); null on a load without it. */
  kvDisk: KvDisk | null;
  memory: Memory;
  /** Null on a load without an expert cache: a 27B load. */
  experts: ExpertResidency | null;
  /** Null on a load without an n-gram table: a 27B load. */
  ngram: Ngram | null;
  /** Null until a request that ran verify rounds ended, or on a load without speculation. */
  speculation: Speculation | null;
  /** Null until the image model admitted a job: a text-only server. */
  images: ImageJobs | null;
  /** Null until the speech model was asked for a transcription. */
  speech: SpeechTranscriptions | null;
  /** Null until the guard watched a request, or with `loop_guard: off`. */
  loopGuard: LoopGuard | null;
  /** Null until a request held to a grammar ended. */
  structured: StructuredOutput | null;
  /** The thinking budget's own closes (GitHub #330); null until the first. Not the guard's `closed`. */
  thinkingForcedCloses: number | null;
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
  conditionBudgetUsedBytes: "ignis_condition_budget_used_bytes",
} as const;

/** The expert residency families (GitHub #301). */
const EXPERT_FAMILIES = {
  hits: "ignis_expert_cache_hits_total",
  misses: "ignis_expert_cache_misses_total",
  prefetchIssued: "ignis_expert_prefetches_issued_total",
  prefetchUsed: "ignis_expert_prefetches_used_total",
  bytesMoved: "ignis_expert_bytes_moved_total",
  stall: "ignis_expert_residency_stall_seconds_total",
  slots: "ignis_expert_cache_slots",
} as const;

/** The n-gram families (GitHub #302). */
const NGRAM_FAMILIES = {
  rows: "ignis_ngram_rows_total",
  reads: "ignis_ngram_reads_total",
  readBytes: "ignis_ngram_read_bytes_total",
} as const;

/** The speculative families (GitHub #307). */
const SPEC_FAMILIES = {
  rounds: "ignis_speculative_rounds_total",
  drafted: "ignis_speculative_drafted_tokens_total",
  accepted: "ignis_speculative_accepted_tokens_total",
  positionDrafted: "ignis_speculative_position_drafted_total",
  positionAccepted: "ignis_speculative_position_accepted_total",
} as const;

/** The image families (GitHub #324). */
const IMAGE_FAMILIES = {
  jobs: "ignis_image_jobs_total",
  images: "ignis_image_images_total",
  duration: "ignis_image_job_duration_seconds",
  phaseSeconds: "ignis_image_job_phase_seconds_total",
  queue: "ignis_image_queue_jobs",
} as const;

/** The speech families (spec speech/02). */
const SPEECH_FAMILIES = {
  transcriptions: "ignis_speech_transcriptions_total",
  audioSeconds: "ignis_speech_audio_seconds_total",
  wait: "ignis_speech_commit_wait_seconds",
  sessions: "ignis_speech_sessions_open",
  queue: "ignis_speech_queue_length",
} as const;

/** The loop guard's families (GitHub #330). */
const LOOP_FAMILIES = {
  watched: "ignis_loop_guard_watched_requests_total",
  looped: "ignis_loop_guard_looped_requests_total",
  loops: "ignis_loop_guard_loops_total",
  rewoundTokens: "ignis_loop_guard_rewound_tokens_total",
  minRatio: "ignis_loop_guard_min_ratio",
} as const;

/** The structured-output families (#593). */
const STRUCTURED_FAMILIES = {
  requests: "ignis_structured_requests_total",
  misses: "ignis_structured_misses_total",
  repairs: "ignis_structured_repairs_total",
  repairTime: "ignis_structured_repair_seconds",
} as const;

/** The thinking budget's forced closes (GitHub #330): absent until the first. */
const THINKING_FORCED_CLOSES = "ignis_thinking_forced_closes_total";

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
  "ignis_kv_disk_bytes",
  "ignis_kv_disk_spills_total",
  "ignis_kv_disk_failures_total",
  "ignis_retained_slots",
  "ignis_retained_slot_skips_total",
  ...Object.values(COUNTERS),
  ...Object.values(MEMORY_GAUGES),
  ...Object.values(RETAINED_HOST),
  ...Object.values(RETAINED_FAMILIES),
  ...Object.values(EXPERT_FAMILIES),
  ...Object.values(NGRAM_FAMILIES),
  ...Object.values(SPEC_FAMILIES),
  ...Object.values(IMAGE_FAMILIES),
  ...Object.values(SPEECH_FAMILIES),
  ...Object.values(LOOP_FAMILIES),
  ...Object.values(STRUCTURED_FAMILIES),
  THINKING_FORCED_CLOSES,
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
    kvDisk: null,
    memory: {
      reserved: Object.fromEntries(VRAM_LINES.map((line) => [line, null])) as Record<VramLine, number | null>,
      budgetBytes: null,
      kvPoolPages: null,
      kvPageBytes: null,
      kvPoolUsedPages: null,
      conditionBudgetUsedBytes: null,
      kvRamArena: { capacity: null, used: null },
      retainedSlots: { capacity: null, inUse: null },
      retainedHost: { slots: null, bytes: null },
      slotSkips: Object.fromEntries(SLOT_SKIP_REASONS.map((reason) => [reason, null])) as Record<SlotSkipReason, number | null>,
    },
    experts: null,
    ngram: null,
    speculation: null,
    images: null,
    speech: null,
    loopGuard: null,
    structured: null,
    thinkingForcedCloses: null,
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

  // The KV-disk tier, all or nothing: a load with it renders its series from the first scrape, a load without renders none.
  if (families.has("ignis_kv_disk_bytes")) {
    snap.kvDisk = {
      capacity: valueOf("ignis_kv_disk_bytes", ["state", "capacity"]),
      used: valueOf("ignis_kv_disk_bytes", ["state", "used"]),
      spills: { device: valueOf("ignis_kv_disk_spills_total", ["from", "device"]), kv_ram: valueOf("ignis_kv_disk_spills_total", ["from", "kv_ram"]) },
      failures: { write: valueOf("ignis_kv_disk_failures_total", ["op", "write"]), read: valueOf("ignis_kv_disk_failures_total", ["op", "read"]) },
    };
  }

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
      stall: { decode: valueOf(EXPERT_FAMILIES.stall, ["phase", "decode"]), prefill: valueOf(EXPERT_FAMILIES.stall, ["phase", "prefill"]) },
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

  if (families.has(SPEC_FAMILIES.rounds)) {
    const positions = (family: string) =>
      Array.from({ length: DRAFT_POSITIONS }, (_, j) => valueOf(family, ["position", String(j + 1)]));
    snap.speculation = {
      rounds: valueOf(SPEC_FAMILIES.rounds),
      drafted: valueOf(SPEC_FAMILIES.drafted),
      accepted: valueOf(SPEC_FAMILIES.accepted),
      positionDrafted: positions(SPEC_FAMILIES.positionDrafted),
      positionAccepted: positions(SPEC_FAMILIES.positionAccepted),
    };
  }

  // The image jobs, all or nothing: the first admitted job renders every series.
  if (families.has(IMAGE_FAMILIES.jobs)) {
    snap.images = {
      jobs: Object.fromEntries(IMAGE_OUTCOMES.map((outcome) => [outcome, valueOf(IMAGE_FAMILIES.jobs, ["outcome", outcome])])) as ImageJobs["jobs"],
      images: valueOf(IMAGE_FAMILIES.images),
      phaseSeconds: Object.fromEntries(IMAGE_PHASES.map((phase) => [phase, valueOf(IMAGE_FAMILIES.phaseSeconds, ["phase", phase])])) as ImageJobs["phaseSeconds"],
      duration: readHistogram(samples(IMAGE_FAMILIES.duration), IMAGE_FAMILIES.duration),
      waiting: valueOf(IMAGE_FAMILIES.queue, ["state", "waiting"]),
      running: valueOf(IMAGE_FAMILIES.queue, ["state", "running"]),
    };
  }

  // The speech model, all or nothing: the first transcription renders every series.
  if (families.has(SPEECH_FAMILIES.transcriptions)) {
    snap.speech = {
      transcriptions: Object.fromEntries(
        SPEECH_EXECUTORS.map((executor) => [
          executor,
          Object.fromEntries(SPEECH_OUTCOMES.map((outcome) => [outcome, samples(SPEECH_FAMILIES.transcriptions).find((s) => s.labels.executor === executor && s.labels.outcome === outcome)?.value ?? null])),
        ]),
      ) as SpeechTranscriptions["transcriptions"],
      audioSeconds: valueOf(SPEECH_FAMILIES.audioSeconds),
      wait: readHistogram(samples(SPEECH_FAMILIES.wait), SPEECH_FAMILIES.wait),
      sessionsOpen: valueOf(SPEECH_FAMILIES.sessions),
      queueLength: valueOf(SPEECH_FAMILIES.queue),
    };
  }

  // The loop guard, all or nothing: the first watched request renders every series.
  if (families.has(LOOP_FAMILIES.watched)) {
    snap.loopGuard = {
      watched: valueOf(LOOP_FAMILIES.watched),
      looped: valueOf(LOOP_FAMILIES.looped),
      loops: Object.fromEntries(LOOP_OUTCOMES.map((outcome) => [outcome, valueOf(LOOP_FAMILIES.loops, ["outcome", outcome])])) as LoopGuard["loops"],
      rewoundTokens: valueOf(LOOP_FAMILIES.rewoundTokens),
      minRatio: readHistogram(samples(LOOP_FAMILIES.minRatio), LOOP_FAMILIES.minRatio),
    };
  }
  // Structured output, all or nothing: the first request held to a grammar renders every series.
  if (families.has(STRUCTURED_FAMILIES.requests)) {
    const byType = (family: string) =>
      Object.fromEntries(STRUCTURED_TYPES.map((type) => [type, valueOf(family, ["type", type])])) as Record<StructuredType, number | null>;
    snap.structured = {
      requests: byType(STRUCTURED_FAMILIES.requests),
      misses: byType(STRUCTURED_FAMILIES.misses),
      repairs: byType(STRUCTURED_FAMILIES.repairs),
      repairTime: readHistogram(samples(STRUCTURED_FAMILIES.repairTime), STRUCTURED_FAMILIES.repairTime),
    };
  }
  snap.thinkingForcedCloses = valueOf(THINKING_FORCED_CLOSES);

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
    case IMAGE_FAMILIES.duration:
    case SPEECH_FAMILIES.wait:
    case LOOP_FAMILIES.minRatio:
    case STRUCTURED_FAMILIES.repairTime:
      return true;
    case STRUCTURED_FAMILIES.requests:
    case STRUCTURED_FAMILIES.misses:
    case STRUCTURED_FAMILIES.repairs:
      return oneOf(STRUCTURED_TYPES, s.labels.type) && Object.keys(s.labels).length === 1;
    case LOOP_FAMILIES.loops:
      return oneOf(LOOP_OUTCOMES, s.labels.outcome) && Object.keys(s.labels).length === 1;
    case IMAGE_FAMILIES.jobs:
      return oneOf(IMAGE_OUTCOMES, s.labels.outcome) && Object.keys(s.labels).length === 1;
    case IMAGE_FAMILIES.phaseSeconds:
      return oneOf(IMAGE_PHASES, s.labels.phase) && Object.keys(s.labels).length === 1;
    case SPEECH_FAMILIES.transcriptions:
      return oneOf(SPEECH_EXECUTORS, s.labels.executor) && oneOf(SPEECH_OUTCOMES, s.labels.outcome) && Object.keys(s.labels).length === 2;
    case IMAGE_FAMILIES.queue:
      return (s.labels.state === "waiting" || s.labels.state === "running") && Object.keys(s.labels).length === 1;
    case "ignis_decisions_total":
      return oneOf(DECISION_TYPES, s.labels.type);
    case "ignis_vram_reserved_bytes":
      return oneOf(VRAM_LINES, s.labels.line);
    case "ignis_kv_ram_arena_bytes":
      return s.labels.state === "capacity" || s.labels.state === "used";
    case "ignis_kv_disk_bytes":
      return s.labels.state === "capacity" || s.labels.state === "used";
    case "ignis_kv_disk_spills_total":
      return oneOf(KV_DISK_SPILL_SOURCES, s.labels.from) && Object.keys(s.labels).length === 1;
    case "ignis_kv_disk_failures_total":
      return oneOf(KV_DISK_OPS, s.labels.op) && Object.keys(s.labels).length === 1;
    case "ignis_retained_slots":
      return s.labels.state === "capacity" || s.labels.state === "in_use";
    case "ignis_retained_slot_skips_total":
      return oneOf(SLOT_SKIP_REASONS, s.labels.reason);
    case EXPERT_FAMILIES.hits:
    case EXPERT_FAMILIES.misses:
      return oneOf(EXPERT_CLASSES, s.labels.class) && oneOf(EXPERT_PHASES, s.labels.phase) && Object.keys(s.labels).length === 2;
    case EXPERT_FAMILIES.bytesMoved:
    case EXPERT_FAMILIES.stall:
      return oneOf(EXPERT_PHASES, s.labels.phase) && Object.keys(s.labels).length === 1;
    case EXPERT_FAMILIES.slots:
      return oneOf(EXPERT_CLASSES, s.labels.class) && (s.labels.state === "capacity" || s.labels.state === "in_use") && Object.keys(s.labels).length === 2;
    case NGRAM_FAMILIES.rows:
      return oneOf(NGRAM_SOURCES, s.labels.source) && Object.keys(s.labels).length === 1;
    case SPEC_FAMILIES.positionDrafted:
    case SPEC_FAMILIES.positionAccepted: {
      const position = Number(s.labels.position);
      return Number.isInteger(position) && position >= 1 && position <= DRAFT_POSITIONS && Object.keys(s.labels).length === 1;
    }
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
    ...KV_DISK_SPILL_SOURCES.map((source) => s.kvDisk?.spills[source] ?? null),
    ...KV_DISK_OPS.map((op) => s.kvDisk?.failures[op] ?? null),
    ...EXPERT_CLASSES.flatMap((cls) => EXPERT_PHASES.flatMap((phase) => [s.experts?.hits[cls][phase] ?? null, s.experts?.misses[cls][phase] ?? null])),
    s.experts?.prefetchIssued ?? null,
    s.experts?.prefetchUsed ?? null,
    ...EXPERT_PHASES.map((phase) => s.experts?.bytesMoved[phase] ?? null),
    ...EXPERT_PHASES.map((phase) => s.experts?.stall[phase] ?? null),
    ...NGRAM_SOURCES.map((source) => s.ngram?.rows[source] ?? null),
    s.ngram?.reads ?? null,
    s.ngram?.readBytes ?? null,
    s.speculation?.rounds ?? null,
    s.speculation?.drafted ?? null,
    s.speculation?.accepted ?? null,
    ...IMAGE_OUTCOMES.map((outcome) => s.images?.jobs[outcome] ?? null),
    s.images?.images ?? null,
    ...IMAGE_PHASES.map((phase) => s.images?.phaseSeconds[phase] ?? null),
    s.images?.duration?.count ?? null,
    ...SPEECH_EXECUTORS.flatMap((executor) => SPEECH_OUTCOMES.map((outcome) => s.speech?.transcriptions[executor][outcome] ?? null)),
    s.speech?.audioSeconds ?? null,
    s.speech?.wait?.count ?? null,
    s.loopGuard?.watched ?? null,
    s.loopGuard?.looped ?? null,
    ...LOOP_OUTCOMES.map((outcome) => s.loopGuard?.loops[outcome] ?? null),
    s.loopGuard?.rewoundTokens ?? null,
    s.loopGuard?.minRatio?.count ?? null,
    ...STRUCTURED_TYPES.flatMap((type) => [s.structured?.requests[type] ?? null, s.structured?.misses[type] ?? null, s.structured?.repairs[type] ?? null]),
    s.structured?.repairTime?.count ?? null,
    s.thinkingForcedCloses,
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
