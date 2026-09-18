// A simulated ignis behind `npm run dev:mock`'s /ui/metrics: requests arrive
// in waves and bursts (an agent fan-out), queue for six lanes, prefill,
// decode, get cancelled or turned away, and the counters and histograms move
// the way the real exposition's do (crates/server/src/metrics.rs, ADR 0017
// and ADR 0030 §Observability, which the memory panel of GitHub #217 reads).
// The simulation advances on each scrape to the scrape's clock, so the
// Monitor has live data without the shared GPU. Development only.

const TTFT_BOUNDS_MS = [50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000];
const DURATION_BOUNDS_MS = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000, 600_000];
const LANES = 6;
const QUEUE_LIMIT = 8;
const STEP_MS = 100;

// What a load reserves, in the shape ADR 0030's plan lays out: eleven lines,
// then the pool the rest of the budget buys. The figures are a 5090-sized
// hq-e8-2b load, rounded; the simulation only needs them to be consistent
// with each other.
const VRAM_BUDGET_BYTES = 31_138_512_896;
const KV_PAGE_BYTES = 1_835_008;
const KV_POOL_PAGES = 4_032;
const KV_PAGE_TOKENS = 256;
const KV_RAM_ARENA_BYTES = 8 * 1024 ** 3;
const RETAINED_SLOTS = 10;
/** One retained image's KV-RAM bytes once it has been spilled. */
const RETAINED_BLOB_BYTES = 228 * 1024 ** 2;
const VRAM_LINES: [string, number][] = [
  ["weights", 17_179_869_184],
  ["cuda_context", 587_202_560],
  ["workspace", 1_342_177_280],
  ["media_embedding", 402_653_184],
  ["sampling", 33_554_432],
  ["decode_graph", 16_777_216],
  ["verify_round", 268_435_456],
  ["drafter_round", 134_217_728],
  ["lane_state", 1_073_741_824],
  ["retained_slots", 2_415_919_104],
  ["residual", 268_435_456],
];
const SKIP_REASONS = ["publish_skipped_no_slot", "capture_skipped_no_slot", "capture_skipped_no_page"] as const;
const RETAINED_FAMILIES: [string, string][] = [
  ["ignis_retained_reused_tokens_total", "Cumulative tokens skipped through retained state, by residency tier."],
  ["ignis_retained_state_hits_total", "Retained state chosen to resume from or brought back, by residency tier."],
  ["ignis_retained_state_misses_total", "First prefill chunks with no retained checkpoint matching in the tier."],
  ["ignis_retained_state_spills_total", "Retained checkpoints and prefixes spilled into the tier."],
  ["ignis_retained_state_discards_total", "Retained checkpoints and prefixes discarded from the tier."],
  ["ignis_retained_state_restores_total", "Retained state restored from the tier."],
];

type Request = {
  arrivedAt: number;
  tokens: number;
  /** The prompt it arrived with: the KV pages it holds start here. */
  promptTokens: number;
  firstAt: number;
  endAt: number;
  cancelAt: number | null;
  sawFirst: boolean;
  tokensPerSecond: number;
  /** Tokens counted as decoded so far. */
  decoded: number;
};

class Histogram {
  buckets: number[];
  sumMs = 0;
  constructor(private bounds: number[]) {
    // One bucket per bound, plus +Inf's own.
    this.buckets = new Array<number>(bounds.length + 1).fill(0);
  }
  observe(ms: number) {
    const rounded = Math.max(0, Math.round(ms));
    const i = this.bounds.findIndex((b) => rounded <= b);
    this.buckets[i === -1 ? this.bounds.length : i]++;
    this.sumMs += rounded;
  }
  render(out: string[], name: string, help: string) {
    out.push(`# HELP ${name} ${help}`, `# TYPE ${name} histogram`);
    let cumulative = 0;
    this.buckets.forEach((count, i) => {
      cumulative += count;
      const le = i < this.bounds.length ? seconds(this.bounds[i]) : "+Inf";
      out.push(`${name}_bucket{le="${le}"} ${cumulative}`);
    });
    out.push(`${name}_sum ${seconds(this.sumMs)}`, `${name}_count ${cumulative}`);
  }
}

/** Milliseconds as seconds without trailing zeros, as the server writes them. */
function seconds(ms: number): string {
  const whole = Math.floor(ms / 1000);
  const frac = ms % 1000;
  return frac === 0 ? String(whole) : `${whole}.${String(frac).padStart(3, "0").replace(/0+$/, "")}`;
}

function gaussian(): number {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

const logNormal = (median: number, spread: number) => median * Math.exp(spread * gaussian());

export function createMetricsSim(start = Date.now()) {
  let clock = start;
  let nextBurst = start + 15_000;
  const waiting: Request[] = [];
  let running: Request[] = [];
  const count = { accepted: 0, completed: 0, cancelled: 0, tokens: 0, decoded: 0, evictions: 0, ramDrops: 0, prefix: 0 };
  const rejected = { full: 0, unknown_model: 0, oversized: 0 };
  const ttft = new Histogram(TTFT_BOUNDS_MS);
  const duration = new Histogram(DURATION_BOUNDS_MS);

  // Retained state: the ledger the Monitor's memory panel reads. A key is
  // `family/tier/kind`, exactly the label pair the exposition carries.
  const retained: Record<string, number> = {};
  const bump = (family: string, tier: string, kind: string, by = 1) => {
    const key = `${family}/${tier}/${kind}`;
    retained[key] = (retained[key] ?? 0) + by;
  };
  const skips: Record<string, number> = { publish_skipped_no_slot: 0, capture_skipped_no_slot: 0, capture_skipped_no_page: 0 };
  /** The slots holding an image, and how many of those were spilled to KV-RAM. */
  const slots = { inUse: 0, spilled: 0 };

  /**
   * One scheduled request's turn with retained state: it looks a checkpoint
   * up, then tries to leave its own behind. A load whose slots are all held
   * cannot, which is the skip counter's whole point.
   */
  function retainedTurn() {
    const roll = Math.random();
    if (roll < 0.3) {
      bump("hits", "device", "checkpoint");
      bump("reusedTokens", "device", "checkpoint", 400 + Math.floor(Math.random() * 3_000));
    } else if (roll < 0.4 && slots.spilled > 0) {
      bump("hits", "kv_ram", "checkpoint");
      bump("restores", "kv_ram", "checkpoint");
      bump("reusedTokens", "kv_ram", "checkpoint", 400 + Math.floor(Math.random() * 3_000));
      slots.spilled--;
    } else {
      bump("misses", "device", "checkpoint");
      if (slots.spilled > 0) bump("misses", "kv_ram", "checkpoint");
    }
    if (Math.random() < 0.25) {
      // A shared prefix published for the siblings behind it.
      if (slots.inUse < RETAINED_SLOTS) {
        slots.inUse++;
        bump("reusedTokens", "device", "prefix", 200 + Math.floor(Math.random() * 1_500));
      } else {
        skips.publish_skipped_no_slot++;
      }
    }
    if (Math.random() < 0.5) {
      // This request's own prompt checkpoint, captured at its opener.
      if (slots.inUse < RETAINED_SLOTS) slots.inUse++;
      else if (Math.random() < 0.7) skips.capture_skipped_no_slot++;
      else skips.capture_skipped_no_page++;
    }
    // Room runs short: the oldest images are spilled to KV-RAM, or dropped.
    while (slots.inUse >= RETAINED_SLOTS && Math.random() < 0.4) {
      const kind = Math.random() < 0.6 ? "checkpoint" : "prefix";
      // An image either moves down a tier or is given up; it is never both.
      if (slots.spilled * RETAINED_BLOB_BYTES + RETAINED_BLOB_BYTES <= KV_RAM_ARENA_BYTES) {
        bump("spills", "kv_ram", kind);
        slots.spilled++;
      } else {
        bump("discards", "device", kind);
      }
      slots.inUse--;
    }
  }

  /** The KV pages a request holds: its prompt and what it has decoded, in whole pages. */
  const pagesOf = (req: Request) => Math.ceil((req.promptTokens + req.decoded) / KV_PAGE_TOKENS);

  function arrive(t: number) {
    const roll = Math.random();
    if (roll < 0.008) return void rejected.unknown_model++;
    if (roll < 0.02) return void rejected.oversized++;
    if (waiting.length >= QUEUE_LIMIT) return void rejected.full++;
    count.accepted++;
    waiting.push({ arrivedAt: t, tokens: Math.max(8, Math.round(logNormal(380, 0.75))), promptTokens: Math.max(64, Math.round(logNormal(2_600, 0.9))), firstAt: 0, endAt: 0, cancelAt: null, sawFirst: false, tokensPerSecond: 0, decoded: 0 });
  }

  function step(t: number) {
    const phase = (t - start) / 1000;
    const perSecond = Math.max(0.02, 0.16 + 0.14 * Math.sin(phase / 70) + 0.06 * Math.sin(phase / 17));
    if (Math.random() < (perSecond * STEP_MS) / 1000) arrive(t);
    if (t >= nextBurst) {
      const fanOut = 4 + Math.floor(Math.random() * 9);
      for (let i = 0; i < fanOut; i++) arrive(t);
      nextBurst = t + 40_000 + Math.random() * 80_000;
    }

    while (running.length < LANES && waiting.length > 0) {
      const req = waiting.shift()!;
      if (Math.random() < 0.45) count.prefix += 512 + Math.floor(Math.random() * 5_500);
      retainedTurn();
      req.firstAt = t + logNormal(320 + 60 * running.length, 0.6);
      req.tokensPerSecond = 58 / (1 + 0.1 * running.length);
      req.endAt = req.firstAt + (req.tokens / req.tokensPerSecond) * 1000;
      req.cancelAt = Math.random() < 0.06 ? t + Math.random() * (req.endAt - t) : null;
      running.push(req);
    }

    running = running.filter((req) => {
      if (t >= req.firstAt) {
        // Decoded tokens count as they stream, like ignis_decoded_tokens_total.
        const due = Math.min(req.tokens, Math.floor(((t - req.firstAt) / 1000) * req.tokensPerSecond) + 1);
        count.decoded += due - req.decoded;
        req.decoded = due;
      }
      if (req.cancelAt !== null && t >= req.cancelAt) {
        count.cancelled++;
        return false;
      }
      if (!req.sawFirst && t >= req.firstAt) {
        req.sawFirst = true;
        ttft.observe(req.firstAt - req.arrivedAt);
      }
      if (t >= req.endAt) {
        count.completed++;
        count.tokens += req.tokens;
        duration.observe(req.endAt - req.arrivedAt);
        return false;
      }
      return true;
    });

    if (running.length === LANES && waiting.length > 2 && Math.random() < 0.004) {
      count.evictions++;
      // The tier only drops a live snapshot once it is itself full, so a drop
      // trails an eviction rather than happening beside it (GitHub #224).
      if (Math.random() < 0.15) count.ramDrops++;
    }
  }

  return {
    /** The exposition at `now`, after simulating up to it (at most two minutes at a time). */
    render(now = Date.now()): string {
      clock = Math.max(clock, now - 120_000);
      while (clock + STEP_MS <= now) {
        clock += STEP_MS;
        step(clock);
      }
      const out: string[] = [];
      const declare = (name: string, type: string, help: string) => out.push(`# HELP ${name} ${help}`, `# TYPE ${name} ${type}`);
      declare("ignis_build_info", "gauge", "Constant build identity with value 1.");
      out.push('ignis_build_info{version="0.1.0-mock"} 1');
      declare("ignis_scheduler_requests", "gauge", "Current requests by observable scheduler state.");
      out.push(`ignis_scheduler_requests{state="waiting"} ${waiting.length}`, `ignis_scheduler_requests{state="running"} ${running.length}`);
      const counters: [string, string, number][] = [
        ["ignis_requests_accepted_total", "Accepted submissions.", count.accepted],
        ["ignis_requests_completed_total", "Completed requests.", count.completed],
        ["ignis_requests_cancelled_total", "Accepted requests cancelled before completion.", count.cancelled],
        ["ignis_generated_tokens_total", "Generated tokens on completed requests.", count.tokens],
        ["ignis_decoded_tokens_total", "Tokens generated so far, counted as each one is emitted.", count.decoded],
        ["ignis_kv_cache_evictions_total", "Cumulative host-tier evictions.", count.evictions],
        [
          "ignis_kv_ram_evictions_total",
          "Live host-tier snapshots dropped from KV-RAM to make room; the request re-prefills from the start.",
          count.ramDrops,
        ],
        ["ignis_prefix_reused_tokens_total", "Cumulative tokens skipped through sibling-prefix reuse.", count.prefix],
      ];
      for (const [name, help, value] of counters) {
        declare(name, "counter", help);
        out.push(`${name} ${value}`);
      }
      for (const [name, help] of RETAINED_FAMILIES) {
        declare(name, "counter", help);
        const family = name.replace(/^ignis_retained_(state_)?/, "").replace(/_total$/, "");
        const key = family === "reused_tokens" ? "reusedTokens" : family;
        for (const tier of ["device", "kv_ram"]) {
          for (const kind of ["checkpoint", "prefix"]) out.push(`${name}{tier="${tier}",kind="${kind}"} ${retained[`${key}/${tier}/${kind}`] ?? 0}`);
        }
      }
      declare("ignis_retained_slot_skips_total", "counter", "Publishes and captures that found no retained slot or no tail page.");
      for (const reason of SKIP_REASONS) out.push(`ignis_retained_slot_skips_total{reason="${reason}"} ${skips[reason]}`);
      declare("ignis_vram_reserved_bytes", "gauge", "Device bytes this load reserved, by the plan line that reserved them.");
      for (const [line, bytes] of VRAM_LINES) out.push(`ignis_vram_reserved_bytes{line="${line}"} ${bytes}`);
      // The pages running requests and retained tail pages hold together, as
      // the scheduler's own accounting counts them (ADR 0030).
      const usedPages = Math.min(KV_POOL_PAGES, running.reduce((a, req) => a + pagesOf(req), 0) + slots.inUse * 4);
      const gauges: [string, string, number][] = [
        ["ignis_vram_budget_bytes", "The device budget the plan was laid out inside.", VRAM_BUDGET_BYTES],
        ["ignis_kv_pool_pages", "Pages the KV pool holds.", KV_POOL_PAGES],
        ["ignis_kv_page_bytes", "One KV page's bytes.", KV_PAGE_BYTES],
        ["ignis_kv_pool_used_pages", "KV pool pages reserved by running requests and retained state.", usedPages],
      ];
      for (const [name, help, value] of gauges) {
        declare(name, "gauge", help);
        out.push(`${name} ${value}`);
      }
      declare("ignis_kv_ram_arena_bytes", "gauge", "The pinned host KV-RAM arena: what it holds, and what is used of it.");
      out.push(`ignis_kv_ram_arena_bytes{state="capacity"} ${KV_RAM_ARENA_BYTES}`, `ignis_kv_ram_arena_bytes{state="used"} ${slots.spilled * RETAINED_BLOB_BYTES}`);
      declare("ignis_retained_slots", "gauge", "Retained slots this load hands out, and how many hold an image.");
      out.push(`ignis_retained_slots{state="capacity"} ${RETAINED_SLOTS}`, `ignis_retained_slots{state="in_use"} ${slots.inUse}`);
      declare("ignis_requests_rejected_total", "counter", "Rejected submissions by fixed reason.");
      for (const [reason, value] of Object.entries(rejected)) out.push(`ignis_requests_rejected_total{reason="${reason}"} ${value}`);
      ttft.render(out, "ignis_request_ttft_seconds", "Submission-to-first-token latency.");
      duration.render(out, "ignis_request_duration_seconds", "Submission-to-completion latency.");
      return `${out.join("\n")}\n`;
    },
  };
}
