// A simulated ignis behind `npm run dev:mock`'s /ui/metrics: requests arrive
// in waves and bursts (an agent fan-out), queue for six lanes, prefill,
// decode, get cancelled or turned away, and the counters and histograms move
// the way the real exposition's do (crates/server/src/metrics.rs, ADR 0017).
// The simulation advances on each scrape to the scrape's clock, so the
// Monitor has live data without the shared GPU. Development only.

const TTFT_BOUNDS_MS = [50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000];
const DURATION_BOUNDS_MS = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000, 600_000];
const LANES = 6;
const QUEUE_LIMIT = 8;
const STEP_MS = 100;

type Request = {
  arrivedAt: number;
  tokens: number;
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
  const count = { accepted: 0, completed: 0, cancelled: 0, tokens: 0, decoded: 0, evictions: 0, prefix: 0 };
  const rejected = { full: 0, unknown_model: 0, oversized: 0 };
  const ttft = new Histogram(TTFT_BOUNDS_MS);
  const duration = new Histogram(DURATION_BOUNDS_MS);

  function arrive(t: number) {
    const roll = Math.random();
    if (roll < 0.008) return void rejected.unknown_model++;
    if (roll < 0.02) return void rejected.oversized++;
    if (waiting.length >= QUEUE_LIMIT) return void rejected.full++;
    count.accepted++;
    waiting.push({ arrivedAt: t, tokens: Math.max(8, Math.round(logNormal(380, 0.75))), firstAt: 0, endAt: 0, cancelAt: null, sawFirst: false, tokensPerSecond: 0, decoded: 0 });
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

    if (running.length === LANES && waiting.length > 2 && Math.random() < 0.004) count.evictions++;
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
        ["ignis_prefix_reused_tokens_total", "Cumulative tokens skipped through sibling-prefix reuse.", count.prefix],
      ];
      for (const [name, help, value] of counters) {
        declare(name, "counter", help);
        out.push(`${name} ${value}`);
      }
      declare("ignis_requests_rejected_total", "counter", "Rejected submissions by fixed reason.");
      for (const [reason, value] of Object.entries(rejected)) out.push(`ignis_requests_rejected_total{reason="${reason}"} ${value}`);
      ttft.render(out, "ignis_request_ttft_seconds", "Submission-to-first-token latency.");
      duration.render(out, "ignis_request_duration_seconds", "Submission-to-completion latency.");
      return `${out.join("\n")}\n`;
    },
  };
}
