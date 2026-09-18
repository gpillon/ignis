import { describe, expect, it } from "vitest";
import { assessHealth, deriveDashboard, deriveMemory, type HealthInput, headerPulse, TOKENS_PER_KV_PAGE } from "./derive.ts";
import type { Point } from "./history.ts";
import { emptySnapshot, type Memory as MemorySeries, type Snapshot } from "./snapshot.ts";

const at = (ms: number, over: Partial<Snapshot>): Point => ({ at: ms, snap: { ...emptySnapshot(), ...over }, scrapeMs: 1 });
const h = (le1: number, inf: number) => ({ bounds: [1, 2, Infinity], cumulative: [le1, inf, inf], sum: inf, count: inf });

describe("deriveDashboard", () => {
  it("has nothing to show before the first scrape", () => {
    expect(deriveDashboard([], 60_000)).toBeNull();
  });

  it("reads the window's gains, per-minute rates and the latest gauges", () => {
    const points = [
      at(0, { accepted: 100, running: 1, waiting: 0, rejected: { full: 0, unknown_model: 0, oversized: 0 } }),
      at(30_000, { accepted: 110, running: 4, waiting: 2, rejected: { full: 1, unknown_model: 0, oversized: 0 } }),
      at(60_000, { accepted: 140, running: 3, waiting: 0, rejected: { full: 3, unknown_model: 1, oversized: 0 } }),
    ];
    const dash = deriveDashboard(points, 60_000)!;
    expect(dash.accepted).toMatchObject({ total: 140, window: 40, perMin: 40 });
    expect(dash.rejected).toMatchObject({ total: 4, window: 4 });
    expect(dash.rejected.byReason.full).toEqual({ total: 3, window: 3 });
    expect([dash.running, dash.waiting, dash.runningPeak, dash.waitingPeak]).toEqual([3, 0, 4, 2]);
    expect(dash.health.level).toBe("saturated");
  });

  it("takes latency quantiles from the observations inside the window only", () => {
    const points = [at(0, { ttft: h(50, 50) }), at(30_000, { ttft: h(50, 50) }), at(60_000, { ttft: h(50, 60) })];
    const dash = deriveDashboard(points, 30_000)!;
    expect(dash.ttft.count).toBe(10);
    expect(dash.ttft.p50).toBeCloseTo(1.5);
    expect(dash.ttft.lifetimeCount).toBe(60);
    expect(dash.ttft.trendP50).toEqual([null, null, dash.ttft.p50]);
  });

  it("gives the header the latest gauges and token rate", () => {
    const points = [at(0, { generatedTokens: 0, running: 2, waiting: 1 }), at(10_000, { generatedTokens: 500, running: 3, waiting: 0 })];
    expect(headerPulse(points)).toEqual({ running: 3, waiting: 0, tokensPerSec: 50, live: false });
    expect(headerPulse([])).toEqual({ running: null, waiting: null, tokensPerSec: null, live: false });
  });

  it("takes throughput from decoded tokens while a long request runs, and per request from completed ones", () => {
    // A request decoding for 10 s: decoded tokens climb, nothing has completed yet.
    const decoding = [
      at(0, { decodedTokens: 0, generatedTokens: 0, completed: 0 }),
      at(5_000, { decodedTokens: 250, generatedTokens: 0, completed: 0 }),
      at(10_000, { decodedTokens: 500, generatedTokens: 0, completed: 0 }),
    ];
    const dash = deriveDashboard(decoding, 60_000)!;
    expect(dash.tokens).toMatchObject({ live: true, spanMs: 10_000, perSec: 50, window: 500, perRequest: null });
    expect(headerPulse(decoding)).toMatchObject({ tokensPerSec: 50, live: true });

    const done = [...decoding, at(15_000, { decodedTokens: 600, generatedTokens: 600, completed: 2 })];
    expect(deriveDashboard(done, 60_000)!.tokens.perRequest).toBe(300);
  });
});


describe("deriveMemory", () => {
  const memoryAt = (ms: number, over: Partial<MemorySeries>, retained?: Snapshot["retained"]): Point => {
    const snap = emptySnapshot();
    return { at: ms, snap: { ...snap, memory: { ...snap.memory, ...over }, retained: retained ?? snap.retained }, scrapeMs: 1 };
  };

  it("reads each live figure against the constant that bounds it", () => {
    const points = [
      memoryAt(0, { kvPoolUsedPages: 100, kvPoolPages: 400, kvRamArena: { capacity: 800, used: 0 }, retainedSlots: { capacity: 10, inUse: 1 } }),
      memoryAt(60_000, { kvPoolUsedPages: 300, kvPoolPages: 400, kvRamArena: { capacity: 800, used: 200 }, retainedSlots: { capacity: 10, inUse: 7 } }),
    ];
    const m = deriveMemory(points, 0);
    expect(m.pagesInUse).toMatchObject({ used: 300, capacity: 400, share: 0.75 });
    expect(m.arenaInUse).toMatchObject({ used: 200, capacity: 800, share: 0.25 });
    expect(m.slotsInUse).toMatchObject({ used: 7, capacity: 10, share: 0.7 });
    expect(m.pagesInUse.series).toEqual([100, 300]);
  });

  it("sparks the window the pill asks for, not the whole history", () => {
    const pages = (ms: number, used: number) => memoryAt(ms, { kvPoolUsedPages: used, kvPoolPages: 400 });
    const points = [pages(0, 10), pages(30_000, 20), pages(60_000, 30)];
    expect(deriveMemory(points, 30_000).pagesInUse.series).toEqual([20, 30]);
    expect(deriveMemory(points, -Infinity).pagesInUse.series).toEqual([10, 20, 30]);
  });

  it("lays the plan out in plan order and leaves the rest of the budget to the KV pool", () => {
    const reserved = Object.fromEntries(Object.keys(emptySnapshot().memory.reserved).map((line) => [line, 0])) as MemorySeries["reserved"];
    Object.assign(reserved, { weights: 600, workspace: 300, residual: 100 });
    const m = deriveMemory([memoryAt(0, { reserved, budgetBytes: 2000, kvPoolPages: 8, kvPageBytes: 100 })], 0);
    expect(m.planned).toBe(true);
    expect(m.lines.map((l) => l.line).slice(0, 3)).toEqual(["weights", "cuda_context", "workspace"]);
    expect(m.linesBytes).toBe(1000);
    expect(m.kvRoomBytes).toBe(1000);
    expect(m.kvPool).toEqual({ pages: 8, pageBytes: 100, bytes: 800, tokens: 8 * TOKENS_PER_KV_PAGE });
    expect(m.spareBytes).toBe(200);
    expect(m.oversubscribed).toBe(false);
  });

  it("will not add a plan up unless the scrape carries all eleven lines", () => {
    // A partial sum would understate the plan and overstate the room beside
    // it; the server writes the eleven together or not at all.
    const partial = { ...emptySnapshot().memory.reserved, weights: 600, workspace: 300 };
    const m = deriveMemory([memoryAt(0, { reserved: partial, budgetBytes: 2000 })], 0);
    expect(m.linesBytes).toBeNull();
    expect(m.kvRoomBytes).toBeNull();
    expect(m.planned).toBe(false);
  });

  it("reports an overrun rather than clamping it away", () => {
    // --allow-vram-oversubscription: the plan is larger than the budget it
    // was laid out in, and the panel has to be able to say so.
    const reserved = Object.fromEntries(Object.keys(emptySnapshot().memory.reserved).map((line) => [line, 100]));
    const m = deriveMemory([memoryAt(0, { reserved: reserved as MemorySeries["reserved"], budgetBytes: 900, kvPoolPages: 2, kvPageBytes: 50 })], 0);
    expect(m.linesBytes).toBe(1100);
    expect(m.kvRoomBytes).toBe(-200);
    expect(m.spareBytes).toBe(-300);
    expect(m.oversubscribed).toBe(true);
  });

  it("has no plan, and no share to compute, on a load that exported none", () => {
    const m = deriveMemory([memoryAt(0, {})], 0);
    expect(m.planned).toBe(false);
    expect(m.budgetBytes).toBeNull();
    expect(m.linesBytes).toBeNull();
    expect(m.kvRoomBytes).toBeNull();
    expect(m.kvPool.bytes).toBeNull();
    expect([m.pagesInUse.share, m.arenaInUse.share, m.slotsInUse.share]).toEqual([null, null, null]);
    expect(m.skips.total).toBeNull();
    expect(m.retained.spills.kv_ram.checkpoint).toEqual({ total: null, window: null });
    expect(deriveMemory([], 0).planned).toBe(false);
  });

  it("gives a capacity of zero no share, so nothing is drawn as full", () => {
    // `--prompt-reuse off` without --retained-slots hands out no slots (#215).
    const m = deriveMemory([memoryAt(0, { retainedSlots: { capacity: 0, inUse: 0 } })], 0);
    expect(m.slotsInUse).toMatchObject({ used: 0, capacity: 0, share: null });
  });

  it("counts each retained family by tier and kind, and the skips beside the slots", () => {
    const matrix = (device: [number, number], kvRam: [number, number]) => ({
      device: { checkpoint: device[0], prefix: device[1] },
      kv_ram: { checkpoint: kvRam[0], prefix: kvRam[1] },
    });
    const empty = emptySnapshot().retained;
    const before = { ...empty, spills: matrix([0, 0], [2, 1]) };
    const after = { ...empty, spills: matrix([0, 0], [6, 4]) };
    const points = [
      memoryAt(0, { slotSkips: { publish_skipped_no_slot: 1, capture_skipped_no_slot: 0, capture_skipped_no_page: 0 } }, before),
      memoryAt(60_000, { slotSkips: { publish_skipped_no_slot: 9, capture_skipped_no_slot: 2, capture_skipped_no_page: 1 } }, after),
    ];
    const m = deriveMemory(points, 0);
    expect(m.retained.spills.kv_ram.checkpoint).toEqual({ total: 6, window: 4 });
    expect(m.retained.spills.kv_ram.prefix).toEqual({ total: 4, window: 3 });
    expect(m.retained.spills.device.checkpoint).toEqual({ total: 0, window: 0 });
    expect(m.skips).toMatchObject({ total: 12, window: 11 });
    expect(m.skips.byReason.publish_skipped_no_slot).toEqual({ total: 9, window: 8 });
  });
});

describe("assessHealth", () => {
  const quiet: HealthInput = {
    windowMs: 300_000,
    running: 0,
    waiting: 0,
    accepted: 0,
    completed: 0,
    cancelled: 0,
    rejected: { full: 0, unknown_model: 0, oversized: 0 },
    evictions: 0,
    ttftP95: null,
  };

  it("is idle without traffic", () => {
    expect(assessHealth(quiet)).toEqual({ level: "idle", summary: "No requests in the last 5 min", notes: [] });
  });

  it("is healthy while serving with nothing queued", () => {
    expect(assessHealth({ ...quiet, running: 3, accepted: 5 })).toMatchObject({ level: "healthy", summary: "3 requests running, nothing queued" });
  });

  it("is busy with a queue, evictions or slow first tokens", () => {
    expect(assessHealth({ ...quiet, running: 6, waiting: 2 })).toMatchObject({ level: "busy", summary: "2 waiting, 6 running" });
    expect(assessHealth({ ...quiet, running: 6, evictions: 1 }).level).toBe("busy");
    expect(assessHealth({ ...quiet, running: 1, ttftP95: 7.5 }).notes).toContain("p95 time to first token 7.50 s");
  });

  it("is saturated when requests are turned away as full, and says why in notes", () => {
    const health = assessHealth({ ...quiet, running: 6, waiting: 8, rejected: { full: 4, unknown_model: 0, oversized: 1 }, completed: 6, cancelled: 2 });
    expect(health.level).toBe("saturated");
    expect(health.summary).toBe("4 requests turned away in the last 5 min");
    expect(health.notes).toEqual([
      "4 requests turned away: engine full",
      "1 request too long to ever fit",
      "8 waiting for a lane",
      "25% of finished requests cancelled",
    ]);
  });
});
