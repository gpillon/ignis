import { describe, expect, it } from "vitest";
import { assessHealth, deriveDashboard, deriveMemory, deriveSpeculation, type HealthInput, headerPulse, TOKENS_PER_KV_PAGE } from "./derive.ts";
import type { Point } from "./history.ts";
import { parseExposition } from "./exposition.ts";
import { FLASH_NEXT_EXPOSITION } from "./fixture.ts";
import { EXPERT_CLASSES, emptySnapshot, type Memory as MemorySeries, readSnapshot, type Snapshot } from "./snapshot.ts";

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

  // GitHub #224: an eviction is a departure from a tier, and each departure
  // lands in the column that says what it cost. The device's spill shows up on
  // the VRAM row as a demotion even though the series is labelled `kv_ram` --
  // arriving there is the same act as leaving the device.
  it("splits evictions by the tier that gave the state up", () => {
    const retainedWith = (over: { discardsDevice?: number; discardsKvRam?: number; spillsKvRam?: number }) => {
      const r = emptySnapshot().retained;
      r.discards.device.checkpoint = over.discardsDevice ?? 0;
      r.discards.kv_ram.prefix = over.discardsKvRam ?? 0;
      r.spills.kv_ram.checkpoint = over.spillsKvRam ?? 0;
      return r;
    };
    const points = [
      at(0, { kvEvictions: 10, kvRamEvictions: 1, retained: retainedWith({}) }),
      at(60_000, {
        kvEvictions: 14,
        kvRamEvictions: 3,
        retained: retainedWith({ discardsDevice: 5, discardsKvRam: 2, spillsKvRam: 7 }),
      }),
    ];
    const ev = deriveDashboard(points, 60_000)!.evictions;

    expect(ev.vram).toMatchObject({ implemented: true });
    expect(ev.vram.live).toMatchObject({ total: 14, window: 4 });
    expect(ev.vram.retained).toMatchObject({ total: 5, window: 5 });
    expect(ev.vram.demoted).toMatchObject({ total: 7, window: 7 });

    expect(ev.ram.live).toMatchObject({ total: 3, window: 2 });
    expect(ev.ram.retained).toMatchObject({ total: 2, window: 2 });
    expect(ev.ram.demoted).toBeNull();

    // No disk tier on this load, so every column is absent rather than zero.
    expect(ev.disk).toEqual({ tier: "disk", implemented: false, live: null, retained: null, demoted: null });
  });

  // Spec vram-budget/03: with the KV-disk tier the disk row reads arrivals
  // into it, and the rows above add what left them for it.
  it("feeds the disk row, and the VRAM and RAM rows, on a load with the KV-disk tier", () => {
    const snap = (over: { evictions: number; ramEvictions: number; spills: [number, number]; retained: [number, number, number]; failures?: number }): Partial<Snapshot> => {
      const r = emptySnapshot().retained;
      r.discards.disk.checkpoint = over.retained[0];
      r.discards.disk.prefix = 1;
      r.spills.disk.checkpoint = over.retained[1];
      r.spills.disk.prefix = over.retained[2];
      r.spills.kv_ram.checkpoint = 0;
      return {
        kvEvictions: over.evictions,
        kvRamEvictions: over.ramEvictions,
        retained: r,
        kvDisk: { capacity: 1000, used: 250, spills: { device: over.spills[0], kv_ram: over.spills[1] }, failures: { write: 0, read: over.failures ?? 0 } },
      };
    };
    const points = [
      at(0, snap({ evictions: 10, ramEvictions: 1, spills: [1, 2], retained: [0, 0, 0] })),
      at(60_000, snap({ evictions: 12, ramEvictions: 1, spills: [4, 7], retained: [3, 4, 2] })),
    ];
    const ev = deriveDashboard(points, 60_000)!.evictions;

    expect(ev.disk.implemented).toBe(true);
    expect(ev.disk.live).toMatchObject({ total: 11, window: 8 });
    expect(ev.disk.retained).toMatchObject({ total: 4, window: 3 });
    expect(ev.disk.demoted).toMatchObject({ total: 6, window: 6 });
    // A live sequence that went device -> disk left VRAM as well.
    expect(ev.vram.live).toMatchObject({ total: 16, window: 5 });
    // RAM's demotion is the retained spills into the disk plus the live snapshots demoted to it.
    expect(ev.ram.demoted).toMatchObject({ total: 6 + 7, window: 6 + 5 });
  });

  it("keeps the VRAM and RAM rows as they were on a load without the tier", () => {
    const points = [at(0, { kvEvictions: 1 }), at(60_000, { kvEvictions: 3 })];
    const ev = deriveDashboard(points, 60_000)!.evictions;
    expect(ev.vram.live).toMatchObject({ total: 3, window: 2 });
    expect(ev.ram.demoted).toBeNull();
    expect(ev.disk.implemented).toBe(false);
  });

  it("meters the disk against its budget only when the tier exists", () => {
    const kvDisk = { capacity: 1000, used: 250, spills: { device: 0, kv_ram: 0 }, failures: { write: 0, read: 0 } };
    expect(deriveMemory([at(0, { kvDisk })], 0).diskInUse).toMatchObject({ used: 250, capacity: 1000, share: 0.25 });
    expect(deriveMemory([at(0, {})], 0).diskInUse).toBeNull();
  });

  // A server too old to export the new series leaves the RAM row blank rather
  // than reading as a tier that dropped nothing.
  it("reports no RAM live figure when the server does not export one", () => {
    const points = [at(0, { kvEvictions: 0 }), at(60_000, { kvEvictions: 2 })];
    const ev = deriveDashboard(points, 60_000)!.evictions;
    expect(ev.ram.live).toMatchObject({ total: null, window: null });
    expect(ev.vram.live).toMatchObject({ total: 2 });
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


describe("deriveDashboard on a Flash-Next load", () => {
  // Ten seconds apart: 900 decode hits and 100 decode misses over 50 decoded
  // tokens, 50 MB copied in, 800 n-gram rows of which 760 from RAM.
  const first = readSnapshot(parseExposition(FLASH_NEXT_EXPOSITION));
  const second = structuredClone(first);
  second.decodedTokens = 12_450;
  second.experts!.hits.gate_up_k2.decode = 4900;
  second.experts!.misses.down_k3.decode = 400;
  second.experts!.bytesMoved = { decode: 6_040_000_000, prefill: 9_010_000_000 };
  second.experts!.stall = { decode: 18.7, prefill: 40.25 };
  second.experts!.prefetchIssued = 920;
  second.experts!.prefetchUsed = 735;
  second.experts!.slots.gate_up_k4.inUse = 260;
  second.ngram = { rows: { hot: 190_760, file: 10_040 }, reads: 6008, readBytes: 24_608_768 };
  const points: Point[] = [
    { at: 0, snap: first, scrapeMs: 1 },
    { at: 10_000, snap: second, scrapeMs: 1 },
  ];

  it("reads the residency figures the owner diagnoses a slow turn by", () => {
    const experts = deriveDashboard(points, 60_000)!.experts!;
    expect(experts.decodeHitShare).toBeCloseTo(0.9);
    expect(experts.prefillHitShare).toBeNull();
    expect(experts.missesPerToken).toBeCloseTo(2);
    expect(experts.bytesPerSec).toBeCloseTo(5_000_000);
    expect(experts.bytesMoved).toEqual({ total: 15_050_000_000, window: 50_000_000 });
    expect(experts.prefetch).toEqual({ issued: { total: 920, window: 20 }, used: { total: 735, window: 15 } });
    // 0.1 s of decode stall over 50 tokens and 10 s.
    expect(experts.stallPerToken).toBeCloseTo(0.002);
    expect(experts.stallShare).toBeCloseTo(0.01);
    expect(experts.stall.total).toBeCloseTo(58.95);
    expect(experts.stall.window).toBeCloseTo(0.1);
    const gateUpK4 = experts.classes.find((c) => c.cls === "gate_up_k4")!;
    expect(gateUpK4.slots).toMatchObject({ used: 260, capacity: 400, share: 0.65 });
    expect(gateUpK4.hits).toEqual({ total: 800, window: 0 });
    expect(experts.classes.map((c) => c.cls)).toEqual(EXPERT_CLASSES);
  });

  it("reads the n-gram rows by source and the file reads behind them", () => {
    const ngram = deriveDashboard(points, 60_000)!.ngram!;
    expect(ngram.hotShare).toBeCloseTo(0.95);
    expect(ngram.rows).toEqual({ total: 200_800, window: 800 });
    expect(ngram.reads).toEqual({ total: 6008, window: 8 });
    expect(ngram.readBytes).toEqual({ total: 24_608_768, window: 32_768 });
    expect(ngram.readBytesPerSec).toBeCloseTo(3276.8);
  });

  it("has neither on a 27B load", () => {
    const dash = deriveDashboard([at(0, {}), at(10_000, { accepted: 1 })], 60_000)!;
    expect(dash.experts).toBeNull();
    expect(dash.ngram).toBeNull();
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

  it("counts Flash-Next's residency and expert cache as lines, and a server without them as 0", () => {
    const reserved = Object.fromEntries(Object.keys(emptySnapshot().memory.reserved).map((line) => [line, 0])) as MemorySeries["reserved"];
    Object.assign(reserved, { weights: 600, residency: 100, expert_cache: 500 });
    const m = deriveMemory([memoryAt(0, { reserved, budgetBytes: 2000, kvPoolPages: 8, kvPageBytes: 100 })], 0);
    expect(m.lines.slice(-2).map((l) => l.line)).toEqual(["residency", "expert_cache"]);
    expect(m.linesBytes).toBe(1200);
    expect(m.expertCacheBytes).toBe(500);
    expect(m.kvRoomBytes).toBe(800);
    expect(m.spareBytes).toBe(0);
    // A server that predates the two lines exports twelve; the plan is whole all the same.
    const old: Record<string, number | null> = { ...reserved, residency: null, expert_cache: null };
    const o = deriveMemory([memoryAt(0, { reserved: old as MemorySeries["reserved"], budgetBytes: 2000 })], 0);
    expect(o.planned).toBe(true);
    expect(o.linesBytes).toBe(600);
    expect(o.expertCacheBytes).toBe(0);
  });

  it("will not add a plan up unless the scrape carries all twelve lines", () => {
    // A partial sum would understate the plan and overstate the room beside
    // it; the server writes the twelve together or not at all.
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
    expect(m.linesBytes).toBe(1400);
    expect(m.kvRoomBytes).toBe(-500);
    expect(m.spareBytes).toBe(-600);
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
      disk: { checkpoint: null, prefix: null },
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
    ramDrops: 0,
    diskReadFailures: 0,
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

  // GitHub #224: a VRAM eviction keeps the request's work in host RAM; a RAM
  // drop destroys it. The one that loses work must not read as the milder of
  // the two.
  it("ranks a snapshot dropped from RAM above an eviction to RAM", () => {
    const evicted = assessHealth({ ...quiet, running: 6, evictions: 3 });
    const dropped = assessHealth({ ...quiet, running: 6, ramDrops: 3 });
    expect(evicted.level).toBe("busy");
    expect(dropped.level).toBe("saturated");
    expect(dropped.summary).toBe("3 snapshots dropped from host RAM in the last 5 min");
    expect(dropped.notes).toContain("3 snapshots dropped from host RAM: re-prefilled from the start");
  });

  // Spec vram-budget/03: a file refused at restore costs the request its
  // prefill, like a snapshot dropped from RAM.
  it("weighs a disk read failure like a RAM drop", () => {
    const failed = assessHealth({ ...quiet, running: 2, diskReadFailures: 2 });
    expect(failed.level).toBe("saturated");
    expect(failed.summary).toBe("2 KV-disk files refused at restore in the last 5 min");
    expect(failed.notes).toContain("2 KV-disk files refused at restore: re-prefilled from the start");
    expect(assessHealth({ ...quiet, running: 2, diskReadFailures: 0 }).level).toBe("healthy");
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

// GitHub #307: acceptance is read over the window, overall and per draft position.
describe("deriveSpeculation", () => {
  const spec = (rounds: number, drafted: number, accepted: number, at1: [number, number], at2: [number, number]) => ({
    rounds,
    drafted,
    accepted,
    positionDrafted: [at1[0], at2[0], 0, 0, 0, 0, 0],
    positionAccepted: [at1[1], at2[1], 0, 0, 0, 0, 0],
  });

  it("reads acceptance, tokens per round and each position's share over the window", () => {
    const points = [at(0, { speculation: spec(100, 200, 120, [100, 70], [100, 50]) }), at(60_000, { speculation: spec(110, 220, 135, [110, 78], [110, 57]) })];
    const s = deriveSpeculation(points, 0);
    expect(s.rounds).toEqual({ total: 110, window: 10 });
    expect(s.acceptance).toBeCloseTo(15 / 20);
    expect(s.tokensPerRound).toBeCloseTo(25 / 10);
    expect(s.byPosition.slice(0, 3)).toEqual([0.8, 0.7, null]);
  });

  it("is absent from the dashboard until a verify round ran", () => {
    expect(deriveDashboard([at(0, {})], 60_000)!.speculation).toBeNull();
  });
});
