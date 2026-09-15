import { describe, expect, it } from "vitest";
import { assessHealth, deriveDashboard, type HealthInput, headerPulse } from "./derive.ts";
import type { Point } from "./history.ts";
import { emptySnapshot, type Snapshot } from "./snapshot.ts";

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
    expect(headerPulse(points)).toEqual({ running: 3, waiting: 0, tokensPerSec: 50 });
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
