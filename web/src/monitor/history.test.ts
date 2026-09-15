import { describe, expect, it } from "vitest";
import { append, increase, increaseOver, isRestart, type Point, rateSeries, rollingRate, windowHistogram } from "./history.ts";
import { emptySnapshot, type Snapshot } from "./snapshot.ts";

const snap = (over: Partial<Snapshot>): Snapshot => ({ ...emptySnapshot(), ...over });
const at = (ms: number, over: Partial<Snapshot>): Point => ({ at: ms, snap: snap(over), scrapeMs: 1 });

describe("append", () => {
  it("keeps the points inside the window, oldest first", () => {
    let points: Point[] = [];
    for (const ms of [0, 5_000, 10_000, 15_000]) points = append(points, at(ms, {}), 10_000);
    expect(points.map((p) => p.at)).toEqual([5_000, 10_000, 15_000]);
  });
});

describe("increase", () => {
  it("is the difference between two readings", () => {
    expect(increase(10, 14)).toBe(4);
  });

  it("counts from zero after a counter went down, never negative", () => {
    expect(increase(40, 3)).toBe(3);
  });

  it("is unknown when either reading is", () => {
    expect(increase(null, 3)).toBeNull();
    expect(increase(3, null)).toBeNull();
  });
});

describe("rateSeries", () => {
  it("gives a per-second rate between successive scrapes", () => {
    const points = [at(0, { accepted: 0 }), at(2_000, { accepted: 4 }), at(4_000, { accepted: 10 })];
    expect(rateSeries(points, (s) => s.accepted)).toEqual([null, 2, 3]);
  });

  it("resets the series on a server restart instead of drawing a negative rate", () => {
    const points = [at(0, { accepted: 50 }), at(5_000, { accepted: 60 }), at(10_000, { accepted: 5 })];
    const rates = rateSeries(points, (s) => s.accepted);
    expect(rates).toEqual([null, 2, 1]);
    expect(rates.every((r) => r === null || r >= 0)).toBe(true);
  });
});

describe("a server restart", () => {
  it("restarts every counter's count, even one that came back higher", () => {
    // `completed` going down marks the restart; `accepted` climbed past its old total.
    const points = [at(0, { accepted: 50, completed: 40 }), at(10_000, { accepted: 60, completed: 2 })];
    expect(rateSeries(points, (s) => s.accepted)).toEqual([null, 6]);
    expect(increaseOver(points, (s) => s.accepted, 0)).toBe(60);
  });

  it("takes the whole new histogram as the window's observations", () => {
    const h = (le1: number, inf: number) => ({ bounds: [1, Infinity], cumulative: [le1, inf], sum: inf, count: inf });
    const points = [at(0, { completed: 9, ttft: h(2, 3) }), at(5_000, { completed: 1, ttft: h(1, 5) })];
    expect(windowHistogram(points, (s) => s.ttft, 0)).toEqual(h(1, 5));
  });
});

describe("rollingRate", () => {
  it("averages each point's rate over the span before it", () => {
    const points = [at(0, { generatedTokens: 0 }), at(10_000, { generatedTokens: 10 }), at(20_000, { generatedTokens: 30 }), at(30_000, { generatedTokens: 30 })];
    expect(rollingRate(points, (s) => s.generatedTokens, 20_000)).toEqual([null, 1, 1.5, 1]);
  });

  it("counts from zero across a restart", () => {
    const points = [at(0, { generatedTokens: 500 }), at(10_000, { generatedTokens: 20 })];
    expect(rollingRate(points, (s) => s.generatedTokens, 60_000)).toEqual([null, 2]);
  });
});

describe("increaseOver", () => {
  it("sums the gains of the scrapes since a moment, across a restart", () => {
    const points = [
      at(0, { completed: 100 }),
      at(5_000, { completed: 110 }),
      at(10_000, { completed: 2 }),
      at(15_000, { completed: 7 }),
    ];
    expect(increaseOver(points, (s) => s.completed, 5_000)).toBe(7);
    expect(increaseOver(points, (s) => s.completed, 0)).toBe(17);
    expect(increaseOver(points.slice(0, 1), (s) => s.completed, 0)).toBeNull();
  });
});

describe("isRestart", () => {
  it("spots counters going down between two scrapes", () => {
    expect(isRestart(snap({ accepted: 9, completed: 8 }), snap({ accepted: 1, completed: 0 }))).toBe(true);
    expect(isRestart(snap({ accepted: 9, completed: 8 }), snap({ accepted: 9, completed: 9 }))).toBe(false);
  });
});

describe("windowHistogram", () => {
  const h = (le1: number, inf: number, sum: number) => ({ bounds: [1, Infinity], cumulative: [le1, inf], sum, count: inf });

  it("is the observations made since a moment", () => {
    const points = [at(0, { ttft: h(1, 2, 2) }), at(5_000, { ttft: h(3, 5, 6) }), at(10_000, { ttft: h(4, 8, 10) })];
    expect(windowHistogram(points, (s) => s.ttft, 5_000)).toEqual(h(1, 3, 4));
    expect(windowHistogram(points, (s) => s.ttft, 0)).toEqual(h(3, 6, 8));
  });

  it("is unknown with fewer than two histograms in reach", () => {
    expect(windowHistogram([at(0, { ttft: h(1, 1, 1) })], (s) => s.ttft, 0)).toBeNull();
  });
});
