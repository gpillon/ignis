import { describe, expect, it } from "vitest";
import { compactTokens, contextUsage } from "./context.ts";
import type { Figures } from "./figures.ts";
import type { LogRow } from "./sessions.ts";

const figures = (promptTokens: number | null, completionTokens: number | null): Figures => ({
  ttftMs: 100,
  decodeTokensPerSec: 50,
  durationMs: 1000,
  promptTokens,
  completionTokens,
  finishReason: "stop",
  partial: false,
});

const row = (n: number, f: Figures | null): LogRow => ({
  n,
  at: "12:00",
  laneTag: "interactive",
  reasoningEffort: "xhigh",
  figures: f,
});

describe("compactTokens", () => {
  it("shortens thousands", () => {
    expect(compactTokens(850)).toBe("850");
    expect(compactTokens(1234)).toBe("1.2K");
    expect(compactTokens(40960)).toBe("41K");
  });
});

describe("contextUsage", () => {
  it("is empty before any reply", () => {
    const usage = contextUsage([], 16384, 40960);
    expect(usage.used).toBe(0);
    expect(usage.usedShare).toBe(0);
    expect(usage.reserveShare).toBeCloseTo(0.4);
    expect(usage.overflows).toBe(false);
  });

  it("counts the last reply that carries usage", () => {
    const log = [row(1, figures(100, 50)), row(2, figures(300, 200)), row(3, null), row(4, figures(null, null))];
    const usage = contextUsage(log, 1000, 10000);
    expect(usage).toMatchObject({ used: 500, promptTokens: 300, completionTokens: 200, reserve: 1000 });
    expect(usage.usedShare).toBeCloseTo(0.05);
  });

  it("skips the rows agents made", () => {
    const log = [row(1, figures(100, 50)), { ...row(2, figures(9, 9)), agent: "scan" }];
    expect(contextUsage(log, null, 1000).used).toBe(150);
  });

  it("flags a next request that would be over the context", () => {
    const log = [row(1, figures(30000, 2000))];
    expect(contextUsage(log, 16384, 40960).overflows).toBe(true);
    expect(contextUsage(log, 8000, 40960).overflows).toBe(false);
  });

  it("clamps the shares to the bar", () => {
    const usage = contextUsage([row(1, figures(9000, 3000))], 20000, 10000);
    expect(usage.usedShare).toBe(1);
    expect(usage.reserveShare).toBe(1);
  });

  it("has no shares without a limit", () => {
    const usage = contextUsage([row(1, figures(10, 5))], null, null);
    expect(usage).toMatchObject({ used: 15, limit: null, usedShare: null, reserveShare: null, overflows: false });
  });
});
