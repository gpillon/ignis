import { describe, expect, it } from "vitest";
import { formatAgo, formatBound, formatCount, formatNumber, formatOffset, formatSeconds, formatShare, niceScale } from "./format.ts";

describe("monitor formatting", () => {
  it("compacts counts", () => {
    expect(formatCount(null)).toBe("—");
    expect(formatCount(1284)).toBe("1,284");
    expect(formatCount(12_345)).toBe("12.3K");
    expect(formatCount(150_000)).toBe("150K");
    expect(formatCount(4_200_000)).toBe("4.2M");
  });

  it("gives measured figures the decimals their size deserves", () => {
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(0.25)).toBe("0.25");
    expect(formatNumber(4.54)).toBe("4.5");
    expect(formatNumber(43.2)).toBe("43");
  });

  it("writes latencies from milliseconds to minutes", () => {
    expect(formatSeconds(0.85)).toBe("850 ms");
    expect(formatSeconds(2.4)).toBe("2.40 s");
    expect(formatSeconds(12.5)).toBe("12.5 s");
    expect(formatSeconds(72)).toBe("1m 12s");
    expect(formatSeconds(120)).toBe("2m");
    expect(formatSeconds(59.7)).toBe("59.7 s");
  });

  it("names bucket bounds, offsets and moments", () => {
    expect([0.05, 2.5, 120, Infinity].map(formatBound)).toEqual(["50ms", "2.5s", "2m", "+Inf"]);
    expect([0, 45_000, 300_000].map(formatOffset)).toEqual(["now", "−45s", "−5m"]);
    expect([400, 4_000, 120_000].map(formatAgo)).toEqual(["just now", "4 s ago", "2 min ago"]);
    expect(formatShare(1, 4)).toBe("25%");
    expect(formatShare(1, 0)).toBe("—");
  });

  it("picks clean axis tops", () => {
    expect(niceScale(0)).toEqual({ top: 1, step: 0.25 });
    expect(niceScale(0, 4, true)).toEqual({ top: 4, step: 1 });
    expect(niceScale(43)).toEqual({ top: 50, step: 10 });
    expect(niceScale(3, 4, true)).toEqual({ top: 3, step: 1 });
    expect(niceScale(0.7)).toEqual({ top: 0.8, step: 0.2 });
  });
});
