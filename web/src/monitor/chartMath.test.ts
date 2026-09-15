import { describe, expect, it } from "vitest";
import { bucketOffset, knownRuns, nearestIndex, stackTops, tipLeft } from "./chartMath.ts";

describe("stackTops", () => {
  it("keeps each series' own values unstacked", () => {
    expect(stackTops([[1, null], [2, 3]], 2, false)).toEqual([[1, null], [2, 3]]);
  });

  it("stacks series, counting an unknown as zero unless every series is unknown", () => {
    expect(stackTops([[1, null, null], [2, 3, undefined]], 3, true)).toEqual([[1, 0, null], [3, 3, null]]);
  });
});

describe("knownRuns", () => {
  it("breaks a line where a value is unknown", () => {
    expect(knownRuns([1, 2, null, 4, undefined, 6, 7])).toEqual([[0, 1], [3], [5, 6]]);
    expect(knownRuns([1, 2, 3], 1)).toEqual([[1, 2]]);
    expect(knownRuns([null, null])).toEqual([]);
  });
});

describe("nearestIndex", () => {
  it("snaps to the closest candidate scrape", () => {
    const times = [0, 5_000, 10_000, 15_000];
    expect(nearestIndex(times, [1, 2, 3], 6_000)).toBe(1);
    expect(nearestIndex(times, [1, 2, 3], -50_000)).toBe(1);
    expect(nearestIndex(times, [], 6_000)).toBeNull();
  });
});

describe("bucketOffset", () => {
  const bounds = [1, 2, 4, Infinity];

  it("places a value inside its bucket", () => {
    expect(bucketOffset(bounds, 0.5)).toBe(0.5);
    expect(bucketOffset(bounds, 1.5)).toBe(1.5);
    expect(bucketOffset(bounds, 3)).toBe(2.5);
  });

  it("puts +Inf values mid-bucket", () => {
    expect(bucketOffset(bounds, 100)).toBe(3.5);
    expect(bucketOffset([1, 2], 5)).toBe(2);
  });
});

describe("tipLeft", () => {
  it("opens right of the anchor, or left when it would overflow", () => {
    expect(tipLeft(100, 1000, 200)).toBe(114);
    expect(tipLeft(900, 1000, 200)).toBe(686);
    expect(tipLeft(50, 200, 196)).toBe(0);
  });
});
