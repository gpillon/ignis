import { describe, expect, it } from "vitest";
import { bucketCounts, histogramIncrease, histogramMean, histogramQuantile } from "./quantile.ts";
import type { Histogram } from "./snapshot.ts";

const hist = (bounds: number[], cumulative: number[], sum = 0): Histogram => ({
  bounds,
  cumulative,
  sum,
  count: cumulative.at(-1) ?? 0,
});

describe("histogramQuantile", () => {
  it("has no quantile without observations", () => {
    expect(histogramQuantile(0.5, hist([1, Infinity], [0, 0]))).toBeNull();
  });

  it("interpolates linearly inside the bucket holding the rank", () => {
    expect(histogramQuantile(0.5, hist([1, 2, Infinity], [0, 10, 10]))).toBeCloseTo(1.5);
    expect(histogramQuantile(0.95, hist([1, 2, Infinity], [0, 10, 10]))).toBeCloseTo(1.95);
  });

  it("starts the first bucket at zero", () => {
    expect(histogramQuantile(0.5, hist([0.1, Infinity], [10, 10]))).toBeCloseTo(0.05);
  });

  it("answers the highest finite bound when the rank falls in +Inf", () => {
    expect(histogramQuantile(0.95, hist([1, Infinity], [5, 10]))).toBe(1);
  });
});

describe("histogram helpers", () => {
  it("turns cumulative buckets into per-bucket counts", () => {
    expect(bucketCounts(hist([1, 2, Infinity], [3, 7, 8]))).toEqual([3, 4, 1]);
  });

  it("averages sum over count", () => {
    expect(histogramMean(hist([1, Infinity], [2, 4], 6))).toBe(1.5);
    expect(histogramMean(hist([1, Infinity], [0, 0]))).toBeNull();
  });

  it("takes what one histogram gained over an earlier one", () => {
    const a = hist([1, Infinity], [2, 3], 2);
    const b = hist([1, Infinity], [5, 9], 11);
    expect(histogramIncrease(a, b)).toEqual(hist([1, Infinity], [3, 6], 9));
  });

  it("treats a count going down as a restart: the later histogram is the gain", () => {
    const before = hist([1, Infinity], [5, 9], 11);
    const after = hist([1, Infinity], [1, 1], 0.5);
    expect(histogramIncrease(before, after)).toEqual(after);
  });
});
