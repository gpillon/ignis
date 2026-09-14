import { describe, expect, it } from "vitest";
import { trackFill } from "./slider.ts";

describe("trackFill", () => {
  it("places the value along the track", () => {
    expect(trackFill(0.7, 0, 2)).toBe("35%");
    expect(trackFill(1, 0, 1)).toBe("100%");
    expect(trackFill(0, 0, 1)).toBe("0%");
  });

  it("clamps a value outside the track", () => {
    expect(trackFill(3, 0, 2)).toBe("100%");
    expect(trackFill(-1, 0, 2)).toBe("0%");
  });

  it("treats an empty track as unfilled", () => {
    expect(trackFill(1, 1, 1)).toBe("0%");
  });
});
