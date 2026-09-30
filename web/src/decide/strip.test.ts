import { describe, expect, it } from "vitest";
import { levelOf, positionOf, stripOf } from "./strip.ts";

// The strip: a long target in a fixed number of columns, marked where a line
// reads as an error or a warning.

describe("levelOf", () => {
  it("reads a level word in capitals, and a level field in any case", () => {
    expect(levelOf("09:15:40 ERROR orders pg: FATAL sorry, too many clients")).toBe("error");
    expect(levelOf("09:14:31 WARN  billing retrying charge")).toBe("warn");
    expect(levelOf('ts=2026-09-28 level=warn msg="slow"')).toBe("warn");
    expect(levelOf('{"severity": "ERROR", "msg": "down"}')).toBe("error");
    expect(levelOf('{"level":"Fatal"}')).toBe("error");
    expect(levelOf("Caused by: java.lang.IllegalStateException: closed")).toBe("error");
    expect(levelOf("TypeError: x is undefined")).toBe("error");
    expect(levelOf("Exception in thread main")).toBe("error");
  });

  it("leaves prose alone: an error in a sentence is not a level", () => {
    expect(levelOf("The error was in the second paragraph, and a warning came later.")).toBe(null);
    expect(levelOf("Error handling is described in the next section.")).toBe(null);
    expect(levelOf("09:14:02 INFO  gateway GET /v1/orders 200 41ms")).toBe(null);
  });
});

describe("stripOf", () => {
  const log = Array.from({ length: 1000 }, (_, i) => (i === 500 ? "ERROR down" : i === 900 ? "WARN slow" : `line ${i}`));

  it("buckets a long target into the width it is given, in order and without a gap", () => {
    const strip = stripOf(log, 120);
    expect(strip.columns).toHaveLength(120);
    expect(strip.columns[0].from).toBe(0);
    for (let c = 1; c < 120; c++) expect(strip.columns[c].from).toBeGreaterThan(strip.columns[c - 1].from);
    expect(Math.max(...strip.columns.map((c) => c.weight))).toBe(1);
  });

  it("marks the column holding each loud line, and counts them", () => {
    const strip = stripOf(log, 100);
    expect(strip.columns[50].level).toBe("error");
    expect(strip.columns[90].level).toBe("warn");
    expect(strip.columns.filter((c) => c.level !== null)).toHaveLength(2);
    expect([strip.errors, strip.warnings]).toEqual([1, 1]);
  });

  it("gives a short target a column per segment, and a blank segment no height", () => {
    const strip = stripOf(["ERROR a", "", "bb"], 120);
    expect(strip.columns.map((c) => c.from)).toEqual([0, 1, 2]);
    expect(strip.columns.map((c) => c.level)).toEqual(["error", null, null]);
    expect(strip.columns[1].weight).toBe(0);
  });

  it("reads a target once per width", () => {
    expect(stripOf(log, 120)).toBe(stripOf(log, 120));
  });
});

describe("positionOf", () => {
  it("puts a segment at its own middle along the strip", () => {
    expect(positionOf(0, 4)).toBe(0.125);
    expect(positionOf(3, 4)).toBe(0.875);
  });
});
