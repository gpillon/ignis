import { describe, expect, it } from "vitest";
import { parseExposition } from "./exposition.ts";
import { IGNIS_EXPOSITION } from "./fixture.ts";
import { readSnapshot } from "./snapshot.ts";

describe("readSnapshot", () => {
  it("reads every ADR 0017 contract series", () => {
    const s = readSnapshot(parseExposition(IGNIS_EXPOSITION));
    expect(s.version).toBe("0.1.0");
    expect(s.waiting).toBe(2);
    expect(s.running).toBe(5);
    expect(s.accepted).toBe(42);
    expect(s.completed).toBe(30);
    expect(s.cancelled).toBe(3);
    expect(s.generatedTokens).toBe(12345);
    expect(s.decodedTokens).toBe(12400);
    expect(s.kvEvictions).toBe(4);
    expect(s.prefixReusedTokens).toBe(8192);
    expect(s.rejected).toEqual({ full: 6, unknown_model: 1, oversized: 0 });
    expect(s.unknown).toEqual([]);
  });

  it("reads a histogram as finite bounds plus +Inf, cumulative counts, sum and count", () => {
    const { ttft, duration } = readSnapshot(parseExposition(IGNIS_EXPOSITION));
    expect(ttft?.bounds).toEqual([0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, Infinity]);
    expect(ttft?.cumulative).toEqual([1, 4, 10, 20, 28, 30, 30, 30, 30, 30, 30, 30, 30]);
    expect(ttft?.sum).toBe(15.3);
    expect(ttft?.count).toBe(30);
    expect(duration?.bounds.at(-2)).toBe(600);
  });

  it("leaves absent series null", () => {
    const s = readSnapshot(parseExposition("ignis_requests_accepted_total 3\n"));
    expect(s.accepted).toBe(3);
    expect(s.version).toBeNull();
    expect(s.running).toBeNull();
    expect(s.decodedTokens).toBeNull();
    expect(s.rejected).toEqual({ full: null, unknown_model: null, oversized: null });
    expect(s.ttft).toBeNull();
  });

  it("lists series outside the contract instead of dropping them", () => {
    const text = `${IGNIS_EXPOSITION}# TYPE process_cpu_seconds_total counter\nprocess_cpu_seconds_total 3\nignis_future{a="b"} 1\n`;
    const s = readSnapshot(parseExposition(text));
    expect(s.unknown).toEqual([
      { name: "process_cpu_seconds_total", labels: {}, value: 3 },
      { name: "ignis_future", labels: { a: "b" }, value: 1 },
    ]);
  });

  it("lists a label value the contract does not name, on a metric it does", () => {
    const text = IGNIS_EXPOSITION.replace(
      'ignis_scheduler_requests{state="running"} 5',
      'ignis_scheduler_requests{state="running"} 5\nignis_scheduler_requests{state="prefilling"} 1',
    ).replace('reason="oversized"} 0', 'reason="oversized"} 0\nignis_requests_rejected_total{reason="shutdown"} 2');
    const s = readSnapshot(parseExposition(text));
    expect(s.running).toBe(5);
    expect(s.unknown).toEqual([
      { name: "ignis_scheduler_requests", labels: { state: "prefilling" }, value: 1 },
      { name: "ignis_requests_rejected_total", labels: { reason: "shutdown" }, value: 2 },
    ]);
  });
});
