import { afterEach, describe, expect, it, vi } from "vitest";
import { currentStreamBudget, HTTP1_STREAM_BUDGET, observedProtocol, streamBudget } from "./slots.ts";

describe("streamBudget", () => {
  it("leaves one connection free for the rest of the page over HTTP/1.1", () => {
    expect(streamBudget("http/1.1")).toBe(HTTP1_STREAM_BUDGET);
    expect(HTTP1_STREAM_BUDGET).toBe(5);
  });

  it("does not cap a multiplexed connection", () => {
    expect(streamBudget("h2")).toBe(Number.POSITIVE_INFINITY);
    expect(streamBudget("h3")).toBe(Number.POSITIVE_INFINITY);
  });

  it("caps a protocol it cannot read, since only h2 and h3 are known to multiplex", () => {
    expect(streamBudget(undefined)).toBe(HTTP1_STREAM_BUDGET);
    expect(streamBudget("")).toBe(HTTP1_STREAM_BUDGET);
    expect(streamBudget("http/1.0")).toBe(HTTP1_STREAM_BUDGET);
    expect(streamBudget("spdy/3.1")).toBe(HTTP1_STREAM_BUDGET);
  });
});

describe("observedProtocol", () => {
  const entry = (name: string, nextHopProtocol: string, startTime: number) =>
    ({ name, nextHopProtocol, startTime }) as PerformanceResourceTiming;

  it("reads the protocol the streams themselves were served over", () => {
    const entries = [
      entry("http://localhost:5173/ui/", "http/1.1", 0),
      entry("http://localhost:5173/v1/chat/completions", "h2", 10),
    ];
    expect(observedProtocol(entries)).toBe("h2");
  });

  it("prefers the newest request, so a changed connection wins", () => {
    const entries = [
      entry("http://localhost:5173/v1/chat/completions", "h2", 10),
      entry("http://localhost:5173/v1/chat/completions", "http/1.1", 20),
    ];
    expect(observedProtocol(entries)).toBe("http/1.1");
  });

  it("falls back to how the page itself was served when no request has been made", () => {
    expect(observedProtocol([entry("http://localhost:5173/ui/", "http/1.1", 0)])).toBe("http/1.1");
  });

  it("reads nothing from an empty timeline", () => {
    expect(observedProtocol([])).toBeUndefined();
  });

  it("ignores an entry the browser would not tell us about", () => {
    // A cross-origin resource without Timing-Allow-Origin reports "".
    expect(observedProtocol([entry("https://cdn.example/x.js", "", 5)])).toBeUndefined();
  });
});

describe("currentStreamBudget", () => {
  const entries = (list: Partial<PerformanceResourceTiming>[]) =>
    vi.spyOn(performance, "getEntriesByType").mockImplementation(((kind: string) =>
      kind === "resource" ? list : []) as typeof performance.getEntriesByType);

  afterEach(() => vi.restoreAllMocks());

  it("caps a page served over HTTP/1.1", () => {
    entries([{ name: "http://localhost:5173/v1/chat/completions", nextHopProtocol: "http/1.1", startTime: 1 }]);
    expect(currentStreamBudget()).toBe(HTTP1_STREAM_BUDGET);
  });

  it("lifts the cap behind a proxy that speaks HTTP/2", () => {
    entries([{ name: "https://ignis.example/v1/chat/completions", nextHopProtocol: "h2", startTime: 1 }]);
    expect(currentStreamBudget()).toBe(Number.POSITIVE_INFINITY);
  });

  it("caps when the browser has no timeline to read", () => {
    entries([]);
    expect(currentStreamBudget()).toBe(HTTP1_STREAM_BUDGET);
  });
});
