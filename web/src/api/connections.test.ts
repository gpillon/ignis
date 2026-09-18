import { afterEach, describe, expect, it, vi } from "vitest";
import { currentStreamBudget, HTTP1_STREAM_BUDGET, observedProtocol, streamBudget, streamsInFlight, withStreamPermit } from "./connections.ts";

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

const ORIGIN = "http://localhost:5173";

describe("observedProtocol", () => {
  const entry = (name: string, nextHopProtocol: string, startTime: number) =>
    ({ name, nextHopProtocol, startTime }) as PerformanceResourceTiming;

  it("reads the protocol the streams themselves were served over", () => {
    const entries = [
      entry("http://localhost:5173/ui/", "h3", 0),
      entry("http://localhost:5173/v1/chat/completions", "h2", 10),
    ];
    expect(observedProtocol(entries, ORIGIN)).toBe("h2");
  });

  it("prefers the newest request, so a changed connection wins", () => {
    const entries = [
      entry("http://localhost:5173/v1/chat/completions", "h2", 10),
      entry("http://localhost:5173/v1/chat/completions", "http/1.1", 20),
    ];
    expect(observedProtocol(entries, ORIGIN)).toBe("http/1.1");
  });

  it("reads nothing before the first chat request, leaving the page's own connection to answer", () => {
    expect(observedProtocol([entry("http://localhost:5173/ui/", "http/1.1", 0)], ORIGIN)).toBeUndefined();
  });

  it("reads nothing from an empty timeline", () => {
    expect(observedProtocol([], ORIGIN)).toBeUndefined();
  });

  it("ignores an entry the browser would not tell us about", () => {
    // A cross-origin resource without Timing-Allow-Origin reports "".
    expect(observedProtocol([entry("https://cdn.example/x.js", "", 5)], ORIGIN)).toBeUndefined();
  });

  it("ignores another origin's connection, which says nothing about ours", () => {
    // Another ignis, proxied over h2, answering the same path.
    const entries = [
      entry("http://localhost:5173/v1/chat/completions", "http/1.1", 3),
      entry("https://ignis.example/v1/chat/completions", "h2", 9),
    ];
    expect(observedProtocol(entries, ORIGIN)).toBe("http/1.1");
    expect(observedProtocol([entry("https://api.tavily.com/search", "h2", 9)], ORIGIN)).toBeUndefined();
  });
});

describe("currentStreamBudget", () => {
  const served = (nextHopProtocol: string, name = `${ORIGIN}/v1/chat/completions`) => {
    vi.stubGlobal("location", { origin: ORIGIN });
    const timed = { name, nextHopProtocol, startTime: 1 } as PerformanceResourceTiming;
    vi.spyOn(performance, "getEntriesByType").mockImplementation(((kind: string) =>
      kind === "resource" ? [timed] : []) as typeof performance.getEntriesByType);
  };

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("caps a page served over HTTP/1.1", () => {
    served("http/1.1");
    expect(currentStreamBudget()).toBe(HTTP1_STREAM_BUDGET);
  });

  it("lifts the cap behind a proxy that speaks HTTP/2", () => {
    served("h2");
    expect(currentStreamBudget()).toBe(Number.POSITIVE_INFINITY);
  });

  it("is not lifted by another origin served over HTTP/2", () => {
    served("h2", "https://api.tavily.com/search");
    expect(currentStreamBudget()).toBe(HTTP1_STREAM_BUDGET);
  });

  it("caps where there is no page to read, as under a test runner", () => {
    expect(currentStreamBudget()).toBe(HTTP1_STREAM_BUDGET);
  });

  it("falls back to how the page itself arrived until a chat request has been served", () => {
    vi.stubGlobal("location", { origin: ORIGIN });
    const page = { name: `${ORIGIN}/ui/`, nextHopProtocol: "h2", startTime: 0 } as PerformanceResourceTiming;
    vi.spyOn(performance, "getEntriesByType").mockImplementation(((kind: string) =>
      kind === "navigation" ? [page] : []) as typeof performance.getEntriesByType);
    expect(currentStreamBudget()).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("withStreamPermit", () => {
  const held = () => {
    let release = () => {};
    const done = new Promise<void>((resolve) => {
      release = resolve;
    });
    return { done, release };
  };

  it("runs no more streams at once than the budget allows", async () => {
    const streams = Array.from({ length: 4 }, held);
    let peak = 0;
    const runs = streams.map((stream) =>
      withStreamPermit(async () => {
        peak = Math.max(peak, streamsInFlight().running);
        await stream.done;
        return "done";
      }, 2),
    );
    await Promise.resolve();
    expect(streamsInFlight()).toEqual({ running: 2, queued: 2 });
    for (const stream of streams) stream.release();
    expect(await Promise.all(runs)).toEqual(["done", "done", "done", "done"]);
    expect(peak).toBe(2);
    expect(streamsInFlight()).toEqual({ running: 0, queued: 0 });
  });

  it("hands the connection to the next in line, so a queued stream starts at once", async () => {
    const first = held();
    const started: string[] = [];
    const running = withStreamPermit(async () => {
      started.push("first");
      await first.done;
    }, 1);
    const queued = withStreamPermit(async () => {
      started.push("second");
    }, 1);
    await Promise.resolve();
    expect(started).toEqual(["first"]);
    first.release();
    await Promise.all([running, queued]);
    expect(started).toEqual(["first", "second"]);
  });

  it("frees the connection when a stream throws", async () => {
    await expect(
      withStreamPermit(() => {
        throw new Error("the engine dropped it");
      }, 1),
    ).rejects.toThrow("the engine dropped it");
    expect(streamsInFlight()).toEqual({ running: 0, queued: 0 });
  });
});
