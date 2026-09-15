import { afterEach, describe, expect, it } from "vitest";
import { forgetKey, saveKey } from "../api/auth.ts";
import { IGNIS_EXPOSITION } from "./fixture.ts";
import { applyScrape, clampInterval, fetchScrape, initialMonitor, monitorVisible, nextDelay, OFF_RETRY_MS } from "./scrape.ts";

const ok = (text = IGNIS_EXPOSITION) => ({ kind: "ok" as const, text });

describe("fetchScrape", () => {
  afterEach(() => forgetKey());

  const answer = (status: number, body = "") => (async () => new Response(body, { status })) as typeof fetch;

  it("reads the exposition from /ui/metrics with the API key", async () => {
    saveKey("sk-live");
    let seen: [string, RequestInit | undefined] | undefined;
    const doFetch = (async (url: string, init?: RequestInit) => {
      seen = [url, init];
      return new Response("x 1\n", { status: 200 });
    }) as typeof fetch;
    expect(await fetchScrape(doFetch)).toEqual({ kind: "ok", text: "x 1\n" });
    expect(seen?.[0]).toBe("/ui/metrics");
    expect(seen?.[1]?.headers).toEqual({ Authorization: "Bearer sk-live" });
  });

  it("tells metrics being off from a missing key and a failure", async () => {
    expect(await fetchScrape(answer(404))).toEqual({ kind: "off" });
    expect(await fetchScrape(answer(401))).toEqual({ kind: "unauthorized" });
    expect(await fetchScrape(answer(502, "bad gateway"))).toEqual({ kind: "failed", message: "HTTP 502" });
    const down = (async () => {
      throw new TypeError("Failed to fetch");
    }) as typeof fetch;
    expect(await fetchScrape(down)).toEqual({ kind: "failed", message: "TypeError: Failed to fetch" });
  });
});

describe("applyScrape", () => {
  it("goes live on the first 200 and keeps the sample", () => {
    const s = applyScrape(initialMonitor(), ok(), 1_000, 4);
    expect(s.availability).toBe("on");
    expect(s.points).toHaveLength(1);
    expect(s.points[0].snap.accepted).toBe(42);
    expect(s.last).toMatchObject({ at: 1_000, scrapeMs: 4, raw: IGNIS_EXPOSITION, errors: [] });
  });

  it("hides on a 404 and asks for the key on a 401", () => {
    expect(applyScrape(initialMonitor(), { kind: "off" }, 1_000, 2).availability).toBe("off");
    expect(applyScrape(initialMonitor(), { kind: "unauthorized" }, 1_000, 2).availability).toBe("unauthorized");
  });

  it("keeps the history when a live server stops answering", () => {
    const live = applyScrape(initialMonitor(), ok(), 1_000, 4);
    const lost = applyScrape(live, { kind: "failed", message: "HTTP 502" }, 6_000, 9);
    expect(lost.availability).toBe("unreachable");
    expect(lost.message).toBe("HTTP 502");
    expect(lost.points).toHaveLength(1);
  });

  it("marks a restart when the counters go down", () => {
    const live = applyScrape(initialMonitor(), ok(), 1_000, 4);
    const restarted = applyScrape(live, ok(IGNIS_EXPOSITION.replace("ignis_requests_accepted_total 42", "ignis_requests_accepted_total 1").replace("ignis_requests_completed_total 30", "ignis_requests_completed_total 0")), 6_000, 4);
    expect(restarted.restarts).toEqual([6_000]);
  });
});

describe("monitorVisible", () => {
  it("offers the Monitor only once /ui/metrics answered", () => {
    expect(monitorVisible(initialMonitor())).toBe(false);
    expect(monitorVisible(applyScrape(initialMonitor(), { kind: "off" }, 0, 1))).toBe(false);
    expect(monitorVisible(applyScrape(initialMonitor(), { kind: "failed", message: "down" }, 0, 1))).toBe(false);
    expect(monitorVisible(applyScrape(initialMonitor(), { kind: "unauthorized" }, 0, 1))).toBe(true);
    const live = applyScrape(initialMonitor(), ok(), 0, 1);
    expect(monitorVisible(live)).toBe(true);
    expect(monitorVisible(applyScrape(live, { kind: "failed", message: "down" }, 5_000, 1))).toBe(true);
  });
});

describe("polling cadence", () => {
  it("keeps the interval between two and five seconds, five by default", () => {
    expect(clampInterval(500)).toBe(2_000);
    expect(clampInterval(3_000)).toBe(3_000);
    expect(clampInterval(60_000)).toBe(5_000);
    expect(initialMonitor().intervalMs).toBe(5_000);
  });

  it("scrapes at the interval while live, and seldom while off or locked", () => {
    const live = applyScrape(initialMonitor(), ok(), 1_000, 4);
    expect(nextDelay({ ...live, intervalMs: 3_000 })).toBe(3_000);
    expect(nextDelay(applyScrape(initialMonitor(), { kind: "off" }, 0, 1))).toBe(OFF_RETRY_MS);
    expect(nextDelay(applyScrape(initialMonitor(), { kind: "unauthorized" }, 0, 1))).toBe(OFF_RETRY_MS);
  });
});
