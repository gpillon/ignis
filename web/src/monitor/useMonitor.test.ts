import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { IGNIS_EXPOSITION } from "./fixture.ts";
import type { ScrapeResult } from "./scrape.ts";
import { OFF_RETRY_MS } from "./scrape.ts";
import { createMonitorStore } from "./useMonitor.ts";

// The shared scraper's timing, on fake timers: it scrapes while someone
// listens, at the interval, never while paused, seldom while metrics are off,
// again when the key changes, and not at all once the last listener leaves.

describe("createMonitorStore", () => {
  let keyChanged: () => void = () => {};
  const onKeyChange = (listener: () => void) => {
    keyChanged = listener;
    return () => (keyChanged = () => {});
  };

  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const counting = (result: ScrapeResult = { kind: "ok", text: IGNIS_EXPOSITION }) => {
    const scrape = vi.fn(async () => result);
    return { scrape, store: createMonitorStore({ scrape, onKeyChange }) };
  };

  it("scrapes on the first listener, then at the interval", async () => {
    const { scrape, store } = counting();
    const stop = store.subscribe(() => {});
    await vi.advanceTimersByTimeAsync(0);
    expect(scrape).toHaveBeenCalledTimes(1);
    expect(store.get().availability).toBe("on");
    await vi.advanceTimersByTimeAsync(5_000);
    expect(scrape).toHaveBeenCalledTimes(2);
    store.setInterval(2_000);
    await vi.advanceTimersByTimeAsync(2_000);
    expect(scrape).toHaveBeenCalledTimes(3);
    stop();
  });

  it("stops while paused and scrapes at once on resume", async () => {
    const { scrape, store } = counting();
    const stop = store.subscribe(() => {});
    await vi.advanceTimersByTimeAsync(0);
    store.setPaused(true);
    await vi.advanceTimersByTimeAsync(20_000);
    expect(scrape).toHaveBeenCalledTimes(1);
    store.setPaused(false);
    await vi.advanceTimersByTimeAsync(0);
    expect(scrape).toHaveBeenCalledTimes(2);
    stop();
  });

  it("looks again only every OFF_RETRY_MS while metrics are off, and at once when the key changes", async () => {
    const { scrape, store } = counting({ kind: "off" });
    const stop = store.subscribe(() => {});
    await vi.advanceTimersByTimeAsync(OFF_RETRY_MS - 1);
    expect(scrape).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(scrape).toHaveBeenCalledTimes(2);
    keyChanged();
    await vi.advanceTimersByTimeAsync(0);
    expect(scrape).toHaveBeenCalledTimes(3);
    stop();
  });

  it("stops polling when the last listener leaves", async () => {
    const { scrape, store } = counting();
    const stopA = store.subscribe(() => {});
    const stopB = store.subscribe(() => {});
    await vi.advanceTimersByTimeAsync(0);
    stopA();
    await vi.advanceTimersByTimeAsync(5_000);
    expect(scrape).toHaveBeenCalledTimes(2);
    stopB();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(scrape).toHaveBeenCalledTimes(2);
  });
});
