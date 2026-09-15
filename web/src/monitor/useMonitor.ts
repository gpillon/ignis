import { useEffect, useState, useSyncExternalStore } from "react";
import { subscribeAuth } from "../api/auth.ts";
import { applyScrape, clampInterval, fetchScrape, initialMonitor, type MonitorState, monitorVisible, nextDelay, type ScrapeResult } from "./scrape.ts";

// The one scraper the page shares: it polls /ui/metrics while anything on
// the page listens, at the chosen interval while metrics answer and every
// OFF_RETRY_MS while they don't, and scrapes again at once when the API key
// changes. The history lives here, in memory; a reload starts it over.

export type MonitorStore = {
  get: () => MonitorState;
  subscribe: (listener: () => void) => () => void;
  setPaused: (paused: boolean) => void;
  setInterval: (ms: number) => void;
};

/** A scraper over `scrape`, re-scraping when `onKeyChange` fires. The page's one is `monitor` below. */
export function createMonitorStore({
  scrape = fetchScrape,
  onKeyChange = subscribeAuth,
}: { scrape?: () => Promise<ScrapeResult>; onKeyChange?: (listener: () => void) => () => void } = {}): MonitorStore {
  let state = initialMonitor();
  const listeners = new Set<() => void>();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let inflight = false;
  let stopKeyWatch: (() => void) | undefined;

  const update = (next: MonitorState) => {
    state = next;
    for (const listener of listeners) listener();
  };

  const schedule = (delayMs: number) => {
    clearTimeout(timer);
    if (listeners.size === 0 || state.paused) return;
    timer = setTimeout(() => void scrapeNow(), delayMs);
  };

  async function scrapeNow() {
    if (inflight || state.paused) return;
    inflight = true;
    clearTimeout(timer);
    const started = performance.now();
    const result = await scrape();
    inflight = false;
    update(applyScrape(state, result, Date.now(), performance.now() - started));
    schedule(nextDelay(state));
  }

  return {
    get: () => state,
    subscribe(listener) {
      listeners.add(listener);
      if (listeners.size === 1) {
        stopKeyWatch = onKeyChange(() => void scrapeNow());
        void scrapeNow();
      }
      return () => {
        listeners.delete(listener);
        if (listeners.size > 0) return;
        clearTimeout(timer);
        stopKeyWatch?.();
      };
    },
    setPaused(paused) {
      update({ ...state, paused });
      if (paused) clearTimeout(timer);
      else void scrapeNow();
    },
    setInterval(ms) {
      update({ ...state, intervalMs: clampInterval(ms) });
      if (!inflight) schedule(nextDelay(state));
    },
  };
}

const monitor = createMonitorStore();

export const setMonitorPaused = monitor.setPaused;
export const setMonitorInterval = monitor.setInterval;

export function useMonitor(): MonitorState {
  return useSyncExternalStore(monitor.subscribe, monitor.get, monitor.get);
}

const visible = () => monitorVisible(monitor.get());

/** Whether to offer the Monitor — a boolean, so its listeners re-render only when it flips. */
export function useMonitorVisible(): boolean {
  return useSyncExternalStore(monitor.subscribe, visible, visible);
}

/** The wall clock, ticking every `ms`: for "scraped 3 s ago". */
export function useNow(ms = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return now;
}
