import { authHeaders } from "../api/auth.ts";
import { type Family, type ParseError, parseExposition } from "./exposition.ts";
import { append, HISTORY_MS, isRestart, type Point } from "./history.ts";
import { readSnapshot } from "./snapshot.ts";

// Scraping `GET /ui/metrics` from the browser (GitHub #165, ADR 0017): the
// Playground's copy of the exposition on the API listener, key-gated when
// ignis runs with --api-key. A 404 means metrics are off, a 401 that the key
// is missing or wrong. The state below is a pure reduction of scrape results
// so the cadence and the history are testable without a timer.

export type ScrapeResult =
  | { kind: "ok"; text: string }
  | { kind: "off" }
  | { kind: "unauthorized" }
  | { kind: "failed"; message: string };

/** `probing` until the first answer; `unreachable` when a scrape failed. */
export type Availability = "probing" | "on" | "off" | "unauthorized" | "unreachable";

export type MonitorState = {
  availability: Availability;
  /** Why the last scrape failed, when it did. */
  message: string | null;
  paused: boolean;
  intervalMs: number;
  points: Point[];
  last: { at: number; raw: string; scrapeMs: number; families: Family[]; errors: ParseError[] } | null;
  /** When a scrape found the counters gone down, inside the history. */
  restarts: number[];
};

/** The poll interval's range (2–5 s) and its default. */
const MIN_INTERVAL_MS = 2_000;
const MAX_INTERVAL_MS = 5_000;
/** How often to look again while metrics are off, locked or never reached. */
export const OFF_RETRY_MS = 30_000;

export function initialMonitor(): MonitorState {
  return {
    availability: "probing",
    message: null,
    paused: false,
    intervalMs: MAX_INTERVAL_MS,
    points: [],
    last: null,
    restarts: [],
  };
}

export function clampInterval(ms: number): number {
  return Math.min(MAX_INTERVAL_MS, Math.max(MIN_INTERVAL_MS, ms));
}

export async function fetchScrape(doFetch: typeof fetch = fetch): Promise<ScrapeResult> {
  try {
    const res = await doFetch("/ui/metrics", { headers: authHeaders(), cache: "no-store" });
    if (res.status === 404) return { kind: "off" };
    if (res.status === 401) return { kind: "unauthorized" };
    if (!res.ok) return { kind: "failed", message: `HTTP ${res.status}` };
    return { kind: "ok", text: await res.text() };
  } catch (err) {
    return { kind: "failed", message: String(err) };
  }
}

/** The state after a scrape that ended at `at` and took `scrapeMs`. */
export function applyScrape(state: MonitorState, result: ScrapeResult, at: number, scrapeMs: number): MonitorState {
  switch (result.kind) {
    case "off":
      return { ...state, availability: "off", message: null };
    case "unauthorized":
      return { ...state, availability: "unauthorized", message: null };
    case "failed":
      return { ...state, availability: "unreachable", message: result.message };
    case "ok": {
      const parsed = parseExposition(result.text);
      const snap = readSnapshot(parsed);
      const previous = state.points.at(-1);
      const points = append(state.points, { at, snap, scrapeMs }, HISTORY_MS);
      const from = points[0].at;
      const restarts = [...state.restarts, ...(previous && isRestart(previous.snap, snap) ? [at] : [])].filter((t) => t >= from);
      return {
        ...state,
        availability: "on",
        message: null,
        points,
        restarts,
        last: { at, raw: result.text, scrapeMs, families: [...parsed.families.values()], errors: parsed.errors },
      };
    }
  }
}

/** A server that answered before and has a history to show, even while a scrape fails. */
function hasLiveHistory(state: MonitorState): boolean {
  return state.availability === "on" || (state.availability === "unreachable" && state.points.length > 0);
}

/**
 * Whether the page offers the Monitor: metrics answered, or answered 401
 * (the Monitor says the key is wanted), or answered before and stopped.
 * A server without --metrics never shows it.
 */
export function monitorVisible(state: MonitorState): boolean {
  return hasLiveHistory(state) || state.availability === "unauthorized";
}

/** How long to wait before the next scrape. */
export function nextDelay(state: MonitorState): number {
  return hasLiveHistory(state) ? clampInterval(state.intervalMs) : OFF_RETRY_MS;
}
