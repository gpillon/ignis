import { CHAT_PATH } from "./request.ts";

// How many requests this browser may keep streaming at once (GitHub #220).
// Over HTTP/1.1 a browser opens at most six connections per origin, and a
// stream holds one for as long as it runs: six agents took all six, and every
// other request on the page — the Monitor's /ui/metrics scrape first of all —
// sat queued until an agent finished. Chrome shows that as a long "Stalled".
// A multiplexed connection (h2, h3) has no such limit, which is what a
// production reverse proxy or `--expose` gives; localhost, served plain, does
// not. So the budget is read from the protocol actually in use rather than
// fixed, and only h2 and h3 lift it: anything we cannot read is capped, since
// guessing wrong the other way freezes the page.
//
// The budget is held here, around every stream, rather than by whoever starts
// them: agents, an agent's own tool round and the JavaScript safety check all
// open one, and a reservation each of them counts separately is no
// reservation at all.

/** Connections a browser opens per origin over HTTP/1.1. */
export const BROWSER_CONNECTIONS_PER_ORIGIN = 6;

/** Connections kept free for the rest of the page: the metrics scrape, a new chat, an asset. */
export const CONNECTIONS_RESERVED_FOR_THE_PAGE = 1;

/** Streams allowed at once over HTTP/1.1. */
export const HTTP1_STREAM_BUDGET = BROWSER_CONNECTIONS_PER_ORIGIN - CONNECTIONS_RESERVED_FOR_THE_PAGE;

/** The ALPN ids that multiplex streams over one connection. */
const MULTIPLEXED = ["h2", "h3"];

/** How many streams may run at once over `protocol`; unbounded when it multiplexes. */
export function streamBudget(protocol: string | undefined): number {
  return protocol !== undefined && MULTIPLEXED.includes(protocol) ? Number.POSITIVE_INFINITY : HTTP1_STREAM_BUDGET;
}

/**
 * The protocol `origin` is being served over, read from `entries`: the newest
 * chat request to that origin, since the streams are what the budget is
 * about. Only that origin counts — a cross-origin resource (a web tool's
 * search, a font) says nothing about the connection the streams share, and an
 * h2 one used to lift the cap on a page served over HTTP/1.1. `undefined`
 * when no chat request has been made yet, or when the browser reports "" for
 * a resource it may not disclose; the caller then falls back to how the page
 * itself arrived.
 *
 * The name is matched before the origin is parsed: a page that has been open
 * for a while carries hundreds of entries, and parsing every one of them to
 * answer this costs more than the answer is worth.
 */
export function observedProtocol(entries: readonly PerformanceResourceTiming[], origin: string): string | undefined {
  let newest: PerformanceResourceTiming | undefined;
  for (const entry of entries) {
    if (entry.nextHopProtocol === "" || !entry.name.includes(CHAT_PATH)) continue;
    if (newest !== undefined && entry.startTime < newest.startTime) continue;
    if (sameOrigin(entry.name, origin)) newest = entry;
  }
  return newest?.nextHopProtocol;
}

function sameOrigin(name: string, origin: string): boolean {
  try {
    return new URL(name, origin).origin === origin;
  } catch {
    return false;
  }
}

/**
 * The budget for this page, measured now. It is read per call, not once: the
 * first answer comes from the page's own connection and later ones from the
 * chat requests, which a proxy in front could serve differently. The browser
 * keeps a bounded timeline, so once the older chat requests are dropped from
 * it this answers from the page's own connection again — the same origin, so
 * the same answer unless the server was replaced under the open page.
 */
export function currentStreamBudget(): number {
  const origin = globalThis.location?.origin;
  if (origin === undefined) return HTTP1_STREAM_BUDGET;
  const streams = performance.getEntriesByType("resource") as PerformanceResourceTiming[];
  const page = performance.getEntriesByType("navigation") as PerformanceResourceTiming[];
  return streamBudget(observedProtocol(streams, origin) ?? page[0]?.nextHopProtocol);
}

export type PermitOptions = {
  /** How many streams may run at once; read once, when the call arrives. */
  budget?: number;
  /** Stops waiting for a connection when the caller gives up. */
  signal?: AbortSignal;
  /** Called when the stream may go, which is at once unless it had to queue. */
  onGranted?: () => void;
};

export type StreamPermits = {
  withPermit: <T>(stream: () => Promise<T>, options?: PermitOptions) => Promise<T>;
  inFlight: () => { running: number; queued: number };
};

/**
 * A budget and the streams holding it. The page shares the one below; a test
 * makes its own so it starts from nothing.
 *
 * A stream that has to wait is handed the connection of whoever finishes
 * first, rather than re-entering the check, so a queued stream starts the
 * moment one ends. A caller that gives up while queued leaves the queue and
 * runs anyway: its request is already aborted, so it opens no connection.
 */
export function createStreamPermits(): StreamPermits {
  let running = 0;
  const waiting: (() => void)[] = [];

  const leave = (grant: () => void) => {
    const at = waiting.indexOf(grant);
    if (at !== -1) waiting.splice(at, 1);
  };

  return {
    inFlight: () => ({ running, queued: waiting.length }),
    async withPermit<T>(stream: () => Promise<T>, options: PermitOptions = {}): Promise<T> {
      const budget = options.budget ?? currentStreamBudget();
      let holds = true;
      if (running >= budget) {
        holds = await new Promise<boolean>((resolve) => {
          const grant = () => resolve(true);
          waiting.push(grant);
          options.signal?.addEventListener("abort", () => {
            leave(grant);
            resolve(false);
          }, { once: true });
        });
      } else running++;
      options.onGranted?.();
      try {
        return await stream();
      } finally {
        if (holds) {
          const next = waiting.shift();
          if (next) next();
          else running--;
        }
      }
    },
  };
}

const permits = createStreamPermits();

/** Runs `stream` on the page's own budget. */
export const withStreamPermit = permits.withPermit;

/** How many streams the page is running or queueing. */
export const streamsInFlight = permits.inFlight;
