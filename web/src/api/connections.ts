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

/** The path every streaming request goes to; `observedProtocol` reads those requests back. */
export const CHAT_PATH = "/v1/chat/completions";

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
 * chat request, or how the page itself arrived. Only that origin counts, since
 * the limit is per origin and a cross-origin resource (a web tool's search,
 * a font) says nothing about the connection the streams share. `undefined`
 * when the timeline says nothing — a browser reports "" for a resource it may
 * not disclose.
 */
export function observedProtocol(entries: readonly PerformanceResourceTiming[], origin: string): string | undefined {
  const ours = entries.filter((entry) => entry.nextHopProtocol !== "" && sameOrigin(entry.name, origin));
  const chats = ours.filter((entry) => entry.name.includes(CHAT_PATH));
  const newest = (chats.length > 0 ? chats : ours).reduce<PerformanceResourceTiming | undefined>(
    (best, entry) => (best === undefined || entry.startTime >= best.startTime ? entry : best),
    undefined,
  );
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
 * keeps a bounded timeline, so once the older entries are dropped this falls
 * back to the page's own connection — the capped answer, never the lifted one.
 */
export function currentStreamBudget(): number {
  const origin = globalThis.location?.origin;
  if (origin === undefined) return HTTP1_STREAM_BUDGET;
  const entries = [
    ...(performance.getEntriesByType("navigation") as PerformanceResourceTiming[]),
    ...(performance.getEntriesByType("resource") as PerformanceResourceTiming[]),
  ];
  return streamBudget(observedProtocol(entries, origin));
}

let streaming = 0;
const waiting: (() => void)[] = [];

/**
 * Runs `stream` once the page can afford another one, and hands the
 * connection straight to whoever is next in line when it ends. `budget` is
 * read when the call arrives, so a page that turns out to be multiplexed
 * stops queueing from the next call on.
 */
export async function withStreamPermit<T>(stream: () => Promise<T>, budget = currentStreamBudget()): Promise<T> {
  if (streaming >= budget) await new Promise<void>((resolve) => waiting.push(resolve));
  else streaming++;
  try {
    return await stream();
  } finally {
    const next = waiting.shift();
    if (next) next();
    else streaming--;
  }
}

/** How many streams are running or queued, for tests. */
export function streamsInFlight(): { running: number; queued: number } {
  return { running: streaming, queued: waiting.length };
}
