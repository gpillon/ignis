// How many chat streams this browser may keep open at once (GitHub #220).
// Over HTTP/1.1 a browser opens at most six connections per origin, and a
// stream holds one for as long as it runs: six agents took all six, and every
// other request on the page — the Monitor's /ui/metrics scrape first of all —
// sat queued until an agent finished. Chrome shows that as a long "Stalled".
// A multiplexed connection (h2, h3) has no such limit, which is what a
// production reverse proxy or `--expose` gives; localhost, served plain, does
// not. So the budget is read from the protocol actually in use rather than
// fixed, and only h2 and h3 lift it: anything we cannot read is capped, since
// guessing wrong the other way freezes the page.

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
 * The protocol the page's own origin is being served over, read from `entries`:
 * the newest chat request, or how the document itself arrived. `undefined` when
 * the timeline says nothing — a browser reports "" for a resource it may not
 * disclose.
 */
export function observedProtocol(entries: readonly PerformanceResourceTiming[]): string | undefined {
  const known = entries.filter((entry) => entry.nextHopProtocol !== "");
  const chats = known.filter((entry) => entry.name.includes("/v1/chat/completions"));
  const newest = (chats.length > 0 ? chats : known).reduce<PerformanceResourceTiming | undefined>(
    (best, entry) => (best === undefined || entry.startTime >= best.startTime ? entry : best),
    undefined,
  );
  return newest?.nextHopProtocol;
}

/**
 * The budget for this page, measured now. It is read per call, not once: the
 * first answer comes from the document's own connection, and later ones from
 * the chat requests, which is what a proxy in front could serve differently.
 */
export function currentStreamBudget(): number {
  if (typeof performance === "undefined" || typeof performance.getEntriesByType !== "function") {
    return HTTP1_STREAM_BUDGET;
  }
  const entries = [
    ...(performance.getEntriesByType("navigation") as PerformanceResourceTiming[]),
    ...(performance.getEntriesByType("resource") as PerformanceResourceTiming[]),
  ];
  return streamBudget(observedProtocol(entries));
}
