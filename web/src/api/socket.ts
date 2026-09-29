import { authHeaders, getAuth, keyRequired } from "./auth.ts";
import { apiErrorMessage } from "./errors.ts";
import { recordEvent, type Timeline } from "../metrics/figures.ts";
import { buildResponseCreate, type InputItem, parseResponseEvent, replyItems, responseItems, type ServerEvent, TOKEN_EVENTS } from "./responses.ts";
import type { ToolCall } from "./sse.ts";
import type { StreamOptions, StreamResult } from "./stream.ts";

// The Playground's conversation on the Responses WebSocket (GitHub #283,
// spec 01 for the wire). One socket carries every stream at once — no
// browser cap on connections (GitHub #220) — each session and each agent on
// a named stream of its own. A request is one `response.create`; its events
// come back as the chunk events `streamChat` hands the turn loop, and its
// timeline is read off the socket.
//
// A stream continues its latest completed response with
// `previous_response_id` and only the new items when the request's history
// is exactly that response's history and output; anything else (an edit, a
// regenerate, a fork, a new socket) sends it all, and an id ignis no longer
// has is retried once in full.
//
// When the socket cannot be opened, the browser does not say why: a probe of
// GET /v1/models tells a missing key (the key prompt) from a server or proxy
// without the socket, for which the page falls back to HTTP. The socket, the
// clock and the probe's `fetch` are injectable, so all of it is testable
// without a server.

export const RESPONSES_PATH = "/v1/responses";

/** The named streams ignis lets one connection carry; past them the page opens another socket. */
export const STREAMS_PER_SOCKET = 32;

/** The part of a WebSocket the transport uses; a test hands in a fake. */
export type SocketLike = {
  send(data: string): void;
  close(): void;
  onopen: ((event: unknown) => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onclose: ((event: { code: number }) => void) | null;
  onerror: ((event: unknown) => void) | null;
};

export type SocketDeps = {
  /** Opens a socket; may throw, as `new WebSocket` does for a key that is not a valid subprotocol. */
  open?: (url: string, protocols: string[]) => SocketLike;
  url?: () => string;
  /** For the probe after a failed upgrade. */
  fetch?: typeof fetch;
  now?: () => number;
};

/** What a request on the socket came to; `unavailable` when the socket cannot be opened and HTTP should carry it. */
export type SocketResult = StreamResult | "unavailable";

export type ResponsesSocket = { stream: (options: StreamOptions) => Promise<SocketResult> };

/** The socket on the page's own origin, `wss` when the page is served over TLS. */
function pageUrl(): string {
  const { protocol, host } = globalThis.location;
  return `${protocol === "https:" ? "wss" : "ws"}://${host}${RESPONSES_PATH}`;
}

/** A stream id as ignis takes it: 1-256 characters of `[A-Za-z0-9_.-]`. */
export function streamIdOf(name: string): string {
  return name.replace(/[^A-Za-z0-9_.-]/g, "-").slice(0, 256) || "stream";
}

/** The credential rides as a subprotocol, next to a name ignis selects: a page cannot set headers on a socket. */
function protocols(): string[] {
  const key = getAuth().key;
  return key ? ["responses", `openai-insecure-api-key.${key}`] : ["responses"];
}

/** What one request's events are handed to; `null` when the connection closed under it. */
type Listener = (event: ServerEvent | null) => void;

type Connection = {
  socket: SocketLike | null;
  /** Resolves true once open, false when the socket closed before it opened. */
  ready: Promise<boolean>;
  closed: boolean;
  /** The named streams this connection has carried: ignis counts every id it has seen. */
  ids: Set<string>;
  /** Requests that picked this connection and have not settled; a full connection closes once there are none. */
  users: number;
  /** Per stream, the request whose events are arriving. */
  listeners: Map<string, Listener>;
  /** Per stream, the latest response that completed: its id, and the history it saw followed by its output. */
  latest: Map<string, { id: string; items: InputItem[] }>;
  /** Why the upgrade failed, asked once for everyone waiting on it. */
  probe?: Promise<Probe>;
};

type Probe = { kind: "key"; message: string } | { kind: "fallback" } | { kind: "unreachable"; message: string };

/** How one `response.create` ended, for the request that sent it. */
type Outcome =
  | { kind: "completed"; id: string; output: InputItem[] }
  | { kind: "ended" }
  | { kind: "stopped" }
  | { kind: "failed"; message: string; notFound?: boolean };

const startsWith = (items: InputItem[], prefix: InputItem[]) =>
  prefix.length < items.length && prefix.every((item, i) => JSON.stringify(item) === JSON.stringify(items[i]));

export function createResponsesSocket(deps: SocketDeps = {}): ResponsesSocket {
  const open = deps.open ?? ((url, offered) => new WebSocket(url, offered) as unknown as SocketLike);
  const url = deps.url ?? pageUrl;
  const doFetch = deps.fetch ?? ((input, init) => fetch(input, init));
  const connections: Connection[] = [];
  // Per stream, the request before this one: a stream carries one at a time,
  // so every event on it belongs to the one request it has on the wire.
  const lines = new Map<string, Promise<void>>();
  let unnamed = 0;

  function connect(): Connection {
    const connection: Connection = { socket: null, ready: Promise.resolve(false), closed: false, ids: new Set(), users: 0, listeners: new Map(), latest: new Map() };
    connection.ready = new Promise<boolean>((resolve) => {
      let socket: SocketLike;
      try {
        socket = open(url(), protocols());
      } catch {
        // A key that is not a valid subprotocol token throws: a failed upgrade like any other.
        connection.closed = true;
        return resolve(false);
      }
      connection.socket = socket;
      socket.onopen = () => resolve(true);
      socket.onerror = () => {};
      socket.onclose = () => {
        connection.closed = true;
        resolve(false);
        for (const listener of [...connection.listeners.values()]) listener(null);
      };
      socket.onmessage = (message) => {
        let event: ServerEvent;
        try {
          event = JSON.parse(String(message.data)) as ServerEvent;
        } catch {
          return;
        }
        // The answer to a cancel of a response that had just ended: nothing waits for it.
        if (event.type === "error" && event.error?.code === "response_not_found") return;
        if (event.stream_id !== undefined) connection.listeners.get(event.stream_id)?.(event);
      };
    });
    connections.push(connection);
    return connection;
  }

  /** The connection carrying `streamId`: the one it is on, else the newest with a free id, else a new one. */
  function connectionFor(streamId: string): Connection {
    const live = connections.filter((c) => !c.closed);
    const connection =
      live.find((c) => c.ids.has(streamId)) ?? live.reverse().find((c) => c.ids.size < STREAMS_PER_SOCKET) ?? connect();
    connection.ids.add(streamId);
    connection.users++;
    return connection;
  }

  /** A request is done with `connection`; a full one nobody is using goes, and its streams move to a new one. */
  function leave(connection: Connection) {
    connection.users--;
    if (connection.users === 0 && connection.ids.size >= STREAMS_PER_SOCKET && !connection.closed) {
      connection.closed = true;
      connection.socket?.close();
    }
    const at = connections.indexOf(connection);
    if (connection.closed && connection.users === 0 && at !== -1) connections.splice(at, 1);
  }

  async function probe(): Promise<Probe> {
    try {
      const res = await doFetch("/v1/models", { headers: authHeaders() });
      if (res.status === 401) return { kind: "key", message: apiErrorMessage(401, await res.text()) };
      return { kind: "fallback" };
    } catch (err) {
      // ignis is not there at all: HTTP would fail the same way, and a restart should find the socket again.
      return { kind: "unreachable", message: `Could not reach ignis: ${String(err)}` };
    }
  }

  /**
   * Sends one `response.create` on `streamId` and follows it to its end. A
   * stop ends it for the caller at once; the cancel goes out as soon as the
   * response has an id, and the stream is free again once ignis has ended it.
   */
  function send(
    connection: Connection,
    streamId: string,
    body: object,
    options: StreamOptions,
    timeline: Timeline,
    now: () => number,
    settled: () => void,
  ): Promise<Outcome> {
    return new Promise<Outcome>((resolve) => {
      let responseId: string | undefined;
      let stopped = false;
      let done = false;
      const reply = { reasoning: "", content: "", toolCalls: [] as ToolCall[] };
      const cancel = (id: string) => connection.socket?.send(JSON.stringify({ type: "response.cancel", response_id: id }));
      const finish = (outcome: Outcome) => {
        if (done) return;
        done = true;
        options.signal?.removeEventListener("abort", stop);
        resolve(outcome);
      };
      const stop = () => {
        stopped = true;
        if (responseId !== undefined) cancel(responseId);
        finish({ kind: "stopped" });
      };
      const end = (outcome: Outcome) => {
        connection.listeners.delete(streamId);
        settled();
        finish(outcome);
      };

      connection.listeners.set(streamId, (event) => {
        if (event === null) return end({ kind: "failed", message: "the connection to ignis closed before the reply finished" });
        const at = now();
        if (event.type === "response.created" && event.response) {
          responseId = event.response.id;
          if (stopped) cancel(responseId);
        }
        if (!stopped) {
          if (event.type === "response.queued") {
            timeline.queuedAt = at;
            options.onQueued?.();
          }
          if (event.type === "response.in_progress") {
            timeline.admittedAt = at;
            options.onStart?.();
          }
          if (TOKEN_EVENTS.has(event.type)) {
            timeline.firstTokenAt ??= at;
            timeline.lastTokenAt = at;
          }
          for (const chunk of parseResponseEvent(event)) {
            recordEvent(timeline, chunk, at);
            if (chunk.kind === "reasoning") reply.reasoning += chunk.text;
            if (chunk.kind === "content") reply.content += chunk.text;
            if (chunk.kind === "tool_call") reply.toolCalls.push(chunk.call);
            options.onEvent(chunk);
          }
        }
        if (event.type === "error") {
          if (event.status === 401) keyRequired();
          const message = event.error?.message ?? "ignis refused the request";
          return end({
            kind: "failed",
            message: event.status !== undefined ? `${event.status}: ${message}` : message,
            notFound: event.error?.code === "previous_response_not_found",
          });
        }
        if (event.type === "response.failed") {
          const error = event.response?.error;
          const message = error?.message ?? "the engine failed the reply";
          return end({ kind: "failed", message: error?.code ? `${error.code}: ${message}` : message });
        }
        if (event.type === "response.completed" && event.response) {
          return end({ kind: "completed", id: event.response.id, output: replyItems(reply) });
        }
        if (event.type === "response.incomplete") {
          return end(event.response?.status === "cancelled" ? { kind: "stopped" } : { kind: "ended" });
        }
      });

      if (options.signal?.aborted) return end({ kind: "stopped" });
      options.signal?.addEventListener("abort", stop, { once: true });
      connection.socket?.send(JSON.stringify(body));
      timeline.sentAt = now();
    });
  }

  async function stream(options: StreamOptions): Promise<SocketResult> {
    const now = options.now ?? deps.now ?? (() => performance.now());
    const streamId = streamIdOf(options.streamId ?? `request-${++unnamed}`);
    const timeline: Timeline = { sentAt: now(), stopped: false };
    const stopped = (): StreamResult => {
      timeline.stopped = true;
      timeline.endedAt = now();
      return { ok: true, timeline };
    };

    // Wait for the stream's previous request to end: ignis would hold this one behind it anyway.
    const before = lines.get(streamId);
    let release!: () => void;
    const mine = new Promise<void>((resolve) => (release = resolve));
    const line = (before ?? Promise.resolve()).then(() => mine);
    lines.set(streamId, line);
    await before;
    const releaseLine = () => {
      release();
      if (lines.get(streamId) === line) lines.delete(streamId);
    };
    if (options.signal?.aborted) {
      releaseLine();
      return stopped();
    }

    const connection = connectionFor(streamId);
    try {
      if (!(await connection.ready)) {
        releaseLine();
        const found = await (connection.probe ??= probe());
        if (found.kind === "fallback") return "unavailable";
        if (found.kind === "key") keyRequired();
        timeline.endedAt = now();
        return { ok: false, message: found.message, timeline };
      }
      if (connection.closed || options.signal?.aborted) {
        releaseLine();
        return connection.closed ? { ok: false, message: "the connection to ignis closed before the request went out", timeline } : stopped();
      }

      const request = options.request;
      const items = responseItems(request.turns);
      const parent = connection.latest.get(streamId);
      connection.latest.delete(streamId);
      const continues = parent !== undefined && startsWith(items, parent.items);
      /** One `response.create`: its outcome for this request, and when the stream is free of it. */
      const attempt = (input: InputItem[], parentId?: string) => {
        let settled!: () => void;
        const ended = new Promise<void>((resolve) => (settled = resolve));
        const body = buildResponseCreate(request, streamId, input, parentId);
        return { ended, outcome: send(connection, streamId, body, options, timeline, now, () => settled()) };
      };
      let sent = continues ? attempt(items.slice(parent.items.length), parent.id) : attempt(items);
      let outcome = await sent.outcome;
      if (outcome.kind === "failed" && outcome.notFound && continues && !connection.closed) {
        // ignis no longer has the parent: the same request, whole.
        await sent.ended;
        sent = attempt(items);
        outcome = await sent.outcome;
      }
      void sent.ended.then(releaseLine);

      if (outcome.kind === "completed") connection.latest.set(streamId, { id: outcome.id, items: [...items, ...outcome.output] });
      if (outcome.kind === "stopped") return stopped();
      timeline.endedAt ??= now();
      if (outcome.kind === "failed") return { ok: false, message: outcome.message, timeline };
      return { ok: true, timeline };
    } finally {
      leave(connection);
    }
  }

  return { stream };
}
