import { authHeaders, keyRequired } from "./auth.ts";
import { withStreamPermit } from "./connections.ts";
import { apiErrorMessage } from "./errors.ts";
import { recordEvent, type Timeline } from "../metrics/figures.ts";
import { buildChatRequest, CHAT_PATH, type ConversationRequest } from "./request.ts";
import { createResponsesSocket, type ResponsesSocket } from "./socket.ts";
import { type ChunkEvent, createSseParser, parseChunk } from "./sse.ts";
import { activeTransport, fallBackToHttp } from "./transport.ts";

// One streaming reply, end to end (GitHub #164): send the page's request,
// hand each event to the caller as it arrives, and keep the timeline the
// figures are computed from. `streamChat` is where every conversation
// request goes; it picks the wire (`transport.ts`): the Responses socket
// (`socket.ts`, GitHub #283), or chat completions over HTTP, read as SSE
// below — which is also where a request goes when the socket cannot be
// opened. Whoever calls it never learns which wire ran.
//
// Over HTTP every stream waits for a connection the page can spare (GitHub
// #220, see `connections.ts`); the wait is before `sentAt`, so a queued
// stream does not report the queue as its own latency. The socket needs no
// such wait: it carries every stream on one connection.

export type StreamOptions = {
  request: ConversationRequest;
  /**
   * The stream this request belongs to on the socket: a session's, an
   * agent's. A stream carries one request at a time and continues its
   * previous reply; a request without one gets a stream of its own.
   */
  streamId?: string;
  onEvent: (event: ChunkEvent) => void;
  signal?: AbortSignal;
  fetch?: typeof fetch;
  now?: () => number;
  /**
   * Called when the request starts: over HTTP when it goes out, which is
   * later than the call when the page had no connection to spare; on the
   * socket when the engine admits it.
   */
  onStart?: () => void;
  /** Called when the engine is full and queues the request (on the socket; over HTTP a full engine answers 503). */
  onQueued?: () => void;
};

export type StreamResult = { ok: true; timeline: Timeline } | { ok: false; message: string; timeline: Timeline };

/** `streamChat` over `socket`; the page's own is below, a test makes one on a fake. */
export function createStreamChat(socket: ResponsesSocket) {
  return async (options: StreamOptions): Promise<StreamResult> => {
    if (activeTransport() === "websocket") {
      const result = await socket.stream(options);
      if (result !== "unavailable") return result;
      fallBackToHttp();
    }
    const { request, streamId: _stream, onQueued: _queued, ...http } = options;
    return streamChatCompletions({ ...http, body: buildChatRequest(request.settings, request.turns, request.extras) });
  };
}

export const streamChat = createStreamChat(createResponsesSocket());

export type ChatStreamOptions = Omit<StreamOptions, "request" | "streamId" | "onQueued"> & { body: unknown };

/** One `POST /v1/chat/completions` stream, on the page's connection budget. */
export async function streamChatCompletions(options: ChatStreamOptions): Promise<StreamResult> {
  return withStreamPermit(() => sendChat(options), { signal: options.signal, onGranted: options.onStart });
}

async function sendChat(options: ChatStreamOptions): Promise<StreamResult> {
  const now = options.now ?? (() => performance.now());
  const doFetch = options.fetch ?? fetch;
  const timeline: Timeline = { sentAt: now(), stopped: false };

  try {
    const response = await doFetch(CHAT_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(options.body),
      signal: options.signal,
    });
    if (response.status === 401) keyRequired();
    if (!response.ok) {
      const text = await response.text();
      timeline.endedAt = now();
      return { ok: false, message: apiErrorMessage(response.status, text), timeline };
    }
    if (!response.body) throw new Error("the response has no body");

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    const parser = createSseParser();
    let failed: string | undefined;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const data of parser.push(value)) {
        const at = now();
        for (const event of parseChunk(data)) {
          if (event.kind === "error") {
            failed = event.message;
            continue;
          }
          recordEvent(timeline, event, at);
          options.onEvent(event);
        }
      }
    }
    timeline.endedAt ??= now();
    // GitHub #296: an engine error is its own chunk, with the engine's message.
    if (failed !== undefined) return { ok: false, message: failed, timeline };
    // ignis ends a stream the engine dropped with a bare `[DONE]`: no finish
    // reason means the reply did not really finish.
    if (timeline.finishReason === undefined) {
      return { ok: false, message: "the stream ended without a finish reason (the engine dropped the request)", timeline };
    }
    return { ok: true, timeline };
  } catch (err) {
    timeline.endedAt = now();
    if (options.signal?.aborted) {
      timeline.stopped = true;
      return { ok: true, timeline };
    }
    return { ok: false, message: String(err), timeline };
  }
}
