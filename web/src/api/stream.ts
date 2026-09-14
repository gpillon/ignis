import { apiErrorMessage } from "./errors.ts";
import type { Timeline } from "../metrics/figures.ts";
import { type ChunkEvent, createSseParser, parseChunk } from "./sse.ts";

// One streaming chat request, end to end (GitHub #164): send, read the SSE
// body as it arrives, hand each event to the caller, and keep the timeline
// the figures are computed from. `fetch` and the clock are injectable so
// this is testable without a server.

export type StreamOptions = {
  body: unknown;
  onEvent: (event: ChunkEvent) => void;
  signal?: AbortSignal;
  fetch?: typeof fetch;
  now?: () => number;
};

export type StreamResult = { ok: true; timeline: Timeline } | { ok: false; message: string; timeline: Timeline };

export async function streamChat(options: StreamOptions): Promise<StreamResult> {
  const now = options.now ?? (() => performance.now());
  const doFetch = options.fetch ?? fetch;
  const timeline: Timeline = { sentAt: now(), stopped: false };

  try {
    const response = await doFetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options.body),
      signal: options.signal,
    });
    if (!response.ok) {
      const text = await response.text();
      timeline.endedAt = now();
      return { ok: false, message: apiErrorMessage(response.status, text), timeline };
    }
    if (!response.body) throw new Error("the response has no body");

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    const parser = createSseParser();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const data of parser.push(value)) {
        const at = now();
        for (const event of parseChunk(data)) {
          if (event.kind === "reasoning" || event.kind === "content" || event.kind === "tool_call") {
            timeline.firstTokenAt ??= at;
            timeline.lastTokenAt = at;
          } else if (event.kind === "finish") {
            timeline.finishReason = event.reason;
          } else if (event.kind === "usage") {
            timeline.usage = event.usage;
          } else if (event.kind === "done") {
            timeline.endedAt = at;
          }
          options.onEvent(event);
        }
      }
    }
    timeline.endedAt ??= now();
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
