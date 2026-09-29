import type { ChunkEvent, Usage } from "../api/sse.ts";

// Per-request figures (GitHub #164). Every time here is observed by the
// browser — read with its clock around the stream, HTTP or the socket
// (GitHub #283) — never an engine-internal timing.

/** When things happened to one request, in milliseconds on one clock. */
export type Timeline = {
  sentAt: number;
  /** The first chunk carrying reasoning or content. */
  firstTokenAt?: number;
  /** The last chunk carrying reasoning or content. */
  lastTokenAt?: number;
  /** `[DONE]`, the end of the body, an error, or the stop. */
  endedAt?: number;
  usage?: Usage;
  finishReason?: string;
  /** The reasoning tokens emitted when the thinking budget forced the block closed; absent when it closed on its own. */
  thinkingForcedAt?: number;
  /** The owner stopped the stream before it ended. */
  stopped: boolean;
  /** The engine was full and queued the request (on the socket; over HTTP a full engine answers 503). */
  queuedAt?: number;
  /** The engine admitted the request: on the socket, its `response.in_progress`. */
  admittedAt?: number;
};

export type Figures = {
  ttftMs: number | null;
  decodeTokensPerSec: number | null;
  durationMs: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  finishReason: string | null;
  /** Where the thinking budget closed the reasoning, in reasoning tokens; absent when the reply closed it itself. */
  thinkingForcedAt?: number;
  /** How long the request waited in the engine's queue; absent when it was admitted at once. Not part of TTFT. */
  queueMs?: number;
  /** Stopped before the end: no usage, so no token counts or rate. */
  partial: boolean;
};

/** What one stream event tells the timeline, at `at`: output, the finish, the usage, the end. */
export function recordEvent(t: Timeline, event: ChunkEvent, at: number): void {
  if (event.kind === "reasoning" || event.kind === "content" || event.kind === "tool_call") {
    t.firstTokenAt ??= at;
    t.lastTokenAt = at;
  } else if (event.kind === "finish") {
    t.finishReason = event.reason;
    if (event.thinkingForcedAt !== undefined) t.thinkingForcedAt = event.thinkingForcedAt;
  } else if (event.kind === "usage") {
    t.usage = event.usage;
  } else if (event.kind === "done") {
    t.endedAt = at;
  }
}

export function computeFigures(t: Timeline): Figures {
  const completion = t.usage?.completion_tokens ?? null;
  const decodeSpanMs =
    t.firstTokenAt !== undefined && t.lastTokenAt !== undefined ? t.lastTokenAt - t.firstTokenAt : 0;
  // The first token closes TTFT; the rate covers the ones after it.
  const decodeTokensPerSec =
    completion !== null && completion > 1 && decodeSpanMs > 0 ? (completion - 1) / (decodeSpanMs / 1000) : null;
  // Time in the engine's queue is back-pressure, not latency: TTFT leaves it out.
  const queueMs = t.queuedAt !== undefined && t.admittedAt !== undefined ? t.admittedAt - t.queuedAt : undefined;
  return {
    ttftMs: t.firstTokenAt !== undefined ? t.firstTokenAt - t.sentAt - (queueMs ?? 0) : null,
    decodeTokensPerSec,
    durationMs: t.endedAt !== undefined ? t.endedAt - t.sentAt : null,
    promptTokens: t.usage?.prompt_tokens ?? null,
    completionTokens: completion,
    finishReason: t.finishReason ?? null,
    ...(t.thinkingForcedAt !== undefined ? { thinkingForcedAt: t.thinkingForcedAt } : {}),
    ...(queueMs !== undefined ? { queueMs } : {}),
    partial: t.stopped,
  };
}

/** `250 ms` below a second, `2.30 s` above, `—` when unknown. */
export function formatMs(ms: number | null): string {
  if (ms === null) return "—";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

/** `25.0 tok/s`, `—` when unknown. */
export function formatRate(tokensPerSec: number | null): string {
  return tokensPerSec === null ? "—" : `${tokensPerSec.toFixed(1)} tok/s`;
}

/** Every figure as display text — one source for the reply line and the session table. */
export function describeFigures(f: Figures) {
  return {
    ttft: formatMs(f.ttftMs),
    decode: formatRate(f.decodeTokensPerSec),
    duration: formatMs(f.durationMs),
    promptTokens: f.promptTokens === null ? "—" : String(f.promptTokens),
    completionTokens: f.completionTokens === null ? "—" : String(f.completionTokens),
    finish: f.partial ? "stopped" : (f.finishReason ?? "—"),
  };
}
