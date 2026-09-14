import type { Usage } from "./sse.ts";

// Per-request figures (GitHub #164). Every time here is HTTP-observed — read
// with the browser's clock around the stream — never an engine-internal
// timing.

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
  /** The owner stopped the stream before it ended. */
  stopped: boolean;
};

export type Figures = {
  ttftMs: number | null;
  decodeTokensPerSec: number | null;
  durationMs: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  finishReason: string | null;
  /** Stopped before the end: no usage, so no token counts or rate. */
  partial: boolean;
};

export function computeFigures(t: Timeline): Figures {
  const completion = t.usage?.completion_tokens ?? null;
  const decodeSpanMs =
    t.firstTokenAt !== undefined && t.lastTokenAt !== undefined ? t.lastTokenAt - t.firstTokenAt : 0;
  // The first token closes TTFT; the rate covers the ones after it.
  const decodeTokensPerSec =
    completion !== null && completion > 1 && decodeSpanMs > 0 ? (completion - 1) / (decodeSpanMs / 1000) : null;
  return {
    ttftMs: t.firstTokenAt !== undefined ? t.firstTokenAt - t.sentAt : null,
    decodeTokensPerSec,
    durationMs: t.endedAt !== undefined ? t.endedAt - t.sentAt : null,
    promptTokens: t.usage?.prompt_tokens ?? null,
    completionTokens: completion,
    finishReason: t.finishReason ?? null,
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
