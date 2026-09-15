import type { LogRow } from "../sessions/sessions.ts";

// How much of the model's context a session fills, from what the browser
// already has: the usage of the session's last complete reply and the
// max_tokens the next request will reserve. ignis refuses a request whose
// prompt plus max_tokens is over the context, so the reservation counts.

export type ContextUsage = {
  /** Prompt plus completion of the last reply with usage; 0 before one. */
  used: number;
  promptTokens: number;
  completionTokens: number;
  /** What the next request reserves on top (its max_tokens), when set. */
  reserve: number | null;
  /** The model's context in tokens, when the server says. */
  limit: number | null;
  /** Shares of the limit for the bar, each clamped to 0..1; null without a limit. */
  usedShare: number | null;
  reserveShare: number | null;
  /** The next request, as set, would be over the context. */
  overflows: boolean;
};

/** `850`, `12.3K`, `41K`: token counts short enough for the bar's label. */
export function compactTokens(n: number): string {
  if (n < 1000) return String(n);
  const k = n / 1000;
  return k < 10 ? `${k.toFixed(1)}K` : `${Math.round(k)}K`;
}

export function contextUsage(log: LogRow[], maxTokens: number | null, limit: number | null): ContextUsage {
  const last = [...log]
    .reverse()
    // Agents' rows measure their own short conversations, not this one.
    .find((row) => row.agent === undefined && row.figures?.promptTokens != null && row.figures.completionTokens != null)
    ?.figures;
  const promptTokens = last?.promptTokens ?? 0;
  const completionTokens = last?.completionTokens ?? 0;
  const used = promptTokens + completionTokens;
  const clamp = (share: number) => Math.min(1, Math.max(0, share));
  const hasLimit = limit !== null && limit > 0;
  return {
    used,
    promptTokens,
    completionTokens,
    reserve: maxTokens,
    limit: hasLimit ? limit : null,
    usedShare: hasLimit ? clamp(used / limit) : null,
    reserveShare: hasLimit ? clamp((maxTokens ?? 0) / limit) : null,
    overflows: hasLimit && maxTokens !== null && used + maxTokens > limit,
  };
}
