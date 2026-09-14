// The chat request the Playground sends (GitHub #164): ignis's own
// `POST /v1/chat/completions`, streaming with the usage chunk, plus the two
// ignis extensions it exposes — `enable_thinking` and the lane tag `class`.

export type LaneTag = "interactive" | "agent";

export type Settings = {
  model: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  /** `null` leaves the cap to the engine. */
  maxTokens: number | null;
  thinking: boolean;
  laneTag: LaneTag;
};

export type Turn = { role: "user" | "assistant"; content: string };

export type ChatRequest = {
  model: string;
  messages: { role: "system" | "user" | "assistant"; content: string }[];
  stream: true;
  stream_options: { include_usage: true };
  temperature: number;
  top_p: number;
  max_tokens?: number;
  enable_thinking: boolean;
  class: LaneTag;
};

/** One message already in the Playground's conversation. */
export type Exchange = Turn & { failed: boolean };

/**
 * What of the conversation goes back to the engine: every exchange except
 * the ones that produced nothing — a prompt whose reply failed or came back
 * empty is dropped together with that reply, so the history never holds two
 * user turns in a row. A reply stopped part-way keeps the text it got.
 * Reasoning is never resent.
 */
export function conversationTurns(entries: Exchange[]): Turn[] {
  const turns: Turn[] = [];
  const empty = (e: Exchange | undefined) => e !== undefined && e.role === "assistant" && (e.failed || !e.content);
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    if (entry.role === "user" && empty(entries[i + 1])) {
      i++;
      continue;
    }
    if (empty(entry)) continue;
    turns.push({ role: entry.role, content: entry.content });
  }
  return turns;
}

export function buildChatRequest(settings: Settings, turns: Turn[]): ChatRequest {
  const system = settings.systemPrompt.trim();
  return {
    model: settings.model,
    messages: [...(system ? [{ role: "system" as const, content: settings.systemPrompt }] : []), ...turns],
    stream: true,
    stream_options: { include_usage: true },
    temperature: settings.temperature,
    top_p: settings.topP,
    ...(settings.maxTokens !== null ? { max_tokens: settings.maxTokens } : {}),
    enable_thinking: settings.thinking,
    class: settings.laneTag,
  };
}
