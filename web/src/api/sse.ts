// Server-sent events for `POST /v1/chat/completions` with `stream: true`
// (GitHub #164). `EventSource` cannot POST, so the page reads the body
// itself: `createSseParser` turns arbitrary text reads into complete event
// payloads, `parseChunk` turns one payload into what it means.

export type Usage = { prompt_tokens: number; completion_tokens: number; total_tokens: number };

/** A complete tool call: ignis streams each one whole, in a single delta. */
export type ToolCall = { id: string; name: string; arguments: string };

export type ChunkEvent =
  | { kind: "reasoning"; text: string }
  | { kind: "content"; text: string }
  | { kind: "tool_call"; call: ToolCall }
  /**
   * The reply ended. `thinkingForcedAt` is ignis's forced-close extension
   * (spec server/08): the reasoning tokens emitted when the thinking budget
   * closed the block. It is absent when the reply closed it itself.
   */
  | { kind: "finish"; reason: string; thinkingForcedAt?: number }
  | { kind: "usage"; usage: Usage }
  | { kind: "done" };

/** Buffers text reads and yields the `data` of each complete event. */
export function createSseParser() {
  let buffer = "";
  return {
    push(text: string): string[] {
      buffer = (buffer + text).replace(/\r\n/g, "\n");
      const events: string[] = [];
      let end: number;
      while ((end = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const data = block
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(line.startsWith("data: ") ? 6 : 5));
        if (data.length > 0) events.push(data.join("\n"));
      }
      return events;
    },
  };
}

type WireChunk = {
  choices?: {
    delta?: {
      reasoning_content?: string;
      content?: string;
      tool_calls?: { id?: string; function?: { name?: string; arguments?: string } }[];
    };
    finish_reason?: string | null;
    /** On the chunk carrying `finish_reason`, and only when the thinking budget forced the close. */
    thinking_budget_forced_at?: unknown;
  }[];
  usage?: Usage | null;
};

/** What one `chat.completion.chunk` payload (or `[DONE]`) carries. */
export function parseChunk(data: string): ChunkEvent[] {
  if (data.trim() === "[DONE]") return [{ kind: "done" }];
  let chunk: WireChunk;
  try {
    chunk = JSON.parse(data) as WireChunk;
  } catch {
    return [];
  }
  const events: ChunkEvent[] = [];
  for (const choice of chunk.choices ?? []) {
    if (choice.delta?.reasoning_content) events.push({ kind: "reasoning", text: choice.delta.reasoning_content });
    if (choice.delta?.content) events.push({ kind: "content", text: choice.delta.content });
    for (const call of choice.delta?.tool_calls ?? []) {
      if (call.id && call.function?.name) {
        events.push({ kind: "tool_call", call: { id: call.id, name: call.function.name, arguments: call.function.arguments ?? "{}" } });
      }
    }
    if (choice.finish_reason) {
      const forcedAt = choice.thinking_budget_forced_at;
      const forced = typeof forcedAt === "number" && Number.isInteger(forcedAt) && forcedAt >= 0;
      events.push({ kind: "finish", reason: choice.finish_reason, ...(forced ? { thinkingForcedAt: forcedAt } : {}) });
    }
  }
  if (chunk.usage) events.push({ kind: "usage", usage: chunk.usage });
  return events;
}
