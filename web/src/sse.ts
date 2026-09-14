// Server-sent events for `POST /v1/chat/completions` with `stream: true`
// (GitHub #164). `EventSource` cannot POST, so the page reads the body
// itself: `createSseParser` turns arbitrary text reads into complete event
// payloads, `parseChunk` turns one payload into what it means.

export type Usage = { prompt_tokens: number; completion_tokens: number; total_tokens: number };

export type ChunkEvent =
  | { kind: "reasoning"; text: string }
  | { kind: "content"; text: string }
  | { kind: "finish"; reason: string }
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
  choices?: { delta?: { reasoning_content?: string; content?: string }; finish_reason?: string | null }[];
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
    if (choice.finish_reason) events.push({ kind: "finish", reason: choice.finish_reason });
  }
  if (chunk.usage) events.push({ kind: "usage", usage: chunk.usage });
  return events;
}
