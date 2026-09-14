import { describe, expect, it } from "vitest";
import type { ChunkEvent } from "./sse.ts";
import { streamChat } from "./stream.ts";

const sse = (delta: object, finish: string | null = null) =>
  `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;

/** A fake `fetch` answering with `parts` as separate body reads. */
function fakeFetch(parts: string[], status = 200): typeof fetch {
  return (async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const part of parts) controller.enqueue(encoder.encode(part));
        controller.close();
      },
    });
    return new Response(body, { status });
  }) as typeof fetch;
}

/** A clock advancing 10 ms per read. */
function ticking() {
  let t = 0;
  return () => (t += 10);
}

describe("streamChat with tool calls", () => {
  it("passes a tool call on and times it as output", async () => {
    const events: ChunkEvent[] = [];
    const call = { index: 0, id: "call_0", type: "function", function: { name: "agent", arguments: '{"prompt":"x"}' } };
    const body = sse({ tool_calls: [call] }) + sse({}, "tool_calls") + "data: [DONE]\n\n";
    const result = await streamChat({ body: {}, fetch: fakeFetch([body]), now: ticking(), onEvent: (e) => events.push(e) });
    expect(result.ok).toBe(true);
    expect(events[0]).toEqual({ kind: "tool_call", call: { id: "call_0", name: "agent", arguments: '{"prompt":"x"}' } });
    expect(result.timeline.firstTokenAt).toBeDefined();
    expect(result.timeline.finishReason).toBe("tool_calls");
  });
});

describe("streamChat", () => {
  it("delivers the events in order and records the request's timeline", async () => {
    const events: ChunkEvent[] = [];
    const whole =
      sse({ reasoning_content: "hm" }) +
      sse({ content: "Hi" }) +
      sse({}, "stop") +
      `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } })}\n\n` +
      "data: [DONE]\n\n";
    // Split mid-event, so the parser has to carry state across reads.
    const parts = [whole.slice(0, 17), whole.slice(17, 90), whole.slice(90)];

    const result = await streamChat({ body: {}, fetch: fakeFetch(parts), now: ticking(), onEvent: (e) => events.push(e) });

    expect(events.map((e) => e.kind)).toEqual(["reasoning", "content", "finish", "usage", "done"]);
    expect(result.ok).toBe(true);
    const t = result.timeline;
    expect(t.stopped).toBe(false);
    expect(t.finishReason).toBe("stop");
    expect(t.usage).toEqual({ prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 });
    expect(t.firstTokenAt).toBeGreaterThan(t.sentAt);
    expect(t.lastTokenAt).toBeGreaterThanOrEqual(t.firstTokenAt!);
    expect(t.endedAt).toBeGreaterThanOrEqual(t.lastTokenAt!);
  });

  it("reassembles a UTF-8 character split across two body reads", async () => {
    const bytes = new TextEncoder().encode(sse({ content: "caffè" }) + sse({}, "stop") + "data: [DONE]\n\n");
    const cut = bytes.indexOf(0xc3) + 1; // inside the two-byte "è"
    const splitFetch = (async () =>
      new Response(
        new ReadableStream<Uint8Array>({
          start(c) {
            c.enqueue(bytes.slice(0, cut));
            c.enqueue(bytes.slice(cut));
            c.close();
          },
        }),
      )) as typeof fetch;
    const texts: string[] = [];
    await streamChat({ body: {}, fetch: splitFetch, now: ticking(), onEvent: (e) => e.kind === "content" && texts.push(e.text) });
    expect(texts).toEqual(["caffè"]);
  });

  it("reports a stream the engine ended without a finish reason as failed", async () => {
    const result = await streamChat({
      body: {},
      fetch: fakeFetch([sse({ content: "Hi" }), "data: [DONE]\n\n"]),
      now: ticking(),
      onEvent: () => {},
    });
    expect(result.ok).toBe(false);
    expect(result.timeline.firstTokenAt).toBeDefined();
  });

  it("returns the API error without streaming", async () => {
    const body = JSON.stringify({ error: { message: "messages must be non-empty", type: "invalid_request_error", code: null } });
    const result = await streamChat({ body: {}, fetch: fakeFetch([body], 400), now: ticking(), onEvent: () => {} });
    expect(result).toMatchObject({ ok: false, message: "400: messages must be non-empty" });
  });

  it("ends a stopped request as partial rather than as an error", async () => {
    const controller = new AbortController();
    const abortingFetch = (async (_url: unknown, init?: RequestInit) => {
      const body = new ReadableStream<Uint8Array>({
        start(c) {
          c.enqueue(new TextEncoder().encode(sse({ content: "Hel" })));
          init?.signal?.addEventListener("abort", () => c.error(new DOMException("aborted", "AbortError")));
        },
      });
      return new Response(body, { status: 200 });
    }) as typeof fetch;

    const result = await streamChat({
      body: {},
      signal: controller.signal,
      fetch: abortingFetch,
      now: ticking(),
      onEvent: (e) => {
        if (e.kind === "content") controller.abort();
      },
    });

    expect(result.ok).toBe(true);
    expect(result.timeline.stopped).toBe(true);
    expect(result.timeline.firstTokenAt).toBeDefined();
    expect(result.timeline.usage).toBeUndefined();
  });
});
