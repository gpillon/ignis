import { existsSync, readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { ServerEvent } from "./responses.ts";
import { createResponsesSocket, type SocketLike } from "./socket.ts";
import type { StreamResult } from "./stream.ts";

// The page's socket transport driven through what ignis really sends (GitHub
// #283): the server's golden transcripts (#282), one event per line as the
// socket sent them, which the server's own socket tests assert so they cannot
// drift. Each stream in a transcript gets a request from the page; each
// request must end as its terminal event says, and what the page rebuilds of
// a reply from the deltas must be what ignis says the reply was.

const DIR = new URL("../../../crates/server/tests/fixtures/responses/", import.meta.url);
const files = existsSync(DIR) ? readdirSync(DIR).filter((name) => name.endsWith(".jsonl")) : [];

const read = (name: string): ServerEvent[] =>
  readFileSync(new URL(name, DIR), "utf8")
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as ServerEvent);

const TERMINAL = new Set(["response.completed", "response.incomplete", "response.failed", "error"]);

type Item = { type: string; content?: { text?: string }[]; arguments?: string };

/** Per response of a transcript: what its deltas add up to, and what its done items say. */
function repliesOf(events: ServerEvent[]) {
  const open = new Map<string, { deltas: Record<string, string>; items: Item[]; admitted: boolean; tokenBeforeAdmission: boolean }>();
  const done: { deltas: Record<string, string>; items: Item[]; tokenBeforeAdmission: boolean }[] = [];
  for (const event of events) {
    const stream = event.stream_id ?? "";
    if (event.type === "response.created") open.set(stream, { deltas: {}, items: [], admitted: false, tokenBeforeAdmission: false });
    const reply = open.get(stream);
    if (!reply) continue;
    if (event.type === "response.in_progress") reply.admitted = true;
    if (event.type.endsWith(".delta")) {
      if (!reply.admitted) reply.tokenBeforeAdmission = true;
      reply.deltas[event.type] = (reply.deltas[event.type] ?? "") + (event.delta ?? "");
    }
    if (event.type === "response.output_item.done" && event.item) reply.items.push(event.item as Item);
    if (TERMINAL.has(event.type)) {
      done.push(reply);
      open.delete(stream);
    }
  }
  return done;
}

describe.skipIf(files.length === 0)("the server's golden transcripts", () => {
  it.each(files)("%s: every request ends as its terminal event says", async (name) => {
    const events = read(name);
    let socket: { emit: (event: ServerEvent) => void; open: () => void } | undefined;
    const transport = createResponsesSocket({
      open: () => {
        const fake: SocketLike = { send: () => {}, close: () => {}, onopen: null, onmessage: null, onclose: null, onerror: null };
        socket = { emit: (event) => fake.onmessage?.({ data: JSON.stringify(event) }), open: () => fake.onopen?.({}) };
        return fake;
      },
      url: () => "ws://ignis/v1/responses",
    });
    const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
    const running = new Map<string, Promise<StreamResult | "unavailable">>();
    const ended: { terminal: ServerEvent; result: Promise<StreamResult | "unavailable"> }[] = [];
    for (const event of events) {
      const stream = event.stream_id;
      if (stream === undefined) continue;
      if (!running.has(stream)) {
        running.set(stream, transport.stream({ request: { settings: SETTINGS, turns: [{ role: "user", content: "x" }] }, streamId: stream, onEvent: () => {} }));
        await tick();
        socket?.open();
        await tick();
      }
      socket?.emit(event);
      if (TERMINAL.has(event.type)) {
        ended.push({ terminal: event, result: running.get(stream)! });
        running.delete(stream);
      }
    }
    expect(ended.length).toBeGreaterThan(0);
    for (const { terminal, result } of ended) {
      const outcome = await result;
      if (outcome === "unavailable") throw new Error("the socket was refused");
      if (terminal.type === "error") {
        expect(outcome).toMatchObject({ ok: false, message: `${terminal.status}: ${terminal.error?.message}` });
      } else if (terminal.type === "response.failed") {
        expect(outcome.ok).toBe(false);
      } else if (terminal.response?.status === "cancelled") {
        expect(outcome).toMatchObject({ ok: true, timeline: { stopped: true } });
      } else {
        expect(outcome.ok).toBe(true);
        const calls = (terminal.response?.output ?? []).some((item) => item.type === "function_call");
        expect(outcome.timeline.finishReason).toBe(
          terminal.type === "response.completed" ? (calls ? "tool_calls" : "stop") : "length",
        );
      }
    }
  });

  it.each(files)("%s: a reply is its deltas, and no token comes before the engine admits it", (name) => {
    for (const reply of repliesOf(read(name))) {
      expect(reply.tokenBeforeAdmission).toBe(false);
      // A response may hold several items of a kind (text resumed after a call): together they are its deltas.
      const joined = (type: string, text: (item: Item) => string | undefined) => {
        const items = reply.items.filter((item) => item.type === type);
        return items.length > 0 ? items.map((item) => text(item) ?? "").join("") : undefined;
      };
      const texts = (item: Item) => item.content?.map((part) => part.text ?? "").join("");
      expect(joined("message", texts)).toBe(reply.deltas["response.output_text.delta"]);
      expect(joined("reasoning", texts)).toBe(reply.deltas["response.reasoning_text.delta"]);
      expect(joined("function_call", (item) => item.arguments)).toBe(reply.deltas["response.function_call_arguments.delta"]);
    }
  });
});

const SETTINGS = {
  model: "m",
  systemPrompt: "",
  temperature: 1,
  topP: 0.95,
  maxTokens: null,
  reasoningEffort: "xhigh" as const,
  thinkingBudget: null,
  laneTag: "interactive" as const,
};
