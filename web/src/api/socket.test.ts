import { beforeEach, describe, expect, it } from "vitest";
import { computeFigures } from "../metrics/figures.ts";
import { forgetKey, getAuth, saveKey } from "./auth.ts";
import type { ConversationRequest, Settings, Turn } from "./request.ts";
import type { ServerEvent } from "./responses.ts";
import { createResponsesSocket, type SocketLike, STREAMS_PER_SOCKET, streamIdOf } from "./socket.ts";
import type { ChunkEvent } from "./sse.ts";
import { createStreamChat, type StreamOptions, streamChatCompletions } from "./stream.ts";
import { chooseTransport, getTransport } from "./transport.ts";

// The socket transport through its public contract (GitHub #283): a fake
// WebSocket stands in for ignis and plays spec 01's side of the wire.

/** A socket the test opens, refuses, drops and speaks through. */
class FakeSocket implements SocketLike {
  sent: Record<string, unknown>[] = [];
  closedByPage = false;
  onopen: SocketLike["onopen"] = null;
  onmessage: SocketLike["onmessage"] = null;
  onclose: SocketLike["onclose"] = null;
  onerror: SocketLike["onerror"] = null;
  constructor(
    readonly url: string,
    readonly protocols: string[],
  ) {}
  send(data: string) {
    this.sent.push(JSON.parse(data) as Record<string, unknown>);
  }
  close() {
    this.closedByPage = true;
    this.onclose?.({ code: 1000 });
  }
  accept() {
    this.onopen?.({});
  }
  /** What the browser shows of a refused upgrade: a close before the open, 1006. */
  refuse() {
    this.onerror?.({});
    this.onclose?.({ code: 1006 });
  }
  drop() {
    this.onclose?.({ code: 1006 });
  }
  emit(event: ServerEvent) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
  /** The `response.create` events sent so far. */
  creates() {
    return this.sent.filter((e) => e.type === "response.create");
  }
}

const settings: Settings = {
  model: "m",
  systemPrompt: "",
  temperature: 1,
  topP: 0.95,
  maxTokens: null,
  reasoningEffort: "xhigh",
  thinkingBudget: null,
  laneTag: "interactive",
};

const request = (turns: Turn[]): ConversationRequest => ({ settings, turns });

const usage = { input_tokens: 5, output_tokens: 3, total_tokens: 8 };

/** ignis's events for one response on `stream`: admitted at once, `text` and optional calls, completed. */
function reply(stream: string, id: string, { reasoning = "", text = "", calls = [] as { call_id: string; name: string; arguments: string }[] } = {}): ServerEvent[] {
  const response = { id, status: "in_progress" };
  const fc = calls.map((c) => ({ type: "function_call", ...c, status: "completed" }));
  return [
    { type: "response.created", response },
    { type: "response.in_progress", response },
    ...(reasoning ? [{ type: "response.reasoning_text.delta", delta: reasoning }] : []),
    ...(text ? [{ type: "response.output_text.delta", delta: text }] : []),
    ...fc.flatMap((item) => [
      { type: "response.function_call_arguments.delta", delta: item.arguments },
      { type: "response.output_item.done", item },
    ]),
    {
      type: "response.completed",
      response: {
        id,
        status: "completed",
        output: [
          ...(reasoning ? [{ type: "reasoning", content: [{ type: "reasoning_text", text: reasoning }] }] : []),
          ...(text ? [{ type: "message", content: [{ type: "output_text", text }] }] : []),
          ...fc,
        ],
        usage,
      },
    },
  ].map((event) => ({ ...event, stream_id: stream }));
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

/** A clock advancing 10 ms per read. */
function ticking() {
  let t = 0;
  return () => (t += 10);
}

function harness(probeStatus: number | "down" = 200) {
  const sockets: FakeSocket[] = [];
  const probes: RequestInit[] = [];
  const socket = createResponsesSocket({
    open: (url, protocols) => {
      const fake = new FakeSocket(url, protocols);
      sockets.push(fake);
      return fake;
    },
    url: () => "ws://ignis/v1/responses",
    fetch: (async (_url: string, init?: RequestInit) => {
      probes.push(init ?? {});
      if (probeStatus === "down") throw new TypeError("Failed to fetch");
      return new Response(probeStatus === 401 ? '{"error":{"message":"incorrect API key provided"}}' : '{"data":[]}', { status: probeStatus });
    }) as typeof fetch,
    now: ticking(),
  });
  /** Starts a request and collects what it hands the turn loop. */
  const start = (turns: Turn[], options: Partial<StreamOptions> = {}) => {
    const events: ChunkEvent[] = [];
    const result = socket.stream({ request: request(turns), onEvent: (e) => events.push(e), ...options });
    return { events, result };
  };
  return { socket, sockets, probes, start };
}

const user = (content: string): Turn => ({ role: "user", content });

beforeEach(() => {
  saveKey("sk-test");
  chooseTransport("websocket");
});

describe("the socket and its key", () => {
  it("opens one socket on the page's origin and sends the key as the credential subprotocol", async () => {
    const h = harness();
    h.start([user("hi")], { streamId: "session-1" });
    await tick();
    expect(h.sockets).toHaveLength(1);
    expect(h.sockets[0].url).toBe("ws://ignis/v1/responses");
    expect(h.sockets[0].protocols).toEqual(["responses", "openai-insecure-api-key.sk-test"]);
  });

  it("offers no credential without a key", async () => {
    forgetKey();
    const h = harness();
    h.start([user("hi")]);
    await tick();
    expect(h.sockets[0].protocols).toEqual(["responses"]);
  });

  it("asks for the key when the upgrade fails and the probe answers 401", async () => {
    const h = harness(401);
    const { result } = h.start([user("hi")]);
    await tick();
    h.sockets[0].refuse();
    expect(await result).toMatchObject({ ok: false, message: "401: incorrect API key provided" });
    expect(h.probes[0].headers).toMatchObject({ Authorization: "Bearer sk-test" });
    expect(getAuth()).toMatchObject({ needsKey: true, rejected: true });
  });
});

describe("the socket failing to open", () => {
  it("hands the request to HTTP when ignis is there but takes no socket, asking once for all who waited", async () => {
    const h = harness(200);
    const first = h.start([user("a")], { streamId: "a" });
    const second = h.start([user("b")], { streamId: "b" });
    await tick();
    h.sockets[0].refuse();
    expect(await first.result).toBe("unavailable");
    expect(await second.result).toBe("unavailable");
    expect(h.probes).toHaveLength(1);
  });

  it("treats a key the socket cannot carry as a failed upgrade", async () => {
    const probes: unknown[] = [];
    const socket = createResponsesSocket({
      open: () => {
        throw new DOMException("invalid subprotocol", "SyntaxError");
      },
      url: () => "ws://ignis/v1/responses",
      fetch: (async () => (probes.push(1), new Response("{}", { status: 200 }))) as typeof fetch,
    });
    expect(await socket.stream({ request: request([user("a")]), onEvent: () => {} })).toBe("unavailable");
    expect(probes).toHaveLength(1);
  });

  it("fails the turn and keeps the socket when the probe gets a 5xx, as a dev proxy answers while ignis restarts", async () => {
    const h = harness(502);
    const streamChat = createStreamChat(h.socket);
    const turn = streamChat({ request: request([user("a")]), onEvent: () => {} });
    await tick();
    h.sockets[0].refuse();
    expect(await turn).toMatchObject({ ok: false, message: expect.stringMatching(/^502/) });
    expect(getTransport()).toMatchObject({ choice: "websocket", fellBack: false, notice: false });
  });

  it("fails the turn without leaving the socket when ignis cannot be reached at all", async () => {
    const h = harness("down");
    const { result } = h.start([user("a")]);
    await tick();
    h.sockets[0].refuse();
    expect(await result).toMatchObject({ ok: false, message: expect.stringMatching(/Could not reach ignis/) });
  });
});

describe("streamChat and the transport", () => {
  const chatBody = 'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
  const httpFetch = (sent: string[]) =>
    (async (url: string) => (sent.push(url), new Response(chatBody, { status: 200 }))) as typeof fetch;

  it("re-sends a turn over HTTP when the socket cannot be opened, and stays there with a notice", async () => {
    const h = harness(200);
    const streamChat = createStreamChat(h.socket);
    const sent: string[] = [];
    const turn = streamChat({ request: request([user("a")]), fetch: httpFetch(sent), onEvent: () => {} });
    await tick();
    h.sockets[0].refuse();
    expect(await turn).toMatchObject({ ok: true });
    expect(sent).toEqual(["/v1/chat/completions"]);
    expect(getTransport()).toMatchObject({ choice: "websocket", fellBack: true, notice: true });

    await streamChat({ request: request([user("b")]), fetch: httpFetch(sent), onEvent: () => {} });
    expect(h.sockets).toHaveLength(1);
    expect(sent).toHaveLength(2);
  });

  it("goes over HTTP when the setting says so, and back to the socket when it says WebSocket again", async () => {
    const h = harness();
    const streamChat = createStreamChat(h.socket);
    chooseTransport("http");
    const sent: string[] = [];
    await streamChat({ request: request([user("a")]), fetch: httpFetch(sent), onEvent: () => {} });
    expect(sent).toHaveLength(1);
    expect(h.sockets).toHaveLength(0);

    chooseTransport("websocket");
    void streamChat({ request: request([user("a")]), fetch: httpFetch(sent), onEvent: () => {} });
    await tick();
    expect(h.sockets).toHaveLength(1);
    expect(sent).toHaveLength(1);
  });
});

describe("a reply on the socket", () => {
  it("is the same events and figures as the same reply over HTTP", async () => {
    const h = harness();
    const call = { call_id: "call_0", name: "agent", arguments: '{"prompt":"x"}' };
    const { events, result } = h.start([user("hi")], { streamId: "session-1" });
    await tick();
    h.sockets[0].accept();
    await tick();
    for (const event of reply("session-1", "resp_1", { reasoning: "hm", text: "Hi", calls: [call] })) h.sockets[0].emit(event);
    const socketResult = await result;

    const sse = (payload: object) => `data: ${JSON.stringify(payload)}\n\n`;
    const choice = (delta: object, finish: string | null = null) => sse({ choices: [{ index: 0, delta, finish_reason: finish }] });
    const body =
      choice({ reasoning_content: "hm" }) +
      choice({ content: "Hi" }) +
      choice({ tool_calls: [{ index: 0, id: "call_0", type: "function", function: { name: "agent", arguments: '{"prompt":"x"}' } }] }) +
      choice({}, "tool_calls") +
      sse({ choices: [], usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 } }) +
      "data: [DONE]\n\n";
    const httpEvents: ChunkEvent[] = [];
    const httpResult = await streamChatCompletions({
      body: {},
      fetch: (async () => new Response(body)) as typeof fetch,
      now: ticking(),
      onEvent: (e) => httpEvents.push(e),
    });

    expect(events).toEqual(httpEvents);
    expect(socketResult).toMatchObject({ ok: true });
    const figures = (r: typeof httpResult) => {
      const { ttftMs: _t, decodeTokensPerSec: _d, durationMs: _m, ...rest } = computeFigures(r.timeline);
      return rest;
    };
    expect(socketResult !== "unavailable" && figures(socketResult)).toEqual(figures(httpResult));
  });

  it("sends the page's request as a response.create on the request's stream", async () => {
    const h = harness();
    h.start([user("hi")], { streamId: "session 1/x" });
    await tick();
    h.sockets[0].accept();
    await tick();
    expect(h.sockets[0].creates()).toEqual([
      {
        type: "response.create",
        stream_id: "session-1-x",
        model: "m",
        input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] }],
        temperature: 1,
        top_p: 0.95,
        reasoning_effort: "xhigh",
        class: "interactive",
      },
    ]);
  });

  it("names a stream within ignis's pattern", () => {
    expect(streamIdOf("session-1.agent-call_0")).toBe("session-1.agent-call_0");
    expect(streamIdOf("a b/c")).toBe("a-b-c");
    expect(streamIdOf("x".repeat(300))).toHaveLength(256);
  });

  it("fails visibly on an error event, with the status the same failure has over HTTP", async () => {
    const h = harness();
    const { result } = h.start([user("hi")], { streamId: "s" });
    await tick();
    h.sockets[0].accept();
    await tick();
    h.sockets[0].emit({
      type: "error",
      stream_id: "s",
      status: 400,
      error: { type: "invalid_request_error", code: "context_length_exceeded", message: "the prompt is too long", param: null },
    });
    expect(await result).toMatchObject({ ok: false, message: "400: the prompt is too long" });
  });

  it("fails visibly when the engine fails the response", async () => {
    const h = harness();
    const { result } = h.start([user("hi")], { streamId: "s" });
    await tick();
    h.sockets[0].accept();
    await tick();
    h.sockets[0].emit({ type: "response.created", stream_id: "s", response: { id: "r", status: "in_progress" } });
    h.sockets[0].emit({
      type: "response.failed",
      stream_id: "s",
      response: { id: "r", status: "failed", error: { code: "request_timeout", message: "the request timed out" } },
    });
    expect(await result).toMatchObject({ ok: false, message: "request_timeout: the request timed out" });
  });
});

describe("queued replies", () => {
  it("fails a queued reply whose admission ignis then refused", async () => {
    const h = harness();
    const { result } = h.start([user("hi")], { streamId: "s" });
    await tick();
    h.sockets[0].accept();
    await tick();
    const socket = h.sockets[0];
    socket.emit({ type: "response.created", stream_id: "s", response: { id: "resp_q0", status: "queued" } });
    socket.emit({ type: "response.queued", stream_id: "s", response: { id: "resp_q0", status: "queued" } });
    socket.emit({
      type: "response.failed",
      stream_id: "s",
      response: { id: "resp_q0", status: "failed", error: { code: "context_length_exceeded", message: "the prompt does not fit" } },
    });
    expect(await result).toMatchObject({ ok: false, message: "context_length_exceeded: the prompt does not fit" });
  });

  it("keeps the id response.created gave a queued reply for its whole life: the cancel and the continuation use it", async () => {
    const h = harness();
    const first = h.start([user("a")], { streamId: "s" });
    await tick();
    h.sockets[0].accept();
    await tick();
    const socket = h.sockets[0];
    socket.emit({ type: "response.created", stream_id: "s", response: { id: "resp_q0", status: "queued" } });
    socket.emit({ type: "response.queued", stream_id: "s", response: { id: "resp_q0", status: "queued" } });
    socket.emit({ type: "response.in_progress", stream_id: "s", response: { id: "resp_12", status: "in_progress" } });
    socket.emit({ type: "response.output_text.delta", stream_id: "s", delta: "b" });
    socket.emit({
      type: "response.completed",
      stream_id: "s",
      response: { id: "resp_12", status: "completed", output: [{ type: "message", content: [{ type: "output_text", text: "b" }] }], usage },
    });
    await first.result;

    const controller = new AbortController();
    h.start([user("a"), { role: "assistant", content: "b" }, user("c")], { streamId: "s", signal: controller.signal });
    await tick();
    expect(socket.creates().at(-1)).toMatchObject({ previous_response_id: "resp_q0" });
    socket.emit({ type: "response.created", stream_id: "s", response: { id: "resp_q1", status: "queued" } });
    socket.emit({ type: "response.in_progress", stream_id: "s", response: { id: "resp_13", status: "in_progress" } });
    controller.abort();
    expect(socket.sent.at(-1)).toEqual({ type: "response.cancel", response_id: "resp_q1" });
  });

  it("says the reply is queued, and counts the queue as queue time rather than TTFT", async () => {
    const h = harness();
    const seen: string[] = [];
    const { result } = h.start([user("hi")], { streamId: "s", onQueued: () => seen.push("queued"), onStart: () => seen.push("start") });
    await tick();
    h.sockets[0].accept();
    await tick();
    const socket = h.sockets[0];
    const queued = { id: "r", status: "queued" };
    socket.emit({ type: "response.created", stream_id: "s", response: queued });
    socket.emit({ type: "response.queued", stream_id: "s", response: queued });
    expect(seen).toEqual(["queued"]);
    socket.emit({ type: "response.in_progress", stream_id: "s", response: { id: "r", status: "in_progress" } });
    expect(seen).toEqual(["queued", "start"]);
    socket.emit({ type: "response.output_text.delta", stream_id: "s", delta: "Hi" });
    socket.emit({ type: "response.completed", stream_id: "s", response: { id: "r", status: "completed", output: [], usage } });
    const done = await result;
    if (done === "unavailable") throw new Error("no socket");
    const t = done.timeline;
    // Ticks: sent, created, queued, in progress, first token.
    expect(t.admittedAt! - t.queuedAt!).toBe(10);
    const figures = computeFigures(t);
    expect(figures.queueMs).toBe(10);
    expect(figures.ttftMs).toBe(t.firstTokenAt! - t.sentAt - 10);
  });
});

describe("streams", () => {
  it("runs twelve streams at once on one socket, each getting its own events", async () => {
    const h = harness();
    const names = Array.from({ length: 12 }, (_, i) => (i < 4 ? `session-${i}` : `session-0.agent-call_${i}`));
    const runs = names.map((name) => h.start([user(name)], { streamId: name }));
    await tick();
    h.sockets[0].accept();
    await tick();
    expect(h.sockets).toHaveLength(1);
    expect(h.sockets[0].creates().map((c) => c.stream_id)).toEqual(names);
    // Interleaved, as a multiplexed socket delivers them.
    const streams = names.map((name, i) => reply(name, `resp_${i}`, { text: `answer ${i}` }));
    for (let step = 0; step < streams[0].length; step++) for (const events of streams) h.sockets[0].emit(events[step]);
    const results = await Promise.all(runs.map((r) => r.result));
    expect(results.every((r) => r !== "unavailable" && r.ok)).toBe(true);
    expect(runs.map((r) => r.events.find((e) => e.kind === "content"))).toEqual(names.map((_, i) => ({ kind: "content", text: `answer ${i}` })));
  });

  it(`opens another socket past ${STREAMS_PER_SOCKET} stream ids, and keeps a stream on the socket it started on`, async () => {
    const h = harness();
    const first = Array.from({ length: STREAMS_PER_SOCKET }, (_, i) => h.start([user("x")], { streamId: `s${i}` }));
    await tick();
    h.sockets[0].accept();
    await tick();
    const extra = h.start([user("x")], { streamId: "one-more" });
    await tick();
    expect(h.sockets).toHaveLength(2);
    h.sockets[1].accept();
    await tick();
    expect(h.sockets[1].creates().map((c) => c.stream_id)).toEqual(["one-more"]);

    for (const event of reply("s0", "resp_s0", { text: "a" })) h.sockets[0].emit(event);
    await first[0].result;
    h.start([user("x"), { role: "assistant", content: "a" }, user("y")], { streamId: "s0" });
    await tick();
    expect(h.sockets[0].creates().at(-1)).toMatchObject({ stream_id: "s0", previous_response_id: "resp_s0" });
    for (const event of reply("one-more", "resp_x", { text: "b" })) h.sockets[1].emit(event);
    await extra.result;
  });
});

describe("continuation", () => {
  const history = [user("a"), { role: "assistant" as const, content: "b", reasoning: "hm" }];

  /** A harness with its socket open and `session` having answered "a" with "b". */
  async function answered() {
    const h = harness();
    const first = h.start([user("a")], { streamId: "session" });
    await tick();
    h.sockets[0].accept();
    await tick();
    for (const event of reply("session", "resp_1", { reasoning: "hm", text: "b" })) h.sockets[0].emit(event);
    await first.result;
    return h;
  }

  it("sends only the new items after the previous response on the same socket", async () => {
    const h = await answered();
    h.start([...history, user("c")], { streamId: "session" });
    await tick();
    expect(h.sockets[0].creates().at(-1)).toMatchObject({
      previous_response_id: "resp_1",
      input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "c" }] }],
    });
  });

  it("continues from the output as ignis holds it: text after a call is not the page's one message, so it all goes", async () => {
    const h = harness();
    const first = h.start([user("a")], { streamId: "s" });
    await tick();
    h.sockets[0].accept();
    await tick();
    const socket = h.sockets[0];
    const call = { type: "function_call", call_id: "c", name: "read_file", arguments: "{}" };
    const output = [{ type: "message", content: [{ type: "output_text", text: "x" }] }, call, { type: "message", content: [{ type: "output_text", text: "y" }] }];
    for (const event of [
      { type: "response.created", response: { id: "r1", status: "in_progress" } },
      { type: "response.in_progress", response: { id: "r1", status: "in_progress" } },
      { type: "response.output_text.delta", delta: "x" },
      { type: "response.output_item.done", item: call },
      { type: "response.output_text.delta", delta: "y" },
      { type: "response.completed", response: { id: "r1", status: "completed", output, usage } },
    ])
      socket.emit({ ...event, stream_id: "s" });
    await first.result;
    const toolCalls = [{ id: "c", name: "read_file", arguments: "{}" }];
    h.start([user("a"), { role: "assistant", content: "xy", toolCalls }, { role: "tool", content: "file", toolCallId: "c" }], { streamId: "s" });
    await tick();
    expect(socket.creates().at(-1)).not.toHaveProperty("previous_response_id");
    expect(socket.creates().at(-1)?.input).toHaveLength(4);
  });

  it("sends the whole history after an edit, a regenerate and a fork", async () => {
    const edited = await answered();
    edited.start([user("a"), { role: "assistant", content: "b, edited", reasoning: "hm" }, user("c")], { streamId: "session" });
    await tick();
    expect(edited.sockets[0].creates().at(-1)).not.toHaveProperty("previous_response_id");
    expect(edited.sockets[0].creates().at(-1)?.input).toHaveLength(4);

    const regenerated = await answered();
    regenerated.start([user("a")], { streamId: "session" });
    await tick();
    expect(regenerated.sockets[0].creates().at(-1)).not.toHaveProperty("previous_response_id");

    const forked = await answered();
    forked.start([...history, user("c")], { streamId: "session-2" });
    await tick();
    expect(forked.sockets[0].creates().at(-1)).not.toHaveProperty("previous_response_id");
  });

  it("retries once in full when ignis no longer has the previous response", async () => {
    const h = await answered();
    const { result } = h.start([...history, user("c")], { streamId: "session" });
    await tick();
    h.sockets[0].emit({
      type: "error",
      stream_id: "session",
      status: 400,
      error: { type: "invalid_request_error", code: "previous_response_not_found", message: "Previous response not found", param: "previous_response_id" },
    });
    await tick();
    const retry = h.sockets[0].creates().at(-1);
    expect(retry).not.toHaveProperty("previous_response_id");
    expect(retry?.input).toHaveLength(4);
    for (const event of reply("session", "resp_2", { text: "d" })) h.sockets[0].emit(event);
    expect(await result).toMatchObject({ ok: true });
    expect(h.sockets[0].creates()).toHaveLength(3);
  });

  it("sends no retry for a request stopped while ignis answered previous_response_not_found", async () => {
    const h = await answered();
    const controller = new AbortController();
    const { result } = h.start([...history, user("c")], { streamId: "session", signal: controller.signal });
    await tick();
    h.sockets[0].emit({
      type: "error",
      stream_id: "session",
      status: 400,
      error: { type: "invalid_request_error", code: "previous_response_not_found", message: "Previous response not found", param: "previous_response_id" },
    });
    controller.abort();
    expect(await result).toMatchObject({ ok: true, timeline: { stopped: true } });
    await tick();
    expect(h.sockets[0].creates()).toHaveLength(2);
  });

  it("sends the whole history after the socket reconnects, or after a reply that did not complete", async () => {
    const h = await answered();
    h.sockets[0].drop();
    h.start([...history, user("c")], { streamId: "session" });
    await tick();
    h.sockets[1].accept();
    await tick();
    expect(h.sockets[1].creates()[0]).not.toHaveProperty("previous_response_id");

    const stopped = await answered();
    const controller = new AbortController();
    const cut = stopped.start([...history, user("c")], { streamId: "session", signal: controller.signal });
    await tick();
    controller.abort();
    await cut.result;
    stopped.start([...history, user("c"), { role: "assistant", content: "half" }, user("d")], { streamId: "session" });
    await tick();
    stopped.sockets[0].emit({ type: "response.created", stream_id: "session", response: { id: "resp_2", status: "in_progress" } });
    stopped.sockets[0].emit({ type: "response.incomplete", stream_id: "session", response: { id: "resp_2", status: "cancelled" } });
    await tick();
    expect(stopped.sockets[0].creates().at(-1)).not.toHaveProperty("previous_response_id");
  });
});

describe("stop", () => {
  it("cancels the reply by id, ends it as stopped at once, and leaves the other streams running", async () => {
    const h = harness();
    const controller = new AbortController();
    const stopped = h.start([user("a")], { streamId: "a", signal: controller.signal });
    const other = h.start([user("b")], { streamId: "b" });
    await tick();
    h.sockets[0].accept();
    await tick();
    const socket = h.sockets[0];
    socket.emit({ type: "response.created", stream_id: "a", response: { id: "resp_a", status: "in_progress" } });
    socket.emit({ type: "response.output_text.delta", stream_id: "a", delta: "Hel" });
    controller.abort();
    const result = await stopped.result;
    expect(result).toMatchObject({ ok: true, timeline: { stopped: true } });
    expect(socket.sent.at(-1)).toEqual({ type: "response.cancel", response_id: "resp_a" });

    // What still arrives for the stopped reply is not shown.
    socket.emit({ type: "response.output_text.delta", stream_id: "a", delta: "lo" });
    socket.emit({ type: "response.incomplete", stream_id: "a", response: { id: "resp_a", status: "cancelled" } });
    expect(stopped.events).toEqual([{ kind: "content", text: "Hel" }]);
    for (const event of reply("b", "resp_b", { text: "still here" })) socket.emit(event);
    expect(await other.result).toMatchObject({ ok: true });
    expect(socket.closedByPage).toBe(false);
  });

  it("stops a reply before ignis has named it, and cancels it once it does", async () => {
    const h = harness();
    const controller = new AbortController();
    const { events, result } = h.start([user("a")], { streamId: "a", signal: controller.signal });
    await tick();
    h.sockets[0].accept();
    await tick();
    controller.abort();
    expect(await result).toMatchObject({ ok: true, timeline: { stopped: true } });
    const socket = h.sockets[0];
    expect(socket.sent.some((e) => e.type === "response.cancel")).toBe(false);
    socket.emit({ type: "response.created", stream_id: "a", response: { id: "resp_a", status: "queued" } });
    expect(socket.sent.at(-1)).toEqual({ type: "response.cancel", response_id: "resp_a" });
    socket.emit({ type: "response.incomplete", stream_id: "a", response: { id: "resp_a", status: "cancelled" } });
    expect(events).toEqual([]);

    // The stream is free again: the next turn goes out.
    h.start([user("b")], { streamId: "a" });
    await tick();
    expect(socket.creates()).toHaveLength(2);
  });

  it("stops a request still waiting for the socket to open, and never sends it", async () => {
    const h = harness();
    const controller = new AbortController();
    const { result } = h.start([user("a")], { streamId: "a", signal: controller.signal });
    await tick();
    controller.abort();
    expect(await result).toMatchObject({ ok: true, timeline: { stopped: true } });
    h.sockets[0].accept();
    await tick();
    expect(h.sockets[0].creates()).toEqual([]);
  });

  it("stops a request still waiting behind its stream's previous one, and never sends it", async () => {
    const h = harness();
    h.start([user("a")], { streamId: "a" });
    await tick();
    h.sockets[0].accept();
    await tick();
    const controller = new AbortController();
    const waiting = h.start([user("b")], { streamId: "a", signal: controller.signal });
    await tick();
    controller.abort();
    expect(await waiting.result).toMatchObject({ ok: true, timeline: { stopped: true } });
    for (const event of reply("a", "resp_a", { text: "done" })) h.sockets[0].emit(event);
    await tick();
    expect(h.sockets[0].creates()).toHaveLength(1);
  });

  it("holds a stream's next request until ignis has ended the stopped one", async () => {
    const h = harness();
    const controller = new AbortController();
    h.start([user("a")], { streamId: "a", signal: controller.signal });
    await tick();
    h.sockets[0].accept();
    await tick();
    controller.abort();
    h.start([user("b")], { streamId: "a" });
    await tick();
    expect(h.sockets[0].creates()).toHaveLength(1);
    h.sockets[0].emit({ type: "response.created", stream_id: "a", response: { id: "resp_a", status: "in_progress" } });
    h.sockets[0].emit({ type: "response.incomplete", stream_id: "a", response: { id: "resp_a", status: "cancelled" } });
    await tick();
    expect(h.sockets[0].creates()).toHaveLength(2);
  });
});

describe("a dropped socket", () => {
  it("fails the replies it carried, and the next request opens a new one", async () => {
    const h = harness();
    const a = h.start([user("a")], { streamId: "a" });
    const b = h.start([user("b")], { streamId: "b" });
    await tick();
    h.sockets[0].accept();
    await tick();
    h.sockets[0].emit({ type: "response.created", stream_id: "a", response: { id: "resp_a", status: "in_progress" } });
    h.sockets[0].drop();
    expect(await a.result).toMatchObject({ ok: false, message: expect.stringMatching(/connection to ignis closed/) });
    expect(await b.result).toMatchObject({ ok: false });

    const next = h.start([user("c")], { streamId: "a" });
    await tick();
    expect(h.sockets).toHaveLength(2);
    h.sockets[1].accept();
    await tick();
    for (const event of reply("a", "resp_c", { text: "back" })) h.sockets[1].emit(event);
    expect(await next.result).toMatchObject({ ok: true });
  });
});
