import { describe, expect, it } from "vitest";
import type { Settings, Turn } from "./request.ts";
import { buildResponseCreate, outputItems, parseResponseEvent, responseItems, type ServerEvent } from "./responses.ts";
import { createSseParser, parseChunk } from "./sse.ts";

const settings: Settings = {
  model: "qwen3.8-27b",
  systemPrompt: "Be terse.",
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 512,
  reasoningEffort: "xhigh",
  thinkingBudget: 2048,
  laneTag: "agent",
};

describe("buildResponseCreate", () => {
  it("carries the system prompt as instructions and the settings under the chat request's names", () => {
    const tool = { type: "function" as const, function: { name: "agent", description: "Run a sub-task.", parameters: { type: "object" } } };
    const body = buildResponseCreate(
      { settings, turns: [{ role: "user", content: "hi" }], extras: { ignisPrompt: "# Tools", tools: [tool] } },
      "session-1",
      responseItems([{ role: "user", content: "hi" }]),
    );
    expect(body).toEqual({
      type: "response.create",
      stream_id: "session-1",
      model: "qwen3.8-27b",
      instructions: "# Tools\n\nBe terse.",
      input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] }],
      tools: [{ type: "function", name: "agent", description: "Run a sub-task.", parameters: { type: "object" } }],
      temperature: 0.7,
      top_p: 0.9,
      max_output_tokens: 512,
      reasoning_effort: "xhigh",
      thinking_budget: 2048,
      class: "agent",
    });
  });

  it("leaves out what the chat request leaves out: a blank prompt, no tools, the engine's cap, the server's budget", () => {
    const body = buildResponseCreate(
      { settings: { ...settings, systemPrompt: " ", maxTokens: null, reasoningEffort: "none" }, turns: [] },
      "s",
      [],
      "resp_7",
    );
    expect(body).toEqual({
      type: "response.create",
      stream_id: "s",
      model: "qwen3.8-27b",
      input: [],
      previous_response_id: "resp_7",
      temperature: 0.7,
      top_p: 0.9,
      reasoning_effort: "none",
      class: "agent",
    });
  });
});

describe("responseItems", () => {
  it("writes each turn as the items the Responses API reads, reasoning included", () => {
    const image = { name: "a.jpg", url: "data:image/jpeg;base64,AA", width: 8, height: 6 };
    const turns: Turn[] = [
      { role: "user", content: "look", images: [image], dateTime: "It is now 01:52." },
      { role: "assistant", content: "Let me check.", reasoning: "hm", toolCalls: [{ id: "call_0", name: "agent", arguments: '{"prompt":"x"}' }] },
      { role: "tool", content: "a cat", toolCallId: "call_0" },
      { role: "assistant", content: "A cat." },
      { role: "user", content: "", images: [image] },
    ];
    expect(responseItems(turns)).toEqual([
      { type: "message", role: "developer", content: [{ type: "input_text", text: "It is now 01:52." }] },
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_image", image_url: "data:image/jpeg;base64,AA" },
          { type: "input_text", text: "look" },
        ],
      },
      { type: "reasoning", content: [{ type: "reasoning_text", text: "hm" }], summary: [] },
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "Let me check." }] },
      { type: "function_call", call_id: "call_0", name: "agent", arguments: '{"prompt":"x"}' },
      { type: "function_call_output", call_id: "call_0", output: "a cat" },
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "A cat." }] },
      { type: "message", role: "user", content: [{ type: "input_image", image_url: "data:image/jpeg;base64,AA" }] },
    ]);
  });

  it("takes a reply's output as the items its turn becomes in the next request, when it has one message", () => {
    const output = [
      { type: "reasoning", id: "rs_0", summary: [], content: [{ type: "reasoning_text", text: "hm" }] },
      { type: "message", id: "msg_0", role: "assistant", status: "completed", content: [{ type: "output_text", text: "Let me look.", annotations: [] }] },
      { type: "function_call", id: "fc_0", call_id: "c", name: "web_search", arguments: "{}", status: "completed" },
    ];
    const turn = { role: "assistant" as const, content: "Let me look.", reasoning: "hm", toolCalls: [{ id: "c", name: "web_search", arguments: "{}" }] };
    expect(outputItems({ id: "r", output })).toEqual(responseItems([turn]));
  });

  it("keeps the output's own order when text follows a call, as ignis holds it", () => {
    const output = [
      { type: "message", content: [{ type: "output_text", text: "Reading it. " }] },
      { type: "function_call", call_id: "c", name: "read_file", arguments: "{}" },
      { type: "message", content: [{ type: "output_text", text: "Then this." }] },
    ];
    expect(outputItems({ id: "r", output }).map((item) => item.type)).toEqual(["message", "function_call", "message"]);
  });
});

/** Every chunk event of a chat SSE body. */
const chatEvents = (body: string) => createSseParser().push(body).flatMap(parseChunk);

const sse = (delta: object, finish: string | null = null, extra: object = {}) =>
  `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish, ...extra }] })}\n\n`;

const usageChunk = `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 9, completion_tokens: 4, total_tokens: 13 } })}\n\n`;

const usage = { input_tokens: 9, output_tokens: 4, total_tokens: 13, input_tokens_details: { cached_tokens: 0 } };

describe("parseResponseEvent", () => {
  it("yields what the equivalent chat stream yields, for thinking, text and a tool call", () => {
    const call = { type: "function_call", id: "fc_1", call_id: "call_0", name: "agent", arguments: '{"prompt":"x"}', status: "completed" };
    const events: ServerEvent[] = [
      { type: "response.created", response: { id: "resp_1", status: "in_progress" } },
      { type: "response.in_progress", response: { id: "resp_1", status: "in_progress" } },
      { type: "response.output_item.added", item: { type: "reasoning" } },
      { type: "response.reasoning_text.delta", delta: "hm" },
      { type: "response.output_item.done", item: { type: "reasoning" } },
      { type: "response.output_item.added", item: { type: "message" } },
      { type: "response.output_text.delta", delta: "Hi" },
      { type: "response.output_text.delta", delta: " there" },
      { type: "response.output_item.done", item: { type: "message" } },
      { type: "response.output_item.added", item: { ...call, arguments: "", status: "in_progress" } },
      { type: "response.function_call_arguments.delta", delta: '{"prompt":"x"}' },
      { type: "response.function_call_arguments.done", item: call },
      { type: "response.output_item.done", item: call },
      {
        type: "response.completed",
        response: { id: "resp_1", status: "completed", output: [{ type: "reasoning" }, { type: "message" }, call], usage },
      },
    ];
    const chat =
      sse({ reasoning_content: "hm" }) +
      sse({ content: "Hi" }) +
      sse({ content: " there" }) +
      sse({ tool_calls: [{ index: 0, id: "call_0", type: "function", function: { name: "agent", arguments: '{"prompt":"x"}' } }] }) +
      sse({}, "tool_calls") +
      usageChunk +
      "data: [DONE]\n\n";
    expect(events.flatMap(parseResponseEvent)).toEqual(chatEvents(chat));
  });

  it("joins the text of several message items into the one reply the chat stream would give", () => {
    const call = { type: "function_call", call_id: "c", name: "read_file", arguments: "{}" };
    const events: ServerEvent[] = [
      { type: "response.output_text.delta", delta: "Reading it. " },
      { type: "response.output_item.done", item: { type: "message" } },
      { type: "response.output_item.done", item: call },
      { type: "response.output_item.added", item: { type: "message" } },
      { type: "response.output_text.delta", delta: "Then this." },
      { type: "response.completed", response: { id: "r", status: "completed", output: [{ type: "message" }, call, { type: "message" }], usage } },
    ];
    const content = events.flatMap(parseResponseEvent).flatMap((e) => (e.kind === "content" ? [e.text] : []));
    expect(content.join("")).toBe("Reading it. Then this.");
    expect(events.flatMap(parseResponseEvent).find((e) => e.kind === "finish")).toEqual({ kind: "finish", reason: "tool_calls" });
  });

  it("ends a completed reply with stop, and a reply cut at max_output_tokens with length", () => {
    const done = (type: string, extra: object) => [
      { type: "response.output_text.delta", delta: "Hi" },
      { type, response: { id: "r", output: [{ type: "message" }], usage, ...extra } },
    ];
    expect(done("response.completed", { status: "completed" }).flatMap(parseResponseEvent)).toEqual(
      chatEvents(sse({ content: "Hi" }) + sse({}, "stop") + usageChunk + "data: [DONE]\n\n"),
    );
    expect(
      done("response.incomplete", { status: "incomplete", incomplete_details: { reason: "max_output_tokens" } }).flatMap(parseResponseEvent),
    ).toEqual(chatEvents(sse({ content: "Hi" }) + sse({}, "length") + usageChunk + "data: [DONE]\n\n"));
  });

  it("carries where the thinking budget closed the reasoning, as the chat stream does", () => {
    const events = parseResponseEvent({
      type: "response.completed",
      response: { id: "r", status: "completed", output: [], usage, thinking_budget_forced_at: 2048 },
    });
    expect(events[0]).toEqual({ kind: "finish", reason: "stop", thinkingForcedAt: 2048 });
  });

  it("gives a cancelled response no finish: the page stopped it", () => {
    const cancelled = { type: "response.incomplete", response: { id: "r", status: "cancelled", usage } };
    expect(parseResponseEvent(cancelled)).toEqual([]);
  });
});
