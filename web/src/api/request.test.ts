import { describe, expect, it } from "vitest";
import { buildChatRequest, conversationTurns, type Settings } from "./request.ts";

describe("conversationTurns", () => {
  const user = (content: string) => ({ role: "user" as const, content, failed: false });
  const reply = (content: string, failed = false) => ({ role: "assistant" as const, content, failed });

  it("resends a clean conversation as it is", () => {
    expect(conversationTurns([user("a"), reply("b"), user("c"), reply("d")])).toEqual([
      { role: "user", content: "a" },
      { role: "assistant", content: "b" },
      { role: "user", content: "c" },
      { role: "assistant", content: "d" },
    ]);
  });

  it("drops a failed exchange, prompt included, so no two user turns follow each other", () => {
    expect(conversationTurns([user("a"), reply("", true), user("b"), reply("c")])).toEqual([
      { role: "user", content: "b" },
      { role: "assistant", content: "c" },
    ]);
  });

  it("carries a prompt's images back with it, so a later request repeats the same bytes", () => {
    const images = [{ name: "a.jpg", url: "data:image/jpeg;base64,AA", width: 8, height: 6 }];
    expect(conversationTurns([{ ...user("look"), images }, reply("a cat")])).toEqual([
      { role: "user", content: "look", images },
      { role: "assistant", content: "a cat" },
    ]);
  });

  it("keeps a reply that only called tools, and the tool results after it", () => {
    const call = { id: "call_0", name: "agent", arguments: "{}" };
    expect(
      conversationTurns([
        user("a"),
        { role: "assistant", content: "", failed: false, toolCalls: [call] },
        { role: "tool", content: "result", failed: false, toolCallId: "call_0" },
        reply("done"),
      ]),
    ).toEqual([
      { role: "user", content: "a" },
      { role: "assistant", content: "", toolCalls: [call] },
      { role: "tool", content: "result", toolCallId: "call_0" },
      { role: "assistant", content: "done" },
    ]);
  });

  it("keeps the moment each turn was sent with, rather than the moment of this request", () => {
    expect(conversationTurns([{ ...user("a"), dateTime: "It is now 01:52." }, reply("b")])).toEqual([
      { role: "user", content: "a", dateTime: "It is now 01:52." },
      { role: "assistant", content: "b" },
    ]);
  });

  it("drops a reply stopped before its first token, and keeps one stopped part-way", () => {
    expect(conversationTurns([user("a"), reply(""), user("b"), reply("half")])).toEqual([
      { role: "user", content: "b" },
      { role: "assistant", content: "half" },
    ]);
  });
});

const settings: Settings = {
  model: "qwen3.8-27b",
  systemPrompt: "Be terse.",
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 512,
  reasoningEffort: "xhigh",
  thinkingBudget: null,
  laneTag: "agent",
};

describe("buildChatRequest", () => {
  it("streams with usage and carries the sampling, reasoning effort and lane tag fields", () => {
    const body = buildChatRequest(settings, [{ role: "user", content: "hi" }]);
    expect(body).toEqual({
      model: "qwen3.8-27b",
      messages: [
        { role: "system", content: "Be terse." },
        { role: "user", content: "hi" },
      ],
      stream: true,
      stream_options: { include_usage: true },
      temperature: 0.7,
      top_p: 0.9,
      max_tokens: 512,
      reasoning_effort: "xhigh",
      class: "agent",
    });
  });

  it("turns thinking off with the `none` effort alone, never a conflicting enable_thinking", () => {
    const body = buildChatRequest({ ...settings, reasoningEffort: "none" }, [{ role: "user", content: "hi" }]);
    expect(body.reasoning_effort).toBe("none");
    expect("enable_thinking" in body).toBe(false);
  });

  it("sends the whole conversation, and no system message when the prompt is blank", () => {
    const body = buildChatRequest({ ...settings, systemPrompt: "  " }, [
      { role: "user", content: "a" },
      { role: "assistant", content: "b" },
      { role: "user", content: "c" },
    ]);
    expect(body.messages).toEqual([
      { role: "user", content: "a" },
      { role: "assistant", content: "b" },
      { role: "user", content: "c" },
    ]);
  });

  it("puts the ignis prompt ahead of the owner's and declares the tools", () => {
    const tool = { type: "function" as const, function: { name: "agent", description: "d", parameters: {} } };
    const body = buildChatRequest(settings, [{ role: "user", content: "hi" }], { ignisPrompt: "Use agents.", tools: [tool] });
    expect(body.messages[0]).toEqual({ role: "system", content: "Use agents.\n\nBe terse." });
    expect(body.tools).toEqual([tool]);
    expect("tools" in buildChatRequest(settings, [{ role: "user", content: "hi" }], { tools: [] })).toBe(false);
  });

  it("sends tool calls and tool results in the OpenAI wire shape", () => {
    const call = { id: "call_0", name: "agent", arguments: '{"prompt":"x"}' };
    const body = buildChatRequest({ ...settings, systemPrompt: "" }, [
      { role: "user", content: "a" },
      { role: "assistant", content: "", toolCalls: [call] },
      { role: "tool", content: "done", toolCallId: "call_0" },
    ]);
    expect(body.messages).toEqual([
      { role: "user", content: "a" },
      { role: "assistant", content: "", tool_calls: [{ id: "call_0", type: "function", function: { name: "agent", arguments: '{"prompt":"x"}' } }] },
      { role: "tool", content: "done", tool_call_id: "call_0" },
    ]);
  });

  it("puts a turn's moment in a developer message ahead of it, and leaves the other turns alone", () => {
    const body = buildChatRequest({ ...settings, systemPrompt: "" }, [
      { role: "user", content: "a", dateTime: "It is now 01:52." },
      { role: "assistant", content: "b" },
      { role: "user", content: "c", dateTime: "It is now 02:04." },
    ]);
    expect(body.messages).toEqual([
      { role: "developer", content: "It is now 01:52." },
      { role: "user", content: "a" },
      { role: "assistant", content: "b" },
      { role: "developer", content: "It is now 02:04." },
      { role: "user", content: "c" },
    ]);
  });

  /**
   * The property the date and time tool's live option rests on: a turn's
   * moment never moves a token of what went before it, so a request finds the
   * last one whole at its head and the engine prefills only the new tail.
   * Rewriting the earlier moments to the present instead would diverge at the
   * first of them and cost a full prefill — 25 s of it at 98K tokens.
   */
  it("grows a conversation by appending, leaving one request a prefix of the next", () => {
    const turns = [
      { role: "user" as const, content: "a", dateTime: "It is now 01:52." },
      { role: "assistant" as const, content: "b" },
      { role: "user" as const, content: "c", dateTime: "It is now 02:04." },
      { role: "assistant" as const, content: "d" },
    ];
    const before = buildChatRequest(settings, turns.slice(0, 2)).messages;
    const after = buildChatRequest(settings, turns).messages;
    expect(after.slice(0, before.length)).toEqual(before);
  });

  it("sends a turn without images as a plain string, so it tokenizes as it always did", () => {
    const body = buildChatRequest(settings, [{ role: "user", content: "hi" }]);
    expect(body.messages.at(-1)).toEqual({ role: "user", content: "hi" });
  });

  it("sends a turn's images as content parts, the images ahead of the text", () => {
    const image = (url: string) => ({ name: "a.jpg", url, width: 4, height: 4 });
    const body = buildChatRequest(settings, [
      { role: "user", content: "what is this?", images: [image("data:image/jpeg;base64,AA"), image("data:image/jpeg;base64,BB")] },
    ]);
    expect(body.messages.at(-1)).toEqual({
      role: "user",
      content: [
        { type: "image_url", image_url: { url: "data:image/jpeg;base64,AA" } },
        { type: "image_url", image_url: { url: "data:image/jpeg;base64,BB" } },
        { type: "text", text: "what is this?" },
      ],
    });
  });

  it("sends an image with no words of its own as the image part alone", () => {
    const body = buildChatRequest(settings, [
      { role: "user", content: "", images: [{ name: "a.jpg", url: "data:image/jpeg;base64,AA", width: 4, height: 4 }] },
    ]);
    expect(body.messages.at(-1)).toEqual({
      role: "user",
      content: [{ type: "image_url", image_url: { url: "data:image/jpeg;base64,AA" } }],
    });
  });

  it("leaves max_tokens out when unset, so the engine's own cap applies", () => {
    const body = buildChatRequest({ ...settings, maxTokens: null }, [{ role: "user", content: "hi" }]);
    expect("max_tokens" in body).toBe(false);
  });
});

describe("buildChatRequest and the thinking budget", () => {
  const hi = [{ role: "user" as const, content: "hi" }];

  it("leaves thinking_budget out on the server default, so the operator's --thinking-budget applies", () => {
    expect("thinking_budget" in buildChatRequest({ ...settings, thinkingBudget: null }, hi)).toBe(false);
  });

  it("sends 0 for no budget, and a token count as that count", () => {
    expect(buildChatRequest({ ...settings, thinkingBudget: 0 }, hi).thinking_budget).toBe(0);
    expect(buildChatRequest({ ...settings, thinkingBudget: 4096 }, hi).thinking_budget).toBe(4096);
  });

  it("sends no budget with thinking off, whatever the setting holds", () => {
    const body = buildChatRequest({ ...settings, reasoningEffort: "none", thinkingBudget: 4096 }, hi);
    expect(body.reasoning_effort).toBe("none");
    expect("thinking_budget" in body).toBe(false);
  });

  it("sends the max effort without any thinking_budget, a number or 0, since max ignores both", () => {
    for (const thinkingBudget of [null, 0, 2048]) {
      const body = buildChatRequest({ ...settings, reasoningEffort: "max", thinkingBudget }, hi);
      expect(body.reasoning_effort).toBe("max");
      expect("thinking_budget" in body).toBe(false);
    }
  });
});
