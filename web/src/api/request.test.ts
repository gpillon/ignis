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

  it("leaves max_tokens out when unset, so the engine's own cap applies", () => {
    const body = buildChatRequest({ ...settings, maxTokens: null }, [{ role: "user", content: "hi" }]);
    expect("max_tokens" in body).toBe(false);
  });
});
