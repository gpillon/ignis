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

  it("leaves max_tokens out when unset, so the engine's own cap applies", () => {
    const body = buildChatRequest({ ...settings, maxTokens: null }, [{ role: "user", content: "hi" }]);
    expect("max_tokens" in body).toBe(false);
  });
});
