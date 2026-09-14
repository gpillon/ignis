import { describe, expect, it } from "vitest";
import { createSseParser, parseChunk } from "./sse.ts";

const chunk = (delta: object, finish: string | null = null) =>
  JSON.stringify({ id: "c", object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason: finish }] });

describe("createSseParser", () => {
  it("returns the data of each complete event", () => {
    const parser = createSseParser();
    expect(parser.push("data: one\n\ndata: two\n\n")).toEqual(["one", "two"]);
  });

  it("holds a partial event until its blank line arrives, across any split", () => {
    const parser = createSseParser();
    expect(parser.push("da")).toEqual([]);
    expect(parser.push("ta: {\"a\":")).toEqual([]);
    expect(parser.push("1}\n")).toEqual([]);
    expect(parser.push("\ndata: [DONE]\n\n")).toEqual(["{\"a\":1}", "[DONE]"]);
  });

  it("accepts CRLF line endings and ignores comments and other fields", () => {
    const parser = createSseParser();
    expect(parser.push(": keep-alive\r\n\r\nevent: x\r\ndata: hi\r\n\r\n")).toEqual(["hi"]);
  });
});

describe("parseChunk", () => {
  it("separates reasoning from content", () => {
    expect(parseChunk(chunk({ reasoning_content: "hmm" }))).toEqual([{ kind: "reasoning", text: "hmm" }]);
    expect(parseChunk(chunk({ content: "Hi" }))).toEqual([{ kind: "content", text: "Hi" }]);
  });

  it("reads the finish reason from the final chunk's empty delta", () => {
    expect(parseChunk(chunk({}, "length"))).toEqual([{ kind: "finish", reason: "length" }]);
  });

  it("reads usage from the trailing usage chunk", () => {
    const data = JSON.stringify({ choices: [], usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 } });
    expect(parseChunk(data)).toEqual([{ kind: "usage", usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 } }]);
  });

  it("reads a whole tool call from its delta", () => {
    const call = { index: 0, id: "call_0", type: "function", function: { name: "agent", arguments: '{"prompt":"x"}' } };
    expect(parseChunk(chunk({ tool_calls: [call] }))).toEqual([
      { kind: "tool_call", call: { id: "call_0", name: "agent", arguments: '{"prompt":"x"}' } },
    ]);
  });

  it("recognises the end marker", () => {
    expect(parseChunk("[DONE]")).toEqual([{ kind: "done" }]);
  });

  it("keeps every part of a chunk that carries several", () => {
    expect(parseChunk(chunk({ reasoning_content: "a", content: "b" }, "stop"))).toEqual([
      { kind: "reasoning", text: "a" },
      { kind: "content", text: "b" },
      { kind: "finish", reason: "stop" },
    ]);
  });
});
