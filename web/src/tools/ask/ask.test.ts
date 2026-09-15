import { describe, expect, it } from "vitest";
import { askToolResult, parseAskCall, type Question } from "./ask.ts";

const call = (args: string) => ({ id: "q1", name: "ask_user", arguments: args });

describe("parseAskCall", () => {
  it("reads the question and keeps only non-empty string options", () => {
    expect(parseAskCall(call('{"question":" Which team? ","options":["Roma", " Lazio ", "", 3]}'))).toEqual({
      callId: "q1",
      question: "Which team?",
      options: ["Roma", "Lazio"],
      status: "waiting",
    });
    expect(parseAskCall(call('{"question":"Why?"}')).options).toEqual([]);
  });

  it("marks bad JSON and a missing question invalid, with a reason", () => {
    expect(parseAskCall(call("{oops"))).toMatchObject({ status: "invalid", error: expect.stringMatching(/not valid JSON/) });
    expect(parseAskCall(call('{"options":["a"]}'))).toMatchObject({ status: "invalid", error: expect.stringMatching(/non-empty "question"/) });
  });
});

describe("askToolResult", () => {
  const q: Question = { callId: "q1", question: "Which team?", options: [], status: "waiting" };

  it("tells the model the answer, a stopped turn, or why it was not asked", () => {
    expect(askToolResult({ ...q, status: "answered", answer: "Roma" })).toBe("The user answered: Roma");
    expect(askToolResult({ ...q, status: "skipped" })).toBe("The user did not answer: the turn was stopped.");
    expect(askToolResult({ ...q, status: "invalid", error: "bad" })).toBe("The question was not asked: bad");
  });
});
