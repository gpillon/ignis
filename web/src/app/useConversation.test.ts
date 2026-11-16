import { describe, expect, it } from "vitest";
import { streamIdOf } from "../api/socket.ts";
import type { Figures } from "../metrics/figures.ts";
import { callsNotRun, canStartTurn, sessionStream } from "./useConversation.ts";

// Whether a reply's tool calls run, and what each gets when they do not (GitHub #583): a reply cut at the token
// limit runs none of them, the closed ones included.

describe("callsNotRun", () => {
  const ended = (finishReason: string): Figures => ({
    ttftMs: 1,
    decodeTokensPerSec: 1,
    durationMs: 1,
    promptTokens: 1,
    completionTokens: 1,
    finishReason,
    partial: false,
  });

  it("runs the calls of a reply that ended on them, within the turn's rounds", () => {
    expect(callsNotRun(ended("tool_calls"), false, 0, 16)).toBeUndefined();
    expect(callsNotRun(ended("tool_calls"), false, 15, 16)).toBeUndefined();
  });

  it("runs none of the calls of a reply cut at the token limit", () => {
    expect(callsNotRun(ended("length"), false, 0, 16)).toBe("Not run: the reply was cut by the token limit.");
  });

  it("runs none once the turn is stopped or out of rounds", () => {
    expect(callsNotRun(ended("tool_calls"), true, 0, 16)).toBe("Stopped before the call ran.");
    expect(callsNotRun(ended("tool_calls"), false, 16, 16)).toBe("Not run: a turn can call tools at most 16 times in a row.");
  });
});

// When a Send may start a turn: the whole of the parallel setting.

describe("canStartTurn", () => {
  it("runs one turn for the whole page while sessions are not parallel", () => {
    expect(canStartTurn(new Set(), 1, false)).toBe(true);
    expect(canStartTurn(new Set([2]), 1, false)).toBe(false);
    expect(canStartTurn(new Set([1]), 1, false)).toBe(false);
  });

  it("lets the other sessions run their own turn in parallel, never a second one in the same session", () => {
    expect(canStartTurn(new Set(), 1, true)).toBe(true);
    expect(canStartTurn(new Set([2]), 1, true)).toBe(true);
    expect(canStartTurn(new Set([2, 3, 4]), 1, true)).toBe(true);
    // A session's history is a line: a second reply streaming into it would fork it.
    expect(canStartTurn(new Set([1, 2]), 1, true)).toBe(false);
  });
});

// Each session is a stream of its own on the socket (GitHub #283): two
// sessions sharing one would run in turn, not in parallel.

describe("sessionStream", () => {
  it("names each session's stream apart, within the ids ignis takes", () => {
    const names = [1, 2, 10, 12].map(sessionStream);
    expect(new Set(names).size).toBe(4);
    for (const name of names) expect(streamIdOf(name)).toBe(name);
  });
});
