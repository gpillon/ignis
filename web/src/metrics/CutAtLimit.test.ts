import { describe, expect, it } from "vitest";
import { endedAtTokenLimit } from "./CutAtLimit.tsx";

// What the conversation's tool loop and an agent's round read before they run a reply's calls.

describe("endedAtTokenLimit", () => {
  it("is true only for a reply whose finish reason is length", () => {
    expect(endedAtTokenLimit({ finishReason: "length" })).toBe(true);
    expect(endedAtTokenLimit({ finishReason: "tool_calls" })).toBe(false);
    expect(endedAtTokenLimit({ finishReason: null })).toBe(false);
    expect(endedAtTokenLimit(undefined)).toBe(false);
  });
});
