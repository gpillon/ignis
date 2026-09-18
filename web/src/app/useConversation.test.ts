import { describe, expect, it } from "vitest";
import { canStartTurn } from "./useConversation.ts";

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
