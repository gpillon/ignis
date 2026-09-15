import { describe, expect, it } from "vitest";
import { isAtBottom } from "./scroll.ts";

describe("isAtBottom", () => {
  // A 600 px viewport over 2000 px of conversation.
  const view = (scrollTop: number) => ({ scrollTop, clientHeight: 600, scrollHeight: 2000 });

  it("is true at the very bottom and within the slack above it", () => {
    expect(isAtBottom(view(1400))).toBe(true);
    expect(isAtBottom(view(1370))).toBe(true);
  });

  it("is false once the reader has scrolled up past the slack", () => {
    expect(isAtBottom(view(1300))).toBe(false);
    expect(isAtBottom(view(0))).toBe(false);
  });

  it("is true when everything fits and there is nothing to scroll", () => {
    expect(isAtBottom({ scrollTop: 0, clientHeight: 600, scrollHeight: 400 })).toBe(true);
  });
});
