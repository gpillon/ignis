import { describe, expect, it } from "vitest";
import { computeFigures, describeFigures, formatMs, formatRate } from "./figures.ts";

describe("describeFigures", () => {
  it("renders every figure, dashes for the unknown ones and `stopped` for a partial request", () => {
    const base = { ttftMs: 250, decodeTokensPerSec: 25, durationMs: 2300, promptTokens: 100, completionTokens: 51, finishReason: "stop", partial: false };
    expect(describeFigures(base)).toEqual({
      ttft: "250 ms",
      decode: "25.0 tok/s",
      duration: "2.30 s",
      promptTokens: "100",
      completionTokens: "51",
      finish: "stop",
    });
    const stopped = describeFigures({ ...base, decodeTokensPerSec: null, promptTokens: null, completionTokens: null, finishReason: null, partial: true });
    expect(stopped).toMatchObject({ decode: "—", promptTokens: "—", completionTokens: "—", finish: "stopped" });
  });
});

describe("formatting", () => {
  it("shows milliseconds below a second, seconds above, and a dash when unknown", () => {
    expect(formatMs(249.6)).toBe("250 ms");
    expect(formatMs(2300)).toBe("2.30 s");
    expect(formatMs(null)).toBe("—");
    expect(formatRate(25)).toBe("25.0 tok/s");
    expect(formatRate(null)).toBe("—");
  });
});

const usage = { prompt_tokens: 100, completion_tokens: 51, total_tokens: 151 };

describe("computeFigures", () => {
  it("derives TTFT, decode rate and duration from a finished request", () => {
    const figures = computeFigures({
      sentAt: 1000,
      firstTokenAt: 1250,
      lastTokenAt: 3250,
      endedAt: 3300,
      usage,
      finishReason: "stop",
      stopped: false,
    });
    expect(figures).toEqual({
      ttftMs: 250,
      // 50 tokens after the first, over the 2 s between first and last.
      decodeTokensPerSec: 25,
      durationMs: 2300,
      promptTokens: 100,
      completionTokens: 51,
      finishReason: "stop",
      partial: false,
    });
  });

  it("has no decode rate for a single-token reply", () => {
    const figures = computeFigures({
      sentAt: 0,
      firstTokenAt: 10,
      lastTokenAt: 10,
      endedAt: 12,
      usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
      finishReason: "length",
      stopped: false,
    });
    expect(figures.decodeTokensPerSec).toBeNull();
    expect(figures.ttftMs).toBe(10);
  });

  it("marks a stopped request partial, without the usage it never received", () => {
    const figures = computeFigures({ sentAt: 0, firstTokenAt: 40, lastTokenAt: 90, endedAt: 100, stopped: true });
    expect(figures).toEqual({
      ttftMs: 40,
      decodeTokensPerSec: null,
      durationMs: 100,
      promptTokens: null,
      completionTokens: null,
      finishReason: null,
      partial: true,
    });
  });

  it("has no TTFT when no token ever arrived", () => {
    const figures = computeFigures({ sentAt: 0, endedAt: 30, stopped: false });
    expect(figures.ttftMs).toBeNull();
    expect(figures.durationMs).toBe(30);
  });
});
