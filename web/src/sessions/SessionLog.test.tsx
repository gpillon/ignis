import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Figures } from "../metrics/figures.ts";
import { SessionLog } from "./SessionLog.tsx";
import type { LogRow } from "./sessions.ts";

// Each request's row says what thinking budget it sent and whether the
// budget closed its reasoning (spec playground/04).

const figures = (thinkingForcedAt?: number): Figures => ({
  ttftMs: 100,
  decodeTokensPerSec: 50,
  durationMs: 1000,
  promptTokens: 10,
  completionTokens: 20,
  finishReason: "stop",
  partial: false,
  ...(thinkingForcedAt !== undefined ? { thinkingForcedAt } : {}),
});

const row = (change: Partial<LogRow>): LogRow => ({ n: 1, at: "12:00", laneTag: "interactive", reasoningEffort: "xhigh", figures: figures(), ...change });

/** The cells of the table's only row, as text. */
function cells(r: LogRow): string[] {
  const html = renderToStaticMarkup(<SessionLog rows={[r]} open onToggle={() => {}} />);
  const body = /<tbody>(.*)<\/tbody>/s.exec(html)?.[1] ?? "";
  return [...body.matchAll(/<td[^>]*>(.*?)<\/td>/gs)].map((m) => m[1].replace(/<[^>]+>/g, ""));
}

const headers = () =>
  [...renderToStaticMarkup(<SessionLog rows={[row({})]} open onToggle={() => {}} />).matchAll(/<th[^>]*>(.*?)<\/th>/g)].map((m) => m[1]);

describe("SessionLog and the thinking budget", () => {
  it("has a budget column beside the thinking effort", () => {
    const h = headers();
    expect(h.indexOf("Budget")).toBe(h.indexOf("Thinking") + 1);
  });

  const budget = (r: LogRow) => cells(r)[headers().indexOf("Budget")];

  it("shows the budget each request sent: a count, off for 0, default for none sent", () => {
    expect(budget(row({ thinkingBudget: 4096 }))).toBe("4096");
    expect(budget(row({ thinkingBudget: 0 }))).toBe("off");
    expect(budget(row({}))).toBe("default");
  });

  it("shows no budget for a request that cannot have one: thinking off, or the max effort", () => {
    expect(budget(row({ reasoningEffort: "none" }))).toBe("—");
    expect(budget(row({ reasoningEffort: "max" }))).toBe("—");
  });

  it("flags a request whose reasoning the budget closed, with where it closed", () => {
    expect(budget(row({ thinkingBudget: 16384, figures: figures(14884) }))).toBe("16384 · reached 14884");
    expect(budget(row({ figures: figures(8192) }))).toBe("default · reached 8192");
  });

  it("keeps the budget on a row that failed, whose figures never came", () => {
    expect(budget(row({ thinkingBudget: 2048, figures: null, error: "503: engine full" }))).toBe("2048");
  });
});
