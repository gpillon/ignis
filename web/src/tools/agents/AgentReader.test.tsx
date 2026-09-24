import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Figures } from "../../metrics/figures.ts";
import { AgentReader } from "./AgentReader.tsx";
import type { AgentRun } from "./agents.ts";

// An agent's reasoning runs across every request it made; the reader marks
// it when the thinking budget closed any of them.

const round = (thinkingForcedAt?: number): Figures => ({
  ttftMs: 100,
  decodeTokensPerSec: 50,
  durationMs: 1000,
  promptTokens: 10,
  completionTokens: 20,
  finishReason: "stop",
  partial: false,
  ...(thinkingForcedAt !== undefined ? { thinkingForcedAt } : {}),
});

const run = (rounds: Figures[]): AgentRun => ({
  callId: "c",
  name: "scan",
  prompt: "look",
  status: "done",
  reasoning: "Thinking about it.",
  content: "Found it.",
  figures: rounds.at(-1),
  rounds,
});

const render = (r: AgentRun) =>
  renderToStaticMarkup(<AgentReader run={r} markdown={false} figures={null} systemPrompt="s" onClose={() => {}} />);

describe("AgentReader and the thinking budget", () => {
  it("marks the reasoning when the budget closed any of the agent's requests", () => {
    const html = render(run([round(2048), round()]));
    expect(html).toContain("Budget reached");
    expect(html).toContain("2048 reasoning tokens");
  });

  it("carries no marker when every request closed its reasoning itself", () => {
    expect(render(run([round(), round()]))).not.toContain("Budget reached");
  });
});
