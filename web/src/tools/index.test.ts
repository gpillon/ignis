import { describe, expect, it } from "vitest";
import { AGENT_TOOL, AGENTS_IGNIS_PROMPT } from "./agents.ts";
import { ignisPrompt, NO_TOOLS, toolExtras } from "./index.ts";

describe("tools", () => {
  it("adds nothing while no tool is on", () => {
    expect(ignisPrompt(NO_TOOLS)).toBe("");
    expect(toolExtras(NO_TOOLS)).toEqual({ ignisPrompt: "", tools: [] });
  });

  it("declares agents and adds their part of the ignis prompt", () => {
    expect(toolExtras({ agents: true })).toEqual({ ignisPrompt: AGENTS_IGNIS_PROMPT, tools: [AGENT_TOOL] });
  });
});
