import type { ToolExtras } from "../request.ts";
import { AGENT_TOOL, AGENTS_IGNIS_PROMPT } from "./agents.ts";

// The Playground's tools. Turning one on declares it to the model and adds
// its part to the ignis system prompt; the conversation loop reacts to its
// calls. Agents is the first.

export type ToolsState = { agents: boolean };

export const NO_TOOLS: ToolsState = { agents: false };

/** The read-only ignis system prompt: what the enabled tools add, in a fixed order. */
export function ignisPrompt(tools: ToolsState): string {
  return [tools.agents ? AGENTS_IGNIS_PROMPT : ""].filter((part) => part !== "").join("\n\n");
}

/** The prompt and definitions a request carries for the enabled tools. */
export function toolExtras(tools: ToolsState): ToolExtras {
  return { ignisPrompt: ignisPrompt(tools), tools: tools.agents ? [AGENT_TOOL] : [] };
}
