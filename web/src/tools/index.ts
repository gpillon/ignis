import type { ToolExtras } from "../api/request.ts";
import { AGENT_TOOL, AGENT_TOOL_NAME, AGENTS_IGNIS_PROMPT } from "./agents/agents.ts";
import { isWebTool, WEB_FETCH_TOOL, WEB_IGNIS_PROMPT, WEB_SEARCH_TOOL } from "./web/web.ts";

// The Playground's tools. Turning one on declares it to the model and adds
// its part to the ignis system prompt; the conversation loop reacts to its
// calls. Agents came first; web (search and fetch) runs in the browser.

export type ToolsState = { agents: boolean; web: boolean };

export const NO_TOOLS: ToolsState = { agents: false, web: false };

/** The read-only ignis system prompt: what the enabled tools add, in a fixed order. */
export function ignisPrompt(tools: ToolsState): string {
  return [tools.agents ? AGENTS_IGNIS_PROMPT : "", tools.web ? WEB_IGNIS_PROMPT : ""]
    .filter((part) => part !== "")
    .join("\n\n");
}

/** The prompt and definitions a request carries for the enabled tools. */
export function toolExtras(tools: ToolsState): ToolExtras {
  return {
    ignisPrompt: ignisPrompt(tools),
    tools: [...(tools.agents ? [AGENT_TOOL] : []), ...(tools.web ? [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] : [])],
  };
}

/**
 * Which runner answers a call, given the tool names the request declared:
 * a declared web tool goes to web; anything else — the agent tool, or a
 * name the model made up — to agents while they are on, else to web, so an
 * unknown call still fails where the reply shows it.
 */
export function routeCall(name: string, available: string[]): "agent" | "web" {
  if (isWebTool(name) && available.includes(name)) return "web";
  return available.includes(AGENT_TOOL_NAME) ? "agent" : "web";
}
