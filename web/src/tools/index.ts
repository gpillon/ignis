import type { ToolExtras } from "../api/request.ts";
import { AGENT_TOOL, AGENT_TOOL_NAME, AGENTS_IGNIS_PROMPT } from "./agents/agents.ts";
import { isWebTool, WEB_FETCH_TOOL, WEB_IGNIS_PROMPT, WEB_SEARCH_TOOL } from "./web/web.ts";

// The Playground's tools. Turning one on declares it to the model and adds
// its part to the ignis system prompt; the conversation loop reacts to its
// calls. Agents came first; web (search and fetch) runs in the browser.
// One switch turns all of them off without losing which ones were chosen.

export type ToolsState = { enabled: boolean; agents: boolean; web: boolean };

export const NO_TOOLS: ToolsState = { enabled: true, agents: false, web: false };

/** The tools in use: each one's own switch, under the switch for all of them. */
function inUse(tools: ToolsState) {
  return { agents: tools.enabled && tools.agents, web: tools.enabled && tools.web };
}

/** The read-only ignis system prompt: what the enabled tools add, in a fixed order. */
export function ignisPrompt(tools: ToolsState): string {
  const on = inUse(tools);
  return [on.agents ? AGENTS_IGNIS_PROMPT : "", on.web ? WEB_IGNIS_PROMPT : ""].filter((part) => part !== "").join("\n\n");
}

/** The prompt and definitions a request carries for the enabled tools. */
export function toolExtras(tools: ToolsState): ToolExtras {
  const on = inUse(tools);
  return {
    ignisPrompt: ignisPrompt(tools),
    tools: [...(on.agents ? [AGENT_TOOL] : []), ...(on.web ? [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] : [])],
  };
}

/** What an agent's requests carry: the session's tools less `agent`, so agents do not start agents. */
export function agentExtras(tools: ToolsState): ToolExtras {
  return toolExtras({ ...tools, agents: false });
}

/**
 * The switch for all tools. Off keeps each tool's choice for when it comes
 * back on; on with no tool chosen turns every tool on.
 */
export function setAllTools(tools: ToolsState, enabled: boolean): ToolsState {
  if (enabled && !tools.agents && !tools.web) return { enabled, agents: true, web: true };
  return { ...tools, enabled };
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
