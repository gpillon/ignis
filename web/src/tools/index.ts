import type { ToolExtras } from "../api/request.ts";
import { AGENT_TOOL, AGENT_TOOL_NAME, AGENTS_IGNIS_PROMPT } from "./agents/agents.ts";
import { ASK_USER_IGNIS_PROMPT, ASK_USER_TOOL, ASK_USER_TOOL_NAME } from "./ask/ask.ts";
import { dateTimePrompt } from "./datetime/datetime.ts";
import { isWebTool, WEB_FETCH_TOOL, WEB_IGNIS_PROMPT, WEB_SEARCH_TOOL } from "./web/web.ts";

// The Playground's tools. Turning one on declares it to the model and adds
// its part to the ignis system prompt; the conversation loop reacts to its
// calls. Date and time declares nothing: it only writes the current moment
// into the prompt. One switch turns all of them off without losing which
// ones were chosen.

export type ToolsState = { enabled: boolean; agents: boolean; web: boolean; askUser: boolean; dateTime: boolean };

export const NO_TOOLS: ToolsState = { enabled: true, agents: false, web: false, askUser: false, dateTime: false };

/** Every tool on: the Playground's default. */
export const ALL_TOOLS: ToolsState = { enabled: true, agents: true, web: true, askUser: true, dateTime: true };

const CHOICES = ["agents", "web", "askUser", "dateTime"] as const;

/** The tools in use: each one's own switch, under the switch for all of them. */
function inUse(tools: ToolsState) {
  return {
    agents: tools.enabled && tools.agents,
    web: tools.enabled && tools.web,
    askUser: tools.enabled && tools.askUser,
    dateTime: tools.enabled && tools.dateTime,
  };
}

/** The read-only ignis system prompt: what the enabled tools add, in a fixed order, as of `now`. */
export function ignisPrompt(tools: ToolsState, now = new Date()): string {
  const on = inUse(tools);
  return [
    on.dateTime ? dateTimePrompt(now) : "",
    on.agents ? AGENTS_IGNIS_PROMPT : "",
    on.web ? WEB_IGNIS_PROMPT : "",
    on.askUser ? ASK_USER_IGNIS_PROMPT : "",
  ]
    .filter((part) => part !== "")
    .join("\n\n");
}

/** The prompt and definitions a request carries for the enabled tools. */
export function toolExtras(tools: ToolsState, now = new Date()): ToolExtras {
  const on = inUse(tools);
  return {
    ignisPrompt: ignisPrompt(tools, now),
    tools: [
      ...(on.agents ? [AGENT_TOOL] : []),
      ...(on.web ? [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] : []),
      ...(on.askUser ? [ASK_USER_TOOL] : []),
    ],
  };
}

/**
 * What an agent's requests carry: the session's tools less `agent` (agents
 * do not start agents) and `ask_user` (only the conversation asks the user).
 */
export function agentExtras(tools: ToolsState, now = new Date()): ToolExtras {
  return toolExtras({ ...tools, agents: false, askUser: false }, now);
}

/**
 * The switch for all tools. Off keeps each tool's choice for when it comes
 * back on; on with no tool chosen turns every tool on.
 */
export function setAllTools(tools: ToolsState, enabled: boolean): ToolsState {
  if (enabled && CHOICES.every((choice) => !tools[choice])) return ALL_TOOLS;
  return { ...tools, enabled };
}

/**
 * Which runner answers a call, given the tool names the request declared:
 * agents, web or ask for a declared tool; unknown for anything else — a tool
 * the model made up never runs, and gets the list of the ones it has.
 */
export function routeCall(name: string, available: string[]): "agent" | "web" | "ask" | "unknown" {
  if (!available.includes(name)) return "unknown";
  if (isWebTool(name)) return "web";
  if (name === ASK_USER_TOOL_NAME) return "ask";
  return name === AGENT_TOOL_NAME ? "agent" : "unknown";
}
