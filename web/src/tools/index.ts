import type { ToolExtras } from "../api/request.ts";
import { AGENT_TOOL, AGENT_TOOL_NAME, AGENTS_IGNIS_PROMPT } from "./agents/agents.ts";
import { ASK_USER_IGNIS_PROMPT, ASK_USER_TOOL, ASK_USER_TOOL_NAME } from "./ask/ask.ts";
import { dateTimePrompt } from "./datetime/datetime.ts";
import type { Attachment } from "./local/attachments.ts";
import {
  attachmentsPrompt,
  CREATE_FILE_TOOL,
  FILES_PROMPT,
  isLocalTool,
  MEMORY_DELETE_TOOL,
  MEMORY_READ_TOOL,
  MEMORY_SAVE_TOOL,
  memoryPrompt,
  PLAN_PROMPT,
  READ_FILE_TOOL,
  RUN_JS_TOOL,
  runJsPrompt,
  UPDATE_PLAN_TOOL,
} from "./local/local.ts";
import { browserMemory, type MemoryNote } from "./local/memory.ts";
import { isWebTool, WEB_FETCH_TOOL, WEB_IGNIS_PROMPT, WEB_SEARCH_TOOL } from "./web/web.ts";

// The Playground's tools. Turning one on declares it to the model and adds
// its part to the ignis system prompt; the conversation loop reacts to its
// calls. Date and time declares nothing: it only writes the current moment
// into the prompt. One switch turns all of them off without losing which
// ones were chosen.

export type ToolsState = {
  enabled: boolean;
  agents: boolean;
  web: boolean;
  askUser: boolean;
  dateTime: boolean;
  runJs: boolean;
  /** An option of run_js, not a tool: review the code with the model before it runs. */
  jsSafetyCheck: boolean;
  plan: boolean;
  memory: boolean;
  files: boolean;
  readFiles: boolean;
};

export const NO_TOOLS: ToolsState = {
  enabled: true,
  agents: false,
  web: false,
  askUser: false,
  dateTime: false,
  runJs: false,
  jsSafetyCheck: true,
  plan: false,
  memory: false,
  files: false,
  readFiles: false,
};

/** Every tool on, the safety check too: the Playground's default. */
export const ALL_TOOLS: ToolsState = {
  enabled: true,
  agents: true,
  web: true,
  askUser: true,
  dateTime: true,
  runJs: true,
  jsSafetyCheck: true,
  plan: true,
  memory: true,
  files: true,
  readFiles: true,
};

const CHOICES = ["agents", "web", "askUser", "dateTime", "runJs", "plan", "memory", "files", "readFiles"] as const;

/** How many tools are in use. */
export const toolsInUse = (tools: ToolsState) => (tools.enabled ? CHOICES.filter((choice) => tools[choice]).length : 0);

/** What the prompt depends on besides the switches; each defaults to now, the saved notes and no attachments. */
export type PromptContext = { now?: Date; notes?: MemoryNote[]; attachments?: Attachment[] };

/** The tools in use: each one's own switch, under the switch for all of them. read_file needs an attached file. */
function inUse(tools: ToolsState, attachments: Attachment[]) {
  const on = (choice: (typeof CHOICES)[number]) => tools.enabled && tools[choice];
  return {
    agents: on("agents"),
    web: on("web"),
    askUser: on("askUser"),
    dateTime: on("dateTime"),
    runJs: on("runJs"),
    plan: on("plan"),
    memory: on("memory"),
    files: on("files"),
    readFiles: on("readFiles") && attachments.length > 0,
  };
}

/** The read-only ignis system prompt: what the enabled tools add, in a fixed order. */
export function ignisPrompt(tools: ToolsState, context: PromptContext = {}): string {
  const attachments = context.attachments ?? [];
  const on = inUse(tools, attachments);
  return [
    on.dateTime ? dateTimePrompt(context.now ?? new Date()) : "",
    on.agents ? AGENTS_IGNIS_PROMPT : "",
    on.web ? WEB_IGNIS_PROMPT : "",
    on.askUser ? ASK_USER_IGNIS_PROMPT : "",
    on.runJs ? runJsPrompt(tools.jsSafetyCheck) : "",
    on.plan ? PLAN_PROMPT : "",
    on.memory ? memoryPrompt(context.notes ?? browserMemory.list()) : "",
    on.files ? FILES_PROMPT : "",
    on.readFiles ? attachmentsPrompt(attachments) : "",
  ]
    .filter((part) => part !== "")
    .join("\n\n");
}

/** The prompt and definitions a request carries for the enabled tools. */
export function toolExtras(tools: ToolsState, context: PromptContext = {}): ToolExtras {
  const on = inUse(tools, context.attachments ?? []);
  return {
    ignisPrompt: ignisPrompt(tools, context),
    tools: [
      ...(on.agents ? [AGENT_TOOL] : []),
      ...(on.web ? [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] : []),
      ...(on.askUser ? [ASK_USER_TOOL] : []),
      ...(on.runJs ? [RUN_JS_TOOL] : []),
      ...(on.plan ? [UPDATE_PLAN_TOOL] : []),
      ...(on.memory ? [MEMORY_SAVE_TOOL, MEMORY_READ_TOOL, MEMORY_DELETE_TOOL] : []),
      ...(on.files ? [CREATE_FILE_TOOL] : []),
      ...(on.readFiles ? [READ_FILE_TOOL] : []),
    ],
  };
}

/**
 * What an agent's requests carry: the session's tools less `agent` (agents
 * do not start agents), `ask_user` (only the conversation asks the user)
 * and `update_plan` (the plan is the conversation's).
 */
export function agentExtras(tools: ToolsState, context: PromptContext = {}): ToolExtras {
  return toolExtras({ ...tools, agents: false, askUser: false, plan: false }, context);
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
 * agents, web, ask or local for a declared tool; unknown for anything else —
 * a tool the model made up never runs, and gets the list of the ones it has.
 */
export function routeCall(name: string, available: string[]): "agent" | "web" | "ask" | "local" | "unknown" {
  if (!available.includes(name)) return "unknown";
  if (isWebTool(name)) return "web";
  if (isLocalTool(name)) return "local";
  if (name === ASK_USER_TOOL_NAME) return "ask";
  return name === AGENT_TOOL_NAME ? "agent" : "unknown";
}
