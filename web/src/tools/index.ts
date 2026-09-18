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
// into the prompt, once per session into the system prompt or — with its
// "update every prompt" option — once per turn, after the history. One switch
// turns all of them off without losing which ones were chosen.

export type ToolsState = {
  enabled: boolean;
  agents: boolean;
  web: boolean;
  askUser: boolean;
  dateTime: boolean;
  /**
   * An option of date and time, not a tool: send every turn its own moment as
   * a developer message after the history, instead of the session's moment in
   * the system prompt. Off keeps the prompt's head still, which is what lets a
   * long conversation reuse its prefix.
   */
  dateTimeLive: boolean;
  runJs: boolean;
  /** An option of run_js, not a tool: review the code with the model before it runs. */
  jsSafetyCheck: boolean;
  plan: boolean;
  memory: boolean;
  files: boolean;
  readFiles: boolean;
  /**
   * An option of every tool, not a tool: how many replies in a row may call
   * tools before the calls stop being run — the conversation's own loop, and
   * each agent's.
   */
  maxRounds: number;
};

/** The rounds a turn gets when nothing else is chosen. */
export const DEFAULT_TOOL_ROUNDS = 16;

export const NO_TOOLS: ToolsState = {
  enabled: true,
  agents: false,
  web: false,
  askUser: false,
  dateTime: false,
  dateTimeLive: false,
  runJs: false,
  jsSafetyCheck: true,
  plan: false,
  memory: false,
  files: false,
  readFiles: false,
  maxRounds: DEFAULT_TOOL_ROUNDS,
};

/** Every tool on, the safety check too: the Playground's default. */
export const ALL_TOOLS: ToolsState = {
  enabled: true,
  agents: true,
  web: true,
  askUser: true,
  dateTime: true,
  dateTimeLive: false,
  runJs: true,
  jsSafetyCheck: true,
  plan: true,
  memory: true,
  files: true,
  readFiles: true,
  maxRounds: DEFAULT_TOOL_ROUNDS,
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
    on.dateTime && !tools.dateTimeLive ? dateTimePrompt(context.now ?? new Date()) : "",
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

/**
 * The developer message a turn sent at `now` carries, or `undefined` when the
 * date and time tool is off or writes the session's moment into the system
 * prompt instead. It rides after the history, the one place in a prompt where
 * the moment can change between turns without moving a token ahead of it.
 */
export function turnDateTime(tools: ToolsState, now: Date): string | undefined {
  return tools.enabled && tools.dateTime && tools.dateTimeLive ? dateTimePrompt(now) : undefined;
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
 *
 * An agent answers one prompt and is gone, so it has no history to put a
 * developer message after: with the date and time tool on it always takes the
 * session's moment in its system prompt, whatever the conversation does. That
 * is also what keeps every agent of a session opening with the same tokens.
 */
export function agentExtras(tools: ToolsState, context: PromptContext = {}): ToolExtras {
  return toolExtras({ ...tools, agents: false, askUser: false, plan: false, dateTimeLive: false }, context);
}

/**
 * The switch for all tools. Off keeps each tool's choice for when it comes
 * back on; on with no tool chosen turns every tool on. The rounds are a
 * setting of the section, not of a tool: they survive either way.
 */
export function setAllTools(tools: ToolsState, enabled: boolean): ToolsState {
  if (enabled && CHOICES.every((choice) => !tools[choice])) return { ...ALL_TOOLS, maxRounds: tools.maxRounds };
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
