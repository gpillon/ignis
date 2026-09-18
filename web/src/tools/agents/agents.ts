import { computeFigures, type Figures } from "../../metrics/figures.ts";
import { buildChatRequest, type ChatRequest, type Settings, type ToolDefinition, type ToolExtras, type Turn } from "../../api/request.ts";
import type { ToolCall } from "../../api/sse.ts";
import { streamChat } from "../../api/stream.ts";
import { type UnknownCall, unknownCall, unknownToolError } from "../errors.ts";
import { isLocalTool, type LocalContext, type LocalRun, localToolResult, runLocalCalls, startedRun } from "../local/local.ts";
import { isWebTool, parseWebCall, runWeb, type WebRun, webToolResult } from "../web/web.ts";

// The `agent` tool: the model hands one self-contained sub-task to an agent,
// a separate request on an agent lane that sees only that task. Several calls
// in one reply run in parallel; each agent's answer goes back to the model as
// the call's tool result. Agents run with the session's sampling and thinking
// settings, and get every other tool that is on — never `agent` itself, so
// agents do not start agents. An agent that calls tools runs its own loop:
// stream, run the calls, send back the results, until it answers without one.

export const AGENT_TOOL_NAME = "agent";

export const AGENT_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: AGENT_TOOL_NAME,
    description:
      "Run one self-contained sub-task on a separate agent, in parallel with every other agent call in the same reply. Returns the agent's answer.",
    parameters: {
      type: "object",
      properties: {
        name: { type: "string", description: "A short label for the sub-task, a few words." },
        prompt: {
          type: "string",
          description: "Complete instructions for the agent: every fact and constraint it needs, and what to return.",
        },
      },
      required: ["name", "prompt"],
    },
  },
};

/** What the agent tool adds to the ignis system prompt of the conversation. */
export const AGENTS_IGNIS_PROMPT = `# Agents
You can delegate work with the \`agent\` tool. Each call starts an agent: a separate model instance that sees only the prompt you write, with no conversation history. It gets your other tools except \`ask_user\` and \`update_plan\`, if you have any, and cannot start agents of its own.
- Agents run in parallel on dedicated lanes. When a task splits into independent parts, call \`agent\` several times in the same reply, one call per part, rather than one after another.
- Make every prompt self-contained: include the facts, text or code the agent needs, and say exactly what it should return.
- Each agent's answer comes back to you as that call's result. Check the results, then write your reply to the user from them.
- Answer directly when the question is simple; agents are for work that benefits from being split.`;

/** The system prompt of an agent without tools. */
export const AGENT_SYSTEM_PROMPT = `You are an agent working for another assistant on one self-contained task. You cannot ask questions and have no tools. Do the task and reply with the result only: complete, accurate and concise, ready to be merged into a larger answer.`;

/** The system prompt of an agent with tools, after what the tools add. */
export const AGENT_TOOLS_SYSTEM_PROMPT = `You are an agent working for another assistant on one self-contained task. You cannot ask questions or start agents. Use your tools when the task needs them, then reply with the result only: complete, accurate and concise, ready to be merged into a larger answer.`;

/** The whole system prompt an agent with `extras` receives, as the reader shows it. */
export function agentSystemPrompt(extras: ToolExtras = {}): string {
  if (!extras.tools?.length) return AGENT_SYSTEM_PROMPT;
  return [extras.ignisPrompt ?? "", AGENT_TOOLS_SYSTEM_PROMPT].filter((part) => part.trim() !== "").join("\n\n");
}

/** ignis admits 8 requests in flight; the main reply has finished while its agents run. */
export const MAX_PARALLEL_AGENTS = 8;

/** How many replies in a row an agent may call tools in before it is failed, when the caller names no budget. */
export const MAX_AGENT_TOOL_ROUNDS = 16;

export type AgentStatus = "queued" | "running" | "done" | "failed" | "stopped";

/** One agent as the Playground shows it and the model is told about it. */
export type AgentRun = {
  callId: string;
  name: string;
  prompt: string;
  status: AgentStatus;
  /** Across every request of the run. */
  reasoning: string;
  /** The text of the run's latest request: its answer, once done. */
  content: string;
  /** When its first request started, on the stream's clock. */
  startedAt?: number;
  /** The latest request's figures. */
  figures?: Figures;
  /** Every finished request's figures, in order. */
  rounds?: Figures[];
  /** The web calls the agent made, in order. */
  web?: WebRun[];
  /** The local tool calls the agent made, in order. */
  local?: LocalRun[];
  /** Calls the agent made to tools it was not given. */
  unknownTools?: UnknownCall[];
  systemPrompt?: string;
  error?: string;
};

export type AgentTask = { callId: string; name: string; prompt: string };

const queued = (task: AgentTask): AgentRun => ({ ...task, status: "queued", reasoning: "", content: "" });

/**
 * A call the model made, read as an agent task, or as a failed run whose
 * error explains what was wrong (an unknown tool, arguments that are not a
 * JSON object with a prompt).
 */
export function parseAgentCall(
  call: ToolCall,
  available: string[] = [AGENT_TOOL_NAME],
): { ok: true; task: AgentTask } | { ok: false; run: AgentRun } {
  const fail = (error: string, name: string) => ({
    ok: false as const,
    run: { ...queued({ callId: call.id, name, prompt: call.arguments }), status: "failed" as const, error },
  });
  if (call.name !== AGENT_TOOL_NAME || !available.includes(AGENT_TOOL_NAME)) {
    return fail(unknownToolError(call.name, available), call.name);
  }
  let args: unknown;
  try {
    args = JSON.parse(call.arguments);
  } catch {
    return fail("The call's arguments are not valid JSON.", AGENT_TOOL_NAME);
  }
  const { name, prompt } = (typeof args === "object" && args !== null ? args : {}) as { name?: unknown; prompt?: unknown };
  const label = typeof name === "string" && name.trim() !== "" ? name.trim() : AGENT_TOOL_NAME;
  if (typeof prompt !== "string" || prompt.trim() === "") return fail('The call needs a non-empty "prompt".', label);
  return { ok: true, task: { callId: call.id, name: label, prompt } };
}

/** An agent's request: its own system prompt and turns, its tools if any, on an agent lane. */
export function agentRequest(settings: Settings, turns: Turn[], extras: ToolExtras = {}): ChatRequest {
  const withTools = (extras.tools?.length ?? 0) > 0;
  return buildChatRequest(
    { ...settings, systemPrompt: withTools ? AGENT_TOOLS_SYSTEM_PROMPT : AGENT_SYSTEM_PROMPT, laneTag: "agent" },
    turns,
    withTools ? extras : {},
  );
}

/** The tool result the model receives for a finished run. */
export function toolResult(run: AgentRun): string {
  if (run.status === "done") return run.content.trim() || "(The agent returned no text.)";
  if (run.status === "failed") return `The agent failed: ${run.error ?? "unknown error"}`;
  const partial = run.content.trim();
  return `The agent was stopped before it finished.${partial ? `\n\nPartial answer:\n\n${partial}` : ""}`;
}

const ORDER: AgentStatus[] = ["running", "queued", "done", "failed", "stopped"];

/** `3 agents: 2 running, 1 done`. */
export function agentSummary(runs: AgentRun[]): string {
  const parts = ORDER.map((status) => [status, runs.filter((r) => r.status === status).length] as const)
    .filter(([, n]) => n > 0)
    .map(([status, n]) => `${n} ${status}`);
  return `${runs.length} ${runs.length === 1 ? "agent" : "agents"}${parts.length ? `: ${parts.join(", ")}` : ""}`;
}

const isEngineFull = (message: string) => /engine_full|all lanes in use/i.test(message);

function abortableSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(timer), resolve()), { once: true });
  });
}

export type RunAgentsOptions = {
  settings: Settings;
  signal: AbortSignal;
  /** Every change to a run, as a whole new run. */
  onUpdate: (run: AgentRun) => void;
  /** The tools agents get: the session's, less `agent`. */
  tools?: ToolExtras;
  /** For the agents' web searches. */
  tavilyKey?: string | null;
  /** For the agents' local tools: the safety check, the attachments, the check log. */
  local?: Omit<LocalContext, "settings" | "signal">;
  limit?: number;
  /** How many replies in a row an agent may call tools in; the session's tool rounds. */
  maxRounds?: number;
  /** A full engine is retried, not failed: agents wait for a lane. */
  retryDelayMs?: number;
  maxRetries?: number;
  stream?: typeof streamChat;
  runWeb?: typeof runWeb;
  now?: () => number;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
};

/** Runs the tasks, at most `limit` at once, and resolves with every run finished (done, failed or stopped). */
export async function runAgents(tasks: AgentTask[], options: RunAgentsOptions): Promise<AgentRun[]> {
  const stream = options.stream ?? streamChat;
  const doRunWeb = options.runWeb ?? runWeb;
  const now = options.now ?? (() => performance.now());
  const sleep = options.sleep ?? abortableSleep;
  const limit = Math.max(1, options.limit ?? MAX_PARALLEL_AGENTS);
  const maxRounds = Math.max(1, Math.floor(options.maxRounds ?? MAX_AGENT_TOOL_ROUNDS));
  const maxRetries = options.maxRetries ?? 60;
  const retryDelayMs = options.retryDelayMs ?? 1000;
  const tools = options.tools ?? {};
  const available = (tools.tools ?? []).map((t) => t.function.name);
  const systemPrompt = agentSystemPrompt(tools);
  const { signal } = options;

  const runs = new Map(tasks.map((task) => [task.callId, { ...queued(task), systemPrompt }]));
  const get = (callId: string) => runs.get(callId)!;
  const set = (callId: string, change: Partial<AgentRun>) => {
    const run = { ...get(callId), ...change };
    runs.set(callId, run);
    options.onUpdate(run);
  };

  /** One request of the run, retried while the engine is full; null once the run was stopped. */
  async function request(task: AgentTask, turns: Turn[], round: number) {
    const before = get(task.callId).reasoning;
    for (let attempt = 0; ; attempt++) {
      if (signal.aborted) {
        set(task.callId, { status: "stopped" });
        return null;
      }
      const calls: ToolCall[] = [];
      // A later request's reasoning is set apart from the earlier ones'.
      let separator = before ? "\n\n" : "";
      const result = await stream({
        body: agentRequest(options.settings, turns, tools),
        signal,
        now,
        // An agent whose stream is waiting for a connection (GitHub #220) is
        // still queued: it turns running when its request actually goes out.
        onStart: () => set(task.callId, { status: "running", reasoning: before, content: "", ...(round === 0 ? { startedAt: now() } : {}) }),
        onEvent: (event) => {
          const run = get(task.callId);
          if (event.kind === "reasoning") {
            set(task.callId, { reasoning: run.reasoning + separator + event.text });
            separator = "";
          }
          if (event.kind === "content") set(task.callId, { content: run.content + event.text });
          if (event.kind === "tool_call") calls.push(event.call);
        },
      });
      if (!result.ok && isEngineFull(result.message) && attempt < maxRetries && !signal.aborted) {
        set(task.callId, { status: "queued" });
        await sleep(retryDelayMs, signal);
        continue;
      }
      return { result, calls };
    }
  }

  /**
   * Runs one request's declared web and local calls at once, answers any
   * other call as an unknown tool, and resolves with each call's result.
   */
  async function runCalls(callId: string, calls: ToolCall[]): Promise<Map<string, string>> {
    const earlierWeb = get(callId).web ?? [];
    const earlierLocal = get(callId).local ?? [];
    const declared = (call: ToolCall) => available.includes(call.name);
    const webCalls = calls.filter((call) => declared(call) && isWebTool(call.name));
    const localCalls = calls.filter((call) => declared(call) && isLocalTool(call.name));
    const unknown = calls
      .filter((call) => !webCalls.includes(call) && !localCalls.includes(call))
      .map((call) => unknownCall(call, available));
    const parsed = webCalls.map((call) => parseWebCall(call, available));
    let web: WebRun[] = parsed.map((p) => (p.ok ? { ...p.task, status: "running" } : p.run));
    let local: LocalRun[] = localCalls.map(startedRun);
    const show = () => set(callId, { web: [...earlierWeb, ...web], local: [...earlierLocal, ...local] });
    if (unknown.length > 0) set(callId, { unknownTools: [...(get(callId).unknownTools ?? []), ...unknown] });
    show();
    await Promise.all([
      doRunWeb(
        parsed.flatMap((p) => (p.ok ? [p.task] : [])),
        {
          tavilyKey: options.tavilyKey ?? null,
          signal,
          onUpdate: (run) => {
            web = web.map((r) => (r.callId === run.callId ? run : r));
            show();
          },
        },
      ),
      runLocalCalls(
        localCalls,
        { jsSafetyCheck: true, attachments: [], ...options.local, settings: options.settings, signal },
        (run) => {
          local = local.map((r) => (r.callId === run.callId ? run : r));
          show();
        },
      ),
    ]);
    return new Map([
      ...web.map((run) => [run.callId, webToolResult(run)] as const),
      ...local.map((run) => [run.callId, localToolResult(run)] as const),
      ...unknown.map((call) => [call.callId, call.error] as const),
    ]);
  }

  async function runOne(task: AgentTask) {
    let turns: Turn[] = [{ role: "user", content: task.prompt }];
    for (let round = 0; ; round++) {
      const answer = await request(task, turns, round);
      if (!answer) return;
      const { result, calls } = answer;
      if (!result.ok) return set(task.callId, { status: "failed", error: result.message });
      const figures = computeFigures(result.timeline);
      set(task.callId, { figures, rounds: [...(get(task.callId).rounds ?? []), figures] });
      if (result.timeline.stopped) return set(task.callId, { status: "stopped" });
      if (calls.length === 0) return set(task.callId, { status: "done" });
      if (round >= maxRounds) {
        return set(task.callId, {
          status: "failed",
          error: `An agent can call tools at most ${maxRounds} times in a row; this one kept calling.`,
        });
      }
      const content = get(task.callId).content;
      const results = await runCalls(task.callId, calls);
      turns = [
        ...turns,
        { role: "assistant", content, toolCalls: calls },
        ...calls.map((call): Turn => ({ role: "tool", content: results.get(call.id) ?? "", toolCallId: call.id })),
      ];
    }
  }

  let next = 0;
  const worker = async () => {
    while (next < tasks.length) await runOne(tasks[next++]);
  };
  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return tasks.map((task) => get(task.callId));
}
