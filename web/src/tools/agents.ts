import { computeFigures, type Figures } from "../figures.ts";
import { buildChatRequest, type ChatRequest, type Settings, type ToolDefinition } from "../request.ts";
import type { ToolCall } from "../sse.ts";
import { streamChat } from "../stream.ts";

// The `agent` tool: the model hands one self-contained sub-task to an agent,
// a separate request on an agent lane that sees only that task. Several calls
// in one reply run in parallel; each agent's answer goes back to the model as
// the call's tool result. Agents get no tools of their own, and run with the
// session's sampling and thinking settings.

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
You can delegate work with the \`agent\` tool. Each call starts an agent: a separate model instance that sees only the prompt you write, with no conversation history and no tools.
- Agents run in parallel on dedicated lanes. When a task splits into independent parts, call \`agent\` several times in the same reply, one call per part, rather than one after another.
- Make every prompt self-contained: include the facts, text or code the agent needs, and say exactly what it should return.
- Each agent's answer comes back to you as that call's result. Check the results, then write your reply to the user from them.
- Answer directly when the question is simple; agents are for work that benefits from being split.`;

/** The system prompt every agent request carries. */
export const AGENT_SYSTEM_PROMPT = `You are an agent working for another assistant on one self-contained task. You cannot ask questions and have no tools. Do the task and reply with the result only: complete, accurate and concise, ready to be merged into a larger answer.`;

/** ignis admits 8 requests in flight; the main reply has finished while its agents run. */
export const MAX_PARALLEL_AGENTS = 8;

export type AgentStatus = "queued" | "running" | "done" | "failed" | "stopped";

/** One agent as the Playground shows it and the model is told about it. */
export type AgentRun = {
  callId: string;
  name: string;
  prompt: string;
  status: AgentStatus;
  reasoning: string;
  content: string;
  /** When its current attempt started, on the stream's clock. */
  startedAt?: number;
  figures?: Figures;
  error?: string;
};

export type AgentTask = { callId: string; name: string; prompt: string };

const queued = (task: AgentTask): AgentRun => ({ ...task, status: "queued", reasoning: "", content: "" });

/**
 * A call the model made, read as an agent task, or as a failed run whose
 * error explains what was wrong (an unknown tool, arguments that are not a
 * JSON object with a prompt).
 */
export function parseAgentCall(call: ToolCall): { ok: true; task: AgentTask } | { ok: false; run: AgentRun } {
  const fail = (error: string, name: string) => ({
    ok: false as const,
    run: { ...queued({ callId: call.id, name, prompt: call.arguments }), status: "failed" as const, error },
  });
  if (call.name !== AGENT_TOOL_NAME) {
    return fail(`Unknown tool "${call.name}": the only tool available is "${AGENT_TOOL_NAME}".`, call.name);
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

/** An agent's request: its own system prompt and task, no tools, on an agent lane. */
export function agentRequest(settings: Settings, prompt: string): ChatRequest {
  return buildChatRequest({ ...settings, systemPrompt: AGENT_SYSTEM_PROMPT, laneTag: "agent" }, [{ role: "user", content: prompt }]);
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
  limit?: number;
  /** A full engine is retried, not failed: agents wait for a lane. */
  retryDelayMs?: number;
  maxRetries?: number;
  stream?: typeof streamChat;
  now?: () => number;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
};

/** Runs the tasks, at most `limit` at once, and resolves with every run finished (done, failed or stopped). */
export async function runAgents(tasks: AgentTask[], options: RunAgentsOptions): Promise<AgentRun[]> {
  const stream = options.stream ?? streamChat;
  const now = options.now ?? (() => performance.now());
  const sleep = options.sleep ?? abortableSleep;
  const limit = Math.max(1, options.limit ?? MAX_PARALLEL_AGENTS);
  const maxRetries = options.maxRetries ?? 60;
  const retryDelayMs = options.retryDelayMs ?? 1000;
  const { signal } = options;

  const runs = new Map(tasks.map((task) => [task.callId, queued(task)]));
  const set = (callId: string, change: Partial<AgentRun>) => {
    const run = { ...runs.get(callId)!, ...change };
    runs.set(callId, run);
    options.onUpdate(run);
  };

  async function runOne(task: AgentTask) {
    for (let attempt = 0; ; attempt++) {
      if (signal.aborted) return set(task.callId, { status: "stopped" });
      set(task.callId, { status: "running", startedAt: now(), reasoning: "", content: "" });
      const result = await stream({
        body: agentRequest(options.settings, task.prompt),
        signal,
        now,
        onEvent: (event) => {
          const run = runs.get(task.callId)!;
          if (event.kind === "reasoning") set(task.callId, { reasoning: run.reasoning + event.text });
          if (event.kind === "content") set(task.callId, { content: run.content + event.text });
        },
      });
      if (!result.ok && isEngineFull(result.message) && attempt < maxRetries && !signal.aborted) {
        set(task.callId, { status: "queued" });
        await sleep(retryDelayMs, signal);
        continue;
      }
      if (!result.ok) return set(task.callId, { status: "failed", error: result.message });
      return set(task.callId, {
        status: result.timeline.stopped ? "stopped" : "done",
        figures: computeFigures(result.timeline),
      });
    }
  }

  let next = 0;
  const worker = async () => {
    while (next < tasks.length) await runOne(tasks[next++]);
  };
  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return tasks.map((task) => runs.get(task.callId)!);
}
