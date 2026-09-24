import { describe, expect, it } from "vitest";
import type { Timeline } from "../../metrics/figures.ts";
import type { ChatRequest, Settings, ToolExtras } from "../../api/request.ts";
import type { StreamOptions, StreamResult } from "../../api/stream.ts";
import { runWeb, WEB_FETCH_TOOL, WEB_IGNIS_PROMPT, WEB_SEARCH_TOOL, type WebRun } from "../web/web.ts";
import {
  AGENT_SYSTEM_PROMPT,
  AGENT_TOOLS_SYSTEM_PROMPT,
  agentRequest,
  type AgentRun,
  agentSummary,
  agentSystemPrompt,
  MAX_AGENT_TOOL_ROUNDS,
  parseAgentCall,
  runAgents,
  toolResult,
} from "./agents.ts";

const settings: Settings = {
  model: "m",
  systemPrompt: "the owner's prompt",
  temperature: 1,
  topP: 0.95,
  maxTokens: 100,
  reasoningEffort: "xhigh",
  thinkingBudget: null,
  laneTag: "interactive",
};

const timeline = (stopped = false): Timeline => ({
  sentAt: 0,
  firstTokenAt: 1,
  lastTokenAt: 11,
  endedAt: 12,
  usage: { prompt_tokens: 5, completion_tokens: 11, total_tokens: 16 },
  finishReason: "stop",
  stopped,
});

const task = (n: number) => ({ callId: `call_${n}`, name: `task ${n}`, prompt: `do ${n}` });

describe("parseAgentCall", () => {
  it("reads name and prompt", () => {
    expect(parseAgentCall({ id: "c", name: "agent", arguments: '{"name":" scan ","prompt":"look"}' })).toEqual({
      ok: true,
      task: { callId: "c", name: "scan", prompt: "look" },
    });
  });

  it("names an unnamed task after the tool", () => {
    const parsed = parseAgentCall({ id: "c", name: "agent", arguments: '{"prompt":"look"}' });
    expect(parsed.ok && parsed.task.name).toBe("agent");
  });

  it("fails an unknown tool, bad JSON and a missing prompt with a reason", () => {
    for (const [call, error] of [
      [{ id: "c", name: "read_file", arguments: "{}" }, /Unknown tool "read_file"/],
      [{ id: "c", name: "agent", arguments: "{oops" }, /not valid JSON/],
      [{ id: "c", name: "agent", arguments: '{"name":"x"}' }, /non-empty "prompt"/],
    ] as const) {
      const parsed = parseAgentCall(call);
      expect(parsed.ok).toBe(false);
      if (!parsed.ok) {
        expect(parsed.run.status).toBe("failed");
        expect(parsed.run.error).toMatch(error);
      }
    }
  });
});

describe("agentRequest", () => {
  it("sends only the task, with the agent system prompt, on the agent lane, without tools", () => {
    const body = agentRequest(settings, [{ role: "user", content: "do it" }]);
    expect(body.messages).toEqual([
      { role: "system", content: AGENT_SYSTEM_PROMPT },
      { role: "user", content: "do it" },
    ]);
    expect(body.class).toBe("agent");
    expect(body.reasoning_effort).toBe("xhigh");
    expect("tools" in body).toBe(false);
  });
});

describe("toolResult and agentSummary", () => {
  const run = (status: AgentRun["status"], extra: Partial<AgentRun> = {}): AgentRun => ({
    callId: "c",
    name: "n",
    prompt: "p",
    status,
    reasoning: "",
    content: "",
    ...extra,
  });

  it("tells the model what each run ended with", () => {
    expect(toolResult(run("done", { content: " answer " }))).toBe("answer");
    expect(toolResult(run("failed", { error: "boom" }))).toBe("The agent failed: boom");
    expect(toolResult(run("stopped", { content: "half" }))).toMatch(/stopped before it finished\.\n\nPartial answer:\n\nhalf/);
  });

  it("summarises the runs", () => {
    expect(agentSummary([run("done"), run("running"), run("done")])).toBe("3 agents: 1 running, 2 done");
    expect(agentSummary([run("queued")])).toBe("1 agent: 1 queued");
  });
});

describe("runAgents", () => {
  it("runs at most `limit` agents at once and collects their answers", async () => {
    let live = 0;
    let peak = 0;
    const bodies: ChatRequest[] = [];
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      bodies.push(o.body as ChatRequest);
      live++;
      peak = Math.max(peak, live);
      await new Promise((r) => setTimeout(r, 5));
      o.onEvent({ kind: "reasoning", text: "hm" });
      o.onEvent({ kind: "content", text: `answer to ${(o.body as ChatRequest).messages[1].content}` });
      live--;
      return { ok: true, timeline: timeline() };
    };
    const runs = await runAgents([task(1), task(2), task(3), task(4)], {
      settings,
      signal: new AbortController().signal,
      onUpdate: () => {},
      limit: 2,
      stream,
    });
    expect(peak).toBe(2);
    expect(bodies).toHaveLength(4);
    expect(runs.map((r) => [r.status, r.content])).toEqual([
      ["done", "answer to do 1"],
      ["done", "answer to do 2"],
      ["done", "answer to do 3"],
      ["done", "answer to do 4"],
    ]);
    expect(runs[0].figures?.completionTokens).toBe(11);
  });

  it("waits and retries while the engine is full", async () => {
    let calls = 0;
    const statuses: string[] = [];
    const stream = async (): Promise<StreamResult> =>
      ++calls === 1
        ? { ok: false, message: "503: the engine cannot admit the request right now (all lanes in use); retry", timeline: timeline() }
        : { ok: true, timeline: timeline() };
    const [run] = await runAgents([task(1)], {
      settings,
      signal: new AbortController().signal,
      onUpdate: (r) => statuses.push(r.status),
      stream,
      sleep: async () => {},
    });
    expect(calls).toBe(2);
    expect(run.status).toBe("done");
    expect(statuses).toContain("queued");
  });

  it("fails a run on any other error", async () => {
    const stream = async (): Promise<StreamResult> => ({ ok: false, message: "400: bad", timeline: timeline() });
    const [run] = await runAgents([task(1)], { settings, signal: new AbortController().signal, onUpdate: () => {}, stream });
    expect(run).toMatchObject({ status: "failed", error: "400: bad" });
  });

  it("stops every run once the signal aborts", async () => {
    const abort = new AbortController();
    abort.abort();
    let calls = 0;
    const stream = async (): Promise<StreamResult> => (calls++, { ok: true, timeline: timeline(true) });
    const runs = await runAgents([task(1), task(2)], { settings, signal: abort.signal, onUpdate: () => {}, stream });
    expect(calls).toBe(0);
    expect(runs.map((r) => r.status)).toEqual(["stopped", "stopped"]);
  });

  it("reports a stream stopped part-way as stopped", async () => {
    const stream = async (): Promise<StreamResult> => ({ ok: true, timeline: timeline(true) });
    const [run] = await runAgents([task(1)], { settings, signal: new AbortController().signal, onUpdate: () => {}, stream });
    expect(run.status).toBe("stopped");
  });
});

describe("agents with tools", () => {
  const webTools: ToolExtras = { ignisPrompt: WEB_IGNIS_PROMPT, tools: [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] };
  const tasksSeen: number[] = [];
  const doneWeb: typeof runWeb = async (tasks, o) => {
    tasksSeen.push(tasks.length);
    return tasks.map((t) => {
      const run: WebRun = { ...t, status: "done", results: [{ title: "T", url: "https://t", content: "c" }] };
      o.onUpdate(run);
      return run;
    });
  };
  const callEvent = (name: string, args: object) => ({ kind: "tool_call" as const, call: { id: "w1", name, arguments: JSON.stringify(args) } });

  it("sends the tools with their prompt, runs the calls and streams again until the agent answers", async () => {
    const bodies: ChatRequest[] = [];
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      bodies.push(o.body as ChatRequest);
      if (bodies.length === 1) {
        o.onEvent({ kind: "reasoning", text: "search first" });
        o.onEvent(callEvent("web_search", { query: "ignis" }));
      } else {
        o.onEvent({ kind: "reasoning", text: "got it" });
        o.onEvent({ kind: "content", text: "final" });
      }
      return { ok: true, timeline: timeline() };
    };
    const [run] = await runAgents([task(1)], {
      settings,
      tools: webTools,
      signal: new AbortController().signal,
      onUpdate: () => {},
      stream,
      runWeb: doneWeb,
    });
    expect(run).toMatchObject({
      status: "done",
      content: "final",
      reasoning: "search first\n\ngot it",
      systemPrompt: agentSystemPrompt(webTools),
    });
    expect(run.rounds).toHaveLength(2);
    expect(run.web?.map((w) => [w.tool, w.status])).toEqual([["web_search", "done"]]);
    expect(bodies[0].class).toBe("agent");
    expect(bodies[0].tools?.map((t) => t.function.name)).toEqual(["web_search", "web_fetch"]);
    expect(bodies[0].messages[0]).toEqual({ role: "system", content: `${WEB_IGNIS_PROMPT}\n\n${AGENT_TOOLS_SYSTEM_PROMPT}` });
    expect(bodies[1].messages.slice(1)).toEqual([
      { role: "user", content: "do 1" },
      {
        role: "assistant",
        content: "",
        tool_calls: [{ id: "w1", type: "function", function: { name: "web_search", arguments: '{"query":"ignis"}' } }],
      },
      { role: "tool", content: "1. T\nhttps://t\nc", tool_call_id: "w1" },
    ]);
  });

  it("answers an agent's call to agent as an unknown tool, without running it", async () => {
    const bodies: ChatRequest[] = [];
    tasksSeen.length = 0;
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      bodies.push(o.body as ChatRequest);
      if (bodies.length === 1) o.onEvent(callEvent("agent", { name: "x", prompt: "y" }));
      else o.onEvent({ kind: "content", text: "ok" });
      return { ok: true, timeline: timeline() };
    };
    const [run] = await runAgents([task(1)], {
      settings,
      tools: webTools,
      signal: new AbortController().signal,
      onUpdate: () => {},
      stream,
      runWeb: doneWeb,
    });
    expect(run.status).toBe("done");
    expect(tasksSeen).toEqual([0]);
    expect(run.web).toEqual([]);
    expect(run.unknownTools?.map((u) => [u.name, u.arguments])).toEqual([["agent", '{"name":"x","prompt":"y"}']]);
    expect(bodies[1].messages.at(-1)?.content).toMatch(/Unknown tool "agent": the tools available are "web_search", "web_fetch"/);
  });

  it("fails an agent that keeps calling tools, after the rounds it was given", async () => {
    let calls = 0;
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      calls++;
      o.onEvent(callEvent("web_search", { query: "again" }));
      return { ok: true, timeline: timeline() };
    };
    const keepCalling = (maxRounds?: number) => {
      calls = 0;
      return runAgents([task(1)], {
        settings,
        tools: webTools,
        signal: new AbortController().signal,
        onUpdate: () => {},
        ...(maxRounds === undefined ? {} : { maxRounds }),
        stream,
        runWeb: doneWeb,
      });
    };

    // The session's rounds: the caller's number, the last round being the one that is not run.
    const [short] = await keepCalling(2);
    expect(calls).toBe(3);
    expect(short).toMatchObject({ status: "failed", error: expect.stringMatching(/at most 2 times/) });

    const [byDefault] = await keepCalling();
    expect(calls).toBe(MAX_AGENT_TOOL_ROUNDS + 1);
    expect(byDefault).toMatchObject({ status: "failed", error: expect.stringMatching(/at most 16 times/) });
  });
});

describe("agents and the thinking budget", () => {
  const webTools: ToolExtras = { ignisPrompt: WEB_IGNIS_PROMPT, tools: [WEB_SEARCH_TOOL] };
  const doneWeb: typeof runWeb = async (tasks, o) =>
    tasks.map((t) => {
      const run: WebRun = { ...t, status: "done", results: [] };
      o.onUpdate(run);
      return run;
    });

  /** The bodies of every request two agents make, the second request of each after one web search. */
  async function bodiesFor(budgeted: Settings) {
    const bodies: ChatRequest[] = [];
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      const body = o.body as ChatRequest;
      bodies.push(body);
      if (body.messages.at(-1)?.role === "user") {
        o.onEvent({ kind: "tool_call", call: { id: `w${bodies.length}`, name: "web_search", arguments: '{"query":"q"}' } });
      }
      return { ok: true, timeline: timeline() };
    };
    await runAgents([task(1), task(2)], {
      settings: budgeted,
      tools: webTools,
      signal: new AbortController().signal,
      onUpdate: () => {},
      stream,
      runWeb: doneWeb,
    });
    return bodies;
  }

  it("sends the turn's budget on every request of every agent, its tool rounds included", async () => {
    const bodies = await bodiesFor({ ...settings, thinkingBudget: 4096 });
    expect(bodies).toHaveLength(4);
    expect(bodies.map((b) => b.thinking_budget)).toEqual([4096, 4096, 4096, 4096]);
  });

  it("sends 0 when the turn has no budget, and nothing on the server default", async () => {
    expect((await bodiesFor({ ...settings, thinkingBudget: 0 })).map((b) => b.thinking_budget)).toEqual([0, 0, 0, 0]);
    expect((await bodiesFor({ ...settings, thinkingBudget: null })).every((b) => !("thinking_budget" in b))).toBe(true);
  });

  it("sends an agent no budget under the max effort, as the turn itself", async () => {
    const bodies = await bodiesFor({ ...settings, reasoningEffort: "max", thinkingBudget: 4096 });
    expect(bodies.every((b) => b.reasoning_effort === "max" && !("thinking_budget" in b))).toBe(true);
  });
});
