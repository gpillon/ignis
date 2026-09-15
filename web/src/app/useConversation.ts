import { useRef, useState } from "react";
import type { ModelState } from "../api/model.ts";
import { buildChatRequest, conversationTurns, type Exchange, type Settings, type ToolExtras } from "../api/request.ts";
import { streamChat } from "../api/stream.ts";
import { computeFigures } from "../metrics/figures.ts";
import {
  addLogRow,
  addMessages,
  createSession,
  editMessage,
  forkSession,
  type LogRow,
  type Message,
  openSession,
  removeSession,
  type SessionList,
  truncateFrom,
  updateMessage,
} from "../sessions/sessions.ts";
import type { PlaygroundSettings } from "../settings/defaults.ts";
import { type AgentRun, parseAgentCall, runAgents, toolResult } from "../tools/agents/agents.ts";
import { routeCall, toolExtras, type ToolsState } from "../tools/index.ts";
import { getTavilyKey } from "../tools/web/tavilyKey.ts";
import { parseWebCall, runWeb, type WebRun, webToolResult } from "../tools/web/web.ts";

// The conversation loop: the sessions, the one reply streaming at a time,
// and the tools a reply calls. Sessions live in memory; a reload starts over.

/** How many replies in a row may call tools before the Playground stops running them. */
const MAX_TOOL_ROUNDS = 8;

const exchangeOf = (m: Message): Exchange => ({
  role: m.role,
  content: m.content,
  failed: m.error !== undefined,
  toolCalls: m.toolCalls,
  toolCallId: m.toolCallId,
});

export function useConversation({ model, settings, tools }: { model: ModelState; settings: PlaygroundSettings; tools: ToolsState }) {
  // Session and message ids. The counter lives in a ref, not in a module
  // variable: a hot reload keeps component state but re-runs the module, and
  // a restarted count would hand out ids still on screen (duplicate keys).
  const ids = useRef(2);
  const newId = () => ids.current++;
  const [list, setList] = useState<SessionList>(() => ({ sessions: [createSession(1)], activeId: 1 }));
  // The session a reply is streaming into; one stream at a time.
  const [streamingId, setStreamingId] = useState<number | null>(null);
  const controller = useRef<AbortController | null>(null);
  // Whether the reader is at the bottom of the conversation: only then do
  // new tokens pull the view down.
  const following = useRef(true);

  const active = list.sessions.find((s) => s.id === list.activeId) ?? list.sessions[0];
  const busy = streamingId !== null;
  /** A new turn can start: nothing streams and the model is known. */
  const canRun = !busy && model.state === "ready";

  function selectSession(id: number) {
    following.current = true;
    setList((l) => ({ ...l, activeId: id }));
  }

  function newSession() {
    following.current = true;
    const id = newId();
    setList((l) => openSession(l, id));
  }

  function deleteSession(sessionId: number) {
    const id = newId();
    setList((l) => removeSession(l, sessionId, id));
  }

  /**
   * One turn, streamed into `sessionId` after `history`: to `prompt` sent as
   * a new user message (send, resend), or to the history as it stands when
   * `prompt` is null (regenerate). A reply that calls agents starts them,
   * hands their answers back and streams the next reply, until one answers
   * without calling a tool. Stop ends the whole turn, agents included.
   */
  async function exchange(sessionId: number, history: Message[], prompt: string | null) {
    if (busy || model.state !== "ready") return;
    const requestSettings: Settings = { ...settings, model: model.id };
    const extras = toolExtras(tools);
    const abort = new AbortController();
    controller.current = abort;
    // Sending is a request to see the answer: follow it from the bottom.
    following.current = true;
    setStreamingId(sessionId);

    let conversation = history;
    if (prompt !== null) {
      const user: Message = { id: newId(), role: "user", content: prompt, reasoning: "", streaming: false };
      setList((l) => ({ ...l, sessions: addMessages(l.sessions, sessionId, [user]) }));
      conversation = [...conversation, user];
    }
    for (let round = 0; ; round++) {
      const reply = await streamReply(sessionId, conversation, requestSettings, extras, abort.signal);
      conversation = [...conversation, reply];
      const calls = reply.toolCalls ?? [];
      if (calls.length === 0 || reply.error) break;
      // Every call gets a result, even one that never ran: the history must stay well-formed.
      if (abort.signal.aborted || round >= MAX_TOOL_ROUNDS) {
        const reason = abort.signal.aborted
          ? "Stopped before the call ran."
          : `Not run: a turn can call tools at most ${MAX_TOOL_ROUNDS} times in a row.`;
        addToolResults(sessionId, calls.map((c) => ({ callId: c.id, content: reason })));
        break;
      }
      conversation = [...conversation, ...(await runToolCalls(sessionId, reply, requestSettings, extras, abort.signal))];
      if (abort.signal.aborted) break;
    }
    controller.current = null;
    setStreamingId(null);
  }

  /** Streams one assistant reply and logs it; resolves with the reply as it ended. */
  async function streamReply(
    sessionId: number,
    conversation: Message[],
    requestSettings: Settings,
    extras: ToolExtras,
    signal: AbortSignal,
  ): Promise<Message> {
    const request = buildChatRequest(requestSettings, conversationTurns(conversation.map(exchangeOf)), extras);
    let reply: Message = { id: newId(), role: "assistant", content: "", reasoning: "", streaming: true };
    setList((l) => ({ ...l, sessions: addMessages(l.sessions, sessionId, [reply]) }));
    const change = (f: (m: Message) => Message) => {
      reply = f(reply);
      const next = reply;
      setList((l) => ({ ...l, sessions: updateMessage(l.sessions, sessionId, next.id, () => next) }));
    };
    const result = await streamChat({
      body: request,
      signal,
      onEvent: (event) => {
        if (event.kind === "reasoning") change((m) => ({ ...m, reasoning: m.reasoning + event.text }));
        if (event.kind === "content") change((m) => ({ ...m, content: m.content + event.text }));
        if (event.kind === "tool_call") change((m) => ({ ...m, toolCalls: [...(m.toolCalls ?? []), event.call] }));
      },
    });
    const figures = result.ok ? computeFigures(result.timeline) : null;
    const error = result.ok ? undefined : result.message;
    change((m) => ({ ...m, streaming: false, figures: figures ?? undefined, error }));
    logRow(sessionId, { laneTag: request.class, reasoningEffort: request.reasoning_effort, figures, error });
    return reply;
  }

  /**
   * Runs a reply's calls — agents and web calls at once — shown on the reply
   * as they go; resolves with the tool results in call order.
   */
  async function runToolCalls(sessionId: number, reply: Message, requestSettings: Settings, extras: ToolExtras, signal: AbortSignal) {
    const calls = reply.toolCalls ?? [];
    const available = (extras.tools ?? []).map((t) => t.function.name);
    const parsed = calls.filter((c) => routeCall(c.name, available) === "agent").map((c) => parseAgentCall(c, available));
    const tasks = parsed.flatMap((p) => (p.ok ? [p.task] : []));
    let runs: AgentRun[] = parsed.map((p) => (p.ok ? { ...p.task, status: "queued", reasoning: "", content: "" } : p.run));
    const webParsed = calls.filter((c) => routeCall(c.name, available) === "web").map((c) => parseWebCall(c, available));
    const webTasks = webParsed.flatMap((p) => (p.ok ? [p.task] : []));
    let webRuns: WebRun[] = webParsed.map((p) => (p.ok ? { ...p.task, status: "running" } : p.run));
    const show = () => {
      const agents = runs;
      const web = webRuns;
      setList((l) => ({ ...l, sessions: updateMessage(l.sessions, sessionId, reply.id, (m) => ({ ...m, agents, web })) }));
    };
    show();
    await Promise.all([
      runAgents(tasks, {
        settings: requestSettings,
        signal,
        onUpdate: (run) => {
          runs = runs.map((r) => (r.callId === run.callId ? run : r));
          show();
        },
      }),
      runWeb(webTasks, {
        tavilyKey: getTavilyKey(),
        signal,
        onUpdate: (run) => {
          webRuns = webRuns.map((r) => (r.callId === run.callId ? run : r));
          show();
        },
      }),
    ]);
    for (const run of runs) {
      if (!tasks.some((t) => t.callId === run.callId) || (!run.figures && !run.error)) continue;
      logRow(sessionId, {
        laneTag: "agent",
        reasoningEffort: requestSettings.reasoningEffort,
        figures: run.figures ?? null,
        error: run.error,
        agent: run.name,
      });
    }
    const results = new Map([
      ...runs.map((run) => [run.callId, toolResult(run)] as const),
      ...webRuns.map((run) => [run.callId, webToolResult(run)] as const),
    ]);
    return addToolResults(sessionId, calls.map((c) => ({ callId: c.id, content: results.get(c.id) ?? "" })));
  }

  function addToolResults(sessionId: number, results: { callId: string; content: string }[]): Message[] {
    const messages = results.map(
      (r): Message => ({ id: newId(), role: "tool", content: r.content, toolCallId: r.callId, reasoning: "", streaming: false }),
    );
    setList((l) => ({ ...l, sessions: addMessages(l.sessions, sessionId, messages) }));
    return messages;
  }

  function logRow(sessionId: number, row: Omit<LogRow, "n" | "at">) {
    setList((l) => ({ ...l, sessions: addLogRow(l.sessions, sessionId, { at: new Date().toLocaleTimeString(), ...row }) }));
  }

  /** Sends `prompt` in the active session, when a turn can start. */
  function send(prompt: string) {
    if (!canRun) return;
    void exchange(active.id, active.messages, prompt);
  }

  /** Drops `messageId` and what follows, then streams again: an edited prompt, or a fresh reply. */
  function rerun(messageId: number, prompt: string | null) {
    const index = active.messages.findIndex((m) => m.id === messageId);
    if (index === -1 || busy || model.state !== "ready") return;
    const sessionId = active.id;
    setList((l) => ({ ...l, sessions: truncateFrom(l.sessions, sessionId, messageId) }));
    void exchange(sessionId, active.messages.slice(0, index), prompt);
  }

  function saveEdit(messageId: number, text: string) {
    const sessionId = active.id;
    setList((l) => ({ ...l, sessions: editMessage(l.sessions, sessionId, messageId, text) }));
  }

  function fork(messageId: number) {
    following.current = true;
    const id = newId();
    const sessionId = active.id;
    setList((l) => forkSession(l, sessionId, messageId, id));
  }

  function stop() {
    controller.current?.abort();
  }

  return {
    list,
    active,
    streamingId,
    busy,
    canRun,
    following,
    selectSession,
    newSession,
    deleteSession,
    send,
    rerun,
    saveEdit,
    fork,
    stop,
  };
}
