import { useRef, useState } from "react";
import type { ModelState } from "../api/model.ts";
import { buildChatRequest, conversationTurns, type Exchange, type Settings, type ToolExtras } from "../api/request.ts";
import type { ToolCall } from "../api/sse.ts";
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
import { askToolResult, parseAskCall } from "../tools/ask/ask.ts";
import { unknownCall } from "../tools/errors.ts";
import { agentExtras, routeCall, toolExtras, type ToolsState } from "../tools/index.ts";
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
  // Questions waiting for the user, by `messageId:callId`: each settles with the answer.
  const waiting = useRef(new Map<string, (answer: string | null) => void>());
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
    // The date and time the tools write into the prompt: when the turn started.
    const startedAt = new Date();
    const extras = toolExtras(tools, startedAt);
    const agentTools = agentExtras(tools, startedAt);
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
    // Calls of the latest reply still without a result, if the loop breaks while they run.
    let unanswered: ToolCall[] = [];
    try {
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
        unanswered = calls;
        conversation = [...conversation, ...(await runToolCalls(sessionId, reply, requestSettings, extras, agentTools, abort.signal))];
        unanswered = [];
        if (abort.signal.aborted) break;
      }
    } catch (err) {
      // A fault in the loop must not leave the page busy for good: it ends the turn, says what broke and
      // answers the calls it left open, so the next turn starts from a well-formed history.
      console.error(err);
      const message = `The Playground failed: ${err instanceof Error ? err.message : String(err)}`;
      if (unanswered.length > 0) addToolResults(sessionId, unanswered.map((c) => ({ callId: c.id, content: message })));
      const failure: Message = { id: newId(), role: "assistant", content: "", reasoning: "", streaming: false, error: message };
      setList((l) => ({
        ...l,
        sessions: addMessages(
          l.sessions.map((s) =>
            s.id === sessionId ? { ...s, messages: s.messages.map((m) => (m.streaming ? { ...m, streaming: false } : m)) } : s,
          ),
          sessionId,
          [failure],
        ),
      }));
    } finally {
      controller.current = null;
      setStreamingId(null);
    }
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
  async function runToolCalls(
    sessionId: number,
    reply: Message,
    requestSettings: Settings,
    extras: ToolExtras,
    agentTools: ToolExtras,
    signal: AbortSignal,
  ) {
    const calls = reply.toolCalls ?? [];
    const available = (extras.tools ?? []).map((t) => t.function.name);
    const parsed = calls.filter((c) => routeCall(c.name, available) === "agent").map((c) => parseAgentCall(c, available));
    const tasks = parsed.flatMap((p) => (p.ok ? [p.task] : []));
    let runs: AgentRun[] = parsed.map((p) => (p.ok ? { ...p.task, status: "queued", reasoning: "", content: "" } : p.run));
    const webParsed = calls.filter((c) => routeCall(c.name, available) === "web").map((c) => parseWebCall(c, available));
    const webTasks = webParsed.flatMap((p) => (p.ok ? [p.task] : []));
    let webRuns: WebRun[] = webParsed.map((p) => (p.ok ? { ...p.task, status: "running" } : p.run));
    const unknownTools = calls.filter((c) => routeCall(c.name, available) === "unknown").map((c) => unknownCall(c, available));
    let questions = calls.filter((c) => routeCall(c.name, available) === "ask").map(parseAskCall);
    const show = () => {
      const agents = runs;
      const web = webRuns;
      const asked = questions;
      setList((l) => ({
        ...l,
        sessions: updateMessage(l.sessions, sessionId, reply.id, (m) => ({ ...m, agents, web, unknownTools, questions: asked })),
      }));
    };
    show();
    // Each waiting question settles when the user answers it (answer) or the turn is stopped.
    const asking = questions
      .filter((q) => q.status === "waiting")
      .map(
        (q) =>
          new Promise<void>((resolve) => {
            const key = `${reply.id}:${q.callId}`;
            const settle = (text: string | null) => {
              waiting.current.delete(key);
              signal.removeEventListener("abort", onAbort);
              questions = questions.map((x) =>
                x.callId !== q.callId ? x : text === null ? { ...x, status: "skipped" } : { ...x, status: "answered", answer: text },
              );
              show();
              resolve();
            };
            const onAbort = () => settle(null);
            if (signal.aborted) return settle(null);
            signal.addEventListener("abort", onAbort, { once: true });
            waiting.current.set(key, settle);
          }),
      );
    await Promise.all([
      runAgents(tasks, {
        settings: requestSettings,
        tools: agentTools,
        tavilyKey: getTavilyKey(),
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
      ...asking,
    ]);
    // A row per request an agent made; a run that failed adds one for its error.
    for (const run of runs) {
      if (!tasks.some((t) => t.callId === run.callId)) continue;
      const row = { laneTag: "agent" as const, reasoningEffort: requestSettings.reasoningEffort, agent: run.name };
      for (const figures of run.rounds ?? []) logRow(sessionId, { ...row, figures });
      if (run.error) logRow(sessionId, { ...row, figures: null, error: run.error });
    }
    const results = new Map([
      ...runs.map((run) => [run.callId, toolResult(run)] as const),
      ...webRuns.map((run) => [run.callId, webToolResult(run)] as const),
      ...unknownTools.map((call) => [call.callId, call.error] as const),
      ...questions.map((q) => [q.callId, askToolResult(q)] as const),
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

  /** The user's answer to a question a reply is waiting on. */
  function answer(messageId: number, callId: string, text: string) {
    waiting.current.get(`${messageId}:${callId}`)?.(text);
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
    answer,
    stop,
  };
}
