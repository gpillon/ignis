import { useRef, useState } from "react";
import type { ModelState } from "../api/model.ts";
import { buildChatRequest, conversationTurns, type Exchange, type Settings, type ToolExtras } from "../api/request.ts";
import type { ToolCall } from "../api/sse.ts";
import { streamChat } from "../api/stream.ts";
import type { PromptImage } from "../conversation/images.ts";
import { computeFigures } from "../metrics/figures.ts";
import {
  addAttachments,
  addLogRow,
  addMessages,
  createSession,
  editMessage,
  forkSession,
  type LogRow,
  type Message,
  openSession,
  removeAttachment,
  removeSession,
  type SessionList,
  truncateFrom,
  updateMessage,
} from "../sessions/sessions.ts";
import type { PlaygroundSettings } from "../settings/defaults.ts";
import { type AgentRun, parseAgentCall, runAgents, toolResult } from "../tools/agents/agents.ts";
import { askToolResult, parseAskCall } from "../tools/ask/ask.ts";
import { unknownCall } from "../tools/errors.ts";
import { type Attachment, attachmentFromFile } from "../tools/local/attachments.ts";
import { type LocalContext, type LocalRun, localToolResult, runLocalCalls, startedRun } from "../tools/local/local.ts";
import { agentExtras, routeCall, toolExtras, type ToolsState, turnDateTime } from "../tools/index.ts";
import { getTavilyKey } from "../tools/web/tavilyKey.ts";
import { parseWebCall, runWeb, type WebRun, webToolResult } from "../tools/web/web.ts";

// The conversation loop: the sessions, the replies streaming into them, and
// the tools a reply calls. Sessions live in memory; a reload starts over.
//
// A session streams one reply at a time — its history is a line, and two
// replies writing into it would fork it. Whether *other* sessions may stream
// meanwhile is the caller's choice (`parallel`): off, the page runs one turn
// and every Send waits for it; on, each session runs its own. Either way the
// browser's stream budget (GitHub #220) queues what the connection cannot
// carry, so parallel sessions share the same five streams on localhost.

/**
 * A turn may start in `sessionId`: nothing streams there, and — unless
 * sessions run in parallel — nothing streams anywhere else either.
 */
export function canStartTurn(streaming: ReadonlySet<number>, sessionId: number, parallel: boolean): boolean {
  return parallel ? !streaming.has(sessionId) : streaming.size === 0;
}

const exchangeOf = (m: Message): Exchange => ({
  role: m.role,
  content: m.content,
  images: m.images,
  failed: m.error !== undefined,
  toolCalls: m.toolCalls,
  toolCallId: m.toolCallId,
  dateTime: m.dateTime,
});

export function useConversation({
  model,
  settings,
  tools,
  parallel,
}: {
  model: ModelState;
  settings: PlaygroundSettings;
  tools: ToolsState;
  /** Sessions other than the one streaming may start their own turn. */
  parallel: boolean;
}) {
  // Session and message ids. The counter lives in a ref, not in a module
  // variable: a hot reload keeps component state but re-runs the module, and
  // a restarted count would hand out ids still on screen (duplicate keys).
  const ids = useRef(2);
  const newId = () => ids.current++;
  const [list, setList] = useState<SessionList>(() => ({ sessions: [createSession(1)], activeId: 1 }));
  // The sessions a reply is streaming into, for the render.
  const [streaming, setStreaming] = useState<ReadonlySet<number>>(() => new Set());
  // The same, as the turns themselves see it: a ref settles two Sends in one
  // tick, which the state — one render behind — would let both through.
  const controllers = useRef(new Map<number, AbortController>());
  const running = () => new Set(controllers.current.keys());
  const [attachError, setAttachError] = useState<string | null>(null);
  // Questions waiting for the user, by `messageId:callId`: each settles with the answer.
  const waiting = useRef(new Map<string, (answer: string | null) => void>());
  // Whether the reader is at the bottom of the conversation: only then do
  // new tokens pull the view down.
  const following = useRef(true);

  const active = list.sessions.find((s) => s.id === list.activeId) ?? list.sessions[0];
  /** Anything at all is streaming: what the header's pulse and the model dot show. */
  const busy = streaming.size > 0;
  const streamingHere = streaming.has(active.id);
  /** A new turn can start in the session on screen, and the model is known. */
  const canRun = canStartTurn(streaming, active.id, parallel) && model.state === "ready";
  /** Send waits: this session streams, or another one does and turns are not parallel. */
  const sendBlocked = !canStartTurn(streaming, active.id, parallel);

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
    // A session that goes takes its turn with it: the loop would otherwise
    // keep a stream open, writing into messages nobody can read.
    controllers.current.get(sessionId)?.abort();
    const id = newId();
    setList((l) => removeSession(l, sessionId, id));
  }

  /**
   * One turn, streamed into `sessionId` after `history`: to `prompt` sent as
   * a new user message (send, resend), with `images` on it when it carries
   * any, or to the history as it stands when `prompt` is null (regenerate).
   * A reply that calls agents starts them, hands their answers back and
   * streams the next reply, until one answers without calling a tool. Stop
   * ends the whole turn, agents included.
   */
  async function exchange(sessionId: number, history: Message[], prompt: string | null, images: PromptImage[] = []) {
    if (!canStartTurn(running(), sessionId, parallel) || model.state !== "ready") return;
    const requestSettings: Settings = { ...settings, model: model.id };
    // The notes and files the tools write into the prompt are as they were when the turn started. The date and
    // time is the session's, not this turn's: it sits ahead of the whole conversation, so a moment that moved
    // between two turns would change the prompt's first tokens and cost a prefill of everything (GitHub #186).
    // The tool's "update every prompt" option is what sends this turn's own moment, from behind the history.
    const session = list.sessions.find((s) => s.id === sessionId);
    const attachments = session?.attachments ?? [];
    const promptContext = { now: session?.startedAt ?? new Date(), attachments };
    const extras = toolExtras(tools, promptContext);
    const agentTools = agentExtras(tools, promptContext);
    // The rounds the turn was started with: the same budget for this loop and for each agent's own.
    const maxRounds = Math.max(1, Math.floor(tools.maxRounds));
    const local: Omit<LocalContext, "settings" | "signal"> = {
      jsSafetyCheck: tools.jsSafetyCheck,
      attachments,
      onCheck: (figures, error) =>
        logRow(sessionId, { laneTag: "agent", reasoningEffort: "none", figures, error, agent: "run_js safety check" }),
    };
    const abort = new AbortController();
    controllers.current.set(sessionId, abort);
    // Sending is a request to see the answer: follow it from the bottom. Only
    // the session on screen, though — a turn running in another one must not
    // drag the reader down.
    if (sessionId === list.activeId) following.current = true;
    setStreaming((s) => new Set(s).add(sessionId));

    let conversation = history;
    if (prompt !== null) {
      const stamp = turnDateTime(tools, new Date());
      const user: Message = {
        id: newId(),
        role: "user",
        content: prompt,
        reasoning: "",
        streaming: false,
        ...(images.length > 0 ? { images } : {}),
        ...(stamp ? { dateTime: stamp } : {}),
      };
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
        if (abort.signal.aborted || round >= maxRounds) {
          const reason = abort.signal.aborted
            ? "Stopped before the call ran."
            : `Not run: a turn can call tools at most ${maxRounds} times in a row.`;
          addToolResults(sessionId, calls.map((c) => ({ callId: c.id, content: reason })));
          break;
        }
        unanswered = calls;
        conversation = [...conversation, ...(await runToolCalls(sessionId, reply, requestSettings, extras, agentTools, local, maxRounds, abort.signal))];
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
      controllers.current.delete(sessionId);
      setStreaming((s) => {
        const rest = new Set(s);
        rest.delete(sessionId);
        return rest;
      });
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
    local: Omit<LocalContext, "settings" | "signal">,
    maxRounds: number,
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
    const localCalls = calls.filter((c) => routeCall(c.name, available) === "local");
    let localRuns: LocalRun[] = localCalls.map(startedRun);
    const show = () => {
      const agents = runs;
      const web = webRuns;
      const asked = questions;
      const localNow = localRuns;
      setList((l) => ({
        ...l,
        sessions: updateMessage(l.sessions, sessionId, reply.id, (m) => ({
          ...m,
          agents,
          web,
          unknownTools,
          questions: asked,
          local: localNow,
        })),
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
        local,
        maxRounds,
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
      runLocalCalls(localCalls, { ...local, settings: requestSettings, signal }, (run) => {
        localRuns = localRuns.map((r) => (r.callId === run.callId ? run : r));
        show();
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
      ...localRuns.map((run) => [run.callId, localToolResult(run)] as const),
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

  /** Sends `prompt`, and the images picked with it, in the active session when a turn can start. */
  function send(prompt: string, images: PromptImage[] = []) {
    if (!canRun) return;
    void exchange(active.id, active.messages, prompt, images);
  }

  /**
   * Drops `messageId` and what follows, then streams again: an edited
   * prompt, or a fresh reply. An edited prompt keeps the images the original
   * was sent with — editing the words is not a reason to take the picture away.
   */
  function rerun(messageId: number, prompt: string | null) {
    const index = active.messages.findIndex((m) => m.id === messageId);
    if (index === -1 || !canStartTurn(running(), active.id, parallel) || model.state !== "ready") return;
    const sessionId = active.id;
    const images = active.messages[index].images ?? [];
    setList((l) => ({ ...l, sessions: truncateFrom(l.sessions, sessionId, messageId) }));
    void exchange(sessionId, active.messages.slice(0, index), prompt, prompt === null ? [] : images);
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

  /** Attaches files to the active session, as text; files that cannot be read are named in `attachError`. */
  async function attach(files: File[]) {
    const sessionId = active.id;
    const taken = active.attachments.map((a) => a.name);
    const added: Attachment[] = [];
    const errors: string[] = [];
    for (const file of files) {
      const result = await attachmentFromFile(file, [...taken, ...added.map((a) => a.name)]);
      if (result.ok) added.push(result.attachment);
      else errors.push(result.error);
    }
    setAttachError(errors.length > 0 ? errors.join(" ") : null);
    if (added.length > 0) setList((l) => ({ ...l, sessions: addAttachments(l.sessions, sessionId, added) }));
  }

  function detach(name: string) {
    const sessionId = active.id;
    setList((l) => ({ ...l, sessions: removeAttachment(l.sessions, sessionId, name) }));
  }

  /** The user's answer to a question a reply is waiting on. */
  function answer(messageId: number, callId: string, text: string) {
    waiting.current.get(`${messageId}:${callId}`)?.(text);
  }

  /** Stops the turn on screen; with nothing streaming here, stops the ones that are. */
  function stop() {
    const here = controllers.current.get(active.id);
    if (here) return here.abort();
    for (const turn of controllers.current.values()) turn.abort();
  }

  return {
    list,
    active,
    streaming,
    streamingHere,
    busy,
    sendBlocked,
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
    attach,
    detach,
    attachError,
    stop,
  };
}
