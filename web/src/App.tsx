import {
  type CSSProperties,
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { AgentReader, AgentStrip } from "./AgentStrip.tsx";
import flame from "./brand/flame.webp";
import wordmark from "./brand/wordmark-light.webp";
import { compactTokens, type ContextUsage, contextUsage } from "./context.ts";
import { computeFigures, describeFigures, type Figures } from "./figures.ts";
import { Markdown } from "./Markdown.tsx";
import {
  buildChatRequest,
  conversationTurns,
  type Exchange,
  REASONING_EFFORTS,
  type ReasoningEffort,
  type Settings,
  type ToolExtras,
} from "./request.ts";
import { isAtBottom } from "./scroll.ts";
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
  type Session,
  type SessionList,
  truncateFrom,
  updateMessage,
} from "./sessions.ts";
import { trackFill } from "./slider.ts";
import { streamChat } from "./stream.ts";
import { AGENT_SYSTEM_PROMPT, type AgentRun, parseAgentCall, runAgents, toolResult } from "./tools/agents.ts";
import { ignisPrompt, NO_TOOLS, toolExtras, type ToolsState } from "./tools/index.ts";

// The Playground (GitHub #164): a streaming chat against ignis's own
// /v1/chat/completions, with per-request figures measured in the browser.
// Sessions, settings and figures live in memory; a reload starts over.

/** How many replies in a row may call tools before the Playground stops running them. */
const MAX_TOOL_ROUNDS = 8;

const exchangeOf = (m: Message): Exchange => ({
  role: m.role,
  content: m.content,
  failed: m.error !== undefined,
  toolCalls: m.toolCalls,
  toolCallId: m.toolCallId,
});

type ModelState =
  | { state: "loading" }
  | { state: "ready"; id: string; contextLimit: number | null }
  | { state: "error"; message: string };

type Drawer = "sessions" | "settings" | null;

const DEFAULT_SETTINGS: Omit<Settings, "model"> = {
  systemPrompt: "",
  temperature: 1,
  topP: 0.95,
  maxTokens: 16384,
  reasoningEffort: "xhigh",
  laneTag: "interactive",
};

const EFFORT_LABELS: Record<ReasoningEffort, string> = { none: "Off", low: "Low", medium: "Medium", xhigh: "X-high" };

const caption = "font-display text-[13px] font-medium text-ash";
const field =
  "w-full rounded-[2px] border border-line bg-surface px-2.5 py-2 text-sm text-ink placeholder:text-ash/70 focus:border-ember focus:outline-none";


export function App() {
  const [model, setModel] = useState<ModelState>({ state: "loading" });
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  // Session and message ids. The counter lives in a ref, not in a module
  // variable: a hot reload keeps component state but re-runs the module, and
  // a restarted count would hand out ids still on screen (duplicate keys).
  const ids = useRef(2);
  const newId = () => ids.current++;
  const [list, setList] = useState<SessionList>(() => ({ sessions: [createSession(1)], activeId: 1 }));
  // The session a reply is streaming into; one stream at a time.
  const [streamingId, setStreamingId] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [logOpen, setLogOpen] = useState(false);
  const [markdown, setMarkdown] = useState(true);
  const [tools, setTools] = useState<ToolsState>(NO_TOOLS);
  // The agent open in the reader: the reply that started it, and its call.
  const [reader, setReader] = useState<{ messageId: number; callId: string } | null>(null);
  const closeReader = useCallback(() => setReader(null), []);
  const [drawer, setDrawer] = useState<Drawer>(null);
  const controller = useRef<AbortController | null>(null);
  const transcript = useRef<HTMLDivElement>(null);
  // Whether the reader is at the bottom of the conversation: only then do
  // new tokens pull the view down.
  const following = useRef(true);

  const active = list.sessions.find((s) => s.id === list.activeId) ?? list.sessions[0];
  const busy = streamingId !== null;
  const readerRun: AgentRun | undefined = reader
    ? active.messages.find((m) => m.id === reader.messageId)?.agents?.find((r) => r.callId === reader.callId)
    : undefined;

  useEffect(() => {
    fetch("/v1/models")
      .then(async (res) => {
        if (!res.ok) throw new Error(`GET /v1/models: ${res.status}`);
        const body = (await res.json()) as { data?: { id: string; max_model_len?: number }[] };
        const id = body.data?.[0]?.id;
        if (!id) throw new Error("GET /v1/models: no model listed");
        setModel({ state: "ready", id, contextLimit: body.data?.[0]?.max_model_len ?? null });
      })
      .catch((err: unknown) => setModel({ state: "error", message: String(err) }));
  }, []);

  useEffect(() => {
    if (!drawer) return;
    const close = (e: globalThis.KeyboardEvent) => e.key === "Escape" && setDrawer(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [drawer]);

  useLayoutEffect(() => {
    const el = transcript.current;
    if (el && following.current) el.scrollTop = el.scrollHeight;
  }, [active.messages, active.id]);

  function selectSession(id: number) {
    following.current = true;
    setList((l) => ({ ...l, activeId: id }));
    setDrawer(null);
    setReader(null);
  }

  function newSession() {
    following.current = true;
    const id = newId();
    setList((l) => openSession(l, id));
    setDrawer(null);
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
      conversation = [...conversation, ...(await runToolCalls(sessionId, reply, requestSettings, abort.signal))];
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

  /** Runs a reply's calls as agents, shown on the reply as they go; resolves with the tool results in call order. */
  async function runToolCalls(sessionId: number, reply: Message, requestSettings: Settings, signal: AbortSignal) {
    const parsed = (reply.toolCalls ?? []).map(parseAgentCall);
    const tasks = parsed.flatMap((p) => (p.ok ? [p.task] : []));
    let runs: AgentRun[] = parsed.map((p) => (p.ok ? { ...p.task, status: "queued", reasoning: "", content: "" } : p.run));
    const show = () => {
      const snapshot = runs;
      setList((l) => ({ ...l, sessions: updateMessage(l.sessions, sessionId, reply.id, (m) => ({ ...m, agents: snapshot })) }));
    };
    show();
    await runAgents(tasks, {
      settings: requestSettings,
      signal,
      onUpdate: (run) => {
        runs = runs.map((r) => (r.callId === run.callId ? run : r));
        show();
      },
    });
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
    return addToolResults(sessionId, runs.map((run) => ({ callId: run.callId, content: toolResult(run) })));
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

  function send() {
    const text = input.trim();
    if (!text || busy || model.state !== "ready") return;
    setInput("");
    void exchange(active.id, active.messages, text);
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

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  const set = <K extends keyof typeof settings>(key: K, value: (typeof settings)[K]) =>
    setSettings((s) => ({ ...s, [key]: value }));

  // ignis refuses greedy sampling with a top_p it would ignore.
  const greedyConflict = settings.temperature === 0 && settings.topP !== 1;
  const streamingHere = streamingId === active.id;

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <header className="z-20 shrink-0 bg-kiln text-[#eae8e4]">
        <div className="flex items-center gap-3 px-4 py-3 sm:gap-4 md:px-6">
          <HeaderButton label="Sessions" onClick={() => setDrawer("sessions")}>
            <IconSessions />
          </HeaderButton>
          <img src={flame} alt="" className="flame h-9 w-auto" data-busy={busy} />
          <img src={wordmark} alt="ignis" className="h-[18px] w-auto" />
          <span className="hidden h-6 w-px bg-kiln-line sm:block" aria-hidden />
          <span className="hidden font-display text-[15px] font-medium tracking-wide text-[#b9bec4] sm:block">Playground</span>
          <ModelStatus model={model} busy={busy} />
          <HeaderButton label="Settings" onClick={() => setDrawer("settings")}>
            <IconSliders />
          </HeaderButton>
        </div>
        <div className="heat" data-busy={busy} aria-hidden />
      </header>

      <div className="relative flex min-h-0 flex-1">
        {drawer && (
          <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => setDrawer(null)} aria-hidden />
        )}

        <aside
          aria-label="Sessions"
          className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-line bg-ground transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-60 lg:translate-x-0 ${drawer === "sessions" ? "translate-x-0" : "max-lg:invisible max-lg:-translate-x-full"}`}
        >
          <div className="p-3">
            <button
              type="button"
              className="cut flex w-full items-center gap-2 bg-surface px-3 py-2.5 font-display text-sm font-medium text-ink hover:bg-line"
              onClick={newSession}
            >
              <IconPlus />
              New session
            </button>
          </div>
          <ul className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-3 pb-3">
            {list.sessions.map((s) => (
              <SessionItem
                key={s.id}
                session={s}
                active={s.id === active.id}
                streaming={s.id === streamingId}
                onSelect={() => selectSession(s.id)}
                onRemove={() => {
                  const id = newId();
                  setList((l) => removeSession(l, s.id, id));
                }}
              />
            ))}
          </ul>
        </aside>

        <main className="flex min-w-0 flex-1 flex-col">
          <div
            ref={transcript}
            onScroll={(e) => (following.current = isAtBottom(e.currentTarget))}
            className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 py-6 md:px-10"
          >
            {active.messages.length === 0 ? (
              <EmptyState model={model} />
            ) : (
              <div className="mx-auto flex w-full max-w-3xl flex-col gap-7">
                {active.messages.map((m, i) => {
                  // A tool result shows through the agent card of the call it answers.
                  if (m.role === "tool") return null;
                  const actions: TurnActions = {
                    canRerun: !busy && model.state === "ready",
                    onSave: (text) => saveEdit(m.id, text),
                    onFork: () => fork(m.id),
                  };
                  // Keyed by session too: a fork shares message ids with its source.
                  const key = `${active.id}:${m.id}`;
                  return m.role === "user" ? (
                    <UserTurn key={key} message={m} actions={actions} onResend={(text) => rerun(m.id, text)} />
                  ) : (
                    <Reply
                      key={key}
                      message={m}
                      markdown={markdown}
                      last={i === active.messages.length - 1}
                      actions={actions}
                      onRegenerate={() => rerun(m.id, null)}
                      openCallId={reader?.messageId === m.id ? reader.callId : null}
                      onOpenAgent={(callId) => setReader({ messageId: m.id, callId })}
                    />
                  );
                })}
              </div>
            )}
          </div>

          <div className="shrink-0 px-4 pb-4 md:px-10">
            <div className="cut mx-auto flex w-full max-w-3xl items-end gap-2 bg-surface p-2 shadow-[inset_0_-2px_0_var(--line)] [--cut-size:14px] focus-within:shadow-[inset_0_-2px_0_var(--ember)]">
              <textarea
                className="max-h-60 min-h-[3.25rem] flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-normal text-ink [field-sizing:content] placeholder:text-ash focus:outline-none"
                rows={2}
                aria-label="Prompt"
                value={input}
                placeholder={model.state === "ready" ? "Ask anything" : "Waiting for the model…"}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
              />
              {streamingHere ? (
                <button
                  type="button"
                  className="cut bg-[#c8161d] px-5 py-2.5 font-display text-sm font-semibold text-white hover:bg-[#a8121a]"
                  onClick={() => controller.current?.abort()}
                >
                  Stop
                </button>
              ) : (
                <button
                  type="button"
                  className="cut bg-[#ff5a1f] px-5 py-2.5 font-display text-sm font-semibold text-[#1c2026] hover:bg-[#ff7a45] disabled:cursor-default disabled:bg-line disabled:text-ash"
                  disabled={busy || !input.trim() || model.state !== "ready"}
                  onClick={() => void send()}
                >
                  Send
                </button>
              )}
            </div>
            <div className="mx-auto mt-2 flex w-full max-w-3xl items-center justify-between gap-4 px-1">
              <p className="min-w-0 text-xs text-ash">
                {busy && !streamingHere
                  ? "A reply is streaming in another session. Send works again once it ends."
                  : "Enter sends. Shift+Enter starts a new line."}
              </p>
              <ContextMeter
                usage={contextUsage(active.log, settings.maxTokens, model.state === "ready" ? model.contextLimit : null)}
              />
            </div>
          </div>
        </main>

        <aside
          aria-label="Settings"
          className={`fixed inset-y-0 right-0 z-40 flex w-72 flex-col gap-6 overflow-y-auto border-l border-line bg-ground px-5 py-5 transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-68 lg:translate-x-0 ${drawer === "settings" ? "translate-x-0" : "max-lg:invisible max-lg:translate-x-full"}`}
        >
          <SystemPromptField value={settings.systemPrompt} onChange={(v) => set("systemPrompt", v)} ignis={ignisPrompt(tools)} />

          <Segmented
            legend="Thinking"
            name="effort"
            value={settings.reasoningEffort}
            options={REASONING_EFFORTS.map((effort) => ({ value: effort, label: EFFORT_LABELS[effort] }))}
            onChange={(v) => set("reasoningEffort", v)}
          />

          <div className="flex flex-col gap-4">
            <Slider label="Temperature" min={0} max={2} step={0.05} value={settings.temperature} onChange={(v) => set("temperature", v)} />
            <Slider label="top_p" min={0} max={1} step={0.05} value={settings.topP} onChange={(v) => set("topP", v)} />
            {greedyConflict && (
              <p className="border-l-2 border-fault pl-2 text-xs leading-snug text-fault">
                Temperature 0 needs top_p at 1, or ignis answers 400.
              </p>
            )}
          </div>

          <label className="flex flex-col gap-2">
            <span className={caption}>Max tokens</span>
            <input
              className={`${field} font-display tabular-nums`}
              type="number"
              min={1}
              placeholder="Engine cap"
              value={settings.maxTokens ?? ""}
              onChange={(e) => set("maxTokens", e.target.value === "" ? null : Number(e.target.value))}
            />
          </label>

          <Segmented
            legend="Lane tag"
            name="lane"
            value={settings.laneTag}
            options={[
              { value: "interactive", label: "Interactive" },
              { value: "agent", label: "Agent" },
            ]}
            onChange={(v) => set("laneTag", v)}
          />

          <ToolsSetting tools={tools} onChange={setTools} />

          <MarkdownSetting on={markdown} onChange={setMarkdown} />
        </aside>
      </div>

      <SessionLog rows={active.log} open={logOpen} onToggle={() => setLogOpen((o) => !o)} />

      {readerRun && (
        <>
          <div className="fixed inset-0 z-40 bg-[#1c2026]/40" onClick={closeReader} aria-hidden />
          <AgentReader
            key={readerRun.callId}
            run={readerRun}
            markdown={markdown}
            systemPrompt={AGENT_SYSTEM_PROMPT}
            figures={readerRun.figures && <Readout figures={readerRun.figures} />}
            onClose={closeReader}
          />
        </>
      )}
    </div>
  );
}

function HeaderButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className="grid size-9 shrink-0 place-items-center text-[#b9bec4] hover:bg-kiln-line hover:text-white lg:hidden"
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function ModelStatus({ model, busy }: { model: ModelState; busy: boolean }) {
  return (
    <span className="ml-auto flex min-w-0 items-center gap-2 font-display text-sm">
      {model.state === "ready" && (
        <>
          <span className={`cut size-2.5 shrink-0 [--cut-size:4px] ${busy ? "bg-[#ff5a1f]" : "bg-[#3ecf8e]"}`} aria-hidden />
          <span className="hidden truncate text-[#eae8e4] sm:inline">{model.id}</span>
          <span className="sr-only">{busy ? "generating" : "ready"}</span>
        </>
      )}
      {model.state === "loading" && <span className="text-[#939ba4]">Loading model…</span>}
      {model.state === "error" && <span className="truncate text-[#ff7b6b]">{model.message}</span>}
    </span>
  );
}

function SessionItem(props: {
  session: Session;
  active: boolean;
  streaming: boolean;
  onSelect: () => void;
  onRemove: () => void;
}) {
  const { session: s, active, streaming } = props;
  const replies = s.log.filter((row) => row.agent === undefined).length;
  return (
    <li
      className={`group relative flex items-stretch ${active ? "bg-surface shadow-[inset_2px_0_0_var(--ember)]" : "hover:bg-surface/60"}`}
    >
      <button
        type="button"
        aria-current={active ? "true" : undefined}
        className="flex min-w-0 flex-1 flex-col gap-0.5 px-3 py-2 text-left"
        onClick={props.onSelect}
      >
        <span className={`truncate text-sm ${active ? "text-ink" : "text-ink/85"}`}>{s.title}</span>
        <span className="flex items-center gap-1.5 font-display text-xs text-ash">
          {streaming && <span className="cut size-2 bg-ember [--cut-size:3px]" aria-hidden />}
          {streaming ? "Streaming" : replies === 0 ? "No replies yet" : replies === 1 ? "1 reply" : `${replies} replies`}
        </span>
      </button>
      <button
        type="button"
        aria-label={`Delete session: ${s.title}`}
        title="Delete session"
        disabled={streaming}
        className={`grid w-8 shrink-0 place-items-center text-ash hover:text-fault focus-visible:opacity-100 disabled:hidden ${active ? "" : "opacity-0 group-hover:opacity-100 max-lg:opacity-100"}`}
        onClick={props.onRemove}
      >
        <IconClose />
      </button>
    </li>
  );
}

function EmptyState({ model }: { model: ModelState }) {
  return (
    <div className="m-auto flex max-w-md flex-col items-center gap-7 py-10 text-center">
      <img src={flame} alt="" className="h-44 w-auto drop-shadow-[0_18px_40px_rgb(200_22_29/0.35)]" />
      <div className="flex flex-col gap-2">
        <p className="font-display text-4xl font-semibold tracking-tight text-ink">
          {model.state === "ready" ? "Light it up" : model.state === "loading" ? "Warming up" : "No model to talk to"}
        </p>
        <p className="text-sm text-ash">
          {model.state === "ready" && (
            <>
              Send a prompt to <span className="font-display text-ink">{model.id}</span>. Timings for each reply show up
              under it and in this session's log below.
            </>
          )}
          {model.state === "loading" && "Asking ignis which model it serves."}
          {model.state === "error" && "Start ignis-server, then reload this page."}
        </p>
      </div>
    </div>
  );
}

/** What every turn can do; `canRerun` is false while any reply streams or without a model. */
type TurnActions = {
  canRerun: boolean;
  onSave: (text: string) => void;
  onFork: () => void;
};

const RERUN_BLOCKED = "Wait for the current reply to end";

function UserTurn({ message: m, actions, onResend }: { message: Message; actions: TurnActions; onResend: (text: string) => void }) {
  const [editing, setEditing] = useState(false);
  if (editing) {
    return (
      <MessageEditor
        initial={m.content}
        onCancel={() => setEditing(false)}
        buttons={[
          {
            label: "Save",
            onClick: (text) => {
              actions.onSave(text);
              setEditing(false);
            },
          },
          {
            label: "Save and resend",
            primary: true,
            disabled: !actions.canRerun,
            title: actions.canRerun ? "Drops everything after this prompt and sends it again" : RERUN_BLOCKED,
            onClick: (text) => {
              setEditing(false);
              onResend(text);
            },
          },
        ]}
      />
    );
  }
  return (
    <div className="group/turn flex flex-col items-end gap-1 self-end md:max-w-[85%]">
      <article className="cut bg-surface px-4 py-3 [--cut-size:12px]">
        <pre className="whitespace-pre-wrap break-words font-sans text-[15px] leading-normal">{m.content}</pre>
      </article>
      <ActionRow className="-mr-2">
        {m.edited && <EditedMark />}
        <ActionButton icon={<IconPencil />} label="Edit" onClick={() => setEditing(true)} />
        <ActionButton icon={<IconFork />} label="Fork" title="Start a new session from this message" onClick={actions.onFork} />
      </ActionRow>
    </div>
  );
}

function Reply(props: {
  message: Message;
  markdown: boolean;
  last: boolean;
  actions: TurnActions;
  onRegenerate: () => void;
  openCallId: string | null;
  onOpenAgent: (callId: string) => void;
}) {
  const { message: m, markdown, actions } = props;
  const [editing, setEditing] = useState(false);
  const thinking = m.streaming && !m.content;
  return (
    <article className="group/turn relative flex flex-col gap-3">
      {/* The i-dot of the wordmark marks what ignis wrote. */}
      <span className="cut absolute top-[0.4em] -left-6 hidden size-2.5 bg-ember [--cut-size:4px] md:block" aria-hidden />
      {m.reasoning && (
        <details className="reasoning" open={thinking}>
          <summary className="cursor-pointer select-none font-display text-[13px] font-medium text-ash hover:text-ink">
            {thinking ? "Thinking…" : "Reasoning"}
          </summary>
          <pre className="mt-2 max-h-80 overflow-y-auto border-l-2 border-line pl-3 whitespace-pre-wrap break-words font-sans text-[13px] leading-normal text-ash">
            {m.reasoning}
          </pre>
        </details>
      )}
      {editing ? (
        <MessageEditor
          initial={m.content}
          onCancel={() => setEditing(false)}
          buttons={[
            {
              label: "Save",
              primary: true,
              onClick: (text) => {
                actions.onSave(text);
                setEditing(false);
              },
            },
          ]}
        />
      ) : (
        (m.content || (m.streaming && !m.reasoning)) &&
        (markdown && m.content ? (
          <Markdown text={m.content} streaming={m.streaming} />
        ) : (
          <pre className="whitespace-pre-wrap break-words font-sans text-[15px] leading-normal">
            {m.content}
            {m.streaming && <span className="caret" aria-hidden />}
          </pre>
        ))
      )}
      {m.agents && m.agents.length > 0 && (
        <AgentStrip runs={m.agents} openCallId={props.openCallId} onOpen={props.onOpenAgent} />
      )}
      {m.error && (
        <p className="border-l-2 border-fault pl-3 text-sm text-fault" role="alert">
          {m.error}
        </p>
      )}
      {m.figures && <Readout figures={m.figures} />}
      {!m.streaming && !editing && (
        <ActionRow className="-mt-1 -ml-2" pinned={props.last}>
          <ActionButton
            icon={<IconRegenerate />}
            label="Regenerate"
            disabled={!actions.canRerun}
            title={actions.canRerun ? "Drops this reply and what follows, then answers the same prompt again" : RERUN_BLOCKED}
            onClick={props.onRegenerate}
          />
          <ActionButton icon={<IconPencil />} label="Edit" onClick={() => setEditing(true)} />
          <ActionButton icon={<IconFork />} label="Fork" title="Start a new session from this reply" onClick={actions.onFork} />
          {m.edited && <EditedMark />}
        </ActionRow>
      )}
    </article>
  );
}

/** A turn's actions: shown on hover or focus on wide screens, always on the last reply and on touch widths. */
function ActionRow({ children, className = "", pinned = false }: { children: ReactNode; className?: string; pinned?: boolean }) {
  return (
    <div
      className={`flex flex-wrap items-center gap-0.5 transition-opacity motion-reduce:transition-none ${pinned ? "" : "lg:opacity-0 lg:group-focus-within/turn:opacity-100 lg:group-hover/turn:opacity-100"} ${className}`}
    >
      {children}
    </div>
  );
}

function ActionButton(props: { icon: ReactNode; label: string; onClick: () => void; disabled?: boolean; title?: string }) {
  return (
    <button
      type="button"
      title={props.title ?? props.label}
      disabled={props.disabled}
      onClick={props.onClick}
      className="flex items-center gap-1.5 px-2 py-1 font-display text-xs font-medium text-ash hover:bg-surface hover:text-ink disabled:cursor-default disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-ash"
    >
      {props.icon}
      {props.label}
    </button>
  );
}

function EditedMark() {
  return (
    <span className="px-2 font-display text-xs text-ash" title="Changed by hand; any figures measure the original">
      Edited
    </span>
  );
}

type EditorButton = { label: string; primary?: boolean; disabled?: boolean; title?: string; onClick: (text: string) => void };

/** In-place message editor. Esc cancels; Ctrl+Enter runs the primary button. */
function MessageEditor({ initial, buttons, onCancel }: { initial: string; buttons: EditorButton[]; onCancel: () => void }) {
  const [draft, setDraft] = useState(initial);
  const empty = draft.trim() === "";
  const primary = buttons.find((b) => b.primary);

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && primary && !primary.disabled && !empty) {
      e.preventDefault();
      primary.onClick(draft);
    }
  }

  return (
    <div className="cut flex w-full flex-col gap-2 bg-surface p-2 shadow-[inset_0_-2px_0_var(--ember)] [--cut-size:12px]">
      <textarea
        autoFocus
        aria-label="Edit message"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKeyDown}
        onFocus={(e) => e.currentTarget.setSelectionRange(e.currentTarget.value.length, e.currentTarget.value.length)}
        className="max-h-[60vh] min-h-24 w-full resize-none bg-transparent px-2 py-1.5 text-[15px] leading-normal text-ink [field-sizing:content] focus:outline-none"
      />
      <div className="flex flex-wrap items-center justify-end gap-2">
        {primary && (
          <span className="mr-auto px-2 text-xs text-ash">
            Esc to cancel, Ctrl+Enter to {primary.label.toLowerCase()}.
          </span>
        )}
        <button type="button" className="px-3 py-1.5 font-display text-sm font-medium text-ash hover:text-ink" onClick={onCancel}>
          Cancel
        </button>
        {buttons.map((b) => (
          <button
            key={b.label}
            type="button"
            title={b.title}
            disabled={b.disabled || empty}
            onClick={() => b.onClick(draft)}
            className={`cut px-4 py-1.5 font-display text-sm disabled:cursor-default disabled:bg-line disabled:text-ash ${
              b.primary
                ? "bg-[#ff5a1f] font-semibold text-[#1c2026] hover:bg-[#ff7a45]"
                : "bg-ground font-medium text-ink hover:bg-line"
            }`}
          >
            {b.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Readout({ figures }: { figures: Figures }) {
  const d = describeFigures(figures);
  const items: [string, string, boolean?][] = [
    ["TTFT", d.ttft],
    ["Decode", d.decode, true],
    ["Total", d.duration],
    ["Tokens", `${d.promptTokens} in, ${d.completionTokens} out`],
    ["Finish", d.finish],
  ];
  return (
    <dl className="flex flex-wrap items-baseline gap-x-5 gap-y-1 border-t border-line pt-2 font-display text-xs">
      {items.map(([k, v, hot]) => (
        <div key={k} className="flex items-baseline gap-1.5">
          <dt className="text-ash">{k}</dt>
          <dd className={`tabular-nums ${hot ? "font-semibold text-ember" : "text-ink"}`}>{v}</dd>
        </div>
      ))}
      <div className="ml-auto text-ash" title="Measured by the browser around the HTTP stream, not engine-internal timings">
        HTTP-observed
      </div>
    </dl>
  );
}

/** The system prompt in two tabs: the owner's, editable, and the ignis one the enabled tools write, read-only. */
function SystemPromptField({ value, onChange, ignis }: { value: string; onChange: (value: string) => void; ignis: string }) {
  const [tab, setTab] = useState<"user" | "ignis">("user");
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-end justify-between gap-2">
        <span id="system-prompt-label" className={caption}>
          System prompt
        </span>
        <div role="tablist" aria-labelledby="system-prompt-label" className="flex gap-3">
          {(["user", "ignis"] as const).map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              id={`system-tab-${t}`}
              aria-selected={tab === t}
              aria-controls="system-prompt-panel"
              onClick={() => setTab(t)}
              className={`flex items-center gap-1.5 border-b-2 pb-0.5 font-display text-[13px] font-medium ${tab === t ? "border-ember text-ink" : "border-transparent text-ash hover:text-ink"}`}
            >
              {t === "user" ? "User" : "ignis"}
              {t === "ignis" && ignis && (
                <>
                  <span className="cut size-1.5 bg-ember [--cut-size:2px]" aria-hidden />
                  <span className="sr-only">(tools add text here)</span>
                </>
              )}
            </button>
          ))}
        </div>
      </div>
      <div id="system-prompt-panel" role="tabpanel" aria-labelledby={`system-tab-${tab}`}>
        {tab === "user" ? (
          <textarea
            className={`${field} resize-y leading-relaxed`}
            rows={5}
            placeholder="None"
            aria-label="Your system prompt"
            value={value}
            onChange={(e) => onChange(e.target.value)}
          />
        ) : ignis ? (
          <textarea
            readOnly
            className={`${field} resize-y border-dashed bg-ground text-xs leading-relaxed text-ash focus:border-line`}
            rows={10}
            aria-label="The ignis system prompt, read-only"
            value={ignis}
          />
        ) : (
          <p className="border border-dashed border-line px-2.5 py-2 text-xs leading-relaxed text-ash">
            Empty. Turn on a tool and ignis writes its instructions here, sent ahead of yours.
          </p>
        )}
      </div>
    </div>
  );
}

function ToolsSetting({ tools, onChange }: { tools: ToolsState; onChange: (tools: ToolsState) => void }) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className={`${caption} mb-2`}>Tools</legend>
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <span id="tool-agents-label" className="font-display text-sm font-semibold text-ink">
            Agents
          </span>
          <span className="text-xs leading-snug text-ash">
            The model can hand sub-tasks to agents that run in parallel on agent lanes.
          </span>
        </div>
        <Switch on={tools.agents} onChange={(agents) => onChange({ ...tools, agents })} labelledBy="tool-agents-label" />
      </div>
    </fieldset>
  );
}

/** The Markdown switch: replies formatted, or their raw text. */
function MarkdownSetting({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  return (
    <div className="mt-auto border-t border-line pt-5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <span id="markdown-label" className="font-display text-sm font-semibold text-ink">
            Markdown in replies
          </span>
          <span className="text-xs leading-snug text-ash">Headings, lists, tables and code, formatted.</span>
        </div>
        <Switch on={on} onChange={onChange} labelledBy="markdown-label" />
      </div>
    </div>
  );
}

function Switch({ on, onChange, labelledBy }: { on: boolean; onChange: (on: boolean) => void; labelledBy: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-labelledby={labelledBy}
      className="mt-0.5 shrink-0 cursor-pointer"
      onClick={() => onChange(!on)}
    >
      <span
        className={`cut relative block h-6 w-11 transition-colors [--cut-size:7px] motion-reduce:transition-none ${on ? "bg-[#ff5a1f]" : "bg-line"}`}
      >
        <span
          className={`cut absolute top-1 left-1 size-4 transition-transform duration-200 [--cut-size:5px] motion-reduce:transition-none ${on ? "translate-x-5 bg-[#1c2026]" : "bg-ink"}`}
        />
      </span>
    </button>
  );
}

function Segmented<T extends string>(props: {
  legend: string;
  name: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className={`${caption} mb-2`}>{props.legend}</legend>
      <div className="flex border border-line bg-surface p-0.5">
        {props.options.map((o) => (
          <label key={o.value} className="relative flex-1">
            <input
              type="radio"
              name={props.name}
              className="peer sr-only"
              checked={props.value === o.value}
              onChange={() => props.onChange(o.value)}
            />
            <span className="block cursor-pointer px-1 py-1.5 text-center font-display text-[13px] font-medium text-ash peer-checked:bg-ink peer-checked:text-ground peer-focus-visible:outline-2 peer-focus-visible:outline-ember hover:text-ink peer-checked:hover:text-ground">
              {o.label}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function Slider(props: { label: string; min: number; max: number; step: number; value: number; onChange: (value: number) => void }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex items-baseline justify-between">
        <span className={caption}>{props.label}</span>
        <output className="font-display text-sm font-medium tabular-nums text-ink">{props.value.toFixed(2)}</output>
      </span>
      <input
        type="range"
        className="range w-full"
        style={{ "--fill": trackFill(props.value, props.min, props.max) } as CSSProperties}
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

function SessionLog({ rows, open, onToggle }: { rows: LogRow[]; open: boolean; onToggle: () => void }) {
  const numeric = ["TTFT", "Decode", "Total", "Prompt", "Completion", "Finish"];
  const replies = rows.filter((row) => row.agent === undefined);
  const agentRows = rows.length - replies.length;
  const last = replies.at(-1)?.figures;
  const count = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  return (
    <section className="z-10 shrink-0 border-t border-line bg-ground" aria-label="This session">
      <div className="flex items-center gap-x-3 px-4 py-2.5 md:px-6">
        <h2 className="font-display text-sm font-semibold">This session</h2>
        <p className="min-w-0 truncate text-xs text-ash">
          {rows.length === 0
            ? "HTTP-observed figures for each reply collect here."
            : `${count(replies.length, "reply", "replies")}${agentRows ? `, ${count(agentRows, "agent request", "agent requests")}` : ""}${last ? `, last decode ${describeFigures(last).decode}` : ""}. HTTP-observed: measured by the browser, not the engine.`}
        </p>
        <button
          type="button"
          aria-expanded={open}
          aria-controls="session-log"
          aria-label={open ? "Collapse this session's log" : "Expand this session's log"}
          title={open ? "Collapse" : "Expand"}
          className="ml-auto grid size-8 shrink-0 place-items-center text-ash hover:bg-surface hover:text-ink"
          onClick={onToggle}
        >
          <IconChevron className={`transition-transform motion-reduce:transition-none ${open ? "rotate-180" : ""}`} />
        </button>
      </div>
      <div
        id="session-log"
        className={`grid transition-[grid-template-rows] duration-200 motion-reduce:transition-none ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}
        inert={!open}
      >
        <div className="min-h-0 overflow-hidden">
          {/* Open, the table scrolls within a quarter of the window, under the title row. */}
          <div className="max-h-[25dvh] overflow-auto px-4 pb-4 md:px-6">
            {rows.length === 0 ? (
              <p className="py-2 text-sm text-ash">Each reply in this session adds a row here.</p>
            ) : (
              <table className="w-full border-collapse font-display text-[13px] tabular-nums">
                <thead className="sticky top-0 bg-ground text-ash">
                  <tr className="border-b border-line">
                    {["#", "Time", "Lane tag", "Thinking"].map((h) => (
                      <Th key={h}>{h}</Th>
                    ))}
                    {numeric.map((h) => (
                      <Th key={h} right>
                        {h}
                      </Th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.n} className="border-b border-line/60 last:border-0 hover:bg-surface">
                      <td className="px-3 py-1.5 text-ash">{row.n}</td>
                      <td className="px-3 py-1.5 whitespace-nowrap">{row.at}</td>
                      <td className="px-3 py-1.5 whitespace-nowrap">
                        {row.agent ? (
                          <>
                            agent <span className="text-ash">{row.agent}</span>
                          </>
                        ) : (
                          row.laneTag
                        )}
                      </td>
                      <td className="px-3 py-1.5">{EFFORT_LABELS[row.reasoningEffort]}</td>
                      {row.figures ? (
                        <FigureCells figures={row.figures} />
                      ) : (
                        <td colSpan={6} className="px-3 py-1.5 text-right text-fault">
                          {row.error}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function ContextMeter({ usage: u }: { usage: ContextUsage }) {
  const n = (v: number) => v.toLocaleString("en-US");
  const label = u.limit === null ? `${compactTokens(u.used)} used` : `${compactTokens(u.used)} / ${compactTokens(u.limit)}`;
  const free = u.limit === null ? null : u.limit - u.used - (u.reserve ?? 0);
  const rows: [string, string, boolean?][] = [
    ["Context", u.limit === null ? "Not reported by ignis" : `${n(u.limit)} tokens`],
    ["Last prompt", n(u.promptTokens)],
    ["Last completion", n(u.completionTokens)],
    ["In use", u.limit === null ? n(u.used) : `${n(u.used)} (${Math.round((u.used / u.limit) * 100)}%)`],
    ["Next reply reserves", u.reserve === null ? "Engine cap" : n(u.reserve)],
    ...(free === null ? [] : [["Left after that", n(free), free < 0] as [string, string, boolean]]),
  ];
  return (
    <div className="group relative shrink-0">
      <button
        type="button"
        aria-describedby="context-details"
        className="flex items-center gap-2.5 py-1 font-display text-xs tabular-nums text-ash hover:text-ink"
      >
        <span>Context</span>
        <span className="relative flex h-1.5 w-28 overflow-hidden bg-line sm:w-40" aria-hidden>
          {u.usedShare !== null && <span className="h-full bg-[#ff5a1f]" style={{ width: `${u.usedShare * 100}%` }} />}
          {u.reserveShare !== null && (
            <span
              className={`h-full ${u.overflows ? "bg-fault" : "bg-[#ff5a1f]/35"}`}
              style={{ width: `${u.reserveShare * 100}%` }}
            />
          )}
        </span>
        <span className={u.overflows ? "text-fault" : "text-ink"}>{label}</span>
      </button>
      <div
        id="context-details"
        role="tooltip"
        className="cut invisible absolute right-0 bottom-full z-30 mb-2 w-72 bg-kiln p-4 text-[#eae8e4] opacity-0 shadow-[0_12px_32px_rgb(0_0_0/0.35)] transition-opacity [--cut-size:12px] group-focus-within:visible group-focus-within:opacity-100 group-hover:visible group-hover:opacity-100 motion-reduce:transition-none"
      >
        <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 font-display text-[13px]">
          {rows.map(([k, v, bad]) => (
            <div key={k} className="contents">
              <dt className="text-[#939ba4]">{k}</dt>
              <dd className={`text-right tabular-nums ${bad ? "text-[#ff7b6b]" : ""}`}>{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 border-t border-kiln-line pt-3 text-xs leading-relaxed text-[#939ba4]">
          {u.overflows
            ? "The next request is over the context, so ignis will refuse it. Lower Max tokens or start a new session."
            : "From the last reply's usage in this session. Reasoning is not sent back, so the next prompt can be shorter."}
        </p>
      </div>
    </div>
  );
}

function Th({ children, right }: { children: ReactNode; right?: boolean }) {
  return <th className={`px-3 py-2 font-medium whitespace-nowrap ${right ? "text-right" : "text-left"}`}>{children}</th>;
}

function FigureCells({ figures }: { figures: Figures }) {
  const d = describeFigures(figures);
  return (
    <>
      {[d.ttft, d.decode, d.duration, d.promptTokens, d.completionTokens, d.finish].map((value, i) => (
        <td key={i} className={`px-3 py-1.5 text-right whitespace-nowrap ${i === 1 ? "font-semibold text-ember" : ""}`}>
          {value}
        </td>
      ))}
    </>
  );
}

const icon = { width: 16, height: 16, viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", strokeWidth: 1.5, "aria-hidden": true } as const;

function IconPencil() {
  return (
    <svg {...icon} width={13} height={13}>
      <path d="M10.5 2.5l3 3L6 13H3v-3z" />
    </svg>
  );
}

function IconFork() {
  return (
    <svg {...icon} width={13} height={13}>
      <circle cx="4" cy="3.5" r="1.5" />
      <circle cx="12" cy="3.5" r="1.5" />
      <circle cx="8" cy="12.5" r="1.5" />
      <path d="M4 5v1c0 1.7 1.3 3 3 3h2c1.7 0 3-1.3 3-3V5M8 9v2" />
    </svg>
  );
}

function IconRegenerate() {
  return (
    <svg {...icon} width={13} height={13}>
      <path d="M13 8a5 5 0 1 1-1.46-3.54M13.5 2v3h-3" />
    </svg>
  );
}

function IconPlus() {
  return (
    <svg {...icon}>
      <path d="M8 3v10M3 8h10" />
    </svg>
  );
}

function IconClose() {
  return (
    <svg {...icon} width={14} height={14}>
      <path d="M4 4l8 8M12 4l-8 8" />
    </svg>
  );
}

function IconChevron({ className }: { className?: string }) {
  return (
    <svg {...icon} className={className}>
      <path d="M4 10l4-4 4 4" />
    </svg>
  );
}

function IconSessions() {
  return (
    <svg {...icon} width={18} height={18}>
      <path d="M2.5 4h11M2.5 8h11M2.5 12h7" />
    </svg>
  );
}

function IconSliders() {
  return (
    <svg {...icon} width={18} height={18}>
      <path d="M2.5 4.5h6M11.5 4.5h2M2.5 11.5h2M7.5 11.5h6" />
      <path d="M8.5 3v3M4.5 10v3" />
    </svg>
  );
}
