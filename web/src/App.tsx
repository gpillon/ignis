import { type KeyboardEvent, useEffect, useLayoutEffect, useRef, useState } from "react";
import { computeFigures, describeFigures, type Figures } from "./figures.ts";
import {
  buildChatRequest,
  conversationTurns,
  type LaneTag,
  REASONING_EFFORTS,
  type ReasoningEffort,
  type Settings,
} from "./request.ts";
import { isAtBottom } from "./scroll.ts";
import { streamChat } from "./stream.ts";

// The Playground (GitHub #164): a streaming chat against ignis's own
// /v1/chat/completions, with per-request figures measured in the browser.
// Everything lives in memory; a reload starts over.

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  reasoning: string;
  streaming: boolean;
  figures?: Figures;
  error?: string;
};

type SessionRow = {
  n: number;
  at: string;
  laneTag: LaneTag;
  reasoningEffort: ReasoningEffort;
  figures: Figures | null;
  error?: string;
};

type ModelState = { state: "loading" } | { state: "ready"; id: string } | { state: "error"; message: string };

const DEFAULT_SETTINGS: Omit<Settings, "model"> = {
  systemPrompt: "",
  temperature: 0.7,
  topP: 0.95,
  maxTokens: 1024,
  reasoningEffort: "medium",
  laneTag: "interactive",
};

const panel = "rounded-lg border border-stone-200 bg-white p-3 dark:border-stone-800 dark:bg-stone-900";
const field =
  "w-full rounded-md border border-stone-300 bg-stone-50 px-2 py-1.5 text-sm text-stone-900 focus:border-orange-600 focus:outline-none dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100";
const label = "flex flex-col gap-1 text-xs font-medium text-stone-500 dark:text-stone-400";
const button = "rounded-md px-3 py-1.5 text-sm font-medium disabled:cursor-default disabled:opacity-50";

let nextId = 1;

export function App() {
  const [model, setModel] = useState<ModelState>({ state: "loading" });
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [messages, setMessages] = useState<Message[]>([]);
  const [session, setSession] = useState<SessionRow[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const list = useRef<HTMLDivElement>(null);
  // Whether the reader is at the bottom of the conversation: only then do
  // new tokens pull the view down.
  const following = useRef(true);

  useEffect(() => {
    fetch("/v1/models")
      .then(async (res) => {
        if (!res.ok) throw new Error(`GET /v1/models: ${res.status}`);
        const body = (await res.json()) as { data?: { id: string }[] };
        const id = body.data?.[0]?.id;
        if (!id) throw new Error("GET /v1/models: no model listed");
        setModel({ state: "ready", id });
      })
      .catch((err: unknown) => setModel({ state: "error", message: String(err) }));
  }, []);

  useLayoutEffect(() => {
    const el = list.current;
    if (el && following.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const update = (id: number, change: (m: Message) => Message) =>
    setMessages((all) => all.map((m) => (m.id === id ? change(m) : m)));

  async function send() {
    const text = input.trim();
    if (!text || busy || model.state !== "ready") return;

    const turns = conversationTurns([
      ...messages.map((m) => ({ role: m.role, content: m.content, failed: m.error !== undefined })),
      { role: "user", content: text, failed: false },
    ]);
    const request = buildChatRequest({ ...settings, model: model.id }, turns);
    const user: Message = { id: nextId++, role: "user", content: text, reasoning: "", streaming: false };
    const reply: Message = { id: nextId++, role: "assistant", content: "", reasoning: "", streaming: true };
    // Sending is a request to see the answer: follow it from the bottom.
    following.current = true;
    setMessages((all) => [...all, user, reply]);
    setInput("");
    setBusy(true);

    const abort = new AbortController();
    controller.current = abort;
    const result = await streamChat({
      body: request,
      signal: abort.signal,
      onEvent: (event) => {
        if (event.kind === "reasoning") update(reply.id, (m) => ({ ...m, reasoning: m.reasoning + event.text }));
        if (event.kind === "content") update(reply.id, (m) => ({ ...m, content: m.content + event.text }));
      },
    });
    controller.current = null;
    setBusy(false);

    const figures = result.ok ? computeFigures(result.timeline) : null;
    const error = result.ok ? undefined : result.message;
    update(reply.id, (m) => ({ ...m, streaming: false, figures: figures ?? undefined, error }));
    setSession((rows) => [
      ...rows,
      {
        n: rows.length + 1,
        at: new Date().toLocaleTimeString(),
        laneTag: request.class,
        reasoningEffort: request.reasoning_effort,
        figures,
        error,
      },
    ]);
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

  return (
    <div className="mx-auto grid max-w-7xl gap-3 px-4 py-3 text-sm md:grid-cols-[16rem_minmax(0,1fr)]">
      <header className="flex items-baseline gap-3 md:col-span-2">
        <h1 className="text-lg font-semibold">Playground</h1>
        <span className="text-stone-500 dark:text-stone-400">
          {model.state === "loading" && "loading model…"}
          {model.state === "ready" && <code className="font-mono">{model.id}</code>}
          {model.state === "error" && <span className="text-red-700 dark:text-red-400">{model.message}</span>}
        </span>
      </header>

      <aside className={`${panel} order-2 flex flex-col gap-3 self-start md:order-none`}>
        <label className={label}>
          System prompt
          <textarea className={`${field} resize-y`} rows={4} value={settings.systemPrompt} onChange={(e) => set("systemPrompt", e.target.value)} />
        </label>
        <label className={label}>
          Thinking effort
          <select className={field} value={settings.reasoningEffort} onChange={(e) => set("reasoningEffort", e.target.value as ReasoningEffort)}>
            {REASONING_EFFORTS.map((effort) => (
              <option key={effort} value={effort}>
                {effort === "none" ? "none (thinking off)" : effort}
              </option>
            ))}
          </select>
        </label>
        <div className="grid grid-cols-2 gap-2">
          <label className={label}>
            temperature
            <input className={field} type="number" min={0} max={2} step={0.05} value={settings.temperature} onChange={(e) => set("temperature", Number(e.target.value))} />
          </label>
          <label className={label}>
            top_p
            <input className={field} type="number" min={0} max={1} step={0.05} value={settings.topP} onChange={(e) => set("topP", Number(e.target.value))} />
          </label>
        </div>
        {greedyConflict && <p className="text-xs text-red-700 dark:text-red-400">temperature 0 needs top_p 1, or ignis answers 400.</p>}
        <label className={label}>
          max_tokens
          <input
            className={field}
            type="number"
            min={1}
            placeholder="engine cap"
            value={settings.maxTokens ?? ""}
            onChange={(e) => set("maxTokens", e.target.value === "" ? null : Number(e.target.value))}
          />
        </label>
        <label className={label}>
          Lane tag
          <select className={field} value={settings.laneTag} onChange={(e) => set("laneTag", e.target.value as LaneTag)}>
            <option value="interactive">Interactive</option>
            <option value="agent">Agent</option>
          </select>
        </label>
        <button
          type="button"
          className={`${button} border border-stone-300 hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800`}
          disabled={busy || messages.length === 0}
          onClick={() => setMessages([])}
        >
          New conversation
        </button>
      </aside>

      <main className={`${panel} flex min-h-[60vh] min-w-0 flex-col`}>
        <div
          ref={list}
          onScroll={(e) => (following.current = isAtBottom(e.currentTarget))}
          className="flex max-h-[65vh] flex-1 flex-col gap-3 overflow-y-auto pr-1"
        >
          {messages.length === 0 && (
            <p className="text-stone-500 dark:text-stone-400">Send a prompt to start. Enter sends, Shift+Enter adds a line.</p>
          )}
          {messages.map((m) => (
            <article key={m.id} className={`rounded-md px-3 py-2 ${m.role === "user" ? "bg-slate-100 dark:bg-slate-800" : ""}`}>
              {m.reasoning && (
                <details className="mb-2 text-stone-500 dark:text-stone-400" open={m.streaming && !m.content}>
                  <summary className="cursor-pointer select-none text-xs font-medium">reasoning</summary>
                  <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-xs">{m.reasoning}</pre>
                </details>
              )}
              {(m.content || m.role === "user") && <pre className="whitespace-pre-wrap break-words font-sans">{m.content}</pre>}
              {m.streaming && !m.content && !m.reasoning && <p className="text-stone-500 dark:text-stone-400">waiting for the first token…</p>}
              {m.error && (
                <p className="text-red-700 dark:text-red-400" role="alert">
                  {m.error}
                </p>
              )}
              {m.figures && <FigureLine figures={m.figures} />}
            </article>
          ))}
        </div>
        <div className="mt-3 flex gap-2">
          <textarea
            className={`${field} flex-1 resize-y`}
            rows={3}
            value={input}
            placeholder={model.state === "ready" ? "Prompt" : "waiting for the model…"}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
          />
          {busy ? (
            <button type="button" className={`${button} self-end bg-red-700 text-white hover:bg-red-800`} onClick={() => controller.current?.abort()}>
              Stop
            </button>
          ) : (
            <button
              type="button"
              className={`${button} self-end bg-orange-700 text-white hover:bg-orange-800`}
              disabled={!input.trim() || model.state !== "ready"}
              onClick={() => void send()}
            >
              Send
            </button>
          )}
        </div>
      </main>

      <section className={`${panel} order-3 md:col-span-2`}>
        <h2 className="mb-2 font-semibold">
          This session <span className="font-normal text-stone-500 dark:text-stone-400">— HTTP-observed, measured by the browser</span>
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse font-mono text-xs">
            <thead className="text-stone-500 dark:text-stone-400">
              <tr className="border-b border-stone-200 dark:border-stone-800">
                {["#", "time", "lane tag", "effort"].map((h) => (
                  <th key={h} className="px-2 py-1 text-left font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
                {["TTFT", "decode", "duration", "prompt", "completion", "finish"].map((h) => (
                  <th key={h} className="px-2 py-1 text-right font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {session.map((row) => (
                <tr key={row.n} className="border-b border-stone-100 last:border-0 dark:border-stone-800/60">
                  <td className="px-2 py-1">{row.n}</td>
                  <td className="px-2 py-1 whitespace-nowrap">{row.at}</td>
                  <td className="px-2 py-1">{row.laneTag}</td>
                  <td className="px-2 py-1">{row.reasoningEffort}</td>
                  {row.figures ? (
                    <FigureCells figures={row.figures} />
                  ) : (
                    <td colSpan={6} className="px-2 py-1 text-right text-red-700 dark:text-red-400">
                      {row.error}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function FigureLine({ figures }: { figures: Figures }) {
  const d = describeFigures(figures);
  return (
    <p className="mt-2 font-mono text-xs text-stone-500 dark:text-stone-400">
      HTTP-observed · TTFT {d.ttft} · {d.decode} · {d.duration} · {d.promptTokens} → {d.completionTokens} tok · {d.finish}
    </p>
  );
}

function FigureCells({ figures }: { figures: Figures }) {
  const d = describeFigures(figures);
  return (
    <>
      {[d.ttft, d.decode, d.duration, d.promptTokens, d.completionTokens, d.finish].map((value, i) => (
        <td key={i} className="px-2 py-1 text-right whitespace-nowrap">
          {value}
        </td>
      ))}
    </>
  );
}
