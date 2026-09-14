import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import { computeFigures, describeFigures, type Figures } from "./figures.ts";
import { buildChatRequest, conversationTurns, type LaneTag, type Settings } from "./request.ts";
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
  thinking: boolean;
  figures: Figures | null;
  error?: string;
};

type ModelState = { state: "loading" } | { state: "ready"; id: string } | { state: "error"; message: string };

const DEFAULT_SETTINGS: Omit<Settings, "model"> = {
  systemPrompt: "",
  temperature: 0.7,
  topP: 0.95,
  maxTokens: 1024,
  thinking: true,
  laneTag: "interactive",
};

let nextId = 1;

export function App() {
  const [model, setModel] = useState<ModelState>({ state: "loading" });
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [messages, setMessages] = useState<Message[]>([]);
  const [session, setSession] = useState<SessionRow[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
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
        thinking: request.enable_thinking,
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
    <div className="shell">
      <header className="top">
        <h1>Playground</h1>
        <span className="model">
          {model.state === "loading" && "loading model…"}
          {model.state === "ready" && <code>{model.id}</code>}
          {model.state === "error" && <span className="error">{model.message}</span>}
        </span>
      </header>

      <aside className="settings">
        <label>
          System prompt
          <textarea rows={4} value={settings.systemPrompt} onChange={(e) => set("systemPrompt", e.target.value)} />
        </label>
        <label>
          temperature
          <input type="number" min={0} max={2} step={0.05} value={settings.temperature} onChange={(e) => set("temperature", Number(e.target.value))} />
        </label>
        <label>
          top_p
          <input type="number" min={0} max={1} step={0.05} value={settings.topP} onChange={(e) => set("topP", Number(e.target.value))} />
        </label>
        {greedyConflict && <p className="error">temperature 0 needs top_p 1, or ignis answers 400.</p>}
        <label>
          max_tokens
          <input
            type="number"
            min={1}
            placeholder="engine cap"
            value={settings.maxTokens ?? ""}
            onChange={(e) => set("maxTokens", e.target.value === "" ? null : Number(e.target.value))}
          />
        </label>
        <label className="inline">
          <input type="checkbox" checked={settings.thinking} onChange={(e) => set("thinking", e.target.checked)} />
          thinking
        </label>
        <label>
          Lane tag
          <select value={settings.laneTag} onChange={(e) => set("laneTag", e.target.value as LaneTag)}>
            <option value="interactive">Interactive</option>
            <option value="agent">Agent</option>
          </select>
        </label>
        <button type="button" className="secondary" disabled={busy || messages.length === 0} onClick={() => setMessages([])}>
          New conversation
        </button>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && <p className="hint">Send a prompt to start. Enter sends, Shift+Enter adds a line.</p>}
          {messages.map((m) => (
            <article key={m.id} className={`message ${m.role}`}>
              {m.reasoning && (
                <details className="reasoning" open={m.streaming && !m.content}>
                  <summary>reasoning</summary>
                  <pre>{m.reasoning}</pre>
                </details>
              )}
              {(m.content || m.role === "user") && <pre className="content">{m.content}</pre>}
              {m.streaming && !m.content && !m.reasoning && <p className="hint">waiting for the first token…</p>}
              {m.error && <p className="error" role="alert">{m.error}</p>}
              {m.figures && <FigureLine figures={m.figures} />}
            </article>
          ))}
          <div ref={bottom} />
        </div>
        <div className="composer">
          <textarea
            rows={3}
            value={input}
            placeholder={model.state === "ready" ? "Prompt" : "waiting for the model…"}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
          />
          {busy ? (
            <button type="button" className="stop" onClick={() => controller.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="button" disabled={!input.trim() || model.state !== "ready"} onClick={() => void send()}>
              Send
            </button>
          )}
        </div>
      </main>

      <section className="session">
        <h2>
          This session <span className="hint">— HTTP-observed, measured by the browser</span>
        </h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>time</th>
                <th>lane tag</th>
                <th>thinking</th>
                <th>TTFT</th>
                <th>decode</th>
                <th>duration</th>
                <th>prompt</th>
                <th>completion</th>
                <th>finish</th>
              </tr>
            </thead>
            <tbody>
              {session.map((row) => (
                <tr key={row.n}>
                  <td>{row.n}</td>
                  <td>{row.at}</td>
                  <td>{row.laneTag}</td>
                  <td>{row.thinking ? "on" : "off"}</td>
                  {row.figures ? <FigureCells figures={row.figures} /> : <td colSpan={6} className="error">{row.error}</td>}
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
    <p className="figures">
      HTTP-observed · TTFT {d.ttft} · {d.decode} · {d.duration} · {d.promptTokens} → {d.completionTokens} tok · {d.finish}
    </p>
  );
}

function FigureCells({ figures }: { figures: Figures }) {
  const d = describeFigures(figures);
  return (
    <>
      <td>{d.ttft}</td>
      <td>{d.decode}</td>
      <td>{d.duration}</td>
      <td>{d.promptTokens}</td>
      <td>{d.completionTokens}</td>
      <td>{d.finish}</td>
    </>
  );
}
