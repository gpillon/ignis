import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import { describeFigures, formatMs } from "./figures.ts";
import { Markdown } from "./Markdown.tsx";
import { isAtBottom } from "./scroll.ts";
import { type AgentRun, type AgentStatus, agentSummary } from "./tools/agents.ts";

// The agents a reply started, as a strip of lanes inside the reply, and the
// reader that opens one of them: its prompt, reasoning and answer, live.

/** A clock that ticks while `active`, for the running agents' elapsed time. */
function useNow(active: boolean) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

const STATUS_LABEL: Record<AgentStatus, string> = {
  queued: "Waiting for a lane",
  running: "Running",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

function statusLine(run: AgentRun, now: number): string {
  if (run.status === "running" && run.startedAt !== undefined) return `Running, ${formatMs(now - run.startedAt)}`;
  if (run.status === "done" && run.figures) return `Done in ${describeFigures(run.figures).duration}`;
  return STATUS_LABEL[run.status];
}

function StatusMark({ status }: { status: AgentStatus }) {
  const color = {
    queued: "bg-line",
    running: "bg-ember animate-pulse motion-reduce:animate-none",
    done: "bg-ink",
    failed: "bg-fault",
    stopped: "bg-ash",
  }[status];
  return <span className={`cut size-2 shrink-0 [--cut-size:3px] ${color}`} aria-hidden />;
}

export function AgentStrip(props: { runs: AgentRun[]; openCallId: string | null; onOpen: (callId: string) => void }) {
  const now = useNow(props.runs.some((r) => r.status === "running"));
  return (
    <section aria-label="Agents" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="font-display text-[13px] font-semibold text-ink">Agents</h3>
        <p className="font-display text-xs text-ash">{agentSummary(props.runs)}</p>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(12rem,1fr))] gap-2">
        {props.runs.map((run) => (
          <AgentCard
            key={run.callId}
            run={run}
            now={now}
            open={run.callId === props.openCallId}
            onOpen={() => props.onOpen(run.callId)}
          />
        ))}
      </div>
    </section>
  );
}

function AgentCard({ run, now, open, onOpen }: { run: AgentRun; now: number; open: boolean; onOpen: () => void }) {
  const d = run.figures ? describeFigures(run.figures) : null;
  const bar = {
    queued: "bg-line",
    running: "",
    done: "bg-ember",
    failed: "bg-fault",
    stopped: "bg-ash",
  }[run.status];
  return (
    <button
      type="button"
      aria-pressed={open}
      title={`Read ${run.name}`}
      onClick={onOpen}
      className={`cut flex min-w-0 flex-col gap-2 bg-surface px-3 pt-2.5 pb-2 text-left [--cut-size:10px] hover:bg-line/50 ${open ? "shadow-[inset_0_0_0_1px_var(--ember)]" : ""}`}
    >
      <span className="flex min-w-0 items-center gap-2">
        <StatusMark status={run.status} />
        <span className="truncate font-display text-sm font-semibold text-ink">{run.name}</span>
      </span>
      <span className="line-clamp-2 min-h-[2lh] text-xs leading-snug text-ash">{run.prompt}</span>
      {run.status === "running" ? (
        <span className="heat block" data-busy="true" aria-hidden />
      ) : (
        <span className={`block h-0.5 ${bar}`} aria-hidden />
      )}
      <span className="flex items-baseline justify-between gap-2 font-display text-xs tabular-nums">
        <span className={run.status === "failed" ? "text-fault" : "text-ash"}>{statusLine(run, now)}</span>
        {d && run.status === "done" && <span className="font-semibold text-ember">{d.decode}</span>}
      </span>
    </button>
  );
}

/** The side reader for one agent: what it was asked, how it reasoned, what it answered. */
export function AgentReader(props: { run: AgentRun; markdown: boolean; figures: ReactNode; systemPrompt: string; onClose: () => void }) {
  const { run, onClose } = props;
  const now = useNow(run.status === "running");
  const thinking = run.status === "running" && !run.content;
  // Like the conversation: new text pulls the view down only while the
  // reader is at the bottom. An agent still writing is followed from the
  // start; a finished one opens at its prompt.
  const body = useRef<HTMLDivElement>(null);
  const following = useRef(run.status === "running" || run.status === "queued");

  useLayoutEffect(() => {
    const el = body.current;
    if (el && following.current) el.scrollTop = el.scrollHeight;
  }, [run.content, run.reasoning, run.status]);

  useEffect(() => {
    const close = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);

  return (
    <aside
      aria-label={`Agent: ${run.name}`}
      className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l border-kiln-line bg-ground shadow-[-24px_0_48px_rgb(0_0_0/0.3)]"
    >
      <header className="shrink-0 bg-kiln text-[#eae8e4]">
        <div className="flex items-start gap-3 px-5 py-4">
          <StatusMark status={run.status} />
          <div className="-mt-1.5 min-w-0 flex-1">
            <h2 className="truncate font-display text-lg font-semibold">{run.name}</h2>
            <p className="font-display text-xs text-[#b9bec4] tabular-nums">
              {statusLine(run, now)}, agent lane
            </p>
          </div>
          <button
            type="button"
            aria-label="Close the agent reader"
            title="Close (Esc)"
            className="-mt-1 -mr-2 grid size-8 place-items-center text-[#b9bec4] hover:bg-kiln-line hover:text-white"
            onClick={onClose}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
              <path d="M4 4l8 8M12 4l-8 8" />
            </svg>
          </button>
        </div>
        <div className="heat" data-busy={run.status === "running"} aria-hidden />
      </header>

      <div
        ref={body}
        onScroll={(e) => (following.current = isAtBottom(e.currentTarget))}
        className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto px-5 py-5"
      >
        <details className="reasoning">
          <summary className="cursor-pointer select-none font-display text-[13px] font-medium text-ash hover:text-ink">
            System prompt
          </summary>
          <pre className="mt-2 border-l-2 border-line pl-3 whitespace-pre-wrap break-words font-sans text-[13px] leading-normal text-ash">
            {props.systemPrompt}
          </pre>
        </details>
        <details className="reasoning" open>
          <summary className="cursor-pointer select-none font-display text-[13px] font-medium text-ash hover:text-ink">
            Prompt
          </summary>
          <pre className="cut mt-2 bg-surface px-3 py-2.5 whitespace-pre-wrap break-words font-sans text-sm leading-normal [--cut-size:10px]">
            {run.prompt}
          </pre>
        </details>
        {run.reasoning && (
          <details className="reasoning" open={thinking}>
            <summary className="cursor-pointer select-none font-display text-[13px] font-medium text-ash hover:text-ink">
              {thinking ? "Thinking…" : "Reasoning"}
            </summary>
            <pre className="mt-2 border-l-2 border-line pl-3 whitespace-pre-wrap break-words font-sans text-[13px] leading-normal text-ash">
              {run.reasoning}
            </pre>
          </details>
        )}
        <section aria-label="Answer" className="flex flex-col gap-3">
          {run.content ? (
            props.markdown ? (
              <Markdown text={run.content} streaming={run.status === "running"} />
            ) : (
              <pre className="whitespace-pre-wrap break-words font-sans text-[15px] leading-normal">
                {run.content}
                {run.status === "running" && <span className="caret" aria-hidden />}
              </pre>
            )
          ) : (
            <p className="text-sm text-ash">
              {run.status === "queued"
                ? "Waiting for a free lane."
                : run.status === "running"
                  ? "No answer yet."
                  : run.status === "failed"
                    ? ""
                    : "The agent returned no text."}
            </p>
          )}
          {run.error && (
            <p className="border-l-2 border-fault pl-3 text-sm text-fault" role="alert">
              {run.error}
            </p>
          )}
          {props.figures}
        </section>
      </div>
    </aside>
  );
}
