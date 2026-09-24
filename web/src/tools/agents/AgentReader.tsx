import { type ReactNode, useEffect, useLayoutEffect, useRef } from "react";
import { BudgetReached } from "../../metrics/BudgetReached.tsx";
import { Markdown } from "../../ui/Markdown.tsx";
import { isAtBottom } from "../../ui/scroll.ts";
import { LocalStrip } from "../local/LocalStrip.tsx";
import { UnknownCalls } from "../UnknownCalls.tsx";
import { WebStrip } from "../web/WebStrip.tsx";
import type { AgentRun } from "./agents.ts";
import { StatusMark, statusLine, useNow } from "./status.tsx";

/** The side reader for one agent: what it was asked, how it reasoned, what it answered. */
export function AgentReader(props: { run: AgentRun; markdown: boolean; figures: ReactNode; systemPrompt: string; onClose: () => void }) {
  const { run, onClose } = props;
  const now = useNow(run.status === "running");
  const thinking = run.status === "running" && !run.content;
  // The reasoning runs across every request the agent made: the latest one the budget closed, if any.
  const forcedAt = run.rounds?.filter((round) => round.thinkingForcedAt !== undefined).at(-1)?.thinkingForcedAt;
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
              {forcedAt !== undefined && <BudgetReached at={forcedAt} />}
            </summary>
            <pre className="mt-2 border-l-2 border-line pl-3 whitespace-pre-wrap break-words font-sans text-[13px] leading-normal text-ash">
              {run.reasoning}
            </pre>
          </details>
        )}
        {run.web && run.web.length > 0 && <WebStrip runs={run.web} />}
        {run.local && run.local.length > 0 && <LocalStrip runs={run.local} />}
        {run.unknownTools && run.unknownTools.length > 0 && <UnknownCalls calls={run.unknownTools} />}
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
