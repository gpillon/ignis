import type { KeyboardEvent } from "react";
import type { ContextUsage } from "../metrics/context.ts";
import { ContextMeter } from "../metrics/ContextMeter.tsx";

/** The prompt box under the conversation: Send, or Stop while this session streams, and the context bar. */
export function Composer(props: {
  value: string;
  onChange: (value: string) => void;
  /** The model is known. */
  ready: boolean;
  /** A reply streams in some session. */
  busy: boolean;
  /** A reply streams in this session. */
  streamingHere: boolean;
  usage: ContextUsage;
  onSend: () => void;
  onStop: () => void;
}) {
  const { busy, streamingHere } = props;

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      props.onSend();
    }
  }

  return (
    <div className="shrink-0 px-4 pb-4 md:px-10">
      <div className="cut mx-auto flex w-full max-w-3xl items-end gap-2 bg-surface p-2 shadow-[inset_0_-2px_0_var(--line)] [--cut-size:14px] focus-within:shadow-[inset_0_-2px_0_var(--ember)]">
        <textarea
          className="max-h-60 min-h-[3.25rem] flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-normal text-ink [field-sizing:content] placeholder:text-ash focus:outline-none"
          rows={2}
          name="prompt"
          aria-label="Prompt"
          value={props.value}
          placeholder={props.ready ? "Ask anything" : "Waiting for the model…"}
          onChange={(e) => props.onChange(e.target.value)}
          onKeyDown={onKeyDown}
        />
        {streamingHere ? (
          <button
            type="button"
            className="cut bg-[#c8161d] px-5 py-2.5 font-display text-sm font-semibold text-white hover:bg-[#a8121a]"
            onClick={props.onStop}
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            className="cut bg-[#ff5a1f] px-5 py-2.5 font-display text-sm font-semibold text-[#1c2026] hover:bg-[#ff7a45] disabled:cursor-default disabled:bg-line disabled:text-ash"
            disabled={busy || !props.value.trim() || !props.ready}
            onClick={props.onSend}
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
        <ContextMeter usage={props.usage} />
      </div>
    </div>
  );
}
