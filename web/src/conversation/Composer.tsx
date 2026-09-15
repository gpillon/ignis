import { type KeyboardEvent, useRef } from "react";
import type { ContextUsage } from "../metrics/context.ts";
import { ContextMeter } from "../metrics/ContextMeter.tsx";
import type { Attachment } from "../tools/local/attachments.ts";
import { IconClose, IconPaperclip } from "../ui/icons.tsx";

/** The prompt box under the conversation: Send, or Stop while this session streams, the session's attached files, and the context bar. */
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
  /** The attachments tool is on: files can be attached, and the attached ones show. */
  canAttach: boolean;
  attachments: Attachment[];
  attachError: string | null;
  onAttach: (files: File[]) => void;
  onDetach: (name: string) => void;
}) {
  const { busy, streamingHere } = props;
  const picker = useRef<HTMLInputElement>(null);

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      props.onSend();
    }
  }

  return (
    <div className="shrink-0 px-4 pb-4 md:px-10">
      {props.canAttach && (props.attachments.length > 0 || props.attachError) && (
        <div className="mx-auto mb-2 flex w-full max-w-3xl flex-wrap items-center gap-2">
          {props.attachments.map((a) => (
            <span key={a.name} className="cut flex min-w-0 items-center gap-1.5 bg-surface py-1 pr-1.5 pl-2.5 text-xs [--cut-size:6px]">
              <span className="min-w-0 truncate font-display font-semibold text-ink">{a.name}</span>
              <span className="shrink-0 font-display tabular-nums text-ash">
                {a.kind === "pdf" ? "PDF, " : ""}
                {a.text.length.toLocaleString()} chars
              </span>
              <button type="button" aria-label={`Remove ${a.name}`} onClick={() => props.onDetach(a.name)} className="shrink-0 text-ash hover:text-fault">
                <IconClose />
              </button>
            </span>
          ))}
          {props.attachError && <span className="text-xs leading-snug text-fault">{props.attachError}</span>}
        </div>
      )}
      <div className="cut mx-auto flex w-full max-w-3xl items-end gap-2 bg-surface p-2 shadow-[inset_0_-2px_0_var(--line)] [--cut-size:14px] focus-within:shadow-[inset_0_-2px_0_var(--ember)]">
        {props.canAttach && (
          <>
            <button
              type="button"
              aria-label="Attach files"
              title="Attach text or PDF files for the model to read"
              className="grid size-10 shrink-0 place-items-center text-ash hover:text-ink"
              onClick={() => picker.current?.click()}
            >
              <IconPaperclip />
            </button>
            <input
              ref={picker}
              type="file"
              name="attachments"
              multiple
              hidden
              onChange={(e) => {
                const files = Array.from(e.target.files ?? []);
                e.target.value = "";
                if (files.length > 0) props.onAttach(files);
              }}
            />
          </>
        )}
        <textarea
          className="max-h-60 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-[15px] leading-normal text-ink [field-sizing:content] placeholder:text-ash focus:outline-none"
          rows={1}
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
