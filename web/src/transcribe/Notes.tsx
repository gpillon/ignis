import { useId, useState } from "react";
import { caption } from "../ui/classes.ts";
import { Markdown } from "../ui/Markdown.tsx";
import { Switch } from "../ui/Switch.tsx";
import { useWatched } from "./assist.ts";
import { jumpOnClick, linkTimes } from "./jump.ts";
import { kindHint, kindOf, NOTE_KINDS, notesMarkdown, notesStatus } from "./notes.ts";
import type { NotesHandle, NotesWriterState } from "./notesWriter.ts";
import type { Transcript } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// The Notes tab of the Transcribe view's right panel (spec speech/07,
// Q124-Q127): the kind of conversation, the button that writes the notes, the
// switch that keeps them up while listening, and the notes as the model wrote
// them, streaming in under the same heat the sheet's live line has.

/** The notes button: what it says, and why it cannot be pressed when it cannot. */
export function notesButton(t: Transcript, writer: NotesWriterState, blocked: string | null): { label: string; disabled: boolean; stops: boolean } {
  if (writer.writing === t.id) return { label: "Stop", disabled: false, stops: true };
  const label = t.notes?.writtenAs && t.notes.writtenAs === kindOf(t) ? "Update notes" : "Write notes";
  const nothing = t.utterances.every((u) => u.text.trim() === "");
  return { label, disabled: blocked !== null || nothing || writer.writing !== null, stops: false };
}

export function Notes({ transcript, state, actions, notes }: { transcript: Transcript; state: TranscribeState; actions: TranscribeActions; notes: NotesHandle }) {
  return <NotesPanel transcript={transcript} state={state} actions={actions} writer={useWatched(notes)} />;
}

export function NotesPanel({ transcript: t, state, actions, writer }: { transcript: Transcript; state: TranscribeState; actions: TranscribeActions; writer: NotesWriterState }) {
  const kindId = useId();
  const liveId = useId();
  const [copied, setCopied] = useState(false);
  const kind = kindOf(t);
  const here = writer.writing === t.id;
  const button = notesButton(t, writer, state.textBlocked);
  const kept = notesMarkdown(t.notes);
  // While a writing streams, its text; before its first token, the notes it replaces, dimmed.
  const shown = here && writer.text !== "" ? writer.text : kept;
  const status = here ? null : notesStatus(t.notes, t.utterances.length);
  const failed = writer.failed?.transcript === t.id && !here ? writer.failed.message : null;

  return (
    <>
      <section aria-labelledby={kindId} className="flex flex-col gap-2">
        <h2 id={kindId} className={caption}>
          Kind of conversation
        </h2>
        <div role="radiogroup" aria-labelledby={kindId} className="grid grid-cols-2 gap-1.5">
          {NOTE_KINDS.map((k) => (
            <button
              key={k.kind}
              type="button"
              role="radio"
              aria-checked={kind === k.kind}
              onClick={() => actions.setNoteKind(k.kind)}
              className={`flex flex-col gap-0.5 border px-2.5 py-2 text-left ${kind === k.kind ? "border-ember bg-surface" : "border-line hover:border-ash"}`}
            >
              <span className={`font-display text-[13px] font-semibold ${kind === k.kind ? "text-ink" : "text-ink/85"}`}>{k.label}</span>
              <span className="text-[11.5px] leading-snug text-ash">{kindHint(k)}</span>
            </button>
          ))}
        </div>
      </section>

      <section aria-label="Writing the notes" className="flex flex-col gap-3">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            disabled={button.disabled}
            onClick={button.stops ? actions.stopNotes : actions.writeNotes}
            className={`cut px-3.5 py-1.5 font-display text-[13px] font-semibold [--cut-size:7px] disabled:cursor-not-allowed disabled:opacity-50 ${button.stops ? "bg-surface text-ink hover:bg-line" : "bg-ink text-ground hover:bg-ember"}`}
          >
            {button.label}
          </button>
          <button
            type="button"
            disabled={kept === ""}
            onClick={() => {
              void navigator.clipboard?.writeText(kept).then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 2_000);
              });
            }}
            className="border border-line px-2.5 py-1.5 font-display text-[13px] font-medium text-ink hover:border-ember disabled:cursor-not-allowed disabled:opacity-50"
          >
            {copied ? "Copied" : "Copy notes"}
          </button>
        </div>
        <div className="flex items-start gap-2.5">
          <Switch on={state.notesLive} onChange={actions.setNotesLive} labelledBy={liveId} />
          <span className="flex flex-col gap-0.5">
            <span id={liveId} className="font-display text-[13px] font-medium text-ink">
              Keep updating while listening
            </span>
            <span className="text-xs leading-snug text-ash">About once a minute while something new is said, and once more at the end.</span>
          </span>
        </div>
        {state.textBlocked && <p className="border-l-2 border-ember pl-2 text-xs leading-snug text-ink">{state.textBlocked}</p>}
        {failed && (
          <p role="status" className="border-l-2 border-fault pl-2 text-xs leading-snug text-fault">
            The notes were not written: {failed}
          </p>
        )}
      </section>

      <article aria-label="Notes" aria-busy={here} className="relative flex flex-col gap-3 pl-3.5" onClickCapture={jumpOnClick}>
        <span aria-hidden="true" className={`absolute inset-y-0 left-0 w-[2px] ${here ? "spine-live" : shown ? "bg-line" : "bg-transparent"}`} />
        {shown ? (
          <div className={`notes ${here && writer.text === "" ? "opacity-50" : ""}`}>
            <Markdown text={linkTimes(shown)} streaming={here} />
          </div>
        ) : (
          <p className="text-xs leading-snug text-ash">
            {here
              ? "The model is reading the transcript."
              : "The text model reads this transcript and writes its notes, in the kind picked above. Nothing goes to it until you press Write notes or turn on Keep updating."}
          </p>
        )}
        {here && shown && writer.text === "" && <p className="text-xs text-ash">The model is reading the transcript.</p>}
        {status && <p className="text-xs leading-snug text-ash">{status}</p>}
      </article>
    </>
  );
}
