import { useId, useState } from "react";
import { caption } from "../ui/classes.ts";
import { Markdown } from "../ui/Markdown.tsx";
import { Switch } from "../ui/Switch.tsx";
import { useWatched } from "./assist.ts";
import { jumpOnClick, linkTimes } from "./jump.ts";
import { kindHint, kindOf, NOTE_KINDS, notesMarkdown, notesStatus } from "./notes.ts";
import type { NotesHandle, NotesWriterState } from "./notesWriter.ts";
import { recapMarkdown } from "./recap.ts";
import type { Transcript } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// The Notes tab of the Transcribe view's right panel (spec speech/07,
// Q124-Q127): the kind of conversation, the button that writes the notes, the
// switch that keeps them up while listening, and the notes as the model wrote
// them, streaming in under the same heat the sheet's live line has. Once a
// recap is written (recap.ts, #441), it stands on top and the running notes
// fold under it; Write the recap writes one at any moment.

type Press = { label: string; disabled: boolean; stops: boolean };

/** Why a writing cannot start: no text model, nothing heard, or a writing under way anywhere. */
const cannotWrite = (t: Transcript, writer: NotesWriterState, blocked: string | null) =>
  blocked !== null || t.utterances.every((u) => u.text.trim() === "") || writer.writing !== null;

/** The notes button: what it says, and why it cannot be pressed when it cannot. */
export function notesButton(t: Transcript, writer: NotesWriterState, blocked: string | null): Press {
  if (writer.writing === t.id && writer.form === "running") return { label: "Stop", disabled: false, stops: true };
  const label = t.notes?.writtenAs && t.notes.writtenAs === kindOf(t) ? "Update notes" : "Write notes";
  return { label, disabled: cannotWrite(t, writer, blocked), stops: false };
}

/** The recap button, as the notes' is: a Stop while the recap is written. */
export function recapButton(t: Transcript, writer: NotesWriterState, blocked: string | null): Press {
  if (writer.writing === t.id && writer.form === "recap") return { label: "Stop", disabled: false, stops: true };
  return { label: "Write the recap", disabled: cannotWrite(t, writer, blocked), stops: false };
}

export function Notes({ transcript, state, actions, notes }: { transcript: Transcript; state: TranscribeState; actions: TranscribeActions; notes: NotesHandle }) {
  return <NotesPanel transcript={transcript} state={state} actions={actions} writer={useWatched(notes)} />;
}

export function NotesPanel({ transcript: t, state, actions, writer }: { transcript: Transcript; state: TranscribeState; actions: TranscribeActions; writer: NotesWriterState }) {
  const kindId = useId();
  const liveId = useId();
  const [copied, setCopied] = useState(false);
  const kind = kindOf(t);
  const here = writer.writing === t.id && writer.form === "running";
  const recapHere = writer.writing === t.id && writer.form === "recap";
  const button = notesButton(t, writer, state.textBlocked);
  const recapPress = recapButton(t, writer, state.textBlocked);
  const kept = notesMarkdown(t.notes);
  const recapKept = recapMarkdown(t.recap);
  // While a writing streams, its text; before its first token, what it replaces, dimmed.
  const shown = here && writer.text !== "" ? writer.text : kept;
  const recapShown = recapHere && writer.text !== "" ? writer.text : recapKept;
  const recapOnTop = recapHere || recapKept !== "";
  // Copy gives the notes meant for after: the recap when there is one, else the running notes.
  const copy = recapKept !== "" ? { text: recapKept, label: "Copy recap" } : { text: kept, label: "Copy notes" };
  const status = here ? null : notesStatus(t.notes, t.utterances.length);
  const recapStatus = recapHere ? null : notesStatus(t.recap, t.utterances.length);
  const failed = writer.failed?.transcript === t.id && writer.writing !== t.id ? writer.failed : null;
  const running = (
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
            : "The text model reads this transcript and writes its notes, in the kind picked above. Nothing goes to it until you press Write notes or Write the recap, or turn on Keep updating."}
        </p>
      )}
      {here && shown && writer.text === "" && <p className="text-xs text-ash">The model is reading the transcript.</p>}
      {status && <p className="text-xs leading-snug text-ash">{status}</p>}
    </article>
  );

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
        <div className="flex flex-wrap items-center gap-1.5">
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
            disabled={recapPress.disabled}
            onClick={recapPress.stops ? actions.stopNotes : actions.writeRecap}
            className={`font-display text-[13px] disabled:cursor-not-allowed disabled:opacity-50 ${recapPress.stops ? "cut bg-surface px-3.5 py-1.5 font-semibold text-ink [--cut-size:7px] hover:bg-line" : "border border-line px-2.5 py-1.5 font-medium text-ink hover:border-ember"}`}
          >
            {recapPress.label}
          </button>
          <button
            type="button"
            disabled={copy.text === ""}
            onClick={() => {
              void navigator.clipboard?.writeText(copy.text).then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 2_000);
              });
            }}
            className="border border-line px-2.5 py-1.5 font-display text-[13px] font-medium text-ink hover:border-ember disabled:cursor-not-allowed disabled:opacity-50"
          >
            {copied ? "Copied" : copy.label}
          </button>
        </div>
        <div className="flex items-start gap-2.5">
          <Switch on={state.notesLive} onChange={actions.setNotesLive} labelledBy={liveId} />
          <span className="flex flex-col gap-0.5">
            <span id={liveId} className="font-display text-[13px] font-medium text-ink">
              Keep updating while listening
            </span>
            <span className="text-xs leading-snug text-ash">About once a minute while something new is said, and the recap when the listening ends.</span>
          </span>
        </div>
        {state.textBlocked && <p className="border-l-2 border-ember pl-2 text-xs leading-snug text-ink">{state.textBlocked}</p>}
        {failed && (
          <p role="status" className="border-l-2 border-fault pl-2 text-xs leading-snug text-fault">
            {failed.form === "recap" ? "The recap was not written" : "The notes were not written"}: {failed.message}
          </p>
        )}
      </section>

      {recapOnTop && (
        <article aria-label="Recap" aria-busy={recapHere} className="relative flex flex-col gap-3 pl-3.5" onClickCapture={jumpOnClick}>
          <span aria-hidden="true" className={`absolute inset-y-0 left-0 w-[2px] ${recapHere ? "spine-live" : "bg-line"}`} />
          <h2 className={caption}>Recap</h2>
          {recapShown && (
            <div className={`notes ${recapHere && writer.text === "" ? "opacity-50" : ""}`}>
              <Markdown text={linkTimes(recapShown)} streaming={recapHere} />
            </div>
          )}
          {recapHere && writer.text === "" && <p className="text-xs text-ash">The model is reading the whole transcript.</p>}
          {recapStatus && <p className="text-xs leading-snug text-ash">{recapStatus}</p>}
        </article>
      )}
      {recapOnTop ? (
        <details>
          <summary className="cursor-pointer select-none font-display text-[13px] font-medium text-ash hover:text-ink">Running notes</summary>
          <div className="mt-3">{running}</div>
        </details>
      ) : (
        running
      )}
    </>
  );
}
