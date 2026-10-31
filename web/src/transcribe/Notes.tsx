import { type KeyboardEvent, useEffect, useId, useRef, useState } from "react";
import { caption } from "../ui/classes.ts";
import { IconChevron, IconClose, IconPlus, IconRegenerate } from "../ui/icons.tsx";
import { InlineMarkdown, Markdown } from "../ui/Markdown.tsx";
import { Switch } from "../ui/Switch.tsx";
import { useWatched } from "./assist.ts";
import { hoursClocks } from "./export.ts";
import { jumpOnClick, linkTimes } from "./jump.ts";
import { changesBetween, type DrawnSection, markedSections, type NoteMark, type NoteRow, nextExpiry, withChanges } from "./marks.ts";
import { kindHint, kindOf, NOTE_KINDS, type NoteLine, type Notes as NotesOf, notesMarkdown, notesStatus } from "./notes.ts";
import type { NotesForm, NotesHandle, NotesWriterState } from "./notesWriter.ts";
import { recapMarkdown, recapSections, recapStatus } from "./recap.ts";
import type { Transcript } from "./transcript.ts";
import type { TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// The Notes tab of the Transcribe view's right panel (spec speech/07,
// Q124-Q127; spec #428): the kind of conversation, the button that updates the
// running notes, the switch that keeps them up while listening, and the notes
// in the kind's live sections, under the same heat the sheet's live line has
// while an update changes them a line at a time. Rewrite from scratch sits in
// a menu. Once a recap is written (recap.ts, #441), it stands on top and the
// running notes fold under it; Write the recap writes one at any moment.
// The user edits both by hand (#436): a click edits a line in place, × on
// hover removes it, + at a section's foot adds one, and their own lines carry
// a hollow bullet where the model's have the ember one.

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

/** Rewrite from scratch: once the model wrote a line of the notes, and only when a writing could start. */
export function rewriteButton(t: Transcript, writer: NotesWriterState, blocked: string | null): { disabled: boolean } {
  return { disabled: cannotWrite(t, writer, blocked) || !(t.notes?.lines ?? []).some((l) => l.author === "model") };
}

/** What marks the notes carry now: kept as the notes change, and let go of when their time is up (marks.ts). */
export type Marked = { marks: NoteMark[]; now: number };

/** The marks of a Transcript's notes: every line the model adds, changes or removes, for a few seconds; none for notes just opened. */
function useMarked(id: number, notes: NotesOf | undefined): Marked {
  const [state, setState] = useState<{ id: number; marks: NoteMark[] }>({ id, marks: [] });
  const [now, setNow] = useState(() => Date.now());
  const seen = useRef({ id, notes });
  useEffect(() => {
    const before = seen.current;
    seen.current = { id, notes };
    if (before.id !== id) return setState({ id, marks: [] });
    const found = changesBetween(before.notes, notes);
    if (found.length === 0) return;
    const at = Date.now();
    setNow(at);
    setState((s) => ({ id, marks: withChanges(s.id === id ? s.marks : [], found, at) }));
  }, [id, notes]);
  const marks = state.id === id ? state.marks : [];
  const expiry = nextExpiry(marks, now);
  useEffect(() => {
    if (expiry === null) return;
    const timer = setTimeout(() => setNow(Date.now()), Math.max(0, expiry - Date.now()) + 20);
    return () => clearTimeout(timer);
  }, [expiry]);
  return { marks, now };
}

/** What is typed in place: a line of the running notes or the recap (as the user saw it), or a new line at a section's foot. */
export type Editing = { form: NotesForm; line: NoteLine } | { form: NotesForm; section: string };

/** A line typed in place: Enter keeps it, Escape lets it go, and leaving the field keeps it too. */
function LineEditor({ text, label, hint, onDone }: { text: string; label: string; hint: string; onDone: (typed: string | null) => void }) {
  const field = useRef<HTMLTextAreaElement>(null);
  const done = useRef(false);
  useEffect(() => {
    const el = field.current;
    el?.focus();
    el?.setSelectionRange(el.value.length, el.value.length);
  }, []);
  const finish = (typed: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(typed);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.nativeEvent.isComposing) {
      e.preventDefault();
      finish(e.currentTarget.value);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      finish(null);
    }
  };
  return (
    <span className="note-editor">
      <textarea
        ref={field}
        aria-label={label}
        rows={1}
        defaultValue={text}
        onKeyDown={onKeyDown}
        onBlur={(e) => finish(e.currentTarget.value)}
        className="w-full resize-none rounded-[2px] border border-line bg-surface px-1.5 py-0.5 text-ink [field-sizing:content] focus:border-ember focus:outline-none"
      />
      <span className="text-[11px] leading-snug text-ash">{hint}</span>
    </span>
  );
}

/** A task's box at a line's start (`[ ] `, `[x] `), as a recap's Action items write them. */
const TASK = /^\[([ xX])\][ \t]+/;

/** One line of the notes: its text, a click away from editing it, and × to remove it; struck while a removal's mark stands. */
function NoteItem({ row, form, editing, onEditing, actions }: { row: NoteRow; form: NotesForm; editing: boolean; onEditing: (e: Editing | null) => void; actions: TranscribeActions }) {
  const { line, mark } = row;
  const shown = hoursClocks(line.text);
  if (editing) {
    return (
      <li data-author={line.author}>
        <LineEditor
          text={shown}
          label={`Line ${line.id} of ${line.section}`}
          hint="Enter saves, Escape cancels; emptied, the line goes."
          onDone={(typed) => {
            onEditing(null);
            if (typed !== null && typed.trim() !== shown) actions.editNote(form, line, typed);
          }}
        />
      </li>
    );
  }
  if (mark === "gone") {
    return (
      <li data-author={line.author} data-mark="gone">
        <del>
          <InlineMarkdown text={shown} />
        </del>
      </li>
    );
  }
  const task = TASK.exec(shown);
  const mine = line.author === "user";
  const edit = () => onEditing({ form, line });
  return (
    <li data-author={line.author} data-mark={mark ?? undefined}>
      {task && <input type="checkbox" disabled checked={task[1] !== " "} />}
      <span
        tabIndex={0}
        title={mine ? "Your line: updates leave it as it is. Click to edit it" : "Edit the line"}
        onClick={(e) => (e.target as Element).closest("a") === null && edit()}
        onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), edit())}
        className="note-text"
      >
        <InlineMarkdown text={linkTimes(task ? shown.slice(task[0].length) : shown)} />
      </span>
      {mine && <span className="sr-only"> (your line)</span>}
      <button type="button" aria-label={`Remove line ${line.id}`} title="Remove the line" onClick={() => actions.removeNote(form, line)} className="note-remove">
        <IconClose />
      </button>
    </li>
  );
}

/**
 * The notes' lines in their sections, each editable in place, and a + at each section's foot. A line being edited stays
 * where the user types it, even when the model removed it meanwhile: their edit wins.
 */
function NoteLines({ sections, form, editing, onEditing, actions }: { sections: DrawnSection[]; form: NotesForm; editing: Editing | null; onEditing: (e: Editing | null) => void; actions: TranscribeActions }) {
  const here = editing?.form === form ? editing : null;
  const typing = here && "line" in here ? here.line : null;
  return (
    <div className="notes">
      <div className="md">
        {sections.map(({ name, rows }) => {
          const kept = typing && typing.section === name && !rows.some((r) => r.line.id === typing.id) ? [...rows, { line: typing, mark: null }] : rows;
          const adding = here !== null && "section" in here && here.section === name;
          return (
            <div key={name} className="note-section">
              <h2>{name}</h2>
              {kept.length > 0 ? (
                <ul>
                  {kept.map((row) => (
                    <NoteItem key={row.line.id} row={row} form={form} editing={typing?.id === row.line.id} onEditing={onEditing} actions={actions} />
                  ))}
                </ul>
              ) : (
                <p>—</p>
              )}
              {adding ? (
                <LineEditor
                  text=""
                  label={`New line in ${name}`}
                  hint="Enter adds it, Escape cancels."
                  onDone={(typed) => {
                    onEditing(null);
                    if (typed !== null && typed.trim() !== "") actions.addNote(form, name, typed);
                  }}
                />
              ) : (
                <button type="button" aria-label={`Add a line to ${name}`} onClick={() => onEditing({ form, section: name })} className="note-add">
                  <IconPlus />
                  <span>Add a line</span>
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function Notes({ transcript, state, actions, notes }: { transcript: Transcript; state: TranscribeState; actions: TranscribeActions; notes: NotesHandle }) {
  return <NotesPanel key={transcript.id} transcript={transcript} state={state} actions={actions} writer={useWatched(notes)} />;
}

export function NotesPanel({
  transcript: t,
  state,
  actions,
  writer,
  marked,
  editing: shownEditing,
}: {
  transcript: Transcript;
  state: TranscribeState;
  actions: TranscribeActions;
  writer: NotesWriterState;
  marked?: Marked;
  editing?: Editing | null;
}) {
  const own = useMarked(t.id, t.notes);
  const { marks, now } = marked ?? own;
  const [ownEditing, setEditing] = useState<Editing | null>(null);
  const editing = shownEditing === undefined ? ownEditing : shownEditing;
  const kindId = useId();
  const liveId = useId();
  const [copied, setCopied] = useState(false);
  const kind = kindOf(t);
  const here = writer.writing === t.id && writer.form === "running";
  const recapHere = writer.writing === t.id && writer.form === "recap";
  const button = notesButton(t, writer, state.textBlocked);
  const recapPress = recapButton(t, writer, state.textBlocked);
  const rewrite = rewriteButton(t, writer, state.textBlocked);
  const menu = useRef<HTMLDetailsElement>(null);
  // The running notes change in place, a line at a time, as an update applies the model's calls.
  const shown = notesMarkdown(t.notes);
  const drawn = markedSections(t.notes, marks, now);
  const recapKept = recapMarkdown(t.recap);
  // While the recap streams, its text; before its first token, the one it replaces, dimmed.
  const recapShown = recapHere && writer.text !== "" ? writer.text : recapKept;
  const recapOnTop = recapHere || recapKept !== "";
  // Copy gives the notes meant for after: the recap when there is one, else the running notes.
  const copy = recapKept !== "" ? { text: recapKept, label: "Copy recap" } : { text: shown, label: "Copy notes" };
  const status = here ? null : notesStatus(t.notes, t.utterances.length, state.notesLive);
  const recapLine = recapHere ? null : recapStatus(t.recap, t.utterances.length);
  const failed = writer.failed?.transcript === t.id && writer.writing !== t.id ? writer.failed : null;
  const running = (
    <article aria-label="Notes" aria-busy={here} className="relative flex flex-col gap-3 pl-3.5" onClickCapture={jumpOnClick}>
      <span aria-hidden="true" className={`absolute inset-y-0 left-0 w-[2px] ${here ? "spine-live" : shown ? "bg-line" : "bg-transparent"}`} />
      {drawn.length > 0 ? (
        <NoteLines sections={drawn} form="running" editing={editing} onEditing={setEditing} actions={actions} />
      ) : (
        <p className="text-xs leading-snug text-ash">
          {here
            ? "The model is reading the transcript."
            : "The text model reads this transcript and writes its notes, in the kind picked above. Nothing goes to it until you press Write notes or Write the recap, or turn on Keep updating."}
        </p>
      )}
      {here && shown && <p className="text-xs text-ash">The model is reading what was said since.</p>}
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
          <details ref={menu} className="group relative">
            <summary className="flex cursor-pointer list-none items-center gap-1 border border-line px-2.5 py-1.5 font-display text-[13px] font-medium text-ink hover:border-ember [&::-webkit-details-marker]:hidden">
              More
              <IconChevron className="rotate-180 transition-transform group-open:rotate-0 motion-reduce:transition-none" />
            </summary>
            <div className="absolute right-0 z-20 mt-1 flex w-60 max-w-[calc(100vw-2rem)] flex-col border border-line bg-surface py-1 shadow-lg">
              <button
                type="button"
                disabled={rewrite.disabled}
                onClick={() => {
                  menu.current?.removeAttribute("open");
                  actions.rewriteNotes();
                }}
                className="flex items-start gap-2 px-3 py-1.5 text-left hover:bg-line disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent"
              >
                <span className="mt-0.5 text-ash">
                  <IconRegenerate />
                </span>
                <span className="flex flex-col gap-0.5">
                  <span className="font-display text-[13px] font-medium text-ink">Rewrite from scratch</span>
                  <span className="text-xs leading-snug text-ash">Drops the model's lines and writes the notes again from the transcript.</span>
                </span>
              </button>
            </div>
          </details>
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
          {recapHere
            ? recapShown && (
                <div className={`notes ${writer.text === "" ? "opacity-50" : ""}`}>
                  <Markdown text={linkTimes(recapShown)} streaming />
                </div>
              )
            : t.recap && (
                <NoteLines
                  sections={markedSections(t.recap, [], now, recapSections(t.recap.writtenAs ?? t.recap.kind).map((s) => s.name))}
                  form="recap"
                  editing={editing}
                  onEditing={setEditing}
                  actions={actions}
                />
              )}
          {recapHere && writer.text === "" && <p className="text-xs text-ash">The model is reading the whole transcript.</p>}
          {recapLine && <p className="text-xs leading-snug text-ash">{recapLine}</p>}
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
