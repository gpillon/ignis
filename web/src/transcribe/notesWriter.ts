import type { ConversationRequest } from "../api/request.ts";
import { type AskDeps, askModel, Follower, settle, Watched } from "./assist.ts";
import { dueForUpdate, LIVE_GAP_MS, type NoteKind, notesRequest, parseNotes, tidyNotes, type Written } from "./notes.ts";
import { parseRecap, recapDue, recapRequest } from "./recap.ts";
import type { Transcript } from "./transcript.ts";

// The notes' writer (spec speech/07, Q124-Q127): one writing at a time, on
// the text model loaded, with no model named, so a request never switches
// one. Its text streams into the state while it writes; the notes land on the
// Transcript the writing started on (`onWritten`), whichever is open by then,
// and a Stop or a failure leaves the notes there were. A full engine is
// waited out a second at a time, as an agent's request is: the notes are a
// side request on an agent lane, behind the chat. The page holds one, so a
// writing goes on while the view is hidden. What lands is the model's
// Markdown read as lines (notes.ts's parseNotes). It writes the recap too
// (recap.ts), on a press and when a listening ends, in place of the notes'
// last writing: one writing at a time of either, on the GPU the chat shares.

/** Which notes a writing writes: the running notes, or the recap. */
export type NotesForm = "running" | "recap";

export type NotesWriterState = {
  /** The Transcript whose notes are being written. */
  writing: number | null;
  /** Which of them: the running notes when none is. */
  form: NotesForm;
  /** What the writing has written so far. */
  text: string;
  /** The last writing that failed, which notes it was, and why. */
  failed: { transcript: number; form: NotesForm; message: string } | null;
};

export type { Written };

export type NotesWriterDeps = AskDeps & {
  onWritten: (transcript: number, written: Written, form: NotesForm) => void;
  now?: () => number;
};

const IDLE: NotesWriterState = { writing: null, form: "running", text: "", failed: null };

export class NotesWriter extends Watched<NotesWriterState> {
  private abort: AbortController | null = null;
  /** When each Transcript's last writing began: keeping up counts from it. */
  private readonly began = new Map<number, number>();
  private readonly follower = new Follower();

  constructor(private readonly deps: NotesWriterDeps) {
    super(IDLE);
  }

  /** Writes `t`'s notes as `kind`, unless a writing is under way. */
  write(t: Transcript, kind: NoteKind, contextLimit: number | null): Promise<void> {
    if (this.state.writing !== null) return Promise.resolve();
    this.began.set(t.id, (this.deps.now ?? Date.now)());
    return this.run(t, kind, "running", notesRequest(t, kind, contextLimit));
  }

  /** Writes `t`'s recap as `kind` from the whole transcript, unless a writing is under way. */
  writeRecap(t: Transcript, kind: NoteKind, contextLimit: number | null): Promise<void> {
    if (this.state.writing !== null) return Promise.resolve();
    return this.run(t, kind, "recap", recapRequest(t, kind, contextLimit));
  }

  private async run(t: Transcript, kind: NoteKind, form: NotesForm, { request, fromMs }: { request: ConversationRequest; fromMs: number | null }): Promise<void> {
    const now = this.deps.now ?? Date.now;
    const abort = new AbortController();
    this.abort = abort;
    this.set({ writing: t.id, form, text: "", failed: null });
    const covered = t.utterances.length;
    const asked = await askModel(request, abort.signal, this.deps, (text) => this.set({ ...this.state, text }));
    const settled = settle(asked, tidyNotes);
    if (settled === "stopped") return;
    this.abort = null;
    if ("failed" in settled) {
      this.set({ ...IDLE, failed: { transcript: t.id, form, message: settled.failed } });
      return;
    }
    const lines = form === "recap" ? parseRecap(settled.text, kind) : parseNotes(settled.text, kind);
    this.deps.onWritten(t.id, { lines, writtenAs: kind, covered, writtenAt: now(), fromMs }, form);
    this.set(IDLE);
  }

  /** Stops the writing under way, the notes' or the recap's; what there was stays. */
  stop() {
    if (!this.abort) return;
    this.abort.abort();
    this.abort = null;
    this.set(IDLE);
  }

  /**
   * Writes `t`'s notes if keeping up says it is time (notes.ts); `final`, as the listening ends, its recap in their
   * place (recap.ts). A writing of `t`'s notes that failed waits the gap before the next, whatever the cadence says: a
   * first writing or another kind is otherwise due at every look.
   */
  keepUp(t: Transcript, kind: NoteKind, contextLimit: number | null, { final = false } = {}): Promise<void> {
    if (final) return recapDue(t, kind) ? this.writeRecap(t, kind, contextLimit) : Promise.resolve();
    const began = this.began.get(t.id);
    const now = (this.deps.now ?? Date.now)();
    const failed = this.state.failed;
    if (failed?.transcript === t.id && failed.form === "running" && began !== undefined && now - began < LIVE_GAP_MS) return Promise.resolve();
    const due = dueForUpdate({ notes: t.notes && { ...t.notes, kind }, utterances: t.utterances.length, sinceLastMs: began === undefined ? null : now - began });
    return due ? this.write(t, kind, contextLimit) : Promise.resolve();
  }

  /**
   * One look while keeping up, every few seconds: `on` is the Transcript a listening appends to, `null` with none.
   * When its listening ends, its recap is written as soon as no writing is under way.
   */
  look(on: number | null, find: (id: number) => Transcript | undefined, kindOf: (t: Transcript) => NoteKind, contextLimit: number | null): Promise<void> {
    const next = this.follower.next(on, this.state.writing !== null);
    const t = next === null ? undefined : find(next.id);
    return t ? this.keepUp(t, kindOf(t), contextLimit, { final: next!.final }) : Promise.resolve();
  }
}

/** The writer as the page shows it (assist.ts's useWatched): the text streams token by token, and only the notes redraw. */
export type NotesHandle = Pick<NotesWriter, "getState" | "subscribe">;
