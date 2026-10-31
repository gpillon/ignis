import { type AskDeps, askModel, Follower, settle, Watched } from "./assist.ts";
import { dueForUpdate, LIVE_GAP_MS, type NoteKind, notesRequest, parseNotes, tidyNotes, type Written } from "./notes.ts";
import type { Transcript } from "./transcript.ts";

// The notes' writer (spec speech/07, Q124-Q127): one writing at a time, on
// the text model loaded, with no model named, so a request never switches
// one. Its text streams into the state while it writes; the notes land on the
// Transcript the writing started on (`onWritten`), whichever is open by then,
// and a Stop or a failure leaves the notes there were. A full engine is
// waited out a second at a time, as an agent's request is: the notes are a
// side request on an agent lane, behind the chat. The page holds one, so a
// writing goes on while the view is hidden. What lands is the model's
// Markdown read as lines (notes.ts's parseNotes).

export type NotesWriterState = {
  /** The Transcript whose notes are being written. */
  writing: number | null;
  /** What the writing has written so far. */
  text: string;
  /** The last writing that failed, and why. */
  failed: { transcript: number; message: string } | null;
};

export type { Written };

export type NotesWriterDeps = AskDeps & {
  onWritten: (transcript: number, written: Written) => void;
  now?: () => number;
};

const IDLE: NotesWriterState = { writing: null, text: "", failed: null };

export class NotesWriter extends Watched<NotesWriterState> {
  private abort: AbortController | null = null;
  /** When each Transcript's last writing began: keeping up counts from it. */
  private readonly began = new Map<number, number>();
  private readonly follower = new Follower();

  constructor(private readonly deps: NotesWriterDeps) {
    super(IDLE);
  }

  /** Writes `t`'s notes as `kind`, unless a writing is under way. */
  async write(t: Transcript, kind: NoteKind, contextLimit: number | null): Promise<void> {
    if (this.state.writing !== null) return;
    const now = this.deps.now ?? Date.now;
    const abort = new AbortController();
    this.abort = abort;
    this.began.set(t.id, now());
    this.set({ writing: t.id, text: "", failed: null });
    const { request, fromMs } = notesRequest(t, kind, contextLimit);
    const covered = t.utterances.length;
    const asked = await askModel(request, abort.signal, this.deps, (text) => this.set({ ...this.state, text }));
    const settled = settle(asked, tidyNotes);
    if (settled === "stopped") return;
    this.abort = null;
    if ("failed" in settled) {
      this.set({ writing: null, text: "", failed: { transcript: t.id, message: settled.failed } });
      return;
    }
    this.deps.onWritten(t.id, { lines: parseNotes(settled.text, kind), writtenAs: kind, covered, writtenAt: now(), fromMs });
    this.set(IDLE);
  }

  /** Stops the writing under way; the notes there were stay. */
  stop() {
    if (!this.abort) return;
    this.abort.abort();
    this.abort = null;
    this.set(IDLE);
  }

  /**
   * Writes `t`'s notes if keeping up says it is time (notes.ts); `final` as the listening ends. A writing of `t` that
   * failed waits the gap before the next, whatever the cadence says: a first writing or another kind is otherwise due
   * at every look.
   */
  keepUp(t: Transcript, kind: NoteKind, contextLimit: number | null, { final = false } = {}): Promise<void> {
    const began = this.began.get(t.id);
    const now = (this.deps.now ?? Date.now)();
    if (!final && this.state.failed?.transcript === t.id && began !== undefined && now - began < LIVE_GAP_MS) return Promise.resolve();
    const due = dueForUpdate({ notes: t.notes && { ...t.notes, kind }, utterances: t.utterances.length, sinceLastMs: began === undefined ? null : now - began, final });
    return due ? this.write(t, kind, contextLimit) : Promise.resolve();
  }

  /**
   * One look while keeping up, every few seconds: `on` is the Transcript a listening appends to, `null` with none.
   * When its listening ends, the notes are written once more as soon as no writing is under way.
   */
  look(on: number | null, find: (id: number) => Transcript | undefined, kindOf: (t: Transcript) => NoteKind, contextLimit: number | null): Promise<void> {
    const next = this.follower.next(on, this.state.writing !== null);
    const t = next === null ? undefined : find(next.id);
    return t ? this.keepUp(t, kindOf(t), contextLimit, { final: next!.final }) : Promise.resolve();
  }
}

/** The writer as the page shows it (assist.ts's useWatched): the text streams token by token, and only the notes redraw. */
export type NotesHandle = Pick<NotesWriter, "getState" | "subscribe">;
