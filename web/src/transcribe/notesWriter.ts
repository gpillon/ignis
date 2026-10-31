import type { ConversationRequest, Turn } from "../api/request.ts";
import type { ToolCall } from "../api/sse.ts";
import { ASSIST_SYSTEM, type AskDeps, askModel, CHARS_PER_TOKEN, Follower, settle, Watched } from "./assist.ts";
import { shownOf, shownSince } from "./contextImages.ts";
import { minutesClock } from "./export.ts";
import { dueForUpdate, LIVE_GAP_MS, type NoteKind, noNotes, type Notes, NOTES_MAX_TOKENS, NOTES_SETTINGS, RUNNING_NOTES, tidyNotes, type Written } from "./notes.ts";
import { parseRecap, recapDue, recapRequest } from "./recap.ts";
import { applyCall, changesSince, fromScratch, laterTurn, notesState, restartAt, runningTask, runningTools, type Told, toldOf } from "./running.ts";
import { taskInput } from "./taskContext.ts";
import type { Transcript } from "./transcript.ts";

// The notes' writer (spec speech/07, Q124-Q127; spec #428): one writing at a
// time, on the text model loaded, with no model named, so a request never
// switches one. A full engine is waited out a second at a time, as an agent's
// request is: the notes are a side request on an agent lane, behind the chat.
// The page holds one, so a writing goes on while the view is hidden.
//
// The running notes are kept by a thread per Transcript (running.ts, ADR
// 0076): its first turn is the context, the title, the transcript so far and
// the notes as they stand; each later update appends what was said since, and
// what the user changed in the transcript meanwhile (a diff against what the
// thread told of it, so a change made during an update goes into the next),
// and runs a few requests, the page applying each call the moment it comes and
// sending its result back with the next. The thread lives here, not in the
// Transcript, so a reload starts it again from the notes, as a kind change, a
// Rewrite from scratch and a prompt past the threshold do. A Stop or a failure
// keeps every call applied, and the next update starts from the notes.
//
// It writes the recap too (recap.ts), on a press and when a listening ends:
// one writing over the whole transcript, its text streaming into the state.

/** Which notes a writing writes: the running notes, or the recap. */
export type NotesForm = "running" | "recap";

export type NotesWriterState = {
  /** The Transcript whose notes are being written. */
  writing: number | null;
  /** Which of them: the running notes when none is. */
  form: NotesForm;
  /** What the recap's writing has written so far. */
  text: string;
  /** The last writing that failed, which notes it was, and why. */
  failed: { transcript: number; form: NotesForm; message: string } | null;
};

export type { Written };

export type NotesWriterDeps = AskDeps & {
  /** A change to a Transcript's running notes: a call applied, or a restart's. The page applies it to the notes it holds. */
  onNotes: (transcript: number, change: (notes: Notes | undefined) => Notes) => void;
  /** A recap written, for the Transcript it started on. */
  onRecap: (transcript: number, written: Written) => void;
  now?: () => number;
};

const IDLE: NotesWriterState = { writing: null, form: "running", text: "", failed: null };

/**
 * A Transcript's notes thread: the kind it keeps, its turns, what each utterance was told as, the context images whose
 * reading it told, and its last prompt.
 */
type Thread = { kind: NoteKind; turns: Turn[]; told: Map<number, Told>; shown: Set<number>; promptTokens: number };

/** The stream a Transcript's thread goes on, so the socket continues it. */
const streamOf = (transcript: number) => `transcribe-notes-${transcript}`;

export class NotesWriter extends Watched<NotesWriterState> {
  private abort: AbortController | null = null;
  /** When each Transcript's last writing began: keeping up counts from it. */
  private readonly began = new Map<number, number>();
  private readonly follower = new Follower();
  /** Each Transcript's notes thread, while it holds: an update under way takes it, and gives it back whole. */
  private readonly threads = new Map<number, Thread>();

  constructor(private readonly deps: NotesWriterDeps) {
    super(IDLE);
  }

  /** Updates `t`'s running notes as `kind`, unless a writing is under way; `scratch`, from scratch: the model's lines dropped first. */
  write(t: Transcript, kind: NoteKind, contextLimit: number | null, { scratch = false } = {}): Promise<void> {
    if (this.state.writing !== null) return Promise.resolve();
    this.began.set(t.id, (this.deps.now ?? Date.now)());
    return this.update(t, kind, contextLimit, scratch);
  }

  /** Writes `t`'s running notes again from scratch: the model's lines dropped, the user's kept, the thread started over. */
  rewrite(t: Transcript, kind: NoteKind, contextLimit: number | null): Promise<void> {
    return this.write(t, kind, contextLimit, { scratch: true });
  }

  /** Writes `t`'s recap as `kind` from the whole transcript, unless a writing is under way. */
  writeRecap(t: Transcript, kind: NoteKind, contextLimit: number | null): Promise<void> {
    if (this.state.writing !== null) return Promise.resolve();
    return this.recap(t, kind, recapRequest(t, kind, contextLimit));
  }

  /** The thread `t`'s update goes on: the one it has, its next turn appended, or a new one; `null` with nothing to tell. */
  private threadFor(t: Transcript, kind: NoteKind, notes: Notes, contextLimit: number | null, restart: boolean): { thread: Thread; fromMs: number | null } | null {
    const held = this.threads.get(t.id);
    this.threads.delete(t.id);
    if (!restart && held && held.kind === kind && held.promptTokens <= restartAt(contextLimit)) {
      // What was said since the last turn, and what the user changed meanwhile: `t` against what the thread told.
      const { said, events } = changesSince(held.told, t);
      // The context images read meanwhile (#437), each a change at its moment.
      events.push(...shownSince(held.shown, t));
      const turn = laterTurn(said, events);
      const told = toldOf(t, held.told);
      const shown = shownOf(t);
      if (turn === "") {
        this.threads.set(t.id, { ...held, told, shown });
        return null;
      }
      return { thread: { ...held, told, shown, turns: [...held.turns, { role: "user", content: turn }] }, fromMs: notes.fromMs };
    }
    // A new thread: the transcript so far, from where its tail fits half the threshold, then the notes as they stand.
    const state = notesState(notes, kind);
    const reserve = NOTES_MAX_TOKENS + Math.ceil(state.length / CHARS_PER_TOKEN);
    const { content, fromMs } = taskInput(t, Math.floor(restartAt(contextLimit) / 2), reserve, minutesClock);
    return { thread: { kind, turns: [{ role: "user", content: `${content}\n${state}` }], told: toldOf(t), shown: shownOf(t), promptTokens: 0 }, fromMs };
  }

  private async update(t: Transcript, kind: NoteKind, contextLimit: number | null, scratch: boolean): Promise<void> {
    const now = this.deps.now ?? Date.now;
    const abort = new AbortController();
    this.abort = abort;
    this.set({ writing: t.id, form: "running", text: "", failed: null });
    const covered = t.utterances.length;
    let notes = t.notes ?? noNotes(kind);
    // Another kind, or Rewrite from scratch: the model's lines go before the thread starts again.
    const restart = scratch || (notes.writtenAs !== null && notes.writtenAs !== kind);
    if (restart) {
      notes = fromScratch(notes, kind);
      this.deps.onNotes(t.id, (n) => fromScratch(n, kind));
    }
    const stamp = (at: number, fromMs: number | null) => (n: Notes): Notes => ({ ...n, writtenAs: kind, writtenAt: at, covered, fromMs });
    const started = this.threadFor(t, kind, notes, contextLimit, restart);
    if (started === null) {
      // Nothing new to tell the model: the notes are as current as they can be.
      this.deps.onNotes(t.id, (n) => stamp(now(), notes.fromMs)(n ?? notes));
      this.abort = null;
      this.set(IDLE);
      return;
    }
    let { thread } = started;
    const settings = { ...NOTES_SETTINGS, systemPrompt: `${ASSIST_SYSTEM}\n\n${runningTask(kind)}` };
    const extras = { tools: runningTools(kind) };
    let idle = 0;
    for (let round = 0; round < RUNNING_NOTES.maxRounds; round++) {
      const request: ConversationRequest = { settings, turns: thread.turns, extras };
      const calls: ToolCall[] = [];
      const results: Turn[] = [];
      let unchanged = 0;
      let prompt: number | null = null;
      const asked = await askModel(request, abort.signal, this.deps, undefined, {
        streamId: streamOf(t.id),
        onEvent: (event) => {
          if (event.kind === "usage") prompt = event.usage.prompt_tokens;
          if (event.kind !== "tool_call") return;
          // Each call is applied as it comes, so the lines change one at a time; the page applies the same call to the
          // notes it holds, so a change made there meanwhile is kept.
          const { call } = event;
          const applied = applyCall(notes, kind, call);
          calls.push(call);
          results.push({ role: "tool", content: applied.result, toolCallId: call.id });
          if (applied.outcome === "unchanged") unchanged++;
          if (applied.outcome !== "applied") return;
          const changed = stamp(now(), started.fromMs);
          notes = changed(applied.notes);
          this.deps.onNotes(t.id, (n) => {
            const there = applyCall(n ?? noNotes(kind), kind, call);
            return there.outcome === "applied" ? changed(there.notes) : there.notes;
          });
        },
      });
      if (asked.ok === "stopped") return;
      if (!asked.ok) {
        this.abort = null;
        this.set({ ...IDLE, failed: { transcript: t.id, form: "running", message: asked.message } });
        return;
      }
      const answer: Turn = { role: "assistant", content: asked.text, ...(calls.length > 0 ? { toolCalls: calls } : {}) };
      const turns = [...thread.turns, answer, ...results];
      thread = { ...thread, turns, promptTokens: prompt ?? Math.ceil((settings.systemPrompt.length + turns.reduce((n, turn) => n + turn.content.length, 0)) / CHARS_PER_TOKEN) };
      if (calls.length === 0) break;
      // Rounds that change nothing, in a row, end the update; past the cap, what is left goes to the next one.
      idle = unchanged === calls.length ? idle + 1 : 0;
      if (idle >= RUNNING_NOTES.idleRounds) break;
    }
    this.abort = null;
    this.deps.onNotes(t.id, (n) => stamp(now(), started.fromMs)(n ?? notes));
    this.threads.set(t.id, thread);
    this.set(IDLE);
  }

  private async recap(t: Transcript, kind: NoteKind, { request, fromMs }: { request: ConversationRequest; fromMs: number | null }): Promise<void> {
    const now = this.deps.now ?? Date.now;
    const abort = new AbortController();
    this.abort = abort;
    this.set({ writing: t.id, form: "recap", text: "", failed: null });
    const covered = t.utterances.length;
    const asked = await askModel(request, abort.signal, this.deps, (text) => this.set({ ...this.state, text }));
    const settled = settle(asked, tidyNotes);
    if (settled === "stopped") return;
    this.abort = null;
    if ("failed" in settled) {
      this.set({ ...IDLE, failed: { transcript: t.id, form: "recap", message: settled.failed } });
      return;
    }
    this.deps.onRecap(t.id, { lines: parseRecap(settled.text, kind), writtenAs: kind, covered, writtenAt: now(), fromMs });
    this.set(IDLE);
  }

  /** Stops the writing under way, the notes' or the recap's; what there was, and every call applied, stays. */
  stop() {
    if (!this.abort) return;
    this.abort.abort();
    this.abort = null;
    this.set(IDLE);
  }

  /**
   * Updates `t`'s notes if keeping up says it is time (notes.ts); `final`, as the listening ends, writes its recap
   * instead (recap.ts). An update of `t`'s notes that failed waits the gap before the next, whatever the cadence says:
   * a first update or another kind is otherwise due at every look.
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

/** The writer as the page shows it (assist.ts's useWatched): the recap's text streams token by token, and only the notes redraw. */
export type NotesHandle = Pick<NotesWriter, "getState" | "subscribe">;
