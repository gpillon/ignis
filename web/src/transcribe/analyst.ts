import { type AskDeps, askModel, assistRequest, Follower, settle, Watched } from "./assist.ts";
import { ANALYSIS_GAP_MS, ANALYSIS_MAX_TOKENS, ANALYSIS_SETTINGS, ANALYSIS_TASK, type Analysis, dueForAnalysis, readAnalysis } from "./suggestions.ts";
import { taskInput } from "./taskContext.ts";
import type { Transcript } from "./transcript.ts";

// The look for suggestions (spec speech/07, Q132): the text model reads a
// Transcript and says its topic, a title and the terms to spell right
// (suggestions.ts), on Suggest now or, while listening, about every minute and
// a half when something new was said, and once more at the end. One look at a
// time, on an agent lane with thinking off, as the notes' writing; a look
// that failed waits the gap before the next. What it found goes to `onFound`
// with the Transcript it read, whichever is open by then.

export type AnalystState = {
  /** The Transcript being read. */
  looking: number | null;
  /** When the next look while listening may come, by Transcript (epoch ms): the last look's start and the gap. */
  nextAt: Record<number, number>;
  /** The last look that failed, and why. */
  failed: { transcript: number; message: string } | null;
};

export type AnalystDeps = AskDeps & {
  /** What a look found, and up to where it read (the end of the last utterance it had, ms on the Transcript's clock). */
  onFound: (transcript: number, found: Analysis, readUntilMs: number) => void;
  now?: () => number;
};

const IDLE: AnalystState = { looking: null, nextAt: {}, failed: null };

export class Analyst extends Watched<AnalystState> {
  private abort: AbortController | null = null;
  /** When each Transcript's last look began, and up to where it read (an utterance deleted since changes nothing). */
  private readonly began = new Map<number, number>();
  private readonly readUntil = new Map<number, number>();
  private readonly follower = new Follower();

  constructor(private readonly deps: AnalystDeps) {
    super(IDLE);
  }

  /** Reads `t` now, unless a look is under way. */
  async analyze(t: Transcript, contextLimit: number | null): Promise<void> {
    if (this.state.looking !== null) return;
    const abort = new AbortController();
    this.abort = abort;
    const began = (this.deps.now ?? Date.now)();
    this.began.set(t.id, began);
    const readUntilMs = Math.max(0, ...t.utterances.map((u) => u.endMs));
    this.readUntil.set(t.id, readUntilMs);
    this.set({ looking: t.id, nextAt: { ...this.state.nextAt, [t.id]: began + ANALYSIS_GAP_MS }, failed: null });
    const { content } = taskInput(t, contextLimit, ANALYSIS_MAX_TOKENS);
    const asked = await askModel(assistRequest(ANALYSIS_SETTINGS, ANALYSIS_TASK, content), abort.signal, this.deps);
    const settled = settle(asked, (text) => text);
    if (settled === "stopped") return;
    this.abort = null;
    if ("failed" in settled) {
      this.set({ ...this.state, looking: null, failed: { transcript: t.id, message: settled.failed } });
      return;
    }
    this.deps.onFound(t.id, readAnalysis(settled.text), readUntilMs);
    this.set({ ...this.state, looking: null });
  }

  stop() {
    if (!this.abort) return;
    this.abort.abort();
    this.abort = null;
    this.set({ ...this.state, looking: null });
  }

  /** Reads `t` if keeping up says it is time; `final` as the listening ends. A failed look waits the gap. */
  keepUp(t: Transcript, contextLimit: number | null, { final = false } = {}): Promise<void> {
    const began = this.began.get(t.id);
    const sinceLastMs = began === undefined ? null : (this.deps.now ?? Date.now)() - began;
    if (!final && this.state.failed?.transcript === t.id && sinceLastMs !== null && sinceLastMs < ANALYSIS_GAP_MS) return Promise.resolve();
    const until = this.readUntil.get(t.id);
    const analyzed = until === undefined ? 0 : t.utterances.filter((u) => u.endMs <= until).length;
    const due = dueForAnalysis({ utterances: t.utterances.length, analyzed, sinceLastMs, final });
    return due ? this.analyze(t, contextLimit) : Promise.resolve();
  }

  /** One look while listening (assist.ts's Follower): `on`, the Transcript a listening appends to. */
  look(on: number | null, find: (id: number) => Transcript | undefined, contextLimit: number | null): Promise<void> {
    const next = this.follower.next(on, this.state.looking !== null);
    const t = next === null ? undefined : find(next.id);
    return t ? this.keepUp(t, contextLimit, { final: next!.final }) : Promise.resolve();
  }

}

/** The look as the page shows it (assist.ts's useWatched). */
export type AnalystHandle = Pick<Analyst, "getState" | "subscribe">;
