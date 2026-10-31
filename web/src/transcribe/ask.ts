import type { ConversationRequest, Settings } from "../api/request.ts";
import { type AskDeps, askModel, assistRequest, settle, Watched } from "./assist.ts";
import { clock } from "./export.ts";
import { type MissedFrom, missedSince } from "./missed.ts";
import { taskInput } from "./taskContext.ts";
import { heardUntil, type Transcript, type Utterance } from "./transcript.ts";

// Ask about this conversation (spec speech/07, Q139): a question about the
// Transcript, answered by the text model from the transcript alone, streamed,
// and kept with the Transcript, latest first. A person waits on it, so the
// interactive lane, thinking off: a quick answer, its moments as [mm:ss],
// which the view turns into links to them (Q141). The answer stands in the
// sheet where it was asked, after the last utterance it read (the owner's
// pick after trying it in the Notes tab).

export const ASK_TASK = [
  "Task: answer the user's question about the conversation, from the transcript alone, briefly, in the language the context says to write in, in Markdown.",
  "Point at the moments you draw on by their time, as [mm:ss]. When the transcript does not say, say so.",
].join("\n");

export const ASK_MAX_TOKENS = 800;

export const ASK_SETTINGS: Omit<Settings, "systemPrompt"> = {
  temperature: 0.3,
  topP: 0.9,
  maxTokens: ASK_MAX_TOKENS,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "interactive",
};

/**
 * A question asked and its answer, as the Transcript keeps them; `untilMs` is the end of the last utterance the
 * answer read, its place in the sheet (an answer kept before it was there stands at the foot).
 */
export type Answered = { id: number; question: string; answer: string; at: number; untilMs?: number };

export function askRequest(t: Transcript, question: string, contextLimit: number | null): ConversationRequest {
  const { content } = taskInput(t, contextLimit, ASK_MAX_TOKENS);
  return assistRequest(ASK_SETTINGS, ASK_TASK, `${content}\nQuestion: ${question.trim()}`);
}

/** The questions a tap asks: what the user sees, and what the model is sent; `missed`, What did I miss?. */
export type QuickQuestion = { label: string; detail?: string; question: string; sent: string; missed?: true };

/**
 * The quick questions of a call. Written in English for the page, their answers asked in the conversation's
 * language. "What did I miss?" reads from `from` (missed.ts: the last five minutes unless the user was away or
 * picked a moment) and names its first utterance by its time, a line the model finds as it is written: asked from
 * the end less five minutes, a time that shares the last line's seconds, the model took one for the other and said
 * nothing came since.
 */
export function quickQuestions(t: Transcript, from: MissedFrom | null = null): QuickQuestion[] {
  const own = "Answer in the language of the conversation.";
  const missed = missedSince(t.utterances, from);
  const since = clock(missed.startMs);
  return [
    {
      label: "What did I miss?",
      detail: missed.detail,
      question: `What was said since ${since}?`,
      sent: `What was said from [${since}] on? The main points, briefly. ${own}`,
      missed: true,
    },
    { label: "Decisions", question: "What has been decided so far, and what is still open?", sent: `What has been decided so far, and what is still open? ${own}` },
    { label: "Who does what", question: "Who agreed to do what, and by when?", sent: `Who agreed to do what, and by when? ${own}` },
  ];
}

/**
 * Where an answer stands among the utterances: after the last it read, the last that had ended by `untilMs`; at the
 * foot for one kept before answers had a place. In time order by start, a call's two lanes can hold an utterance
 * that began before the question and ended after it: it stands above the answer, unread, not the other way round.
 */
export function answerAt(utterances: Utterance[], untilMs: number | undefined): number {
  if (untilMs === undefined) return utterances.length;
  return utterances.reduce((at, u, i) => (u.endMs <= untilMs ? i + 1 : at), 0);
}

/** `answered`'s next, latest first. */
export function withAnswer(answered: Answered[] | undefined, question: string, answer: string, at: number, untilMs?: number): Answered[] {
  const all = answered ?? [];
  return [{ id: Math.max(0, ...all.map((a) => a.id)) + 1, question, answer, at, ...(untilMs === undefined ? {} : { untilMs }) }, ...all];
}

/** A stored Transcript's answers, those this page can read; a place it cannot read is dropped, not the answer. */
export function readAnswered(value: unknown): Answered[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value
    .filter(
      (a): a is Answered =>
        typeof a === "object" && a !== null && typeof a.id === "number" && typeof a.question === "string" && typeof a.answer === "string" && typeof a.at === "number",
    )
    .map(({ untilMs, ...a }) => (typeof untilMs === "number" && Number.isFinite(untilMs) ? { ...a, untilMs } : a));
}

export type AskerState = {
  /** The Transcript asked about, the question, up to where it reads, and the answer so far. */
  transcript: number | null;
  question: string;
  untilMs: number;
  text: string;
  writing: boolean;
  failed: string | null;
};

const IDLE: AskerState = { transcript: null, question: "", untilMs: 0, text: "", writing: false, failed: null };

export type AskerDeps = AskDeps & { onAnswered: (transcript: number, question: string, answer: string, untilMs: number) => void };

/** One question at a time; the answer streams, and lands on the Transcript asked about. */
export class Asker extends Watched<AskerState> {
  private abort: AbortController | null = null;

  constructor(private readonly deps: AskerDeps) {
    super(IDLE);
  }

  /** Asks `question` about `t`; the model is sent `sent` in its place when given (a quick question's). */
  async ask(t: Transcript, question: string, contextLimit: number | null, sent?: string): Promise<void> {
    if (this.state.writing || question.trim() === "") return;
    const abort = new AbortController();
    this.abort = abort;
    this.set({ transcript: t.id, question: question.trim(), untilMs: heardUntil(t.utterances), text: "", writing: true, failed: null });
    const asked = await askModel(askRequest(t, sent ?? question, contextLimit), abort.signal, this.deps, (text) => this.set({ ...this.state, text }));
    const settled = settle(asked);
    if (settled === "stopped") return;
    this.abort = null;
    if ("failed" in settled) {
      this.set({ ...this.state, writing: false, text: "", failed: settled.failed });
      return;
    }
    this.deps.onAnswered(t.id, this.state.question, settled.text, this.state.untilMs);
    this.set(IDLE);
  }

  stop() {
    this.abort?.abort();
    this.abort = null;
    this.set(IDLE);
  }
}

export type AskerHandle = Pick<Asker, "getState" | "subscribe">;
