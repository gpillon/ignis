import type { ConversationRequest, Settings } from "../api/request.ts";
import { type AskDeps, askModel, assistRequest, settle, Watched } from "./assist.ts";
import { toText } from "./export.ts";
import { contextBlock } from "./taskContext.ts";
import type { Transcript, Utterance } from "./transcript.ts";

// Rewrite (spec speech/07, Q138): the user asks the text model to rebuild one
// utterance the speech model misheard, from the context the Transcript is
// told, the conversation around it, and how the misheard words sound. The
// answer is a proposal, the words it changed shown, which the user accepts or
// discards; accepted, the utterance keeps its text as heard, as a fix does.
// Asked by a person who waits, it goes on the interactive lane, with a little
// thinking: on the owner's own misheard line, thinking off rebuilt the name
// two times in three and once changed a dialect word, thinking low three times
// in three, the dialect kept, in about 2 s (probed on the 27B, 2026-10-27).

export const REWRITE_TASK = [
  "Task: the user names one utterance that speech recognition misheard. Write what was actually said, from the conversation around it, the context given, and how the misheard words sound when spoken: a name or product heard as other words or a wrong spelling, a word heard as one that sounds alike.",
  "Keep its language and every word that is plausible as heard; change only what was misheard. Never add, drop or rephrase content.",
  "Write only the corrected utterance, without its time or speaker, and nothing else. If nothing was misheard, write it unchanged.",
].join("\n");

export const REWRITE_SETTINGS: Omit<Settings, "systemPrompt"> = {
  temperature: 0.3,
  topP: 0.9,
  maxTokens: 1_500,
  reasoningEffort: "low",
  thinkingBudget: null,
  laneTag: "interactive",
};

/** How much of the conversation goes with it: utterances before, and after. */
const BEFORE = 20;
const AFTER = 5;

/** The request: the context told, the conversation around the utterance, the utterance named last. */
export function rewriteRequest(t: Transcript, utterance: number): ConversationRequest | null {
  const index = t.utterances.findIndex((u) => u.id === utterance);
  if (index === -1) return null;
  const around = { ...t, utterances: t.utterances.slice(Math.max(0, index - BEFORE), index + AFTER + 1) };
  const line = toText({ ...t, utterances: [t.utterances[index]] }).trim();
  const content = `${contextBlock(t)}Transcript, around the utterance:
${toText(around).trimEnd()}

Misheard utterance: ${line}`;
  return assistRequest(REWRITE_SETTINGS, REWRITE_TASK, content);
}

/** The model's answer as a text: no quotes, no label, time or `speaker` name it may have repeated. */
export function tidyRewrite(answer: string, speaker: string | null): string {
  let text = answer
    .trim()
    .replace(/^(misheard utterance|corrected utterance)\s*:\s*/i, "")
    .replace(/^\[~?\d{1,2}:\d{2}(?::\d{2})?\]\s*/, "");
  if (speaker !== null && text.toLocaleLowerCase().startsWith(`${speaker.toLocaleLowerCase()}:`)) text = text.slice(speaker.length + 1);
  return text
    .trim()
    .replace(/^["“«](.*)["”»]$/s, "$1")
    .trim();
}

const bare = (word: string) => word.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");

/** A rewrite accepted: the utterance's new text, its text as heard kept, the words it put in marked, as a fix's. */
export function acceptRewrite(t: Transcript, utterance: number, text: string): Transcript {
  return {
    ...t,
    utterances: t.utterances.map((u) => {
      if (u.id !== utterance || text === u.text) return u;
      const added = wordDiff(u.text, text)
        .filter((part) => part.kind === "added")
        .map((part) => bare(part.text))
        .filter((word) => word !== "");
      return { ...u, text, heard: u.heard ?? u.text, fixed: [...new Set([...(u.fixed ?? []), ...added])] };
    }),
  };
}

/** A word of a rewrite as the proposal shows it. */
export type DiffPart = { text: string; kind: "same" | "added" | "removed" };

/** The words `after` changed from `before`: kept, added, removed, in reading order (a longest common subsequence). */
export function wordDiff(before: string, after: string): DiffPart[] {
  const a = before.split(/\s+/).filter(Boolean);
  const b = after.split(/\s+/).filter(Boolean);
  const lcs = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--) lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  const parts: DiffPart[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) parts.push({ text: a[i++], kind: "same" }), j++;
    else if (j < b.length && (i === a.length || lcs[i][j + 1] >= lcs[i + 1][j])) parts.push({ text: b[j++], kind: "added" });
    else parts.push({ text: a[i++], kind: "removed" });
  }
  return parts;
}

export type RewriterState = {
  /** The utterance being rewritten or proposed for, and its Transcript. */
  transcript: number | null;
  utterance: number | null;
  writing: boolean;
  /** The model's text for it, once it came. */
  proposal: string | null;
  failed: string | null;
};

const IDLE: RewriterState = { transcript: null, utterance: null, writing: false, proposal: null, failed: null };

/** One rewrite at a time, the user's; a new one replaces a proposal waiting. */
export class Rewriter extends Watched<RewriterState> {
  private abort: AbortController | null = null;

  constructor(private readonly deps: AskDeps = {}) {
    super(IDLE);
  }

  async rewrite(t: Transcript, u: Utterance): Promise<void> {
    if (this.state.writing) return;
    const request = rewriteRequest(t, u.id);
    if (!request) return;
    const abort = new AbortController();
    this.abort = abort;
    this.set({ transcript: t.id, utterance: u.id, writing: true, proposal: null, failed: null });
    const asked = await askModel(request, abort.signal, this.deps);
    const speaker = t.speakers.find((s) => s.id === u.speaker)?.name ?? null;
    const settled = settle(asked, (text) => tidyRewrite(text, speaker));
    if (settled === "stopped") return;
    this.abort = null;
    this.set({ ...this.state, writing: false, ...("failed" in settled ? { failed: settled.failed } : { proposal: settled.text }) });
  }

  /** Stops a rewrite under way, or closes the proposal or the failure shown. */
  close() {
    this.abort?.abort();
    this.abort = null;
    this.set(IDLE);
  }
}

export type RewriterHandle = Pick<Rewriter, "getState" | "subscribe">;
