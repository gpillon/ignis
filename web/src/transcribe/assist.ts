import { useSyncExternalStore } from "react";
import { abortableSleep, isEngineFull } from "../api/busy.ts";
import type { ConversationRequest, Settings } from "../api/request.ts";
import type { ChunkEvent } from "../api/sse.ts";
import { streamChat, type StreamOptions, type StreamResult } from "../api/stream.ts";
import { linesWithImages } from "./contextImages.ts";
import type { Transcript } from "./transcript.ts";

// What the Transcribe view asks the text model (spec speech/07, Q125, Q134):
// the notes and the suggestions are one shape of request. The system prompt is
// a constant preamble and the task, which stays the same from one request of
// that task to the next; the user turn is the title and the transcript as its
// `.txt`, which a listening only appends to. The engine reuses a prompt only
// whole: a request whose prompt begins with an earlier one's skips that much
// prefill (measured on the 27B, 2026-10-27: 448 of 565 tokens). So each task
// keeps its own chain, the notes theirs and the suggestions theirs; a task
// after the transcript would end every prompt differently and nothing would be
// reused, and two tasks cannot share one.

export const ASSIST_SYSTEM = [
  "You help someone with a conversation while it is being transcribed. The user message holds its transcript; your task follows.",
  "The transcript comes from speech recognition, one utterance a line: `[mm:ss] Speaker: text`. Expect misheard words. Speakers are labels such as \"Speaker 1\", \"Me\" or \"Others\" unless someone named them; a name said aloud may be a speaker's. A line `[mm:ss] (image) ...` is a picture the conversation was looking at from that moment, as it was read. The conversation may still be going on.",
  "Use only what the transcript says: never invent names, numbers or outcomes.",
].join("\n\n");

/** The prompt and what a task writes are allowed of the model's context, beside the transcript. */
const PROMPT_TOKENS = 1_000;
/** Characters a token is taken to hold, low for safety: an Italian token holds about four. */
export const CHARS_PER_TOKEN = 3;

/**
 * The transcript as the model reads it: the title, then the `.txt` (`time` writing each moment), each context image
 * read a line at its moment (contextImages.ts). When the whole would not fit `contextLimit` with `answerTokens` to
 * write, the tail that does, and `fromMs` the moment of its first line.
 */
export function transcriptInput(
  t: Transcript,
  contextLimit: number | null,
  answerTokens: number,
  time?: (ms: number) => string,
): { content: string; fromMs: number | null } {
  const header = `${t.title}\n\n`;
  const budget = contextLimit === null ? Infinity : Math.max(0, (contextLimit - answerTokens - PROMPT_TOKENS) * CHARS_PER_TOKEN) - header.length;
  const lines = linesWithImages(t, time);
  let used = 0;
  let first = lines.length;
  while (first > 0 && used + lines[first - 1].line.length <= budget) used += lines[--first].line.length;
  const kept = lines.slice(first);
  const cut = kept.length < lines.length && lines.slice(0, first).some((l) => l.line !== "");
  return { content: header + kept.map((l) => l.line).join(""), fromMs: cut && kept.length > 0 ? kept[0].atMs : null };
}

/** A request: the preamble and the task as the system prompt, the transcript as the user turn. */
export function assistRequest(settings: Omit<Settings, "systemPrompt">, task: string, transcript: string): ConversationRequest {
  return { settings: { ...settings, systemPrompt: `${ASSIST_SYSTEM}\n\n${task}` }, turns: [{ role: "user", content: transcript }] };
}

export type Asked = { ok: true; text: string } | { ok: false; message: string } | { ok: "stopped" };

export type AskDeps = {
  stream?: (options: StreamOptions) => Promise<StreamResult>;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
};

/** How many times a full engine is waited out, a second each, before the request says so. */
const BUSY_TRIES = 60;

/**
 * One answer of the model, its text as it streams to `onText`; a full engine waited out a second at a time, as an
 * agent's request is. `stopped` once `signal` aborts, whatever came. `onEvent` hears every event as it comes (a tool
 * call, the usage), and `streamId` names the stream a thread of requests continues on the socket.
 */
export async function askModel(
  request: ConversationRequest,
  signal: AbortSignal,
  deps: AskDeps = {},
  onText?: (text: string) => void,
  { onEvent, streamId }: { onEvent?: (event: ChunkEvent) => void; streamId?: string } = {},
): Promise<Asked> {
  const stream = deps.stream ?? streamChat;
  const sleep = deps.sleep ?? abortableSleep;
  for (let attempt = 1; ; attempt++) {
    let text = "";
    const result = await stream({
      request,
      signal,
      ...(streamId === undefined ? {} : { streamId }),
      onEvent: (event) => {
        if (signal.aborted) return;
        onEvent?.(event);
        if (event.kind !== "content") return;
        text += event.text;
        onText?.(text);
      },
    });
    if (signal.aborted) return { ok: "stopped" };
    if (!result.ok && isEngineFull(result.message) && attempt < BUSY_TRIES) {
      await sleep(1_000, signal);
      if (signal.aborted) return { ok: "stopped" };
      continue;
    }
    return result.ok ? { ok: true, text } : { ok: false, message: result.message };
  }
}

/**
 * A listening followed from one look to the next, every few seconds: the Transcript it appends to, and once its
 * listening ends, that Transcript owed one last look as soon as nothing is under way.
 */
export class Follower {
  private followed: number | null = null;
  private ended: number | null = null;

  /** The Transcript to look at now, `on` or one whose listening ended (`final`); none while `busy`. */
  next(on: number | null, busy: boolean): { id: number; final: boolean } | null {
    if (this.followed !== null && this.followed !== on) this.ended = this.followed;
    this.followed = on;
    if (busy) return null;
    const final = this.ended !== null;
    const id = this.ended ?? on;
    this.ended = null;
    return id === null ? null : { id, final };
  }
}

/** A state the page watches: the notes' writer's, the look's. */
export class Watched<S> {
  private readonly listeners = new Set<() => void>();

  constructor(protected state: S) {}

  getState = (): S => this.state;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  protected set(state: S) {
    this.state = state;
    for (const listener of this.listeners) listener();
  }
}

/** A watched state, for what shows it: only that redraws as it changes. */
export function useWatched<S>(handle: Pick<Watched<S>, "getState" | "subscribe">): S {
  return useSyncExternalStore(handle.subscribe, handle.getState, handle.getState);
}

/**
 * What a side request's answer comes to: stopped (the Stop said so already), the text tidied, or why there is none,
 * "The model wrote nothing." for an answer that tidied to nothing.
 */
export function settle(asked: Asked, tidy: (text: string) => string = (text) => text.trim()): "stopped" | { text: string } | { failed: string } {
  if (asked.ok === "stopped") return "stopped";
  if (!asked.ok) return { failed: asked.message };
  const text = tidy(asked.text);
  return text === "" ? { failed: "The model wrote nothing." } : { text };
}
