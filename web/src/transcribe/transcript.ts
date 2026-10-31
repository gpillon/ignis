// A Transcript (spec speech/07): the record of one conversation heard in the
// Transcribe view, its utterances in time order, each with its times, its
// speaker and its text, and never its audio (GLOSSARY.md).
//
// - **Speakers** are the page's labels, not the diarizer's channels. A
//   session's channel (`stream:index`) maps to a label the first time it
//   speaks, numbered Speaker 1, 2… in that order; a new session (a Resume, a
//   Continue) numbers its channels afresh, so a voice that comes back is a new
//   speaker until the user merges it. On a call the microphone is Me and the
//   shared audio, without its speakers, Others: one label each, whatever the
//   session.
// - **Corrections** are mappings in the page and nothing goes back to the
//   server (Q109): a rename names a label, past and future; a merge maps the
//   channels of both to one label, so the future utterances of either follow;
//   a move and a split are one-off. A split has no audio to read, so its time
//   is proportional to the characters and marked approximate (Q98).
//
// Every function returns a new Transcript and leaves the one it was given alone.

import type { Answered } from "./ask.ts";
import type { CallContext } from "./context.ts";
import type { ContextImage } from "./contextImages.ts";
import type { Notes } from "./notes.ts";
import type { Suggestions } from "./suggestions.ts";

export type SourceKind = "microphone" | "call" | "file";

/** A label: a voice the diarizer told apart, or a call's Me and Others. */
export type Speaker = { id: number; name: string; role: "voice" | "me" | "others" };

export type Utterance = {
  /** Unique within its Transcript. */
  id: number;
  /** From the start of the Transcript's clock. */
  startMs: number;
  endMs: number;
  /** A split guessed the time (Q98): written with `~`. */
  approxStart?: true;
  approxEnd?: true;
  /** Its label; `null` with no speakers. */
  speaker: number | null;
  text: string;
  /** The language the model heard it in, by ISO code, when the server says (Q131). */
  language?: string;
  /** Its text as it was before a fix (Q133): absent while none was accepted. */
  heard?: string;
  /** The words fixes put in it, to mark them. */
  fixed?: string[];
};

export type Transcript = {
  id: number;
  /** "" until its first listening titles it, unless the user named it first. */
  title: string;
  /** Its first listening's source; `null` before one. */
  source: SourceKind | null;
  /** The wall-clock start of its first live listening, in epoch ms; `null` for a file's. */
  startedAt: number | null;
  /** The listened time so far: where the next listening's clock starts (a Continue, Q97). */
  clockMs: number;
  utterances: Utterance[];
  speakers: Speaker[];
  /** A channel, `stream:index`, or `me` and `others` on a call, to its label. */
  channels: Record<string, number>;
  /** The voices numbered so far: the next is Speaker `voices + 1`, whatever was merged or renamed since. */
  voices: number;
  /** Its notes (notes.ts): absent until a kind is picked or notes are written. */
  notes?: Notes;
  /** Its recap (recap.ts), kept as lines as the notes are, its kind the one it was written as: absent until one is. */
  recap?: Notes;
  /** What its transcription is told (context.ts), as last applied: absent until it is. */
  context?: CallContext;
  /** What was suggested for it, and what became of each (suggestions.ts). */
  suggestions?: Suggestions;
  /** Up to where the text model's look last read it (Q140): the end of the last utterance it had, on its clock. */
  checkedUntilMs?: number;
  /** The questions asked about it and their answers, latest first (Q139). */
  asked?: Answered[];
  /** The pictures added to it, each at its moment (contextImages.ts): absent until one is. */
  images?: ContextImage[];
};

/** Who an utterance's audio is, as the listening knows it. */
export type Voice = { kind: "speaker"; index: number } | { kind: "me" } | { kind: "others" };

/** An utterance as the listening hands it over: times on the Transcript's clock. */
export type Heard = { stream: string; startMs: number; endMs: number; text: string; voice: Voice | null; language?: string };

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Two digits: a clock's minutes and seconds. */
export const two = (n: number) => String(n).padStart(2, "0");

/** "25 Oct" and "15:42", in English whatever the browser's locale: a title and an export read the same everywhere. */
export const dayAndTime = (at: Date) => ({ day: `${at.getDate()} ${MONTHS[at.getMonth()]}`, time: `${two(at.getHours())}:${two(at.getMinutes())}` });

/** "25 Oct 15:42 · Call" (Q101); a file is named by its name. */
export function titleFor(source: SourceKind, at: Date, fileName?: string): string {
  const what = source === "file" ? (fileName ?? "File") : source === "call" ? "Call" : "Microphone";
  const { day, time } = dayAndTime(at);
  return `${day} ${time} · ${what}`;
}

/** What a listening starting says of itself. */
export type ListeningStart = { source: SourceKind; at: Date; fileName?: string };

/** A Transcript with nothing heard in it; begun by a listening when one is given. */
export function createTranscript(id: number, start?: ListeningStart): Transcript {
  const empty: Transcript = { id, title: "", source: null, startedAt: null, clockMs: 0, utterances: [], speakers: [], channels: {}, voices: 0 };
  return start ? beginListening(empty, start) : empty;
}

/**
 * A listening starts on `t`. The first titles it "date time · source", unless the user named it already
 * (Q101); a live one keeps its wall-clock start (Q100). A Continue keeps the title and the start it has.
 */
export function beginListening(t: Transcript, { source, at, fileName }: ListeningStart): Transcript {
  const first = t.source === null;
  return {
    ...t,
    title: first && t.title === "" ? titleFor(source, at, fileName) : t.title,
    source: t.source ?? source,
    startedAt: t.startedAt ?? (source === "file" ? null : at.getTime()),
  };
}

const nextId = (items: { id: number }[]) => Math.max(0, ...items.map((i) => i.id)) + 1;

/** A new label: a numbered voice, or Me or Others. */
function newSpeaker(t: Transcript, role: Speaker["role"]): { transcript: Transcript; id: number } {
  const id = nextId(t.speakers);
  const voices = role === "voice" ? t.voices + 1 : t.voices;
  const name = role === "me" ? "Me" : role === "others" ? "Others" : `Speaker ${voices}`;
  return { transcript: { ...t, speakers: [...t.speakers, { id, name, role }], voices }, id };
}

const channelOf = (stream: string, voice: Voice) => (voice.kind === "speaker" ? `${stream}:${voice.index}` : voice.kind);

/** How many sessions' voices the Transcript holds: more than one is a Resume or a Continue, whose voices are numbered anew. */
export const sessionsWithSpeakers = (t: Transcript): number =>
  new Set(Object.keys(t.channels).filter((channel) => channel.includes(":")).map((channel) => channel.split(":")[0])).size;

/**
 * Where a Continue's clock starts: the clock the last listening left, or past the last utterance when it left none
 * (a reload in the middle of a listening keeps the utterances but never hands the clock back).
 */
export const clockOf = (t: Transcript): number => Math.max(t.clockMs, ...t.utterances.map((u) => u.endMs));

/** The label a channel speaks under, made the first time it speaks. */
function labelFor(t: Transcript, stream: string, voice: Voice): { transcript: Transcript; id: number } {
  const channel = channelOf(stream, voice);
  const known = t.channels[channel];
  if (known !== undefined) return { transcript: t, id: known };
  const made = newSpeaker(t, voice.kind === "speaker" ? "voice" : voice.kind);
  return { transcript: { ...made.transcript, channels: { ...made.transcript.channels, [channel]: made.id } }, id: made.id };
}

/** In time order: after every utterance that starts no later. */
function placed(utterances: Utterance[], utterance: Utterance): Utterance[] {
  const at = utterances.findIndex((u) => u.startMs > utterance.startMs);
  return at === -1 ? [...utterances, utterance] : [...utterances.slice(0, at), utterance, ...utterances.slice(at)];
}

/** The end of the last utterance heard: what a look at the Transcript now reads up to. */
export const heardUntil = (utterances: Utterance[]): number => Math.max(0, ...utterances.map((u) => u.endMs));

export function addUtterance(t: Transcript, heard: Heard): Transcript {
  let next = t;
  let speaker: number | null = null;
  if (heard.voice) {
    const label = labelFor(t, heard.stream, heard.voice);
    next = label.transcript;
    speaker = label.id;
  }
  const utterance: Utterance = {
    id: nextId(next.utterances),
    startMs: heard.startMs,
    endMs: heard.endMs,
    speaker,
    text: heard.text,
    ...(heard.language === undefined ? {} : { language: heard.language }),
  };
  return { ...next, utterances: placed(next.utterances, utterance) };
}

const withUtterance = (t: Transcript, id: number, change: (u: Utterance) => Utterance): Transcript => ({
  ...t,
  utterances: t.utterances.map((u) => (u.id === id ? change(u) : u)),
});

/** The label a correction names: one of the Transcript's, or a new voice. */
function target(t: Transcript, to: number | "new"): { transcript: Transcript; id: number } {
  return to === "new" ? newSpeaker(t, "voice") : { transcript: t, id: to };
}

/** Every utterance of the speaker, past and future; a blank name keeps the one it had. */
export function renameSpeaker(t: Transcript, id: number, name: string): Transcript {
  const trimmed = name.trim();
  if (trimmed === "") return t;
  return { ...t, speakers: t.speakers.map((s) => (s.id === id ? { ...s, name: trimmed } : s)) };
}

/** One utterance to another speaker, or a new one: one-off, its channel still speaks as before. */
export function moveUtterance(t: Transcript, utterance: number, to: number | "new"): Transcript {
  const label = target(t, to);
  return withUtterance(label.transcript, utterance, (u) => ({ ...u, speaker: label.id }));
}

/** `from` becomes `into`: its utterances, and the channels that spoke as it, so its future utterances follow. */
export function mergeSpeakers(t: Transcript, from: number, into: number): Transcript {
  if (from === into) return t;
  return {
    ...t,
    speakers: t.speakers.filter((s) => s.id !== from),
    utterances: t.utterances.map((u) => (u.speaker === from ? { ...u, speaker: into } : u)),
    channels: Object.fromEntries(Object.entries(t.channels).map(([channel, label]) => [channel, label === from ? into : label])),
  };
}

/**
 * Two utterances where one was: the text before `at` and the text after it, the second to `to`. With no audio
 * kept, the time between is proportional to the characters before it, and approximate (Q98).
 */
export function splitUtterance(t: Transcript, utterance: number, at: number, to: number | "new"): Transcript {
  const whole = t.utterances.find((u) => u.id === utterance);
  if (!whole) return t;
  const first = whole.text.slice(0, at).trimEnd();
  const second = whole.text.slice(at).trimStart();
  if (first === "" || second === "") return t;
  const label = target(t, to);
  const cut = Math.round(whole.startMs + ((whole.endMs - whole.startMs) * at) / whole.text.length);
  // A fix is undone on the whole utterance: its halves keep the text as fixed, not the way back.
  const { approxEnd, heard: _heard, fixed: _fixed, ...start } = whole;
  const before: Utterance = { ...start, text: first, endMs: cut, approxEnd: true };
  const after: Utterance = {
    id: nextId(label.transcript.utterances),
    startMs: cut,
    endMs: whole.endMs,
    approxStart: true,
    ...(approxEnd ? { approxEnd } : {}),
    speaker: label.id,
    text: second,
  };
  const index = label.transcript.utterances.findIndex((u) => u.id === utterance);
  const utterances = [...label.transcript.utterances];
  utterances.splice(index, 1, before, after);
  return { ...label.transcript, utterances };
}

export const escapeRegExp = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** `from` as a word or words of its own, whatever its case: not inside a longer word. */
export const wordsOf = (from: string) => new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegExp(from)}(?![\\p{L}\\p{N}])`, "giu");

/** The words fixes put in, each as words of their own and as written: a mark's, not a fix's. */
export const fixedWordsOf = (fixed: string[]) => new RegExp(`(?<![\\p{L}\\p{N}])(${fixed.map(escapeRegExp).join("|")})(?![\\p{L}\\p{N}])`, "gu");

/**
 * A fix (Q133): `from` becomes `to` in each of `utterances` that holds it, as words of their own. Each keeps the
 * text it had before its first fix, and the words fixed in, to be marked and undone.
 */
export function fixUtterances(t: Transcript, from: string, to: string, utterances: number[]): Transcript {
  return {
    ...t,
    utterances: t.utterances.map((u) => {
      if (!utterances.includes(u.id) || !wordsOf(from).test(u.text)) return u;
      return { ...u, text: u.text.replace(wordsOf(from), () => to), heard: u.heard ?? u.text, fixed: [...new Set([...(u.fixed ?? []), to])] };
    }),
  };
}

/** Undoes an utterance's fixes: its text as it was heard. An edit since is undone with them. */
export function undoFixes(t: Transcript, utterance: number): Transcript {
  return withUtterance(t, utterance, ({ heard, fixed: _fixed, ...u }) => (heard === undefined ? u : { ...u, text: heard }));
}

/** Takes an utterance out (Q137); nothing else changes, its speaker kept for the others. */
export function removeUtterance(t: Transcript, utterance: number): Transcript {
  return { ...t, utterances: t.utterances.filter((u) => u.id !== utterance) };
}

/** Puts a removed utterance back where its time places it: what Undo does. */
export function restoreUtterance(t: Transcript, utterance: Utterance): Transcript {
  if (t.utterances.some((u) => u.id === utterance.id)) return t;
  return { ...t, utterances: placed(t.utterances, utterance) };
}

export function editText(t: Transcript, utterance: number, text: string): Transcript {
  return withUtterance(t, utterance, (u) => ({ ...u, text }));
}
