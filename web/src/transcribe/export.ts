import { dayAndTime, type Transcript, two, type Utterance } from "./transcript.ts";

// A Transcript out of the view (spec speech/07, "Out of the view", Q95),
// computed in the page from its utterances as edited, each speaker by the
// name it has now. Times are the Transcript's, from its start: `mm:ss`, and
// `h:mm:ss` past an hour (Q100). A split's guessed time is marked `~` where a
// line shows it; a cue writes it as it is. An utterance whose text was
// emptied is left out.

/** `mm:ss` from the start, `h:mm:ss` past an hour. */
export function clock(ms: number): string {
  const seconds = Math.floor(ms / 1_000);
  const h = Math.floor(seconds / 3_600);
  const m = Math.floor((seconds % 3_600) / 60);
  const s = seconds % 60;
  return h > 0 ? `${h}:${two(m)}:${two(s)}` : `${two(m)}:${two(s)}`;
}

/**
 * `mm:ss` in minutes from the start, past the hour too (`64:09`): how the notes thread tells the model a moment, which
 * keeps it from reading an hour as minutes (#429).
 */
export function minutesClock(ms: number): string {
  const seconds = Math.floor(ms / 1_000);
  return `${two(Math.floor(seconds / 60))}:${two(seconds % 60)}`;
}

/** `text` with each `[mm:ss]` past the hour, as the notes thread's model writes it, as `[h:mm:ss]`. */
export const hoursClocks = (text: string): string =>
  text.replace(/\[(~?)(\d{2,}):(\d{2})\]/g, (whole, approx: string, m: string, s: string) => (Number(m) < 60 ? whole : `[${approx}${clock((Number(m) * 60 + Number(s)) * 1_000)}]`));

/** An utterance's start as a line shows it: `~` when a split guessed it. */
export const startOf = (u: Utterance, time: (ms: number) => string = clock) => `${u.approxStart ? "~" : ""}${time(u.startMs)}`;

/** A cue's time: `hh:mm:ss,mmm` for SubRip, `hh:mm:ss.mmm` for WebVTT. */
function cueTime(ms: number, separator: "," | "."): string {
  const whole = Math.max(0, Math.round(ms));
  const h = Math.floor(whole / 3_600_000);
  const m = Math.floor((whole % 3_600_000) / 60_000);
  const s = Math.floor((whole % 60_000) / 1_000);
  return `${two(h)}:${two(m)}:${two(s)}${separator}${String(whole % 1_000).padStart(3, "0")}`;
}

/** The utterances worth writing, each with its speaker's name; `null` with no speakers. */
function lines(t: Transcript): { u: Utterance; text: string; speaker: string | null }[] {
  const names = new Map(t.speakers.map((s) => [s.id, s.name]));
  return t.utterances
    .map((u) => ({ u, text: u.text.trim(), speaker: u.speaker === null ? null : (names.get(u.speaker) ?? null) }))
    .filter((line) => line.text !== "");
}

/** "Started 25 Oct 2026, 15:42": a live listening's wall-clock start. */
export function startedLine(startedAt: number): string {
  const at = new Date(startedAt);
  const { day, time } = dayAndTime(at);
  return `Started ${day} ${at.getFullYear()}, ${time}`;
}

/** The clipboard's and `.txt`'s: `[mm:ss] Speaker: text`, one line per utterance; `time` writes the moment. */
export function toText(t: Transcript, time: (ms: number) => string = clock): string {
  return lines(t)
    .map(({ u, text, speaker }) => `[${startOf(u, time)}] ${speaker === null ? "" : `${speaker}: `}${text}\n`)
    .join("");
}

/** `.srt`: a numbered cue per utterance, from its start to its end, the speaker before the text. */
export function toSrt(t: Transcript): string {
  return lines(t)
    .map(({ u, text, speaker }, i) => `${i + 1}\n${cueTime(u.startMs, ",")} --> ${cueTime(u.endMs, ",")}\n${speaker === null ? "" : `${speaker}: `}${text}\n`)
    .join("\n");
}

const escaped = (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** `.vtt`: a cue per utterance, the speaker in a `<v Speaker>` voice tag. */
export function toVtt(t: Transcript): string {
  const cues = lines(t).map(
    ({ u, text, speaker }) => `${cueTime(u.startMs, ".")} --> ${cueTime(u.endMs, ".")}\n${speaker === null ? "" : `<v ${escaped(speaker)}>`}${escaped(text)}\n`,
  );
  return `WEBVTT\n\n${cues.join("\n")}`;
}

/** A file name from a title: what the downloads are saved as, and what Send to workspace names its file. */
export const fileStem = (title: string) => title.replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "transcript";
