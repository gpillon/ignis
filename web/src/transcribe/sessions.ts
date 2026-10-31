import { createTranscript, type Transcript } from "./transcript.ts";

// The Transcripts the view lists (spec speech/07, Q101), the Decide tab's
// rules over them (decide/sessions.ts): one open at a time, a new one reused
// while nothing is heard in it. Every function returns a new list and leaves
// the one it was given alone. The page's store keeps them (stored.ts).

export type TranscriptList = { transcripts: Transcript[]; activeId: number };

export const UNTITLED = "New transcript";

export function startList(id = 1): TranscriptList {
  return { transcripts: [createTranscript(id)], activeId: id };
}

/** An id nothing in the list is using. */
export const freeTranscriptId = (list: TranscriptList): number => Math.max(0, ...list.transcripts.map((t) => t.id)) + 1;

export const activeOf = (list: TranscriptList): Transcript => list.transcripts.find((t) => t.id === list.activeId) ?? list.transcripts[0];

export const titleOf = (t: Transcript): string => t.title || UNTITLED;

/** Nothing heard in it yet, nor a picture added to it before anyone spoke (#431). */
export const isEmpty = (t: Transcript): boolean => t.utterances.length === 0 && !t.images?.length;

/** A Transcript to listen into: the open one while nothing is heard in it, otherwise `newId` at the top. */
export function openTranscript(list: TranscriptList, newId: number): TranscriptList {
  if (isEmpty(activeOf(list))) return list;
  return { transcripts: [createTranscript(newId), ...list.transcripts], activeId: newId };
}

/** Drops a Transcript: the open one gives way to whichever took its place; the last leaves a new `newId`. */
export function removeTranscript(list: TranscriptList, id: number, newId: number): TranscriptList {
  const index = list.transcripts.findIndex((t) => t.id === id);
  if (index === -1) return list;
  const transcripts = list.transcripts.filter((t) => t.id !== id);
  if (transcripts.length === 0) return startList(newId);
  if (id !== list.activeId) return { transcripts, activeId: list.activeId };
  return { transcripts, activeId: transcripts[Math.min(index, transcripts.length - 1)].id };
}

export function updateTranscript(list: TranscriptList, id: number, change: (t: Transcript) => Transcript): TranscriptList {
  return { ...list, transcripts: list.transcripts.map((t) => (t.id === id ? change(t) : t)) };
}
