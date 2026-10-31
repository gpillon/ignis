import { restoreOrder, type Stamp } from "../store/docs.ts";
import { readAnswered } from "./ask.ts";
import { readContext } from "./context.ts";
import { readContextImages } from "./contextImages.ts";
import { readNotes } from "./notes.ts";
import { readSuggestions } from "./suggestions.ts";
import { isEmpty } from "./sessions.ts";
import type { Transcript } from "./transcript.ts";

// A Transcript as the page's store keeps it (spec speech/07, "The
// Transcript", Q86): a new kind of session, a document of
// `transcribe-sessions` under a key of its own (store/docs.ts), the chat's and
// the Decide tab's rules over it. It is plain data already: its utterances'
// text and times, its speakers and the channels they map, its corrections, its
// context images by their pictures' ids in the store (contextImages.ts). Its
// audio never reaches it: the page keeps none. An utterance is written as it
// completes (store/useStoredSessions.ts); the one under way is the listening's,
// not the Transcript's, so a reload keeps every completed utterance and ends
// the listening.

export const TRANSCRIBE_SESSIONS = "transcribe-sessions";

/** The schema this page writes. A document of a later one was written by a newer page, and is left alone. */
export const TRANSCRIBE_VERSION = 1;

export type StoredTranscript = Omit<Transcript, "id"> & {
  version: typeof TRANSCRIBE_VERSION;
  key: string;
  /** Where it sorts (store/docs.ts). */
  order: number;
};

/** A Transcript with nothing heard in it is not written: an empty one is what a new page opens with anyway. */
export const worthKeeping = (t: Transcript): boolean => !isEmpty(t);

export function toStored({ id: _id, ...t }: Transcript, { key, order }: Stamp): StoredTranscript {
  return { ...t, version: TRANSCRIBE_VERSION, key, order };
}

function readStored(value: unknown): StoredTranscript | null {
  if (typeof value !== "object" || value === null) return null;
  const doc = value as Partial<StoredTranscript>;
  if (doc.version !== TRANSCRIBE_VERSION || typeof doc.key !== "string" || typeof doc.order !== "number") return null;
  if (!Array.isArray(doc.utterances) || !Array.isArray(doc.speakers) || typeof doc.channels !== "object") return null;
  return doc as StoredTranscript;
}

/** The stored Transcripts, newest first, with new ids from the page's `nextId` up, and the stamp of each by its new id. */
export function restoreTranscripts(values: unknown[], nextId: number): { transcripts: Transcript[]; stamps: Map<number, Stamp>; nextId: number } {
  const docs = values.map(readStored).filter((d): d is StoredTranscript => d !== null);
  const ordered = restoreOrder(docs, nextId);
  const stamps = new Map<number, Stamp>();
  const transcripts = ordered.items.map(({ version: _version, key, order, notes, context, suggestions, asked, ...t }): Transcript => {
    stamps.set(t.id, { key, order });
    // Notes, a context or suggestions this page cannot read are dropped, not the Transcript they hang on.
    const readN = readNotes(notes);
    const readC = readContext(context);
    const readS = readSuggestions(suggestions);
    const checked = typeof t.checkedUntilMs === "number" && Number.isFinite(t.checkedUntilMs) ? { checkedUntilMs: t.checkedUntilMs } : {};
    const { checkedUntilMs: _checked, images, ...rest } = t;
    const readA = readAnswered(asked);
    // Context images likewise: those it cannot read are dropped, and the Transcript and its other images stay.
    const readI = readContextImages(images);
    return {
      ...rest,
      ...checked,
      ...(readN ? { notes: readN } : {}),
      ...(readC ? { context: readC } : {}),
      ...(readS ? { suggestions: readS } : {}),
      ...(readA ? { asked: readA } : {}),
      ...(readI?.length ? { images: readI } : {}),
    };
  });
  return { transcripts, stamps, nextId: ordered.nextId };
}
