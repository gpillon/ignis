import { type ImageId, unreferenced } from "../store/imageStore.ts";
import { clock, minutesClock, toText } from "./export.ts";
import type { ListeningState } from "./listening.ts";
import { clockOf, type Transcript, type Utterance } from "./transcript.ts";

// A transcript's context images (spec #428, GLOSSARY.md): a picture added at a
// moment of its clock (a slide, a shared screen, a whiteboard), cropped and
// blacked out first, that tells what the conversation was looking at from
// then on. The picture is in the page's store, once by its SHA-256 as the
// chat's are (store/pictures.ts); the transcript names it by that id, at its
// moment, beside its size. Several to a transcript. The text model with vision
// reads each once (imageReading.ts, #437), and its reading, not the picture,
// is what the tasks are told from then on, a line at its moment.

/** What the text model read in a context image, once: what it shows, its topic, the names and the terms on it. */
export type ImageReading = {
  shows: string;
  topic: string | null;
  names: string[];
  terms: string[];
  /** When it was read, epoch ms. */
  readAt: number;
};

export type ContextImage = {
  /** Unique within its transcript. */
  id: number;
  /** The picture, in the page's store. */
  image: ImageId;
  /** The moment on the transcript's clock it holds from. */
  atMs: number;
  width: number;
  height: number;
  /** When it was added, epoch ms. */
  addedAt: number;
  /** What the text model read in it: absent until it is read. */
  reading?: ImageReading;
};

/**
 * The moment a picture added now holds from: the listening's clock while it listens into `t`, else where `t`'s clock
 * stopped; the start while nothing is heard in it yet, so a picture shown before anyone spoke holds for the whole
 * conversation.
 */
export function imageMoment(t: Transcript, listening: Pick<ListeningState, "phase" | "transcript" | "clockMs">): number {
  if (t.utterances.length === 0) return 0;
  return listening.phase !== "idle" && listening.transcript === t.id ? listening.clockMs : clockOf(t);
}

export function addContextImage(t: Transcript, added: Omit<ContextImage, "id" | "addedAt">, at: number): Transcript {
  const images = t.images ?? [];
  return { ...t, images: [...images, { id: Math.max(0, ...images.map((i) => i.id)) + 1, ...added, addedAt: at }] };
}

/** The pictures a transcript's images name in the page's store, each once. */
export const picturesOf = (t: Transcript): Set<ImageId> => new Set((t.images ?? []).map((i) => i.image));

/**
 * The pictures deleting `gone` frees: those no transcript `left` names, nor anything `elsewhere` holds (the chat's
 * sessions, the Image view's, the workspaces'): the store keeps every view's pictures once. None while `elsewhere`
 * cannot say yet, a view that has not read its sessions.
 */
export function freedBy(gone: Transcript, left: Transcript[], elsewhere: Iterable<ImageId> | null): ImageId[] {
  if (elsewhere === null) return [];
  return unreferenced(picturesOf(gone), new Set<ImageId>([...left.flatMap((t) => [...picturesOf(t)]), ...elsewhere]));
}

/** Where a picture of moment `atMs` stands among the utterances: before the first that starts at that moment or after. */
export function imageAt(utterances: Utterance[], atMs: number): number {
  const at = utterances.findIndex((u) => u.startMs >= atMs);
  return at === -1 ? utterances.length : at;
}

/**
 * The picture a paste in the view adds, if any: the first image it carries. A paste into a text field that carries
 * text too is the field's (a copy from a document can carry both): only a paste the field could not take is a picture.
 */
export function pastedPicture(files: File[], { intoField, hasText }: { intoField: boolean; hasText: boolean }): File | null {
  if (intoField && hasText) return null;
  return files.find((f) => f.type.startsWith("image/")) ?? null;
}

/** `t` with its image `id` read: the reading on it, from then on what the tasks are told of it. */
export function withReading(t: Transcript, id: number, reading: ImageReading): Transcript {
  return { ...t, images: (t.images ?? []).map((i) => (i.id === id ? { ...i, reading } : i)) };
}

/** A sentence's end: what the model wrote, with a full stop when it gave none. */
const ended = (text: string) => (/[.!?…]$/.test(text) ? text : `${text}.`);

/** What a reading says, as the tasks and the `.md` are told it: what the picture shows, then its topic, names and terms. */
export function readingText(r: ImageReading): string {
  return [
    ...(r.shows === "" ? [] : [ended(r.shows)]),
    ...(r.topic === null ? [] : [`Topic: ${ended(r.topic)}`]),
    ...(r.names.length > 0 ? [`Names: ${r.names.join(", ")}.`] : []),
    ...(r.terms.length > 0 ? [`Terms: ${r.terms.join(", ")}.`] : []),
  ].join(" ");
}

/** A read image's line, `[mm:ss] (image) ...`, `time` writing its moment; no line break. */
export const readingLine = (image: ContextImage, time: (ms: number) => string = clock): string =>
  `[${time(image.atMs)}] (image) ${image.reading ? readingText(image.reading) : ""}`.trimEnd();

/** `t`'s images read, in the order of their moments. */
export const readImages = (t: Transcript): ContextImage[] =>
  (t.images ?? []).filter((i) => i.reading !== undefined).sort((a, b) => a.atMs - b.atMs || a.id - b.id);

/**
 * The text the tasks read of `t` (#437): its `.txt` lines, `time` writing each moment, with each read image's line
 * where the spine shows it, before the first utterance at its moment or after; each line with its moment. An emptied
 * utterance's line is "".
 */
export function linesWithImages(t: Transcript, time?: (ms: number) => string): { atMs: number; line: string }[] {
  const images = readImages(t);
  const lines: { atMs: number; line: string }[] = [];
  const imagesAt = (place: number) => {
    for (const image of images) if (imageAt(t.utterances, image.atMs) === place) lines.push({ atMs: image.atMs, line: `${readingLine(image, time)}\n` });
  };
  t.utterances.forEach((u, i) => {
    imagesAt(i);
    lines.push({ atMs: u.startMs, line: toText({ ...t, utterances: [u] }, time) });
  });
  imagesAt(t.utterances.length);
  return lines;
}

/** The images of `t` whose reading a notes thread starting from it, or appending a turn of it, tells: those read. */
export const shownOf = (t: Transcript): Set<number> => new Set(readImages(t).map((i) => i.id));

/**
 * The images read since a notes thread told the ones `shown`, one event each for its next turn, in the order of their
 * moments: `Shown: [mm:ss] (image) ...`, in total minutes as the thread tells every moment.
 */
export const shownSince = (shown: Set<number>, t: Transcript): string[] =>
  readImages(t)
    .filter((i) => !shown.has(i.id))
    .map((i) => `Shown: ${readingLine(i, minutesClock)}`);

const count = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0;
const strings = (v: unknown): v is string[] => Array.isArray(v) && v.every((s) => typeof s === "string");

/** A stored image's reading, or null for one this page cannot read: the image is read again. */
function readReadingOf(value: unknown): ImageReading | null {
  if (typeof value !== "object" || value === null) return null;
  const r = value as Partial<ImageReading>;
  const fine = typeof r.shows === "string" && (r.topic === null || typeof r.topic === "string") && strings(r.names) && strings(r.terms) && count(r.readAt);
  return fine ? (r as ImageReading) : null;
}

/** A stored transcript's images: those this page can read, or null for a value that is no list of them. */
export function readContextImages(value: unknown): ContextImage[] | null {
  if (!Array.isArray(value)) return null;
  return value
    .filter((v): v is ContextImage => {
      if (typeof v !== "object" || v === null) return false;
      const i = v as Partial<ContextImage>;
      return count(i.id) && typeof i.image === "string" && i.image !== "" && count(i.atMs) && count(i.width) && count(i.height) && count(i.addedAt);
    })
    .map(({ reading, ...i }) => {
      const read = reading === undefined ? null : readReadingOf(reading);
      return read ? { ...i, reading: read } : i;
    });
}
