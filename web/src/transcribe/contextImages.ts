import { type ImageId, unreferenced } from "../store/imageStore.ts";
import type { ListeningState } from "./listening.ts";
import { clockOf, type Transcript, type Utterance } from "./transcript.ts";

// A transcript's context images (spec #428, GLOSSARY.md): a picture added at a
// moment of its clock (a slide, a shared screen, a whiteboard), cropped and
// blacked out first, that tells what the conversation was looking at from
// then on. The picture is in the page's store, once by its SHA-256 as the
// chat's are (store/pictures.ts); the transcript names it by that id, at its
// moment, beside its size. Several to a transcript, and nothing reads them
// yet: the text model's reading is a later ticket's (#437).

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

const count = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0;

/** A stored transcript's images: those this page can read, or null for a value that is no list of them. */
export function readContextImages(value: unknown): ContextImage[] | null {
  if (!Array.isArray(value)) return null;
  return value.filter((v): v is ContextImage => {
    if (typeof v !== "object" || v === null) return false;
    const i = v as Partial<ContextImage>;
    return count(i.id) && typeof i.image === "string" && i.image !== "" && count(i.atMs) && count(i.width) && count(i.height) && count(i.addedAt);
  });
}
