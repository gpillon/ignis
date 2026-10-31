// Images the user sends with a prompt (GitHub #174): each one rides on the
// user message as an `image_url` content part, so the engine's vision path
// sees it the way ignis's own tests send one.
//
// A picked file is re-encoded in the browser before it becomes a part: the
// whole conversation goes back on the wire every turn, so a 6 MB photo would
// be paid again at each one. Bounding the long edge and re-encoding as JPEG
// costs a redraw once and keeps every later request small. The bytes are
// then fixed for the life of the message — a resend repeats the same data
// URI, which is what leaves the engine a media prefix to reuse.

import type { Mark } from "../markup/markup.ts";
import type { ImageId } from "../store/imageStore.ts";

/** An image attached to one prompt, as the wire carries it. */
export type PromptImage = {
  name: string;
  /** A `data:` URI: what `image_url.url` sends. */
  url: string;
  /** After the redraw. */
  width: number;
  height: number;
  /**
   * The picture as it was picked, at its own size, where the chat keeps it: the picture tool edits this one, not
   * the redraw for the vision tower. It reaches the page's store when the prompt is sent (`kept`), or when the
   * picture is sent to the Image view.
   */
  full?: Blob;
  /** The marks it was drawn with, over `source`, the picture before them; `region` is the area they mark. */
  markup?: { source: Blob; marks: Mark[]; keepOutside: boolean; region: Blob };
  /**
   * Where the page's store keeps `full` and the markup's pictures, by id, once the prompt is sent: the session is
   * kept without its Blobs, and a reload reads them back from here (sessions/stored.ts).
   */
  kept?: { full?: ImageId; markup?: { source: ImageId; region: ImageId; marks: Mark[]; keepOutside: boolean } };
};

/**
 * A picture of the conversation as its actions take it: the user's, or one the picture tool made. `id` names it in
 * the page's store when it is shown from there (a workspace's picture, #394) rather than from a data URL.
 */
export type ChatPicture = { name: string; url: string; full?: Blob; id?: ImageId };

/** What can be done with a picture of the conversation: mark it into the prompt box, or open it in the Image view. */
export type PictureAction = "mark" | "image";

/** The longest side an attached image keeps: enough detail for the vision tower, small on the wire. */
export const MAX_EDGE = 1568;

/** JPEG quality for the re-encode; a photograph survives it, and the data URI stays a fraction of the original. */
const QUALITY = 0.85;

/** What a picked file may weigh before it is refused, ahead of any decoding. */
const MAX_FILE_BYTES = 32 << 20;

/**
 * The type to build a `File` with, out of what a fetch reported (GitHub #256).
 *
 * A `Blob` whose type says nothing useful is not the empty string: a server
 * with no branch for the extension answers `application/octet-stream`, which
 * is truthy and is not an image type -- so `blob.type || fallback` keeps the
 * wrong one and `imageFromFile` refuses a picture it could have read. Only a
 * type that actually names an image is trusted; everything else falls back to
 * what the caller knows the bytes are.
 */
export const imageMime = (declared: string | undefined, fallback: string): string =>
  declared?.startsWith("image/") ? declared : fallback;

/** A picked file as a prompt image of the chat: the redraw for the wire, and the file itself beside it. */
export async function chatImageFromFile(file: Blob, name?: string): Promise<{ ok: true; image: PromptImage } | { ok: false; error: string }> {
  const named = file instanceof File ? file : new File([file], name ?? "picture.png", { type: file.type || "image/png" });
  const read = await imageFromFile(named);
  return read.ok ? { ok: true, image: { ...read.image, full: named } } : read;
}

/** Why a picked or pasted file cannot be a picture, ahead of any decoding, or null: the chat's limits, a context image's too (#431). */
export function pictureRefusal(file: Blob, named: string): string | null {
  if (!file.type.startsWith("image/")) return `${named} is not an image.`;
  if (file.size > MAX_FILE_BYTES) return `${named} is larger than ${MAX_FILE_BYTES >> 20} MB.`;
  return null;
}

/** A picked file as a prompt image, or why it cannot be one. */
export async function imageFromFile(file: File): Promise<{ ok: true; image: PromptImage } | { ok: false; error: string }> {
  const named = file.name || "The image";
  const refused = pictureRefusal(file, named);
  if (refused !== null) return { ok: false, error: refused };
  try {
    const bitmap = await createImageBitmap(file);
    try {
      const { width, height } = fitted(bitmap.width, bitmap.height);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("the browser gave no 2d canvas");
      context.drawImage(bitmap, 0, 0, width, height);
      // A picture with transparency loses it here: the model is shown what a
      // viewer on white would see, which is what the alpha stood for.
      const url = canvas.toDataURL("image/jpeg", QUALITY);
      if (!url.startsWith("data:image/jpeg;base64,")) throw new Error("the browser encoded no JPEG");
      return { ok: true, image: { name: file.name || "image", url, width, height } };
    } finally {
      bitmap.close();
    }
  } catch (err) {
    return { ok: false, error: `${named} could not be read: ${err instanceof Error ? err.message : String(err)}` };
  }
}

/** A picture redrawn for the model as an attached one is, a workspace's that it reads (#394); rejects with why it cannot be. */
export async function redrawn(blob: Blob, name: string): Promise<{ url: string; width: number; height: number }> {
  const read = await imageFromFile(new File([blob], name, { type: imageMime(blob.type, "image/png") }));
  if (!read.ok) throw new Error(read.error);
  return { url: read.image.url, width: read.image.width, height: read.image.height };
}

/** `width` by `height` scaled down to fit `MAX_EDGE`: never up, never below one pixel. */
export function fitted(width: number, height: number): { width: number; height: number } {
  const scale = Math.min(1, MAX_EDGE / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

/** What a data URI's payload weighs once decoded — for what the composer shows. */
export function dataUriBytes(url: string): number {
  const payload = url.slice(url.indexOf(",") + 1);
  const padding = payload.endsWith("==") ? 2 : payload.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((payload.length * 3) / 4) - padding);
}
