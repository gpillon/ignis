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

/** An image attached to one prompt, as the wire carries it. */
export type PromptImage = {
  name: string;
  /** A `data:` URI: what `image_url.url` sends. */
  url: string;
  /** After the redraw. */
  width: number;
  height: number;
};

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

/** A picked file as a prompt image, or why it cannot be one. */
export async function imageFromFile(file: File): Promise<{ ok: true; image: PromptImage } | { ok: false; error: string }> {
  const named = file.name || "The image";
  if (!file.type.startsWith("image/")) return { ok: false, error: `${named} is not an image.` };
  if (file.size > MAX_FILE_BYTES) return { ok: false, error: `${named} is larger than ${MAX_FILE_BYTES >> 20} MB.` };
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
