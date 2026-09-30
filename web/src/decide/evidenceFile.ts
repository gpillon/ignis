// A file from the reader's disk as the Decide tab's evidence: a log, a
// document or a JSON file, dropped on the tab or chosen from it — what a
// `locate` over a real log is tried on (GitHub #278).
//
// A file is a *source* of the text and JSON shapes, not a fourth one: its text
// is the evidence exactly as a pasted one would be, and the name rides beside
// it only so the bench can show a card instead of a textarea holding
// megabytes. Nothing about the file reaches the wire. A picture goes the way
// the image picker's do.

import { imageFromFile, type PromptImage } from "../conversation/images.ts";
import { parseOrdered } from "./json.ts";
import type { Evidence } from "./model.ts";

/** Where a text or JSON evidence came from, when it came from a file. */
export type FileSource = { name: string; size: number };

/**
 * The most a text file may weigh. The image picker's own cap, and several
 * times the longest prompt the engine reads (about 1M tokens, 3–4 MB), so a
 * log the endpoint could answer is never refused here.
 */
export const MAX_TEXT_FILE_BYTES = 32 << 20;

export type Loaded =
  | { ok: true; evidence: Extract<Evidence, { mode: "text" | "json" }> }
  | { ok: true; image: PromptImage }
  | { ok: false; error: string };

/** `file` as evidence, or why it cannot be. Images are re-encoded the way the picker's are. */
export async function evidenceFromFile(file: File): Promise<Loaded> {
  if (isImageFile(file)) {
    const read = await imageFromFile(file);
    return read.ok ? { ok: true, image: read.image } : read;
  }
  if (file.size > MAX_TEXT_FILE_BYTES) {
    return { ok: false, error: `${file.name} is ${formatBytes(file.size)}, and this tab loads files up to ${formatBytes(MAX_TEXT_FILE_BYTES)}.` };
  }
  let text: string;
  try {
    text = await file.text();
  } catch (error) {
    return { ok: false, error: `${file.name} could not be read: ${error instanceof Error ? error.message : String(error)}` };
  }
  return evidenceFromText({ name: file.name, size: file.size }, text);
}

/**
 * A file's text as evidence: JSON when it parses to an object or an array,
 * text otherwise.
 *
 * The shape is told by the content and not the extension. A `.json` that does
 * not parse is still a text somebody wants to search, and a JSON Lines file
 * does not parse as one value, so its lines are what a `locate` reads — which
 * is right for it. A NUL byte is what says a file is not text at all.
 *
 * Windows line ends become `\n`: the server cuts lines on `\n` exactly, so a
 * CRLF log would carry a `\r` at the end of every segment and pay a token for
 * each one.
 */
export function evidenceFromText(file: FileSource, raw: string): Loaded {
  if (raw.includes("\0")) return { ok: false, error: `${file.name} is not a text file.` };
  const text = raw.replace(/\r\n/g, "\n");
  if (text.trim() === "") return { ok: false, error: `${file.name} is empty.` };
  const first = text.trimStart()[0];
  if ((first === "{" || first === "[") && parseOrdered(text).ok) return { ok: true, evidence: { mode: "json", text, file } };
  return { ok: true, evidence: { mode: "text", text, file } };
}

const IMAGE_NAME = /\.(png|jpe?g|webp|gif|bmp|avif)$/i;

export const isImageFile = (file: File): boolean => file.type.startsWith("image/") || IMAGE_NAME.test(file.name);

/** A size as a reader says it: bytes, then KB and MB in powers of 1024. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${trim(bytes / 1024)} KB`;
  return `${trim(bytes / (1024 * 1024))} MB`;
}

const trim = (value: number): string => (value >= 100 ? Math.round(value).toString() : value.toFixed(1).replace(/\.0$/, ""));
