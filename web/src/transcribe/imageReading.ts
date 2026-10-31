import type { ConversationRequest, Settings } from "../api/request.ts";
import type { ImageId } from "../store/imageStore.ts";
import { type AskDeps, askModel, settle, Watched } from "./assist.ts";
import { termProblem } from "./context.ts";
import { type ContextImage, type ImageReading, withReading } from "./contextImages.ts";
import { clock } from "./export.ts";
import { analysisOffers, suggest } from "./suggestions.ts";
import { contextBlock } from "./taskContext.ts";
import type { Transcript } from "./transcript.ts";

// Reading a context image (spec #428, "Context images"): the text model with
// vision reads each picture once, in a request of its own, the picture on the
// user turn behind the transcript's context, and says what it shows, its
// topic, and the names and terms on it. The reading lands on the image, and
// from then on it is the reading, a line at the image's moment, that the
// tasks are told (contextImages.ts): the picture is never sent again. One
// reading at a time, on an agent's lane with thinking off, as the notes'
// writing; with no text model with vision, a picture waits to be read.

export type { ImageReading };

export const READING_TASK = [
  "You read a picture shown during a conversation that is being transcribed: a slide, a shared screen, a whiteboard. What you read tells the transcription how to spell the words on it, and the notes what the conversation was looking at.",
  "Write what the picture shows, one item a line, in exactly these forms, in this order, and nothing else:",
  "SHOWS: what it shows, in one or two short sentences, with the figures on it that matter, in the language the context says to write in",
  "TOPIC: what it is about, in a few words, in the language the context says to write in",
  "NAME: each person, company, product or project named on it, spelled exactly as written there (one a line, at most ten)",
  "TERM: each technical term or acronym on it, spelled exactly as written there (one a line, at most ten)",
  "Use only what the picture shows: never invent names, numbers or outcomes. Write a NAME or TERM line only where the picture shows one.",
].join("\n");

/** The tokens a reading may take. */
export const READING_MAX_TOKENS = 500;

export const READING_SETTINGS: Omit<Settings, "systemPrompt"> = {
  temperature: 0.2,
  topP: 0.9,
  maxTokens: READING_MAX_TOKENS,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "agent",
};

/** What a reading keeps of what the picture shows, and of its topic. */
const SHOWS_CHARS = 400;
const TOPIC_CHARS = 120;
/** The names, and the terms, a reading keeps. */
const LISTED = 10;

/** A picture as the model is shown it: redrawn as the chat's are (conversation/images.ts). */
export type DrawnPicture = { url: string; width: number; height: number };

/** The request reading `image` of `t`: the context, then when the picture was shown, the picture on that turn. */
export function readingRequest(t: Transcript, image: ContextImage, picture: DrawnPicture): ConversationRequest {
  const at = clock(image.atMs);
  return {
    settings: { ...READING_SETTINGS, systemPrompt: READING_TASK },
    turns: [{ role: "user", content: `${contextBlock(t)}The picture was shown at [${at}] of the conversation.`, images: [{ name: `Context image at ${at}`, ...picture }] }],
  };
}

/** The model's lines, read leniently as the look's are (suggestions.ts): `null` when none is in a form. */
export function readReading(text: string): Omit<ImageReading, "readAt"> | null {
  let shows = "";
  let topic: string | null = null;
  const names: string[] = [];
  const terms: string[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.replace(/^[\s*>-]+/, "").replace(/\*\*/g, "").trim();
    const m = /^(SHOWS|TOPIC|NAME|TERM)\s*:\s*(.+)$/i.exec(line);
    if (!m) continue;
    const value = m[2].trim().replace(/^["“']|["”']$/g, "").trim();
    if (value === "" || /^none\.?$/i.test(value)) continue;
    const label = m[1].toUpperCase();
    if (label === "SHOWS") shows ||= value.slice(0, SHOWS_CHARS);
    else if (label === "TOPIC") topic ??= value.replace(/\.$/, "").slice(0, TOPIC_CHARS);
    else {
      const word = value.replace(/\.$/, "").trim();
      const list = label === "NAME" ? names : terms;
      const known = [...names, ...terms].some((w) => w.toLocaleLowerCase() === word.toLocaleLowerCase());
      if (!known && termProblem(word) === null && word.split(/\s+/).length <= 4 && list.length < LISTED) list.push(word);
    }
  }
  return shows === "" && topic === null && names.length === 0 && terms.length === 0 ? null : { shows, topic, names, terms };
}

/**
 * `t` once its image `id` was read: the reading on the image, and what it read offered as the look's findings are
 * (suggestions.ts), the names and the terms as terms to spell right and the topic while none is told; a kind accepted
 * always is applied at once, which is how they reach what the transcription is told.
 */
export function landed(t: Transcript, id: number, reading: ImageReading, now: number): Transcript {
  const found = { topic: reading.topic, title: null, terms: [...reading.names, ...reading.terms], fixes: [] };
  return analysisOffers(t, found).reduce((x, offer) => suggest(x, offer, now), withReading(t, id, reading));
}

export type ImageReaderState = {
  /** The image being read, by its Transcript and its id. */
  reading: { transcript: number; image: number } | null;
  /** The readings that failed in this page, and why: none is tried again until Read again. */
  failed: { transcript: number; image: number; message: string }[];
};

export type ImageReaderDeps = AskDeps & {
  /** A picture of the page's store as the model is shown it; rejects with why it cannot be. */
  picture: (id: ImageId) => Promise<DrawnPicture>;
  /** An image read: the Transcript it is in, its id, and what was read. */
  onRead: (transcript: number, image: number, reading: ImageReading) => void;
  now?: () => number;
};

const IDLE: ImageReaderState = { reading: null, failed: [] };

/** An image in this page: a deleted Transcript's id, and its images', can be given again, not when it was added. */
const keyOf = (transcript: number, image: ContextImage) => `${transcript}:${image.id}:${image.addedAt}`;

/** One reading at a time; an image is read once, and a failed one waits for Read again. */
export class ImageReader extends Watched<ImageReaderState> {
  private abort: AbortController | null = null;
  /** The images read in this page: never read again, whether the page's Transcript holds the reading yet or not. */
  private readonly done = new Set<string>();
  /** A keepUp goes on, from one image to the next: another call leaves the images to it. */
  private running = false;

  constructor(private readonly deps: ImageReaderDeps) {
    super(IDLE);
  }

  /**
   * Reads, one after the other, every image of `transcripts()` not read yet, while `canRead()` says a text model with
   * vision serves; nothing while a reading is under way, which goes on to the next itself.
   */
  async keepUp(transcripts: () => Transcript[], canRead: () => boolean): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      for (;;) {
        if (!canRead()) return;
        const next = this.next(transcripts());
        if (!next || (await this.read(next.t, next.image)) === "stopped") return;
      }
    } finally {
      this.running = false;
    }
  }

  /** The first image of `transcripts` with no reading, neither read in this page nor failed. */
  private next(transcripts: Transcript[]): { t: Transcript; image: ContextImage } | null {
    for (const t of transcripts) {
      for (const image of t.images ?? []) {
        const failed = this.state.failed.some((f) => f.transcript === t.id && f.image === image.id);
        if (!image.reading && !failed && !this.done.has(keyOf(t.id, image))) return { t, image };
      }
    }
    return null;
  }

  private async read(t: Transcript, image: ContextImage): Promise<"read" | "failed" | "stopped"> {
    const abort = new AbortController();
    this.abort = abort;
    this.set({ ...this.state, reading: { transcript: t.id, image: image.id } });
    const fail = (message: string): "failed" => {
      this.abort = null;
      this.set({ reading: null, failed: [...this.state.failed, { transcript: t.id, image: image.id, message }] });
      return "failed";
    };
    let picture: DrawnPicture;
    try {
      picture = await this.deps.picture(image.image);
    } catch (err) {
      return abort.signal.aborted ? "stopped" : fail(`The picture could not be drawn: ${err instanceof Error ? err.message : String(err)}`);
    }
    if (abort.signal.aborted) return "stopped";
    const settled = settle(await askModel(readingRequest(t, image, picture), abort.signal, this.deps));
    if (settled === "stopped") return "stopped";
    if ("failed" in settled) return fail(settled.failed);
    const reading = readReading(settled.text);
    if (!reading) return fail("The model wrote no reading.");
    this.abort = null;
    this.done.add(keyOf(t.id, image));
    this.deps.onRead(t.id, image.id, { ...reading, readAt: (this.deps.now ?? Date.now)() });
    this.set({ ...this.state, reading: null });
    return "read";
  }

  /** An image whose reading failed, to be read again at the next keepUp. */
  retry(transcript: number, image: number) {
    this.set({ ...this.state, failed: this.state.failed.filter((f) => f.transcript !== transcript || f.image !== image) });
  }

  /** Stops the reading under way; its image is read again later. */
  stop() {
    if (!this.abort) return;
    this.abort.abort();
    this.abort = null;
    this.set({ ...this.state, reading: null });
  }
}

/** The reader as the page shows it (assist.ts's useWatched). */
export type ImageReaderHandle = Pick<ImageReader, "getState" | "subscribe">;

/** Where an image's reading stands, as the spine says it. */
export type ReadingStatus =
  | { kind: "read"; reading: ImageReading }
  | { kind: "reading" }
  | { kind: "failed"; message: string }
  /** A text model with vision serves: its turn comes. */
  | { kind: "queued" }
  /** No text model with vision serves: it is read once one does. */
  | { kind: "waiting" };

/** Where the reading of Transcript `transcript`'s `image` stands, by the reader's state and whether it can read now. */
export function readingStatus(transcript: number, image: ContextImage, state: ImageReaderState, canRead: boolean): ReadingStatus {
  if (image.reading) return { kind: "read", reading: image.reading };
  if (state.reading?.transcript === transcript && state.reading.image === image.id) return { kind: "reading" };
  const failed = state.failed.find((f) => f.transcript === transcript && f.image === image.id);
  if (failed) return { kind: "failed", message: failed.message };
  return canRead ? { kind: "queued" } : { kind: "waiting" };
}
