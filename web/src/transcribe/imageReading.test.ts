import { describe, expect, it } from "vitest";
import type { ConversationRequest } from "../api/request.ts";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { Analyst } from "./analyst.ts";
import { Asker } from "./ask.ts";
import { addContextImage, readingLine, withReading } from "./contextImages.ts";
import { EMPTY_CONTEXT } from "./context.ts";
import { type ImageReading, ImageReader, landed, READING_SETTINGS, READING_TASK, readReading } from "./imageReading.ts";
import { toMarkdown } from "./markdown.ts";
import { NotesWriter } from "./notesWriter.ts";
import { Rewriter } from "./rewrite.ts";
import { accept, openSuggestions, type SuggestionKind } from "./suggestions.ts";
import { contextBlock } from "./taskContext.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// Reading a context image (spec #428, "Context images", seam 4): the text
// model with vision reads each picture once, in a request of its own, the
// picture on the user turn and the context told; its terms, names, topic and
// what it shows land on the image, and the picture is never sent again. With
// no vision model the reading waits, and runs once one serves.

const SLIDE = "a".repeat(64);
const BOARD = "b".repeat(64);

function call(): Transcript {
  let t = { ...createTranscript(4, { source: "call", at: new Date(2026, 9, 31, 10, 0) }), title: "31 Oct 10:00 · Call" };
  t = addUtterance(t, { stream: "mic-1", startMs: 2_000, endMs: 5_000, text: "Vedete la slide del budget?", voice: { kind: "me" } });
  t = addUtterance(t, { stream: "shared-2", startMs: 6_000, endMs: 9_000, text: "Sì, quella del terzo trimestre.", voice: { kind: "speaker", index: 0 } });
  return addContextImage(t, { image: SLIDE, atMs: 5_500, width: 1280, height: 720 }, 1_000);
}

const READ = ["SHOWS: A slide titled Budget Q3, costs per team.", "TOPIC: Budget del terzo trimestre", "NAME: Anna Rossi", "TERM: OKR", "TERM: Kubernetes"].join("\n");

/** A reader on a fake stream, its pictures drawn by name, and the readings landed. */
function reader(answers: (string | Error)[]) {
  const asked: StreamOptions[] = [];
  const drawn: string[] = [];
  const read: [number, number, ImageReading][] = [];
  const stream = async (options: StreamOptions): Promise<StreamResult> => {
    asked.push(options);
    const answer = answers.shift() ?? "";
    const timeline = { sentAt: 0, stopped: false };
    if (answer instanceof Error) return { ok: false, message: answer.message, timeline };
    options.onEvent({ kind: "content", text: answer });
    return { ok: true, timeline };
  };
  const r = new ImageReader({
    stream,
    sleep: async () => {},
    now: () => 7_000,
    picture: async (id) => {
      drawn.push(id);
      return { url: `data:image/jpeg;base64,${id.slice(0, 4)}`, width: 1280, height: 720 };
    },
    onRead: (transcript, image, reading) => read.push([transcript, image, reading]),
  });
  return { r, asked, drawn, read };
}

describe("reading a context image", () => {
  it("reads it once, in a request of its own: the picture on the user turn with the context, never the transcript", async () => {
    const { r, asked, drawn, read } = reader([READ]);
    const t = call();
    await r.keepUp(() => [t], () => true);
    expect(asked).toHaveLength(1);
    const { request, streamId } = asked[0];
    expect(streamId).toBeUndefined();
    expect(request.settings).toEqual({ ...READING_SETTINGS, systemPrompt: READING_TASK });
    expect(request.turns).toHaveLength(1);
    const [turn] = request.turns;
    expect(turn.role).toBe("user");
    expect(turn.images).toEqual([{ name: "Context image at 00:05", url: "data:image/jpeg;base64,aaaa", width: 1280, height: 720 }]);
    expect(turn.content.startsWith(contextBlock(t))).toBe(true);
    expect(turn.content).toContain("[00:05]");
    expect(turn.content).not.toContain("Vedete la slide");
    expect(drawn).toEqual([SLIDE]);
    expect(read).toEqual([
      [4, 1, { shows: "A slide titled Budget Q3, costs per team.", topic: "Budget del terzo trimestre", names: ["Anna Rossi"], terms: ["OKR", "Kubernetes"], readAt: 7_000 }],
    ]);
  });

  it("never reads it again: not when asked again before the page holds the reading, nor once it does", async () => {
    const { r, asked } = reader([READ, READ]);
    const t = call();
    await r.keepUp(() => [t], () => true);
    // The reading has not landed on the page's transcript yet.
    await r.keepUp(() => [t], () => true);
    const held = { ...t, images: t.images!.map((i) => ({ ...i, reading: { shows: "A slide.", topic: null, names: [], terms: [], readAt: 7_000 } })) };
    await r.keepUp(() => [held], () => true);
    expect(asked).toHaveLength(1);
  });

  it("reads each image of each transcript, one request each, one at a time", async () => {
    const { r, asked, read } = reader([READ, "SHOWS: A whiteboard.", "SHOWS: A diagram."]);
    const t = addContextImage(call(), { image: BOARD, atMs: 8_000, width: 800, height: 600 }, 2_000);
    const other = addContextImage(createTranscript(9), { image: BOARD, atMs: 0, width: 800, height: 600 }, 3_000);
    await r.keepUp(() => [t, other], () => true);
    expect(asked).toHaveLength(3);
    expect(asked.map((a) => a.request.turns[0].images?.length)).toEqual([1, 1, 1]);
    expect(read.map(([transcript, image]) => [transcript, image])).toEqual([
      [4, 1],
      [4, 2],
      [9, 1],
    ]);
  });

  it("reads the image of a new transcript that took a deleted one's id", async () => {
    const { r, asked } = reader([READ, READ]);
    await r.keepUp(() => [call()], () => true);
    // Transcript 4 deleted, and the next new one numbered 4 again, its first image numbered 1 again.
    const again = addContextImage({ ...createTranscript(4), title: "Again" }, { image: BOARD, atMs: 0, width: 800, height: 600 }, 5_000);
    await r.keepUp(() => [again], () => true);
    expect(asked).toHaveLength(2);
  });

  it("leaves the images to a reading already going on, which reads each once", async () => {
    const { r, asked } = reader([READ, "SHOWS: A whiteboard."]);
    const t = addContextImage(call(), { image: BOARD, atMs: 8_000, width: 800, height: 600 }, 2_000);
    await Promise.all([r.keepUp(() => [t], () => true), r.keepUp(() => [t], () => true)]);
    expect(asked.map((a) => a.request.turns[0].images?.[0].url)).toEqual(["data:image/jpeg;base64,aaaa", "data:image/jpeg;base64,bbbb"]);
  });

  it("waits while no text model with vision serves, the image kept, and reads it once one does", async () => {
    const { r, asked, read } = reader([READ]);
    const t = call();
    let vision = false;
    await r.keepUp(() => [t], () => vision);
    await r.keepUp(() => [t], () => vision);
    expect(asked).toHaveLength(0);
    expect(r.getState()).toEqual({ reading: null, failed: [] });
    vision = true;
    await r.keepUp(() => [t], () => vision);
    expect(asked).toHaveLength(1);
    expect(read).toHaveLength(1);
  });

  it("says why a reading failed, leaves it until Read again, and goes on to the next image", async () => {
    const { r, asked, read } = reader([new Error("vision is disabled"), "SHOWS: A whiteboard.", READ]);
    const t = addContextImage(call(), { image: BOARD, atMs: 8_000, width: 800, height: 600 }, 2_000);
    await r.keepUp(() => [t], () => true);
    expect(r.getState()).toEqual({ reading: null, failed: [{ transcript: 4, image: 1, message: "vision is disabled" }] });
    expect(read.map(([, image]) => image)).toEqual([2]);
    await r.keepUp(() => [t], () => true);
    expect(asked).toHaveLength(2);
    r.retry(4, 1);
    await r.keepUp(() => [t], () => true);
    expect(asked).toHaveLength(3);
    expect(read.map(([, image]) => image)).toEqual([2, 1]);
    expect(r.getState().failed).toEqual([]);
  });

  it("fails an answer in no form, and a picture the store no longer holds, without asking", async () => {
    const { r, asked } = reader(["I see a slide."]);
    await r.keepUp(() => [call()], () => true);
    expect(r.getState().failed).toEqual([{ transcript: 4, image: 1, message: "The model wrote no reading." }]);
    const lost = new ImageReader({ stream: async () => ({ ok: true, timeline: { sentAt: 0, stopped: false } }), picture: () => Promise.reject(new Error("not in this browser")), onRead: () => {} });
    await lost.keepUp(() => [call()], () => true);
    expect(lost.getState().failed).toEqual([{ transcript: 4, image: 1, message: "The picture could not be drawn: not in this browser" }]);
    expect(asked).toHaveLength(1);
  });
});

describe("a reading as the model writes it", () => {
  it("is read leniently: bullets and bold around a line, a NONE, the names and the terms each once, ten at most", () => {
    const answer = ["- **SHOWS:** Una lavagna con lo schema della rete.", "TOPIC: Rete dell'ufficio.", "NAME: none", "* TERM: VPN", "TERM: vpn", ...Array.from({ length: 12 }, (_, i) => `TERM: T${i}`), "Notes: ignored"].join("\n");
    expect(readReading(answer)).toEqual({
      shows: "Una lavagna con lo schema della rete.",
      topic: "Rete dell'ufficio",
      names: [],
      terms: ["VPN", "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
    });
    expect(readReading("NONE")).toBeNull();
  });
});

describe("where a reading reaches", () => {
  const reading: ImageReading = { shows: "A slide titled Budget Q3, costs per team.", topic: "Budget del terzo trimestre", names: ["Anna Rossi"], terms: ["OKR"], readAt: 7_000 };
  const read = (t: Transcript, id = 1) => withReading(t, id, reading);

  /** What each task's request reads: its user turn. */
  async function sent(ask: (stream: (o: StreamOptions) => Promise<StreamResult>) => Promise<void>): Promise<string> {
    const seen: ConversationRequest[] = [];
    await ask(async (o) => {
      seen.push(o.request);
      return { ok: true, timeline: { sentAt: 0, stopped: false } };
    });
    expect(seen).toHaveLength(1);
    return seen[0].turns[0].content;
  }

  it("is a line marked as an image at its moment, between the utterances, in the look's text and Ask's", async () => {
    const t = read(call());
    const line = "[00:05] (image) A slide titled Budget Q3, costs per team. Topic: Budget del terzo trimestre. Names: Anna Rossi. Terms: OKR.";
    const between = `[00:02] Me: Vedete la slide del budget?\n${line}\n[00:06] Speaker 1: Sì, quella del terzo trimestre.\n`;
    expect(await sent((stream) => new Analyst({ stream, onFound: () => {} }).analyze(t, null))).toContain(between);
    expect(await sent((stream) => new Asker({ stream, onAnswered: () => {} }).ask(t, "Cosa diceva la slide?", null))).toContain(between);
    // Not read yet: no line, the picture never sent.
    const unread = call();
    const look = await sent((stream) => new Analyst({ stream, onFound: () => {} }).analyze(unread, null));
    expect(look).not.toContain("(image)");
  });

  it("reaches Rewrite's text at its moment, with the picture shown when the conversation around the utterance began", async () => {
    let t = { ...createTranscript(4), title: "Call" };
    for (let i = 0; i < 30; i++) t = addUtterance(t, { stream: "s", startMs: i * 10_000, endMs: i * 10_000 + 4_000, text: `Frase ${i}.`, voice: { kind: "me" } });
    // The first, shown and replaced before the conversation around utterance 26 began; the second, on screen when it
    // began; the third, shown during it.
    t = addContextImage(t, { image: SLIDE, atMs: 1_000, width: 1, height: 1 }, 1);
    t = addContextImage(t, { image: BOARD, atMs: 21_000, width: 1, height: 1 }, 2);
    t = addContextImage(t, { image: SLIDE, atMs: 121_000, width: 1, height: 1 }, 3);
    t = withReading(withReading(withReading(t, 1, { ...reading, shows: "The agenda." }), 2, { ...reading, shows: "The team." }), 3, { ...reading, shows: "The budget." });
    const text = await sent((stream) => new Rewriter({ stream }).rewrite(t, t.utterances[25]));
    expect(text).not.toContain("The agenda.");
    expect(text).toContain(`${readingLine(t.images![1])}\n[00:50] Me: Frase 5.\n`);
    expect(text).toContain(`[02:00] Me: Frase 12.\n${readingLine(t.images![2])}\n[02:10] Me: Frase 13.\n`);
  });

  /** A notes writer on a fake stream that calls nothing, and the requests it sent. */
  function notes() {
    const asked: StreamOptions[] = [];
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      asked.push(o);
      return { ok: true, timeline: { sentAt: 0, stopped: false } };
    };
    return { w: new NotesWriter({ stream, onNotes: () => {}, onRecap: () => {} }), asked };
  }

  it("reaches the notes thread's next turn as one change at its moment, in total minutes, and is told once", async () => {
    const { w, asked } = notes();
    const t = call();
    await w.write(t, "work", null);
    expect(asked[0].request.turns[0].content).not.toContain("(image)");
    await w.write(read(t), "work", null);
    expect(asked).toHaveLength(2);
    expect(asked[1].request.turns.at(-1)).toEqual({
      role: "user",
      content: "Changed since:\n- Shown: [00:05] (image) A slide titled Budget Q3, costs per team. Topic: Budget del terzo trimestre. Names: Anna Rossi. Terms: OKR.",
    });
    // The running task and the preamble say what such a line is, in the system block every request of the thread carries.
    expect(asked[1].request.settings.systemPrompt).toContain("A picture the conversation was looking at comes there too, as Shown: with what was read in it.");
    expect(asked[1].request.settings.systemPrompt).toContain("A line `[mm:ss] (image) ...` is a picture the conversation was looking at from that moment, as it was read.");
    // Nothing said, changed or read since: no request.
    await w.write(read(t), "work", null);
    expect(asked).toHaveLength(2);
  });

  it("reaches the recap's request once, among the pictures shown, after the whole transcript", async () => {
    const { w, asked } = notes();
    let t = addContextImage(call(), { image: BOARD, atMs: 0, width: 800, height: 600 }, 2_000);
    t = withReading(read(t), 2, { shows: "The agenda.", topic: null, names: [], terms: [], readAt: 7_500 });
    await w.writeRecap(t, "work", null);
    const content = asked[0].request.turns[0].content;
    const shown = [
      "Pictures shown during the conversation, as read at their moments:",
      "[00:00] (image) The agenda.",
      "[00:05] (image) A slide titled Budget Q3, costs per team. Topic: Budget del terzo trimestre. Names: Anna Rossi. Terms: OKR.",
    ].join("\n");
    expect(content).toContain(`[00:06] Speaker 1: Sì, quella del terzo trimestre.\n\n${shown}\n\nNotes the user took during the conversation: none.`);
    expect(content.split("(image)")).toHaveLength(3);
  });

  it("is a line of the .md at its moment, before an answer asked at the same place; an image not read has none", () => {
    const answered = { ...read(call()), asked: [{ id: 1, question: "Il budget?", answer: "Quello del Q3.", at: 9_000, untilMs: 5_000 }] };
    expect(toMarkdown(answered)).toBe(
      [
        "# 31 Oct 10:00 · Call",
        "",
        "Started 31 Oct 2026, 10:00",
        "",
        "**Me** (00:02): Vedete la slide del budget?",
        "",
        "[00:05] (image) A slide titled Budget Q3, costs per team. Topic: Budget del terzo trimestre. Names: Anna Rossi. Terms: OKR.",
        "",
        "> **Il budget?**",
        ">",
        "> Quello del Q3.",
        "",
        "**Speaker 1** (00:06): Sì, quella del terzo trimestre.",
        "",
      ].join("\n"),
    );
    const { images: _, ...none } = call();
    expect(toMarkdown(call())).toBe(toMarkdown(none));
  });

  it("offers its terms and names, and its topic while none is told, as suggestions in the Call tab", () => {
    const t = landed(call(), 1, reading, 8_000);
    expect(t.images![0].reading).toEqual(reading);
    expect(openSuggestions(t).map(({ id: _, ...s }) => s)).toEqual([
      { kind: "topic", topic: "Budget del terzo trimestre", status: "open" },
      { kind: "terms", terms: ["Anna Rossi", "OKR"], status: "open" },
    ]);
    // Accepted, the terms reach what the transcription is told.
    const termsId = openSuggestions(t).find((s) => s.kind === "terms")!.id;
    expect(accept(t, termsId, 9_000).context?.terms).toEqual(["Anna Rossi", "OKR"]);
    const told = { ...call(), context: { ...EMPTY_CONTEXT, topic: "Budget", terms: ["OKR"] } };
    expect(openSuggestions(landed(told, 1, reading, 8_000)).map(({ id: _, ...s }) => s)).toEqual([{ kind: "terms", terms: ["Anna Rossi"], status: "open" }]);
  });

  it("applies its terms and topic as they come once Always was picked for their kinds", () => {
    const always = { ...call(), suggestions: { items: [], always: ["terms", "topic"] as SuggestionKind[] } };
    const t = landed(always, 1, reading, 8_000);
    expect(openSuggestions(t)).toEqual([]);
    expect(t.context).toMatchObject({ topic: "Budget del terzo trimestre", terms: ["Anna Rossi", "OKR"], appliedAt: 8_000 });
  });

  it("is in a thread's first turn at its moment when the thread starts with it read, and not told again", async () => {
    const { w, asked } = notes();
    const t = read(call());
    await w.write(t, "work", null);
    expect(asked[0].request.turns[0].content).toContain("[00:02] Me: Vedete la slide del budget?\n[00:05] (image) A slide titled Budget Q3");
    await w.write(t, "work", null);
    expect(asked).toHaveLength(1);
  });
});
