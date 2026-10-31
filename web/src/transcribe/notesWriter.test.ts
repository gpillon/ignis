import { describe, expect, it } from "vitest";
import type { ChunkEvent } from "../api/sse.ts";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { ASSIST_SYSTEM } from "./assist.ts";
import { EMPTY_CONTEXT } from "./context.ts";
import { LIVE_GAP_MS, noNotes, notesTask, type Notes } from "./notes.ts";
import { type NotesForm, NotesWriter, type Written } from "./notesWriter.ts";
import { recapTask } from "./recap.ts";
import { contextBlock } from "./taskContext.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// The notes' writer (spec speech/07, Q124-Q127): one writing at a time, on
// the text model, its text as it streams; the notes land on the Transcript it
// started on, a Stop leaves the notes there were, and a full engine is waited
// out. Keeping up writes when notes.ts's cadence says so.

function heard(n: number, id = 1): Transcript {
  let t = { ...createTranscript(id), title: "27 Oct 15:42 · Call" };
  for (let i = 0; i < n; i++) t = addUtterance(t, { stream: "s", startMs: i * 5_000, endMs: i * 5_000 + 2_000, text: `Frase ${i}.`, voice: { kind: "me" } });
  return t;
}

/** A stream that answers each request with the next script: chunks, then a result. */
function fakeStream(scripts: { chunks?: string[]; result?: Partial<StreamResult> & { ok: boolean; message?: string }; hold?: true }[]) {
  const asked: StreamOptions[] = [];
  let release: () => void = () => {};
  const stream = async (options: StreamOptions): Promise<StreamResult> => {
    asked.push(options);
    const script = scripts.shift() ?? { chunks: [] };
    for (const text of script.chunks ?? []) options.onEvent({ kind: "content", text } satisfies ChunkEvent);
    if (script.hold) {
      await new Promise<void>((resolve) => {
        release = resolve;
        options.signal?.addEventListener("abort", () => resolve(), { once: true });
      });
    }
    const timeline = { sentAt: 0, stopped: options.signal?.aborted ?? false };
    if (script.result && !script.result.ok) return { ok: false, message: script.result.message ?? "failed", timeline };
    return { ok: true, timeline };
  };
  return { stream, asked, release: () => release() };
}

function writer(scripts: Parameters<typeof fakeStream>[0], now = () => 1_000) {
  const fake = fakeStream(scripts);
  const written: [number, Written, NotesForm][] = [];
  const slept: number[] = [];
  const w = new NotesWriter({
    stream: fake.stream,
    now,
    sleep: async (ms) => void slept.push(ms),
    onWritten: (id, notes, form) => written.push([id, notes, form]),
  });
  return { w, fake, written, slept };
}

describe("writing a Transcript's notes", () => {
  it("sends the kind's prompt and the transcript, streams the text, and lands it as lines on the Transcript it started on", async () => {
    const { w, fake, written } = writer([{ chunks: ["## Summary\n", "- Si parte.\n- Dal budget [00:05].\n\n## Topics\n—"] }]);
    const seen: string[] = [];
    w.subscribe(() => seen.push(w.getState().text));
    await w.write(heard(3, 7), "work", null);
    expect(fake.asked[0].request.settings.systemPrompt).toBe(`${ASSIST_SYSTEM}\n\n${notesTask("work")}`);
    expect(fake.asked[0].request.turns).toEqual([{ role: "user", content: contextBlock(heard(3, 7)) + "27 Oct 15:42 · Call\n\n[00:00] Me: Frase 0.\n[00:05] Me: Frase 1.\n[00:10] Me: Frase 2.\n" }]);
    expect(seen).toContain("## Summary\n- Si parte.\n- Dal budget [00:05].\n\n## Topics\n—");
    expect(written).toEqual([
      [
        7,
        {
          lines: [
            { section: "Summary", text: "Si parte." },
            { section: "Summary", text: "Dal budget [00:05]." },
          ],
          writtenAs: "work",
          covered: 3,
          writtenAt: 1_000,
          fromMs: null,
        },
        "running",
      ],
    ]);
    expect(w.getState()).toEqual({ writing: null, form: "running", text: "", failed: null });
  });

  it("writes one at a time", async () => {
    const { w, fake } = writer([{ chunks: ["## A"], hold: true }]);
    const first = w.write(heard(3, 1), "general", null);
    await w.write(heard(3, 2), "general", null);
    expect(fake.asked).toHaveLength(1);
    expect(w.getState().writing).toBe(1);
    fake.release();
    await first;
  });

  it("leaves the notes there were on a Stop", async () => {
    const { w, fake, written } = writer([{ chunks: ["## Mezza"], hold: true }]);
    const writing = w.write(heard(3), "general", null);
    expect(w.getState().text).toBe("## Mezza");
    w.stop();
    await writing;
    expect(fake.asked[0].signal?.aborted).toBe(true);
    expect(written).toEqual([]);
    expect(w.getState()).toEqual({ writing: null, form: "running", text: "", failed: null });
  });

  it("says why a writing failed, and that a model which wrote nothing did", async () => {
    const { w, written } = writer([{ result: { ok: false, message: "ignis answered 409: model_kind_mismatch" } }, { chunks: ["  "] }]);
    await w.write(heard(3, 4), "general", null);
    expect(w.getState().failed).toEqual({ transcript: 4, form: "running", message: "ignis answered 409: model_kind_mismatch" });
    await w.write(heard(3, 4), "general", null);
    expect(w.getState().failed).toEqual({ transcript: 4, form: "running", message: "The model wrote nothing." });
    expect(written).toEqual([]);
  });

  it("waits out a full engine, a second at a time", async () => {
    const { w, slept, written } = writer([{ result: { ok: false, message: "engine_full: all lanes in use" } }, { chunks: ["## Ok"] }]);
    await w.write(heard(3), "general", null);
    expect(slept).toEqual([1_000]);
    expect(written).toHaveLength(1);
  });
});

describe("keeping the notes up", () => {
  const notes = (covered: number): Notes => ({ ...noNotes("general"), lines: [{ id: "N1", section: "Summary", text: "A", author: "model" }], nextId: 2, writtenAs: "general", covered, writtenAt: 0 });

  it("writes when the cadence says so, from this Transcript's last writing", async () => {
    let clock = 0;
    const { w, fake } = writer([{ chunks: ["## A"] }, { chunks: ["## B"] }], () => clock);
    await w.keepUp(heard(2), "general", null);
    expect(fake.asked).toHaveLength(0);
    await w.keepUp(heard(3), "general", null);
    expect(fake.asked).toHaveLength(1);
    clock = LIVE_GAP_MS - 1;
    await w.keepUp({ ...heard(4), notes: notes(3) }, "general", null);
    expect(fake.asked).toHaveLength(1);
    clock = LIVE_GAP_MS;
    await w.keepUp({ ...heard(4), notes: notes(3) }, "general", null);
    expect(fake.asked).toHaveLength(2);
  });

  it("waits the gap after a writing that failed, even where the cadence would write at once", async () => {
    let clock = 0;
    const { w, fake } = writer([{ result: { ok: false, message: "ignis answered 500" } }, { chunks: ["## B"] }], () => clock);
    await w.keepUp(heard(3), "general", null);
    clock = 5_000;
    await w.keepUp(heard(3), "general", null);
    expect(fake.asked).toHaveLength(1);
    clock = LIVE_GAP_MS;
    await w.keepUp(heard(3), "general", null);
    expect(fake.asked).toHaveLength(2);
  });

  it("follows a listening, and writes the recap in place of the notes when it ends, after the writing under way", async () => {
    let clock = 0;
    const { w, fake } = writer([{ chunks: ["## A"], hold: true }, { chunks: ["## B"] }], () => clock);
    const transcripts = new Map([[1, heard(3, 1)]]);
    const look = (on: number | null) => w.look(on, (id) => transcripts.get(id), () => "general", null);
    const first = look(1);
    expect(fake.asked).toHaveLength(1);
    transcripts.set(1, { ...heard(4, 1), notes: notes(3) });
    // The listening ends while the first notes are still being written: the last ones wait for them.
    await look(null);
    expect(fake.asked).toHaveLength(1);
    fake.release();
    await first;
    clock = 1_000;
    await look(null);
    expect(fake.asked).toHaveLength(2);
    expect(fake.asked[1].request.settings.systemPrompt).toBe(`${ASSIST_SYSTEM}\n\n${recapTask("general")}`);
    expect(fake.asked[1].request.turns[0].content).toContain("Frase 3.");
    await look(null);
    expect(fake.asked).toHaveLength(2);
  });

  it("writes the recap once at the end, even when the notes covered every utterance, and none when a recap already did", async () => {
    const { w, fake, written } = writer([{ chunks: ["## Summary\n- Fine."] }]);
    const transcripts = new Map([[1, { ...heard(4, 1), notes: notes(4) }]]);
    const look = (on: number | null) => w.look(on, (id) => transcripts.get(id), () => "general", null);
    await look(1);
    expect(fake.asked).toHaveLength(0);
    await look(null);
    expect(fake.asked).toHaveLength(1);
    expect(written).toEqual([[1, { lines: [{ section: "Summary", text: "Fine." }], writtenAs: "general", covered: 4, writtenAt: 1_000, fromMs: null }, "recap"]]);
    await look(null);
    expect(fake.asked).toHaveLength(1);
    // A recap pressed after the last utterance: the end has nothing to add.
    transcripts.set(2, { ...heard(4, 2), notes: notes(4), recap: { ...notes(4), writtenAt: 500 } });
    await look(2);
    await look(null);
    expect(fake.asked).toHaveLength(1);
  });
});

describe("writing the recap", () => {
  /** A Work call whose running notes hold a line of the model's, one of the user's, and one the user removed. */
  function call(): Transcript {
    let t: Transcript = { ...createTranscript(5), title: "27 Oct 15:42 · Call", context: { ...EMPTY_CONTEXT, topic: "The spring launch's budget" } };
    t = addUtterance(t, { stream: "s", startMs: 0, endMs: 2_000, text: "Il budget è di 40k.", voice: { kind: "me" } });
    t = addUtterance(t, { stream: "s", startMs: 5_000, endMs: 7_000, text: "Ada segue gli acquisti.", voice: { kind: "others" } });
    const notes: Notes = {
      ...noNotes("work"),
      lines: [
        { id: "N1", section: "Summary", text: "The budget is 50k.", author: "model" },
        { id: "N2", section: "People", text: "Ada leads procurement.", author: "user" },
      ],
      removed: [{ id: "N3", section: "People", text: "Speaker 3 is a vendor.", author: "model" }],
      nextId: 4,
      writtenAs: "work",
      covered: 2,
      writtenAt: 500,
    };
    return { ...t, notes };
  }

  it("sends one request: the recap's task, then the context, the whole transcript and only the user's own running lines, taken as true", async () => {
    const { w, fake, written } = writer([{ chunks: ["## Summary\n- Il budget è di 40k [00:00].\n\n", "## Figures & dates\n- 40k: il budget [00:00]."] }]);
    const t = call();
    const seen: string[] = [];
    w.subscribe(() => seen.push(`${w.getState().form}: ${w.getState().text}`));
    await w.writeRecap(t, "work", null);
    expect(fake.asked).toHaveLength(1);
    const { settings, turns } = fake.asked[0].request;
    expect(settings.systemPrompt).toBe(`${ASSIST_SYSTEM}\n\n${recapTask("work")}`);
    expect(settings.maxTokens).toBe(3_000);
    expect(turns).toEqual([
      {
        role: "user",
        content:
          contextBlock(t) +
          "27 Oct 15:42 · Call\n\n[00:00] Me: Il budget è di 40k.\n[00:05] Others: Ada segue gli acquisti.\n\n" +
          "Notes the user took during the conversation, to be taken as true:\n- People: Ada leads procurement.",
      },
    ]);
    expect(turns[0].content).toContain("Topic: The spring launch's budget");
    expect(turns[0].content).not.toContain("50k");
    expect(turns[0].content).not.toContain("vendor");
    expect(seen).toContain("recap: ## Summary\n- Il budget è di 40k [00:00].\n\n");
    expect(written).toEqual([
      [
        5,
        {
          lines: [
            { section: "Summary", text: "Il budget è di 40k [00:00]." },
            { section: "Figures & dates", text: "40k: il budget [00:00]." },
          ],
          writtenAs: "work",
          covered: 2,
          writtenAt: 1_000,
          fromMs: null,
        },
        "recap",
      ],
    ]);
    expect(w.getState()).toEqual({ writing: null, form: "running", text: "", failed: null });
  });

  it("says when the user took no notes", async () => {
    const { w, fake } = writer([{ chunks: ["## Summary\n- Ok."] }]);
    await w.writeRecap(heard(3), "general", null);
    expect(fake.asked[0].request.turns[0].content.endsWith("[00:10] Me: Frase 2.\n\nNotes the user took during the conversation: none.")).toBe(true);
  });

  it("is one writing at a time with the notes, and says why it failed", async () => {
    const { w, fake } = writer([{ chunks: ["## A"], hold: true }, { result: { ok: false, message: "ignis answered 500" } }]);
    const notesUnderWay = w.write(heard(3, 1), "general", null);
    await w.writeRecap(heard(3, 1), "general", null);
    expect(fake.asked).toHaveLength(1);
    fake.release();
    await notesUnderWay;
    await w.writeRecap(heard(3, 1), "general", null);
    expect(w.getState().failed).toEqual({ transcript: 1, form: "recap", message: "ignis answered 500" });
  });
});
