import { describe, expect, it } from "vitest";
import type { ChunkEvent } from "../api/sse.ts";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { ASSIST_SYSTEM } from "./assist.ts";
import { LIVE_GAP_MS, noNotes, notesTask, type Notes } from "./notes.ts";
import { NotesWriter, type Written } from "./notesWriter.ts";
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
  const written: [number, Written][] = [];
  const slept: number[] = [];
  const w = new NotesWriter({
    stream: fake.stream,
    now,
    sleep: async (ms) => void slept.push(ms),
    onWritten: (id, notes) => written.push([id, notes]),
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
      ],
    ]);
    expect(w.getState()).toEqual({ writing: null, text: "", failed: null });
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
    expect(w.getState()).toEqual({ writing: null, text: "", failed: null });
  });

  it("says why a writing failed, and that a model which wrote nothing did", async () => {
    const { w, written } = writer([{ result: { ok: false, message: "ignis answered 409: model_kind_mismatch" } }, { chunks: ["  "] }]);
    await w.write(heard(3, 4), "general", null);
    expect(w.getState().failed).toEqual({ transcript: 4, message: "ignis answered 409: model_kind_mismatch" });
    await w.write(heard(3, 4), "general", null);
    expect(w.getState().failed).toEqual({ transcript: 4, message: "The model wrote nothing." });
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
    const { w, fake } = writer([{ chunks: ["## A"] }, { chunks: ["## B"] }, { chunks: ["## C"] }], () => clock);
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
    await w.keepUp({ ...heard(5), notes: notes(4) }, "general", null, { final: true });
    expect(fake.asked).toHaveLength(3);
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

  it("follows a listening, and writes once more when it ends, after the writing under way", async () => {
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
    expect(fake.asked[1].request.turns[0].content).toContain("Frase 3.");
    await look(null);
    expect(fake.asked).toHaveLength(2);
  });
});
