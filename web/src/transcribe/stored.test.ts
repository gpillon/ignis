import { describe, expect, it } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import { openIndexedDbStore } from "../store/indexeddb.ts";
import { createMemoryStore } from "../store/memory.ts";
import type { Notes } from "./notes.ts";
import { activeOf, openTranscript, removeTranscript, startList, titleOf, UNTITLED } from "./sessions.ts";
import { restoreTranscripts, TRANSCRIBE_SESSIONS, TRANSCRIBE_VERSION, toStored, worthKeeping } from "./stored.ts";
import { addUtterance, beginListening, clockOf, createTranscript, mergeSpeakers, type Transcript } from "./transcript.ts";
import { addContextImage } from "./contextImages.ts";

// The Transcripts as the view lists them and the page's store keeps them
// (spec speech/07, "The Transcript", Q86, Q101): a new kind of session, saved
// as each utterance completes, text, times, speakers and corrections, and
// never audio.

function heardTwice(id = 3): Transcript {
  let t = createTranscript(id, { source: "call", at: new Date(2026, 9, 25, 15, 42) });
  t = addUtterance(t, { stream: "mic-1", startMs: 0, endMs: 900, text: "Ci sei?", voice: { kind: "me" } });
  t = addUtterance(t, { stream: "shared-2", startMs: 1_000, endMs: 2_000, text: "Sì.", voice: { kind: "speaker", index: 0 } });
  return { ...t, clockMs: 2_400 };
}

describe("the Transcripts listed", () => {
  it("open with one untitled Transcript; a new one is reused while nothing is heard in it", () => {
    const list = startList(1);
    expect(titleOf(activeOf(list))).toBe(UNTITLED);
    expect(openTranscript(list, 2)).toBe(list);
    const heard = { transcripts: [heardTwice(1)], activeId: 1 };
    const opened = openTranscript(heard, 2);
    expect(opened.transcripts.map((t) => t.id)).toEqual([2, 1]);
    expect(opened.activeId).toBe(2);
  });

  it("removing the open one opens the next, the last leaves a new one", () => {
    const list = { transcripts: [heardTwice(2), heardTwice(1)], activeId: 2 };
    expect(removeTranscript(list, 2, 9)).toEqual({ transcripts: [list.transcripts[1]], activeId: 1 });
    expect(removeTranscript({ transcripts: [heardTwice(1)], activeId: 1 }, 1, 9).transcripts.map((t) => t.id)).toEqual([9]);
  });

  it("a listening titles an untitled Transcript by its date, time and source, and a live one keeps its wall-clock start; a Continue keeps both", () => {
    const at = new Date(2026, 9, 25, 15, 42);
    const begun = beginListening(createTranscript(1), { source: "microphone", at });
    expect(begun).toMatchObject({ title: "25 Oct 15:42 · Microphone", source: "microphone", startedAt: at.getTime() });
    const renamed = beginListening({ ...createTranscript(1), title: "Standup" }, { source: "call", at });
    expect(renamed.title).toBe("Standup");
    const later = new Date(2026, 9, 26, 9, 0);
    expect(beginListening(heardTwice(), { source: "microphone", at: later })).toMatchObject({ title: "25 Oct 15:42 · Call", startedAt: at.getTime() });
    expect(beginListening(createTranscript(1), { source: "file", at, fileName: "talk.mp3" })).toMatchObject({ title: "25 Oct 15:42 · talk.mp3", startedAt: null });
  });
});

describe("where a Continue starts", () => {
  it("is the clock the last listening left, or past the last utterance a reload kept when it left none", () => {
    expect(clockOf(heardTwice())).toBe(2_400);
    // A reload mid-listening keeps the utterances but never hands the clock back.
    expect(clockOf({ ...heardTwice(), clockMs: 0 })).toBe(2_000);
    expect(clockOf(createTranscript(1))).toBe(0);
  });
});

describe("a Transcript in the store", () => {
  it("is plain data, its corrections with it, and comes back as it was", async () => {
    const original = mergeSpeakers(heardTwice(), 2, 1);
    const stored = toStored(original, { key: "key-1", order: 1 });
    expect(stored).toMatchObject({ version: TRANSCRIBE_VERSION, key: "key-1", order: 1 });
    expect(stored).not.toHaveProperty("id");
    // Held to the store's contract, on each adapter: a document of its own collection, a copy in and a copy out.
    for (const store of [createMemoryStore(), openIndexedDbStore(new IDBFactory(), "transcribe-test")]) {
      await store.putDoc(TRANSCRIBE_SESSIONS, stored.key, stored);
      const [back] = restoreTranscripts(await store.listDocs(TRANSCRIBE_SESSIONS), 1).transcripts;
      expect(back).toEqual({ ...original, id: back.id });
    }
    expect(JSON.stringify(stored)).not.toMatch(/audio|pcm/i);
  });

  it("keeps its notes as lines (Q124), the user's and the removed ones with them, and drops notes it cannot read without dropping the Transcript", async () => {
    const notes: Notes = {
      kind: "work",
      lines: [
        { id: "N1", section: "Summary", text: "Si parte dal budget.", author: "model" },
        { id: "N3", section: "Decisions", text: "Tetto a 40k [00:01]", author: "user" },
      ],
      removed: [{ id: "N2", section: "Topics", text: "Il pranzo", author: "model" }],
      nextId: 4,
      writtenAs: "work",
      covered: 2,
      writtenAt: 1_761_580_000_000,
      fromMs: null,
    };
    const original = { ...heardTwice(), notes };
    for (const store of [createMemoryStore(), openIndexedDbStore(new IDBFactory(), "transcribe-notes-test")]) {
      await store.putDoc(TRANSCRIBE_SESSIONS, "k", toStored(original, { key: "k", order: 1 }));
      const [back] = restoreTranscripts(await store.listDocs(TRANSCRIBE_SESSIONS), 1).transcripts;
      expect(back).toEqual({ ...original, id: back.id });
    }
    const garbled = { ...toStored(original, { key: "k", order: 1 }), notes: { ...notes, lines: [{ id: "N1", section: "Summary", text: 3, author: "model" }] } };
    const [kept] = restoreTranscripts([garbled], 1).transcripts;
    expect(kept.utterances).toHaveLength(2);
    expect(kept).not.toHaveProperty("notes");
  });

  it("turns notes an older page kept as Markdown into lines on first read, all of them the model's", () => {
    const old = { kind: "work", text: "## Summary\nSi parte dal budget.\n\n## Action items\n- [ ] Ada manda le slide [00:01]\n\n## Decisions\n—", writtenAs: "work", covered: 2, writtenAt: 1_761_580_000_000, fromMs: null };
    const [back] = restoreTranscripts([{ ...toStored(heardTwice(), { key: "k", order: 1 }), notes: old }], 1).transcripts;
    expect(back.notes).toEqual({
      kind: "work",
      lines: [
        { id: "N1", section: "Summary", text: "Si parte dal budget.", author: "model" },
        { id: "N2", section: "Action items", text: "[ ] Ada manda le slide [00:01]", author: "model" },
      ],
      removed: [],
      nextId: 3,
      writtenAs: "work",
      covered: 2,
      writtenAt: 1_761_580_000_000,
      fromMs: null,
    });
    const [kept] = restoreTranscripts([{ ...toStored(heardTwice(), { key: "k", order: 1 }), notes: { ...old, text: 3 } }], 1).transcripts;
    expect(kept.utterances).toHaveLength(2);
    expect(kept).not.toHaveProperty("notes");
  });

  it("keeps what it is told (Q128), and drops a context it cannot read without dropping the Transcript", () => {
    const context = { languages: { mode: "force" as const, codes: ["it"] }, topic: "Supporto IT", terms: ["PostgreSQL"], other: "", appliedAt: 1_761_580_000_000 };
    const [back] = restoreTranscripts([toStored({ ...heardTwice(), context }, { key: "k", order: 1 })], 1).transcripts;
    expect(back.context).toEqual(context);
    const [kept] = restoreTranscripts([{ ...toStored(heardTwice(), { key: "k", order: 1 }), context: "Standup." }], 1).transcripts;
    expect(kept.utterances).toHaveLength(2);
    expect(kept).not.toHaveProperty("context");
  });

  it("keeps where the text model's look read it up to (Q140)", () => {
    const [back] = restoreTranscripts([toStored({ ...heardTwice(), checkedUntilMs: 2_000 }, { key: "k", order: 1 })], 1).transcripts;
    expect(back.checkedUntilMs).toBe(2_000);
    const [none] = restoreTranscripts([{ ...toStored(heardTwice(), { key: "k", order: 1 }), checkedUntilMs: "x" }], 1).transcripts;
    expect(none).not.toHaveProperty("checkedUntilMs");
  });

  it("is worth keeping once an utterance is heard in it", () => {
    expect(worthKeeping(createTranscript(1))).toBe(false);
    expect(worthKeeping(heardTwice())).toBe(true);
  });

  it("keeps its context images, each by its picture's id at its moment, and drops those it cannot read without dropping the Transcript (#431)", async () => {
    const slide = "c".repeat(64);
    let original = addContextImage(heardTwice(), { image: slide, atMs: 1_500, width: 1280, height: 720 }, 1_761_900_000_000);
    original = addContextImage(original, { image: slide, atMs: 2_400, width: 1280, height: 720 }, 1_761_900_060_000);
    const stored = toStored(original, { key: "k", order: 1 });
    for (const store of [createMemoryStore(), openIndexedDbStore(new IDBFactory(), "transcribe-images-test")]) {
      await store.putDoc(TRANSCRIBE_SESSIONS, stored.key, stored);
      const [back] = restoreTranscripts(await store.listDocs(TRANSCRIBE_SESSIONS), 1).transcripts;
      expect(back.images).toEqual(original.images);
    }
    const [some] = restoreTranscripts([{ ...stored, images: [stored.images?.[0], { id: 2, image: 7 }] }], 1).transcripts;
    expect(some.images).toEqual([original.images?.[0]]);
    const [kept] = restoreTranscripts([{ ...stored, images: "a slide" }], 1).transcripts;
    expect(kept.utterances).toHaveLength(2);
    expect(kept).not.toHaveProperty("images");
  });

  it("is worth keeping, and is not reused as a new one, once a picture is added before anyone spoke (#431)", () => {
    const shown = addContextImage(createTranscript(1, { source: "call", at: new Date(2026, 9, 31, 10, 0) }), { image: "d".repeat(64), atMs: 0, width: 640, height: 480 }, 1);
    expect(worthKeeping(shown)).toBe(true);
    expect(openTranscript({ transcripts: [shown], activeId: 1 }, 2).activeId).toBe(2);
  });

  it("comes back newest first with new ids, and leaves out what it cannot read", () => {
    const older = toStored(heardTwice(9), { key: "a", order: 2 });
    const newer = toStored(heardTwice(2), { key: "b", order: 9 });
    const { transcripts, stamps, nextId } = restoreTranscripts([older, null, { version: TRANSCRIBE_VERSION + 1 }, newer, { version: 1 }], 5);
    expect(transcripts.map((t) => t.id)).toEqual([6, 5]);
    expect(stamps.get(6)).toEqual({ key: "b", order: 9 });
    expect(nextId).toBe(7);
  });
});
