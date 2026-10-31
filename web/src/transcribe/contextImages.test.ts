import { describe, expect, it } from "vitest";
import { addContextImage, freedBy, imageAt, imageMoment, pastedPicture, picturesOf, readContextImages } from "./contextImages.ts";
import { IDLE, type ListeningState } from "./listening.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// A transcript's context images (spec #428): a picture added at a moment of
// its clock, kept by its id in the page's store, several to a transcript, and
// shown on the spine at that moment.

const SLIDE = "a".repeat(64);
const BOARD = "b".repeat(64);

function call(): Transcript {
  let t = createTranscript(4, { source: "call", at: new Date(2026, 9, 31, 10, 0) });
  t = addUtterance(t, { stream: "mic-1", startMs: 2_000, endMs: 5_000, text: "Vedete la slide?", voice: { kind: "me" } });
  t = addUtterance(t, { stream: "shared-2", startMs: 6_000, endMs: 9_000, text: "Sì, quella del budget.", voice: { kind: "speaker", index: 0 } });
  return { ...t, clockMs: 12_000 };
}

const listening = (state: Partial<ListeningState>): ListeningState => ({ ...IDLE, ...state });

describe("the moment an image is added at", () => {
  it("is the listening's clock while it listens into the transcript", () => {
    expect(imageMoment(call(), listening({ phase: "listening", transcript: 4, clockMs: 7_500 }))).toBe(7_500);
    expect(imageMoment(call(), listening({ phase: "paused", transcript: 4, clockMs: 9_200 }))).toBe(9_200);
  });

  it("is where the transcript stopped when nothing listens into it, another transcript's listening or none", () => {
    expect(imageMoment(call(), IDLE)).toBe(12_000);
    expect(imageMoment(call(), listening({ phase: "listening", transcript: 9, clockMs: 40_000 }))).toBe(12_000);
  });

  it("is the start before anyone has spoken, so the picture holds from there", () => {
    const begun = createTranscript(4, { source: "call", at: new Date(2026, 9, 31, 10, 0) });
    expect(imageMoment(begun, listening({ phase: "listening", transcript: 4, clockMs: 20_000 }))).toBe(0);
  });
});

describe("the images of a transcript", () => {
  it("are several, each with an id of its own and its moment, the same picture twice named twice", () => {
    let t = addContextImage(call(), { image: SLIDE, atMs: 5_500, width: 1280, height: 720 }, 1_000);
    t = addContextImage(t, { image: BOARD, atMs: 10_000, width: 800, height: 600 }, 2_000);
    t = addContextImage(t, { image: SLIDE, atMs: 11_000, width: 1280, height: 720 }, 3_000);
    expect(t.images).toEqual([
      { id: 1, image: SLIDE, atMs: 5_500, width: 1280, height: 720, addedAt: 1_000 },
      { id: 2, image: BOARD, atMs: 10_000, width: 800, height: 600, addedAt: 2_000 },
      { id: 3, image: SLIDE, atMs: 11_000, width: 1280, height: 720, addedAt: 3_000 },
    ]);
    // The store keeps each picture once, however many times it is named.
    expect([...picturesOf(t)]).toEqual([SLIDE, BOARD]);
  });
});

describe("where an image stands on the spine", () => {
  it("is before the first utterance that starts at its moment or after: at the start, between two, or at the foot", () => {
    const { utterances } = call();
    expect(imageAt(utterances, 0)).toBe(0);
    expect(imageAt(utterances, 2_000)).toBe(0);
    // Added while the first was being said: after it.
    expect(imageAt(utterances, 3_000)).toBe(1);
    expect(imageAt(utterances, 12_000)).toBe(2);
  });
});

describe("the images as the store reads them", () => {
  it("are those this page can read, and nothing for what is not a list", () => {
    const good = { id: 1, image: SLIDE, atMs: 5_500, width: 1280, height: 720, addedAt: 1_000 };
    expect(readContextImages([good, { ...good, id: 2, image: 42 }, { ...good, id: 3, atMs: -1 }, null])).toEqual([good]);
    expect(readContextImages("a picture")).toBeNull();
    expect(readContextImages(undefined)).toBeNull();
  });
});

describe("the pictures a transcript's deletion frees", () => {
  it("are those no other transcript names, nor the chat, the Image view or a workspace; none while those cannot say", () => {
    const gone = addContextImage(addContextImage(call(), { image: SLIDE, atMs: 0, width: 1, height: 1 }, 1), { image: BOARD, atMs: 0, width: 1, height: 1 }, 2);
    const other = addContextImage(createTranscript(5), { image: BOARD, atMs: 0, width: 1, height: 1 }, 3);
    expect(freedBy(gone, [other], [])).toEqual([SLIDE]);
    expect(freedBy(gone, [], [SLIDE])).toEqual([BOARD]);
    expect(freedBy(gone, [], null)).toEqual([]);
  });
});

describe("a paste in the view", () => {
  const shot = new File(["png"], "image.png", { type: "image/png" });
  const notes = new File(["text"], "notes.txt", { type: "text/plain" });

  it("adds the picture it carries, wherever the focus is", () => {
    expect(pastedPicture([notes, shot], { intoField: false, hasText: false })).toBe(shot);
    expect(pastedPicture([shot], { intoField: true, hasText: false })).toBe(shot);
  });

  it("leaves a paste with no picture alone, and a field the text that came with a picture", () => {
    expect(pastedPicture([notes], { intoField: false, hasText: true })).toBeNull();
    expect(pastedPicture([shot], { intoField: true, hasText: true })).toBeNull();
  });
});
