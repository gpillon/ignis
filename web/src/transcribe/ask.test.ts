import { describe, expect, it } from "vitest";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { ASK_TASK, Asker, answerAt, askRequest, quickQuestions, readAnswered, withAnswer } from "./ask.ts";
import { ASSIST_SYSTEM } from "./assist.ts";
import { contextBlock } from "./taskContext.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// Ask about this conversation (spec speech/07, Q139): the transcript, the
// question last, on the interactive lane with thinking off; the answer
// streams, and lands on the Transcript asked about.

function heard(): Transcript {
  let t = { ...createTranscript(3), title: "27 Oct 15:42 · Call" };
  t = addUtterance(t, { stream: "s", startMs: 4_000, endMs: 6_000, text: "Il backup si fa venerdì.", voice: { kind: "me" } });
  return t;
}

describe("a question's request", () => {
  it("is the task, then the transcript and the question last", () => {
    const request = askRequest(heard(), "  Quando il backup? ", null);
    expect(request.settings).toMatchObject({ systemPrompt: `${ASSIST_SYSTEM}\n\n${ASK_TASK}`, reasoningEffort: "none", laneTag: "interactive" });
    expect(request.turns[0].content).toBe(contextBlock(heard()) + "27 Oct 15:42 · Call\n\n[00:04] Me: Il backup si fa venerdì.\n\nQuestion: Quando il backup?");
  });
});

describe("the answers", () => {
  it("are kept latest first, and read back those this page can read", () => {
    const one = withAnswer(undefined, "Quando?", "Venerdì [00:04].", 1);
    const two = withAnswer(one, "Chi?", "Me.", 2);
    expect(two.map((a) => [a.id, a.question])).toEqual([
      [2, "Chi?"],
      [1, "Quando?"],
    ]);
    expect(readAnswered([...two, { id: "x" }])).toEqual(two);
    expect(readAnswered("x")).toBeUndefined();
    // Up to where each read, its place in the sheet: kept, or dropped alone when it cannot be read.
    const placed = withAnswer(undefined, "Quando?", "Venerdì.", 1, 6_000);
    expect(placed[0].untilMs).toBe(6_000);
    expect(readAnswered(placed)).toEqual(placed);
    expect(readAnswered([{ ...placed[0], untilMs: "x" }])).toEqual([{ id: 1, question: "Quando?", answer: "Venerdì.", at: 1 }]);
  });

  it("stand after the last utterance they read, or at the foot", () => {
    const t = addUtterance(heard(), { stream: "s", startMs: 10_000, endMs: 12_000, text: "Poi?", voice: { kind: "me" } });
    expect(answerAt(t.utterances, 6_000)).toBe(1);
    expect(answerAt(t.utterances, 12_000)).toBe(2);
    expect(answerAt(t.utterances, 0)).toBe(0);
    expect(answerAt(t.utterances, undefined)).toBe(2);
  });

  it("stand below every utterance they read on a call, one of the other lane begun before and ended after above them", () => {
    // In time order by start: Me at 4-6 s, the far side's long one at 5-20 s, Me again at 10-12 s; asked at 12 s.
    let t = addUtterance(heard(), { stream: "c", startMs: 5_000, endMs: 20_000, text: "Allora, per il rilascio…", voice: { kind: "speaker", index: 0 } });
    t = addUtterance(t, { stream: "s", startMs: 10_000, endMs: 12_000, text: "Poi?", voice: { kind: "me" } });
    expect(t.utterances.map((u) => u.startMs)).toEqual([4_000, 5_000, 10_000]);
    expect(answerAt(t.utterances, 12_000)).toBe(3);
  });

  it("stream, and land on the Transcript asked about, with up to where they read", async () => {
    const answered: [number, string, string, number][] = [];
    const seen: string[] = [];
    const stream = async (options: StreamOptions): Promise<StreamResult> => {
      for (const text of ["Venerdì ", "[00:04]."]) options.onEvent({ kind: "content", text });
      return { ok: true, timeline: { sentAt: 0, stopped: false } };
    };
    const asker = new Asker({ stream, onAnswered: (id, q, a, until) => answered.push([id, q, a, until]) });
    asker.subscribe(() => seen.push(asker.getState().text));
    await asker.ask(heard(), "Quando il backup?", null);
    expect(seen).toContain("Venerdì [00:04].");
    expect(answered).toEqual([[3, "Quando il backup?", "Venerdì [00:04].", 6_000]]);
    expect(asker.getState().writing).toBe(false);
  });

  it("show a quick question as asked and send the model its own wording", async () => {
    const sent: string[] = [];
    const stream = async (options: StreamOptions): Promise<StreamResult> => {
      sent.push(String(options.request.turns.at(-1)?.content));
      options.onEvent({ kind: "content", text: "Niente." });
      return { ok: true, timeline: { sentAt: 0, stopped: false } };
    };
    const answered: string[] = [];
    const asker = new Asker({ stream, onAnswered: (_id, q) => answered.push(q) });
    await asker.ask(heard(), "Decisions so far?", null, "What has been decided? Answer in the language of the conversation.");
    expect(sent[0]).toMatch(/Question: What has been decided\? Answer in the language of the conversation\.$/);
    expect(answered).toEqual(["Decisions so far?"]);
  });
});

describe("the quick questions", () => {
  it("ask for the conversation's language, and What did I miss? from the first utterance of the last five minutes", () => {
    let t = heard();
    t = addUtterance(t, { stream: "s", startMs: 300_000, endMs: 303_000, text: "Ripartiamo.", voice: { kind: "me" } });
    t = addUtterance(t, { stream: "s", startMs: 340_000, endMs: 345_000, text: "Fatto.", voice: { kind: "me" } });
    const [missed, decisions, owners] = quickQuestions(t);
    // 345 s less five minutes is 45 s: the first utterance ending after it began at 05:00.
    expect(missed).toMatchObject({ label: "What did I miss?", question: "What was said since 05:00?" });
    expect(missed.sent).toBe("What was said from [05:00] on? The main points, briefly. Answer in the language of the conversation.");
    expect(decisions.sent).toMatch(/Answer in the language of the conversation\.$/);
    expect(owners.label).toBe("Who does what");
    // A call shorter than five minutes: all of it.
    expect(quickQuestions(heard())[0].question).toBe("What was said since 00:04?");
  });
});
