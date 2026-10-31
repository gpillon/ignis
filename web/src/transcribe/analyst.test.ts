import { describe, expect, it } from "vitest";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { Analyst } from "./analyst.ts";
import { ASSIST_SYSTEM } from "./assist.ts";
import { ANALYSIS_GAP_MS, ANALYSIS_TASK, type Analysis } from "./suggestions.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// The look for suggestions (spec speech/07, Q132): one at a time, its own
// task in the system prompt, the transcript after it; what it found goes to
// the Transcript it read; a failed look waits the gap.

function heard(n: number, id = 1): Transcript {
  let t = { ...createTranscript(id), title: "27 Oct 15:42 · Call" };
  for (let i = 0; i < n; i++) t = addUtterance(t, { stream: "s", startMs: i * 5_000, endMs: i * 5_000 + 2_000, text: `Frase ${i}.`, voice: { kind: "me" } });
  return t;
}

function analyst(answers: (string | Error)[], now = () => 0) {
  const asked: StreamOptions[] = [];
  const found: [number, Analysis, number][] = [];
  const stream = async (options: StreamOptions): Promise<StreamResult> => {
    asked.push(options);
    const answer = answers.shift() ?? "NONE";
    const timeline = { sentAt: 0, stopped: false };
    if (answer instanceof Error) return { ok: false, message: answer.message, timeline };
    options.onEvent({ kind: "content", text: answer });
    return { ok: true, timeline };
  };
  const a = new Analyst({ stream, now, sleep: async () => {}, onFound: (id, f, until) => found.push([id, f, until]) });
  return { a, asked, found };
}

describe("a look for suggestions", () => {
  it("sends its task and the transcript, and hands what it found to the Transcript it read", async () => {
    const { a, asked, found } = analyst(["TOPIC: Supporto IT\nTERM: PostgreSQL"]);
    await a.analyze(heard(6, 4), null);
    expect(asked[0].request.settings.systemPrompt).toBe(`${ASSIST_SYSTEM}\n\n${ANALYSIS_TASK}`);
    expect(asked[0].request.turns[0].content).toContain("- Write in: English.");
    expect(asked[0].request.turns[0].content).toContain("\n\n27 Oct 15:42 · Call\n\n[00:00] Me: Frase 0.\n");
    expect(found).toEqual([[4, { topic: "Supporto IT", title: null, terms: ["PostgreSQL"], fixes: [] }, 27_000]]);
    expect(a.getState()).toEqual({ looking: null, nextAt: { 4: ANALYSIS_GAP_MS }, failed: null });
  });

  it("says why one failed, and waits the gap before the next while listening", async () => {
    let clock = 0;
    const { a, asked } = analyst([new Error("ignis answered 500"), "NONE"], () => clock);
    await a.keepUp(heard(6), null);
    expect(a.getState().failed).toEqual({ transcript: 1, message: "ignis answered 500" });
    clock = 5_000;
    await a.keepUp(heard(9), null);
    expect(asked).toHaveLength(1);
    clock = ANALYSIS_GAP_MS;
    await a.keepUp(heard(9), null);
    expect(asked).toHaveLength(2);
  });

  it("counts what it has not read by time, so a deleted utterance leaves the new ones to read", async () => {
    let clock = 0;
    const { a, asked } = analyst(["NONE", "NONE"], () => clock);
    await a.keepUp(heard(6), null);
    // Two of the six deleted, then two new ones: the end's look still reads them.
    const t = heard(8);
    const after = { ...t, utterances: [t.utterances[0], t.utterances[1], t.utterances[2], t.utterances[3], t.utterances[6], t.utterances[7]] };
    clock = 1_000;
    await a.keepUp(after, null, { final: true });
    expect(asked).toHaveLength(2);
  });

  it("follows a listening and looks once more when it ends", async () => {
    let clock = 0;
    const { a, asked } = analyst(["NONE", "NONE"], () => clock);
    const transcripts = new Map([[1, heard(6)]]);
    await a.look(1, (id) => transcripts.get(id), null);
    expect(asked).toHaveLength(1);
    transcripts.set(1, heard(7));
    clock = 1_000;
    await a.look(1, (id) => transcripts.get(id), null);
    expect(asked).toHaveLength(1);
    await a.look(null, (id) => transcripts.get(id), null);
    expect(asked).toHaveLength(2);
  });
});
