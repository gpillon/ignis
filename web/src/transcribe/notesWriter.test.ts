import { describe, expect, it } from "vitest";
import type { ChunkEvent } from "../api/sse.ts";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { ASSIST_SYSTEM, CHARS_PER_TOKEN } from "./assist.ts";
import { EMPTY_CONTEXT } from "./context.ts";
import { addLine, editLine, removeLine } from "./noteEdits.ts";
import { LIVE_GAP_MS, noNotes, type Notes, RUNNING_NOTES } from "./notes.ts";
import { NotesWriter, type Written } from "./notesWriter.ts";
import { recapTask } from "./recap.ts";
import { acceptRewrite } from "./rewrite.ts";
import { runningTask } from "./running.ts";
import { contextBlock } from "./taskContext.ts";
import {
  addUtterance,
  createTranscript,
  editText,
  fixUtterances,
  mergeSpeakers,
  moveUtterance,
  removeUtterance,
  renameSpeaker,
  restoreUtterance,
  splitUtterance,
  type Transcript,
} from "./transcript.ts";

// The notes' writer (spec #428, "Running notes: the notes thread"): one
// append-only thread with the text model per Transcript, its four tools
// applied by the page a call at a time, the result of each sent back in one
// line; and the recap, one writing over the whole transcript. One writing at a
// time, a Stop leaves what was applied, a full engine is waited out, and
// keeping up writes when notes.ts's cadence says so.

function heard(n: number, id = 1): Transcript {
  let t = { ...createTranscript(id), title: "27 Oct 15:42 · Call" };
  for (let i = 0; i < n; i++) t = addUtterance(t, { stream: "s", startMs: i * 5_000, endMs: i * 5_000 + 2_000, text: `Frase ${i}.`, voice: { kind: "me" } });
  return t;
}

/** A model's tool call as the stream hands it over. */
const call = (id: string, name: string, args: unknown): ChunkEvent => ({ kind: "tool_call", call: { id, name, arguments: JSON.stringify(args) } });

/** A request's reported prompt, as the usage chunk gives it. */
const prompt = (tokens: number): ChunkEvent => ({ kind: "usage", usage: { prompt_tokens: tokens, completion_tokens: 10, total_tokens: tokens + 10 } });

type Script = { chunks?: string[]; events?: ChunkEvent[]; result?: Partial<StreamResult> & { ok: boolean; message?: string }; hold?: true };

/** A stream that answers each request with the next script: chunks and events, then a result. */
function fakeStream(scripts: Script[]) {
  const asked: StreamOptions[] = [];
  let release: () => void = () => {};
  const stream = async (options: StreamOptions): Promise<StreamResult> => {
    asked.push(options);
    const script = scripts.shift() ?? { chunks: [] };
    for (const text of script.chunks ?? []) options.onEvent({ kind: "content", text } satisfies ChunkEvent);
    for (const event of script.events ?? []) options.onEvent(event);
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

/** A writer on a fake stream, and the page it writes to: each Transcript's running notes, and the recaps landed. */
function writer(scripts: Script[], now = () => 1_000) {
  const fake = fakeStream(scripts);
  const page = new Map<number, Notes | undefined>();
  const recaps: [number, Written][] = [];
  const slept: number[] = [];
  const w = new NotesWriter({
    stream: fake.stream,
    now,
    sleep: async (ms) => void slept.push(ms),
    onNotes: (id, change) => page.set(id, change(page.get(id))),
    onRecap: (id, written) => recaps.push([id, written]),
  });
  /** `t` as the page holds it now: its notes the ones the writer left. */
  const held = (t: Transcript): Transcript => (page.has(t.id) ? { ...t, notes: page.get(t.id) } : (page.set(t.id, t.notes), t));
  return { w, fake, page, recaps, slept, held };
}

/** The request each round sent: its turns. */
const turnsOf = (asked: StreamOptions[], round: number) => asked[round].request.turns;

describe("the notes thread", () => {
  it("starts with the running task and four tools, then the context, the title, the transcript and that the notes are empty", async () => {
    const { w, fake, page, held } = writer([{ events: [call("c1", "add_line", { section: "Now", text: "Si parte dal budget [00:05]." })] }, {}]);
    const t = held(heard(3, 7));
    await w.write(t, "work", null);
    expect(fake.asked).toHaveLength(2);
    const { settings, extras } = fake.asked[0].request;
    expect(settings.systemPrompt).toBe(`${ASSIST_SYSTEM}\n\n${runningTask("work")}`);
    expect(settings).toMatchObject({ reasoningEffort: "none", laneTag: "agent" });
    expect(extras?.tools?.map((tool) => tool.function.name)).toEqual(["add_line", "replace_line", "remove_line", "rewrite_section"]);
    const first = { role: "user", content: contextBlock(t) + "27 Oct 15:42 · Call\n\n[00:00] Me: Frase 0.\n[00:05] Me: Frase 1.\n[00:10] Me: Frase 2.\n\nThe notes are empty." };
    expect(turnsOf(fake.asked, 0)).toEqual([first]);
    // The call is applied, and its result goes back in one line with the next request.
    expect(turnsOf(fake.asked, 1)).toEqual([
      first,
      { role: "assistant", content: "", toolCalls: [{ id: "c1", name: "add_line", arguments: JSON.stringify({ section: "Now", text: "Si parte dal budget [00:05]." }) }] },
      { role: "tool", content: "Added N1 to Now.", toolCallId: "c1" },
    ]);
    expect(page.get(7)).toEqual({
      ...noNotes("work"),
      lines: [{ id: "N1", section: "Now", text: "Si parte dal budget [00:05].", author: "model" }],
      nextId: 2,
      writtenAs: "work",
      covered: 3,
      writtenAt: 1_000,
    });
    expect(w.getState()).toEqual({ writing: null, form: "running", text: "", failed: null });
  });

  it("appends: a later update's request extends the last one's with what was said since", async () => {
    const { w, fake, held } = writer([{ events: [call("c1", "add_line", { section: "People", text: "Me: apre la call." })] }, {}, {}]);
    await w.write(held(heard(3)), "work", null);
    await w.write(held(heard(5)), "work", null);
    expect(fake.asked).toHaveLength(3);
    expect(turnsOf(fake.asked, 2)).toEqual([...turnsOf(fake.asked, 1), { role: "assistant", content: "" }, { role: "user", content: "Said since [00:15]:\n[00:15] Me: Frase 3.\n[00:20] Me: Frase 4.\n" }]);
    expect(fake.asked[2].request.settings).toEqual(fake.asked[0].request.settings);
    expect(fake.asked[2].request.extras).toEqual(fake.asked[0].request.extras);
  });

  /** A long call: `n` utterances of about a hundred tokens each, `Frase i.` first in each. */
  function long(n: number): Transcript {
    let t = { ...createTranscript(1), title: "27 Oct 15:42 · Call" };
    for (let i = 0; i < n; i++) t = addUtterance(t, { stream: "s", startMs: i * 20_000, endMs: i * 20_000 + 18_000, text: `Frase ${i}. ${"parola ".repeat(44)}`, voice: { kind: "me" } });
    return t;
  }

  it("starts with the whole transcript when it fits the model's context, past the threshold too", async () => {
    // About 43,000 tokens of transcript, at CHARS_PER_TOKEN: more than the threshold's half, less than the context.
    for (const contextLimit of [262_144, null]) {
      const { w, fake, page, held } = writer([{}]);
      await w.write(held(long(400)), "work", contextLimit);
      const first = turnsOf(fake.asked, 0)[0].content;
      expect(first).toContain("\n\n[00:00] Me: Frase 0. ");
      expect(first).toContain("\n[133:00] Me: Frase 399. ");
      expect(page.get(1)?.fromMs).toBeNull();
    }
  });

  it("starts from where the transcript's tail fits the model's context, less the threshold the thread grows by", async () => {
    const { w, fake, page, held } = writer([{}]);
    await w.write(held(long(400)), "work", 32_768);
    const first = turnsOf(fake.asked, 0)[0].content;
    expect(first).not.toContain("Frase 0. ");
    expect(first).toContain("\n[133:00] Me: Frase 399. ");
    // Half the context is the threshold on a context this small: the first turn fills most of the other half.
    expect(first.length).toBeLessThanOrEqual(16_384 * CHARS_PER_TOKEN);
    expect(first.length).toBeGreaterThan(12_000 * CHARS_PER_TOKEN);
    // The notes say from when: the first utterance it carries.
    const [, m, s] = first.match(/\n\[(\d+):(\d+)\] Me: Frase/)!;
    expect(page.get(1)?.fromMs).toBe((Number(m) * 60 + Number(s)) * 1_000);
  });

  it("ends an update when two rounds in a row change nothing", async () => {
    const same = (id: string) => ({ events: [call(id, "replace_line", { id: "N1", text: "Il budget." })] });
    const { w, fake, page, held } = writer([{ events: [call("c1", "add_line", { section: "Now", text: "Il budget." })] }, same("c2"), same("c3"), same("c4")]);
    await w.write(held(heard(3)), "work", null);
    expect(fake.asked).toHaveLength(3);
    expect(turnsOf(fake.asked, 2).at(-1)).toEqual({ role: "tool", content: "Unchanged: N1 already reads so. When the notes are right, answer with no call.", toolCallId: "c2" });
    expect(page.get(1)?.lines).toEqual([{ id: "N1", section: "Now", text: "Il budget.", author: "model" }]);
    expect(w.getState().writing).toBeNull();
  });

  it("counts a round whose calls were refused, or not JSON, as one that changed nothing", async () => {
    // A section the model will not cut below its cap: the same over-cap rewrite again and again, then arguments cut short.
    const overCap = (id: string) => call(id, "rewrite_section", { section: "Now", lines: ["Il budget.", "Le date."] });
    const { w, fake, held } = writer([
      { events: [call("c1", "add_line", { section: "Now", text: "Il budget." })] },
      { events: [overCap("c2")] },
      { events: [overCap("c3"), { kind: "tool_call", call: { id: "c4", name: "add_line", arguments: '{"section": "Now"' } }] },
      { events: [overCap("c5")] },
    ]);
    await w.write(held(heard(3)), "work", null);
    expect(fake.asked).toHaveLength(3);
    expect(turnsOf(fake.asked, 2).filter((turn) => turn.role === "tool").map((turn) => turn.content)).toEqual([
      "Added N1 to Now.",
      "Refused: 2 lines; Now holds 1 at most. Send 1 or fewer: merge lines that belong together, or drop the least useful.",
    ]);
    expect(w.getState().writing).toBeNull();
  });

  it("counts a round that changed a line as one that did, whatever else it refused", async () => {
    const { w, fake, held } = writer([
      { events: [call("c1", "add_line", { section: "Decisions", text: "Si parte." })] },
      { events: [call("c2", "add_line", { section: "Decisions", text: "Si parte." }), call("c3", "add_line", { section: "Now", text: "Il budget." })] },
      { events: [call("c4", "add_line", { section: "Now", text: "Le date." })] },
      { events: [call("c5", "add_line", { section: "Now", text: "Le date." })] },
      {},
    ]);
    await w.write(held(heard(3)), "work", null);
    // Refused, applied with a refusal, refused, refused: the last two end the update.
    expect(fake.asked).toHaveLength(4);
  });

  it("ends an update at the round cap, and what is left goes to the next one", async () => {
    const add = (n: number) => ({ events: [call(`c${n}`, "add_line", { section: "Figures & dates", text: `${n}0k di budget.` })] });
    const { w, fake, page, held } = writer([1, 2, 3, 4, 5, 6, 7].map(add));
    await w.write(held(heard(3)), "work", null);
    expect(RUNNING_NOTES.maxRounds).toBe(6);
    expect(fake.asked).toHaveLength(6);
    expect(page.get(1)?.lines.map((l) => l.text)).toEqual(["10k di budget.", "20k di budget.", "30k di budget.", "40k di budget.", "50k di budget.", "60k di budget."]);
    // The next update picks up after the last result, with what was said since.
    await w.write(held(heard(4)), "work", null);
    expect(fake.asked).toHaveLength(8);
    const sixth = [...turnsOf(fake.asked, 5), { role: "assistant", content: "", toolCalls: [{ id: "c6", name: "add_line", arguments: JSON.stringify({ section: "Figures & dates", text: "60k di budget." }) }] }, { role: "tool", content: "Added N6 to Figures & dates.", toolCallId: "c6" }];
    expect(turnsOf(fake.asked, 6)).toEqual([...sixth, { role: "user", content: "Said since [00:15]:\n[00:15] Me: Frase 3.\n" }]);
    expect(page.get(1)?.lines).toHaveLength(7);
  });
});

describe("the page's guards", () => {
  /** A Work call's running notes: a line of the model's and one of the user's in People, Now full, one line the user removed. */
  const kept = (): Notes => ({
    ...noNotes("work"),
    lines: [
      { id: "N1", section: "People", text: "Ada: acquisti.", author: "model" },
      { id: "N2", section: "People", text: "Bea: fornitore.", author: "user" },
      { id: "N3", section: "Now", text: "Il budget.", author: "model" },
    ],
    removed: [{ id: "N4", section: "People", text: "Speaker 3: un fornitore.", author: "model" }],
    nextId: 5,
    writtenAs: "work",
    covered: 3,
    writtenAt: 500,
  });

  it("strips copied ids, and refuses what cannot stand, each in one line, leaving the notes as they were", async () => {
    const calls: ChunkEvent[] = [
      call("c1", "add_line", { section: "People", text: "N1: Carlo: finanza." }),
      call("c2", "add_line", { section: "People", text: "ada: acquisti." }),
      call("c3", "add_line", { section: "Figures & dates", text: "x".repeat(241) }),
      call("c4", "add_line", { section: "Decisions", text: "Si parte." }),
      call("c5", "replace_line", { id: "N9", text: "Chi?" }),
      call("c6", "replace_line", { id: "N1", text: "  " }),
      call("c7", "add_line", { section: "Now", text: "Le date." }),
      call("c8", "rewrite_section", { section: "People", lines: ["Ada: acquisti."] }),
      call("c9", "rewrite_section", { section: "Now", lines: ["Le date.", "Il budget."] }),
      call("c10", "replace_line", { id: "N2", text: "Bea: la fornitrice." }),
      call("c11", "remove_line", { id: "N2" }),
      call("c12", "add_line", { section: "People", text: "Speaker 3: un fornitore." }),
      call("c13", "replace_line", { id: "N3", text: "Il budget. (N3)" }),
      { kind: "tool_call", call: { id: "c14", name: "add_line", arguments: '{"section": "Now"' } },
      call("c15", "clear_notes", {}),
    ];
    const { w, fake, page, held } = writer([{ events: calls }, {}]);
    await w.write(held({ ...heard(3), notes: kept() }), "work", null);
    expect(
      turnsOf(fake.asked, 1)
        .filter((turn) => turn.role === "tool")
        .map((turn) => turn.content),
    ).toEqual([
      "Added N5 to People.",
      "Refused: People already has that line.",
      "Refused: the line is 241 characters; a line holds 240 at most. Write one short point.",
      "Refused: no section named 'Decisions'. Sections: Now, People, Figures & dates, Asked of me, Commitments heard.",
      "Refused: no line 'N9'.",
      "Refused: the text is empty; remove_line removes a line.",
      "Refused: Now is full (1 line). Rewrite it with rewrite_section, merging what belongs together and folding this in.",
      "Refused: People has 2 lines, not 20: change single lines. rewrite_section is for a full section.",
      "Refused: 2 lines; Now holds 1 at most. Send 1 or fewer: merge lines that belong together, or drop the least useful.",
      "Refused: N2 is the user's: leave it as it is.",
      "Refused: N2 is the user's: leave it as it is.",
      "Refused: the user removed that line from People; never write it again.",
      "Unchanged: N3 already reads so. When the notes are right, answer with no call.",
      "Refused: the arguments are not JSON.",
      "Refused: no tool 'clear_notes'.",
    ]);
    // Only the first call changed the notes, its id stripped from the text.
    expect(page.get(1)).toEqual({ ...kept(), lines: [...kept().lines, { id: "N5", section: "People", text: "Carlo: finanza.", author: "model" }], nextId: 6, writtenAt: 1_000 });
  });

  it("rewrites a full section, a line kept word for word keeping its id, the user's lines left where they are", async () => {
    const asked = ["Il preventivo?", "La data?", "Il listino?", "Il contratto?", "La fattura?", "Il budget?"];
    const full: Notes = {
      ...noNotes("work"),
      lines: [...asked.map((text, i) => ({ id: `N${i + 1}`, section: "Asked of me", text, author: "model" as const })), { id: "N7", section: "Asked of me", text: "Il mio orario?", author: "user" }],
      nextId: 8,
      writtenAs: "work",
      covered: 3,
      writtenAt: 500,
    };
    const lines = ["Il preventivo?", "La data?", "N3 Il listino?", "Il mio orario?", "La consegna?"];
    const { w, fake, page, held } = writer([{ events: [call("c1", "rewrite_section", { section: "Asked of me", lines })] }, {}]);
    await w.write(held({ ...heard(3), notes: full }), "work", null);
    expect(turnsOf(fake.asked, 1).at(-1)?.content).toBe("Asked of me rewritten: 3 kept, 1 new, 3 removed.");
    expect(page.get(1)?.lines.map((l) => `${l.id} ${l.text}`)).toEqual(["N7 Il mio orario?", "N1 Il preventivo?", "N2 La data?", "N3 Il listino?", "N8 La consegna?"]);
    expect(page.get(1)?.nextId).toBe(9);
  });

  it("tells the model a moment in total minutes, past the hour too", async () => {
    const late = addUtterance(heard(3), { stream: "s", startMs: 3_849_000, endMs: 3_851_000, text: "Ci vediamo alle cinque.", voice: { kind: "me" } });
    const { w, fake, held } = writer([{}, {}]);
    await w.write(held(heard(3)), "work", null);
    await w.write(held(late), "work", null);
    expect(turnsOf(fake.asked, 1).at(-1)).toEqual({ role: "user", content: "Said since [64:09]:\n[64:09] Me: Ci vediamo alle cinque.\n" });
    expect(runningTask("work")).toContain("Point at a moment by its time, as [mm:ss], in minutes from the start: [64:09] is sixty-four minutes in.");
  });
});

describe("starting the thread again", () => {
  /** Running notes of a Work call: a line of the model's, one of the user's in People and one in Figures & dates, one removed. */
  const kept = (): Notes => ({
    ...noNotes("work"),
    lines: [
      { id: "N1", section: "Now", text: "Il budget.", author: "model" },
      { id: "N2", section: "People", text: "Bea: fornitore.", author: "user" },
      { id: "N3", section: "Figures & dates", text: "40k: il tetto.", author: "user" },
    ],
    removed: [{ id: "N4", section: "People", text: "Speaker 3: un fornitore.", author: "model" }],
    nextId: 5,
    writtenAs: "work",
    covered: 3,
    writtenAt: 500,
  });

  /** The first turn of a thread started over, after the transcript so far. */
  const snapshot = [
    "The notes so far, each line with its id:",
    "Now:\nN1 Il budget.",
    "People:\nN2 (the user's) Bea: fornitore.",
    "Figures & dates:\nN3 (the user's) 40k: il tetto.",
    "Asked of me:\n(empty)",
    "Commitments heard:\n(empty)",
    "Lines the user removed, never to be written again:\n- People: Speaker 3: un fornitore.",
  ].join("\n\n");

  it("after a reload, from a snapshot of the notes: each line with its id, the user's marked, the ones they removed listed", async () => {
    const { w, fake, held } = writer([{}]);
    const t = held({ ...heard(4), notes: kept() });
    await w.write(t, "work", null);
    expect(turnsOf(fake.asked, 0)).toEqual([{ role: "user", content: `${contextBlock(t)}27 Oct 15:42 · Call\n\n[00:00] Me: Frase 0.\n[00:05] Me: Frase 1.\n[00:10] Me: Frase 2.\n[00:15] Me: Frase 3.\n\n${snapshot}` }]);
  });

  it("once its prompt grew past the threshold beyond its first request's", async () => {
    const base = 5_000;
    const { w, fake, held } = writer([
      { events: [prompt(base)] },
      { events: [prompt(base + RUNNING_NOTES.restartTokens - 100)] },
      { events: [prompt(base + RUNNING_NOTES.restartTokens + 1)] },
      {},
    ]);
    await w.write(held({ ...heard(3), notes: kept() }), "work", null);
    await w.write(held(heard(4)), "work", null);
    expect(turnsOf(fake.asked, 1)).toHaveLength(3);
    await w.write(held(heard(5)), "work", null);
    expect(turnsOf(fake.asked, 2)).toHaveLength(5);
    await w.write(held(heard(6)), "work", null);
    expect(turnsOf(fake.asked, 3)).toHaveLength(1);
    expect(turnsOf(fake.asked, 3)[0].content).toContain("[00:25] Me: Frase 5.\n\nThe notes so far, each line with its id:");
  });

  it("once the turn it would append takes it past the threshold: the request never outgrows the model's context", async () => {
    const base = 5_000;
    const { w, fake, held } = writer([{ events: [prompt(base)] }, { events: [prompt(base + RUNNING_NOTES.restartTokens - 5)] }, {}]);
    await w.write(held({ ...heard(3), notes: kept() }), "work", null);
    await w.write(held(heard(4)), "work", null);
    expect(turnsOf(fake.asked, 1)).toHaveLength(3);
    // The turn it would append, what was said since [00:20], is more than five tokens.
    await w.write(held(heard(5)), "work", null);
    expect(turnsOf(fake.asked, 2)).toHaveLength(1);
  });

  it("not at every update when its first request alone passed the threshold", async () => {
    const base = RUNNING_NOTES.restartTokens + 50_000;
    const { w, fake, held } = writer([{ events: [prompt(base)] }, { events: [prompt(base + 1_000)] }, {}]);
    await w.write(held({ ...heard(3), notes: kept() }), "work", null);
    await w.write(held(heard(4)), "work", null);
    expect(turnsOf(fake.asked, 1)).toHaveLength(3);
    await w.write(held(heard(5)), "work", null);
    expect(turnsOf(fake.asked, 2)).toHaveLength(5);
  });

  it("sooner on a model whose context is small: once it grew half of it", async () => {
    const { w, fake, held } = writer([{ events: [prompt(2_000)] }, { events: [prompt(2_000 + 16_385)] }, {}]);
    await w.write(held(heard(3)), "work", 32_768);
    await w.write(held(heard(4)), "work", 32_768);
    expect(turnsOf(fake.asked, 1)).toHaveLength(3);
    await w.write(held(heard(5)), "work", 32_768);
    expect(turnsOf(fake.asked, 2)).toHaveLength(1);
  });

  it("on another kind, in its sections: the model's lines gone, the user's under a section of the same name or a closing Other", async () => {
    const { w, fake, page, held } = writer([{}, {}]);
    await w.write(held({ ...heard(3), notes: kept() }), "work", null);
    const t = held({ ...heard(3), notes: { ...page.get(1)!, kind: "general" } });
    await w.write(t, "general", null);
    expect(fake.asked[1].request.settings.systemPrompt).toBe(`${ASSIST_SYSTEM}\n\n${runningTask("general")}`);
    expect(fake.asked[1].request.extras?.tools?.[0].function.parameters).toMatchObject({ properties: { section: { enum: ["Now", "People", "Points", "Asked of me", "Open questions"] } } });
    expect(page.get(1)?.lines).toEqual([
      { id: "N2", section: "People", text: "Bea: fornitore.", author: "user" },
      { id: "N3", section: "Other", text: "40k: il tetto.", author: "user" },
    ]);
    expect(page.get(1)).toMatchObject({ kind: "general", writtenAs: "general", nextId: 5, removed: kept().removed });
    expect(turnsOf(fake.asked, 1)).toHaveLength(1);
    expect(turnsOf(fake.asked, 1)[0].content).toContain(
      [
        "Now:\n(empty)",
        "People:\nN2 (the user's) Bea: fornitore.",
        "Points:\n(empty)",
        "Asked of me:\n(empty)",
        "Open questions:\n(empty)",
        "Other:\nN3 (the user's) 40k: il tetto.",
      ].join("\n\n"),
    );
  });

  it("on Rewrite from scratch: no line of the model's kept, and the thread new", async () => {
    const { w, fake, page, held } = writer([{ events: [call("c1", "add_line", { section: "People", text: "Ada: acquisti." })] }, {}, { events: [call("c2", "add_line", { section: "Now", text: "Le date." })] }, {}]);
    await w.write(held({ ...heard(3), notes: kept() }), "work", null);
    expect(page.get(1)?.lines.filter((l) => l.author === "model")).toHaveLength(2);
    await w.rewrite(held(heard(4)), "work", null);
    expect(turnsOf(fake.asked, 2)).toHaveLength(1);
    expect(turnsOf(fake.asked, 2)[0].content).toContain("Now:\n(empty)\n\nPeople:\nN2 (the user's) Bea: fornitore.");
    expect(page.get(1)?.lines).toEqual([
      { id: "N2", section: "People", text: "Bea: fornitore.", author: "user" },
      { id: "N3", section: "Figures & dates", text: "40k: il tetto.", author: "user" },
      { id: "N6", section: "Now", text: "Le date.", author: "model" },
    ]);
  });
});

describe("what changed in the transcript since the last update", () => {
  /** A call of two voices taking turns, Speaker 1 and Speaker 2: utterance `i + 1` is said at `5i` seconds. */
  function voices(n: number): Transcript {
    let t = { ...createTranscript(1), title: "27 Oct 15:42 · Call" };
    for (let i = 0; i < n; i++) t = addUtterance(t, { stream: "s", startMs: i * 5_000, endMs: i * 5_000 + 2_000, text: `Frase ${i}.`, voice: { kind: "speaker", index: i % 2 } });
    return t;
  }

  /** The turn the update after the first one appended: what it told the model. */
  async function laterTurnOf(first: Transcript, then: Transcript): Promise<string | undefined> {
    const { w, fake, held } = writer([{}, {}]);
    await w.write(held(first), "work", null);
    await w.write(held(then), "work", null);
    expect(turnsOf(fake.asked, 1)).toHaveLength(3);
    return turnsOf(fake.asked, 1).at(-1)?.content;
  }

  it("tells an utterance edited, with its time and its new text, after what was said since", async () => {
    const edited = editText(voices(4), 2, "Frase uno, corretta.");
    expect(await laterTurnOf(voices(3), edited)).toBe("Said since [00:15]:\n[00:15] Speaker 2: Frase 3.\n\nChanged since:\n- Corrected: [00:05] Speaker 2: Frase uno, corretta. It was: Frase 1.");
    // The running task says what the block is, in the system block every request of the thread carries.
    expect(runningTask("work")).toContain(
      "- Each user turn brings what was said since the last one and, under Changed since, what the user changed in the transcript meanwhile: an utterance corrected, moved to another speaker, split, deleted or put back, a speaker renamed or merged into another; and the notes' lines the user edited, added or removed by hand.",
    );
  });

  it("tells each utterance an accepted fix changed, and one Rewrite took, as corrected", async () => {
    const fixed = fixUtterances(voices(3), "Frase", "Fase", [1, 3]);
    expect(await laterTurnOf(voices(3), fixed)).toBe("Changed since:\n- Corrected: [00:00] Speaker 1: Fase 0. It was: Frase 0.\n- Corrected: [00:10] Speaker 1: Fase 2. It was: Frase 2.");
    const rewritten = acceptRewrite(voices(3), 3, "Frase due, riscritta.");
    expect(await laterTurnOf(voices(3), rewritten)).toBe("Changed since:\n- Corrected: [00:10] Speaker 1: Frase due, riscritta. It was: Frase 2.");
  });

  it("tells an utterance moved to another speaker, and whose it was", async () => {
    expect(await laterTurnOf(voices(3), moveUtterance(voices(3), 2, 1))).toBe("Changed since:\n- Moved: [00:05] Speaker 1: Frase 1. It was Speaker 2's.");
    // Moved and edited both: one event, with the line it was.
    const both = editText(moveUtterance(voices(3), 2, 1), 2, "Frase uno.");
    expect(await laterTurnOf(voices(3), both)).toBe("Changed since:\n- Corrected: [00:05] Speaker 1: Frase uno. It was: Speaker 2: Frase 1.");
  });

  it("tells an utterance deleted, or its text emptied, by the line it was", async () => {
    expect(await laterTurnOf(voices(3), removeUtterance(voices(3), 2))).toBe("Changed since:\n- Deleted: [00:05] Speaker 2: Frase 1.");
    expect(await laterTurnOf(voices(3), editText(voices(3), 2, " "))).toBe("Changed since:\n- Deleted: [00:05] Speaker 2: Frase 1.");
  });

  it("tells an utterance put back once its deletion was told, and nothing when the Undo came before the update", async () => {
    const { w, fake, held } = writer([{}, {}, {}]);
    const t = voices(3);
    await w.write(held(t), "work", null);
    const gone = removeUtterance(t, 2);
    await w.write(held(gone), "work", null);
    await w.write(held(restoreUtterance(gone, t.utterances[1])), "work", null);
    expect(turnsOf(fake.asked, 2).at(-1)?.content).toBe("Changed since:\n- Put back: [00:05] Speaker 2: Frase 1.");
    // Deleted and put back between two updates: nothing changed.
    await w.write(held(restoreUtterance(removeUtterance(voices(3), 3), t.utterances[2])), "work", null);
    expect(fake.asked).toHaveLength(3);
  });

  it("tells an utterance split as one change, its halves with their times; a split of one not told yet is said", async () => {
    const longer = addUtterance(voices(2), { stream: "s", startMs: 10_000, endMs: 12_000, text: "Il budget è 40k. Per me va bene.", voice: { kind: "speaker", index: 0 } });
    const split = splitUtterance(longer, 3, 17, 2);
    expect(await laterTurnOf(longer, split)).toBe("Changed since:\n- Split: [00:10] Speaker 1: Il budget è 40k. | [~00:11] Speaker 2: Per me va bene.");
    // Split again, its second half in two more: still one change.
    const thrice = splitUtterance(split, 4, 7, 1);
    expect(await laterTurnOf(longer, thrice)).toBe("Changed since:\n- Split: [00:10] Speaker 1: Il budget è 40k. | [~00:11] Speaker 2: Per me | [~00:11] Speaker 1: va bene.");
    expect(await laterTurnOf(voices(2), split)).toBe("Said since [00:10]:\n[00:10] Speaker 1: Il budget è 40k.\n[~00:11] Speaker 2: Per me va bene.\n");
  });

  it("tells a speaker renamed as one change, and leaves the notes' lines to the model", async () => {
    const notes: Notes = { ...noNotes("work"), lines: [{ id: "N1", section: "People", text: "Speaker 2: acquisti.", author: "model" }], nextId: 2, writtenAs: "work", covered: 3, writtenAt: 500 };
    const { w, fake, page, held } = writer([{}, {}]);
    await w.write(held({ ...voices(3), notes }), "work", null);
    await w.write(held(renameSpeaker(voices(4), 2, "Anna")), "work", null);
    expect(turnsOf(fake.asked, 1).at(-1)?.content).toBe("Said since [00:15]:\n[00:15] Anna: Frase 3.\n\nChanged since:\n- Renamed: Speaker 2, heard from [00:05], is Anna.");
    // The page never puts the new name in a line: the model does, through its tools.
    expect(page.get(1)?.lines).toEqual(notes.lines);
  });

  it("tells a speaker merged into another as one change, and an utterance of theirs moved elsewhere on its own", async () => {
    const t = voices(5);
    expect(await laterTurnOf(t, mergeSpeakers(t, 2, 1))).toBe("Changed since:\n- Merged: Speaker 2, heard from [00:05], is Speaker 1.");
    const elsewhere = mergeSpeakers(moveUtterance(t, 4, "new"), 2, 1);
    expect(await laterTurnOf(t, elsewhere)).toBe("Changed since:\n- Merged: Speaker 2, heard from [00:05], is Speaker 1.\n- Moved: [00:15] Speaker 3: Frase 3. It was Speaker 2's.");
    // Both renamed, then merged: each told by the name the model knows them by.
    const named = mergeSpeakers(renameSpeaker(renameSpeaker(t, 1, "Anna"), 2, "Bea"), 2, 1);
    expect(await laterTurnOf(t, named)).toBe("Changed since:\n- Renamed: Speaker 1, heard from [00:00], is Anna.\n- Merged: Speaker 2, heard from [00:05], is Anna.");
  });

  it("tells a change made while an update runs in the update after it", async () => {
    const { w, fake, held } = writer([{ hold: true }, {}]);
    const t = voices(3);
    const updating = w.write(held(t), "work", null);
    // The user corrects while the request is under way, and presses Update notes: the update goes on as it started.
    const edited = editText(t, 1, "Frase zero, corretta.");
    await w.write(held(edited), "work", null);
    expect(fake.asked).toHaveLength(1);
    fake.release();
    await updating;
    await w.write(held(edited), "work", null);
    expect(turnsOf(fake.asked, 1).at(-1)?.content).toBe("Changed since:\n- Corrected: [00:00] Speaker 1: Frase zero, corretta. It was: Frase 0.");
  });

  it("starts the thread again from the transcript as it stands: a change before a restart is in its first turn, and not told again", async () => {
    // An update whose second request's prompt grew past the threshold beyond its first's.
    const first = { events: [call("c1", "add_line", { section: "Now", text: "Il budget." }), prompt(1_000)] };
    const { w, fake, held } = writer([first, { events: [prompt(1_000 + RUNNING_NOTES.restartTokens + 1)] }, {}, {}]);
    await w.write(held(voices(3)), "work", null);
    const edited = renameSpeaker(editText(voices(3), 2, "Frase uno, corretta."), 2, "Anna");
    await w.write(held(edited), "work", null);
    expect(turnsOf(fake.asked, 2)).toHaveLength(1);
    expect(turnsOf(fake.asked, 2)[0].content).toContain("[00:05] Anna: Frase uno, corretta.\n");
    expect(turnsOf(fake.asked, 2)[0].content).not.toContain("Changed since");
    const next = addUtterance(edited, { stream: "s", startMs: 15_000, endMs: 17_000, text: "Frase 3.", voice: { kind: "speaker", index: 1 } });
    await w.write(held(next), "work", null);
    expect(turnsOf(fake.asked, 3).at(-1)?.content).toBe("Said since [00:15]:\n[00:15] Anna: Frase 3.\n");
  });

  it("tells the last utterance deleted apart from a new one heard under its id", async () => {
    const gone = removeUtterance(voices(3), 3);
    const next = addUtterance(gone, { stream: "s", startMs: 15_000, endMs: 17_000, text: "Nuova.", voice: { kind: "speaker", index: 0 } });
    expect(next.utterances.at(-1)?.id).toBe(3);
    expect(await laterTurnOf(voices(3), next)).toBe("Said since [00:15]:\n[00:15] Speaker 1: Nuova.\n\nChanged since:\n- Deleted: [00:10] Speaker 1: Frase 2.");
  });
});

describe("the notes edited by hand", () => {
  /** The results a round's request carried back: each call's one-line answer, in order. */
  const resultsOf = (asked: StreamOptions[], round: number) =>
    turnsOf(asked, round)
      .filter((turn) => turn.role === "tool")
      .map((turn) => turn.content);

  /** Running notes the model wrote: Ada and Carlo in People, the budget in Now. */
  const kept = (): Notes => ({
    ...noNotes("work"),
    lines: [
      { id: "N1", section: "People", text: "Ada: acquisti.", author: "model" },
      { id: "N2", section: "Now", text: "Il budget.", author: "model" },
      { id: "N3", section: "People", text: "Carlo: finanza.", author: "model" },
    ],
    nextId: 4,
    writtenAs: "work",
    covered: 3,
    writtenAt: 500,
  });

  it("lets an edit made while an update runs win: a call arriving on that line after it is refused, the line left the user's", async () => {
    const { w, fake, page, held } = writer([
      { events: [call("c1", "add_line", { section: "People", text: "Ada: acquisti." })], hold: true },
      { events: [call("c2", "replace_line", { id: "N1", text: "Ada: acquisti e logistica." }), call("c3", "remove_line", { id: "N1" })] },
      {},
    ]);
    const updating = w.write(held(heard(3)), "work", null);
    const seen = page.get(1)!.lines[0];
    w.editNotes(1, (n) => editLine(n, seen, "Ada: logistica."));
    fake.release();
    await updating;
    expect(resultsOf(fake.asked, 2).slice(-2)).toEqual(["Refused: N1 is the user's: leave it as it is.", "Refused: N1 is the user's: leave it as it is."]);
    expect(page.get(1)?.lines).toEqual([{ id: "N1", section: "People", text: "Ada: logistica.", author: "user" }]);
  });

  it("gives a line the user adds while an update runs an id of its own: the model's next line takes the one after, and each result names the line it made", async () => {
    const { w, fake, page, held } = writer([
      { events: [call("c1", "add_line", { section: "People", text: "Ada: acquisti." })], hold: true },
      { events: [call("c2", "add_line", { section: "People", text: "Carlo: finanza." })] },
      { events: [call("c3", "replace_line", { id: "N3", text: "Carlo: finanza e budget." }), call("c4", "replace_line", { id: "N2", text: "Bea." })] },
      {},
    ]);
    const updating = w.write(held(heard(3)), "work", null);
    w.editNotes(1, (n) => addLine(n, "People", "Bea: fornitore."));
    fake.release();
    await updating;
    expect(resultsOf(fake.asked, 3)).toEqual(["Added N1 to People.", "Added N3 to People.", "N3 updated.", "Refused: N2 is the user's: leave it as it is."]);
    expect(page.get(1)?.lines).toEqual([
      { id: "N1", section: "People", text: "Ada: acquisti.", author: "model" },
      { id: "N2", section: "People", text: "Bea: fornitore.", author: "user" },
      { id: "N3", section: "People", text: "Carlo: finanza e budget.", author: "model" },
    ]);
    expect(page.get(1)?.nextId).toBe(4);
  });

  it("tells the user's edits, additions and removals in the next turn, a line once as it stands, and nothing for a line added and removed meanwhile", async () => {
    const { w, fake, page, held } = writer([{}, { events: [call("c1", "add_line", { section: "Now", text: "Il budget." })] }, {}]);
    await w.write(held({ ...heard(3), notes: kept() }), "work", null);
    const [ada, budget, carlo] = kept().lines;
    w.editNotes(1, (n) => editLine(n, ada, "Ada: logistica."));
    w.editNotes(1, (n) => editLine(n, page.get(1)!.lines[0], "Ada: responsabile logistica."));
    w.editNotes(1, (n) => addLine(n, "Figures & dates", "40k: il tetto."));
    w.editNotes(1, (n) => removeLine(n, budget));
    w.editNotes(1, (n) => addLine(n, "People", "Dario."));
    w.editNotes(1, (n) => removeLine(n, page.get(1)!.lines.at(-1)!));
    await w.write(held(heard(4)), "work", null);
    expect(turnsOf(fake.asked, 1).at(-1)?.content).toBe(
      [
        "Said since [00:15]:\n[00:15] Me: Frase 3.\n",
        "Changed since:",
        "- Edited by the user in People: N1 (the user's) Ada: responsabile logistica. It was: Ada: acquisti.",
        "- Added by the user to Figures & dates: N4 (the user's) 40k: il tetto.",
        "- Removed by the user from Now, never to be written again: N2 Il budget.",
      ].join("\n"),
    );
    // The line they removed is not written again.
    expect(resultsOf(fake.asked, 2)).toEqual(["Refused: the user removed that line from Now; never write it again."]);
    expect(page.get(1)?.lines).toEqual([{ ...ada, text: "Ada: responsabile logistica.", author: "user" }, carlo, { id: "N4", section: "Figures & dates", text: "40k: il tetto.", author: "user" }]);
    // Told once: an update with nothing new asks nothing.
    await w.write(held(heard(4)), "work", null);
    expect(fake.asked).toHaveLength(3);
  });

  it("tells an edit made while an update runs in the update after it", async () => {
    const { w, fake, page, held } = writer([{ events: [call("c1", "add_line", { section: "People", text: "Ada: acquisti." })], hold: true }, {}, {}]);
    const updating = w.write(held(heard(3)), "work", null);
    w.editNotes(1, (n) => editLine(n, page.get(1)!.lines[0], "Ada: logistica."));
    fake.release();
    await updating;
    await w.write(held(heard(3)), "work", null);
    expect(turnsOf(fake.asked, 2).at(-1)?.content).toBe("Changed since:\n- Edited by the user in People: N1 (the user's) Ada: logistica. It was: Ada: acquisti.");
  });

  it("lists the user's lines and the ones they removed in a thread started again, and tells them nothing more", async () => {
    const { w, fake, page, held } = writer([{}, {}, {}]);
    await w.write(held({ ...heard(3), notes: kept() }), "work", null);
    const [ada, budget] = kept().lines;
    w.editNotes(1, (n) => removeLine(editLine(n, ada, "Ada: logistica."), budget));
    await w.rewrite(held(heard(3)), "work", null);
    const first = turnsOf(fake.asked, 1);
    expect(first).toHaveLength(1);
    expect(first[0].content).toContain("People:\nN1 (the user's) Ada: logistica.\n\nFigures & dates:");
    expect(first[0].content).toContain("Lines the user removed, never to be written again:\n- Now: Il budget.");
    expect(page.get(1)?.lines).toEqual([{ ...ada, text: "Ada: logistica.", author: "user" }]);
    await w.write(held(heard(4)), "work", null);
    expect(turnsOf(fake.asked, 2).at(-1)?.content).toBe("Said since [00:15]:\n[00:15] Me: Frase 3.\n");
  });

  it("changes the page's notes alone while no update runs", () => {
    const { w, page, held } = writer([]);
    held({ ...heard(3), notes: kept() });
    w.editNotes(1, (n) => addLine(n, "Now", "Le date."));
    expect(page.get(1)?.lines.at(-1)).toEqual({ id: "N4", section: "Now", text: "Le date.", author: "user" });
  });
});

describe("an update as the user sees it", () => {
  it("lands each line as its call comes, while the answer is still on its way", async () => {
    const { w, fake, page, held } = writer([{ events: [call("c1", "add_line", { section: "Now", text: "Il budget." }), call("c2", "add_line", { section: "People", text: "Ada: acquisti." })], hold: true }, {}]);
    const updating = w.write(held(heard(3)), "work", null);
    expect(w.getState()).toMatchObject({ writing: 1, form: "running" });
    expect(page.get(1)?.lines.map((l) => l.text)).toEqual(["Il budget.", "Ada: acquisti."]);
    fake.release();
    await updating;
    expect(w.getState().writing).toBeNull();
  });

  it("keeps what was applied on a Stop, and starts the thread again from the notes next time", async () => {
    const { w, fake, page, held } = writer([{ events: [call("c1", "add_line", { section: "Now", text: "Il budget." })], hold: true }, {}]);
    const updating = w.write(held(heard(3)), "work", null);
    w.stop();
    await updating;
    expect(fake.asked[0].signal?.aborted).toBe(true);
    expect(page.get(1)?.lines).toEqual([{ id: "N1", section: "Now", text: "Il budget.", author: "model" }]);
    expect(w.getState()).toEqual({ writing: null, form: "running", text: "", failed: null });
    await w.write(held(heard(4)), "work", null);
    expect(turnsOf(fake.asked, 1)).toHaveLength(1);
    expect(turnsOf(fake.asked, 1)[0].content).toContain("[00:15] Me: Frase 3.\n\nThe notes so far, each line with its id:\n\nNow:\nN1 Il budget.");
  });

  it("keeps what was applied when a request fails, and says why", async () => {
    const { w, fake, page, held } = writer([{ events: [call("c1", "add_line", { section: "Now", text: "Il budget." })] }, { result: { ok: false, message: "ignis answered 500" } }, {}]);
    await w.write(held(heard(3, 4)), "work", null);
    expect(w.getState().failed).toEqual({ transcript: 4, form: "running", message: "ignis answered 500" });
    expect(page.get(4)?.lines.map((l) => l.text)).toEqual(["Il budget."]);
    await w.write(held(heard(3, 4)), "work", null);
    expect(turnsOf(fake.asked, 2)).toHaveLength(1);
    expect(w.getState().failed).toBeNull();
  });

  it("asks nothing when nothing was said since, and says the notes are current", async () => {
    let clock = 1_000;
    const { w, fake, page, held } = writer([{}], () => clock);
    await w.write(held(heard(3)), "work", null);
    clock = 90_000;
    await w.write(held(heard(3)), "work", null);
    expect(fake.asked).toHaveLength(1);
    expect(page.get(1)).toMatchObject({ writtenAs: "work", writtenAt: 90_000, covered: 3 });
  });

  it("writes one at a time, and waits out a full engine, a second at a time", async () => {
    const { w, fake, slept, held } = writer([{ hold: true }, { result: { ok: false, message: "engine_full: all lanes in use" } }, {}]);
    const first = w.write(held(heard(3, 1)), "work", null);
    await w.write(held(heard(3, 2)), "work", null);
    expect(fake.asked).toHaveLength(1);
    fake.release();
    await first;
    await w.write(held(heard(3, 2)), "work", null);
    expect(slept).toEqual([1_000]);
    expect(fake.asked).toHaveLength(3);
  });
});

describe("keeping the notes up", () => {
  const notes = (covered: number): Notes => ({ ...noNotes("general"), lines: [{ id: "N1", section: "Now", text: "A", author: "model" }], nextId: 2, writtenAs: "general", covered, writtenAt: 0 });

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
    const { w, fake, recaps } = writer([{ chunks: ["## Summary\n- Fine."] }]);
    const transcripts = new Map([[1, { ...heard(4, 1), notes: notes(4) }]]);
    const look = (on: number | null) => w.look(on, (id) => transcripts.get(id), () => "general", null);
    await look(1);
    expect(fake.asked).toHaveLength(0);
    await look(null);
    expect(fake.asked).toHaveLength(1);
    expect(recaps).toEqual([[1, { lines: [{ section: "Summary", text: "Fine." }], writtenAs: "general", covered: 4, writtenAt: 1_000, fromMs: null }]]);
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
    const { w, fake, recaps } = writer([{ chunks: ["## Summary\n- Il budget è di 40k [00:00].\n\n", "## Figures & dates\n- 40k: il budget [00:00]."] }]);
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
    expect(recaps).toEqual([
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
      ],
    ]);
    expect(w.getState()).toEqual({ writing: null, form: "running", text: "", failed: null });
  });

  it("sends the running lines the user edited or added by hand as notes they took, and none they removed", async () => {
    const { w, fake, held } = writer([{ chunks: ["## Summary\n- Ok."] }]);
    const t = held(call());
    const [budget] = t.notes!.lines;
    w.editNotes(5, (n) => addLine(editLine(n, budget, "The budget is 40k."), "Figures & dates", "40k: the cap."));
    w.editNotes(5, (n) => removeLine(n, n.lines[1]));
    await w.writeRecap(held(t), "work", null);
    expect(fake.asked[0].request.turns[0].content.endsWith("to be taken as true:\n- Summary: The budget is 40k.\n- Figures & dates: 40k: the cap.")).toBe(true);
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
