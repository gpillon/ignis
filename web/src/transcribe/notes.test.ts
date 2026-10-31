import { afterEach, describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import {
  chooseNoteKind,
  dueForUpdate,
  FIRST_LIVE_UTTERANCES,
  kindHint,
  kindOf,
  LIVE_GAP_MS,
  NOTE_KIND_KEY,
  NOTE_KINDS,
  noNotes,
  NOTES_SETTINGS,
  notesMarkdown,
  notesStatus,
  parseNotes,
  readNotes,
  RUNNING_NOTES,
  textModelBlocked,
  reloadNoteKind,
  tidyNotes,
  withWriting,
  type Notes,
} from "./notes.ts";
import { runningTask } from "./running.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// Notes on a Transcript (spec speech/07, Q124-Q127): the text model reads the
// transcript and writes the sections a kind of conversation wants, on a press
// or, kept up while listening, about once a minute.

function call(lines: [string, string][]): Transcript {
  let t = { ...createTranscript(1, { source: "call", at: new Date(2026, 9, 27, 15, 42) }), title: "27 Oct 15:42 · Call" };
  lines.forEach(([who, text], i) => {
    t = addUtterance(t, { stream: "s", startMs: i * 10_000, endMs: i * 10_000 + 4_000, text, voice: who === "me" ? { kind: "me" } : { kind: "others" } });
  });
  return t;
}

afterEach(() => {
  Object.defineProperty(globalThis, "localStorage", { value: undefined, configurable: true, writable: true });
  reloadNoteKind();
});

const written = (patch: Partial<Notes> = {}): Notes => ({
  kind: "work",
  lines: [{ id: "N1", section: "Summary", text: "ok", author: "model" }],
  removed: [],
  nextId: 2,
  writtenAs: "work",
  covered: 4,
  writtenAt: new Date(2026, 9, 27, 16, 5).getTime(),
  fromMs: null,
  ...patch,
});

describe("the kinds of conversation", () => {
  it("show their running notes' sections in short on their tiles, without the one line of what is on now", () => {
    expect(NOTE_KINDS.map(kindHint)).toEqual([
      "People, points, asked of me, open questions",
      "People, figures & dates, asked of me, commitments heard",
      "Players & sides, score, moments, rules in question",
      "Concepts, terms, questions, homework & dates",
    ]);
  });

  it("are General, Work call, Game session and Lesson, each with its recap's sections and its running notes' live ones, a cap each", () => {
    expect(NOTE_KINDS.map((k) => k.label)).toEqual(["General", "Work call", "Game session", "Lesson"]);
    const sections = (kind: string) => NOTE_KINDS.find((k) => k.kind === kind)?.sections.map((s) => s.name);
    expect(sections("work")).toEqual(["Summary", "Topics", "People", "Decisions", "Action items", "Open questions"]);
    expect(sections("game")).toEqual(["Games", "Players", "Results", "Highlights"]);
    const running = (kind: string) => NOTE_KINDS.find((k) => k.kind === kind)?.running.map((s) => `${s.name} ${s.cap}`);
    expect(running("work")).toEqual(["Now 1", "People 20", "Figures & dates 12", "Asked of me 6", "Commitments heard 10"]);
    expect(running("general")).toEqual(["Now 1", "People 20", "Points 10", "Asked of me 6", "Open questions 8"]);
    expect(running("game")).toEqual(["Now playing 1", "Players & sides 12", "Score 8", "Moments 10", "Rules in question 6"]);
    expect(running("lesson")).toEqual(["Now 1", "Concepts 12", "Terms 20", "Questions 8", "Homework & dates 8"]);
  });

  it("start at the one this browser picked last, General at first; a Transcript keeps its own", () => {
    const storage = new Map<string, string>();
    const local = { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => void storage.set(key, value) };
    Object.defineProperty(globalThis, "localStorage", { value: local, configurable: true, writable: true });
    reloadNoteKind();
    const t = call([]);
    expect(kindOf(t)).toBe("general");
    chooseNoteKind("game");
    expect(storage.get(NOTE_KIND_KEY)).toBe("game");
    expect(kindOf(t)).toBe("game");
    expect(kindOf({ ...t, notes: noNotes("lesson") })).toBe("lesson");
    storage.set(NOTE_KIND_KEY, "party");
    reloadNoteKind();
    expect(kindOf(t)).toBe("general");
  });

  it("ask the model to keep their live sections, each with its cap, through the tools, in the language the context names", () => {
    const prompt = runningTask("work");
    expect(prompt.startsWith("Keep the running notes of a work call or meeting: notes that help the user follow it and take part while it goes on.")).toBe(true);
    expect(prompt).toContain(
      "## Now (1 line at most)\nwhat is being discussed right now, in one line.\n\n## People (20 lines at most)\none line a person who spoke or was named:",
    );
    expect(prompt).toContain("## Commitments heard (10 lines at most)\nwho said they would do what, and by when if it was said.");
    const at = (heading: string) => prompt.indexOf(`## ${heading} (`);
    expect(at("Figures & dates")).toBeLessThan(at("Asked of me"));
    expect(prompt).toContain("- Write the lines in the language the context says to write in.");
    expect(prompt).toContain("- A short phrase repeated word for word, again and again, is speech-recognition noise: never note it.");
    expect(prompt).toContain("- A line marked (the user's) is one the user wrote: never change or remove it");
    expect(prompt).not.toContain("## Summary");
    expect(runningTask("game")).toContain("## Now playing (1 line at most)\nthe game being played and where it stands, in one line.");
  });

  it("send no thinking, on the agent lane, and never at greedy with a top_p ignis would refuse", () => {
    expect(NOTES_SETTINGS).toMatchObject({ reasoningEffort: "none", laneTag: "agent", thinkingBudget: null });
    expect(NOTES_SETTINGS.temperature === 0 && NOTES_SETTINGS.topP !== 1).toBe(false);
  });

  it("keep their limits in one place: the threshold a thread starts again past, the rounds of an update, a line's length", () => {
    expect(RUNNING_NOTES).toEqual({ restartTokens: 64_000, maxRounds: 6, idleRounds: 2, lineChars: 240 });
  });
});

describe("keeping the notes up while listening", () => {
  it("writes the first once a few utterances are in, then about once a minute while something new was said", () => {
    expect(FIRST_LIVE_UTTERANCES).toBe(3);
    expect(dueForUpdate({ notes: undefined, utterances: 2, sinceLastMs: null })).toBe(false);
    expect(dueForUpdate({ notes: undefined, utterances: 3, sinceLastMs: null })).toBe(true);
    expect(dueForUpdate({ notes: written({ covered: 4 }), utterances: 4, sinceLastMs: LIVE_GAP_MS * 2 })).toBe(false);
    expect(dueForUpdate({ notes: written({ covered: 4 }), utterances: 5, sinceLastMs: LIVE_GAP_MS - 1 })).toBe(false);
    expect(dueForUpdate({ notes: written({ covered: 4 }), utterances: 5, sinceLastMs: LIVE_GAP_MS })).toBe(true);
  });

  it("writes them again as soon as another kind is picked", () => {
    expect(dueForUpdate({ notes: written({ covered: 4, kind: "game" }), utterances: 4, sinceLastMs: 1_000 })).toBe(true);
  });
});

describe("what the panel says", () => {
  it("is when the notes were updated, and how many utterances came since while they are not kept up", () => {
    expect(notesStatus(written(), 4, false)).toBe("Updated at 16:05.");
    expect(notesStatus(written(), 7, false)).toBe("Updated at 16:05. 3 more since.");
    expect(notesStatus(written(), 7, true)).toBe("Updated at 16:05.");
    // Utterances deleted since leave fewer than the notes covered: nothing came since, never a negative count.
    expect(notesStatus(written({ covered: 4 }), 2, false)).toBe("Updated at 16:05.");
    expect(notesStatus(written({ fromMs: 125_000 }), 4, false)).toBe("Updated at 16:05.");
  });

  it("says which kind they were written as, when another is picked", () => {
    expect(notesStatus(written({ kind: "lesson" }), 4, false)).toBe("Written as Work call notes at 16:05. Write notes to redo them as Lesson notes.");
    expect(notesStatus(undefined, 4, false)).toBeNull();
  });

  it("says why the notes and the suggestions cannot ask the text model", () => {
    const ready: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 262_144, kind: "text" };
    expect(textModelBlocked(ready)).toBeNull();
    expect(textModelBlocked({ state: "unloaded", id: "qwen3.8-27b", contextLimit: null })).toBeNull();
    expect(textModelBlocked({ ...ready, kind: "image" })).toBe("The image model is loaded: the notes and the suggestions need a text model. Switch in the Server view.");
    expect(textModelBlocked({ state: "loading", to: "qwen3.8-27b", since: 0 })).toBe("A model is loading: the notes and the suggestions wait until it serves.");
    expect(textModelBlocked({ state: "error", message: "down" })).toBe("No text model serves: the notes and the suggestions need one. Load it in the Server view.");
    expect(textModelBlocked({ state: "reading" })).toBe("Reading which model is loaded.");
  });
});

describe("the notes as kept", () => {
  it("are what the model wrote, without a fence around it", () => {
    expect(tidyNotes("```markdown\n## Summary\n- ok\n```\n")).toBe("## Summary\n- ok");
    expect(tidyNotes("  ## Summary\n- `x`\n")).toBe("## Summary\n- `x`");
  });

  it("are lines: a bullet or a paragraph each, under the level-2 heading it sits in, the kind's headings as the kind spells them", () => {
    const answer = [
      "## Summary",
      "The team sets the budget for the spring launch.",
      "It closes at noon.",
      "",
      "## topics",
      "- Budget: [01:10] the cap is 40k",
      "  and travel is left out",
      "* Hiring",
      "  - a nested point",
      "1. Numbered",
      "",
      "## Action items",
      "- [ ] Ada sends the slides [03:05]",
      "",
      "## Decisions",
      "—",
      "",
      "## Open questions",
      "- —",
      "---",
      "## Rischi",
      "- Il fornitore tarda",
    ].join("\n");
    expect(parseNotes(answer, "work")).toEqual([
      { section: "Summary", text: "The team sets the budget for the spring launch. It closes at noon." },
      { section: "Topics", text: "Budget: [01:10] the cap is 40k and travel is left out" },
      { section: "Topics", text: "Hiring" },
      { section: "Topics", text: "a nested point" },
      { section: "Topics", text: "Numbered" },
      { section: "Action items", text: "[ ] Ada sends the slides [03:05]" },
      { section: "Rischi", text: "Il fornitore tarda" },
    ]);
  });

  it("keep what came before any heading, under a closing Other, and read a fenced answer and CRLF line ends", () => {
    expect(parseNotes("```markdown\nNotes of the call:\r\n\r\n## Summary\r\n### Launch\r\nThe cap is set.\r\n- ok\n```", "general")).toEqual([
      { section: "Other", text: "Notes of the call:" },
      { section: "Summary", text: "Launch" },
      { section: "Summary", text: "The cap is set." },
      { section: "Summary", text: "ok" },
    ]);
    expect(parseNotes("## Summary\n—", "general")).toEqual([]);
  });

  it("take a writing's lines in place of the model's, a line written again word for word keeping its id, a new one the next id; the user's lines and removals stay", () => {
    const before = written({
      lines: [
        { id: "N1", section: "Summary", text: "The budget.", author: "model" },
        { id: "N2", section: "Topics", text: "Budget: the cap is 40k", author: "model" },
        { id: "N3", section: "Topics", text: "Hiring waits", author: "user" },
        { id: "N5", section: "Decisions", text: "No travel", author: "model" },
      ],
      removed: [{ id: "N4", section: "Topics", text: "Lunch", author: "model" }],
      nextId: 6,
    });
    const after = withWriting(before, {
      lines: [
        { section: "Summary", text: "The budget and the hiring." },
        { section: "Topics", text: "Budget: the cap is 40k" },
        { section: "Decisions", text: "Cap at 40k [02:10]" },
      ],
      writtenAs: "work",
      covered: 9,
      writtenAt: 2_000,
      fromMs: null,
    });
    expect(after).toEqual({
      ...before,
      lines: [
        { id: "N6", section: "Summary", text: "The budget and the hiring.", author: "model" },
        { id: "N2", section: "Topics", text: "Budget: the cap is 40k", author: "model" },
        { id: "N7", section: "Decisions", text: "Cap at 40k [02:10]", author: "model" },
        { id: "N3", section: "Topics", text: "Hiring waits", author: "user" },
      ],
      nextId: 8,
      covered: 9,
      writtenAt: 2_000,
    });
    // A Transcript's first notes: the kind they were written as, from N1.
    expect(withWriting(undefined, { lines: [{ section: "Games", text: "Chess, from [00:10]." }], writtenAs: "game", covered: 3, writtenAt: 1_000, fromMs: 5_000 })).toEqual({
      kind: "game",
      lines: [{ id: "N1", section: "Games", text: "Chess, from [00:10].", author: "model" }],
      removed: [],
      nextId: 2,
      writtenAs: "game",
      covered: 3,
      writtenAt: 1_000,
      fromMs: 5_000,
    });
  });

  it("give back Markdown, what the tab shows and Copy notes copies: the kind's live sections in order, `—` for an empty one, others after and Other last, no mark of who wrote a line", () => {
    const notes = written({
      lines: [
        { id: "N5", section: "Other", text: "Portare il listino.", author: "user" },
        { id: "N3", section: "Rischi", text: "Il fornitore tarda", author: "model" },
        { id: "N1", section: "Figures & dates", text: "40k: the cap [01:10]", author: "model" },
        { id: "N2", section: "Now", text: "The team sets the budget.", author: "model" },
        { id: "N4", section: "Figures & dates", text: "15 Nov: the deadline [64:09]", author: "user" },
      ],
      nextId: 6,
    });
    expect(notesMarkdown(notes)).toBe(
      [
        "## Now\n- The team sets the budget.",
        "## People\n—",
        // A time past the hour, as the model writes it in total minutes, reads h:mm:ss.
        "## Figures & dates\n- 40k: the cap [01:10]\n- 15 Nov: the deadline [1:04:09]",
        "## Asked of me\n—",
        "## Commitments heard\n—",
        "## Rischi\n- Il fornitore tarda",
        "## Other\n- Portare il listino.",
      ].join("\n\n"),
    );
    expect(notesMarkdown({ ...noNotes("game"), writtenAs: "game" })).toBe("## Now playing\n—\n\n## Players & sides\n—\n\n## Score\n—\n\n## Moments\n—\n\n## Rules in question\n—");
    expect(notesMarkdown(noNotes("work"))).toBe("");
    expect(notesMarkdown(undefined)).toBe("");
  });

  it("read back from the store as written, and a malformed one as none", () => {
    const line = written().lines[0];
    expect(readNotes(written())).toEqual(written());
    expect(readNotes(written({ lines: [{ ...line, author: "user" }], removed: [{ id: "N2", section: "Topics", text: "Hiring", author: "model" }], nextId: 3 }))).toEqual(
      written({ lines: [{ ...line, author: "user" }], removed: [{ id: "N2", section: "Topics", text: "Hiring", author: "model" }], nextId: 3 }),
    );
    expect(readNotes({ ...written(), kind: "party" })).toBeUndefined();
    expect(readNotes({ ...written(), lines: [{ ...line, text: 3 }] })).toBeUndefined();
    expect(readNotes({ ...written(), lines: [{ ...line, text: " " }] })).toBeUndefined();
    expect(readNotes({ ...written(), lines: [{ ...line, id: "7" }] })).toBeUndefined();
    expect(readNotes({ ...written(), lines: [{ ...line, author: "someone" }] })).toBeUndefined();
    expect(readNotes({ ...written(), removed: "N1" })).toBeUndefined();
    // An id given twice, or one the next line would take again.
    expect(readNotes({ ...written(), lines: [line, { ...line, text: "again" }], nextId: 2 })).toBeUndefined();
    expect(readNotes({ ...written(), removed: [line] })).toBeUndefined();
    expect(readNotes({ ...written(), nextId: 1 })).toBeUndefined();
    expect(readNotes(null)).toBeUndefined();
  });

  it("written by an older page as Markdown, become lines the first time they are read, all of them the model's", () => {
    const at = new Date(2026, 9, 27, 16, 5).getTime();
    const old = { kind: "work", text: "## Summary\nThe budget for the launch.\n\n## Decisions\n- Cap at 40k [02:10]\n- No travel\n\n## Riassunto\n- ok", writtenAs: "work", covered: 4, writtenAt: at, fromMs: 125_000 };
    expect(readNotes(old)).toEqual({
      kind: "work",
      lines: [
        { id: "N1", section: "Summary", text: "The budget for the launch.", author: "model" },
        { id: "N2", section: "Decisions", text: "Cap at 40k [02:10]", author: "model" },
        { id: "N3", section: "Decisions", text: "No travel", author: "model" },
        { id: "N4", section: "Riassunto", text: "ok", author: "model" },
      ],
      removed: [],
      nextId: 5,
      writtenAs: "work",
      covered: 4,
      writtenAt: at,
      fromMs: 125_000,
    });
    // A kind picked before any notes were written.
    expect(readNotes({ kind: "lesson", text: "", writtenAs: null, covered: 0, writtenAt: null, fromMs: null })).toEqual(noNotes("lesson"));
    expect(readNotes({ ...old, text: 3 })).toBeUndefined();
    expect(readNotes({ ...old, writtenAs: "party" })).toBeUndefined();
  });
});
