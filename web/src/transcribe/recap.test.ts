import { describe, expect, it } from "vitest";
import { noNotes, type Notes } from "./notes.ts";
import { readRecap, recapDue, recapMarkdown, recapRequest, recapSections, recapTask, withRecap } from "./recap.ts";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

function heard(texts: string[]): Transcript {
  let t = { ...createTranscript(1), title: "27 Oct 15:42 · Call" };
  texts.forEach((text, i) => {
    t = addUtterance(t, { stream: "s", startMs: i * 10_000, endMs: i * 10_000 + 4_000, text, voice: { kind: "me" } });
  });
  return t;
}

const recap = (patch: Partial<Notes> = {}): Notes => ({
  ...noNotes("work"),
  lines: [{ id: "N1", section: "Summary", text: "The budget.", author: "model" }],
  nextId: 2,
  writtenAs: "work",
  covered: 2,
  writtenAt: 1_000,
  ...patch,
});

// The recap (spec #428, "The recap"): one writing over the whole transcript,
// in the kind's sections with a few additions, its rules asking for the main
// threads by weight, and the user's own running lines taken as true.

describe("the recap's sections", () => {
  it("are the kind's, with Figures & dates for a Work call, and Terms and Homework & dates for a Lesson", () => {
    const names = (kind: Parameters<typeof recapSections>[0]) => recapSections(kind).map((s) => s.name);
    expect(names("work")).toEqual(["Summary", "Topics", "People", "Figures & dates", "Decisions", "Action items", "Open questions"]);
    expect(names("lesson")).toEqual(["Topics", "Key concepts", "Terms", "Questions", "To review", "Homework & dates"]);
    expect(names("general")).toEqual(["Summary", "Key points", "Next steps"]);
    expect(names("game")).toEqual(["Games", "Players", "Results", "Highlights"]);
  });

  it("are asked for as the probe measured them: the recap's rules first, then the notes' own", () => {
    expect(recapTask("work")).toBe(
      [
        "Write the recap of a work call or meeting: these sections, in this order, each under a level-2 Markdown heading (`## `):",
        "## Summary\nthe purpose of the call and where it got to, in two to four sentences.",
        "## Topics\neach subject discussed, with what was said of it in a few words.",
        "## People\neach person who spoke or was named, with their role or stance where the conversation shows it.",
        "## Figures & dates\nevery figure said (an amount, a percentage, a quantity, a date, a deadline), one a line, with what it refers to and the time it was said.",
        "## Decisions\nwhat was agreed, with the time it was said.",
        "## Action items\nwho does what, and by when if it was said, with the time it was said, as a task list (`- [ ] `).",
        "## Open questions\nwhat was raised and left unresolved.",
        [
          "Rules:",
          "- The Summary and the Topics carry the main threads by the weight the conversation gave them, not every aside.",
          "- Decisions are what was agreed; Action items what someone committed to do; People everyone who spoke or was named.",
          "- Notes the user took during the conversation are true: use them, and let them correct what the transcript seems to say.",
          "- Keep each heading exactly as given. Write the text under them in the language the context says to write in.",
          "- Short bullet points. No preamble, no closing remark, nothing outside the sections.",
          "- A section with nothing in it yet is `—` alone.",
          "- Point at a moment by its time, as [mm:ss].",
          "- Write the notes as of the transcript's last line.",
        ].join("\n"),
      ].join("\n\n"),
    );
    const lesson = recapTask("lesson");
    expect(lesson).toContain("## Key concepts\neach idea explained, with its explanation in a sentence.\n\n## Terms\nthe terms introduced, each with what it means.\n\n## Questions");
    expect(lesson).toContain("## To review\nwhat to study or look up after.\n\n## Homework & dates\nhomework, readings, exams and their dates.\n\nRules:");
  });
});

describe("the recap's request", () => {
  it("gives what the context images showed after the transcript and before the notes the user took", () => {
    const t = heard(["Here is the slide.", "The cap is forty."]);
    const { request } = recapRequest(t, "work", null, ["[00:00] (image) A slide: Q3 budget, cap 40k."]);
    expect(request.turns[0].content.endsWith(
      "[00:10] Me: The cap is forty.\n\nPictures shown during the conversation, as read at their moments:\n[00:00] (image) A slide: Q3 budget, cap 40k.\n\nNotes the user took during the conversation: none.",
    )).toBe(true);
  });

  it("sends the transcript from where its tail fits the model's context, the notes the user took kept whole", () => {
    const long = heard(Array.from({ length: 400 }, (_, i) => `Sentence number ${i} of the long call.`));
    const theirs = { ...noNotes("work"), lines: [{ id: "N1", section: "People", text: "Ada leads procurement.", author: "user" as const }], nextId: 2 };
    const { request, fromMs } = recapRequest({ ...long, notes: theirs }, "work", 8_000);
    expect(fromMs).not.toBeNull();
    expect(request.turns[0].content).not.toContain("Sentence number 0 ");
    expect(request.turns[0].content).toContain("Sentence number 399 ");
    expect(request.turns[0].content.endsWith("to be taken as true:\n- People: Ada leads procurement.")).toBe(true);
  });
});

describe("the end of a listening", () => {
  it("writes the recap once something was heard, unless a recap of the kind picked covers every utterance already", () => {
    expect(recapDue(heard([]), "work")).toBe(false);
    expect(recapDue(heard([" "]), "work")).toBe(false);
    expect(recapDue(heard(["One.", "Two."]), "work")).toBe(true);
    expect(recapDue({ ...heard(["One.", "Two."]), recap: recap({ covered: 2 }) }, "work")).toBe(false);
    expect(recapDue({ ...heard(["One.", "Two.", "Three."]), recap: recap({ covered: 2 }) }, "work")).toBe(true);
    expect(recapDue({ ...heard(["One.", "Two."]), recap: recap({ covered: 2 }) }, "lesson")).toBe(true);
  });
});

describe("the recap as kept", () => {
  it("takes a rewrite's lines in place of the model's, keeping the user's lines, as the kind it was written as", () => {
    const before = recap({
      lines: [
        { id: "N1", section: "Summary", text: "The budget.", author: "model" },
        { id: "N2", section: "Decisions", text: "Cap at 40k [02:10]", author: "user" },
      ],
      nextId: 3,
    });
    expect(withRecap(before, { lines: [{ section: "Topics", text: "Concepts [00:10]" }], writtenAs: "lesson", covered: 5, writtenAt: 2_000, fromMs: null })).toEqual({
      ...before,
      kind: "lesson",
      lines: [
        { id: "N3", section: "Topics", text: "Concepts [00:10]", author: "model" },
        { id: "N2", section: "Decisions", text: "Cap at 40k [02:10]", author: "user" },
      ],
      nextId: 4,
      writtenAs: "lesson",
      covered: 5,
      writtenAt: 2_000,
    });
    expect(withRecap(undefined, { lines: [{ section: "Games", text: "Chess." }], writtenAs: "game", covered: 3, writtenAt: 1_000, fromMs: null })).toEqual({
      ...noNotes("game"),
      lines: [{ id: "N1", section: "Games", text: "Chess.", author: "model" }],
      nextId: 2,
      writtenAs: "game",
      covered: 3,
      writtenAt: 1_000,
    });
  });

  it("gives back Markdown in the recap's sections, in order, `—` for an empty one", () => {
    const kept = recap({
      lines: [
        { id: "N2", section: "Figures & dates", text: "40k: the cap [02:10]", author: "model" },
        { id: "N1", section: "Summary", text: "The budget.", author: "user" },
      ],
      nextId: 3,
    });
    expect(recapMarkdown(kept)).toBe(
      [
        "## Summary\n- The budget.",
        "## Topics\n—",
        "## People\n—",
        "## Figures & dates\n- 40k: the cap [02:10]",
        "## Decisions\n—",
        "## Action items\n—",
        "## Open questions\n—",
      ].join("\n\n"),
    );
    expect(recapMarkdown(undefined)).toBe("");
  });

  it("reads back from the store as written, one never written or malformed as none", () => {
    expect(readRecap(recap())).toEqual(recap());
    expect(readRecap(noNotes("work"))).toBeUndefined();
    expect(readRecap({ ...recap(), lines: [{ id: "N1", section: "Summary", text: "", author: "model" }] })).toBeUndefined();
  });
});
