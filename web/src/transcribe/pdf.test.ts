import { afterEach, describe, expect, it } from "vitest";
import { EMPTY_CONTEXT } from "./context.ts";
import { addContextImage } from "./contextImages.ts";
import { noNotes, type Notes } from "./notes.ts";
import { choosePdfOptions, lastPdfOptions, PDF_OPTIONS_KEY, type PdfOptions, pageRule, pdfHints, printDocument, reloadPdfOptions } from "./pdf.ts";
import { addUtterance, createTranscript, editText, renameSpeaker, splitUtterance, type Transcript } from "./transcript.ts";

// The PDF of a transcript (spec #428, "Exports"): what it prints is a document
// computed from the transcript and the dialog's three options; the print view
// only draws it.

/** A call: Me, then a voice of the call renamed Ada, then Me again, split by hand. */
function call(): Transcript {
  let t = createTranscript(1, { source: "call", at: new Date(2026, 9, 25, 15, 42) });
  t = addUtterance(t, { stream: "mic-1", startMs: 1_200, endMs: 3_900, text: "Mi sentite?", voice: { kind: "me" } });
  t = addUtterance(t, { stream: "call-1", startMs: 4_480, endMs: 9_020, text: "Sì, forte e chiaro.", voice: { kind: "speaker", index: 0 } });
  t = addUtterance(t, { stream: "mic-1", startMs: 61_000, endMs: 65_000, text: "Allora partiamo. Va bene.", voice: { kind: "me" } });
  t = renameSpeaker(t, t.speakers[1].id, "Ada");
  return splitUtterance(t, t.utterances[2].id, 16, t.speakers[1].id);
}

const written = (kind: Notes["kind"], lines: [string, string][]): Notes => ({
  ...noNotes(kind),
  lines: lines.map(([section, text], i) => ({ id: `N${i + 1}`, section, text, author: "model" as const })),
  nextId: lines.length + 1,
  writtenAs: kind,
  covered: 3,
  writtenAt: 1_000,
});

const answer = (id: number, question: string, text: string, untilMs?: number) => ({ id, question, answer: text, at: id, untilMs });

const ALL: PdfOptions = { recap: true, context: true, answers: true };
const NONE: PdfOptions = { recap: false, context: false, answers: false };

describe("the PDF's document, always", () => {
  it("opens on a header with the title, the start, the duration and the speakers who spoke", () => {
    const [header] = printDocument(call(), NONE);
    expect(header).toEqual({
      kind: "header",
      title: "25 Oct 15:42 · Call",
      facts: [
        { label: "Started", value: "25 Oct 2026, 15:42" },
        { label: "Duration", value: "01:05" },
        { label: "Speakers", value: "Me, Ada" },
      ],
    });
  });

  it("carries every line with its time and its speaker, a split's guessed time marked ~, an emptied one left out", () => {
    let t = call();
    t = editText(t, t.utterances[0].id, "  ");
    expect(printDocument(t, NONE).slice(1)).toEqual([
      { kind: "utterance", id: t.utterances[1].id, time: "00:04", speaker: "Ada", text: "Sì, forte e chiaro." },
      { kind: "utterance", id: t.utterances[2].id, time: "01:01", speaker: "Me", text: "Allora partiamo." },
      { kind: "utterance", id: t.utterances[3].id, time: "~01:03", speaker: "Ada", text: "Va bene." },
    ]);
  });

  it("states no start for a file, and no speakers when none was told apart", () => {
    let t = createTranscript(1, { source: "file", at: new Date(2026, 9, 25, 15, 42), fileName: "talk.mp3" });
    t = addUtterance(t, { stream: "f-1", startMs: 0, endMs: 2_000, text: "Benvenuti.", voice: null });
    expect(printDocument(t, ALL)).toEqual([
      { kind: "header", title: "25 Oct 15:42 · talk.mp3", facts: [{ label: "Duration", value: "00:02" }] },
      { kind: "utterance", id: t.utterances[0].id, time: "00:00", speaker: null, text: "Benvenuti." },
    ]);
  });

  it("sets each context image before the first line said at its moment or after, one shown first at the start, one after the last at the end", () => {
    let t = call();
    const picture = (image: string, atMs: number) => (t = addContextImage(t, { image, atMs, width: 1280, height: 720 }, 0));
    picture("slide-late", 70_000);
    picture("agenda", 0);
    picture("budget", 30_000);
    const shown = printDocument(t, NONE)
      .slice(1)
      .map((b) => (b.kind === "image" ? `image ${b.image} ${b.time} ${b.width}×${b.height}` : b.kind === "utterance" ? b.time : b.kind));
    expect(shown).toEqual(["image agenda 00:00 1280×720", "00:01", "00:04", "image budget 00:30 1280×720", "01:01", "~01:03", "image slide-late 01:10 1280×720"]);
  });
});

describe("the PDF's document, on request", () => {
  it("sets Ask's answers, in the order asked, after the lines each read, past a picture of the same place; one with no moment at the end, one never answered left out", () => {
    const t = {
      ...addContextImage(call(), { image: "budget", atMs: 30_000, width: 800, height: 600 }, 0),
      asked: [
        answer(4, "Unanswered?", "  ", 9_020),
        answer(2, "Who spoke last?", "Ada, at [01:03].\n\nBriefly.", 9_020),
        answer(1, "Before it all?", "Nothing.", 0),
        answer(3, "Old one?", "Kept before places."),
      ],
    };
    const shown = printDocument(t, { ...NONE, answers: true })
      .slice(1)
      .map((b) => (b.kind === "answer" ? `${b.question} ${b.time ?? "-"}` : b.kind === "image" ? "image" : b.kind === "utterance" ? b.time : b.kind));
    expect(shown).toEqual(["Before it all? 00:00", "00:01", "00:04", "image", "Who spoke last? 00:09", "01:01", "~01:03", "Old one? -"]);
    expect(printDocument(t, { ...NONE, answers: true })).toContainEqual({ kind: "answer", id: 2, time: "00:09", question: "Who spoke last?", answer: "Ada, at [01:03].\n\nBriefly." });
    expect(printDocument(t, NONE).some((b) => b.kind === "answer")).toBe(false);
  });

  it("sets the context in short after the header: the topic, the terms and the languages, nothing else told, under a Transcript heading over the lines", () => {
    const context = { languages: { mode: "free" as const, codes: ["it", "en"] }, topic: " Il budget del trimestre ", terms: ["Ada Rossi", "OKR"], other: "Parla piano.", appliedAt: 1 };
    const blocks = printDocument({ ...call(), context }, { ...NONE, context: true });
    expect(blocks.slice(1, 3)).toEqual([
      {
        kind: "context",
        facts: [
          { label: "Topic", value: "Il budget del trimestre" },
          { label: "Terms", value: "Ada Rossi, OKR" },
          { label: "Languages", value: "Italian, English" },
        ],
      },
      { kind: "heading", text: "Transcript" },
    ]);
    expect(blocks[3]).toMatchObject({ kind: "utterance", time: "00:01" });
    // Only what was told: a forced language alone, no topic, no terms.
    const forced = { ...EMPTY_CONTEXT, languages: { mode: "force" as const, codes: ["it"] } };
    expect(printDocument({ ...call(), context: forced }, { ...NONE, context: true })[1]).toEqual({ kind: "context", facts: [{ label: "Languages", value: "Italian" }] });
  });

  it("leaves the context out when nothing was told, or it was not asked for", () => {
    const told = { ...EMPTY_CONTEXT, topic: "Budget" };
    for (const t of [call(), { ...call(), context: EMPTY_CONTEXT }, { ...call(), context: { ...EMPTY_CONTEXT, other: "Only more." } }]) {
      expect(printDocument(t, ALL).map((b) => b.kind)).not.toContain("context");
    }
    expect(printDocument({ ...call(), context: told }, NONE).map((b) => b.kind)).not.toContain("context");
  });

  it("sets the running notes after the context and before the transcript, in the kind's sections, when there is no recap", () => {
    const t = { ...call(), context: { ...EMPTY_CONTEXT, topic: "Budget" }, notes: written("general", [["Key points", "The line works."]]) };
    const blocks = printDocument(t, ALL);
    expect(blocks.map((b) => b.kind).slice(0, 5)).toEqual(["header", "context", "notes", "heading", "utterance"]);
    expect(blocks[2]).toEqual({ kind: "notes", title: "Notes", markdown: "## Summary\n—\n\n## Key points\n- The line works.\n\n## Next steps\n—" });
  });

  it("sets the recap in place of the running notes once there is one, in the recap's sections", () => {
    const t = { ...call(), notes: written("work", [["Summary", "Running."]]), recap: written("work", [["Figures & dates", "300 euro [00:04]"]]) };
    const notes = printDocument(t, { ...NONE, recap: true })[1];
    expect(notes).toMatchObject({ kind: "notes", title: "Recap" });
    expect(notes.kind === "notes" && notes.markdown).toBe(
      ["## Summary\n—", "## Topics\n—", "## People\n—", "## Figures & dates\n- 300 euro [00:04]", "## Decisions\n—", "## Action items\n—", "## Open questions\n—"].join("\n\n"),
    );
  });

  it("leaves the notes out when none were written, or they were not asked for", () => {
    for (const t of [call(), { ...call(), notes: noNotes("work") }]) expect(printDocument(t, ALL).map((b) => b.kind)).not.toContain("notes");
    const t = { ...call(), notes: written("work", [["Summary", "Running."]]), recap: written("work", [["Summary", "Recap."]]) };
    expect(printDocument(t, NONE).map((b) => b.kind)).not.toContain("notes");
  });
});

describe("the PDF's footer", () => {
  it("reads ignis, the title and page n/m at the foot of every page", () => {
    expect(pageRule("25 Oct 15:42 · Call")).toContain(`@bottom-center { content: "ignis · 25 Oct 15:42 · Call · page " counter(page) "/" counter(pages);`);
  });

  it("is on A4 pages, set by the PDF alone: printing anything else in the page keeps the browser's own", () => {
    expect(pageRule("Call")).toMatch(/^@page \{ size: A4; margin: [^;]+; @bottom-center/);
  });

  it("holds a title with quotes, a backslash, a line break or a closing tag as text", () => {
    const rule = pageRule('Il "piano" C:\\dati\n</style><b>');
    expect(rule).toContain(`content: "ignis · Il \\"piano\\" C:\\\\dati \\3c /style>\\3c b> · page "`);
    expect(rule).not.toContain("</style");
    expect(rule).not.toContain("\n");
  });
});

describe("the PDF dialog's options", () => {
  it("say what each would add to this transcript's PDF", () => {
    const t = {
      ...call(),
      context: { ...EMPTY_CONTEXT, topic: "Budget", languages: { mode: "free" as const, codes: ["it"] } },
      recap: written("work", [["Summary", "Recap."]]),
      asked: [answer(1, "Who?", "Ada.", 9_020), answer(2, "When?", "Now.", 9_020)],
    };
    expect(pdfHints(t)).toEqual({
      recap: "The recap, before the transcript.",
      context: "Topic and languages, after the header.",
      answers: "2 answers, boxed at the moments they were asked.",
    });
    expect(pdfHints({ ...t, recap: undefined, notes: written("work", [["Summary", "Running."]]), asked: [answer(1, "Who?", "Ada.")] })).toMatchObject({
      recap: "No recap yet: the running notes, before the transcript.",
      answers: "1 answer, boxed at the moment it was asked.",
    });
  });

  it("say when there is nothing for one to add", () => {
    expect(pdfHints(call())).toEqual({ recap: "No recap or notes written yet.", context: "Nothing told: no topic, terms or languages.", answers: "No answers yet." });
  });
});

describe("the PDF dialog's choice", () => {
  afterEach(() => {
    Object.defineProperty(globalThis, "localStorage", { value: undefined, configurable: true, writable: true });
    reloadPdfOptions();
  });

  it("is everything on the first time, then the last one made in this browser; a stored value it cannot read is everything again", () => {
    const storage = new Map<string, string>();
    const local = { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => void storage.set(key, value) };
    Object.defineProperty(globalThis, "localStorage", { value: local, configurable: true, writable: true });
    reloadPdfOptions();
    expect(lastPdfOptions()).toEqual(ALL);
    choosePdfOptions({ recap: true, context: false, answers: true });
    expect(storage.get(PDF_OPTIONS_KEY)).toBe("recap,answers");
    reloadPdfOptions();
    expect(lastPdfOptions()).toEqual({ recap: true, context: false, answers: true });
    choosePdfOptions(NONE);
    reloadPdfOptions();
    expect(lastPdfOptions()).toEqual(NONE);
    storage.set(PDF_OPTIONS_KEY, "answers,recap,images");
    reloadPdfOptions();
    expect(lastPdfOptions()).toEqual(ALL);
  });

  it("lasts the page where the browser keeps nothing", () => {
    reloadPdfOptions();
    choosePdfOptions({ recap: false, context: true, answers: false });
    expect(lastPdfOptions()).toEqual({ recap: false, context: true, answers: false });
  });
});

describe("the PDF's document, for every choice of the dialog", () => {
  /** A call with all a PDF may carry: a context, running notes and a recap, a picture, and two answers. */
  const rich = (): Transcript => ({
    ...addContextImage(call(), { image: "budget", atMs: 30_000, width: 800, height: 600 }, 0),
    context: { ...EMPTY_CONTEXT, topic: "Budget" },
    notes: written("work", [["Summary", "Running."]]),
    recap: written("work", [["Summary", "Recap."]]),
    asked: [answer(1, "Who spoke last?", "Ada.", 9_020), answer(2, "Old one?", "Kept before places.")],
  });
  const lines = "utterance utterance image utterance utterance";
  const answered = "utterance utterance image answer utterance utterance answer";

  it.each([
    [{ recap: false, context: false, answers: false }, `header ${lines}`],
    [{ recap: false, context: false, answers: true }, `header ${answered}`],
    [{ recap: false, context: true, answers: false }, `header context heading ${lines}`],
    [{ recap: false, context: true, answers: true }, `header context heading ${answered}`],
    [{ recap: true, context: false, answers: false }, `header notes heading ${lines}`],
    [{ recap: true, context: false, answers: true }, `header notes heading ${answered}`],
    [{ recap: true, context: true, answers: false }, `header context notes heading ${lines}`],
    [{ recap: true, context: true, answers: true }, `header context notes heading ${answered}`],
  ])("with %o prints %s", (options, kinds) => {
    expect(printDocument(rich(), options).map((b) => b.kind).join(" ")).toBe(kinds);
  });
});
