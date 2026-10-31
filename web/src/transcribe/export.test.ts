import { describe, expect, it } from "vitest";
import { clock, toSrt, toText, toVtt } from "./export.ts";
import { toMarkdown } from "./markdown.ts";
import { noNotes, type Notes } from "./notes.ts";
import { addUtterance, createTranscript, editText, renameSpeaker, splitUtterance, type Transcript } from "./transcript.ts";

// The five exports (spec speech/07, "Out of the view", Q95): computed in the
// page from the utterances as edited, speakers by their names.

/** A call: Me, then a voice of the call renamed Ada, then Me again, split by hand. */
function call(): Transcript {
  let t = createTranscript(1, { source: "call", at: new Date(2026, 9, 25, 15, 42) });
  t = addUtterance(t, { stream: "mic-1", startMs: 1_200, endMs: 3_900, text: "Mi sentite?", voice: { kind: "me" } });
  t = addUtterance(t, { stream: "call-1", startMs: 4_480, endMs: 9_020, text: "Sì, forte e chiaro.", voice: { kind: "speaker", index: 0 } });
  t = addUtterance(t, { stream: "mic-1", startMs: 61_000, endMs: 65_000, text: "Allora partiamo. Va bene.", voice: { kind: "me" } });
  t = renameSpeaker(t, t.speakers[1].id, "Ada");
  t = splitUtterance(t, t.utterances[2].id, 16, t.speakers[1].id);
  return editText(t, t.utterances[0].id, "Mi sentite bene?");
}

describe("clock", () => {
  it("is mm:ss from the start, h:mm:ss past an hour", () => {
    expect(clock(0)).toBe("00:00");
    expect(clock(65_999)).toBe("01:05");
    expect(clock(3_599_000)).toBe("59:59");
    expect(clock(3_723_000)).toBe("1:02:03");
  });
});

describe("the exports", () => {
  it("as text and for the clipboard: one line per utterance, [mm:ss] Speaker: text, the edit carried, a split time marked ~", () => {
    expect(toText(call())).toBe(
      ["[00:01] Me: Mi sentite bene?", "[00:04] Ada: Sì, forte e chiaro.", "[01:01] Me: Allora partiamo.", "[~01:03] Ada: Va bene."].join("\n") + "\n",
    );
  });

  it("as Markdown: the title and the start in a header, then **Speaker** (mm:ss): text", () => {
    expect(toMarkdown(call())).toBe(
      [
        "# 25 Oct 15:42 · Call",
        "",
        "Started 25 Oct 2026, 15:42",
        "",
        "**Me** (00:01): Mi sentite bene?",
        "",
        "**Ada** (00:04): Sì, forte e chiaro.",
        "",
        "**Me** (01:01): Allora partiamo.",
        "",
        "**Ada** (~01:03): Va bene.",
        "",
      ].join("\n"),
    );
  });

  it("as SubRip: a numbered cue per utterance from its start to its end, the speaker before the text", () => {
    expect(toSrt(call())).toBe(
      [
        "1",
        "00:00:01,200 --> 00:00:03,900",
        "Me: Mi sentite bene?",
        "",
        "2",
        "00:00:04,480 --> 00:00:09,020",
        "Ada: Sì, forte e chiaro.",
        "",
        "3",
        "00:01:01,000 --> 00:01:03,560",
        "Me: Allora partiamo.",
        "",
        "4",
        "00:01:03,560 --> 00:01:05,000",
        "Ada: Va bene.",
        "",
      ].join("\n"),
    );
  });

  it("as WebVTT: a cue per utterance, the speaker in a voice tag, the text escaped", () => {
    let t = call();
    t = editText(t, t.utterances[1].id, "Sì <forte> & chiaro.");
    expect(toVtt(t)).toBe(
      [
        "WEBVTT",
        "",
        "00:00:01.200 --> 00:00:03.900",
        "<v Me>Mi sentite bene?",
        "",
        "00:00:04.480 --> 00:00:09.020",
        "<v Ada>Sì &lt;forte&gt; &amp; chiaro.",
        "",
        "00:01:01.000 --> 00:01:03.560",
        "<v Me>Allora partiamo.",
        "",
        "00:01:03.560 --> 00:01:05.000",
        "<v Ada>Va bene.",
        "",
      ].join("\n"),
    );
  });

  it("without speakers, writes the text alone, and leaves out an utterance whose text was emptied", () => {
    let t = createTranscript(1, { source: "file", at: new Date(2026, 9, 25, 15, 42), fileName: "talk.mp3" });
    t = addUtterance(t, { stream: "f-1", startMs: 0, endMs: 2_000, text: "Benvenuti.", voice: null });
    t = addUtterance(t, { stream: "f-1", startMs: 3_000, endMs: 4_000, text: "Rumore.", voice: null });
    t = editText(t, t.utterances[1].id, "  ");
    expect(toText(t)).toBe("[00:00] Benvenuti.\n");
    expect(toMarkdown(t)).toBe("# 25 Oct 15:42 · talk.mp3\n\n(00:00): Benvenuti.\n");
    expect(toSrt(t)).toBe("1\n00:00:00,000 --> 00:00:02,000\nBenvenuti.\n");
    expect(toVtt(t)).toBe("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nBenvenuti.\n");
  });
});

// The `.md` with the notes and Ask's answers (spec #428, "Exports"); the other
// three exports never carry them.

const written = (kind: Notes["kind"], lines: [string, string][]): Notes => ({
  ...noNotes(kind),
  lines: lines.map(([section, text], i) => ({ id: `N${i + 1}`, section, text, author: "model" as const })),
  nextId: lines.length + 1,
  writtenAs: kind,
  covered: 3,
  writtenAt: 1_000,
});

const answer = (id: number, question: string, text: string, untilMs?: number) => ({ id, question, answer: text, at: id, untilMs });

describe("the Markdown export with notes and answers", () => {
  it("is as before with no notes and no answers", () => {
    expect(toMarkdown({ ...call(), notes: noNotes("work"), asked: [] })).toBe(toMarkdown(call()));
  });

  it("carries the running notes in the kind's sections after the start and before the transcript", () => {
    const md = toMarkdown({ ...call(), notes: written("work", [["Summary", "A check of the line."]]) });
    expect(md.startsWith("# 25 Oct 15:42 · Call\n\nStarted 25 Oct 2026, 15:42\n\n## Summary\n- A check of the line.\n\n## ")).toBe(true);
    expect(md).toContain("\n—\n");
    expect(md.indexOf("## Transcript")).toBeGreaterThan(md.lastIndexOf("\n—"));
    expect(md.indexOf("**Me** (00:01)")).toBeGreaterThan(md.indexOf("## Transcript"));
  });

  it("carries the recap, not the running notes, when there is one", () => {
    const md = toMarkdown({ ...call(), notes: written("work", [["Summary", "Running."]]), recap: written("work", [["Summary", "Recap."]]) });
    expect(md).toContain("- Recap.");
    expect(md).not.toContain("Running.");
  });

  it("sets each answer between the utterances at its moment, a quoted block titled by its question, and one with no moment at the end", () => {
    const asked = [answer(2, "Who spoke last?", "Ada, at [01:03].\n\nBriefly.", 9_020), answer(1, "Before it all?", "Nothing.", 0), answer(3, "Old one?", "Kept before places.")];
    expect(toMarkdown({ ...call(), asked })).toBe(
      [
        "# 25 Oct 15:42 · Call",
        "",
        "Started 25 Oct 2026, 15:42",
        "",
        "> **Before it all?**",
        ">",
        "> Nothing.",
        "",
        "**Me** (00:01): Mi sentite bene?",
        "",
        "**Ada** (00:04): Sì, forte e chiaro.",
        "",
        "> **Who spoke last?**",
        ">",
        "> Ada, at [01:03].",
        ">",
        "> Briefly.",
        "",
        "**Me** (01:01): Allora partiamo.",
        "",
        "**Ada** (~01:03): Va bene.",
        "",
        "> **Old one?**",
        ">",
        "> Kept before places.",
        "",
      ].join("\n"),
    );
  });

  it("leaves .txt, .srt and .vtt as they are with notes, a recap and answers kept", () => {
    const plain = call();
    const t = { ...plain, notes: written("work", [["Summary", "x"]]), recap: written("work", [["Summary", "y"]]), asked: [answer(1, "Q?", "A.", 9_020)] };
    expect(toText(t)).toBe(toText(plain));
    expect(toSrt(t)).toBe(toSrt(plain));
    expect(toVtt(t)).toBe(toVtt(plain));
  });
});
