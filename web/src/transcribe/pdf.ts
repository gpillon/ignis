import { isSpeechLanguage, languageName } from "../speech/languages.ts";
import type { ImageId } from "../store/imageStore.ts";
import { answerAt } from "./ask.ts";
import type { CallContext } from "./context.ts";
import { imageAt } from "./contextImages.ts";
import { clock, startedOn, startOf } from "./export.ts";
import { notesMarkdown } from "./notes.ts";
import { recapMarkdown } from "./recap.ts";
import { titleOf } from "./sessions.ts";
import { clockOf, type Transcript } from "./transcript.ts";
import { viewerChoice } from "./viewerChoice.ts";

// The PDF of a transcript (spec #428, "Exports"): what it prints is computed
// here, from the transcript and the three options its dialog asks, as a list
// of blocks in the order they print; the print view (PrintView.tsx) only draws
// them, and the browser's print dialog saves the file. A header and the lines
// always; the context in short, the notes and Ask's answers on request.

/** What a PDF carries beside the header and the lines: each on or off. */
export type PdfOptions = { recap: boolean; context: boolean; answers: boolean };

export const PDF_OPTIONS_KEY = "ignis.transcribePdf";

const PARTS = ["recap", "context", "answers"] as const;

/** The options as this browser keeps them: the parts on, in order, comma-separated; `none` with none on. */
const keptAs = (options: PdfOptions) => PARTS.filter((p) => options[p]).join(",") || "none";

/** Every choice the dialog can keep, each of the eight once. */
const KEPT = Array.from({ length: 1 << PARTS.length }, (_, bits) => keptAs({ recap: (bits & 1) !== 0, context: (bits & 2) !== 0, answers: (bits & 4) !== 0 }));

/** The dialog's last choice, this browser's, as the notes' kind is: everything on at first. */
const picked = viewerChoice<string>({ key: PDF_OPTIONS_KEY, choices: KEPT, fallback: keptAs({ recap: true, context: true, answers: true }) });

export function lastPdfOptions(): PdfOptions {
  const on = picked.get().split(",");
  return { recap: on.includes("recap"), context: on.includes("context"), answers: on.includes("answers") };
}
export const choosePdfOptions = (options: PdfOptions) => picked.choose(keptAs(options));
export const reloadPdfOptions = picked.reload;

/** A label and its value: the header's start, duration and speakers; the context's topic, terms and languages. */
export type Fact = { label: string; value: string };

export type PrintBlock =
  | { kind: "header"; title: string; facts: Fact[] }
  /** The context in short: the topic, the terms and the languages the transcription was told, not what it was told under More. */
  | { kind: "context"; facts: Fact[] }
  /** The recap, else the running notes, as Markdown in their sections. */
  | { kind: "notes"; title: "Recap" | "Notes"; markdown: string }
  | { kind: "heading"; text: string }
  | { kind: "utterance"; id: number; time: string; speaker: string | null; text: string }
  | { kind: "image"; id: number; time: string; image: ImageId; width: number; height: number }
  /** An answer of Ask's, boxed, its question the title; `time` the moment it read up to, `null` for one kept before answers had one. */
  | { kind: "answer"; id: number; time: string | null; question: string; answer: string };

/** The context in short, a fact each of the topic, the terms and the languages told; none when nothing of them was. */
function contextFacts(c: CallContext | undefined): Fact[] {
  if (!c) return [];
  const languages = c.languages.codes.filter(isSpeechLanguage).map(languageName);
  return [
    ...(c.topic.trim() === "" ? [] : [{ label: "Topic", value: c.topic.trim() }]),
    ...(c.terms.length === 0 ? [] : [{ label: "Terms", value: c.terms.join(", ") }]),
    ...(languages.length === 0 ? [] : [{ label: "Languages", value: languages.join(", ") }]),
  ];
}

/** The notes meant for after, as Markdown in their sections: the recap when there is one, else the running notes. */
function notesBlock(t: Transcript): PrintBlock[] {
  const recap = recapMarkdown(t.recap);
  if (recap !== "") return [{ kind: "notes", title: "Recap", markdown: recap }];
  const running = notesMarkdown(t.notes);
  return running === "" ? [] : [{ kind: "notes", title: "Notes", markdown: running }];
}

/**
 * The blocks a PDF of `t` prints, in order: the header; the context in short and the notes when asked for, with a
 * Transcript heading under them; the lines with the context images, and Ask's answers when asked for, at their
 * moments.
 */
export function printDocument(t: Transcript, options: PdfOptions): PrintBlock[] {
  const names = new Map(t.speakers.map((s) => [s.id, s.name]));
  const said = t.utterances.filter((u) => u.text.trim() !== "");
  const spoke = t.speakers.filter((s) => said.some((u) => u.speaker === s.id)).map((s) => s.name);
  const facts: Fact[] = [
    ...(t.startedAt === null ? [] : [{ label: "Started", value: startedOn(t.startedAt) }]),
    { label: "Duration", value: clock(clockOf(t)) },
    ...(spoke.length === 0 ? [] : [{ label: "Speakers", value: spoke.join(", ") }]),
  ];
  const context = options.context ? contextFacts(t.context) : [];
  const before: PrintBlock[] = [...(context.length > 0 ? [{ kind: "context" as const, facts: context }] : []), ...(options.recap ? notesBlock(t) : [])];

  const body: PrintBlock[] = [];
  // The context images in the order of their moments, each before the first utterance at its moment or after (the sheet's place).
  const images = [...(t.images ?? [])].sort((x, y) => x.atMs - y.atMs || x.id - y.id);
  // Ask's answers in the order asked, each after the utterances it read, past a picture of the same place (the sheet's order).
  const answers = options.answers ? [...(t.asked ?? [])].sort((x, y) => x.id - y.id).filter((a) => a.answer.trim() !== "") : [];
  const asidesAt = (place: number) => {
    for (const i of images) {
      if (imageAt(t.utterances, i.atMs) === place) body.push({ kind: "image", id: i.id, time: clock(i.atMs), image: i.image, width: i.width, height: i.height });
    }
    for (const a of answers) {
      if (answerAt(t.utterances, a.untilMs) === place) {
        body.push({ kind: "answer", id: a.id, time: a.untilMs === undefined ? null : clock(a.untilMs), question: a.question.trim(), answer: a.answer.trim() });
      }
    }
  };
  t.utterances.forEach((u, place) => {
    asidesAt(place);
    const text = u.text.trim();
    if (text !== "") body.push({ kind: "utterance", id: u.id, time: startOf(u), speaker: u.speaker === null ? null : (names.get(u.speaker) ?? null), text });
  });
  asidesAt(t.utterances.length);

  const heading: PrintBlock[] = before.length > 0 && body.length > 0 ? [{ kind: "heading", text: "Transcript" }] : [];
  return [{ kind: "header", title: titleOf(t), facts }, ...before, ...heading, ...body];
}

/** "a", "a and b", "a, b and c". */
const listed = (words: string[]) => (words.length < 2 ? words.join("") : `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`);

/** What each of the dialog's options would add to a PDF of `t`, the line under it. */
export function pdfHints(t: Transcript): Record<keyof PdfOptions, string> {
  const all = printDocument(t, { recap: true, context: true, answers: true });
  const notes = all.find((b) => b.kind === "notes");
  const context = all.find((b) => b.kind === "context");
  const answers = all.filter((b) => b.kind === "answer").length;
  const told = context ? context.facts.map((f, i) => (i === 0 ? f.label : f.label.toLocaleLowerCase())) : [];
  return {
    recap: !notes ? "No recap or notes written yet." : notes.title === "Recap" ? "The recap, before the transcript." : "No recap yet: the running notes, before the transcript.",
    context: told.length > 0 ? `${listed(told)}, after the header.` : "Nothing told: no topic, terms or languages.",
    answers:
      answers === 0 ? "No answers yet." : answers === 1 ? "1 answer, boxed at the moment it was asked." : `${answers} answers, boxed at the moments they were asked.`,
  };
}

/** `text` as a CSS string's content: a quote, a backslash and a `<` escaped, so a title is never read as anything else. */
const cssText = (text: string) => text.replace(/[\r\n]+/g, " ").replace(/[\\"]/g, (c) => `\\${c}`).replace(/</g, "\\3c ");

/**
 * The print view's page rule: A4, and the footer, "ignis · title · page n/m", in a page margin box, where alone the
 * page counters count. It names the title, so the print view writes it for each PDF, and it holds only while the
 * view stands: printing anything else in the page keeps the browser's own pages.
 */
export function pageRule(title: string): string {
  const content = `"ignis · ${cssText(title)} · page " counter(page) "/" counter(pages)`;
  const footer = `@bottom-center { content: ${content}; font-family: "Chakra Petch", ui-sans-serif, sans-serif; font-size: 8pt; color: #5c646e; }`;
  return `@page { size: A4; margin: 16mm 16mm 18mm; ${footer} }`;
}
