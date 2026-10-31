import type { ConversationRequest, Settings } from "../api/request.ts";
import { assistRequest, CHARS_PER_TOKEN } from "./assist.ts";
import { readImages, readingLine } from "./contextImages.ts";
import { type DraftLine, type NoteKind, type NoteLine, type Notes, NOTES_SETTINGS, notesMarkdown, type NoteSection, parseLines, readNotes, sectionsTask, specOf, withWriting, type Written } from "./notes.ts";
import { clock } from "./export.ts";
import { taskInput } from "./taskContext.ts";
import { dayAndTime, type Transcript } from "./transcript.ts";

// The recap (spec #428, "The recap"): the notes written once from the whole
// transcript, for what comes after the conversation. Running notes kept a
// minute at a time stay poor at anything weighed against the whole call; one
// writing over all of it, told the context, got the Summary, the Decisions,
// the Action items and the People right (#429's probe). It is written when a
// listening ends, in place of the notes' last writing, and on a press.

/** The sections a kind's recap adds to the kind's own, each after the section named. */
const RECAP_ADDITIONS: Partial<Record<NoteKind, { after: string; section: NoteSection }[]>> = {
  work: [
    {
      after: "People",
      section: { name: "Figures & dates", ask: "every figure said (an amount, a percentage, a quantity, a date, a deadline), one a line, with what it refers to and the time it was said" },
    },
  ],
  lesson: [
    { after: "Key concepts", section: { name: "Terms", ask: "the terms introduced, each with what it means" } },
    { after: "To review", section: { name: "Homework & dates", ask: "homework, readings, exams and their dates" } },
  ],
};

/** A kind's recap's sections, in order: the kind's, with the recap's additions. */
export function recapSections(kind: NoteKind): NoteSection[] {
  const additions = RECAP_ADDITIONS[kind] ?? [];
  return specOf(kind).sections.flatMap((s) => [s, ...additions.filter((a) => a.after === s.name).map((a) => a.section)]);
}

/** What the recap is asked beyond the notes' rules. */
const RECAP_RULES = [
  "- The Summary and the Topics carry the main threads by the weight the conversation gave them, not every aside.",
  "- Decisions are what was agreed; Action items what someone committed to do; People everyone who spoke or was named.",
  "- Notes the user took during the conversation are true: use them, and let them correct what the transcript seems to say.",
];

/** The recap's task: its sections and how to write them, the recap's rules first. */
export function recapTask(kind: NoteKind): string {
  return sectionsTask(`Write the recap of ${specOf(kind).about}`, recapSections(kind), RECAP_RULES);
}

/** The tokens a recap may take: the probe's, twice the notes', for its added sections over a whole call. */
export const RECAP_MAX_TOKENS = 3_000;

export const RECAP_SETTINGS: Omit<Settings, "systemPrompt"> = { ...NOTES_SETTINGS, maxTokens: RECAP_MAX_TOKENS };

/**
 * The user's own running lines, as notes they took and the recap takes as true. The model's running lines are never
 * sent, nor the ones the user removed: a mistake the notes made live does not carry over into the recap.
 */
function userNotes(t: Transcript): string {
  const theirs = (t.notes?.lines ?? []).filter((l) => l.author === "user");
  if (theirs.length === 0) return "Notes the user took during the conversation: none.";
  return `Notes the user took during the conversation, to be taken as true:\n${theirs.map((l) => `- ${l.section}: ${l.text}`).join("\n")}`;
}

/**
 * The lines the user wrote in the recap and the ones they removed from it, when there are any: a new recap keeps the
 * first as they are (withRecap), so it is told neither to repeat them nor to contradict them, nor to write the others
 * again (#436).
 */
function userRecap(recap: Notes | undefined): string[] {
  const listed = (lines: NoteLine[]) => lines.map((l) => `- ${l.section}: ${l.text}`).join("\n");
  const theirs = (recap?.lines ?? []).filter((l) => l.author === "user");
  const removed = recap?.removed ?? [];
  return [
    ...(theirs.length > 0 ? [`Lines the user wrote in the recap, which stay in it as they are: they are true; write nothing that contradicts them, and do not write them again:\n${listed(theirs)}`] : []),
    ...(removed.length > 0 ? [`Lines the user removed from the recap, never to be written again:\n${listed(removed)}`] : []),
  ];
}

/**
 * A recap of `t` as `kind`, and where its transcript starts when the start did not fit. Its user turn is the context,
 * the title and the whole transcript, then what the context images showed (`readings`, a line each as their reading
 * gives it, `[mm:ss] (image) ...`: every image of `t` read, unless given), the notes the user took, and the lines they
 * wrote in the recap or removed from it. The readings stand there whole, whatever of the transcript fits, and not again
 * among its utterances.
 */
export function recapRequest(
  t: Transcript,
  kind: NoteKind,
  contextLimit: number | null,
  readings: string[] = readImages(t).map((i) => readingLine(i)),
): { request: ConversationRequest; fromMs: number | null } {
  const shown = readings.length > 0 ? [`Pictures shown during the conversation, as read at their moments:\n${readings.join("\n")}`] : [];
  const after = [...shown, userNotes(t), ...userRecap(t.recap)].join("\n\n");
  const { content, fromMs } = taskInput({ ...t, images: [] }, contextLimit, RECAP_MAX_TOKENS + Math.ceil(after.length / CHARS_PER_TOKEN));
  return { request: assistRequest(RECAP_SETTINGS, recapTask(kind), `${content}\n${after}`), fromMs };
}

/** The recap's answer as lines, in its sections as the kind's recap spells them (notes.ts's parseLines). */
export const parseRecap = (answer: string, kind: NoteKind): DraftLine[] => parseLines(answer, recapSections(kind).map((s) => s.name));

/**
 * A Transcript's recap once a writing lands (notes.ts's withWriting): its lines in place of the model's, the user's
 * lines and the ones they removed kept, and the kind it was written as its own.
 */
export const withRecap = (recap: Notes | undefined, written: Written): Notes => withWriting(recap && { ...recap, kind: written.writtenAs }, written);

/** The recap as Markdown, what the tab shows and Copy recap gives: the recap's sections in order (notesMarkdown). */
export const recapMarkdown = (recap: Notes | undefined): string => notesMarkdown(recap, recap && recapSections(recap.writtenAs ?? recap.kind));

const counted = (n: number) => (n === 1 ? "1 utterance" : `${n} utterances`);

/** The line under the recap: when it was written, from how much, and what came since; `null` before one is. */
export function recapStatus(recap: Notes | undefined, utterances: number): string | null {
  if (!recap?.writtenAs || recap.writtenAt === null) return null;
  const time = dayAndTime(new Date(recap.writtenAt)).time;
  const from = recap.fromMs === null ? "" : `, from ${clock(recap.fromMs)} on: the start did not fit the model's context`;
  const fresh = Math.max(0, utterances - recap.covered);
  return `Written at ${time} from ${counted(recap.covered)}${from}.${fresh > 0 ? ` ${fresh} more since.` : ""}`;
}

/** A stored Transcript's recap, as notes.ts reads notes; `undefined` for one this page cannot read, or never written. */
export function readRecap(value: unknown): Notes | undefined {
  const recap = readNotes(value);
  return recap?.writtenAs ? recap : undefined;
}

/**
 * Whether a listening's end writes `t`'s recap as `kind`: once something was heard, unless a recap of that kind
 * already covers every utterance (one pressed after the last).
 */
export function recapDue(t: Transcript, kind: NoteKind): boolean {
  if (t.utterances.every((u) => u.text.trim() === "")) return false;
  const recap = t.recap;
  return !recap?.writtenAs || recap.writtenAs !== kind || recap.covered < t.utterances.length;
}
