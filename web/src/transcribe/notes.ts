import { canChat, imageLoaded, loadUnderWay, type ModelState } from "../api/model.ts";
import type { ConversationRequest, Settings } from "../api/request.ts";
import { assistRequest } from "./assist.ts";
import { taskInput } from "./taskContext.ts";
import { clock } from "./export.ts";
import { dayAndTime, type Transcript } from "./transcript.ts";
import { viewerChoice } from "./viewerChoice.ts";

// Notes on a Transcript (spec speech/07, Q124-Q127): the text model loaded
// reads the transcript and writes the notes a kind of conversation wants, a
// Markdown section each (the server has no structured output, #287). They are
// written on a press, or kept up while listening: about once a minute while
// something new was said, and once more at the end.
//
// The request is assist.ts's shape, which the suggestions share (Q134): the
// preamble and this kind's sections as the system prompt, the title and the
// transcript as its `.txt` as the user turn, which a listening only appends
// to, so the next writing reuses what the engine read. The notes written
// before are not sent back: each writing starts from the transcript, so a
// correction holds in the next one and nothing the model got wrong compounds.

export type NoteKind = "general" | "work" | "game" | "lesson";

/** A section the model fills, and what it asks for. */
export type NoteSection = { name: string; ask: string };

export type NoteKindSpec = { kind: NoteKind; label: string; about: string; sections: NoteSection[] };

export const NOTE_KINDS: NoteKindSpec[] = [
  {
    kind: "general",
    label: "General",
    about: "a conversation",
    sections: [
      { name: "Summary", ask: "what the conversation is about and where it got to, in two to four sentences" },
      { name: "Key points", ask: "what was said that matters, a bullet each" },
      { name: "Next steps", ask: "what someone said they, or someone else, will do" },
    ],
  },
  {
    kind: "work",
    label: "Work call",
    about: "a work call or meeting",
    sections: [
      { name: "Summary", ask: "the purpose of the call and where it got to, in two to four sentences" },
      { name: "Topics", ask: "each subject discussed, with what was said of it in a few words" },
      { name: "People", ask: "each person who spoke or was named, with their role or stance where the conversation shows it" },
      { name: "Decisions", ask: "what was agreed, with the time it was said" },
      { name: "Action items", ask: "who does what, and by when if it was said, with the time it was said, as a task list (`- [ ] `)" },
      { name: "Open questions", ask: "what was raised and left unresolved" },
    ],
  },
  {
    kind: "game",
    label: "Game session",
    about: "people playing games together",
    sections: [
      { name: "Games", ask: "each game or match played, in order, with the time it started" },
      { name: "Players", ask: "who played, and on which side or team where it shows" },
      { name: "Results", ask: "who won or lost what, with the scores where they were said" },
      { name: "Highlights", ask: "the moments worth remembering: plays, turnarounds, jokes" },
    ],
  },
  {
    kind: "lesson",
    label: "Lesson",
    about: "a lesson, a lecture or a talk",
    sections: [
      { name: "Topics", ask: "the subjects covered, in order, with the time each began" },
      { name: "Key concepts", ask: "each idea explained, with its explanation in a sentence" },
      { name: "Questions", ask: "what was asked, and the answer when one came" },
      { name: "To review", ask: "what to study or look up after" },
    ],
  },
];

export const specOf = (kind: NoteKind): NoteKindSpec => NOTE_KINDS.find((k) => k.kind === kind) ?? NOTE_KINDS[0];

/** What a kind's tile says under its name: its sections, the summary left out of a long list, four at most. */
export function kindHint(spec: NoteKindSpec): string {
  const names = spec.sections.map((s) => s.name).filter((name, _, all) => all.length <= 4 || name !== "Summary");
  const shown = names.slice(0, 4).map((name, i) => (i === 0 ? name : name.toLocaleLowerCase()));
  return shown.join(", ");
}

export const NOTE_KIND_KEY = "ignis.transcribeNoteKind";

/** The kind last picked, this browser's: what a Transcript with no notes yet starts at. General at first. */
const picked = viewerChoice<NoteKind>({ key: NOTE_KIND_KEY, choices: NOTE_KINDS.map((k) => k.kind), fallback: "general" });
export const lastNoteKind = picked.get;
export const chooseNoteKind = picked.choose;
export const reloadNoteKind = picked.reload;

/** The kind a Transcript's next notes are written as. */
export const kindOf = (t: Transcript): NoteKind => t.notes?.kind ?? lastNoteKind();

/** A Transcript's notes before any are written. */
export const noNotes = (kind: NoteKind): Notes => ({ kind, text: "", writtenAs: null, covered: 0, writtenAt: null, fromMs: null });

/** A Transcript's notes, as its document keeps them. */
export type Notes = {
  /** The kind picked for this Transcript: what the next writing asks for. */
  kind: NoteKind;
  /** The notes last written, as the model wrote them; "" before any. */
  text: string;
  /** The kind `text` was written as; `null` before any. */
  writtenAs: NoteKind | null;
  /** How many of the Transcript's utterances `text` was written from. */
  covered: number;
  /** When it was written, in epoch ms. */
  writtenAt: number | null;
  /** Where the transcript sent began, when its start did not fit the model's context; `null` when all of it went. */
  fromMs: number | null;
};

/** The task: the kind's sections and how to write them (assist.ts). */
export function notesTask(kind: NoteKind): string {
  const spec = specOf(kind);
  const sections = spec.sections.map((s) => `## ${s.name}\n${s.ask}.`).join("\n\n");
  return [
    `Write the notes of ${spec.about}: these sections, in this order, each under a level-2 Markdown heading (\`## \`):`,
    sections,
    [
      "Rules:",
      "- Keep each heading exactly as given. Write the text under them in the language the context says to write in.",
      "- Short bullet points. No preamble, no closing remark, nothing outside the sections.",
      "- A section with nothing in it yet is `—` alone.",
      "- Point at a moment by its time, as [mm:ss].",
      "- Write the notes as of the transcript's last line.",
    ].join("\n"),
  ].join("\n\n");
}

/** The tokens a writing may take. */
export const NOTES_MAX_TOKENS = 1_500;

export const NOTES_SETTINGS: Omit<Settings, "systemPrompt"> = {
  temperature: 0.3,
  topP: 0.9,
  maxTokens: NOTES_MAX_TOKENS,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "agent",
};

/** A writing of `t`'s notes as `kind`, and where its transcript starts when the start did not fit. */
export function notesRequest(t: Transcript, kind: NoteKind, contextLimit: number | null): { request: ConversationRequest; fromMs: number | null } {
  const { content, fromMs } = taskInput(t, contextLimit, NOTES_MAX_TOKENS);
  return { request: assistRequest(NOTES_SETTINGS, notesTask(kind), content), fromMs };
}

/** Keeping up: the first notes once this many utterances are in. */
export const FIRST_LIVE_UTTERANCES = 3;
/** Keeping up: the least time from one writing's start to the next. */
export const LIVE_GAP_MS = 60_000;

/**
 * Whether a listening that keeps the notes up writes them now. `sinceLastMs`: since this Transcript's last writing
 * began, `null` if none did; `final`, the listening just ended.
 */
export function dueForUpdate({ notes, utterances, sinceLastMs, final = false }: { notes: Notes | undefined; utterances: number; sinceLastMs: number | null; final?: boolean }): boolean {
  const covered = notes?.writtenAs ? notes.covered : 0;
  const fresh = utterances - covered;
  if (notes?.writtenAs && notes.writtenAs !== notes.kind) return utterances > 0;
  if (fresh <= 0) return false;
  if (final) return true;
  if (covered === 0) return fresh >= FIRST_LIVE_UTTERANCES;
  return sinceLastMs === null || sinceLastMs >= LIVE_GAP_MS;
}

const counted = (n: number) => (n === 1 ? "1 utterance" : `${n} utterances`);

/** The line under the notes: when they were written, from how much, and what came since; `null` before any. */
export function notesStatus(notes: Notes | undefined, utterances: number): string | null {
  if (!notes?.writtenAs || notes.writtenAt === null) return null;
  const time = dayAndTime(new Date(notes.writtenAt)).time;
  if (notes.writtenAs !== notes.kind) {
    return `Written as ${specOf(notes.writtenAs).label} notes at ${time}. Write notes to redo them as ${specOf(notes.kind).label} notes.`;
  }
  const from = notes.fromMs === null ? "" : `, from ${clock(notes.fromMs)} on: the start did not fit the model's context`;
  const fresh = Math.max(0, utterances - notes.covered);
  return `Written at ${time} from ${counted(notes.covered)}${from}.${fresh > 0 ? ` ${fresh} more since.` : ""}`;
}

/**
 * Why the notes and the suggestions cannot ask the text model now; `null` while one serves, or would load again on
 * the request.
 */
export function textModelBlocked(model: ModelState): string | null {
  if (canChat(model)) return null;
  if (model.state === "reading") return "Reading which model is loaded.";
  if (imageLoaded(model)) return "The image model is loaded: the notes and the suggestions need a text model. Switch in the Server view.";
  if (loadUnderWay(model)) return "A model is loading: the notes and the suggestions wait until it serves.";
  return "No text model serves: the notes and the suggestions need one. Load it in the Server view.";
}

/** The model's notes as kept: trimmed, and out of a fence when it wrapped them in one. */
export function tidyNotes(text: string): string {
  const trimmed = text.trim();
  const fenced = /^```(?:markdown|md)?\n([\s\S]*?)\n```$/.exec(trimmed);
  return (fenced ? fenced[1] : trimmed).trim();
}

const isKind = (value: unknown): value is NoteKind => NOTE_KINDS.some((k) => k.kind === value);
const isTime = (value: unknown) => value === null || (typeof value === "number" && Number.isFinite(value));

/** A stored Transcript's notes, or `undefined` when they are not notes this page wrote. */
export function readNotes(value: unknown): Notes | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const n = value as Record<string, unknown>;
  const ok =
    isKind(n.kind) &&
    typeof n.text === "string" &&
    (n.writtenAs === null || isKind(n.writtenAs)) &&
    typeof n.covered === "number" &&
    isTime(n.writtenAt) &&
    isTime(n.fromMs);
  return ok ? (n as Notes) : undefined;
}
