import { canChat, imageLoaded, loadUnderWay, type ModelState } from "../api/model.ts";
import type { Settings } from "../api/request.ts";
import { hoursClocks } from "./export.ts";
import { dayAndTime, type Transcript } from "./transcript.ts";
import { viewerChoice } from "./viewerChoice.ts";

// Notes on a Transcript (spec speech/07, Q124-Q127; spec #428): the text
// model loaded keeps the notes a kind of conversation wants, as lines in
// sections. They come in two forms. The running notes are kept while the
// conversation goes on, in the kind's live sections, through a thread with
// the model that changes them a line at a time (running.ts, ADR 0076); they
// are updated on a press, or kept up while listening: about once a minute
// while something new was said. The recap is written once from the whole
// transcript, in the kind's sections, when a listening ends and on a press
// (recap.ts).

export type NoteKind = "general" | "work" | "game" | "lesson";

/** A section the model fills, and what it asks for. */
export type NoteSection = { name: string; ask: string };

/** A section of the running notes: what it asks for, and how many of the model's lines it holds at most. */
export type RunningSection = NoteSection & { cap: number };

/**
 * A kind of conversation: the sections its recap is written in (recap.ts adds to them), and the live sections its
 * running notes keep, five at most, each holding the current state or facts that add up (spec #428).
 */
export type NoteKindSpec = { kind: NoteKind; label: string; about: string; sections: NoteSection[]; running: RunningSection[] };

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
    running: [
      { name: "Now", cap: 1, ask: "what is being talked about right now, in one line" },
      { name: "People", cap: 20, ask: "one line a person who spoke or was named: who they are, as the conversation shows it. Add a person the first time they come up, never remove one" },
      { name: "Points", cap: 10, ask: "the points made, one a line, with who made them" },
      { name: "Asked of me", cap: 6, ask: "questions or requests put to Me and not answered yet; remove a line once Me has answered it" },
      { name: "Open questions", cap: 8, ask: "what was raised and is still open; remove a line once it is settled" },
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
    running: [
      { name: "Now", cap: 1, ask: "what is being discussed right now, in one line" },
      {
        name: "People",
        cap: 20,
        ask: "one line a person who spoke or was named: who they are, their role or stance as the conversation shows it. Add a person the first time they come up, complete their line as more is learned, never remove one",
      },
      { name: "Figures & dates", cap: 12, ask: "every figure said (an amount, a percentage, a quantity, a date, a deadline), one a line, with what it refers to" },
      { name: "Asked of me", cap: 6, ask: "questions or requests put to Me and not answered yet; remove a line once Me has answered it" },
      { name: "Commitments heard", cap: 10, ask: "who said they would do what, and by when if it was said" },
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
    running: [
      { name: "Now playing", cap: 1, ask: "the game being played and where it stands, in one line" },
      { name: "Players & sides", cap: 12, ask: "one line a player: their side, team or role" },
      { name: "Score", cap: 8, ask: "the score or the standings as they are now" },
      { name: "Moments", cap: 10, ask: "the moments that mattered, with their time" },
      { name: "Rules in question", cap: 6, ask: "rules asked about or disputed, and how they were settled" },
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
    running: [
      { name: "Now", cap: 1, ask: "the subject being taught right now, in one line" },
      { name: "Concepts", cap: 12, ask: "the concepts taught, one a line, with a line of explanation" },
      { name: "Terms", cap: 20, ask: "the terms introduced, one a line, with what they mean" },
      { name: "Questions", cap: 8, ask: "the questions asked, with the answer if one was given" },
      { name: "Homework & dates", cap: 8, ask: "homework, readings, exams and their dates" },
    ],
  },
];

export const specOf = (kind: NoteKind): NoteKindSpec => NOTE_KINDS.find((k) => k.kind === kind) ?? NOTE_KINDS[0];

/** What a kind's tile says under its name: its running notes' sections, the first (what is on now) left out of a long list, four at most. */
export function kindHint(spec: NoteKindSpec): string {
  const names = spec.running.map((s) => s.name).filter((_, i, all) => all.length <= 4 || i > 0);
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
export const noNotes = (kind: NoteKind): Notes => ({ kind, lines: [], removed: [], nextId: 1, writtenAs: null, covered: 0, writtenAt: null, fromMs: null });

/** Who wrote a line of the notes: the model, or the user by hand. */
export type NoteAuthor = "model" | "user";

/** A line of the notes: its id (`N1`, `N2`, never given twice), its section, its text, and who wrote it. */
export type NoteLine = { id: string; section: string; text: string; author: NoteAuthor };

/** A line as the model's answer gives it, before it has an id. */
export type DraftLine = Pick<NoteLine, "section" | "text">;

/** A Transcript's notes, as its document keeps them. */
export type Notes = {
  /** The kind picked for this Transcript: what the next writing asks for. */
  kind: NoteKind;
  /** The lines, each section's in order; none before any are written. */
  lines: NoteLine[];
  /** The lines the user removed, as they were: they never come back. */
  removed: NoteLine[];
  /** The number the next line's id takes, so an id is never given twice. */
  nextId: number;
  /** The kind the lines were written as; `null` before any. */
  writtenAs: NoteKind | null;
  /** How many of the Transcript's utterances the lines were written from. */
  covered: number;
  /** When it was written, in epoch ms. */
  writtenAt: number | null;
  /** Where the transcript sent began, when its start did not fit the model's context; `null` when all of it went. */
  fromMs: number | null;
};

/** A writing in sections (the recap): what to write, its sections in order, its own rules ahead of the common ones. */
export function sectionsTask(opening: string, sections: NoteSection[], rules: string[] = []): string {
  return [
    `${opening}: these sections, in this order, each under a level-2 Markdown heading (\`## \`):`,
    sections.map((s) => `## ${s.name}\n${s.ask}.`).join("\n\n"),
    [
      "Rules:",
      ...rules,
      "- Keep each heading exactly as given. Write the text under them in the language the context says to write in.",
      "- Short bullet points. No preamble, no closing remark, nothing outside the sections.",
      "- A section with nothing in it yet is `—` alone.",
      "- Point at a moment by its time, as [mm:ss].",
      "- Write the notes as of the transcript's last line.",
    ].join("\n"),
  ].join("\n\n");
}

/** The tokens an answer of the notes thread may take. */
export const NOTES_MAX_TOKENS = 1_500;

export const NOTES_SETTINGS: Omit<Settings, "systemPrompt"> = {
  temperature: 0.3,
  topP: 0.9,
  maxTokens: NOTES_MAX_TOKENS,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "agent",
};

/**
 * The running notes' limits, beside each live section's cap (`RunningSection.cap`), as #429's replay measured them on
 * the live model.
 */
export const RUNNING_NOTES = {
  /** A thread whose last request's prompt passed this many tokens starts again from a snapshot of the notes. */
  restartTokens: 64_000,
  /** The requests one update makes at most; what is left goes to the next update. */
  maxRounds: 6,
  /** Rounds in a row that change nothing before an update ends. */
  idleRounds: 2,
  /** The longest line, in characters: a longer one is refused. */
  lineChars: 240,
};

/** Keeping up: the first notes once this many utterances are in. */
export const FIRST_LIVE_UTTERANCES = 3;
/** Keeping up: the least time from one writing's start to the next. */
export const LIVE_GAP_MS = 60_000;

/**
 * Whether a listening that keeps the notes up writes them now. `sinceLastMs`: since this Transcript's last writing
 * began, `null` if none did. The listening's end writes the recap instead (recap.ts).
 */
export function dueForUpdate({ notes, utterances, sinceLastMs }: { notes: Notes | undefined; utterances: number; sinceLastMs: number | null }): boolean {
  const covered = notes?.writtenAs ? notes.covered : 0;
  const fresh = utterances - covered;
  if (notes?.writtenAs && notes.writtenAs !== notes.kind) return utterances > 0;
  if (fresh <= 0) return false;
  if (covered === 0) return fresh >= FIRST_LIVE_UTTERANCES;
  return sinceLastMs === null || sinceLastMs >= LIVE_GAP_MS;
}

/**
 * The line under the running notes: when they were last updated and, while they are not kept up (`live` off), how many
 * utterances came since; `null` before any.
 */
export function notesStatus(notes: Notes | undefined, utterances: number, live: boolean): string | null {
  if (!notes?.writtenAs || notes.writtenAt === null) return null;
  const time = dayAndTime(new Date(notes.writtenAt)).time;
  if (notes.writtenAs !== notes.kind) {
    return `Written as ${specOf(notes.writtenAs).label} notes at ${time}. Write notes to redo them as ${specOf(notes.kind).label} notes.`;
  }
  const fresh = Math.max(0, utterances - notes.covered);
  return `Updated at ${time}.${!live && fresh > 0 ? ` ${fresh} more since.` : ""}`;
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

/** The section of a line the model wrote under no heading. */
export const OTHER_SECTION = "Other";

const SECTION_HEADING = /^##[ \t]+(.+?)[ \t#]*$/;
const OTHER_HEADING = /^#{1,6}[ \t]+(.+?)[ \t#]*$/;
const BULLET = /^[ \t]*(?:[-*+]|\d+[.)])[ \t]+(.*)$/;
const RULE = /^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
/** What a section with nothing in it holds (sectionsTask). */
const NOTHING = /^[—–-]?$/;

/**
 * Notes an older page kept as the model's Markdown, as lines: a line per bullet or paragraph, in the section of the
 * level-2 heading above it, a heading of the kind's written as the kind spells it. A section's `—` is no line; what
 * sits under no heading goes under Other.
 */
export const parseNotes = (answer: string, kind: NoteKind): DraftLine[] => parseLines(answer, specOf(kind).sections.map((s) => s.name));

/** A writing's answer as lines (parseNotes), a heading among `names` written as `names` spells it. */
export function parseLines(answer: string, names: string[]): DraftLine[] {
  const lines: DraftLine[] = [];
  let section = OTHER_SECTION;
  let open: string[] | null = null;
  const close = () => {
    const text = open?.join(" ").trim() ?? "";
    if (!NOTHING.test(text)) lines.push({ section, text });
    open = null;
  };
  for (const row of tidyNotes(answer.replace(/\r\n?/g, "\n")).split("\n")) {
    const heading = SECTION_HEADING.exec(row);
    const other = OTHER_HEADING.exec(row);
    const bullet = BULLET.exec(row);
    if (heading) {
      close();
      const name = heading[1].trim();
      section = names.find((n) => n.toLocaleLowerCase() === name.toLocaleLowerCase()) ?? name;
    } else if (row.trim() === "" || RULE.test(row)) {
      close();
    } else if (other) {
      // Any other heading is a line of its own.
      close();
      open = [other[1].trim()];
      close();
    } else if (bullet) {
      close();
      open = [bullet[1].trim()];
    } else {
      (open ??= []).push(row.trim());
    }
  }
  close();
  return lines;
}

/** What a finished writing puts in a Transcript's notes: its answer's lines, the kind it was written as, and from what. */
export type Written = { lines: DraftLine[]; writtenAs: NoteKind; covered: number; writtenAt: number; fromMs: number | null };

/**
 * `notes` once a writing lands: its lines in place of the model's, a line written again word for word in its section
 * keeping its id, a new one the next id. The user's lines and the lines they removed stay as they are, and a line of
 * the writing that is one of them, word for word in its section, is left out (#436).
 */
export function withWriting(notes: Notes | undefined, { lines: drafts, ...written }: Written): Notes {
  const base = notes ?? noNotes(written.writtenAs);
  const before = base.lines.filter((l) => l.author === "model");
  const theirs = [...base.lines.filter((l) => l.author === "user"), ...base.removed];
  const isTheirs = (d: DraftLine) => theirs.some((l) => l.section === d.section && l.text.toLocaleLowerCase() === d.text.toLocaleLowerCase());
  let nextId = base.nextId;
  const lines = drafts.filter((d) => !isTheirs(d)).map((d): NoteLine => {
    const same = before.findIndex((l) => l.section === d.section && l.text === d.text);
    const id = same >= 0 ? before.splice(same, 1)[0].id : `N${nextId++}`;
    return { id, ...d, author: "model" };
  });
  return { ...base, ...written, lines: [...lines, ...base.lines.filter((l) => l.author === "user")], nextId };
}

/** The sections notes are shown in: `named` in order, then any other a line sits in, Other last. */
export function sectionOrder(named: string[], lines: NoteLine[]): string[] {
  const sections = [...new Set([...named, ...lines.map((l) => l.section).filter((s) => s !== OTHER_SECTION)])];
  return lines.some((l) => l.section === OTHER_SECTION) && !sections.includes(OTHER_SECTION) ? [...sections, OTHER_SECTION] : sections;
}

/**
 * The notes as Markdown, what the tab shows and Copy notes gives: the live sections of the kind they were written as,
 * in order, `—` under one with no line, then any other section a line sits in, Other last; nothing says who wrote a
 * line, and a time past the hour reads `h:mm:ss`. "" before any are written. `named`: the sections in order, when
 * they are not the running notes' (the recap's).
 */
export function notesMarkdown(notes: Notes | undefined, named?: NoteSection[]): string {
  if (!notes || (notes.writtenAs === null && notes.lines.length === 0)) return "";
  const kinds = (named ?? specOf(notes.writtenAs ?? notes.kind).running).map((s) => s.name);
  return sectionOrder(kinds, notes.lines)
    .map((section) => {
      const lines = notes.lines.filter((l) => l.section === section).map((l) => `- ${hoursClocks(l.text)}`);
      return `## ${section}\n${lines.length > 0 ? lines.join("\n") : "—"}`;
    })
    .join("\n\n");
}

const isKind = (value: unknown): value is NoteKind => NOTE_KINDS.some((k) => k.kind === value);
const isTime = (value: unknown) => value === null || (typeof value === "number" && Number.isFinite(value));

const LINE_ID = /^N([1-9]\d*)$/;

function readLine(value: unknown): NoteLine | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const { id, section, text, author } = value as Record<string, unknown>;
  const ok = typeof id === "string" && LINE_ID.test(id) && typeof section === "string" && section !== "" && typeof text === "string" && text.trim() !== "";
  return ok && (author === "model" || author === "user") ? { id, section, text, author } : undefined;
}

function readLines(value: unknown): NoteLine[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const lines = value.map(readLine);
  return lines.every((l) => l !== undefined) ? lines : undefined;
}

/**
 * A stored Transcript's notes, or `undefined` when they are not notes this page wrote. Notes an older page kept as the
 * model's Markdown become lines here, all of them the model's.
 */
export function readNotes(value: unknown): Notes | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const n = value as Record<string, unknown>;
  const ok =
    isKind(n.kind) &&
    (n.writtenAs === null || isKind(n.writtenAs)) &&
    typeof n.covered === "number" &&
    isTime(n.writtenAt) &&
    isTime(n.fromMs);
  if (!ok) return undefined;
  const kept = { kind: n.kind, writtenAs: n.writtenAs, covered: n.covered, writtenAt: n.writtenAt, fromMs: n.fromMs } as Omit<Notes, "lines" | "removed" | "nextId">;
  if (n.lines === undefined && typeof n.text === "string") {
    const lines = parseNotes(n.text, kept.writtenAs ?? kept.kind).map((d, i): NoteLine => ({ id: `N${i + 1}`, ...d, author: "model" }));
    return { ...kept, lines, removed: [], nextId: lines.length + 1 };
  }
  const lines = readLines(n.lines);
  const removed = readLines(n.removed);
  const nextId = n.nextId;
  if (!lines || !removed || typeof nextId !== "number" || !Number.isInteger(nextId) || nextId < 1) return undefined;
  // Every id once, and below the next: an id given twice would point at two lines.
  const ids = [...lines, ...removed].map((l) => Number(LINE_ID.exec(l.id)![1]));
  if (new Set(ids).size !== ids.length || ids.some((id) => id >= nextId)) return undefined;
  return { ...kept, lines, removed, nextId };
}
