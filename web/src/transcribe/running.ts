import type { ToolDefinition } from "../api/request.ts";
import type { ToolCall } from "../api/sse.ts";
import { minutesClock, toText } from "./export.ts";
import { type NoteKind, type NoteLine, noNotes, type Notes, OTHER_SECTION, RUNNING_NOTES, type RunningSection, sectionOrder, specOf } from "./notes.ts";
import type { Transcript, Utterance } from "./transcript.ts";

// The running notes (spec #428, "Running notes: the notes thread"; ADR 0076):
// the notes kept while a conversation goes on, in the kind's live sections, by
// one append-only thread with the text model per Transcript. The model changes
// them through four tools; the page applies each call to the lines and answers
// it in one line, whatever the model does: an id it copies into a text is
// stripped, a duplicate or overlong line is refused, a full section takes a
// rewrite and nothing smaller, and the user's own lines are never the model's
// to change. The shape, the prompt and the guards are the ones #429's replay
// measured on the live model (scripts/notes-replay/replay.py).

/** A kind's live sections, in order. */
export const runningSections = (kind: NoteKind): RunningSection[] => specOf(kind).running;

const counted = (n: number) => (n === 1 ? "1 line" : `${n} lines`);

/** The running task: the kind's live sections, each with its cap and what it asks for, and how the tools keep them. */
export function runningTask(kind: NoteKind): string {
  const sections = runningSections(kind)
    .map((s) => `## ${s.name} (${counted(s.cap)} at most)\n${s.ask}.`)
    .join("\n\n");
  const rules = [
    "Rules:",
    "- The notes live outside this conversation, one line at a time, and change only through the tools: add_line adds a line to a section, replace_line rewrites a line by its id, remove_line removes one, rewrite_section rewrites a whole section. A tool's result says what it did.",
    "- Each user turn brings what was said since the last one and, under Changed since, what the user changed in the transcript meanwhile: an utterance corrected, moved to another speaker, split, deleted or put back, a speaker renamed or merged into another. Change only what it makes wrong, stale or incomplete; a speaker goes by the name they have now. A line still true stays as it is, word for word. When nothing changes, call no tool.",
    "- Keep the notes short: a section holds at most the lines given above. Change single lines; only when a section is full, rewrite it with rewrite_section: merge, drop what is stale, fold in what is new; a line kept word for word keeps its id.",
    "- A line is one short point; a figure gets a line of its own. Never write a line's id in its text.",
    "- A line marked (the user's) is one the user wrote: never change or remove it; it does not count among its section's lines. A line the user removed is never written again.",
    "- Write the lines in the language the context says to write in.",
    "- Point at a moment by its time, as [mm:ss], in minutes from the start: [64:09] is sixty-four minutes in.",
    "- A short phrase repeated word for word, again and again, is speech-recognition noise: never note it.",
    "- Make every change the turn calls for, one tool call each: several calls in one answer are fine. When the results come back, make any change still missing, or answer with no call once the notes are right.",
    "- Answer only with tool calls, never with text.",
  ].join("\n");
  return `Keep the running notes of ${specOf(kind).about}: notes that help the user follow it and take part while it goes on. These sections, in this order:\n\n${sections}\n\n${rules}`;
}

/** The four tools, the same for every request of a kind's thread, so the engine reuses the prompt whole. */
export function runningTools(kind: NoteKind): ToolDefinition[] {
  const sections = runningSections(kind).map((s) => s.name);
  return [
    {
      type: "function",
      function: {
        name: "add_line",
        description: "Adds one line at the end of a section of the notes.",
        parameters: {
          type: "object",
          properties: { section: { type: "string", enum: sections }, text: { type: "string", description: "The line, without a bullet marker." } },
          required: ["section", "text"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "replace_line",
        description: "Rewrites one line of the notes, by its id.",
        parameters: { type: "object", properties: { id: { type: "string", description: "The line's id, such as N7." }, text: { type: "string" } }, required: ["id", "text"] },
      },
    },
    {
      type: "function",
      function: {
        name: "remove_line",
        description: "Removes one line of the notes, by its id.",
        parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
      },
    },
    {
      type: "function",
      function: {
        name: "rewrite_section",
        description:
          "Rewrites a whole section of the notes, only once it is full: these lines, at most the section's limit, take the place of the ones it has. A line kept word for word keeps its id.",
        parameters: {
          type: "object",
          properties: { section: { type: "string", enum: sections }, lines: { type: "array", items: { type: "string" } } },
          required: ["section", "lines"],
        },
      },
    },
  ];
}

/** What one call came to: the notes after it, whether it changed them, and the one line the model is answered with. */
export type Applied = { notes: Notes; outcome: "applied" | "unchanged" | "refused"; result: string };

const same = (a: string, b: string) => a.toLocaleLowerCase() === b.toLocaleLowerCase();
const inSection = (notes: Notes, section: string) => notes.lines.filter((l) => l.section === section);
const theModels = (notes: Notes, section: string) => inSection(notes, section).filter((l) => l.author === "model");
const removedFrom = (notes: Notes, section: string, text: string) => notes.removed.some((l) => l.section === section && same(l.text, text));

/** `text` trimmed, without the ids of the notes' lines the model copied into it: at its start, or in brackets. */
function stripIds(notes: Notes, text: string): string {
  const trimmed = text.trim();
  const ids = notes.lines.map((l) => l.id).join("|");
  if (ids === "") return trimmed;
  return trimmed
    .replace(new RegExp(`^(?:[([]?(?:${ids})[)\\]]?[:.)]?\\s+)+`), "")
    .replace(new RegExp(`\\s*[([](?:${ids})[)\\]]`, "g"), "")
    .trim();
}

/** Why `text` cannot stand in `section` (`but` aside), or `null`: empty, too long, there already, or removed by the user. */
function refusal(notes: Notes, section: string, text: string, but?: string): string | null {
  const length = [...text].length;
  if (length === 0) return "Refused: the text is empty; remove_line removes a line.";
  if (length > RUNNING_NOTES.lineChars) return `Refused: the line is ${length} characters; a line holds ${RUNNING_NOTES.lineChars} at most. Write one short point.`;
  if (inSection(notes, section).some((l) => l.id !== but && same(l.text, text))) return `Refused: ${section} already has that line.`;
  if (removedFrom(notes, section, text)) return `Refused: the user removed that line from ${section}; never write it again.`;
  return null;
}

/**
 * One call of the model applied to the running notes of `kind`: the notes it leaves and its one-line result. A refused
 * or unchanging call leaves the notes as they were. A line's id is never given twice.
 */
export function applyCall(notes: Notes, kind: NoteKind, call: ToolCall): Applied {
  const refused = (result: string): Applied => ({ notes, outcome: "refused", result });
  const unchanged = (result: string): Applied => ({ notes, outcome: "unchanged", result });
  let args: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(call.arguments || "{}");
    args = typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : {};
  } catch {
    return refused("Refused: the arguments are not JSON.");
  }
  const textOf = (value: unknown) => stripIds(notes, typeof value === "string" ? value : "");
  const sections = runningSections(kind);
  const section = typeof args.section === "string" ? sections.find((s) => same(s.name, args.section as string)) : undefined;
  const noSection = () => refused(`Refused: no section named '${String(args.section)}'. Sections: ${sections.map((s) => s.name).join(", ")}.`);

  switch (call.name) {
    case "add_line": {
      if (!section) return noSection();
      const text = textOf(args.text);
      const bad = refusal(notes, section.name, text);
      if (bad) return refused(bad);
      if (theModels(notes, section.name).length >= section.cap) {
        return refused(`Refused: ${section.name} is full (${counted(section.cap)}). Rewrite it with rewrite_section, merging what belongs together and folding this in.`);
      }
      const id = `N${notes.nextId}`;
      const line: NoteLine = { id, section: section.name, text, author: "model" };
      return { notes: { ...notes, lines: [...notes.lines, line], nextId: notes.nextId + 1 }, outcome: "applied", result: `Added ${id} to ${section.name}.` };
    }
    case "replace_line":
    case "remove_line": {
      const line = notes.lines.find((l) => l.id === args.id);
      if (!line) return refused(`Refused: no line '${String(args.id)}'.`);
      if (line.author === "user") return refused(`Refused: ${line.id} is the user's: leave it as it is.`);
      if (call.name === "remove_line") return { notes: { ...notes, lines: notes.lines.filter((l) => l !== line) }, outcome: "applied", result: `Removed ${line.id}.` };
      const text = textOf(args.text);
      if (text === line.text) return unchanged(`Unchanged: ${line.id} already reads so. When the notes are right, answer with no call.`);
      const bad = refusal(notes, line.section, text, line.id);
      if (bad) return refused(bad);
      return { notes: { ...notes, lines: notes.lines.map((l) => (l === line ? { ...l, text } : l)) }, outcome: "applied", result: `${line.id} updated.` };
    }
    case "rewrite_section": {
      if (!section) return noSection();
      if (!Array.isArray(args.lines)) return refused("Refused: lines is the list of the section's new lines.");
      const theirs = inSection(notes, section.name).filter((l) => l.author === "user");
      // What cannot stand is left out, as an add of it would be refused: an empty or overlong line, one given twice,
      // one the user wrote or removed.
      const texts: string[] = [];
      for (const raw of args.lines) {
        const text = textOf(raw);
        const fits = text !== "" && [...text].length <= RUNNING_NOTES.lineChars;
        if (fits && !texts.some((t) => same(t, text)) && !theirs.some((l) => same(l.text, text)) && !removedFrom(notes, section.name, text)) texts.push(text);
      }
      if (texts.length > section.cap) return refused(`Refused: ${texts.length} lines; ${section.name} holds ${section.cap} at most.`);
      const old = theModels(notes, section.name);
      if (old.length < section.cap) {
        return refused(`Refused: ${section.name} has ${counted(old.length)}, not ${section.cap}: change single lines. rewrite_section is for a full section.`);
      }
      let nextId = notes.nextId;
      const kept = new Set<string>();
      const written = texts.map((text): NoteLine => {
        const was = old.find((l) => l.text === text && !kept.has(l.id));
        if (was) {
          kept.add(was.id);
          return was;
        }
        return { id: `N${nextId++}`, section: section.name, text, author: "model" };
      });
      if (kept.size === old.length && written.length === old.length) return unchanged(`Unchanged: ${section.name} already reads so. When the notes are right, answer with no call.`);
      const lines = [...notes.lines.filter((l) => !old.includes(l)), ...written];
      return {
        notes: { ...notes, lines, nextId },
        outcome: "applied",
        result: `${section.name} rewritten: ${kept.size} kept, ${written.length - kept.size} new, ${old.length - kept.size} removed.`,
      };
    }
    default:
      return refused(`Refused: no tool '${call.name}'.`);
  }
}

/**
 * The notes as a thread starting from them is told: empty, or each line with its id under its section, the user's
 * marked as theirs; then the lines the user removed.
 */
export function notesState(notes: Notes, kind: NoteKind): string {
  const parts: string[] = [];
  if (notes.lines.length === 0) parts.push("The notes are empty.");
  else {
    const sections = sectionOrder(
      runningSections(kind).map((s) => s.name),
      notes.lines,
    );
    const shown = sections.map((section) => {
      const lines = inSection(notes, section).map((l) => `${l.id} ${l.author === "user" ? "(the user's) " : ""}${l.text}`);
      return `${section}:\n${lines.length > 0 ? lines.join("\n") : "(empty)"}`;
    });
    parts.push("The notes so far, each line with its id:", shown.join("\n\n"));
  }
  if (notes.removed.length > 0) parts.push(`Lines the user removed, never to be written again:\n${notes.removed.map((l) => `- ${l.section}: ${l.text}`).join("\n")}`);
  return parts.join("\n\n");
}

/** What the model is told of utterances heard: each one's line, from its start. */
export type Said = { startMs: number; line: string };

/**
 * What a thread told the model of an utterance: its start, its speaker and the name it had, the text, and its line
 * without the line break, "" while the model knows of no line (its text emptied, or the utterance deleted).
 */
export type Told = { startMs: number; speaker: number | null; who: string | null; text: string; line: string };

/** An utterance of `t`'s line as the thread tells it, in total minutes: "" for an emptied one. */
const lineOf = (t: Transcript, u: Utterance): string => toText({ ...t, utterances: [u] }, minutesClock);

/**
 * Whether `u` is the utterance the thread told as `was`: the same id from the same start; not an id the page gave again
 * once the last utterance was deleted.
 */
const isTold = (u: Utterance | undefined, was: Told | undefined): boolean => u !== undefined && was !== undefined && was.startMs === u.startMs;

/** `t`'s utterances by id. */
const byIdOf = (t: Transcript) => new Map(t.utterances.map((u) => [u.id, u]));

/**
 * What a thread tells the model of `t`'s utterances, by id, starting from it or appending a turn of it: each utterance
 * as it stands, and, from what it told `before`, each one since deleted as no line, so a deletion is told once and an
 * Undo after it as put back.
 */
export function toldOf(t: Transcript, before?: Map<number, Told>): Map<number, Told> {
  const names = new Map(t.speakers.map((s) => [s.id, s.name]));
  const byId = byIdOf(t);
  const told = new Map<number, Told>();
  for (const [id, was] of before ?? []) if (!isTold(byId.get(id), was)) told.set(id, { ...was, line: "" });
  for (const u of t.utterances) {
    const who = u.speaker === null ? null : (names.get(u.speaker) ?? null);
    told.set(u.id, { startMs: u.startMs, speaker: u.speaker, who, text: u.text.trim(), line: lineOf(t, u).trimEnd() });
  }
  return told;
}

/**
 * What a later turn tells of `t`, against what the thread `told` before: the utterances said since, and one event a
 * change the user made meanwhile (#434), each with its time and its new text: first a speaker renamed or merged into
 * another, by the name the model knows them by; then, in the order of their times, an utterance corrected (an edit, an
 * accepted fix, a Rewrite), moved to another speaker, split, deleted or put back. A rename is the model's to carry into
 * the notes' lines: the page never puts a name in one.
 */
export function changesSince(told: Map<number, Told>, t: Transcript): { said: Said[]; events: string[] } {
  const byId = byIdOf(t);
  // The speakers the model knows, by the name it knows and the first time it heard them; one gone from the speakers
  // was merged into the one its utterances are now.
  const known = new Map<number, { who: string; at: number; now: Map<number, number> }>();
  for (const [id, was] of told) {
    if (was.line === "" || was.speaker === null || was.who === null) continue;
    const speaker = known.get(was.speaker) ?? { who: was.who, at: was.startMs, now: new Map<number, number>() };
    speaker.at = Math.min(speaker.at, was.startMs);
    const u = byId.get(id);
    if (u?.speaker != null && isTold(u, was)) speaker.now.set(u.speaker, (speaker.now.get(u.speaker) ?? 0) + 1);
    known.set(was.speaker, speaker);
  }
  const names = new Map(t.speakers.map((s) => [s.id, s.name]));
  const events: string[] = [];
  const mergedInto = new Map<number, number>();
  for (const [id, { who, at, now }] of known) {
    const name = names.get(id);
    const heard = `${who}, heard from [${minutesClock(at)}],`;
    if (name !== undefined) {
      if (name !== who) events.push(`Renamed: ${heard} is ${name}.`);
      continue;
    }
    const into = [...now].sort((a, b) => b[1] - a[1])[0]?.[0];
    if (into === undefined || !names.has(into)) continue;
    mergedInto.set(id, into);
    events.push(`Merged: ${heard} is ${names.get(into)}.`);
  }
  // A split keeps its first half's id and puts the rest right after it, under new ids, each start guessed (`~`): the
  // utterances so placed after one told are its halves, not something said.
  const halves = new Map<number, Utterance[]>();
  const halved = new Set<Utterance>();
  let split: Utterance | null = null;
  for (const u of t.utterances) {
    if (isTold(u, told.get(u.id))) split = u;
    else if (split && u.approxStart) {
      halves.set(split.id, [...(halves.get(split.id) ?? []), u]);
      halved.add(u);
    } else split = null;
  }
  const said: Said[] = [];
  const changes: { at: number; event: string }[] = [];
  const change = (at: number, event: string) => changes.push({ at, event });
  for (const u of t.utterances) {
    const was = told.get(u.id);
    const line = lineOf(t, u);
    if (was === undefined || !isTold(u, was)) {
      if (line !== "" && !halved.has(u)) said.push({ startMs: u.startMs, line });
      continue;
    }
    const rest = halves.get(u.id) ?? [];
    if (rest.length > 0) {
      const lines = [u, ...rest].map((x) => lineOf(t, x).trimEnd()).filter((l) => l !== "");
      change(u.startMs, `Split: ${lines.join(" | ")}`);
      continue;
    }
    const now = line.trimEnd();
    if (was.line === "" || now === "") {
      // Emptied is deleted, as far as the model knows: it is told no line of it.
      if (now !== was.line) change(u.startMs, now === "" ? `Deleted: ${was.line}` : `Put back: ${now}`);
      continue;
    }
    const moved = u.speaker !== was.speaker && (was.speaker === null || mergedInto.get(was.speaker) !== u.speaker);
    if (u.text.trim() !== was.text) change(u.startMs, `Corrected: ${now} It was: ${moved && was.who !== null ? `${was.who}: ` : ""}${was.text}`);
    else if (moved) change(u.startMs, `Moved: ${now}${was.who === null ? "" : ` It was ${was.who}'s.`}`);
  }
  for (const [id, was] of told) if (was.line !== "" && !isTold(byId.get(id), was)) change(was.startMs, `Deleted: ${was.line}`);
  changes.sort((a, b) => a.at - b.at);
  return { said, events: [...events, ...changes.map((c) => c.event)] };
}

/**
 * A later turn of the thread: what was said since the last one, then what changed meanwhile, one event a line (the
 * transcript's changes, changesSince's; the user's edits to the notes). "" with neither.
 */
export function laterTurn(said: Said[], events: string[] = []): string {
  const parts: string[] = [];
  if (said.length > 0) parts.push(`Said since [${minutesClock(said[0].startMs)}]:\n${said.map((s) => s.line).join("")}`);
  if (events.length > 0) parts.push(`Changed since:\n${events.map((e) => `- ${e}`).join("\n")}`);
  return parts.join("\n");
}

/**
 * The notes as a restart in `kind` from scratch leaves them (a kind change, Rewrite from scratch): no line of the
 * model's, the user's each under the live section of the same name, else under a closing Other, and the lines they
 * removed still listed.
 */
export function fromScratch(notes: Notes | undefined, kind: NoteKind): Notes {
  const base = notes ?? noNotes(kind);
  const sections = runningSections(kind);
  const lines = base.lines
    .filter((l) => l.author === "user")
    .map((l) => ({ ...l, section: sections.find((s) => same(s.name, l.section))?.name ?? OTHER_SECTION }));
  return { ...base, kind, lines, writtenAs: null, covered: 0, writtenAt: null, fromMs: null };
}

/**
 * The prompt past which a thread starts again from a snapshot: the running notes' threshold, or half the model's
 * context when that is smaller, so a thread never outgrows it.
 */
export const restartAt = (contextLimit: number | null): number =>
  Math.min(RUNNING_NOTES.restartTokens, contextLimit === null ? Infinity : Math.floor(contextLimit / 2));
