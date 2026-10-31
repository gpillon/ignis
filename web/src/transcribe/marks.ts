import { type NoteLine, type Notes, sectionOrder, specOf } from "./notes.ts";

// What an update changed in the running notes, shown for a moment (spec #428,
// "The notes view"): a line added or changed carries the ember mark and a
// removed one stays struck and dimmed, each for about three seconds. The rule
// is here and pure; Notes.tsx keeps the marks and draws them. What the user
// does by hand is never marked: they saw it happen (#436).

/** How long a mark stays, in ms. */
export const MARK_MS = 3_000;

/** One line that differs between two states of the notes. */
export type NoteChange = { kind: "added" | "changed" | "removed"; line: NoteLine; index: number };

/** A line marked until `until` (epoch ms): fresh when added or changed, gone when removed (`line` as it was, `index` its place in its section). */
export type NoteMark = { id: string; kind: "fresh" | "gone"; until: number; line: NoteLine; index: number };

const indexIn = (notes: Notes | undefined, line: NoteLine) => (notes?.lines ?? []).filter((l) => l.section === line.section).findIndex((l) => l.id === line.id);

/**
 * The model's lines that were added, changed (text or section) or removed going from `before` to `after`; the user's own
 * lines are not marked, nor a line of the model's the user made theirs or removed.
 */
export function changesBetween(before: Notes | undefined, after: Notes | undefined): NoteChange[] {
  const was = new Map((before?.lines ?? []).map((l) => [l.id, l]));
  const is = new Map((after?.lines ?? []).map((l) => [l.id, l]));
  const theirs = new Set((after?.removed ?? []).map((l) => l.id));
  const found: NoteChange[] = [];
  for (const line of is.values()) {
    const old = was.get(line.id);
    if (line.author !== "model") continue;
    if (!old) found.push({ kind: "added", line, index: indexIn(after, line) });
    else if (old.text !== line.text || old.section !== line.section) found.push({ kind: "changed", line, index: indexIn(after, line) });
  }
  for (const line of was.values()) if (line.author === "model" && !is.has(line.id) && !theirs.has(line.id)) found.push({ kind: "removed", line, index: indexIn(before, line) });
  return found;
}

/** The marks after `changes` at `at`: each line's latest change counts, so a line changed twice keeps its mark until the last one's time is up; the spent ones go. */
export function withChanges(marks: NoteMark[], changes: NoteChange[], at: number): NoteMark[] {
  const hit = new Set(changes.map((c) => c.line.id));
  const kept = marks.filter((m) => m.until > at && !hit.has(m.id));
  return [...kept, ...changes.map((c): NoteMark => ({ id: c.line.id, kind: c.kind === "removed" ? "gone" : "fresh", until: at + MARK_MS, line: c.line, index: c.index }))];
}

/** The marks still standing at `now`. */
export const standing = (marks: NoteMark[], now: number): NoteMark[] => marks.filter((m) => m.until > now);

/** When the next standing mark runs out, or `null` when none stands. */
export const nextExpiry = (marks: NoteMark[], now: number): number | null => {
  const left = standing(marks, now).map((m) => m.until);
  return left.length === 0 ? null : Math.min(...left);
};

/** A line as the view draws it: one of the notes, fresh while its mark stands, or one removed, struck while its mark stands. */
export type NoteRow = { line: NoteLine; mark: "fresh" | "gone" | null };

/** A section as the view draws it: its name, and its rows in order. */
export type DrawnSection = { name: string; rows: NoteRow[] };

/**
 * The notes as the view draws them, in notes.ts's `notesMarkdown` order: `named` (the live sections of the kind the
 * notes were written as, unless given), then any other a line sits in, Other last; each line with its mark, a line of
 * the user's never marked, and each removed line still standing back where it was, or last when its neighbours have
 * moved on. None before any notes are written.
 */
export function markedSections(notes: Notes | undefined, marks: NoteMark[], now: number, named?: string[]): DrawnSection[] {
  const live = standing(marks, now);
  if (!notes || (notes.writtenAs === null && notes.lines.length === 0 && live.length === 0)) return [];
  const fresh = new Set(live.filter((m) => m.kind === "fresh").map((m) => m.id));
  const gone = live.filter((m) => m.kind === "gone" && !notes.lines.some((l) => l.id === m.id));
  const names = named ?? specOf(notes.writtenAs ?? notes.kind).running.map((s) => s.name);
  return sectionOrder(names, [...notes.lines, ...gone.map((m) => m.line)]).map((name) => {
    const rows = notes.lines.filter((l) => l.section === name).map((line): NoteRow => ({ line, mark: line.author === "model" && fresh.has(line.id) ? "fresh" : null }));
    for (const m of gone.filter((g) => g.line.section === name).sort((a, b) => a.index - b.index)) rows.splice(Math.min(m.index, rows.length), 0, { line: m.line, mark: "gone" });
    return { name, rows };
  });
}
