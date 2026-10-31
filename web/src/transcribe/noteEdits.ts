import type { NoteLine, Notes } from "./notes.ts";

// The notes edited by hand (spec #428, "Editing by hand"): in the running
// notes and in the recap, a line the user edits or adds is theirs, and the
// model never touches it again; a line they remove goes to the removed lines,
// as it was, and never comes back. Each edit names the line as the user saw
// it, so it wins over what the model did to that line meanwhile.

/** A line as typed, on one line. */
const tidy = (text: string) => text.replace(/\s+/g, " ").trim();

/** `notes` without `line` (as the user saw it), kept among the removed lines as it stands, or as seen when it is gone already. */
export function removeLine(notes: Notes, line: NoteLine): Notes {
  if (notes.removed.some((l) => l.id === line.id)) return notes;
  const current = notes.lines.find((l) => l.id === line.id);
  return { ...notes, lines: notes.lines.filter((l) => l.id !== line.id), removed: [...notes.removed, current ?? line] };
}

/**
 * `notes` with `line` (as the user saw it) reading `text`, now the user's: in its place, or at its section's end when
 * the model removed it meanwhile, under the id it had. Emptied, it is removed; the text the user saw changes nothing.
 */
export function editLine(notes: Notes, line: NoteLine, text: string): Notes {
  const typed = tidy(text);
  if (typed === "") return removeLine(notes, line);
  if (typed === line.text || notes.removed.some((l) => l.id === line.id)) return notes;
  const theirs = (l: NoteLine): NoteLine => ({ ...l, text: typed, author: "user" });
  if (!notes.lines.some((l) => l.id === line.id)) return { ...notes, lines: [...notes.lines, theirs(line)] };
  return { ...notes, lines: notes.lines.map((l) => (l.id === line.id ? theirs(l) : l)) };
}

/** `notes` with the user's line `text` at the end of `section`, under the next id; an empty one adds nothing. */
export function addLine(notes: Notes, section: string, text: string): Notes {
  const typed = tidy(text);
  if (typed === "") return notes;
  return { ...notes, lines: [...notes.lines, { id: `N${notes.nextId}`, section, text: typed, author: "user" }], nextId: notes.nextId + 1 };
}
