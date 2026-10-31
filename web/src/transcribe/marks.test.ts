import { describe, expect, it } from "vitest";
import { changesBetween, MARK_MS, markedMarkdown, nextExpiry, standing, withChanges } from "./marks.ts";
import { type NoteLine, noNotes, type Notes, notesMarkdown } from "./notes.ts";

// Which lines of the running notes are marked, and until when (spec #428,
// "The notes view", stories 3 and 4).

const line = (id: string, text: string, section = "Now", author: NoteLine["author"] = "model"): NoteLine => ({ id, section, text, author });
const notes = (...lines: NoteLine[]): Notes => ({ ...noNotes("work"), lines, writtenAs: "work", nextId: 20 });
const T = 1_000_000;

describe("changesBetween", () => {
  it("finds the lines added, changed in text or section, and removed", () => {
    const before = notes(line("N1", "a"), line("N2", "b"), line("N3", "c"), line("N4", "d"));
    const after = notes(line("N1", "a"), line("N2", "b2"), line("N3", "c", "Decisions"), line("N5", "e"));
    const found = changesBetween(before, after);
    expect(found.map((c) => [c.kind, c.line.id])).toEqual([
      ["changed", "N2"],
      ["changed", "N3"],
      ["added", "N5"],
      ["removed", "N4"],
    ]);
    expect(found.find((c) => c.line.id === "N4")?.index).toBe(3);
  });

  it("finds nothing when the lines are as they were, and treats missing notes as empty", () => {
    const same = notes(line("N1", "a"));
    expect(changesBetween(same, same)).toEqual([]);
    expect(changesBetween(undefined, undefined)).toEqual([]);
    expect(changesBetween(undefined, same).map((c) => c.kind)).toEqual(["added"]);
  });

  it("leaves the user's own lines unmarked", () => {
    expect(changesBetween(notes(), notes(line("N1", "mine", "Now", "user")))).toEqual([]);
    expect(changesBetween(notes(line("N1", "mine", "Now", "user")), notes())).toEqual([]);
  });
});

describe("withChanges and standing", () => {
  const added = (id: string) => changesBetween(notes(), notes(line(id, "x")));

  it("marks each change of one update for three seconds", () => {
    const marks = withChanges([], [...added("N1"), ...changesBetween(notes(line("N2", "y")), notes())], T);
    expect(marks.map((m) => [m.id, m.kind, m.until])).toEqual([
      ["N1", "fresh", T + MARK_MS],
      ["N2", "gone", T + MARK_MS],
    ]);
    expect(standing(marks, T + MARK_MS - 1)).toHaveLength(2);
    expect(standing(marks, T + MARK_MS)).toHaveLength(0);
  });

  it("keeps a line changed twice marked until the last change's time is up", () => {
    const first = withChanges([], added("N1"), T);
    const twice = withChanges(first, changesBetween(notes(line("N1", "x")), notes(line("N1", "x2"))), T + 2_000);
    expect(twice).toHaveLength(1);
    expect(standing(twice, T + MARK_MS + 1)).toHaveLength(1);
    expect(standing(twice, T + 2_000 + MARK_MS)).toHaveLength(0);
  });

  it("lets a removal take over a line's mark, and drops spent marks", () => {
    const first = withChanges([], added("N1"), T);
    const removed = withChanges(first, changesBetween(notes(line("N1", "x")), notes()), T + 1_000);
    expect(removed.map((m) => m.kind)).toEqual(["gone"]);
    expect(withChanges(removed, added("N2"), T + 10_000).map((m) => m.id)).toEqual(["N2"]);
  });

  it("says when the next mark runs out", () => {
    const marks = withChanges(withChanges([], added("N1"), T), added("N2"), T + 1_000);
    expect(nextExpiry(marks, T)).toBe(T + MARK_MS);
    expect(nextExpiry(marks, T + MARK_MS)).toBe(T + 1_000 + MARK_MS);
    expect(nextExpiry(marks, T + 1_000 + MARK_MS)).toBeNull();
  });
});

describe("markedMarkdown", () => {
  const before = notes(line("N1", "a"), line("N2", "b"), line("N3", "c"));
  const after = notes(line("N1", "a"), line("N3", "c2"), line("N4", "d"));
  const marks = withChanges([], changesBetween(before, after), T);

  it("is the plain notes when no mark stands", () => {
    expect(markedMarkdown(after, [], T).markdown).toBe(notesMarkdown(after));
    expect(markedMarkdown(after, marks, T + MARK_MS)).toEqual({ markdown: notesMarkdown(after), items: {} });
    expect(markedMarkdown(undefined, [], T)).toEqual({ markdown: "", items: {} });
  });

  it("flags the fresh lines and puts a removed one back in its place, struck", () => {
    const { markdown, items } = markedMarkdown(after, marks, T + 1);
    const rows = markdown.split("\n");
    expect(rows.slice(0, 6)).toEqual(["## Now", "- a", "- ~~b~~", "- c2", "- d", ""]);
    expect(items).toEqual({ 3: "gone", 4: "fresh", 5: "fresh" });
  });

  it("shows a section that only a removed line keeps, and no dash while it stands", () => {
    const only = notes(line("N1", "a", "Decisions"));
    const m = withChanges([], changesBetween(only, notes()), T);
    const { markdown, items } = markedMarkdown(notes(), m, T);
    expect(markdown).toContain("## Decisions\n- ~~a~~");
    expect(Object.values(items)).toEqual(["gone"]);
  });
});
