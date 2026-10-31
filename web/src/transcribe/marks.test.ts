import { describe, expect, it } from "vitest";
import { changesBetween, type DrawnSection, MARK_MS, markedSections, nextExpiry, standing, withChanges } from "./marks.ts";
import { type NoteLine, noNotes, type Notes } from "./notes.ts";

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

  it("leaves unmarked what the user did by hand: a line of the model's they made theirs, or removed", () => {
    const before = notes(line("N1", "a"), line("N2", "b"));
    const after = { ...notes(line("N2", "b2", "Now", "user")), removed: [line("N1", "a")] };
    expect(changesBetween(before, after)).toEqual([]);
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

describe("markedSections", () => {
  const before = notes(line("N1", "a"), line("N2", "b"), line("N3", "c"));
  const after = notes(line("N1", "a"), line("N3", "c2"), line("N4", "d"));
  const marks = withChanges([], changesBetween(before, after), T);
  /** Each section's name and its lines' texts, a mark before a marked one. */
  const drawn = (sections: DrawnSection[]) => sections.map((s) => [s.name, ...s.rows.map((r) => (r.mark ? `${r.mark} ${r.line.text}` : r.line.text))]);

  it("are the notes in the kind's live sections, others after, nothing marked when no mark stands", () => {
    const empty = [["People"], ["Figures & dates"], ["Asked of me"], ["Commitments heard"]];
    expect(drawn(markedSections(after, [], T))).toEqual([["Now", "a", "c2", "d"], ...empty]);
    expect(markedSections(after, marks, T + MARK_MS)).toEqual(markedSections(after, [], T));
    expect(drawn(markedSections(notes(line("N1", "e", "Other"), line("N2", "f", "Decisions")), [], T))).toEqual([["Now"], ...empty, ["Decisions", "f"], ["Other", "e"]]);
    expect(markedSections(undefined, [], T)).toEqual([]);
    expect(markedSections(noNotes("work"), [], T)).toEqual([]);
  });

  it("marks the fresh lines and puts a removed one back in its place, struck", () => {
    expect(drawn(markedSections(after, marks, T + 1))[0]).toEqual(["Now", "a", "gone b", "fresh c2", "fresh d"]);
  });

  it("shows a section that only a removed line keeps", () => {
    const only = notes(line("N1", "a", "Decisions"));
    const m = withChanges([], changesBetween(only, notes()), T);
    expect(drawn(markedSections(notes(), m, T)).at(-1)).toEqual(["Decisions", "gone a"]);
  });

  it("leaves a line the user made theirs unmarked, while its mark of the model's change stands", () => {
    const mine = notes(line("N1", "a"), line("N3", "mine", "Now", "user"), line("N4", "d"));
    expect(drawn(markedSections(mine, marks, T + 1))[0]).toEqual(["Now", "a", "gone b", "mine", "fresh d"]);
  });

  it("draws the recap in the sections named", () => {
    expect(drawn(markedSections(after, [], T, ["Summary", "Now"]))).toEqual([["Summary"], ["Now", "a", "c2", "d"]]);
  });
});
