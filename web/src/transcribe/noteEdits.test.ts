import { describe, expect, it } from "vitest";
import { addLine, editLine, removeLine } from "./noteEdits.ts";
import { type NoteLine, noNotes, type Notes, readNotes } from "./notes.ts";

// The notes edited by hand (spec #428, "Editing by hand", stories 13-18):
// a line the user edits or adds is theirs, one they remove is kept aside so it
// never comes back, and an edit wins over what the model did to the line
// while the user was typing.

const line = (id: string, text: string, section = "People", author: NoteLine["author"] = "model"): NoteLine => ({ id, section, text, author });
const notes = (...lines: NoteLine[]): Notes => ({ ...noNotes("work"), lines, nextId: 6, writtenAs: "work", covered: 3, writtenAt: 500 });

describe("editing a line", () => {
  it("makes it the user's, in its place, its id kept", () => {
    const before = notes(line("N1", "Ada: acquisti."), line("N2", "Bea: fornitore."));
    const after = editLine(before, before.lines[0], "  Ada: responsabile acquisti. ");
    expect(after.lines).toEqual([line("N1", "Ada: responsabile acquisti.", "People", "user"), line("N2", "Bea: fornitore.")]);
    expect(after.nextId).toBe(6);
    expect(after.removed).toEqual([]);
  });

  it("keeps it on one line", () => {
    const before = notes(line("N1", "Ada."));
    expect(editLine(before, before.lines[0], "Ada:\nacquisti,\n\tdal 2024.").lines[0].text).toBe("Ada: acquisti, dal 2024.");
  });

  it("changes nothing when the text is the one the user saw, so an edit closed unchanged leaves a model's line the model's", () => {
    const before = notes(line("N1", "Ada: acquisti."));
    expect(editLine(before, before.lines[0], "Ada: acquisti. ")).toBe(before);
  });

  it("removes a line the user emptied, as × does", () => {
    const before = notes(line("N1", "Ada: acquisti."), line("N2", "Bea: fornitore."));
    expect(editLine(before, before.lines[0], "  ")).toEqual(removeLine(before, before.lines[0]));
  });

  it("wins over the model: a line it rewrote meanwhile takes the user's text, one it removed comes back as the user's at its section's end", () => {
    const seen = line("N1", "Ada: acquisti.");
    const rewritten = notes(line("N1", "Ada: acquisti e logistica."), line("N2", "Bea: fornitore."));
    expect(editLine(rewritten, seen, "Ada: logistica.").lines[0]).toEqual(line("N1", "Ada: logistica.", "People", "user"));
    const gone = notes(line("N2", "Bea: fornitore."), line("N3", "Il budget.", "Now"));
    expect(editLine(gone, seen, "Ada: logistica.").lines).toEqual([...gone.lines, line("N1", "Ada: logistica.", "People", "user")]);
  });

  it("leaves a line the user removed removed", () => {
    const before = notes(line("N2", "Bea: fornitore."));
    const removed = removeLine(before, line("N1", "Ada: acquisti."));
    expect(editLine(removed, line("N1", "Ada: acquisti."), "Ada.")).toBe(removed);
  });
});

describe("adding a line", () => {
  it("puts the user's line at its section's end, under the next id", () => {
    const before = notes(line("N1", "Ada: acquisti."), line("N2", "Il budget.", "Now"));
    const after = addLine(before, "People", " Carlo: finanza. ");
    expect(after.lines).toEqual([...before.lines, line("N6", "Carlo: finanza.", "People", "user")]);
    expect(after.nextId).toBe(7);
  });

  it("adds nothing for an empty line", () => {
    const before = notes(line("N1", "Ada: acquisti."));
    expect(addLine(before, "People", " \n ")).toBe(before);
  });
});

describe("removing a line", () => {
  it("keeps it aside as it was, the model's or the user's, so it never comes back", () => {
    const before = notes(line("N1", "Ada: acquisti."), line("N2", "Bea: fornitore.", "People", "user"));
    const after = removeLine(removeLine(before, before.lines[0]), before.lines[1]);
    expect(after.lines).toEqual([]);
    expect(after.removed).toEqual(before.lines);
    expect(after.nextId).toBe(6);
  });

  it("keeps aside the line the user saw when the model removed it meanwhile, and a line once", () => {
    const seen = line("N1", "Ada: acquisti.");
    const before = notes(line("N2", "Bea: fornitore."));
    const after = removeLine(before, seen);
    expect(after.removed).toEqual([seen]);
    expect(removeLine(after, seen)).toBe(after);
  });

  it("keeps the user's lines and the removed ones, with their author, across a reload", () => {
    const before = notes(line("N1", "Ada: acquisti."), line("N2", "Bea: fornitore."));
    const edited = removeLine(addLine(editLine(before, before.lines[0], "Ada: logistica."), "Now", "Il listino."), before.lines[1]);
    expect(readNotes(JSON.parse(JSON.stringify(edited)))).toEqual(edited);
    expect(edited.lines.map((l) => l.author)).toEqual(["user", "user"]);
  });
});
