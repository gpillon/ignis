import { describe, expect, it } from "vitest";
import { createMemoryStore, type MemoryNote, titleOf } from "./memory.ts";
import { COOL_AFTER_MS, heat, highlight, savedAgo, visibleNotes } from "./memoryView.ts";

const now = Date.parse("2026-09-15T12:00:00Z");
const at = (msAgo: number) => new Date(now - msAgo).toISOString();
const note = (id: string, title: string, body: string, msAgo: number): MemoryNote => ({ id, title, body, savedAt: at(msAgo) });

describe("heat and savedAgo", () => {
  it("cools from 100 to 0 over a week", () => {
    expect(heat(at(0), now)).toBe(100);
    expect(heat(at(COOL_AFTER_MS / 2), now)).toBe(50);
    expect(heat(at(COOL_AFTER_MS * 3), now)).toBe(0);
    expect(heat("not a date", now)).toBe(0);
  });

  it("says how long ago, then the date", () => {
    expect(savedAgo(at(10_000), now)).toBe("just now");
    expect(savedAgo(at(5 * 60_000), now)).toBe("5 minutes ago");
    expect(savedAgo(at(3 * 3_600_000), now)).toBe("3 hours ago");
    expect(savedAgo(at(26 * 3_600_000), now)).toBe("yesterday");
    expect(savedAgo("2026-08-02T10:00:00Z", now)).toBe("2 Aug");
    expect(savedAgo("2025-08-02T10:00:00Z", now)).toBe("2 Aug 2025");
  });
});

describe("visibleNotes and highlight", () => {
  const notes = [
    note("m1", "Language", "Prefers answers in Italian.", 3000),
    note("m2", "Team", "Supports AS Roma.", 2000),
    note("m3", "Project", "Works on ignis, in Rust.", 1000),
  ];

  it("keeps notes with every word in the title or body, in either order", () => {
    expect(visibleNotes(notes, "", "newest").map((n) => n.id)).toEqual(["m3", "m2", "m1"]);
    expect(visibleNotes(notes, "", "oldest").map((n) => n.id)).toEqual(["m1", "m2", "m3"]);
    expect(visibleNotes(notes, "project  RUST", "newest").map((n) => n.id)).toEqual(["m3"]);
    expect(visibleNotes(notes, "m2", "newest").map((n) => n.id)).toEqual(["m2"]);
  });

  it("marks where the words matched", () => {
    expect(highlight("Supports AS Roma.", "roma")).toEqual([
      { text: "Supports AS ", match: false },
      { text: "Roma", match: true },
      { text: ".", match: false },
    ]);
    expect(highlight("a+b", "a+")).toEqual([
      { text: "a+", match: true },
      { text: "b", match: false },
    ]);
    expect(highlight("text", " ")).toEqual([{ text: "text", match: false }]);
  });
});

describe("memory store", () => {
  it("takes a missing title from the body, and cuts a long one", () => {
    const store = createMemoryStore(null);
    expect(store.save("  ", "Prefers Italian.\nTechnical terms stay in English.", new Date(now))).toMatchObject({
      id: "m1",
      title: "Prefers Italian.",
      body: "Prefers Italian.\nTechnical terms stay in English.",
    });
    expect(titleOf("x".repeat(100))).toHaveLength(80);
  });

  it("rewrites a note as saved now, and restores a removed one in its place", () => {
    const store = createMemoryStore(null);
    store.save("One", "first", new Date(now - 5000));
    store.save("Two", "second", new Date(now - 4000));
    expect(store.update("m1", " One, edited ", "first, edited", new Date(now))).toBe(true);
    expect(store.list()[0]).toEqual({ id: "m1", title: "One, edited", body: "first, edited", savedAt: new Date(now).toISOString() });
    expect(store.update("m1", " ", " ")).toBe(false);
    expect(store.update("m9", "x", "y")).toBe(false);

    const second = store.list()[1];
    store.remove("m2");
    store.restore(second);
    store.restore(second);
    expect(store.list().map((n) => n.id)).toEqual(["m2", "m1"]);
  });

  it("reads notes saved before titles, with their first line as the title", () => {
    const storage = {
      getItem: () => JSON.stringify([{ id: "m4", note: "Supports AS Roma.\nSince 1990.", savedAt: "2026-09-01T00:00:00Z" }, { bad: true }]),
      setItem: () => {},
    };
    expect(createMemoryStore(storage).list()).toEqual([
      { id: "m4", title: "Supports AS Roma.", body: "Supports AS Roma.\nSince 1990.", savedAt: "2026-09-01T00:00:00Z" },
    ]);
  });
});
