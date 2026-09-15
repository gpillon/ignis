import type { MemoryNote } from "./memory.ts";

// How the memory manager shows notes: their heat (how recently they were
// saved), when, which ones a search keeps, and where a search matched.

/** How long a note takes to cool from ember to ash. */
export const COOL_AFTER_MS = 7 * 24 * 60 * 60 * 1000;

/** 100 for a note saved just now, falling to 0 once it is a week old. */
export function heat(savedAt: string, now: number): number {
  const age = now - Date.parse(savedAt);
  if (!Number.isFinite(age)) return 0;
  return Math.round(100 * (1 - Math.min(1, Math.max(0, age) / COOL_AFTER_MS)));
}

/** `just now`, `5 minutes ago`, `yesterday`, then the date. */
export function savedAgo(savedAt: string, now: number): string {
  const at = Date.parse(savedAt);
  if (!Number.isFinite(at)) return "";
  const seconds = Math.max(0, Math.round((now - at) / 1000));
  if (seconds < 45) return "just now";
  const relative = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  if (seconds < 3600) return relative.format(-Math.max(1, Math.round(seconds / 60)), "minute");
  if (seconds < 86_400) return relative.format(-Math.round(seconds / 3600), "hour");
  if (seconds < 7 * 86_400) return relative.format(-Math.round(seconds / 86_400), "day");
  const sameYear = new Date(at).getFullYear() === new Date(now).getFullYear();
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", ...(sameYear ? {} : { year: "numeric" }) }).format(at);
}

export type NoteOrder = "newest" | "oldest";

const words = (query: string) => query.toLowerCase().split(/\s+/).filter(Boolean);

/** The notes holding every word of the query in their title or body (or with it as their id), newest or oldest first. */
export function visibleNotes(notes: MemoryNote[], query: string, order: NoteOrder): MemoryNote[] {
  const wanted = words(query);
  const kept = notes.filter((n) => {
    const text = `${n.title}\n${n.body}`.toLowerCase();
    return wanted.every((w) => text.includes(w) || n.id.toLowerCase() === w);
  });
  const sorted = [...kept].sort((a, b) => Date.parse(a.savedAt) - Date.parse(b.savedAt));
  return order === "newest" ? sorted.reverse() : sorted;
}

/** The text in pieces, the ones matching a word of the query marked. */
export function highlight(text: string, query: string): { text: string; match: boolean }[] {
  const wanted = words(query).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (wanted.length === 0) return [{ text, match: false }];
  const alternatives = wanted.join("|");
  const isMatch = new RegExp(`^(?:${alternatives})$`, "i");
  return text
    .split(new RegExp(`(${alternatives})`, "gi"))
    .filter((piece) => piece !== "")
    .map((piece) => ({ text: piece, match: isMatch.test(piece) }));
}
