import { useSyncExternalStore } from "react";

// Memory: notes the model keeps across sessions, in this browser's
// localStorage. A note is a short title and a body. Only the titles go into
// the ignis system prompt, so memory stays cheap however much it holds; the
// model reads a body with memory_read when a title looks relevant.

export type MemoryNote = { id: string; title: string; body: string; savedAt: string };

const STORAGE_KEY = "ignis.memory";

/** The titles the prompt lists, newest kept first, cut to this many characters in all. */
export const MEMORY_PROMPT_LIMIT = 4000;
export const MEMORY_TITLE_LIMIT = 80;

export type MemoryStore = {
  list(): MemoryNote[];
  /** An empty title is taken from the body's first line. */
  save(title: string, body: string, now?: Date): MemoryNote;
  /** Rewrites a note; an edited note counts as saved now. */
  update(id: string, title: string, body: string, now?: Date): boolean;
  remove(id: string): boolean;
  /** Puts back a note that was removed (undo). */
  restore(note: MemoryNote): void;
  clear(): void;
  subscribe(listener: () => void): () => void;
};

type Storage = { getItem(key: string): string | null; setItem(key: string, value: string): void };

/** A title from `text`: its first line, cut to MEMORY_TITLE_LIMIT. */
export function titleOf(text: string): string {
  const line = text.trim().split("\n")[0].trim();
  return line.length > MEMORY_TITLE_LIMIT ? `${line.slice(0, MEMORY_TITLE_LIMIT - 1).trimEnd()}…` : line;
}

/** A stored note, including one saved before notes had titles (its first line becomes the title). */
function fromStored(value: unknown): MemoryNote | null {
  if (typeof value !== "object" || value === null) return null;
  const n = value as { id?: unknown; title?: unknown; body?: unknown; note?: unknown; savedAt?: unknown };
  if (typeof n.id !== "string") return null;
  const savedAt = typeof n.savedAt === "string" ? n.savedAt : new Date(0).toISOString();
  if (typeof n.title === "string") return { id: n.id, title: n.title, body: typeof n.body === "string" ? n.body : "", savedAt };
  if (typeof n.note === "string") return { id: n.id, title: titleOf(n.note), body: n.note, savedAt };
  return null;
}

export function createMemoryStore(storage: Storage | null): MemoryStore {
  let notes = load();
  const listeners = new Set<() => void>();

  function load(): MemoryNote[] {
    try {
      const parsed = JSON.parse(storage?.getItem(STORAGE_KEY) ?? "[]") as unknown;
      return Array.isArray(parsed) ? parsed.map(fromStored).filter((n): n is MemoryNote => n !== null) : [];
    } catch {
      return [];
    }
  }

  function commit(next: MemoryNote[]) {
    notes = next;
    try {
      storage?.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // No storage (private window, blocked site data): the notes last this page.
    }
    for (const listener of listeners) listener();
  }

  const clean = (title: string, body: string) => ({ title: titleOf(title.trim() || body), body: body.trim() });

  return {
    list: () => notes,
    save(title, body, now = new Date()) {
      const n = notes.reduce((max, x) => Math.max(max, Number(x.id.slice(1)) || 0), 0) + 1;
      const saved = { id: `m${n}`, ...clean(title, body), savedAt: now.toISOString() };
      commit([...notes, saved]);
      return saved;
    },
    update(id, title, body, now = new Date()) {
      if (!notes.some((x) => x.id === id) || (title.trim() === "" && body.trim() === "")) return false;
      commit(notes.map((x) => (x.id === id ? { ...x, ...clean(title, body), savedAt: now.toISOString() } : x)));
      return true;
    },
    remove(id) {
      if (!notes.some((x) => x.id === id)) return false;
      commit(notes.filter((x) => x.id !== id));
      return true;
    },
    restore(note) {
      if (notes.some((x) => x.id === note.id)) return;
      commit([...notes, note].sort((a, b) => Date.parse(a.savedAt) - Date.parse(b.savedAt)));
    },
    clear() {
      commit([]);
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

function browserStorage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export const browserMemory = createMemoryStore(browserStorage());

export function useMemoryNotes(store: MemoryStore = browserMemory): MemoryNote[] {
  return useSyncExternalStore(store.subscribe, store.list, store.list);
}
