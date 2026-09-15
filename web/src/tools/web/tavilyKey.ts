import { useSyncExternalStore } from "react";

// The Tavily API key for `web_search`, kept in this browser's localStorage.
// It is sent to Tavily only, never to ignis.

const STORAGE_KEY = "ignis.tavilyKey";

let key: string | null = readStored();
const listeners = new Set<() => void>();

function readStored(): string | null {
  try {
    return globalThis.localStorage?.getItem(STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

export function getTavilyKey(): string | null {
  return key;
}

/** Keeps `next` (trimmed) from now on; an empty key forgets it. */
export function setTavilyKey(next: string) {
  key = next.trim() || null;
  try {
    if (key) globalThis.localStorage?.setItem(STORAGE_KEY, key);
    else globalThis.localStorage?.removeItem(STORAGE_KEY);
  } catch {
    // No storage (private window, blocked site data): the key lasts this page.
  }
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useTavilyKey(): string | null {
  return useSyncExternalStore(subscribe, getTavilyKey, getTavilyKey);
}
