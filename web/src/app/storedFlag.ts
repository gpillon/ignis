import { useCallback, useState } from "react";

// A switch in the settings panel that this browser remembers. Without one of
// these a switch is `useState`, which forgets on every reload — the setting
// looks like it never took.
//
// Storage is per browser and per origin, and it can be absent or throw (a
// private window, blocked site data), so every read and write is guarded and
// the flag simply falls back to its default when storage is not there. The
// two guarded halves are `readFlag` / `writeFlag`; the hook below is glue.

/** Every stored flag's key, so the names live in one place rather than at their call sites. */
export const STORED_FLAGS = {
  /** Send in one session while another still streams. */
  parallel: "ignis.parallelSessions",
} as const;

/**
 * What this browser has stored under `key`, or `fallback` where it has
 * nothing. Nothing stored and `false` stored are different facts: a flag the
 * user turned off has to survive a default of on.
 */
export function readFlag(key: string, fallback: boolean): boolean {
  try {
    const stored = globalThis.localStorage?.getItem(key);
    return stored === null || stored === undefined ? fallback : stored === "1";
  } catch {
    return fallback;
  }
}

/** Store `on` under `key`. A browser without storage keeps nothing and says so by doing nothing. */
export function writeFlag(key: string, on: boolean): void {
  try {
    globalThis.localStorage?.setItem(key, on ? "1" : "0");
  } catch {
    // No storage: the choice lasts this page, which is what it did before.
  }
}

/**
 * A boolean that survives a reload: the value, and a setter that writes it
 * through. `fallback` is what a browser with nothing stored starts at.
 */
export function useStoredFlag(key: string, fallback: boolean): [boolean, (on: boolean) => void] {
  const [on, setOn] = useState(() => readFlag(key, fallback));
  const set = useCallback(
    (next: boolean) => {
      setOn(next);
      writeFlag(key, next);
    },
    [key],
  );
  return [on, set];
}
