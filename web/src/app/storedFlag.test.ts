import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readFlag, STORED_FLAGS, writeFlag } from "./storedFlag.ts";

const KEY = STORED_FLAGS.parallel;

/** A browser's storage, as much of it as a flag touches. */
function fakeStorage(fail = false) {
  const items = new Map<string, string>();
  return {
    getItem: (k: string) => {
      if (fail) throw new Error("blocked site data");
      return items.get(k) ?? null;
    },
    setItem: (k: string, v: string) => {
      if (fail) throw new Error("blocked site data");
      items.set(k, v);
    },
  };
}

function useStorage(storage: unknown) {
  Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true, writable: true });
}

describe("storedFlag", () => {
  beforeEach(() => useStorage(fakeStorage()));
  afterEach(() => useStorage(undefined));

  it("starts at the default when this browser has stored nothing", () => {
    expect(readFlag(KEY, false)).toBe(false);
    expect(readFlag(KEY, true)).toBe(true);
  });

  // The bug this exists for: the switch was `useState`, so turning it on
  // lasted until the next reload and looked like it had never taken.
  it("reads back what was written, so a reload keeps the choice", () => {
    writeFlag(KEY, true);
    expect(readFlag(KEY, false)).toBe(true);
  });

  // Off stored and off by default are the same value but not the same fact:
  // a flag the user turned off has to survive a default of on.
  it("keeps a stored off against a default of on", () => {
    writeFlag(KEY, false);
    expect(readFlag(KEY, true)).toBe(false);
  });

  it("falls back to the default rather than throwing when storage throws", () => {
    useStorage(fakeStorage(true));
    expect(() => writeFlag(KEY, true)).not.toThrow();
    expect(readFlag(KEY, true)).toBe(true);
    expect(readFlag(KEY, false)).toBe(false);
  });

  it("falls back to the default where there is no storage at all", () => {
    useStorage(undefined);
    expect(() => writeFlag(KEY, true)).not.toThrow();
    expect(readFlag(KEY, true)).toBe(true);
  });
});
