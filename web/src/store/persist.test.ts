import { describe, expect, it, vi } from "vitest";
import { askForPersistence, type StorageManagerLike } from "./persist.ts";

// The Playground asks the browser to keep its storage, once, when it is not
// kept yet; whatever the browser answers, nothing here throws.

const manager = (persisted: boolean, grants: boolean) =>
  ({ persisted: vi.fn(async () => persisted), persist: vi.fn(async () => grants) }) satisfies StorageManagerLike;

describe("asking for persistent storage", () => {
  it("asks when storage is not persistent, and reports a grant", async () => {
    const storage = manager(false, true);
    expect(await askForPersistence(storage)).toBe("granted");
    expect(storage.persist).toHaveBeenCalledTimes(1);
  });

  it("reports a refusal", async () => {
    const storage = manager(false, false);
    expect(await askForPersistence(storage)).toBe("refused");
    expect(storage.persist).toHaveBeenCalledTimes(1);
  });

  it("does not ask when storage is already persistent", async () => {
    const storage = manager(true, true);
    expect(await askForPersistence(storage)).toBe("persistent");
    expect(storage.persist).not.toHaveBeenCalled();
  });

  it("goes on where the browser has no storage manager", async () => {
    expect(await askForPersistence(undefined)).toBe("unsupported");
    expect(await askForPersistence({} as StorageManagerLike)).toBe("unsupported");
  });

  it("goes on when a call rejects", async () => {
    const persistRejects = { persisted: async () => false, persist: vi.fn(async () => Promise.reject(new Error("no"))) };
    expect(await askForPersistence(persistRejects)).toBe("failed");
    const persistedRejects = { persisted: async () => Promise.reject(new Error("no")), persist: vi.fn(async () => true) };
    expect(await askForPersistence(persistedRejects)).toBe("failed");
    expect(persistedRejects.persist).not.toHaveBeenCalled();
  });

  it("goes on when a call throws at once", async () => {
    const persistedThrows = {
      persisted: (): Promise<boolean> => {
        throw new Error("no");
      },
      persist: vi.fn(async () => true),
    };
    expect(await askForPersistence(persistedThrows)).toBe("failed");
    expect(persistedThrows.persist).not.toHaveBeenCalled();
  });
});
