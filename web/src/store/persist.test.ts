import { describe, expect, it, vi } from "vitest";
import {
  askAtStart,
  askForPersistence,
  askViaNotifications,
  type NotificationsLike,
  notificationsStep,
  readKeeping,
  type StorageManagerLike,
  startSettled,
} from "./persist.ts";

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

describe("the ask at start", () => {
  it("is what the Settings section waits on before it reads the state", async () => {
    let answer: (kept: boolean) => void = () => {};
    const storage = { persisted: () => new Promise<boolean>((r) => (answer = r)), persist: vi.fn(async () => true) };
    const asked = askAtStart(storage);
    let settled = false;
    void startSettled().then(() => (settled = true));
    await Promise.resolve();
    expect(settled).toBe(false);
    answer(true);
    expect(await asked).toBe("persistent");
    await startSettled();
    expect(settled).toBe(true);
  });
});

describe("reading whether the storage is kept", () => {
  it("says kept, best-effort and unsupported, and survives a rejecting call", async () => {
    expect(await readKeeping(manager(true, true))).toBe("kept");
    expect(await readKeeping(manager(false, true))).toBe("best-effort");
    expect(await readKeeping(undefined)).toBe("unsupported");
    expect(await readKeeping({ persisted: async () => Promise.reject(new Error("no")), persist: async () => true })).toBe("best-effort");
  });
});

const notifier = (permission: NotificationPermission, answer: NotificationPermission | Error = permission) =>
  ({
    permission,
    requestPermission: vi.fn(async () => {
      if (answer instanceof Error) throw answer;
      return answer;
    }),
  }) satisfies NotificationsLike;

describe("the notifications step", () => {
  it("is available only while the permission is undecided", () => {
    expect(notificationsStep(notifier("default"))).toBe("available");
    expect(notificationsStep(notifier("granted"))).toBe("granted");
    expect(notificationsStep(notifier("denied"))).toBe("denied");
    expect(notificationsStep(undefined)).toBe("unsupported");
  });

  it("asks the permission, then the storage again: refused, then granted", async () => {
    const storage = { persisted: vi.fn(async () => false), persist: vi.fn().mockResolvedValueOnce(true) };
    const n = notifier("default", "granted");
    expect(await askViaNotifications(storage, n)).toEqual({ permission: "granted", persistence: "granted" });
    expect(n.requestPermission).toHaveBeenCalledTimes(1);
  });

  it("reports a refusal that notifications did not change", async () => {
    const result = await askViaNotifications(manager(false, false), notifier("default", "granted"));
    expect(result).toEqual({ permission: "granted", persistence: "refused" });
  });

  it("asks the storage again even when the permission is denied in the prompt", async () => {
    const storage = manager(false, false);
    expect(await askViaNotifications(storage, notifier("default", "denied"))).toEqual({ permission: "denied", persistence: "refused" });
    expect(storage.persist).toHaveBeenCalledTimes(1);
  });

  it("asks only the storage again when the permission is already decided or the API is missing", async () => {
    const storage = manager(false, false);
    const denied = notifier("denied");
    expect(await askViaNotifications(storage, denied)).toEqual({ permission: "denied", persistence: "refused" });
    expect(await askViaNotifications(storage, notifier("granted"))).toEqual({ permission: "granted", persistence: "refused" });
    expect(await askViaNotifications(storage, undefined)).toEqual({ permission: "unsupported", persistence: "refused" });
    expect(denied.requestPermission).not.toHaveBeenCalled();
    expect(storage.persist).toHaveBeenCalledTimes(3);
  });

  it("leaves the step open when the question is closed without an answer", async () => {
    const result = await askViaNotifications(manager(false, false), notifier("default", "default"));
    expect(result).toEqual({ permission: "default", persistence: "refused" });
  });

  it("goes on when a call rejects", async () => {
    const storage = manager(false, true);
    expect(await askViaNotifications(storage, notifier("default", new Error("no")))).toEqual({ permission: "failed", persistence: "granted" });
    const persistFails = { persisted: async () => false, persist: async () => Promise.reject(new Error("no")) };
    expect((await askViaNotifications(persistFails, notifier("default", "granted"))).persistence).toBe("failed");
  });
});
