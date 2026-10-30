// Asks the browser to keep the page's storage. Storage is best-effort until
// the browser says otherwise: when the disk runs short it may evict the
// origin's data whole, with no prompt. A grant ends that; a refusal leaves
// things as they were, so the page goes on and keeps saving. Nothing here
// throws. The page asks once at start and never waits on the answer; the
// Settings section waits on that ask before it reads the state, and asks
// again on the user's clicks.

/** The part of the browser's `StorageManager` this module uses. */
export type StorageManagerLike = { persisted(): Promise<boolean>; persist(): Promise<boolean> };

/**
 * What came of asking: `persistent` when the storage already was, so nothing was asked; `failed` when a call
 * rejected or threw; `unsupported` with no storage manager (an insecure origin, an old browser).
 */
export type Persistence = "granted" | "refused" | "persistent" | "failed" | "unsupported";

const usable = (storage: StorageManagerLike | undefined): storage is StorageManagerLike =>
  typeof storage?.persisted === "function" && typeof storage.persist === "function";

/** Asks `storage` (the browser's `navigator.storage`) to persist, when it is not persistent yet. */
export async function askForPersistence(storage: StorageManagerLike | undefined): Promise<Persistence> {
  if (!usable(storage)) return "unsupported";
  try {
    if (await storage.persisted()) return "persistent";
    return (await storage.persist()) ? "granted" : "refused";
  } catch {
    return "failed";
  }
}

let atStart: Promise<Persistence> | null = null;

/** The ask the page makes once at start (main.tsx), kept so that the Settings section reads the state after it. */
export function askAtStart(storage: StorageManagerLike | undefined): Promise<Persistence> {
  atStart = askForPersistence(storage);
  return atStart;
}

/** Settles once the start's ask has its answer; at once when the page made none. */
export const startSettled = (): Promise<unknown> => atStart ?? Promise.resolve();

/** Whether the browser keeps the storage. */
export type Keeping = "kept" | "best-effort" | "unsupported";

/**
 * `kept` when `persisted()` says so, `best-effort` when it says not or cannot answer, `unsupported` with no storage
 * manager.
 */
export async function readKeeping(storage: StorageManagerLike | undefined): Promise<Keeping> {
  if (!usable(storage)) return "unsupported";
  try {
    return (await storage.persisted()) ? "kept" : "best-effort";
  } catch {
    return "best-effort";
  }
}

/** The part of the browser's `Notification` this module uses: its permission, and the request for it. */
export type NotificationsLike = {
  permission: NotificationPermission;
  requestPermission(): Promise<NotificationPermission>;
};

/**
 * Where the notifications step stands: `available` when the permission is still to be asked (`default`), `granted`
 * or `denied` as the browser has it, `unsupported` with no Notification API.
 */
export type NotificationsStep = "available" | "granted" | "denied" | "unsupported";

export function notificationsStep(notifications: NotificationsLike | undefined): NotificationsStep {
  if (typeof notifications?.requestPermission !== "function") return "unsupported";
  if (notifications.permission === "granted") return "granted";
  if (notifications.permission === "denied") return "denied";
  return "available";
}

/**
 * What came of the notifications step: the permission as the user left it, or as it already stood (`failed` when the
 * request broke, `unsupported` with no Notification API), then the second ask.
 */
export type ViaNotifications = {
  permission: NotificationPermission | "failed" | "unsupported";
  persistence: Persistence;
};

/**
 * A Chromium-based browser counts the notifications permission as a sign a site matters. Asks for it while it is
 * undecided, then asks to keep the storage again, whatever the answer. Only on a click: the Playground sends no
 * notifications and never calls this by itself.
 */
export async function askViaNotifications(
  storage: StorageManagerLike | undefined,
  notifications: NotificationsLike | undefined,
): Promise<ViaNotifications> {
  const step = notificationsStep(notifications);
  let permission: ViaNotifications["permission"] = step === "unsupported" ? "unsupported" : notifications!.permission;
  if (step === "available") {
    try {
      permission = await notifications!.requestPermission();
    } catch {
      permission = "failed";
    }
  }
  return { permission, persistence: await askForPersistence(storage) };
}
