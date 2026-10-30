// Asks the browser to keep the page's storage. Storage is best-effort until
// the browser says otherwise: when the disk runs short it may evict the
// origin's data whole, with no prompt. A grant ends that; a refusal leaves
// things as they were, so the page goes on and keeps saving. Nothing here
// throws, and the page never waits on the answer.

/** The part of the browser's `StorageManager` this module uses. */
export type StorageManagerLike = { persisted(): Promise<boolean>; persist(): Promise<boolean> };

/**
 * What came of asking: `persistent` when the storage already was, so nothing was asked; `failed` when a call
 * rejected or threw; `unsupported` with no storage manager (an insecure origin, an old browser).
 */
export type Persistence = "granted" | "refused" | "persistent" | "failed" | "unsupported";

/** Asks `storage` (the browser's `navigator.storage`) to persist, when it is not persistent yet. */
export async function askForPersistence(storage: StorageManagerLike | undefined): Promise<Persistence> {
  if (typeof storage?.persisted !== "function" || typeof storage.persist !== "function") return "unsupported";
  try {
    if (await storage.persisted()) return "persistent";
    return (await storage.persist()) ? "granted" : "refused";
  } catch {
    return "failed";
  }
}

/**
 * Whether the browser keeps the storage: `kept` when `persisted()` says so, `best-effort` when it says not (or
 * cannot answer), `unsupported` with no storage manager. Never throws.
 */
export type Keeping = "kept" | "best-effort" | "unsupported";

export async function readKeeping(storage: StorageManagerLike | undefined): Promise<Keeping> {
  if (typeof storage?.persisted !== "function" || typeof storage.persist !== "function") return "unsupported";
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

/** What came of the notifications step: the permission the user gave (`failed` if the request broke), then the second ask. */
export type ViaNotifications = { permission: NotificationPermission | "failed"; persistence: Persistence };

/**
 * A Chromium-based browser counts the notifications permission as a sign a site matters. Asks for it, then asks to
 * keep the storage again. Only on a click: the Playground sends no notifications and never calls this by itself.
 * Nothing is asked when the step is not `available`, and a denied request still gets the second ask.
 */
export async function askViaNotifications(
  storage: StorageManagerLike | undefined,
  notifications: NotificationsLike | undefined,
): Promise<ViaNotifications> {
  if (notificationsStep(notifications) !== "available") return { permission: notifications?.permission ?? "denied", persistence: "unsupported" };
  let permission: ViaNotifications["permission"];
  try {
    permission = await notifications!.requestPermission();
  } catch {
    permission = "failed";
  }
  return { permission, persistence: await askForPersistence(storage) };
}
