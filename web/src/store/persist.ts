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
