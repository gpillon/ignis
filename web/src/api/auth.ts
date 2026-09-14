import { useSyncExternalStore } from "react";
import { apiErrorMessage } from "./errors.ts";

// The API key, for an ignis started with `--api-key`: kept in this browser's
// localStorage and sent as `Authorization: Bearer <key>` on every /v1
// request. Any 401 flips the page to the key prompt (KeyPage); the rest of
// the page keeps its state behind it.

const STORAGE_KEY = "ignis.apiKey";

export type AuthState = {
  /** The key sent with each request, if any. */
  key: string | null;
  /** ignis answered 401: the key prompt is showing. */
  needsKey: boolean;
  /** The key prompt is showing because ignis refused the key that was sent. */
  rejected: boolean;
};

let state: AuthState = { key: readStored(), needsKey: false, rejected: false };
const listeners = new Set<() => void>();

function readStored(): string | null {
  try {
    return globalThis.localStorage?.getItem(STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

function writeStored(key: string | null) {
  try {
    if (key) globalThis.localStorage?.setItem(STORAGE_KEY, key);
    else globalThis.localStorage?.removeItem(STORAGE_KEY);
  } catch {
    // No storage (private window, blocked site data): the key lasts this page.
  }
}

function update(next: AuthState) {
  state = next;
  for (const listener of listeners) listener();
}

export function getAuth(): AuthState {
  return state;
}

export function subscribeAuth(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useAuth(): AuthState {
  return useSyncExternalStore(subscribeAuth, getAuth, getAuth);
}

/** The header carrying the key, or none. */
export function authHeaders(): Record<string, string> {
  return state.key ? { Authorization: `Bearer ${state.key}` } : {};
}

/** ignis answered 401: ask for a key. A key that was sent is dropped as refused. */
export function keyRequired() {
  if (state.needsKey) return;
  writeStored(null);
  update({ key: null, needsKey: true, rejected: state.key !== null });
}

/** Use `key` from now on (the prompt checked it first). */
export function saveKey(key: string) {
  writeStored(key);
  update({ key, needsKey: false, rejected: false });
}

/** Drop the saved key and go back to the prompt. */
export function forgetKey() {
  writeStored(null);
  update({ key: null, needsKey: true, rejected: false });
}

export type KeyCheck = { ok: true } | { ok: false; message: string };

/** Whether ignis accepts `key`: one GET /v1/models carrying it. */
export async function checkKey(key: string, doFetch: typeof fetch = fetch): Promise<KeyCheck> {
  try {
    const res = await doFetch("/v1/models", { headers: { Authorization: `Bearer ${key}` } });
    if (res.ok) return { ok: true };
    if (res.status === 401) return { ok: false, message: "ignis refused this key." };
    return { ok: false, message: apiErrorMessage(res.status, await res.text()) };
  } catch (err) {
    return { ok: false, message: `Could not reach ignis: ${String(err)}` };
  }
}
