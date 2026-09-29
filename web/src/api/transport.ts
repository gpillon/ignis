import { useSyncExternalStore } from "react";

// Which wire the Playground's conversation goes over (GitHub #283): the
// Responses WebSocket by default, HTTP chat completions when the owner picks
// it in General, or when the socket cannot be opened (an older ignis, a proxy
// that refuses upgrades) — then for the rest of the page's life, said once.
// The choice is this browser's; the fallback is this page's. The Decide tab
// is not a conversation and keeps its own HTTP endpoint.

export type Transport = "websocket" | "http";

const STORAGE_KEY = "ignis.transport";

/** `make web-mock` serves no socket: there the page starts on HTTP, and falling back says nothing. */
const MOCK = import.meta.env.MODE === "mock";

export type TransportState = {
  /** What the owner picked. */
  choice: Transport;
  /** The socket could not be opened: HTTP carries the conversation until the owner picks WebSocket again or reloads. */
  fellBack: boolean;
  /** The notice saying so is showing. */
  notice: boolean;
};

let state: TransportState = { choice: readStored(), fellBack: false, notice: false };
const listeners = new Set<() => void>();

function readStored(): Transport {
  try {
    const stored = globalThis.localStorage?.getItem(STORAGE_KEY);
    if (stored === "websocket" || stored === "http") return stored;
  } catch {
    // No storage: the default.
  }
  return MOCK ? "http" : "websocket";
}

function update(next: TransportState) {
  state = next;
  for (const listener of listeners) listener();
}

export function getTransport(): TransportState {
  return state;
}

export function subscribeTransport(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useTransport(): TransportState {
  return useSyncExternalStore(subscribeTransport, getTransport, getTransport);
}

/** The wire the next request goes over. */
export function activeTransport(): Transport {
  return state.choice === "websocket" && !state.fellBack ? "websocket" : "http";
}

/** The owner's pick, remembered by this browser. Picking WebSocket tries the socket again. */
export function chooseTransport(choice: Transport) {
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, choice);
  } catch {
    // No storage: the choice lasts this page.
  }
  update({ choice, fellBack: choice === "websocket" ? false : state.fellBack, notice: false });
}

/** The socket could not be opened and ignis is there: HTTP from now on, and a notice the first time. */
export function fallBackToHttp() {
  if (state.fellBack) return;
  update({ ...state, fellBack: true, notice: !MOCK });
}

export function dismissTransportNotice() {
  update({ ...state, notice: false });
}
