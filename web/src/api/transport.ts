import { useSyncExternalStore } from "react";
import { readFlag, STORED_FLAGS, writeFlag } from "../app/storedFlag.ts";

// Which wire the Playground's conversation goes over (GitHub #283): the
// Responses WebSocket by default, HTTP chat completions when the owner picks
// it in General, or when the socket cannot be opened (an older ignis, a proxy
// that refuses upgrades) — then until the page reloads or the owner picks
// WebSocket again, said once. The choice is this browser's; the fallback is
// this page's. The Decide tab
// is not a conversation and keeps its own HTTP endpoint.

export type Transport = "websocket" | "http";

/** `make web-mock` serves no socket: there the page talks HTTP whatever the setting says, and falling back says nothing. */
const MOCK = import.meta.env.MODE === "mock";

export type TransportState = {
  /** What the owner picked. */
  choice: Transport;
  /** The socket could not be opened: HTTP carries the conversation until the owner picks WebSocket again or reloads. */
  fellBack: boolean;
  /** The notice saying so is showing. */
  notice: boolean;
};

let state: TransportState = { choice: readFlag(STORED_FLAGS.httpTransport, false) ? "http" : "websocket", fellBack: false, notice: false };
const listeners = new Set<() => void>();

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
  return state.choice === "websocket" && !state.fellBack && !MOCK ? "websocket" : "http";
}

/** The owner's pick, remembered by this browser. Picking WebSocket tries the socket again. */
export function chooseTransport(choice: Transport) {
  writeFlag(STORED_FLAGS.httpTransport, choice === "http");
  update({ choice, fellBack: choice === "websocket" ? false : state.fellBack, notice: false });
}

/** The socket could not be opened and ignis is there: HTTP from now on, and a notice the first time. */
export function fallBackToHttp() {
  if (state.fellBack) return;
  update({ ...state, fellBack: true, notice: true });
}

export function dismissTransportNotice() {
  update({ ...state, notice: false });
}
