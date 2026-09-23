// The one call the Decide tab makes (GitHub #247): `POST /v1/decide` with
// every question in the request, because that is what the endpoint is for —
// the evidence is prefilled once and the questions share it. A per-question
// call would hide the only property worth showing.

import { authHeaders, keyRequired } from "../api/auth.ts";
import { apiErrorMessage } from "../api/errors.ts";

export const DECIDE_PATH = "/v1/decide";

/** One digit of a constrained answer, and the model's own confidence in it. */
export type DigitDraw = { digit: number; probability: number };

/** One question's answer. The tag is `decide.rs`'s `Answer`. */
export type Answer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number }
  | { type: "score"; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number }
  | { type: "number"; number: number; uncertainty: number; digits: DigitDraw[] }
  // A `scalar` closes its own object (GitHub #255), so `text` is what the
  // model actually wrote and `value` is what that spelling parsed to — `3`
  // and `3.0` are the same number and not the same answer. `uncertainty` is
  // in units of the value, and the trace carries the digits alone: the
  // decimal point, the sign and the closing brace were steps of the run but
  // hold no place, so a trace shorter than `text` is right.
  | { type: "scalar"; value: number; text: string; uncertainty: number; digits: DigitDraw[] }
  // A `point` and a `box` say which method answered them (GitHub #260, #263).
  // A `head` answer is read in one pass off the calibrated heads and carries
  // no digit trace, only the `region` it was read from; a `chain` answer
  // carries the trace. On a head answer `uncertainty` is the reading's
  // **resolution** and not a spread: half an image token where a head set
  // resolved a peak inside its cell (GitHub #264), one whole token where the
  // pointing head answered alone. `extent` is the box the head set outlines
  // around a head point — absent on every other answer, including a head
  // point off the pointing head alone, which has no set to outline with.
  | {
      type: "point" | "box";
      method?: "head" | "chain";
      pixels: Record<string, number>;
      normalized: Record<string, number>;
      uncertainty: Record<string, number>;
      region?: { cells: number; share: number };
      extent?: Record<string, number>;
      digits?: Record<string, DigitDraw[]>;
    }
  | { type: "error"; code: string; message: string };

export type DecideResponse = {
  model: string;
  /** Keyed by the ids the request chose — alphabetically sorted, so panels read the request's order instead. */
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
};

/** A finished run: the body, the wall-clock it took, and the bytes it came back as. */
export type Run = {
  response: DecideResponse;
  /** Measured in this browser, around the fetch. */
  elapsedMs: number;
  /** The raw body, for the reader who wants the wire. */
  raw: string;
};

export type Sent = { ok: true; run: Run } | { ok: false; message: string };

/** Send `body` (already-serialized JSON) and read the answer, or say why there is none. */
export async function decide(body: string, signal?: AbortSignal): Promise<Sent> {
  const started = performance.now();
  let res: Response;
  try {
    res = await fetch(DECIDE_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body,
      signal,
    });
  } catch (error) {
    if (signal?.aborted) return { ok: false, message: "The decision was stopped." };
    return { ok: false, message: `ignis is not answering: ${String(error)}` };
  }
  const raw = await res.text();
  const elapsedMs = performance.now() - started;
  if (res.status === 401) {
    keyRequired();
    return { ok: false, message: "ignis wants the API key." };
  }
  if (!res.ok) return { ok: false, message: apiErrorMessage(res.status, raw) };
  try {
    return { ok: true, run: { response: JSON.parse(raw) as DecideResponse, elapsedMs, raw } };
  } catch {
    return { ok: false, message: "ignis answered 200 with something that is not JSON." };
  }
}

/** A score's level keys in index order: they arrive as strings, where "10" sorts before "2". */
export function levelOrder(probabilities: Record<string, number>): string[] {
  return Object.keys(probabilities).sort((a, b) => Number(a) - Number(b));
}

/** The axes a point or box answers on, in the order the shape reads. */
export const AXES: Record<"point" | "box", string[]> = { point: ["x", "y"], box: ["x0", "y0", "x1", "y1"] };
