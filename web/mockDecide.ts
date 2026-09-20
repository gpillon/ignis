import { type JsonNode, parseOrdered, writeOrdered } from "./src/decide/json.ts";

// A fake `/v1/decide` for `npm run dev:mock` (GitHub #247): enough of the real
// answer shapes to build the Decide tab against without the shared GPU.
// Development only — never part of the build.
//
// It reads the request through the tab's own order-preserving parser, not
// `JSON.parse`, so an option order like `3, 1, 2` stays that order here too and
// the winner is really the option the request declared first.
//
// Every figure is derived from a hash of the question, so a run is repeatable
// and two questions never look the same. A question whose instructions contain
// "/error" answers with a per-question error instead, which is how the panel's
// error row is exercised; "/full" refuses the whole request the way an engine
// at capacity does.

type Digit = { digit: number; probability: number };

/** A stable 0-1 from a string: the mock's only source of variety. */
function hashed(seed: string, salt = 0): number {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

const entry = (node: JsonNode, key: string): JsonNode | undefined =>
  node.kind === "object" ? node.entries.find((e) => e.key === key)?.value : undefined;

const text = (node: JsonNode | undefined): string => {
  if (node === undefined) return "";
  if (node.kind === "string") return node.value;
  if (node.kind === "number" || node.kind === "boolean") return String(node.value);
  if (node.kind === "null") return "";
  return writeOrdered(node, 0);
};

/** Weights that sum to 1, one per slot, with a clear winner and a long thin tail. */
function distribute(seed: string, count: number): number[] {
  const raw = Array.from({ length: count }, (_, i) => 0.02 + hashed(seed, i + 1) ** 3);
  const lead = Math.floor(hashed(seed, 991) * count);
  raw[lead] += 1.4;
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map((w) => w / total);
}

/** The digits of one axis, with the falling confidence a real trace shows. */
function digitsFor(seed: string, count: number): Digit[] {
  return Array.from({ length: count }, (_, place) => ({
    digit: Math.floor(hashed(seed, place + 17) * 10),
    // Leading places are read confidently and the units are not, which is the
    // shape the measured traces have.
    probability: Number(Math.max(0.12, 0.99 - place * (0.16 + hashed(seed, place + 71) * 0.2)).toFixed(4)),
  }));
}

const valueOf = (digits: Digit[]): number => Number(digits.map((d) => d.digit).join(""));
/** `sigma = sum((1 - p_k) * 10^place)`, as `numbers.rs` computes it. */
const sigmaOf = (digits: Digit[]): number =>
  digits.reduce((sum, d, i) => sum + (1 - d.probability) * 10 ** (digits.length - 1 - i), 0);

/** A JPEG's pixel size, read off its SOF marker, so a point lands where it says it does. */
function jpegSize(dataUri: string): { width: number; height: number } | null {
  const comma = dataUri.indexOf(",");
  if (comma < 0) return null;
  let bytes: Buffer;
  try {
    bytes = Buffer.from(dataUri.slice(comma + 1), "base64");
  } catch {
    return null;
  }
  let at = 2;
  while (at + 9 < bytes.length) {
    if (bytes[at] !== 0xff) {
      at++;
      continue;
    }
    const marker = bytes[at + 1];
    // SOF0, SOF1, SOF2: the frame header carries the size.
    if (marker >= 0xc0 && marker <= 0xc2) {
      return { height: bytes.readUInt16BE(at + 5), width: bytes.readUInt16BE(at + 7) };
    }
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      at += 2;
      continue;
    }
    at += 2 + bytes.readUInt16BE(at + 2);
  }
  return null;
}

/** The image the request submitted, if its `state` carried one. */
function stateImage(state: JsonNode | undefined): { width: number; height: number } | null {
  if (!state || state.kind !== "array") return null;
  for (const part of state.items) {
    const url = entry(part, "image_url");
    const inner = url ? entry(url, "url") : undefined;
    if (inner?.kind === "string") return jpegSize(inner.value);
  }
  return null;
}

/** The mock's answer to one request body, or the refusal it stands in for. */
export function mockDecide(raw: string): { status: number; body: unknown } {
  const parsed = parseOrdered(raw || "null");
  if (!parsed.ok || parsed.node.kind !== "object") {
    return { status: 400, body: { error: { type: "invalid_request_error", code: "bad_json", message: "the body is not a JSON object" } } };
  }
  const state = entry(parsed.node, "state");
  const questions = entry(parsed.node, "questions");
  if (!questions || questions.kind !== "object" || questions.entries.length === 0) {
    return {
      status: 422,
      body: { error: { type: "invalid_request_error", code: "no_questions", message: "`questions` must carry at least one question" } },
    };
  }
  const evidence = text(state);
  if (evidence.includes("/full")) {
    return { status: 503, body: { error: { type: "server_error", code: "engine_full", message: "the engine is at capacity" } } };
  }

  const image = stateImage(state);
  const answers: Record<string, unknown> = {};
  let generated = 0;

  for (const { key: id, value: question } of questions.entries) {
    const kind = text(entry(question, "type"));
    const instructions = text(entry(question, "instructions"));
    const seed = `${id}|${kind}|${instructions}`;
    if (instructions.includes("/error")) {
      answers[id] = { type: "error", code: "engine_full", message: "the engine refused this question's admission" };
      continue;
    }
    const criteria = entry(question, "criteria");
    const digits = Math.max(1, Math.min(6, Number(text(entry(question, "digits"))) || 3));

    if (kind === "noul" || kind === "boolean") {
      answers[id] = { type: "noul", noul: Number((0.04 + hashed(seed) * 0.93).toFixed(4)) };
      continue;
    }
    if (kind === "choice") {
      const keys = criteria?.kind === "object" ? criteria.entries.map((e) => e.key) : ["a", "b"];
      const weights = distribute(seed, keys.length);
      const probabilities: Record<string, number> = {};
      keys.forEach((k, i) => (probabilities[k] = Number(weights[i].toFixed(4))));
      const top = Math.max(...weights);
      answers[id] = {
        type: "choice",
        choice: keys[weights.indexOf(top)],
        probabilities,
        confidence: Number(top.toFixed(4)),
      };
      continue;
    }
    if (kind === "score") {
      const levels = criteria?.kind === "array" ? criteria.items.map(text) : ["low", "high"];
      const weights = distribute(seed, levels.length);
      const probabilities: Record<string, number> = {};
      const legend: Record<string, string> = {};
      levels.forEach((label, i) => {
        probabilities[String(i)] = Number(weights[i].toFixed(4));
        legend[String(i)] = label;
      });
      const mean = weights.reduce((sum, p, i) => sum + p * i, 0);
      const variance = weights.reduce((sum, p, i) => sum + p * (i - mean) ** 2, 0);
      const half = Math.max(0.5, (levels.length - 1) / 2);
      answers[id] = {
        type: "score",
        score: Number(mean.toFixed(4)),
        legend,
        probabilities,
        confidence: Number(Math.max(0, 1 - Math.sqrt(variance) / half).toFixed(4)),
      };
      continue;
    }
    if (kind === "number") {
      const trace = digitsFor(seed, digits);
      generated += digits;
      answers[id] = { type: "number", number: valueOf(trace), uncertainty: Number(sigmaOf(trace).toFixed(4)), digits: trace };
      continue;
    }
    if (kind === "point" || kind === "box") {
      if (!image) {
        answers[id] = { type: "error", code: "state_carries_no_image", message: "a point is a position on an image, and this `state` carried none" };
        continue;
      }
      const axes = kind === "point" ? ["x", "y"] : ["x0", "y0", "x1", "y1"];
      const scale = 10 ** digits - 1;
      const pixels: Record<string, number> = {};
      const normalized: Record<string, number> = {};
      const uncertainty: Record<string, number> = {};
      const trace: Record<string, Digit[]> = {};
      for (const [index, axis] of axes.entries()) {
        const own = digitsFor(`${seed}|${axis}`, digits);
        // A box reads left/top in the first half of the frame and right/bottom
        // in the second, so the rectangle it draws is not inside out.
        let value = valueOf(own);
        if (kind === "box") value = index < 2 ? Math.round(value * 0.45) : Math.round(scale * 0.55 + value * 0.4);
        const side = axis.startsWith("x") ? image.width : image.height;
        pixels[axis] = Math.round((value / scale) * side);
        normalized[axis] = value;
        uncertainty[axis] = Number(((sigmaOf(own) / scale) * side).toFixed(4));
        trace[axis] = own;
        generated += digits;
      }
      answers[id] = { type: kind, pixels, normalized, uncertainty, digits: trace };
      continue;
    }
    answers[id] = { type: "error", code: "unknown_type", message: `the mock does not answer a ${kind}` };
  }

  return {
    status: 200,
    body: {
      model: "mock-model",
      answers,
      // A readout generates nothing, so this stays 0 unless a number, point or
      // box was asked for — the same honesty the real endpoint keeps.
      usage: { input_tokens: Math.ceil(raw.length / 3.6), output_tokens: generated },
    },
  };
}
