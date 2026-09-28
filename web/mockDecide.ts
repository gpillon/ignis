import { type JsonNode, parseOrdered, writeOrdered } from "./src/decide/json.ts";
import { cutTarget, FOUND_THRESHOLD, POINTER_SHARE, resolve, type Target } from "./src/decide/locate.ts";
import { type Compression, COMPRESSIONS, LOCATE_KINDS, LOCATE_METHODS, type LocateMethod, type ResolvedKind } from "./src/decide/model.ts";

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
// at capacity does; "/absent" answers a `locate` "not found" on the routes
// that carry `found`.

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
/**
 * `sigma = sum((1 - p_k) * 10^place)`, as `numbers.rs` computes it, where
 * `fraction` is how many of the trailing digits fell after a decimal point.
 *
 * A `number` has no point and passes zero; a `scalar` cannot know a digit's
 * place until the point has been seen, which is why `scalar.rs` parses the
 * run before it weights it.
 */
const sigmaAt = (digits: Digit[], fraction: number): number =>
  digits.reduce((sum, d, i) => sum + (1 - d.probability) * 10 ** (digits.length - fraction - 1 - i), 0);
const sigmaOf = (digits: Digit[]): number => sigmaAt(digits, 0);

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

/** An OpenAI content-parts list, which is the one `state` a `locate` refuses. */
const isParts = (state: JsonNode): boolean =>
  state.kind === "array" && state.items.length > 0 && state.items.every((part) => entry(part, "type") !== undefined);

const refuse = (code: string, message: string) => ({ status: 422, body: { error: { type: "invalid_request_error", code, message } } });

/** The heads a served `locate` votes with: every share is a whole number of their votes. */
const VOTERS = 32;
/** `RANKING_LEN`: the most segments a `locate`'s ranking lists. */
const RANKING_LEN = 5;
/** The candidates the heads keep for a shortlist's last choice (spec 22). */
const SHORTLIST = 16;
/** `AUTO_SEGMENTS`: how many non-blank segments `auto` reads. */
const AUTO_SEGMENTS = 2000;

/** Whether `value` is one of `values`, narrowing it to their type. */
const isOneOf = <T extends string>(values: readonly T[], value: string): value is T => (values as readonly string[]).includes(value);

/** What a `locate` asked for, validated as `prepare_locate` does: `null` is `auto`, and the compression its kind's own. */
type Ask = { kind: ResolvedKind | null; method: LocateMethod; compression: Compression | null };

type Refusal = { ok: false; code: string; message: string };

/**
 * A `locate`'s `method`, `kind` and `compression` (GitHub #278), refused in
 * the server's order and words: an unknown value naming the accepted ones,
 * then the two combinations that never apply — a fold under a vote, a fold of
 * prose named as such.
 */
function locateAsk(id: string, question: JsonNode): { ok: true; ask: Ask } | Refusal {
  const q = JSON.stringify(id);
  const named = (key: string): string | null => {
    const node = entry(question, key);
    return node === undefined ? null : text(node);
  };
  const method = named("method") ?? "shortlist";
  if (!isOneOf(LOCATE_METHODS, method)) {
    return {
      ok: false,
      code: "method_unknown",
      message: `question ${q} asks for method ${JSON.stringify(method)}; a locate's accepted values are "shortlist" and "vote"`,
    };
  }
  const kind = named("kind") ?? "auto";
  if (!isOneOf(LOCATE_KINDS, kind)) {
    return {
      ok: false,
      code: "kind_unknown",
      message: `question ${q} asks for kind ${JSON.stringify(kind)}; the accepted values are "auto", "log", "prose" and "records"`,
    };
  }
  const compression = named("compression");
  if (compression !== null && !isOneOf(COMPRESSIONS, compression)) {
    return {
      ok: false,
      code: "compression_unknown",
      message: `question ${q} asks for compression ${JSON.stringify(compression)}; the accepted values are "template_fold" and "none"`,
    };
  }
  if (compression === "template_fold" && method === "vote") {
    return {
      ok: false,
      code: "compression_unsupported",
      message: `question ${q}: the vote reads the text as it is — over a fold it read 28 of 58 real-log questions — so "template_fold" is refused with "vote"; ask for "shortlist", or for "none"`,
    };
  }
  if (compression === "template_fold" && kind === "prose") {
    return {
      ok: false,
      code: "compression_unsupported",
      message: `question ${q}: prose does not fold into templates, so "template_fold" is refused with "prose"; ask for "none", or omit \`compression\``,
    };
  }
  return { ok: true, ask: { kind: kind === "auto" ? null : kind, method, compression } };
}

/**
 * The kind a target resolves to and the compression it gets, as
 * `resolve_locate` says them: `auto` told, a named kind the target
 * contradicts refused, and a fold of what `auto` told is prose refused.
 */
function resolveLocate(id: string, ask: Ask, target: Target): { ok: true; kind: ResolvedKind; compression: Compression } | Refusal {
  const q = JSON.stringify(id);
  if (ask.kind === "records" && !target.records) {
    return {
      ok: false,
      code: "kind_mismatch",
      message: `question ${q} names kind "records", and its target is not an array of two or more JSON objects; ask for "log" or "prose", or omit \`kind\``,
    };
  }
  if ((ask.kind === "log" || ask.kind === "prose") && target.records) {
    return {
      ok: false,
      code: "kind_mismatch",
      message: `question ${q} names kind "${ask.kind}", and its target is an array of JSON objects, which is read as records; to fold it, ask for kind "records" with compression "template_fold"`,
    };
  }
  const kind = ask.kind ?? autoKind(target);
  const compression = ask.compression ?? (kind === "log" ? "template_fold" : "none");
  if (kind === "prose" && compression === "template_fold") {
    return {
      ok: false,
      code: "compression_unsupported",
      message: `question ${q}: its target reads as prose, which does not fold into templates, so "template_fold" is refused; ask for "none", or omit \`compression\``,
    };
  }
  return { ok: true, kind, compression };
}

/**
 * `auto`, approximated. A records array is told exactly, by its shape. The
 * rest the server tells by folding the first 2,000 non-blank segments into
 * templates and saying `log` when at least half of them fall in templates of
 * two or more; the mock has no fold, so it calls two segments one template
 * when their first two words match with every digit masked — near enough that
 * a timestamped log reads as a log and prose as prose, and no more than that.
 */
function autoKind(target: Target): ResolvedKind {
  if (target.records) return "records";
  const shapes = target.segments
    .filter((_, index) => target.owns[index])
    .slice(0, AUTO_SEGMENTS)
    .map((segment) => segment.trim().split(/\s+/).slice(0, 2).join(" ").replace(/\d+/g, "0"));
  const counts = new Map<string, number>();
  for (const shape of shapes) counts.set(shape, (counts.get(shape) ?? 0) + 1);
  const templated = shapes.filter((shape) => (counts.get(shape) ?? 0) >= 2).length;
  return templated >= shapes.length / 2 ? "log" : "prose";
}

/**
 * Every segment as the caller sent it: a line as a string, an element as
 * itself. An element is read off the state and not off the cut, which keeps
 * a string element's raw text — `JSON.parse` of that is not the element.
 */
function segmentValues(state: JsonNode, within: string, target: Target): unknown[] {
  if (target.unit === "line") return target.segments;
  const resolved = resolve(state, within);
  const items = resolved.ok && resolved.node.kind === "array" ? resolved.node.items : [];
  return items.map((item) => JSON.parse(writeOrdered(item, 0)) as unknown);
}

/**
 * A `vote` over `target`: the calibrated vote, faked. Every head votes for
 * one segment that owns a token — a blank line is never named — most of them
 * for one winner, the rest spread over a few runners-up, so the ranking and
 * the context both have something to show. It names the kind the target
 * resolved to, reads as it is whatever the kind, and points at its winner
 * alone (GitHub #278).
 */
function voteAnswer(seed: string, target: Target, values: unknown[], kind: ResolvedKind) {
  const owning = target.owns.flatMap((own, index) => (own ? [index] : []));
  const votes = new Map<number, number>();
  const winner = owning[Math.floor(hashed(seed, 607) * owning.length)];
  votes.set(winner, 12 + Math.floor(hashed(seed, 613) * 16));
  for (let head = [...votes.values()][0]; head < VOTERS; head++) {
    const pick = owning[Math.floor(hashed(seed, 700 + head) * owning.length)];
    votes.set(pick, (votes.get(pick) ?? 0) + 1);
  }
  // Most votes first. The server breaks a tie toward the best-ranked head,
  // which the mock has none of, so it breaks one toward the earlier segment.
  const ranking = [...votes.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, RANKING_LEN)
    .map(([segment, count]) => ({ segment, share: count / VOTERS }));
  const top = ranking[0].segment;
  return {
    type: "locate",
    kind,
    method: "vote",
    compression: "none",
    segment: top,
    value: values[top],
    confidence: ranking[0].share,
    ranking,
    pointers: [{ segment: top, value: values[top], share: ranking[0].share }],
  };
}

/**
 * A `shortlist` over `target` (GitHub #278), faked. The heads' candidates are
 * up to sixteen segments that own a token — never a blank one, and in prose
 * never a `# ` title — drawn by the seed and kept in document order; the
 * labelled choice's probabilities over them have one clear pick and a thin
 * tail, so a question points at one segment or at several. Under a fold every
 * share is scaled by a first level's pick, as the server's are.
 *
 * `found` is on the three measured routes only — `log` + `template_fold`,
 * `prose` + `none`, `records` + `none`. A fifth of those questions come back
 * not found by the seed alone, and every one whose instructions say
 * "/absent" does: no segment, no pointer, and the ranking kept.
 */
function shortlistAnswer(seed: string, target: Target, values: unknown[], kind: ResolvedKind, compression: Compression, absent: boolean) {
  const owning = target.owns.flatMap((own, index) => (own ? [index] : []));
  const readable = kind === "prose" ? owning.filter((index) => !target.segments[index].startsWith("# ")) : owning;
  const candidates = [...(readable.length > 0 ? readable : owning)]
    .sort((a, b) => hashed(seed, 1000 + a) - hashed(seed, 1000 + b))
    .slice(0, SHORTLIST)
    .sort((a, b) => a - b);
  const weights = distribute(seed, candidates.length);
  const scale = compression === "template_fold" ? 0.7 + hashed(seed, 809) * 0.3 : 1;
  const share = (at: number) => Number((weights[at] * scale).toFixed(4));
  // By probability, the earlier candidate first on a tie, as the server orders them.
  const order = candidates.map((_, at) => at).sort((a, b) => weights[b] - weights[a] || a - b);
  const measured = (kind === "log" && compression === "template_fold") || (kind !== "log" && compression === "none");
  const found = !measured
    ? undefined
    : absent || hashed(seed, 881) < 0.2
      ? Number((0.05 + hashed(seed, 883) * 0.4).toFixed(4))
      : Number((0.55 + hashed(seed, 887) * 0.44).toFixed(4));
  const named = found === undefined || found >= FOUND_THRESHOLD;
  const pick = candidates[order[0]];
  return {
    type: "locate",
    kind,
    method: "shortlist",
    compression,
    ...(found === undefined ? {} : { found }),
    segment: named ? pick : null,
    value: named ? values[pick] : null,
    confidence: named ? share(order[0]) : null,
    ranking: order.slice(0, RANKING_LEN).map((at) => ({ segment: candidates[at], share: share(at) })),
    pointers: named
      ? order
          .filter((at, rank) => rank === 0 || share(at) >= POINTER_SHARE)
          .map((at) => ({ segment: candidates[at], value: values[candidates[at]], share: share(at) }))
      : [],
  };
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
    // GitHub #278: a locate's `kind` and `compression` on anything else are
    // refused, never ignored.
    if (kind !== "locate" && entry(question, "kind") !== undefined) {
      return refuse("kind_unsupported", `question ${JSON.stringify(id)} is a ${kind}: \`kind\` names the reading a \`locate\`'s text gets`);
    }
    if (kind !== "locate" && entry(question, "compression") !== undefined) {
      return refuse("compression_unsupported", `question ${JSON.stringify(id)} is a ${kind}: \`compression\` names what a \`locate\`'s text is read as`);
    }
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
    if (kind === "scalar") {
      // Here `digits` is a ceiling and absent asks for the widest run, so the
      // mock picks its own width under it — that is the whole behaviour the
      // tab is being built against.
      const ceiling = Math.max(1, Math.min(15, Number(text(entry(question, "digits"))) || 8));
      const width = 1 + Math.floor(hashed(seed, 313) * ceiling);
      const trace = digitsFor(seed, width);
      // Some of them fractional and fewer of them negative, so the decimal and
      // the sign are both reachable without the card.
      const fraction = width > 1 && hashed(seed, 419) < 0.45 ? 1 : 0;
      const sign = hashed(seed, 523) < 0.25 ? "-" : "";
      // A model writes `0.5` and never `07`: a leading zero only stands before
      // the point. Corrected on the **trace** and not on the spelling, because
      // the card shows both and a digit the text does not carry reads there as
      // the fault it would be.
      if (width - fraction > 1 && trace[0].digit === 0) trace[0] = { ...trace[0], digit: 1 };
      const spelled = trace.map((d) => d.digit).join("");
      const written = `${sign}${spelled.slice(0, width - fraction)}${fraction ? `.${spelled.slice(width - fraction)}` : ""}`;
      // What it wrote, plus the brace that closed the object.
      generated += [...written].length + 1;
      answers[id] = {
        type: "scalar",
        value: Number(written),
        text: written,
        uncertainty: Number(sigmaAt(trace, fraction).toFixed(6)),
        digits: trace,
      };
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
    if (kind === "locate") {
      // Everything a caller can get wrong about a locate refuses the whole
      // request before the first prefill, as it does on the server: its
      // route's own fields first, then the state, then the kind the target
      // resolves to.
      const asked = locateAsk(id, question);
      if (!asked.ok) return refuse(asked.code, asked.message);
      if (!state || isParts(state)) {
        return refuse("locate_needs_json_state", "a `locate` reads the lines of a string or the elements of an array, and this `state` is content parts");
      }
      const withinNode = entry(question, "within");
      const within = withinNode?.kind === "string" ? withinNode.value : "";
      const cut = cutTarget(state, within);
      if (!cut.ok) return refuse(cut.code, cut.message);
      const route = resolveLocate(id, asked.ask, cut.target);
      if (!route.ok) return refuse(route.code, route.message);
      const values = segmentValues(state, within, cut.target);
      answers[id] =
        asked.ask.method === "vote"
          ? voteAnswer(seed, cut.target, values, route.kind)
          : shortlistAnswer(seed, cut.target, values, route.kind, route.compression, instructions.includes("/absent"));
      // A locate generates nothing; its baselines and choices are prefill, not output.
      continue;
    }
    answers[id] = { type: "error", code: "unknown_type", message: `the mock does not answer a ${kind}` };
  }

  return {
    status: 200,
    body: {
      model: "mock-model",
      answers,
      // A readout generates nothing, so this stays 0 unless a number, scalar,
      // point or box was asked for — the same honesty the real endpoint keeps.
      usage: { input_tokens: Math.ceil(raw.length / 3.6), output_tokens: generated },
    },
  };
}
