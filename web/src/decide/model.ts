// The decision the Decide tab is building (GitHub #247): one evidence and an
// ordered list of typed questions, the request body it serializes to, and the
// validation that mirrors `decide.rs`'s refusals so the builder cannot send a
// request the endpoint will answer 422 to.
//
// Questions and a `choice`'s options are **arrays**, never objects. The server
// reads both in order of appearance because each option is bound to the answer
// token at its position, so a different order is a different prompt — and a
// JavaScript object cannot carry that order (`json.ts` says why). The body is
// written through `writeOrdered` for the same reason.

import type { PromptImage } from "../conversation/images.ts";
import {
  asText,
  type JsonEntry,
  type JsonNode,
  jsonArray,
  jsonNull,
  jsonObject,
  jsonString,
  parseOrdered,
  saysNothing,
  writeOrdered,
} from "./json.ts";

/** The seven primitives `decide.rs` serves. */
export const PRIMITIVES = ["noul", "choice", "score", "number", "scalar", "point", "box"] as const;
export type Primitive = (typeof PRIMITIVES)[number];

/** What each primitive answers with, in the tab's own words. */
export const PRIMITIVE_BLURB: Record<Primitive, string> = {
  noul: "Yes or no, as the probability of yes.",
  choice: "One option from a set you declare, with the distribution over all of them.",
  score: "A value across ordered levels — the weighted average, so it can land between two.",
  number: "A whole number, generated one digit at a time into a field you declare.",
  scalar: "A number that ends when it is complete — it may be fractional or negative, and you need not say how wide.",
  point: "A position on the image, in its own pixels.",
  box: "A rectangle on the image, in its own pixels.",
};

/**
 * How a `point` is answered (GitHub #260), and `null` for "whichever the load
 * serves" — which is `head` when the loaded artifact has a calibrated pointing
 * head and `chain` when it has none. The tab cannot know which of those a load
 * is, so the default is sent as an absent field rather than guessed at here,
 * and the answer says which ran.
 */
export const POINT_METHODS = ["head", "chain"] as const;
export type PointMethod = (typeof POINT_METHODS)[number];

/** What each method is for, in the tab's own words. */
export const POINT_METHOD_BLURB: Record<PointMethod, string> = {
  head: "One pass: the calibrated pointing head's attention over the image, with no decode round. Coarse — one image token — and on a labelled target it marks where the label begins, not its centre.",
  chain: "The digit chain: one decode round per digit. Finer than one image token, and it carries a per-digit trace.",
};

/**
 * The primitives whose `digits` is a **width**: a field for a `number`, an
 * axis scale for a `point` or a `box`. A `scalar` generates as they do, but
 * its `digits` is a ceiling it may close early out of — a different field on
 * the question and a different refusal — so it is not one of these.
 */
export const FIXED_WIDTH: Primitive[] = ["number", "point", "box"];
export const hasFixedWidth = (kind: Primitive) => FIXED_WIDTH.includes(kind);
/** `point` and `box` answer in pixels of the submitted image, so they need one. */
export const isSpatial = (kind: Primitive) => kind === "point" || kind === "box";

/** `MAX_OPTIONS` in `decide.rs`: measured this far, not a property of the model. */
export const MAX_OPTIONS = 256;
/** `MIN_SCORE_LEVELS`. */
export const MIN_LEVELS = 2;
/** `numbers::DIGITS`, inclusive. */
export const MIN_DIGITS = 1;
export const MAX_DIGITS = 6;
/** `numbers::DEFAULT_DIGITS`. */
export const DEFAULT_DIGITS = 3;
/**
 * `scalar::DIGITS`, inclusive, and wider than a field's on purpose: a width
 * must be filled and a ceiling need not, so raising it costs a caller who
 * does not reach it nothing. Fifteen is where an `f64` stops carrying a
 * decimal exactly.
 */
export const MAX_CEILING = 15;
/** `scalar::DEFAULT_DIGITS`: what an empty ceiling asks for, which is not the maximum. */
export const DEFAULT_CEILING = 8;

/** One `choice` option: the key the answer comes back under, and what the prompt says it means. */
export type Option = {
  key: string;
  /** Blank sends `null`, which is the server's "the key is its own description". */
  description: string;
};

/**
 * One question under construction.
 *
 * Every primitive's fields are present at once, and only the current kind's
 * are sent. Switching a question from `choice` to `score` and back therefore
 * keeps the options that were already written — the type selector is not a
 * destructive control.
 */
export type Question = {
  /** React's key, and the handle validation reports against. Never sent. */
  uid: string;
  /** The id the answer comes back under. */
  id: string;
  kind: Primitive;
  /** A string in the builder; an object or array when it came from the JSON editor. */
  instructions: JsonNode;
  /** `noul`: what a yes and a no mean. Blank is omitted, which is the server's own default. */
  yes: string;
  no: string;
  /** `choice`, in declared order. */
  options: Option[];
  /** `score`, in level order. */
  levels: string[];
  /** `number`, `point`, `box`: the width of the field, or the scale of the axes. */
  digits: number;
  /**
   * `point`: which method answers it, and `null` for the load's own default.
   * Sent only on a `point` — `decide.rs` refuses `method` on every other
   * primitive, a `box` most pointedly of all, because the head's region is
   * not a box.
   */
  method: PointMethod | null;
  /**
   * `scalar`: at most this many digits, and `null` for the server's own
   * default of `DEFAULT_CEILING`. A ceiling and not a width — the run closes
   * itself as soon as the number is complete — so blank is the honest default
   * for the caller this primitive exists for, who does not know the
   * magnitude.
   */
  ceiling: number | null;
  /** Fields the JSON editor carried that the builder does not edit; re-emitted as they were. */
  extras: JsonEntry[];
};

/** The evidence, in the three shapes this tab offers for `state`. */
export type Evidence =
  | { mode: "text"; text: string }
  | { mode: "json"; text: string }
  | { mode: "image"; images: PromptImage[]; text: string };

export type EvidenceMode = Evidence["mode"];

/**
 * What the evidence shapes *not* in use last held, so switching mode and back
 * is not a loss. Never sent, and reset whenever the whole draft is replaced —
 * a freshly loaded example has no history of its own.
 */
export type Spare = { text: string; json: string; images: PromptImage[]; words: string };

export const EMPTY_SPARE: Spare = { text: "", json: "", images: [], words: "" };

/** `evidence` moved aside, so the mode it is leaving can be returned to. */
export function setAside(spare: Spare, evidence: Evidence): Spare {
  if (evidence.mode === "text") return { ...spare, text: evidence.text };
  if (evidence.mode === "json") return { ...spare, json: evidence.text };
  return { ...spare, images: evidence.images, words: evidence.text };
}

/** The evidence a mode is returned to, out of what was set aside. */
export function restore(spare: Spare, mode: EvidenceMode): Evidence {
  if (mode === "text") return { mode, text: spare.text };
  if (mode === "json") return { mode, text: spare.json };
  return { mode, images: spare.images, text: spare.words };
}

/** The whole request under construction. */
export type Draft = {
  evidence: Evidence;
  questions: Question[];
  /** Top-level fields the JSON editor carried besides `state` and `questions`. */
  extras: JsonEntry[];
};

let counter = 0;
const nextUid = () => `q${++counter}`;

export function newQuestion(kind: Primitive, id: string): Question {
  return {
    uid: nextUid(),
    id,
    kind,
    instructions: jsonString(""),
    yes: "",
    no: "",
    options: kind === "choice" ? [emptyOption(), emptyOption()] : [],
    levels: kind === "score" ? ["", "", ""] : [],
    digits: DEFAULT_DIGITS,
    method: null,
    ceiling: null,
    extras: [],
  };
}

export const emptyOption = (): Option => ({ key: "", description: "" });

/** An id nothing in `questions` is using yet: `answer`, then `answer_2`, … */
export function freeId(questions: Question[], stem: string): string {
  const taken = new Set(questions.map((q) => q.id));
  if (!taken.has(stem)) return stem;
  for (let n = 2; ; n++) if (!taken.has(`${stem}_${n}`)) return `${stem}_${n}`;
}

export const EMPTY_DRAFT: Draft = { evidence: { mode: "text", text: "" }, questions: [], extras: [] };

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * One reason this draft is not sendable.
 *
 * `code` is the refusal `decide.rs` would answer with, so a fault here and a
 * 422 from the server read as the same fault. Two codes have no server
 * counterpart and say so: `blank_question_id` (stricter than the endpoint,
 * which would accept `""` as an id nobody can use) and `invalid_state_json`
 * (this editor's own parse).
 */
export type Fault = {
  code: string;
  message: string;
  /** The question it belongs to, when it belongs to one. */
  uid?: string;
};

/** The thinking controls `decide.rs` answers 422 for: accepted only to be refused. */
const THINKING_FIELDS = ["enable_thinking", "reasoning_effort", "preserve_thinking", "chat_template_kwargs"];

export function validate(draft: Draft): Fault[] {
  const faults: Fault[] = [];

  if (draft.evidence.mode === "json") {
    const parsed = parseOrdered(draft.evidence.text.trim() || "null");
    if (!parsed.ok) {
      faults.push({
        code: "invalid_state_json",
        message: `The evidence is not JSON: ${parsed.error.message} (line ${parsed.error.line}, column ${parsed.error.column}).`,
      });
    }
  }

  for (const extra of draft.extras) {
    if (THINKING_FIELDS.includes(extra.key)) {
      faults.push({
        code: "thinking_refused",
        message: `Remove \`${extra.key}\`: a decision's prompt ends where its answer is read, so the endpoint refuses a request that asks for thinking.`,
      });
    }
  }

  if (draft.questions.length === 0) {
    faults.push({ code: "no_questions", message: "Add a question: a decision with nothing to decide is refused." });
    return faults;
  }

  const seen = new Set<string>();
  for (const question of draft.questions) {
    const name = question.id.trim() === "" ? "This question" : `Question "${question.id}"`;
    if (question.id.trim() === "") {
      faults.push({ code: "blank_question_id", message: "Name this question: its answer comes back under the name.", uid: question.uid });
    } else if (seen.has(question.id)) {
      faults.push({ code: "duplicate_question", message: `${name} is named twice; each answer needs its own name.`, uid: question.uid });
    }
    seen.add(question.id);

    if (saysNothing(question.instructions)) {
      faults.push({ code: "empty_instructions", message: `${name} asks nothing: write what the model should decide.`, uid: question.uid });
    }

    faults.push(...methodFaults(question, name));
    if (question.kind === "choice") faults.push(...choiceFaults(question, name));
    if (question.kind === "score") faults.push(...scoreFaults(question, name));
    if (hasFixedWidth(question.kind)) faults.push(...digitFaults(question, name));
    if (question.kind === "scalar") faults.push(...ceilingFaults(question, name));
    if (isSpatial(question.kind) && !hasImage(draft.evidence)) {
      faults.push({
        code: "state_carries_no_image",
        message: `${name} answers in pixels of an image, and the evidence carries none. Switch the evidence to Image.`,
        uid: question.uid,
      });
    }
  }
  return faults;
}

/**
 * `method` on a question that has one way of being answered.
 *
 * The builder only offers the control on a `point`, so what this catches is a
 * body written in the JSON editor — where a `method` on a `box` is the
 * mistake worth naming, because it reads as if the head could draw one.
 */
function methodFaults(question: Question, name: string): Fault[] {
  const faults: Fault[] = [];
  // Only `readQuestion` can put a `method` here, and only one the two names
  // do not cover.
  const unknown = question.extras.find((entry) => entry.key === "method");
  if (unknown) {
    faults.push({
      code: "method_unknown",
      message: `${name} asks for the method ${JSON.stringify(asText(unknown.value))}; the accepted values are "head" and "chain".`,
      uid: question.uid,
    });
  }
  if (question.method !== null && question.kind !== "point") {
    const why =
      question.kind === "box"
        ? "a box cannot come out of the pointing head — its region is not a box — so a box is always the chain's"
        : "`method` chooses how a point is answered, and this primitive has one way";
    faults.push({ code: "method_unsupported", message: `${name} is a ${question.kind}: ${why}.`, uid: question.uid });
  }
  return faults;
}

function choiceFaults(question: Question, name: string): Fault[] {
  const faults: Fault[] = [];
  const keys = question.options.map((o) => o.key.trim()).filter((key) => key !== "");
  if (question.options.length === 0) {
    faults.push({ code: "no_options", message: `${name} declares no options.`, uid: question.uid });
  }
  if (keys.length < question.options.length) {
    faults.push({ code: "unclean_option", message: `${name} has an option with no name; the name is what the answer comes back as.`, uid: question.uid });
  }
  const duplicate = keys.find((key, index) => keys.indexOf(key) !== index);
  if (duplicate !== undefined) {
    faults.push({ code: "duplicate_option", message: `${name} declares the option "${duplicate}" twice.`, uid: question.uid });
  }
  if (question.options.length > MAX_OPTIONS) {
    faults.push({
      code: "too_many_options",
      message: `${name} declares ${question.options.length} options; ${MAX_OPTIONS} is as far as this has been measured.`,
      uid: question.uid,
    });
  }
  return faults;
}

function scoreFaults(question: Question, name: string): Fault[] {
  const faults: Fault[] = [];
  if (question.levels.length < MIN_LEVELS) {
    faults.push({
      code: "too_few_levels",
      message: `${name} declares ${question.levels.length} level${question.levels.length === 1 ? "" : "s"}; a score needs at least ${MIN_LEVELS}.`,
      uid: question.uid,
    });
  }
  if (question.levels.some((level) => level.trim() === "")) {
    faults.push({ code: "malformed_criteria", message: `${name} has a level with no description.`, uid: question.uid });
  }
  return faults;
}

/** The widths `numbers::DIGITS` serves, which a scalar's ceiling shares. */
const inDigitRange = (digits: number) => Number.isInteger(digits) && digits >= MIN_DIGITS && digits <= MAX_DIGITS;

function digitFaults(question: Question, name: string): Fault[] {
  if (inDigitRange(question.digits)) return [];
  return [
    {
      code: "digits_out_of_range",
      message: `${name} asks for ${question.digits} digits; ${MIN_DIGITS} to ${MAX_DIGITS} is the range served.`,
      uid: question.uid,
    },
  ];
}

/**
 * A `scalar`'s ceiling, which is allowed to be absent — that is the whole
 * point of the primitive — and which reaches further than a `number`'s width,
 * because a ceiling need not be filled. `decide.rs` refuses the rest under
 * the same code as a width.
 */
function ceilingFaults(question: Question, name: string): Fault[] {
  const { ceiling } = question;
  if (ceiling === null || (Number.isInteger(ceiling) && ceiling >= MIN_DIGITS && ceiling <= MAX_CEILING)) return [];
  return [
    {
      code: "digits_out_of_range",
      message: `${name} allows at most ${ceiling} digits; ${MIN_DIGITS} to ${MAX_CEILING} is the range a scalar serves.`,
      uid: question.uid,
    },
  ];
}

export const hasImage = (evidence: Evidence) => evidence.mode === "image" && evidence.images.length > 0;

/** The image a `point` or `box` answers against — the first one, as the prompt carries it. */
export const evidenceImage = (evidence: Evidence): PromptImage | null =>
  evidence.mode === "image" ? (evidence.images[0] ?? null) : null;

// ---------------------------------------------------------------------------
// The request body
// ---------------------------------------------------------------------------

/**
 * The draft as the request body, entries in the order they are written here.
 *
 * `model` is not sent: the endpoint routes to the loaded model when it is
 * absent, and this tab has exactly one to talk to.
 */
export function requestNode(draft: Draft): JsonNode {
  return jsonObject([
    { key: "state", value: stateNode(draft.evidence) },
    ...draft.extras,
    { key: "questions", value: jsonObject(draft.questions.map((q) => ({ key: q.id, value: questionNode(q) }))) },
  ]);
}

export const requestBody = (draft: Draft): string => writeOrdered(requestNode(draft), 2);

function stateNode(evidence: Evidence): JsonNode {
  if (evidence.mode === "text") return jsonString(evidence.text);
  if (evidence.mode === "json") {
    const parsed = parseOrdered(evidence.text.trim() || "null");
    // An unparseable state is a fault `validate` already reports; writing the
    // text verbatim keeps the editor's own bytes on the page instead of
    // inventing a value nobody typed.
    return parsed.ok ? parsed.node : jsonString(evidence.text);
  }
  // Content parts: the images first and the text behind them, the order the
  // chat path and `classify_vision_readout_gpu.rs` both send.
  const parts: JsonNode[] = evidence.images.map((image) =>
    jsonObject([
      { key: "type", value: jsonString("image_url") },
      { key: "image_url", value: jsonObject([{ key: "url", value: jsonString(image.url) }]) },
    ]),
  );
  if (evidence.text.trim() !== "") {
    parts.push(jsonObject([{ key: "type", value: jsonString("text") }, { key: "text", value: jsonString(evidence.text) }]));
  }
  return jsonArray(parts);
}

function questionNode(question: Question): JsonNode {
  const entries: JsonEntry[] = [
    { key: "type", value: jsonString(question.kind) },
    { key: "instructions", value: question.instructions },
  ];
  const criteria = criteriaNode(question);
  if (criteria) entries.push({ key: "criteria", value: criteria });
  if (hasFixedWidth(question.kind)) entries.push({ key: "digits", value: { kind: "number", value: question.digits } });
  // A `point` alone carries `method`, and only when one was chosen: absent is
  // what the server reads as "whichever this load serves", and no value says
  // that.
  if (question.kind === "point" && question.method !== null) {
    entries.push({ key: "method", value: jsonString(question.method) });
  }
  // A scalar's ceiling is omitted when it has none: absent is what the server
  // reads as "the widest run you serve", and there is no value to write that
  // says it.
  if (question.kind === "scalar" && question.ceiling !== null) {
    entries.push({ key: "digits", value: { kind: "number", value: question.ceiling } });
  }
  return jsonObject([...entries, ...question.extras]);
}

function criteriaNode(question: Question): JsonNode | null {
  switch (question.kind) {
    case "noul": {
      // A blank description is omitted rather than sent empty: absent is what
      // the server reads as "use the default", and an empty string is the one
      // shape it refuses.
      const entries: JsonEntry[] = [];
      if (question.yes.trim() !== "") entries.push({ key: "true", value: jsonString(question.yes) });
      if (question.no.trim() !== "") entries.push({ key: "false", value: jsonString(question.no) });
      return entries.length ? jsonObject(entries) : null;
    }
    case "choice":
      return jsonObject(
        question.options.map((option) => ({
          key: option.key,
          // `null` is Jev's own "this option needs no extra detail", and the
          // key becomes its own description.
          value: option.description.trim() === "" ? jsonNull : jsonString(option.description),
        })),
      );
    case "score":
      return jsonArray(question.levels.map(jsonString));
    default:
      // A constrained decode declares no options; `criteria` on one is refused.
      return null;
  }
}

// ---------------------------------------------------------------------------
// Reading a body back into the builder
// ---------------------------------------------------------------------------

export type Read = { ok: true; draft: Draft } | { ok: false; message: string };

/** A request body as a draft, or why it is not one. Order survives: `json.ts` parses, not `JSON.parse`. */
export function readRequest(text: string): Read {
  const parsed = parseOrdered(text);
  if (!parsed.ok) return { ok: false, message: `${parsed.error.message} (line ${parsed.error.line}, column ${parsed.error.column})` };
  if (parsed.node.kind !== "object") return { ok: false, message: "the request must be a JSON object" };

  const entries = parsed.node.entries;
  const questionsEntry = entries.find((e) => e.key === "questions");
  if (!questionsEntry) return { ok: false, message: "the request needs a `questions` object" };
  if (questionsEntry.value.kind !== "object") return { ok: false, message: "`questions` must be a JSON object of id to question" };
  const stateEntry = entries.find((e) => e.key === "state");

  const questions: Question[] = [];
  for (const entry of questionsEntry.value.entries) {
    const read = readQuestion(entry.key, entry.value);
    if (!read.ok) return read;
    questions.push(read.question);
  }

  return {
    ok: true,
    draft: {
      evidence: readEvidence(stateEntry?.value),
      questions,
      // `model` is dropped: this tab talks to the loaded model, and keeping a
      // stale name would send a request to something that is not there.
      extras: entries.filter((e) => e.key !== "state" && e.key !== "questions" && e.key !== "model"),
    },
  };
}

function readEvidence(node: JsonNode | undefined): Evidence {
  if (!node) return { mode: "text", text: "" };
  if (node.kind === "string") return { mode: "text", text: node.value };
  if (node.kind === "array" && node.items.every(isContentPart)) {
    const images: PromptImage[] = [];
    let text = "";
    for (const item of node.items) {
      if (item.kind !== "object") continue;
      const type = item.entries.find((e) => e.key === "type")?.value;
      if (type?.kind === "string" && type.value === "text") {
        const value = item.entries.find((e) => e.key === "text")?.value;
        if (value?.kind === "string") text = text ? `${text}\n${value.value}` : value.value;
        continue;
      }
      const url = item.entries.find((e) => e.key === "image_url")?.value;
      const inner = url?.kind === "object" ? url.entries.find((e) => e.key === "url")?.value : undefined;
      if (inner?.kind === "string") images.push({ name: "evidence", url: inner.value, width: 0, height: 0 });
    }
    return { mode: "image", images, text };
  }
  return { mode: "json", text: writeOrdered(node, 2) };
}

/** Whether every item looks like an OpenAI content part, which is what makes the list an image state. */
function isContentPart(node: JsonNode): boolean {
  if (node.kind !== "object") return false;
  const type = node.entries.find((e) => e.key === "type")?.value;
  return type?.kind === "string" && (type.value === "text" || type.value === "image_url");
}

type ReadQuestion = { ok: true; question: Question } | { ok: false; message: string };

function readQuestion(id: string, node: JsonNode): ReadQuestion {
  if (node.kind !== "object") return { ok: false, message: `question ${JSON.stringify(id)} must be a JSON object` };
  const at = (key: string) => node.entries.find((e) => e.key === key)?.value;
  const typeNode = at("type");
  if (typeNode?.kind !== "string") return { ok: false, message: `question ${JSON.stringify(id)} needs a \`type\`` };
  const named = typeNode.value;
  const kind = ALIASES[named] ?? ((PRIMITIVES as readonly string[]).includes(named) ? (named as Primitive) : null);
  if (!kind) return { ok: false, message: `question ${JSON.stringify(id)} has the unknown type ${JSON.stringify(named)}` };
  const question = newQuestion(kind, id);
  const instructions = at("instructions") ?? at("question");
  if (instructions) question.instructions = instructions;
  const digits = at("digits");
  // The same wire field lands on a different question field for a scalar,
  // because there it is a ceiling and not a width.
  if (digits?.kind === "number") {
    if (kind === "scalar") question.ceiling = digits.value;
    else question.digits = digits.value;
  }
  // A spelling neither method covers is kept as it was written rather than
  // dropped: the server refuses it naming the two it accepts, and a body that
  // round-trips through this editor has to still earn that refusal.
  const method = at("method");
  if (method?.kind === "string" && (POINT_METHODS as readonly string[]).includes(method.value)) {
    question.method = method.value as PointMethod;
  } else if (method) {
    question.extras = [{ key: "method", value: method }];
  }
  const criteria = at("criteria") ?? at("options");
  if (criteria) applyCriteria(question, criteria);
  // Anything else the caller wrote stays on the question and goes back out as
  // it came in, so a round-trip through this editor loses nothing.
  question.extras = [...question.extras, ...node.entries.filter((e) => !READ_KEYS.includes(e.key))];
  return { ok: true, question };
}

/** `decide.rs`'s serde aliases: their names and ours are the same field. */
const ALIASES: Record<string, Primitive | undefined> = { boolean: "noul" };
const READ_KEYS = ["type", "instructions", "question", "criteria", "options", "digits", "method"];

function applyCriteria(question: Question, criteria: JsonNode) {
  if (question.kind === "score" && criteria.kind === "array") {
    question.levels = criteria.items.map(asText);
    return;
  }
  if (criteria.kind !== "object") return;
  if (question.kind === "noul") {
    const yes = criteria.entries.find((e) => e.key === "true")?.value;
    const no = criteria.entries.find((e) => e.key === "false")?.value;
    question.yes = yes && yes.kind === "string" ? yes.value : "";
    question.no = no && no.kind === "string" ? no.value : "";
    return;
  }
  question.options = criteria.entries.map((entry) => ({
    key: entry.key,
    description: entry.value.kind === "null" ? "" : asText(entry.value),
  }));
}

