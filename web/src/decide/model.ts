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
import { type Cut, cutTarget } from "./locate.ts";

/** The eight primitives `decide.rs` serves. */
export const PRIMITIVES = ["noul", "choice", "score", "number", "scalar", "point", "box", "locate"] as const;
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
  locate: "Which line of a text, sentence of a document or element of a list answers the instruction — the calibrated heads narrow it to a few candidates and a labelled choice picks one, with nothing generated or written into the evidence.",
};

/**
 * How a `point` or a `box` is answered (GitHub #260, #263) — `decide.rs`'s
 * `SpatialMethod` — and `null` for "whichever the load serves".
 *
 * The two primitives do not default alike, which is why the tab never guesses
 * the absent value: a `point` is `head` on a load with a calibrated pointing
 * head and `chain` on one without, and a `box` is `chain` on every load
 * (`BOX_DEFAULT_METHOD`), because the head box was not better than the chain
 * on a small button. So `head` is a `box`'s opt-in, and the answer says which
 * ran either way.
 */
export const SPATIAL_METHODS = ["head", "chain"] as const;
export type SpatialMethod = (typeof SPATIAL_METHODS)[number];

/**
 * What each method is for, in the tab's own words, and per primitive: the
 * same name does a different thing on each. A head `point` reads a position
 * off the heads; a head `box` reads the set's extent, and is the answer you
 * have to ask for.
 */
export const SPATIAL_METHOD_BLURB: Record<"point" | "box", Record<SpatialMethod, string>> = {
  point: {
    head: "One pass: the calibrated heads' attention over the image, with no decode round. Where the load has a head set the point is the centre of the object the set outlines; with the pointing head alone it is coarser — one image token — and on a labelled target it marks where the label begins, not the centre.",
    chain: "The digit chain: one decode round per digit. Finer than one image token, and it carries a per-digit trace.",
  },
  box: {
    head: "One pass: the head set's extent around the pointing head's point, with no decode round, in the same pixels and on the same scale as the chain's. A load with no calibrated head set refuses it rather than answering by the chain.",
    chain: "The digit chain: one decode round per digit, four edges' worth, and it carries a per-digit trace. This is what a box answers with unless it asks for the head.",
  },
};

/**
 * What an absent `method` means, per primitive — the empty choice's own label
 * and the sentence under it.
 *
 * A `point`'s default is the **load's**, which this tab cannot resolve; a
 * `box`'s is the endpoint's, and is `chain` whatever the load is. Naming them
 * the same way would say a box is load-dependent when it is not.
 */
export const DEFAULT_METHOD: Record<"point" | "box", { label: string; blurb: string }> = {
  point: {
    label: "this load's own",
    blurb:
      "The head where the loaded artifact has a calibrated pointing head, and the chain where it has none. Asking for head on a load that has none is refused rather than answered by the chain.",
  },
  box: {
    label: "the endpoint's",
    blurb:
      "The chain, on every load: the head box was measured no better than it on a small button, so the head is a box's opt-in. Asking for head on a load with no calibrated head set is refused rather than answered by the chain.",
  },
};

/**
 * A `locate`'s route (GitHub #278, spec 22): which reading its text gets
 * (`kind`), how it is answered (`method`) and what its text is read as
 * (`compression`) — three enums on the wire, each optional.
 *
 * `null` on the question is the endpoint's default, sent as no field at all,
 * and it is the empty choice of the selector, as a point's `method` is. For
 * `kind` and `method` that default has a name of its own — `auto`,
 * `shortlist` — so the empty choice is labelled with it and the name is not
 * offered a second time: `prepare_locate` reads the name and the absence in
 * one arm, so they are one request. `compression` has no fixed default — it
 * follows the kind the text resolves to — so its empty choice says that, and
 * both values are offered.
 */
export const LOCATE_KINDS = ["auto", "log", "prose", "records"] as const;
export type LocateKind = (typeof LOCATE_KINDS)[number];
/** A kind an answer can name: the one `auto` told, never `auto` itself. */
export type ResolvedKind = Exclude<LocateKind, "auto">;
export const LOCATE_METHODS = ["shortlist", "vote"] as const;
export type LocateMethod = (typeof LOCATE_METHODS)[number];
export const COMPRESSIONS = ["template_fold", "none"] as const;
export type Compression = (typeof COMPRESSIONS)[number];

/** What each kind reads, in the tab's own words; `auto`'s is the empty choice's. */
export const LOCATE_KIND_BLURB: Record<LocateKind, string> = {
  auto: "Told for you, and named in the answer: records for an array of two or more JSON objects; otherwise the text is folded, and it is a log when at least half of its first 2,000 non-blank segments share a template, prose when fewer do.",
  log: "Lines of a log: the end heads read each line where it ends, and the log is folded into templates first unless you say otherwise, so a million tokens answer in seconds. Refused on an array of JSON objects, which is read as records.",
  prose: "Sentences of a document: the sum heads read every key of a sentence, the choice sees each candidate inside its paragraph, and a two-part answer comes back as several pointers. Never folded; refused on an array of JSON objects.",
  records: "The elements of an array of JSON objects, each shown to the choice as one line of JSON. Refused on any other target.",
};

/** What each method does; `shortlist`'s is the empty choice's. */
export const LOCATE_METHOD_BLURB: Record<LocateMethod, string> = {
  shortlist:
    "The default: the calibrated heads narrow the text to a few candidates and a labelled choice picks among them, with nothing generated. The shares are that choice's probabilities, and on the three measured routes it also says whether anything answers at all.",
  vote: "The head vote served before: one prefill over the whole target, at most 4,554 tokens on the served artifact, and the shares are the heads' votes. It always names a segment, carries no found, and reads the text as it is, so a fold is refused under it.",
};

/** What each compression does, and what the empty choice leaves it to. */
export const COMPRESSION_BLURB: Record<Compression, string> = {
  template_fold:
    "Fold first: templates and their values, so nothing near the text's length is prefilled. A log's default; a records array folds too — faster, less often right, and with no found. A fold drops the lines' order, their neighbours and every time, and it is refused on prose and under a vote.",
  none: "Read the text as it is, in windows of 200,000 tokens past that, keeping the context across lines that a fold removes. The first question pays each window's prefill; a log read this way carries no found.",
};

export const DEFAULT_COMPRESSION = {
  label: "the kind's own",
  blurb: "Whatever the kind reads best with: a log is folded into templates, prose and records are read as they are. A vote reads every text as it is.",
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
   * `point`, `box`: which method answers it, and `null` for the default the
   * endpoint would pick. Sent only from those two — `decide.rs` refuses
   * `method` on every other primitive, which reads one position and has one
   * way of reading it — and kept on the question regardless, so switching a
   * question's type and switching back does not lose the choice.
   */
  method: SpatialMethod | null;
  /**
   * `scalar`: at most this many digits, and `null` for the server's own
   * default of `DEFAULT_CEILING`. A ceiling and not a width — the run closes
   * itself as soon as the number is complete — so blank is the honest default
   * for the caller this primitive exists for, who does not know the
   * magnitude.
   */
  ceiling: number | null;
  /**
   * `locate`: the JSON Pointer to the part of the evidence it searches, and
   * `""` for the whole of it — the server's absent field, so an empty one is
   * not sent. Kept on the question whatever its type, like `method`, and sent
   * only from a `locate`: `decide.rs` refuses `within` on every other type.
   */
  within: string;
  /**
   * `locate` (GitHub #278): the kind of text it reads, and `null` for `auto`.
   * Not `kind`, which is the primitive: the server's own field is `text_kind`,
   * renamed `kind` on the wire. Kept whatever the type, like `within`, and
   * sent only from a `locate`: `decide.rs` refuses `kind` on every other one.
   */
  textKind: ResolvedKind | null;
  /**
   * `locate`: `vote`, and `null` for the endpoint's `shortlist`. It is the
   * wire's `method`, and a field apart from a point's all the same — the key
   * names another vocabulary there — as a scalar's ceiling is a field apart
   * from a number's width.
   */
  locateMethod: Exclude<LocateMethod, "shortlist"> | null;
  /** `locate`: `template_fold` or `none`, and `null` for the one its kind reads best with. */
  compression: Compression | null;
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
    within: "",
    textKind: null,
    locateMethod: null,
    compression: null,
    extras: [],
  };
}

export const emptyOption = (): Option => ({ key: "", description: "" });

/** The wire key each of a locate's route fields is sent under. */
const ROUTE_KEY = { textKind: "kind", locateMethod: "method", compression: "compression" } as const;

/**
 * `question` with one of a locate's route fields chosen (GitHub #278), and
 * whatever the JSON editor kept under the same wire key dropped: a value no
 * name covered is replaced by the one chosen, not sent beside it as a second
 * `method`.
 */
export function chooseRoute<K extends keyof typeof ROUTE_KEY>(question: Question, field: K, value: Question[K]): Question {
  return { ...question, [field]: value, extras: question.extras.filter((entry) => entry.key !== ROUTE_KEY[field]) };
}

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
    faults.push(...routeFieldFaults(question, name));
    if (question.kind === "choice") faults.push(...choiceFaults(question, name));
    if (question.kind === "score") faults.push(...scoreFaults(question, name));
    if (hasFixedWidth(question.kind)) faults.push(...digitFaults(question, name));
    if (question.kind === "scalar") faults.push(...ceilingFaults(question, name));
    if (question.kind === "locate") faults.push(...locateFaults(draft.evidence, question, name));
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
 * A `method` spelling the question's own type does not name.
 *
 * A known `method` on a primitive that does not take it is not a fault: it is
 * kept on the question, so a trip through another type and back does not lose
 * it, and ignored there — `requestBody` sends a point's only from a point or
 * a box, and a locate's only from a locate.
 */
function methodFaults(question: Question, name: string): Fault[] {
  const faults: Fault[] = [];
  // Only `readQuestion` can put a `method` here, and only one the question's
  // type does not name: a locate's are `shortlist` and `vote` (GitHub #278),
  // everything else's `head` and `chain`.
  const unknown = question.extras.find((entry) => entry.key === "method");
  if (unknown) {
    const accepted = question.kind === "locate" ? '"shortlist" and "vote"' : '"head" and "chain"';
    faults.push({
      code: "method_unknown",
      message: `${name} asks for the method ${JSON.stringify(asText(unknown.value))}; the accepted values are ${accepted}.`,
      uid: question.uid,
    });
  }
  return faults;
}

/**
 * A `kind` or a `compression` no value covers (GitHub #278), which only
 * `readQuestion` can put here, kept as written so the body still earns the
 * server's refusal: an unknown value on a `locate`, and the field itself on
 * every other type — `decide.rs` never ignores a field a caller wrote.
 */
function routeFieldFaults(question: Question, name: string): Fault[] {
  const locate = question.kind === "locate";
  const faults: Fault[] = [];
  for (const { key, value } of question.extras) {
    const written = JSON.stringify(asText(value));
    if (key === "kind") {
      faults.push(
        locate
          ? {
              code: "kind_unknown",
              message: `${name} asks for the kind ${written}; the accepted values are "auto", "log", "prose" and "records".`,
              uid: question.uid,
            }
          : { code: "kind_unsupported", message: `${name} is a ${question.kind}: \`kind\` names the reading a locate's text gets.`, uid: question.uid },
      );
    }
    if (key === "compression") {
      faults.push(
        locate
          ? {
              code: "compression_unknown",
              message: `${name} asks for the compression ${written}; the accepted values are "template_fold" and "none".`,
              uid: question.uid,
            }
          : {
              code: "compression_unsupported",
              message: `${name} is a ${question.kind}: \`compression\` names what a locate's text is read as.`,
              uid: question.uid,
            },
      );
    }
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

/**
 * A `locate`'s own refusals, reported before the send: the target it names
 * has to exist, be a string or a non-empty array, and hold two segments with
 * text in them. A JSON evidence that does not parse is already a fault of its
 * own, and cutting it would only say the same thing twice.
 *
 * And its route's (GitHub #278): the two combinations that never apply — a
 * fold under a vote, a fold of prose — and a named kind the target
 * contradicts. One refusal stays the server's: a fold of a text `auto` tells
 * is prose, which only the fold itself can say.
 */
function locateFaults(evidence: Evidence, question: Question, name: string): Fault[] {
  const faults: Fault[] = [];
  const fault = (code: string, message: string) => faults.push({ code, message: `${name} ${message}`, uid: question.uid });
  if (question.compression === "template_fold" && question.locateMethod === "vote") {
    fault(
      "compression_unsupported",
      'asks for a vote over a fold: the vote reads the text as it is — over a fold it read 28 of 58 real-log questions — so "template_fold" is refused with "vote". Ask for the shortlist, or for "none".',
    );
  }
  if (question.compression === "template_fold" && question.textKind === "prose") {
    fault("compression_unsupported", 'asks to fold prose, which does not fold into templates. Ask for "none", or leave the compression to the kind.');
  }
  const cut = locateTarget(evidence, question.within);
  if (cut === null) return faults;
  if (!cut.ok) return [...faults, { code: cut.code, message: `${name}: ${cut.message}`, uid: question.uid }];
  if (question.textKind === "records" && !cut.target.records) {
    fault("kind_mismatch", 'names kind "records", and its target is not an array of two or more JSON objects. Ask for "log" or "prose", or leave it to auto.');
  }
  if ((question.textKind === "log" || question.textKind === "prose") && cut.target.records) {
    fault(
      "kind_mismatch",
      `names kind "${question.textKind}", and its target is an array of JSON objects, which is read as records. To fold it, ask for kind "records" with compression "template_fold".`,
    );
  }
  return faults;
}

/**
 * The target a `locate` would read in this evidence, cut into its segments —
 * or the refusal, or `null` when the evidence is JSON that does not parse.
 *
 * Text evidence is a JSON string on the wire, so its segments are its lines.
 * Image evidence is content parts, which a `locate` refuses whatever `within`
 * says.
 */
export function locateTarget(evidence: Evidence, within: string): Cut | null {
  if (evidence.mode === "image") {
    return {
      ok: false,
      code: "locate_needs_json_state",
      message: "a locate reads the lines of a text or the elements of a JSON array, and this evidence is an image. Switch the evidence to Text or JSON.",
    };
  }
  if (evidence.mode === "text") return cutTarget(jsonString(evidence.text), within);
  const parsed = parseOrdered(evidence.text.trim() || "null");
  return parsed.ok ? cutTarget(parsed.node, within) : null;
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
  // A `point` and a `box` carry `method`, and only when one was chosen:
  // absent is what the server reads as "the default for this primitive on
  // this load", and no value says that.
  if (isSpatial(question.kind) && question.method !== null) {
    entries.push({ key: "method", value: jsonString(question.method) });
  }
  // A scalar's ceiling is omitted when it has none: absent is what the server
  // reads as "the widest run you serve", and there is no value to write that
  // says it.
  if (question.kind === "scalar" && question.ceiling !== null) {
    entries.push({ key: "digits", value: { kind: "number", value: question.ceiling } });
  }
  // The whole evidence is what an absent `within` searches, and no pointer
  // needs writing to say it.
  if (question.kind === "locate" && question.within !== "") {
    entries.push({ key: "within", value: jsonString(question.within) });
  }
  // A locate's route (GitHub #278), each field only when one was chosen:
  // absent is `auto`, the `shortlist`, and the compression the kind the text
  // resolves to reads best with — none of which a value written here could
  // say without pinning a default the endpoint owns.
  if (question.kind === "locate") {
    if (question.textKind !== null) entries.push({ key: "kind", value: jsonString(question.textKind) });
    if (question.locateMethod !== null) entries.push({ key: "method", value: jsonString(question.locateMethod) });
    if (question.compression !== null) entries.push({ key: "compression", value: jsonString(question.compression) });
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
  // A spelling the question's type does not name is kept as it was written
  // rather than dropped: the server refuses it naming the two it accepts, and
  // a body that round-trips through this editor has to still earn that
  // refusal. Which two depends on the type (GitHub #278): a locate's
  // `shortlist` and `vote`, everything else's `head` and `chain` — so a
  // `head` on a locate is kept, not read as a point's and left off the wire.
  const method = at("method");
  if (method?.kind === "string" && kind === "locate" && isOneOf(LOCATE_METHODS, method.value)) {
    // `shortlist` is the empty choice: the server reads it and an absent
    // field alike.
    question.locateMethod = method.value === "shortlist" ? null : method.value;
  } else if (method?.kind === "string" && kind !== "locate" && isOneOf(SPATIAL_METHODS, method.value)) {
    question.method = method.value;
  } else if (method) {
    question.extras = [{ key: "method", value: method }];
  }
  // A pointer that is not a string is kept as written, for the server to
  // refuse, as a `method` neither name covers is.
  const within = at("within");
  if (within?.kind === "string") question.within = within.value;
  else if (within) question.extras = [...question.extras, { key: "within", value: within }];
  // A locate's `kind` and `compression` (GitHub #278) are read whatever the
  // type, as `within` is, and a value neither covers is kept as written.
  // `auto` is the empty choice, as `shortlist` is.
  const textKind = at("kind");
  if (textKind?.kind === "string" && isOneOf(LOCATE_KINDS, textKind.value)) {
    question.textKind = textKind.value === "auto" ? null : textKind.value;
  } else if (textKind) {
    question.extras = [...question.extras, { key: "kind", value: textKind }];
  }
  const compression = at("compression");
  if (compression?.kind === "string" && isOneOf(COMPRESSIONS, compression.value)) question.compression = compression.value;
  else if (compression) question.extras = [...question.extras, { key: "compression", value: compression }];
  const criteria = at("criteria") ?? at("options");
  if (criteria) applyCriteria(question, criteria);
  // Anything else the caller wrote stays on the question and goes back out as
  // it came in, so a round-trip through this editor loses nothing.
  question.extras = [...question.extras, ...node.entries.filter((e) => !READ_KEYS.includes(e.key))];
  return { ok: true, question };
}

/** `decide.rs`'s serde aliases: their names and ours are the same field. */
const ALIASES: Record<string, Primitive | undefined> = { boolean: "noul" };
const READ_KEYS = ["type", "instructions", "question", "criteria", "options", "digits", "method", "within", "kind", "compression"];

/** Whether `value` is one of `values`, narrowing it to their type. */
const isOneOf = <T extends string>(values: readonly T[], value: string): value is T => (values as readonly string[]).includes(value);

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

