// A `locate`'s target, cut the way the server cuts it (GitHub #277, #275).
//
// `crate::locate::evidence_within` resolves `within` (an RFC 6901 pointer)
// against the `state` and segments what it names: a string's lines, split on
// `\n` exactly, or a non-empty array's elements. The answer names a segment
// by its index and nothing else, so the tab has to cut the target itself to
// show a ranking as text — and cutting it before the send is also what lets
// the builder report the five refusals the endpoint would answer 422 with.
//
// Read through the ordered tree, not `JSON.parse`: a key the state writes
// twice is a refusal (`locate_within_ambiguous`), and a JavaScript object
// would keep one copy and hide the fault.

import { type JsonNode, writeOrdered } from "./json.ts";

/** What a segment is called: a string's lines, an array's items. */
export type Unit = "line" | "item";

/** The target a `locate` reads, cut into its segments. */
export type Target = {
  unit: Unit;
  /** Each segment as the caller sent it: a line as a string, an element as its JSON. */
  segments: string[];
  /** Whether each segment owns a token: an empty or all-whitespace one keeps its index and owns none. */
  owns: boolean[];
};

export type Cut = { ok: true; target: Target } | { ok: false; code: string; message: string };

/** The fewest key-owning segments a `locate` is served with. */
export const MIN_SEGMENTS = 2;

/**
 * `state` cut at `within`, or the refusal the endpoint would answer.
 *
 * `within` is sent as written — the empty string is the root, the server's
 * absent field — so a pointer with stray spaces is refused here exactly as
 * there.
 */
export function cutTarget(state: JsonNode, within: string): Cut {
  const resolved = resolve(state, within);
  if (!resolved.ok) return resolved;
  const target = resolved.node;
  if (target.kind === "string") {
    const lines = target.value.split("\n");
    return counted({ unit: "line", segments: lines, owns: lines.map((line) => line.trim() !== "") });
  }
  if (target.kind === "array" && target.items.length > 0) {
    return counted({
      unit: "item",
      segments: target.items.map((item) => (item.kind === "string" ? item.value : writeOrdered(item, 0))),
      owns: target.items.map((item) => item.kind !== "string" || item.value.trim() !== ""),
    });
  }
  return {
    ok: false,
    code: "locate_target_unsegmentable",
    message: `\`within\` ${where(within)} names ${kindOf(target)}; a locate reads the lines of a string or the elements of a non-empty array.`,
  };
}

function counted(target: Target): Cut {
  const owning = target.owns.filter(Boolean).length;
  if (owning >= MIN_SEGMENTS) return { ok: true, target };
  return {
    ok: false,
    code: "locate_too_few_segments",
    message: `The target has ${owning} ${target.unit}${owning === 1 ? "" : "s"} with any text in it; a locate needs at least ${MIN_SEGMENTS} to choose between.`,
  };
}

/** How a pointer reads in a message: the root has no spelling of its own. */
const where = (within: string) => (within === "" ? "(the whole state)" : JSON.stringify(within));

type Resolved = { ok: true; node: JsonNode } | { ok: false; code: string; message: string };

/** RFC 6901, refusing a key written twice rather than choosing a copy — `resolve` in `locate.rs`. */
function resolve(state: JsonNode, pointer: string): Resolved {
  if (pointer === "") return { ok: true, node: state };
  const malformed: Resolved = {
    ok: false,
    code: "locate_within_malformed",
    message: `\`within\` ${where(pointer)} is not a JSON Pointer: it must be empty or start with /, and ~ may only be written ~0 or ~1.`,
  };
  if (!pointer.startsWith("/")) return malformed;
  const notFound: Resolved = { ok: false, code: "locate_within_not_found", message: `\`within\` ${where(pointer)} names nothing in the evidence.` };
  let at = state;
  for (const raw of pointer.slice(1).split("/")) {
    const token = unescape(raw);
    if (token === null) return malformed;
    if (at.kind === "object") {
      const matches = at.entries.filter((entry) => entry.key === token);
      if (matches.length === 0) return notFound;
      if (matches.length > 1) {
        return {
          ok: false,
          code: "locate_within_ambiguous",
          message: `\`within\` ${where(pointer)} passes through the key ${JSON.stringify(token)}, which the evidence writes more than once.`,
        };
      }
      at = matches[0].value;
    } else if (at.kind === "array") {
      // Digits only, no leading zero, and never `-`: the element past the end
      // exists for writing, not for reading.
      if (!/^(0|[1-9][0-9]*)$/.test(token)) return notFound;
      const index = Number(token);
      if (index >= at.items.length) return notFound;
      at = at.items[index];
    } else {
      return notFound;
    }
  }
  return { ok: true, node: at };
}

/** A reference token unescaped: `~1` is `/`, `~0` is `~`, any other `~` is no pointer. */
function unescape(raw: string): string | null {
  if (/~(?![01])/.test(raw)) return null;
  return raw.replace(/~1/g, "/").replace(/~0/g, "~");
}

function kindOf(node: JsonNode): string {
  switch (node.kind) {
    case "null":
      return "null";
    case "boolean":
      return "a boolean";
    case "number":
      return "a number";
    case "string":
      return "a string";
    case "array":
      return node.items.length === 0 ? "an empty array" : "an array";
    case "object":
      return "an object";
  }
}
