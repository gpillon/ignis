// JSON that keeps the order its keys were written in (GitHub #247).
//
// `JSON.parse` cannot carry it: a JavaScript object hoists integer-like keys
// to the front and sorts them ascending, so `{"3":…,"1":…,"2":…}` parses with
// its keys in the order 1, 2, 3 — and for a `choice`'s `criteria` that is not
// a presentation detail but a different prompt, because each option is bound
// to the answer token at its position (`decide.rs`, `Ordered`). A reviver
// does not help; it is called in the already-reordered order.
//
// So the Decide tab reads and writes JSON through this instead of the
// built-in: a recursive-descent parser into an ordered tree, and a printer
// that emits the entries as they stand. The tree is also what the builder's
// model holds for the values it does not itself edit — an `instructions`
// written as an object survives a round-trip as itself.

/** A JSON value, with every object's entries in written order. */
export type JsonNode =
  | { kind: "object"; entries: JsonEntry[] }
  | { kind: "array"; items: JsonNode[] }
  | { kind: "string"; value: string }
  | { kind: "number"; value: number }
  | { kind: "boolean"; value: boolean }
  | { kind: "null" };

export type JsonEntry = { key: string; value: JsonNode };

export const jsonNull: JsonNode = { kind: "null" };
export const jsonString = (value: string): JsonNode => ({ kind: "string", value });
export const jsonObject = (entries: JsonEntry[]): JsonNode => ({ kind: "object", entries });
export const jsonArray = (items: JsonNode[]): JsonNode => ({ kind: "array", items });

/** Where a parse gave up, and on what. */
export type JsonError = { message: string; line: number; column: number };

export type Parsed = { ok: true; node: JsonNode } | { ok: false; error: JsonError };

export function parseOrdered(text: string): Parsed {
  const reader = new Reader(text);
  try {
    reader.spaces();
    const node = reader.value();
    reader.spaces();
    if (!reader.done()) reader.fail("unexpected text after the value");
    return { ok: true, node };
  } catch (thrown) {
    if (thrown instanceof ParseFailure) return { ok: false, error: thrown.error };
    throw thrown;
  }
}

class ParseFailure extends Error {
  constructor(readonly error: JsonError) {
    super(error.message);
  }
}

class Reader {
  private at = 0;

  constructor(private readonly text: string) {}

  done(): boolean {
    return this.at >= this.text.length;
  }

  fail(message: string): never {
    // Lines and columns are 1-based, the way an editor counts them.
    const before = this.text.slice(0, this.at);
    const line = before.split("\n").length;
    const column = this.at - (before.lastIndexOf("\n") + 1) + 1;
    throw new ParseFailure({ message, line, column });
  }

  spaces() {
    while (this.at < this.text.length && " \t\r\n".includes(this.text[this.at])) this.at++;
  }

  private take(literal: string): boolean {
    if (!this.text.startsWith(literal, this.at)) return false;
    this.at += literal.length;
    return true;
  }

  value(): JsonNode {
    if (this.done()) this.fail("the document ends where a value was expected");
    const char = this.text[this.at];
    if (char === "{") return this.object();
    if (char === "[") return this.array();
    if (char === '"') return { kind: "string", value: this.string() };
    if (this.take("true")) return { kind: "boolean", value: true };
    if (this.take("false")) return { kind: "boolean", value: false };
    if (this.take("null")) return jsonNull;
    return this.number();
  }

  private object(): JsonNode {
    this.at++; // {
    const entries: JsonEntry[] = [];
    this.spaces();
    if (this.take("}")) return { kind: "object", entries };
    for (;;) {
      this.spaces();
      if (this.text[this.at] !== '"') this.fail("a key must be a quoted string");
      const key = this.string();
      this.spaces();
      if (!this.take(":")) this.fail(`no ":" after the key ${JSON.stringify(key)}`);
      this.spaces();
      // Duplicates are kept, not merged: which copy wins is a parser's
      // choice, and the model refuses a duplicate by name rather than
      // silently keeping one.
      entries.push({ key, value: this.value() });
      this.spaces();
      if (this.take(",")) continue;
      if (this.take("}")) return { kind: "object", entries };
      this.fail('expected "," or "}"');
    }
  }

  private array(): JsonNode {
    this.at++; // [
    const items: JsonNode[] = [];
    this.spaces();
    if (this.take("]")) return { kind: "array", items };
    for (;;) {
      this.spaces();
      items.push(this.value());
      this.spaces();
      if (this.take(",")) continue;
      if (this.take("]")) return { kind: "array", items };
      this.fail('expected "," or "]"');
    }
  }

  private string(): string {
    this.at++; // "
    let out = "";
    for (;;) {
      if (this.done()) this.fail("the document ends inside a string");
      const char = this.text[this.at++];
      if (char === '"') return out;
      if (char !== "\\") {
        if (char < " ") this.fail("a control character must be escaped inside a string");
        out += char;
        continue;
      }
      const escape = this.text[this.at++];
      const simple: Record<string, string> = { '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t" };
      if (escape in simple) {
        out += simple[escape];
      } else if (escape === "u") {
        const hex = this.text.slice(this.at, this.at + 4);
        if (!/^[0-9a-fA-F]{4}$/.test(hex)) this.fail("a \\u escape needs four hex digits");
        out += String.fromCharCode(Number.parseInt(hex, 16));
        this.at += 4;
      } else {
        this.at--;
        this.fail(`unknown escape \\${escape ?? ""}`);
      }
    }
  }

  private number(): JsonNode {
    const start = this.at;
    if (this.text[this.at] === "-") this.at++;
    while (this.at < this.text.length && /[0-9]/.test(this.text[this.at])) this.at++;
    if (this.text[this.at] === ".") {
      this.at++;
      while (this.at < this.text.length && /[0-9]/.test(this.text[this.at])) this.at++;
    }
    if (this.text[this.at] === "e" || this.text[this.at] === "E") {
      this.at++;
      if (this.text[this.at] === "+" || this.text[this.at] === "-") this.at++;
      while (this.at < this.text.length && /[0-9]/.test(this.text[this.at])) this.at++;
    }
    const raw = this.text.slice(start, this.at);
    const value = Number(raw);
    if (raw === "" || !Number.isFinite(value)) {
      this.at = start;
      this.fail("not a value");
    }
    return { kind: "number", value };
  }
}

/** The node as JSON text, entries in their own order. `indent` 0 writes one line. */
export function writeOrdered(node: JsonNode, indent = 2): string {
  const write = (n: JsonNode, depth: number): string => {
    const pad = indent ? "\n" + " ".repeat(indent * (depth + 1)) : "";
    const close = indent ? "\n" + " ".repeat(indent * depth) : "";
    const gap = indent ? " " : "";
    switch (n.kind) {
      case "object":
        if (n.entries.length === 0) return "{}";
        return `{${n.entries.map((e) => `${pad}${JSON.stringify(e.key)}:${gap}${write(e.value, depth + 1)}`).join(",")}${close}}`;
      case "array":
        if (n.items.length === 0) return "[]";
        return `[${n.items.map((item) => `${pad}${write(item, depth + 1)}`).join(",")}${close}]`;
      case "string":
        return JSON.stringify(n.value);
      case "number":
        return String(n.value);
      case "boolean":
        return String(n.value);
      case "null":
        return "null";
    }
  };
  return write(node, 0);
}

/** The node as a plain JavaScript value — for display, where order no longer matters. */
export function plain(node: JsonNode): unknown {
  switch (node.kind) {
    case "object":
      return Object.fromEntries(node.entries.map((e) => [e.key, plain(e.value)]));
    case "array":
      return node.items.map(plain);
    default:
      return node.kind === "null" ? null : node.value;
  }
}

/** A plain value as a node. Object keys keep the order `Object.entries` reports. */
export function fromPlain(value: unknown): JsonNode {
  if (value === null || value === undefined) return jsonNull;
  if (typeof value === "string") return { kind: "string", value };
  if (typeof value === "number") return { kind: "number", value: Number.isFinite(value) ? value : 0 };
  if (typeof value === "boolean") return { kind: "boolean", value };
  if (Array.isArray(value)) return { kind: "array", items: value.map(fromPlain) };
  return { kind: "object", entries: Object.entries(value).map(([key, v]) => ({ key, value: fromPlain(v) })) };
}

/** Whether the node says nothing at all — `instructions_are_empty` in `decide.rs`. */
export function saysNothing(node: JsonNode): boolean {
  switch (node.kind) {
    case "null":
      return true;
    case "string":
      return node.value.trim() === "";
    case "array":
      return node.items.length === 0;
    case "object":
      return node.entries.length === 0;
    default:
      return false;
  }
}

/** A node's text if it is a string, and its JSON otherwise — what a one-line readout shows. */
export function asText(node: JsonNode): string {
  return node.kind === "string" ? node.value : writeOrdered(node, 0);
}

/** The first key that appears more than once, if any. */
export function duplicateKey(entries: JsonEntry[]): string | null {
  const seen = new Set<string>();
  for (const entry of entries) {
    if (seen.has(entry.key)) return entry.key;
    seen.add(entry.key);
  }
  return null;
}
