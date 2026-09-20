import { describe, expect, it } from "vitest";
import { asText, duplicateKey, fromPlain, parseOrdered, plain, saysNothing, writeOrdered } from "./json.ts";

const parse = (text: string) => {
  const read = parseOrdered(text);
  if (!read.ok) throw new Error(`${read.error.message} at ${read.error.line}:${read.error.column}`);
  return read.node;
};

describe("parseOrdered", () => {
  it("keeps integer-like keys in the order they were written, where JSON.parse would sort them", () => {
    const text = '{"3":"c","1":"a","2":"b"}';
    // The whole reason this parser exists: a different option order is a
    // different prompt, and the built-in loses it here.
    expect(Object.keys(JSON.parse(text) as object)).toEqual(["1", "2", "3"]);
    expect(writeOrdered(parse(text), 0)).toBe(text);
  });

  it("keeps duplicate keys rather than merging them, so a duplicate can be reported", () => {
    const node = parse('{"a":1,"a":2}');
    expect(node.kind === "object" && node.entries.map((e) => e.key)).toEqual(["a", "a"]);
    expect(duplicateKey(node.kind === "object" ? node.entries : [])).toBe("a");
  });

  it("reads the scalars, nesting and escapes", () => {
    expect(plain(parse('{"a":[1,-2.5,1e3,true,false,null],"b":"x\\n\\u00e9\\"y"}'))).toEqual({
      a: [1, -2.5, 1000, true, false, null],
      b: 'x\né"y',
    });
  });

  it("reads an empty object and an empty array", () => {
    expect(writeOrdered(parse("{}"), 0)).toBe("{}");
    expect(writeOrdered(parse("[ ]"), 0)).toBe("[]");
  });

  it("names the line and column where it gave up", () => {
    const read = parseOrdered('{\n  "a": 1,\n  "b" 2\n}');
    expect(read.ok).toBe(false);
    if (read.ok) return;
    expect(read.error.message).toContain('no ":" after the key "b"');
    expect(read.error).toMatchObject({ line: 3, column: 7 });
  });

  it("refuses trailing text, an unquoted key and an unterminated string", () => {
    for (const bad of ['{"a":1} extra', "{a:1}", '{"a":"x', "", "{", '{"a":1,}']) {
      expect(parseOrdered(bad).ok, bad).toBe(false);
    }
  });
});

describe("writeOrdered", () => {
  it("round-trips a document through the pretty form", () => {
    const text = '{"b":{"z":[1,{"y":null}],"a":"s"},"a":true}';
    expect(writeOrdered(parse(writeOrdered(parse(text), 2)), 0)).toBe(text);
  });

  it("indents nested values and keeps empty containers on one line", () => {
    expect(writeOrdered(parse('{"a":{"b":[1]},"c":{},"d":[]}'))).toBe(
      ['{', '  "a": {', '    "b": [', '      1', '    ]', '  },', '  "c": {},', '  "d": []', '}'].join("\n"),
    );
  });
});

describe("fromPlain", () => {
  it("turns a plain value into a node and back", () => {
    expect(plain(fromPlain({ a: [1, "x", null, false] }))).toEqual({ a: [1, "x", null, false] });
  });

  it("reads undefined as null, since a request field is either written or absent", () => {
    expect(plain(fromPlain(undefined))).toBe(null);
  });
});

describe("saysNothing", () => {
  it("is true for the shapes decide.rs calls empty instructions", () => {
    for (const text of ["null", '""', '"  "', "[]", "{}"]) expect(saysNothing(parse(text)), text).toBe(true);
    for (const text of ['"a"', "0", "false", "[1]", '{"a":1}']) expect(saysNothing(parse(text)), text).toBe(false);
  });
});

describe("asText", () => {
  it("shows a string verbatim and anything else as its JSON", () => {
    expect(asText(parse('"hello"'))).toBe("hello");
    expect(asText(parse('{"a": 1}'))).toBe('{"a":1}');
  });
});
