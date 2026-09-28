import { describe, expect, it } from "vitest";
import { jsonString, parseOrdered } from "./json.ts";
import { cutTarget } from "./locate.ts";

// A `locate`'s target cut the way `crate::locate::evidence_within` cuts it
// (GitHub #277): the answer is only an index, so the tab's cut and the
// server's have to agree segment for segment, and every refusal the server
// answers for a target has to be one the builder reports first.

const json = (text: string) => {
  const parsed = parseOrdered(text);
  if (!parsed.ok) throw new Error(parsed.error.message);
  return parsed.node;
};

const codeOf = (state: string, within: string) => {
  const cut = cutTarget(json(state), within);
  return cut.ok ? "ok" : cut.code;
};

describe("cutTarget", () => {
  it("cuts a string into its lines on \\n exactly, keeping a \\r in its line", () => {
    const cut = cutTarget(jsonString("first\r\nsecond\nthird"), "");
    expect(cut).toEqual({
      ok: true,
      target: { unit: "line", segments: ["first\r", "second", "third"], owns: [true, true, true], records: false },
    });
  });

  it("keeps a blank line's index and gives it no key", () => {
    const cut = cutTarget(jsonString("a\n  \n\nb"), "");
    expect(cut.ok && cut.target.owns).toEqual([true, false, false, true]);
    expect(cut.ok && cut.target.segments.length).toBe(4);
  });

  it("cuts an array into its elements, each as its JSON, and a blank string element owns nothing", () => {
    const cut = cutTarget(json('[{"id": 1, "b": [2]}, " ", "x", 3]'), "");
    expect(cut).toEqual({
      ok: true,
      target: { unit: "item", segments: ['{"id":1,"b":[2]}', " ", "x", "3"], owns: [true, false, true, true], records: false },
    });
  });

  it("calls an array of two or more objects a records array, unless every one is shaped like a content part", () => {
    // GitHub #278, spec 22 § `auto`: `is_records_array`, which is what `auto`
    // reads as `records` and what a named kind is checked against.
    const records = (state: string) => {
      const cut = cutTarget(json(state), "");
      return cut.ok && cut.target.records;
    };
    expect(records('[{"id": 1}, {"id": 2}]')).toBe(true);
    // One object with a string `type` does not make it parts; all of them do.
    expect(records('[{"type": "a", "id": 1}, {"id": 2}]')).toBe(true);
    expect(records('[{"type": "a"}, {"type": "b"}]')).toBe(false);
    // A `type` that is not a string is a field like any other.
    expect(records('[{"type": 1}, {"type": 2}]')).toBe(true);
    expect(records('[{"id": 1}, "x"]')).toBe(false);
    expect(records('["x", "y"]')).toBe(false);
    expect(records('"a\\nb"')).toBe(false);
  });

  it("follows a pointer through objects and arrays, unescaping ~1 and ~0", () => {
    const state = json('{"a/b": {"c~d": ["x\\ny", "p\\nq\\nr"]}}');
    const cut = cutTarget(state, "/a~1b/c~0d/1");
    expect(cut.ok && cut.target.segments).toEqual(["p", "q", "r"]);
  });

  it("refuses what is not a pointer", () => {
    expect(codeOf('{"log": "a\\nb"}', "log")).toBe("locate_within_malformed");
    expect(codeOf('{"log": "a\\nb"}', "/lo~2g")).toBe("locate_within_malformed");
    expect(codeOf('{"log": "a\\nb"}', "/log~")).toBe("locate_within_malformed");
  });

  it("refuses a pointer that names nothing — a missing key, an index past the end, a leading zero, the element past the end, a step into a string", () => {
    const state = '{"items": ["a", "b"], "log": "a\\nb"}';
    expect(codeOf(state, "/item")).toBe("locate_within_not_found");
    expect(codeOf(state, "/items/2")).toBe("locate_within_not_found");
    expect(codeOf(state, "/items/01")).toBe("locate_within_not_found");
    expect(codeOf(state, "/items/-")).toBe("locate_within_not_found");
    expect(codeOf(state, "/log/0")).toBe("locate_within_not_found");
  });

  it("refuses a pointer through a key the state writes twice, rather than choosing a copy", () => {
    expect(codeOf('{"log": "a\\nb", "log": "c\\nd"}', "/log")).toBe("locate_within_ambiguous");
  });

  it("refuses a target that is neither a string nor a non-empty array, naming what it is", () => {
    const cut = cutTarget(json('{"meta": {"a": 1}, "none": []}'), "/meta");
    expect(cut.ok).toBe(false);
    if (!cut.ok) {
      expect(cut.code).toBe("locate_target_unsegmentable");
      expect(cut.message).toContain("an object");
    }
    expect(codeOf('{"none": []}', "/none")).toBe("locate_target_unsegmentable");
    expect(codeOf('{"n": 4}', "")).toBe("locate_target_unsegmentable");
  });

  it("refuses a target with fewer than two segments that own a key", () => {
    expect(cutTarget(jsonString("only one line"), "").ok).toBe(false);
    expect(codeOf('["x", "", "  "]', "")).toBe("locate_too_few_segments");
    expect(codeOf('["x", "y"]', "")).toBe("ok");
  });
});
