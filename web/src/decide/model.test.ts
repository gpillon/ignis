import { describe, expect, it } from "vitest";
import { jsonString, parseOrdered } from "./json.ts";
import {
  type Draft,
  EMPTY_DRAFT,
  EMPTY_SPARE,
  type Evidence,
  freeId,
  newQuestion,
  type Primitive,
  type Question,
  readRequest,
  requestBody,
  restore,
  setAside,
  validate,
} from "./model.ts";

const image = { name: "a.jpg", url: "data:image/jpeg;base64,AA", width: 40, height: 20 };

function question(id: string, kind: Primitive, instructions = "Decide.", extra: Partial<Question> = {}): Question {
  return { ...newQuestion(kind, id), instructions: jsonString(instructions), ...extra };
}

const draft = (questions: Question[], evidence: Evidence = { mode: "text", text: "Evidence." }): Draft => ({
  evidence,
  questions,
  extras: [],
});

const codes = (d: Draft) => validate(d).map((f) => f.code);

describe("requestBody", () => {
  it("writes a choice's options in the order they were declared, where an object would sort them", () => {
    const body = requestBody(
      draft([
        question("pick", "choice", "Which one?", {
          options: [
            { key: "3", description: "third" },
            { key: "1", description: "first" },
            { key: "2", description: "" },
          ],
        }),
      ]),
    );
    // On the bytes, not on an object: `JSON.stringify` would emit 1, 2, 3.
    expect(body.indexOf('"3"')).toBeLessThan(body.indexOf('"1"'));
    expect(body.indexOf('"1"')).toBeLessThan(body.indexOf('"2"'));
    // A blank description sends null, which the server reads as "the key
    // describes itself".
    expect(body).toContain('"2": null');
  });

  it("writes the questions in the order they were added", () => {
    const body = requestBody(draft([question("b", "noul"), question("a", "noul")]));
    expect(body.indexOf('"b"')).toBeLessThan(body.indexOf('"a"'));
  });

  it("puts the evidence first and sends no model, so the loaded one answers", () => {
    const body = requestBody(draft([question("a", "noul")]));
    expect(body.indexOf('"state"')).toBeLessThan(body.indexOf('"questions"'));
    expect(body).not.toContain('"model"');
  });

  it("omits a noul's criteria when neither description was written, and sends only the one that was", () => {
    expect(requestBody(draft([question("a", "noul")]))).not.toContain("criteria");
    const one = requestBody(draft([question("a", "noul", "Urgent?", { yes: "Time-sensitive" })]));
    expect(one).toContain('"true": "Time-sensitive"');
    expect(one).not.toContain('"false"');
  });

  it("sends a score's levels as an ordered array and no digits", () => {
    const body = requestBody(draft([question("a", "score", "How much?", { levels: ["Low", "High"] })]));
    expect(body).toContain('"criteria": [\n        "Low",\n        "High"\n      ]');
    expect(body).not.toContain("digits");
  });

  it("sends digits for a constrained decode and never criteria, which that primitive refuses", () => {
    const body = requestBody(draft([question("n", "number", "How many?", { digits: 4, options: [{ key: "x", description: "y" }] })]));
    expect(body).toContain('"digits": 4');
    expect(body).not.toContain("criteria");
  });

  it("sends a scalar's ceiling only when one was written, because absent is what asks for the widest run", () => {
    const none = requestBody(draft([question("s", "scalar", "How many hours?")]));
    expect(none).not.toContain("digits");
    expect(none).not.toContain("criteria");
    const capped = requestBody(draft([question("s", "scalar", "How many hours?", { ceiling: 2 })]));
    expect(capped).toContain('"digits": 2');
  });

  it("does not let a scalar borrow the width a number left on the question", () => {
    // The two fields are separate on purpose: switching a question from
    // `number` to `scalar` must not send the field width as a ceiling.
    const body = requestBody(draft([question("s", "scalar", "How many hours?", { digits: 4 })]));
    expect(body).not.toContain("digits");
  });

  it("sends a JSON evidence as itself, not as a string", () => {
    const body = requestBody(draft([question("a", "noul")], { mode: "json", text: '{"order":"A-1"}' }));
    expect(body).toContain('"state": {\n    "order": "A-1"\n  }');
  });

  it("sends an image evidence as content parts, images first and the text behind them", () => {
    const body = requestBody(draft([question("a", "noul")], { mode: "image", images: [image], text: "The mark." }));
    const state = (JSON.parse(body) as { state: { type: string }[] }).state;
    expect(state.map((part) => part.type)).toEqual(["image_url", "text"]);
  });

  it("sends an image evidence with no words as the image alone", () => {
    const body = requestBody(draft([question("a", "noul")], { mode: "image", images: [image], text: "  " }));
    const state = (JSON.parse(body) as { state: { type: string }[] }).state;
    expect(state.map((part) => part.type)).toEqual(["image_url"]);
  });
});

describe("validate", () => {
  it("passes a draft the endpoint would serve", () => {
    expect(
      codes(
        draft([
          question("a", "noul", "Urgent?", { yes: "Yes it is", no: "No" }),
          question("b", "choice", "Which?", { options: [{ key: "x", description: "" }] }),
          question("c", "score", "How much?", { levels: ["Low", "High"] }),
          question("d", "number", "How many?"),
        ]),
      ),
    ).toEqual([]);
  });

  it("refuses a draft with no questions", () => {
    expect(codes(EMPTY_DRAFT)).toEqual(["no_questions"]);
  });

  it("names a blank question id, which the endpoint would accept and nobody could use", () => {
    expect(codes(draft([question("", "noul")]))).toEqual(["blank_question_id"]);
  });

  it("names a duplicate question id", () => {
    expect(codes(draft([question("a", "noul"), question("a", "noul")]))).toEqual(["duplicate_question"]);
  });

  it("names empty instructions, in every shape decide.rs calls empty", () => {
    for (const instructions of ['""', '"   "', "null", "[]", "{}"]) {
      const read = parseOrdered(instructions);
      if (!read.ok) throw new Error(instructions);
      expect(codes(draft([{ ...question("a", "noul"), instructions: read.node }])), instructions).toEqual(["empty_instructions"]);
    }
  });

  it("names a choice with no options, a blank option name and a duplicate one", () => {
    expect(codes(draft([question("a", "choice", "Which?", { options: [] })]))).toEqual(["no_options"]);
    expect(codes(draft([question("a", "choice", "Which?", { options: [{ key: " ", description: "x" }] })]))).toEqual(["unclean_option"]);
    expect(
      codes(
        draft([
          question("a", "choice", "Which?", {
            options: [
              { key: "x", description: "" },
              { key: "x", description: "" },
            ],
          }),
        ]),
      ),
    ).toEqual(["duplicate_option"]);
  });

  it("names a choice past the measured option ceiling", () => {
    const options = Array.from({ length: 257 }, (_, i) => ({ key: `o${i}`, description: "" }));
    expect(codes(draft([question("a", "choice", "Which?", { options })]))).toEqual(["too_many_options"]);
  });

  it("names a score with too few levels and one with a blank level", () => {
    expect(codes(draft([question("a", "score", "How much?", { levels: ["Only"] })]))).toEqual(["too_few_levels"]);
    expect(codes(draft([question("a", "score", "How much?", { levels: ["Low", " "] })]))).toEqual(["malformed_criteria"]);
  });

  it("names digits outside the range the endpoint serves", () => {
    for (const digits of [0, 7, 2.5]) {
      expect(codes(draft([question("a", "number", "How many?", { digits })])), String(digits)).toEqual(["digits_out_of_range"]);
    }
    for (const digits of [1, 3, 6]) {
      expect(codes(draft([question("a", "number", "How many?", { digits })])), String(digits)).toEqual([]);
    }
  });

  it("accepts a scalar with no ceiling, and names one outside the range", () => {
    expect(codes(draft([question("a", "scalar", "How many hours?")]))).toEqual([]);
    for (const ceiling of [0, 16, 2.5]) {
      expect(codes(draft([question("a", "scalar", "How many hours?", { ceiling })])), String(ceiling)).toEqual([
        "digits_out_of_range",
      ]);
    }
    // Widths a `number` refuses are legal ceilings: a field must be filled
    // and a ceiling need not.
    for (const ceiling of [1, 3, 6, 8, 15]) {
      expect(codes(draft([question("a", "scalar", "How many hours?", { ceiling })])), String(ceiling)).toEqual([]);
    }
  });

  it("names a point or a box whose evidence carries no image", () => {
    for (const kind of ["point", "box"] as const) {
      expect(codes(draft([question("a", kind, "Where?")])), kind).toEqual(["state_carries_no_image"]);
      expect(codes(draft([question("a", kind, "Where?")], { mode: "image", images: [image], text: "" })), kind).toEqual([]);
    }
  });

  it("names a JSON evidence that is not JSON, with the place it broke", () => {
    const faults = validate(draft([question("a", "noul")], { mode: "json", text: "{oops}" }));
    expect(faults.map((f) => f.code)).toEqual(["invalid_state_json"]);
    expect(faults[0].message).toContain("line 1");
  });

  it("refuses a pasted thinking control rather than sending one the endpoint answers 422 to", () => {
    const d = { ...draft([question("a", "noul")]), extras: [{ key: "enable_thinking", value: { kind: "boolean" as const, value: true } }] };
    expect(codes(d)).toEqual(["thinking_refused"]);
  });

  it("carries the offending question's handle, so the builder can point at it", () => {
    const bad = question("", "noul");
    expect(validate(draft([bad]))[0].uid).toBe(bad.uid);
  });
});

describe("readRequest", () => {
  it("reads a body back with its option order intact", () => {
    const body = '{"state":"s","questions":{"pick":{"type":"choice","instructions":"Which?","criteria":{"3":"c","1":null,"2":"b"}}}}';
    const read = readRequest(body);
    if (!read.ok) throw new Error(read.message);
    expect(read.draft.questions[0].options.map((o) => o.key)).toEqual(["3", "1", "2"]);
    // A null description comes back blank, which is what sends null again.
    expect(read.draft.questions[0].options[1].description).toBe("");
    expect(requestBody(read.draft)).toContain('"3"');
  });

  it("round-trips every primitive", () => {
    const original = draft([
      question("a", "noul", "Urgent?", { yes: "Y", no: "N" }),
      question("b", "choice", "Which?", { options: [{ key: "x", description: "the x" }] }),
      question("c", "score", "How much?", { levels: ["Low", "High"] }),
      question("d", "number", "How many?", { digits: 5 }),
      question("e", "scalar", "How many hours?", { ceiling: 4 }),
      question("f", "point", "Where?"),
      question("g", "box", "Bound it."),
    ]);
    const read = readRequest(requestBody(original));
    if (!read.ok) throw new Error(read.message);
    expect(requestBody(read.draft)).toBe(requestBody(original));
  });

  it("reads a scalar's digits back as a ceiling, and keeps the absence of one", () => {
    const capped = readRequest('{"state":"s","questions":{"s":{"type":"scalar","instructions":"How many?","digits":2}}}');
    if (!capped.ok) throw new Error(capped.message);
    expect(capped.draft.questions[0].ceiling).toBe(2);
    // The width a `number` would use is untouched, so switching the type in
    // the builder does not inherit a ceiling as a field.
    expect(capped.draft.questions[0].digits).toBe(3);

    const open = readRequest('{"state":"s","questions":{"s":{"type":"scalar","instructions":"How many?"}}}');
    if (!open.ok) throw new Error(open.message);
    expect(open.draft.questions[0].ceiling).toBeNull();
    expect(requestBody(open.draft)).not.toContain("digits");
  });

  it("reads our own field names as the aliases decide.rs accepts them as", () => {
    const read = readRequest('{"state":"s","questions":{"a":{"type":"boolean","question":"Urgent?","options":{"true":"Y"}}}}');
    if (!read.ok) throw new Error(read.message);
    expect(read.draft.questions[0]).toMatchObject({ kind: "noul", yes: "Y" });
  });

  it("reads a string state as text, a content-part list as an image, and anything else as JSON", () => {
    const text = readRequest('{"state":"hello","questions":{"a":{"type":"noul","instructions":"?"}}}');
    expect(text.ok && text.draft.evidence.mode).toBe("text");
    const parts = readRequest(
      '{"state":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AA"}},{"type":"text","text":"look"}],"questions":{"a":{"type":"noul","instructions":"?"}}}',
    );
    expect(parts.ok && parts.draft.evidence).toMatchObject({ mode: "image", text: "look" });
    const object = readRequest('{"state":{"order":"A-1"},"questions":{"a":{"type":"noul","instructions":"?"}}}');
    expect(object.ok && object.draft.evidence.mode).toBe("json");
  });

  it("keeps a field it does not edit, on the request and on a question", () => {
    const body = '{"state":"s","note":"mine","questions":{"a":{"type":"noul","instructions":"?","mine":7}}}';
    const read = readRequest(body);
    if (!read.ok) throw new Error(read.message);
    const out = requestBody(read.draft);
    expect(out).toContain('"note": "mine"');
    expect(out).toContain('"mine": 7');
  });

  it("drops a pasted model name, since this tab talks to the loaded model", () => {
    const read = readRequest('{"state":"s","model":"jev-latest","questions":{"a":{"type":"noul","instructions":"?"}}}');
    if (!read.ok) throw new Error(read.message);
    expect(requestBody(read.draft)).not.toContain("jev-latest");
  });

  it("says what is wrong instead of guessing", () => {
    expect(readRequest("{oops}")).toMatchObject({ ok: false });
    expect(readRequest('"a string"').ok).toBe(false);
    expect(readRequest('{"state":"s"}')).toMatchObject({ ok: false, message: expect.stringContaining("`questions`") });
    expect(readRequest('{"questions":[]}')).toMatchObject({ ok: false, message: expect.stringContaining("JSON object") });
    expect(readRequest('{"questions":{"a":{"instructions":"?"}}}')).toMatchObject({ ok: false, message: expect.stringContaining("`type`") });
    expect(readRequest('{"questions":{"a":{"type":"vibe","instructions":"?"}}}')).toMatchObject({
      ok: false,
      message: expect.stringContaining("unknown type"),
    });
  });
});

describe("setAside and restore", () => {
    it("gives each mode back what it last held, and never what another mode held", () => {
    let spare = setAside(EMPTY_SPARE, { mode: "text", text: "prose" });
    spare = setAside(spare, { mode: "json", text: '{"a":1}' });
    spare = setAside(spare, { mode: "image", images: [image], text: "the words" });
    expect(restore(spare, "text")).toEqual({ mode: "text", text: "prose" });
    expect(restore(spare, "json")).toEqual({ mode: "json", text: '{"a":1}' });
    expect(restore(spare, "image")).toEqual({ mode: "image", images: [image], text: "the words" });
  });

  it("starts each mode empty, so a freshly replaced draft carries no history", () => {
    expect(restore(EMPTY_SPARE, "text")).toEqual({ mode: "text", text: "" });
    expect(restore(EMPTY_SPARE, "image")).toEqual({ mode: "image", images: [], text: "" });
  });
});

describe("freeId", () => {
  it("takes the stem when it is free and counts up when it is not", () => {
    expect(freeId([], "answer")).toBe("answer");
    expect(freeId([question("answer", "noul")], "answer")).toBe("answer_2");
    expect(freeId([question("answer", "noul"), question("answer_2", "noul")], "answer")).toBe("answer_3");
  });
});

// How a point or a box is answered (GitHub #260, #263). Absent is not a
// missing choice: it is the default the endpoint would pick, and the two do
// not pick alike — a point follows the load (head where the artifact has a
// calibrated pointing head, chain where it has none) and a box is the chain
// on every load. Neither is resolvable from here, so neither is guessed at.
describe("a spatial question's method", () => {
  const withImage = (questions: Question[]) => draft(questions, { mode: "image", images: [image], text: "" });

  it("sends nothing when none was chosen, and the name when one was", () => {
    expect(requestBody(withImage([question("p", "point", "Where?")]))).not.toContain("method");
    const asked = requestBody(withImage([question("p", "point", "Where?", { method: "chain" })]));
    expect(asked).toContain('"method": "chain"');
  });

  it("is a box's too, and says so on the wire", () => {
    // GitHub #263: a head box is the head set's extent, read in the same one
    // pass, and it is an opt-in — so the field has to leave the tab.
    expect(requestBody(withImage([question("b", "box", "Where?", { method: "head" })]))).toContain('"method": "head"');
    // Absent is still absent: `chain` is what a box gets by default, but
    // writing the name would pin a default the endpoint owns.
    expect(requestBody(withImage([question("b", "box", "Where?")]))).not.toContain("method");
  });

  it("round-trips without doubling the field", () => {
    const body = '{"state":"s","questions":{"p":{"type":"box","instructions":"Where?","digits":3,"method":"head"}}}';
    const read = readRequest(body);
    if (!read.ok) throw new Error(read.message);
    expect(read.draft.questions[0].method).toBe("head");
    expect(read.draft.questions[0].extras).toEqual([]);
    expect(requestBody(read.draft).match(/"method"/g)).toHaveLength(1);
  });

  it("ignores a method left over on a primitive that has one way of being answered", () => {
    // Choosing `head` on a point and then switching the type must not block
    // the request: the field is simply not sent from there, where the
    // endpoint would refuse it.
    const scalar = draft([question("s", "scalar", "How hot?", { method: "head" })]);
    expect(validate(scalar).filter((f) => f.code.startsWith("method"))).toEqual([]);
    expect(requestBody(scalar)).not.toContain("method");
    const noul = draft([question("n", "noul", "Well?", { method: "chain" })]);
    expect(validate(noul).filter((f) => f.code.startsWith("method"))).toEqual([]);
    expect(requestBody(noul)).not.toContain("method");
  });

  it("keeps a spelling neither method covers, and refuses it naming both", () => {
    const read = readRequest('{"state":"s","questions":{"p":{"type":"point","instructions":"Where?","method":"attention"}}}');
    if (!read.ok) throw new Error(read.message);
    expect(read.draft.questions[0].method).toBeNull();
    // The body still carries what was written, so the server still answers the
    // 422 this fault is the mirror of.
    expect(requestBody(read.draft)).toContain('"method": "attention"');
    const faults = validate(read.draft);
    expect(faults.map((f) => f.code)).toContain("method_unknown");
    expect(faults.find((f) => f.code === "method_unknown")?.message).toContain('"head" and "chain"');
  });
});
