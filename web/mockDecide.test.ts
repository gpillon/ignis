import { describe, expect, it } from "vitest";
import { mockDecide } from "./mockDecide.ts";

// The dev mock (GitHub #247). What matters here is that it answers in the
// shapes the tab reads, and that it honours the two properties the tab is
// built around: the declared option order, and `output_tokens` staying 0 for a
// readout.

type Body = {
  model: string;
  answers: Record<string, Record<string, unknown>>;
  usage: { input_tokens: number; output_tokens: number };
};

const send = (body: unknown) => {
  const raw = typeof body === "string" ? body : JSON.stringify(body);
  const { status, body: out } = mockDecide(raw);
  return { status, body: out as Body };
};

describe("mockDecide", () => {
  it("answers a noul with a probability and generates nothing", () => {
    const { status, body } = send({ state: "s", questions: { a: { type: "noul", instructions: "Urgent?" } } });
    expect(status).toBe(200);
    expect(body.answers.a.type).toBe("noul");
    expect(body.answers.a.noul).toBeGreaterThan(0);
    expect(body.answers.a.noul).toBeLessThan(1);
    expect(body.usage.output_tokens).toBe(0);
  });

  it("picks a choice from the options the request declared, integer-like keys included", () => {
    // Read through the ordered parser, so the option at position 0 really is
    // the one written first. The response map itself is unordered, exactly as
    // the server's `BTreeMap` is — the tab reads order off the request.
    const raw = '{"state":"s","questions":{"pick":{"type":"choice","instructions":"Which?","criteria":{"3":"c","1":"a","2":"b"}}}}';
    const { body } = send(raw);
    const probabilities = body.answers.pick.probabilities as Record<string, number>;
    expect(Object.keys(probabilities).sort()).toEqual(["1", "2", "3"]);
    expect(Object.values(probabilities).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 2);
    const winner = body.answers.pick.choice as string;
    expect(probabilities[winner]).toBe(Math.max(...Object.values(probabilities)));
  });

  it("answers a score between its levels, with a legend and one probability per level", () => {
    const { body } = send({ state: "s", questions: { s: { type: "score", instructions: "How much?", criteria: ["Low", "Mid", "High"] } } });
    const answer = body.answers.s as { score: number; legend: Record<string, string>; probabilities: Record<string, number> };
    expect(Object.keys(answer.legend)).toEqual(["0", "1", "2"]);
    expect(answer.score).toBeGreaterThanOrEqual(0);
    expect(answer.score).toBeLessThanOrEqual(2);
    expect(Object.keys(answer.probabilities)).toHaveLength(3);
  });

  it("generates a digit per place for a number, and counts them", () => {
    const { body } = send({ state: "s", questions: { n: { type: "number", instructions: "How many?", digits: 4 } } });
    const answer = body.answers.n as { number: number; digits: { digit: number; probability: number }[] };
    expect(answer.digits).toHaveLength(4);
    expect(String(answer.number).length).toBeLessThanOrEqual(4);
    // Confidence falls across the places, the way a measured trace does.
    expect(answer.digits[0].probability).toBeGreaterThan(answer.digits[3].probability);
    expect(body.usage.output_tokens).toBe(4);
  });

  it("lets a scalar choose its own width under the ceiling, and bills the brace that closed it", () => {
    const { body } = send({ state: "s", questions: { s: { type: "scalar", instructions: "How many hours?", digits: 4 } } });
    const answer = body.answers.s as { type: string; value: number; text: string; uncertainty: number; digits: { digit: number }[] };
    expect(answer.type).toBe("scalar");
    expect(answer.value).toBe(Number(answer.text));
    // The trace is the digits alone: the point and the sign were steps of the
    // run and hold no place.
    // The trace and the spelling are the same digits in the same order: a
    // column the text does not carry is the one fault this card exists to
    // show, so the mock must never manufacture one.
    expect(answer.text.replace(/[-.]/g, "")).toBe(answer.digits.map((d) => d.digit).join(""));
    expect(answer.digits.length).toBeLessThanOrEqual(4);
    expect(body.usage.output_tokens).toBe(answer.text.length + 1);
  });

  it("answers a scalar with no ceiling at all, which is the request this primitive exists for", () => {
    const { body } = send({ state: "s", questions: { s: { type: "scalar", instructions: "How much?" } } });
    const answer = body.answers.s as { type: string; digits: unknown[] };
    expect(answer.type).toBe("scalar");
    // Nothing declared asks for the server's default ceiling, which is 8 and
    // not the 15 it would serve on request.
    expect(answer.digits.length).toBeLessThanOrEqual(8);
  });

  it("refuses a point when the state carried no image, on that question alone", () => {
    const { body } = send({
      state: "just words",
      questions: { p: { type: "point", instructions: "Where?" }, a: { type: "noul", instructions: "Urgent?" } },
    });
    expect(body.answers.p).toMatchObject({ type: "error", code: "state_carries_no_image" });
    expect(body.answers.a.type).toBe("noul");
  });

  it("answers a point in the pixels of the JPEG it was handed", () => {
    const state = [{ type: "image_url", image_url: { url: jpeg(64, 32) } }];
    const { body } = send({ state, questions: { p: { type: "point", instructions: "Where?", digits: 3 } } });
    const answer = body.answers.p as { type: string; pixels: Record<string, number> };
    expect(answer.type).toBe("point");
    expect(answer.pixels.x).toBeGreaterThanOrEqual(0);
    expect(answer.pixels.x).toBeLessThanOrEqual(64);
    expect(answer.pixels.y).toBeLessThanOrEqual(32);
  });

  it("draws a box that is not inside out", () => {
    const state = [{ type: "image_url", image_url: { url: jpeg(100, 100) } }];
    const { body } = send({ state, questions: { b: { type: "box", instructions: "Bound it." } } });
    const answer = body.answers.b as { pixels: Record<string, number> };
    expect(answer.pixels.x1).toBeGreaterThan(answer.pixels.x0);
    expect(answer.pixels.y1).toBeGreaterThan(answer.pixels.y0);
  });

  it("fails one question on /error and the whole request on /full", () => {
    const one = send({ state: "s", questions: { a: { type: "noul", instructions: "/error" }, b: { type: "noul", instructions: "ok" } } });
    expect(one.body.answers.a.type).toBe("error");
    expect(one.body.answers.b.type).toBe("noul");
    expect(send({ state: "/full", questions: { a: { type: "noul", instructions: "?" } } }).status).toBe(503);
  });

  it("refuses a body that is not a request", () => {
    expect(send("{oops}").status).toBe(400);
    expect(send({ state: "s", questions: {} }).status).toBe(422);
  });

  it("answers the same question the same way twice", () => {
    const request = { state: "s", questions: { a: { type: "noul", instructions: "Urgent?" } } };
    expect(send(request).body.answers).toEqual(send(request).body.answers);
  });
});

/** A minimal JPEG carrying nothing but a SOF0 frame header of the given size. */
function jpeg(width: number, height: number): string {
  const sof = Buffer.alloc(19);
  sof.writeUInt16BE(0xffd8, 0); // SOI
  sof.writeUInt16BE(0xffc0, 2); // SOF0
  sof.writeUInt16BE(11, 4); // segment length
  sof.writeUInt8(8, 6); // precision
  sof.writeUInt16BE(height, 7);
  sof.writeUInt16BE(width, 9);
  return `data:image/jpeg;base64,${sof.toString("base64")}`;
}
