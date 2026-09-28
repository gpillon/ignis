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

  it("answers a vote with a segment that owns a key, in whole votes of the heads, and generates nothing", () => {
    // The vote is pinned by name (spec 22): the shortlist is the default now.
    const state = { log: "a start\n\nb middle\nc end", tickets: [{ id: 1 }, { id: 2 }, { id: 3 }] };
    const { status, body } = send({
      state,
      questions: {
        line: { type: "locate", instructions: "Which line?", within: "/log", method: "vote" },
        item: { type: "locate", instructions: "Which ticket?", within: "/tickets", method: "vote" },
      },
    });
    expect(status).toBe(200);
    const line = body.answers.line as unknown as Located;
    // The blank line is segment 1, and a locate never names it.
    expect([0, 2, 3]).toContain(line.segment);
    expect(line.value).toBe(state.log.split("\n")[line.segment!]);
    expect(line.confidence).toBe(line.ranking[0].share);
    expect(line.ranking.length).toBeLessThanOrEqual(5);
    for (const { share } of line.ranking) expect(Number.isInteger(share * 32)).toBe(true);
    const shares = line.ranking.map((r) => r.share);
    expect(shares).toEqual([...shares].sort((a, b) => b - a));
    // GitHub #278: it names what answered it, reads as it is, carries no
    // `found`, and points at its winner alone.
    expect(line).toMatchObject({ method: "vote", compression: "none" });
    expect(line).not.toHaveProperty("found");
    expect(line.pointers).toEqual([{ segment: line.segment, value: line.value, share: line.confidence }]);
    const item = body.answers.item as unknown as Located;
    expect(item.kind).toBe("records");
    expect(item.value).toEqual(state.tickets[item.segment!]);
    expect(body.usage.output_tokens).toBe(0);
  });

  it("refuses a locate over content parts, and a within that names nothing, the whole request", () => {
    const parts = [{ type: "text", text: "a\nb" }];
    expect(send({ state: parts, questions: { l: { type: "locate", instructions: "Which?" } } }).status).toBe(422);
    const missing = send({ state: { log: "a\nb" }, questions: { l: { type: "locate", instructions: "Which?", within: "/logs" } } });
    expect(missing.status).toBe(422);
    expect(JSON.stringify(missing.body)).toContain("locate_within_not_found");
  });

  it("answers the same question the same way twice", () => {
    const request = { state: "s", questions: { a: { type: "noul", instructions: "Urgent?" } } };
    expect(send(request).body.answers).toEqual(send(request).body.answers);
  });
});

/** A `locate` answer as the mock writes it (GitHub #278). */
type Located = {
  type: "locate";
  kind: string;
  method: string;
  compression: string;
  found?: number;
  segment: number | null;
  value: unknown;
  confidence: number | null;
  ranking: { segment: number; share: number }[];
  pointers: { segment: number; value: unknown; share: number }[];
};

// A locate's route (GitHub #278, spec 22): every combination the server
// serves answers with its shape, `found` on the three measured routes alone,
// "not found" among them; every one it refuses is refused.
describe("mockDecide's locate routes", () => {
  const LOG = [
    "09:14:02 INFO  gateway GET /v1/orders 200 41ms",
    "09:14:07 INFO  auth    user 6121 signed in",
    "09:15:40 ERROR orders  pg: FATAL sorry, too many clients already",
    "",
    "09:15:41 ERROR orders  POST /v1/orders 500 12ms",
    "09:16:03 INFO  deploy  replica api-2 healthy",
    "09:17:10 INFO  deploy  rollout of api v2.41.0 complete",
  ].join("\n");
  const PROSE = [
    "# The kiln",
    "A kiln is a thermally insulated chamber used for firing clay.",
    "Its temperature is raised slowly over many hours.",
    "",
    "# The glaze",
    "Glaze is a layer of glass fused to a ceramic body.",
    "It melts only when the kiln is hot enough.",
  ].join("\n");
  const TICKETS = [
    { id: 311, subject: "Export button greyed out" },
    { id: 312, subject: "Billed twice for March" },
    { id: 313, subject: "2FA code never arrives" },
  ];
  const state = { log: LOG, prose: PROSE, tickets: TICKETS, notes: ["first note", "second note", "third note"] };

  const ask = (question: Record<string, unknown>, id = "q") => {
    const { status, body } = send({ state, questions: { [id]: { type: "locate", instructions: "Which one?", ...question } } });
    return { status, answer: body.answers?.[id] as unknown as Located, error: (body as unknown as { error?: { code: string; message: string } }).error };
  };

  /** What every answer holds whatever its route: a ranking best first, and pointers at 0.05 or more with their values. */
  const wellFormed = (answer: Located, values: unknown[]) => {
    expect(answer.type).toBe("locate");
    expect(answer.ranking.length).toBeGreaterThan(0);
    expect(answer.ranking.length).toBeLessThanOrEqual(5);
    const shares = answer.ranking.map((r) => r.share);
    expect(shares).toEqual([...shares].sort((a, b) => b - a));
    if (answer.segment === null) {
      expect(answer.value).toBeNull();
      expect(answer.confidence).toBeNull();
      expect(answer.pointers).toEqual([]);
      return;
    }
    expect(answer.value).toEqual(values[answer.segment]);
    expect(answer.ranking[0]).toEqual({ segment: answer.segment, share: answer.confidence });
    expect(answer.pointers[0]).toEqual({ segment: answer.segment, value: answer.value, share: answer.confidence });
    for (const pointer of answer.pointers.slice(1)) expect(pointer.share).toBeGreaterThanOrEqual(0.05);
    for (const pointer of answer.pointers) expect(pointer.value).toEqual(values[pointer.segment]);
  };

  const logLines = LOG.split("\n");
  const proseLines = PROSE.split("\n");

  it("answers with no fields by the defaults, and names each of them", () => {
    const log = ask({ within: "/log" }).answer;
    expect(log).toMatchObject({ kind: "log", method: "shortlist", compression: "template_fold" });
    wellFormed(log, logLines);
    const prose = ask({ within: "/prose" }).answer;
    expect(prose).toMatchObject({ kind: "prose", method: "shortlist", compression: "none" });
    wellFormed(prose, proseLines);
    const records = ask({ within: "/tickets" }).answer;
    expect(records).toMatchObject({ kind: "records", method: "shortlist", compression: "none" });
    wellFormed(records, TICKETS);
  });

  it("never names a blank line, nor a prose title", () => {
    for (const instructions of ["One?", "Two?", "Three?", "Four?", "Five?"]) {
      const log = ask({ within: "/log", instructions, compression: "none" }).answer;
      expect(log.ranking.map((r) => r.segment), instructions).not.toContain(3);
      const prose = ask({ within: "/prose", instructions }).answer;
      for (const { segment } of prose.ranking) expect(proseLines[segment], instructions).not.toMatch(/^# |^$/);
    }
  });

  it("carries found on the three measured routes and on no other", () => {
    const routes: [Record<string, unknown>, unknown[], boolean][] = [
      [{ within: "/log", kind: "log", compression: "template_fold" }, logLines, true],
      [{ within: "/prose", kind: "prose", compression: "none" }, proseLines, true],
      [{ within: "/tickets", kind: "records", compression: "none" }, TICKETS, true],
      [{ within: "/log", kind: "log", compression: "none" }, logLines, false],
      [{ within: "/tickets", kind: "records", compression: "template_fold" }, TICKETS, false],
      [{ within: "/log", method: "vote" }, logLines, false],
      [{ within: "/prose", method: "vote", compression: "none" }, proseLines, false],
      [{ within: "/tickets", method: "vote" }, TICKETS, false],
    ];
    for (const [question, values, finds] of routes) {
      const { status, answer } = ask(question);
      const route = JSON.stringify(question);
      expect(status, route).toBe(200);
      expect(answer.found !== undefined, route).toBe(finds);
      wellFormed(answer, values);
      // A route without `found` always names a segment.
      if (!finds) expect(answer.segment, route).not.toBeNull();
    }
  });

  it("answers not found under 0.5 on a measured route — no segment, no pointer, the ranking kept", () => {
    for (const within of ["/log", "/prose", "/tickets"]) {
      const { answer } = ask({ within, instructions: "Which one says the moon landed? /absent" });
      expect(answer.found, within).toBeLessThan(0.5);
      expect(answer.segment, within).toBeNull();
      expect(answer.value, within).toBeNull();
      expect(answer.confidence, within).toBeNull();
      expect(answer.pointers, within).toEqual([]);
      expect(answer.ranking.length, within).toBeGreaterThan(0);
    }
    // And names one all the same where the route has no `found` to say so.
    expect(ask({ within: "/log", compression: "none", instructions: "/absent" }).answer.segment).not.toBeNull();
    expect(ask({ within: "/log", method: "vote", instructions: "/absent" }).answer.segment).not.toBeNull();
  });

  it("answers found at 0.5 or more with the pick, and comes back not found for some questions by the seed alone", () => {
    const answers = Array.from({ length: 40 }, (_, i) => ask({ within: "/log", instructions: `Question ${i}?` }).answer);
    const found = answers.filter((a) => (a.found ?? 0) >= 0.5);
    const absent = answers.filter((a) => (a.found ?? 1) < 0.5);
    expect(found.length).toBeGreaterThan(0);
    expect(absent.length).toBeGreaterThan(0);
    for (const answer of found) {
      expect(answer.segment).not.toBeNull();
      wellFormed(answer, logLines);
    }
    for (const answer of absent) expect(answer.segment).toBeNull();
    // Some of them point at more than one segment, and a fold scales every
    // share by its first level's pick, so none reaches 1.
    expect(found.some((a) => a.pointers.length > 1)).toBe(true);
  });

  it("tells auto's kind: records by shape, a log from prose by the lines' shapes — and never answers auto", () => {
    expect(ask({ within: "/tickets" }).answer.kind).toBe("records");
    expect(ask({ within: "/log" }).answer.kind).toBe("log");
    expect(ask({ within: "/prose" }).answer.kind).toBe("prose");
    // An array of strings is not a records array: its elements' text decides.
    const notes = ask({ within: "/notes" }).answer;
    expect(notes.kind).not.toBe("records");
    wellFormed(notes, state.notes);
    expect(ask({ within: "/log", kind: "auto" }).answer.kind).toBe("log");
  });

  it("answers an element of an array of strings as the string itself", () => {
    const answer = ask({ within: "/notes", method: "vote" }).answer;
    expect(state.notes).toContain(answer.value);
    expect(answer.pointers[0].value).toBe(answer.value);
  });

  it("refuses an unknown value naming the accepted ones", () => {
    const method = ask({ within: "/log", method: "head" });
    expect(method.status).toBe(422);
    expect(method.error).toMatchObject({ code: "method_unknown" });
    expect(method.error?.message).toContain('"shortlist" and "vote"');
    const kind = ask({ within: "/log", kind: "table" });
    expect(kind.error).toMatchObject({ code: "kind_unknown" });
    expect(kind.error?.message).toContain('"auto", "log", "prose" and "records"');
    const compression = ask({ within: "/log", compression: "zip" });
    expect(compression.error).toMatchObject({ code: "compression_unknown" });
    expect(compression.error?.message).toContain('"template_fold" and "none"');
  });

  it("refuses a fold under a vote, a fold of prose named or told, and names the question", () => {
    for (const question of [
      { within: "/log", method: "vote", compression: "template_fold" },
      { within: "/log", kind: "prose", compression: "template_fold" },
      { within: "/prose", compression: "template_fold" },
    ]) {
      const { status, error } = ask(question);
      expect(status, JSON.stringify(question)).toBe(422);
      expect(error?.code, JSON.stringify(question)).toBe("compression_unsupported");
      expect(error?.message).toContain('question "q"');
    }
    // A vote left to its kind's compression is served: a vote reads as it is.
    expect(ask({ within: "/log", method: "vote" }).status).toBe(200);
  });

  it("refuses a kind the target contradicts, both ways", () => {
    expect(ask({ within: "/log", kind: "records" }).error).toMatchObject({ code: "kind_mismatch" });
    expect(ask({ within: "/notes", kind: "records" }).error).toMatchObject({ code: "kind_mismatch" });
    for (const kind of ["log", "prose"]) {
      const { error } = ask({ within: "/tickets", kind });
      expect(error, kind).toMatchObject({ code: "kind_mismatch" });
      expect(error?.message).toContain('kind "records" with compression "template_fold"');
    }
  });

  it("refuses kind and compression on every other type", () => {
    const kind = send({ state: "s", questions: { a: { type: "noul", instructions: "Urgent?", kind: "log" } } });
    expect(kind.status).toBe(422);
    expect(JSON.stringify(kind.body)).toContain("kind_unsupported");
    const compression = send({ state: "s", questions: { a: { type: "choice", instructions: "Which?", compression: "none" } } });
    expect(JSON.stringify(compression.body)).toContain("compression_unsupported");
  });

  it("answers every route the same way twice", () => {
    for (const question of [{ within: "/log" }, { within: "/prose" }, { within: "/tickets", method: "vote" }]) {
      expect(ask(question).answer, JSON.stringify(question)).toEqual(ask(question).answer);
    }
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
