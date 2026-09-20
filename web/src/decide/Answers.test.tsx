import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Answers } from "./Answers.tsx";
import { jsonString } from "./json.ts";
import { type Draft, type Evidence, newQuestion, type Primitive, type Question } from "./model.ts";
import type { Answer, DecideResponse, Run } from "./request.ts";

// The answer panels (GitHub #247). What these check is that each primitive is
// read in the order the *request* declared, since the response's own order is
// alphabetical — and that every figure is on the page as text, not only as a
// mark.

const image = { name: "a.jpg", url: "data:image/jpeg;base64,AA", width: 400, height: 200 };

function question(id: string, kind: Primitive, extra: Partial<Question> = {}): Question {
  return { ...newQuestion(kind, id), instructions: jsonString("Decide."), ...extra };
}

function render(questions: Question[], answers: Record<string, Answer>, evidence: Evidence = { mode: "text", text: "e" }, generated = 0) {
  const draft: Draft = { evidence, questions, extras: [] };
  const response: DecideResponse = { model: "m", answers, usage: { input_tokens: 312, output_tokens: generated } };
  const run: Run = { response, elapsedMs: 84.4, raw: JSON.stringify(response) };
  return renderToStaticMarkup(<Answers draft={draft} run={run} />);
}

/** The order the given needles appear in the markup. */
const order = (html: string, needles: string[]) => [...needles].sort((a, b) => html.indexOf(a) - html.indexOf(b));

describe("Answers", () => {
  it("leads with the cost and says a readout generated nothing", () => {
    const html = render([question("a", "noul")], { a: { type: "noul", noul: 0.86 } });
    expect(html).toContain("312");
    expect(html).toContain("84");
    expect(html).toContain("Nothing was generated");
  });

  it("says the generated tokens are real when a constrained decode was asked", () => {
    const html = render([question("n", "number")], { n: { type: "number", number: 12, uncertainty: 1.4, digits: [] } }, undefined, 3);
    expect(html).toContain("generates a digit per step");
  });

  it("reads a noul as the probability of yes, with the caller's own descriptions", () => {
    const html = render([question("a", "noul", { yes: "Time-sensitive", no: "Not urgent" })], { a: { type: "noul", noul: 0.862 } });
    expect(html).toContain("0.862");
    expect(html).toContain("Time-sensitive");
    expect(html).toContain("Not urgent");
  });

  it("falls back to Yes and No when a noul declared no descriptions", () => {
    const html = render([question("a", "noul")], { a: { type: "noul", noul: 0.2 } });
    expect(html).toContain("0.200");
    expect(html).toContain("No");
  });

  it("lists a choice in the order the request declared, not the order it came back", () => {
    const options = [
      { key: "zulu", description: "last alphabetically, first declared" },
      { key: "alpha", description: "first alphabetically, last declared" },
    ];
    const html = render([question("pick", "choice", { options })], {
      pick: { type: "choice", choice: "alpha", probabilities: { alpha: 0.7, zulu: 0.3 }, confidence: 0.7 },
    });
    // Matched on the descriptions: the winner's own name also heads the panel.
    expect(order(html, ["first alphabetically", "last alphabetically"])).toEqual(["last alphabetically", "first alphabetically"]);
    expect(html).toContain("0.700");
    expect(html).toContain("0.300");
  });

  it("still shows an option the response carried that the request did not declare", () => {
    const html = render([question("pick", "choice", { options: [{ key: "a", description: "" }] })], {
      pick: { type: "choice", choice: "surprise", probabilities: { a: 0.4, surprise: 0.6 }, confidence: 0.6 },
    });
    expect(html).toContain("surprise");
  });

  it("orders a score's levels by index, where a string sort would put 10 before 2", () => {
    const levels = Array.from({ length: 12 }, (_, i) => `level ${i}`);
    const probabilities: Record<string, number> = {};
    const legend: Record<string, string> = {};
    levels.forEach((label, i) => {
      probabilities[String(i)] = i === 11 ? 0.5 : 0.5 / 11;
      legend[String(i)] = label;
    });
    const html = render([question("s", "score", { levels })], {
      s: { type: "score", score: 10.4, legend, probabilities, confidence: 0.8 },
    });
    expect(order(html, ["level 2", "level 10", "level 11"])).toEqual(["level 2", "level 10", "level 11"]);
  });

  it("puts a score's pointer between the two levels it falls between", () => {
    const html = render([question("s", "score", { levels: ["Calm", "Frustrated", "Very angry"] })], {
      s: { type: "score", score: 1.6, legend: { "0": "Calm", "1": "Frustrated", "2": "Very angry" }, probabilities: { "0": 0.05, "1": 0.3, "2": 0.65 }, confidence: 0.7 },
    });
    // 1.6 across three levels is 80% of the axis: past Frustrated, short of Very angry.
    expect(html).toContain("left:80%");
    expect(html).toContain("1.60");
  });

  it("shows a number with its uncertainty and one column per place", () => {
    const digits = [
      { digit: 3, probability: 0.98 },
      { digit: 1, probability: 0.74 },
      { digit: 4, probability: 0.31 },
    ];
    const html = render([question("n", "number")], { n: { type: "number", number: 314, uncertainty: 87.5, digits } }, undefined, 3);
    expect(html).toContain("314");
    expect(html).toContain("87.5");
    for (const p of ["0.98", "0.74", "0.31"]) expect(html).toContain(p);
    expect(html).toContain("not a bound");
  });

  it("draws a point at the answer's pixels, on the image the request submitted", () => {
    const html = render(
      [question("p", "point")],
      { p: { type: "point", pixels: { x: 120, y: 64 }, normalized: { x: 300, y: 320 }, uncertainty: { x: 4.5, y: 6 }, digits: {} } },
      { mode: "image", images: [image], text: "" },
      6,
    );
    expect(html).toContain('viewBox="0 0 400 200"');
    expect(html).toContain('cx="120"');
    expect(html).toContain('cy="64"');
    // The figures are in the table too, so the drawing is never the only copy.
    expect(html).toContain(">120<");
    expect(html).toContain(">300<");
    expect(html).toContain("not a bound");
  });

  it("draws a box and its per-edge uncertainty band", () => {
    const html = render(
      [question("b", "box")],
      {
        b: {
          type: "box",
          pixels: { x0: 10, y0: 20, x1: 110, y1: 120 },
          normalized: { x0: 25, y0: 100, x1: 275, y1: 600 },
          uncertainty: { x0: 2, y0: 3, x1: 4, y1: 5 },
          digits: {},
        },
      },
      { mode: "image", images: [image], text: "" },
      12,
    );
    expect(html).toContain('width="100"');
    expect(html).toContain('height="100"');
    // The band is the nominal rect grown by each edge's own sigma.
    expect(html).toContain('width="106"');
  });

  it("says a point has no image to draw on rather than drawing nowhere", () => {
    const html = render([question("p", "point")], {
      p: { type: "point", pixels: { x: 1, y: 1 }, normalized: { x: 1, y: 1 }, uncertainty: { x: 0, y: 0 }, digits: {} },
    });
    expect(html).toContain("no longer on this page");
  });

  it("renders a failed question beside its siblings' answers rather than in place of them", () => {
    const html = render([question("a", "noul"), question("b", "noul")], {
      a: { type: "noul", noul: 0.5 },
      b: { type: "error", code: "engine_full", message: "the engine refused this question" },
    });
    expect(html).toContain("0.500");
    expect(html).toContain("engine_full");
    expect(html).toContain("the engine refused this question");
  });

  it("says so when an answer is missing altogether", () => {
    const html = render([question("a", "noul")], {});
    expect(html).toContain("The other answers stand");
  });

  it("says once that the answer mass is not in the body, and points at the Monitor", () => {
    const html = render([question("a", "noul")], { a: { type: "noul", noul: 0.5 } });
    expect(html).toContain("answer mass");
    expect(html).toContain("the Monitor charts it");
  });
});
