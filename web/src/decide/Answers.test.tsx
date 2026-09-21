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
  it("heads a panel with what was asked, and keeps the answer name as a label beside it", () => {
    const asked = "Which team should handle this?";
    const html = render([{ ...question("department", "choice", { options: [{ key: "a", description: "" }] }), instructions: jsonString(asked) }], {
      department: { type: "choice", choice: "a", probabilities: { a: 1 }, confidence: 1 },
    });
    expect(html).toContain(`>${asked}</h3>`);
    expect(html).toContain('title="The key this answer came back under"');
    expect(html).toContain("department");
  });

  it("sets the question in the ink and the headline figure in the ember, and rules one answer off from the next", () => {
    const html = render([question("a", "noul"), question("b", "noul"), question("c", "noul")], {
      a: { type: "noul", noul: 0.1 },
      b: { type: "noul", noul: 0.2 },
      c: { type: "noul", noul: 0.3 },
    });
    // The question is text and wears a text token; the ember is spent on the
    // one figure the panel exists to report.
    expect(html).toMatch(/<h3 class="[^"]*font-bold[^"]*text-ink[^"]*">/);
    expect(html).toMatch(/<p class="[^"]*text-\[28px\][^"]*text-ember[^"]*">No<\/p>/);
    // A rule between panels, and not above the first one.
    const items = html.match(/<li class="[^"]*"><section>/g) ?? [];
    expect(items).toHaveLength(3);
    expect(items[0]).toBe('<li class=""><section>');
    expect(items.slice(1).every((li) => li.includes("border-t"))).toBe(true);
  });

  it("falls back to the answer name when a question asked nothing", () => {
    const html = render([{ ...question("bare", "noul"), instructions: jsonString("") }], { bare: { type: "noul", noul: 0.5 } });
    expect(html).toMatch(/<h3[^>]*>bare<\/h3>/);
  });

  it("shows the image as a preview that opens at full size, not as a column-wide picture", () => {
    const html = render(
      [question("p", "point")],
      { p: { type: "point", pixels: { x: 12, y: 8 }, normalized: { x: 30, y: 40 }, uncertainty: { x: 1, y: 1 }, digits: {} } },
      { mode: "image", images: [image], text: "" },
      6,
    );
    expect(html).toContain("max-h-56");
    expect(html).toContain("Full size");
    expect(html).toContain("cursor-zoom-in");
    // The picture and its figures read side by side, the picture centred in
    // its half while the caption spans it.
    expect(html).toContain("sm:grid-cols-2");
    expect(html).toContain("mx-auto block w-fit");
    // The caveat is about the figures as much as the drawing, so it runs under
    // both columns rather than under the picture.
    expect(html).toMatch(/<p class="[^"]*sm:col-span-2[^"]*">\s*The halo is the model/);
  });

  it("closes with the cost, under the answers it paid for, and says a readout generated nothing", () => {
    const html = render([{ ...question("a", "noul"), instructions: jsonString("Urgent?") }], { a: { type: "noul", noul: 0.86 } });
    expect(html).toContain("312");
    expect(html).toContain("84");
    expect(html).toContain("Nothing was generated");
    // The receipt comes last: after the answer, and above the raw body.
    expect(html.indexOf("Urgent?")).toBeLessThan(html.indexOf("prompt tokens"));
    expect(html.indexOf("prompt tokens")).toBeLessThan(html.indexOf("answer mass"));
    expect(html.indexOf("answer mass")).toBeLessThan(html.indexOf("Response body"));
  });

  it("says the generated tokens are real when a constrained decode was asked", () => {
    const html = render([question("n", "number")], { n: { type: "number", number: 12, uncertainty: 1.4, digits: [] } }, undefined, 3);
    expect(html).toContain("generates a digit per step");
  });

  it("leads a noul with the option that won, in the caller's own words, and reads the confidence under it", () => {
    const yes = render([question("a", "noul", { yes: "Time-sensitive", no: "Not urgent" })], { a: { type: "noul", noul: 0.862 } });
    expect(yes).toContain(">Time-sensitive</p>");
    expect(yes).not.toContain("Not urgent");
    // `noul` carries no confidence of its own: the winner's own probability is
    // it, and the raw p stays on the page beside it.
    expect(yes).toContain("0.862");

    const no = render([question("a", "noul", { yes: "Time-sensitive", no: "Not urgent" })], { a: { type: "noul", noul: 0.138 } });
    expect(no).toContain(">Not urgent</p>");
    // 1 - p for a no, and the raw p still says which way it was read.
    expect(no).toContain("0.862");
    expect(no).toContain("p(yes) = 0.138");
  });

  it("falls back to Yes and No when a noul declared no descriptions", () => {
    const html = render([question("a", "noul")], { a: { type: "noul", noul: 0.2 } });
    expect(html).toContain(">No</p>");
    expect(html).toContain("p(yes) = 0.200");
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
    // n + 1 ticks for n levels: a level is the band between two of them.
    // Three levels, four ticks — at 0%, a third, two thirds, and the right
    // edge, which is drawn from the edge because `left: 100%` never shows.
    expect((html.match(/h-1\.5 w-px bg-line/g) ?? [])).toHaveLength(4);
    expect(html).toContain("left:0%");
    expect(html).toContain("left:33.33333333333333%");
    expect(html).toContain("left:66.66666666666666%");
    expect(html).toContain("right-0 h-1.5 w-px bg-line");
  });

  it("draws a score's levels on the probability's own scale, the same as a choice's rows", () => {
    const html = render([question("s", "score", { levels: ["Low", "High"] })], {
      s: { type: "score", score: 0.4, legend: { "0": "Low", "1": "High" }, probabilities: { "0": 0.6, "1": 0.4 }, confidence: 0.6 },
    });
    // Absolute, not normalized to the peak: normalizing would have made the
    // winning level 100% tall and unreadable beside a choice panel. The value a
    // bar grows to is on the element from the first paint.
    expect(html).toContain("--fill:60%");
    expect(html).toContain("--fill:40%");
  });

  it("says what a confidence means, and says a score's differently from a choice's", () => {
    const choice = render([question("c", "choice", { options: [{ key: "a", description: "" }] })], {
      c: { type: "choice", choice: "a", probabilities: { a: 1 }, confidence: 1 },
    });
    expect(choice).toContain("the winner&#x27;s own share");

    const score = render([question("s", "score", { levels: ["Low", "High"] })], {
      s: { type: "score", score: 0.5, legend: { "0": "Low", "1": "High" }, probabilities: { "0": 0.5, "1": 0.5 }, confidence: 0 },
    });
    // A score's is the spread, not the tallest bar — the one thing a
    // winner's-share confidence would hide.
    expect(score).toContain("how tightly the levels cluster");
    expect(score).toContain("Not the tallest bar");
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

  it("shows a scalar's spelling and the number that spelling parsed to, because they are not the same fact", () => {
    const html = render(
      [question("s", "scalar")],
      { s: { type: "scalar", value: 3, text: "3.0", uncertainty: 0.0035, digits: [{ digit: 3, probability: 0.99 }, { digit: 0, probability: 0.9 }] } },
      undefined,
      4,
    );
    // What it wrote heads the panel; what it parses to reads under it.
    expect(html).toContain(">3.0");
    expect(html).toContain(">3</span>");
  });

  it("prints a scalar's uncertainty in the value's own units, where a fixed decimal place would read 0.0", () => {
    const html = render(
      [question("s", "scalar")],
      { s: { type: "scalar", value: 12.4, text: "12.40", uncertainty: 0.0035, digits: [] } },
      undefined,
      6,
    );
    expect(html).toContain("0.0035");
    expect(html).not.toContain("± 0.0<");
    // Units, never a share: no bar and no percentage anywhere on the panel.
    expect(html).not.toContain("confidence");
    expect(html).not.toContain("--fill");
  });

  it("counts a scalar's rounds as what it wrote plus the brace, beside the field a number would have filled", () => {
    const html = render(
      [question("s", "scalar", { ceiling: 6 })],
      { s: { type: "scalar", value: -0.75, text: "-0.75", uncertainty: 0.0001, digits: [{ digit: 0, probability: 0.99 }, { digit: 7, probability: 0.98 }, { digit: 5, probability: 0.97 }] } },
      undefined,
      6,
    );
    // `-0.75` is six tokens: five characters and the closing brace.
    expect(html).toContain(">6</span> rounds");
    expect(html).toContain("a number 6 wide spends 6");
    // The sign and the point were steps of the run and hold no place, so the
    // trace is three columns against a five-character spelling — right, not
    // a gap.
    expect((html.match(/digit-land/g) ?? [])).toHaveLength(3);
    expect(html).toContain("a trace shorter than the spelling is right");
  });

  it("reads a scalar with no ceiling against the widest field the endpoint serves", () => {
    const html = render([question("s", "scalar")], { s: { type: "scalar", value: 3, text: "3", uncertainty: 0, digits: [] } }, undefined, 2);
    expect(html).toContain(">2</span> rounds");
    expect(html).toContain("a number 6 wide spends 6");
  });

  it("renders a scalar's own per-answer errors as errors, not as a blank card", () => {
    for (const code of ["malformed_scalar", "too_many_digits"]) {
      const html = render([question("s", "scalar")], { s: { type: "error", code, message: "the run is not a number" } });
      expect(html, code).toContain(code);
      expect(html, code).toContain("the run is not a number");
    }
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
    // White over black and never a hue: the mark lands on a picture nobody
    // chose for it. The strokes are in screen pixels, so the same crosshair is
    // readable in the preview and at full size.
    expect(html).toContain('stroke="#fff"');
    expect(html).toContain('stroke="#000"');
    expect(html).toContain('vector-effect="non-scaling-stroke"');
    // Sized off the image's shorter side (200), with a gap at the answer's own pixel.
    expect(html).toContain("M102,64H116 M124,64H138");
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
    // Four axes read two to a row, not as a column taller than the picture.
    expect(html).toContain("grid grid-cols-2");
    expect(html).toContain('stroke="#fff"');
    expect(html).toContain('vector-effect="non-scaling-stroke"');
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
