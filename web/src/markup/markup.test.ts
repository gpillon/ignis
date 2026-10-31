import { describe, expect, it } from "vitest";
import { colorName, ellipseOf, extendStroke, featherOf, isArea, isMark, promptStub, rectOf, redactionPlan, toImagePoint } from "./markup.ts";

// The markup editor's geometry (markup/markup.ts): marks in the picture's own
// pixels whatever it is shown at, and the prompt's first words for them.

describe("the marks' geometry", () => {
  it("maps a pointer on the shown picture to the picture's own pixels, held inside it", () => {
    const rect = { left: 100, top: 50, width: 500, height: 250 };
    const natural = { width: 2000, height: 1000 };
    expect(toImagePoint(350, 175, rect, natural)).toEqual([1000, 500]);
    expect(toImagePoint(50, 400, rect, natural)).toEqual([0, 1000]);
  });

  it("adds a point to a stroke only once the pointer has moved far enough", () => {
    const start: [number, number][] = [[10, 10]];
    expect(extendStroke(start, [10.5, 10.5], 2)).toBe(start);
    expect(extendStroke(start, [13, 10], 2)).toEqual([
      [10, 10],
      [13, 10],
    ]);
  });

  it("reads an ellipse from either corner of its drag, and drops one too small to mean anything", () => {
    expect(ellipseOf([300, 100], [100, 300])).toEqual({ cx: 200, cy: 200, rx: 100, ry: 100 });
    expect(isMark({ kind: "ellipse", color: "#fff", width: 4, from: [0, 0], to: [3, 30] })).toBe(false);
    expect(isMark({ kind: "ellipse", color: "#fff", width: 4, from: [0, 0], to: [30, 30] })).toBe(true);
    expect(isMark({ kind: "stroke", color: "#fff", width: 4, points: [[1, 1]] })).toBe(true);
  });

  it("feathers the kept edge by about half a percent of the short side", () => {
    expect(featherOf(2048, 2048)).toBe(12);
    expect(featherOf(100, 80)).toBe(2);
  });
});

describe("the prompt's first words for the marks", () => {
  it("match how the marks went", () => {
    expect(promptStub("drawn", { picture: 2 }, "#ff2a2a")).toBe("In image 2, change what is marked in red: ");
    expect(promptStub("mask", { picture: 1, mask: 2 }, "#ff2a2a")).toBe("In image 1, change only the area that is white in image 2: ");
    expect(colorName("#123456")).toBe("the marked colour");
    expect(promptStub("drawn", { picture: 1 }, "#000000")).toBe("In image 1, change what is marked in black: ");
  });
});

// A context image's editing (spec #428): a crop and black-outs, as rectangles
// in the picture's own pixels, made into the picture that is kept.
describe("the crop and the black-outs", () => {
  const picture = { width: 200, height: 100 };

  it("draw a rectangle from either corner of the drag, widened to whole pixels and held inside the picture", () => {
    expect(rectOf([10.4, 20.6], [3.2, 5.5], picture)).toEqual({ x: 3, y: 5, width: 8, height: 16 });
    expect(rectOf([-5, 90], [30, 140], picture)).toEqual({ x: 0, y: 90, width: 30, height: 10 });
  });

  it("take no area from a click, only from a drag", () => {
    expect(isArea(rectOf([40.2, 40.2], [40.6, 40.6], picture))).toBe(false);
    expect(isArea(rectOf([40, 40], [42, 42], picture))).toBe(true);
  });

  it("keep the whole picture, nothing painted, when nothing is drawn", () => {
    expect(redactionPlan([], picture)).toEqual({ crop: { x: 0, y: 0, width: 200, height: 100 }, blackouts: [] });
  });

  it("keep the last crop drawn, and paint each black-out where it falls in it, one outside it dropped", () => {
    const plan = redactionPlan(
      [
        { kind: "blackout", rect: { x: 10, y: 10, width: 30, height: 20 } },
        { kind: "crop", rect: { x: 20, y: 0, width: 100, height: 50 } },
        { kind: "blackout", rect: { x: 150, y: 60, width: 20, height: 20 } },
        { kind: "blackout", rect: { x: 100, y: 70, width: 40, height: 40 } },
        { kind: "crop", rect: { x: 0, y: 5, width: 120, height: 80 } },
      ],
      picture,
    );
    expect(plan.crop).toEqual({ x: 0, y: 5, width: 120, height: 80 });
    // In the kept picture's own pixels: the first as it was drawn, the last cut at the crop's edges.
    expect(plan.blackouts).toEqual([
      { x: 10, y: 5, width: 30, height: 20 },
      { x: 100, y: 65, width: 20, height: 15 },
    ]);
  });
});
