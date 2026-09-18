import { describe, expect, it } from "vitest";
import { dataUriBytes, fitted, imageFromFile, MAX_EDGE } from "./images.ts";

describe("fitted", () => {
  it("leaves an image that already fits alone, rather than blowing it up", () => {
    expect(fitted(320, 200)).toEqual({ width: 320, height: 200 });
  });

  it("brings the long edge down to the bound and keeps the aspect ratio", () => {
    expect(fitted(MAX_EDGE * 2, MAX_EDGE)).toEqual({ width: MAX_EDGE, height: MAX_EDGE / 2 });
    expect(fitted(MAX_EDGE, MAX_EDGE * 4)).toEqual({ width: MAX_EDGE / 4, height: MAX_EDGE });
  });

  it("never rounds a very thin image away to nothing", () => {
    expect(fitted(20000, 3).height).toBe(1);
  });
});

describe("dataUriBytes", () => {
  it("reads the decoded size back off the payload, padding included", () => {
    expect(dataUriBytes("data:image/jpeg;base64,YWJj")).toBe(3);
    expect(dataUriBytes("data:image/jpeg;base64,YWJjZA==")).toBe(4);
    expect(dataUriBytes("data:image/jpeg;base64,YWJjZGU=")).toBe(5);
  });
});

describe("imageFromFile", () => {
  it("refuses a file that is not an image, naming it", async () => {
    const file = new File(["not a picture"], "notes.txt", { type: "text/plain" });
    const result = await imageFromFile(file);
    expect(result).toEqual({ ok: false, error: "notes.txt is not an image." });
  });
});
