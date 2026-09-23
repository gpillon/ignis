import { describe, expect, it } from "vitest";
import { dataUriBytes, fitted, imageFromFile, imageMime, MAX_EDGE } from "./images.ts";

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

describe("imageMime", () => {
  it("keeps a type that names an image", () => {
    expect(imageMime("image/webp", "image/png")).toBe("image/webp");
    expect(imageMime("image/jpeg", "image/png")).toBe("image/jpeg");
  });

  it("falls back on a type that is truthy and still not an image (GitHub #256)", () => {
    // What a server with no branch for the extension answers: truthy, so
    // `blob.type || fallback` kept it and the picture was refused.
    expect(imageMime("application/octet-stream", "image/webp")).toBe("image/webp");
  });

  it("falls back on nothing at all", () => {
    expect(imageMime("", "image/webp")).toBe("image/webp");
    expect(imageMime(undefined, "image/webp")).toBe("image/webp");
  });
});
