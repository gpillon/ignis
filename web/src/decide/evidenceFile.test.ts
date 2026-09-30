import { describe, expect, it } from "vitest";
import { evidenceFromFile, evidenceFromText, formatBytes, isImageFile, MAX_TEXT_FILE_BYTES } from "./evidenceFile.ts";

// A file from disk as the Decide tab's evidence: the shape is told by the
// content, the text is the evidence as it was written, and the name rides
// beside it.

const file = { name: "app.log", size: 120 };

describe("evidenceFromText", () => {
  it("reads a log as text, and keeps the file it came from", () => {
    expect(evidenceFromText(file, "09:14 INFO up\n09:15 ERROR down\n")).toEqual({
      ok: true,
      evidence: { mode: "text", text: "09:14 INFO up\n09:15 ERROR down\n", file },
    });
  });

  it("gives a Windows log the line ends the server cuts on", () => {
    const read = evidenceFromText(file, "a\r\nb\r\nc");
    expect(read.ok && "evidence" in read && read.evidence.text).toBe("a\nb\nc");
  });

  it("reads a JSON object or array as a record, and anything that does not parse as text", () => {
    const shape = (name: string, text: string) => {
      const read = evidenceFromText({ name, size: text.length }, text);
      return read.ok && "evidence" in read ? read.evidence.mode : null;
    };
    expect(shape("order.json", '{"order": "A-4471"}')).toBe("json");
    expect(shape("tickets.txt", '  [{"id": 1}, {"id": 2}]')).toBe("json");
    // The extension does not decide: a broken .json is still a text to search,
    // and JSON Lines is lines — which is what a locate reads.
    expect(shape("broken.json", '{"order": ')).toBe("text");
    expect(shape("events.jsonl", '{"a": 1}\n{"a": 2}')).toBe("text");
  });

  it("refuses a binary file and an empty one, naming the file", () => {
    expect(evidenceFromText({ name: "core.dump", size: 3 }, "a\0b")).toEqual({ ok: false, error: "core.dump is not a text file." });
    expect(evidenceFromText({ name: "blank.log", size: 2 }, " \n")).toEqual({ ok: false, error: "blank.log is empty." });
  });
});

describe("evidenceFromFile", () => {
  it("reads a text file", async () => {
    const read = await evidenceFromFile(new File(["one\ntwo"], "two.log", { type: "text/plain" }));
    expect(read).toEqual({ ok: true, evidence: { mode: "text", text: "one\ntwo", file: { name: "two.log", size: 7 } } });
  });

  it("refuses a file over the tab's cap before reading a byte of it", async () => {
    let read = false;
    const huge = {
      name: "huge.log",
      type: "text/plain",
      size: MAX_TEXT_FILE_BYTES + 1,
      text: async () => {
        read = true;
        return "";
      },
    } as unknown as File;
    expect(await evidenceFromFile(huge)).toEqual({ ok: false, error: "huge.log is 32 MB, and this tab loads files up to 32 MB." });
    expect(read).toBe(false);
  });
});

describe("isImageFile", () => {
  it("knows a picture by its type or, failing that, its name", () => {
    expect(isImageFile(new File([""], "shot", { type: "image/png" }))).toBe(true);
    expect(isImageFile(new File([""], "shot.JPG"))).toBe(true);
    expect(isImageFile(new File([""], "app.log", { type: "text/plain" }))).toBe(false);
  });
});

describe("formatBytes", () => {
  it("says a size the way a file manager does", () => {
    expect(formatBytes(812)).toBe("812 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(6.1 * 1024 * 1024)).toBe("6.1 MB");
    expect(formatBytes(310 * 1024 * 1024)).toBe("310 MB");
  });
});
