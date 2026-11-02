// A command in the guide installs the latest release, never a version written
// into the page: the Install page's `$v = "v0.8.0"` kept installing 0.8.0 after
// 0.9.5 was out. So no code block of the guide, or of the root README, names a
// version. The contributing pages are the exception: their examples are the
// release tooling's own arguments (`make changelog FROM=v0.1.1 TO=v0.1.2`).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const guide = fileURLToPath(new URL("../../user/", import.meta.url));
const readme = fileURLToPath(new URL("../../../README.md", import.meta.url));

// changelog/ is written by the build, not committed (src/changelog.mjs); contributing/ is the exception above.
const EXEMPT = ["changelog", "contributing"].map((dir) => path.join(guide, dir));

/** Every .md and .mdx page of the guide outside EXEMPT, as absolute paths. */
function pages(dir = guide) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return EXEMPT.includes(full) ? [] : pages(full);
    return /\.mdx?$/.test(entry.name) ? [full] : [];
  });
}

const VERSION = /(?<![\d.])v?\d+\.\d+\.\d+(?![\d.])/;

/** The lines of the text's fenced code blocks that name a version: x.y.z, with or without a v. */
function versionsInCode(text) {
  const found = [];
  let fence = null;
  for (const line of text.split(/\r?\n/)) {
    const marker = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
    if (marker && !fence) {
      fence = marker[1];
      continue;
    }
    if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length && marker[2].trim() === "") {
      fence = null;
      continue;
    }
    if (fence && VERSION.test(line)) found.push(line.trim());
  }
  return found;
}

describe("versionsInCode", () => {
  it("finds a version inside a code block, in either spelling", () => {
    const text = ['```powershell', '$v = "v0.8.0"', "```", "", "```sh", "v=0.8.0", "```"].join("\n");
    expect(versionsInCode(text)).toEqual(['$v = "v0.8.0"', "v=0.8.0"]);
  });
  it("leaves prose, addresses and a longer fence's inner fences alone", () => {
    const text = ["Since 0.8.0 ignis serves it.", "````", "```python", "http://127.0.0.1:8000", "```", "````"].join("\n");
    expect(versionsInCode(text)).toEqual([]);
  });
});

describe.each([readme, ...pages()].map((file) => [path.relative(guide, file), file]))("%s", (_, file) => {
  it("names no version in a code block", () => {
    expect(versionsInCode(fs.readFileSync(file, "utf8"))).toEqual([]);
  });
});
