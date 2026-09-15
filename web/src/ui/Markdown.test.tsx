import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Markdown, textOf } from "./Markdown.tsx";

const html = (text: string) => renderToStaticMarkup(<Markdown text={text} />);

describe("Markdown", () => {
  it("renders headings, emphasis, inline code and lists", () => {
    const out = html("## Title\n\n**bold** and `code`\n\n- one\n- two\n\n1. first");
    expect(out).toContain("<h2>Title</h2>");
    expect(out).toContain("<strong>bold</strong>");
    expect(out).toContain("<code>code</code>");
    expect(out).toContain("<li>one</li>");
    expect(out).toMatch(/<ol>\s*<li>first<\/li>/);
  });

  it("renders GitHub tables and strikethrough", () => {
    const out = html("| model | tok/s |\n|---|---:|\n| qwen | 68 |\n\n~~old~~");
    expect(out).toContain('<div class="md-table"><table>');
    expect(out).toMatch(/<td[^>]*>68<\/td>/);
    expect(out).toContain("<del>old</del>");
  });

  it("puts a fenced block in a code block labelled with its language", () => {
    const out = html("```rust\nfn main() {}\n```");
    expect(out).toContain("<span>rust</span>");
    expect(out).toContain("fn main() {}");
    expect(out).toContain(">Copy</button>");
  });

  it("does not render raw HTML", () => {
    const out = html('<script>alert(1)</script>\n\n<img src="x" onerror="alert(1)">');
    expect(out).not.toContain("<script");
    expect(out).not.toContain("<img");
  });

  it("drops unsafe link targets and opens links in a new tab", () => {
    expect(html("[x](javascript:alert(1))")).not.toContain("javascript:");
    expect(html("[docs](https://example.com)")).toContain('href="https://example.com" target="_blank"');
  });

  it("turns images into links instead of loading them", () => {
    const out = html("![a chart](https://example.com/chart.png)");
    expect(out).not.toContain("<img");
    expect(out).toContain('href="https://example.com/chart.png"');
    expect(out).toContain(">a chart</a>");
  });

  it("marks a streaming reply for the caret", () => {
    expect(renderToStaticMarkup(<Markdown text="hi" streaming />)).toContain('data-streaming="true"');
  });
});

describe("textOf", () => {
  it("collects the text of nested children", () => {
    expect(textOf(["a", 1, <b key="b">c{"d"}</b>, null])).toBe("a1cd");
  });
});
