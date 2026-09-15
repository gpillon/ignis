import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { HtmlPreview, isHtmlName, withPrintTitle } from "./HtmlPreview.tsx";
import { Markdown, normalizeMath } from "./Markdown.tsx";

const html = (text: string) => renderToStaticMarkup(<Markdown text={text} />);

describe("formulas", () => {
  it("renders $$ on its own line as display math and $…$ inline", () => {
    const out = html("Pressure:\n\n$$P = \\rho \\cdot g \\cdot h$$\n\nwith $\\rho$ the density.");
    expect(out).toContain('class="katex-display"');
    expect(out).toMatch(/<span class="katex"><span class="katex-mathml"><math[^>]*><semantics><mrow><mi>ρ<\/mi>/);
  });

  it("reads LaTeX's \\( \\) and \\[ \\] delimiters too, but never inside code", () => {
    expect(normalizeMath("a \\(x^2\\) b")).toBe("a $x^2$ b");
    expect(normalizeMath("\\[\\int f\\]")).toBe("\n$$\n\\int f\n$$\n");
    expect(normalizeMath("  $$E = mc^2$$  ")).toBe("$$\nE = mc^2\n$$");
    const code = "`\\(kept\\)`\n\n```\n\\[kept\\]\n$$kept$$\n```";
    expect(normalizeMath(code)).toBe(code);
    expect(html("\\[a + b\\]")).toContain('class="katex-display"');
  });
});

describe("HTML previews", () => {
  it("names the PDF after the file when the page has no title", () => {
    expect(withPrintTitle("<html><head><meta charset=utf-8></head><body>x</body></html>", "report.html")).toBe(
      "<html><head><title>report</title><meta charset=utf-8></head><body>x</body></html>",
    );
    expect(withPrintTitle("<h1>x</h1>", "a<b>.htm")).toBe("<title>a&lt;b&gt;</title><h1>x</h1>");
    expect(withPrintTitle("<title>Kept</title><p>x</p>", "report.html")).toBe("<title>Kept</title><p>x</p>");
  });

  it("offers a preview on an html code block only", () => {
    expect(html("```html\n<h1>Hi</h1>\n```")).toContain(">Preview</button>");
    expect(html("```js\nlet a\n```")).not.toContain(">Preview</button>");
  });

  it("shows a file's preview first, in a sandbox without the page's origin", () => {
    const out = renderToStaticMarkup(<HtmlPreview html="<h1>Hi</h1>" name="hello.html" />);
    expect(out).toContain('sandbox="allow-scripts allow-forms"');
    expect(out).not.toContain("allow-same-origin");
    expect(out).toContain('aria-selected="true"');
    expect(out).toMatch(/<iframe[^>]*srcDoc="&lt;h1&gt;Hi&lt;\/h1&gt;"/i);
    expect(out).toContain(">Expand preview</button>");
    expect(out).toContain(">Save as PDF</button>");
    expect(isHtmlName("page.HTML")).toBe(true);
    expect(isHtmlName("notes.md")).toBe(false);
  });
});
