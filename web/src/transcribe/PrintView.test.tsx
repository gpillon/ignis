import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { StaticPictures } from "../image/pictures.tsx";
import { EMPTY_CONTEXT } from "./context.ts";
import { addContextImage } from "./contextImages.ts";
import { noNotes } from "./notes.ts";
import { printDocument } from "./pdf.ts";
import { PdfDialog, PrintedTranscript } from "./PrintView.tsx";
import { addUtterance, createTranscript, renameSpeaker, type Transcript } from "./transcript.ts";

// The PDF's print view and its dialog (spec #428, "Exports"), drawn as static
// markup: the view draws the document pdf.ts computes, nothing more.

function call(): Transcript {
  let t = createTranscript(1, { source: "call", at: new Date(2026, 9, 25, 15, 42) });
  t = addUtterance(t, { stream: "mic-1", startMs: 1_200, endMs: 3_900, text: "Mi sentite?", voice: { kind: "me" } });
  t = addUtterance(t, { stream: "call-1", startMs: 4_480, endMs: 9_020, text: "Sì, <forte> e chiaro.", voice: { kind: "speaker", index: 0 } });
  t = renameSpeaker(t, t.speakers[1].id, "Ada");
  t = addContextImage(t, { image: "slide", atMs: 4_000, width: 1280, height: 720 }, 0);
  return {
    ...t,
    context: { ...EMPTY_CONTEXT, topic: "Il budget", terms: ["OKR"] },
    recap: { ...noNotes("general"), lines: [{ id: "N1", section: "Key points", text: "The line **works**.", author: "model" }], nextId: 2, writtenAs: "general", covered: 2, writtenAt: 1 },
    asked: [{ id: 1, question: "Who answered?", answer: "Ada, at [00:04].", at: 1, untilMs: 9_020 }],
  };
}

const drawn = (t: Transcript, options = { recap: true, context: true, answers: true }) =>
  renderToStaticMarkup(
    <StaticPictures urls={{ "full:slide": "blob:slide-full", "thumb:slide": "blob:slide-thumb" }}>
      <PrintedTranscript blocks={printDocument(t, options)} />
    </StaticPictures>,
  );

describe("the print view", () => {
  it("draws the header: the ignis mark, the title, the start, the duration and the speakers", () => {
    const html = drawn(call());
    expect(html).toMatch(/<img[^>]*class="pdf-mark"/);
    expect(html).toContain("<h1 class=\"pdf-title\">25 Oct 15:42 · Call</h1>");
    expect(html).toContain("<dt>Started</dt><dd>25 Oct 2026, 15:42</dd>");
    expect(html).toContain("<dt>Duration</dt><dd>00:09</dd>");
    expect(html).toContain("<dt>Speakers</dt><dd>Me, Ada</dd>");
  });

  it("draws the context in short, then the notes under their title in their sections, then the Transcript heading", () => {
    const html = drawn(call());
    const at = (text: string) => html.indexOf(text);
    expect(html).toContain("<dt>Topic</dt><dd>Il budget</dd>");
    expect(html).toContain("<dt>Terms</dt><dd>OKR</dd>");
    expect(html).toMatch(/<h2 class="pdf-heading">Recap<\/h2>/);
    expect(html).toContain("<h2>Key points</h2>");
    expect(html).toContain("<strong>works</strong>");
    expect(at("<dt>Topic</dt>")).toBeLessThan(at(">Recap<"));
    expect(at(">Recap<")).toBeLessThan(at(">Transcript<"));
    expect(at(">Transcript<")).toBeLessThan(at("Mi sentite?"));
  });

  it("draws every line with its time and its speaker in bold, the text as written", () => {
    const html = drawn(call(), { recap: false, context: false, answers: false });
    expect(html).toMatch(/<time class="pdf-time">00:01<\/time><p><b class="pdf-speaker">Me<\/b> Mi sentite\?<\/p>/);
    expect(html).toContain("<b class=\"pdf-speaker\">Ada</b> Sì, &lt;forte&gt; e chiaro.");
    expect(html).not.toContain(">Transcript<");
  });

  it("draws a context image at its moment, the whole picture, not its thumbnail", () => {
    const html = drawn(call());
    expect(html).toMatch(/<time class="pdf-time">00:04<\/time><figure><img src="blob:slide-full" alt="Context image at 00:04"/);
    expect(html).not.toContain("blob:slide-thumb");
    expect(html.indexOf("blob:slide-full")).toBeLessThan(html.indexOf("forte"));
  });

  it("marks a picture still being read, so the print waits for it", () => {
    const html = renderToStaticMarkup(
      <StaticPictures urls={{}}>
        <PrintedTranscript blocks={printDocument(call(), { recap: false, context: false, answers: false })} />
      </StaticPictures>,
    );
    expect(html).toMatch(/<span[^>]*aria-label="Context image at 00:04"[^>]*data-pending="true"/);
  });

  it("draws an answer as a box titled by its question, at its moment", () => {
    const html = drawn(call());
    expect(html).toMatch(/<li class="pdf-row pdf-answer"><time class="pdf-time">00:09<\/time><section aria-label="Answer to: Who answered\?"><h3>Who answered\?<\/h3>/);
    expect(html).toContain("Ada, at [00:04].");
    expect(html.indexOf("Who answered?")).toBeGreaterThan(html.indexOf("forte"));
  });

  it("writes the footer's page rule with the title", () => {
    expect(drawn(call())).toContain('content: "ignis · 25 Oct 15:42 · Call · page " counter(page) "/" counter(pages)');
  });
});

describe("the PDF dialog", () => {
  const dialog = (options = { recap: true, context: false, answers: true }) =>
    renderToStaticMarkup(<PdfDialog transcript={call()} options={options} onOptions={() => {}} onPrint={() => {}} onCancel={() => {}} />);

  it("offers Recap, the context in short and Ask's answers, each checked as chosen, with what it adds", () => {
    const html = dialog();
    expect(html).toMatch(/<input type="checkbox"[^>]*checked=""[^>]*\/>.*Recap/s);
    expect(html).toContain("Recap");
    expect(html).toContain("Context in short");
    expect(html).toContain("Ask&#x27;s answers");
    expect(html.match(/checked=""/g)).toHaveLength(2);
    expect(html).toContain("The recap, before the transcript.");
    expect(html).toContain("Topic and terms, after the header.");
    expect(html).toContain("1 answer, boxed at the moment it was asked.");
  });

  it("prints through the browser's print dialog, and says the header, the lines and the pictures always go", () => {
    const html = dialog();
    expect(html).toMatch(/<button[^>]*>Print…<\/button>/);
    expect(html).toContain("Cancel");
    expect(html).toContain("The header, the lines and the context images are always in it.");
    expect(html).toContain("print dialog");
  });
});
