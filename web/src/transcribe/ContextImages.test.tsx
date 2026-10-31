import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { StaticPictures } from "../image/pictures.tsx";
import { ImageAside, ImagesCard } from "./ContextImages.tsx";
import { addContextImage, type ContextImage, withReading } from "./contextImages.ts";
import { type ImageReaderState, readingStatus } from "./imageReading.ts";
import { Sheet } from "./Sheet.tsx";
import { addUtterance, createTranscript, type Transcript } from "./transcript.ts";

// A context image's reading in the view (#437): on the spine, under the
// picture, what was read in it, or that it is being read, waits for a text
// model with vision, or failed with Read again; the Call tab's Images card
// says how many wait and why.

const SLIDE = "e".repeat(64);
const IDLE: ImageReaderState = { reading: null, failed: [] };

function shown(): Transcript {
  const t = addUtterance(createTranscript(3), { stream: "s", startMs: 4_000, endMs: 8_000, text: "Ecco la slide.", voice: { kind: "me" } });
  return addContextImage(addContextImage(t, { image: SLIDE, atMs: 2_000, width: 1280, height: 720 }, 1), { image: SLIDE, atMs: 9_000, width: 1280, height: 720 }, 2);
}

const aside = (image: ContextImage, state: ImageReaderState, canRead: boolean, onRetry?: () => void) =>
  renderToStaticMarkup(
    <StaticPictures urls={{}}>
      <ImageAside image={image} status={readingStatus(3, image, state, canRead)} onRetry={onRetry} />
    </StaticPictures>,
  );

describe("a context image's reading on the spine", () => {
  it("shows what was read under the picture: what it shows, its topic, the names and the terms", () => {
    const t = withReading(shown(), 1, { shows: "A slide titled Budget Q3.", topic: "Budget", names: ["Anna Rossi"], terms: ["OKR", "Kubernetes"], readAt: 1 });
    const html = aside(t.images![0], IDLE, true);
    expect(html).toContain("A slide titled Budget Q3.");
    expect(html).toMatch(/>Topic<\/dt><dd[^>]*>Budget<\/dd>/);
    expect(html).toMatch(/>Names<\/dt><dd[^>]*>Anna Rossi<\/dd>/);
    expect(html).toMatch(/>Terms<\/dt><dd[^>]*>OKR, Kubernetes<\/dd>/);
    expect(html).not.toContain("Read again");
  });

  it("says its reading waits while no text model with vision serves, and that it runs once one does", () => {
    const html = aside(shown().images![0], IDLE, false);
    expect(html).toContain("Not read yet: no text model with vision serves. It is read once one does.");
  });

  it("says it is being read, or waits its turn, while one serves", () => {
    const reading = aside(shown().images![0], { reading: { transcript: 3, image: 1 }, failed: [] }, true);
    expect(reading).toContain('aria-busy="true"');
    expect(reading).toContain("Reading the picture.");
    expect(aside(shown().images![1], { reading: { transcript: 3, image: 1 }, failed: [] }, true)).toContain("To be read.");
  });

  it("says why a reading failed, with Read again", () => {
    const html = aside(shown().images![0], { reading: null, failed: [{ transcript: 3, image: 1, message: "vision is disabled" }] }, true, () => {});
    expect(html).toContain("The reading failed: vision is disabled.");
    expect(html).toContain(">Read again</button>");
  });
});

describe("the sheet's images", () => {
  const none = () => {};
  const actions = { moveUtterance: none, splitUtterance: none, editText: none, undoFixes: none, deleteUtterance: none, undoDelete: none, holdUndo: none, rewriteUtterance: none, acceptRewrite: none, closeRewrite: none, forgetAnswer: none, askFrom: none, readImageAgain: none };
  const sheet = (t: Transcript, imageReading?: ImageReaderState, canRead?: boolean) =>
    renderToStaticMarkup(
      <StaticPictures urls={{}}>
        <Sheet transcript={t} live={[]} phase="idle" call={false} here actions={actions} deleted={null} hits={[]} current={0} rewrite={null} canRewrite readTo={null} seek={0} autoCheck asking={null} imageReading={imageReading} canRead={canRead} />
      </StaticPictures>,
    );

  it("say under each picture where its reading stands, by the reader and the text model", () => {
    const t = withReading(shown(), 2, { shows: "The closing slide.", topic: null, names: [], terms: [], readAt: 1 });
    expect(sheet(t)).toContain("Not read yet: no text model with vision serves.");
    expect(sheet(t)).toContain("The closing slide.");
    const html = sheet(t, { reading: null, failed: [{ transcript: 3, image: 1, message: "The model wrote no reading." }] }, true);
    expect(html).toContain("The reading failed: The model wrote no reading.");
    expect(html).toContain(">Read again</button>");
  });
});

describe("the Call tab's images", () => {
  const card = (t: Transcript, canRead: boolean) =>
    renderToStaticMarkup(<ImagesCard transcript={t} sharing={false} canRead={canRead} adding={{ open: () => {}, capture: () => {}, capturing: false, error: null }} />);

  it("say the text model with vision reads each picture once, and how many wait for one", () => {
    expect(card(shown(), true)).toContain("The text model with vision reads each picture once");
    expect(card(shown(), false)).toContain("2 images wait to be read: no text model with vision serves.");
    const one = withReading(shown(), 1, { shows: "A slide.", topic: null, names: [], terms: [], readAt: 1 });
    expect(card(one, false)).toContain("1 image waits to be read: no text model with vision serves.");
    expect(card(one, true)).not.toContain("to be read: no text model");
  });
});
