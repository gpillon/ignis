import { type KeyboardEvent, useEffect, useReducer, useRef, useState } from "react";
import { createPortal } from "react-dom";
import flame from "../brand/flame.webp";
import { usePictureUrl } from "../image/pictures.tsx";
import type { ImageId } from "../store/imageStore.ts";
import { Markdown } from "../ui/Markdown.tsx";
import { sheetKeyDown } from "../ui/Sheet.tsx";
import { fileStem } from "./export.ts";
import { pageRule, pdfHints, type PdfOptions, type PrintBlock, printDocument } from "./pdf.ts";
import { titleOf } from "./sessions.ts";
import type { Transcript } from "./transcript.ts";

// A transcript's PDF in the view (spec #428, "Exports"): the Download menu's
// PDF opens a dialog over what the PDF carries; its Print draws the document
// pdf.ts computes in a sheet of its own, which the page prints alone while it
// stands (styles.css, `.print-sheet`), once its pictures and fonts are in, and
// the browser's print dialog saves the file. Nothing new is bundled for it.

/** The dialog's options, in order, each with its name. */
const OPTIONS: [keyof PdfOptions, string][] = [
  ["recap", "Recap"],
  ["context", "Context in short"],
  ["answers", "Ask's answers"],
];

/** What the PDF carries beside the header and the lines, asked under the Download menu; `options` as last chosen when it opens. */
export function PdfDialog({
  transcript,
  options,
  onOptions,
  onPrint,
  onCancel,
}: {
  transcript: Transcript;
  options: PdfOptions;
  onOptions: (options: PdfOptions) => void;
  onPrint: () => void;
  onCancel: () => void;
}) {
  const hints = pdfHints(transcript);
  return (
    <div
      role="dialog"
      aria-label="Download as PDF"
      onKeyDown={(e: KeyboardEvent<HTMLDivElement>) => sheetKeyDown(e, { onClose: onCancel })}
      className="absolute top-full right-0 z-20 mt-1 flex w-72 flex-col gap-3 border border-line bg-surface p-3.5 shadow-lg"
    >
      <h2 className="font-display text-[13px] font-semibold text-ink">PDF</h2>
      <fieldset className="flex flex-col gap-2.5">
        <legend className="sr-only">What the PDF carries</legend>
        {OPTIONS.map(([key, label]) => (
          <label key={key} className="flex cursor-pointer items-start gap-2.5">
            <input
              type="checkbox"
              checked={options[key]}
              onChange={(e) => onOptions({ ...options, [key]: e.currentTarget.checked })}
              className="mt-[3px] size-3.5 shrink-0 cursor-pointer accent-ember"
            />
            <span className="flex flex-col gap-0.5">
              <span className="font-display text-[13px] font-medium text-ink">{label}</span>
              <span className="text-xs leading-snug text-ash">{hints[key]}</span>
            </span>
          </label>
        ))}
      </fieldset>
      <p className="text-xs leading-snug text-ash">The header, the lines and the context images are always in it.</p>
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          autoFocus
          onClick={onPrint}
          className="cut bg-ink px-3 py-1.5 font-display text-[13px] font-semibold text-ground [--cut-size:7px] hover:bg-ember"
        >
          Print…
        </button>
        <button type="button" onClick={onCancel} className="px-1.5 font-display text-[13px] font-medium text-ash hover:text-ink">
          Cancel
        </button>
      </div>
      <p className="text-xs leading-snug text-ash">In the browser's print dialog that opens, choose to save as PDF.</p>
    </div>
  );
}

/** A context image as the PDF prints it: the whole picture, or, while it is read from the store, a box the print waits on. */
function PrintPicture({ id, alt }: { id: ImageId; alt: string }) {
  const url = usePictureUrl()(id, "full");
  return url ? <img src={url} alt={alt} className="pdf-picture" /> : <span role="img" aria-label={alt} data-pending="true" className="pdf-picture pdf-picture-pending" />;
}

/** One of the transcript's rows: its time, then the line, a picture or an answer beside the spine. */
function Line({ block }: { block: Extract<PrintBlock, { kind: "utterance" | "image" | "answer" }> }) {
  switch (block.kind) {
    case "utterance":
      return (
        <li className="pdf-row pdf-line">
          <time className="pdf-time">{block.time}</time>
          <p>
            {block.speaker !== null && <b className="pdf-speaker">{block.speaker}</b>}
            {block.speaker !== null ? ` ${block.text}` : block.text}
          </p>
        </li>
      );
    case "image":
      return (
        <li className="pdf-row pdf-image">
          <time className="pdf-time">{block.time}</time>
          <figure>
            <PrintPicture id={block.image} alt={`Context image at ${block.time}`} />
            <figcaption>
              Context image, {block.width}×{block.height}
            </figcaption>
          </figure>
        </li>
      );
    case "answer":
      return (
        <li className="pdf-row pdf-answer">
          <time className="pdf-time">{block.time}</time>
          <section aria-label={`Answer to: ${block.question}`}>
            <h3>{block.question}</h3>
            <Markdown text={block.answer} />
          </section>
        </li>
      );
  }
}

/** The document a PDF prints, drawn as pdf.ts computed it: the header, the context, the notes, then the lines. */
export function PrintedTranscript({ blocks }: { blocks: PrintBlock[] }) {
  const lines = blocks.filter((b) => b.kind === "utterance" || b.kind === "image" || b.kind === "answer");
  return (
    <article className="pdf">
      {blocks.map((block, i) => {
        switch (block.kind) {
          case "header":
            return (
              <header key={i} className="pdf-head">
                <style>{pageRule(block.title)}</style>
                <img src={flame} alt="ignis" className="pdf-mark" />
                <div>
                  <h1 className="pdf-title">{block.title}</h1>
                  <dl className="pdf-facts">
                    {block.facts.map((f) => (
                      <div key={f.label}>
                        <dt>{f.label}</dt>
                        <dd>{f.value}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </header>
            );
          case "context":
            return (
              <section key={i} aria-label="Context" className="pdf-context">
                <h2 className="pdf-heading">Context</h2>
                <dl className="pdf-told">
                  {block.facts.map((f) => (
                    <div key={f.label}>
                      <dt>{f.label}</dt>
                      <dd>{f.value}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            );
          case "notes":
            return (
              <section key={i} aria-label={block.title} className="pdf-notes">
                <h2 className="pdf-heading">{block.title}</h2>
                <Markdown text={block.markdown} />
              </section>
            );
          case "heading":
            return (
              <h2 key={i} className="pdf-heading">
                {block.text}
              </h2>
            );
          default:
            return null;
        }
      })}
      {lines.length > 0 && (
        <ol aria-label="Utterances" className="pdf-lines">
          {lines.map((block) => (
            <Line key={`${block.kind}-${block.id}`} block={block} />
          ))}
        </ol>
      )}
    </article>
  );
}

/** The class the page's root takes while a PDF's sheet stands: the print stylesheet prints the sheet alone. */
const PRINTING = "printing-transcript";

/** The faces the sheet prints in, loaded before the print: the page may not have used all of them yet. */
const PRINT_FONTS = ['600 20px "Chakra Petch"', '500 12px "Chakra Petch"', '400 12px "Instrument Sans Variable"', '400 12px "JetBrains Mono Variable"'];

/** How long the print waits on a picture the store cannot read: then it prints without it. */
const PICTURES_WAIT_MS = 5_000;

/**
 * A PDF of `transcript` with `options`, as it stood at the press: the sheet, drawn on the page's body, opens the
 * browser's print dialog once its pictures and fonts are in. While it stands the page's title is the file name the
 * downloads take, which the dialog offers to save as; `onDone` once the print dialog closes.
 */
export function PdfPrint({ transcript, options, onDone }: { transcript: Transcript; options: PdfOptions; onDone: () => void }) {
  const [blocks] = useState(() => printDocument(transcript, options));
  const [name] = useState(() => fileStem(titleOf(transcript)));
  const sheet = useRef<HTMLDivElement>(null);
  const done = useRef(onDone);
  done.current = onDone;
  const [, landed] = useReducer((n: number) => n + 1, 0);
  const [waited, setWaited] = useState(false);
  const printed = useRef(false);

  useEffect(() => {
    const page = document.documentElement;
    const title = document.title;
    page.classList.add(PRINTING);
    document.title = name;
    const after = () => done.current();
    window.addEventListener("afterprint", after);
    // A picture's load does not bubble: caught on its way down, it draws the sheet again, which looks again.
    const node = sheet.current;
    node?.addEventListener("load", landed, true);
    node?.addEventListener("error", landed, true);
    const timer = setTimeout(() => setWaited(true), PICTURES_WAIT_MS);
    return () => {
      page.classList.remove(PRINTING);
      document.title = title;
      window.removeEventListener("afterprint", after);
      node?.removeEventListener("load", landed, true);
      node?.removeEventListener("error", landed, true);
      clearTimeout(timer);
    };
  }, [name]);

  useEffect(() => {
    const node = sheet.current;
    if (!node || printed.current) return;
    const reading = node.querySelector("[data-pending]") !== null || [...node.querySelectorAll("img")].some((img) => !img.complete);
    if (reading && !waited) return;
    printed.current = true;
    void Promise.all(PRINT_FONTS.map((font) => document.fonts.load(font)))
      .catch(() => [])
      .then(() => window.print());
  });

  return createPortal(
    <div ref={sheet} className="print-sheet">
      <PrintedTranscript blocks={blocks} />
    </div>,
    document.body,
  );
}
