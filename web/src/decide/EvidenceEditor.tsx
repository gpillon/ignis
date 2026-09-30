import { useMemo, useRef, useState } from "react";
import { imageFromFile, type PromptImage } from "../conversation/images.ts";
import { caption, field } from "../ui/classes.ts";
import { IconClose, IconImage, IconPaperclip } from "../ui/icons.tsx";
import { type FileSource, formatBytes } from "./evidenceFile.ts";
import { Lightbox } from "./marks.tsx";
import { type Evidence, type EvidenceMode, locateTarget, restore, setAside, type Spare } from "./model.ts";
import { stripOf, STRIP_WIDTH } from "./strip.ts";
import { Strip } from "./Strip.tsx";

// The evidence (GitHub #247): what every question in the request is asked
// about, and what makes a fan-out cheap — it is prefilled once and the
// questions share it.
//
// Three shapes, all three of which `state` accepts: prose, a JSON value that
// goes in as itself, and content parts so the evidence can be an image.
// Switching mode keeps what the other modes held, so a wrong click costs
// nothing.
//
// A text or a JSON evidence can come from a file (`evidenceFile.ts`), and then
// the bench shows the file instead of its text: what it is, how long, where
// its errors are, and how it begins and ends. A textarea holding a
// many-megabyte log would put the whole file between every key and its
// character, and nobody reads a log by scrolling a textarea anyway.

const MODES: { mode: EvidenceMode; label: string; hint: string }[] = [
  { mode: "text", label: "Text", hint: "Prose: a message, a review, a transcript." },
  { mode: "json", label: "JSON", hint: "A record: it goes in as an object or an array, not as a quoted string." },
  {
    mode: "image",
    label: "Image",
    // The server binds a spatial answer to the *first* image (`media.rs`,
    // `source_pixels`), so several images have no single frame between them.
    hint: "A picture, with optional words. A point or a box answers in the pixels of the first one.",
  },
];

export function EvidenceEditor({
  evidence,
  spare,
  onChange,
  onSpare,
  onFiles,
  fileError,
  invalid,
}: {
  evidence: Evidence;
  /** What the modes not in use last held; the caller owns it, and clears it with the draft. */
  spare: Spare;
  onChange: (evidence: Evidence) => void;
  onSpare: (spare: Spare) => void;
  /** Files chosen here; the caller loads them, as it does the ones dropped anywhere on the tab. */
  onFiles: (files: File[]) => void;
  /** Why the last file did not load. */
  fileError: string | null;
  /** The parse error of a JSON evidence, when there is one. */
  invalid?: string;
}) {
  const picker = useRef<HTMLInputElement>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const hint = MODES.find((m) => m.mode === evidence.mode)?.hint;

  function switchTo(mode: EvidenceMode) {
    if (mode === evidence.mode) return;
    const kept = setAside(spare, evidence);
    onSpare(kept);
    onChange(restore(kept, mode));
  }

  async function addFiles(files: File[]) {
    if (evidence.mode !== "image") return;
    const results = await Promise.all(files.map(imageFromFile));
    const errors = results.flatMap((r) => (r.ok ? [] : [r.error]));
    setImageError(errors.length ? errors.join(" ") : null);
    const added = results.flatMap((r) => (r.ok ? [r.image] : []));
    if (added.length) onChange({ ...evidence, images: [...evidence.images, ...added] });
  }

  return (
    <section>
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-display text-[15px] font-semibold text-ink">Evidence</h2>
        <div role="group" aria-label="Evidence shape" className="flex">
          {MODES.map(({ mode, label }) => (
            <button
              key={mode}
              type="button"
              aria-pressed={evidence.mode === mode}
              onClick={() => switchTo(mode)}
              className={`px-2.5 py-1 font-display text-[12px] font-medium ${
                evidence.mode === mode ? "bg-ink text-ground" : "text-ash hover:bg-line hover:text-ink"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {evidence.mode !== "image" && evidence.file && (
        <FileCard
          evidence={evidence}
          file={evidence.file}
          onEdit={() => onChange({ mode: evidence.mode, text: evidence.text })}
          onRemove={() => onChange({ mode: evidence.mode, text: "" })}
        />
      )}

      {evidence.mode === "text" && !evidence.file && (
        <textarea
          value={evidence.text}
          onChange={(e) => onChange({ mode: "text", text: e.target.value })}
          name="evidence-text"
          aria-label="The evidence"
          placeholder="Paste the message, the review, the transcript…"
          rows={5}
          className={`${field} mt-2 resize-y`}
        />
      )}

      {evidence.mode === "json" && !evidence.file && (
        <>
          <textarea
            value={evidence.text}
            onChange={(e) => onChange({ mode: "json", text: e.target.value })}
            name="evidence-json"
            aria-label="The evidence, as JSON"
            placeholder={'{\n  "order": "A-4471"\n}'}
            rows={8}
            spellCheck={false}
            className={`${field} mt-2 resize-y font-mono text-[13px] ${invalid ? "border-warn" : ""}`}
          />
          {invalid && <p className="mt-1 text-[12px] leading-snug text-warn">{invalid}</p>}
        </>
      )}

      {evidence.mode === "image" && (
        <ImagePicker
          images={evidence.images}
          words={evidence.text}
          error={imageError}
          onWords={(text) => onChange({ ...evidence, text })}
          onAdd={(files) => void addFiles(files)}
          onRemove={(index) => onChange({ ...evidence, images: evidence.images.filter((_, i) => i !== index) })}
        />
      )}

      {fileError && evidence.mode !== "image" && <p className="mt-1 text-[12px] leading-snug text-warn">{fileError}</p>}

      <div className="mt-1.5 flex items-start justify-between gap-3">
        <p className="max-w-[68ch] text-[12px] leading-snug text-ash">{hint}</p>
        {/* The image picker has a button of its own, which takes pictures only. */}
        {evidence.mode !== "image" && (
          <>
            <button
              type="button"
              onClick={() => picker.current?.click()}
              title="A log, a document or a JSON file. You can also drop one anywhere on this tab."
              className="-mr-1 -mt-0.5 flex shrink-0 items-center gap-1 px-1 py-0.5 font-display text-[12px] font-medium text-ember hover:underline [&_svg]:size-[15px]"
            >
              <IconPaperclip />
              {evidence.file ? "Open another" : "Open a file"}
            </button>
            <input
              ref={picker}
              type="file"
              name="evidence-file"
              aria-label="Choose a file as the evidence"
              className="hidden"
              onChange={(e) => {
                onFiles([...(e.target.files ?? [])]);
                e.target.value = "";
              }}
            />
          </>
        )}
      </div>
    </section>
  );
}

/** The most segments of the head and of the tail the card shows. */
const HEAD = 4;
const TAIL = 2;

/**
 * A loaded file, in place of its text: its name, how many segments a `locate`
 * would cut it into, its size, the strip, and how it begins and ends.
 *
 * The count is the root target's — lines of a text, elements of a JSON array
 * — cut the way the server cuts it, so the figure here is the one an answer's
 * "line N of M" will say. A JSON object has no root target, and counts lines.
 * Indices start at 0, as the answers' do.
 */
function FileCard({
  evidence,
  file,
  onEdit,
  onRemove,
}: {
  evidence: Extract<Evidence, { mode: "text" | "json" }>;
  file: FileSource;
  onEdit: () => void;
  onRemove: () => void;
}) {
  const { segments, unit } = useMemo(() => {
    const cut = locateTarget(evidence, "");
    return cut?.ok ? cut.target : { segments: evidence.text.split("\n"), unit: "line" as const };
  }, [evidence]);
  const total = segments.length;
  const head = segments.slice(0, Math.min(HEAD, total));
  const tailFrom = Math.max(head.length, total - TAIL);
  const hidden = tailFrom - head.length;
  return (
    <div className="cut mt-2 bg-surface">
      <div className="flex items-start gap-3 px-3 pt-2.5">
        <div className="min-w-0 flex-1">
          <p className="truncate font-display text-[15px] font-semibold text-ink" title={file.name}>
            {file.name}
          </p>
          <p className="mt-0.5 flex flex-wrap gap-x-3 text-[12px] tabular-nums text-ash">
            <span>
              <span className="text-ink">{total.toLocaleString("en-US")}</span> {plural(total, unit)}
            </span>
            <span>{formatBytes(file.size)}</span>
            <span>read as {evidence.mode === "json" ? "JSON" : "text"}</span>
          </p>
        </div>
        <div className="-mr-1 flex shrink-0 items-center">
          <button
            type="button"
            onClick={onEdit}
            title="Put the text in an editor. The text stays; only the card goes."
            className="px-2 py-1 font-display text-[12px] font-medium text-ash hover:text-ink"
          >
            Edit the text
          </button>
          <button
            type="button"
            onClick={onRemove}
            aria-label={`Remove ${file.name}`}
            className="grid size-7 place-items-center text-ash hover:bg-line hover:text-ink"
          >
            <IconClose />
          </button>
        </div>
      </div>

      <div className="px-3 pb-2.5 pt-3">
        <FileStrip segments={segments} unit={unit} />
      </div>

      <ol className="bg-kiln py-1.5 font-mono text-[11.5px] leading-[1.65] text-[#c9ccd1]" aria-label={`How ${file.name} begins and ends`}>
        {head.map((text, index) => (
          <PreviewLine key={index} index={index} text={text} />
        ))}
        {hidden > 0 && (
          <li className="px-3 font-display text-[11px] text-[#939ba4]">
            {hidden.toLocaleString("en-US")} more {plural(hidden, unit)}
          </li>
        )}
        {segments.slice(tailFrom).map((text, i) => (
          <PreviewLine key={tailFrom + i} index={tailFrom + i} text={text} />
        ))}
      </ol>
    </div>
  );
}

const plural = (count: number, unit: "line" | "item") => (count === 1 ? unit : `${unit}s`);

/** The card's strip, tinted by level, with a legend for its colours when there is anything in them to read. */
function FileStrip({ segments, unit }: { segments: readonly string[]; unit: "line" | "item" }) {
  const { errors, warnings } = stripOf(segments, STRIP_WIDTH);
  const label =
    `${segments.length.toLocaleString("en-US")} ${plural(segments.length, unit)}, each column as tall as its ${unit}s are long` +
    (errors || warnings ? `; ${errors} read as errors and ${warnings} as warnings` : "");
  return (
    <>
      <Strip segments={segments} tinted label={label} className="h-9" />
      {(errors > 0 || warnings > 0) && (
        <p className="mt-1.5 flex flex-wrap gap-x-4 text-[11.5px] tabular-nums text-ash">
          {errors > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="size-2 bg-ember" aria-hidden />
              {errors.toLocaleString("en-US")} {plural(errors, unit)} {errors === 1 ? "reads as an error" : "read as errors"}
            </span>
          )}
          {warnings > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="size-2 bg-warn" aria-hidden />
              {warnings.toLocaleString("en-US")} {warnings === 1 ? "as a warning" : "as warnings"}
            </span>
          )}
        </p>
      )}
    </>
  );
}

function PreviewLine({ index, text }: { index: number; text: string }) {
  return (
    <li className="flex gap-2.5 px-3">
      <span className="w-9 shrink-0 text-right tabular-nums text-[#6b737d]">{index}</span>
      <span className="min-w-0 flex-1 truncate" title={text}>
        {text || " "}
      </span>
    </li>
  );
}

function ImagePicker({
  images,
  words,
  error,
  onWords,
  onAdd,
  onRemove,
}: {
  images: PromptImage[];
  words: string;
  error: string | null;
  onWords: (text: string) => void;
  onAdd: (files: File[]) => void;
  onRemove: (index: number) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  // The thumbnail is a thumbnail; the evidence itself is worth a proper look.
  const [zoom, setZoom] = useState<PromptImage | null>(null);
  return (
    <div className="mt-2">
      {/* A picture dropped here is the tab's to load (`DecideView`), like any other file. */}
      <div className="flex flex-wrap items-center gap-2 border border-dashed border-line p-2">
        {images.map((image, index) => (
          <span key={index} className="relative block">
            <button type="button" onClick={() => setZoom(image)} title={`Open ${image.name} at its own size`} className="block cursor-zoom-in">
              <img src={image.url} alt={image.name} className="cut block h-20 w-auto border border-line [--cut-size:6px]" />
            </button>
            <button
              type="button"
              aria-label={`Remove ${image.name}`}
              onClick={() => onRemove(index)}
              className="absolute right-0 top-0 grid size-5 place-items-center bg-kiln text-[#eae8e4] hover:bg-fault"
            >
              <IconClose />
            </button>
          </span>
        ))}
        <button
          type="button"
          onClick={() => input.current?.click()}
          className="flex items-center gap-1.5 px-2 py-1.5 font-display text-[13px] font-medium text-ember hover:underline"
        >
          <IconImage />
          {images.length ? "Add another" : "Choose an image"}
        </button>
        <input
          ref={input}
          type="file"
          name="evidence-image"
          aria-label="Add an evidence image"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            onAdd([...(e.target.files ?? [])]);
            e.target.value = "";
          }}
        />
      </div>
      {error && <p className="mt-1 text-[12px] leading-snug text-warn">{error}</p>}
      {zoom && (
        <Lightbox onClose={() => setZoom(null)}>
          <img src={zoom.url} alt={zoom.name} className="block max-h-[82vh] w-auto max-w-full" />
        </Lightbox>
      )}
      <label className="mt-2 block">
        <span className={`${caption} block`}>Words with the image, if any</span>
        <input
          value={words}
          onChange={(e) => onWords(e.target.value)}
          name="evidence-words"
          placeholder="A screenshot of the checkout page."
          className={`${field} mt-1`}
        />
      </label>
    </div>
  );
}
