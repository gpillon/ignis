import { useRef, useState } from "react";
import { imageFromFile, type PromptImage } from "../conversation/images.ts";
import { caption, field } from "../ui/classes.ts";
import { IconClose, IconImage } from "../ui/icons.tsx";
import { Lightbox } from "./marks.tsx";
import { type Evidence, type EvidenceMode, restore, setAside, type Spare } from "./model.ts";

// The evidence (GitHub #247): what every question in the request is asked
// about, and what makes a fan-out cheap — it is prefilled once and the
// questions share it.
//
// Three shapes, all three of which `state` accepts: prose, a JSON value that
// goes in as itself, and content parts so the evidence can be an image.
// Switching mode keeps what the other modes held, so a wrong click costs
// nothing.

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
  invalid,
}: {
  evidence: Evidence;
  /** What the modes not in use last held; the caller owns it, and clears it with the draft. */
  spare: Spare;
  onChange: (evidence: Evidence) => void;
  onSpare: (spare: Spare) => void;
  /** The parse error of a JSON evidence, when there is one. */
  invalid?: string;
}) {
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

      {evidence.mode === "text" && (
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

      {evidence.mode === "json" && (
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

      <p className="mt-1.5 max-w-[68ch] text-[12px] leading-snug text-ash">{hint}</p>
    </section>
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
  const [over, setOver] = useState(false);
  // The thumbnail is a thumbnail; the evidence itself is worth a proper look.
  const [zoom, setZoom] = useState<PromptImage | null>(null);
  return (
    <div className="mt-2">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          onAdd([...e.dataTransfer.files]);
        }}
        className={`flex flex-wrap items-center gap-2 border border-dashed p-2 ${over ? "border-ember bg-ember/5" : "border-line"}`}
      >
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
