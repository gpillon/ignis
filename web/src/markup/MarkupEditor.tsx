import { type ReactNode, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Switch } from "../ui/Switch.tsx";
import { actionShape } from "../ui/classes.ts";
import {
  drawMarks,
  drawRedactions,
  extendStroke,
  isArea,
  isMark,
  type Mark,
  MARK_COLORS,
  type MarkedPicture,
  type MarkMode,
  type MarkTool,
  type Point,
  rectOf,
  type RedactedPicture,
  type Redaction,
  type RedactTool,
  renderMarks,
  renderRedactions,
  toImagePoint,
} from "./markup.ts";

// The markup editor (markup.ts says what the marks are for): a picture
// full-screen on the kiln, a circle and a brush in a few colours, undo and
// clear, how the marks travel — drawn on the picture, or (where the caller
// offers it) as a mask image of their own — and whether the result keeps to
// the marked area. The image view and the chat both open it. The Transcribe
// view opens it with its other two tools, a crop and a black-out, for a
// context image (spec #428): `RedactEditor`. It is a portal on the body:
// every surface of the Playground is a `.cut`, whose clip-path would cut an
// overlay away.

/** The picture to mark: shown clean, with the marks it already has. */
export type MarkupSource = { url: string; marks?: Mark[]; mode?: MarkMode; keepOutside?: boolean };

const chip = (on: boolean) =>
  `px-3 py-1.5 font-display text-[13px] font-medium ${on ? "bg-[#eae8e4] text-[#1c2026]" : "text-[#b9bec4] hover:bg-[#2e343c] hover:text-white"}`;

/** The picture's own size, and how many of its pixels one screen pixel covers. */
type Picture = { natural: { width: number; height: number }; perScreenPixel: number };

/** Where the pointer is on the picture, in its own pixels. */
type FramePointer = {
  down: (p: Point, picture: Picture) => void;
  move: (p: Point, picture: Picture) => void;
  up: (p: Point, picture: Picture) => void;
  cancel: () => void;
};

/**
 * Both tool sets' frame: the picture full-screen on the kiln with a canvas over it at the screen's own pixel
 * density, the tools above and the way out below. Escape leaves without saving; Ctrl+Z takes the last change back.
 */
function EditorFrame({
  url,
  title,
  onClose,
  onUndo,
  overlay,
  pointer,
  tools,
  footer,
}: {
  url: string;
  title: string;
  onClose: () => void;
  onUndo: () => void;
  /** Draws over the picture, `scale` being canvas pixels per picture pixel, `natural` the picture's own size. */
  overlay: (ctx: CanvasRenderingContext2D, scale: number, natural: Picture["natural"]) => void;
  pointer: FramePointer;
  tools: ReactNode;
  footer: ReactNode;
}) {
  const image = useRef<HTMLImageElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState({ width: 0, height: 0 });

  useEffect(() => dialog.current?.focus(), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        onUndo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onUndo]);

  // The canvas follows the picture's shown size, at the screen's own pixel density.
  useEffect(() => {
    const img = image.current;
    if (!img) return;
    const observer = new ResizeObserver(() => setShown({ width: img.clientWidth, height: img.clientHeight }));
    observer.observe(img);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    const img = image.current;
    if (!c || !img || shown.width === 0 || img.naturalWidth === 0) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(shown.width * dpr);
    c.height = Math.round(shown.height * dpr);
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
    overlay(ctx, (shown.width / img.naturalWidth) * dpr, { width: img.naturalWidth, height: img.naturalHeight });
  }, [overlay, shown]);

  const natural = () => ({ width: image.current?.naturalWidth ?? 1, height: image.current?.naturalHeight ?? 1 });
  /** Screen pixels as picture pixels: a brush keeps its look whatever the picture's size. */
  const picture = (): Picture => ({ natural: natural(), perScreenPixel: natural().width / Math.max(1, shown.width) });

  function point(e: React.PointerEvent<HTMLCanvasElement>) {
    return toImagePoint(e.clientX, e.clientY, e.currentTarget.getBoundingClientRect(), natural());
  }

  return createPortal(
    <div
      ref={dialog}
      role="dialog"
      aria-modal="true"
      aria-label={title}
      tabIndex={-1}
      className="monitor-kiln fixed inset-0 z-50 flex flex-col text-[#eae8e4] outline-none"
    >
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3 border-b border-[#2e343c] px-4 py-3 md:px-6">
        <h2 className="font-display text-lg font-semibold">{title}</h2>
        {tools}
      </div>

      <div className="flex min-h-0 flex-1 items-center justify-center p-4">
        <div className="relative inline-block">
          <img
            ref={image}
            src={url}
            alt={title}
            className="stage-checker block max-h-[calc(100dvh-11rem)] max-w-[calc(100vw-2rem)] select-none"
            draggable={false}
            onLoad={(e) => setShown({ width: e.currentTarget.clientWidth, height: e.currentTarget.clientHeight })}
          />
          <canvas
            ref={canvas}
            aria-label="Marks"
            className="absolute inset-0 size-full cursor-crosshair touch-none"
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              pointer.down(point(e), picture());
            }}
            onPointerMove={(e) => pointer.move(point(e), picture())}
            onPointerUp={(e) => pointer.up(point(e), picture())}
            onPointerCancel={pointer.cancel}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-[#2e343c] px-4 py-3 md:px-6">{footer}</div>
    </div>,
    document.body,
  );
}

export function MarkupEditor({
  source,
  title,
  modes,
  maskLabel = "As a mask",
  maskRoom = true,
  onSave,
  onClose,
}: {
  source: MarkupSource;
  /** `Mark #3`, `Mark the picture`. */
  title: string;
  /** The ways the caller can send the marks; one means no choice is shown. */
  modes: MarkMode[];
  /** What the mask will be called in the prompt, `As a mask, image 2`. */
  maskLabel?: string;
  /** The request has room for one more picture, the mask. */
  maskRoom?: boolean;
  /** The marks saved; null takes them off, the picture clean again. `color` is the last mark's, for a prompt stub. */
  onSave: (marked: MarkedPicture | null, color: string) => void;
  onClose: () => void;
}) {
  const had = source.marks !== undefined;
  const [marks, setMarks] = useState<Mark[]>(source.marks ?? []);
  const [drafting, setDrafting] = useState<Mark | null>(null);
  const [tool, setTool] = useState<MarkTool>("ellipse");
  const [color, setColor] = useState<string>(MARK_COLORS[0].value);
  const [width, setWidth] = useState(8);
  const [mode, setMode] = useState<MarkMode>(source.mode && modes.includes(source.mode) ? source.mode : modes[0]);
  const [keep, setKeep] = useState(source.keepOutside ?? true);
  const [saving, setSaving] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const keepLabel = useId();

  async function save() {
    setSaving(true);
    setFailure(null);
    try {
      onSave(marks.length === 0 ? null : await renderMarks(source.url, marks, mode, keep), color);
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err));
      setSaving(false);
    }
  }

  return (
    <EditorFrame
      url={source.url}
      title={title}
      onClose={onClose}
      onUndo={() => setMarks((current) => current.slice(0, -1))}
      overlay={(ctx, scale) => drawMarks(ctx, drafting ? [...marks, drafting] : marks, scale)}
      pointer={{
        down: (p, { perScreenPixel }) => {
          const w = width * perScreenPixel;
          setDrafting(tool === "stroke" ? { kind: "stroke", color, width: w, points: [p] } : { kind: "ellipse", color, width: w, from: p, to: p });
        },
        move: (p, { perScreenPixel }) => {
          if (!drafting) return;
          setDrafting(drafting.kind === "stroke" ? { ...drafting, points: extendStroke(drafting.points, p, 1.5 * perScreenPixel) } : { ...drafting, to: p });
        },
        up: () => {
          if (drafting && isMark(drafting)) setMarks((m) => [...m, drafting]);
          setDrafting(null);
        },
        cancel: () => setDrafting(null),
      }}
      tools={
        <>
          <div role="radiogroup" aria-label="Tool" className="flex border border-[#2e343c]">
            {(["ellipse", "stroke"] as const).map((t) => (
              <button key={t} type="button" role="radio" aria-checked={tool === t} className={chip(tool === t)} onClick={() => setTool(t)}>
                {t === "ellipse" ? "Circle" : "Brush"}
              </button>
            ))}
          </div>
          <div role="radiogroup" aria-label="Colour" className="flex items-center gap-1.5">
            {MARK_COLORS.map((c) => (
              <button
                key={c.value}
                type="button"
                role="radio"
                aria-checked={color === c.value}
                aria-label={c.name}
                className={`size-6 border-2 ${color === c.value ? "border-[#eae8e4]" : "border-[#3e454f]"}`}
                style={{ background: c.value }}
                onClick={() => setColor(c.value)}
              />
            ))}
          </div>
          <label className="flex items-center gap-2 text-[13px] text-[#b9bec4]">
            Width
            <input
              type="range"
              min={2}
              max={64}
              value={width}
              onChange={(e) => setWidth(Number(e.target.value))}
              className="range w-28"
              style={{ "--fill": `${((width - 2) / 62) * 100}%` } as React.CSSProperties}
            />
          </label>
          <div className="ml-auto flex items-center">
            <button type="button" className={chip(false)} disabled={marks.length === 0} onClick={() => setMarks((m) => m.slice(0, -1))}>
              Undo
            </button>
            <button type="button" className={chip(false)} disabled={marks.length === 0} onClick={() => setMarks([])}>
              Clear
            </button>
          </div>
        </>
      }
      footer={
        <>
          {modes.length > 1 && (
            <div role="radiogroup" aria-label="Send the marks" className="flex border border-[#2e343c]">
              <button type="button" role="radio" aria-checked={mode === "drawn"} className={chip(mode === "drawn")} onClick={() => setMode("drawn")}>
                Drawn on the picture
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={mode === "mask"}
                disabled={!maskRoom && mode !== "mask"}
                className={`${chip(mode === "mask")} disabled:cursor-not-allowed disabled:opacity-40`}
                onClick={() => setMode("mask")}
              >
                {maskLabel}
              </button>
            </div>
          )}
          <div className="flex items-center gap-3">
            <Switch on={keep} onChange={setKeep} labelledBy={keepLabel} />
            <span id={keepLabel} className="text-sm">
              Change only the marked area
            </span>
          </div>
          <p className="min-w-0 flex-1 basis-64 text-[13px] text-[#939ba4]">
            {mode === "drawn" ? "The model sees your marks on the picture: say in the prompt what they mean." : "The picture goes clean, followed by the marked area in white on black."}
            {keep ? " Outside the marks, the result keeps this picture's own pixels." : ""}
          </p>
          {failure && <p className="text-sm text-[#ff7b6b]">{failure}</p>}
          <div className="ml-auto flex items-center gap-3">
            <button type="button" className={`${actionShape} border border-[#3e454f] px-4 py-2 text-[#eae8e4] hover:border-[#ff5a1f]`} onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              disabled={saving || (marks.length === 0 && !had)}
              className={`${actionShape} bg-[#eae8e4] px-5 py-2 text-[#1c2026] hover:bg-[#ff5a1f]`}
              onClick={() => void save()}
            >
              {saving ? "Saving…" : marks.length === 0 && had ? "Remove marks" : "Save marks"}
            </button>
          </div>
        </>
      }
    />
  );
}

/**
 * The editor's other two tools (spec #428): **Crop** keeps the area dragged, the last one drawn; **Black out** paints
 * the area dragged solid black. Undo and Clear take them back while editing; saving makes the picture they show
 * (`renderRedactions`), and only that one is handed on: the original, and what is under a black-out, never are.
 */
export function RedactEditor({
  url,
  title,
  saveLabel,
  onSave,
  onClose,
}: {
  /** The picture as it came, shown clean. */
  url: string;
  title: string;
  /** `Add image`. */
  saveLabel: string;
  onSave: (edited: RedactedPicture) => void | Promise<void>;
  onClose: () => void;
}) {
  const [edits, setEdits] = useState<Redaction[]>([]);
  const [drafting, setDrafting] = useState<{ kind: RedactTool; from: Point; to: Point } | null>(null);
  const [tool, setTool] = useState<RedactTool>("crop");
  const [saving, setSaving] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  async function save() {
    setSaving(true);
    setFailure(null);
    try {
      await onSave(await renderRedactions(url, edits));
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err));
      setSaving(false);
    }
  }

  const drafted = (d: NonNullable<typeof drafting>, natural: Picture["natural"]): Redaction => ({ kind: d.kind, rect: rectOf(d.from, d.to, natural) });
  const cropped = edits.some((e) => e.kind === "crop");
  const blacked = edits.filter((e) => e.kind === "blackout").length;

  return (
    <EditorFrame
      url={url}
      title={title}
      onClose={onClose}
      onUndo={() => setEdits((current) => current.slice(0, -1))}
      overlay={(ctx, scale, natural) => drawRedactions(ctx, drafting ? [...edits, drafted(drafting, natural)] : edits, natural, scale)}
      pointer={{
        down: (p) => setDrafting({ kind: tool, from: p, to: p }),
        move: (p) => drafting && setDrafting({ ...drafting, to: p }),
        up: (p, { natural }) => {
          if (drafting) {
            const edit = drafted({ ...drafting, to: p }, natural);
            if (isArea(edit.rect)) setEdits((current) => [...current, edit]);
          }
          setDrafting(null);
        },
        cancel: () => setDrafting(null),
      }}
      tools={
        <>
          <div role="radiogroup" aria-label="Tool" className="flex border border-[#2e343c]">
            {(["crop", "blackout"] as const).map((t) => (
              <button key={t} type="button" role="radio" aria-checked={tool === t} className={chip(tool === t)} onClick={() => setTool(t)}>
                {t === "crop" ? "Crop" : "Black out"}
              </button>
            ))}
          </div>
          <p className="text-[13px] text-[#939ba4]">{tool === "crop" ? "Drag the area to keep." : "Drag over what must not be kept."}</p>
          <div className="ml-auto flex items-center">
            <button type="button" className={chip(false)} disabled={edits.length === 0} onClick={() => setEdits((current) => current.slice(0, -1))}>
              Undo
            </button>
            <button type="button" className={chip(false)} disabled={edits.length === 0} onClick={() => setEdits([])}>
              Clear
            </button>
          </div>
        </>
      }
      footer={
        <>
          <p className="min-w-0 flex-1 basis-64 text-[13px] text-[#939ba4]">
            {cropped ? "Cropped" : "The whole picture"}
            {blacked > 0 ? `, ${blacked === 1 ? "1 area" : `${blacked} areas`} blacked out` : ""}. Only the picture as shown is kept: a black-out is painted into its
            pixels, and the original is never stored.
          </p>
          {failure && <p className="text-sm text-[#ff7b6b]">{failure}</p>}
          <div className="ml-auto flex items-center gap-3">
            <button type="button" className={`${actionShape} border border-[#3e454f] px-4 py-2 text-[#eae8e4] hover:border-[#ff5a1f]`} onClick={onClose}>
              Cancel
            </button>
            <button type="button" disabled={saving} className={`${actionShape} bg-[#eae8e4] px-5 py-2 text-[#1c2026] hover:bg-[#ff5a1f]`} onClick={() => void save()}>
              {saving ? "Saving…" : saveLabel}
            </button>
          </div>
        </>
      }
    />
  );
}
