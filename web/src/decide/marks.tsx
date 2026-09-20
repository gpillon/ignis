import { type CSSProperties, type ReactNode, useEffect, useState } from "react";
import type { PromptImage } from "../conversation/images.ts";
import { IconClose, IconExpand } from "../ui/icons.tsx";
import type { DigitDraw } from "./request.ts";

// The marks the answers are drawn with (GitHub #247), in the palette's tokens
// and in the Monitor's idiom: plain SVG or a plain div, thin, and never the
// only place a number appears.
//
// Every distribution is **one hue**. Bar length already encodes the
// probability, so shading a bar by the same number would spend the only free
// channel on information the mark already carries; the winner is marked by its
// label, not by a second colour.

/**
 * The value a mark grows to, carried as a custom property so it is on the
 * element from the first paint — the animation is then "0% until grown", and
 * what the mark *means* is readable in the DOM whether it has grown or not.
 */
const fill = (share: number, extra: CSSProperties = {}): CSSProperties =>
  ({ "--fill": `${Math.max(0, Math.min(1, share)) * 100}%`, ...extra }) as CSSProperties;

/** True one frame after mount, which is what lets a width transition run. */
function useGrown(): boolean {
  const [grown, setGrown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setGrown(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return grown;
}

/** One probability, as a track and a fill. The number itself is the caller's to print. */
export function Bar({ value, dim = false }: { value: number; dim?: boolean }) {
  const grown = useGrown();
  return (
    <span className="relative block h-2 w-full bg-line/70" aria-hidden>
      <span
        className="bar-grow bar-end absolute inset-y-0 left-0 bg-ember"
        style={fill(value, { width: grown ? "var(--fill)" : "0%", opacity: dim ? 0.55 : 1 })}
      />
    </span>
  );
}

/** A `noul`: the probability of yes, as the figure and the bar under it. */
export function NoulMark({ value, yes, no }: { value: number; yes: string; no: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-4">
        <p className="font-display text-[34px] font-semibold leading-none tabular-nums text-ink">{value.toFixed(3)}</p>
        <p className="text-right text-[13px] leading-snug text-ash">{value >= 0.5 ? yes : no}</p>
      </div>
      <div className="mt-2">
        <Bar value={value} />
      </div>
      <div className="mt-1.5 flex justify-between font-display text-[11px] text-ash">
        <span>0 · {no}</span>
        <span>{yes} · 1</span>
      </div>
    </div>
  );
}

/** One row of a distribution: the option, its bar, its probability. */
export function DistributionRow({
  label,
  detail,
  value,
  winner,
}: {
  label: string;
  detail?: string;
  value: number;
  winner: boolean;
}) {
  return (
    <li className="grid grid-cols-[minmax(0,7rem)_1fr_3.2rem] items-center gap-x-3 gap-y-1 py-1">
      <span className="min-w-0">
        <span className={`block truncate text-[13px] ${winner ? "font-semibold text-ink" : "text-ash"}`} title={detail ?? label}>
          {winner && <span className="mr-1.5 inline-block size-1.5 bg-ember align-middle" aria-hidden />}
          {label}
        </span>
      </span>
      <Bar value={value} dim={!winner} />
      <span className="text-right font-display text-[13px] tabular-nums text-ink">{value.toFixed(3)}</span>
    </li>
  );
}

/**
 * A `score`: the distribution over the levels, and the expected value riding
 * the axis between them.
 *
 * The pointer is the one second mark in the whole panel, and it earns itself:
 * the score is the weighted average of the level indices, so it genuinely
 * lands between the levels the bars sit on — `1.6` for
 * `{0: 0.05, 1: 0.3, 2: 0.65}` is not any one level's answer.
 */
export function ScoreMark({
  score,
  levels,
  probabilities,
}: {
  score: number;
  levels: { index: number; label: string; probability: number }[];
  /** The bar heights, already in level order. */
  probabilities: number[];
}) {
  const grown = useGrown();
  const last = Math.max(1, levels.length - 1);
  const pointer = `${(Math.max(0, Math.min(last, score)) / last) * 100}%`;
  const peak = Math.max(0.0001, ...probabilities);
  return (
    <div>
      <p className="font-display text-[34px] font-semibold leading-none tabular-nums text-ink">{score.toFixed(2)}</p>
      <div className="mt-3 flex h-16 items-end gap-1" aria-hidden>
        {levels.map((level) => (
          <span key={level.index} className="flex min-w-0 flex-1 flex-col justify-end">
            {/* On the probability's own 0-1 scale, not normalized to the
                peak: a choice's bars beside it read the same way, and a flat
                distribution should look flat. */}
            <span className="bar-grow w-full bg-ember" style={fill(level.probability, { height: grown ? "var(--fill)" : "0%", opacity: 0.85 })} />
          </span>
        ))}
      </div>
      {/* The axis: a hairline with a tick per level and the pointer on it. */}
      <div className="relative mt-1.5 h-4">
        <span className="absolute inset-x-0 top-0 h-px bg-line" aria-hidden />
        {levels.map((level) => (
          <span
            key={level.index}
            className="absolute top-0 h-1.5 w-px bg-line"
            style={{ left: `${(level.index / last) * 100}%` }}
            aria-hidden
          />
        ))}
        {/* The answer itself, so it is in place from the first paint; the bars
            behind it are what grow. */}
        <span
          className="absolute top-0 -ml-[5px] size-0 border-x-[5px] border-t-[7px] border-x-transparent border-t-ember"
          style={{ left: pointer }}
          aria-hidden
        />
      </div>
      <ol className="mt-1 flex gap-1 text-[11px] leading-tight text-ash">
        {levels.map((level) => (
          <li key={level.index} className="min-w-0 flex-1">
            <span className={`block truncate ${level.probability === peak ? "text-ink" : ""}`} title={level.label}>
              {level.label}
            </span>
            <span className="block font-display tabular-nums">{level.probability.toFixed(3)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/**
 * The digit trace of a constrained decode: one column per place, the digit it
 * committed and how sure it was.
 *
 * The places land left to right, one per 40 ms, because that is how they were
 * generated — a forced token per step. The measured trace falls across a
 * number (hundreds near 0.99, units as low as 0.15), which is the model
 * reporting its own resolution, so the columns are drawn to make that shape
 * visible rather than averaged into one figure.
 */
export function DigitTrace({ digits, offset = 0 }: { digits: DigitDraw[]; offset?: number }) {
  return (
    <ol className="flex items-end gap-1" aria-hidden>
      {digits.map((draw, index) => (
        <li key={index} className="digit-land flex w-6 flex-col items-center gap-1" style={{ animationDelay: `${(offset + index) * 40}ms` }}>
          <span className="font-mono text-[15px] leading-none tabular-nums text-ink">{draw.digit}</span>
          <span className="relative block h-6 w-full bg-line/70">
            <span className="absolute inset-x-0 bottom-0 bg-ember" style={{ height: `${draw.probability * 100}%` }} />
          </span>
          <span className="font-display text-[10px] leading-none tabular-nums text-ash">{draw.probability.toFixed(2)}</span>
        </li>
      ))}
    </ol>
  );
}

/**
 * A point or a box on the image the request submitted.
 *
 * The SVG shares the image's own coordinate system, so the answer's pixels are
 * drawn as pixels whatever size the image is displayed at — which is how the
 * same drawing serves both the preview and the full-size view.
 *
 * A preview, because an answer panel is a column of answers and one picture
 * should not own the column. Clicking it opens the picture at its own size,
 * where a crosshair a few pixels wide is actually readable.
 *
 * The halo is the model's declared uncertainty and is labelled as that, never
 * as a bound: on the measured sample it covers the true error on four axes of
 * six.
 */
export function ImageMark({
  image,
  children,
  caption,
}: {
  image: PromptImage;
  /** Drawn in the image's pixel coordinates; `natural` is handed down through the viewBox. */
  children: (natural: { width: number; height: number }) => ReactNode;
  caption: ReactNode;
}) {
  const [natural, setNatural] = useState<{ width: number; height: number } | null>(
    image.width > 0 && image.height > 0 ? { width: image.width, height: image.height } : null,
  );
  const [open, setOpen] = useState(false);

  // One plate, drawn twice: the preview and the full-size view differ only in
  // how tall the image is allowed to be, so the marks cannot drift apart.
  const plate = (limit: string) => (
    <span className="relative inline-block align-top bg-ground">
      <img
        src={image.url}
        alt=""
        className={`block w-auto max-w-full ${limit}`}
        onLoad={(e) => setNatural({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
      />
      {natural && (
        <svg viewBox={`0 0 ${natural.width} ${natural.height}`} preserveAspectRatio="none" className="absolute inset-0 size-full" aria-hidden>
          {children(natural)}
        </svg>
      )}
    </span>
  );

  return (
    <figure className="m-0">
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="Open the picture at its own size"
        className="group relative block max-w-full cursor-zoom-in"
      >
        {plate("max-h-56")}
        <span className="absolute bottom-1 right-1 flex items-center gap-1 bg-kiln/85 px-1.5 py-0.5 font-display text-[11px] text-[#eae8e4] opacity-70 transition-opacity group-hover:opacity-100">
          <IconExpand />
          Full size
        </span>
      </button>
      <figcaption className="mt-2 text-[12px] leading-snug text-ash">{caption}</figcaption>
      {open && <Lightbox onClose={() => setOpen(false)}>{plate("max-h-[82vh]")}</Lightbox>}
    </figure>
  );
}

/**
 * A picture at its own size, over the page. Escape and a click outside close
 * it; nothing under it moves, so going back is exactly where you were.
 */
export function Lightbox({ onClose, children }: { onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const key = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-[#1c2026]/88 p-4" onClick={onClose} role="presentation">
      <div className="relative max-h-full max-w-full" onClick={(e) => e.stopPropagation()}>
        {children}
        <button
          type="button"
          onClick={onClose}
          aria-label="Close the picture"
          className="absolute right-0 top-0 grid size-8 place-items-center bg-kiln text-[#eae8e4] hover:bg-ember"
        >
          <IconClose />
        </button>
      </div>
    </div>
  );
}

/** The crosshair and the uncertainty halo of a `point`, in image pixels. */
export function PointGlyph({
  x,
  y,
  sigmaX,
  sigmaY,
  scale,
}: {
  x: number;
  y: number;
  sigmaX: number;
  sigmaY: number;
  /** Pixels per stroke unit, so a hairline stays a hairline on a large image. */
  scale: number;
}) {
  const arm = 14 * scale;
  return (
    <>
      <ellipse cx={x} cy={y} rx={Math.max(sigmaX, 1)} ry={Math.max(sigmaY, 1)} fill="var(--ember)" opacity="0.18" />
      <line x1={x - arm} y1={y} x2={x + arm} y2={y} stroke="var(--ember)" strokeWidth={2 * scale} />
      <line x1={x} y1={y - arm} x2={x} y2={y + arm} stroke="var(--ember)" strokeWidth={2 * scale} />
      <circle cx={x} cy={y} r={3 * scale} fill="var(--ember)" />
    </>
  );
}

/** A `box`, with the band its per-edge uncertainty spans. */
export function BoxGlyph({
  pixels,
  sigma,
  scale,
}: {
  pixels: Record<string, number>;
  sigma: Record<string, number>;
  scale: number;
}) {
  const x0 = pixels.x0 ?? 0;
  const y0 = pixels.y0 ?? 0;
  const x1 = pixels.x1 ?? 0;
  const y1 = pixels.y1 ?? 0;
  const outer = {
    x: x0 - (sigma.x0 ?? 0),
    y: y0 - (sigma.y0 ?? 0),
    width: x1 - x0 + (sigma.x0 ?? 0) + (sigma.x1 ?? 0),
    height: y1 - y0 + (sigma.y0 ?? 0) + (sigma.y1 ?? 0),
  };
  return (
    <>
      <rect {...outer} fill="var(--ember)" opacity="0.14" />
      <rect x={x0} y={y0} width={Math.max(0, x1 - x0)} height={Math.max(0, y1 - y0)} fill="none" stroke="var(--ember)" strokeWidth={2 * scale} />
    </>
  );
}
