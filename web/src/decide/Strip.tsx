import { useMemo } from "react";
import { positionOf, STRIP_WIDTH, stripOf } from "./strip.ts";

// The strip (`strip.ts`) drawn: one bar per column, as tall as its lines are
// long, anchored at the foot like a seismograph trace. Tinted, it marks the
// columns holding an error line in ember and a warning in amber — the file
// card's reading of a log. Untinted, it is the ground a `locate`'s answer
// marks its pick and its candidates on, so the only colour on it is theirs.

export function Strip({
  segments,
  width = STRIP_WIDTH,
  tinted = false,
  pick,
  candidates = [],
  label,
  className = "h-9",
}: {
  segments: readonly string[];
  /** The most columns to draw. */
  width?: number;
  /** Colour the columns by the loudest level in them. */
  tinted?: boolean;
  /** The segment the answer named. */
  pick?: number | null;
  /** Other segments the answer weighed. */
  candidates?: number[];
  /** What the strip says, for a reader who cannot see it. */
  label: string;
  className?: string;
}) {
  const strip = useMemo(() => stripOf(segments, width), [segments, width]);
  const count = strip.columns.length;
  // A gap that reads as a gap at both extremes: a hairline between 120 thin
  // bars, and not a gulf between a dozen wide ones.
  const gap = count > 60 ? 0.3 : 0.12;
  return (
    <div role="img" aria-label={label} className={`relative ${className}`}>
      <svg viewBox={`0 0 ${count} 100`} preserveAspectRatio="none" className="block size-full" aria-hidden>
        {strip.columns.map((column, c) => {
          const height = column.weight === 0 ? 5 : 16 + 84 * column.weight;
          const tone = tinted && column.level === "error" ? "fill-ember" : tinted && column.level === "warn" ? "fill-warn" : "fill-ash/30";
          return <rect key={c} x={c + gap / 2} y={100 - height} width={1 - gap} height={height} className={tone} />;
        })}
      </svg>
      {candidates.map((segment) => (
        <span
          key={segment}
          className="absolute inset-y-0 w-px bg-ink/45"
          style={{ left: `${positionOf(segment, segments.length) * 100}%` }}
          aria-hidden
        />
      ))}
      {pick !== undefined && pick !== null && (
        <span
          className="absolute -inset-y-1 w-[2px] -translate-x-1/2 bg-ember shadow-[0_0_0_2px_var(--ground)]"
          style={{ left: `${positionOf(pick, segments.length) * 100}%` }}
          aria-hidden
        />
      )}
    </div>
  );
}
