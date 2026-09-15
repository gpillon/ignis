import { type KeyboardEvent, type ReactNode, useEffect, useId, useRef, useState } from "react";
import { bucketOffset, knownRuns, nearestIndex, stackTops, tipLeft } from "./chartMath.ts";
import { formatOffset, niceScale } from "./format.ts";

// The Monitor's charts, drawn as plain SVG in the palette's tokens: a time
// chart (lines or stacked areas, one y-axis, a crosshair that snaps to the
// nearest scrape), a sparkline, and a histogram's distribution. Every hover
// readout is also reachable by keyboard; the numbers themselves are in the
// tiles and the series table, so a tooltip never gates a value. The geometry
// is in chartMath.ts.

export type Series = { key: string; label: string; color: string; values: (number | null)[] };

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

const clock = (t: number) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });

const PAD = { left: 44, right: 12, top: 12, bottom: 22 };
const TIP_WIDTH = 196;

export function TimeChart({
  label,
  times,
  from,
  to,
  series,
  stacked = false,
  area = stacked,
  integer = false,
  height = 180,
  format,
  markers = [],
  live = false,
}: {
  label: string;
  times: number[];
  from: number;
  to: number;
  series: Series[];
  stacked?: boolean;
  /** A wash under each line (always on when stacked). */
  area?: boolean;
  integer?: boolean;
  height?: number;
  format: (v: number) => string;
  /** Moments to mark: server restarts. */
  markers?: number[];
  /** Breathe a ring on the latest point. */
  live?: boolean;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hoverAt, setHoverAt] = useState<number | null>(null);
  const clipId = `tc${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const plotW = Math.max(1, width - PAD.left - PAD.right);
  const plotH = height - PAD.top - PAD.bottom;
  const span = Math.max(1, to - from);

  const tops = stackTops(
    series.map((s) => s.values),
    times.length,
    stacked,
  );
  const visible: number[] = [];
  for (let i = 0; i < times.length; i++) if (times[i] >= from && times[i] <= to) visible.push(i);
  // One scrape before the window keeps the lines running to its left edge.
  const first = Math.max(0, (visible[0] ?? times.length) - 1);
  let max = 0;
  for (const column of tops) for (const i of visible) max = Math.max(max, column[i] ?? 0);
  const scale = niceScale(max, 4, integer);
  const x = (t: number) => PAD.left + ((t - from) / span) * plotW;
  const y = (v: number) => PAD.top + plotH - (Math.min(v, scale.top) / scale.top) * plotH;

  const line = (k: number, run: number[]) => run.map((i, n) => `${n ? "L" : "M"}${x(times[i]).toFixed(1)},${y(tops[k][i]!).toFixed(1)}`).join("");
  const wash = (k: number, run: number[]) => {
    const base = (i: number) => (stacked && k > 0 ? (tops[k - 1][i] ?? 0) : 0);
    const back = [...run].reverse().map((i) => `L${x(times[i]).toFixed(1)},${y(base(i)).toFixed(1)}`).join("");
    return `${line(k, run)}${back}Z`;
  };

  const hover = hoverAt === null ? null : nearestIndex(times, visible, hoverAt);

  const onKey = (e: KeyboardEvent) => {
    if (!visible.length) return;
    const pos = hover === null ? visible.length : visible.indexOf(hover);
    let next: number;
    if (e.key === "ArrowLeft") next = Math.max(0, pos - 1);
    else if (e.key === "ArrowRight") next = Math.min(visible.length - 1, pos + 1);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = visible.length - 1;
    else if (e.key === "Escape") return setHoverAt(null);
    else return;
    e.preventDefault();
    setHoverAt(times[visible[next]]);
  };

  const tickStep = span <= 60_000 ? 15_000 : span <= 300_000 ? 60_000 : 180_000;
  const ticks: number[] = [];
  for (let off = 0; off <= span + 1; off += tickStep) ticks.push(off);
  const grid: number[] = [];
  for (let v = 0; v <= scale.top + scale.step / 1000; v += scale.step) grid.push(v);
  const lastIndex = visible.at(-1);
  const liveSeries = stacked ? series.length - 1 : 0;
  const latest = series
    .map((s) => {
      const v = lastIndex === undefined ? null : s.values[lastIndex];
      return `${s.label} ${v === null || v === undefined ? "unknown" : format(v)}`;
    })
    .join(", ");

  return (
    <div
      ref={ref}
      role="group"
      tabIndex={0}
      aria-label={`${label}. Latest: ${latest}. Arrow keys read earlier scrapes.`}
      className="relative w-full outline-offset-2 select-none"
      style={{ height }}
      onPointerMove={(e) => {
        const px = e.clientX - e.currentTarget.getBoundingClientRect().left;
        setHoverAt(from + ((px - PAD.left) / plotW) * span);
      }}
      onPointerLeave={() => setHoverAt(null)}
      onKeyDown={onKey}
      onBlur={() => setHoverAt(null)}
    >
      {width > 0 && (
        <svg width={width} height={height} className="block" aria-hidden>
          <defs>
            <clipPath id={clipId}>
              <rect x={PAD.left} y={0} width={plotW + 6} height={height} />
            </clipPath>
          </defs>
          {grid.map((v) => (
            <g key={v}>
              <line x1={PAD.left} x2={width - PAD.right} y1={y(v)} y2={y(v)} style={{ stroke: "var(--line)" }} strokeWidth={1} opacity={v === 0 ? 1 : 0.6} />
              <text x={PAD.left - 8} y={y(v)} dy="0.32em" textAnchor="end" className="font-display tabular-nums" fontSize={11} style={{ fill: "var(--ash)" }}>
                {format(v)}
              </text>
            </g>
          ))}
          {ticks.map((off) => {
            const tx = x(to - off);
            const anchor = off === 0 ? "end" : tx - PAD.left < 20 ? "start" : "middle";
            return (
              <text key={off} x={tx} y={height - 6} textAnchor={anchor} className="font-display" fontSize={11} style={{ fill: "var(--ash)" }}>
                {formatOffset(off)}
              </text>
            );
          })}
          <g clipPath={`url(#${clipId})`}>
            {series.map((s, k) =>
              area || stacked
                ? knownRuns(tops[k], first).map((run) => <path key={`${s.key}-a${run[0]}`} d={wash(k, run)} style={{ fill: s.color }} opacity={stacked ? 0.22 : 0.12} />)
                : null,
            )}
            {series.map((s, k) =>
              knownRuns(tops[k], first).map((run) => (
                <path key={`${s.key}-l${run[0]}`} d={line(k, run)} fill="none" style={{ stroke: s.color }} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              )),
            )}
            {markers
              .filter((m) => m >= from && m <= to)
              .map((m) => (
                <g key={m}>
                  <line x1={x(m)} x2={x(m)} y1={PAD.top} y2={PAD.top + plotH} style={{ stroke: "var(--ember)" }} strokeWidth={1} />
                  <text x={x(m) + 4} y={PAD.top + 8} fontSize={10} className="font-display" style={{ fill: "var(--ember)" }}>
                    restart
                  </text>
                </g>
              ))}
          </g>
          {live && lastIndex !== undefined && tops[liveSeries][lastIndex] !== null && hover === null && (
            <g>
              <circle className="live-ring" cx={x(times[lastIndex])} cy={y(tops[liveSeries][lastIndex]!)} r={4} style={{ fill: series[liveSeries].color }} />
              <circle cx={x(times[lastIndex])} cy={y(tops[liveSeries][lastIndex]!)} r={4} strokeWidth={2} style={{ fill: series[liveSeries].color, stroke: "var(--surface)" }} />
            </g>
          )}
          {hover !== null && (
            <g>
              <line x1={x(times[hover])} x2={x(times[hover])} y1={PAD.top} y2={PAD.top + plotH} style={{ stroke: "var(--ash)" }} strokeWidth={1} opacity={0.7} />
              {series.map((s, k) =>
                tops[k][hover] === null ? null : (
                  <circle key={s.key} cx={x(times[hover])} cy={y(tops[k][hover]!)} r={4} strokeWidth={2} style={{ fill: s.color, stroke: "var(--surface)" }} />
                ),
              )}
            </g>
          )}
          {visible.length < 2 && (
            <text x={PAD.left + plotW / 2} y={PAD.top + plotH / 2} textAnchor="middle" fontSize={12} style={{ fill: "var(--ash)" }}>
              Collecting scrapes…
            </text>
          )}
        </svg>
      )}
      {hover !== null && (
        <Tip left={tipLeft(x(times[hover]), width, TIP_WIDTH)}>
          <div className="mb-1.5 flex justify-between font-display text-[11px] text-ash">
            <span className="tabular-nums">{clock(times[hover])}</span>
            <span>{formatOffset(to - times[hover])}</span>
          </div>
          {(stacked ? [...series].reverse() : series).map((s) => (
            <TipRow key={s.key} color={s.color} label={s.label} value={s.values[hover] === null || s.values[hover] === undefined ? "—" : format(s.values[hover]!)} />
          ))}
          {stacked && series.length > 1 && <TipRow label="Total" value={tops[series.length - 1][hover] === null ? "—" : format(tops[series.length - 1][hover]!)} />}
        </Tip>
      )}
    </div>
  );
}

function Tip({ left, children }: { left: number; children: ReactNode }) {
  return (
    <div
      className="pointer-events-none absolute top-1 z-10 border border-line bg-surface/95 px-3 py-2 shadow-[0_8px_24px_rgb(0_0_0/0.18)] backdrop-blur-sm"
      style={{ left, width: TIP_WIDTH }}
    >
      {children}
    </div>
  );
}

function TipRow({ color, label, value }: { color?: string; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 py-0.5 text-xs">
      {color ? <span className="h-[2px] w-3 shrink-0" style={{ background: color }} /> : <span className="w-3 shrink-0" />}
      <span className="truncate text-ash">{label}</span>
      <span className="ml-auto font-display text-[13px] font-semibold tabular-nums text-ink">{value}</span>
    </div>
  );
}

/** A chart's key, from the series it draws: a mark in each colour, the name, and optionally a latest value. */
export function Legend({ series, swatch = "line", value }: { series: Series[]; swatch?: "line" | "area"; value?: (s: Series) => string }) {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
      {series.map((s) => (
        <li key={s.key} className="flex items-center gap-1.5">
          {swatch === "area" ? (
            <span className="size-2.5 rounded-[2px]" style={{ background: s.color }} aria-hidden />
          ) : (
            <span className="h-[2px] w-3.5" style={{ background: s.color }} aria-hidden />
          )}
          <span className="text-ash">{s.label}</span>
          {value && <span className="font-display text-[13px] font-semibold tabular-nums text-ink">{value(s)}</span>}
        </li>
      ))}
    </ul>
  );
}

/** A trend without axes: the shape of the recent past, its latest point marked. */
export function Sparkline({ values, color, height = 32 }: { values: (number | null)[]; color: string; height?: number }) {
  const known = values.filter((v): v is number => v !== null);
  if (values.length < 2 || known.length === 0) return <div style={{ height }} aria-hidden className="border-b border-line" />;
  const max = Math.max(...known) || 1;
  const x = (i: number) => (i / (values.length - 1)) * 100;
  const y = (v: number) => height - 2 - (v / max) * (height - 5);
  let lastIndex = values.length - 1;
  while (values[lastIndex] === null) lastIndex--;
  return (
    <div className="relative" style={{ height }} aria-hidden>
      <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" className="block h-full w-full overflow-visible">
        {knownRuns(values).map((run) => {
          const d = run.map((i, n) => `${n ? "L" : "M"}${x(i)},${y(values[i]!)}`).join("");
          return (
            <g key={run[0]}>
              <path d={`${d}L${x(run.at(-1)!)},${height}L${x(run[0])},${height}Z`} style={{ fill: color }} opacity={0.12} />
              <path d={d} fill="none" style={{ stroke: color }} strokeWidth={1.5} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
            </g>
          );
        })}
      </svg>
      <span
        className="absolute size-2 -translate-1/2 rounded-full ring-2 ring-surface"
        style={{ left: `${x(lastIndex)}%`, top: `${(y(values[lastIndex]!) / height) * 100}%`, background: color }}
      />
    </div>
  );
}

/** A histogram's buckets as columns, with quantile marks laid over them. */
export function Distribution({
  label,
  bounds,
  counts,
  color,
  marks = [],
  height = 150,
  formatBound,
  empty,
}: {
  label: string;
  bounds: number[];
  counts: number[];
  color: string;
  marks?: { label: string; value: number | null }[];
  height?: number;
  formatBound: (le: number) => string;
  /** Said in place of the columns when nothing was observed. */
  empty: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const n = counts.length;
  const total = counts.reduce((a, b) => a + b, 0);
  const pad = { left: 36, right: 6, top: 18, bottom: 22 };
  const plotW = Math.max(1, width - pad.left - pad.right);
  const plotH = height - pad.top - pad.bottom;
  const band = plotW / Math.max(1, n);
  const scale = niceScale(Math.max(0, ...counts), 3, true);
  const y = (v: number) => pad.top + plotH - (v / scale.top) * plotH;
  const bw = Math.max(2, Math.min(24, band - 2));
  const base = pad.top + plotH;
  const column = (i: number) => {
    const h = (plotH * counts[i]) / scale.top;
    if (h <= 0) return null;
    const x0 = pad.left + i * band + (band - bw) / 2;
    const top = base - h;
    const r = Math.min(4, h, bw / 2);
    return `M${x0},${base}V${top + r}Q${x0},${top} ${x0 + r},${top}H${x0 + bw - r}Q${x0 + bw},${top} ${x0 + bw},${top + r}V${base}Z`;
  };
  const markX = (v: number) => pad.left + bucketOffset(bounds, v) * band;
  const grid: number[] = [];
  for (let v = 0; v <= scale.top; v += scale.step) grid.push(v);
  const range = (i: number) => `${i === 0 ? "0" : formatBound(bounds[i - 1])} – ${formatBound(bounds[i])}`;
  const peak = counts.indexOf(Math.max(...counts));

  return (
    <div
      ref={ref}
      role="group"
      tabIndex={0}
      aria-label={total ? `${label}: ${total} requests; the most in ${range(peak)}. Arrow keys read each bucket.` : `${label}: ${empty}`}
      className="relative w-full select-none"
      style={{ height }}
      onPointerLeave={() => setHover(null)}
      onBlur={() => setHover(null)}
      onKeyDown={(e) => {
        if (n === 0) return;
        if (e.key === "ArrowRight") setHover((h) => Math.min(n - 1, (h ?? -1) + 1));
        else if (e.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? n) - 1));
        else if (e.key === "Escape") setHover(null);
        else return;
        e.preventDefault();
      }}
    >
      {width > 0 && (
        <svg width={width} height={height} className="block" aria-hidden>
          {grid.map((v) => (
            <g key={v}>
              <line x1={pad.left} x2={width - pad.right} y1={y(v)} y2={y(v)} style={{ stroke: "var(--line)" }} opacity={v === 0 ? 1 : 0.6} />
              <text x={pad.left - 8} y={y(v)} dy="0.32em" textAnchor="end" fontSize={11} className="font-display tabular-nums" style={{ fill: "var(--ash)" }}>
                {v}
              </text>
            </g>
          ))}
          {counts.map((_, i) => {
            const d = column(i);
            return d ? <path key={i} d={d} style={{ fill: color }} opacity={hover === null || hover === i ? 1 : 0.45} /> : null;
          })}
          {counts.map((_, i) =>
            band >= 38 || i % 2 === 0 || i === n - 1 ? (
              <text key={i} x={pad.left + i * band + band / 2} y={height - 6} textAnchor="middle" fontSize={10.5} className="font-display" style={{ fill: "var(--ash)" }}>
                {formatBound(bounds[i])}
              </text>
            ) : null,
          )}
          {total > 0 &&
            marks.map((m, k) =>
              m.value === null ? null : (
                <g key={m.label}>
                  <line x1={markX(m.value)} x2={markX(m.value)} y1={pad.top - 4} y2={base} style={{ stroke: "var(--ink)" }} strokeWidth={1.25} opacity={0.8} />
                  <text
                    x={markX(m.value) + (k % 2 ? 4 : -4)}
                    y={pad.top - 7}
                    textAnchor={k % 2 ? "start" : "end"}
                    fontSize={10.5}
                    className="font-display font-semibold"
                    style={{ fill: "var(--ink)" }}
                  >
                    {m.label}
                  </text>
                </g>
              ),
            )}
          {total === 0 && (
            <text x={pad.left + plotW / 2} y={pad.top + plotH / 2} textAnchor="middle" fontSize={12} style={{ fill: "var(--ash)" }}>
              {empty}
            </text>
          )}
          {counts.map((_, i) => (
            <rect key={i} x={pad.left + i * band} y={pad.top} width={band} height={plotH} fill="transparent" onPointerEnter={() => setHover(i)} />
          ))}
        </svg>
      )}
      {hover !== null && total > 0 && (
        <Tip left={tipLeft(pad.left + hover * band + band / 2, width, TIP_WIDTH, 12)}>
          <div className="mb-1 font-display text-[11px] text-ash">{range(hover)}</div>
          <TipRow color={color} label="Requests" value={String(counts[hover])} />
          <TipRow label="Share" value={`${Math.round((100 * counts[hover]) / total)}%`} />
        </Tip>
      )}
    </div>
  );
}
