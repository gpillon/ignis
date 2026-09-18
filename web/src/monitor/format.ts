// Display text for the Monitor: counts, rates, latencies, histogram bounds
// and moments relative to the latest scrape. `—` whenever a value is unknown.

const DASH = "—";

const trim = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)).replace(/\.0$/, "");

/** `1,284`, `12.3K`, `4.2M`. */
export function formatCount(n: number | null): string {
  if (n === null || !Number.isFinite(n)) return DASH;
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${trim(n / 1e9)}B`;
  if (abs >= 1e6) return `${trim(n / 1e6)}M`;
  if (abs >= 1e4) return `${trim(n / 1e3)}K`;
  return Math.round(n).toLocaleString("en-US");
}

/** A measured figure (a rate, a mean) with as many decimals as its size deserves: `0.25`, `4.5`, `43`, `12.3K`. */
export function formatNumber(n: number | null): string {
  if (n === null || !Number.isFinite(n)) return DASH;
  if (n === 0) return "0";
  const abs = Math.abs(n);
  if (abs < 1) return n.toFixed(2);
  if (abs < 10) return n.toFixed(1);
  if (abs < 1000) return n.toFixed(0);
  return formatCount(n);
}

/**
 * Memory in the binary units the plan is laid out in: `384 MiB`, `16 GiB`,
 * `29.69 GiB`. Two decimals below 100 of a unit, because a GiB is coarse
 * enough that rounding it whole would hide a quarter of a load’s workspace.
 */
export function formatBytes(bytes: number | null): string {
  if (bytes === null || !Number.isFinite(bytes)) return DASH;
  const abs = Math.abs(bytes);
  const units: [number, string][] = [
    [1024 ** 3, "GiB"],
    [1024 ** 2, "MiB"],
    [1024, "KiB"],
  ];
  const [scale, unit] = units.find(([s]) => abs >= s) ?? [1, "B"];
  const value = bytes / scale;
  const text = unit === "B" || Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2).replace(/\.?0+$/, "");
  return `${text} ${unit}`;
}

/** `850 ms`, `2.40 s`, `12.5 s`, `1m 12s`. */
export function formatSeconds(s: number | null): string {
  if (s === null || !Number.isFinite(s)) return DASH;
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 10) return `${s.toFixed(2)} s`;
  if (s < 60) return `${s.toFixed(1)} s`;
  const total = Math.round(s);
  const rest = total % 60;
  return rest ? `${Math.floor(total / 60)}m ${rest}s` : `${total / 60}m`;
}

/** A histogram bucket's upper bound: `50ms`, `2.5s`, `2m`, `+Inf`. */
export function formatBound(le: number): string {
  if (le === Infinity) return "+Inf";
  if (le < 1) return `${Math.round(le * 1000)}ms`;
  if (le < 60) return `${le}s`;
  return `${trim(le / 60)}m`;
}

/** How long ago, for the scrape clock: `just now`, `4 s ago`, `2 min ago`. */
export function formatAgo(ms: number): string {
  if (ms < 1500) return "just now";
  if (ms < 60_000) return `${Math.round(ms / 1000)} s ago`;
  return `${Math.round(ms / 60_000)} min ago`;
}

/** An axis offset from now: `now`, `−45s`, `−5m`. */
export function formatOffset(ms: number): string {
  if (ms <= 0) return "now";
  if (ms < 60_000) return `−${Math.round(ms / 1000)}s`;
  return `−${trim(ms / 60_000)}m`;
}

/** A window length: `1 min`, `15 min`, `30 s`. */
export function formatWindow(ms: number): string {
  return ms < 60_000 ? `${Math.round(ms / 1000)} s` : `${trim(ms / 60_000)} min`;
}

/** A share of a whole as a percentage, `—` for an empty whole. */
export function formatShare(part: number | null, whole: number | null): string {
  if (part === null || whole === null || whole <= 0) return DASH;
  const pct = (100 * part) / whole;
  return `${pct < 10 && pct > 0 ? pct.toFixed(1) : pct.toFixed(0)}%`;
}

/** A clean axis: the top value and step for `ticks` gridlines over `[0, max]`. */
export function niceScale(max: number, ticks = 4, integer = false): { top: number; step: number } {
  if (!(max > 0)) return integer ? { top: ticks, step: 1 } : { top: 1, step: 1 / ticks };
  // The finest nice step (1, 2, 2.5, 5 × a power of ten) that needs at most one gridline more than asked.
  const mag = 10 ** Math.floor(Math.log10(max / ticks));
  for (const nice of [1, 2, 2.5, 5, 10, 20, 25, 50, 100]) {
    const step = integer ? Math.max(1, Math.ceil(nice * mag)) : nice * mag;
    const intervals = Math.ceil(max / step - 1e-9);
    if (intervals <= ticks + 1) return { top: intervals * step, step };
  }
  return { top: max, step: max / ticks };
}
