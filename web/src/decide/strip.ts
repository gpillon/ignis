// The whole of a long target at a glance: its segments bucketed into a fixed
// number of columns, each as tall as its lines are long and marked where a
// line in it reads as an error or a warning. The file card shows it for what
// was loaded, and a `locate`'s answer for where in the text the pick sits —
// "line 48,213 of 120,004" is a number, and the strip is the place.

/** How loud the loudest segment in a column is. */
export type Level = "error" | "warn" | null;

export type Column = {
  /** The first segment the column covers; the next column's `from` is where it ends. */
  from: number;
  /** Its segments' mean length over the loudest column's, in (0, 1]; 0 when every segment is blank. */
  weight: number;
  level: Level;
};

export type Strip = {
  columns: Column[];
  /** How many segments read as errors, and as warnings — the legend's figures. */
  errors: number;
  warnings: number;
};

// Bare level words only in capitals: "error" in a sentence is prose, and a
// strip of a document tinted by it would be saying something false. A level
// written as a field (`level=warn`, `"severity": "ERROR"`) is a level whatever
// its case, and so is a thrown type heading its message (`TypeError: …`,
// `java.io.IOException: …`), which is how a stack trace says it.
const ERROR_WORD = /\b(FATAL|ERROR|ERR|CRIT|CRITICAL|PANIC|SEVERE|EMERG|ALERT)\b|\bException\b|\b[A-Z]\w*(?:Exception|Error)(?::|$)|^Traceback \(most recent call last\)/;
const WARN_WORD = /\b(WARN|WARNING)\b/;
const LEVEL_FIELD = /(?:\blevel|\blvl|\bseverity)"?\s*[=:]\s*"?(error|err|fatal|crit|critical|panic|warn|warning)\b/i;

/** What `line` says of its own level, if anything. */
export function levelOf(line: string): Level {
  const field = LEVEL_FIELD.exec(line);
  if (field) return field[1].toLowerCase().startsWith("warn") ? "warn" : "error";
  if (ERROR_WORD.test(line)) return "error";
  if (WARN_WORD.test(line)) return "warn";
  return null;
}

/** The most columns a strip draws: a hairline each across the bench, and still a bar each across the answers. */
export const STRIP_WIDTH = 120;

const cache = new WeakMap<readonly string[], Map<number, Strip>>();

/**
 * `segments` in at most `width` columns. Fewer segments than columns get one
 * column each, so a short text is not stretched into a false texture.
 *
 * Kept per segments array: the target a `locate` cut is cached by the evidence
 * it came from, so re-rendering an answer does not re-read a whole log.
 */
export function stripOf(segments: readonly string[], width: number): Strip {
  const byWidth = cache.get(segments) ?? new Map<number, Strip>();
  cache.set(segments, byWidth);
  const hit = byWidth.get(width);
  if (hit) return hit;

  const count = Math.max(1, Math.min(width, segments.length));
  const levels = segments.map(levelOf);
  const columns: Column[] = [];
  const means: number[] = [];
  for (let c = 0; c < count; c++) {
    const from = Math.floor((c * segments.length) / count);
    const to = Math.floor(((c + 1) * segments.length) / count);
    let chars = 0;
    let level: Level = null;
    for (let i = from; i < to; i++) {
      chars += segments[i].length;
      if (levels[i] === "error") level = "error";
      else if (levels[i] === "warn" && level === null) level = "warn";
    }
    means.push(to > from ? chars / (to - from) : 0);
    columns.push({ from, weight: 0, level });
  }
  const loudest = Math.max(...means);
  for (let c = 0; c < count; c++) columns[c].weight = loudest > 0 ? means[c] / loudest : 0;

  const strip = {
    columns,
    errors: levels.filter((l) => l === "error").length,
    warnings: levels.filter((l) => l === "warn").length,
  };
  byWidth.set(width, strip);
  return strip;
}

/** Where segment `index` of `total` sits along the strip, as a fraction of its width, at the segment's middle. */
export const positionOf = (index: number, total: number): number => (total <= 0 ? 0 : (index + 0.5) / total);
