// The Prometheus text exposition format 0.0.4, parsed in the browser
// (GitHub #165): `# HELP` and `# TYPE` lines, samples with labels, and a
// histogram's `_bucket`/`_sum`/`_count` series grouped under the histogram
// they belong to. A line it cannot read is reported, not thrown, so one bad
// line never blanks the panel.

export type MetricType = "counter" | "gauge" | "histogram" | "summary" | "untyped";

export type Sample = { name: string; labels: Record<string, string>; value: number };

export type Family = { name: string; type: MetricType; help: string; samples: Sample[] };

export type ParseError = { line: number; text: string; reason: string };

export type Exposition = { families: Map<string, Family>; errors: ParseError[] };

const TYPES = new Set<MetricType>(["counter", "gauge", "histogram", "summary", "untyped"]);
const NAME = /^[a-zA-Z_:][a-zA-Z0-9_:]*$/;
const SUFFIXES: Record<string, string[]> = {
  histogram: ["_bucket", "_sum", "_count"],
  summary: ["_sum", "_count"],
};

export function parseExposition(text: string): Exposition {
  const families = new Map<string, Family>();
  const errors: ParseError[] = [];
  const family = (name: string): Family => {
    let f = families.get(name);
    if (!f) {
      f = { name, type: "untyped", help: "", samples: [] };
      families.set(name, f);
    }
    return f;
  };

  text.split("\n").forEach((raw, index) => {
    const line = raw.trim();
    const fail = (reason: string) => errors.push({ line: index + 1, text: raw, reason });
    if (line === "") return;

    if (line.startsWith("#")) {
      const meta = /^#\s*(HELP|TYPE)\s+(\S+)(?:\s+(.*))?$/.exec(line);
      if (!meta) {
        // A free comment, unless it tries to be a HELP or TYPE line.
        if (/^#\s*(HELP|TYPE)\b/.test(line)) fail("HELP or TYPE without a metric name");
        return;
      }
      const [, kind, name, rest = ""] = meta;
      if (!NAME.test(name)) return fail(`invalid metric name "${name}"`);
      if (kind === "HELP") {
        family(name).help = unescapeHelp(rest);
        return;
      }
      const type = rest.trim() as MetricType;
      if (!TYPES.has(type)) return fail(`unknown metric type "${rest.trim()}"`);
      family(name).type = type;
      return;
    }

    const sample = parseSample(line);
    if (typeof sample === "string") return fail(sample);
    family(familyOf(sample.name, families)).samples.push(sample);
  });

  return { families, errors };
}

/** The family a sample belongs to: a declared histogram or summary its suffix names, else its own name. */
function familyOf(name: string, families: Map<string, Family>): string {
  for (const [type, suffixes] of Object.entries(SUFFIXES)) {
    for (const suffix of suffixes) {
      if (!name.endsWith(suffix)) continue;
      const base = name.slice(0, -suffix.length);
      if (families.get(base)?.type === type) return base;
    }
  }
  return name;
}

/** One sample line, or why it is not one. */
function parseSample(line: string): Sample | string {
  const nameMatch = /^[a-zA-Z_:][a-zA-Z0-9_:]*/.exec(line);
  if (!nameMatch) return "a sample starts with a metric name";
  const name = nameMatch[0];
  let rest = line.slice(name.length);
  const labels: Record<string, string> = {};

  if (rest.startsWith("{")) {
    let i = 1;
    for (;;) {
      while (rest[i] === " ") i++;
      if (rest[i] === "}") {
        i++;
        break;
      }
      const key = /^[a-zA-Z_][a-zA-Z0-9_]*/.exec(rest.slice(i));
      if (!key) return "a label name was expected";
      i += key[0].length;
      while (rest[i] === " ") i++;
      if (rest[i] !== "=" || rest[i + 1] !== '"') return `label "${key[0]}" has no quoted value`;
      i += 2;
      let value = "";
      for (;;) {
        const c = rest[i];
        if (c === undefined) return "a label value is not closed";
        if (c === '"') break;
        if (c === "\\") {
          const next = rest[i + 1];
          value += next === "n" ? "\n" : (next ?? "");
          i += 2;
          continue;
        }
        value += c;
        i++;
      }
      i++;
      labels[key[0]] = value;
      while (rest[i] === " ") i++;
      if (rest[i] === ",") {
        i++;
        continue;
      }
      if (rest[i] === "}") {
        i++;
        break;
      }
      return "labels are not closed";
    }
    rest = rest.slice(i);
  }

  // The value, then an optional millisecond timestamp.
  const fields = rest.trim().split(/\s+/);
  if (rest.trim() === "" || fields.length > 2) return "a sample has one value and an optional timestamp";
  const value = parseValue(fields[0]);
  if (value === null) return `"${fields[0]}" is not a number`;
  if (fields.length === 2 && !/^-?\d+$/.test(fields[1])) return `timestamp "${fields[1]}" is not whole milliseconds`;
  return { name, labels, value };
}

function parseValue(text: string): number | null {
  if (text === "NaN") return Number.NaN;
  if (text === "+Inf" || text === "Inf") return Infinity;
  if (text === "-Inf") return -Infinity;
  if (!/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(text)) return null;
  return Number(text);
}

function unescapeHelp(text: string): string {
  return text.replace(/\\(n|\\)/g, (_, c: string) => (c === "n" ? "\n" : "\\"));
}
