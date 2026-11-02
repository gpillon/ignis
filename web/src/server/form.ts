import type { CatalogRow, ConfigDocument, FieldDecl, OriginRow } from "./routes.ts";

// The Server view's form (spec playground/05 §Configuration, §Applying), as
// pure functions over the declarations and the running configuration: which
// control a field gets, which value is in force and where it was read from,
// and the draft of changes that becomes one PATCH /v1/config. The server is
// the judge of a value: nothing here checks one, a row sends what was typed.

/** The names people give the scopes the server declares: the one thing here that is not read from it. */
const FAMILY_NAMES: Record<string, string> = { qwen38: "Qwen3.8-27B", qwen38flashnext: "Qwen3.8-Flash-Next" };

/** A family's scope name as people say it; a scope with no name here is shown as the server spells it. */
export const familyName = (scope: string) => FAMILY_NAMES[scope] ?? scope;

const GROUP_TITLES: Record<string, string> = {
  server: "Server",
  model: "Model",
  vram: "VRAM",
  reuse: "Prompt reuse",
  switch: "Model switch",
  spec: "Speculative decoding",
  vision: "Vision",
  media: "Media",
  download: "Download",
  ngram: "N-gram table",
  kv_disk: "KV-disk",
};

/** A group's heading; a group this page does not know yet is shown by its key. */
export function groupTitle(group: string): string {
  return GROUP_TITLES[group] ?? group;
}

export type Control =
  | { kind: "switch" }
  | { kind: "segmented"; options: string[] }
  | { kind: "select"; options: string[] }
  /** `hint`: the kind and the rule; `placeholder`: "auto" for an unset value. */
  | { kind: "text"; hint: string; placeholder: string };

/** The control a declaration calls for, for a field whose value in force is `value`. */
export function controlFor(decl: FieldDecl, value: unknown): Control {
  if (decl.kind === "bool") return { kind: "switch" };
  if (decl.rule && "one_of" in decl.rule) {
    // The server lists canonical forms; a value in force it spells otherwise (an alias set from the file) is listed too, to show as chosen.
    const named = decl.rule.one_of;
    const shown = typeof value === "string" ? value : null;
    const options = shown !== null && !named.some((option) => option.toLowerCase() === shown.toLowerCase()) ? [...named, shown] : named;
    return options.length <= 4 ? { kind: "segmented", options } : { kind: "select", options };
  }
  const hint = decl.rule_text ? `${decl.kind}, ${decl.rule_text}` : decl.kind;
  return { kind: "text", hint, placeholder: value === null || value === undefined ? "auto" : textOf(value) };
}

/**
 * How a row shows its field: with a control, or read-only because it is set
 * at start (not patchable) or because it is `switch.known_models`, which a
 * patch only adds to. A scope shows no field its family does not take
 * (`inScope`).
 */
export type RowMode = { kind: "edit" } | { kind: "fixed" } | { kind: "list" };

export function rowMode(decl: FieldDecl): RowMode {
  if (!decl.patchable) return { kind: "fixed" };
  if (decl.file_key === "switch.known_models") return { kind: "list" };
  return { kind: "edit" };
}

/** A value in force and where an edit of it goes: a family's section, or the group (`null`). */
export type InForce = { value: unknown; scope: string | null };

/** Every family the server declares, by scope name: those a field is limited to, those the catalog's artifacts belong to, and the running one. */
export function familiesOf(fields: FieldDecl[], catalog: CatalogRow[] | null, running: string | null): string[] {
  const named = fields.flatMap((decl) => decl.families ?? []);
  const served = (catalog ?? []).flatMap((row) => row.family ?? []);
  return [...new Set([...named, ...served, ...(running ? [running] : [])])].sort();
}

export const section = (doc: ConfigDocument, group: string, scope: string) => {
  const found = doc[group]?.[scope];
  return found !== null && typeof found === "object" ? (found as Record<string, unknown>) : undefined;
};

/** The one family a field is only taken by; `null` for a field every family takes. */
const onlyFamily = (decl: FieldDecl) => (decl.families?.length === 1 ? decl.families[0] : null);

/**
 * Whether a scope shows the field: All models (`null`) every field; a family
 * only what it can set for itself, a field that takes a value per family and
 * that it takes, or one only it takes. A field with one value for every model
 * is All models', and so is the place to change it.
 */
export function inScope(decl: FieldDecl, scope: string | null): boolean {
  if (scope === null) return true;
  return onlyFamily(decl) === scope || (decl.scoped && (decl.families === null || decl.families.includes(scope)));
}

/** The value a family runs a field with: its own section's when it has one, else the group's. */
export function resolved(doc: ConfigDocument, decl: FieldDecl, family: string): unknown {
  const own = section(doc, decl.group, family);
  return own && decl.name in own ? own[decl.name] : doc[decl.group]?.[decl.name];
}

/**
 * Where a family's value comes from (spec playground/05 §Follow-up 2): the
 * operator's own, in the file or a live change, which a shared edit does not
 * reach and a removal takes away; a profile's, or a flag's or env var's,
 * which a shared edit does reach and nothing here removes; or the shared one.
 */
export type Origin = { kind: "own" } | { kind: "profile"; name?: string } | { kind: "set"; spelling?: string } | { kind: "shared" };

/**
 * A family's value of `decl` by its origin. Without the origins read, a
 * family section that holds the field is the family's own, as the file's
 * shape alone tells it.
 */
export function originOf(decl: FieldDecl, doc: ConfigDocument | null, origins: OriginRow[] | null, family: string): Origin {
  const row = origins?.find((r) => r.key === decl.file_key && r.scope === family);
  if (!row) {
    const own = doc ? section(doc, decl.group, family) : undefined;
    return own !== undefined && decl.name in own ? { kind: "own" } : { kind: "shared" };
  }
  if (!row.scoped) return { kind: "shared" };
  switch (row.source) {
    case "file":
    case "live":
      return { kind: "own" };
    case "profile":
      return { kind: "profile", name: row.spelling };
    case "flag":
    case "env":
      return { kind: "set", spelling: row.spelling };
    case "default":
      return { kind: "shared" };
  }
}

/** A family that has a value of its own for a field, the value it runs, and where it comes from. */
type Scoped = { family: string; value: unknown; origin: Origin };

/**
 * What a row shows in a scope (`null` is "All models"), spec playground/05
 * §Follow-up. `inForce` is the value the control starts from and the section
 * an edit goes to; `shared` is the group's value, which "Use shared value"
 * falls back to; `note` says what the value is: its origin in a family's
 * scope, or which families have a value of their own.
 */
export type RowView = {
  mode: RowMode;
  inForce: InForce;
  shared: unknown;
  note:
    | { kind: "scoped"; origin: Origin }
    | { kind: "only"; family: string }
    | { kind: "overridden"; by: Scoped[] }
    | null;
};

export function rowView(decl: FieldDecl, doc: ConfigDocument | null, origins: OriginRow[] | null, scope: string | null): RowView {
  const mode = rowMode(decl);
  const shared = doc?.[decl.group]?.[decl.name];
  const only = onlyFamily(decl);
  if (only !== null) {
    // Only ever in its family's section, whichever scope shows it.
    const own = doc ? section(doc, decl.group, only) : undefined;
    return { mode, shared, inForce: { value: own?.[decl.name], scope: only }, note: scope === null ? { kind: "only", family: only } : null };
  }
  if (!decl.scoped) {
    // Shown in All models only (`inScope`), which says nothing of a field every family shares.
    return { mode, shared, inForce: { value: shared, scope: null }, note: null };
  }
  if (scope === null) {
    // The families the origins name for the field; unread, those whose section holds it.
    const names = origins ? origins.filter((r) => r.key === decl.file_key && r.scope !== null).map((r) => r.scope!) : Object.keys(doc?.[decl.group] ?? {});
    const by = names
      .map((family) => ({ family, value: doc ? resolved(doc, decl, family) : undefined, origin: originOf(decl, doc, origins, family) }))
      .filter(({ origin }) => origin.kind !== "shared");
    return { mode, shared, inForce: { value: shared, scope: null }, note: by.length > 0 ? { kind: "overridden", by } : null };
  }
  const own = doc ? section(doc, decl.group, scope) : undefined;
  const has = own !== undefined && decl.name in own;
  return { mode, shared, inForce: { value: has ? own[decl.name] : shared, scope }, note: { kind: "scoped", origin: originOf(decl, doc, origins, scope) } };
}

/** A value as a note quotes it: unset is "auto". */
export function labelOf(value: unknown): string {
  return value === null || value === undefined ? "auto" : shownOf(value);
}

/** The lines a row says about its value, given its pending edit (`entry`) in this scope, if any. */
export function noteLines(view: RowView, entry: DraftEntry | undefined): string[] {
  const note = view.note;
  if (note === null) return [];
  switch (note.kind) {
    case "scoped":
      if (entry?.removal) return ["Back to the shared value once applied"];
      return [entry ? "Own value" : originLine(note.origin)];
    case "only":
      return [`${familyName(note.family)} only`];
    case "overridden":
      return note.by.map(overrideLine);
  }
}

const setBy = (spelling: string | undefined) => spelling ?? "a flag or an env var";

function originLine(origin: Origin): string {
  switch (origin.kind) {
    case "own":
      return "Own value";
    case "profile":
      return origin.name ? `From profile ${origin.name}` : "From profile";
    case "set":
      return `Set by ${setBy(origin.spelling)}`;
    case "shared":
      return "Shared";
  }
}

/** What a shared edit does to one family with a value of its own, as All models says it. */
function overrideLine({ family, value, origin }: Scoped): string {
  const name = familyName(family);
  if (origin.kind === "profile") return `Replaces the profile's ${labelOf(value)} for ${name}`;
  if (origin.kind === "set") return `${name} is set by ${setBy(origin.spelling)}`;
  return `${name} has its own value: ${labelOf(value)}`;
}

/** Whether "Use shared value" is on offer: the family has an own value (file or live) and none is on its way out. */
export function canUseShared(view: RowView, entry: DraftEntry | undefined): boolean {
  return view.note?.kind === "scoped" && view.note.origin.kind === "own" && !entry?.removal;
}

/** A value as a text field shows it: unset is empty. */
export function textOf(value: unknown): string {
  if (value === null || value === undefined) return "";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

/** A value shown with no control. */
export function shownOf(value: unknown): string {
  if (value === null || value === undefined) return "Not set";
  if (typeof value === "boolean") return value ? "On" : "Off";
  return textOf(value);
}

/**
 * One edited field: where it goes (a family's section, or the group), what
 * was typed or chosen, whether it needs a reload, and whether it removes the
 * family's own value instead (`null` in the patch).
 */
export type DraftEntry = { group: string; name: string; scope: string | null; value: string | boolean; reload: boolean; removal?: true };

/** The edits, by `draftKey`. A cleared text field stays here so the field can show it empty. */
export type Draft = Record<string, DraftEntry>;

/** A field's key in the draft: the group's and a family's section hold separate edits of one field. */
export const draftKey = (group: string, name: string, scope: string | null) => (scope === null ? `${group}.${name}` : `${group}.${scope}.${name}`);

const same = (decl: FieldDecl, value: string | boolean, current: unknown) => {
  if (typeof value === "boolean") return value === current;
  const typed = value.trim();
  const shown = textOf(current);
  return decl.rule && "one_of" in decl.rule ? typed.toLowerCase() === shown.toLowerCase() : typed === shown;
};

/**
 * The draft with `decl` set to `value`. The edit goes where `inForce` says
 * (the family's section or the group); a value equal to the one in force
 * drops the entry, so the draft holds only real changes.
 */
export function editDraft(draft: Draft, decl: FieldDecl, inForce: InForce, value: string | boolean): Draft {
  const key = draftKey(decl.group, decl.name, inForce.scope);
  const { [key]: _dropped, ...rest } = draft;
  if (same(decl, value, inForce.value)) return rest;
  return { ...rest, [key]: { group: decl.group, name: decl.name, scope: inForce.scope, value, reload: decl.reload_required } };
}

/** The draft with the family's own value of `decl` on its way out: `null` in its section. */
export function removeDraft(draft: Draft, decl: FieldDecl, scope: string): Draft {
  return { ...draft, [draftKey(decl.group, decl.name, scope)]: { group: decl.group, name: decl.name, scope, value: "", reload: decl.reload_required, removal: true } };
}

/** The draft without the edit or removal of `decl` in `scope`. */
export function dropDraft(draft: Draft, decl: FieldDecl, scope: string | null): Draft {
  const { [draftKey(decl.group, decl.name, scope)]: _dropped, ...rest } = draft;
  return rest;
}

/**
 * The entries a PATCH carries: every one but a cleared text field — PATCH
 * reads an unset value as no change, so a removal is `null`, never empty
 * (spec playground/05, Out of Scope).
 */
export function changes(draft: Draft): Draft {
  return Object.fromEntries(Object.entries(draft).filter(([, entry]) => entry.removal || typeof entry.value === "boolean" || entry.value.trim() !== ""));
}

/** The PATCH /v1/config body: only the changed fields, nested as the config file nests them; a removal is `null` in the family's section. */
export function patchBody(draft: Draft): Record<string, Record<string, unknown>> {
  const body: Record<string, Record<string, unknown>> = {};
  for (const entry of Object.values(changes(draft))) {
    const group = (body[entry.group] ??= {});
    const at = entry.scope === null ? group : ((group[entry.scope] ??= {}) as Record<string, unknown>);
    at[entry.name] = entry.removal ? null : typeof entry.value === "string" ? entry.value.trim() : entry.value;
  }
  return body;
}

/**
 * What the page has read of the running configuration, and who runs it: the
 * declarations, the document and where its values come from, the running
 * family and every family there is.
 */
export type Basis = { fields: FieldDecl[]; doc: ConfigDocument | null; origins: OriginRow[] | null; family: string | null; families: string[] };

const declOf = (fields: FieldDecl[], entry: DraftEntry) => fields.find((d) => d.group === entry.group && d.name === entry.name);

/**
 * The configuration after the draft, as far as the page can tell: a removal
 * drops the family's own value, and a shared value replaces a family's value
 * that is not its own, as the server resolves it (a patch outranks a profile,
 * a flag and an env var).
 */
function applied(draft: Draft, basis: Basis & { doc: ConfigDocument }): ConfigDocument {
  const after: ConfigDocument = Object.fromEntries(Object.entries(basis.doc).map(([group, fields]) => [group, { ...fields }]));
  for (const entry of Object.values(changes(draft))) {
    const group = (after[entry.group] ??= {});
    if (entry.scope === null) {
      group[entry.name] = entry.value;
      const decl = declOf(basis.fields, entry);
      for (const family of basis.families) {
        const sectioned = section(after, entry.group, family);
        if (decl && sectioned && entry.name in sectioned && originOf(decl, basis.doc, basis.origins, family).kind !== "own") {
          const { [entry.name]: _replaced, ...rest } = sectioned;
          group[family] = rest;
        }
      }
      continue;
    }
    const at = (group[entry.scope] = { ...(group[entry.scope] as object | undefined) }) as Record<string, unknown>;
    if (entry.removal) delete at[entry.name];
    else at[entry.name] = entry.value;
  }
  return after;
}

const norm = (value: unknown) => textOf(value).trim().toLowerCase();

/**
 * Whether an entry changes what the running family runs with: a family's own
 * value (file or live) outranks a shared edit, and anything else resolves
 * to the edited value. With no family or configuration known, it is assumed to.
 */
function reaches(entry: DraftEntry, after: ConfigDocument | null, basis: Basis): boolean {
  const { doc, family } = basis;
  const decl = declOf(basis.fields, entry);
  if (!doc || !after || family === null || !decl) return true;
  if (entry.scope === null && originOf(decl, doc, basis.origins, family).kind === "own") return false;
  return norm(resolved(doc, decl, family)) !== norm(resolved(after, decl, family));
}

const appliedTo = (draft: Draft, basis: Basis) => (basis.doc ? applied(draft, { ...basis, doc: basis.doc }) : null);

/**
 * Whether applying the draft reloads the running model: a changed or removed
 * `reload_required` value that resolves differently for the running family.
 * A value for the other family, or a group key the running family has its
 * own value for, is a live change; and when any value does reload, the whole
 * patch goes through the reload.
 */
export function needsReload(draft: Draft, basis: Basis): boolean {
  const after = appliedTo(draft, basis);
  return Object.values(changes(draft)).some((entry) => entry.reload && reaches(entry, after, basis));
}

/**
 * The values that wait for another model, by family, as `Takes effect when
 * <family> loads` lists them: a value in a family's section waits for that
 * family; a group key the running family has its own value for waits for
 * each other family that has none.
 */
export function waitingForOthers(draft: Draft, basis: Basis): { family: string; keys: string[] }[] {
  const { doc, family } = basis;
  const after = appliedTo(draft, basis);
  if (!doc || !after || family === null) return [];
  const others = basis.families.filter((f) => f !== family);
  const by = new Map<string, string[]>();
  for (const entry of Object.values(changes(draft))) {
    if (reaches(entry, after, basis)) continue;
    const decl = declOf(basis.fields, entry);
    const goes = entry.scope !== null ? [entry.scope] : others.filter((f) => !decl || originOf(decl, doc, basis.origins, f).kind !== "own");
    for (const f of goes) by.set(f, [...(by.get(f) ?? []), draftKey(entry.group, entry.name, entry.scope)]);
  }
  return [...by].map(([f, keys]) => ({ family: f, keys }));
}

/** The dock's line for values that wait for another model. */
export const waitingLine = (entry: { family: string; keys: string[] }) => `Takes effect when ${familyName(entry.family)} loads: ${entry.keys.join(", ")}`;

/** The primary action's label, which says what will happen before the click. */
export function primaryLabel(reload: boolean, id: string): string {
  return reload ? `Apply and reload ${id}` : "Apply";
}

const escaped = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * The file keys of the fields a refusal names — as `<group>.<name>` or in a
 * family section, `<group>.<family>.<name>` — never a key that only begins
 * like a longer one (`server.metrics` in `server.metrics_bind`).
 */
export function namedKeys(message: string, fields: FieldDecl[]): Set<string> {
  const named = new Set<string>();
  for (const decl of fields) {
    const key = new RegExp(`(?<![\\w.])${escaped(decl.group)}\\.(?:\\w+\\.)?${escaped(decl.name)}(?!\\w)`);
    if (key.test(message)) named.add(decl.file_key);
  }
  return named;
}

/** A description's first sentence, and the rest (empty when there is none). */
export function firstSentence(text: string): [string, string] {
  const split = /^(.+?\.)\s+(?=[A-Z`])(.*)$/s.exec(text);
  return split ? [split[1], split[2]] : [text, ""];
}

/**
 * How a field meets the Configuration search: `marks`, the spans of its name
 * the words cover; `opens`, a word is only in the description's part behind
 * "More"; `ids`, a word is only in the flag or the environment variable, which
 * the row then shows.
 */
export type FieldMatch = { marks: [number, number][]; opens: boolean; ids: boolean };

/**
 * How `decl` meets `query`, or `null` when a word of it is in none of the
 * field's `group.name` key, description, flag and environment variable. Case
 * does not count, and every word must be in one of them; a word is taken where
 * the row shows it first. An empty query lets every field through.
 */
export function fieldMatches(decl: FieldDecl, query: string): FieldMatch | null {
  const key = decl.file_key.toLowerCase();
  const [first, rest] = firstSentence(decl.description).map((part) => part.toLowerCase());
  const ids = `${decl.flag} ${decl.env}`.toLowerCase();
  // The name is the key's tail: a span of the key over it is a span of the name.
  const start = key.length - decl.name.length;
  const spans: [number, number][] = [];
  const match: FieldMatch = { marks: [], opens: false, ids: false };
  for (const word of query.toLowerCase().split(/\s+/).filter(Boolean)) {
    if (key.includes(word)) {
      for (let at = key.indexOf(word); at !== -1; at = key.indexOf(word, at + 1)) {
        const from = Math.max(at, start);
        if (from < at + word.length) spans.push([from - start, at + word.length - start]);
      }
    } else if (first.includes(word)) continue;
    else if (rest.includes(word)) match.opens = true;
    else if (ids.includes(word)) match.ids = true;
    else return null;
  }
  for (const [from, to] of spans.sort((a, b) => a[0] - b[0])) {
    const last = match.marks.at(-1);
    if (last && from <= last[1]) last[1] = Math.max(last[1], to);
    else match.marks.push([from, to]);
  }
  return match;
}

/** `switch.drain_timeout` in force: the seconds a switch or a reload gives a running reply. */
export function drainTimeout(doc: ConfigDocument | null): number | null {
  const raw = doc?.switch?.drain_timeout;
  if (raw === null || raw === undefined) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}
