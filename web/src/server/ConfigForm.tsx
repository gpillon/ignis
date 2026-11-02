import { Fragment, type ReactNode, useState } from "react";
import { field, fieldLook } from "../ui/classes.ts";
import { IconClose, IconSearch } from "../ui/icons.tsx";
import { Segmented } from "../ui/Segmented.tsx";
import { Switch } from "../ui/Switch.tsx";
import { ConfigFileButton } from "./ConfigFile.tsx";
import {
  canUseShared,
  changes,
  controlFor,
  type Draft,
  draftKey,
  familiesOf,
  familyName,
  type FieldMatch,
  fieldMatches,
  firstSentence,
  groupTitle,
  type InForce,
  inScope,
  labelOf,
  noteLines,
  rowView,
  shownOf,
  textOf,
} from "./form.ts";
import type { CatalogRow, ConfigDocument, FieldDecl, OriginRow } from "./routes.ts";
import type { Loaded } from "./useServer.ts";

// The running configuration as a form (spec playground/05 §Configuration):
// one surface, a section per group in declaration order, a row per visible
// field with the control its declaration calls for. The one colour it spends
// is the heat mark on a field whose change reloads the model — the heat of
// the header's bar, which runs while that reload does. No popup lives in the
// clipped (`.cut`) surface: a choice of more than four is a native select.
//
// A scope selector at the top picks whose values the rows show and edit (spec
// playground/05 §Follow-up): "All models" the shared ones, a family its own
// where it has any, and only the fields a family can set for itself
// (`inScope`): one value for every model is All models' to show and change. It
// opens on All models; the running family's segment is filled with the heat.
//
// A search beside it narrows the rows to the fields it finds (by key, flag,
// environment variable or description) and the groups that hold them. It only
// hides rows: a change on a hidden one stays in the draft, and in the patch.

/** What a row's controls ask of the draft: an edit, the family's own value removed, or that removal taken back. */
export type RowEdits = {
  onEdit: (decl: FieldDecl, inForce: InForce, value: string | boolean) => void;
  onRemove: (decl: FieldDecl, scope: string) => void;
  onUndo: (decl: FieldDecl, scope: string) => void;
};

export function ConfigForm({
  fields,
  config,
  origins,
  catalog,
  family,
  draft,
  marked,
  initialScope,
  initialQuery = "",
  ...edits
}: {
  fields: Loaded<FieldDecl[]>;
  config: Loaded<ConfigDocument>;
  /** Where each value comes from; the rows read the file's shape alone until it is read, or when the server has no such route. */
  origins: Loaded<OriginRow[]>;
  /** Names the families whose artifacts the server can load. */
  catalog: Loaded<CatalogRow[]>;
  /** The running family's scope name, from the catalog's loaded row; unknown without it. */
  family: string | null;
  draft: Draft;
  marked: string[];
  /** The scope the form opens on: a family, or `null` for "All models". Unset, All models. */
  initialScope?: string | null;
  /** The search the form opens with; empty, every field shows. */
  initialQuery?: string;
} & RowEdits) {
  const [scope, setScope] = useState<string | null>(initialScope ?? null);
  const [query, setQuery] = useState(initialQuery);
  return (
    <section aria-labelledby="server-config" className="flex flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-1">
        <div className="min-w-0">
          {/* The tab above names it: the heading stays for a screen reader's outline. */}
          <h2 id="server-config" className="sr-only">
            Configuration
          </h2>
          <p className="text-sm text-ash">
            What the server runs with, for all models or for one. A change is sent as one patch, and the server judges every value in it.
          </p>
        </div>
        {config.state === "ok" && <ConfigFileButton />}
      </div>
      <Body fields={fields} config={config} origins={origins} catalog={catalog} family={family} scope={scope} onScope={setScope} query={query} onQuery={setQuery} draft={draft} marked={marked} {...edits} />
    </section>
  );
}

/** "All models", then a family each, the running one marked: a segmented row from `sm`, filled with the heat; a select below it, in words. */
function ScopeSelector({ families, running, scope, onScope }: { families: string[]; running: string | null; scope: string | null; onScope: (scope: string | null) => void }) {
  const options: { value: string; label: string; accent?: string }[] = [
    { value: "all", label: "All models" },
    ...families.map((f) => ({ value: f, label: familyName(f), accent: f === running ? "running" : undefined })),
  ];
  const value = scope ?? "all";
  const pick = (next: string) => onScope(next === "all" ? null : next);
  return (
    <div>
      <div className="hidden sm:block">
        <Segmented legend="Scope" legendHidden name="server-scope" value={value} options={options} onChange={pick} />
      </div>
      <select aria-label="Scope" name="server-scope-select" value={value} onChange={(e) => pick(e.target.value)} className={`w-full ${fieldLook} sm:hidden`}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.accent ? `${option.label} (${option.accent})` : option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function Body({
  fields,
  config,
  origins,
  catalog,
  family,
  scope,
  onScope,
  query,
  onQuery,
  draft,
  marked,
  ...edits
}: Parameters<typeof ConfigForm>[0] & { scope: string | null; onScope: (scope: string | null) => void; query: string; onQuery: (query: string) => void }) {
  if (config.state === "error" && config.refusal.status === 501) {
    return <Notice>This server has no running configuration to show. The model above can still be switched.</Notice>;
  }
  if (fields.state === "error") return <Notice>The fields could not be read: {fields.refusal.message}</Notice>;
  if (config.state === "error") return <Notice>The configuration could not be read: {config.refusal.message}</Notice>;
  if (fields.state === "loading" || config.state === "loading") return <p className="py-6 text-sm text-ash">Reading the configuration…</p>;

  const shown = fields.value.filter((decl) => inScope(decl, scope));
  const groups = [...new Set(shown.map((decl) => decl.group))];
  const found = shown.flatMap((decl) => {
    const match = fieldMatches(decl, query);
    return match ? [{ decl, match }] : [];
  });
  const holding = new Set(found.map(({ decl }) => decl.group));
  const changed = changes(draft);
  const jump = (group: string) => document.getElementById(`server-group-${group}`)?.scrollIntoView({ block: "start" });
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-stretch sm:gap-4">
        <div className="min-w-0 sm:flex-1">
          <ScopeSelector families={familiesOf(fields.value, catalog.state === "ok" ? catalog.value : null, family)} running={family} scope={scope} onScope={onScope} />
        </div>
        <SearchBox query={query} count={found.length} onQuery={onQuery} />
      </div>
      {scope !== null && (
        <p className="text-sm text-ash">
          {`Only what ${familyName(scope)} can set for itself. The rest is in `}
          <button type="button" onClick={() => onScope(null)} className="text-ember hover:underline">
            All models
          </button>
          .
        </p>
      )}
      <div className="lg:grid lg:grid-cols-[11rem_minmax(0,1fr)] lg:items-start lg:gap-6">
        <nav aria-label="Configuration groups" className="sticky top-4 hidden lg:block">
          <ul className="flex flex-col">
            {groups.map((group) => {
              const count = Object.values(changed).filter((entry) => entry.group === group).length;
              return (
                <li key={group}>
                  {/* A group the search finds nothing in stays, dimmed, so the list does not jump; its changes still count. */}
                  <button
                    type="button"
                    disabled={!holding.has(group)}
                    onClick={() => jump(group)}
                    className="flex w-full items-baseline justify-between gap-2 px-3 py-1.5 text-left text-sm text-ash enabled:hover:bg-surface enabled:hover:text-ink disabled:opacity-40"
                  >
                    {groupTitle(group)}
                    {count > 0 && <span className="text-xs text-ember tabular-nums">{count}</span>}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>
        <div className="cut min-w-0 bg-surface [--cut-size:14px]">
          {query.trim() !== "" && found.length === 0 && (
            <p className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-6 text-sm text-ash md:px-5">
              <span>No field matches “{query.trim()}”.</span>
              <button type="button" onClick={() => onQuery("")} className="text-ember hover:underline">
                Clear search
              </button>
            </p>
          )}
          {groups.filter((group) => holding.has(group)).map((group) => (
            <section
              key={group}
              id={`server-group-${group}`}
              aria-labelledby={`server-group-${group}-title`}
              className="scroll-mt-4 border-t border-line first:border-t-0"
            >
              <h3 id={`server-group-${group}-title`} className="flex items-baseline gap-2 px-4 pt-5 pb-2 md:px-5">
                <span className="font-display text-[17px] font-semibold">{groupTitle(group)}</span>
                <code className="font-mono text-xs text-ash">{group}</code>
              </h3>
              {found
                .filter(({ decl }) => decl.group === group)
                .map(({ decl, match }) => (
                  <FieldRow
                    key={decl.file_key}
                    decl={decl}
                    match={match}
                    config={config.value}
                    origins={origins.state === "ok" ? origins.value : null}
                    scope={scope}
                    draft={draft}
                    marked={marked.includes(decl.file_key)}
                    {...edits}
                  />
                ))}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

/** The search over the fields, in the look of the Transcribe view's: the count inside it, Escape or ✕ to clear. */
function SearchBox({ query, count, onQuery }: { query: string; count: number; onQuery: (query: string) => void }) {
  const searching = query.trim() !== "";
  return (
    <div
      role="search"
      className={`flex min-h-[34px] items-center gap-1.5 rounded-[2px] border bg-surface pr-1 pl-2.5 sm:w-72 sm:shrink-0 ${searching ? "border-ember/70" : "border-line"} focus-within:border-ember`}
    >
      <span className="text-ash" aria-hidden="true">
        <IconSearch />
      </span>
      <input
        type="search"
        name="server-search"
        aria-label="Search the configuration"
        value={query}
        placeholder="Find a field"
        autoComplete="off"
        spellCheck={false}
        onChange={(e) => onQuery(e.currentTarget.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") onQuery("");
        }}
        className="min-w-0 flex-1 bg-transparent py-1.5 text-sm text-ink placeholder:text-ash focus:outline-none [&::-webkit-search-cancel-button]:hidden"
      />
      <span aria-live="polite" className={`shrink-0 font-display text-xs tabular-nums ${count === 0 ? "text-fault" : "text-ash"}`}>
        {searching ? (count === 0 ? "None" : count === 1 ? "1 field" : `${count} fields`) : ""}
      </span>
      {searching && (
        <button type="button" aria-label="Clear the search" onClick={() => onQuery("")} className="grid size-6 place-items-center text-ash hover:text-ink">
          <IconClose />
        </button>
      )}
    </div>
  );
}

/** `text` with the spans a search matched marked. */
function Marked({ text, marks }: { text: string; marks: [number, number][] }) {
  const parts: ReactNode[] = [];
  let at = 0;
  for (const [from, to] of marks) {
    parts.push(text.slice(at, from), <mark key={from} className="bg-ember/20 text-ink">{text.slice(from, to)}</mark>);
    at = to;
  }
  parts.push(text.slice(at));
  return <>{parts}</>;
}

function Notice({ children }: { children: ReactNode }) {
  return <p className="cut bg-surface px-4 py-3 text-sm text-ash [--cut-size:10px]">{children}</p>;
}

function FieldRow({
  decl,
  match,
  config,
  origins,
  scope,
  draft,
  marked,
  onEdit,
  onRemove,
  onUndo,
}: {
  decl: FieldDecl;
  /** How the search found it: the name's spans to mark, and what to show of why. */
  match: FieldMatch;
  config: ConfigDocument;
  origins: OriginRow[] | null;
  /** The scope the form shows: a family's scope name, or `null` for "All models". */
  scope: string | null;
  draft: Draft;
  marked: boolean;
} & RowEdits) {
  const view = rowView(decl, config, origins, scope);
  const { mode, inForce } = view;
  const entry = draft[draftKey(decl.group, decl.name, inForce.scope)];
  // A removal on its way out shows the shared value the family returns to.
  const shown = entry ? (entry.removal ? view.shared : entry.value) : inForce.value;
  const id = `server-field-${decl.file_key}`;
  const nameId = `${id}-name`;
  const notes = mode.kind === "edit" ? noteLines(view, entry) : [];
  return (
    <div
      className={`relative grid gap-x-6 gap-y-2 border-t border-line px-4 py-3.5 md:grid-cols-[minmax(0,1fr)_minmax(0,17rem)] md:px-5 ${marked ? "bg-fault/5" : ""}`}
    >
      {(entry || marked) && <span className={`absolute inset-y-0 left-0 w-[3px] ${marked ? "bg-fault" : "bg-ember"}`} aria-hidden />}
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span id={nameId} className="font-mono text-[13px] font-semibold break-all text-ink">
            <Marked text={decl.name} marks={match.marks} />
          </span>
          {decl.reload_required && mode.kind === "edit" && <ReloadMark />}
        </div>
        {match.ids && (
          <p className="mt-0.5 flex flex-wrap gap-x-3 font-mono text-xs break-all text-ash">
            <span>{decl.flag}</span>
            <span>{decl.env}</span>
          </p>
        )}
        <Description text={decl.description} opens={match.opens} />
        {notes.length > 0 && (
          <ul className="mt-1 flex flex-col gap-0.5 text-xs text-ash">
            {notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        )}
        {mode.kind === "edit" && inForce.scope !== null && canUseShared(view, entry) && (
          <button type="button" onClick={() => onRemove(decl, inForce.scope!)} className="mt-1 text-xs text-ember hover:underline">
            Use shared value ({labelOf(view.shared)})
          </button>
        )}
        {mode.kind === "edit" && inForce.scope !== null && entry?.removal && (
          <button type="button" onClick={() => onUndo(decl, inForce.scope!)} className="mt-1 text-xs text-ember hover:underline">
            Keep own value
          </button>
        )}
        {marked && <p className="mt-1 text-xs text-fault">Refused by the server</p>}
      </div>
      <div className="min-w-0">
        {mode.kind === "edit" && (
          <Control decl={decl} id={id} nameId={nameId} value={shown} inForce={entry?.removal ? view.shared : inForce.value} marked={marked} onEdit={(value) => onEdit(decl, inForce, value)} />
        )}
        {mode.kind === "fixed" && <ReadOnly value={shownOf(inForce.value)} note="Set at start" />}
        {mode.kind === "list" && <KnownModels value={inForce.value} />}
      </div>
    </div>
  );
}

/** The heat of the header's bar, small: a change here reloads the model. */
function ReloadMark() {
  return (
    <span className="flex items-center gap-1.5 text-xs text-ash" title="A change to this field reloads the model">
      <span className="cut h-2 w-5 [--cut-size:3px]" style={{ background: "var(--heat)" }} aria-hidden />
      reload
    </span>
  );
}

/** The first sentence, the rest behind "More"; open from the start when the search found its word there (`opens`), until toggled. */
function Description({ text, opens }: { text: string; opens: boolean }) {
  const [toggled, setOpen] = useState<boolean | null>(null);
  const open = toggled ?? opens;
  const [first, rest] = firstSentence(text);
  return (
    <p className="mt-1 text-sm text-ash">
      <Prose text={first} />
      {rest && open && (
        <>
          {" "}
          <Prose text={rest} />
        </>
      )}
      {rest && (
        <button type="button" aria-expanded={open} onClick={() => setOpen(!open)} className="ml-1.5 text-ember hover:underline">
          {open ? "Less" : "More"}
        </button>
      )}
    </p>
  );
}

/** A description's `code` spans, in the mono face. */
function Prose({ text }: { text: string }) {
  return (
    <>
      {text.split("`").map((part, i) =>
        i % 2 === 1 ? (
          <code key={i} className="font-mono text-[0.9em] text-ink">
            {part}
          </code>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}

function ReadOnly({ value, note }: { value: string; note: string }) {
  return (
    <div className="md:text-right">
      <p className="font-mono text-[13px] break-all text-ink">{value}</p>
      <p className="text-xs text-ash">{note}</p>
    </div>
  );
}

/** `switch.known_models`, read-only: a patch only adds to it. */
function KnownModels({ value }: { value: unknown }) {
  const entries = value !== null && typeof value === "object" ? Object.entries(value as Record<string, unknown>) : [];
  return (
    <div>
      {entries.length === 0 ? (
        <p className="text-sm text-ash">None</p>
      ) : (
        <ul className="flex flex-col gap-1 font-mono text-[12px]">
          {entries.map(([id, path]) => (
            <li key={id} className="min-w-0">
              <span className="text-ink">{id}</span> <span className="break-all text-ash">{textOf(path)}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-1 text-xs text-ash">Read-only here</p>
    </div>
  );
}

function Control({
  decl,
  id,
  nameId,
  value,
  inForce,
  marked,
  onEdit,
}: {
  decl: FieldDecl;
  id: string;
  nameId: string;
  /** The draft's value, else the one in force. */
  value: unknown;
  inForce: unknown;
  marked: boolean;
  onEdit: (value: string | boolean) => void;
}) {
  const control = controlFor(decl, inForce);
  const chosen = (options: string[]) => options.find((option) => option.toLowerCase() === textOf(value).toLowerCase()) ?? null;
  switch (control.kind) {
    case "switch":
      return (
        <div className="md:flex md:justify-end">
          <Switch on={value === true} onChange={onEdit} labelledBy={nameId} />
        </div>
      );
    case "segmented":
      return (
        <Segmented
          legend={decl.name}
          legendHidden
          name={decl.file_key}
          value={chosen(control.options)}
          options={control.options.map((option) => ({ value: option, label: option }))}
          onChange={onEdit}
        />
      );
    case "select": {
      const current = chosen(control.options);
      return (
        <select
          id={id}
          name={decl.file_key}
          aria-labelledby={nameId}
          aria-invalid={marked || undefined}
          value={current ?? ""}
          onChange={(e) => onEdit(e.target.value)}
          // A variant, not a second border colour: of two base utilities Tailwind's order wins, not the class string's.
          className={`w-full ${fieldLook} aria-invalid:border-fault`}
        >
          {current === null && (
            <option value="" disabled>
              auto
            </option>
          )}
          {control.options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      );
    }
    case "text":
      return (
        <div>
          <input
            id={id}
            name={decl.file_key}
            aria-labelledby={nameId}
            aria-describedby={`${id}-hint`}
            aria-invalid={marked || undefined}
            autoComplete="off"
            spellCheck={false}
            value={textOf(value)}
            placeholder={control.placeholder}
            onChange={(e) => onEdit(e.target.value)}
            className={`${field} font-mono aria-invalid:border-fault`}
          />
          <p id={`${id}-hint`} className="mt-1 text-xs text-ash">
            {control.hint}
          </p>
        </div>
      );
  }
}
