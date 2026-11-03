import { authHeaders, keyRequired } from "../api/auth.ts";
import type { ModelKind } from "../api/model.ts";

// The routes the Server view talks to (spec playground/05, GitHub #316),
// beside GET /v1/models (api/model.ts): what each config field is, the
// running configuration and a change to it, the models a switch can go to,
// and the switch itself, by id. Each answer is read into a value or the
// server's own message; a 401 shows the key prompt, as everywhere else.

/** The rule a value must meet, as `GET /v1/config/fields` structures it; `rule_text` says it in words. */
export type Rule = null | { min: number | null; max: number | null } | { multiple_of: number } | { one_of: string[] };

/** One visible field's declaration (ADR 0046's `FieldMeta`). */
export type FieldDecl = {
  group: string;
  name: string;
  /** `<group>.<name>`, as the config file nests it and a `400` names it. */
  file_key: string;
  flag: string;
  env: string;
  /** `bool`, `tokens`, `bytes|tokens`, … */
  kind: string;
  /** The field may be unset: `null`, shown as "auto". */
  nullable: boolean;
  default: unknown;
  rule: Rule;
  /** The rule as `help --fields` prints it; empty with none. */
  rule_text: string;
  /** `null` for every family, else the scope names (`qwen38`, `qwen38flashnext`) that take it. */
  families: string[] | null;
  /** It may be given per family, in a `<group>.<family>` section. */
  scoped: boolean;
  patchable: boolean;
  reload_required: boolean;
  description: string;
};

/** A config document in the file's own shape: `<group>: { <field>: value, <family>: { <field>: value } }`. */
export type ConfigDocument = Record<string, Record<string, unknown>>;

/**
 * Where one value of one field comes from (`GET /v1/config/origins`, spec
 * playground/05 §Follow-up 2): a row per visible field and scope, `null` for
 * the shared value, else a family. `source` is the source of the value that
 * resolves there, `scoped` that it was set for this family specifically, and
 * `spelling` where, as the startup source log spells it: a flag, an env var,
 * a file key and path, or a profile's name.
 */
export type OriginRow = {
  /** `<group>.<name>`, a declaration's `file_key`. */
  key: string;
  scope: string | null;
  source: "default" | "profile" | "file" | "env" | "flag" | "live";
  scoped: boolean;
  spelling?: string;
};

/** One model the server can switch to or fetch (`GET /v1/models/catalog`). No path: the page switches by id. */
export type CatalogRow = {
  id: string;
  source: "built-in" | "operator" | "known";
  on_disk: "complete" | "partial" | "absent";
  /** An entry's total, or a known model's artifact on disk; absent otherwise. */
  bytes?: number;
  /** The scope name the artifact's header names, when it is on disk; the running family for the loaded row. */
  family?: string;
  /** The model kind (GitHub #448): the artifact header's when it is on disk, else what the catalog tells of the entry; absent when neither tells. An older server sends none. */
  kind?: ModelKind | "diarizer";
  /** The license the entry's weights are published under (ADR 0078): its identifier and a link to its text. Absent for a known model, for an operator entry that states none, and from an older server. */
  license?: { id: string; url: string };
  loaded: boolean;
  switchable: boolean;
  /** `false` when this card's class is below the entry's `card_class` (ADR 0064): no switch to it, no download of it. An older server sends neither field. */
  available?: boolean;
  /** Why it is not available, naming both classes; only with `available: false`. */
  reason?: string;
  /** Its download through `POST /v1/models/download`: running now, or the last one failed. An older server sends none of the three. */
  download?: "none" | "running" | "failed";
  /** Why the last download failed; only with `download: "failed"`. */
  download_reason?: string;
  /** What of it is on disk now, partial files included. */
  bytes_on_disk?: number;
  /** The id the server started on, loaded from this entry's files under download.path whatever id serves it (GitHub #380): removed, the next start looks for it again. */
  start_model?: string;
  /** The ids `switch.known_models` names at this entry's files, the start model's apart: removed, a switch to any of them is refused until it is downloaded again. */
  known_models?: string[];
};

/** Why the card cannot load a row (ADR 0064), as the server words it; null when it can. The one wording every view shows. */
export const unavailableReason = (row: CatalogRow): string | null => (row.available === false ? (row.reason ?? "Too large for this card") : null);

/** A `PATCH /v1/config` that was applied (`200`) or began a reload (`202`). */
export type PatchApplied = {
  applied: "live" | "reload";
  persisted: boolean;
  path?: string;
  reason?: string;
  /** `<key> (set by <spelling>)`: a flag or an env var outranks the file at the next start. */
  shadowed_at_restart?: string[];
  switching?: { from: string; to: string };
  /** The configuration now running, after a live change. */
  config?: ConfigDocument;
};

/** A refusal: the HTTP status, and the server's own `error.message` and `error.code`. */
export type Refusal = { status: number; message: string; code?: string };

export type Answer<T> = { ok: true; status: number; value: T } | ({ ok: false } & Refusal);

/** The server's `{"error": {message, code}}`, or the raw body, or the status alone. */
export function refusalOf(status: number, body: string): Refusal {
  try {
    const error = (JSON.parse(body) as { error?: { message?: unknown; code?: unknown } }).error;
    if (typeof error?.message === "string" && error.message) {
      return { status, message: error.message, ...(typeof error.code === "string" ? { code: error.code } : {}) };
    }
  } catch {
    // Not JSON: the raw body below.
  }
  const text = body.trim();
  return { status, message: text || `${status}` };
}

async function callText<T>(path: string, init: RequestInit, read: (text: string) => T, fetcher: typeof fetch): Promise<Answer<T>> {
  try {
    const res = await fetcher(path, { ...init, headers: { ...authHeaders(), ...(init.headers as Record<string, string> | undefined) } });
    if (res.status === 401) keyRequired();
    const text = await res.text();
    if (!res.ok) return { ok: false, ...refusalOf(res.status, text) };
    return { ok: true, status: res.status, value: read(text) };
  } catch (err) {
    return { ok: false, status: 0, message: `Could not reach ignis: ${String(err)}` };
  }
}

const call = <T,>(path: string, init: RequestInit, read: (body: unknown) => T, fetcher: typeof fetch) =>
  callText(path, init, (text) => read(JSON.parse(text)), fetcher);

/** The list envelope both new routes answer with, as `GET /v1/models` does. */
const listData = <T,>(body: unknown) => ((body as { data?: T[] }).data ?? []) as T[];

export const readFields = (fetcher: typeof fetch = fetch) => call("/v1/config/fields", {}, listData<FieldDecl>, fetcher);

export const readConfig = (fetcher: typeof fetch = fetch) => call("/v1/config", {}, (body) => body as ConfigDocument, fetcher);

/** Where each value of the configuration in force comes from. */
export const readOrigins = (fetcher: typeof fetch = fetch) => call("/v1/config/origins", {}, listData<OriginRow>, fetcher);

/** The configuration in force as the file it amounts to (`?format=yaml`), as text. */
export const readConfigFile = (fetcher: typeof fetch = fetch) => callText("/v1/config?format=yaml", {}, (text) => text, fetcher);

/** Where the server downloads models, and the bytes free to it on that volume (GitHub #453). */
export type CatalogDisk = { path: string; free_bytes: number };

/** The catalog's rows, with the envelope's `disk` beside them: absent from an older server, or when it cannot be read. */
export type CatalogList = CatalogRow[] & { disk?: CatalogDisk };

const catalogList = (body: unknown): CatalogList => {
  const disk = (body as { disk?: CatalogDisk }).disk;
  return Object.assign(listData<CatalogRow>(body), disk ? { disk } : {});
};

export const readCatalog = (fetcher: typeof fetch = fetch) => call("/v1/models/catalog", {}, catalogList, fetcher);

/** Switch to `id` by id alone: the server resolves it to a known model or a catalog entry on disk. */
export const switchTo = (id: string, fetcher: typeof fetch = fetch) =>
  call(
    "/v1/models/switch",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: id }) },
    (body) => body as { from: string; to: string },
    fetcher,
  );

/** Download the catalog entry `id` onto the server, in the background: GET /v1/models/catalog follows it. */
export const startDownload = (id: string, fetcher: typeof fetch = fetch) =>
  call(
    "/v1/models/download",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: id }) },
    (body) => body as { model: string; bytes: number; bytes_on_disk: number },
    fetcher,
  );

/** Stop the running download (GitHub #377); what it fetched stays, for its Resume. */
export const cancelDownload = (fetcher: typeof fetch = fetch) =>
  call(
    "/v1/models/download/cancel",
    { method: "POST" },
    (body) => body as { model: string; bytes: number; bytes_on_disk: number },
    fetcher,
  );

export const patchConfig = (body: object, fetcher: typeof fetch = fetch) =>
  call(
    "/v1/config",
    { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    (answer) => answer as PatchApplied,
    fetcher,
  );

/** A removal that took place (GitHub #380): what went, what it freed, and the files left because another entry lists them. */
export type ModelRemoved = {
  status: string;
  model: string;
  bytes_freed: number;
  files_removed: string[];
  files_kept: { file: string; by: string }[];
};

/** Delete the catalog entry `id`'s files, finished or partial, from the server's models folder; the server answers once they are gone. */
export const removeModel = (id: string, fetcher: typeof fetch = fetch) =>
  call(
    "/v1/models/remove",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: id }) },
    (body) => body as ModelRemoved,
    fetcher,
  );
