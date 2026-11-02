import type { LastSwitch } from "./src/api/model.ts";
import { CATALOG, CONFIG, FIELDS, originRows, PROVENANCE, type Provenance, SCOPES } from "./src/server/fixture.ts";
import { familyName, resolved, section } from "./src/server/form.ts";
import type { CatalogDisk, CatalogRow, ConfigDocument, FieldDecl } from "./src/server/routes.ts";

// A fake of the routes the Server view talks to, for `npm run dev:mock`
// (spec playground/05 §Mock): GET /v1/config/fields, GET and PATCH
// /v1/config, GET /v1/config/origins, GET /v1/models/catalog, POST /v1/models/switch and the model
// state GET /v1/models reports. Development only — never part of the build.
//
// It starts as the 27B under the `rtx5090` profile (src/server/fixture.ts). A
// switch, or a PATCH naming a field that needs a reload, reads `switching`
// for three seconds, then `last_switch` says how it ended; meanwhile a PATCH
// gets the gate's 503 and a second switch a 409, as the server answers them.
//
// Triggers: a value "/error" is refused with a 400 naming that field's key;
// a switch to an id containing "fail" reads `failed` for a second (the old
// model being loaded again) and ends `restored` with a reason; one to an id
// containing "dead" ends `failed` for good, the old model gone and nothing
// serving (no `loaded` row, the torn-down model `switchable` again) until a
// switch to a model that loads; a reload whose
// patch holds a value "/fail" ends the same way, the configuration unchanged.
// A patch of `model.thinking_budget` is shadowed at restart, as if the mock
// had started with IGNIS_MODEL_THINKING_BUDGET set.
//
// Per-model values (spec playground/05 §Follow-up): a patch reloads only if a
// `reload_required` field resolves differently for the loaded family, so a
// value in the other family's section, or a group key the loaded family
// overrides, is a `200` live. `null` removes: a family's own value (the group's
// applies again) or a group key (back to the field's default). `GET /v1/config`
// shows a family's value only where it differs from the group's, as the
// server does. `?format=yaml` renders the configuration as a YAML document.
//
// Origins (spec playground/05 §Follow-up 2): the 27B's `vram.headroom_bytes`
// is the `rtx5090` profile's and `model.thinking_budget` is set by an env var
// (src/server/fixture.ts); an edit makes a value `live`. A shared edit
// replaces a family's value that is not the operator's own, a `null` cannot
// remove one, and a group-key value for a field only the other family takes
// is refused, as the server refuses it.
//
// Downloads (GitHub #376): POST /v1/models/download runs for ten seconds, its
// row's `bytes_on_disk` growing from what is on disk to the whole entry, one at
// a time. An operator entry fails after a second with a 401's reason, as a
// private repository does without its token. POST /v1/models/download/cancel
// stops it (GitHub #377): what landed stays as a partial entry, the next
// download's starting point. POST /v1/models/remove (GitHub #380) takes an
// entry's files, finished or partial, and refuses the loaded model and the one
// downloading.

const SWITCH_MS = 3_000;
const RESTORE_MS = 1_000;
const DOWNLOAD_MS = 10_000;
const REFUSED_MS = 1_000;
const CONFIG_PATH = "ignis.yaml";
const COUNTS = new Set(["tokens", "seconds", "slots", "percent", "lanes", "MiB", "rows", "tokens|off"]);

/** One value of a patch; `remove` is its `null`. */
type Entry = { decl: FieldDecl; scope: string | null; value: unknown; remove?: true };
type Running = { from: string; to: string; until: number; failure?: string; dead?: string; patch?: Entry[] };
export type MockAnswer = { status: number; body: unknown };

const isObject = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);

const refuse = (status: number, type: string, code: string, message: string): MockAnswer => ({ status, body: { error: { message, type, code } } });

/** The serving gate's `503 model_switching` while a switch from `from` to `to` runs, as the server words it. */
const switching = ({ from, to }: { from: string; to: string }): MockAnswer =>
  refuse(503, "server_error", "model_switching", `the server is switching models from ${from} to ${to}; retry once GET /v1/models reports \`serving\``);

const keyOf = (entry: Entry) => (entry.scope === null ? entry.decl.file_key : `${entry.decl.group}.${entry.scope}.${entry.decl.name}`);

/** A value as the config file writes it back: a bool, a count, a choice in its own spelling, else the text. */
function canonical(decl: FieldDecl, value: unknown): unknown {
  if (typeof value !== "string") return value;
  const text = value.trim();
  if (decl.kind === "bool") {
    const lower = text.toLowerCase();
    if (["true", "on", "1"].includes(lower)) return true;
    if (["false", "off", "0"].includes(lower)) return false;
  }
  if (decl.rule && "one_of" in decl.rule) return decl.rule.one_of.find((name) => name.toLowerCase() === text.toLowerCase()) ?? text;
  if (COUNTS.has(decl.kind) && /^\d+$/.test(text)) return Number(text);
  return text;
}

const sameValue = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** The configuration with every family value, also one equal to the shared one, and where each value was set. */
type State = { store: ConfigDocument; from: Provenance };

const provenanceKey = (decl: FieldDecl, scope: string | null) => (scope === null ? decl.file_key : `${decl.group}.${scope}.${decl.name}`);

/** What `GET /v1/config` answers: a family's value only where it differs from the group's. */
function viewOf(store: ConfigDocument): ConfigDocument {
  const doc = structuredClone(store);
  for (const decl of FIELDS.filter((d) => d.scoped)) {
    for (const scope of SCOPES) {
      const own = section(doc, decl.group, scope);
      if (own && decl.name in own && sameValue(own[decl.name], doc[decl.group][decl.name])) delete own[decl.name];
    }
  }
  for (const [group, fields] of Object.entries(doc)) {
    for (const scope of SCOPES) if (section(doc, group, scope) && Object.keys(fields[scope] as object).length === 0) delete fields[scope];
  }
  return doc;
}

const plain = (text: string) => /^[A-Za-z0-9_./~-][A-Za-z0-9_./:~-]*$/.test(text) && !/^(true|false|null|yes|no|on|off|~|[-+]?[0-9.]+)$/i.test(text);

const scalar = (value: unknown) => (typeof value === "string" ? (plain(value) ? value : JSON.stringify(value)) : JSON.stringify(value ?? null));

/** A configuration document as YAML, two spaces to a level. */
export function yamlOf(value: Record<string, unknown>, depth = 0): string {
  const pad = "  ".repeat(depth);
  return Object.entries(value)
    .map(([key, entry]) => {
      if (!isObject(entry)) return `${pad}${key}: ${scalar(entry)}\n`;
      return Object.keys(entry).length === 0 ? `${pad}${key}: {}\n` : `${pad}${key}:\n${yamlOf(entry, depth + 1)}`;
    })
    .join("");
}

/** Why the mock's speech model cannot serve, in the server's words (speech/mod.rs): the Playground's mic is struck. */
export const SPEECH_REASON = "no speech model files for `qwen3-asr-1.7b` under download.path; fetch them with `ignis-server model download qwen3-asr-1.7b`";

/** Why the mock's diarizer cannot serve, in the server's words (speech/diarizing.rs): the Server view shows it (spec speech/08). */
export const DIARIZER_REASON = "no diarizer files for `nemotron-3-diarization` under download.path; fetch them with `ignis-server model download nemotron-3-diarization`";

export function createServerSim() {
  const state: State = { store: structuredClone(CONFIG) as ConfigDocument, from: structuredClone(PROVENANCE) };
  let loaded = "qwen3.8-27b";
  let running: Running | null = null;
  // A target that did not load: `failed` while the old model is loaded again.
  let restoring: { reason: string; until: number; last: LastSwitch } | null = null;
  let last: LastSwitch | undefined;
  // A switch that left nothing serving: `loaded` is then only the model torn down.
  let dead: { reason: string } | null = null;
  // The download under way, the entries one fetched, and why an entry's last one failed.
  let download: { id: string; from: number; total: number; since: number; until: number; fails: boolean } | null = null;
  const landed = new Set<string>();
  const failures = new Map<string, string>();
  // What a cancelled download left on disk, by entry (GitHub #377).
  const partials = new Map<string, number>();
  // The entries a removal took, until a download lands them again (GitHub #380).
  const removed = new Set<string>();
  const onDisk = (row: CatalogRow) => (row.on_disk === "complete" && !removed.has(row.id)) || landed.has(row.id);

  const familyOf = (id: string) => CATALOG.find((row) => row.id === id)?.family ?? "qwen38";

  /** A row's download as the catalog reads it at `now`: running with its bytes, fetched, or failed with its reason. */
  function downloadOf(row: CatalogRow, now: number): Partial<CatalogRow> {
    if (download?.id === row.id) {
      const done = download.fails ? 0 : Math.min(1, (now - download.since) / DOWNLOAD_MS);
      return { download: "running", bytes_on_disk: Math.round(download.from + (download.total - download.from) * done) };
    }
    if (landed.has(row.id)) return { on_disk: "complete", bytes_on_disk: row.bytes };
    const kept: Partial<CatalogRow> = partials.has(row.id)
      ? { on_disk: "partial", bytes_on_disk: partials.get(row.id) }
      : removed.has(row.id)
        ? { on_disk: "absent", bytes_on_disk: 0 }
        : {};
    const reason = failures.get(row.id);
    return reason ? { ...kept, download: "failed", download_reason: reason } : kept;
  }

  function advance(now: number) {
    if (download && now >= download.until) {
      const ended = download;
      download = null;
      if (!ended.fails) landed.add(ended.id);
      else {
        failures.set(
          ended.id,
          `GET https://huggingface.co/acme/${ended.id}/resolve/main/${ended.id}.ninfer: 401 Unauthorized: the endpoint wants a token and none was sent: set IGNIS_DOWNLOAD_TOKEN (download.token), or HF_TOKEN for the server (read at its start) or for \`model download\``,
        );
      }
    }
    if (running && now >= running.until) {
      const ended = running;
      running = null;
      if (ended.dead) {
        dead = { reason: ended.dead };
        last = { from: ended.from, to: ended.to, outcome: "failed", reason: ended.dead };
      } else if (ended.failure) {
        restoring = { reason: ended.failure, until: ended.until + RESTORE_MS, last: { from: ended.from, to: ended.to, outcome: "restored", reason: ended.failure } };
      } else {
        loaded = ended.to;
        dead = null;
        if (ended.patch) apply(ended.patch);
        last = { from: ended.from, to: ended.to, outcome: "switched" };
      }
    }
    if (restoring && now >= restoring.until) {
      last = restoring.last;
      restoring = null;
    }
  }

  const status = () => (running ? "switching" : restoring || dead ? "failed" : "serving");

  /** Whether a family's value of a field is the operator's own: the file's or a live change, which a shared edit does not reach. */
  const ownValue = (into: State, decl: FieldDecl, family: string) => {
    const source = into.from[provenanceKey(decl, family)]?.source ?? "file";
    return source === "file" || source === "live";
  };

  function apply(entries: Entry[], into: State = state) {
    for (const { decl, scope, value, remove } of entries) {
      const group = (into.store[decl.group] ??= {});
      const key = provenanceKey(decl, scope);
      if (remove) {
        // A profile's, a flag's or an env var's value is not in the file, so nothing removes it.
        if (scope === null) group[decl.name] = decl.default;
        else if (ownValue(into, decl, scope)) delete section(into.store, decl.group, scope)?.[decl.name];
        else continue;
        delete into.from[key];
      } else {
        const at = scope === null ? group : ((group[scope] ??= {}) as Record<string, unknown>);
        if (decl.file_key === "switch.known_models" && isObject(value)) at[decl.name] = { ...(at[decl.name] as object), ...value };
        else at[decl.name] = canonical(decl, value);
        into.from[key] = { source: "live" };
        // A patched shared value outranks what a family has from the profile, a flag or an env var.
        if (scope === null) {
          for (const family of SCOPES) {
            if (ownValue(into, decl, family)) continue;
            delete section(into.store, decl.group, family)?.[decl.name];
            delete into.from[provenanceKey(decl, family)];
          }
        }
      }
      if (scope !== null && section(into.store, decl.group, scope) && Object.keys(group[scope] as object).length === 0) delete group[scope];
    }
  }

  /** The changes a PATCH body names, or the 400 the server answers it with. */
  function readPatch(body: unknown, running: string): Entry[] | MockAnswer {
    const invalid = (message: string) => refuse(400, "invalid_request_error", "config_invalid", message);
    if (!isObject(body)) return invalid("the body must be a config document: `{ <group>: { <field>: value } }`");
    const find = (group: string, name: string) => FIELDS.find((decl) => decl.group === group && decl.name === name);
    const entries: Entry[] = [];
    for (const [group, fields] of Object.entries(body)) {
      if (!isObject(fields)) return invalid(`\`${group}\` must be a map of its fields`);
      for (const [key, value] of Object.entries(fields)) {
        if (SCOPES.includes(key) && isObject(value)) {
          for (const [name, scoped] of Object.entries(value)) {
            const decl = find(group, name);
            if (!decl) return invalid(`unknown field \`${group}.${key}.${name}\``);
            if (!decl.scoped && !decl.families?.includes(key)) return invalid(`\`${group}.${key}.${name}\`: \`${group}.${name}\` has no family sections`);
            entries.push({ decl, scope: key, value: scoped, ...(scoped === null ? { remove: true as const } : {}) });
          }
          continue;
        }
        const decl = find(group, key);
        if (!decl) return invalid(`unknown field \`${group}.${key}\``);
        // A field only one family takes lives in that family's section, as the server writes it; a value the running family cannot take is refused.
        const only = decl.families?.length === 1 ? decl.families[0] : null;
        if (only !== null && only !== running && value !== null && !sameValue(canonical(decl, value), decl.default)) {
          return invalid(`\`${decl.file_key} ${String(value)}\`: ${familyName(running)} does not take it (a ${familyName(only)} option)`);
        }
        entries.push({ decl, scope: only, value, ...(value === null ? { remove: true as const } : {}) });
      }
    }
    if (entries.length === 0) return invalid("the patch names no field");
    const fixed = entries.filter((entry) => !entry.decl.patchable).map((entry) => `\`${keyOf(entry)}\``);
    if (fixed.length > 0) {
      return refuse(400, "invalid_request_error", "config_field_not_patchable", `${fixed.join(", ")} cannot be changed while the server runs (not patchable); nothing in the patch was applied`);
    }
    const refused = entries.find((entry) => entry.value === "/error");
    if (refused) return invalid(`\`${keyOf(refused)}\` expects a ${refused.decl.kind} value, got \`/error\` (the mock refuses "/error"); nothing in the patch was applied`);
    return entries;
  }

  // The kind a family's own id names (`ModelFamily::of_model_id`): all the
  // server says of a model it has not loaded, since it reads no header there.
  const FAMILY_KINDS: Record<string, string> = { "qwen3.8-27b": "text", "qwen3.8-flash-next": "text", "ignis-image-2.1-turbo": "image" };
  // The servable rest of GET /v1/models (spec model-switch/03): every row on
  // disk but the loaded one, in catalog order.
  const servable = () =>
    CATALOG.filter((row) => onDisk(row) && row.id !== loaded).map((row) => ({
      id: row.id,
      object: "model",
      owned_by: "ignis",
      active: false,
      ...(FAMILY_KINDS[row.id] ? { kind: FAMILY_KINDS[row.id] } : {}),
    }));

  return {
    fields: () => ({ object: "list" as const, data: FIELDS }),

    /** GET /v1/config: the running configuration, and the model status its header carries. */
    config(now: number): { body: ConfigDocument; status: string } {
      advance(now);
      return { body: viewOf(state.store), status: status() };
    },

    /** GET /v1/config?format=...: `json` (or none) is the document, `yaml` the file it amounts to, any other format a 400. */
    configAs(format: string | null, now: number): { modelStatus: string; http: number; type: string; text: string } {
      const { body, status } = this.config(now);
      if (format === null || format === "json") return { modelStatus: status, http: 200, type: "application/json", text: JSON.stringify(body) };
      if (format === "yaml") return { modelStatus: status, http: 200, type: "application/yaml", text: yamlOf(body) };
      const refused = refuse(400, "invalid_request_error", "config_invalid", `unknown format \`${format}\`: use \`json\` or \`yaml\``);
      return { modelStatus: status, http: 400, type: "application/json", text: JSON.stringify(refused.body) };
    },

    /** GET /v1/config/origins: where each value in force comes from. */
    origins(now: number): { object: "list"; data: ReturnType<typeof originRows> } {
      advance(now);
      return { object: "list", data: originRows(FIELDS, state.store, state.from) };
    },

    /** GET /v1/models: the loaded model, then the servable rest (spec model-switch/03). */
    models(now: number): Record<string, unknown> {
      advance(now);
      const family = familyOf(loaded);
      const context = Number((state.store.model?.[family] as Record<string, unknown> | undefined)?.max_context ?? state.store.model?.max_context);
      return {
        object: "list",
        data: dead ? [] : [{ id: loaded, object: "model", owned_by: "ignis", active: true, kind: "text", max_model_len: context }, ...servable()],
        status: status(),
        ...(running ? { switching: { from: running.from, to: running.to } } : {}),
        ...(restoring || dead ? { reason: (restoring ?? dead)!.reason } : {}),
        ...(last ? { last_switch: last } : {}),
        // The mock has no Realtime socket: its speech model is one whose files are not on disk (spec speech/02), and
        // so is its diarizer (spec speech/08).
        speech: {
          available: false,
          reason: SPEECH_REASON,
          diarizer: { id: "nemotron-3-diarization", loaded: false, available: false, reason: DIARIZER_REASON },
        },
      };
    },

    catalog(now: number): { object: "list"; data: CatalogRow[]; disk: CatalogDisk } {
      advance(now);
      return {
        object: "list",
        data: CATALOG.map((row) => {
          const isLoaded = !dead && row.id === loaded;
          // A switch can begin while serving or after a failed one, never during one.
          return { ...row, loaded: isLoaded, switchable: !running && !isLoaded && onDisk(row), ...downloadOf(row, now) };
        }),
        // GitHub #453: the room left under download.path, a made-up 412 GiB.
        disk: { path: String(CONFIG.download.path), free_bytes: 412 * 1024 ** 3 },
      };
    },

    patch(body: unknown, now: number): MockAnswer {
      advance(now);
      if (running) {
        return switching(running);
      }
      const unready = restoring ?? dead;
      if (unready) return refuse(503, "server_error", "server_not_ready", `a model switch failed and no model is serving (${unready.reason}); retry shortly`);
      const entries = readPatch(body, familyOf(loaded));
      if (!Array.isArray(entries)) return entries;
      const shadowed = entries.filter((entry) => keyOf(entry) === "model.thinking_budget").map((entry) => `${keyOf(entry)} (set by IGNIS_MODEL_THINKING_BUDGET)`);
      const shadowing = shadowed.length > 0 ? { shadowed_at_restart: shadowed } : {};
      // A reload only for what the loaded model runs with: apply the patch to a copy and compare what the loaded family resolves.
      const family = familyOf(loaded);
      const after = structuredClone(state);
      apply(entries, after);
      const reloads = entries.some((entry) => entry.decl.reload_required && !sameValue(resolved(state.store, entry.decl, family), resolved(after.store, entry.decl, family)));
      if (reloads) {
        const failure = entries.some((entry) => entry.value === "/fail")
          ? `the reload of ${loaded} failed: the mock refuses a value "/fail"; the previous configuration was loaded again`
          : undefined;
        running = { from: loaded, to: loaded, until: now + SWITCH_MS, patch: entries, failure };
        return {
          status: 202,
          body: {
            applied: "reload",
            persisted: false,
            path: CONFIG_PATH,
            reason: "written once the reloaded model serves; not if the reload fails",
            ...shadowing,
            switching: { from: loaded, to: loaded },
          },
        };
      }
      apply(entries);
      return { status: 200, body: { applied: "live", persisted: true, path: CONFIG_PATH, ...shadowing, config: viewOf(state.store) } };
    },

    /** POST /v1/models/switch by id alone (an `artifact` beside it is ignored here). */
    switchTo(body: unknown, now: number): MockAnswer {
      advance(now);
      if (running) {
        return refuse(409, "invalid_request_error", "switch_in_progress", `a switch from ${running.from} to ${running.to} is already under way; switches are not queued`);
      }
      const id = isObject(body) && typeof body.model === "string" ? body.model.trim() : "";
      if (!id) return refuse(400, "invalid_request_error", "invalid_request_error", "`model` must name the id to serve it under");
      const row = CATALOG.find((r) => r.id === id);
      if (!row) {
        return refuse(404, "model_not_found", "model_not_found", `unknown model: ${id} is neither in switch.known_models nor a catalog entry; GET /v1/models/catalog lists the models this server can switch to`);
      }
      if (!onDisk(row)) {
        return refuse(
          404,
          "model_not_found",
          "model_not_found",
          row.source === "known"
            ? `model ${id} is in switch.known_models but its artifact is not a file; fix the path or put the file there`
            : `model ${id} is in the catalog but not on disk (download.path); fetch it with \`ignis-server model download ${id}\`, a switch never starts a download`,
        );
      }
      const failure = id.includes("fail") ? `loading ${id} failed: the mock refuses every id containing "fail"; ${loaded} was loaded again` : undefined;
      const gone = id.includes("dead") ? `loading ${id} failed and ${loaded} could not be loaded again: the mock leaves nothing serving for every id containing "dead"` : undefined;
      running = { from: loaded, to: id, until: now + SWITCH_MS, failure: gone ? undefined : failure, dead: gone };
      restoring = null;
      return { status: 202, body: { status: "switching", from: loaded, to: id } };
    },

    /** POST /v1/models/download (GitHub #376): a catalog entry not on disk, in the background, one at a time. */
    downloadModel(body: unknown, now: number): MockAnswer {
      advance(now);
      if (running) {
        return switching(running);
      }
      const id = isObject(body) && typeof body.model === "string" ? body.model.trim() : "";
      if (!id) return refuse(400, "invalid_request_error", "invalid_request_error", "`model` must name the catalog entry to download");
      const row = CATALOG.find((r) => r.id === id && r.source !== "known");
      if (!row) {
        return refuse(404, "model_not_found", "model_not_found", `unknown model: ${id} is not a catalog entry, and only catalog entries are downloaded; GET /v1/models/catalog lists them`);
      }
      if (onDisk(row)) {
        return refuse(409, "invalid_request_error", "model_already_downloaded", `${id} is already on disk under download.path; POST /v1/models/switch with it loads it`);
      }
      if (download) {
        return refuse(409, "invalid_request_error", "download_in_progress", `a download of ${download.id} is already running; downloads are one at a time, never queued`);
      }
      const fails = row.source === "operator";
      const from = partials.get(id) ?? (removed.has(id) ? 0 : (row.bytes_on_disk ?? 0));
      failures.delete(id);
      download = { id, from, total: row.bytes ?? 0, since: now, until: now + (fails ? REFUSED_MS : DOWNLOAD_MS), fails };
      return { status: 202, body: { status: "downloading", model: id, bytes: row.bytes ?? 0, bytes_on_disk: from } };
    },

    /** POST /v1/models/download/cancel (GitHub #377): stop the running download; what landed stays, partial. Answered during a switch too. */
    cancelDownload(now: number): MockAnswer {
      advance(now);
      const row = download && CATALOG.find((r) => r.id === download!.id);
      if (!download || !row) {
        return refuse(409, "invalid_request_error", "download_not_running", "no download is running; GET /v1/models/catalog shows each entry's state");
      }
      const kept = downloadOf(row, now).bytes_on_disk ?? download.from;
      partials.set(row.id, kept);
      download = null;
      return { status: 200, body: { status: "cancelled", model: row.id, bytes: row.bytes ?? 0, bytes_on_disk: kept } };
    },

    /** POST /v1/models/remove (GitHub #380): an entry's files, finished or partial, never the model loaded or the one downloading. */
    removeModel(body: unknown, now: number): MockAnswer {
      advance(now);
      if (running) {
        return switching(running);
      }
      const id = isObject(body) && typeof body.model === "string" ? body.model.trim() : "";
      if (!id) return refuse(400, "invalid_request_error", "invalid_request_error", "`model` must name the catalog entry to remove");
      const row = CATALOG.find((r) => r.id === id && r.source !== "known");
      if (!row) {
        return refuse(404, "model_not_found", "model_not_found", `unknown model: ${id} is not a catalog entry, and only a catalog entry's files are removed; GET /v1/models/catalog lists them`);
      }
      if (id === loaded && !dead) {
        return refuse(409, "invalid_request_error", "model_in_use", `${id} is the model loaded (${loaded}), which serves from its files; load another model before removing it`);
      }
      if (download?.id === id) {
        return refuse(409, "invalid_request_error", "model_in_use", `${id} is being downloaded; cancel its download (POST /v1/models/download/cancel), or let it end, before removing it`);
      }
      const there = downloadOf(row, now).bytes_on_disk ?? row.bytes_on_disk ?? 0;
      if (there === 0) {
        return refuse(409, "invalid_request_error", "model_not_downloaded", `nothing of ${id} is under download.path (./models): there is nothing to remove`);
      }
      removed.add(id);
      landed.delete(id);
      partials.delete(id);
      failures.delete(id);
      return { status: 200, body: { status: "removed", model: id, bytes_freed: there, files_removed: [`${id}.ninfer`], files_kept: [] } };
    },
  };
}
