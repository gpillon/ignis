import { familiesOf } from "./form.ts";
import type { CatalogRow, ConfigDocument, FieldDecl, OriginRow, Rule } from "./routes.ts";

// What the Server view reads, as ignis serves it (spec playground/05): every
// visible field's declaration in declaration order — taken from `ignis-server
// help --fields --format json`, so the kinds, rules, defaults and
// descriptions are the real ones — a running configuration of the 27B under
// the `rtx5090` profile, and a catalog. The tests render the view over these,
// and `npm run dev:mock` serves them.

const dashed = (name: string) => name.replaceAll("_", "-");

/** The rule in words, as `Validator::describe` writes it. */
function ruleText(rule: Rule): string {
  if (rule === null) return "";
  if ("one_of" in rule) return `one of ${rule.one_of.join(", ")}`;
  if ("multiple_of" in rule) return `nonzero multiple of ${rule.multiple_of}`;
  if (rule.min !== null && rule.max !== null) return `${rule.min}..=${rule.max}`;
  if (rule.min !== null) return `>= ${rule.min}`;
  return rule.max !== null ? `<= ${rule.max}` : "";
}

/**
 * A declaration as the server writes it: the flag, env var and file key
 * spelled from the group and the name as `FieldMeta` spells them, and
 * `nullable` for the `Opt<…>` kinds — the fields whose default is unset.
 */
function decl(
  group: string,
  name: string,
  kind: string,
  value: unknown,
  attrs: { rule?: Rule; families?: string[]; scoped?: boolean; patchable?: boolean; reload_required?: boolean },
  description: string,
): FieldDecl {
  const rule = attrs.rule ?? null;
  return {
    group,
    name,
    file_key: `${group}.${name}`,
    flag: `--${dashed(group)}-${dashed(name)}`,
    env: `IGNIS_${group.toUpperCase()}_${name.toUpperCase()}`,
    kind,
    nullable: value === null,
    default: value,
    rule,
    rule_text: ruleText(rule),
    families: attrs.families ?? null,
    scoped: attrs.scoped ?? false,
    patchable: attrs.patchable ?? false,
    reload_required: attrs.reload_required ?? false,
    description,
  };
}

/** Every visible field, in declaration order (the API key and the download token are not). */
export const FIELDS: FieldDecl[] = [
  decl("server", "bind", "text", "127.0.0.1:8000", {},
    "The API's listen address (host:port). Localhost on the OpenAI port by default; never changed at runtime — a rebind is a restart."),
  decl("server", "expose", "expose mode", null, { rule: { one_of: ["cloudflare-quick"] } },
    "Expose the server beyond its bind address: `cloudflare-quick` opens a public https://*.trycloudflare.com URL, printed at start. An exposed server always requires an API key — `auto` when none is set."),
  decl("server", "metrics", "bool", true, {},
    "Serve Prometheus metrics on their own listener (`metrics_bind`), and at /ui/metrics unless the Playground is off. On by default; `false` serves neither."),
  decl("server", "metrics_bind", "text", "127.0.0.1:9464", {},
    "The metrics listener's address: loopback by default, no API key, never exposed, never the API's `bind`; refused with `metrics` off. The default address taken (a second server on the host) is a warning, and the server starts without the listener; another address that cannot be bound refuses the start."),
  decl("server", "ui", "bool", true, {},
    "Serve the Playground at /ui/. A binary built without web/dist serves a page saying how to build it."),
  decl("server", "request_timeout", "seconds", 120, { rule: { min: 1, max: 3600 }, patchable: true },
    "How long a non-streaming request waits for its completion before the handler gives up with a 504, in seconds. A ceiling against a fat-fingered value, not a real operating point."),
  decl("server", "system_message_policy", "policy", "merge", { rule: { one_of: ["merge", "strict"] }, patchable: true },
    "Where `system` messages go: `merge` joins a leading run into the system prompt and keeps a later one as its own block in place; `strict` answers 400 for a system message that is not first."),
  decl("server", "developer_message_policy", "policy", "inplace", { rule: { one_of: ["inplace", "into-system", "after-system", "one-after-system", "reject"] }, patchable: true },
    "Where `developer` messages go: `inplace`, `into-system`, `after-system`, `one-after-system` or `reject`. A leading developer message is the system prompt except under `reject`."),
  decl("model", "id", "text", null, {},
    "The id the model is served under. Unset, a load is served under its own model's id (qwen3.8-27b or qwen3.8-flash-next); without an artifact, qwen3.8-27b. An id naming the other model than the artifact's is refused."),
  decl("model", "artifact", "path", null, {},
    "The .ninfer artifact to load. Unset, the model is looked for under `download.path`, and fetched when it is not there."),
  decl("model", "max_context", "tokens", 40960, { rule: { min: 1, max: null }, scoped: true, patchable: true, reload_required: true },
    "The largest prompt + generation a single request may reserve, in tokens. On the 27B, bounded by its attention's envelope for the KV format."),
  decl("model", "default_max_tokens", "tokens", 38912, { scoped: true, patchable: true, reload_required: true },
    "The max_tokens of a request that sends none, its reasoning included, never past what its prompt leaves of `max_context`; an explicit cap always wins. 0 = none, up to the context."),
  decl("model", "kv_format", "kv format", "hq-e8-2b", { rule: { one_of: ["bf16", "hq-e8-2b"] }, scoped: true, patchable: true, reload_required: true },
    "The KV storage format: `hq-e8-2b` (the serving default) or `bf16` (the retained oracle format)."),
  decl("model", "prefill_chunk", "tokens", 1024, { rule: { multiple_of: 128 }, scoped: true, patchable: true, reload_required: true },
    "The prefill chunk width, in tokens."),
  decl("model", "decode_share", "percent", null, { rule: { min: 0, max: 99 }, scoped: true, patchable: true, reload_required: true },
    "The percent of the model's time decoding lanes keep while a prompt prefills. Unset, the model's own: 25 on both. 0 is one decode round per chunk, 50 splits time evenly."),
  decl("model", "enable_thinking", "bool", true, { patchable: true },
    "Whether a request that says nothing thinks."),
  decl("model", "reasoning_effort", "reasoning effort", null, { rule: { one_of: ["none", "minimal", "low", "medium", "high", "xhigh", "max"] }, patchable: true },
    "The reasoning_effort of a request that names none. Unset, the template's own default."),
  decl("model", "thinking_budget", "tokens|off", 32768, { patchable: true },
    "The reasoning tokens a request may spend before the model's close is forced; `off` = no budget. A request's own thinking_budget overrides it."),
  decl("model", "rope_scaling", "rope scaling", "auto", { scoped: true, patchable: true, reload_required: true },
    "The text rotary table: `auto` (the default), `none` (the linear table) or `yarn:F[,t=..][,bf=..][,bs=..]`, which rescales the checkpoint's trained 262,144-position envelope — what a context past it needs. `auto` decides per model from its `model.max_context`: none up to 262,144, else YaRN at `max_context / 262,144`. Both models run the table."),
  decl("model", "adapt_context_step", "tokens", 65536, { rule: { multiple_of: 64 }, families: ["qwen38"], patchable: true, reload_required: true },
    "The 27B's adaptation (ADR 0063): when its VRAM plan refuses a `max_context` the profile or the default gave, the load lowers it this many tokens at a time, never below this many -- a start whose plan does not hold one step is refused. A whole number of KV pages (64 tokens)."),
  decl("vram", "kv_pool_bytes", "bytes|tokens", null, { scoped: true, patchable: true, reload_required: true },
    "The KV pool: a byte count (K/M/G suffix read) or tokens as `<n>tok`, `<n>Ktok`, `<n>Mtok`. Unset, the KV pool policy's: the rest of the VRAM budget when every weight is on the device (the 27B); 524,288 tokens shared by the lanes when Flash-Next's experts stream, the rest to the expert cache. Refused below one `max_context` sequence and a page per retained slot."),
  decl("vram", "headroom_bytes", "bytes", "1536M", { scoped: true, patchable: true, reload_required: true },
    "What a derived VRAM budget leaves to the desktop and every other process on the card: the budget is the memory free at start less this. Not with `budget_bytes`."),
  decl("vram", "budget_bytes", "bytes", null, { rule: { min: 1, max: null }, scoped: true, patchable: true, reload_required: true },
    "The device memory the whole process may hold, weights included; refused above free memory. Unset, derived from `headroom_bytes`."),
  decl("vram", "allow_oversubscription", "bool", false, { scoped: true, patchable: true, reload_required: true },
    "Start above free memory, with a warning. Needs `budget_bytes`."),
  decl("vram", "allow_expert_cache_below_floor", "bool", false, { families: ["qwen38flashnext"], patchable: true, reload_required: true },
    "Start Flash-Next with an expert cache below its 12 GiB floor, with a warning — decode slows sharply below it."),
  decl("vram", "pinned_expert_bytes", "bytes", 0, { families: ["qwen38flashnext"], patchable: true, reload_required: true },
    "Flash-Next's calibration-hottest experts held in the expert cache for the whole load, outside its LRU, with no copy in the pinned host pool: whole experts, hottest first, until they reach this many bytes. The pool takes that much less RAM; the LRU has those slots fewer (on the 5090: nothing measurable at 1 GiB, ~1% of three-lane decode at 2 GiB). 0 pins none; the `rtx5090` profile pins 2 GiB. A card that holds every expert keeps none in RAM anyway and ignores it."),
  decl("vram", "adapt_spare_tokens", "tokens", 65536, { families: ["qwen38"], patchable: true, reload_required: true },
    "The 27B's adaptation (ADR 0063): at more than one decode lane, a context the load lowers is kept only when the KV pool holds it and a spare reservation of this many tokens, room for a second request; 0, one sequence is enough. At one lane the plan's own floor decides."),
  decl("reuse", "prompt", "bool", true, { scoped: true, patchable: true, reload_required: true },
    "Cross-request state reuse. Off, no prompt checkpoint is captured or claimed, and no prefix is shared unless `retained_device` or `retained_host` gives slots for it."),
  decl("reuse", "retained_device", "slots", null, { scoped: true, patchable: true, reload_required: true },
    "Retained slots in VRAM, reserved in the VRAM plan and handed out first. Unset, 0. With prompt reuse off, a count shares heads between live siblings only."),
  decl("reuse", "retained_host", "slots", null, { scoped: true, patchable: true, reload_required: true },
    "Retained slots in one pinned host block reserved at start, ~222 MiB each, a PCIe copy per capture and per claim. Unset, the model's own: two per decode lane on the 27B, 8 on Flash-Next; 0 with prompt reuse off."),
  decl("reuse", "retained_interactive_ttl", "seconds", 300, { scoped: true, patchable: true, reload_required: true },
    "Idle seconds after which a main-conversation checkpoint in KV-RAM ranks as a subagent's. Needs prompt reuse on."),
  decl("reuse", "kv_host_pool_bytes", "bytes", "2G", { scoped: true, patchable: true, reload_required: true },
    "The KV-RAM host tier's budget: page-locked whole at start and held for the life of the load, so it is RAM the process holds even idle (and the figure Windows reports as its shared GPU memory). 0 disables the tier."),
  decl("switch", "drain_timeout", "seconds", 30, { rule: { min: 0, max: 3600 }, patchable: true },
    "How long a model switch lets the old model's running requests finish before it cancels them, in seconds; 0 cancels them at once."),
  decl("switch", "allow_implicit", "bool", true, { patchable: true },
    "Whether a request whose model names another model `known_models` lists switches the server to it, then is served on it. Off refuses it as an unknown model is. An explicit switch is not affected."),
  decl("switch", "known_models", "id=path;...", {}, { patchable: true },
    "The models a request may switch to by naming them, each with the artifact it loads from: `<id>=<path>` (the flag is repeatable, the env var takes `;`-separated pairs, a file a map). The model the server starts on joins them once its load says its id."),
  decl("spec", "backend", "backend", null, { rule: { one_of: ["dflash2", "mtp", "off"] }, scoped: true, patchable: true, reload_required: true },
    "The speculative backend: `dflash2` (the 27B's drafter), `mtp` (Flash-Next's head, with its companion container beside the artifact), or `off`. Unset, no speculation."),
  decl("spec", "draft_tokens", "tokens", null, { rule: { min: 1, max: 7 }, scoped: true, patchable: true, reload_required: true },
    "The draft window: required with `dflash2`; with `mtp`, the most drafts a lane verifies (default 2). Needs a backend."),
  decl("spec", "draft_rows", "rows", null, { rule: { min: 0, max: 32 }, families: ["qwen38flashnext"], patchable: true, reload_required: true },
    "Flash-Next's draft row budget: the rows a verify round takes across lanes, 0 (= 8) or 2..=32; past 8, five to eight lanes draft too."),
  decl("spec", "decode_lanes", "lanes", null, { rule: { min: 1, max: 8 }, scoped: true, patchable: true, reload_required: true },
    "The decode lanes: the sequences a model decodes at once. On the 27B each lane reserves its state (~233 MB with `dflash2` under hq-e8-2b) and its share of the round buffers, and takes two of the default host retained slots. On Flash-Next they decode sharing the KV pool — min(524,288 tokens, lanes x `max_context`), never below one `max_context` and a page per retained slot. Unset, the model's own: 8 on the 27B, 3 on Flash-Next."),
  decl("spec", "draft_head", "head", null, { rule: { one_of: ["full", "shortlist", "auto"] }, scoped: true, patchable: true, reload_required: true },
    "The drafter's proposal head: `full` (the target's output head), `shortlist` (the artifact's Q4 head over the 131,072 most frequent tokens, ~357 MB more VRAM) or `auto` (`full` on every card class: the shortlist frees no VRAM and measured slower). Needs a backend; `mtp` has only `full`. Unset, `full`."),
  decl("vision", "enabled", "bool", false, { families: ["qwen38"], patchable: true, reload_required: true },
    "Load the vision tower and reserve its workspace."),
  decl("vision", "max_tokens", "tokens", 32768, { rule: { min: 1, max: 1048576 }, families: ["qwen38"], patchable: true, reload_required: true },
    "Merged vision tokens per request. An image over it is shrunk to fit, aspect kept; refused only when its aspect ratio cannot fit, or when several images total over it. Needs vision on."),
  decl("vision", "embedding_pool_mib", "MiB", null, { rule: { min: 1, max: 65536 }, families: ["qwen38"], patchable: true, reload_required: true },
    "Encoded images kept for reuse, in MiB, where `vision.residency` puts the pool: VRAM, or pinned host RAM under `host`. Unset, one envelope-wide embedding; a pool below that is raised to it. Needs vision on."),
  decl("vision", "residency", "vision residency", "host", { rule: { one_of: ["host", "device"] }, families: ["qwen38"], patchable: true, reload_required: true },
    "Where the vision tower's weights and the embedding pool live: `device`, in VRAM for the life of the load (today's encode latency), or `host`, in pinned host RAM, the weights copied to the GPU for each image encode (~300 MB and the pool less VRAM, so more KV pool, a few ms more an encode). A `host` start the host RAM cannot hold is refused, never moved to the device. Needs vision on."),
  decl("media", "allow_private_network", "bool", false, { families: ["qwen38"], patchable: true, reload_required: true },
    "Fetch image URLs on private, loopback, link-local, multicast and CGNAT addresses. Off, so an exposed server cannot be used to probe the operator's LAN. Needs vision on."),
  decl("media", "cache_mib", "MiB", 1024, { rule: { min: 0, max: 65536 }, families: ["qwen38"], patchable: true, reload_required: true },
    "Prepared images kept for reuse, in MiB; 0 keeps none. Needs vision on."),
  decl("download", "enabled", "bool", true, {},
    "Fetch the model when no artifact is on disk — asked first when stdin is a terminal, downloaded straight away when it is not. Off keeps the placeholder template, and refuses POST /v1/models/download too: the server never uses the network."),
  decl("download", "path", "path", "./models", {},
    "Where a fetched model lands, and where one fetched earlier is found."),
  decl("download", "endpoint", "text", "https://huggingface.co", {},
    "Where a fetched model's files come from: Hugging Face, or a mirror answering its {repo}/resolve/{revision}/{file} route (an Artifactory or Nexus proxy, a static server laid out the same way). The pins still say whether the bytes are the published ones."),
  decl("download", "catalog", "path", null, {},
    "An operator catalog: a YAML or JSON file of models fetched beside the built-in ones, in the same format; an id the built-in catalog has is refused. A relative path is read from the directory of the config file that names it."),
  decl("download", "from_ui", "bool", true, {},
    "Let POST /v1/models/download, the Server view's Download, fetch a catalog entry while the server runs, under the same API key as PATCH /v1/config; it needs `enabled` too. Off refuses it, and `model download` on the server's machine still fetches."),
  decl("download", "free_margin", "bytes", "1G", {},
    "What a download must leave free on the volume of `path` beyond the bytes it still misses: the Server view's Download, `model download` and the start's own fetch are refused, before any byte is written, when the free space is below the two together."),
  decl("ngram", "persist", "bool", true, { patchable: true, reload_required: true },
    "Persist Flash-Next's n-gram hot rows between loads."),
  decl("ngram", "persist_path", "model|auto|dir", "model", { patchable: true, reload_required: true },
    "Where the n-gram cache goes: `model` (beside the artifact), `auto` (Windows LOCALAPPDATA/ignis/cache/ngram, Linux XDG_CACHE_HOME/ignis/ngram or HOME/.cache/ignis/ngram), or a directory."),
  decl("ngram", "hot_bytes", "bytes|auto", null, { families: ["qwen38flashnext"], patchable: true, reload_required: true },
    "The n-gram rows held in RAM, every other row read from NVMe when a step needs it: a byte count, or `auto` — what the host plan leaves after its other lines and the margin, less 256 MiB for the load, the whole ~29 GB table when that fits, never below the default. Unset, 1 GiB."),
  decl("kv_disk", "bytes", "bytes", null, { scoped: true, patchable: true, reload_required: true },
    "KV-disk's budget: a ceiling, cut at start to the volume's free space less 10 GiB. Unset, the model's own: 4G on Flash-Next, 0 (off) on the 27B."),
  decl("kv_disk", "path", "model|auto|dir", "model", { patchable: true, reload_required: true },
    "Where KV-disk's files go: `model` (beside the artifact), `auto` (the per-user cache directory under kv-disk), or a directory. They go in an ignis-kv-disk/<pid>-<nonce> directory there, removed at shutdown."),
];

/**
 * `GET /v1/config` on the 27B under the `rtx5090` profile: the general value
 * of every field, and a family's section wherever its value differs — a field
 * only one family takes is only ever in that family's section.
 *
 * The 27B's `headroom_bytes` is the profile's (1200M against the general
 * 1536M); every other family value is the operator's own, in the file.
 */
export const CONFIG: ConfigDocument = {
  server: {
    bind: "127.0.0.1:8000",
    expose: null,
    metrics: true,
    metrics_bind: "127.0.0.1:9464",
    ui: true,
    request_timeout: 120,
    system_message_policy: "merge",
    developer_message_policy: "inplace",
  },
  model: {
    id: null,
    artifact: null,
    max_context: 40960,
    default_max_tokens: 38912,
    kv_format: "hq-e8-2b",
    prefill_chunk: 1024,
    decode_share: null,
    enable_thinking: true,
    reasoning_effort: null,
    thinking_budget: 32768,
    rope_scaling: "auto",
    qwen38: { max_context: 786432 },
    qwen38flashnext: { max_context: 589824 },
  },
  vram: {
    kv_pool_bytes: null,
    headroom_bytes: "1536M",
    budget_bytes: null,
    allow_oversubscription: false,
    qwen38: { headroom_bytes: "1200M" },
    qwen38flashnext: { allow_expert_cache_below_floor: false, pinned_expert_bytes: "2G" },
  },
  reuse: { prompt: true, retained_device: null, retained_host: null, retained_interactive_ttl: 300, kv_host_pool_bytes: "2G" },
  switch: {
    drain_timeout: 30,
    allow_implicit: true,
    known_models: { "qwen3.8-27b": "models/qwen3_8_27b_nvfp4full-v2.ninfer", "qwen3.8-27b-dead": "F:/scratch/qwen3_8_27b-dead.ninfer", "qwen3.8-27b-fail": "F:/scratch/qwen3_8_27b-fail.ninfer" },
  },
  spec: {
    backend: null,
    draft_tokens: null,
    decode_lanes: null,
    draft_head: null,
    qwen38: { backend: "dflash2", draft_tokens: 7 },
    qwen38flashnext: { draft_rows: null, decode_lanes: 3 },
  },
  vision: { qwen38: { enabled: true, max_tokens: 32768, embedding_pool_mib: null, residency: "host" } },
  media: { qwen38: { allow_private_network: false, cache_mib: 1024 } },
  download: { enabled: true, path: "./models", endpoint: "https://huggingface.co", catalog: "./catalog.yaml", from_ui: true, free_margin: "1G" },
  ngram: { persist: true, persist_path: "model", qwen38flashnext: { hot_bytes: null } },
  kv_disk: { bytes: null, path: "model" },
};

/**
 * `GET /v1/models/catalog` with the 27B loaded: the three built-in entries
 * (their totals are the catalog's file sizes summed), an operator entry not
 * fetched, and a known model that is no catalog entry.
 */
export const CATALOG: CatalogRow[] = [
  { id: "qwen3.8-27b", source: "built-in", on_disk: "complete", bytes: 19406949449, family: "qwen38", loaded: true, switchable: false, download: "none", bytes_on_disk: 19406949449 },
  { id: "qwen3.8-27b-abliterated", source: "built-in", on_disk: "partial", bytes: 19406984175, loaded: false, switchable: false, download: "none", bytes_on_disk: 6442450944 },
  { id: "qwen3.8-flash-next", source: "built-in", on_disk: "complete", bytes: 72799980571, family: "qwen38flashnext", loaded: false, switchable: true, download: "none", bytes_on_disk: 72799980571 },
  { id: "qwen3.8-27b-custom", source: "operator", on_disk: "absent", bytes: 19406949449, loaded: false, switchable: false, download: "none", bytes_on_disk: 0 },
  { id: "qwen3.8-27b-dead", source: "known", on_disk: "complete", bytes: 19406942468, family: "qwen38", loaded: false, switchable: true, download: "none", bytes_on_disk: 19406942468 },
  { id: "qwen3.8-27b-fail", source: "known", on_disk: "complete", bytes: 19406942468, family: "qwen38", loaded: false, switchable: true, download: "none", bytes_on_disk: 19406942468 },
];

/** The families the fixture's server declares, read from its declarations and catalog as the page reads them. */
export const SCOPES = familiesOf(FIELDS, CATALOG, null);

/** Where a value was set, by `<group>.<name>` or `<group>.<family>.<name>`; a value not named here is the file's, or the default's. */
export type Provenance = Record<string, { source: OriginRow["source"]; spelling?: string }>;

export const PROVENANCE: Provenance = {
  "vram.qwen38.headroom_bytes": { source: "profile", spelling: "rtx5090" },
  "vram.qwen38flashnext.pinned_expert_bytes": { source: "profile", spelling: "rtx5090" },
  "model.thinking_budget": { source: "env", spelling: "IGNIS_MODEL_THINKING_BUDGET" },
};

const CONFIG_PATH = "ignis.yaml";

/**
 * `GET /v1/config/origins` for a configuration: a row per visible field and
 * scope, the shared one first, then each family that takes the field (the
 * shared row of a field one family takes holds the default, as no group value is set). `doc`
 * holds every family value, also one equal to the shared one, which is why
 * the server's own answer could not tell it from a shared one.
 */
export function originRows(fields: FieldDecl[], doc: ConfigDocument, from: Provenance): OriginRow[] {
  const rows: OriginRow[] = [];
  for (const decl of fields) {
    for (const scope of [null, ...(decl.families ?? SCOPES)]) {
      const section = scope === null ? undefined : doc[decl.group]?.[scope];
      const own = section !== null && typeof section === "object" && decl.name in (section as object);
      const key = scope === null ? `${decl.group}.${decl.name}` : `${decl.group}.${scope}.${decl.name}`;
      const sharedKey = `${decl.group}.${decl.name}`;
      const set = from[key] ?? (own ? undefined : scope === null ? undefined : from[sharedKey]);
      const value = own ? (section as Record<string, unknown>)[decl.name] : doc[decl.group]?.[decl.name];
      const source = set?.source ?? (JSON.stringify(value ?? null) === JSON.stringify(decl.default ?? null) ? "default" : "file");
      rows.push({
        key: decl.file_key,
        scope,
        source,
        scoped: own,
        ...(set?.spelling ? { spelling: set.spelling } : source === "file" ? { spelling: `${own ? key : sharedKey} in ${CONFIG_PATH}` } : {}),
      });
    }
  }
  return rows;
}

/** `GET /v1/config/origins` for the fixture's configuration. */
export const ORIGINS: OriginRow[] = originRows(FIELDS, CONFIG, PROVENANCE);
