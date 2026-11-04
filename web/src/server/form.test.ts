import { describe, expect, it } from "vitest";
import { CATALOG, CONFIG, FIELDS, ORIGINS } from "./fixture.ts";
import {
  changes,
  controlFor,
  type Draft,
  drainTimeout,
  editDraft,
  type Basis,
  familiesOf,
  familyName,
  fieldMatches,
  firstSentence,
  inScope,
  namedKeys,
  patchBody,
  canUseShared,
  dropDraft,
  needsReload,
  noteLines,
  primaryLabel,
  removeDraft,
  rowMode,
  rowView,
  shownOf,
  waitingForOthers,
  waitingLine,
} from "./form.ts";
import type { FieldDecl, OriginRow } from "./routes.ts";

// The Server view's form (spec playground/05 §Configuration, §Applying): the
// control a declaration calls for, the value in force, and the draft that
// becomes one PATCH.

const field = (key: string): FieldDecl => {
  const found = FIELDS.find((f) => f.file_key === key);
  if (!found) throw new Error(key);
  return found;
};

/** Edit `key` to `value` in `scope` (the 27B's by default; `null` is "All models"), as a row does. */
const edit = (draft: Draft, key: string, value: string | boolean, scope: string | null = "qwen38") =>
  editDraft(draft, field(key), rowView(field(key), CONFIG, ORIGINS, scope).inForce, value);

const view = (key: string, scope: string | null, origins: OriginRow[] | null = ORIGINS) => rowView(field(key), CONFIG, origins, scope);

/** The origins with the row of `key` in `scope` replaced. */
const withRow = (key: string, scope: string, row: Partial<OriginRow>): OriginRow[] =>
  ORIGINS.map((r) => (r.key === key && r.scope === scope ? { ...r, ...row } : r));

const basis = (family: string | null, origins: OriginRow[] | null = ORIGINS): Basis => ({ fields: FIELDS, doc: CONFIG, origins, family, families: familiesOf(FIELDS, CATALOG, family) });

describe("controlFor", () => {
  it("is the switch for a bool, segments up to four values, a select past that, else a text field", () => {
    expect(controlFor(field("model.enable_thinking"), true)).toEqual({ kind: "switch" });
    expect(controlFor(field("server.system_message_policy"), "merge")).toEqual({ kind: "segmented", options: ["merge", "strict"] });
    expect(controlFor(field("model.kv_format"), "hq-e8-2b")).toEqual({ kind: "segmented", options: ["bf16", "hq-e8-2b"] });
    expect(controlFor(field("server.developer_message_policy"), "inplace")).toEqual({
      kind: "select",
      options: ["inplace", "into-system", "after-system", "one-after-system", "reject"],
    });
    expect(controlFor(field("model.reasoning_effort"), null)).toMatchObject({ kind: "select" });
  });

  it("lists a value in force the rule does not name (an alias set from the file), so it shows as the choice", () => {
    expect(controlFor(field("model.kv_format"), "hq_e8_2b")).toEqual({ kind: "segmented", options: ["bf16", "hq-e8-2b", "hq_e8_2b"] });
    // A spelling that differs in case only is the same choice.
    expect(controlFor(field("model.kv_format"), "BF16")).toEqual({ kind: "segmented", options: ["bf16", "hq-e8-2b"] });
    // Unset is no value to list.
    expect(controlFor(field("model.reasoning_effort"), null)).toMatchObject({ options: expect.not.arrayContaining([null]) });
  });

  it("gives a text field the kind and the rule as its hint, and \"auto\" as the placeholder of an unset value", () => {
    expect(controlFor(field("server.request_timeout"), 120)).toEqual({ kind: "text", hint: "seconds, 1..=3600", placeholder: "120" });
    expect(controlFor(field("model.prefill_chunk"), 1024)).toMatchObject({ hint: "tokens, nonzero multiple of 128" });
    expect(controlFor(field("model.max_context"), 786432)).toMatchObject({ hint: "tokens, >= 1" });
    expect(controlFor(field("vram.kv_pool_bytes"), null)).toEqual({ kind: "text", hint: "bytes|tokens", placeholder: "auto" });
    // The draft row budget's range as the server declares it (GitHub #329 item 5: up to 32).
    expect(controlFor(field("spec.draft_rows"), null)).toEqual({ kind: "text", hint: "rows, 0..=32", placeholder: "auto" });
  });
});

describe("rowMode", () => {
  it("reads a field that is not patchable as set at start, with no control", () => {
    expect(rowMode(field("server.bind"))).toEqual({ kind: "fixed" });
    expect(rowMode(field("download.path"))).toEqual({ kind: "fixed" });
  });

  it("gives a patchable field its control", () => {
    expect(rowMode(field("spec.draft_rows"))).toEqual({ kind: "edit" });
    expect(rowMode(field("vision.enabled"))).toEqual({ kind: "edit" });
  });

  it("shows the known models read-only: a patch only adds to them", () => {
    expect(rowMode(field("switch.known_models"))).toEqual({ kind: "list" });
  });
});

describe("inScope", () => {
  it("shows every field in All models", () => {
    for (const key of ["server.request_timeout", "model.max_context", "vision.enabled", "spec.draft_rows"]) expect(inScope(field(key), null), key).toBe(true);
  });

  it("shows in a family's scope only what that family can set for itself: a per-family field, or one only it takes", () => {
    expect(inScope(field("model.max_context"), "qwen38")).toBe(true);
    expect(inScope(field("vram.headroom_bytes"), "qwen38")).toBe(true);
    expect(inScope(field("vision.enabled"), "qwen38")).toBe(true);
    // One value for every model, set at start or live: All models'.
    expect(inScope(field("server.request_timeout"), "qwen38")).toBe(false);
    expect(inScope(field("model.id"), "qwen38")).toBe(false);
    // Another family's own.
    expect(inScope(field("spec.draft_rows"), "qwen38")).toBe(false);
    expect(inScope(field("spec.draft_rows"), "qwen38flashnext")).toBe(true);
    expect(inScope(field("vision.enabled"), "qwen38flashnext")).toBe(false);
  });
});

describe("rowView", () => {
  it("in a family's scope reads a scoped field from its section when it has one, else the group's, and says which", () => {
    expect(view("model.max_context", "qwen38")).toMatchObject({ inForce: { value: 786432, scope: "qwen38" }, shared: 40960, note: { kind: "scoped", origin: { kind: "own" } } });
    expect(view("model.max_context", "qwen38flashnext").inForce).toEqual({ value: 589824, scope: "qwen38flashnext" });
    // No own value: the group's, and an edit still goes to the family's section.
    expect(view("model.kv_format", "qwen38")).toMatchObject({ inForce: { value: "hq-e8-2b", scope: "qwen38" }, note: { kind: "scoped", origin: { kind: "shared" } } });
    expect(view("spec.backend", "qwen38flashnext")).toMatchObject({ inForce: { value: null, scope: "qwen38flashnext" }, note: { kind: "scoped", origin: { kind: "shared" } } });
  });

  it("says the card decided a value it runs in place of the one left to its default (GitHub #477), also for a field only one model takes", () => {
    const decided = withRow("model.adapt_context_step", "qwen38", { source: "card", spelling: "NVIDIA GeForce RTX 3090 (8.6)" });
    for (const scope of ["qwen38", null]) {
      const card = view("model.adapt_context_step", scope, decided);
      expect(card.note).toEqual({ kind: "scoped", origin: { kind: "card", name: "NVIDIA GeForce RTX 3090 (8.6)" } });
      expect(noteLines(card, undefined)).toEqual(["Decided by the card, NVIDIA GeForce RTX 3090 (8.6)"]);
      expect(canUseShared(card, undefined)).toBe(false);
    }
    expect(view("model.adapt_context_step", null).note).toEqual({ kind: "only", family: "qwen38" });
  });

  it("reads a family's value by its origin: the profile's and a flag's are neither own nor removable", () => {
    const profile = view("vram.headroom_bytes", "qwen38");
    expect(profile).toMatchObject({ inForce: { value: "1200M", scope: "qwen38" }, shared: "1536M", note: { kind: "scoped", origin: { kind: "profile", name: "rtx5090" } } });
    expect(noteLines(profile, undefined)).toEqual(["From profile rtx5090"]);
    expect(canUseShared(profile, undefined)).toBe(false);
    // An edit of it is the family's own value from then on.
    expect(noteLines(profile, edit({}, "vram.headroom_bytes", "1G")["vram.qwen38.headroom_bytes"])).toEqual(["Own value"]);

    const flagged = view("vram.headroom_bytes", "qwen38", withRow("vram.headroom_bytes", "qwen38", { source: "flag", spelling: "--qwen38-vram-headroom-bytes" }));
    expect(noteLines(flagged, undefined)).toEqual(["Set by --qwen38-vram-headroom-bytes"]);
    expect(canUseShared(flagged, undefined)).toBe(false);
    const env = view("vram.headroom_bytes", "qwen38", withRow("vram.headroom_bytes", "qwen38", { source: "env", spelling: "IGNIS_QWEN38_VRAM_HEADROOM_BYTES" }));
    expect(noteLines(env, undefined)).toEqual(["Set by IGNIS_QWEN38_VRAM_HEADROOM_BYTES"]);

    // Only a file or live value is the operator's own.
    for (const source of ["file", "live"] as const) {
      const own = view("vram.headroom_bytes", "qwen38", withRow("vram.headroom_bytes", "qwen38", { source }));
      expect(noteLines(own, undefined)).toEqual(["Own value"]);
      expect(canUseShared(own, undefined)).toBe(true);
    }
  });

  it("shows an own value equal to the shared one as own, which the file's shape alone cannot", () => {
    // GET /v1/config leaves the 27B's kv_format out of its section when it equals the shared one; the origins still say it is the file's.
    const own = withRow("model.kv_format", "qwen38", { source: "file", scoped: true });
    const kv = view("model.kv_format", "qwen38", own);
    expect(kv.inForce).toEqual({ value: "hq-e8-2b", scope: "qwen38" });
    expect(noteLines(kv, undefined)).toEqual(["Own value"]);
    expect(canUseShared(kv, undefined)).toBe(true);
    expect(noteLines(view("model.kv_format", null, own), undefined)).toEqual(["Qwen3.8-27B has its own value: hq-e8-2b"]);
  });

  it("without the origins reads the file's shape alone: a family section is the family's own", () => {
    expect(noteLines(view("model.max_context", "qwen38", null), undefined)).toEqual(["Own value"]);
    expect(noteLines(view("model.kv_format", "qwen38", null), undefined)).toEqual(["Shared"]);
    expect(noteLines(view("model.max_context", null, null), undefined)).toEqual(["Qwen3.8-27B has its own value: 786432", "Qwen3.8-Flash-Next has its own value: 589824"]);
  });

  it("in All models says what a shared edit does to each family with a value of its own", () => {
    // The profile's value is replaced by a shared edit; a flag's or an env var's value is named.
    expect(noteLines(view("vram.headroom_bytes", null), undefined)).toEqual(["Replaces the profile's 1200M for Qwen3.8-27B"]);
    const flagged = withRow("vram.headroom_bytes", "qwen38", { source: "flag", spelling: "--qwen38-vram-headroom-bytes" });
    expect(noteLines(view("vram.headroom_bytes", null, flagged), undefined)).toEqual(["Qwen3.8-27B is set by --qwen38-vram-headroom-bytes"]);
    // A family on the shared value is not named.
    expect(noteLines(view("vram.allow_oversubscription", null), undefined)).toEqual([]);
  });

  it("in All models shows and edits the group value, and names each family that has its own", () => {
    const context = view("model.max_context", null);
    expect(context.inForce).toEqual({ value: 40960, scope: null });
    expect(noteLines(context, undefined)).toEqual(["Qwen3.8-27B has its own value: 786432", "Qwen3.8-Flash-Next has its own value: 589824"]);
    expect(noteLines(view("spec.backend", null), undefined)).toEqual(["Qwen3.8-27B has its own value: dflash2"]);
    expect(noteLines(view("model.kv_format", null), undefined)).toEqual([]);
  });

  it("edits the group's value of a field every family shares, whichever scope asks, and says nothing of it", () => {
    expect(view("server.request_timeout", "qwen38")).toMatchObject({ inForce: { value: 120, scope: null }, note: null });
    expect(noteLines(view("server.request_timeout", null), undefined)).toEqual([]);
    // A field with no control says nothing about its sharing.
    expect(noteLines(view("server.bind", "qwen38"), undefined)).toEqual([]);
  });

  it("reads a field only one family takes from its family's section, in any scope that shows it", () => {
    expect(view("vision.enabled", "qwen38").inForce).toEqual({ value: true, scope: "qwen38" });
    expect(view("vision.enabled", null).inForce).toEqual({ value: true, scope: "qwen38" });
    expect(noteLines(view("vision.enabled", null), undefined)).toEqual(["Qwen3.8-27B only"]);
    expect(noteLines(view("vision.enabled", "qwen38"), undefined)).toEqual([]);
    // It is editable while the other family runs: the section is its family's.
    expect(view("spec.draft_rows", null)).toMatchObject({ mode: { kind: "edit" }, inForce: { value: null, scope: "qwen38flashnext" } });
    expect(noteLines(view("spec.draft_rows", null), undefined)).toEqual(["Qwen3.8-Flash-Next only"]);
  });

  it("has no value without a configuration", () => {
    expect(rowView(field("server.request_timeout"), null, null, "qwen38").inForce).toEqual({ value: undefined, scope: null });
    expect(rowView(field("model.max_context"), null, null, null).note).toBeNull();
  });

  it("marks a pending edit as an own value, and a pending removal as a return to the shared one", () => {
    const kv = view("model.kv_format", "qwen38");
    const edited = edit({}, "model.kv_format", "bf16")["model.qwen38.kv_format"];
    expect(noteLines(kv, edited)).toEqual(["Own value"]);
    expect(canUseShared(kv, edited)).toBe(false);
    const context = view("model.max_context", "qwen38");
    expect(canUseShared(context, undefined)).toBe(true);
    const removal = removeDraft({}, field("model.max_context"), "qwen38")["model.qwen38.max_context"];
    expect(noteLines(context, removal)).toEqual(["Back to the shared value once applied"]);
    expect(canUseShared(context, removal)).toBe(false);
  });
});

describe("the draft", () => {
  it("holds only what differs from the value in force, compared as the field reads it", () => {
    let draft: Draft = {};
    draft = edit(draft, "server.request_timeout", "60");
    expect(Object.keys(changes(draft))).toEqual(["server.request_timeout"]);
    // Typed back to the value in force, it is no change at all.
    draft = edit(draft, "server.request_timeout", "120");
    expect(draft).toEqual({});
    // A choice is compared without case, as the server compares it.
    expect(edit({}, "server.system_message_policy", "MERGE")).toEqual({});
    expect(edit({}, "model.enable_thinking", true)).toEqual({});
  });

  it("keeps a cleared text field on screen but sends nothing for it: a PATCH cannot unset a field", () => {
    const draft = edit({}, "server.request_timeout", "  ");
    expect(draft["server.request_timeout"]?.value).toBe("  ");
    expect(changes(draft)).toEqual({});
    expect(patchBody(draft)).toEqual({});
  });
});

describe("patchBody", () => {
  it("nests only the changed fields as the config file nests them, a scoped edit under the family it was made for", () => {
    let draft: Draft = {};
    draft = edit(draft, "server.request_timeout", " 60 ");
    draft = edit(draft, "model.enable_thinking", false);
    draft = edit(draft, "model.max_context", "524288");
    draft = edit(draft, "model.kv_format", "bf16");
    draft = edit(draft, "vision.max_tokens", "16384");
    expect(patchBody(draft)).toEqual({
      server: { request_timeout: "60" },
      model: { enable_thinking: false, qwen38: { max_context: "524288", kv_format: "bf16" } },
      vision: { qwen38: { max_tokens: "16384" } },
    });
  });

  it("holds edits of one field from several scopes at once, and a removal as null in its family's section", () => {
    let draft: Draft = {};
    draft = edit(draft, "model.max_context", "50000", null);
    draft = edit(draft, "model.kv_format", "bf16", "qwen38");
    draft = removeDraft(draft, field("model.max_context"), "qwen38flashnext");
    draft = edit(draft, "spec.draft_rows", "4", null);
    expect(Object.keys(draft)).toEqual(["model.max_context", "model.qwen38.kv_format", "model.qwen38flashnext.max_context", "spec.qwen38flashnext.draft_rows"]);
    expect(Object.keys(changes(draft))).toHaveLength(4);
    expect(patchBody(draft)).toEqual({
      model: { max_context: "50000", qwen38: { kv_format: "bf16" }, qwen38flashnext: { max_context: null } },
      spec: { qwen38flashnext: { draft_rows: "4" } },
    });
    expect(Object.keys(dropDraft(draft, field("model.max_context"), "qwen38flashnext"))).not.toContain("model.qwen38flashnext.max_context");
  });

  it("replaces a removal by an edit, and an edit by a removal, of the same field in the same scope", () => {
    const removed = removeDraft({}, field("model.max_context"), "qwen38");
    expect(patchBody(edit(removed, "model.max_context", "524288"))).toEqual({ model: { qwen38: { max_context: "524288" } } });
    expect(patchBody(removeDraft(edit({}, "model.max_context", "524288"), field("model.max_context"), "qwen38"))).toEqual({ model: { qwen38: { max_context: null } } });
  });

  it("sends a scoped field under the group when it is edited in All models", () => {
    expect(patchBody(edit({}, "spec.backend", "mtp", null))).toEqual({ spec: { backend: "mtp" } });
  });
});

describe("the reload a draft causes", () => {
  const reload = (draft: Draft, family: string | null = "qwen38", origins: OriginRow[] | null = ORIGINS) => needsReload(draft, basis(family, origins));
  const waits = (draft: Draft, family: string | null = "qwen38") => waitingForOthers(draft, basis(family));

  it("reloads for a reload_required value that resolves differently for the running family", () => {
    expect(reload(edit({}, "model.max_context", "524288"))).toBe(true);
    expect(reload(edit({}, "model.kv_format", "bf16", null))).toBe(true);
    expect(reload(edit({}, "vision.enabled", false))).toBe(true);
    // A live field never does.
    expect(reload(edit({}, "server.request_timeout", "60"))).toBe(false);
    expect(reload({})).toBe(false);
  });

  it("does not reload for a value of the other family, which waits for it", () => {
    const draft = edit({}, "model.max_context", "524288", "qwen38flashnext");
    expect(reload(draft)).toBe(false);
    expect(waits(draft)).toEqual([{ family: "qwen38flashnext", keys: ["model.qwen38flashnext.max_context"] }]);
    expect(waitingLine(waits(draft)[0])).toBe("Takes effect when Qwen3.8-Flash-Next loads: model.qwen38flashnext.max_context");
    // A field only that family takes is the same.
    const rows = edit({}, "spec.draft_rows", "4", null);
    expect(reload(rows)).toBe(false);
    expect(waits(rows)).toEqual([{ family: "qwen38flashnext", keys: ["spec.qwen38flashnext.draft_rows"] }]);
    // And from the other side.
    expect(reload(edit({}, "model.max_context", "524288", "qwen38"), "qwen38flashnext")).toBe(false);
    expect(waits(edit({}, "model.max_context", "524288", "qwen38"), "qwen38flashnext")).toEqual([{ family: "qwen38", keys: ["model.qwen38.max_context"] }]);
  });

  it("does not reload for a group key the running family overrides, which reaches only a family that does not", () => {
    // The 27B and Flash-Next both have their own max_context: the shared one is run by neither.
    const both = edit({}, "model.max_context", "50000", null);
    expect(reload(both)).toBe(false);
    expect(waits(both)).toEqual([]);
    // Only the 27B overrides the backend: Flash-Next takes the shared one when it loads.
    const backend = edit({}, "spec.backend", "mtp", null);
    expect(reload(backend)).toBe(false);
    expect(waits(backend)).toEqual([{ family: "qwen38flashnext", keys: ["spec.backend"] }]);
    // Seen from Flash-Next, which has no own backend, the same edit reloads.
    expect(reload(backend, "qwen38flashnext")).toBe(true);
  });

  it("reloads for a shared edit of a value the profile gave the running family, which the edit replaces", () => {
    // rtx5090 gives the 27B headroom 1200M; the patched shared value outranks it.
    const headroom = edit({}, "vram.headroom_bytes", "1G", null);
    expect(reload(headroom)).toBe(true);
    expect(waits(headroom)).toEqual([]);
    // A flag's value or an env var's is replaced the same way, until restart.
    for (const source of ["flag", "env"] as const) {
      expect(reload(headroom, "qwen38", withRow("vram.headroom_bytes", "qwen38", { source, spelling: "x" }))).toBe(true);
    }
    // Replaced by the value it already has, nothing changes for the model.
    expect(reload(edit({}, "vram.headroom_bytes", "1200M", null))).toBe(false);
  });

  it("does not reload for a shared edit the running family's own value outranks, hidden from the file's shape or not", () => {
    // The 27B's own kv_format equals the shared one, so GET /v1/config shows no section: only the origins say it is the file's.
    const own = withRow("model.kv_format", "qwen38", { source: "file", scoped: true });
    const draft = edit({}, "model.kv_format", "bf16", null);
    expect(reload(draft, "qwen38", own)).toBe(false);
    expect(reload(draft, "qwen38", ORIGINS)).toBe(true);
    expect(reload(draft, "qwen38flashnext", own)).toBe(true);
    expect(waitingForOthers(draft, basis("qwen38", own))).toEqual([{ family: "qwen38flashnext", keys: ["model.kv_format"] }]);
  });

  it("reloads for a removal of the running family's own value when the shared one differs", () => {
    expect(reload(removeDraft({}, field("model.max_context"), "qwen38"))).toBe(true);
    expect(reload(removeDraft({}, field("spec.backend"), "qwen38"))).toBe(true);
    expect(reload(removeDraft({}, field("model.max_context"), "qwen38flashnext"))).toBe(false);
    expect(waits(removeDraft({}, field("model.max_context"), "qwen38flashnext"))).toEqual([{ family: "qwen38flashnext", keys: ["model.qwen38flashnext.max_context"] }]);
    // A removal against a group edit in the same draft resolves to the edited value.
    let draft = edit({}, "model.max_context", "786432", null);
    draft = removeDraft(draft, field("model.max_context"), "qwen38");
    expect(reload(draft)).toBe(false);
  });

  it("reloads when any value reaches the running model, and still lists what waits", () => {
    let draft = edit({}, "model.max_context", "524288", "qwen38flashnext");
    expect(reload(draft)).toBe(false);
    draft = edit(draft, "model.kv_format", "bf16");
    expect(reload(draft)).toBe(true);
    expect(waits(draft)).toEqual([{ family: "qwen38flashnext", keys: ["model.qwen38flashnext.max_context"] }]);
  });

  it("assumes a reload when the running family or the configuration is not known, and lists nothing as waiting", () => {
    const draft = edit({}, "model.max_context", "524288", "qwen38flashnext");
    expect(reload(draft, null)).toBe(true);
    expect(waits(draft, null)).toEqual([]);
    expect(needsReload(draft, { ...basis("qwen38"), doc: null })).toBe(true);
  });
});

describe("primaryLabel", () => {
  it("is Apply, and names the reload when the draft causes one", () => {
    expect(primaryLabel(false, "qwen3.8-27b")).toBe("Apply");
    expect(primaryLabel(true, "qwen3.8-27b")).toBe("Apply and reload qwen3.8-27b");
    // A cleared reload field is no change, so no reload either.
    expect(needsReload(edit({}, "model.max_context", ""), basis("qwen38"))).toBe(false);
  });
});

describe("namedKeys", () => {
  it("marks every row whose file key the refusal names, in a family section too, and no row a longer key contains", () => {
    const message =
      "`model.qwen38.max_context` must be at least 1, got 0; `server.metrics_bind` cannot be changed while the server runs; `server.request_timeout` expects a second count, got `x`";
    expect([...namedKeys(message, FIELDS)].sort()).toEqual(["model.max_context", "server.metrics_bind", "server.request_timeout"]);
  });
});

describe("text", () => {
  it("splits a description after its first sentence, for the rest behind More", () => {
    expect(firstSentence(field("model.id").description)).toEqual([
      "The id the model is served under.",
      "Unset, a load is served under its own model's id (qwen3.8-27b or qwen3.8-flash-next); without an artifact, qwen3.8-27b. An id naming the other model than the artifact's is refused.",
    ]);
    expect(firstSentence(field("model.enable_thinking").description)).toEqual(["Whether a request that says nothing thinks.", ""]);
  });

  it("derives the families from the declarations and the catalog, never a list kept here", () => {
    expect(familiesOf(FIELDS, CATALOG, "qwen38")).toEqual(["qwen38", "qwen38flashnext"]);
    expect(familiesOf([], [], null)).toEqual([]);
    expect(familiesOf(FIELDS, null, null)).toEqual(["qwen38", "qwen38flashnext"]);
    // The catalog names a family no declaration does, and the running one is always there.
    const third = [...CATALOG, { id: "m", source: "known" as const, on_disk: "complete" as const, family: "qwen4", loaded: false, switchable: true }];
    expect(familiesOf(FIELDS, third, "qwen38")).toEqual(["qwen38", "qwen38flashnext", "qwen4"]);
    expect(familiesOf([], [], "qwen9")).toEqual(["qwen9"]);
  });

  it("names the families as people do, and shows a value with no control as words", () => {
    expect(familyName("qwen38")).toBe("Qwen3.8-27B");
    expect(familyName("qwen38flashnext")).toBe("Qwen3.8-Flash-Next");
    // A scope with no display name is shown as the server spells it.
    expect(familyName("qwen4")).toBe("qwen4");
    expect(shownOf(null)).toBe("Not set");
    expect(shownOf(true)).toBe("On");
    expect(shownOf("127.0.0.1:8000")).toBe("127.0.0.1:8000");
    expect(shownOf(40960)).toBe("40960");
  });
});

describe("fieldMatches", () => {
  const maxContext = field("model.max_context");
  const none = { marks: [], opens: false, ids: false };

  it("lets every field through an empty search, marking nothing", () => {
    expect(fieldMatches(maxContext, "")).toEqual(none);
    expect(fieldMatches(maxContext, "   ")).toEqual(none);
  });

  it("marks where the name holds a word, whatever its case, and needs every word somewhere", () => {
    expect(fieldMatches(maxContext, "CONTEXT")).toEqual({ ...none, marks: [[4, 11]] });
    expect(fieldMatches(maxContext, "context max")).toEqual({ ...none, marks: [[0, 3], [4, 11]] });
    expect(fieldMatches(maxContext, "max nowhere")).toBeNull();
  });

  it("finds a field by its group.name key, as a refusal names it, marking the part of the name it covers", () => {
    expect(fieldMatches(maxContext, "model.max_context")).toEqual({ ...none, marks: [[0, 11]] });
    expect(fieldMatches(maxContext, "model.max")).toEqual({ ...none, marks: [[0, 3]] });
    // The group alone is in the section's heading: nothing in the name to mark.
    expect(fieldMatches(maxContext, "model")).toEqual(none);
  });

  it("finds a field by its description, and opens it when the word is only behind More", () => {
    expect(fieldMatches(maxContext, "reserve")).toEqual(none);
    expect(fieldMatches(field("model.id"), "flash-next")).toEqual({ ...none, opens: true });
  });

  it("finds a field by its flag or its environment variable, which the row then shows", () => {
    expect(fieldMatches(maxContext, "max-context")).toEqual({ ...none, ids: true });
    expect(fieldMatches(maxContext, "ignis_model_max_context")).toEqual({ ...none, ids: true });
    expect(fieldMatches(maxContext, "context --model")).toEqual({ marks: [[4, 11]], opens: false, ids: true });
  });
});

describe("drainTimeout", () => {
  it("is switch.drain_timeout in force, or unknown without a configuration", () => {
    expect(drainTimeout(CONFIG)).toBe(30);
    expect(drainTimeout({ switch: { drain_timeout: 0 } })).toBe(0);
    expect(drainTimeout(null)).toBeNull();
  });
});
