# 01 — config: one declaration per field (flag, env, file key, default, validator)

GitHub: TBD (to open alongside this spec). ADR: `docs/adr/0046-one-declaration-per-config-field.md`.

## Problem Statement

`crates/server/src/config.rs` (3,875 lines) hand-matches ~50 CLI flags in one
`while` loop and separately resolves each against one of 54 env vars with the
repeated shape `flag.or_else(|| env("IGNIS_X")).unwrap_or(default))` — a
flag, its env var, its default and its validation are four things written by
hand, in up to four different places in the file, for every field. At least
one has already drifted: `IGNIS_PROMETHEUS` and `IGNIS_TELEMETRY` exist as
env vars with no flag found to match either. There is no config file and no
grouping — a few flags share an informal prefix (`--vision-*`, `--media-*`,
`--retained-*`, `--draft-*`), most do not.

The runtime model switch (#305, spec `model-switch/01`) is what forced the
issue. The 27B and Qwen3.8-Flash-Next want opposite things from the same
knob: the 27B's production profile gives the KV pool everything left after
weights (`--kv-pool-bytes` resident), Flash-Next starves the KV pool so the
expert cache gets the room instead. A switch today carries forward one
`Config`, set once at process start, to whichever model loads next;
`config::fit_to_family` (`crates/server/src/config.rs:1193-1219`) only
*drops* a flag the target family cannot take at all (`--vision`, `--spec`,
`--draft-rows`, `--decode-lanes`, `--allow-expert-cache-below-floor`,
`--ngram-hot-bytes`) — it never *resizes* one both families take but want
different. Measured directly: starting the server on the 27B's production
profile (`--kv-host-pool-bytes 8G`) and switching to Flash-Next sent
Flash-Next's own host-plan check
(`crates/core/src/residency/plan.rs::plan_host`) the 27B's 8 GiB arena size
instead of a Flash-Next-shaped one, and the switch refused on a real machine
— not a bug in the switch (independently verified: the old model's pool is
torn down before the new one allocates anything), but one process-wide value
asked to serve two models with different resource shapes.

## Solution

Declare every configurable parameter once. That one declaration produces its
CLI flag, its env var, its config-file key, its default, its validator, its
applicable model families and its help text. A config file (JSON or YAML,
one struct, 1:1) joins the existing flag/env sources at a defined place in
the precedence order. Fields that two model families want sized differently
gain a family-scoped variant of their flag/env/file key; fields only one
family has at all are refused for the other, generically, from the same
declaration. A small `macro_rules!` block is the one piece of generated code
this adds — one invocation per field, expanding to both the struct field and
its metadata, so the two can never drift apart because they are the same
written thing.

## User Stories

1. As a maintainer, I want every config field's flag, env var, default and
   validator declared in one place, so that `IGNIS_PROMETHEUS`-style drift
   (an env var or flag with no counterpart) cannot happen again.
2. As the owner, I want flags grouped by concern
   (`server`/`model`/`vram`/`reuse`/`switch`/`spec`/`vision`/`media`/`download`/`ngram`/`kv_disk`)
   with one naming shape across all three surfaces
   (`--<group>-<field>` / `IGNIS_<GROUP>_<FIELD>` / `<group>.<field>` in the
   file), so that related knobs read as related instead of a flat
   alphabet-soup list.
3. As the owner, I want a config file (JSON or YAML, my choice, 1:1 with the
   same struct) named by `--config <path>` / `IGNIS_CONFIG`, so that a
   server's whole configuration can be versioned and reviewed as one
   artifact instead of a long command line.
4. As the owner, I want flag > env > config file > default, so that an
   operator's explicit override always wins regardless of what a file or the
   environment says.
5. As the owner, I want some fields overridable per model family
   (`QWEN38` / `QWEN38FLASHNEXT` — plain identifiers, no sanitizing, tied to
   the existing `ModelFamily` enum rather than to a served model id's
   spelling), so that the 27B and Flash-Next each keep their own resource
   shape across a switch, closing the exact gap #305 hit.
6. As the owner, I want the family scope to be a tiebreaker *within* a
   source, not a fourth independent axis, so that
   `<family>-flag > flag > <family>-env > env > <family>-config > config > default`
   is the whole precedence rule — the command line always wins regardless of
   scope, and within any one source the more specific value wins.
7. As the owner, I want a field that only one family has at all
   (`--spec dflash2`: 27B only; MTP: Flash-Next only) refused outright if set
   for the wrong family — the same hard failure `--ngram-hot-bytes` on a 27B
   start already gives today — rather than silently accepted and later
   dropped, so a config mistake is caught when it is made.
8. As the owner, I want that same field, set validly for the family that is
   active and then switched *away* from, dropped with a log line rather than
   refusing the whole switch, so that `fit_to_family`'s existing, correct
   behavior (`ignis.model.switch_flags_dropped`) is preserved, not changed —
   only made to come from the field's own declaration instead of a
   hand-maintained list.
9. As a maintainer, I want each field's validator to be one of a small typed
   set (`Range`, `MultipleOf`, `OneOf`, or `Regex` for string-shaped fields
   only), so that "a nonzero multiple of 128" (`--prefill-chunk`,
   `crates/server/src/config.rs:1243`) is expressible — a regex over the raw
   string cannot state a numeric-multiple constraint, so it is not the
   general mechanism.
10. As an operator, I want `ignis-server help` and `ignis-server help
    --fields` (every field: group, flag, env, file key, default, validator,
    applicable families, description), so that the running binary is its own
    up-to-date reference instead of a hand-maintained table in
    `docs/user/README.md` that can fall behind.
11. As an operator, I want `ignis-server generate-config` (no flags: every
    default; with flags: those values resolved the same way a start would)
    to write a config file, so that today's command line can be frozen into
    a reviewable, versionable file in one step.
12. As an operator, I want `ignis-server generate-config --dry-run` to run
    the same resolution and validation and print the result without writing
    anything, so that a flag combination or an existing config file can be
    checked for correctness without touching disk, the GPU, or an artifact.
13. As an operator, I want the bare invocation (`ignis-server --artifact
    ...`) to still start the server with no new leading subcommand, so that
    `make start`'s generated command line does not have to change shape for
    this.
14. As a maintainer, I want the field declaration to be a local
    `macro_rules!` block, not a proc-macro crate, so that what each field
    expands to stays readable top to bottom in one file, diffable, and
    needs no `cargo expand` step to inspect.
15. As a maintainer, I want this to depend on no new CLI/config-layering
    crate (`clap`, `figment`, `config-rs`) beyond `serde` and one of
    `serde_json`/`serde_yaml` (already idiomatic, needed for the file
    format regardless), so that this codebase's narrative, field-specific
    error messages stay ours to write, not a library's generic ones.

## Implementation Decisions

### The field declaration

One `macro_rules!` invocation per field, per group. Expands to:
- the struct field on that group's `Config` sub-struct (plain Rust type,
  `Serialize`/`Deserialize` derived on the whole sub-struct for the file
  format);
- a `FieldMeta` entry (group, field name via `stringify!`, type tag,
  default, description, `Validator`, `applicable_to`, whether a family
  override exists for it) pushed into that group's `const FIELDS: &[FieldMeta]`.

```rust
enum Validator {
    None,
    Range { min: Option<i64>, max: Option<i64> },
    MultipleOf(u64),
    OneOf(&'static [&'static str]),
    Regex(&'static str), // string-typed fields only
}

enum Applicability {
    AllFamilies,
    Only(&'static [ignis_core::compute::ModelFamily]),
}
```

A per-type resolver trait (`Bool`, `U64`, `Bytes` — the existing `K`/`M`/`G`
suffix parser, `Duration`, `String`, `OneOf`) is what the macro calls per
field rather than inlining parsing logic in the macro itself — keeps the
macro's own expansion small and each type's parsing testable on its own,
independent of which field uses it.

### Groups

`server` (bind, api-key, expose, metrics, metrics-bind, ui, request-timeout,
system/developer-message-policy), `model` (model, artifact, max-context,
kv-format, prefill-chunk, decode-share, thinking defaults, rope-scaling),
`vram` (kv-pool-bytes, vram-headroom-bytes, vram-budget-bytes,
allow-vram-oversubscription, allow-expert-cache-below-floor), `reuse`
(prompt-reuse, retained-pool-bytes, retained-slots, retained-device,
retained-host, retained-interactive-ttl, kv-host-pool-bytes — the arena is
prompt reuse's, not the VRAM pool's), `switch` (allow-model-switch,
switch-drain-timeout, known-model), `spec` (spec, draft-tokens, draft-rows,
decode-lanes, draft-head), `vision`, `media`, `download`
(model-download(-path)), `ngram` (persist-ngram-cache(-path),
ngram-hot-bytes), `kv_disk` (kv-disk-bytes, kv-disk-path).

### Family scope

Two families today, `QWEN38` and `QWEN38FLASHNEXT`
(`ignis_core::compute::ModelFamily::Qwen38_27b`/`FlashNext`), spelled plainly
— no character sanitizing, because the scope is the family (two fixed,
clean identifiers), never a served model id (`qwen3.8-flash-next`'s dot and
hyphens would need escaping if the scope were per-id instead). A
family-overridable field gets, beside its group's general
flag/env/file-key, one more of each per family:
`--qwen38-reuse-kv-host-pool-bytes` / `IGNIS_QWEN38_REUSE_KV_HOST_POOL_BYTES`
/ `reuse.qwen38.kv_host_pool_bytes`, and the `qwen38flashnext` equivalents.
Resolution per AC 6; the family-scoped source is checked first, at each
precedence level, falling back to the group-general one at that same level
before moving to the next source.

### `applicable_to`

A field with `Applicability::Only(&[family])` is refused — a `ConfigError`
naming the field and the family, the same shape `--ngram-hot-bytes` on a 27B
start already produces — if an explicit value (flag, env, or file, scoped or
general) is given while that family is the one being configured. This
replaces `fit_to_family`'s hand-written drop list
(`crates/server/src/config.rs:1193-1219`) with a generic check driven by the
same field metadata `help --fields` reads; the *behavior* at a runtime
switch is unchanged — a field valid for the model being switched away from
is dropped and logged (`ignis.model.switch_flags_dropped`), never refused,
because the value was never wrong, only no longer applicable.

### The config file

Each group's struct derives `Serialize`/`Deserialize`; the top-level
`ConfigFile` composes them by group name as nested keys, matching the flag's
own `<group>-<field>` shape (`<group>: { <field>: value }`). JSON or YAML
read through the same `Deserialize` impl — `serde_json`/`serde_yaml`, no
bespoke parser. `--config <path>` / `IGNIS_CONFIG` name the file; a missing
file named by env is a refusal, by flag likewise (an operator who named a
file meant it to exist).

### The CLI surface

Bare invocation serves, unchanged in shape (AC 13). New subcommands:
`help`, `help --fields` (table: group, flag, env, file key, default,
validator, applicable families, description — one row per field, machine-
and human-readable), `version` (already existed as a flag; keeps also
working as `--version`/`-V` for muscle memory), `generate-config [--dry-run]
[--out <path>] [the same flags a start would take]`. `generate-config`
resolves flags/env/defaults exactly as a start would (no artifact load, no
GPU, no config-file read of its own — it *produces* one), applies every
validator, and on success either writes the result (`--out`, format from
its extension; stdout if omitted) or, under `--dry-run`, prints it without
writing. A validation failure reports the same `ConfigError` a start would
have refused with.

### Precedence resolution

One generic function per primitive kind, taking the up-to-six candidate
raw strings in precedence order (family-flag, flag, family-env, env,
family-file, file, each `Option<&str>`) plus the default, returning the
parsed, validated value or the `ConfigError` the first non-empty, invalid
candidate produced (precedence order decides which candidate is "first,"
not which fails first — a later, lower-precedence candidate being invalid
is never reached once an earlier one resolves).

## Testing Decisions

- **Exhaustiveness**: a test iterates every group's `FIELDS` table and
  confirms it has exactly one entry per struct field (by field name) and no
  orphans — the generic version of what `IGNIS_PROMETHEUS` broke; run once
  per group, not three thousand asserts written by hand.
- **Precedence**: for a representative field in each source combination
  (flag-only, env-only, file-only, default-only, family-flag over flag,
  family-env over env, family-file over file, flag over family-env — the
  "command line always wins regardless of scope" case) — table-driven, one
  table per precedence rule, not one test per field.
- **Validators**: each `Validator` variant gets a unit test independent of
  which field uses it (a `Range` test, a `MultipleOf` test, etc.), plus one
  pinned regression per field that has a real validator today
  (`--prefill-chunk`'s 128 multiple, `--decode-lanes`'s 1..=8, `--kv-format`'s
  `OneOf`).
- **`applicable_to`**: a field `Only([FlashNext])` set explicitly while
  configuring the 27B is refused (mirrors the existing
  `ngram_hot_bytes_parse_and_refuse_the_27b` test, generalized); the same
  field, valid at start and switched away from, is dropped and logged, not
  refused (existing `model_switch.rs` test coverage, re-pointed at the new
  mechanism).
- **The config file**: round-trip (`generate-config` with no flags, read
  back by a start) for both JSON and YAML; a file naming an unknown field or
  the wrong type for one is a `ConfigError` naming the field, not a serde
  panic or a silent default.
- **`help --fields`**: a golden test that every field present in the table
  also appears in the printed output (not a snapshot of the whole text,
  which would break on every wording tweak) — the same exhaustiveness
  property as the first bullet, from the CLI side.
- **The gate**: `cargo test` stays green, CPU-only, workspace-wide, as
  today — nothing here touches the GPU or needs an artifact.

## Out of Scope

- **Per-served-model-id overrides** (as opposed to per-family). Two
  families cover today's two models; a served id's own spelling is not
  env-var-safe without sanitizing, and nothing today needs finer than
  family scope. Revisit only if a family ever serves more than one
  materially-different artifact at once.
- **A third config source beyond flag/env/file** (a remote config service, a
  database). Not asked for.
- **`clap`, `figment`, `config-rs`, or any other CLI/layering crate.**
  Rejected in the ADR; the home-grown macro is judged small enough.
- **Renaming `ModelFamily`'s two variants** or adding a third family. This
  spec threads the existing enum through config scoping; it does not touch
  what families exist.
- **Changing any field's actual default value or validation range** as part
  of this move. The rename and regrouping is mechanical; a value change is a
  separate decision, made separately if ever needed.
- **Migrating existing deployments' flags automatically.** The rename is
  breaking, accepted; a compatibility shim that accepts both old and new
  flag spellings is not built.

## Further Notes

The macro's expansion should stay inspectable without `cargo expand` —
`macro_rules!` keeps this true by construction (no separate crate, no
hidden codegen pass), which is worth protecting if a later refactor is
tempted to reach for a proc-macro once the field count grows: the point of
this design was specifically to avoid that infrastructure while still
removing the duplication that caused `IGNIS_PROMETHEUS` to drift in the
first place.
