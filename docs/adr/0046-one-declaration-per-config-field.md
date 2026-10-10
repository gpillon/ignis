# ADR 0046 — one declaration per config field: flag, env, file key, default, validator

## Status

Accepted (2026-10-10): decided in conversation with the owner while testing
the runtime model switch (#305) against a real client. Spec:
`docs/specs/config-v2/01-one-declaration-per-field.md`. **Amended same day**
by `docs/specs/config-v2/02-live-config-profiles-and-discovery.md`: the same
one-declaration-per-field mechanism grows three more consumers (`GET`/`PATCH
/v1/config`, hardware `--profile`s, config-file auto-discovery) and three
new `FieldMeta` attributes (`visible`, `patchable`, `reload_required`) that
drive them — no change to the core mechanism decided here.

## Context

`crates/server/src/config.rs` (3,875 lines) has ~50 CLI flags and 54 env
vars, hand-matched in one `while` loop and resolved field by field with the
repeated shape `flag.or_else(|| env("IGNIS_X")).unwrap_or(default)`. There is
no config file, no grouping (a few flags share an informal prefix —
`--vision-*`, `--media-*`, `--retained-*` — most do not), and at least one
flag/env pair has already drifted (`IGNIS_PROMETHEUS` and `IGNIS_TELEMETRY`
exist as env vars with no corresponding flag found).

The runtime model switch (#305, spec `model-switch/01`) exposed why this
matters beyond tidiness: the 27B and Qwen3.8-Flash-Next have opposite
resource philosophies — the 27B's default is to give the KV pool everything
left after weights, Flash-Next's default is to starve the KV pool so the
expert cache gets the room. A switch today carries forward one shared
`Config` (set once at process start) to whatever model loads next;
`config::fit_to_family` only *drops* flags the target family cannot take, it
never *resizes* one for the target. Running the server with the 27B's
production profile (`--kv-host-pool-bytes 8G`) and switching to Flash-Next
sent Flash-Next's own host-plan check (`crates/core/src/residency/plan.rs`)
the 27B's 8 GiB arena size, not Flash-Next's own natural one, and the switch
refused on a real machine (`HostPlanError::BelowMargin`, `kv_ram_arena`
crossing the line) — not because of a bug in the switch mechanism (already
verified: the old model's pool is torn down before the new one allocates),
but because one process-wide value was asked to serve two models that want
different things from the same knob.

## Decision

**Every configurable parameter is declared once**, and that one declaration
produces its CLI flag, its env var, its config-file key, its default, its
validator and its help text — never four things written by hand in four
places that can drift apart.

**Groups replace the flat flag namespace.** `server`, `model`, `vram`,
`reuse` (prompt reuse + the KV-RAM arena), `switch`, `spec` (speculation),
`vision`, `media`, `download`, `ngram`, `kv_disk`. A field's flag is
`--<group>-<field>`, its env var `IGNIS_<GROUP>_<FIELD>`, its config-file key
`<group>.<field>` (nested object, not a flat dotted string) — one shape, three
surfaces.

**A config file, JSON or YAML, one struct.** Every group struct derives
`Serialize`/`Deserialize`; the file is whichever format, 1:1, no bespoke
schema. `--config <path>` / `IGNIS_CONFIG` names it.

**Precedence:** flag > env > config file > default. Two models want
different values for some fields (the problem above): scope is the second
axis. Within each source, a family-scoped value wins over the group's
general one:

```
<family>-flag > flag > <family>-env > env > <family>-config > config > default
```

**Families, not model ids, carry the per-family scope.** Two families today,
`QWEN38` and `QWEN38FLASHNEXT` — plain identifiers, no sanitizing needed,
because they tie directly to the existing `ModelFamily` enum rather than to
whatever a served model id happens to be spelled (`qwen3.8-flash-next`'s dot
and hyphens are not valid env-var characters, and model ids are not fixed to
two anyway — families are). `--qwen38-reuse-kv-host-pool-bytes` /
`IGNIS_QWEN38_REUSE_KV_HOST_POOL_BYTES`, `--qwen38flashnext-reuse-kv-host-pool-bytes`
/ `IGNIS_QWEN38FLASHNEXT_REUSE_KV_HOST_POOL_BYTES`, and a config-file
`reuse.qwen38.kv_host_pool_bytes` / `reuse.qwen38flashnext.kv_host_pool_bytes`
beside the group's general `reuse.kv_host_pool_bytes`.

**A field that only one family has at all is a different thing from a field
both families want sized differently.** `applicable_to: AllFamilies |
Only(&[ModelFamily])` on every field. `--spec dflash2` (`Only([Qwen38_27b])`)
or MTP (`Only([FlashNext])`) set for the wrong family is refused outright —
the same hard failure `--ngram-hot-bytes` already gives a 27B start today,
now a generic check instead of a one-off panic. Set for the right family and
then switched *away* from, it is dropped and logged
(`ignis.model.switch_flags_dropped`), exactly as `fit_to_family` does today —
this ADR makes that drop list schema-driven instead of hand-maintained, it
does not change the behavior.

**Validators are a small typed enum, not a regex escape hatch for
everything:** `Range{min,max}`, `MultipleOf(u64)` (`--prefill-chunk`'s
nonzero-multiple-of-128), `OneOf(&[&str])` (`--kv-format`'s `bf16` /
`hq-e8-2b`), and `Regex(&str)` kept only for string-shaped fields (paths,
ids) where the others don't apply. A regex cannot express "a multiple of
128" over a parsed integer, so it is not the general mechanism.

**One declaration per field, via `macro_rules!`, not a proc-macro crate.** A
hand-maintained table alongside the real struct fields is still two things a
person must keep in sync — a test can catch drift after the fact, it cannot
prevent writing the field twice. A declarative macro that expands one
invocation into both the struct field and its `FieldMeta` entry (names via
`stringify!`) removes the duplication at the source, while staying textual
and local — readable top to bottom, no second compilation pass, no
`cargo expand` required to see what it does. A full proc-macro crate was
considered and rejected as more infrastructure than eleven groups of a few
fields each need. Adopting an existing crate for this (Rust's `clap` for the
CLI half, `figment`/`config-rs` for the layered-source half, the rough
equivalent of Python's `pydantic-settings`/`dynaconf` the owner named) was
considered and rejected too: none of them produce this codebase's narrative,
field-specific error messages verbatim, and the owner judged a home-grown
macro small enough not to need the dependency.

**The CLI gains subcommands, minimally.** Bare invocation still serves —
`make start`'s generated command line does not grow a new leading word.
`help` (not only `--help`) and `help --fields` (every field: group, flag,
env, config key, default, validator, `applicable_to`, description) are new.
`generate-config [--dry-run] [flags...]` resolves the same flag/env/default
pipeline used to start a server and writes the result as a config file
(stdout, or an explicit `--out <path>`, format from its extension); with no
flags it emits every default. `--dry-run` runs the same resolution and
validation and prints the result without writing anything — a config file's
or a flag combination's correctness check that needs no artifact and no GPU.

## Consequences

- **Breaking, accepted by the owner.** Every existing flag and env var is
  renamed into its group (`--kv-host-pool-bytes` becomes
  `--reuse-kv-host-pool-bytes`, `IGNIS_KV_HOST_POOL_BYTES` becomes
  `IGNIS_REUSE_KV_HOST_POOL_BYTES`, etc.). The blast radius: `config.rs`
  itself (effectively rewritten), `main.rs`, the Makefile's knob-to-flag
  translation (`MAX_CONTEXT`, `KV_HOST_POOL_BYTES`, `MODEL_FAMILY`, …),
  `docs/user/README.md`'s flag table, `local.mk.example`, and every test
  across the workspace that builds a `Config` via `args(&[...])` /
  `env_map(&[...])`. The spec enumerates what moves; nothing here is done
  silently.
- **The original problem (27B vs. Flash-Next wanting different arena sizes)
  is solved by the per-family scope, not by a one-off flag or a smaller
  global compromise value.** Each family keeps its own natural resource
  shape across a switch in either direction, which was not possible with one
  shared `Config`.
- **`fit_to_family`'s hand-written drop list is replaced by the generic
  `applicable_to` check** — same observable behavior, declared once per
  field instead of maintained as a parallel list that can fall out of sync
  with the fields it is supposed to cover.
- **`help --fields` becomes the source of truth for what the server
  accepts**, probably replacing (or generating) the flag table in
  `docs/user/README.md` rather than the two staying hand-synced.
- **`generate-config` gives an operator a way to freeze today's flags/env
  into a reproducible file**, and `--dry-run` gives a way to validate one
  without starting anything — useful on its own, independent of the
  multi-model motivation.
