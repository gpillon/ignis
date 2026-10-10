# 02 — live config over HTTP, hardware profiles, and config-file auto-discovery

GitHub: #311 (same ticket as spec 01; "few tickets" — one ticket, two spec
files). Builds on `docs/specs/config-v2/01-one-declaration-per-field.md`;
read that first — this spec adds three consumers of the same field
declaration, it does not change the mechanism.

## Problem Statement

Spec 01 gives every config field one flag, one env var, one file key, one
default and one validator. Three things came up once that existed:

1. **The runtime model switch (#305) can already reload a model; nothing
   lets an operator *change* its configuration without hand-editing a file
   and restarting, or crafting a `POST /v1/models/switch` body that still
   only names `artifact`/`model`.** An HTTP surface to read and change the
   live config is the natural next step, and it must reuse the switch
   mechanism already proven, not invent a second one.
2. **The owner plans to support GPUs beyond the RTX 5090.** Several fields'
   right *default* is hardware-shaped (VRAM headroom, the expert-cache
   floor, how many decode lanes fit) — today that shape lives in one
   developer's head and the Makefile's own defaults, not in the config
   system spec 01 just defined.
3. **Nothing points ignis at a config file unless told to.** `--config` /
   `IGNIS_CONFIG` (spec 01) are explicit; an operator who drops a file next
   to the binary or in a per-user config directory, the way most CLIs
   already work, gets nothing unless they remember the flag.

## Solution

Three `FieldMeta` attributes (`visible`, `patchable`, `reload_required`)
added to every field's single declaration (spec 01's macro) drive a new
`GET`/`PATCH /v1/config` pair: `GET` returns only fields marked `visible`
(whitelist, not a redaction step — a field simply absent if not whitelisted,
`api_key` never marked `visible` at all; `bind` *is*, nothing about it is
secret). `PATCH` always persists what it is given to the config file in
use (spec 01's `--config`/auto-discovery), and applies it one of two ways,
never mixed: if every patched field is `patchable` and none is
`reload_required`, it takes effect immediately, live, no model touched; if
even one patched field is `reload_required`, the **whole patch** — every
field in the request, not only the reload-requiring ones — goes through
the same atomic teardown-then-load `model_switch::switch` #305 already
proved, now parameterized on a config diff rather than only on a different
artifact.

A `--profile <name>` names a bundle of hardware-shaped defaults, slotting in
just above the hardcoded default in spec 01's precedence chain — a smarter
default source, not a new override tier above flag/env/config.

A config file is looked for at a short, fixed list of conventional paths
when `--config`/`IGNIS_CONFIG` name none, and the server always prints
where its config came from at startup — a discovered path, the explicit one
named, or that there is none — so "why is this value what it is" is never a
mystery.

## User Stories

1. As the owner, I want `GET /v1/config` to show the server's current
   effective configuration, so that I can inspect a running server without
   SSH-ing in to read a file.
2. As the owner, I want `GET /v1/config` to show only fields explicitly
   marked `visible` — `api_key` never among them, `bind` always among them
   — so that the endpoint cannot leak a secret by oversight; the absent
   field, not a masked value, is what protects it.
3. As the owner, I want `PATCH /v1/config` to always write what it accepts
   to the config file the server is using, so that a live change survives
   the next restart, not only the current process.
4. As the owner, I want a patch containing only `patchable`,
   non-`reload_required` fields (`request-timeout`, message policies,
   `allow-model-switch`, `known-model` additions, `switch-drain-timeout`)
   applied immediately with no model touched, so that routine operational
   tuning never pays a reload it does not need.
5. As the owner, I want a patch containing even one `reload_required`
   field to apply its *entire* request as one atomic reload — never some
   fields live and others pending — so that the server is never left in a
   state that matches neither the old config nor the new one.
6. As the owner, I want a field marked `patchable: false` (`api_key`,
   `bind`, `metrics-bind`, `ui`) refused by `PATCH` outright, so that an
   authenticated request cannot lock me out of my own server by changing
   the key it was authenticated with, or ask for a socket rebind the
   reload mechanism was never built to do.
7. As the owner, I want a `PATCH` whose reload fails to leave the previous
   model serving and the file **not** overwritten with the failed values,
   so that a bad patch degrades exactly like a bad explicit switch (#305
   AC 9) — never a stopped server, never a file that describes a config
   nothing is running.
8. As the owner, I want `--profile <name>` to seed hardware-shaped defaults
   (VRAM headroom, the expert-cache floor, decode lanes, retained slots)
   below every flag/env/config-file value, so that supporting a card
   other than the RTX 5090 is choosing a profile, not re-deriving every
   number by hand.
9. As the owner, I want a profile definable in the config file itself
   (a `profiles:` section), not only as a handful of built-in Rust presets,
   so that a profile for a card I own and the maintainers do not is still
   possible.
10. As the owner, I want profile and model-family scope to compose (a
    profile sets the hardware shape, a family override sets the per-model
    shape on top), so that "this card, these two models" is fully
    expressible without a third override mechanism.
11. As an operator, I want the server to look for a config file at a short,
    fixed list of conventional paths when `--config`/`IGNIS_CONFIG` name
    none, so that dropping a file where the binary already expects to find
    one works without extra flags.
12. As an operator, I want the server to print where its configuration came
    from at every startup — the discovered path, the explicitly named one,
    or that none was found — so that "why is this value what it is" is
    answered by the first line of the log, not a guess.
13. As a maintainer, I want `visible`/`patchable`/`reload_required` to be
    three independent attributes on the same field declaration spec 01
    already has (not three new mechanisms), so that `GET`/`PATCH
    /v1/config` cost the field table three booleans each, not a parallel
    classification list that can drift from it the way `fit_to_family`'s
    hand-written drop list already did once.
14. As a maintainer, I want the CLI's `config generate`/`config
    print`/`config patch` (below) to resolve through the exact same
    function `PATCH /v1/config` does, so that "what the file-only tool
    would produce" and "what the live endpoint would apply" are
    provably the same code, not two paths that can disagree.

## Implementation Decisions

### `GET /v1/config`

Returns the current effective config (spec 01's resolved `Config`,
post-family-override, post-profile), filtered to fields with
`visible: true`. No api key field exists in the response at all when
`visible: false` — not a string, not `null`, the key is absent, so a
client cannot distinguish "hidden" from "absent" and neither tells it
anything.

### `PATCH /v1/config`

Body: the same nested, grouped shape the config file uses (spec 01,
`<group>: { <field>: value, "<family>": { <field>: value } }`), partial —
only the fields being changed.

1. Validate every field in the body against its own `Validator` (spec 01)
   and its `applicable_to` (refuse a field the active/target family cannot
   take, same as a bad explicit value today).
2. Refuse the whole patch, nothing applied, if any field is
   `patchable: false`.
3. Classify what remains: if no field is `reload_required`, apply all of
   them directly to the live `Server`/`ActiveModel` state (the small set
   of fields this can mean: request timeout, message policies, the
   model-switch group's own knobs, default thinking) and return `200` with
   the new effective config.
4. If any field *is* `reload_required`, build the full target config (the
   active config, patched) and run it through `model_switch::switch`
   exactly as `POST /v1/models/switch` does — same gate, same drain, same
   teardown-then-load, same rollback-to-the-old-model on failure (#305) —
   except the artifact/model stay the **same**; what changes is the config
   the next load resolves with. Returns `202`, `GET /v1/models` (`status`)
   and `GET /v1/config` both reflect the switch in progress and its
   outcome, as they do for an explicit one.
5. **The write to the config file happens only after the change is known
   to have taken (step 3's direct apply, or step 4's switch succeeding)** —
   a failed reload leaves the file matching what is actually running, per
   AC 7. The file write reuses the exact function `config patch` (CLI,
   below) uses to merge a change into an existing file.
6. No config file in use at startup (no `--config`, `IGNIS_CONFIG`, nor an
   auto-discovered one): the patch still applies (live or reload, per
   steps 3/4), and the response marks `persisted: false` with the reason,
   rather than refusing a live change just because there is nowhere to
   write it down.

`model_switch::ArtifactLoader` (`crates/server/src/model_switch.rs`) is
extended to carry an optional config diff alongside its fixed
`artifact`/`model`, applied to `self.options` before `fit_to_family` runs —
the one structural change to the existing switch machinery; everything
else in `begin`'s five steps is unchanged (module doc,
`crates/server/src/model_switch.rs:1-47`).

### `--profile <name>`

Resolves to a named bundle of field defaults — a few built into the binary
(`rtx5090`, today's existing numbers, as the implicit default profile so an
unset `--profile` changes nothing for the owner's own machine) and any
number defined in the config file's own `profiles:` section (same shape as
a family override: a partial set of `<group>.<field>` values). Precedence
(extends spec 01 AC 6):

```
<family>-flag > flag > <family>-env > env > <family>-config
  > config > <profile>-default > hardcoded-default
```

A profile is orthogonal to family scope: a profile answers "what kind of
card is this," a family override answers "what does this model want" —
both apply, family wins where they'd disagree (it is closer to the top of
the chain above the profile's own position).

### `applicable_to` × `visible`/`patchable`/`reload_required`

Four independent attributes per field now (spec 01's `applicable_to` plus
this spec's three), all on the one declaration a field's `macro_rules!`
invocation already produces (spec 01 §The field declaration) — no new
macro, three more arguments to the existing one.

### Config-file auto-discovery

Checked, in order, only when neither `--config` nor `IGNIS_CONFIG` names
one: the working directory (`./ignis.config.yaml`, `./ignis.config.json`),
then a per-user config directory (`%APPDATA%\ignis\config.yaml` on Windows,
`$XDG_CONFIG_HOME/ignis/config.yaml` or `~/.config/ignis/config.yaml` on
Linux — matching the release's own two platforms, ADR 0032). First match
wins; none found is not an error, it is the "no config file" case spec 01
already handles (flags/env/profile/default only). The startup log always
states the source — `ignis.config.source` naming `explicit` (the path),
`discovered` (the path), or `none` — printed before the model load begins,
so it is visible even if the load itself then fails.

### The CLI: `config generate` / `config print` / `config patch`

Three verbs (spec 01's `generate-config` is renamed `config generate` to
sit under this grouping; `help`/`help --fields`/`version`/bare-invocation-
serves are unchanged from spec 01):

- **`config generate [--format json|yaml] [--out <path>] [--force]
  [--dry-run] [the usual flags]`** — resolves flags/env/profile/defaults
  (no existing file read, ever) and writes a fresh file. Refuses to
  overwrite an existing file at the target path unless `--force`.
  `--format` is explicit and wins over whatever `--out`'s extension would
  suggest; required when writing to stdout (no extension to infer from).
  `--dry-run` resolves and validates, prints, writes nothing.
- **`config print [--file <path>] [the usual flags]`** — resolves
  file-if-present (default path per auto-discovery's first candidate when
  `--file` is omitted) + env + the flags given on *this* invocation, and
  prints the effective result. Never writes. Answers "what would actually
  run" — the same question `--dry-run` answers for a hypothetical fresh
  file, now for "this file plus whatever I just typed."
- **`config patch [--file <path>] [--out <path>] [the usual flags]`** —
  same resolution as `print`, but validates and **writes** (in place, or
  to `--out`). Reuses every field's own flag (`--reuse-kv-host-pool-bytes
  2G`, not a separate `--set group.field=value` syntax — one field table
  drives `serve`, `generate`, `print` and `patch` identically; only the
  action taken with the resolved values differs). No field changes given
  on the command line still re-resolves file + env and writes the
  result — this is the whole of what an earlier draft called `config
  save`; no separate verb is needed; refuses if `--file`/the default path
  names nothing (`config generate` runs first, by design — `patch`
  modifies, it does not create).

## Testing Decisions

- **`GET /v1/config`**: `api_key` absent from the response body entirely
  (not `null`, not a masked string) when the server was started with one;
  `bind` present. A field added to the group table without `visible` set
  explicitly defaults to *not* visible — a test instantiates a throwaway
  field descriptor with no `visible` given and asserts it is excluded,
  pinning the whitelist's fail-closed default.
- **`PATCH /v1/config`**: a `patchable`-only, non-`reload_required` body
  applied with no `ignis.model.switch_*` event emitted (direct apply,
  engine untouched); a body containing one `reload_required` field drives
  the full `model_switch::switch` path (reuses #305's mock-engine harness,
  asserting the *other*, otherwise-hot fields in the same body landed too —
  the "never mixed" rule, AC 5); a body naming a `patchable: false` field
  refused with nothing applied, including the hot fields that were also in
  it; a reload that fails leaves both the old model serving (existing #305
  coverage) **and** the config file unchanged (new: read the file after a
  failed patch, assert it still matches the pre-patch value).
- **Profiles**: a built-in profile's values resolve exactly where no
  flag/env/config names a field (precedence table, extending spec 01's);
  a config-file-defined custom profile resolves identically to a built-in
  one from the resolver's point of view (same code path, not a special
  case for "ours" vs. "theirs").
- **Auto-discovery**: cwd found before the user config dir when both
  exist; an explicit `--config`/`IGNIS_CONFIG` short-circuits discovery
  entirely (never even stats the candidate paths); the startup log line
  is asserted for all three source states (`explicit`/`discovered`/`none`).
- **The CLI**: `config generate` refuses an existing file without
  `--force`; `config patch` refuses when no file exists (names `config
  generate` as the fix, in the error text); `config print` never touches
  disk (a test asserts the target path's mtime is unchanged, or that it
  does not exist at all when it shouldn't); `config patch` with zero field
  flags still rewrites the file (the `save`-by-another-name case) and its
  content changes only if env/an existing stale value actually differs
  from what re-resolution now produces.
- **The gate**: `cargo test` stays green, CPU-only, workspace-wide; nothing
  here needs the GPU — the reload path is proven through the same
  mock-engine harness #305 already built.

## Out of Scope

- **An HTTP endpoint to manage profiles** (`POST /v1/profiles` or similar).
  Profiles are config-file content for now; a dedicated API for them is a
  later decision if the file ever becomes awkward for it.
- **Remote/networked config stores.** Spec 01 already ruled this out; this
  spec's `GET`/`PATCH /v1/config` is local-process state over the
  existing HTTP surface, not a new kind of source.
- **A `POST /v1/config` “save” endpoint distinct from `PATCH`.** Settled in
  conversation: `PATCH` with no `reload_required` fields already persists,
  and a CLI `config patch` with zero field flags already is the save case
  — no separate verb, HTTP or CLI, is added.
- **Hot-swapping `--bind` or `--metrics-bind` without a process restart.**
  `patchable: false` on both; rebinding a listening socket is not what the
  model-reload mechanism this spec reuses was built to do, and nothing
  here asks it to.
- **Auto-discovery searching more than the two conventional locations**, or
  searching recursively up a directory tree (as some tools do for a
  project root). Two fixed candidates, first match, nothing cleverer.

## Further Notes

The three new `FieldMeta` attributes read, at a glance, like they could
collapse into fewer — `patchable: false` implies "never `reload_required`
matters, it's refused regardless" for `api_key`/`bind`, so one might ask
why both exist. They stay separate because they answer different
questions a field can have only one real answer to, but a reader
shouldn't have to infer the second from the first: `visible` is about
*reading*, `patchable`/`reload_required` are about *writing*, and a field
could in principle be `visible: true, patchable: false` (show it, never
let `PATCH` touch it — `bind`, exactly) or `visible: false, patchable:
true` (no field happens to want this combination yet, but the attributes
should not be coupled just because today's fields don't exercise every
corner of the matrix).
