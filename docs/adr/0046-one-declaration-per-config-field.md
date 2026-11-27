---
status: accepted
---

# One declaration per config field: flag, env, file key, default, validator

Every configurable parameter is declared once, and that one declaration produces its CLI flag, its env var, its
config-file key, its default, its validator and its help text: never four things written by hand in four places that
can drift apart, as a flag and its env var already had in the flat, hand-matched set this replaced, which had no
config file and no grouping. The runtime model switch showed why it matters beyond tidiness: the 27B gives the KV
pool everything left after the weights, Flash-Next starves the KV pool so that the expert cache gets the room, and
one process-wide value, set once at start, was asked to serve two models that want different things from the same
knob. A switch carried the 27B's arena size to Flash-Next's host plan and was refused, with no bug in the switch.

- **Groups replace the flat flag namespace.** A field's flag is `--<group>-<field>`, its env var
  `IGNIS_<GROUP>_<FIELD>`, its config-file key `<group>.<field>` (a nested object, not a flat dotted string): one
  shape, three surfaces.
- **A config file, JSON or YAML**, whichever format, 1:1 with the groups, with no bespoke schema; `--config <path>`
  or `IGNIS_CONFIG` names it.
- **Precedence** is flag > env > config file > default, and scope is the second axis: within each source a
  family-scoped value wins over the group's general one,
  `<family>-flag > flag > <family>-env > env > <family>-config > config > default`. The sources added since, a
  hardware profile and a live change, have their places in the order `crates/server/src/config/source.rs` states.
- **Families, not model ids, carry the per-family scope**: plain identifiers that need no sanitizing, tied to the
  `ModelFamily` enum, where a served model id holds characters an env var cannot (the dot and hyphens of
  `qwen3.8-flash-next`) and model ids are not a fixed set, as families are.
- **A field only one family has at all is a different thing from a field both families want sized differently**,
  and each field declares the families it applies to. Set for the wrong family it is refused outright, by one
  generic check instead of a panic of that field's own; set for the right family and then switched away from, it is
  dropped and logged (`ignis.model.switch_flags_dropped`), from a drop list the declarations give instead of one
  kept by hand, with the behaviour that list had. An explicit value equal to a field's default counts as unnamed, so
  a file `config generate` wrote with every default starts either model (ADR 0063's adaptation reads a value's
  source instead: one the operator wrote never adapts, even at the default).
- **A family judges only the fields of its own slot.** The text or image model is one slot; the speech model and
  the diarizer, loaded beside it, are slots of their own. So a field only the diarizer takes is judged against the
  diarizer, never against the active text model: a 27B start that sets one is not refused.
- **A value both models take is not one family's field.** `--spec-backend mtp` is both models' value, since the 27B
  drafts with its MTP head too (ADR 0095), so a switch to the 27B keeps an MTP setting instead of dropping it. A
  rule only the loading model decides runs where the configuration is fitted to it, never where every model's
  values are validated, or a value for one would refuse a start of the other.
- **Validators are a small typed set** (a numeric range, a nonzero multiple of a step, one of a set of spellings),
  not a regex escape hatch for everything: a regex cannot express "a multiple of 128" over a parsed integer, so it is
  not the general mechanism. There is no regex validator at all: no field has a pattern to match, and it would
  bring in the `regex` crate for nothing.
- **The declaration is a `macro_rules!` macro, not a proc-macro crate.** A hand-maintained table beside the real
  struct fields is still two things a person must keep in sync: a test can catch the drift after the fact, it cannot
  prevent writing the field twice. A declarative macro that expands one written entry into both the struct field
  and its `FieldMeta` entry, the names taken from the field's own identifier, removes the duplication at the source
  while staying textual and local: readable top to bottom, no second compilation pass, no `cargo expand` to see
  what it does.
- **The same declarations serve the live configuration, the hardware profiles and the discovery of a config
  file**: a declaration also says whether `GET /v1/config` shows the field, whether `PATCH /v1/config` may change it
  and whether a change needs a reload, with no change to the mechanism.
- **The CLI gains verbs, minimally.** A bare invocation still serves, so the command line `make start` generates
  does not grow a leading word. `help --fields` lists every field with what its declaration says.
  `config generate` resolves flags, environment and defaults through the pipeline that starts a server and writes
  the result as a config file, every default when no flag is given: a way to freeze today's flags and environment
  into a reproducible file. Its `--dry-run` runs the same resolution and validation and prints the result without
  writing anything: a check of a config file or a flag combination that needs no artifact and no GPU. Both are
  useful on their own, apart from the two models.

## Considered Options

- **A one-off flag for the arena, or a smaller global value both models could live with**: rejected for the
  per-family scope, under which each family keeps its own natural resource shape across a switch in either
  direction, which one shared configuration could not give.
- **A proc-macro crate**: rejected as more infrastructure than a few groups of a few fields each need.
- **An existing crate** (`clap` for the CLI half, `figment` or `config-rs` for the layered sources, the rough
  equivalent of Python's `pydantic-settings` or `dynaconf`, which the owner named): rejected. None of them produces
  this codebase's narrative, field-specific error messages verbatim, and the owner judged a home-grown macro small
  enough not to need the dependency.

## Consequences

- Breaking, accepted by the owner: every flag and env var that existed was renamed into its group.
- `help --fields` is the source of truth for what the server accepts, and the guide's reference of fields is
  generated from the same table (ADR 0066) instead of kept in sync by hand.
