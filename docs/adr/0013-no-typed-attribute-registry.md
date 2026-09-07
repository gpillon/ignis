# ADR 0013 — No typed attribute registry; discipline + regression tests only

## Status

Accepted (2026-09-07, grilling session on `.scratch/Ignis Structured Logging
Specification.md`).

## Context

The logging spec's §18 ("Attribute stability") requires that once a
structured attribute like `ignis.scheduler.queue_depth` is externally useful,
its name and type stay stable (e.g. it must not silently become a formatted
string later). `tracing`'s field macros (`field = value`) are stringly-typed
at the call site — nothing at compile time stops a second call site from
logging the same attribute name with a different type.

The alternative considered was a small typed registry in `crates/logging`
(e.g. `attr::QUEUE_DEPTH: u64` constants/newtypes) that would force attribute
name+type agreement at compile time for "contract" attributes.

## Decision

**No typed attribute registry.** Ignis relies on call-site discipline plus
the regression tests already required by the spec itself (§39, test #17:
"structured attributes retain native types"). This was an explicit choice,
not a default left unexamined: for a single-owner project with no external
consumers of these attributes yet, the ceremony of a typed registry costs
more up front than the type-drift risk it prevents. This is orthogonal to
critical-path performance — it is a compile-time/authoring-time tradeoff
only, and has zero cost or benefit on the prefill/decode hot path either way.

## Consequences

- Attribute type drift, if it happens, is caught by tests (or later, in
  review) rather than by the compiler.
- If/when ignis gains external consumers of specific attributes (dashboards,
  alerting, another team), revisit this — a typed registry for just the
  attributes that became a real external contract is the natural next step,
  not a wholesale rewrite.
