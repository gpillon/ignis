---
status: accepted
---

# No typed attribute registry; discipline + regression tests only

Once a structured log attribute is useful outside ignis its name and its type must stay stable (it must not silently
become a formatted string later), and `tracing`'s field macros are stringly typed at the call site: nothing at
compile time stops a second call site from logging the same attribute name with a different type. ignis holds that
stability by call-site discipline and a regression test that structured attributes retain their native types, with
no typed registry of attributes. The choice was explicit, not a default left unexamined: for a single-owner project
with no external consumer of these attributes yet, the ceremony of a typed registry costs more up front than the
type drift it prevents. It is a trade-off of compile time and authoring only, with no cost or benefit on the prefill
and decode hot path either way.

## Considered Options

- **A small typed registry in `crates/logging`** (constants or newtypes such as `attr::QUEUE_DEPTH: u64`) that
  forces name and type to agree at compile time for the attributes that are a contract: rejected for the cost above.

## Consequences

- Attribute type drift, if it happens, is caught by the tests, or later in review, not by the compiler.
- When ignis gains external consumers of specific attributes (dashboards, alerting, another team), this is
  revisited: a typed registry for just the attributes that became a real external contract is the natural next
  step, not a wholesale rewrite.
