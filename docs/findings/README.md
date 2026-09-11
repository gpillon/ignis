# Findings

This collection preserves durable, reusable, evidence-backed results that can
inform future work in this repository. To add, update, or supersede a finding,
follow the [authoring convention](../agents/findings.md).

## Classification

- `discovery`: durable observations from exploring the codebase, architecture,
  runtime, or workflow.
- `research`: conclusions drawn from identifiable external sources.
- `experiment`: results from reproducible tests, benchmarks, or measurements.

## Status

- `current`: the finding remains usable.
- `superseded`: the finding is retained as historical context and links to the
  material that replaces it.

## Index

| Finding | Kind | Scope | Observed | Status | Superseded by | Summary |
|---|---|---|---|---|---|---|
| [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md) | discovery | kernel / paged KV cache, scheduler capacity | 2026-09-11 | current | none | A sequence-token costs 9,216 bytes under hq-e8-2b against 65,536 under BF16 (7.11x), so 8 lanes at 40,960 context need ~3.02 GB instead of 20 GiB. |
