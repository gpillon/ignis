# Findings

This document is the single source of truth for creating, updating, and
superseding material in `docs/findings/`. A finding preserves knowledge and
evidence that future work can reuse; it does not track work or implicitly make
an architectural decision.

## Admission

Create a finding only when the result is all of the following:

- durable beyond the task that produced it;
- reusable in future repository work;
- backed by evidence that another reader can inspect or reproduce;
- useful to a future intervention in this repository.

Keep task-specific notes or unverified hypotheses with the work that produced
them. Promote them into a finding only after every admission condition is
satisfied.

## Kind

Choose the kind that describes the result's primary evidence:

- `discovery`: observations established by exploring the codebase,
  architecture, runtime, or workflow;
- `research`: conclusions derived from identifiable external sources;
- `experiment`: results established by reproducible tests, benchmarks, or
  measurements.

When evidence spans kinds, choose the kind that supports the central conclusion
and include the other evidence in the document.

## Artifact boundaries

| Content | Authoritative location |
|---|---|
| Work status and work to be done | GitHub issue |
| Acceptance criteria and implementation spec | `.scratch/<feature>/specs/` |
| Architectural decision | `docs/adr/` |
| Domain term or concept | root `CONTEXT.md` |
| Temporary material, raw logs, and experiment output | `.scratch/` |
| Reusable finding that is not yet a decision | `docs/findings/` |

One task may produce both temporary evidence in `.scratch/` and a durable
synthesis in `docs/findings/`. The finding links to useful raw material rather
than copying voluminous logs or output.

## File and document format

Keep `docs/findings/` flat until its index becomes concretely difficult to
navigate. Name each finding `YYYY-MM-DD-short-descriptive-slug.md`, using the
observation date and a lowercase hyphenated slug.

Every finding uses this format:

```markdown
# Title

- Kind: discovery | research | experiment
- Status: current | superseded
- Observed: YYYY-MM-DD
- Last verified: YYYY-MM-DD
- Scope: component / subsystem
- Related: issue, ADR, spec, finding, or `none`
- Superseded by: link or `none`

## Question

## Evidence

## Finding

## Implications

## Limits and unknowns

## Follow-ups
```

Use `Evidence` for observations, identifiable sources, measurements, commands,
and other verifiable results. In `Finding`, distinguish observed facts from
inferences explicitly. Use `Limits and unknowns` to state what the evidence
does not establish. `Follow-ups` may link future work, while GitHub remains the
source of truth for its status. Keep raw logs and voluminous output in
`.scratch/` and reference them when useful.

Set `Last verified` to the most recent date on which the evidence and conclusion
were checked. Use repository-relative links for `Related` and `Superseded by`
when the target is in the repository, and full links for GitHub issues or
external sources.

## Index

The index in [`docs/findings/README.md`](../findings/README.md) contains exactly
one row per finding. Add or update the row in the same change as the finding.
Copy `Kind`, `Scope`, `Observed`, `Status`, and `Superseded by` from the finding's
metadata. Link `Finding` to the document and summarize its conclusion in one
concise sentence so an agent can evaluate relevance without opening it. Use
`none` under `Superseded by` for a current finding.

## Lifecycle

A finding is either `current` or `superseded`. Preserve a historically relevant
finding rather than silently rewriting a conclusion that later evidence
invalidates or replaces. To supersede it:

1. Change its `Status` to `superseded`.
2. Set `Superseded by` to the replacing finding or promoted artifact.
3. Update the finding's index row with both changes.
4. Add a backlink from the replacement when possible.

Purely editorial corrections do not require supersession. A change that
invalidates or replaces the conclusion does.

## Promotion

A finding may lead to an ADR, an implementation spec, a GitHub issue, an update
to `CONTEXT.md`, or a subsequent finding. Link the finding and promoted artifact
in both directions when possible.

The finding remains the evidence and historical context. The promoted artifact
becomes authoritative for its own concern: an ADR for a decision, a spec for an
implementation contract, a GitHub issue for work status, and `CONTEXT.md` for
domain terminology.

## Authoring procedure

1. Check the proposed result against every admission condition. This step is
   complete when durability, reuse, evidence, and future repository value are
   each demonstrated by the material to be written.
2. Choose `discovery`, `research`, or `experiment` from the primary evidence.
   This step is complete when exactly one allowed `Kind` describes the central
   conclusion.
3. Create the flat, date-prefixed file with the required sections and metadata.
   This step is complete when the path matches
   `docs/findings/YYYY-MM-DD-short-descriptive-slug.md` and every required field
   and heading is present.
4. Write the result with evidence, conclusion, implications, and limits kept
   distinct. This step is complete when each claim is traceable to evidence,
   inferences are labeled, and unsupported scope is recorded under
   `Limits and unknowns`.
5. Add exactly one row to the index. This step is complete when the row links to
   the finding and exposes its kind, scope, observation date, status,
   supersession target, and conclusion without opening the document.
6. Verify metadata, links, and lifecycle state. This step is complete when all
   metadata agrees with the body and index, every link resolves, and the status
   matches the supersession fields.
7. Connect any promoted or superseded artifacts. This step is complete when the
   forward links exist and backlinks have been added wherever the target can be
   edited.

## Completion

The work is complete when the finding satisfies the admission criterion, uses
the required filename and format, has exactly one accurate index row, keeps raw
bulk material outside `docs/findings/`, and has valid metadata, lifecycle state,
and links. Any work it proposes is represented and tracked by its authoritative
artifact rather than by the finding.
