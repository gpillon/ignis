# Findings

This document is the single source of truth for creating, updating,
correcting, superseding and archiving material in `docs/findings/`, and for
its generated index. A finding preserves knowledge and
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
| Spec, and its tickets' acceptance criteria | the spec's GitHub issue and its sub-issues (before 2026-10-25, a file under `docs/specs/`) |
| Architectural decision | `docs/adr/` |
| Byte-level, ABI or wire-format contract that code cites as normative | `docs/formats/` |
| Domain term or concept | root `GLOSSARY.md` |
| Temporary material, raw logs, and experiment output | `.scratch/` |
| Reusable finding that is not yet a decision | `docs/findings/` |

One task may produce both temporary evidence in `.scratch/` and a durable
synthesis in `docs/findings/`. The finding links to useful raw material rather
than copying voluminous logs or output.

## File and document format

`docs/findings/` stays flat, and a finding's file never moves or changes its
name: the guide and the other records link findings, and the guide's build
refuses a dead link. Archiving is a status and a section of the index, never a
folder. Name each finding `YYYY-MM-DD-short-descriptive-slug.md`, using the
observation date and a lowercase hyphenated slug.

Every finding uses this format:

```markdown
# Title

- Kind: discovery | research | experiment
- Status: current | corrected | superseded | archived
- Observed: YYYY-MM-DD
- Last verified: YYYY-MM-DD
- Scope: component / subsystem
- Topic: <slug>
- Related: issue, ADR, spec, finding, or `none`
- Superseded by: link(s) or `none`
- Corrected by: link(s)
- Summary: one sentence, at most 240 characters

## Question

## Evidence

## Finding

## Implications

## Limits and unknowns

## Follow-ups
```

The header is that list, in that order, between the title and the first blank
line. `Corrected by` is there only when `Status` is `corrected`. A long `Scope`
or `Related` may wrap onto indented continuation lines; `Summary` stays on one
line.

Use `Evidence` for observations, identifiable sources, measurements, commands,
and other verifiable results. In `Finding`, distinguish observed facts from
inferences explicitly. Use `Limits and unknowns` to state what the evidence
does not establish. `Follow-ups` may link future work, while GitHub remains the
source of truth for its status. Keep raw logs and voluminous output in
`.scratch/` and reference them when useful.

A profiler capture is not raw material you can commit. An Nsight Systems
`.nsys-rep` (and its `.sqlite`) records the profiled process's **environment**,
so it carries every API key that was set when the run started — GitHub's push
protection caught exactly that here. Commit the command that produced the
capture and the report derived from it, and leave the capture on disk;
`.gitignore` refuses the file types that are known to carry one.

## Header fields

- `Observed` is the date of the observation, the one the file name carries.
- `Last verified` is the date the claim was last checked against evidence: a
  rerun, a new measurement, a reread of the code it describes. It equals
  `Observed` until such a check, and changes only with one, never for an
  editorial edit or a change of status. It is a bare date: what was checked
  again goes in the body.
- `Topic` is one slug from the table below, the area a reader would look
  under. The index lists the topics in this order. Adding a topic means adding
  it to `TOPICS` in `mk/findings-index.py` and to this table, in the same
  change; the index check refuses the two out of step.
- `Summary` is one sentence of at most 240 characters, on one line: the
  conclusion as it holds now, with its key number and the condition it was
  measured under (card, model, width, context), so that a reader can judge
  from the index alone whether to open the finding. A `corrected` finding's
  summary is the revised conclusion; an `archived` one's says why the line of
  research is closed.
- `Related`, `Superseded by` and `Corrected by` use repository-relative links
  when the target is in the repository, and full links for GitHub issues or
  external sources. A relative link in `Superseded by` or `Corrected by` must
  resolve to a file that exists.

| Topic | What it holds |
|---|---|
| `decode-27b` | the 27B decode round, its kernels and speculation (DFlash2) |
| `flash-next` | Flash-Next: MoE experts, residency, MTP, verify rows |
| `prefill` | prefill: chunk width, TTFT, prompt reuse |
| `kv-and-state` | the KV cache, its formats and the state tiers (KV-RAM, KV-disk, live moves) |
| `artifacts-and-gates` | model artifacts, the 16 GB trellis and the quality gates |
| `host-and-gpu` | the host, VRAM and GPU generations (WDDM, PCIe, NVML, sm_86 and sm_89, other backends) |
| `decide-options` | `/v1/decide`: options, number, scalar, prompt and reuse |
| `decide-pointing` | `/v1/decide`: point and box, attention heads |
| `decide-locate` | `/v1/decide`: locate |
| `reasoning-and-context` | reasoning, loops and long context |
| `vision` | vision encode and fan-out |
| `speech` | speech: ASR, diarization, notes |
| `image` | the image model |
| `harness-and-build` | the test harness, bench, build and transport |

## Index

[`docs/findings/README.md`](../findings/README.md) lists the usable findings,
`current` and `corrected`, one section per topic, newest first: the date, the
linked title and the `Summary`, and for a corrected one the link to what
revised it. [`docs/findings/ARCHIVE.md`](../findings/ARCHIVE.md) lists the
`superseded` and `archived` ones the same way, next to what replaced them.

`mk/findings-index.py` generates both from the findings' headers: run
`make records` (or `python mk/findings-index.py --write`, `python3` on Linux)
in the same change as the finding, and never edit either index by hand.
`make records-check` (`--check`), which CI runs, refuses a header field
missing, out of order or outside its vocabulary, a `Summary` missing, on more
than one line or over 240 characters, a status whose links do not match it, a
link in `Superseded by` or `Corrected by` to a file that does not exist, and
an index out of sync with the findings. `--selftest` is the tool's own test.

## Lifecycle

A finding has one of four statuses. Preserve a historically relevant finding
rather than silently rewriting a conclusion that later evidence revises,
invalidates or replaces. Purely editorial corrections change no status.

- `current`: usable as it is. `Superseded by: none`.
- `corrected`: the conclusion still holds and a reader may act on it, but a
  number, a threshold or a part of it was revised later. `Corrected by` links
  what revised it (a later finding, an ADR, an issue or a commit), the
  `Summary` states what holds now, and `Superseded by` is `none`. Keep the
  original text and mark the revised passage with a dated note that links the
  same record; the status, not a banner, tells the reader.
- `superseded`: the conclusion no longer holds, or another record now answers
  its question. `Superseded by` links the replacement, a finding, an ADR or
  the promoted artifact, with a relative link; add a backlink from the
  replacement when possible. The `Summary` keeps what the finding concluded.
- `archived`: a closed line of research: a negative result, a design not
  implemented, an exploration not followed. Nothing replaced it, so
  `Superseded by` is `none`, and the `Summary` says why the line is closed.

To choose: a finding whose conclusion a reader can still act on, after a
revised number, is `corrected`; one whose conclusion would mislead them, or
that another record has taken over, is `superseded`; one that closed its own
line of research, without a successor, is `archived`. A change of status edits
the header (its links and, when needed, the `Summary`) and regenerates the
index in the same change.

## Promotion

A finding may lead to an ADR, a spec, a GitHub issue, an update
to `GLOSSARY.md`, or a subsequent finding. Link the finding and promoted artifact
in both directions when possible.

The finding remains the evidence and historical context. The promoted artifact
becomes authoritative for its own concern: an ADR for a decision, a spec for
what to build, a GitHub issue for work status, and `GLOSSARY.md` for
domain terminology.

## Authoring procedure

1. Check the proposed result against every admission condition. This step is
   complete when durability, reuse, evidence, and future repository value are
   each demonstrated by the material to be written.
2. Choose `discovery`, `research`, or `experiment` from the primary evidence.
   This step is complete when exactly one allowed `Kind` describes the central
   conclusion.
3. Create the flat, date-prefixed file with the required sections and header.
   This step is complete when the path matches
   `docs/findings/YYYY-MM-DD-short-descriptive-slug.md` and every required field,
   in its order, and every heading is present.
4. Write the result with evidence, conclusion, implications, and limits kept
   distinct. This step is complete when each claim is traceable to evidence,
   inferences are labeled, unsupported scope is recorded under
   `Limits and unknowns`, and the `Summary` states the conclusion with its key
   number and condition in one sentence of at most 240 characters.
5. Regenerate the index with `make records`. This step is complete when the
   change carries the regenerated `docs/findings/README.md` and `ARCHIVE.md`,
   and neither was edited by hand.
6. Verify the header, links, and lifecycle state. This step is complete when
   `make records-check` passes, the header agrees with the body, every link
   resolves, and the status matches its `Superseded by` and `Corrected by`.
7. Connect any promoted, superseded or corrected artifacts. This step is
   complete when the forward links exist, backlinks have been added wherever
   the target can be edited, and a corrected passage carries its dated note.

## Completion

The work is complete when the finding satisfies the admission criterion, uses
the required filename and format, keeps raw bulk material outside
`docs/findings/`, passes `make records-check`, and ships with the regenerated
index in the same change. Any work it proposes is represented and tracked by
its authoritative artifact rather than by the finding.
