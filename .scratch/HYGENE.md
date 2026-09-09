# `.scratch` hygiene review (2026-09-08)

Two passes: (1) structural cleanliness of `.scratch`, (2) whether content
written into `.scratch` actually got carried to its permanent home (GitHub
issue, ADR, `CONTEXT.md`, `ROADMAP.md`) or dead-ended here. Review only — no
files moved or changed as part of this audit.

## Pass 1 — structure

Placement of `.scratch` itself is correct and documented (`AGENTS.md`,
`docs/agents/issue-tracker.md`): GitHub owns issues/status/blocking,
`.scratch/<feature>/specs/` owns durable spec text, `.scratch/` root also
holds temp artifacts/experiments.

7 of 8 feature dirs are clean and consistent: `artifact/`, `bench/`, `core/`,
`kernel-abi/`, `kernel-port/`, `runtime/`, `server/` — each `spec.md` +
`specs/NN-name.md`.

Inconsistencies:

1. **`logging` feature breaks the pattern.** No `.scratch/logging/` dir —
   loose at scratch root instead: `Ignis Structured Logging Specification.md`
   (spec.md equivalent), `logging-roadmap.md` (phase plan), and
   `issue-phase1..4-logging-*.md` (specs/NN-name.md equivalent, but named
   "issue-*" even though these are specs, not issues, per your own
   convention).
2. **`spec-cli-config.md`** — lone spec, no feature folder, inverted name
   order (`spec-X` vs `X/spec.md`).
3. **`runtime/`** mixes durable `spec.md`/`specs/` with 6 gitignored log
   files (`*.log`, `.gitignore:13`) — harmless for git, just visual noise;
   the only feature dir doing this.
4. **`debug-67-server.{err,out}.log`** — 0 bytes, gitignored, disk-only
   clutter.

## Pass 2 — did the content get carried out, or is it stuck here?

Checked every `specs/NN-*.md` against `gh issue list/view`, every feature
`spec.md` against `docs/adr/*` and `CONTEXT.md`, and `ROADMAP.md`'s
master-ticket column against actual issue state.

**Integrated (verified, no action needed):**

- All 7 clean feature dirs: every spec carries `GitHub: #N`, all closed
  except active Phase-2 work (#63). kernel-abi's specs self-mark
  `> SUPERSEDED (2026-09-05)` pointing at `runtime/specs/01` (#36,
  closed) — intentionally superseded, not stuck.
- Logging: phases map 1:1 to issues #78–81 (open, `ready-for-agent`,
  not started — expected). ADRs 0011/0012/0013 exist and match. `CONTEXT.md`
  has the `## Observability` section reflecting the decisions. Loose file
  layout (see Pass 1 #1) but the content did land.
- `spec-cli-config.md` — no `GitHub:` header (tagging gap), but content
  matches closed issue #77 by title/git-log. Integrated, just mislabeled.
- `ROADMAP.md` master-ticket column — #36/#63/#64/#65/#66 all exist, right
  states. "to write when X lands" placeholders are for future spec docs,
  not missing tickets — fine.
- `REVIEW-2026-09-05.md` — its §7 decisions trace to ADR 0009/0010 or are
  baked into the ROADMAP phase policy. Meant to stay as the rationale
  doc; ROADMAP is the live pointer.

**STUCK IN SCRATCH — written, never carried anywhere permanent:**

1. **`KERNEL-WIP.md`** — 4 open design questions (GDN state layout
   FP32-vs-bf16, warp-tiling design, Q/K L2-norm placement,
   chunked-vs-per-token prefill). This is exactly the design surface for
   Phase 2's chunked-GDN work (issues #84, #86; spec
   `runtime/specs/02-real-prefill.md`). Checked both — zero mention of
   KERNEL-WIP, warp-tiling, or state-layout. The investigation's
   conclusions never made it into the tickets that need them.

2. **`ROADMAP-kernel-perf.md`** — cited 3× by `REVIEW-2026-09-05.md`
   (§2.1, WI-K2) as the source-of-truth for kernel perf work items. The
   file does not exist anywhere in the repo. Either lost or never written.

## Suggested follow-ups (not yet actioned)

- Fold `KERNEL-WIP.md`'s open questions into `runtime/specs/02-real-prefill.md`
  and/or issue #86, then retire the file.
- Chase down or recreate `ROADMAP-kernel-perf.md`, or strip the dead
  references from `REVIEW-2026-09-05.md` if the work items moved elsewhere.
- Nest logging into `.scratch/logging/spec.md` + `specs/NN-name.md` to match
  the other 7 features; rename `issue-phaseN-*` → `NN-name.md`.
- Give `spec-cli-config.md` a home (own folder, or fold into `server/` since
  CLI config is server-adjacent) and add the missing `GitHub: #77` header.
