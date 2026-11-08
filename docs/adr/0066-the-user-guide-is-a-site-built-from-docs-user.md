---
status: accepted
---

# ADR 0066 — the user guide is a Starlight site built from `docs/user/` alone, published on a release tag

## Status

Accepted (2026-10-24, owner decisions in the session that opened GitHub #347):
- an Actions workflow, not GitHub Pages' deploy from `/docs`;
- Starlight, with VitePress as the fallback if it could not read `docs/user/` in place;
- English only;
- published on a `v*.*.*` tag;
- four reading paths: user, advanced user, developer (who builds on the API), contributor (who works on ignis).

Amended 2026-10-24 (GitHub #348): the site deploys once the release is published, not when its tag is pushed,
and two parts of it are produced by tools rather than written: the Changelog section and the Playground's
screenshots. See **Amendment** below.

Related:
- [ADR 0036](0036-the-v1-surface-documents-itself.md): the `/v1` surface documents itself;
- [ADR 0046](0046-one-declaration-per-config-field.md): one declaration per config field;
- [`docs/agents/user-docs.md`](../agents/user-docs.md): the rule that keeps the guide current.

## Context

- **The user guide was one 2,045-line file**, `docs/user/README.md`, beside the 645-line root README. It was
  complete and current — 158 of 1,102 commits since 2026-10-01 touched it — but it had no search and no
  navigation beyond anchors, and nothing told a first-time user where to start.
- **Some of it the code already knows.** The field table prints every config field (`help --fields`), the
  OpenAPI document lists every route (ADR 0036), and the built-in profiles, the built-in catalog and the metrics
  exposition are tables in the binary. A hand-copied list of any of them drifts.
- **`docs/` holds more than the guide**: ADRs, specs, findings, reviews and the agents' conventions. GitHub
  Pages' deploy from `/docs` would publish all of it.
- **The release workflow ignores `docs/**`** (ADR 0032, amended 2026-10-12): a docs-only push starts no build.

## Decision

1. **`docs/user/` is the guide and the only published tree.** It is a tree of Markdown pages by kind — start,
   guides, concepts, configuration, Playground, API, reference, troubleshooting, contributing — and four reading
   paths over the same pages. A page reads on GitHub as it is; `docs/user/README.md` is the folder's index there
   and is not a page of the site.
2. **`docs/site/` builds it with Starlight**: the guide is its one content collection, loaded where it lives;
   search is Pagefind's; the look is the Playground's (tokens, fonts, the kiln band and the wordmark).
3. **One link syntax for both readers.** A page links with a relative path to a Markdown file. At build time a
   page link becomes its route and a link out of `docs/user/` becomes the GitHub URL of the file at the published
   ref; a link to a missing page, a missing anchor or a missing file fails the build.
4. **What the code can write is generated and committed.** `ignis_server::user_docs` renders the config fields,
   the profiles, the catalog, the metrics, the HTTP routes and the command line into `docs/user/reference/`;
   `cargo test` holds the committed pages equal to the rendering and `make docs-reference` rewrites them.
   Committed rather than built in the workflow, so the pages are in the diff that changes them and the workflow
   needs no Rust build.
5. **`.github/workflows/docs.yml` builds on every push that touches the guide or the site, and deploys on a
   `v*.*.*` tag** (and on `workflow_dispatch`), so the published guide describes the released binary.
   (Amended: once the tag's release is published.)

## Considered options

- **Deploy from `/docs`.** Rejected: it publishes the internal trees with the guide.
- **mdBook.** Rejected once the goals were a wiki with the Playground's look, reading paths and components: its
  theme is a template override and it has no components.
- **Generating the reference in the workflow from a built binary.** Rejected: the workflow would compile the
  server, and a stale page would be invisible to `cargo test` and to review.
- **Publishing on every push to main.** Rejected: the site would describe unreleased behaviour.

## Consequences

- Every user-visible change updates `docs/user/` in the same commit (`AGENTS.md`, `docs/agents/user-docs.md`).
- A page's file name is its route: lowercase words joined by dashes, `index.md` for a section's landing page.
- A test reads `docs/user/reference/`, an exception to the release workflow's "no test reads docs/": a code
  change that makes those pages stale runs the release workflow's `cargo test`, and a push that edits only the
  pages runs that one test in `.github/workflows/docs-reference.yml`.
- The landing page's cards link by route, outside what the link plugin sees; a test holds each to a page.
- The site's dependencies are pinned in `docs/site/package-lock.json`; the workflow installs with `npm ci`.

## Amendment (2026-10-24, GitHub #348)

1. **The Changelog section is read from the GitHub releases at build time**, one page per `v*.*.*` release,
   newest first (`docs/site/src/changelog.mjs`). The notes live on GitHub only: the release workflow writes the
   install text, and what changed is written by hand into the release after it is published. A page generated
   and committed would be stale the moment the notes are edited, and would add a step to every release. The
   pages land in `docs/user/changelog/`, which git ignores. A CI build that cannot read the releases fails; a
   local one writes a placeholder.
2. **The site deploys when the release workflow's run for a `v*.*.*` tag succeeds** (`workflow_run`), and again
   when that release is edited (`release: edited`), not on the tag push: at the push the release does not
   exist yet, so its notes would be missing from the section. The release is created with `GITHUB_TOKEN`,
   whose `release: published` starts no workflow. Only the newest release is deployed; a re-run or an edit of
   an older one builds and stops. `workflow_dispatch` still deploys the chosen ref.
3. **The Playground's screenshots are taken by a script** (`make docs-screenshots`,
   `docs/site/scripts/screenshots.mjs`) that drives the Playground in the installed Chrome through
   playwright-core, against the dev mock (no GPU, the same answers each run) or a running server
   (`PLAYGROUND_URL`). Each picture is a named scene, and a change that alters a pictured screen retakes its
   scene. The pictures are committed: the workflow has no Chrome and no server to take them with.
