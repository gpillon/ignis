---
status: accepted
---

# The user guide is a Starlight site built from `docs/user/` alone, published with each release

`docs/user/` is the guide and the only published tree. `docs/site/` builds it with Starlight, which gives it the
search (Pagefind's), the navigation and the place to start that a single file lacked, and the site is published when
a release is, so the published guide describes the released binary. `docs/` holds more than the guide (ADRs,
findings, the agents' conventions), so the site is built by a workflow from that one folder. The guide is English
only.

- **One source for both readers.** A page reads on GitHub as it is, and links with a relative path to a Markdown
  file: the build turns a page link into its route and a link out of `docs/user/` into the GitHub URL of the file at
  the published ref, and a link to a missing page, a missing anchor or a missing file fails the build. Reading paths
  walk the same pages. The site loads the guide where it lives, as its one content collection.
- **What the code can write is generated and committed.** `ignis_server::user_docs` renders `docs/user/reference/`
  from the tables the binary runs on (the config fields of ADR 0046, the routes of ADR 0036, the profiles, the
  catalog, the metrics, the command line), since a hand-copied list of any of them drifts, and `cargo test` holds
  the committed pages equal to the rendering. Committed rather than built in the workflow, so the pages are in the
  diff that changes them and the workflow needs no Rust build.
- **The site deploys once the tag's release is published, and again when that release is edited**, not when the tag
  is pushed: at the push the release does not exist yet, so its notes would be missing from the Changelog section.
  Only the newest release is deployed: a re-run or an edit of an older one builds and stops.
- **The Changelog section is read from the GitHub releases at build time and never committed.** The notes live on
  GitHub only and are written by hand into the release after it is published, so a committed page would be stale the
  moment the notes are edited, and would add a step to every release.
- **The Playground's screenshots are taken by a script and committed** (`make docs-screenshots`). Each picture is a
  named scene, and a change that alters a pictured screen retakes its scene; the workflow has no Chrome and no
  server to take them with.

## Considered Options

- **GitHub Pages' deploy from `/docs`**: rejected. It publishes the internal trees with the guide.
- **mdBook**: rejected once the goals were a wiki with the Playground's look, reading paths and components. Its
  theme is a template override and it has no components.
- **Generating the reference in the workflow from a built binary**: rejected. The workflow would compile the server,
  and a stale page would be invisible to `cargo test` and to review.
- **Publishing on every push to main**: rejected. The site would describe unreleased behaviour.

## Consequences

- Every user-visible change updates `docs/user/` in the same commit (`AGENTS.md`, `docs/agents/user-docs.md`).
- A test reads `docs/user/reference/`, the exception to the release workflow's rule that no test reads `docs/`
  (ADR 0032): a code change that makes those pages stale runs the release workflow's `cargo test`, and a push that
  edits only the pages runs that one test in `.github/workflows/docs-reference.yml`.
