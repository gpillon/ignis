# Issue tracker: GitHub (gpillon/ignis)

Issues for this repo live in **GitHub Issues** on `github.com/gpillon/ignis`,
managed with the `gh` CLI. Local specs and work-in-progress plans live under
`.scratch/<feature-slug>/` in this repo (one directory per feature: `spec.md`
+ `issues/`), and each GitHub issue links back to its local spec.

## Conventions

- One issue per ticket; issue titles mirror the local ticket files in
  `.scratch/`.
- The five triage labels (see `triage-labels.md`) are applied at issue
  creation; a ticket's local `Status:` line mirrors the label state.
- "Publish to the issue tracker" = create a GitHub issue in `gpillon/ignis`
  (`gh issue create`), with the local `.scratch/` spec as the issue body or a
  linked file.
- "Fetch the relevant ticket" = `gh issue view <n>`; read the linked
  `.scratch/<feature-slug>/` files for the full spec.

## Auth

Repo-local git credential helper (see `.git/config`:
`credential.https://github.com.helper = store --file ...`). The token lives in
the user's credential file, **never** in URLs or committed files.

## PRs as a request surface

Disabled by default — external PRs are not part of the triage queue. A
maintainer who wants that can enable it here.