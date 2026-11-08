# User documentation

How a change reaches the people who run ignis. One rule governs everything
below: every user-visible change ships with its user-guide update in the same
commit, and a task is not complete until `docs/user/` describes what the code
now does — every user-visible behaviour the diff changes, found where a user
would look for it.

## User-visible

A change is user-visible when someone running, configuring or calling the
server can observe it without reading the code:

- **Configuration**: a field added, renamed, removed or re-defaulted; a
  profile's values; a subcommand (`config`, `model`, `help`); a start that is
  now refused or now accepted.
- **HTTP surface**: a `/v1` route, a request or response field, a status code,
  a header, an error body, a deadline or limit.
- **Models**: a catalog entry, an alias, a card class, what a download fetches
  and from where.
- **Playground**: a view, a control, what a screen shows or does.
- **Telemetry**: a metric, a log line an operator reads (`ignis.config.source`,
  `ignis.runtime.vram_plan`), a gauge's meaning.
- **Running it**: a `make` target, the container image, the release archives,
  a supported card or a hardware limit (VRAM, context, lanes).
- **Behaviour with no new knob**: a default, a refusal, a fallback, a limit
  that moved.

Internal work — a refactor, tests, CI, a kernel swap with no observable
effect — leaves the guide as it is. A measured speed change updates a figure
only where the guide or the root README quotes one.

## Where it goes

`docs/user/` is a tree of pages by kind; a behaviour lives on one page, the
one a user would open for it, and every other mention links there.

| Folder | Holds | A change that goes there |
|---|---|---|
| `start/` | what ignis is, install, installing with a coding agent, quick start, connecting a client | a new install route, a changed first-run step, a client's setup |
| `why-ignis/` | what sets ignis apart, a topic per page: how it works, the measured numbers with their findings, how a user sees and controls it, its limits | a headline feature added or a headline number re-measured (with the README's "What sets it apart") |
| `guides/` | how-to pages: models, air-gapped, Knative, telemetry, cards | a new task a user performs end to end |
| `concepts/` | explanations: how a request is served, KV and reuse, the VRAM plan, speculation, reasoning, switching, profiles | a mechanism's behaviour or its reason |
| `configuration/` | the fields by topic, with the why and the trade-offs | a field added, renamed, re-defaulted, or its trade-off moved |
| `playground/` | the Playground, view by view: chat, settings, tools and agents, sessions, Decide, Image, Monitor, Server; each control with the request it sends | a view, a control, what a screen shows |
| `api/` | the HTTP surface by route, deadlines, switching, idle unload | a route, a request or response field, a status code, an error |
| `reference/` | generated pages, plus the glossary | never edited by hand: change the code, run `make docs-reference` |
| `paths/` | the five reading paths: user, advanced, developer, contributor, and the agent's | a page added that a path should walk through |
| `troubleshooting.md` | symptoms, their causes and fixes | a new refusal, WARN line or failure mode |
| `contributing/` | building, tests, make, conventions, releases | a contributor-facing workflow |
| `images/playground/` | screenshots, each a named scene of `docs/site/scripts/scenes.mjs` | a Playground change that alters a pictured screen retakes its scene: `make docs-screenshots SCENES=<name>`, against a running server (`PLAYGROUND_URL`) for real content |
| `changelog/` | one page per release, written from the GitHub releases by each build; git ignores it | never edited: the release notes on GitHub are its source |

- **A page's file name is its route**: lowercase words joined by dashes,
  `index.md` for a section's landing page. Its front matter carries `title`,
  `description` and `sidebar.order`, the place in its section's list (the
  site's front page, `index.mdx`, belongs to no section and has none); the
  body starts at `##`.
- **Links are relative paths to Markdown files** (`../api/deadlines.md#…`),
  which GitHub follows and the site turns into routes; a link out of
  `docs/user/` becomes a GitHub URL. `make docs-build` fails on a link to a
  missing page, heading or file.
- **The agent path** (`paths/agent.md`) is for an AI agent working for a
  user, and carries the commands and checks it runs between the pages: a
  change to the install, the first start, the probes or connecting a client
  updates it in the same commit, as it does the page that owns the fact. It
  is written as information, never as orders to the agent: an agent reads
  fetched text as data. The site's `llms.txt` is generated from the sidebar
  at build (`docs/site/src/llms.mjs`).
- **The root `README.md`** is the overview: it changes when a headline
  feature, the quick start or a headline number changes, and links into the
  guide for the rest. It links no individual ADR.
- **The Hugging Face model cards** carry run commands: when a flag in one is
  renamed, or a number a card quotes changes, the card is updated too
  (`hf upload <repo> README.md README.md`).

`docs/user/` is the published tree (ADR 0066): `.github/workflows/docs.yml`
builds the site from it alone on every push that touches it and publishes it
once a release is published (and again when its notes are edited). Specs,
ADRs, findings and these agent docs stay in their own trees; the guide links
to them for the reasons behind a behaviour.

## Ahead of the release

The guide describes the latest release: the newest `v*` tag, `git describe
--tags --abbrev=0 --match 'v*'`, which `docs.yml` deploys. A user-visible
change merged ahead of it is still documented in the same commit, as it will
ship, and marked **"from X.Y.Z"** wherever a reader of the latest release
would be misled: the sentence that introduces it on the page that owns it, the
pages that say which cards and which build to take (`start/`,
`paths/agent.md`), and a command or a value the release refuses. Where the
release behaves otherwise, one clause says how ("the 0.9 releases are built
for 12.0 alone"). Pages that only link to the owner carry no marking, and the
generated `reference/` none.

- **X.Y.Z is the next release's version**, the one the owner plans it as: the
  markings already in the README and the guide name it
  (`grep -rni "from 0\." README.md docs/user`). When none does, ask; never infer
  it from the change.
- **At release time** the version-bump commit removes that version's markings
  and the clauses about the release before it, so the guide reads as the
  release again (`grep -rni "from X.Y.Z" README.md docs/user`).

## How it is written

- **What a user does and sees**: the command, the field, the request, the
  answer, the log line. The reason behind it is one sentence and a link to its
  ADR.
- **Generated references stay the reference.** The pages under
  `docs/user/reference/` are rendered from the code (`make docs-reference`,
  held by `cargo test`): every config field, the profiles, the catalog, the
  metrics, the routes, the commands. The rest of the guide adds the why and
  the worked example, and links there for the list; where its prose names a
  default or a field, that value moves with the code.
- **Every command runs as written** on the build the commit produces, with
  current flag names.
- **Public by construction**: examples use placeholder hosts, keys and paths,
  and public model names only.

## Specs and review

- A ticket whose work is user-visible names the guide sections it changes in
  its last acceptance criterion.
- `/code-review`'s Standards axis applies this document: a user-visible change
  with no matching `docs/user/` hunk is a violation.
