# AGENTS.md

This file is for agents working **on** ignis: its code, its guide, its issues.
An agent here for a user of ignis (to explain it, install it, or connect a
client to it) wants [the agent path](docs/user/paths/agent.md) instead.
An agent working in a fork of ignis reads
[Working in a fork](docs/user/contributing/forks.md) first: what goes upstream.

## Agent skills

### Issue tracker

GitHub Issues on `gpillon/ignis`, through the `gh` CLI: a spec is an issue and
its tickets are sub-issues with native blocking, planned on a public project
board. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles and the two categories as same-name labels, plus the
repo's `idea` and `quick` markers. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.

## Repo conventions

### Scratch

`.scratch/` is the untracked scratchpad, local to each clone. What must outlive
the task moves into the repo, to the home Artifact boundaries in
`docs/agents/findings.md` names.

### Coding standards

Writing, reviewing or merging a change, or running a skill that writes files:
`CODING_STANDARDS.md`.

### Findings

Recording or consulting a durable finding from discovery, research or an
experiment: the index `docs/findings/README.md`, and `docs/agents/findings.md`.

### ADRs

Recording, amending, numbering or citing a decision: the index
`docs/adr/README.md`, and `docs/agents/domain.md`.

### Testing

Every code change ships with a test, and the task is done only when
`cargo test` passes workspace-wide. Before writing a test or reporting a code
change done: `docs/agents/testing.md`.

One run fits the card at a time, and the loser dies with no diagnostic. Before
anything on the GPU (the profile, a bench run, a gate leg, one `--ignored`
test): `make gpu-status`. If your own model is served from this card (a local
agent), the process holding it is you: never launch GPU work or stop it; hand
the GPU step to the user (the recipe: `docs/agents/testing.md`). If you run
elsewhere, whatever holds the card is someone else's: wait or ask, never stop
a process you did not start. Not sure which: ask.

### User documentation

Every user-visible change ships with its `docs/user/` update in the same
commit. Before reporting a change done: `docs/agents/user-docs.md`.

### Make

An agent runs the server as a daemon, `make start` then `make stop`: a
foreground target holds the shell until Ctrl+C. Rebuilding, scripting or
restarting it, or changing the `Makefile`: `docs/user/contributing/make.md`.

### Orchestrated builds

Coordinating implementer agents on a spec's tickets:
`docs/agents/orchestrated-builds.md`.
