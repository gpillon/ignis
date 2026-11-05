# AGENTS.md

This file is for agents working **on** ignis: its code, its guide, its issues.
An agent here for a user of ignis (to explain it, install it, or connect a
client to it) wants [the agent path](docs/user/paths/agent.md) instead.
An agent working in a fork of ignis reads
[Working in a fork](docs/user/contributing/forks.md) first: what goes upstream.

## Agent skills

### Issue tracker

GitHub Issues on `gpillon/ignis`, through the `gh` CLI: a spec is an issue and
its tickets are sub-issues with native blocking. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles and the two categories as same-name labels, plus the
repo's `idea` and `quick` markers. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.

## Repo conventions

### Scratch

`.scratch/` is the untracked scratchpad, local to each clone, for temporary
artifacts, experiments, raw logs and workflow output. When something there
must outlive the task, it moves into the repo: Artifact boundaries in
`docs/agents/findings.md` names the usual homes.

### Coding standards

Writing, reviewing or merging a change, or running a skill that writes files:
`CODING_STANDARDS.md`.

### Findings

When recording or consulting durable findings from discovery, research, or
experiments, start from `docs/findings/README.md`.

### ADRs

When recording a decision in `docs/adr/`, write it in the `/domain-modeling`
skill's short format, with the bare title as its heading: the release notes
take the number from the file name. An ADR takes the next free number when it
merges into `main`; if `main` took that number first, the ADR is renumbered
with every line that cites it (b61a224f). When a skill's convention and the
repo's disagree: ADR 0070 lists where the repo differs.

### Testing

Every code change ships with a test, and the task is done only when
`cargo test` passes workspace-wide. Before writing a test or reporting a code
change done: `docs/agents/testing.md`.

Before launching **anything** on the GPU (the profile, a bench run, a gate
leg, one `--ignored` test): check with `make gpu-status` that nothing else
holds the card, from any worktree or agent session. The 5090 fits one run at
a time, and the loser dies with no diagnostic.

### User documentation

Every user-visible change ships with its `docs/user/` update in the same
commit. Before reporting a change done: `docs/agents/user-docs.md` says
whether it is user-visible, where its guide update goes and how it is written.

### Make

An agent runs the server as a daemon, `make start` then `make stop`: a
foreground target holds the shell until Ctrl+C. `start` and `run` launch only
the last build and refuse a stale one, naming the changed files; `make build`
is the fix. Before changing the Makefile: ADR 0027.

Rebuilding a running server on Windows: `make stop`, then `make build`, then
`make start`, each only once the last succeeded, since Windows locks a running
exe and `make build` fails on it, as `make redeploy` does. Scripting
`make start`: redirect it to a file from bash (`> file 2>&1`), since the
server inherits a pipe (`| Tee-Object`, `| tail`) and holds it open for as
long as it runs. Restarting a server whose log holds an incident: copy
`ignis-server.log` and its `.err` out of `.scratch/serve/` first, since
`make start` rewrites both.

### Orchestrated builds

Coordinating implementer agents on a spec's tickets:
`docs/agents/orchestrated-builds.md`.
