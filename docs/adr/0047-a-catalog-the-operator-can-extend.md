# ADR 0047 — a model catalog the operator can extend, and a `model` command to fetch from it

## Status

Accepted (2026-10-10, owner decision in conversation — GitHub #313). Spec:
`docs/specs/model-download/02-catalog-and-model-command.md`. **Amends ADR
0033**: its pins, its `.part`/resume/verify transfer and its start-time
behaviour stand; what changes is the one option it rejected, "a `models.json`
next to the binary", which comes back in a narrower form.

## Context

ADR 0033 made `ignis-server` fetch its own model from a registry pinned in
the binary (`crates/server/src/download.rs`). Three weeks on, that registry
has one entry (the 27B) while the owner has published three repos: the 27B,
its huihui-abliterated variant, and Qwen3.8-Flash-Next with its MTP companion
container — four files, 72.8 GB. The other two are fetched by a `hf download`
line in a README, and the abliterated one is served under the same id as the
27B (`make UNCENSORED=1` swaps only the artifact path), so a client cannot
tell which of the two answered.

Two other needs came with it. An air-gapped or corporate operator reaches
Hugging Face through a mirror (Artifactory/Nexus HF proxies, an internal
static server), if at all, and may need a token. And an operator with
artifacts of their own wants the same verified fetch for them. ADR 0033
rejected an external model list outright — "a file an attacker can edit is
not a trust anchor" — and a sidecar is still fetched with no pin at all.

## Decision

**The registry becomes a catalog, in two layers with two different trust
standings.**

- **The built-in catalog** ships inside the binary as a data file
  (`include_str!`), in the same format an operator writes. Its entries are
  the owner's releases, and their pins are the trust anchor ADR 0033 made
  them: nothing outside the binary can change them.
- **The operator catalog** is a separate file named by one config field,
  `download.catalog`. Its entries have the standing `--model-artifact`
  already has — the operator's own word about the operator's own files — and
  never a built-in entry's: **an operator entry reusing a built-in id is
  refused at start**, so no editable file can silently re-pin an official id.
  It is a separate file, not a config section, because a catalog is shared
  across an organisation's machines while a config is one machine's, and
  because a list of structured entries has no flag or env-var form (ADR
  0046's one-declaration-per-field covers the path to it, not its contents).

**Where the bytes come from is a separate axis from which models exist.**
`download.endpoint` (default `https://huggingface.co`) is per machine; any
server answering Hugging Face's `{repo}/resolve/{revision}/{file}` route is
an endpoint, so a mirror of the owner's models needs no catalog at all —
the binary's pins still say whether the bytes are the owner's. A token is
`download.token` (`IGNIS_DOWNLOAD_TOKEN`, redacted like `server.api_key`);
`HF_TOKEN` is read only when the endpoint is Hugging Face itself, so a
personal Hugging Face credential is never handed to a mirror, and no token is
forwarded across a redirect to another host.

**An entry pins every file it needs, at a fixed revision.** Id (the served
id), repo, `revision`, the artifact to load, and a list of files each with
its byte count and SHA-256 — sidecars and companion containers included. A
sidecar is no longer fetched unverified. `revision` is required for every
entry; the built-in ones name a commit, so a republish never breaks a binary
already shipped. Small files are fetched first: a repo that cannot serve them
fails in a second, as ADR 0033's "sidecar first" intended.

**The served id is the entry's id.** Artifacts of one model family are told
apart by the served id, never by the family, which stays what the artifact's
own header says and is not declared in the catalog. The abliterated variant
is `qwen3.8-27b-abliterated`.

**`ignis-server model download [<id>…] | --all` and `model list`.** The
command shares the catalog and the transfer with the start-time fetch, but it
is an explicit yes: it asks nothing, ignores `download.enabled` (which gates
only the implicit start-time fetch), and runs on a build without `cuda` — the
gate ADR 0033 put on the start path keeps CI off the network by construction,
and a typed command cannot fire by accident. With no id it fetches the
configured `model.id`. `model verify` and `model convert` are reserved names,
not part of this decision.

**A catalog entry already on disk is a known model.** Its id joins the
switch's known models (an explicit `switch.known_models` entry wins), so a
`model download` is enough for a running server to switch to it. A switch
to a catalog model that is not on disk is refused as before and never starts
a download.

## Consequences

- Adding one of the owner's models is a row in the built-in catalog file and
  a new binary, as before; adding an operator's is a row in their own file,
  with no binary.
- A started server with `--model-artifact` and no `--model-id` is still
  served under its family's default id — an explicit path stays the
  operator's word, and no file name is matched against the catalog. The
  everyday start goes by id instead (`make UNCENSORED=1` serves
  `qwen3.8-27b-abliterated`).
- `qwen3.8-flash-next` downloads its MTP companion with it even when MTP is
  off: an entry is the release, not a load configuration.
- An air-gapped machine with no endpoint at all gets the files carried in
  (`model download --out <dir>` on any machine, no GPU needed) and finds them
  under `download.path` like any other.

## Considered Options

- **The catalog as a section of the config file** — rejected: a config is one
  machine's, a catalog an organisation's, and its entries have no flag form.
- **An operator entry overriding a built-in id** — rejected: it is the
  silent re-pin ADR 0033 exists to prevent; a mirror of the owner's models is
  an endpoint, a variant of them is a new id.
- **A generic per-file URL instead of an endpoint** — rejected: every mirror
  the owner named speaks Hugging Face's route, and a static server can lay its
  files out the same way.
- **Declaring the family in each entry** — rejected: a second source of truth
  that can only disagree with the artifact's header by mistake.
- **Recognising a named artifact path as a catalog entry by its file name**
  — rejected: a rename would change the served id without a word.
