---
status: accepted
---

# A model catalog the operator can extend, and a `model` command to fetch from it

The server fetched its own model from a list pinned in the binary (ADR 0033), and three needs outgrew it. The
owner's other releases were fetched by hand, and an operator with artifacts of their own wants the same verified
fetch for them. An air-gapped or corporate operator reaches Hugging Face through a mirror, if at all, and may need
a token. And two artifacts of one family were served under one id, so a client could not tell which of the two
answered. So the list becomes a **catalog, in two layers with two different trust standings**, where the bytes come
from becomes a setting of its own, and a `model` command fetches from it. ADR 0033's pins, its verified transfer
and its start-time behaviour stand.

- **The built-in catalog** ships inside the binary, in the same format an operator writes. Its entries are the
  owner's releases, and their pins are the trust anchor ADR 0033 made them: nothing outside the binary can change
  them.
- **The operator catalog** is a separate file named by one config field, `download.catalog`. Its entries have the
  standing `--model-artifact` already has, the operator's own word about the operator's own files, and never a
  built-in entry's: an operator entry reusing a built-in id is refused at start, so no editable file can silently
  re-pin an official id.
- **Where the bytes come from is a separate axis from which models exist.** `download.endpoint` is per machine,
  and any server answering Hugging Face's `{repo}/resolve/{revision}/{file}` route is an endpoint, so a mirror of
  the owner's models needs no catalog at all: the binary's pins still say whether the bytes are the owner's. A
  token is `download.token`, redacted as `server.api_key` is. `HF_TOKEN` is read only when the endpoint is Hugging
  Face itself, so a personal Hugging Face credential is never handed to a mirror, and no token is forwarded across
  a redirect to another host.
- **An entry pins every file it needs, at a fixed revision**: its id, its repo, its `revision`, the artifact to
  load, and a list of files each with its byte count and SHA-256, sidecars and companion containers included, so a
  sidecar is never fetched unverified. `revision` is required of every entry, and the built-in ones name a commit,
  so a republish never breaks a binary already shipped. Small files are fetched first: a repo that cannot serve
  them fails in a second. An entry is the release, not a load configuration: every file of it is fetched, a
  companion the load will not use included.
- **The served id is the entry's id.** Artifacts of one model family are told apart by the served id, never by the
  family. The family stays what the artifact's own header says: where a catalog entry states one, it is a claim
  the header verifies (ADR 0064). What an entry states of the cards it loads on, and the other ids it answers to
  there (its aliases), is ADR 0064's and ADR 0079's.
- **`ignis-server model download` and `model list`** share the catalog and the transfer with the start-time fetch.
  The download is an explicit yes: it asks nothing, ignores `download.enabled`, which gates only the implicit
  start-time fetch, and runs on a build without `cuda`. The gate ADR 0033 put on the start path keeps CI off the
  network by construction, and a typed command cannot fire by accident.
- **A catalog entry already on disk is a known model.** Its id joins the switch's known models, an explicit
  `switch.known_models` entry winning, so a `model download` is enough for a running server to switch to it. A
  switch to a catalog model that is not on disk is refused and never starts a download.
- **A named artifact stays the operator's word.** A server started with `--model-artifact` and no `--model-id` is
  served under its family's default id, and no file name is matched against the catalog.

## Considered Options

- **The catalog as a section of the config file.** A config is one machine's and a catalog is shared across an
  organisation's machines, and a list of structured entries has no flag or env-var form: ADR 0046's one
  declaration per field covers the path to the catalog, not its contents.
- **An operator entry overriding a built-in id.** It is the silent re-pin ADR 0033 exists to prevent: a mirror of
  the owner's models is an endpoint, a variant of them is a new id.
- **A generic per-file URL instead of an endpoint.** Every mirror the owner named speaks Hugging Face's route, and
  a static server can lay its files out the same way.
- **Recognising a named artifact path as a catalog entry by its file name.** A rename would change the served id
  without a word.

## Consequences

- Adding one of the owner's models is a row in the built-in catalog and a new binary; adding an operator's is a row
  in their own file, with no binary.
