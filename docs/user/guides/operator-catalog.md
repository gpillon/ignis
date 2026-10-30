---
title: "Your own models: an operator catalog"
description: "Add models to the catalog with a YAML or JSON file in the built-in catalog's format."
sidebar:
  order: 2
---

`--download-catalog` names a YAML or JSON file in the built-in catalog's own
format. Its entries are added after the built-in ones, fetched and verified the
same way, and listed as `operator`:

```yaml
models:
  - id: acme-qwen3.8-27b-ft          # the served id
    repo: acme/qwen-ft-ninfer          # <owner>/<name>
    revision: v1                       # commit, tag or branch; required
    artifact: acme_ft.ninfer           # the file the server loads
    # kind: speech                     # optional: "speech" or "diarizer" (see below); a text or image model states none
    card_class: 32                     # optional: the smallest card class it loads on, in GB: 16, 32 or 96
    # family: qwen3.8-27b              # optional: its model family, by the family's own id (see below)
    # aliases: [qwen3.8-27b]           # optional: the ids it answers to on a smaller card: its family's default id, or another entry's of its family
    files:                             # every file, all pinned
      - { name: acme_ft.ninfer.graft.json, bytes: 48211,       sha256: 9c1e… }
      - { name: acme_ft.ninfer,            bytes: 19406942468, sha256: abb1… }
```

The whole file is refused — the start, and every `model` command — when it
cannot be read, or when an entry has an unknown key, an empty or duplicate id
(case-insensitive), **an id the built-in catalog already has**, a `repo` not of
the form `<owner>/<name>`, an empty `revision`, an `artifact` not among its
`files`, no sidecar (the artifact's name plus `.graft.json` or
`.conversion.json`), a `sha256` that is not 64 lowercase hex digits, a `bytes`
of 0, a file name that is not a plain name (a path separator, `..`, an
absolute path), a `card_class` that is not 16, 32 or 96, a `family` that is
not a model family's default id (`qwen3.8-27b`, `qwen3.8-flash-next`,
`ignis-image-2.1-turbo`, `nemotron-3-diarization`) or that contradicts what
the entry's id or an alias says, no `family` where one is required (below), or
an alias that is neither a model family's default id nor another entry's id
(matched exactly), is the entry's own id, names another family than the
entry's, names an entry whose family nothing states, or is already another
entry's — a built-in one's included. A relative `--download-catalog` set in a config file is read
from that file's directory; from a flag or an env var, from the working
directory. An entry with no `card_class` states no class: it is never refused
by class, so a catalog written before card classes still starts, and only the
VRAM plan says it does not fit.

`kind: speech` is the one kind a catalog states: it makes the entry a
[speech model](../api/transcriptions.md) (a Qwen3-ASR decoder and its audio
encoder), which is never switched to. Such an entry takes no sidecar; instead
its `artifact` is the decoder, a `.gguf`, and its `files` hold the encoder, a
`mmproj-*.gguf` beside it — the catalog is refused without one. `kind: diarizer` makes the entry the
[diarizer](../configuration/speech.md#the-diarizer) that gives a live session its
speakers: one GGUF, the `artifact`, and nothing else in `files` (no sidecar,
no card class), which is never switched to and is loaded beside the speech model.
`diarizer.model: auto` is the catalog's one diarizer entry. Any other
`kind` is refused: a text or an image model's kind is read from its artifact.

A built-in id cannot be re-pinned by a file: a mirror of the owner's models is
an endpoint, a variant of them a new id. Your entry may answer to a built-in
default id as an alias — your own 16 GB build answering to `qwen3.8-27b` —
since the built-in entry keeps its id and pins and still serves it wherever it
fits: the alias is consulted only on a card below its class, and the answer
names your entry.

## An entry's family, and an alias of your own

An alias resolves only within one model family, so the catalog must know the
family of every entry an alias involves: the entry that lists it, and the
entry whose id it is. An id or an alias that is a family's default id says it:
`qwen3.8-27b-16g`'s alias `qwen3.8-27b` makes it a Qwen3.8-27B. For any other,
the entry states `family`, the family's default id
([ADR 0074](../../adr/0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md)).
That is how an alias can name another entry's id: your own fine-tune and its
16 GB build answer to one id on every card, as `qwen3.8-27b-16g` answers to
`qwen3.8-27b`.

```yaml
models:
  - id: acme-27b-ft                    # names no family: the 16 GB build's alias needs it to state one
    repo: acme/qwen-ft-ninfer
    revision: v1
    artifact: acme_ft.ninfer
    card_class: 32
    family: qwen3.8-27b
    files: [ … ]
  - id: acme-27b-ft-16g
    repo: acme/qwen-ft-16g-ninfer
    revision: v1
    artifact: acme_ft_16g.ninfer
    card_class: 16
    family: qwen3.8-27b                # its id and its alias name no family either
    aliases: [acme-27b-ft]             # answers to acme-27b-ft on a card below 32 GB
    files: [ … ]
```

On a 16 GB card a start, a switch and a request naming `acme-27b-ft` are
served by `acme-27b-ft-16g`, under its own id; on a 32 GB card or larger, and
on a card whose class is not known, by `acme-27b-ft` itself. `GET /v1/models`
and `GET /v1/models/catalog` list `acme-27b-ft` as unavailable on the 16 GB
card, the reason naming the entry that serves it there. The built-in `qwen3.8-27b-abliterated` states
`family: qwen3.8-27b` for the same reason: an alias may name it.

- `family` is required only where an alias involves the entry and neither its
  id nor its aliases is a family's default id. An entry no alias involves
  needs none, so a catalog written before `family` existed loads unchanged.
- Stated, it must agree with the entry's id and its aliases when they name a
  family, and an alias must name an entry of the same family: the catalog is
  refused otherwise, naming the entry.
- `family` is the catalog's word, and the artifact's header stays the
  authority. The entry that answers to an alias must hold an artifact whose
  header names the alias's family, read before anything stops once the
  artifact is on disk. One whose header names another family does not answer:
  the switch or the request is refused `409 model_above_card_class`, naming
  both families, and a start is refused before the load. Named by its own id,
  the entry is served as it is.
