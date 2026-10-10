# 02 — the model catalog, a configurable endpoint, and `ignis-server model download`

GitHub: #313. ADR 0047 (amends ADR 0033). Builds on
`docs/specs/model-download/01-model-download.md`: the start-time decision,
the `.part`/resume/verify transfer and the `[y/N]` question stay; this spec
grows what they run over. Glossary: `CONTEXT.md` § *Models, catalog &
download* (**Model family**, **Served id**, **Catalog**, **Catalog entry**,
**Endpoint**).

## Problem Statement

`download.rs`'s `REGISTRY` holds one entry, the 27B, while three repos are
published: the 27B, its abliterated variant and Flash-Next (with its MTP
companion container). The other two are fetched with `hf download` by hand,
and the abliterated one is served as `qwen3.8-27b`, indistinguishable from the
27B to a client. The URL is fixed to `https://huggingface.co/…/resolve/main/…`
with no token, so a mirror, a private repo or an air-gapped machine cannot use
it, and an operator's own artifacts cannot use the verified fetch at all. The
sidecar is fetched with no pin. Nothing fetches a model except a server start.

## Solution

- The registry becomes the **catalog**: a **built-in catalog** compiled into
  the binary from a data file, plus an optional **operator catalog** file
  named by `download.catalog`, in the same format.
- A **catalog entry** pins every file it needs (sidecars and companion
  included) with byte count and SHA-256, at a fixed `revision`, and names the
  artifact to load. Its `id` is the served id.
- `download.endpoint` and `download.token` say where the bytes come from.
- `ignis-server model download [<id>…] | --all` and `ignis-server model list`.
- A catalog entry on disk is a known model for the switch.

## Implementation Decisions

### Module

`crates/server/src/download.rs` becomes a module directory:
`download/mod.rs` (the start-time decision `artifact_source` and the question),
`download/catalog.rs` (entry types, parsing, validation, the merge of the two
layers), `download/transfer.rs` (`Downloader`), and the built-in catalog as
`download/catalog.yaml`, embedded with `include_str!`. `REGISTRY` becomes
`BUILT_IN_CATALOG` (parsed once), `ModelEntry` becomes `CatalogEntry` with
owned fields; `Copy` and the `&'static ModelEntry` in
`ArtifactSource::Download` go. `artifact_source` stays a pure function over
the merged catalog (spec 01 AC1).

### The catalog file

YAML or JSON by extension, like the config file:

```yaml
models:
  - id: acme-qwen3.8-27b-ft          # the served id
    repo: acme/qwen-ft-ninfer          # <owner>/<name>
    revision: v1                       # commit, tag or branch; required
    artifact: acme_ft.ninfer           # the file the server loads
    files:                             # every file, all pinned
      - { name: acme_ft.ninfer.graft.json, bytes: 48211,       sha256: 9c1e… }
      - { name: acme_ft.ninfer,            bytes: 19406942468, sha256: abb1… }
```

Refused, by name, when the catalog is loaded (at start and by every `model`
subcommand): an unknown key; an empty or duplicate id (case-insensitive,
within the file); **an operator id equal to a built-in one**; a `repo` not of
the form `<owner>/<name>`; an empty `revision`; an `artifact` not among
`files`; no file named the artifact's name plus one of
`loader::SIDECAR_SUFFIXES`; a `sha256` that is not 64 lowercase hex digits; a
`bytes` of 0; **a file name that is not a plain name** (a path separator,
`..`, or an absolute path — a catalog must never write outside the
destination directory). A `download.catalog` that names a missing or
unreadable file refuses the start. A relative `download.catalog` resolves
against the directory of the config file that set it, and against the
working directory when it came from a flag or env var.

### The built-in catalog

Every value from the Hugging Face API at the pinned revision; a large file's
SHA-256 is its LFS oid, a sidecar's was computed from the file served at that
revision. Files listed small first.

| id | repo @ revision | artifact | files (bytes, sha256) |
|---|---|---|---|
| `qwen3.8-27b` | `gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer` @ `e961b419b672e183aa55df8c4b975abc82006e8a` | `qwen3_8_27b_nvfp4full-v2.ninfer` | `….ninfer.graft.json` 6,981 `b87e3c005fb1daf7d6b208da52790f95a3f478732f767f87417a5d7fc23491d1`; `qwen3_8_27b_nvfp4full-v2.ninfer` 19,406,942,468 `abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af` |
| `qwen3.8-27b-abliterated` | `gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer` @ `ef3216949e42c42fc713feffb2e05849d57f0f3b` | `qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer` | `….ninfer.graft.json` 41,707 `09b0df892c314563192b0a4f102139939d4ef6261fcd210439d4e2636d12e6fe`; the `.ninfer` 19,406,942,468 `18954280c794cb2ff1fc24ada8158de1df11bf0a0a2ea63f109045af48905ef1` |
| `qwen3.8-flash-next` | `gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis` @ `8d94db83cd8e58d6be31e796723e5e35b960638c` | `qwen3_8_flash_next_trellis_a25-v2.ninfer` | `qwen3_8_flash_next_mtp_3p0-v2.ninfer.conversion.json` 179,029 `a5b2c10fe0745cfe6b8e2914135e44903cfb80acb3f0db21e549e76641f14127`; `qwen3_8_flash_next_trellis_a25-v2.ninfer.conversion.json` 1,183,942 `9858d9fb03dd66fa2558b5dc6aa6463ee70f00024ffbd746545b231c9514f7a0`; `qwen3_8_flash_next_mtp_3p0-v2.ninfer` 1,037,905,920 `432fe5c0838027047c5b46c4fde526f3d3ef3d126f00fed1213921f8705988d2`; `qwen3_8_flash_next_trellis_a25-v2.ninfer` 71,760,711,680 `12e3b265b0678ba935a4b05d1b612ba4b68b3a96c3026d818ad65838985fd864` |

The companion container is found by the loader beside the main one under its
fixed name (`packer::MTP_ARTIFACT_FILE_NAME`), so the flat destination holds.
The repos' `.sha256` files are never fetched (ADR 0033).

### Endpoint and token

Three fields in the `download` group (ADR 0046 declaration, flag/env/file key
as usual), none patchable over `PATCH /v1/config`:

- `download.endpoint` — default `https://huggingface.co`. A file's URL is
  `{endpoint}/{repo}/resolve/{revision}/{file}`, the revision
  percent-encoded.
- `download.token` — sent as `Authorization: Bearer …` to the endpoint. A
  secret: redacted wherever `server.api_key` is (`config print`,
  `GET /v1/config`, logs).
- `download.catalog` — the operator catalog's path, unset by default.

Which token a request carries is one pure function of (endpoint,
`download.token`, `HF_TOKEN`): `download.token` when set; else `HF_TOKEN`
**only when the endpoint's host is `huggingface.co`**; else none. No token is
sent on a redirect to another host (Hugging Face redirects large files to its
CDN).

### The transfer

Every file of an entry goes through the verified path today's artifact has:
streamed to `<name>.part`, hashed while written, renamed only when byte count
and digest are the pinned ones; a short body keeps its `.part` for a `Range`
resume; a wrong digest, an overlong body or a `416` on resume discard it.
Sidecars included — `fetch_sidecar`'s unverified write goes. Files are fetched
in ascending byte order, so a repo that cannot serve its small files fails
before the large one. A file already at its final name with its pinned byte
count is skipped without hashing (hashing what is on disk is `model verify`'s,
not this spec's); one at its final name with another byte count is refused by
name and left alone.

### Start-time fetch

Unchanged in shape (spec 01): `--model-artifact` set wins; no `cuda` → no
fetch; an id outside the catalog → placeholder; the entry's artifact on disk →
load it; `download.enabled` false → placeholder; otherwise ask on a TTY,
fetch without one. Now over the merged catalog, fetching the whole entry. The
question names the **endpoint** and the repo, not a hard-coded
`https://huggingface.co`.

### `ignis-server model download` / `model list`

Both go through `config::resolve_with` like `config generate` does, so config
discovery, `--config`, env vars and flags (`--download-endpoint`, …) all
apply. Bare `ignis-server` still serves; `make config` output is unchanged.

- `model download [<id>…] [--all] [--out <dir>]` — fetches each named entry
  (none named: the configured `model.id`; `--all`: every entry) into `--out`
  or `download.path`. Never asks; ignores `download.enabled`; runs on a build
  without `cuda`. Progress lines on stderr as it runs; the artifact paths on
  stdout when done; exit 0 only when every entry ended verified. An id outside
  the catalog is an error naming the ids there are.
- `model list [--format json]` — one row per entry: id, built-in or operator,
  repo@revision, total size, whether it is on disk under `download.path`
  (complete / partial / absent), and its family when its artifact is on disk
  (read from the header), `—` otherwise.

`model verify` and `model convert` are reserved: `model <anything else>` is an
"unknown command" error listing `download` and `list`.

### Known models

When `switch.allow_model_switch` is on, a request naming a catalog entry's id
whose artifact is under `download.path` **at the time of the request** is a
known model: a `model download` into a running server's `download.path` makes
it switchable with no restart. An explicit `switch.known_models` entry wins
for its id. The catalog's contribution is derived, never written back to the
config file by `PATCH /v1/config`. A catalog id whose artifact is not on disk
is refused as today — a switch never starts a download.

### Make and docs

- `make UNCENSORED=1` serves `qwen3.8-27b-abliterated` (`make config` shows
  the `--model-id`), still from its artifact path; the Makefile's
  `hf download` hint becomes the `ignis-server model download <id>` line.
- `docs/user/README.md`: the `model` command, the three new fields, the
  catalog file format, and the air-gapped recipes — files carried in
  (`model download --out`), a mirror (`download.endpoint` + token, no
  catalog), own models (a catalog). `README.md`'s `hf download` lines become
  `model download`.
- Spec 01: the flag names #311 renamed (`--no-model-download` →
  `--download-enabled false`, `--model-download-path` → `--download-path`,
  `--model` → `--model-id`, `--artifact` → `--model-artifact`) are updated in
  place. `download.rs`'s doc comment loses `models/*.README.md` (no such
  files) and "the v1 specialization" (no such glossary entry).

## Acceptance

1. The built-in catalog is a data file embedded in the binary; a test parses
   it and checks every invariant in *The catalog file*, and that it holds
   exactly the three entries of the table above with those values.
2. An operator catalog named by `download.catalog` (YAML and JSON) adds its
   entries; each refusal listed in *The catalog file* is a test that names
   the offending entry or key, the built-in-id collision and the non-plain
   file name among them.
3. A relative `download.catalog` resolves against the config file's directory
   when the config file set it.
4. `download.endpoint` and `download.token` exist as ADR 0046 fields; a
   file's URL carries the endpoint, repo, revision and file name (tested
   against the loopback server of `tests/model_download.rs`).
5. The token function: `download.token` wins; `HF_TOKEN` only for a
   `huggingface.co` endpoint; none otherwise — a unit test per cell.
6. A redirect to another host does not carry the `Authorization` header —
   tested with two loopback listeners.
7. `download.token` is redacted in `config print` and `GET /v1/config`, and
   never appears in a log line.
8. Every file of an entry, sidecars included, is verified before rename: a
   sidecar served with the wrong digest is discarded and the fetch fails.
9. A multi-file entry is fetched in ascending byte order, each file resumable
   from its own `.part`; a complete file already at its final name is
   skipped, one with a different byte count is refused and left in place.
10. `artifact_source` remains pure over the merged catalog; spec 01's table
    still passes, plus a cell for an operator entry.
11. The start-time question names the configured endpoint.
12. `ignis-server model download <id>` fetches the entry into
    `download.path`, `--out` elsewhere; no id fetches `model.id`; `--all`
    fetches every entry; an unknown id fails naming the known ones; it asks
    nothing, ignores `download.enabled`, and works in a build without `cuda`.
    Exit code nonzero on any failure; progress on stderr.
13. `ignis-server model list` (and `--format json`) shows every entry with
    its layer, repo@revision, size, on-disk state and family-or-`—`.
14. A catalog entry whose artifact is on disk when a request names it is
    switchable with implicit switching on; an explicit `switch.known_models`
    entry wins; nothing catalog-derived is written to the config file.
15. `make UNCENSORED=1 config` shows `--model-id qwen3.8-27b-abliterated`.
16. Docs as in *Make and docs*, spec 01's flag names current.
17. `cargo test --workspace` passes. No GPU, no `--features cuda`, no
    `--ignored` run is needed for any of the above.

## Testing Decisions

The decision, the catalog parser and the token rule are pure functions with
unit tests. The transfer, the endpoint URL shape and the redirect are driven
against loopback `axum` servers (`tests/model_download.rs`), never the
internet. The CLI is driven through `resolve_with` with in-memory files, as
`config generate` is.

## Out of Scope

`model verify`, `model convert`; publishing to Hugging Face; a per-entry
endpoint or a generic per-file URL mode; signing the catalog; a switch that
downloads; hashing files already on disk; mirroring the repos' `.sha256`
files.
