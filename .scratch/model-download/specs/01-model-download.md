# Model download at start (`--no-model-download`, `--model-download-path`)

An `ignis-server` started on a machine that has no `.ninfer` artifact is
useless: the operator gets the placeholder template and a mock backend, and
nothing says where the real model comes from. The artifact is published on
Hugging Face (the owner's own repo, the one `models/*.README.md` documents),
so the server can fetch it itself.

**Interactive** (stdin is a TTY): ask before spending 19.4 GB of disk and
bandwidth. **Non-interactive** (a container, a daemon, CI): nobody can
answer, so it downloads.

## Surface

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--model-download` / `--no-model-download` | `IGNIS_MODEL_DOWNLOAD` | on | May the server download a missing artifact at all |
| `--model-download-path <dir>` | `IGNIS_MODEL_DOWNLOAD_PATH` | `./models` | Where a downloaded artifact lands |

The destination is **flat**, one file per model, exactly what
`hf download <repo> <file> --local-dir models` produces:
`<dir>/qwen3_8_27b_nvfp4full-v2.ninfer` plus its `.graft.json` sidecar. A
machine that already downloaded the artifact by hand — or that has the
shared store symlinked at `./models`, as this one does — is therefore
already "found", and nothing is re-downloaded.

## Registry

One static table, `crates/server/src/download.rs`. The model id
(`--model`, default `qwen3.8-27b`) is the key:

| Model | Repo | Artifact | Bytes | SHA-256 |
|---|---|---|---|---|
| `qwen3.8-27b` | `gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer` | `qwen3_8_27b_nvfp4full-v2.ninfer` (+ `.graft.json`) | 19,406,942,468 | `abb1e120…fd9af` |

The size and digest are **pinned in the binary**, never read from the repo:
the published `.sha256` file lives in the same trust domain as the artifact
it describes, so it can attest nothing about it.

## When the download happens

The trigger is the *unset* `--artifact` case only, in a `--features cuda`
build:

```
--artifact <path> given:
  the path is the operator's word — a missing file refuses the start as it
  always has, with one added hint line naming --model-download-path.

--artifact unset, built with cuda:
  <download-path>/<artifact file> exists  -> load it
  missing, downloads off                  -> placeholder + MockCompute (as today)
  missing, model not in the registry      -> placeholder + MockCompute (as today)
  missing, stdin is a TTY                 -> ask [y/N]; no -> placeholder
  missing, stdin is not a TTY             -> download, then load

--artifact unset, built without cuda:
  placeholder + MockCompute, always. A binary that cannot run the model has
  no use for 19.4 GB of weights, and this is what keeps `make mock`,
  `cargo test` and every CPU CI job from ever touching the network.
```

## Acceptance criteria

1. **AC1 — the decision is pure.** `download::artifact_source` takes the
   config values, an `exists` probe, the TTY answer and a registry lookup,
   and returns `Use` / `Download` / `Placeholder`; every cell of the table
   above is a unit test, with no filesystem and no network.
2. **AC2 — flags and env resolve like every other knob.** Flag beats env
   beats default, `IGNIS_MODEL_DOWNLOAD` accepts `1/true/on` and
   `0/false/off` (anything else refuses the start), and `--help` names both
   flags.
3. **AC3 — the artifact is verified before it is used.** The download
   streams to `<file>.part`, hashes as it writes, and renames to the final
   name only when the digest and the byte count match the pinned values.
   Nothing unverified is ever left under the artifact's real name. What
   becomes of the `.part` depends on what went wrong: a body that stopped
   early is a dropped connection, so it is kept for AC4 to resume; a body
   that arrived whole and hashes to something else — or one longer than the
   pin, or a partial the source can no longer satisfy — is not this artifact
   at all, so it is deleted rather than resumed forever.
4. **AC4 — an interrupted download resumes.** A second run sends
   `Range: bytes=<part len>-`, continues the digest over the bytes already
   on disk, and ends with the same verified file.
5. **AC5 — the sidecar comes first.** `<artifact>.graft.json` is fetched
   before the 19.4 GB body, so a repo that cannot serve it fails in a second
   instead of an hour. The loader's existing sidecar/checksum gate then runs
   unchanged.
6. **AC6 — the prompt goes to stderr.** stdout carries the plain lines
   `mk/windows/common.ps1` parses (the generated API key, the public URL); a
   question must not land among them. EOF on stdin reads as "no".
7. **AC7 — the container default is documented.** README's artifact-less
   `podman run` smoke test gains `--no-model-download`, and the flag table
   and the `main.rs` module doc gain both flags.

## Seam

`download.rs` is the module: the registry, the pure decision, and a
`Downloader` whose base URL is injectable, so the HTTP path is tested against
a local `axum` server (Range, truncation, a tampered body) with no network
and no GPU. `main` calls the decision once and either loads, downloads then
loads, or falls through to the placeholder it already had.
