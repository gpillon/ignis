---
title: "Get a model"
description: "The catalog, `model download`, letting the server fetch its model, the endpoint and token, and pointing at an artifact you already have."
sidebar:
  order: 1
---

The engine loads a `.ninfer` artifact: weights, tokenizer and chat template in
one container, verified against its provenance sidecar (`.graft.json` or
`.conversion.json`) at load. A mismatch refuses the start.

## The catalog

The models the server knows how to fetch are its **catalog**
([ADR 0047](../../adr/0047-a-catalog-the-operator-can-extend.md)). Twelve entries
are built into the binary, eight text and image models, three speech models and a
diarizer, each pinned file by file — byte count and SHA-256 — at a fixed
revision of its Hugging Face repository:

| Id | Repository | Size |
|---|---|---|
| `qwen3.8-27b` (the default) | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer) | 18.1 GiB |
| `qwen3.8-27b-abliterated` | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer) | 18.1 GiB |
| `qwen3.8-flash-next` | [`gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis`](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis) | 68.2 GiB (with its MTP and vision companions, from 0.11.0; 67.8 GiB with its MTP companion in 0.10.0) |
| `ignis-image-2.1-turbo` | [`gpillon/ignis-image-2.1-turbo`](https://huggingface.co/gpillon/ignis-image-2.1-turbo) | 15.6 GiB (the [image model](../api/images.md); non-commercial licence) |
| `qwen3.8-27b-16g` | [`gpillon/Qwen3.8-27B-trellis-3bpw-ignis`](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-ignis) | 12.5 GiB (the 27B for 16 GB cards) |
| `qwen3.8-27b-abliterated-16g` | [`gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis`](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis) | 12.5 GiB (the [uncensored variant](uncensored.md) for 16 GB cards) |
| `swift-1.5-27b` | [`gpillon/Swift-1.5-Qwen3.8-27B-nvfp4-ignis`](https://huggingface.co/gpillon/Swift-1.5-Qwen3.8-27B-nvfp4-ignis) | 18.1 GiB (Swift 1.5, a [fine-tune](fine-tunes.md) of the 27B; a license that restricts commercial use) |
| `swift-1.5-27b-16g` | [`gpillon/Swift-1.5-Qwen3.8-27B-trellis-3bpw-ignis`](https://huggingface.co/gpillon/Swift-1.5-Qwen3.8-27B-trellis-3bpw-ignis) | 12.5 GiB (Swift 1.5 for 16 GB cards) |
| `qwen3-asr-1.7b` | [`gpillon/Qwen3-ASR-1.7B-GGUF-ignis`](https://huggingface.co/gpillon/Qwen3-ASR-1.7B-GGUF-ignis) | 1.5 GiB (the [speech model](../api/transcriptions.md), `speech.model: auto`: a Q4_K_M decoder and a Q8_0 audio encoder) |
| `qwen3-asr-1.7b-q8` | [`gpillon/Qwen3-ASR-1.7B-GGUF-ignis`](https://huggingface.co/gpillon/Qwen3-ASR-1.7B-GGUF-ignis) | 2.3 GiB (the same with a Q8_0 decoder) |
| `qwen3-asr-0.6b` | [`gpillon/Qwen3-ASR-0.6B-GGUF-ignis`](https://huggingface.co/gpillon/Qwen3-ASR-0.6B-GGUF-ignis) | 0.9 GiB (the smaller speech model, for a weak CPU) |
| `nemotron-3-diarization` | [`gpillon/Nemotron-3-Diarization-GGUF-ignis`](https://huggingface.co/gpillon/Nemotron-3-Diarization-GGUF-ignis) | 0.2 GiB (the [diarizer](../configuration/speech.md#the-diarizer), `diarizer.model: auto`; OpenMDW-1.1 licence) |

The id is the **served id**: what `--model-id` names, what `/v1/models`
reports and what a request puts in `model`.

Every entry states the **license** its weights are published under: an
identifier as the entry's Hugging Face card names it at the pinned revision,
and a link to the text. The 27B's own four entries and the speech
models are `apache-2.0`, Swift 1.5's two
[`swift-open-license-1.0`](fine-tunes.md#its-license), Flash-Next is `qwen-community-1.0`, the image model
`qwen-research` and the diarizer `openmdw-1.1`; the
[generated reference](../reference/models.md) lists each entry's. It is shown
beside the entry in the [Server view](../playground/server.md#models), in
`model list` (a `LICENSE` column, and `license` in `--format json`) and in
`model download`, and `GET /v1/models/catalog` reports it. Whoever downloads
is bound by the license, as with any download from the hub; no step asks to
accept it, and a download fetches only the entry's pinned files, never a
license file. An [operator entry](operator-catalog.md#an-entrys-license) may
state one too.

The three `qwen3-asr-*` entries are the **speech models** (`kind: speech`): two
GGUF files each, a decoder and its audio encoder, with no sidecar. They run on
the CPU beside the loaded model, so they state no card class and no card refuses
them, and a switch never goes to one. `ignis-server model download qwen3-asr-1.7b`
fetches the default; the two 1.7B entries share one encoder file.

The [diarizer](../configuration/speech.md#the-diarizer) that labels a live
session's speakers is an entry of its own kind, `kind: diarizer`: one GGUF, no
sidecar, no card class, never switched to. The built-in one is
`nemotron-3-diarization`; an [operator's catalog](operator-catalog.md) can state
another, and `ignis-server model download <id>` fetches it like any entry.

Each entry states the smallest **card class** it loads on
([ADR 0064](../../adr/0064-a-catalog-entry-states-its-card-class.md)): 32 GB,
the RTX 5090's class, or 16 GB for a built-in text or image model, none for
the speech models and the diarizer;
[Where each model loads](../reference/models.md#where-each-model-loads) gives
each entry on every card, by its class and generation.
On a card of a smaller class (16 or 24 GB), the server refuses an entry above it before unloading or fetching
anything: a switch to it and a request naming it are answered
`409 model_above_card_class`, naming both classes, while the loaded model
serves on; a start with no artifact that would load it, and its
`model download`, are refused; `GET /v1/models`, `GET /v1/models/catalog` and
the Playground list it as unavailable, with the reason. An entry may answer to
its family's default id as an **alias**: the 16 GB 27B answers to
`qwen3.8-27b` on a 16 or 24 GB card, so a client configured with
`qwen3.8-27b` is served by it, and the answer names it — only an entry whose
artifact is of the 27B's family, by the artifact's header, answers to it; one
of another family is refused, naming both. An alias may also be another
entry's id of the same family: `qwen3.8-27b-abliterated-16g` answers to
`qwen3.8-27b-abliterated` on a 16 or 24 GB card, `swift-1.5-27b-16g` to
`swift-1.5-27b` ([fine-tunes](fine-tunes.md)), and an [operator's catalog](operator-catalog.md#an-entrys-family-and-an-alias-of-your-own)
does the same with an entry's `family`. `GET /v1/models` lists the entry
under its own id, never under an alias. A model loaded from a path named
for the load itself — `--model-artifact`, a switch naming its `artifact` — is
your word and is not fitted to the card: only the VRAM plan refuses it. A
catalog id you also list in `--switch-known-models` is fitted all the same:
the class is its catalog entry's. A model id no entry holds has no class, and
a card whose class is not known refuses nothing.

An entry may also state the oldest **GPU generation** it loads on,
`generation: ampere`, `ada` or `blackwell`
([ADR 0079](../../adr/0079-a-catalog-entry-states-the-oldest-gpu-generation-it-loads-on.md)).
The server reads the card's generation from its compute capability at start:
8.6 is Ampere (RTX 30), 8.9 Ada (RTX 40), 12.x Blackwell (RTX 50, RTX PRO).
An entry that states none loads on every generation the build runs on. On an
RTX 3090 or RTX 4090, `qwen3.8-27b`, `qwen3.8-27b-abliterated` and
`swift-1.5-27b` are served by their 16 GB entries through their aliases, and an entry no alias serves is refused as for the class, with
`409 model_above_gpu_generation`: a switch to it, a request naming it, a start
with no artifact that would load it and its `model download`. The message
names what the entry needs and the card: its name, compute capability and
generation. When the class is above the card's too, the code is
`model_above_gpu_generation` and the message names both, since a larger card
of the same generation would not load it either
([troubleshooting](../troubleshooting.md#409-model_above_gpu_generation)). The
lists mark it unavailable with the reason, as for the class. A card whose
compute capability could not be read refuses nothing by generation. A load the
card has no route for, such as an NVFP4 artifact named by its path, is still
refused by name at the load, whatever a catalog says.

`ignis-server model list` shows every entry, how much of it is on disk, and
whether the card of the machine it runs on loads it:

```
ignis-server model list                 # id, built-in or operator, repo@revision, size, complete/partial/absent, family, available, license
ignis-server model list --format json
```

`AVAILABLE` is `no` for an entry the card cannot load, by class or by
generation, and each such entry's reason follows the table, naming the entry
that serves its id on this card when one does; `--format json` gives
`available` and `reason`, as `GET /v1/models/catalog` does. On a machine with
no card read, every entry is available.

## Fetching one: `ignis-server model download`

```
ignis-server model download                          # the configured --model-id (qwen3.8-27b by default)
ignis-server model download qwen3.8-flash-next       # one or more ids
ignis-server model download --all                    # every entry (refused when the card cannot load one)
ignis-server model download qwen3.8-27b --out /mnt/usb   # somewhere other than --download-path
```

Every file of the entry — the artifact, its sidecar, Flash-Next's companion
containers — is streamed to a `.part` file, checked against the size and
SHA-256 pinned in the catalog, and renamed into place only then, small files
first. An interrupted download resumes from its `.part`; a tampered one is
discarded. A file already at its name with its pinned size is kept; one with
another size is refused and left alone. The command never asks, ignores
`--download-enabled`, and needs no GPU and no NVIDIA driver: a build without
`cuda` fetches as well, and so does the CUDA build — the container image
included — on a machine with no driver, such as a Kubernetes CPU node. The
CUDA build links none of the driver's libraries; it reaches them only when a
model loads, and a GPU load on a machine without the driver is refused as
`no NVIDIA driver (libcuda.so.1) on this machine` (`nvcuda.dll` on Windows).
A driver too old for the build's CUDA runtime (a lower CUDA major version than
the runtime's; a lower minor is fine) is refused naming both versions and asking
to update the NVIDIA driver.
Before it fetches an entry, it prints the entry's license, its identifier and
the link to its text (`<id>: license <license>: <url>`, on stderr with the
progress), for an entry that states one; nothing asks to accept it, and you
are bound by it as with any download from the hub.
Progress goes to stderr, the artifact paths to stdout, and the exit code is 0
only when every entry verified. It reads the same configuration a start does
(config file, environment, flags). It is fitted to the card of the machine it
runs on: an id an entry answers to there fetches that entry
(`model download qwen3.8-27b` on an RTX 3090 fetches `qwen3.8-27b-16g`), and
an entry above the card's class or its generation refuses the whole command
before anything is fetched — fetch it on the machine that will run it, or on
one without a GPU.

A model fetched into a running server's `--download-path` is switchable at
once: a request naming its id switches the server to it, as a
`--switch-known-models` entry would (`--switch-allow-implicit`, on by default;
an explicit `--switch-known-models` entry wins for its id). A request naming a
catalog id that is not on disk is refused as an unknown model — a switch never
starts a download.

A client configured with two known models reloads the server back and forth: opencode sends its `small_model` requests (titles, summaries) to the id named by `small_model`, so a `model` and a `small_model` naming different ignis ids switch the server on alternate requests, 13-33 s each. The same holds for a front end that picks its ids from `GET /v1/models`, which lists every servable model: Open WebUI's task model (titles, tags, follow-ups) or Hermes set to another listed id than the chat's does it too. Point both at the same id, or start with `--switch-allow-implicit false`. The Playground's chat and Decide never cause a switch: their requests name no model, so the server serves the one loaded, and the header and the context limit follow it, read again from `GET /v1/models` when a turn starts. The Playground switches only when asked to: **Load** on its Server view, **Generate** and **Edit** on its Image view with a text model loaded ([in the Playground](../api/images.md)), the chat's `make_image` tool (to the image model and back), and the chat's picker for the next message's model ([the Playground](../playground/index.md)); each of these switches with `POST /v1/models/switch` or by naming the image model, never by a chat request naming a model.

### One writer per entry

Every fetch of an entry holds the entry's lock for as long as it runs: the
command's, a server's start, and a server's download from its Server view
alike. The lock is `<artifact>.lock`, beside the entry's files in the folder
it fetches into. A second fetch of the same entry into the same folder is
refused before it fetches anything, so two never write one `.part`, with a
message naming who holds the entry, a server or `ignis-server model download`,
and its process id:

```
qwen3.8-27b: failed: qwen3.8-27b is already being downloaded into ./models, by a server (process 4120), which holds ./models/qwen3_8_27b_nvfp4full-v2.ninfer.lock: one writer per entry, so this one fetched nothing; try again once that download ends
```

The holder writes who it is into the `.lock` file while it holds it, and the
refused side reads it there; a lock whose holder has written nothing yet names
either of the two.

The command exits 1, the start is refused, and the Server view's **Download**
answers `409 download_locked`; try again once the other fetch ends. The
**Download** answers it even on a disk short of
[room](#room-on-the-disk---download-free-margin), so no refusal for room stays
on the row of an entry the other fetch then lands. Two
different entries are fetched at the same time. The lock is the operating
system's, `flock` on Linux and `LockFileEx` on Windows: it ends with the
fetch, whether the fetch verified, failed or was cancelled, and with the
process holding it, killed or crashed, so it is never cleared by hand. The
`.lock` file stays in the folder, emptied of its note, for the next fetch of
the entry.

## Downloading from the Server view: `POST /v1/models/download`

A running server downloads a catalog entry itself, while it keeps serving:
**Download** on the [Server view](../playground/server.md#downloading-a-model)'s
row, or the same request from any client
([ADR 0071](../../adr/0071-the-server-downloads-and-removes-models-from-the-server-view.md)):

```
curl -X POST http://127.0.0.1:8000/v1/models/download \
  -H "Authorization: Bearer $IGNIS_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-flash-next"}'
```

It answers `202 {"status": "downloading", "model", "bytes", "bytes_on_disk"}`
at once, and fetches every file of the entry into `--download-path` with the
same transfer as `model download`: resumed from a `.part`, verified against the
pins, small files first, the token sent to the endpoint only. Only a catalog
entry is downloaded, the built-in ones and an operator catalog's alike, never
an arbitrary repository, and the request carries no token: the server uses
the one it read at start ([endpoint and token](#where-the-bytes-come-from-endpoint-and-token)).
`GET /v1/models/catalog` follows it: the entry's row carries `download`
(`running`, then `none`; `failed` with a `download_reason` when it failed) and
`bytes_on_disk`, what of it is there, partial files included. Once every file
verified, the row is `on_disk: complete` and `switchable`, `GET /v1/models`
lists it, and `POST /v1/models/switch` loads it: the server never switches to
it by itself, and a switch still never starts a download. A failed download's
reason stays on the row, in memory, until the entry's next download or a
restart, or until the entry is complete on disk another way (`model download`
fetched it): a complete row reads no failure. Nothing about a download is
written down, and nothing resumes by
itself at the next start: a partial entry's next download, the Server view's
**Resume**, resumes it ([Cancelling and resuming](#cancelling-and-resuming-post-v1modelsdownloadcancel)).

| Answer | When |
|---|---|
| `202` | The download began. |
| `400` | `model` is empty. |
| `401 invalid_api_key` | The server has an API key and the request did not carry it: the same key as `PATCH /v1/config`. |
| `403 download_disabled` | `--download-enabled false` (the server never uses the network) or `--download-from-ui false`; the message names which. |
| `404 model_not_found` | `model` is not a catalog entry. |
| `409 download_in_progress` | A download already runs, named in the message: one at a time, never queued. |
| `409 download_locked` | Another fetch of this entry into `--download-path` holds its lock: `model download`, or another server on the same folder ([one writer per entry](#one-writer-per-entry)). |
| `409 model_above_card_class` | The entry's `card_class` is above this card's. |
| `409 model_above_gpu_generation` | The entry's `generation` is newer than this card's GPU generation, its class above the card's or not. |
| `409 model_already_downloaded` | Every file of it is already on disk. |
| `500 download_failed` | The download could not begin: the server could not build its HTTP client, or could not open the entry's lock file under `--download-path` (a folder it cannot create or write to); the message says which. Nothing was fetched, and the row keeps no reason. |
| `503 model_switching` | A switch is under way; retry once it lands. |
| `507 download_no_room` | The models folder's volume has less free space than the entry's bytes still missing plus `--download-free-margin`; the row keeps the reason ([Room on the disk](#room-on-the-disk---download-free-margin)). |

A failure is the transfer's own reason, as `model download` prints it. A
gated repository's `401` or `403` names the token and where it is read from:
`IGNIS_DOWNLOAD_TOKEN` (`download.token`), or `HF_TOKEN` when the endpoint is
huggingface.co, set where the server starts; the token itself is never shown,
logged or sent by the page. The log has the transfer's own
`ignis.model.download_started`, `ignis.model.download_progress` and
`ignis.model.downloaded` events, as a start's fetch logs them, and
`ignis.model.download_failed` with the reason.

`--download-from-ui false` (`download.from_ui`, on by default) turns this route
off, so the Server view can be exposed without exposing downloads; `model
download` on the server's machine still fetches. Neither switch can be changed
by `PATCH /v1/config`.

### Cancelling and resuming: `POST /v1/models/download/cancel`

```
curl -X POST http://127.0.0.1:8000/v1/models/download/cancel \
  -H "Authorization: Bearer $IGNIS_API_KEY"
```

stops the running download, the one at a time, with no body to send, and
answers once it has stopped: `200 {"status": "cancelled", "model", "bytes",
"bytes_on_disk"}`. What it fetched stays under `--download-path`: every file it
verified at its name, and the `.part` of the file it was writing, with every
byte that arrived. The entry's row then reads `on_disk: partial` and
`download: none`, with those `bytes_on_disk`, and its next
`POST /v1/models/download` (the Server view's **Resume**) resumes it: it
fetches no verified file again, asks for the rest of the `.part` with `Range`,
and checks every file against its pins before renaming it into place. A
download cut short by a dropped connection or a restart leaves a partial entry
too. Nothing resumes it by itself, at a restart either: it stays partial until
a download of it, through the API or `model download`, continues it. The one
exception is the model the server is started to serve, which a start fetches
when it is missing, as [below](#letting-the-server-fetch-it).

A cancel never touches the loaded model, so the server answers it during a
model switch too, as it answers `GET /v1/models`: a download begun before the
switch is stopped while the switch runs. A new download waits for the switch
(`503 model_switching`).

| Answer | When |
|---|---|
| `200` | The download stopped; its partial files stay. |
| `401 invalid_api_key` | The server has an API key and the request did not carry it: the same key as `PATCH /v1/config`. |
| `409 download_not_running` | No download runs: none began, or the one that ran had already landed or failed, which the message says. |
| `503 server_not_ready` | The loaded model's first traversal has not run yet; retry shortly. |

The log has `ignis.model.download_cancelled` at INFO, with the entry's `model`,
`repo` and `endpoint`, its `bytes` and the `bytes_on_disk` it keeps for the
resume. A cancel is no failure: it logs no `ignis.model.download_failed`, and
the row carries no `download_reason` for it.

## Removing a model: `POST /v1/models/remove`

A running server removes a catalog entry's files from its models folder, to
free the disk: **Remove** on the [Server view](../playground/server.md#removing-a-model)'s
row, after a confirmation, or the same request from any client
([ADR 0071](../../adr/0071-the-server-downloads-and-removes-models-from-the-server-view.md)):

```
curl -X POST http://127.0.0.1:8000/v1/models/remove \
  -H "Authorization: Bearer $IGNIS_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-flash-next"}'
```

It deletes every file of the entry under `--download-path`, finished or
partial, each at its name and its `.part`, so a partly downloaded entry's
unfinished download goes too, and answers once they are gone:
`200 {"status": "removed", "model", "bytes_freed", "files_removed",
"files_kept"}`. Nothing else is touched: not a known model's own path, and not
the entry's `<artifact>.lock`, which stays empty for the next fetch
([one writer per entry](#one-writer-per-entry)). A file another entry also
lists stays while that entry is on disk, every file of it at its name, or being
downloaded, by the server or by `model download`: `files_kept` names it with
that entry (`{"file", "by"}`), and the entry removed then reads `partial`. The
entry's lock is held while its files go, so no fetch of it begins meanwhile.
Its `GET /v1/models/catalog` row then reads `on_disk: absent` (or `partial`),
`bytes_on_disk` what is left, and `download: none`, its last failure
forgotten, and `GET /v1/models` no longer lists it; its next download fetches
it again.

The start model may go. Its catalog row names it in `start_model` while the
server's start loaded it from these files, under whatever id it serves, and the
next start looks for it under `--download-path` again: it fetches it, asking
first on a terminal, as [below](#letting-the-server-fetch-it); with
`--download-enabled false` it comes up on the placeholder template; and a start
whose `--model-artifact` names the removed file refuses to start. A row lists in
`known_models` the ids `--switch-known-models` names at one of its files, its
own or another: removed, a switch to any of them is refused until it is
downloaded again. Both are matched by file, so the Server view's confirmation
names a model served from the entry's files under another id too.

| Answer | When |
|---|---|
| `200` | The files were deleted. |
| `400` | `model` is empty. |
| `401 invalid_api_key` | The server has an API key and the request did not carry it: the same key as `PATCH /v1/config`. |
| `403 removal_disabled` | `--download-from-ui false`. |
| `404 model_not_found` | `model` is not a catalog entry: a known model is a path the operator gave, and only a catalog entry's files are removed. |
| `409 model_in_use` | The entry is the model loaded (also unloaded while idle, or served from its files under another id), the model a load is loading (by its id or from its files), the [speech model](../api/transcriptions.md) or the [diarizer](../configuration/speech.md#the-diarizer) loaded beside it, or the entry the server is downloading; the message says which. Cancel the download, load another model, or wait for the speech worker to unload it (`--speech-idle-unload` seconds after its last transcription), first. |
| `409 download_locked` | `model download`, or another server on the same folder, is fetching the entry. |
| `409 model_not_downloaded` | Nothing of the entry is under `--download-path`. |
| `500 removal_failed` | A file could not be deleted; the message names it, and what was deleted before it. |
| `503 model_switching` | A switch is under way, which may be loading this entry; retry once it lands. |

The log has `ignis.model.removed` at INFO, with the entry's `model`, `repo`,
the `dir`, the `bytes` freed, the `files` deleted and the files `kept`. A
refused removal logs nothing. `--download-from-ui false` turns this route off
with the download routes; `--download-enabled false` does not, since a removal
uses no network.

## Letting the server fetch it

A GPU build started **without** `--model-artifact` / `IGNIS_MODEL_ARTIFACT`
looks for the `--model-id` entry's artifact under `--download-path` (default
`./models`, flat, under the names the repository publishes) and fetches the
whole entry when it is not there
([ADR 0033](../../adr/0033-the-server-fetches-its-own-model.md)):

```
ignis-server                                      # asks first: the size, the source, the destination
ignis-server --model-id qwen3.8-flash-next        # the same, for another catalog entry
ignis-server --download-enabled false             # never fetches: the placeholder template and the CPU mock
ignis-server --download-path /weights
```

On a terminal you are asked (`[y/N]`, on stderr, naming the endpoint the bytes
come from); without one — a container, a daemon, CI — nobody can answer, so it
downloads, through the same verified transfer as `model download`. A failed
fetch refuses the start. A build without `--features cuda` never downloads
weights it could not run.

A `model download` and a download the server did land on the same file names —
as does a `hf download <repo> --revision <rev> --local-dir models` done by
hand — so any of them satisfies the others.

## Room on the disk: `--download-free-margin`

Before it writes a byte, every download checks that the volume holding
`--download-path` has room for it: the free space must cover the entry's bytes
still missing (a partial entry counts only what its `.part` files and finished
files do not hold yet) plus `--download-free-margin` (`IGNIS_DOWNLOAD_FREE_MARGIN`,
`download.free_margin`, `1G` by default, a byte count with `K`/`M`/`G`). The
Server view's **Download**, `model download` and the start's own fetch all make
the same check, so a transfer never fills the disk halfway through. The free
space is what the system grants this process on that volume
(`GetDiskFreeSpaceExW` on Windows, `statvfs` on Linux), read at the folder,
or at the nearest folder above it when the first download has yet to create it.
An entry with nothing missing writes nothing and is never refused.

Short of room, the download is refused naming both numbers:
`not enough room for <id> under <dir>: <n> bytes needed (<missing> still
missing plus download.free_margin <margin>), <free> bytes free (… GiB needed, …
GiB free)`. `model download` prints it as the entry's failure and fetches the
next entry named, exiting nonzero; the start logs it on
`ignis.model.download_failed` and refuses to start; `POST /v1/models/download`
answers `507 download_no_room`, and the entry's row in `GET /v1/models/catalog`
keeps it as its `download_reason`, `download: failed`, until its next download,
a restart, or the entry complete on disk another way. Free some space on that
volume, or lower the margin and restart: the
field cannot be changed by `PATCH /v1/config`.

## Where the bytes come from: endpoint and token

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--download-endpoint <url>` | `IGNIS_DOWNLOAD_ENDPOINT` | `https://huggingface.co` | The server a file is fetched from, as `{endpoint}/{repo}/resolve/{revision}/{file}` — Hugging Face, or any mirror answering the same route. |
| `--download-token <token>` | `IGNIS_DOWNLOAD_TOKEN` | unset | Sent as `Authorization: Bearer …` to the endpoint. Never shown by `GET /v1/config` or `config print`, never logged. |
| `--download-catalog <path>` | `IGNIS_DOWNLOAD_CATALOG` | unset | An operator catalog ([Your own models: an operator catalog](operator-catalog.md)). |

The token is `--download-token` when it is set; otherwise `HF_TOKEN`, but only
when the endpoint is `huggingface.co` itself — a personal Hugging Face
credential is never handed to a mirror. No token is sent across a redirect to
another host (Hugging Face sends a large file from its CDN). None of the three
can be changed by `PATCH /v1/config`.

## Pointing at one you already have

```
set IGNIS_MODEL_ARTIFACT=./models/qwen3_8_27b_nvfp4full-v2.ninfer   # Windows
export IGNIS_MODEL_ARTIFACT=./models/qwen3_8_27b_nvfp4full-v2.ninfer
```

A named path is the operator's word: missing, it refuses the start, and it is
never matched against the catalog. Served with no `--model-id`, it takes its
family's default id.

The artifact carries a grafted DFlash2 drafter module, which `--spec-backend dflash2`
loads for speculative decoding and which costs no VRAM when speculation is off.
