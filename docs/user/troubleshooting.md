---
title: "Troubleshooting"
description: "Symptoms and error messages, grouped by when they appear, with the cause and the fix for each."
sidebar:
  order: 1
---

Each entry starts with what you see: the message as the server prints it, or the
behaviour. Messages are quoted from the server; a figure in one, such as a byte
count, differs from run to run. Error bodies on `/v1` are OpenAI's
`{"error": {message, type, code}}`; the `code` is what a client should match
([The API](api/index.md)).

## Starting

### A start under `--profile auto` is refused, naming `--profile`

The card's total memory is not one the built-in profiles serve (under 20 GiB,
28 to 40, 88 to 104), or NVML could not read it. Name the profile that fits the
card, or `none` for the hardcoded defaults
([Profiles and defaults](concepts/profiles.md)).

### `no NVIDIA driver (libcuda.so.1) on this machine`

On Windows the library is `nvcuda.dll`. The full message continues: "a GPU load
needs the host's NVIDIA driver (in a container, the NVIDIA container runtime
provides it on a GPU node)".

The CUDA build links none of the driver's libraries and reaches them only when
a model loads, so the server and `model download` start without a driver and a
GPU load is refused. Install the driver on the host. In a container, pass the
GPU in (`--device nvidia.com/gpu=all` for podman, `--gpus all` for docker) and
check the host has the NVIDIA container runtime
([Install](start/install.md#the-container-image)).

On a machine that has no GPU on purpose (a Kubernetes CPU node), the same image
can still fetch models: `ignis-server model download <id>`
([Get a model](guides/models.md#fetching-one-ignis-server-model-download)).

### `the NVIDIA driver on this machine supports CUDA X but this build's CUDA runtime is Y: update the NVIDIA driver to one that supports CUDA Z or newer`

The driver's CUDA major version is lower than the build's runtime. A lower
minor version is fine; a lower major is refused. Update the NVIDIA driver on
the host. The Linux tarball also needs the CUDA 13 runtime installed
([Install](start/install.md#release-archives)).

### The server starts but the text is placeholder and every reply ends with `finish_reason: "length"`

That is the CPU mock. The server runs it when it was built without
`--features cuda`, when `--download-enabled false` is set with no artifact, or
in the container smoke test that passes `--download-enabled false` ([Install](start/install.md#the-container-image)).
The release archives and container images of v0.9.0 and v0.9.1 were built
without `cuda` by mistake, so they serve the mock whatever the model: use
v0.9.2 or later ([#408](https://github.com/gpillon/ignis/issues/408)).
The mock has no EOS token, so every completion ends `length`. Start a GPU build
with a model: [Quick start](start/quick-start.md).

### `VRAM budget: free memory is unreadable`

The server reads the card's free memory before it creates the CUDA context and
refuses to start when that fails. Check that `nvidia-smi` runs and shows the
card on this machine. The Spark's unified memory is one reason this cannot work
([Why not DGX Spark?](guides/dgx-spark.md)).

### `the VRAM budget of … bytes (--vram-budget-bytes) is more than the … bytes free at start`

You named a `--vram-budget-bytes` larger than the memory free on the card.
Another program holding VRAM is the usual cause: check what is using the card
(`nvidia-smi`; in a checkout, `make gpu-status`). Lower the budget, free the
card, or pass `--vram-allow-oversubscription` to start anyway; on Windows that
pages ([VRAM budget](configuration/vram-and-kv-pool.md)).

Without `--vram-budget-bytes` the budget is the free memory at start minus
`--vram-headroom-bytes` (`1536M`), so a busy card gives a smaller budget
instead of this refusal, and the next message may follow.

### `the KV pool of … (--vram-kv-pool-bytes) holds … pages of 64 tokens … fewer than the … it must hold`

The VRAM plan cannot hold one full context. The message ends with the knobs
that fix it: "name a larger --vram-kv-pool-bytes, or a smaller
--model-max-context, --reuse-retained-device or --reuse-retained-host". The
plan is printed as `ignis.runtime.vram_plan` (`ignis.runtime.flash_next_plan`
on Flash-Next) ([VRAM budget and the KV pool](configuration/vram-and-kv-pool.md#the-kv-pool),
[ADR 0030](../adr/0030-device-memory-reserved-at-load.md)). On a card that
another program shares, closing that program is the other fix.

### Flash-Next refuses the start because the expert cache is below 12 GiB

Below 12 GiB the expert cache refuses the start, because decode slows sharply
there. The message names the knobs that lift it: a smaller `--vram-kv-pool-bytes`,
`--model-prefill-chunk` or `--vram-headroom-bytes`, or fewer
`--reuse-retained-device`. `--vram-allow-expert-cache-below-floor` starts anyway
with an `ignis.runtime.expert_cache_below_floor` warning. A cache too small for
one decode step's experts is refused whatever the flag says
([VRAM budget](configuration/vram-and-kv-pool.md#the-kv-pool)).

### `the metrics listener's default address could not be bound`

This is the WARN `ignis.server.metrics_unavailable`: another server on the host
holds `127.0.0.1:9464`. The server runs without the metrics listener, and
`/ui/metrics` still answers. Name another address with `--server-metrics-bind`,
or turn metrics off with `--server-metrics false`. An address you named that
cannot be bound refuses the start instead
([Access](configuration/access.md)).

### `make run` or `make start` refuses a stale binary

`run` and `start` only launch the last build. The refusal names the files that
changed; `make build` is the fix ([Working with make](contributing/make.md)).
Before GPU work, `make gpu-status` shows who holds the card, and a CUDA `run`
or `start` runs a GPU guard that refuses when it is held (`GPU_CHECK=0` skips
it).

## Loading a model

### The first start downloads for a long time, or asks `[y/N]`

With no `--model-artifact`, a GPU build looks for the model under
`--download-path` (default `./models`) and fetches the whole entry when it is
not there: 18.1 GiB for the default 27B, 67.8 GiB for Flash-Next. On a terminal
it asks first; in a container, a daemon or CI nobody can answer, so it
downloads. Every file is checked against its pinned size and SHA-256, an
interrupted download resumes from its `.part`, and a failed fetch refuses the
start. Downloading ahead with `model download` (no GPU needed), pointing
`--download-path` at a directory you keep, and mirrors for air-gapped machines
are in [Get a model](guides/models.md#letting-the-server-fetch-it) and
[Air-gapped machines](guides/air-gapped.md).

### The model is downloaded again on every container start

With no `IGNIS_MODEL_ARTIFACT` the image fetches into its own working
directory (`/home/ignis/models`) and loses it with the container. Mount a
directory and pass `--download-path /models`
([Install](start/install.md#the-container-image)).

### The load fails reading the model directory

The server runs as `ignis` (uid 10001). A model directory, or a Hugging Face
cache, that root wrote with `0700` directories is not readable by it. Make
what you mount readable by uid 10001
([Install](start/install.md#the-container-image)).

### A file at its name is refused and left alone

A download keeps a file at its name with its pinned size and refuses one with
another size without touching it. Delete or move the wrong file and run
`ignis-server model download <id>` again. A `--model-artifact` path that is
missing, or an artifact that does not verify against its provenance sidecar
(`.graft.json` or `.conversion.json`), refuses the start
([Get a model](guides/models.md)).

### `… is already being downloaded into …, by … (process …)`

Another fetch of the same entry into the same folder holds its lock, and the
message names it with its process id: `a server`, downloading it from its
Server view or at its start (or removing it), or `ignis-server model download`.
`model download` then fails that entry and exits 1, and a start that would
fetch it is refused; nothing was fetched or written. Wait for the other fetch
to end, or stop that process, and run again: a partial download resumes from
its `.part`. A killed or crashed fetch leaves no lock behind, and the
`<artifact>.lock` file it names stays for the next fetch
([one writer per entry](guides/models.md#one-writer-per-entry)).

### The Server view's Download is refused, or fails

**Download** (`POST /v1/models/download`) answers with the server's reason,
shown in the model band
([Downloading from the Server view](guides/models.md#downloading-from-the-server-view-post-v1modelsdownload)):

- `403 download_disabled`: `--download-enabled false` or
  `--download-from-ui false`, named in the message; the row then offers the
  `model download` command to copy. Turn the field on and restart the server,
  or run the command on the server's machine.
- `409 download_in_progress`: one download runs at a time; its row shows its
  progress. Download the next one once it ends.
- `409 download_locked`: `ignis-server model download`, or another server on
  the same `--download-path`, is fetching the entry. Download it once that
  ends, or let the command land it.
- `409 model_already_downloaded`: the entry is on disk; **Load** switches to it.
- `409 model_above_card_class`: the card cannot load the entry (**Not for this
  card**).
- `500 download_failed`: the download could not begin, the message says why:
  most often the entry's `<artifact>.lock` cannot be opened, because the
  server cannot create or write to `--download-path`. Give the server's user
  write access to that folder, or point `--download-path` at one it can write.
- `507 download_no_room`: the models folder has no room for the entry; the
  reason shows under its row instead of the band
  ([`not enough room for … under …`](#not-enough-room-for--under-)).

A download that began and failed keeps its reason under its row,
`Download failed: …`, until its next download or a restart, or until the entry
is complete on disk another way (`ignis-server model download`). `GET …: 401
Unauthorized: the endpoint wants a token and none was sent`, or `403 Forbidden:
the endpoint refused the token sent`, is a gated or private repository: set
`IGNIS_DOWNLOAD_TOKEN` (`download.token`), or `HF_TOKEN` for huggingface.co,
where the server starts, restart it, and press **Download** again; the page
never takes a token. Any other reason is the transfer's, as `model download`
prints it, and the row's **Resume** (its **Download**, when nothing landed)
continues from what is on disk.

### The Server view's Remove is refused

**Remove** (`POST /v1/models/remove`) answers with the server's reason, shown in
the model band
([Removing a model](guides/models.md#removing-a-model-post-v1modelsremove)):

- `409 model_in_use`: the entry is the model loaded (also unloaded while idle,
  or served from its files under another id), the one a load is loading, the
  speech model or the diarizer loaded beside it, or the one the
  server is downloading. Load another model first, **Cancel** its download, or wait for
  the speech worker to unload it (`--speech-idle-unload` seconds after its last
  transcription).
- `409 download_locked`: `ignis-server model download`, or another server on
  the same `--download-path`, is fetching the entry. Remove it once that ends.
- `409 model_not_downloaded`: nothing of it is under `--download-path`; a row
  that still reads `On disk` is a known model's path, which is never removed.
- `403 removal_disabled`: `--download-from-ui false`; delete the files on the
  server's machine, or turn the field on and restart the server.
- `500 removal_failed`: a file could not be deleted, named in the message with
  what was deleted before it: on Windows, a process holding it open. Close it
  and press **Remove** again.

After a removal the band names any file kept because another entry also lists
it: that entry is on disk or being downloaded, and the entry removed reads
partly downloaded until the other one goes too.

### `not enough room for … under …`

A download, from the Server view, `ignis-server model download` or the start's
own fetch, checks before it writes a byte that the volume holding
`--download-path` has room for the entry's bytes still missing plus
`--download-free-margin` (`1G` by default), and is refused when it has not:
`<n> bytes needed (<missing> still missing plus download.free_margin <margin>),
<free> bytes free`. Nothing was written. Free that much space on the volume, or
point `--download-path` at a larger one, or lower `--download-free-margin` and
restart the server (it is read at start), then download again: what is already
on disk counts, so a partial entry needs only the rest
([Room on the disk](guides/models.md#room-on-the-disk---download-free-margin)).

### `not enough RAM: the load holds … and keeps 4 GiB free for the system, so it needs … available`

The host plan refused the load, before anything was pinned. Flash-Next needs
about 46-47 GB of available RAM; the 27B about 18.6 GB under the `make` defaults
on a 32 GB card.
In a Linux container on cgroup v2, "available" is capped by the container's
memory limit, so give the container more. The message names the plan line to
shrink and the flag that shrinks it, and the same refusal in exact bytes is the
event `ignis.runtime.flash_next_host_plan_refused` (the 27B's is
`ignis.runtime.qwen27b_host_plan_refused`; after a switch,
`GET /v1/models` carries `last_switch.host_plan`). Close other applications,
or lower the retained slots, the KV-RAM arena, `--ngram-hot-bytes` or the
margin; each costs something
([Host RAM for Flash-Next](configuration/host-ram.md#host-ram-for-flash-next),
[for the 27B](configuration/host-ram.md#host-ram-for-the-27b)).

When it names `crossing_line: load_staging`, no flag shrinks those layer
buffers: close other applications. When it names `crossing_line:
vision_tower` or `vision_pool`, the 27B's vision tower or its embedding pool
does not fit in pinned RAM under `--vision-residency host`: start with
`--vision-residency device` to keep both in VRAM, or without vision
([Where the vision tower and the embedding pool live](configuration/vision.md#where-the-vision-tower-and-the-embedding-pool-live)).

### `--spec-backend mtp` refuses the start

Flash-Next's MTP head needs its companion container
(`qwen3_8_flash_next_mtp_3p0-v2.ninfer`) beside the artifact. `model download
qwen3.8-flash-next` fetches both
([Flash-Next speculation](configuration/flash-next-mtp.md)).

### A flag is refused for this model

A field only one model takes is refused at start when set for the other
(`--vision-enabled` and Flash-Next, the n-gram flags and the 27B,
`--spec-decode-lanes` and the 27B). Remove it, or give it in the model's own
family section of the config file. On a model switch it is dropped instead,
with a log line ([Configuration](configuration/index.md)).

## Serving requests

### `401` on every request

An API key is set and the request does not carry it. Send
`Authorization: Bearer <key>`; a request refused `401` loads nothing
([Connect a client](start/connect-a-client.md#the-api-key)).

### `503 server_not_ready`

The model is still loading, or the first warm-up request has not run. Every
`/v1` route answers this with `Retry-After: 5` until it has; `/health` answers
`200` throughout. Wait and retry; point a load balancer at `/ready`, never a
restarting probe ([Deadlines](api/deadlines.md)). One case differs: after a failed switch that could reload nothing, `/ready`
answers it with no `Retry-After`, and only a successful switch helps.

### `503 engine_full`

Every lane is in use. The response carries `Retry-After`, an estimate of when a
lane frees (1 to 60 s). The OpenAI SDKs honour it; other clients should retry
after it. Only the Responses API's WebSocket mode queues a request instead
([The API](api/index.md)).

### `503 model_switching`

A switch or a reload is running, and every other `/v1` route answers this with
`Retry-After: 5` until it ends. `GET /v1/models` shows `status: "switching"`
with `switching: {from, to}`. Retry. If it keeps happening, see
[the switch storm](#the-server-keeps-switching-between-two-models) below
([Switching models](api/model-switching.md)).

### `409 switch_in_progress`

A second `POST /v1/models/switch` while one runs, or a switch sent while an
idle unload was tearing the model down. Wait until `GET /v1/models` says
`serving` (or `unloaded`) and send it again
([Idle unload](api/idle-unload.md)).

### `503 model_switch_failed` or `503 model_load_failed`

`model_switch_failed`: the switch a request began did not land. There is no
`Retry-After`, because each retry would unload and reload the serving model
again. The `reason` says why; for a host-RAM refusal it is the message above.
`GET /v1/models` shows `status: "failed"` with a `reason` and `last_switch`
with the outcome (`refused`, `restored` or `failed`).

`model_load_failed`: the model was unloaded while idle and loading it again
failed. The next request tries again ([Switching models](api/model-switching.md),
[Idle unload](api/idle-unload.md)).

### `503 server_stopping`

The server received SIGTERM while a load was running, or while the request
waited for one, and will not begin another. Send the request again once the
server is back ([Install](start/install.md#the-container-image)). The process
exits at once while it is still downloading or loading; its last line is
`ignis.process.stopped`, naming the `phase` it was in.

### `404 model_not_found`

The `model` is not loaded, not listed with `--switch-known-models`, not a
catalog entry whose artifact is on disk, or `--switch-allow-implicit false` is
set. These are exactly the ids `GET /v1/models` does not list. For a catalog
entry that is not on disk, the message names the
`ignis-server model download <id>` that fetches it; a switch never starts a
download ([Get a model](guides/models.md), [Switching models](api/model-switching.md)).

### `409 model_above_card_class`

The request or the switch names a catalog entry that needs a larger card than
this one, such as Flash-Next on a 16 GB card. The message names both classes,
and the loaded model keeps serving. Name an entry the card loads on
([Get a model](guides/models.md)). When an entry that fits answers to the id as
an alias but its artifact's header names another model family than the id's,
the message names both families: that entry's files, or the `family` its
catalog states, are wrong ([an operator catalog](guides/operator-catalog.md#an-entrys-family-and-an-alias-of-your-own)).

### `409 model_kind_mismatch`

A route of the wrong kind for the loaded model (an image route with a text
model loaded, or the reverse). The message names both kinds and the loaded id. Name a
known model of the route's kind and the server switches to it, or switch
explicitly ([Image generation](api/images.md)).

### `503 speech_unavailable`

The speech model cannot serve, and the message says why, with no `Retry-After`
(waiting changes nothing): `speech.model` is `off`; its files are not under
`download.path` (fetch them with `ignis-server model download qwen3-asr-1.7b`;
nothing downloads by itself); the CPU lacks AVX2; or the model could not be
loaded (the message carries the reason). `GET /v1/models` carries the same
reason in its `speech` object ([Speech to text](api/transcriptions.md)).

### `400` on a transcription naming `file`

The format is not one the server decodes: WAV, FLAC, MP3, AAC, and the Ogg and
Matroska/WebM containers holding FLAC are. Opus (what a browser's recorder
produces) is not decoded yet: send WAV, or convert it. A clip longer than
`speech.max_clip_seconds` (1200 s) is refused too, and never split.

### A live transcription session will not open, or will not commit

- `400` naming `intent`: the URL must be `/v1/realtime?intent=transcription`
  with no `model` in it. A `model` in the URL opens a Realtime conversation
  session, which the server does not serve.
- `503 engine_full` with `Retry-After`: `speech.max_sessions` sessions are open;
  close one, or raise the field.
- `503 speech_unavailable`: the speech model is off, missing or cannot run
  here; the message says why. A browser cannot read a refused upgrade's
  status, so ask the route with a plain `GET` (no upgrade): it is answered what
  the upgrade would be, or `400 upgrade_required` when the socket could open,
  which then points at a proxy that does not carry WebSockets.
- A commit answered `input_audio_buffer_commit_empty` although audio was sent:
  the session runs `server_vad`, the default, and heard no speech since its
  last item (or already committed it on a silence). Let the server commit, or
  send `turn_detection: null` to commit yourself
  ([turn detection](api/realtime-transcription.md#turn-detection)).
- Items cut at short pauses, or speech never heard: raise
  `silence_duration_ms`, or lower `threshold` (higher in a noisy room); the
  server's defaults are `speech.vad_*`.
- An `error` refusing `turn_detection` `semantic_vad`: it is not served. Send
  `server_vad`, or `null` and commit with `input_audio_buffer.commit`.
- A WARN `ignis.speech.stable_text_revised`: the speech executor's transcript
  did not begin with the stable text it had already sent as deltas, so that
  item's deltas do not join to its `completed.transcript` (which is still the
  model's text). The executor promises never to revise stable text: report it,
  with the server's version.
- A WARN `ignis.speech.transcription` with `outcome: "failed"`: the speech
  model failed that transcription or item; its `reason` is the executor's own
  words (a load that failed names the file). The client got `engine_error`, or
  `speech_unavailable` for a load.
- The server refuses to start with `ignis.config.speech_invalid`:
  `speech.model` names a catalog entry that is not a speech model, or a
  `speech.compat_names` entry is an id of the catalog in use (an operator's
  `--download-catalog` among it). The error names the field and the id.
- An `error` `ignis_audio_queue_full` refusing an append: the session holds
  `speech.max_queued_audio_seconds` (120) of audio that the speech model or
  the diarizer has not computed yet, so the client sends faster than this CPU
  computes. The append went nowhere and the session stays open: send it again
  once the session catches up, pace the audio nearer real time, or raise the
  field ([sending audio](api/realtime-transcription.md#sending-audio)).
- An `error` `ignis_diarizer_unavailable` refusing
  `session.ignis.diarization`: the diarizer cannot serve, and the message says
  why (`diarizer.model` is `off`, the catalog has no diarizer entry, its files
  are not downloaded, the CPU lacks AVX2); `GET /v1/models` carries the same
  reason in `speech.diarizer`, with the fix for each
  ([turning it on](api/transcriptions.md#turning-it-on)). The session stays
  open, without speakers. Refused naming `turn_detection` instead: speakers
  need `server_vad`, so the session is on manual commit.
- An `error` `ignis_diarizer_failed`, with a WARN `ignis.speech.diarizer_failed`
  in the log: the diarizer failed in the middle of a session (its weights would
  not load, or a step failed); the WARN's `reason` is its own words. The
  session goes on without speakers, and the items waiting for theirs complete
  without one. A load that names a file points at the download; anything else,
  report it with the server's version.
- The server refuses to start with `ignis.config.diarizer_invalid`:
  `diarizer.model` names a catalog entry that is not a diarizer (a text, image
  or speech model's id, in the catalog in use, an operator's
  `--download-catalog` among it). The error names the field and the id: name a
  `kind: diarizer` entry, `auto` or `off`.

### `vision_disabled`, or `media_budget_exceeded`

`vision is disabled for this server`: an `image_url` part was sent and the
server runs without `--vision-enabled`, whatever the artifact holds. Start with
`--vision-enabled`. `media_budget_exceeded`: several images together are past
`--vision-max-tokens` (32,768 by default); a single oversized image is
downscaled instead ([Vision](configuration/vision.md)).

### A `400` naming a `param`

`unsupported_value` refuses a field whose value asks for something the server
does not do; `invalid_sampling_parameter` an out-of-range sampling value, which
is never clamped. Fix or remove the field the `param` names
([Chat completions](api/chat-completions.md)).

### The answer stops short with `finish_reason: "length"`

Either the `max_tokens` you sent was reached, or, with none sent, the
`--model-default-max-tokens` default (38,912, reasoning included) was. Send a
larger `max_tokens`, or start with `--model-default-max-tokens 0` for none. The
reasoning spends the same budget as the answer, and the thinking budget always
leaves 2,048 tokens for the answer ([Chat completions](api/chat-completions.md),
[Prompting](configuration/prompting.md)).

The deadline can also cut an answer short: it is then a `200` with
`finish_reason: "length"` and the header `ignis-stopped-by: request_timeout`.

### `504 request_timeout`, or a response `incomplete` with `max_output_tokens`

The request ran past `--server-request-timeout` or `--server-stream-timeout`
(2400 s each by default; `0` is none). With `--server-timeout-reply partial`
(the default) a chat answer is returned with what was generated; with `error`
it is a `504 request_timeout`. A decision's question and an image job are an
error either way. The log line `ignis.request.cancelled` says `cause: timeout`
([Deadlines](api/deadlines.md),
[ADR 0058](../adr/0058-every-route-has-a-deadline-and-a-passed-one-answers-what-was-generated.md)).

### The reasoning shows `[LOOP DETECTED, REWINDING n TOKENS / m CHARS · h]`

That is the loop guard: it saw the model repeating itself in its reasoning,
took the repeated text back and nudged the model on. It is working as
designed, and `usage` reports the tokens as `rewound_tokens`. To only log
loops, set `--model-loop-guard log`; `off` turns the guard off
([Prompting](configuration/prompting.md)).

## Performance

### The first request after a quiet spell takes 13-33 seconds

The model was unloaded for being idle (`--model-idle-unload`, 300 s by
default) and the request loads it again. `GET /v1/models` showed
`status: "unloaded"`. Raise the value or set it to `0` to never unload; it can
be patched live. A benchmark should always run with `0`
([Idle unload](api/idle-unload.md)).

### The server keeps switching between two models

A client with two different model ids makes the server reload back and forth,
13-33 s per switch. opencode's `small_model`, Open WebUI's task model and
Hermes set to another listed id do it. Make both ids the same, or start with
`--switch-allow-implicit false`
([Connect a client](start/connect-a-client.md#two-model-ids-make-the-server-switch),
[Switching models](api/model-switching.md)).

### Flash-Next gets slower with MTP on several lanes

On a 5090 the MTP head pays at one active request and costs 7-15% at two or
three lanes. It is off by default there; leave it off when several agents
decode together, or use `--spec-draft-rows 3`
([Flash-Next speculation](configuration/flash-next-mtp.md)).

### Flash-Next prefill is slow with the n-gram table on disk

N-gram rows not held in RAM are read from NVMe when a step needs them. Raise
`--ngram-hot-bytes` if the host has RAM to spare (`auto` takes what the plan
leaves). A start that is slow because it rebuilds the rows is the next entry
([Host RAM](configuration/host-ram.md#host-ram-for-flash-next),
[n-gram cache](configuration/ngram-cache.md)).

## Container

### WARN `n-gram cache not writable; reading the artifact`

The artifact's directory is a read-only mount, or one uid 10001 cannot write,
and `--ngram-persist-path` defaults to `model`. The server still starts, but
every start rebuilds the n-gram hot rows from the artifact. Give it a writable
directory (`-v /path/to/ignis-cache:/cache -e IGNIS_NGRAM_PERSIST_PATH=/cache`),
owned by uid 10001 (rootless podman: `podman unshare chown 10001 /path/to/ignis-cache`)
([Install](start/install.md#the-container-image),
[n-gram cache](configuration/ngram-cache.md)).

### WARN `ignis.kv_disk.off`

"the volume has no room above its margin: the KV-disk tier is off". The tier's
files go next to the artifact by default (`--kv-disk-path model`). The volume
has no room above the 10 GiB margin, or the directory is a read-only mount
([Install](start/install.md#the-container-image)). The load runs without KV-disk: a sequence KV-RAM cannot hold is
prefilled again instead of restored. Point `--kv-disk-path` at a writable
volume with space, or accept it on the 27B, where the tier is off by default
([KV-disk](configuration/kv-disk.md)).

### The Prometheus listener is unreachable from outside the container

The keyless listener stays on the container's own loopback. Add
`-e IGNIS_SERVER_METRICS_BIND=0.0.0.0:9464 -p 9464:9464`; the Playground's
Monitor reads `/ui/metrics` on port 8000 either way
([Install](start/install.md#the-container-image)).

### The pod does not scale to zero

An open Responses WebSocket counts as a connection in flight, and a visible
Monitor tab polls `/ui/metrics` and counts as traffic. The server closes a
socket left idle for `--server-socket-idle-close` seconds (300 by default);
lower it. A hidden tab sends nothing ([Run behind Knative](guides/knative.md)).

### The container does not stop on SIGTERM promptly

While the model downloads or loads, the process exits at once; once it serves
it drains in-flight requests and unloads the model first. A second signal ends
it at once (`phase: shutting_down`). Its last line `ignis.process.stopped`
names the phase. Do not put an init in front: the server is PID 1 on purpose
([Install](start/install.md#the-container-image)).

## Still stuck

- The startup line `ignis.config.source` says which config file was used, and
  `ignis-server config print` shows what would run. `GET /v1/config/origins`
  says where each value comes from ([Configuration](configuration/index.md)).
- `IGNIS_LOG_LEVEL` raises the log level; the lifecycle of a request is the
  `ignis.request.*` events ([Telemetry](guides/telemetry.md)).
- Every config field: [Configuration reference](reference/configuration.md).
- How the server plans memory and schedules requests:
  [How ignis serves](concepts/how-ignis-serves.md).
