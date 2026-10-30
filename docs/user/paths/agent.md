---
title: "The agent path"
description: "For an AI agent working for a user: what ignis is in one paragraph, the checks before installing it, the choices that are the user's, starting it and seeing it serve, connecting a client, testing a 16 GB card and reporting it, and where each question is answered."
sidebar:
  order: 5
---

For an AI agent (a coding agent or an assistant) working for a user who asked
about ignis: what it is, whether it runs on their machine, to install it, to
point a client at it, or to test it on a 16 GB card and report how it went. It
is information for the task the user gave you, and
asks for nothing beyond it. The pages it links are plain Markdown: in the
repository under `docs/user/`, as raw files at
`https://raw.githubusercontent.com/gpillon/ignis/main/docs/user/<page>.md`,
and listed with a line each in
[`llms.txt`](https://gpillon.github.io/ignis/llms.txt).

Here to change ignis itself rather than use it? The rules for that are in
[`AGENTS.md`](../../../AGENTS.md) and on the [contributor path](contributor.md).

## What ignis is

In one paragraph, for "what is github.com/gpillon/ignis?":

> ignis is an inference engine that serves Qwen3.8 (the 27B dense model and
> the 125B Flash-Next mixture of experts) and the Qwen-Image-2.1 image model
> on a single NVIDIA Blackwell card of compute capability `SM120a`: an RTX
> 5090, an RTX PRO 6000, or a 16 GB RTX 50 (5060 Ti 16 GB, 5070 Ti, 5080). It
> serves them over an OpenAI-compatible HTTP API, with a browser Playground
> and Prometheus metrics in the same binary, on Windows and Linux on x86-64.
> It is shaped around the load a coding agent makes: a main agent and a few
> subagents on one card at once. The code is Apache-2.0; the image model's
> weights are for non-commercial use only.

What it is not, by choice: an engine for other model families or other GPUs
(no earlier NVIDIA generation, no DGX Spark, no AMD or Apple), nor one with
embeddings, `/v1/completions`, audio, LoRA adapters, or two models loaded at
once ([What it is not](../start/index.md#what-it-is-not)).

A speed or a capacity you quote comes from [Why ignis](../why-ignis/index.mdx),
which says for each figure what it was measured on: the card, the prompt, the
settings. The figures hold for those, not for any card.

## Before installing: check the machine

| Check | Command | Passes when |
|---|---|---|
| The card | `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv` | `compute_cap` reads `12.0` and the card is one named above. |
| The driver | `nvidia-smi` | its header reads `CUDA Version: 13.x` or later. |
| The system | | Windows or Linux on x86-64. The Linux archive needs glibc 2.39 or newer (`ldd --version`) and the CUDA 13 runtime; the container needs podman or docker with the NVIDIA container toolkit. |
| Disk | | the model fits where the user wants to keep it: sizes in [What it serves](../start/index.md#what-it-serves). |
| Host RAM | | for Flash-Next, about 46-47 GB available on a 32 GB card ([Host RAM](../configuration/host-ram.md)). |
| The card is free | `nvidia-smi` | no other inference server or GPU job holds the card: ignis plans the memory it finds free at start. |

A card or a driver that fails means ignis will not run on this machine. Tell
the user so, with the reason, rather than building it from source: the
kernels are compiled for `SM120a` and nothing else
([Why not DGX Spark?](../guides/dgx-spark.md) explains it for the closest case).

## What the user decides

Ask before the install does anything large or visible:

1. **Which model.** The default is `qwen3.8-27b`; on a 16 GB card the server
   serves its 16 GB entry, `qwen3.8-27b-16g`, under that id.
   `qwen3.8-flash-next` needs the host RAM above.
   `qwen3.8-27b-abliterated` does not refuse anything:
   [its page](../guides/uncensored.md) comes first; on a 16 GB card its 16 GB
   entry, `qwen3.8-27b-abliterated-16g`, serves it. `ignis-image-2.1-turbo`
   makes images and its weights are non-commercial
   ([What it serves](../start/index.md#what-it-serves)).
2. **Where the model is kept, and the download itself.** A model is tens of
   GiB. A server started with no terminal on its stdin, which is how an agent's
   shell tool usually runs a command, does not ask before fetching: it
   downloads ([Letting the server fetch it](../guides/models.md#letting-the-server-fetch-it)).
   Ask first, then fetch it yourself into the directory the user chose.
3. **The release archive or the container** ([Install](../start/install.md)).
4. **Who can reach it.** By default the server listens on `127.0.0.1:8000`
   only. Reaching it from other machines means another bind address and an
   API key ([Access](../configuration/access.md)).
5. **Whether to stop what holds the card**, when the last check found
   something.
6. **For a test of a 16 GB card**, the plan, then the advanced part, posting
   the report, and keeping ignis or removing it
   ([Testing a 16 GB card](#testing-a-16-gb-card)).

## Install

1. **The build.** On Windows the `.zip`. On Linux the container image where
   podman or docker can give it the GPU, otherwise the `.tar.gz`.
2. **The version.** The latest release is
   `gh release view --repo gpillon/ignis --json tagName --jq .tagName`, or the
   `tag_name` of `https://api.github.com/repos/gpillon/ignis/releases/latest`.
   The `develop` prerelease is a rolling development build: only when the user
   asks for it.
3. **Download, check, unpack**, with the commands on
   [Install](../start/install.md#release-archives) for that version: the
   SHA-256 check is part of them. On Linux,
   `ldd ./ignis-server | grep 'not found'` prints nothing when every library is
   there; `libcudart.so.13` is the one a fresh machine lacks, and installing it
   needs administrator rights, which is the user's call.
4. **The model**, the one the user chose, into the directory they chose; no
   GPU is needed for it, and every file is checked against its pinned size and
   SHA-256:

   ```
   ignis-server model download qwen3.8-27b --out <dir>
   ```

   On a 16 GB card the id to fetch is `qwen3.8-27b-16g`. An interrupted
   download resumes from its `.part` file
   ([Get a model](../guides/models.md#fetching-one-ignis-server-model-download)).

For the container, [Quick start](../start/quick-start.md) has the command, with
the model directory mounted and kept; under docker compose,
[`docker-compose.yml.example`](../start/install.md#with-docker-compose) is the
same container as a service.

## Start it, and see it serve

The server runs in the foreground until it is stopped. Start it so that it
does not hold your shell, and keep its output: a start that is refused says
why in its last lines.

```sh
nohup ./ignis-server --download-path <dir> > ignis-server.log 2>&1 &
```

```powershell
Start-Process .\ignis-server.exe -ArgumentList '--download-path', '<dir>' `
  -RedirectStandardOutput ignis-server.log -RedirectStandardError ignis-server.err.log -WindowStyle Hidden
```

`--download-path` points the server at the directory the model was fetched
into. Then:

1. `GET http://127.0.0.1:8000/health` answers `200` once the server listens.
2. `GET http://127.0.0.1:8000/ready` answers `503 server_not_ready` with
   `Retry-After: 5` while the model loads, and `200` once it serves
   ([Deadlines](../api/deadlines.md)). A cold load of the 27B took 13-33 s on
   a 5090.
3. `GET http://127.0.0.1:8000/v1/models` lists the loaded model, and one chat
   completion answers ([Quick start](../start/quick-start.md#3-send-a-request)).
4. The user's own way in is the Playground, at <http://127.0.0.1:8000/ui/>.

A start that is refused names its cause: the message is a heading in
[Troubleshooting](../troubleshooting.md). After 300 seconds without a
request the server unloads the model and loads it again on the next request,
which waits for the load ([Idle unload](../api/idle-unload.md)). Ctrl+C or
SIGTERM stops it after the requests in flight; killing the process frees the
card too.

## Connect a client

Three values: the base URL `http://127.0.0.1:8000/v1`, the model id
`GET /v1/models` lists, and the API key if the user set one.
[Connect a client](../start/connect-a-client.md) has them for the OpenAI
SDKs, curl, opencode and chat front ends. Asked to use ignis as your own
model, set the same three in your harness's OpenAI-compatible provider.

- **One model id for every request.** A client that names two ids in turn,
  such as a separate model for titles and summaries, makes the server reload
  back and forth
  ([Two model ids make the server switch](../start/connect-a-client.md#two-model-ids-make-the-server-switch)).
- **A stable system prompt.** Prompt reuse keys on the start of the prompt:
  a timestamp or a counter in the system prompt makes every turn prefill the
  whole conversation again
  ([The KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md)).

## Testing a 16 GB card

The `rtx50-16gb` profile is derived from an RTX 5090 held inside a 16 GB
budget: no real 16 GB card has run it
([the profile](../concepts/profiles.md#rtx50-16gb)). A user who asks you to
test ignis on their RTX 5060 Ti 16 GB, 5070 Ti or 5080 and report how it went
gives the project its first real numbers. The reports are collected on
[issue #360](https://github.com/gpillon/ignis/issues/360), one comment per
card. The test is the install above with the 16 GB model, a short ladder of
settings for when the default is refused, and a report; after the 13.4 GB
download it takes 15 to 30 minutes.

### The card, and the plan the user approves

```
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,display_active,driver_version --format=csv
```

A 16 GB card reads a `memory.total` near 16,300 MiB. An RTX 50 with 8 or
12 GB passes the other checks, but the 16 GB model does not fit it: the test
ends there, with that reason. `memory.used` and `display_active`, read before
ignis starts, go into the report. The memory the desktop and other programs
hold is what most decides whether the default fits: estimated from the
5090's plans, the default keeps 262,144 tokens of context with about 0.3 GiB
held, adapts to 131,072 with 1 GiB and is refused with 3 GiB. The test is what
measures it.

Before anything large, the user gets one plan to approve:

- where ignis goes, where the model goes (`qwen3.8-27b-16g`, 13.4 GB), and
  the archive or the container;
- what holds the card now, which the test leaves running;
- that the server is started and stopped several times with different
  settings, on `127.0.0.1` only;
- how long it takes;
- that an advanced part is offered once the base report is written, and is
  their choice then;
- that the report is shown to them whole before anything is posted, and that
  at the end they choose to keep ignis or remove what the test installed.

Their yes covers the base test and the report. What waits for the user's word
(below) still waits.

### The base test

1. **Install** and fetch `qwen3.8-27b-16g` as in [Install](#install).
2. **The default first**: a start with no setting named (the profile is
   `auto`, and the load adapts itself), as in
   [Start it, and see it serve](#start-it-and-see-it-serve). The report keeps
   this attempt whatever it does: whether the default holds on a real card is
   the first thing the project wants to know.
3. **When a start is refused**, the next attempt adds one setting to the ones
   before it, until a start serves:

   | Attempt | Adds | What it trades |
   |---|---|---|
   | 1 | nothing | |
   | 2 | `--model-adapt-context-step 16384` | The load lowers the context in steps of 16,384 instead of 65,536, down to 16,384, so it can run a context the default's steps skip over. |
   | 3 | `--spec-backend off` | DFlash2's weights and buffers go to the KV pool; decode is slower. |
   | 4 | `--vram-headroom-bytes 512M` | The profile leaves 1,200 MiB for the desktop. With 512 MiB, a desktop that grows after the start makes Windows page the card's memory and decode collapse; the bench's figures show it. |

   A card refused at attempt 4 is reported as refused, with all four
   attempts: below 16,384 tokens a coding agent has no room to work. For each
   attempt the report keeps its flags and its outcome. For a start that
   served, it keeps `max_model_len`, `decode_lanes` and `adapted` from
   `GET /v1/models`. For a refused one, it keeps the refusal from the last
   lines of the log, which carries the plan's deficit and every adaptation
   tried ([when a profile does not fit](../concepts/profiles.md#when-a-profile-does-not-fit)),
   with each path replaced by `<dir>`.
4. **The smoke check**, on the first start that serves. Two greedy requests
   with thinking off:

   ```bash
   curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 17 * 23? Answer with the number only."}],"enable_thinking":false,"temperature":0,"max_tokens":16}'
   curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write a Python function that reverses a string."}],"enable_thinking":false,"temperature":0,"max_tokens":256}'
   ```

   The first passes when `choices[0].message.content` contains `391`; of the
   second, the report keeps the first five lines. A server that answers
   wrongly is a result of its own, not a refusal: the ladder does not go on,
   and the report says so.
5. **The bench**, with nothing else using the server:

   ```
   ignis-bench small-card --endpoint http://127.0.0.1:8000 --out small-card.json
   ```

   It takes several minutes on a 16 GB card, most of them the 64K
   time-to-first-token cell, which is longer than a shell tool's usual
   timeout. It runs the way the server does, without holding the shell, with
   its progress on stderr ([Measure a 16 GB card](../guides/measure-a-16gb-card.md)).

### The report

The report is in English, whatever language the conversation is in: the
project reads it, and so do other owners of the same card. It is written to a
local file and carries these fields and nothing else:

| Field | From |
|---|---|
| Operating system and version | Windows: `(Get-CimInstance Win32_OperatingSystem).Caption` and `.Version`. Linux: `PRETTY_NAME` in `/etc/os-release`, and `uname -r` |
| Driver, CUDA version | `driver_version` above; the header of `nvidia-smi` |
| CPU | Windows: `(Get-CimInstance Win32_Processor).Name`. Linux: `Model name` in `lscpu` |
| Host RAM | the total, rounded to the GB |
| Card | `name` and `memory.total` |
| VRAM held before the start | `memory.used` in MiB, and `display_active`; no process names |
| ignis | `ignis-server version`; archive or container |
| Attempts | per attempt, as above |
| Smoke check | pass or fail, and the five lines |
| The bench | its record, which carries no key, path or address ([what it never records](../guides/measure-a-16gb-card.md#what-it-never-records)) |

Nothing else goes in: no user or host name, no path, no IP address, no GPU
UUID or serial number, no process name, no environment variable. The summary
line reads the record's `decode.coding.median_tokens_per_s`,
`decode.agent.median_tokens_per_s`, `decode.speculative.acceptance` and the
`median_ttft_ms` of each `ttft` cell:

````markdown
### <card> · <OS> · default: works | adapted | refused · runs with: <flags, or "the default">

| | |
|---|---|
| OS | <name and version> |
| Driver / CUDA | <driver> / <CUDA version> |
| CPU | <model> |
| Host RAM | <n> GB |
| Card | <name>, <memory.total> MiB |
| VRAM held before the start | <n> MiB, display active: <yes or no> |
| ignis | <version>, <archive or container> |

| # | Flags | Outcome | Context | Lanes | Adapted, or the refusal |
|---|---|---|---|---|---|
| 1 | (the default) | ... | ... | ... | ... |

**Smoke check:** 17 * 23 -> <answer> (<pass or fail>)

```python
<the first five lines>
```

**Summary:** decode <n> tok/s on the coding prompts, <n> tok/s on the agent
turns; DFlash2 acceptance <n>; time to first token <n> s at 8K, <n> s at 64K.

<details><summary>ignis-bench small-card</summary>

```json
<the whole record>
```

</details>

**Notes from the user:** <anything they want to add, or "none">
````

The user reads it whole before it goes anywhere. Posting it is their call: with
their yes and a `gh` already signed in (`gh auth status`), it goes up with
`gh issue comment 360 --repo gpillon/ignis --body-file <report>`; otherwise
they paste it into [issue #360](https://github.com/gpillon/ignis/issues/360)
themselves.

### The advanced part

It is offered once the user has seen the base report, as their choice. It
takes about 30 minutes, which they can stop at any time. Each step starts
from the settings the base test served with:

- **The context ceiling.** `--model-max-context` named in steps of 16,384
  above the context the base ran, up to 262,144, until a start is refused. A
  named context does not adapt, so the last one that served is the ceiling.
- **One lane against two.** `--spec-decode-lanes 1`, then `2`, at the same
  context, each measured with
  `ignis-bench small-card --cells 8192`.
- **DFlash2 on against off**, at the same context, measured the same way. It
  shows whether speculation pays on this card.
- **The default on a freer card**, only when the base needed a setting. The
  user closes what holds the card, or moves the display to the integrated
  GPU, `memory.used` is read again, and the default is tried again.

Between runs, a server left 300 seconds without a request unloads the model,
and the next request waits for it to load again
([Idle unload](../api/idle-unload.md)): a slow first answer is that, not a
hang. The results go into the same report, under an `Advanced` heading, and
end with the best settings found: the start command that runs them, and the
config file below.

### Keeping it, or removing it

At the end the user chooses to keep ignis or to remove it.

- **Keeping it**: the start command with the settings that worked, the
  Playground at <http://127.0.0.1:8000/ui/>, and
  [Connect a client](../start/connect-a-client.md). A config file lets a bare
  `ignis-server` start with those settings. It goes where a start looks with
  no `--config`: `%APPDATA%\ignis\config.yaml` on Windows,
  `$XDG_CONFIG_HOME/ignis/config.yaml` or `~/.config/ignis/config.yaml` on
  Linux ([the config file](../configuration/index.md)). It holds only the
  model's directory and the settings the test named, not every field: a value
  in the file comes before the profile, so a file of every default (what
  `config generate` writes) would hold them over `rtx50-16gb` and over a
  later release's defaults. `config patch` writes such a file, and checks it:
  the file starts as `{}`, then

  ```
  ignis-server config patch --file <path> --download-path <dir> --model-adapt-context-step 16384
  ```

  The user sees the path, the reason and the file's contents before it is
  written. A file already at that path is not replaced: they see what would
  change, and decide. `ignis-server config print` then shows what a bare start
  runs. With the container, the file goes in the mounted model directory and
  `IGNIS_CONFIG` names it.
- **Removing it**: the test keeps, from its first step, a list of what it
  created: the ignis directory, the model it fetched, the logs, the config
  file, and for the container its image, container and volume. A directory
  the user already had is not on it, only what the test added inside. The
  user sees every file and directory on the list before anything is deleted,
  and the deleting waits for their yes, with the server stopped first. The
  report stays until it is posted.
  Anything installed with administrator rights, such as the CUDA runtime, is
  named and stays, unless the user asks for it to go.

### Without a shell

An assistant that cannot run commands follows the same steps by giving the
user one command at a time and reading the output they paste back. The plan,
the ladder and the report are the same.

## Where each question is answered

| The user asks about | Read |
|---|---|
| What it is, what it runs on, what it does not do | [What ignis is](../start/index.md) |
| How fast, how much context, measured on what | [Why ignis](../why-ignis/index.mdx) |
| The models, their sizes, where they come from | [Get a model](../guides/models.md), [the catalog](../reference/models.md) |
| A setting, its default, what it trades | [Configuration](../configuration/index.md), [every config field](../reference/configuration.md) |
| An endpoint, a field, a status code | [The API](../api/index.md), [HTTP routes](../reference/http-routes.md); a running server serves `/v1/openapi.json` |
| A refusal, an error, a WARN line | [Troubleshooting](../troubleshooting.md) |
| The Playground | [The Playground](../playground/index.md) |
| How it serves a request, and why | [How ignis serves a request](../concepts/how-ignis-serves.md) |
| What changed in a release | [The releases](https://github.com/gpillon/ignis/releases) |
| Testing a 16 GB card and reporting it | [Testing a 16 GB card](#testing-a-16-gb-card), [Measure a 16 GB card](../guides/measure-a-16gb-card.md), [issue #360](https://github.com/gpillon/ignis/issues/360) |

## What waits for the user's word

- Downloading a model, or anything else of that size.
- Installing a driver, the CUDA runtime, or anything else that needs
  administrator rights.
- Stopping a process that holds the card.
- Binding beyond `127.0.0.1`, opening a firewall port, or exposing the server
  any other way.
- Building from source instead of taking a release: it needs a CUDA toolkit
  and a long build ([Building from source](../contributing/building.md)).
- The uncensored model, and the image model for anything commercial.
- Posting a report, or anything else, to GitHub or any other site.
- Writing a config file, and deleting what an install or a test put on the
  machine.
