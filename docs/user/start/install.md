---
title: "Install"
description: "Get ignis: the native binaries for Windows and Linux, or the container image, and what each needs on the host."
sidebar:
  order: 2
---

Every `v*` release publishes ignis three ways, all from the same compile:

| Build | Runs on | The host needs |
|---|---|---|
| **Windows `.zip`** | Windows on x86-64 | an NVIDIA driver new enough for CUDA 13; the zip carries the CUDA runtime itself |
| **Linux `.tar.gz`** | Linux on x86-64, glibc 2.39 or newer (Ubuntu 24.04 or a distribution of its age) | an NVIDIA driver new enough for CUDA 13, and the CUDA 13 runtime (`libcudart.so.13`) |
| **Container image** `ghcr.io/gpillon/ignis` | `linux/amd64`, under podman or docker with the NVIDIA container toolkit | the NVIDIA driver; the image carries the CUDA runtime |

All three are compiled for `SM120a` only: the RTX 5090, the RTX PRO 6000 and
the 16 GB RTX 50 class ([what it runs on](index.md#what-it-runs-on)). None
carries a model: the server fetches one on first start, or you point it at a
`.ninfer` file you already have ([Get a model](../guides/models.md)). To build
from a checkout instead, see [Building](../contributing/building.md).

A coding agent can do all of this for you, asking you the choices that are
yours: [Install with a coding agent](install-with-an-agent.md).

**Check the driver first.** `nvidia-smi` prints, in its header, the highest
CUDA version the driver supports: it must read `CUDA Version: 13.x` or later.
A driver that reports 12.x cannot run any of the three; update it from NVIDIA.

## Release archives

The native binaries, for running ignis directly on the machine with no
container. They are on the
[Releases page](https://github.com/gpillon/ignis/releases), each beside its
SHA-256:

- `ignis-v<version>-x86_64-pc-windows-msvc.zip`
- `ignis-v<version>-x86_64-unknown-linux-gnu.tar.gz`

Each unpacks into one folder of the same name, holding:

| File | What it is |
|---|---|
| `ignis-server` (`.exe` on Windows) | The server: the OpenAI-compatible API, the Playground at `/ui/` and the metrics |
| `ignis-bench` | The measuring tools: the canary suite, trace replay, and `small-card`, the report a 16 GB card owner sends ([measure a 16 GB card](../guides/measure-a-16gb-card.md)) |
| `ignis-artifact-inspect` | Prints a `.ninfer` artifact's identity, its content hash and its contents: `ignis-artifact-inspect <file>.ninfer` |
| `cudart64_*.dll` (Windows only) | The CUDA 13 runtime, which NVIDIA lets a program ship |
| `README.md`, `LICENSE`, `NOTICE`, `NOTICE-kernel` | The licence (Apache-2.0) and the vendored kernels' own terms |

### On Windows

In PowerShell, download, check and unpack the zip (the commands use the
[GitHub CLI](https://cli.github.com/); downloading both files from the Releases
page works the same):

```powershell
$v = "v0.8.0"
$zip = "ignis-$v-x86_64-pc-windows-msvc.zip"
gh release download $v --repo gpillon/ignis --pattern "$zip*"
(Get-FileHash $zip).Hash -eq (Get-Content "$zip.sha256").Split(" ")[0]   # True
Expand-Archive $zip -DestinationPath .
cd "ignis-$v-x86_64-pc-windows-msvc"
.\ignis-server.exe
```

Nothing else is installed: the folder can live anywhere, and deleting it
removes ignis. On Windows the display driver pages video memory out to system
RAM instead of failing an allocation, so the server keeps 1.5 GiB of the card
free for the desktop and the browser; a desktop that holds more makes the 27B
start with a shorter context rather than slow down
([Windows and WDDM](../concepts/vram-plan.md#windows-and-wddm)).

### On Linux

The binaries are built on Ubuntu 24.04 and load the CUDA 13 runtime
(`libcudart.so.13`) from the host, which the driver does not provide. Install
it from NVIDIA's CUDA repository for your distribution, or with a CUDA 13
toolkit, and make sure the loader finds it (its `lib64` on `LD_LIBRARY_PATH`
if it is not in a system path). Then:

```sh
v=v0.8.0
tarball=ignis-$v-x86_64-unknown-linux-gnu.tar.gz
gh release download $v --repo gpillon/ignis --pattern "$tarball*"
sha256sum -c "$tarball.sha256"
tar -xzf "$tarball"
cd "ignis-$v-x86_64-unknown-linux-gnu"
ldd ./ignis-server | grep 'not found'    # prints nothing when every library is there
./ignis-server
```

`ldd` names any library the loader cannot find; `libcudart.so.13` is the one
a fresh machine is usually missing.

### The first start

With no arguments the server listens on `127.0.0.1:8000`, chooses the
built-in profile for the card's memory (`--profile auto`,
[Profiles](../concepts/profiles.md)), and, with no model on disk, offers to
fetch the default one, `qwen3.8-27b`, into `./models` beside where you ran it.
On a terminal it asks first (`[y/N]`), naming the size, the source and the
destination; the 27B is 18.1 GiB, and on a 16 GB card its 16 GB entry, 12.5
GiB. Every file is checked against its pinned size and SHA-256 before use.

- **Keep the model elsewhere**: `--download-path D:\models` (or
  `/srv/models`) fetches there and looks there on the next start.
- **Fetch ahead of time**, with no GPU needed:
  `ignis-server model download qwen3.8-27b --out <dir>`.
- **A model you already have**: `--model-artifact <file>.ninfer`.
- **Reach it from other machines**: `--server-bind 0.0.0.0:8000`, with an API
  key ([access](../configuration/access.md)).
- **Every setting** is a flag, an `IGNIS_*` variable and a config-file key:
  `ignis-server help --fields` lists them ([Configuration](../configuration/index.md)).

Once the model serves, the Playground is at <http://127.0.0.1:8000/ui/>, the
API at `http://127.0.0.1:8000/v1`, and Prometheus at
`http://127.0.0.1:9464/metrics`. **Ctrl+C** stops the server: it finishes the
requests in flight, unloads the model and exits.

### Development builds

Every push to `main` also publishes the same archives as a rolling `develop`
prerelease on the Releases page, and the image as
`ghcr.io/gpillon/ignis:develop`. That is a development build, **not a
release**: it is replaced by the next push and never carries a version.

## The container image

A `v*` tag publishes a `linux/amd64` image to `ghcr.io/gpillon/ignis`, tagged
with the version and `latest`. The image carries the CUDA runtime but no
driver: the host's NVIDIA driver is injected by the container runtime, and the
model is mounted, never baked in.

The `develop` tag is the rolling development build ([development builds](#development-builds)).

```
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e IGNIS_MODEL_ARTIFACT=/models/qwen3_8_27b_nvfp4full-v2.ninfer \
  ghcr.io/gpillon/ignis:latest
```

(`docker`: `--gpus all` in place of `--device`.) Every flag has an `IGNIS_*`
environment variable; anything after the image name is passed to the server.
The image sets no default command, so the Playground is on — the server's own
default. So are metrics: the Playground's Monitor reads them at `/ui/metrics`
on port 8000, while the keyless Prometheus listener stays on the container's
own loopback. To scrape it from outside, add
`-e IGNIS_SERVER_METRICS_BIND=0.0.0.0:9464 -p 9464:9464`.

The server is the image's entrypoint, so it is the container's PID 1, and it
needs no init in front of it: `ignis-server` (not its `model download`
subcommand) stops on SIGTERM (or Ctrl+C) from the moment it starts. Before it
serves — while the model downloads or loads — the process exits at once,
which frees the GPU, its last line `ignis.process.stopped` naming the
`phase` it was in; once it serves it drains in-flight requests and unloads
the model first. A load while serving (a switch, a reload, the load after an
idle unload) is not waited for either: a request waiting for it is answered
`503 server_stopping`, and so is one that would begin a load (`POST
/v1/models/switch`, a `PATCH /v1/config` reload, a request naming another
model or needing the unloaded one); once the drain is over the process exits
under the load, its last line naming the `phase` `GET /v1/models` reports it
in (`switching`, or `loading` after an idle unload). A second signal while the
shutdown runs ends the process at once (`phase: shutting_down`). Its probes
are [`/health` and `/ready`](../api/index.md).

With no `IGNIS_MODEL_ARTIFACT` the image fetches the model into its own working
directory (`/home/ignis/models`) and loses it with the container. Mount a
directory and point the download at it to keep what it fetches:

```
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models \
  ghcr.io/gpillon/ignis:latest --download-path /models
```

**The image's user.** The server runs as `ignis` (uid 10001), never as root,
so whatever it reads under a mount must be readable by uid 10001. A model
directory or Hugging Face cache that root wrote with `0700` directories is
not readable, and the load fails on it. Two Flash-Next features also *write*
next to the artifact by default (`model`): the n-gram cache
(`--ngram-persist-path`) and the [KV-disk](../configuration/kv-disk.md) tier (`--kv-disk-path`).
On a read-only mount, like the `:ro` one above, or one uid 10001 cannot write,
the server still starts, with two costs:
- every start rebuilds the n-gram hot rows from the artifact, with the WARN
  `n-gram cache not writable; reading the artifact`;
- the load runs without KV-disk, with the WARN `ignis.kv_disk.off`.

To keep both, give them a directory uid 10001 can write (rootless podman maps
uids: there, `podman unshare chown 10001 /path/to/ignis-cache`):

```
mkdir -p /path/to/ignis-cache && sudo chown 10001 /path/to/ignis-cache
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models:ro -v /path/to/ignis-cache:/cache \
  -e IGNIS_MODEL_ARTIFACT=/models/qwen3_8_flash_next_trellis_a25-v2.ninfer \
  -e IGNIS_NGRAM_PERSIST_PATH=/cache -e IGNIS_KV_DISK_PATH=/cache \
  ghcr.io/gpillon/ignis:latest
```

To smoke-test the image with no GPU and no model — the deterministic CPU mock
([ADR 0006](../../adr/0006-exclusive-gpu-testing.md)) — say so:

```
podman run --rm -p 8000:8000 ghcr.io/gpillon/ignis:latest --download-enabled false
```

`Containerfile` builds the same thing locally (`podman build -t ignis:dev .`).
Its `artifacts` stage is what CI exports the Linux tarball from, so the release
binaries and the image binaries are the same build
([ADR 0032](../../adr/0032-release-through-the-container-build.md)).

### With docker compose

[`docker-compose.yml.example`](../../../docker-compose.yml.example), at the
root of the repository, is the same container as a compose service: the card
through the NVIDIA container toolkit, the model downloaded into `./models` and
kept there, the API and the Playground on `127.0.0.1:8000`, and a stop grace
period long enough for the drain. Its comments carry the alternatives: a model
you already have, an API key, the metrics port, and the CDI device for podman.

```
curl -fLo docker-compose.yml https://raw.githubusercontent.com/gpillon/ignis/main/docker-compose.yml.example
mkdir -p models && sudo chown 10001 models
docker compose up -d
docker compose logs -f ignis
```

The port is published on the host's loopback only, since the API has no key
until one is set: to reach it from other machines, set `IGNIS_SERVER_API_KEY`
([access](../configuration/access.md)) and drop `127.0.0.1:` from `ports`.
`docker compose down` stops the server, after the drain, and leaves the model
in `./models`.
