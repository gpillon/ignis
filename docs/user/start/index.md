---
title: "What ignis is"
description: "What ignis serves, on which cards and operating systems, what comes with it, and what it deliberately does not do."
sidebar:
  order: 1
---

ignis is an inference engine for one family of models on NVIDIA RTX cards. It
loads a Qwen3.8 model onto a single card, Blackwell, Ampere or Ada, and serves
it over an OpenAI-compatible HTTP API. The server is one binary (`ignis-server`): the
API, a browser Playground and Prometheus metrics ship in it.

It is shaped around the load a coding agent makes: one main agent plus a few
subagents sending requests to the same card at once. The 27B decodes eight
requests ("lanes") in one batch-wide round, the KV cache is budgeted in bytes
rather than in sequences, and the state of a conversation survives the request
that built it. How that works is in
[How ignis serves](../concepts/how-ignis-serves.md).

## What it serves

Twelve entries are built into the server's [catalog](../guides/models.md#the-catalog):
the eight text and image models below, three speech models and a diarizer. It
loads one of the eight at a time and switches between them without a restart.

| Model | Id | What it is | Size on disk |
|---|---|---|---|
| Qwen3.8-27B | `qwen3.8-27b` (the default) | A dense text model, NVFP4, with a DFlash2 draft module for speculative decoding and an optional vision tower for image input. | 18.1 GiB |
| Qwen3.8-27B for 16 GB cards | `qwen3.8-27b-16g` | The same 27B with a 3-bit trellis body, for the 16 GB class; it answers to `qwen3.8-27b` on a 16 GB card. | 12.5 GiB |
| Qwen3.8-27B, uncensored | `qwen3.8-27b-abliterated` | The same weights with an abliteration applied. It does not refuse, so [read its page](../guides/uncensored.md) before exposing it. | 18.1 GiB |
| Qwen3.8-27B, uncensored, for 16 GB cards | `qwen3.8-27b-abliterated-16g` | The 16 GB artifact with the same abliteration; it answers to `qwen3.8-27b-abliterated` on a 16 GB card. | 12.5 GiB |
| Swift 1.5 | `swift-1.5-27b` | UkisAI's post-training of the 27B, a [fine-tune](../guides/fine-tunes.md). Its license does not cover commercial use by an organization whose gross revenue was US$1,000,000 or more in its last fiscal year: [read its page](../guides/fine-tunes.md#its-license). | 18.1 GiB |
| Swift 1.5 for 16 GB cards | `swift-1.5-27b-16g` | Swift 1.5 with a 3-bit trellis body; it answers to `swift-1.5-27b` on a 16 GB card. | 12.5 GiB |
| Qwen3.8-Flash-Next | `qwen3.8-flash-next` | A 125B mixture-of-experts model with 6B active parameters, its experts re-quantized to 2.5 bits per weight on average. A VRAM cache holds the hot experts and the rest stream from host RAM. | 68.1 GiB with its MTP and vision companions |
| Qwen-Image-2.1-Turbo | `ignis-image-2.1-turbo` | The [image model](../api/images.md): generation and edits from up to ten input images. Its weights are for non-commercial use only. | 15.6 GiB |

You can add models of your own with an
[operator catalog](../guides/operator-catalog.md).

## What it runs on

- **Cards.** By GPU generation (Ampere is RTX 30, Ada RTX 40, Blackwell
  RTX 50 and RTX PRO):
  - *Blackwell (compute capability 12.0).* One RTX 5090 (32 GB) or one RTX
    PRO 6000 Blackwell (96 GB) loads every model. The 16 GB class (RTX 5060 Ti
    16 GB, 5070 Ti, 5080) loads the 16 GB entries (the 27B, its uncensored
    variant, Swift 1.5), on a profile derived rather than measured.
  - *Ampere.* The RTX 3090 and the RTX 3090 Ti (8.6, 24 GB) are
    supported, on the 16 GB entries (the 27B, its uncensored variant, Swift
    1.5).
  - *Ada.* The RTX 4090 (24 GB) and the 16 GB RTX 40 cards (RTX
    4060 Ti 16 GB, RTX 4080) are verified for correctness, on the same entries:
    the release's [sm_86 gate](../contributing/releases.md#the-sm_86-gate) (the
    model's quality, the kernel tests and a smoke test of the server) passed on
    an Ada Lovelace card (sm_89)
    ([the finding](../../findings/2026-11-08-sm86-gate-on-the-release-range.md)).
  - *Blackwell only.* The NVFP4 builds (the 27B, its uncensored variant,
    Swift 1.5), Flash-Next and the image model, which also need a card of
    32 GB or more, by their card class. On an RTX 30 or RTX 40 card the server
    refuses them, and the NVFP4 builds' ids are served by their 16 GB entries.
  - *The prefill on an RTX 30 or RTX 40 card.* Unset, the 27B reads a prompt
    through the **dense prefill** (each trellis matrix decoded once per chunk
    and multiplied on the integer MMA), the **integer band** of the prompt
    attention (INT8 scores) and ignis's own gated delta net prefill kernels,
    which every card runs.
    `--model-prefill-linear fp16` and `--model-prefill-attention fp16` name
    the fused prefill and the FP16 band instead
    ([Context and speculation](../configuration/context-and-speculation.md)).
  - This guide gives no speed for an RTX 30 or RTX 40 card.

  Each catalog entry states the smallest card class and the oldest GPU
  generation it loads on, and the server refuses one above the card before it
  unloads or fetches anything:
  [Where each model loads](../reference/models.md#where-each-model-loads) is
  the table, every built-in model on every card above. A release is built
  for compute capabilities 8.6 and 12.0 (`86;120a`). Other cards, including
  the DGX Spark, are not
  supported; [why not DGX Spark?](../guides/dgx-spark.md) gives the reasoning.
  A start on one is refused before anything loads, naming the card, its
  compute capability and the ones the build supports
  ([Troubleshooting](../troubleshooting.md#this-builds-kernels-run-on-compute-capability-120-rtx-50-rtx-pro-built-for-120a-the-card-is--89-which-they-were-not-built-for)).
- **Operating systems.** Windows and Linux on x86-64. A `v*` release publishes
  a Windows `.zip`, a Linux `.tar.gz` and a `linux/amd64` container image
  ([Install](install.md)). There is no ARM build.
- **Driver.** The host's NVIDIA driver must support the CUDA major version the
  build was compiled with (the Linux tarball needs the CUDA 13 runtime). The
  container image carries no driver; the container runtime injects the
  host's.
- **Host RAM.** Flash-Next needs about 46-47 GB of available RAM on a 32 GB
  card, the 27B about 18.6 GB there under the `make` defaults
  ([Host RAM](../configuration/host-ram.md)).
- **One model on the card at a time.** The VRAM plan is decided at load for a
  single model; a switch is a full reload.

Without a GPU build, or with `--download-enabled false`, the server runs a
deterministic CPU mock. It exercises the protocol and the scheduler and
generates no real text. Use it to smoke-test a client, not to get answers.

## What you get

- **An OpenAI-compatible API** under `/v1`: [chat completions](../api/chat-completions.md)
  (streaming and not), [image generation and edits](../api/images.md), model
  listing and [switching](../api/model-switching.md), token counting
  ([`/v1/tokenize`](../api/count.md)). [The API](../api/index.md) lists every
  route, and `/v1/docs/` on a running server is an interactive reference.
- **The Responses API**, over HTTP and over a WebSocket
  ([Responses](../api/responses.md)).
- **A System One-compatible decision API**, `/v1/systemone` and `/v1/decide`,
  that answers typed questions over text or an image from one prefill without
  generating tokens ([Decisions](../api/decide.md)).
- **The models**, each a public Hugging Face repository the server fetches
  and checks by itself: the 27B ([NVFP4](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer),
  [for 16 GB cards](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-ignis),
  [uncensored](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer),
  [uncensored for 16 GB cards](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis)),
  [Flash-Next](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis)
  and the [image model](https://huggingface.co/gpillon/ignis-image-2.1-turbo)
  ([Get a model](../guides/models.md)).
- **The Playground** at `/ui/`: chat with agents, Decide, an Image studio, a
  Monitor and a Server view for loading models and changing the configuration
  ([The Playground](../playground/index.md)).
- **Metrics.** Prometheus text on a loopback listener (`127.0.0.1:9464`), and
  the same exposition at `/ui/metrics` for the Monitor
  ([Telemetry](../guides/telemetry.md)).
- **Operations behaviour**: idle unload that frees the card after five
  minutes ([Idle unload](../api/idle-unload.md)), per-request
  [deadlines](../api/deadlines.md), `/health` and `/ready` probes, and
  an API key when you want one ([Access](../configuration/access.md)).

## What it is not

These are choices, not gaps:

- No embeddings and no generic reranker; a second model on the card is not
  planned for. Classification goes through `/v1/decide`.
- No `/v1/completions` and no `/v1/batches`.
- No audio out (speech is transcribed, never spoken), and no LoRA adapters.
- No more than one model loaded at once.
- Not a general engine: other model families are not served.

A request field ignis does not support is refused with a `400` naming it,
never dropped silently ([Chat completions](../api/chat-completions.md)).

## Where to go next

1. [Install](install.md): the container image and the release archives.
2. [Quick start](quick-start.md): from nothing to a first answer.
3. [Connect a client](connect-a-client.md): the SDKs, curl, opencode and
   front ends.
4. [Get a model](../guides/models.md): the catalog, downloads and your own
   artifacts.
5. [Configuration](../configuration/index.md): fields, profiles and the config
   file; [every field](../reference/configuration.md) in one table.
6. [Troubleshooting](../troubleshooting.md), when something does not start.
