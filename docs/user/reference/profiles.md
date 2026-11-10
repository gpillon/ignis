---
title: "Built-in profiles"
description: "The hardware profiles built into ignis-server and every value each one sets, general and per model."
sidebar:
  order: 2
  badge:
    text: generated
    variant: note
editUrl: false
---

:::note
Generated from the code by `make docs-reference` (`crates/server/src/user_docs.rs`); an edit here is overwritten, and `cargo test` fails while this page and the code disagree.
:::

A profile is a layer of defaults for one kind of card, between the config file and the built-in defaults: any flag, env var or file value still wins over it. An unset `--profile` is `auto`, and `auto` takes the built-in profile of the card's class, read from its total memory; `--profile none` skips the layer. A config file's `profiles:` section defines more, or shadows these ([Profiles and defaults](../concepts/profiles.md)). `rtx50-16gb` is the former name of `rtx-16gb` and is still accepted everywhere a profile is named, as the same profile.

A profile follows the card's memory, not its generation, and what a card loads follows from both: [Where each model loads](models.md#where-each-model-loads) gives each built-in model on each card ignis runs on ([What it runs on](../start/index.md#what-it-runs-on)).

## rtx-16gb

`--profile auto` picks it on a card of the 16 GB class: a total memory below 20 GiB.

| Field | Model | Value |
|---|---|---|
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-27B | `262144` |
| [`model.prefill_chunk`](configuration.md#modelprefill_chunk) | Qwen3.8-Flash-Next | `2048` |
| [`model.prefill_chunk_solo`](configuration.md#modelprefill_chunk_solo) | Qwen3.8-27B | `1024` |
| [`spec.backend`](configuration.md#specbackend) | Qwen3.8-27B | `dflash2` |
| [`spec.decode_lanes`](configuration.md#specdecode_lanes) | Qwen3.8-27B | `2` |
| [`spec.draft_head`](configuration.md#specdraft_head) | Qwen3.8-27B | `auto` |
| [`spec.draft_tokens`](configuration.md#specdraft_tokens) | Qwen3.8-27B | `7` |
| [`vision.enabled`](configuration.md#visionenabled) | Qwen3.8-27B | `true` |
| [`vision.max_tokens`](configuration.md#visionmax_tokens) | every model | `4096` |
| [`vision.residency`](configuration.md#visionresidency) | every model | `host` |
| [`vram.headroom_bytes`](configuration.md#vramheadroom_bytes) | Qwen3.8-27B | `1200M` |

## rtx-24gb

`--profile auto` picks it on a card of the 24 GB class: a total memory from 20 to under 28 GiB.

| Field | Model | Value |
|---|---|---|
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-27B | `1048576` |
| [`model.prefill_chunk`](configuration.md#modelprefill_chunk) | Qwen3.8-27B | `2048` |
| [`model.prefill_chunk`](configuration.md#modelprefill_chunk) | Qwen3.8-Flash-Next | `4096` |
| [`model.prefill_chunk_solo`](configuration.md#modelprefill_chunk_solo) | Qwen3.8-27B | `2048` |
| [`reuse.retained_device`](configuration.md#reuseretained_device) | Qwen3.8-27B | `2` |
| [`spec.backend`](configuration.md#specbackend) | Qwen3.8-27B | `dflash2` |
| [`spec.draft_tokens`](configuration.md#specdraft_tokens) | Qwen3.8-27B | `7` |
| [`vision.enabled`](configuration.md#visionenabled) | Qwen3.8-27B | `true` |
| [`vision.residency`](configuration.md#visionresidency) | every model | `host` |
| [`vram.headroom_bytes`](configuration.md#vramheadroom_bytes) | Qwen3.8-27B | `1200M` |

## rtx5090

`--profile auto` picks it on a card of the 32 GB class: a total memory from 28 to 40 GiB.

| Field | Model | Value |
|---|---|---|
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-27B | `917504` |
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-Flash-Next | `589824` |
| [`model.prefill_chunk`](configuration.md#modelprefill_chunk) | Qwen3.8-27B | `2048` |
| [`model.prefill_chunk`](configuration.md#modelprefill_chunk) | Qwen3.8-Flash-Next | `8192` |
| [`spec.backend`](configuration.md#specbackend) | Qwen3.8-27B | `dflash2` |
| [`spec.decode_lanes`](configuration.md#specdecode_lanes) | Qwen3.8-Flash-Next | `3` |
| [`spec.draft_tokens`](configuration.md#specdraft_tokens) | Qwen3.8-27B | `7` |
| [`vision.enabled`](configuration.md#visionenabled) | Qwen3.8-27B | `true` |
| [`vision.residency`](configuration.md#visionresidency) | every model | `host` |
| [`vram.headroom_bytes`](configuration.md#vramheadroom_bytes) | Qwen3.8-27B | `1200M` |
| [`vram.headroom_bytes`](configuration.md#vramheadroom_bytes) | every model | `1536M` |
| [`vram.pinned_expert_bytes`](configuration.md#vrampinned_expert_bytes) | every model | `2G` |

## rtxpro6000

`--profile auto` picks it on a card of the 96 GB class: a total memory from 88 to 104 GiB.

| Field | Model | Value |
|---|---|---|
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-27B | `1048576` |
| [`model.max_context`](configuration.md#modelmax_context) | Qwen3.8-Flash-Next | `1048576` |
| [`model.prefill_chunk`](configuration.md#modelprefill_chunk) | Qwen3.8-27B | `2048` |
| [`model.prefill_chunk`](configuration.md#modelprefill_chunk) | Qwen3.8-Flash-Next | `8192` |
| [`ngram.hot_bytes`](configuration.md#ngramhot_bytes) | every model | `auto` |
| [`spec.backend`](configuration.md#specbackend) | Qwen3.8-27B | `dflash2` |
| [`spec.backend`](configuration.md#specbackend) | Qwen3.8-Flash-Next | `mtp` |
| [`spec.decode_lanes`](configuration.md#specdecode_lanes) | Qwen3.8-Flash-Next | `8` |
| [`spec.draft_tokens`](configuration.md#specdraft_tokens) | Qwen3.8-27B | `7` |
| [`spec.draft_tokens`](configuration.md#specdraft_tokens) | Qwen3.8-Flash-Next | `3` |
| [`vision.enabled`](configuration.md#visionenabled) | Qwen3.8-27B | `true` |
| [`vision.residency`](configuration.md#visionresidency) | every model | `device` |
