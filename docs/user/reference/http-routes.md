---
title: "HTTP routes"
description: "Every route of the HTTP surface, from the OpenAPI document the server builds from its handlers."
sidebar:
  order: 5
  badge:
    text: generated
    variant: note
editUrl: false
---

:::note
Generated from the code by `make docs-reference` (`crates/server/src/user_docs.rs`); an edit here is overwritten, and `cargo test` fails while this page and the code disagree.
:::

The routes as the server's own OpenAPI document lists them: a route cannot exist without its entry ([ADR 0036](../../adr/0036-the-v1-surface-documents-itself.md)). A running server serves the full document, schemas included, at `/v1/openapi.json` and a browsable page at `/v1/docs/`, both without a key. [The API](../api/index.md) explains each route.

## audio

| Method | Path | What it does |
|---|---|---|
| POST | `/v1/audio/transcriptions` | A transcript from audio |
| GET | `/v1/realtime` | A Realtime transcription session |

## chat

| Method | Path | What it does |
|---|---|---|
| POST | `/v1/chat/completions` | A chat completion, streaming or not |

## config

| Method | Path | What it does |
|---|---|---|
| GET | `/v1/config` | The running configuration |
| PATCH | `/v1/config` | Change the running configuration |
| GET | `/v1/config/fields` | The configuration's fields |
| GET | `/v1/config/origins` | Where each configuration value comes from |

## connector

| Method | Path | What it does |
|---|---|---|
| GET | `/v1/connector/skill` | The ignis skill, as a tar.gz |
| POST | `/v1/mcp` | The connector's MCP server |

## decide

| Method | Path | What it does |
|---|---|---|
| POST | `/v1/decide` | A typed decision, read rather than generated |

## images

| Method | Path | What it does |
|---|---|---|
| POST | `/v1/images/edits` | Images from a prompt and condition images |
| POST | `/v1/images/generations` | Images from a prompt |

## models

| Method | Path | What it does |
|---|---|---|
| POST | `/v1/ignis/models/load` | Load the current model without generating tokens |
| POST | `/v1/ignis/models/unload` | Release the idle current model from memory |
| GET | `/v1/models` | The servable models, the loaded one first |
| GET | `/v1/models/catalog` | The models this server can switch to or fetch |
| POST | `/v1/models/download` | Download a catalog entry in the background |
| POST | `/v1/models/download/cancel` | Stop the running download, keeping its partial files |
| POST | `/v1/models/remove` | Remove a catalog entry's files from the models folder |
| POST | `/v1/models/switch` | Replace the loaded model |
| GET | `/v1/models/{model}` | Retrieve one servable model |

## responses

| Method | Path | What it does |
|---|---|---|
| GET | `/v1/responses` | The Responses API's WebSocket mode |
| POST | `/v1/responses` | A response, the Responses API |

## tokenize

| Method | Path | What it does |
|---|---|---|
| POST | `/v1/detokenize` | Token ids back to text |
| POST | `/v1/tokenize` | A prompt's token count, without serving it |
