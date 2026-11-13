---
title: "Reference"
description: "The lists the code writes about itself — config fields, profiles, models, metrics, routes, commands, JSON Schema keywords — and the glossary."
sidebar:
  order: 0
  label: "Overview"
---

The pages marked *generated* are written by the code, from the same tables the
server runs on, and committed beside the rest of the guide. They cannot name
a field, a route or a metric the binary does not have: `cargo test` fails
while a page and the code disagree, and `make docs-reference` rewrites them.
The pages around them explain; these list.

| Page | What it lists | Read from |
|---|---|---|
| [Every config field](configuration.md) | Every field by group: flag, env var, default, rule, the models it applies to, whether it changes live | the field table, as `ignis-server help --fields` prints it |
| [Built-in profiles](profiles.md) | Every value each hardware profile sets | the built-in profiles |
| [Built-in models](models.md) | Every model the server can fetch, its pinned revision, every file's size and SHA-256 | the built-in catalog |
| [Metrics](metrics.md) | Every Prometheus metric, its type and meaning | the metrics listener's own exposition |
| [HTTP routes](http-routes.md) | Every route of the HTTP surface | the OpenAPI document the handlers build |
| [Command line](cli.md) | The commands and subcommands | `ignis-server --help` |
| [JSON Schema keywords](json-schema.md) | What a `json_schema` in `response_format` may carry: the keywords held, the formats, the annotations, the keywords inert without `strict`, OpenAI's four limits | the grammar's own lists, each inert keyword and limit compiled |
| [Glossary](glossary.md) | The terms the guide uses | written by hand |

A running server also describes itself: `ignis-server help --fields --format
json` for the fields, `GET /v1/openapi.json` and the browsable `/v1/docs/` for
the HTTP surface, `GET /v1/config/fields` for the fields its Server view can
change.
