---
title: "Command line"
description: "ignis-server's commands and subcommands, as its --help prints them."
sidebar:
  order: 6
  badge:
    text: generated
    variant: note
editUrl: false
---

:::note
Generated from the code by `make docs-reference` (`crates/server/src/user_docs.rs`); an edit here is overwritten, and `cargo test` fails while this page and the code disagree.
:::

The commands as `ignis-server --help` prints them. Every field's flag is in [every config field](configuration.md); `ignis-server help --fields --format json` gives the same table to a tool.

```text
ignis-server: the OpenAI-compatible HTTP entrypoint

USAGE:
    ignis-server [--config <path>] [--profile <name>] [FIELD FLAGS]
    ignis-server help [--fields [--format text|json]]
    ignis-server version
    ignis-server config generate [--format json|yaml] [--out <path>] [--force] [--dry-run] [FIELD FLAGS]
    ignis-server config print [--file <path>] [--format json|yaml] [FIELD FLAGS]
    ignis-server config patch [--file <path>] [--out <path>] [FIELD FLAGS]
    ignis-server model download [<id>...] [--all] [--out <dir>] [FIELD FLAGS]
    ignis-server model list [--format text|json] [FIELD FLAGS]

Every field has a flag (--<group>-<field>), an env var (IGNIS_<GROUP>_<FIELD>)
and a config-file key (<group>.<field>). Precedence, highest first: flag, env
var, config file, profile, built-in default; a field's qwen38 / qwen38flashnext
value wins within each. `help --fields` prints every field in full.

--profile defaults to `auto`: the built-in profile of the card's total memory
(16 GB: rtx-16gb, 24 GB: rtx-24gb, 32 GB: rtx5090, 96 GB: rtxpro6000); a card of no class is refused, naming --profile. A named profile
(or `none`) always wins.
```
