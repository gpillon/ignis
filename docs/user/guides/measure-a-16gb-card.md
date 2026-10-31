---
title: "Measure a 16 GB card"
description: "Run `ignis-bench small-card` against your server and post the record on issue #360: the card, the VRAM plan, what the load adapted, decode speed and time to first token. A coding agent can run the whole test for you."
sidebar:
  order: 5
---

The `rtx50-16gb` profile is derived, not measured: the project has no 16 GB
card ([the profile](../concepts/profiles.md#rtx50-16gb)). If you run the 27B
on an RTX 5060 Ti 16 GB, a 5070 Ti or a 5080, one command against your running
server writes a record of what it does there, and that record, posted on
[issue #360](https://github.com/gpillon/ignis/issues/360), is how the profile
gets real numbers.

A coding agent can run the whole test for you, the install included: it
tries the default, tries a few settings if the default is refused, and writes
the report for you to read before anything is posted
([Install with a coding agent](../start/install-with-an-agent.md#test-a-16-gb-card)).

## Run it

Start the server as you normally do, let it load the model, then:

```bash
ignis-bench small-card --endpoint http://127.0.0.1:8000 --out small-card.json
```

`ignis-bench` ships beside `ignis-server` in the release archives
([Install](../start/install.md#release-archives)) and in the container image.
With the container, run it inside, where both of the server's listeners are
(`<container>` is its name or id, `podman ps`), and keep the record from its
output:

```bash
podman exec <container> ignis-bench small-card --endpoint http://127.0.0.1:8000 > small-card.json
```

A container the runtime gave no `nvidia-smi` records the card as `null`, with
a note; the rest is measured as usual.

The record is printed on stdout and, with `--out`, written to that file too;
the progress goes to stderr. On the project's 5090 inside a 16 GB budget it
took two minutes, most of it the 64K time-to-first-token cell, which a 16 GB
card's slower prefill makes longer; `--cells 8192` leaves that cell out.

Measure the server with nothing else using it: the decode figures are one
request at a time, and another client's requests would land in them.

## What it records

- **The card**: its name and memory, and the PCIe link it served at beside the
  most the card and the slot allow, from `nvidia-smi` (read right after the
  decode requests, since an idle link slows down).
- **The load**: the server's version, the model and the profile, the VRAM plan
  line by line with its budget and the KV pool's pages (from the metrics
  listener), the configuration values that shape the load and where each came
  from (default, profile, file, env or flag).
- **What the load adapted**: the context and the decode lanes it runs at beside
  the ones configured, and every step it took to fit the card
  (`adaptation.adapted`, empty when it took none) — the 27B under `rtx50-16gb`
  in a 13 GiB budget (16 GiB less 3 GiB of desktop, set by hand on an RTX 5090
  to imitate a 16 GB card) took
  `max_context 262144 -> 65536 (adapt_context_step 65536, adapt_spare_tokens
  65536)` ([when a profile does not fit](../concepts/profiles.md#when-a-profile-does-not-fit)).
- **Decode speed**: tokens per second on the coding prompts and on three
  synthetic agent turns (a system prompt, tool calls and their results, a
  coding request), 256 tokens each, and how many draft tokens DFlash2 got
  accepted meanwhile. The coding prompts run with thinking off; the agent turns
  think, because a tool call reaches the client whole and only the reasoning
  before it streams token by token. An agent turn that is still reasoning at
  256 tokens makes the server log `generation produced reasoning but no
  content`, which is expected here.
- **Time to first token** at 8,192 and 65,536 prompt tokens, three cold samples
  each after a warm-up. A cell longer than the context the load runs is not
  measured: the record says so and names the largest cell that fits, which
  `--cells` asks for. A sample that is not provably cold (the server served
  part of it from its cache, counted its prompt differently, or failed it) is
  void: the cell's `void_samples` counts it, `all_cold` is false, and
  `median_ttft_ms` is the median of the cold samples alone.
- **Notes**: every piece it could not read, and why.

`--metrics` points at the metrics listener when it is not where the server's
`server.metrics_bind` says (the endpoint's host on that port); `--metrics off`
skips it, and the plan lines are then empty. `--samples` and
`--decode-tokens` change the sample counts. A piece it cannot read is `null`,
never a failed run.

## What it never records

No API key, no path, no address but the endpoint's, written without any
`user:password@`. The configuration is a fixed list of fields, never the whole
document. The preset sends no key either: a server started with
`--server-api-key` answers it `401`, and the record then says nothing was
measured. Run it against a server without a key.

## Send it

Comment on [issue #360](https://github.com/gpillon/ignis/issues/360), the one
that collects the 16 GB reports, starting with your card's name, and paste the
whole file in a fenced block:

````markdown
```json
{
  "schema": "ignis-small-card/1",
  ...
}
```
````

Say anything the record cannot know: what else holds the card (a desktop, a
second application), the operating system, and any flag you started the server
with that you would like looked at.
