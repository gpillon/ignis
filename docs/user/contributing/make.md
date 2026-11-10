---
title: "Working with make"
description: "The make targets a developer and an agent use, and how a run is guarded."
sidebar:
  order: 4
---

The `Makefile` wraps all of the above (GNU make and a POSIX `sh` — on Windows,
Git for Windows' `usr\bin` on `PATH`). `make` alone lists every target.

| Target | What it does |
|---|---|
| `make doctor` | Toolchain, rust target, artifact, web deps. |
| `make build` / `make release` | Build the server (CUDA=1 builds the kernel leaf too) / a clean-slate release build. |
| `make dev` | Build, then run in the foreground. |
| `make run` | Run the last build; fails, naming the changed files, if it is stale. |
| `make mock` | `make dev` on the CPU mock: no GPU, no kernel leaf, no artifact. |
| `make start` / `stop` / `restart` / `status` / `logs` | The background server; it outlives the terminal. Log and pid under `.scratch/serve/`. |
| `make redeploy` | Build, and only if it succeeds: stop + start. On Windows the build fails while the server runs, since Windows locks a running exe: `stop`, `build`, `start` instead, each once the one before it succeeded. |
| `make dev-ui` / `run-ui` | Server plus Playground hot reload as one session; Ctrl+C stops both. |
| `make web-dev` / `web-mock` | Vite alone, against a running engine or an in-process fake one. |
| `make smoke` / `canary` | One `/v1/models` + a short completion / the canary suite. |
| `make metrics` | Scrape the metrics listener (on by default; a server started with `METRICS=0` has none). |
| `make test` / `test-all` / `ci` | `cargo test` / plus web typecheck and vitest / what a CI job runs. |
| `make gpu-status` / `gpu-guard` / `gpu-profile` | Who holds the card / refuse when it is held / the explicit GPU test profile. |
| `make watch CUDA=0` | Rebuild + restart on every Rust change (cargo-watch). |
| `make clean` / `distclean` | This profile's binary / everything. |

Knobs go on the command line (`make dev CUDA=0 PROFILE=dev`,
`make dev-ui SPEC=off MAX_CONTEXT=40960`) or in an untracked `local.mk`
(`local.mk.example`). `make config` prints what is in force.

`make MODEL=image` serves the [image model](../api/images.md) from
`F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer` (`ARTIFACT`
names another) as `ignis-image-2.1-turbo`. It has no context, lanes, prefill
chunk, speculation, KV format or vision, so `make` passes none of those flags;
the settings every model shares (`BIND`, `API_KEY`, `IDLE_UNLOAD`, ...) apply as
before. `make config MODEL=image` prints the server command.

With `CUDA=1` the server starts with the 27B's context left to the profile
(`MAX_CONTEXT` empty: 917,504 tokens under `rtx5090`, adapted down to what the
card has free, [Adaptation](../configuration/index.md)) and the server's `auto` rope scaling
(`yarn:3.5` there, the table that stretches the checkpoint's trained 262,144
positions to the context; `MAX_CONTEXT=524288` names a context, `yarn:2`, that
never adapts; the gate legs ran at `MAX_CONTEXT=262144 ROPE_SCALING=none`) --
`KV_FORMAT=hq-e8-2b`,
`SPEC=dflash2` with
`DRAFT_TOKENS=7`, `PREFILL_CHUNK` empty (the server's: the `rtx5090` profile's 2048), `PREFILL_CHUNK_SOLO` empty (the server's `auto`, [the solo width](../configuration/context-and-speculation.md#the-solo-width)), `KV_HOST_POOL_BYTES=8G` (the 27B's; Flash-Next's own arena
is `FN_KV_HOST_POOL_BYTES=2G`, passed on every start so a switch to it gets it
too). `REQUEST_TIMEOUT` and `STREAM_TIMEOUT` are empty: the server's 2400 s
deadlines ([Deadlines](../api/deadlines.md)). `HOST_MARGIN_BYTES` sets Flash-Next's host
margin (empty: the server's 4G).
`KV_DISK_BYTES` and `KV_DISK_PATH` set [KV-disk](../configuration/kv-disk.md) (`make config`
prints its budget and directory). `SPEC=off` turns
speculation off (`SPEC=` leaves it to the server, and the `rtx5090` profile
speculates with dflash2). `VISION=1` loads the vision tower, `VISION=0`
leaves it out over a profile that loads it (`--vision-enabled false`), and an empty
`VISION` leaves it to the profile: every built-in profile loads it for the 27B
and leaves it out on Flash-Next, where `VISION=1` loads its vision companion's
tower from 0.11.0 (0.10.0 refuses it there) ([Vision](../configuration/vision.md)). `VISION_MAX_TOKENS` is passed whenever
it is set, and a start with vision off for both models refuses it;
from 0.11.0, with an empty `VISION` under a built-in profile it applies to the 27B alone, and the start's
`ignis.config.vision_options` line says so; `UNCENSORED=1` loads
[the uncensored variant](../guides/uncensored.md) in place of the default
image, served as `qwen3.8-27b-abliterated`; `ROPE_SCALING=yarn:4` pins the
factor (empty is `auto`, from `MAX_CONTEXT`; `none` keeps the trained table); the metrics listener is on (`METRICS=0` starts without it); `API_KEY=` and `EXPOSE=` set
the access flags; `IDLE_UNLOAD=<secs>` sets [idle unload](../api/idle-unload.md)
(`IDLE_UNLOAD=0` for a measurement); `LOOP_GUARD=off` turns the reasoning
loop guard off (the server's default rewinds a loop; a measurement passes
`off`, as the A/B and TTFT session scripts do); `ARGS='…'` passes anything
verbatim.

A CUDA `run` or `start` first runs a GPU guard (the preflight plus a check for
other Ignis GPU work), skippable with `GPU_CHECK=0`.

`start`, like `run`, launches the last build and refuses a stale one, naming
the files changed since: `make build` is the fix. It rewrites
`ignis-server.log` and its `.err` in `.scratch/serve/`, so before restarting a
server whose log holds an incident, copy both out. A script runs `make start`
with its output sent to a file (`make start > file 2>&1` from bash): the
server inherits a pipe (`| Tee-Object`, `| tail`) and holds it open for as long
as it runs.

A change to the `Makefile` starts from
[ADR 0027](../../adr/0027-make-as-the-dev-entrypoint.md), the decision that
made `make` the entry point.
