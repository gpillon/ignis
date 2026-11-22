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
| `make test` / `test-all` / `ci` | `cargo test` / plus web typecheck, vitest and the connector skill's script tests / what a CI job runs. |
| `make test-connector` | From 0.11.0: the connector skill's script tests, `node --test` in `connector/`; Node alone, no `npm`. |
| `make test-machine-local` | From 0.11.0: the test targets `mk/machine-local-tests.txt` lists, which read a fixture only the owner's machine holds and skip on CI, run for real; once per release, before the tag ([Cutting a release](releases.md)). CPU only, no GPU lock; it ends with the skip lines the tests printed. |
| `make grammar-venv` | From 0.11.0: the Python venv at `.scratch/grammar-venv` with the independent JSON Schema validator the structured-output tests check every served answer with. Without it those checks are skipped, saying so; CI installs it, and there a missing one fails them. |
| `make gpu-status` / `gpu-guard` / `gpu-profile` | Who holds the card, the GPU lock's holder among it / refuse when it is held / the explicit GPU test profile, whole or (from 0.11.0) only the tests it is given. |
| `make gpu-run CMD="…"` | From 0.11.0: one sh line under the GPU lock, for GPU work no other target runs (an example run in process, a server started by hand for a bench): waits for the lock up to `GPU_LOCK_WAIT` seconds, runs the guard, then the command as written (`$HOME` reaches sh, not make). |
| `make watch CUDA=0` | Rebuild + restart on every Rust change (cargo-watch); from 0.11.0 `CUDA=1` runs it under the GPU lock. |
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
`DRAFT_TOKENS` empty (the server's: the card's profile gives the window, 15 on the RTX 5090, 7 elsewhere), `PREFILL_CHUNK` empty (the server's: the `rtx5090` profile's 2048), `PREFILL_CHUNK_SOLO` empty (the server's `auto`, [the solo width](../configuration/context-and-speculation.md#the-solo-width)), `KV_HOST_POOL_BYTES=8G` (the 27B's; Flash-Next's own arena
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

From 0.11.0 the card also has a GPU lock: one per card, a record that names its
holder (pid, worktree, command, since when), in `%ProgramData%\ignis\gpu-lock\`
on Windows and `/tmp/ignis-gpu-lock/` on Linux (`IGNIS_GPU_LOCK_DIR` moves it).
A CUDA `start`, `run` or `run-ui` takes it before the server starts and hands it
to the server, which holds it until it exits; `stop` frees it, and so does a
start that fails. `make watch CUDA=1` holds it for cargo-watch, which restarts
the server, until it exits. A lock someone else holds refuses the start at
once, naming the holder, and `GPU_CHECK=0` does not skip it; the guard refuses
for it too. On Windows the guard of a start runs once the start holds the lock.
`make kernel-test`, `make gpu-run` and the GPU profile wait for the lock
instead, up to `GPU_LOCK_WAIT` seconds (900), then fail naming the holder.
A holder that has exited, or whose pid now names another process, holds
nothing. `make gpu-status` names the holder, and `make gpu-lock-selftest`
tests the lock and the scripts around it on stand-ins, with no card.

`start`, like `run`, launches the last build and refuses a stale one, naming
the files changed since: `make build` is the fix. It rewrites
`ignis-server.log` and its `.err` in `.scratch/serve/`, so before restarting a
server whose log holds an incident, copy both out. A script runs `make start`
with its output sent to a file (`make start > file 2>&1` from bash): the
server inherits a pipe (`| Tee-Object`, `| tail`) and holds it open for as long
as it runs.

On Linux `start`, `stop` and `status` run `mk/linux/server.py` (Python 3,
`PYTHON`). `start` refuses while the server it recorded is alive or something
already listens on `BIND`, waits for `/ready` rather than `/v1/models`, and
stops the server again if it is not ready within `READY_TIMEOUT`, where Windows
leaves it running. It keeps the previous log and `.err` beside the new ones,
with a timestamp suffix, instead of rewriting them, and does not repeat a
generated API key or public URL on the console: they are in the log. `stop`
sends SIGTERM and waits up to 30 seconds; `FORCE=1` also stops your other
`ignis-server` processes. `dev-ui` / `run-ui` run `mk/linux/devui.py`, which
refuses, waits and keeps the logs as `start` does, but repeats a generated API
key and public URL on the console, as Windows does. However the session ends
(Ctrl+C, Vite exiting, the terminal closed, make killed, even with `kill -9`),
the server gets SIGTERM, its graceful shutdown, and is killed if it still runs
20 seconds later; a second Ctrl+C ends it at once. `gpu-profile` runs
`scripts/gpu-profile.sh` on Linux and `scripts/gpu-profile.ps1` on Windows:
the same three stages, the same marker, the same exit codes. Its switches go
in `GPU_PROFILE_ARGS`: `-SkipKernelBuild`, `-SkipCargoTests` and
`-ThresholdMiB 4096` on Windows, `--skip-kernel-build`, `--skip-cargo-tests`
and `--threshold-mib 4096` on Linux. From 0.11.0, `-Test a_test,b_test` on
Windows and `--test a_test --test b_test` on Linux run only those `#[ignore]`d
tests, each alone, under the lock, the preflight and the marker
(`make gpu-profile GPU_PROFILE_ARGS='-Test a_test,b_test'`); a name that is no
ignored test fails the run before any test runs. The Linux script has had no run on a real
card yet, only its selftest on stand-ins (`mk/linux/gpu-profile-selftest.sh`).

A change to the `Makefile` starts from
[ADR 0027](../../adr/0027-make-as-the-dev-entrypoint.md), the decision that
made `make` the entry point.
