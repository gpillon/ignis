# Testing

How changes to this repo are verified. One rule governs everything
below: every code change ships with a test, and a task is not complete
until the test suite is green.

## The gate

- `cargo test` from the workspace root — all five crates
  (`artifact`, `core`, `server`, `bench`, `vendor`).
- Machine-local smoke tests skip gracefully when their fixture is
  absent (e.g. `crates/artifact/tests/real_artifact.rs` needs the
  artifact in `F:\ai\q38`). A skip counts as green; a failure does not.
- **Exception — compute work: a skip is never green.** For anything on the
  forward pass (the kernel leaf, the step ABI, the program), a test that
  self-skips on a busy GPU or on a kernel error proves nothing; that pattern
  is what let a broken forward stay green for two tickets
  (`.scratch/REVIEW-2026-09-05.md` §4.1). Those tests belong to the explicit
  **GPU profile** (below), which requires the GPU free and *fails* on any
  kernel error, busy GPU, or missing fixture. The default `cargo test` stays
  CPU-only (GitHub #38).
- Kernel work (`kernel/`, CUDA) is not exercised by `cargo test`: build and
  run the leaf's own op-test executable (CTest) with `kernel/build.ps1 -Test`,
  and verify end to end through the GPU profile. The one kernel-adjacent thing
  `cargo test` *does* check is the vendored subtree's integrity
  (`crates/vendor`): a vendored file edited without a recorded patch turns the
  workspace red (ADR 0010, `kernel/vendor/VENDOR.md`).
- A `--features cuda` build rebuilds the leaf for you.
  `crates/artifact/build.rs` watches every file under `kernel/src`,
  `kernel/include`, `kernel/tests`, `kernel/vendor/src` and
  `kernel/vendor/include` (plus `CMakeLists.txt` and `build.ps1`) and runs
  `kernel/build.ps1` whenever one changes; a build with nothing to do costs
  a few seconds. Do not trust a binary built any other way to carry your
  kernel change: until GitHub #94 that script rebuilt only when
  `kernel/build/*.lib` was *missing*, so an edited `.cu` was linked from
  the previous archive with every test around it still green. That is what
  invalidated the first G2 gate run (#93).

## The GPU test profile (ADR 0006, GitHub #38)

ignis and ninfer (the reference engine, also the coding agent's own model
runner) cannot share the RTX 5090 — a single engine's footprint already
takes ~28 GB of the 32 GB card (ADR 0006). So every GPU-touching Rust test
is `#[ignore]`d and runs only when asked for explicitly: `cargo test` never
touches the GPU on its own.

A GPU test decides what a busy GPU, a kernel error, or a missing fixture
means by asking [`ignis_core::gpu_profile`] — never by returning early on
its own:

- **Outside the profile** (`IGNIS_GPU_PROFILE` unset): those conditions
  print `SKIP: ...` and the test returns — a quick local check without
  stopping ninfer.
- **Under the profile** (`IGNIS_GPU_PROFILE=1`): the *same* conditions are
  hard failures (`panic!`), never a skip. This is the only mode whose green
  result counts for a gate.

Use it as `gpu_profile::check_rc(rc, "ignis_<op>")` for a kernel return
code, `check_compute_err(&e, "...")` for a `ComputeError`, and
`skip_or_fail("...")` for a missing fixture or an absent GPU. Its own unit
tests (in `crates/core/src/gpu_profile.rs`) pin the skip/fail decision on
CPU with an injected rc — no GPU needed, so they run in the default suite.

GPU-gated Rust tests use the `cuda` feature and `#[ignore]`: P1-21's
`gqa_layer_gpu` checks the BF16 exception and NVFP4 GQA layers against the
f64 oracle. They must use the helper above rather than reinventing a
self-skip. The GPU-side coverage also includes the kernel leaf's own op-test
executable (`kernel/build.ps1 -Test`), which the same runbook applies to.

### Running the profile: `scripts/gpu-profile.ps1`

ADR 0006 calls the guard "a preflight check in the `bench`/test harness",
not a script a developer must remember to run standalone. So the profile
takes **two** things, not one: `IGNIS_GPU_PROFILE=1` *and* a preflight pass
on record. `scripts/gpu-preflight.ps1` writes a marker file
(`$env:TEMP\ignis-gpu-preflight.ok`) when it passes and clears it when it
refuses; `active()` reads that marker and **panics** if the variable is set
without a recent one. Setting the variable by hand therefore fails loudly
instead of running un-preflighted while ninfer may hold the card.

`crates/core` has no FFI and no GPU access of its own (GitHub #39 removed
the flat C-ABI surface), so the GPU is inspected by the script and the
verdict is carried across to Rust by that marker.

`scripts/gpu-profile.ps1` is the entry point that ties it together: it runs
the preflight, and only on a pass sets `IGNIS_GPU_PROFILE=1` and runs the
GPU-gated work. It consumes the marker — both it and
the env var are cleared before the script exits, pass or fail — so one
preflight authorizes exactly one run. A pass also ages out after 30 minutes,
which only matters if a run was killed before it could clean up.
**This script is the normal, documented way to run the GPU profile.**

It runs three stages and reports each one's wall time at the end, so a slow
run says where the time went without anyone adding up per-test output
(GitHub #135):

| Stage | What it runs | Parallelism |
|-------|--------------|-------------|
| `kernel/build.ps1 -Test` | the leaf's own op tests (CTest) | CTest's own |
| `cpu f64 layer oracle` | `layer_reference_real`, `--ignored` | libtest default |
| `gpu tests (serialized)` | everything else `--ignored` | `--test-threads=1` |

The middle stage exists because the f64 layer oracle touches no GPU: it
reads the stored weights through a memory map and evaluates them on the CPU.
It was paying for ADR 0006's exclusivity that it never needed, so it now
runs at full parallelism and the serialized stage skips it by name. It still
runs under `IGNIS_GPU_PROFILE=1`, so a missing artifact is a hard failure
there too.

**The test profile is optimized** (`[profile.dev]` and `[profile.test]`,
`opt-level = 2`). The f64 oracles are CPU-bound on scalar `f64` with
per-element NVFP4/W8 dequantization, and at the default `opt-level = 0` they
dominated the serialized stage. Rust applies no fast-math, so every `f64`
result and every measured tolerance is bit-identical to an unoptimized
build; `debug-assertions` and the overflow checks that follow them stay on.
The oracle also spreads its output rows over threads, which for the same
reason cannot move a bit: a row's sum is computed by the same code in the
same order whatever the split, pinned by a unit test that compares schedules
bit-for-bit and by `layer_reference_real` still matching the committed
fixtures, which were recorded by the scalar path.

**"Free" means one process on the card, not just "ninfer is stopped."** The
preflight inspects the GPU once, before the run starts; nothing stops a
second GPU-touching process from entering while the sweep is under way. One
clean profile run peaks at **26.1 GiB of the 5090's 32.6 GiB** (measured
2026-09-13, sampled every 15s across two consecutive runs), so the headroom
left for anything else is about 6 GiB — less than one more artifact load. A
test binary that loses that race dies the way GitHub #145 recorded it: an
abrupt process exit, `exit code: 1`, no panic text and nothing printed
despite `--nocapture`, and the same test passing immediately when re-run
alone. Two unrelated tests died that way in one session, which is what a
contended card looks like from inside a test log — there is no diagnostic
that says so.

So before launching **any** GPU work — the profile, a bench run, a gate
leg, a single `--ignored` test — check that nobody has already launched
some. That includes another agent session in another worktree: the
worktrees are separate, the card is not. `nvidia-smi` (or
`scripts/gpu-preflight.ps1`) plus `tasklist` for stray `*_gpu-*.exe`,
`ignis-server`, `ignis-bench` and `ninfer*` processes answers it in one
step.

Runbook:

```powershell
# 0. Check nobody else is already running GPU work -- another agent session
#    in another worktree counts. nvidia-smi for held memory, tasklist for
#    stray *_gpu-*.exe / ignis-server / ignis-bench / ninfer processes.
# 1. Stop ninfer (frees the VRAM the GPU profile needs).
# 2. Preflight + profile in one step: refuses to proceed while the GPU is
#    held (and names the offending process); on a free GPU, runs the leaf's
#    op tests and the #[ignore]d Rust GPU tests under IGNIS_GPU_PROFILE=1,
#    then clears it.
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu-profile.ps1
# -ThresholdMiB <n>    forwarded to gpu-preflight.ps1
# -SkipKernelBuild     Rust tests only, skip kernel/build.ps1 -Test
# -SkipCargoTests      kernel leaf only, skip both cargo stages
# 3. Restart ninfer.
```

`scripts/gpu-preflight.ps1` still exists as the standalone check
(`scripts/gpu-profile.ps1` calls it) for a quick "is the GPU free" query
that doesn't run anything.

## The canary oracle fixture (P1-05, GitHub #41)

`crates/bench/tests/fixtures/oracle_canary.json` is the committed fixture
G1 is measured against: the reference engine's (`ninfer-serve`, greedy,
the `qwen3_8_27b_nvfp4full-v2.ninfer` artifact) completions on the canary
suite (`crates/bench/src/canary.rs::CANARIES`), 32 greedy tokens per
prompt. `ignis-bench oracle compare` diffs a candidate engine's tokens
against it (§"Oracle (two levels)", spec `01-device-resident-forward`).

Re-record it (needs the GPU and the reference stack, ADR 0006 — stop
`ignis-server` first):

```powershell
# 1. Start the reference engine on the same artifact, greedy, thinking off
#    (thinking on burns the whole token budget on the reasoning channel,
#    leaving no content tokens to compare). This is specific to the oracle,
#    which diffs answer tokens; `canary`, `ttft`, `g3` and `g4` read both
#    channels and need no such flag (GitHub #137).
F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe `
  F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --model-id qwen3.8-27b-nvfp4full-v2 --host 127.0.0.1 --port 8080 `
  --greedy --no-thinking --max-context 8192 --max-concurrency 1 --kv-capacity auto

# 2. Record the fixture.
cargo run -p ignis-bench -- oracle record `
  --endpoint http://127.0.0.1:8080 `
  --artifact F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --out crates/bench/tests/fixtures/oracle_canary.json --max-tokens 32

# 3. Self-check: compare the fixture against a fresh live recording (must be
#    100% — greedy + fixed seed on the same artifact is deterministic).
cargo run -p ignis-bench -- oracle compare `
  --fixture crates/bench/tests/fixtures/oracle_canary.json `
  --endpoint http://127.0.0.1:8080 `
  --artifact F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer

# 4. Stop ninfer-serve (frees the GPU) and commit the updated fixture.
```

## The G2 measurement instrument (P2-05, GitHub #87)

`ignis-bench ttft` measures time to first token at an **exact** prompt
length against any OpenAI-compatible endpoint, and `ignis-bench g2` turns
two such records into the G2 verdict (ADR 0015). Both engines are measured
by this one instrument; only the runs themselves need the GPU.

Every sample — the warmup included — gets its own deterministically
generated prompt, distinct from the first content token, and reads the
engine's own computed-prefill-token count back to prove the prefix was
cold. A sample whose computed prefill is not the prompt's own length is
**void** and fails its cell, and `g2` refuses a verdict over a record with
a void sample, a missing cell, or a session that does not match its
counterpart's.

The gate run (GitHub #88) needs the GPU exclusively and the reference
stopped and restarted around it (ADR 0006). One session, one shared
`--session` value:

```powershell
$Session = "g2-$(Get-Date -Format yyyyMMddTHHmmssZ)"
$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"

# 1. The reference, in the owner's production profile (hq-e8-2b KV, 1024
#    prefill chunk, CUDA graphs) -- the bar actually experienced, measured
#    as it is actually run (ADR 0015).
cargo run -p ignis-bench -- ttft `
  --endpoint http://127.0.0.1:8080 --artifact $Artifact `
  --cells 8192,32768 --label reference --profile "hq-e8-2b KV, 1024 chunk, graphs" `
  --session $Session --out .scratch/g2-reference.json

# 2. Stop the reference (it owns the GPU), start ignis-server on the same
#    artifact, then measure it on the same cells in the same session.
cargo run -p ignis-bench -- ttft `
  --endpoint http://127.0.0.1:8000 --artifact $Artifact `
  --cells 8192,32768 --label ignis --profile "BF16 KV, 1024 chunk, eager" `
  --session $Session --out .scratch/g2-ignis.json

# 3. The verdict. `--note` records what is *not* equal between the two
#    engines, rather than correcting for it.
cargo run -p ignis-bench -- g2 `
  --ours .scratch/g2-ignis.json --ref .scratch/g2-reference.json `
  --note "KV format differs: ignis BF16, reference hq-e8-2b" `
  --out .scratch/g2-verdict.json

# 4. Restart the reference.
```

`ignis-server` takes the engine shape the run needs as flags (each over its
own env var, each validated before any loader work starts): `--prefill-chunk`
(default 1024, a nonzero multiple of 128), `--max-context` (default
40960 — a 32K prompt plus an 8K generation budget), `--kv-format` (`bf16`
or `hq-e8-2b`, default `hq-e8-2b` since GitHub #123 wired its attention
routes — ADR 0022's serving default) and
`--kv-pool-bytes` (default auto: 4 GiB, raised if one `--max-context`
sequence would not fit). The paged-KV pool is sized in **bytes**; the
resident-token capacity that budget buys is derived from the format and
logged at load as `ignis.runtime.kv_pool`, so a run's KV profile is read off
that event rather than computed from a flag.

Speculation is a load option too (GitHub #150): `--spec dflash2
--draft-tokens N` (N in 1..7, both flags or neither) binds the drafter's 66
`dflash2/*` objects and reserves its window pool (8 lanes x 80 MiB);
`ignis.model.loaded` records it as `speculation`. Absent, the load, its plan
and its VRAM report are exactly what they were before the option existed.
`crates/core/tests/speculative_load_gpu.rs` pins the VRAM delta.

**BF16 is the oracle format (ADR 0022).** Every correctness check in the GPU
profile asks for it by name — `--kv-format bf16` at the server, and
`ignis_core::KvFormat::Bf16` on both `load_qwen38_27b` and `SeqPoolBudget`
in a test that drives the leaf directly. A test that inherits the default
runs hq, which is right for the serving-shape checks (the HTTP surface, the
TTFT instrument, the `CudaLeaf` smoke test) and wrong for anything carrying
a derived tolerance.

## The KV-format A/B: 2x2 over engine and format (GitHub #139)

What it answers: how much the hq-e8-2b KV format costs against BF16, whether
that cost is ours or the format's, and what the card is actually doing while it
is paid. The result of the 2026-09-13 run is
[`docs/findings/2026-09-13-hq-vs-bf16-decode-cost.md`](../findings/2026-09-13-hq-vs-bf16-decode-cost.md);
this is how to redo it.

**The one rule that makes it a comparison.** Give every leg the *same KV token
capacity*. A BF16 token costs 65,536 bytes against hq-e8-2b's 9,216 (7.11x, see
[hq-e8-2b KV capacity](../findings/2026-09-11-hq-e8-2b-kv-capacity.md)), so the
same byte budget buys 7.11x more pages under hq and a "just change the format"
run confounds the format with its page count. 65,536 tokens is the value to
pick: it is 1,024 pages, the geometry spec 03's ITL table and #114's cap
arithmetic assume, and the ITL fixture's peak of 65,344 tokens fits it with 192
to spare. Under hq that means asking for the smaller pool explicitly
(`--kv-pool-bytes 576M`), because its default 4 GiB buys 465,984 tokens.

For the record: shrinking the hq pool from 7,281 pages to 1,024 moved ITL p50 by
0.04% and p95 by 0.05%, so the page count is not itself a performance term —
but that is a measured result, not an assumption to skip the matching on.

Needs the GPU exclusively (ADR 0006) and `ninfer-serve` stopped for the ignis
legs. Four runs, ~4 minutes each.

```powershell
$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"
$Corpus   = "F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids"
cargo build --release -p ignis-server -p ignis-bench --features cuda

# Per leg: start the telemetry sampler, start the engine, run the cell, stop both.
# Stop the sampler as soon as ignis-bench exits -- see the pitfall below.
nvidia-smi --format=csv,noheader -l 1 `
  --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu `
  > .scratch/gpu-<leg>.csv

# ignis, hq-e8-2b / BF16 (--kv-format bf16, and drop --kv-pool-bytes: BF16's
# 4 GiB default already lands on 65,536 tokens)
.\target\x86_64-pc-windows-msvc\release\ignis-server.exe --artifact $Artifact `
  --bind 127.0.0.1:8000 --kv-format hq-e8-2b --max-context 40960 `
  --kv-pool-bytes 576M --prefill-chunk 1024 --request-timeout 900

# the reference, same two formats
F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe $Artifact `
  --host 127.0.0.1 --port 8080 --kv-dtype hq-e8-2b --max-context 40960 `
  --kv-capacity 65536 --max-concurrency 8 --prefill-chunk 1024

# the cell, once per leg (port 8000 for ignis, 8080 for the reference)
.\target\x86_64-pc-windows-msvc\release\ignis-bench.exe g3 `
  --endpoint http://127.0.0.1:8000 --artifact $Artifact --corpus $Corpus `
  --label ignis --profile "hq-e8-2b KV, 1024 chunk, max-context 40960, 65536-token pool" `
  --session "kvab-<leg>-$(Get-Date -Format yyyyMMddTHHmmssZ)" `
  --out .scratch/g3-<leg>.json
```

Each engine logs the capacity it resolved — `ignis.runtime.kv_pool`'s
`page_count`, and ninfer's `KV capacity explicit resolved=... tokens pages=...`.
Check both say 1,024 pages / 65,536 tokens before believing the leg.

Then reduce each capture over the ITL cell, which is the last cell a `g3` run
measures and whose length the record itself states
(`window_end_ms - window_start_ms`):

```powershell
python scripts/gpu-telemetry-summary.py .scratch/gpu-<leg>.csv --label "<leg> ITL" --last 90
```

**Pitfall: a sampler left running past the cell dilutes the window.** Idle
samples pull the means down hard — the same hq leg reads 96% util / 306 W over
its ITL cell and 83% / 274 W if eleven seconds of idle tail are included. Stop
the sampler when the bench exits, or pass `--from`/`--to` instead of `--last`.

**Pitfall: a cold machine is not a measurement.** The first `g3` run of a
session came in 14% under the records committed the day before and converged on
them over four runs (C=1 46.6 -> 51.3 -> 51.0 -> 54.2 against 55.6). Discard the
first leg, or run one throwaway before the matrix.

**This is not live/live and cannot decide a gate.** Each leg is its own session,
so `g3-gate` will refuse to build a verdict from these records, by design (ADR
0015, ADR 0021). The A/B answers a question about *where time goes*; a verdict
needs both engines measured back to back under one `--session`.

## The harness's own request deadline (GitHub #138)

Every `ignis-bench` subcommand drives its endpoint through one
`HttpEndpoint`, which sets an **explicit** per-request deadline: 1800 s by
default, and `IGNIS_BENCH_REQUEST_TIMEOUT=<seconds>` overrides it (`0`
removes it entirely). Each run prints the deadline it is using on stderr
before its first request.

This is not a knob a normal run needs — it exists because the *implicit*
one cost a gate session. `reqwest::blocking::Client::new()` carries an
undeclared 30-second total timeout that covers reading the response body,
so a streamed measurement request lives entirely inside it. The G4 needle
cell at 131,072 tokens (~30 s of prefill on this hardware) died on it
against **both** engines, every attempt, reporting `read SSE: error
decoding response body` — which reads like an engine fault and was chased
as one across four launches. The 65,536-token cell, at ~15 s, passed every
time.

Two consequences for anyone measuring here: the deadline is now the
harness's own declared number rather than a library default, and a request
that does hit it says so by name instead of surfacing as a decode error.
Raise it (or set `0`) for a cell whose legitimate wall time could approach
half an hour; never lower it to "make a run finish".

## Where tests live

- Unit tests: `#[cfg(test)]` modules, in the file they test.
- Integration tests: `crates/<crate>/tests/`, one file per behavior,
  named for what it verifies (cf. `real_artifact.rs`).

## Writing the tests

- New behavior: its test lands with the change, before the task is
  reported done — and it exercises the new path, not the old world
  (a test that already passed before the change proves nothing about
  it).
- Bug fix: the regression test goes first — reproduce the bug, watch
  the test go red, then fix.
- The standard loop is red-green, driven by the `/tdd` skill.

## Done means

1. every new or changed behavior has a test that covers it
2. `cargo test` passes workspace-wide (machine-local skips allowed)
3. `cargo check --workspace --features cuda --tests` is clean. The
   cuda-gated test targets (`#![cfg(feature = "cuda")]`) are the ones
   `cargo test` never compiles, so a signature change can leave them
   uncompilable and nothing says so until someone runs the GPU profile
   — which needs a free 5090 and so runs rarely. GitHub #132 did exactly
   that: it added a third argument to `apply_chat_template`, missed two
   call sites under `crates/server/tests/`, and blocked the next GPU
   profile outright (found and fixed in #119). This check costs ~30s,
   needs no GPU, and reports *every* broken site rather than stopping at
   the first — which is how the second of those two was found at all.
4. kernel changes: `kernel/build.ps1` clean + the leaf's op tests green +
   the GPU profile green on a free 5090 (`scripts/gpu-profile.ps1` —
   preflight passed, `IGNIS_GPU_PROFILE=1` for the run — never a skip)
