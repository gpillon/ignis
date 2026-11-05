# Testing

How changes to this repo are verified. One rule governs everything
below: every code change ships with a test, and a task is not complete
until the test suite is green.

## The gate

- `cargo test` from the workspace root — all five crates
  (`artifact`, `core`, `server`, `bench`, `vendor`). CI runs exactly this on
  Linux and on Windows, and the release builds do not start until both legs
  are green (`.github/workflows/release.yml`, ADR 0032). That is possible
  only because the default suite is CPU-only; everything below about the GPU
  profile stays on the machine with the 5090.
- Machine-local smoke tests skip gracefully when their fixture is
  absent (e.g. `crates/artifact/tests/real_artifact.rs` needs the
  artifact in `F:\ai\q38`). A skip counts as green; a failure does not.
- **Exception — compute work: a skip is never green.** For anything on the
  forward pass (the kernel leaf, the step ABI, the program), a test that
  self-skips on a busy GPU or on a kernel error proves nothing; that pattern
  is what let a broken forward stay green for two tickets
  (`docs/REVIEW-2026-09-05.md` §4.1). Those tests belong to the explicit
  **GPU profile** (below), which requires the GPU free and *fails* on any
  kernel error, busy GPU, or missing fixture. The default `cargo test` stays
  CPU-only (GitHub #38).
- Kernel work (`kernel/`, CUDA) is not exercised by `cargo test`: build and
  run the leaf's own op-test executable (CTest) with `kernel/build.ps1 -Test`,
  and verify end to end through the GPU profile. On a card below compute
  capability 12.0, with a build that covers it (`IGNIS_CUDA_ARCHITECTURES`,
  #456), run CTest as `ctest -LE sm120a-only`: the tests that label leaves out
  exercise the kernels only the sm_120a set holds, and fail there by design.
  `ignis_kernel_gdn_gating_proj_tests` runs cooperative schedules sized for
  the RTX 5090's 170 SMs: on a card that cannot hold one resident (fewer SMs,
  or an sm_86 build's fewer CTAs per SM) it skips each one the planner's own
  residency query refuses there and names it in a `SKIP` line of its output
  (#478, a portability patch under ADR 0077). Its `budget` lines give each
  cooperative kernel's CTAs per SM; on compute capability 12.0 it fails if
  one is not the value the planner was tuned on, and on the 5090 if it skips
  any schedule. The one kernel-adjacent thing
  `cargo test` *does* check is the vendored subtree's integrity
  (`crates/vendor`): a vendored file edited without a recorded patch turns the
  workspace red (ADR 0010, `kernel/vendor/VENDOR.md`).
- Playground work (`web/`, ADR 0026) is not exercised by `cargo test`
  either: its logic lives in pure modules with vitest tests, so a change
  there also needs `npm --prefix web test` and `npm --prefix web run build`
  (the typecheck) green — `make test-web web-build`. `cargo test` covers only the server's `/ui` routes.
- A `--features cuda` build rebuilds the leaf for you.
  `crates/artifact/build.rs` watches every file under `kernel/src`,
  `kernel/include`, `kernel/tests`, `kernel/vendor/src` and
  `kernel/vendor/include` (plus `CMakeLists.txt` and `build.ps1`) and runs
  `kernel/build.ps1` whenever one changes; a build with nothing to do costs
  a few seconds. It passes `-ArchivesOnly` (`--archives-only` to
  `build.sh`), so a cargo build compiles the two archives the link uses and
  none of the CTest executables (#460); a hand run of the script, `-Test`
  and `make kernel-test` build the whole project, tests included. The first
  CTest run after a cargo-only build therefore compiles the test
  executables once. Do not trust a binary built any other way to carry your
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
| `gpu tests (serialized)` | everything else `--ignored`, but `ignis-speech` and the server's speech legs that touch no GPU (`$NotGpuLegs`) | `--test-threads=1` |

The middle stage exists because the f64 layer oracle touches no GPU: it
reads the stored weights through a memory map and evaluates them on the CPU.
It was paying for ADR 0006's exclusivity that it never needed, so it now
runs at full parallelism and the serialized stage skips it by name. It still
runs under `IGNIS_GPU_PROFILE=1`, so a missing artifact is a hard failure
there too. The server's speech legs that touch no GPU are skipped by name as
well (`$NotGpuLegs`: the dictations through the route on the CPU and the
diarizer's legs, which need the study disk, and the speech catalog's
download); each binary tests that its legs are listed.

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
step; `make gpu-status` prints both, and `make gpu-guard` turns them into a
pass/fail.

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

`make gpu-profile` runs the same script; `GPU_PROFILE_ARGS` forwards its
switches (`make gpu-profile GPU_PROFILE_ARGS=-SkipKernelBuild`).

`scripts/gpu-preflight.ps1` still exists as the standalone check
(`scripts/gpu-profile.ps1` calls it) for a quick "is the GPU free" query
that doesn't run anything.

## Testing one area

- Recording the canary oracle fixture or comparing an engine against it (G1),
  or reading the canary suite's thinking-only verdicts:
  [testing/canary-oracle.md](testing/canary-oracle.md).
- Recording or scoring the vision canary fixture (multimodal prompts against
  the reference's answers): [testing/vision-canary.md](testing/vision-canary.md).
- Testing `/v1/decide` (`locate` over a text state, the shortlist, the
  entailment fixture): [testing/decide.md](testing/decide.md).
- Testing the image model's media against Pillow (the resize fixtures, the
  WebP condition image): [testing/image-model.md](testing/image-model.md).
- Testing the speech model (the speech oracle, `server_vad`'s VAD oracle, their
  legs on the study disk): [testing/speech.md](testing/speech.md).
- Running a bench or a gate leg (TTFT and G2, the multimodal TTFT cell, the
  KV-format A/B, the harness's request deadline):
  [testing/bench.md](testing/bench.md).

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
- Logging on the prefill/decode path: the hotpath lint
  (`crates/logging/src/hotpath_lint.rs`) fails `cargo test` on an
  INFO-or-above `tracing` call or a raw `println!`/`eprintln!`/`print!` in
  its hand-maintained list of hot-path files. It makes the hot-path logging
  constraint checkable in review, not only provable later at the G4 gate; a
  `hotpath-lint-allow:` comment marks a reviewed, intentional exception, never
  a silent one.

## GPU tests that compare output or load an option

Every ignis leg of a bench or a gate runs with the loop guard off:
`--model-loop-guard off`, or `LOOP_GUARD=off` on `make start` (`make config`
prints the flag). The server's default, `rewind`, takes a looping reasoning
back and nudges the model, so wherever it acts the output, the token counts
and the timing differ from a reference that has no guard (ADR 0052). The
in-process GPU tests that compare output (`openai_http_gpu.rs`,
`hq_canary_gpu.rs`, the `LiveServer` ones) build their server without a config
and run with it off; one built from flags or a config gets `rewind`, and a new
one that compares output passes `--model-loop-guard off`.

`ignis-server` takes the engine shape the run needs as flags (each over its
own env var, each validated before any loader work starts): `--model-prefill-chunk`
(default 1024, a nonzero multiple of 128), `--model-max-context` (default
40960 — a 32K prompt plus an 8K generation budget), `--model-kv-format` (`bf16`
or `hq-e8-2b`, default `hq-e8-2b` since GitHub #123 wired its attention
routes — ADR 0022's serving default) and
`--vram-kv-pool-bytes` (default: the KV pool policy's, ADR 0045 -- the rest of the
VRAM budget on the 27B, GitHub #210; 524,288 tokens shared by the lanes on
Flash-Next with its experts streaming; the plan, logged as
`ignis.runtime.vram_plan` or `ignis.runtime.flash_next_plan` with its
`kv_pool_policy`, `kv_pool_pages` and `kv_pool_tokens`, refuses a start that
cannot hold one `--model-max-context` sequence and a page per retained slot). The
pool is named in bytes or in tokens (`512Ktok`); the plan turns either into
pages at the model's own bytes per token (9,216 on the 27B under hq-e8-2b,
65,536 under BF16; 4,224 on Flash-Next under hq-e8-2b, KV and indexer keys).
On the 27B the resident-token capacity is logged at load as
`ignis.runtime.kv_pool`, so a run's KV profile is read off the events rather
than computed from a flag. A request that names no cap generates at most
`--model-default-max-tokens` (38,912; `0` = up to the context), and a test that
needs a request to run to the context sends `max_tokens` or that `0`.

Speculation is a load option too (GitHub #150): `--spec-backend dflash2
--spec-draft-tokens N` (N in 1..7, both flags or neither) binds the drafter's 66
`dflash2/*` objects and grows the prefill scratch by the drafter's context
append; its sequence pool carries the drafter window per slot (40 MiB each
slot, GitHub #152); `ignis.model.loaded` records it as
`speculation`. Absent, the load, its plan and its VRAM report are exactly what
they were before the option existed. `crates/core/tests/speculative_load_gpu.rs`
pins the VRAM delta; `crates/core/tests/dflash2_window_gpu.rs` checks a
prefill's window against a host f64 recomputation from the target's taps.
`crates/core/tests/speculative_round_gpu.rs` proves the verify round (GitHub
#153) with a fake drafter under the test-only `verify-only` backend, whose
sequence pool must be built for that backend too.
`crates/core/tests/dflash2_round_gpu.rs` proves the real drafter in the round
(GitHub #155): greedy spec-on against spec-off on the canary prompts at widths
1, 4 and 8, its graph replay against eager, a lane at extent 0 leaving its
window untouched -- and, continued by a round or by a prefill, writing that
round's anchor into its window before anything else (GitHub #157, checked
against a lane that never ran at extent 0) -- and a mid-generation
snapshot/restore. Every stream is cut
at the turn's first `<|im_end|>`/`<|endoftext|>`, where the server stops: past
it the model loops on chat markers and the verify and decode tilings part by
several logits on text no request emits. It prints acceptance
per round and the mean committed tokens per full-window round against the
reference's 3.4–5.75 band — read that line from the profile's output; a mean
below the band is a finding to file, not a test failure. The drafter's rewrite
checkpoint is retired (blob format 5): `dflash2_window_gpu.rs` asserts a
snapshot no longer lists it. `crates/runtime/tests/cuda_leaf_dflash2_gpu.rs` drives the same
drafter end to end through `RuntimeCompute`: the leaf passes no drafts, and
every round's speculative counters match the lane's extent and committed run.
The acceptance replay of recorded agent turns (GitHub #267,
`crates/core/tests/dflash2_replay_seams_gpu.rs`) walks a turn in
teacher-forced rounds. The recorded token after a lane's run is its pending
token, so it must be a vocabulary id; the pad past a recording's end is taken
only after a stop id, and leaves no pending token until a prefill gives one.
Only greedy lanes may be forced, and only on 27B loads.

Vision is a load option too (GitHub #177): `--vision-enabled` (with
`--vision-max-tokens N`, default 32,768 merged tokens) binds the 333 `vision/*`
objects in their stored Q4/Q5/Q6/W8/BF16 formats and reserves, inside
`ignis_model_load` and so before the sequence pool is built, the encoder
workspace for one item of `min(item bound, V)` and the embedding pool floored
at one `[5120, V]` item for `V = min(max_context, N)`. The item bound is the
processor's (`ProcessorOptions::max_item_tokens`, 16,384 for this artifact),
which the server hands to the load as `vision_item_max_tokens`;
`kernel/tests/test_vision_workspace.cpp` pins the encoder's workspace for a
32,768-token item. The encoder workspace is not an arena of its own
(GitHub #212): the prefill scratch is sized for the larger of the two, and
the VRAM plan shows it as one `workspace` line. `ignis.runtime.kv_pool`
reports what vision adds — the output transient plus the scratch's growth —
as `vision_reserved_bytes` beside the pool's `token_capacity`.
`crates/runtime/tests/vision_shared_workspace_gpu.rs` checks, on a DFlash2
load, that an image request, an image prefill interleaved with decoding text
lanes and a three-image prompt whose encodes land between prefill chunks
generate the tokens recorded before the arena was shared
(`IGNIS_RECORD_VISION_WORKSPACE_TOKENS=1` re-records them).
Absent, nothing vision-related is
bound or allocated. `crates/core/tests/vision_load_gpu.rs` pins the VRAM delta
(weights plus the reported reservation) and prints the reservation at the
default envelope; `kernel/tests/test_model_load_vision_options.cpp` pins the
envelope check and the leaf's vision schema host-side.

Since GitHub #195 `--vision-enabled` combines with `--spec-backend dflash2`: both scopes bind
in one load, the drafter's context append takes the span's KV positions (what
the reference's own prefill sink captures on a multimodal span), and the verify
round rotates its columns at `position + rope_delta` from its own staging, the
way a decode round already did. `crates/server/tests/vision_dflash2_gpu.rs` is
that load's check — the canary floor on it, greedy verify rounds against the
same load's own plain decode rounds, and the acceptance on the text after an
image against the 3.4–5.75 band. The verify round's rope staging exists only
when vision is loaded, so a text-only speculative load allocates nothing new.

Since GitHub #193 multimodal requests share prefixes under a media-aware
identity (the match key, ADR 0029). `crates/server/tests/vision_prefix_reuse_gpu.rs`
is its check on the real model: siblings over one system prompt sending
same-size flat swatches — identical token ids — where the other colour shares
only the system block and still names its own colour, and the same colour
reuses past the image, through a retained checkpoint with prompt reuse on and
through the sibling prefix cache alone with it off.

Since GitHub #249 a request whose image spans the opener's page floor lends the
pages below the opener to a pages-only link at its capture on the 27B too
(ADR 0029). `crates/server/tests/vision_checkpoint_link_gpu.rs`
holds the claimants standing on that link, which ends inside the picture, to a
cold prefill split at the same boundaries (the serving chunk set to the first
turn's opener, the next turn ending at its own): the identical repeat against
the first turn, and the next turn from the device and back from KV-RAM, token
for token. An unsplit cold prefill is printed beside them, not asserted. Its
second test is the common next turn, whose floor lies past the picture: it
publishes a prefix chained over the link and captures over that, and both it
and the turn after it, which stands on the chain, are held to a cold prefill.
There the serving chunk is half a page and every opener the conversation is
cut at is a multiple of it, so the cold prefill cut every chunk is cut where
each claimant's whole history was; the test compares the chunks' ends too.

`crates/server/tests/vision_mixed_load_gpu.rs` (GitHub #181) is all of the
above at once, in the serving shape: one load behind the HTTP router with
`--vision-enabled`, `--spec-backend dflash2`, prompt reuse on, a 128-token prefill chunk, and
every image fetched by URL from a loopback server. Three text lanes decode
while an image request prefills between their rounds; same-size swatch
siblings each name their own colour; and with the KV pool sized to exactly one
4,096-token context, an Interactive request reserving all of it evicts the one
Agent image request to KV-RAM, which is restored and still describes its image.
The request log it prints names the evicted and restored request id and each
request's `spec.*` counters.

**The verify round's rope delta is a deliberate departure from the reference.**
The reference's DFlash2 round passes its proposal positions for both the cache
and the rotation and carries no rope delta in its decode ingress at all, while
its ordinary decode batch and its MTP round both add the sequence's — so on the
reference a multimodal sequence's verify columns and its decode rounds
disagree. The spec (`docs/specs/vision/01-image-input.md`, §Compute seam)
asks for `rope_position = position + rope_delta` on *every* decode and verify
round, and ignis follows the spec. Expect a live/live comparison against the
reference with `--vision --spec dflash2` to show the reference's own
divergence, not ours; the vision canary fixture was recorded without
speculation, so it stays the right oracle either way.

**BF16 is the oracle format (ADR 0022).** Every correctness check in the GPU
profile asks for it by name — `--model-kv-format bf16` at the server, and
`ignis_core::KvFormat::Bf16` on both `load_qwen38_27b` and `SeqPoolBudget`
in a test that drives the leaf directly. A test that inherits the default
runs hq, which is right for the serving-shape checks (the HTTP surface, the
TTFT instrument, the `CudaLeaf` smoke test) and wrong for anything carrying
a derived tolerance.

## The sm_86 gate

A release built for `86;120a` runs on RTX 30 cards that no CI runner and no
default suite covers. Before a release whose range touched the kernel leaf or
the server's architecture module, the release's coordinator runs this gate on
an sm_86 card (spec #474, #484). The range's check:

```
git diff --stat "$(git describe --tags --abbrev=0 --match 'v*')".. -- kernel/ crates/server/src/config/arch.rs
```

Output means the gate runs; none means it does not. A failure of any part
stops the release. Check `make gpu-status` first, as for any GPU run.

1. **Quality**, by the method of
   `docs/findings/2026-11-03-first-light-on-sm86.md`: #340's gate 1 windows,
   16 of 4,096 tokens, dumped by the `span_logits` example at chunk 1024, KV
   `hq-e8-2b`, on the route an 8.x card runs unset (spec #503, #515): the dense
   prefill, the integer band and ignis's own gated delta net prefill, named by
   `IGNIS_MODEL_PREFILL_LINEAR=int8 IGNIS_MODEL_PREFILL_ATTENTION=int8
   IGNIS_GDN_PREFILL=ours` (the example reads the variables, not the card's
   default), scored top-64 against the BF16 checkpoint (`kl_top64` in
   `crates/bench/src/flash_next/kld.rs`). The reference is a set, about 90 MB,
   that holds the BF16 model's 64 most probable next tokens per position and
   the windows themselves. It stays out of the repo:
   `tools/kld-gate/ref_top64.py` builds it from the BF16 checkpoint on the
   RTX 5090 (`tools/kld-gate/README.md`), and the coordinator carries it to
   the card. The limits are derived from the RTX 5090 on the same route, the
   same windows, chunk and reference set: the same dump there under the same
   three variables, scored by the same command, gives a top-64 KLD of
   TODO(#515 coordinator: the 5090's top-64 KLD on this route) and a top-1 of
   TODO(#515 coordinator: the 5090's top-1 on this route). Limits, on the `all`
   row of the scorer's table: `KLD top64` at most 1.05 times the 5090's, `top-1`
   at least 0.5 points under the 5090's. A change that alters the 5090's
   numerics on this route records new 5090 figures in a finding first, and the
   limits follow them. (The limits of the FP16 band, 0.04016 and 92.92% from
   the first-light finding's 5090 figures of 0.03825 and 93.42%, gated the
   releases before 0.10.0; `IGNIS_MODEL_PREFILL_LINEAR=fp16
   IGNIS_MODEL_PREFILL_ATTENTION=fp16` still runs that route, one option away
   on the card, and is not the gate.)
2. **The leaf's tests**: built for `86;120a`, then `ctest -LE sm120a-only`.
   Every test passes. In `ignis_kernel_gdn_gating_proj_tests` the `budget`
   lines and the `SKIP` lines that name the cooperative schedules the card
   cannot hold resident (#478) are expected; a failure is not.
3. **Smoke**: a server started under `--profile auto` (on a 24 GB card,
   `rtx-24gb`, which turns DFlash2 and vision on; its `GET /v1/config` reports
   `prefill_linear` and `prefill_attention` as `int8`, the card their source)
   answers a greedy chat completion, a small image and a large image, each correctly: `Paris`, red,
   blue. The small image is the smallest the processor accepts: it upscales
   every image to the 65,536-pixel floor, so 256 patches at the vision tower
   and 64 tokens after the merge; record that count, the request's
   `media.vision_tokens` in the server's log. The tower's W8 shapes have no
   small-token route in the vendored dispatch (SIMT up to 16 tokens, the
   merger's up to 4 or 5, then `mma_r64_c128`), so no image reaches a kernel
   only the sm_120a set holds. The large image sits at the processor's item
   bound, 16,384 tokens (4096 x 4096 pixels, a merged token being 32 x 32),
   under the profile's `vision.max_tokens`.

The commands, in order:

```sh
# On the sm_86 card, from the release's checkout, the set ref27b-top64/ beside it
# (tools/kld-gate/README.md) and 32.5 GB free for the dump. ignis-server is the
# release candidate's; on Linux every cargo command here also needs
# CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu.
export IGNIS_CUDA_ARCHITECTURES='86;120a'
ignis-server model download qwen3.8-27b-16g      # into ./models
IGNIS_MODEL_PREFILL_LINEAR=int8 IGNIS_MODEL_PREFILL_ATTENTION=int8 IGNIS_GDN_PREFILL=ours \n    cargo run --release -p ignis-core --features cuda --example span_logits -- \
    models/qwen3_8_27b_trellis_sc30-v2.ninfer hq-e8-2b ref27b-top64/windows.json dump-sm86 1024
python3 tools/kld-gate/score_top64.py ref27b-top64 dump-sm86 --json gate-sm86.json
```

```sh
bash kernel/build.sh            # kernel\build.ps1 on Windows: the whole project, tests included
ctest --test-dir kernel/build -LE sm120a-only --output-on-failure
```

```sh
ignis-server --profile auto      # its ignis.config.source line names rtx-24gb
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"qwen3.8-27b","temperature":0,"enable_thinking":false,"max_tokens":64,"messages":[{"role":"user","content":"What is the capital of France? One word."}]}'
# Two solid-colour PNGs: 32 x 32 red, upscaled to the floor; 4096 x 4096 blue, the item bound.
python3 - <<'EOF'
import struct, zlib
def png(path, side, rgb):
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = (b"\0" + bytes(rgb) * side) * side
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
png("small.png", 32, (220, 30, 30))
png("large.png", 4096, (30, 60, 220))
EOF
for image in small large; do
  printf '{"model":"qwen3.8-27b","temperature":0,"enable_thinking":false,"max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"What colour is this image? One word."},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]}]}' \
    "$(base64 -w0 "$image.png")" > "$image.json"
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d "@$image.json"
done
```

The release binary is also checked to carry code for 8.6 and 12.0, whatever
list the build was given (`mk/cuda-build-check.sh --require`, in CI, #482).

The same check holds the `86` cubin to the full-rate rule (spec #503, #512): an
RTX 30 or RTX 40 card runs some tensor-core forms at half the rate of others, and
a kernel the sm_8x prefill work added uses only the full-rate ones, INT8 and HMMA
with an f16 accumulator. `mk/full-rate-kernels.txt` is the one place that lists
the covered kernels and #488's exceptions. The check dumps the `sm_86` SASS with
`cuobjdump` (the CUDA toolkit's, wherever nvcc is) and refuses a covered kernel
that holds HMMA with an f32 accumulator, any BF16, TF32 or FP8/FP6/FP4 form, a
warpgroup or tensor-memory MMA or a form it does not know, or that holds no
integer MMA where the list asks for it; the refusal names the kernel and the
opcode. It also refuses a listed kernel that no binary carries (a rename must
not skip the rule) and a dump in which the negative control, the existing window
kernel's f32-accumulate form, passes: a check that cannot fail is not a pass.

The gate records correctness only: KLD, top-1, pass or fail per part, and the
small image's patch count. Never a speed, never a detail of the machine. The
same procedure, for contributors, is in
`docs/user/contributing/releases.md`.

### A new kernel and the full-rate rule

A kernel added to the sm_8x path joins the list in the same commit, with one line
in `mk/full-rate-kernels.txt`: its role (`mma` for a GEMM or band kernel, which
must hold integer MMA, or `cuda` for a kernel that runs on the CUDA cores), its
name without template arguments (every instantiation is held to the rule) and its
source file. A test in `crates/server/tests/cuda_build_recipes.rs` fails when a
file the list names declares a kernel it does not list, and when a listed kernel
is not in its file. A kernel that predates the rule and stays as it is takes an
`exempt` line whose comment names the issue that decided it, never a reason of
its own; #488's kernels are the only ones. A form the rule refuses is fixed in the
kernel, not excused in the list. The selftest (`make cuda-build-check-selftest`)
runs the whole rule on SASS text and needs no GPU or CUDA toolkit; to run the
check by hand on a build, `IGNIS_CUDA_ARCHITECTURES='86;120a' bash
mk/cuda-build-check.sh --require '86;120a' <binary>...`.

## Done means

1. every new or changed behavior has a test that covers it
2. `cargo test` passes workspace-wide (machine-local skips allowed)
3. `cargo check --workspace --features cuda,logits-capture --tests` is
   clean (`make check CUDA=1`). The
   cuda-gated test targets (`#![cfg(feature = "cuda")]`) are the ones
   `cargo test` never compiles, so a signature change can leave them
   uncompilable and nothing says so until someone runs the GPU profile
   — which needs a free 5090 and so runs rarely. GitHub #132 did exactly
   that: it added a third argument to `apply_chat_template`, missed two
   call sites under `crates/server/tests/`, and blocked the next GPU
   profile outright (found and fixed in #119). This check costs ~30s,
   needs no GPU, and reports *every* broken site rather than stopping at
   the first — which is how the second of those two was found at all.
   `logits-capture` adds the one GPU test the profile never builds
   (`flash_next_rope_logits_gpu.rs`, bit-exact on the card that recorded
   its capture, ADR 0059), which nothing else would compile.
4. kernel changes: `kernel/build.ps1` clean + the leaf's op tests green +
   the GPU profile green on a free 5090 (`scripts/gpu-profile.ps1` —
   preflight passed, `IGNIS_GPU_PROFILE=1` for the run — never a skip)
