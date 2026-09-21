# 01 — GPU profile speed

GitHub: #135

`scripts/gpu-profile.ps1` runs two stages: `kernel/build.ps1 -Test` (the
kernel CTest suite) and `cargo test --workspace --features cuda -- --ignored
--test-threads=1` (the GPU-gated Rust tests, serialized by ADR 0006).
The run recorded on #135 (2026-09-12) put the kernel stage at 65.52 s and
the ten slowest Rust cases at ~718 s combined, with many more in the 8–16 s
range on top. The Rust stage is serial by design, so those times are
additive.

This spec is about the **harness**, not the engine. Nothing here may change
what a test asserts.

## What the run is actually spending

Four causes, measured or read off the code rather than guessed. The kernel
stage is already built `-DCMAKE_BUILD_TYPE=Release`; its 32.73 s
`ignis_hq_codec_test` is sweep-bound, not flag-bound, and is out of scope.

### Lever 1 — the Rust GPU tests run at opt-level 0

The workspace `Cargo.toml` carries no `[profile.test]` or `[profile.dev]`
override, so every `--ignored` test runs unoptimized. This costs most where
a test is CPU-bound on scalar `f64`: `f64_reference`'s `Matrix::product`
dequantizes NVFP4 and W8 **per element** inside its inner loop.

Measured 2026-09-12, `gqa_and_gdn_references_cover_two_tokens`
(`layer_reference_real`, no GPU involved):

| profile | time |
|---|---|
| dev, as the GPU profile runs it | 123.94 s |
| release | ~25 s (35.42 s wall incl. 9.72 s compile) |

Roughly 5x. Rust applies no fast-math, so raising `opt-level` leaves `f64`
results bit-identical and every measured tolerance untouched. Keep
`debug-assertions` on: the point is speed, not fewer checks.

The same oracle path dominates `gqa_layer_gpu` (111.44 s),
`gdn_layer_gpu` (67.06 s) and `step_degenerate_gpu` (54.41 s).

### Lever 2 — a CPU-only oracle is serialized inside the GPU stage

`layer_reference_real` has no `cuda` cfg gate, makes no CUDA call, and reads
the artifact through a memory map. Its own doc comment already says to run
it in release. It nonetheless lands in the `--ignored` sweep and is
serialized under `--test-threads=1` for a GPU exclusivity it does not need,
adding its full wall time to the serial stage.

It is also the one `#[ignore]`d test in the sweep that self-skips with a
bare `eprintln!("skip: ...")` instead of asking `ignis_core::gpu_profile` —
so under `IGNIS_GPU_PROFILE=1` a missing artifact passes silently, which
`docs/agents/testing.md` forbids for everything else in the profile.

### Lever 3 — the f64 oracle is scalar and single-threaded

No `rayon` anywhere in the workspace. `f64_reference` walks output rows in a
plain `for` loop. The work is CPU-only and embarrassingly parallel over
rows, so `--test-threads=1` is not a constraint on it: the GPU is idle while
it runs.

### Lever 4a — one `materialize()` per process, and nobody measures it

`MaterializedArtifact` has no `Drop` of its own: its arena needs the
`Device` that produced it, so it is released either by dropping the
`CudaDevice` or by an explicit `release_arena(&mut device)`. The GPU tests
already encode the consequence as a rule, stated in `sampling_gpu.rs`,
`chunked_prefill_gpu.rs` and `decode_graph_gpu.rs`: one live `materialize()`
per process, *including across a second `#[test]` fn in the same binary*,
because `--test-threads=1` keeps them all in one process. That constraint is
why most GPU test files hold exactly one `#[test]`, which is in turn why the
sweep pays roughly one full upload per binary.

`MaterializationStats::upload_seconds` is computed inside `materialize()`
and read by nothing — no log, no assertion, no print. **The size of this
prize is currently unknown**, and it is what decides whether 4b is worth
its risk.

`model_load_gpu.rs` is the one file paying more than it needs: three
`#[test]` fns, each creating its own `CudaDevice` and its own
`materialize()`. It is not leaking — each releases its arena explicitly —
but it pays three uploads where `decode_graph_gpu.rs`'s one-test shape pays
one.

### Lever 4b — one process for the whole GPU stage

The ask behind #135: materialize once, run every GPU test in series against
those weights, release at the end — Jest's `beforeAll` / `afterAll` around a
test block.

The split the idea needs already exists and is clean:

- `materialize()` uploads the ~16 GiB of weights. **Immutable afterwards**,
  so sharing them cannot leak state between tests.
- `load_qwen38_27b()` reserves the leaf's scratch arena and is parameterized
  by `prefill_chunk_tokens`, `max_context_tokens` and `kv_format`. These
  genuinely differ per test, and `chunk_decomposition_gpu`,
  `chunked_prefill_gpu` and `model_load_gpu` sweep them deliberately.
  `Model` has a real `Drop` (`ignis_model_free`), so churning handles is cheap.

So the shareable unit is **the weights, not the model handle**. Teardown is
already expressible: `release_arena(&mut device)` frees the weights on
demand, so the "unload at the end" half of the ask needs no new machinery.

What blocks it is the process boundary: a Rust integration-test file is its
own binary, and VRAM is not shared across processes. `#[test]` also has no
setup or teardown hook, and a lazily-initialized `static` is never dropped,
so there is no `beforeAll`/`afterAll` in libtest. The real equivalent is
`harness = false` plus a hand-written `main` (optionally `libtest-mimic`),
which keeps libtest-compatible output while owning setup, ordering and
teardown.

**The risk is isolation, not state.** A CUDA context that takes an
unrecoverable fault (an illegal address, cf. #26) is sticky: every later
call in that context returns the same error until the process exits. Today
the per-binary boundary contains that. Merged, the first hard failure
cascades into false reds for every test after it, and the profile stops
naming the test that actually broke. For a repo whose doctrine is that a
green result must mean something, making red meaningless is the same defect
mirrored. Secondary risks: unwinding a panic through the FFI teardown chain
in an order the leaf never expected (#71 was exactly a teardown-ordering
bug), and losing per-test reproducibility, since rerunning one test no
longer replays the same device history.

Therefore 4b must land behind 4a's measurement and must answer the cascade
explicitly.

## Delivered

Branch `issue-135-gpu-profile-speed`. Levers 1, 2, 3 and 4a are in; 4b is
declined below on the strength of 4a's measurement.

- **Profile** (`Cargo.toml`): `[profile.dev]` and `[profile.test]` at
  `opt-level = 2`, `debug-assertions` explicitly on. This is wider than the
  criterion's "GPU-gated Rust tests build optimized" and deliberately so:
  the oracle lives in a library crate, an integration test links that
  library built under `dev`, and Cargo offers no narrower knob that reaches
  it. Ordinary `cargo build` is optimized as a side effect.
- **Parallel oracle** (`f64_reference`): `Matrix::product` splits its output
  rows over scoped threads through a new `fill_rows` helper; the row body is
  untouched, so the schedule cannot move a bit. `product_with_threads` takes
  the thread count so a unit test can compare schedules bit-for-bit, and
  small matrices stay serial below 2^16 elements. The test covers **all
  three decode arms** — BF16, NVFP4 and W8G32 — over synthetic payloads
  built from the real geometries, at thread counts coprime with the row
  counts plus one far above them. Covering only BF16 would have left the
  default CPU gate proving nothing about two of the three arms, since
  `layer_reference_real` is `#[ignore]`d and needs the real artifact.
- **CPU oracle out of the serialized stage**: `layer_reference_real` now
  decides a missing artifact through `ignis_core::gpu_profile` (a
  dev-dependency cycle back onto `ignis-core`, the one Cargo allows) instead
  of a bare `eprintln!`, and `scripts/gpu-profile.ps1` runs it as its own
  full-parallelism stage while the serialized sweep skips it by name. That
  stage asks for the same `cuda` feature set as the serialized one: without
  it cargo builds the whole dependency tree a second time, which cost 30-45 s
  and swallowed most of this lever's gain. The `--skip` couples the script to
  a Rust fn name, so the script first asks cargo to list the tests and fails
  loudly if the name stops resolving, rather than silently returning the
  oracle to the serialized sweep.
- **One upload in `model_load_gpu`**: its three `#[test]` fns became one,
  sharing a single `CudaDevice` and `materialize()`, and it prints the
  upload figure that had been recorded and never read. Every assertion from
  the three survives verbatim. Two consequences worth naming rather than
  burying: the three names #135's own timing table refers to
  (`real_nvfp4full_model_load_binds_every_text_scope_object`,
  `larger_prefill_chunk_reserves_more_program_vram`,
  `an_unaffordable_prefill_chunk_fails_the_load_naming_the_shortfall`) are
  now one, `the_real_artifact_loads_and_its_program_scratch_tracks_the_prefill_chunk`;
  and a failure in an early section now hides the later ones — 4b's own
  cascade objection in miniature, accepted here at three properties in one
  binary where it was refused at forty tests in one process, and already the
  repo's established shape in `decode_graph_gpu.rs` and `sampling_gpu.rs`.
- **Per-stage timing** in `scripts/gpu-profile.ps1`, reported from `finally`
  so a failed run still says where its time went, plus `--nocapture` on the
  serialized stage: libtest swallows a passing test's stdout, which had been
  hiding every diagnostic that stage exists to produce.

Measured on a free 5090, full profile green, never a skip:

| | before (#135, 2026-09-12) | after |
|---|---|---|
| `gqa_and_gdn_references_cover_two_tokens` | 123.94 s | 3.86 s |
| `gqa_layers_match_f64_reference` | 111.44 s | ~11 s |
| `gdn_layer_matches_f64_reference` | 67.06 s | ~10 s |
| `degenerate_program_matches_f64_reference_for_four_tokens` | 54.41 s | ~10 s |
| whole run, all three stages | not recorded | 643.5 s |

The GPU-bound cases are unchanged, as expected:
`chunked_and_per_token_prefill_agree_on_a_long_prompt` stayed at ~188 s.

That the parallel oracle is bit-identical is not only asserted by the new
unit test: `layer_reference_real` still matches the committed
`layer_reference_*.bin` fixtures, and those were recorded by the scalar
path.

### 4b decision: declined, with the measurement

`materialize()` costs **~8.4 s** for 17.21 GB H2D (`upload_seconds` read at
8.511, 8.491 and 8.302 over three runs, 2026-09-12). Roughly twenty
GPU-gated binaries each pay it once, so about 170 s of the 564 s serialized
stage is repeated upload — call it 30% of that stage and a quarter of the
644 s run.

That is not enough to buy what a single-process stage costs. A CUDA context
that takes an unrecoverable fault is sticky, so merging ~40 tests into one
process turns the first hard failure into a cascade of false reds that no
longer names the test that broke. The profile's whole purpose is that its
verdict means something, and this trades a fifth of the wall time for a
verdict that cannot be read on the day it matters most.

The cheaper part of the same prize is available without that trade, because
the repo already has the mechanism: one `#[test]` fn running several
properties over one shared `materialize()`, as `decode_graph_gpu.rs`,
`sampling_gpu.rs` and now `model_load_gpu.rs` do. Consolidating the test
*files* that share a load shape — the Bf16 layer and program oracles are the
obvious group — would recover most of those 170 s while keeping a cascade
contained to one group. That is a rewrite of many test files and changes
test identities that gate records refer to by name, so it belongs in its own
issue rather than here.

## Acceptance

- [ ] The GPU-gated Rust tests build optimized. `cargo test` at the default
      CPU gate is unaffected in behavior, and `debug-assertions` stay on for
      the test profile.
- [ ] `gqa_and_gdn_references_cover_two_tokens` runs in well under half its
      recorded 123.94 s, with its assertions and tolerances unchanged.
- [ ] The f64 layer oracle is parallel across output rows and returns
      bit-identical results to the current scalar path, pinned by a test
      that compares the two on real weights.
- [ ] The CPU-only layer oracle no longer runs inside the serialized GPU
      stage, and it decides a missing artifact through
      `ignis_core::gpu_profile` rather than a bare `eprintln!` skip, so an
      absent fixture is a hard failure under `IGNIS_GPU_PROFILE=1`.
- [ ] `materialize()`'s upload time is reported (its existing
      `upload_seconds` reaches a log or the test output), so a profile run
      shows what the ~28 repeated uploads cost.
- [ ] `model_load_gpu.rs` performs one `CudaDevice` + `materialize()` for
      its three tests, in the shape `decode_graph_gpu.rs` already uses.
- [ ] `scripts/gpu-profile.ps1` prints an end-to-end wall time per stage, so
      the next run is comparable against #135's numbers without re-reading
      per-test output.
- [ ] A single-process GPU stage is delivered **or** declined in writing,
      with the decision resting on the measured upload cost from above. If
      delivered: one `materialize()` serves the whole stage, the first hard
      failure aborts the run and is reported as the failing test rather than
      producing a cascade of false reds, and teardown releases the device
      explicitly instead of relying on process exit.
- [ ] `cargo test` green workspace-wide, `cargo check --workspace --features
      cuda --tests` clean, and the full GPU profile green on a free 5090 —
      never a skip (`docs/agents/testing.md`, "Done means").
- [ ] `docs/agents/testing.md` records however the profile is run at the end
      of this work.

## Out of scope

- Any change to what a test asserts: no test removed, no test skipped, no
  tolerance widened, no coverage traded for time. The tolerances are
  measured values (#96 exists because an inherited one hid a real fault).
- The kernel CTest stage. It is already Release; `ignis_hq_codec_test`'s
  32.73 s is its sample sweep, a separate question.
- Narrowing `bind_text_scope_27b` so a layer test uploads only the layers it
  touches. It would cut the upload sharply, but `load_qwen38_27b` validates
  the full topology, so it is a model-load ABI change, not a harness change.
  Worth its own issue if 4a's measurement shows upload dominates.
- Sharing materialized VRAM across processes (CUDA IPC handles). A large
  escalation for the same prize 4b reaches inside one process.
- `--test-threads=1` for the GPU-touching tests. ADR 0006 is not negotiable
  here.
