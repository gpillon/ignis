# The GPU profile's wall time was mostly unoptimized CPU f64, and what remains is one model upload per test binary

- Kind: experiment
- Status: current
- Observed: 2026-09-12
- Last verified: 2026-09-12
- Scope: test harness / GPU profile, f64 layer oracle, artifact materialization
- Related: [GitHub #135](https://github.com/gpillon/ignis/issues/135),
  [ADR 0006](../adr/0006-exclusive-gpu-testing.md),
  [testing](../agents/testing.md),
  spec `.scratch/gpu-profile-speed/specs/01-gpu-profile-speed.md`
- Superseded by: none

## Question

`scripts/gpu-profile.ps1` had grown slow enough to throttle iteration, and
the serialized `--ignored` stage (ADR 0006) makes every case additive. Two
things were open. Where does the time actually go, and is it worth merging
the GPU test binaries into a single process so the ~16 GiB of weights are
uploaded once for the whole stage rather than once per binary?

## Evidence

Reported run, 2026-09-12, before any change: kernel CTest 65.52 s, and the
ten slowest Rust cases summing to ~718 s with many more in the 8-16 s band.

Two host-side causes were read off the code rather than guessed. The
workspace carried no `[profile.dev]` or `[profile.test]` override, so every
test ran at `opt-level = 0`; and `f64_reference`'s `Matrix::product`
dequantized NVFP4 and W8 per element in a scalar, single-threaded loop.
`layer_reference_real` is the clearest case because it touches no GPU at
all: it reads stored weights through a memory map and evaluates them on the
CPU, yet it was serialized inside the GPU sweep.

| `gqa_and_gdn_references_cover_two_tokens` | wall |
| --- | ---: |
| `opt-level = 0`, scalar rows (as the profile ran it) | 123.94 s |
| optimized, scalar rows | ~25 s |
| optimized, rows over threads | 3.86 s |

The middle row is the one that does not reproduce from this commit, because
the state it measures no longer exists. It was taken at the parent commit
(8b27864, before any change here) with
`cargo test -p ignis-artifact --release --test layer_reference_real -- --ignored`,
whose 35.42 s wall included 9.72 s of compilation. It is recorded only to
apportion the 32x between the two causes; the first and last rows are what
the conclusion rests on.

After both changes, on a free RTX 5090 with the full profile green:

| stage | wall |
| --- | ---: |
| `kernel/build.ps1 -Test` | 76.1 s |
| cpu f64 layer oracle | 3.8 s |
| gpu tests (serialized) | 563.6 s |
| total | 643.5 s |

The CPU oracle stage is 3.8 s only because it asks for the same `cuda`
feature set as the serialized stage. An earlier shape of the script omitted
it and cargo rebuilt the whole dependency tree under a second feature set,
which put that stage at 33-49 s for a 3.86 s test. Splitting a stage out of
the serialization buys nothing if the split makes cargo build twice.

Within the serialized stage, roughly 520 s is test wall time and the
remainder is compile and link. The GPU-bound cases did not move:
`chunked_and_per_token_prefill_agree_on_a_long_prompt` stayed at ~188 s. The
CPU-bound oracles collapsed: 111.44 s to ~11 s for the GQA layer, 67.06 s to
~10 s for GDN, 54.41 s to ~10 s for the degenerate program.

The upload cost came from `MaterializationStats::upload_seconds`, which the
materializer had been recording since it was written and which nothing read.
Printed from `model_load_gpu` under the profile:

```
ignis.profile.materialize: upload_seconds=8.511 h2d_bytes=17206863800 device_capacity_bytes=17206994432
```

Roughly twenty GPU-gated binaries each pay that once.

Every figure above regenerates itself, which is why no log is kept for them
(`*.log` is git-ignored anyway). On a free 5090:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu-profile.ps1
```

The stage table is printed at the end of the run, and the upload line comes
from `model_load_gpu` under `--nocapture`. Three full runs on 2026-09-12
read the upload at 8.511 s, 8.491 s and 8.302 s, so it repeats within 2.5%.
Their totals were 787.6 s, 653.7 s and 643.5 s; the first was a cold build
and the second still carried the duplicate-feature-set compile described
above, so the spread between them is build time rather than test time.

## Finding

Observed: the pre-change profile's dominant *host* cost was an unoptimized,
single-threaded f64 oracle, not GPU work. Raising the profile's optimization
level and spreading the oracle's output rows over threads took its worst
case from 123.94 s to 3.86 s, a factor of 32, with no change to any
assertion or tolerance.

Observed: one full artifact upload costs ~8.4 s for 17.21 GB, about 2.0
GB/s. Repeated once per test binary, that is roughly 170 s of the 564 s
serialized stage — about 30% of that stage and a quarter of the 644 s run.

Inference: sharing that upload across the whole stage is real but modest
value, and the only way to reach it is one process for every GPU test. That
costs isolation. A CUDA context that takes an unrecoverable fault is sticky,
so the first hard failure would cascade into false reds across every test
after it and the run would no longer name the test that broke. Trading a
fifth of the wall time for a verdict that cannot be read on the day it fails
is the wrong side of this repository's testing doctrine.

Observed, and the reason a cheaper path exists: a `MaterializedArtifact`'s
arena can be released on demand with `release_arena`, and one `#[test]` fn
can already drive many properties over one shared `materialize()` — the
shape `decode_graph_gpu.rs` and `sampling_gpu.rs` use, and which
`model_load_gpu.rs` adopted here, dropping two of its three uploads.

Inference: consolidating the test *files* that share a load shape recovers
most of the 170 s while keeping any cascade contained to one group. That is
the recommended direction over a single-process harness.

## Implications

- The profile's optimization level is load-bearing for its wall time, not an
  incidental build setting. Reverting it costs roughly two minutes per run.
- Bit-identity under optimization and threading is not assumed. Rust applies
  no fast-math, each row is summed by the same code in the same order
  whatever the thread split, a unit test compares schedules bit-for-bit, and
  `layer_reference_real` still matches fixtures recorded by the scalar path.
- `--nocapture` matters on the serialized stage: libtest swallows a passing
  test's stdout, which had been hiding every diagnostic that stage produces,
  including the chunk-decomposition sweep tables and this upload figure.

## Limits and unknowns

- The upload figure is one reading on one host. This machine caps the 5090 at
  PCIe Gen 3 x16 (see [Sequence snapshot transfer
  cost](2026-09-12-sequence-snapshot-transfer-cost.md)), so 2.0 GB/s is this
  host's number, not the card's ceiling; a Gen 5 host would shrink the prize
  further and strengthen the same conclusion.
- "Roughly twenty binaries" is counted from the test sources, not instrumented
  per binary. Only `model_load_gpu` reports its upload today.
- The sticky-context cascade is standard CUDA behavior and the repository has
  hit the fault class (GitHub #26), but it was not reproduced as a cascade
  here. Declining the single-process stage rests on the small measured prize
  as much as on that risk.
- Nothing here measures the kernel CTest stage, whose 32.73 s
  `ignis_hq_codec_test` is a sample sweep.

## Follow-ups

- Consolidating GPU test files that share a load shape into fewer one-test
  binaries, for most of the remaining ~170 s.
- Narrowing `bind_text_scope_27b` so a layer oracle uploads only the layers
  it touches. It would cut the upload sharply, but `load_qwen38_27b`
  validates the full topology, so it is a model-load ABI change rather than a
  harness one.
