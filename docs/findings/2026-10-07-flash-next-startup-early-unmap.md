# Early unmapping and a persistent n-gram cache shorten Flash-Next startup

- Kind: experiment
- Status: current
- Observed: 2026-10-07
- Last verified: 2026-10-07
- Scope: core / Flash-Next engine startup and n-gram hot-row loading on Windows
- Related: [previous startup measurement](2026-10-06-flash-next-on-the-5090.md), [mapped-file read slowdown](2026-10-06-flash-next-decode-round.md)
- Superseded by: none

## Question

Can an isolated lifetime change shorten startup without changing weights, hot rows or cache budgets?

## Evidence

Worktree flash-next-load-time; branch experiment/flash-next-load-time; base de08637.
Original flash-next remains clean at de08637; no artifact rewritten.
The probe links the existing unchanged CUDA archives read-only, without rebuilding the original kernel.
Shared GPU lock load-probe held for the runs, released afterwards; GPU and foreign processes checked at each start.

Probe calls FlashNextEngine::load, then generates eight greedy tokens from the first 128 tokens of reference window 0.
Prefill 1024, context 40960, three lanes, expert cache 16,000,000,000 bytes, default 1-GiB hot cache,
hq-e8-2b KV, MTP k=2, draft-row budget 3; no host retained slots or KV-RAM tier.

| Order | Mode | Engine load (s) | N-gram preparation (s) |
|---|---|---:|---:|
| 1 | baseline1 | 113.712 | 75.125 |
| 2 | early-unmap1 | 63.578 | 29.201 |
| 3 | early-unmap2 | 72.437 | 38.415 |
| 4 | baseline2 | 110.791 | 76.078 |


Mean engine startup: 112.252 -> 68.008 s, saving 44.244 s (39.4%).
Mean n-gram preparation: 75.602 -> 33.808 s (55.3% shorter).
Original run 1: open/bind/MTP 0.896 s, device weights 2.430 s, expert allocation 3.599 s,
expert reads 28.814 s, warm start 1.656 s, n-gram 75.125 s, program 0.035 s, sequence pool 0.003 s, graphs 0.065 s.

All four outputs: [5854, 63, 318, 1719, 3817, 579, 1765, 2234]; decode mask 7, verify mask 1.
Fixture regression compares every cached byte and gathered hot/cold, duplicate and boundary rows.
Validation: cargo test --workspace completed successfully (2295 passed, 0 failed, 3 ignored), including the real-artifact SHA checks and real n-gram staging test.

Reproduce from this worktree in PowerShell, with GPU exclusively reserved and checked first:

    $env:IGNIS_LOAD_EXPERIMENT_KERNEL_LIB_DIR = 'F:/ai/opencode/.inference-qwen-worktrees/flash-next/kernel/build'
    cargo build -p ignis-core --features cuda --release --example flash_next_load_probe
    $env:PATH = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/x64;' + $env:PATH
    Remove-Item Env:IGNIS_LOAD_EXPERIMENT -ErrorAction SilentlyContinue
    ./target/x86_64-pc-windows-msvc/release/examples/flash_next_load_probe.exe
    $env:IGNIS_LOAD_EXPERIMENT = 'early-unmap'
    ./target/x86_64-pc-windows-msvc/release/examples/flash_next_load_probe.exe

Logs: .scratch/baseline1.log and .scratch/{early-unmap1,early-unmap2,baseline2}.{out,err}.log.
Machine-readable checked comparison: .scratch/results.json.
The build override is experiment infrastructure, not a production bypass of the kernel rebuild checks.

## Finding

Observed: closing the Reader mapping before the existing hot-row direct reads substantially reduces that phase.
Inference: Windows mapped-file/cache overhead causes the slowdown, consistent with the independent decode-round experiment.
The internal Windows mechanism was not traced.

Hot-row preparation dominates startup here; graph capture and warm-start copies do not.

## Implications

Copy table descriptors, hash buffers and hot-row IDs into owned memory, close the mapping,
then load the exact same rows using the existing independent direct readers.

## Limits and unknowns

Two runs per mode in A-B-B-A order; no reboot or forced page-cache purge.
Original n-gram times stayed 75-76 s even when run last.
This measures engine initialization, not server process-to-HTTP-readiness.
The server loader has an analogous Reader lifetime, but its scheduler, retained slots and KV-RAM tier were not measured.
Eight matching tokens are a smoke check, not full output-quality validation.
Only this Windows machine and these budgets are covered; the variant remains isolated and opt-in.

## Follow-ups

If selected for integration, apply to the server loader and measure HTTP readiness with matched settings.

## Follow-up: persistent hot-row cache (2026-10-07)

The isolated implementation now closes the mapping unconditionally in both
`FlashNextEngine` and the server's `FlashNextLeaf`, after extracting the main
artifact identity and loading the MTP companion. The original `flash-next`
worktree, branch and executable remain untouched.

Persistence is on by default; the server exposes `--persist-ngram-cache
true|false` and `--persist-ngram-cache-path auto|directory`, with the matching
`IGNIS_` environment variables. See [user reference](../user/README.md#flash-next-n-gram-startup-cache).

The compact file stores an 80-byte versioned header (magic/version, key,
payload length and SHA-256) followed by exactly the selected row bytes.
The key binds the canonical source path, file length and modified/created
stamps, container directory hash, table layout, selected ordered row IDs,
and the packer's optional `ngram_cache_source` record. New packs compute
that record from ordered n-gram object descriptions and verified DONE digests
plus slice boundaries, without a second read of the 29 GB table. It survives
packing interruptions and repeated finalization. The existing real artifact
has no such record: this experiment exercises the legacy stamp-based path,
and does not modify or repack it. Preserving all source stamps and recorded
digests while editing payload bytes requires independent source verification.

Readers validate exact length and checksum; writers hold a nonblocking OS
lock and publish a synced temporary file through rename. Missing, invalid,
unwritable or busy caches fall back to ordinary row loading. Disabled
persistence performs no cache reads/writes. Old keyed files are not pruned.

### Measurements

Same probe parameters and greedy smoke prompt as the initial experiment:
prefill 1024, context 40960, three lanes, 16,000,000,000-byte expert cache,
default 1 GiB hot-row budget, MTP k=2 / verify rows=3, no retained or KV-RAM
host pools. All runs are new processes; no OS page-cache purge or reboot.
No build/test was active during the GPU timing runs. A shared exclusive
`.swarm/gpu.lock` was held for each run.

| Mode | Engine load (s) | N-gram phase (s) | Persistent result |
|---|---:|---:|---|
| Early unmap, persistence disabled | 68.156 | 29.175 | no disk cache accessed |
| Early unmap, first cache miss | 67.952 | 34.037 | saved |
| Early unmap, cache hit 1 | 41.199 | 6.815 | restored, checksum verified |
| Early unmap, cache hit 2 | 43.656 | 6.824 | restored, checksum verified |

The mean cache-hit load is 42.428 s: 37.8% below this session's disabled-cache
run, and 62.2% below the original mapping-retained mean of 112.252 s.
The n-gram phase falls by 76.6% from 29.175 to 6.820 s. The first persistence
miss adds 4.862 s to that phase; different expert-read times account for the
nearly equal whole-engine miss/disabled times. The saved file is
1,028,050,730 bytes (payload 1,028,050,650, 11,422,785 selected rows).

Every run generated `[5854,63,318,1719,3817,579,1765,2234]`, with decode
graph mask 7 and verify graph mask 1. These are smoke checks, not a complete
model quality acceptance run.

### Real server verification

Started/stopped through `make start` / `make stop`, isolated port 8019,
cache path `.scratch/ngram-cache`, 4G VRAM headroom, MTP k=2 / rows=3,
prompt reuse off, retained host 0 and host KV pool 0. Make's Flash-Next
defaults here were context 131072 and prefill 8192; therefore its readiness
time is a separate check, not a matched baseline comparison.

The server logged a real cache hit and `ignis.model.loaded` on the GPU;
`/v1/models` returned `qwen3.8-flash-next` with context 131072 after 46.4 s.
The chat prompt `Scrivi solo: cache pronta` returned `cache pronta` (2 tokens,
finish stop, 18 prompt tokens); server counters show MTP 2 drafts, 1 accepted.
Only the managed server PID 47772 was stopped. The shared lock was removed
and VRAM returned to ~2.2 GiB desktop usage.

Raw evidence: `.scratch/cache-{disabled,miss,hit1,hit2}.{out,err}.log`, the
matching JSON records, `.scratch/cache-server-start.log`,
`.scratch/cache-server-smoke/ignis-server.log`,
`.scratch/cache-server-smoke.json`, and `.scratch/cache-server-stop.log`.
These temporary logs are untracked. Windows runtime is verified; both OS
path resolution branches are unit tested, but no Linux runtime was run here.

### Final validation

`cargo test --workspace -j 4` passed: 2,301 tests passed, 0 failed, 3 ignored,
214 result groups. This includes source/content invalidation, disabled mode,
both OS cache directory branches, checksum/version/truncation recovery,
unwritable and busy writer fallback, hot-row restoration versus direct rows,
budget changes, CLI precedence/invalid values, packer resume digest stability,
and the existing real artifact and n-gram checks.

The CUDA server and load probe built successfully in release mode against
the unchanged original CUDA archives. Only the new module was rustfmt'd;
`git diff --check` passed. Existing linker CRT warnings were unchanged.
Evidence: `.scratch/cache-build-final.log`, focused cache/packer/table logs,
`.scratch/cache-workspace-tests.log` and `.scratch/cache-workspace-summary.json`.
No changes were merged or committed into the original worktree.

## Integration cleanup

Removed the experimental CUDA build override, `[LOAD-PROBE]` instrumentation,
duplicate `from_owned_artifact` API and unrelated formatting changes. The
standalone probe source is archived under untracked `.scratch/`; its Cargo
example entry is removed. The branch now contains the persistent cache,
early unmapping in engine/server, packer source identity, tests and user docs.
Historical timing evidence above used the diagnostic build before cleanup.

Post-cleanup validation: `cargo test --workspace -j 4` passed again, with
2,301 passed, 0 failed and 3 ignored across 214 result groups. The disabled
cache fixture now exercises the same owned-reader production entry point;
there is no separate experimental ownership API. The normal CUDA build
script is restored byte-for-byte from the base commit. No additional GPU
run was performed for this diagnostic-only cleanup. Evidence:
`.scratch/cleanup-workspace-tests.log` and `.scratch/cleanup-workspace-summary.json`.
