# The GPU profile fits one run on the card, and the loser dies silently

- Kind: experiment
- Status: current
- Observed: 2026-09-13
- Last verified: 2026-09-13
- Scope: test harness / GPU profile, GPU exclusivity
- Related: [GitHub #145](https://github.com/gpillon/ignis/issues/145),
  [GitHub #128](https://github.com/gpillon/ignis/issues/128),
  [GPU profile wall time](2026-09-12-gpu-profile-wall-time.md),
  `docs/agents/testing.md`, ADR 0006
- Superseded by: none

## Question

#145 recorded the GPU profile's serialized Rust sweep crashing on 2 of 3
attempts during #128's gate session, each time in a different unrelated test,
each time with an abrupt process exit, `exit code: 1`, and no panic text
despite `--nocapture`. Its leading hypothesis was Windows/WDDM device-memory
pressure accumulating over a long sequential run. Is the profile flaky on this
machine, and does it have to be fixed before a gate run?

## Evidence

Three consecutive profile runs on 2026-09-13, main worktree, ninfer stopped,
nothing else on the card, GPU at 2,355 MiB idle baseline before the first:

| run | invocation | exit | `gpu tests (serialized)` | tests |
|---|---|---|---|---|
| 1 | full (kernel CTest included) | 0 | 711.5 s | 39 passed, 0 failed |
| 2 | `-SkipKernelBuild` | 0 | 623.6 s | 39 passed, 0 failed |
| 3 | `-SkipKernelBuild` | 0 | 582.8 s | 39 passed, 0 failed |

Zero failures, zero panics, zero skips across all three. Logs:
`.scratch/issue-145/gpu-profile-a{1,2,3}.log`.

Both tests #145 saw die ran and passed in every attempt. In particular
`prefill_chunk_and_traversal_sweeps` completes its whole width sweep including
**width 4096**, the exact transition it died on in #145's attempt 3.

Device memory sampled every 15 s across runs 2 and 3 (86 samples,
`.scratch/issue-145/gpumem.csv`), on a 32,607 MiB card:

```
15:40:56   2,461 MiB   idle
15:42:29  20,833 MiB   run 2 under way
15:49:18  26,072 MiB   run 2 peak
15:54:37   2,972 MiB   between runs
15:59:10  26,074 MiB   run 3 peak
16:02:12   2,464 MiB   idle
```

The crashed binary is byte-identical to today's: #145's attempt 3 log names
`chunk_decomposition_gpu-6a5ac7122f671a00.exe`, and run 3 today executes the
same hash. #145's crash logs are timestamped 2026-09-12 20:04-20:34, after
every commit of that day (the last was 17:27), so the tree was not moving
under them. No script in `scripts/` kills a process.

#128's session ran **two agent legs concurrently against one worktree** and is
the session that produced both crashes; its own record notes an episode of two
`gpu-profile.ps1` processes launched at once.

## Finding

**Observed.** The profile is green three times out of three on an exclusive
card, the code is unchanged since the crashes, memory returns to the 2.4-2.5
GiB idle baseline between runs rather than drifting upward, and one run peaks
at 26.1 GiB — 80% of the card.

**Inferred.** The accumulating-pressure hypothesis is not supported: nothing
accumulates across runs, and within a run the sweep reaches its peak and
releases it. What the numbers do support is a headroom argument. A single run
leaves roughly 6 GiB free, less than one more artifact load (~17.4 GiB leaf
VRAM at the sweep's own reported figures), so a second GPU-touching process
entering mid-run — another profile, an `ignis-server`, a bench leg — cannot
fit. The loser of that race dies exactly as #145 describes: killed below the
Rust panic machinery, so no panic text, no assertion, nothing on stdout, and a
clean pass when the same test is re-run alone afterwards. Two unrelated tests
crashing in one session and none in three uncontended runs fits contention;
it does not fit a defect in either test.

**Inferred.** The preflight cannot prevent this. It inspects the GPU once,
before the run, and writes a marker; nothing re-checks while the sweep is
under way.

## Implications

- The GPU profile does not need a fix before a gate run. The condition it
  needs is the one ADR 0006 already asks for, read strictly: **one process on
  the card**, not merely "ninfer is stopped".
- Exclusivity is a property of the *machine*, not of a worktree. Two agent
  sessions in two worktrees share one 5090, and neither preflight sees the
  other.
- `docs/agents/testing.md` carries this as runbook step 0, and `CONTEXT.md`'s
  **GPU profile** entry now says what *free* means.

## Limits and unknowns

- Three clean runs exclude the 2-in-3 failure rate #145 recorded (that rate
  would produce three consecutive clean runs about 3.7% of the time). They do
  not exclude a lower rate.
- No crash was reproduced deliberately: contention is inferred from the
  headroom figure and the session's known concurrency, not demonstrated by
  running two profiles at once on purpose.
- The exit status was never captured from the dying test process itself, only
  as cargo reported it (`exit code: 1`). What signal or driver-level failure
  ended it is still unidentified.
- Measured on one machine, one card, one driver.

## Follow-ups

- A mid-run guard (a periodic re-check, or a machine-wide lock a second run
  would have to acquire) would turn this from a convention into something
  enforced. Not filed; the convention is documented first.
