# Single entry point for the explicit GPU test profile (ADR 0006, GitHub #38).
#
# ADR 0006's Decision calls the guard "a preflight check in the bench/test
# harness" -- not a script a developer must remember to run standalone.
# `scripts/gpu-preflight.ps1` alone doesn't enforce that: nothing stops
# someone setting IGNIS_GPU_PROFILE=1 and running tests without ever running
# it. This script is the fix: it *is* the harness entry point -- it runs the
# preflight, and only sets IGNIS_GPU_PROFILE=1 and runs the GPU-gated work if
# the preflight passes. Follow the runbook (docs/agents/testing.md) and this
# is the only command after "stop ninfer" and before "restart ninfer".
#
# crates/core has no FFI and no GPU access (GitHub #39 removed the flat-C-ABI
# surface), so the GPU itself can only be inspected from here. The guard is
# not merely conventional, though: the preflight records its pass in a marker
# file and ignis_core::gpu_profile refuses the profile without a recent one,
# so `$env:IGNIS_GPU_PROFILE = "1"; cargo test ... -- --ignored` by hand fails
# loudly instead of running un-preflighted. This script consumes the marker
# (deleted in the finally block below) so it authorizes exactly this run.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu-profile.ps1
#   ... -ThresholdMiB 4096       # forwarded to gpu-preflight.ps1
#   ... -SkipKernelBuild         # skip kernel/build.ps1 -Test (Rust tests only)
#   ... -SkipCargoTests          # skip both cargo stages (kernel leaf only)
#
# Three stages, each timed and reported at the end (GitHub #135):
#   1. kernel/build.ps1 -Test          the leaf's own op tests (CTest)
#   2. cpu f64 layer oracle            CPU-only, full libtest parallelism
#   3. gpu tests (serialized)          --ignored, --test-threads=1 (ADR 0006)
# Stage 2 touches no GPU, so it does not belong inside stage 3's serialization;
# stage 3 skips it by name.
#
# Exit codes: 1 if the preflight refuses (its own message says why); otherwise
# the exit code of the first stage that fails, or 0 if all pass.
# IGNIS_GPU_PROFILE is always cleared before this script exits, pass or fail.

param(
    [int]$ThresholdMiB = 8192,
    [switch]$SkipKernelBuild,
    [switch]$SkipCargoTests
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repository = Split-Path -Parent $ScriptDir
# The preflight's pass marker (see gpu-preflight.ps1): this run consumes it,
# so it must not outlive the run and authorize a later un-preflighted one.
$MarkerPath = Join-Path $env:TEMP "ignis-gpu-preflight.ok"

# The CPU-only f64 layer oracle (GitHub #56). It reads the stored weights
# through a memory map and evaluates them on the CPU -- no CUDA call at all --
# so it runs as its own stage at libtest's default parallelism instead of
# adding its wall time to the serialized GPU sweep below, which it never
# needed (GitHub #135). The sweep skips it by name.
$CpuOracleTest = "gqa_and_gdn_references_cover_two_tokens"

# Per-stage wall time (GitHub #135): the profile's cost is the point of this
# script, so it reports where the time went instead of leaving the reader to
# add up per-test output.
$Timings = [ordered]@{}

# A GPU-gated step that failed: report it and leave with its own exit code
# (not a flattened 1 -- the caller wants to know which code came back).
function FailWithCode([string]$What, [int]$Code) {
    Write-Error "GPU profile: $What failed (exit $Code)."
    exit $Code
}

# Run one stage, record its wall time, and fail the run with the stage's own
# exit code. The timing is recorded before the failure check, so a failed run
# still reports how long the stage took.
function Invoke-Stage([string]$Name, [scriptblock]$Body) {
    Write-Host "GPU profile: $Name"
    $watch = [Diagnostics.Stopwatch]::StartNew()
    & $Body
    $code = $LASTEXITCODE
    $watch.Stop()
    $script:Timings[$Name] = $watch.Elapsed.TotalSeconds
    Write-Host ("GPU profile: {0} -- {1:N1}s" -f $Name, $watch.Elapsed.TotalSeconds)
    if ($code -ne 0) { FailWithCode $Name $code }
}

function Write-StageSummary {
    if ($script:Timings.Count -eq 0) { return }
    Write-Host ""
    Write-Host "GPU profile: stage wall time"
    foreach ($name in $script:Timings.Keys) {
        Write-Host ("  {0,10:N1}s  {1}" -f $script:Timings[$name], $name)
    }
    $total = ($script:Timings.Values | Measure-Object -Sum).Sum
    Write-Host ("  {0,10:N1}s  total" -f $total)
}

& (Join-Path $ScriptDir "gpu-preflight.ps1") -ThresholdMiB $ThresholdMiB
if ($LASTEXITCODE -ne 0) {
    Write-Error "GPU profile: preflight refused (see above) -- not running the GPU profile."
    exit 1
}

Push-Location $Repository
try {
    $env:IGNIS_GPU_PROFILE = "1"
    Write-Host "GPU profile: preflight passed -- running the profile."

    if (-not $SkipKernelBuild) {
        Invoke-Stage "kernel/build.ps1 -Test" {
            & (Join-Path $Repository "kernel/build.ps1") -Test
        }
    }

    if (-not $SkipCargoTests) {
        # Stage 3 excludes the CPU oracle by test-name string, which couples
        # this script to a Rust fn name. A rename would silently put the
        # oracle back inside the serialization -- the exact cost #135
        # removed, with nothing to signal it -- so check the name still
        # resolves before relying on it.
        $listed = & cargo test -p ignis-artifact --features cuda --test layer_reference_real -- --list
        if ($LASTEXITCODE -ne 0) { FailWithCode "listing the CPU oracle's tests" $LASTEXITCODE }
        if (($listed -join "`n") -notmatch [regex]::Escape($CpuOracleTest)) {
            Write-Error ("GPU profile: '{0}' is no longer a test in layer_reference_real, so stage 3's --skip would stop matching and re-run the CPU oracle inside the serialized sweep. Update the CpuOracleTest variable in this script." -f $CpuOracleTest)
            exit 1
        }

        # The CPU-only oracle, at full parallelism and off the GPU's critical
        # path. IGNIS_GPU_PROFILE is already set, so a missing artifact is a
        # hard failure here too (ADR 0006, docs/agents/testing.md).
        #
        # `--features cuda` matches stage 3's feature set: without it cargo
        # builds the whole dependency tree a second time under a different
        # set, which cost more than this stage's own test does. `--nocapture`
        # for the same reason stage 3 has it.
        Invoke-Stage "cpu f64 layer oracle" {
            & cargo test -p ignis-artifact --features cuda --test layer_reference_real -- --ignored --nocapture
        }

        # `--test-threads=1`: a GPU-gated binary with more than one test (e.g.
        # `openai_http_gpu.rs`) would otherwise run its tests concurrently
        # under libtest's default N-core parallelism -- each independently
        # loading the full artifact onto the single RTX 5090 (ADR 0006),
        # overcommitting the 32 GB card into shared/system GPU memory
        # (GitHub #75; #71 fixed the per-test teardown leak but never wired
        # serialization into this script).
        #
        # `--skip`: the stage above already ran the CPU oracle, and it is the
        # one `#[ignore]`d test in the sweep that touches no GPU.
        #
        # `--nocapture`: libtest swallows a passing test's stdout, which hid
        # every diagnostic this stage exists to produce -- the chunk-
        # decomposition sweep tables (GitHub #92) and the materialize upload
        # figure (GitHub #135) among them. Nothing interleaves at
        # `--test-threads=1`, so there is no reason to capture here.
        Invoke-Stage "gpu tests (serialized)" {
            & cargo test --workspace --features cuda -- --ignored --test-threads=1 --nocapture --skip $CpuOracleTest
        }
    }

    Write-Host "GPU profile: done. Restart ninfer (runbook step 4, docs/agents/testing.md)."
    exit 0
} finally {
    # In `finally`, so a run that failed a stage still reports where its time
    # went -- that is usually exactly what the reader wants to know.
    Write-StageSummary
    Remove-Item Env:\IGNIS_GPU_PROFILE -ErrorAction SilentlyContinue
    Remove-Item $MarkerPath -ErrorAction SilentlyContinue
    Pop-Location
}
