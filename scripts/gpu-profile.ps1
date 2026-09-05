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
# surface), so this stays a process-level guard -- a script wrapping the
# commands, not a Rust harness feature -- same as ADR 0006 always described.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu-profile.ps1
#   ... -ThresholdMiB 4096       # forwarded to gpu-preflight.ps1
#   ... -SkipKernelBuild         # skip kernel/build.ps1 -Test (Rust GPU tests only)
#   ... -SkipCargoTests          # skip cargo test --workspace -- --ignored (kernel leaf only)
#
# Exit codes: 1 if the preflight refuses (its own message says why); otherwise
# the exit code of the first GPU-gated step that fails, or 0 if both pass.
# IGNIS_GPU_PROFILE is always cleared before this script exits, pass or fail.

param(
    [int]$ThresholdMiB = 8192,
    [switch]$SkipKernelBuild,
    [switch]$SkipCargoTests
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repository = Split-Path -Parent $ScriptDir

& (Join-Path $ScriptDir "gpu-preflight.ps1") -ThresholdMiB $ThresholdMiB
if ($LASTEXITCODE -ne 0) {
    Write-Error "GPU profile: preflight refused (see above) -- not running the GPU profile."
    exit 1
}

Push-Location $Repository
try {
    $env:IGNIS_GPU_PROFILE = "1"

    if (-not $SkipKernelBuild) {
        Write-Host "GPU profile: preflight passed -- kernel/build.ps1 -Test"
        & (Join-Path $Repository "kernel/build.ps1") -Test
        if ($LASTEXITCODE -ne 0) {
            Write-Error "GPU profile: kernel/build.ps1 -Test failed (exit $LASTEXITCODE)."
            exit $LASTEXITCODE
        }
    }

    if (-not $SkipCargoTests) {
        Write-Host "GPU profile: cargo test --workspace -- --ignored"
        & cargo test --workspace -- --ignored
        if ($LASTEXITCODE -ne 0) {
            Write-Error "GPU profile: cargo test --workspace -- --ignored failed (exit $LASTEXITCODE)."
            exit $LASTEXITCODE
        }
    }

    Write-Host "GPU profile: done. Restart ninfer (runbook step 4, docs/agents/testing.md)."
    exit 0
} finally {
    Remove-Item Env:\IGNIS_GPU_PROFILE -ErrorAction SilentlyContinue
    Pop-Location
}
