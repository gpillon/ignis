# GitHub #92 criterion 1: drive the per-chunk wall-time decomposition.
#
# Two passes over crates/core/tests/chunk_decomposition_gpu.rs on a free card
# (ADR 0006), each preceded by its own preflight so neither runs on a marker
# the other consumed:
#
#   A. clean     -- IGNIS_CHUNK_PROFILE unset. The chunk-width sweep with no
#                   instrumentation in the leaf at all, so the fixed
#                   per-chunk cost it regresses out is not an artifact of
#                   the events pass B records.
#   B. profiled  -- IGNIS_CHUNK_PROFILE set. Same sweep, plus one JSONL
#                   record per chunk (and per layer) from run_program_chunk.
#
# Output lands in .scratch/issue-92/: pass-a-clean.txt, pass-b-profiled.txt,
# chunks.jsonl.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .scratch/issue-92/run.ps1

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repository = Split-Path -Parent (Split-Path -Parent $Here)
$MarkerPath = Join-Path $env:TEMP "ignis-gpu-preflight.ok"
$Jsonl = Join-Path $Here "chunks.jsonl"

function Invoke-Pass([string]$Label, [string]$LogName) {
    & (Join-Path $Repository "scripts/gpu-preflight.ps1")
    if ($LASTEXITCODE -ne 0) { Write-Error "#92: preflight refused before pass $Label."; exit 1 }
    $env:IGNIS_GPU_PROFILE = "1"
    Write-Host "#92: pass $Label -- cargo test chunk_decomposition_gpu"
    # Windows PowerShell 5.1 wraps a native command's stderr in ErrorRecords,
    # which $ErrorActionPreference = "Stop" would turn into a terminating
    # error on cargo's ordinary progress output. Relax it around the call and
    # read the exit code instead.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & cargo test -p ignis-core --features cuda --test chunk_decomposition_gpu `
        -- --ignored --nocapture --test-threads=1 |
        Tee-Object -FilePath (Join-Path $Here $LogName)
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    Remove-Item Env:\IGNIS_GPU_PROFILE -ErrorAction SilentlyContinue
    Remove-Item $MarkerPath -ErrorAction SilentlyContinue
    if ($code -ne 0) { Write-Error "#92: pass $Label failed (exit $code)."; exit $code }
}

Push-Location $Repository
try {
    Invoke-Pass "A (clean)" "pass-a-clean.txt"

    Remove-Item $Jsonl -ErrorAction SilentlyContinue
    $env:IGNIS_CHUNK_PROFILE = $Jsonl
    $env:IGNIS_CHUNK_PROFILE_LAYERS = "1"
    try {
        Invoke-Pass "B (profiled)" "pass-b-profiled.txt"
    } finally {
        Remove-Item Env:\IGNIS_CHUNK_PROFILE -ErrorAction SilentlyContinue
        Remove-Item Env:\IGNIS_CHUNK_PROFILE_LAYERS -ErrorAction SilentlyContinue
    }

    Write-Host "#92: done. Records in $Jsonl"
} finally {
    Remove-Item Env:\IGNIS_GPU_PROFILE -ErrorAction SilentlyContinue
    Remove-Item $MarkerPath -ErrorAction SilentlyContinue
    Pop-Location
}
