# GPU preflight for the explicit GPU test profile (ADR 0006, GitHub #38).
#
# ignis and ninfer cannot share the RTX 5090 (ADR 0006: a single engine's
# footprint already runs ~28 GB of the 32 GB card, so two full-size engines
# never fit). This script refuses to start the GPU profile while another
# process is holding a meaningful amount of GPU memory, and names the
# offending process (ninfer first, since it is the expected culprit) so the
# developer knows what to stop.
#
# Exit 0: the GPU is free. A marker file ($env:TEMP\ignis-gpu-preflight.ok) is
#         written to record the pass -- ignis_core::gpu_profile refuses the
#         profile without a recent one, so IGNIS_GPU_PROFILE=1 on its own is
#         not enough (ADR 0006 wants the preflight in the harness, not in the
#         developer's memory). scripts/gpu-profile.ps1 deletes the marker when
#         the run ends; docs/agents/testing.md has the runbook.
# Exit 1: the GPU is busy, or nvidia-smi could not be queried -- the message
#         names the offending process (or explains why the check could not
#         run). Any earlier marker is removed first, so a refused preflight
#         never leaves an older pass standing.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu-preflight.ps1
#   ... -ThresholdMiB 4096   # override the busy threshold

param(
    [int]$ThresholdMiB = 8192
)

$ErrorActionPreference = "Stop"

# The pass this run records, read by ignis_core::gpu_profile (which resolves
# the same path via std::env::temp_dir()). Cleared up front: until this run
# passes, no pass is on record.
$MarkerPath = Join-Path $env:TEMP "ignis-gpu-preflight.ok"
Remove-Item $MarkerPath -ErrorAction SilentlyContinue

function Fail([string]$Message) {
    Write-Error $Message
    exit 1
}

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Fail "GPU preflight: nvidia-smi not found on PATH -- cannot verify the GPU is free. Install the NVIDIA driver / CUDA toolkit, or check manually before running the GPU profile."
}

# The named reference engine (ADR 0006's expected culprit): if it is running,
# name it directly rather than relying on the memory-usage heuristic below.
$ninfer = Get-Process -Name "ninfer*" -ErrorAction SilentlyContinue
if ($ninfer) {
    $names = ($ninfer | ForEach-Object { "$($_.ProcessName) (pid $($_.Id))" }) -join ", "
    Fail "GPU preflight: ninfer is running ($names). Stop it before the GPU profile -- the runbook is 'stop ninfer -> preflight -> GPU profile -> restart ninfer' (ADR 0006, docs/agents/testing.md)."
}

$raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
if (-not $? -or -not $raw) {
    Fail "GPU preflight: the nvidia-smi query failed -- cannot verify the GPU is free."
}
$usedMiB = [int]([string]$raw | Select-Object -First 1).Trim()

if ($usedMiB -gt $ThresholdMiB) {
    Write-Host "GPU preflight: $usedMiB MiB in use (threshold $ThresholdMiB MiB)."
    $procs = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null
    if ($procs) {
        Write-Host "Processes nvidia-smi reports on the GPU:"
        $procs | ForEach-Object { Write-Host "  $_" }
    }
    Fail "GPU preflight: $usedMiB MiB in use (> $ThresholdMiB MiB) -- another process is holding VRAM. Stop it before running the GPU profile (ADR 0006)."
}

[DateTimeOffset]::UtcNow.ToString("o") | Set-Content -Path $MarkerPath -Encoding ascii
Write-Host "GPU preflight: free ($usedMiB MiB used, threshold $ThresholdMiB MiB). Pass recorded at $MarkerPath; run scripts/gpu-profile.ps1 to use it."
exit 0
