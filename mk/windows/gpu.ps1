# GPU status and guard for the Makefile (the Windows GPU_* hooks,
# mk/os/windows.mk). The 5090 fits one engine or GPU test at a time, and the
# loser of a race dies with no diagnostic (docs/agents/testing.md), so:
#
#   -Action status   VRAM in use, nvidia-smi's compute apps, and every process
#                    that may hold the card (ninfer*, ignis-server,
#                    ignis-bench, *_gpu-* test binaries)
#   -Action guard    exit 1 while the card is held: scripts/gpu-preflight.ps1
#                    (ninfer + a VRAM threshold) plus the ignis processes the
#                    preflight does not look for. The preflight's pass marker
#                    is removed afterwards: starting a server must not leave
#                    an authorization for the GPU profile behind.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('status', 'guard')]
    [string]$Action,
    [int]$ThresholdMiB = 8192
)

$ErrorActionPreference = 'Stop'
$Repo = Resolve-Path (Join-Path $PSScriptRoot '..\..')

# The driver installs nvidia-smi into System32, which a PATH handed down
# through make + sh does not always carry: fall back to that location.
$NvidiaSmi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (-not $NvidiaSmi) {
    $candidate = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
    if (Test-Path $candidate) {
        $NvidiaSmi = $candidate
        $env:PATH = "$(Split-Path -Parent $candidate);$env:PATH"
    }
}

function Get-GpuHolders {
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -like 'ninfer*' -or
        $_.ProcessName -eq 'ignis-server' -or
        $_.ProcessName -eq 'ignis-bench' -or
        $_.ProcessName -like '*_gpu-*'
    }
}

function Show-Holders($holders) {
    foreach ($h in $holders) {
        Write-Host ("  pid {0,-7} {1}" -f $h.Id, $h.ProcessName)
    }
}

switch ($Action) {
    'status' {
        if ($NvidiaSmi) {
            # No per-process VRAM here: under WDDM nvidia-smi lists every
            # desktop app with used_memory [N/A], which says nothing.
            & $NvidiaSmi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
        } else {
            Write-Host "nvidia-smi not found on PATH"
        }
        $holders = @(Get-GpuHolders)
        if ($holders.Count -gt 0) {
            Write-Host "processes that may hold the card:"
            Show-Holders $holders
        } else {
            Write-Host "no ninfer / ignis-server / ignis-bench / *_gpu-* process running"
        }
        exit 0
    }

    'guard' {
        $holders = @(Get-GpuHolders | Where-Object { $_.ProcessName -notlike 'ninfer*' })
        if ($holders.Count -gt 0) {
            Write-Host "GPU guard: refused -- ignis GPU work is already running (another worktree or session counts):"
            Show-Holders $holders
            Write-Host "  make stop (FORCE=1 for one make did not start), or wait; GPU_CHECK=0 skips this guard"
            exit 1
        }

        # A standing pass means scripts/gpu-profile.ps1 is (or was until a
        # kill) mid-run: its tests read the marker, so it is never touched.
        $marker = Join-Path $env:TEMP 'ignis-gpu-preflight.ok'
        if ((Test-Path $marker) -and ((Get-Date) - (Get-Item $marker).LastWriteTime).TotalMinutes -lt 30) {
            Write-Host "GPU guard: refused -- a GPU profile pass is on record ($marker): a profile run looks in progress"
            Write-Host "  wait for it; a pass older than 30 minutes is ignored. GPU_CHECK=0 skips this guard"
            exit 1
        }
        $code = 1
        try {
            # The preflight's own success line points at gpu-profile.ps1,
            # which is not what is happening here: silence the host stream.
            & (Join-Path $Repo 'scripts\gpu-preflight.ps1') -ThresholdMiB $ThresholdMiB 6>$null
            $code = $LASTEXITCODE
        } catch {
            Write-Host "GPU guard: refused -- $($_.Exception.Message)"
            Write-Host "  GPU_CHECK=0 skips this guard (only if you know the card is free)"
            $code = 1
        } finally {
            Remove-Item $marker -ErrorAction SilentlyContinue
        }
        if ($code -eq 0) { Write-Host "GPU guard: card is free" }
        exit $code
    }
}
