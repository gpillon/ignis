# GPU status and guard for the Makefile (the Windows GPU_* hooks,
# mk/os/windows.mk). The 5090 fits one engine or GPU test at a time, and the
# loser of a race dies with no diagnostic (docs/agents/testing.md), so:
#
#   -Action status   VRAM in use, nvidia-smi's compute apps, every process
#                    that may hold the card (ninfer*, ignis-server,
#                    ignis-bench, *_gpu-* test binaries) and the GPU lock's
#                    holder (mk/windows/gpulock.ps1)
#   -Action guard    exit 1 while the card is held: the GPU lock, held by any
#                    process but -Holder's (a caller that holds it already),
#                    refused at once naming its holder; then
#                    scripts/gpu-preflight.ps1
#                    (ninfer + a VRAM threshold) plus the ignis processes the
#                    preflight does not look for. The preflight's pass marker
#                    is removed afterwards: starting a server must not leave
#                    an authorization for the GPU profile behind.
#   -Action run      the sh line in $env:IGNIS_MK_GPU_CMD under the GPU lock
#                    (make gpu-run, make kernel-test): waits up to
#                    -WaitSeconds for the lock, runs the guard, then the
#                    command in a kill-on-close job, and frees the lock when
#                    it exits; the run's exit code is the command's.
#
# The lock is added beside the marker and the process check, not in place of
# them: ignis_core::gpu_profile reads the marker, and a worktree on an older
# commit knows only those two.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('status', 'guard', 'run')]
    [string]$Action,
    [int]$ThresholdMiB = 8192,
    [int]$Holder = 0,
    [int]$WaitSeconds = 900
)

$ErrorActionPreference = 'Stop'
$Repo = Resolve-Path (Join-Path $PSScriptRoot '..\..')
. (Join-Path $PSScriptRoot 'gpulock.ps1')

function Show-GpuLock {
    $held = Get-GpuLockHolder
    if ($held) { "GPU lock: held ($(Get-GpuLockPath)) by $(Format-GpuLockHolder $held)" } else { "GPU lock: free ($(Get-GpuLockPath))" }
}

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
        Write-Host (Show-GpuLock)
        exit 0
    }

    'guard' {
        $held = Get-GpuLockHolder
        if ($held -and [int]$held.pid -ne $Holder) {
            Write-Host "GPU guard: refused -- $(Show-GpuLock)"
            Write-Host "  wait or ask (docs/agents/testing.md, who holds the card); the lock is free once its holder exits"
            exit 1
        }

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

    'run' {
        $command = $env:IGNIS_MK_GPU_CMD
        if (-not $command) {
            Write-Host 'gpu.ps1 run: no command (make gpu-run CMD="...")'
            exit 2
        }
        try {
            $hold = Lock-Gpu -Command "gpu-run: $command" -WaitSeconds $WaitSeconds
        } catch {
            Write-Host $_.Exception.Message
            exit 1
        }
        $p = $null
        $code = 1
        try {
            & (Get-Process -Id $PID).Path -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -Action guard -ThresholdMiB $ThresholdMiB -Holder $PID
            if ($LASTEXITCODE -ne 0) { exit 1 }
            . (Join-Path $PSScriptRoot 'common.ps1')
            $job = New-KillOnCloseJob
            # sh reads the line from the environment, so it crosses no quoting.
            $sh = (Get-Command sh -ErrorAction Stop).Source
            $p = Start-Process -FilePath $sh -ArgumentList '-c', '"eval \"$IGNIS_MK_GPU_CMD\""' -NoNewWindow -PassThru
            $null = $p.Handle
            Add-ProcessToJob $job $p
            while (-not $p.HasExited) { Start-Sleep -Milliseconds 200 }
            $code = $p.ExitCode
        } finally {
            # Ctrl+C: the command got it too; give it the time to stop.
            if ($p -and -not $p.HasExited) { Stop-Gently $p 20 'the command' }
            Unlock-Gpu $hold
        }
        exit $code
    }
}
