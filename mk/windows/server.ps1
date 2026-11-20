# Background ignis-server control for the Makefile (the Windows SERVER_*
# hooks, mk/os/windows.mk). Inputs: the IGNIS_MK_* environment, see
# common.ps1.
#
# `start` is the one deliberately persistent launch: the server gets a hidden
# console of its own and outlives make and the terminal, until `make stop`.
# A Ctrl+C while start is still waiting for readiness stops it again.
#
# The stop is a hard kill: a hidden background process has no console to
# deliver the Ctrl+C that triggers the server's graceful shutdown.
#
# `run` is `make run` with CUDA=1: the server in the foreground, on this
# console, holding the GPU lock until it exits.
#
# With IGNIS_MK_GPU_LOCK=1 (CUDA=1) start and run take the GPU lock before the
# server starts, refused at once naming its holder, run the guard under it
# with IGNIS_MK_GPU_GUARD=1, hand it to the server's pid and free it if the
# start fails; stop frees the lock a stopped server holds.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('start', 'stop', 'status', 'run')]
    [string]$Action
)

. (Join-Path $PSScriptRoot 'common.ps1')

switch ($Action) {
    'start' {
        Assert-NoServer
        $hold = Lock-GpuForServer 'make start'
        $p = $null
        $outcome = $null
        try {
            $p = Start-IgnisServer
            $hold = Move-GpuLockToServer $hold $p
            $outcome = Wait-IgnisReady $p
        } finally {
            # No outcome: the wait was interrupted (Ctrl+C), or the hand-over failed.
            if (-not $outcome -and $p) {
                Stop-Gently $p 0 'ignis-server'
                Remove-Item $PidFile -ErrorAction SilentlyContinue
                Write-Host "not ready (interrupted, or the GPU lock's hand-over failed): the server was stopped"
            }
            # Ready, or still loading after the timeout: the server keeps the lock.
            if ($outcome -ne 'ready' -and $outcome -ne 'timeout') { Unlock-Gpu $hold }
        }
        switch ($outcome) {
            'ready' { exit 0 }
            'timeout' { Write-Host "  still running in the background: make logs, make stop"; exit 1 }
            default { exit 1 }
        }
    }

    'stop' {
        $targets = @()
        $managed = Get-ManagedServer
        if ($managed) { $targets += $managed }
        if ($Force) {
            $targets += @(Get-Process -Name $ServerName -ErrorAction SilentlyContinue |
                Where-Object { -not $managed -or $_.Id -ne $managed.Id })
        }
        if ($targets.Count -eq 0) {
            $stray = @(Get-Process -Name $ServerName -ErrorAction SilentlyContinue)
            if ($stray.Count -gt 0) {
                $ids = ($stray | ForEach-Object { $_.Id }) -join ', '
                Write-Host "no server started by make; another ignis-server is running (pid $ids): make stop FORCE=1"
            } else {
                Write-Host "no ignis-server running"
            }
            Remove-Item $PidFile -ErrorAction SilentlyContinue
            exit 0
        }
        foreach ($t in $targets) {
            $held = Get-GpuLockIdentity $t.Id
            Stop-Gently $t 0 'ignis-server'
            Unlock-Gpu $held
        }
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        exit 0
    }

    'status' {
        $managed = Get-ManagedServer
        if ($managed) {
            Write-Host ("process: ignis-server pid {0}, started by make at {1:u}, {2:N0} MiB working set" -f `
                $managed.Id, $managed.StartTime, ($managed.WorkingSet64 / 1MB))
        }
        $others = @(Get-Process -Name $ServerName -ErrorAction SilentlyContinue |
            Where-Object { -not $managed -or $_.Id -ne $managed.Id })
        foreach ($o in $others) {
            Write-Host "process: ignis-server pid $($o.Id), not started by make (foreground run or another session)"
        }
        if (-not $managed -and $others.Count -eq 0) {
            Write-Host "process: no ignis-server running"
            if (Test-Path $PidFile) { Remove-Item $PidFile -ErrorAction SilentlyContinue }
        }
        exit 0
    }

    'run' {
        $hold = Lock-GpuForServer 'make run'
        $startArgs = @{ FilePath = $Bin }
        if ($ServerArgs) { $startArgs.ArgumentList = $ServerArgs }
        # A Ctrl+C reaches the server too: its graceful shutdown gets the time.
        exit (Invoke-Foreground $startArgs 'ignis-server' $hold -HandOver)
    }
}
