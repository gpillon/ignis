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

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action
)

. (Join-Path $PSScriptRoot 'common.ps1')

switch ($Action) {
    'start' {
        Assert-NoServer
        $p = Start-IgnisServer
        $outcome = $null
        try {
            $outcome = Wait-IgnisReady $p
        } finally {
            # No outcome: the wait was interrupted (Ctrl+C).
            if (-not $outcome) {
                Stop-Gently $p 0 'ignis-server'
                Remove-Item $PidFile -ErrorAction SilentlyContinue
                Write-Host "interrupted before ready: the server was stopped"
            }
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
            $targets += @(Get-Process -Name 'ignis-server' -ErrorAction SilentlyContinue |
                Where-Object { -not $managed -or $_.Id -ne $managed.Id })
        }
        if ($targets.Count -eq 0) {
            $stray = @(Get-Process -Name 'ignis-server' -ErrorAction SilentlyContinue)
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
            Stop-Gently $t 0 'ignis-server'
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
        $others = @(Get-Process -Name 'ignis-server' -ErrorAction SilentlyContinue |
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
}
