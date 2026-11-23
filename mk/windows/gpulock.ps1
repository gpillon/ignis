# The GPU lock on Windows (ADR 0006, GitHub #707), the counterpart of
# mk/linux/gpulock.py. Dot-source it.
#
# One lock per card, keyed by $env:IGNIS_GPU_CARD ("0", the one card today,
# GitHub #387): a JSON file in $env:IGNIS_GPU_LOCK_DIR (default
# $env:ProgramData\ignis\gpu-lock, one place for the whole machine) that names
# its holder: the pid, the process's name and start time, the worktree, the
# command and the time the hold began. A record whose pid is gone, or now names
# another process (another name or start time), is free, so a holder killed
# before its finally block leaves nothing standing. Every change of the record
# (take, hand over, release), and every read, is made while a file beside it
# is held open exclusively, so two that find one stale record cannot both take
# it.
#
# It is added beside the preflight marker and the process check of
# mk/windows/gpu.ps1, not in place of them: ignis_core::gpu_profile reads the
# marker, and a worktree on an older commit knows only those two.
#
#   Lock-Gpu -Command <text> [-WaitSeconds <n>] [-ProcessId <pid>]
#            hold it (this process by default); waits up to n seconds (0:
#            refuses at once), then throws naming the holder
#   Move-GpuLock <hold> <identity>   hand a hold to another process
#   Unlock-Gpu <identity>            free it, only if that process holds it
#   Get-GpuLockHolder                the live holder's record, or $null

$GpuLockRepo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Get-GpuLockPath {
    $dir = if ($env:IGNIS_GPU_LOCK_DIR) { $env:IGNIS_GPU_LOCK_DIR } else { Join-Path $env:ProgramData 'ignis\gpu-lock' }
    $card = if ($env:IGNIS_GPU_CARD) { $env:IGNIS_GPU_CARD } else { '0' }
    Join-Path $dir "gpu-$card.lock"
}

# A running process as the lock names it, or $null when it is gone. The start
# time is UTC ticks; a process whose start time cannot be read (another user's,
# elevated) keeps an empty one, and its name alone then tells it apart.
function Get-GpuLockIdentity([int]$ProcessId) {
    $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $p) { return $null }
    $started = ''
    try { $started = [string]$p.StartTime.ToUniversalTime().Ticks } catch { }
    [ordered]@{ pid = $p.Id; name = $p.ProcessName; started = $started }
}

function Test-GpuLockSame($Record, $Holder) {
    $Record -and $Holder -and [string]$Record.pid -eq [string]$Holder.pid -and
        $Record.name -eq $Holder.name -and [string]$Record.started -eq [string]$Holder.started
}

# Whether the record's holder still runs: the same pid, name and start time
# (an unreadable start time counts as the same: free only when proven). That
# is the lock's safe side: a lock wrongly held refuses a run naming its holder,
# for a human to resolve, while one wrongly free lets two runs onto the card
# and the loser dies with no diagnostic.
function Test-GpuLockLive($Record) {
    if (-not $Record) { return $false }
    $now = Get-GpuLockIdentity ([int]$Record.pid)
    if (-not $now -or $now.name -ne $Record.name) { return $false }
    return -not ($now.started -and $Record.started -and $now.started -ne [string]$Record.started)
}

function Read-GpuLock([string]$Path) {
    try { return Get-Content -LiteralPath $Path -Raw -ErrorAction Stop | ConvertFrom-Json }
    catch { return $null }
}

function Format-GpuLockHolder($Record) {
    "pid $($Record.pid) ($($Record.name)) since $($Record.since), worktree $($Record.worktree): $($Record.command)"
}

# Run $Body with the file beside the lock open exclusively: the critical
# section of a change or a read. The OS closes it if this process dies.
function Invoke-GpuLockChange([string]$Path, [scriptblock]$Body) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $mutex = $null
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while (-not $mutex) {
        try {
            $mutex = [IO.File]::Open("$Path.mutex", 'OpenOrCreate', 'ReadWrite', 'None')
        } catch [IO.IOException] {
            if ([DateTime]::UtcNow -gt $deadline) { throw "GPU lock: $Path.mutex stayed open for 30s" }
            Start-Sleep -Milliseconds 20
        }
    }
    try { & $Body } finally { $mutex.Dispose() }
}

# Written to a temporary file and renamed over the lock, so a reader never sees
# half a record.
function Write-GpuLockRecord([string]$Path, $Record) {
    $temporary = "$Path.$([guid]::NewGuid().ToString('N'))"
    [IO.File]::WriteAllText($temporary, ($Record | ConvertTo-Json -Compress))
    if (Test-Path -LiteralPath $Path) { [IO.File]::Replace($temporary, $Path, [NullString]::Value) } else { [IO.File]::Move($temporary, $Path) }
}

function Lock-Gpu([string]$Command, [int]$WaitSeconds = 0, [int]$ProcessId = $PID) {
    $path = Get-GpuLockPath
    $me = Get-GpuLockIdentity $ProcessId
    if (-not $me) { throw "GPU lock: pid $ProcessId is not running" }
    $record = [ordered]@{
        pid = $me.pid; name = $me.name; started = $me.started; worktree = $GpuLockRepo
        command = $Command; since = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    }
    # The wait is timed on a Stopwatch: the wall clock moves in ticks, and a
    # wait timed on it could end before $WaitSeconds have passed (#728).
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $told = $false
    while ($true) {
        $state = @{}
        Invoke-GpuLockChange $path {
            $state.held = Read-GpuLock $path
            if (-not (Test-GpuLockLive $state.held)) {
                Write-GpuLockRecord $path $record
                $state.mine = $record
            } elseif (Test-GpuLockSame $state.held $me) {
                $state.mine = $state.held
            }
        }
        if ($state.mine) { return $state.mine }
        $left = $WaitSeconds - $clock.Elapsed.TotalSeconds
        if ($left -le 0) { throw "GPU lock: the card is held ($path) by $(Format-GpuLockHolder $state.held)" }
        if (-not $told) {
            Write-Host "GPU lock: waiting up to ${WaitSeconds}s for the card, held ($path) by $(Format-GpuLockHolder $state.held)"
            $told = $true
        }
        Start-Sleep -Milliseconds ([int][Math]::Min(1000, $left * 1000))
    }
}

# Hand the lock $Hold holds to the process $To (an identity), keeping the
# hold's worktree, command and start; $false if $Hold no longer holds it.
function Move-GpuLock($Hold, $To) {
    $path = Get-GpuLockPath
    $state = @{ moved = $false }
    Invoke-GpuLockChange $path {
        $held = Read-GpuLock $path
        if (Test-GpuLockSame $held $Hold) {
            Write-GpuLockRecord $path ([ordered]@{
                pid = $To.pid; name = $To.name; started = $To.started; worktree = $held.worktree
                command = $held.command; since = $held.since
            })
            $state.moved = $true
        }
    }
    $state.moved
}

function Unlock-Gpu($Holder) {
    $path = Get-GpuLockPath
    if (-not $Holder -or -not (Test-Path -LiteralPath $path)) { return }
    Invoke-GpuLockChange $path {
        if (Test-GpuLockSame (Read-GpuLock $path) $Holder) { Remove-Item -LiteralPath $path -ErrorAction SilentlyContinue }
    }
}

# Read under the mutex, as every change is made: on PowerShell 5.1 a read
# beside the writer's File.Replace can fail, the card then read as free, or
# make the replace throw. No lock directory: no lock was ever taken.
function Get-GpuLockHolder {
    $path = Get-GpuLockPath
    if (-not (Test-Path -LiteralPath (Split-Path -Parent $path))) { return $null }
    $state = @{}
    Invoke-GpuLockChange $path { $state.record = Read-GpuLock $path }
    if (Test-GpuLockLive $state.record) { return $state.record }
    return $null
}
