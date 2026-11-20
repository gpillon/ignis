# The behavioural test of the GPU lock on Windows (GitHub #707): mk/windows/gpulock.ps1 and the
# scripts around it, mk/windows/gpu.ps1 (status, guard, run), mk/windows/server.ps1 (start,
# stop, run), mk/windows/devui.ps1 and scripts/gpu-profile.ps1 (the whole run, -Test). It
# copies them into a temporary tree and runs them on stand-ins: sleeping PowerShells as
# holders, a stub server compiled here that answers any request and copies the lock's record
# as it does, a cargo.cmd and a kernel/build.ps1 that record their arguments, and an
# nvidia-smi.cmd first on PATH. It checks that the lock names its holder, that a refusal is at
# once and a wait ends at its limit, both naming the holder, that a dead holder or a reused pid
# is free, that two takers of one stale record never both win, that a start hands the lock to
# the server and a stop or a failed start frees it, and the profile's stages and named tests
# under it. TEMP, TMP and IGNIS_GPU_LOCK_DIR point under a temporary directory it removes, so
# the profile's marker and the machine's lock are never read or written; no stand-in carries
# a name the guards look for (ninfer*, ignis-server, ignis-bench, *_gpu-*).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File mk/windows/gpu-lock-selftest.ps1

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PowerShell = (Get-Process -Id $PID).Path
$script:failed = 0
function Assert-That([bool]$Ok, [string]$What) {
    if ($Ok) { return }
    $script:failed++
    Write-Host "  FAIL: $What"
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("ignis-gpu-lock-selftest-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
$tree = Join-Path $tmp 'repo'
$bin = Join-Path $tmp 'bin'
$state = Join-Path $tmp 'state'
foreach ($d in @("$tree\scripts", "$tree\mk\windows", "$tree\kernel", "$tree\web\node_modules\vite\bin", $bin, $state, "$tmp\tmp", "$tmp\run")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Copy-Item "$Repo\scripts\gpu-profile.ps1", "$Repo\scripts\gpu-preflight.ps1" "$tree\scripts\"
Copy-Item "$Repo\mk\windows\gpu.ps1", "$Repo\mk\windows\gpulock.ps1", "$Repo\mk\windows\common.ps1", "$Repo\mk\windows\server.ps1", "$Repo\mk\windows\devui.ps1" "$tree\mk\windows\"
Set-Content -Path "$tree\web\node_modules\vite\bin\vite.js" -Value '' -Encoding ascii

$env:TEMP = "$tmp\tmp"
$env:TMP = "$tmp\tmp"
$env:IGNIS_GPU_LOCK_DIR = "$tmp\lock"
$env:STUB_DIR = $state
$env:PATH = "$bin;$env:PATH"
foreach ($name in @('IGNIS_GPU_CARD', 'IGNIS_MK_FORCE', 'IGNIS_GPU_PROFILE', 'IGNIS_MK_GPU_LOCK')) { Remove-Item "Env:\$name" -ErrorAction SilentlyContinue }
$Lock = "$tmp\lock\gpu-0.lock"
$Marker = "$tmp\tmp\ignis-gpu-preflight.ok"
. "$tree\mk\windows\gpulock.ps1"
if (-not (Get-GpuLockPath).StartsWith($tmp) -or -not [IO.Path]::GetTempPath().StartsWith($tmp)) {
    Write-Host "gpu-lock-selftest: the lock ($(Get-GpuLockPath)) or TEMP ($([IO.Path]::GetTempPath())) is outside $tmp`: not running"
    exit 1
}

# The stand-ins.
Set-Content -Path "$bin\nvidia-smi.cmd" -Encoding ascii -Value @'
@echo off
echo %* | findstr /c:"memory.used" >nul
if not errorlevel 1 (
  if defined STUB_USED (echo %STUB_USED%) else (echo 100)
  exit /b 0
)
echo stub-gpu
'@
Set-Content -Path "$bin\cargo.cmd" -Encoding ascii -Value @'
@echo off
setlocal
if not defined STUB_SWEEP_RC set STUB_SWEEP_RC=0
if not defined STUB_ORACLE_RC set STUB_ORACLE_RC=0
if exist "%TEMP%\ignis-gpu-preflight.ok" (set marker=yes) else (set marker=no)
if exist "%IGNIS_GPU_LOCK_DIR%\gpu-0.lock" (set lock=yes) else (set lock=no)
if "%lock%"=="yes" if not exist "%STUB_DIR%\lock-seen.json" copy /y "%IGNIS_GPU_LOCK_DIR%\gpu-0.lock" "%STUB_DIR%\lock-seen.json" >nul
>>"%STUB_LOG%" echo cargo %* ^| profile=%IGNIS_GPU_PROFILE% marker=%marker% lock=%lock%
echo %* | findstr /c:" --list" >nul
if not errorlevel 1 (
  if not defined STUB_NO_ORACLE echo gqa_and_gdn_references_cover_two_tokens: test
  if defined STUB_LISTED for %%n in (%STUB_LISTED%) do echo %%n: test
  exit /b 0
)
if defined STUB_FAIL_TEST (
  echo %* | findstr /c:" %STUB_FAIL_TEST%" | findstr /c:"--exact" >nul
  if not errorlevel 1 exit /b 5
)
echo %* | findstr /c:" --workspace" >nul
if not errorlevel 1 exit /b %STUB_SWEEP_RC%
exit /b %STUB_ORACLE_RC%
'@
Set-Content -Path "$tree\kernel\build.ps1" -Encoding ascii -Value @'
$marker = if (Test-Path "$env:TEMP\ignis-gpu-preflight.ok") { 'yes' } else { 'no' }
$lock = if (Test-Path "$env:IGNIS_GPU_LOCK_DIR\gpu-0.lock") { 'yes' } else { 'no' }
Add-Content -Path $env:STUB_LOG -Value "kernel $($args -join ' ') | profile=$env:IGNIS_GPU_PROFILE marker=$marker lock=$lock"
exit [int]$env:STUB_KERNEL_RC
'@
Add-Type -OutputAssembly "$bin\stub-server.exe" -OutputType ConsoleApplication -TypeDefinition @'
using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

// The server's stand-in: records its pid, then answers any request with 200 and copies the
// lock's record as it stands; STUB_SERVER=exit-early exits with 7 before it serves, and
// STUB_SERVER=run copies the record after 1.5 s and exits with 3.
public static class StubServer {
    static void SeeLock(string state) {
        string lockFile = Path.Combine(Environment.GetEnvironmentVariable("IGNIS_GPU_LOCK_DIR"), "gpu-0.lock");
        for (int i = 0; i < 20; i++) {
            try { File.Copy(lockFile, Path.Combine(state, "lock-seen.json"), true); return; }
            catch (FileNotFoundException) { return; }
            catch (IOException) { Thread.Sleep(50); }
        }
    }

    public static int Main(string[] args) {
        string state = Environment.GetEnvironmentVariable("STUB_DIR");
        string mode = Environment.GetEnvironmentVariable("STUB_SERVER") ?? "";
        File.WriteAllText(Path.Combine(state, "server.pid"), System.Diagnostics.Process.GetCurrentProcess().Id.ToString());
        if (mode == "exit-early") { Thread.Sleep(300); return 7; }
        if (mode == "run") { Thread.Sleep(1500); SeeLock(state); return 3; }
        string bind = args[Array.IndexOf(args, "--server-bind") + 1];
        TcpListener listener = new TcpListener(IPAddress.Loopback, int.Parse(bind.Substring(bind.LastIndexOf(':') + 1)));
        listener.Start();
        byte[] answer = Encoding.ASCII.GetBytes("HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");
        while (true) {
            using (TcpClient client = listener.AcceptTcpClient()) {
                NetworkStream stream = client.GetStream();
                stream.Read(new byte[8192], 0, 8192);
                SeeLock(state);
                stream.Write(answer, 0, answer.Length);
            }
        }
    }
}
'@

$sleepers = New-Object System.Collections.ArrayList
function Start-Sleeper([int]$Seconds = 300) {
    $p = Start-Process -FilePath $PowerShell -ArgumentList '-NoProfile', '-Command', "Start-Sleep $Seconds" -WindowStyle Hidden -PassThru
    [void]$sleepers.Add($p)
    $p
}
function Get-FreePort {
    $l = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
    $l.Start(); $port = $l.LocalEndpoint.Port; $l.Stop(); $port
}
function Get-Field([string]$Path, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Path)) { return '' }
    [string](Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json).$Name
}
function Reset-State { Get-ChildItem $state | Remove-Item -Force -Recurse }
function Add-Setting([hashtable]$Settings, [string]$Name, [string]$Value) { $c = $Settings.Clone(); $c[$Name] = $Value; $c }

# Invoke-Script <script> <arguments> [<environment>]: the script in a PowerShell of its own, as
# make runs it; sets $code, $out (stdout and stderr), $secs and $childPid.
function Invoke-Script([string]$Script, [string[]]$Arguments = @(), [hashtable]$Environment = @{}) {
    $saved = @{}
    foreach ($k in $Environment.Keys) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        [Environment]::SetEnvironmentVariable($k, $Environment[$k])
    }
    try {
        $o = Join-Path $tmp 'child.out'
        $e = Join-Path $tmp 'child.err'
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process -FilePath $PowerShell -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e `
            -ArgumentList (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$Script`"") + $Arguments)
        $null = $p.Handle
        $p.WaitForExit()
        $script:secs = $watch.Elapsed.TotalSeconds
        $script:code = $p.ExitCode
        $script:childPid = $p.Id
        $script:out = [string](Get-Content $o -Raw) + [string](Get-Content $e -Raw)
    } finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}
function Assert-Code([string]$What, [int]$Expected) { Assert-That ($code -eq $Expected) "$What`: exit $code, expected $Expected`:`n$out" }
function Assert-Says([string]$What, [string]$Text) { Assert-That ($out.Contains($Text)) "$What`: no '$Text' in:`n$out" }

# Whether a guard here would refuse for a process the selftest did not start: then the cases
# that need the guard to pass are skipped, saying so.
$othersOnCard = @(Get-Process | Where-Object {
    $_.ProcessName -like 'ninfer*' -or $_.ProcessName -eq 'ignis-server' -or $_.ProcessName -eq 'ignis-bench' -or $_.ProcessName -like '*_gpu-*'
}).Count -gt 0
if ($othersOnCard) { Write-Host "  (GPU processes run on this machine: the cases where the guard passes are skipped)" }

try {
    # --- the lock ---------------------------------------------------------------------------
    Write-Host "gpu-lock-selftest: the lock"
    Assert-That ($null -eq (Get-GpuLockHolder)) "no lock yet: a holder is named"
    $first = Start-Sleeper
    $hold = Lock-Gpu -Command 'gpu-profile -Test a_test' -ProcessId $first.Id
    Assert-That ((Get-Field $Lock pid) -eq [string]$first.Id) "acquire: the record names pid $(Get-Field $Lock pid), not $($first.Id)"
    Assert-That ((Get-Field $Lock worktree) -eq $tree) "acquire: the record's worktree is $(Get-Field $Lock worktree)"
    Assert-That ((Get-Field $Lock started) -eq [string]$first.StartTime.ToUniversalTime().Ticks) "acquire: the record's start time is not the process's"
    Assert-That ((Get-Field $Lock command) -eq 'gpu-profile -Test a_test') "acquire: the record's command is $(Get-Field $Lock command)"
    Assert-That ([string](Get-GpuLockHolder).pid -eq [string]$first.Id) "held: the holder is not named"

    $second = Start-Sleeper
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $said = try { Lock-Gpu -Command 'make start' -ProcessId $second.Id | Out-Null; '' } catch { $_.Exception.Message }
    Assert-That ($said -like "GPU lock: the card is held ($Lock) by pid $($first.Id) (powershell) since *, worktree ${tree}: gpu-profile -Test a_test") "a second taker, no wait: '$said'"
    Assert-That ($watch.Elapsed.TotalSeconds -lt 2) "a second taker, no wait: it took $($watch.Elapsed.TotalSeconds)s"
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $said = try { (Lock-Gpu -Command 'gpu-run' -WaitSeconds 2 -ProcessId $second.Id 6>&1) | Out-String } catch { $_.Exception.Message }
    Assert-That ($said -like "GPU lock: the card is held ($Lock) by pid $($first.Id)*") "a wait that runs out: '$said'"
    Assert-That ($watch.Elapsed.TotalSeconds -ge 2) "a wait that runs out: it gave up after $($watch.Elapsed.TotalSeconds)s"
    Unlock-Gpu (Get-GpuLockIdentity $second.Id)
    Assert-That ((Get-Field $Lock pid) -eq [string]$first.Id) "a release by a process that does not hold it freed the lock"
    $again = Lock-Gpu -Command 'again' -ProcessId $first.Id
    Assert-That ($again.command -eq 'gpu-profile -Test a_test') "the holder takes it again: the hold was rewritten"

    # A holder that exits during the wait: the waiter takes it.
    Unlock-Gpu $hold
    $brief = Start-Sleeper 2
    $null = Lock-Gpu -Command 'brief' -ProcessId $brief.Id
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $taken = Lock-Gpu -Command 'gpu-run' -WaitSeconds 20 -ProcessId $second.Id 6>$null
    Assert-That ([string]$taken.pid -eq [string]$second.Id -and $watch.Elapsed.TotalSeconds -lt 20) "a wait that ends with the holder: $($taken | ConvertTo-Json -Compress)"

    # A dead holder, a reused pid (another name, another start time): free.
    Stop-Process -Id $second.Id -Force; $second.WaitForExit()
    Assert-That ($null -eq (Get-GpuLockHolder)) "a dead holder is named"
    $null = Lock-Gpu -Command 'after the dead one' -ProcessId $first.Id
    Assert-That ((Get-Field $Lock pid) -eq [string]$first.Id) "taking a dead holder's lock failed"
    [IO.File]::WriteAllText($Lock, (@{ pid = $first.Id; name = 'another'; started = [string]$first.StartTime.ToUniversalTime().Ticks } | ConvertTo-Json -Compress))
    Assert-That ($null -eq (Get-GpuLockHolder)) "a pid that names another program is held"
    [IO.File]::WriteAllText($Lock, (@{ pid = $first.Id; name = 'powershell'; started = '1' } | ConvertTo-Json -Compress))
    Assert-That ($null -eq (Get-GpuLockHolder)) "a pid that names a process started later is held"

    # Five takers of one stale record at once: exactly one holds it.
    [IO.File]::WriteAllText($Lock, (@{ pid = 999999; name = 'gone'; started = '1' } | ConvertTo-Json -Compress))
    $racers = @(1..5 | ForEach-Object {
        Start-Process -FilePath $PowerShell -WindowStyle Hidden -PassThru -ArgumentList '-NoProfile', '-Command',
            ". '$tree\mk\windows\gpulock.ps1'; try { `$null = Lock-Gpu -Command race; Set-Content ('$state\race.' + `$PID) 'won'; Start-Sleep 6 } catch { Set-Content ('$state\race.' + `$PID) 'lost' }"
    })
    $racers | ForEach-Object { $null = $_.Handle; $_.WaitForExit() }
    $won = @(Get-ChildItem "$state\race.*" | Where-Object { (Get-Content $_.FullName -Raw).Trim() -eq 'won' }).Count
    $lost = @(Get-ChildItem "$state\race.*" | Where-Object { (Get-Content $_.FullName -Raw).Trim() -eq 'lost' }).Count
    Assert-That ($won -eq 1 -and $lost -eq 4) "five takers of a stale record: $won won, $lost lost"
    Reset-State

    # One lock per card.
    $env:IGNIS_GPU_CARD = '1'
    $null = Lock-Gpu -Command 'card 1' -ProcessId $first.Id
    Remove-Item Env:\IGNIS_GPU_CARD
    Assert-That (Test-Path "$tmp\lock\gpu-1.lock") "another card: no gpu-1.lock"
    Remove-Item "$tmp\lock\gpu-1.lock"

    # --- gpu.ps1: status, guard, run --------------------------------------------------------
    Write-Host "gpu-lock-selftest: gpu.ps1"
    Remove-Item $Lock -ErrorAction SilentlyContinue
    $hold = Lock-Gpu -Command 'gpu-profile' -ProcessId $first.Id
    Invoke-Script "$tree\mk\windows\gpu.ps1" @('-Action', 'status')
    Assert-Says "gpu.ps1 status" "GPU lock: held ($Lock) by pid $($first.Id) (powershell)"
    Invoke-Script "$tree\mk\windows\gpu.ps1" @('-Action', 'guard')
    Assert-Code "the guard, the lock held" 1
    Assert-Says "the guard, the lock held" "GPU guard: refused -- GPU lock: held ($Lock) by pid $($first.Id)"
    Invoke-Script "$tree\mk\windows\gpu.ps1" @('-Action', 'guard', '-Holder', $first.Id)
    Assert-That (-not $out.Contains('GPU lock: held')) "the guard, the lock held by its caller: refused for it:`n$out"
    if (-not $othersOnCard) { Assert-Code "the guard, the lock held by its caller" 0 }
    Unlock-Gpu $hold
    Invoke-Script "$tree\mk\windows\gpu.ps1" @('-Action', 'status')
    Assert-Says "gpu.ps1 status, free" "GPU lock: free ($Lock)"
    Invoke-Script "$tree\mk\windows\gpu.ps1" @('-Action', 'guard') @{ STUB_USED = '9000' }
    Assert-Code "the guard, the card's memory held" 1
    Assert-That (-not (Test-Path $Marker)) "the guard left a preflight pass behind"

    $cmd = 'cp "$IGNIS_GPU_LOCK_DIR/gpu-0.lock" "$STUB_DIR/cmd.lock"; exit 7'
    if (-not $othersOnCard) {
        Invoke-Script "$tree\mk\windows\gpu.ps1" @('-Action', 'run', '-WaitSeconds', '5') @{ IGNIS_MK_GPU_CMD = $cmd }
        Assert-Code "gpu.ps1 run" 7
        Assert-That ((Get-Field "$state\cmd.lock" pid) -eq [string]$childPid) "gpu.ps1 run: the command ran without the lock under the run's pid ($childPid): $(Get-Content "$state\cmd.lock" -Raw -ErrorAction SilentlyContinue)"
        Assert-That ((Get-Field "$state\cmd.lock" command) -eq "gpu-run: $cmd") "gpu.ps1 run: the hold does not name the command"
        Assert-That (-not (Test-Path $Lock)) "gpu.ps1 run: the lock outlived the run"
    }
    Reset-State
    $hold = Lock-Gpu -Command 'gpu-profile' -ProcessId $first.Id
    Invoke-Script "$tree\mk\windows\gpu.ps1" @('-Action', 'run', '-WaitSeconds', '1') @{ IGNIS_MK_GPU_CMD = $cmd }
    Assert-Code "gpu.ps1 run, held" 1
    Assert-Says "gpu.ps1 run, held" "GPU lock: waiting up to 1s for the card"
    Assert-Says "gpu.ps1 run, held" "the card is held ($Lock) by pid $($first.Id)"
    Assert-That (-not (Test-Path "$state\cmd.lock")) "gpu.ps1 run, held: the command ran"
    Unlock-Gpu $hold
    Invoke-Script "$tree\mk\windows\gpu.ps1" @('-Action', 'run', '-WaitSeconds', '1') @{ IGNIS_MK_GPU_CMD = $cmd; STUB_USED = '9000' }
    Assert-Code "gpu.ps1 run, the guard refuses" 1
    Assert-That (-not (Test-Path "$state\cmd.lock")) "gpu.ps1 run, the guard refuses: the command ran"
    Assert-That (-not (Test-Path $Lock)) "gpu.ps1 run, the guard refuses: the lock was left behind"

    # --- server.ps1 and devui.ps1: the server holds the lock, a stop or a failed start frees it
    Write-Host "gpu-lock-selftest: server.ps1, devui.ps1"
    $port = Get-FreePort
    $server = @{
        IGNIS_MK_BIN = "$bin\stub-server.exe"; IGNIS_MK_ARGS = "--server-bind 127.0.0.1:$port"
        IGNIS_MK_PID = "$tmp\run\ignis-server.pid"; IGNIS_MK_LOG = "$tmp\run\ignis-server.log"
        IGNIS_MK_URL = "http://127.0.0.1:$port"; IGNIS_MK_READY_TIMEOUT = '10'; IGNIS_MK_GPU_LOCK = '1'
    }
    Invoke-Script "$tree\mk\windows\server.ps1" @('-Action', 'start') $server
    Assert-Code "start" 0
    $spid = (Get-Content "$state\server.pid" -Raw -ErrorAction SilentlyContinue)
    Assert-That ($spid -and (Get-Field "$state\lock-seen.json" pid) -eq $spid.Trim()) "start: the server saw the lock name pid $(Get-Field "$state\lock-seen.json" pid), not its own ($spid)"
    Assert-That ((Get-Field $Lock name) -eq 'stub-server') "start: the record names $(Get-Field $Lock name)"
    Invoke-Script "$tree\mk\windows\server.ps1" @('-Action', 'stop') $server
    Assert-Code "stop" 0
    Assert-That (-not (Test-Path $Lock)) "stop: the lock is still there: $(Get-Content $Lock -Raw -ErrorAction SilentlyContinue)"
    Assert-That (-not (Get-Process -Id $spid -ErrorAction SilentlyContinue)) "stop: the server still runs"

    Reset-State
    $hold = Lock-Gpu -Command 'gpu-profile' -ProcessId $first.Id
    Invoke-Script "$tree\mk\windows\server.ps1" @('-Action', 'start') $server
    Assert-Code "start, the lock held" 1
    Assert-Says "start, the lock held" "the card is held ($Lock) by pid $($first.Id)"
    Assert-That (-not $out.Contains('GPU lock: waiting')) "start, the lock held: it waited"
    Assert-That (-not (Test-Path "$state\server.pid")) "start, the lock held: the server was started"
    Invoke-Script "$tree\mk\windows\devui.ps1" @() $server
    Assert-Code "run-ui, the lock held" 1
    Assert-Says "run-ui, the lock held" "the card is held ($Lock) by pid $($first.Id)"
    Assert-That (-not (Test-Path "$state\server.pid")) "run-ui, the lock held: the server was started"
    Unlock-Gpu $hold

    Invoke-Script "$tree\mk\windows\server.ps1" @('-Action', 'start') (Add-Setting $server STUB_SERVER 'exit-early')
    Assert-Code "start, the server exits early" 1
    Assert-That (-not (Test-Path $Lock)) "start, the server exits early: the lock is still there: $(Get-Content $Lock -Raw -ErrorAction SilentlyContinue)"

    Reset-State
    Invoke-Script "$tree\mk\windows\server.ps1" @('-Action', 'run') (Add-Setting $server STUB_SERVER 'run')
    Assert-Code "run" 3
    $spid = (Get-Content "$state\server.pid" -Raw -ErrorAction SilentlyContinue)
    Assert-That ($spid -and (Get-Field "$state\lock-seen.json" pid) -eq $spid.Trim()) "run: the server saw the lock name pid $(Get-Field "$state\lock-seen.json" pid), not its own ($spid)"
    Assert-That ((Get-Field "$state\lock-seen.json" command) -like 'make run: *') "run: the hold does not say make run"
    Assert-That (-not (Test-Path $Lock)) "run: the lock outlived the server"

    if (Get-Command node -ErrorAction SilentlyContinue) {
        # Vite's stand-in is node on an empty vite.js: it exits at once, which ends the session.
        Reset-State
        Invoke-Script "$tree\mk\windows\devui.ps1" @() $server
        $spid = (Get-Content "$state\server.pid" -Raw -ErrorAction SilentlyContinue)
        Assert-That ($spid -and (Get-Field "$state\lock-seen.json" pid) -eq $spid.Trim()) "run-ui: the server saw the lock name pid $(Get-Field "$state\lock-seen.json" pid), not its own ($spid):`n$out"
        Assert-That (-not (Test-Path $Lock)) "run-ui: the lock outlived the session"
    } else {
        Write-Host "  (no node on PATH: the run-ui session case is skipped)"
    }

    Reset-State
    Invoke-Script "$tree\mk\windows\server.ps1" @('-Action', 'start') (Add-Setting $server IGNIS_MK_GPU_LOCK '0')
    Assert-Code "start without the lock" 0
    Assert-That (-not (Test-Path $Lock)) "start without the lock took it"
    Invoke-Script "$tree\mk\windows\server.ps1" @('-Action', 'stop') $server

    # --- scripts/gpu-profile.ps1: the lock for the whole run, -Test -------------------------
    Write-Host "gpu-lock-selftest: gpu-profile.ps1"
    $log = "$tmp\calls.log"
    function Invoke-Profile([string[]]$Arguments, [hashtable]$Environment = @{}) {
        Remove-Item $log, "$state\lock-seen.json" -ErrorAction SilentlyContinue
        New-Item -ItemType File -Path $log | Out-Null
        Invoke-Script "$tree\scripts\gpu-profile.ps1" $Arguments (Add-Setting $Environment STUB_LOG $log)
        $script:calls = @(Get-Content $log)
    }
    $on = '| profile=1 marker=yes lock=yes'
    $listing = "cargo test -p ignis-artifact --features cuda --test layer_reference_real -- --list $on"
    if (-not $othersOnCard) {
        Invoke-Profile @()
        Assert-Code "profile, a pass" 0
        Assert-That ($calls.Count -eq 4 -and $calls[0] -eq "kernel -Test $on" -and $calls[1] -eq $listing -and
            $calls[2] -like "cargo test -p ignis-artifact * --ignored --nocapture $on" -and $calls[3] -like "cargo test --workspace * $on") "profile, a pass: the stages are not as expected:`n$($calls -join "`n")"
        Assert-That ((Get-Field "$state\lock-seen.json" pid) -eq [string]$childPid) "profile, a pass: the stages ran without the lock under the profile's pid"
        Assert-That (-not (Test-Path $Lock) -and -not (Test-Path $Marker)) "profile, a pass: the lock or the marker outlived the run"

        Invoke-Profile @('-Test', 'a_gpu_test,b_gpu_test') @{ STUB_LISTED = 'a_gpu_test,b_gpu_test' }
        Assert-Code "profile, two tests" 0
        $named = "cargo test --workspace --exclude ignis-speech --features cuda --no-fail-fast -- --ignored --exact --test-threads=1 --nocapture"
        Assert-That ($calls.Count -eq 3 -and
            $calls[0] -eq "cargo test --workspace --exclude ignis-speech --features cuda -- --list --ignored --exact a_gpu_test b_gpu_test $on" -and
            $calls[1] -eq "$named a_gpu_test $on" -and $calls[2] -eq "$named b_gpu_test $on") "profile, two tests: the calls are not as expected:`n$($calls -join "`n")"
        Assert-Says "profile, two tests" "gpu test a_gpu_test"
        Assert-Says "profile, two tests" "gpu test b_gpu_test"
        Assert-That ((Get-Field "$state\lock-seen.json" command) -eq 'gpu-profile -Test a_gpu_test,b_gpu_test') "profile, two tests: the hold does not name them"
        Assert-That (-not (Test-Path $Lock) -and -not (Test-Path $Marker)) "profile, two tests: the lock or the marker outlived the run"

        Invoke-Profile @('-Test', 'a_gpu_test,b_gpu_test') @{ STUB_LISTED = 'a_gpu_test' }
        Assert-Code "profile, a name that is no #[ignore]d test" 1
        Assert-That ($calls.Count -eq 1) "profile, a name that is no #[ignore]d test: tests ran:`n$($calls -join "`n")"
        Assert-Says "profile, a name that is no #[ignore]d test" "no #[ignore]d test is named 'b_gpu_test'"

        Invoke-Profile @('-Test', 'a_gpu_test,b_gpu_test') @{ STUB_LISTED = 'a_gpu_test,b_gpu_test'; STUB_FAIL_TEST = 'a_gpu_test' }
        Assert-That ($code -ne 0 -and $calls.Count -eq 2) "profile, the first named test fails: exit $code, calls:`n$($calls -join "`n")"
        Assert-That (-not (Test-Path $Lock) -and -not (Test-Path $Marker)) "profile, a failed test: the lock or the marker outlived the run"
    }

    Invoke-Profile @() @{ STUB_USED = '9000' }
    Assert-Code "profile, the preflight refuses" 1
    Assert-That ($calls.Count -eq 0) "profile, the preflight refuses: a stage ran"
    Assert-That (-not (Test-Path $Lock)) "profile, the preflight refuses: the lock was left behind"

    $hold = Lock-Gpu -Command 'make start' -ProcessId $first.Id
    Set-Content -Path $Marker -Value 'another run'
    Invoke-Profile @('-WaitSeconds', '2')
    Assert-Code "profile, the lock held" 1
    Assert-That ($secs -ge 2) "profile, the lock held: it gave up after $secs s"
    Assert-Says "profile, the lock held" "GPU lock: waiting up to 2s for the card, held ($Lock) by pid $($first.Id)"
    Assert-Says "profile, the lock held" "the card is held ($Lock) by pid $($first.Id)"
    Assert-That ($calls.Count -eq 0) "profile, the lock held: a stage ran"
    Assert-That ((Get-Content $Marker -Raw -ErrorAction SilentlyContinue).Trim() -eq 'another run') "profile, the lock held: the marker another run owns was touched"
    Assert-That ((Get-Field $Lock pid) -eq [string]$first.Id) "profile, the lock held: it freed a lock it never held"
    Remove-Item $Marker
    Unlock-Gpu $hold
} finally {
    foreach ($p in $sleepers) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    if (Test-Path "$tmp\run\ignis-server.pid") {
        $left = Get-Process -Id ([int](Get-Content "$tmp\run\ignis-server.pid" -Raw)) -ErrorAction SilentlyContinue
        if ($left -and $left.ProcessName -eq 'stub-server') { Stop-Process -Id $left.Id -Force }
    }
    Get-Process -Name 'stub-server' -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$tmp*" } | Stop-Process -Force
    Start-Sleep -Milliseconds 500
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

if ($script:failed -gt 0) {
    Write-Host "gpu-lock-selftest (Windows): $($script:failed) failure(s)"
    exit 1
}
Write-Host "gpu-lock-selftest (Windows): ok"
