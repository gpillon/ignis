# Shared helpers for the Makefile's Windows server scripts (server.ps1,
# devui.ps1). Dot-source it. Inputs come from the environment so the server
# flag string crosses sh and PowerShell without being re-quoted:
#
#   IGNIS_MK_BIN            the server executable
#   IGNIS_MK_ARGS           its flags, one string
#   IGNIS_MK_PID            pid file
#   IGNIS_MK_LOG            stdout log (stderr goes to <log>.err)
#   IGNIS_MK_URL            base URL probed for readiness (/v1/models)
#   IGNIS_MK_READY_TIMEOUT  seconds to wait for readiness
#   IGNIS_MK_FORCE          "1": stop also kills an ignis-server make did not start

$ErrorActionPreference = 'Stop'

$Bin = $env:IGNIS_MK_BIN
$ServerArgs = $env:IGNIS_MK_ARGS
$PidFile = $env:IGNIS_MK_PID
$LogFile = $env:IGNIS_MK_LOG
$Url = $env:IGNIS_MK_URL
$Timeout = if ($env:IGNIS_MK_READY_TIMEOUT) { [int]$env:IGNIS_MK_READY_TIMEOUT } else { 600 }
$Force = $env:IGNIS_MK_FORCE -eq '1'

# The server make started, if its pid file still names a live ignis-server.
function Get-ManagedServer {
    if (-not (Test-Path $PidFile)) { return $null }
    $id = 0
    if (-not [int]::TryParse((Get-Content $PidFile -Raw).Trim(), [ref]$id)) { return $null }
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq 'ignis-server') { return $p }
    return $null
}

function Test-Ready {
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "$Url/v1/models" | Out-Null
        return $true
    } catch {
        # A 401 is a server that is up and wants an API key (--api-key).
        $response = $_.Exception.Response
        return ($null -ne $response -and [int]$response.StatusCode -eq 401)
    }
}

# `--api-key auto` and `--expose`: the server prints the key it generated
# and its public URL once, into its stdout log; repeat them on the console.
function Show-GeneratedKey {
    if (-not (Test-Path $LogFile)) { return }
    $lines = @()
    $key = Select-String -Path $LogFile -Pattern 'generated API key: (\S+)' | Select-Object -Last 1
    if ($key) { $lines += "API key (generated for this run): $($key.Matches[0].Groups[1].Value)" }
    $url = Select-String -Path $LogFile -Pattern 'public URL: (\S+)' | Select-Object -Last 1
    if ($url) { $lines += "Public URL: $($url.Matches[0].Groups[1].Value)" }
    $ui = Select-String -Path $LogFile -Pattern 'Playground: (\S+)' | Select-Object -Last 1
    if ($ui) { $lines += "Public Playground: $($ui.Matches[0].Groups[1].Value)" }
    if ($lines.Count -gt 0) {
        Write-Host ""
        $lines | ForEach-Object { Write-Host $_ }
        Write-Host ""
    }
}

function Show-LogTail([int]$Lines = 40) {
    foreach ($f in @($LogFile, "$LogFile.err")) {
        if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) {
            Write-Host "--- $f (last $Lines lines)"
            Get-Content $f -Tail $Lines | ForEach-Object { Write-Host $_ }
        }
    }
}

# Exit 1 unless no ignis-server is running at all: a second one would lose
# the port (and, on the GPU, the card).
function Assert-NoServer {
    $managed = Get-ManagedServer
    if ($managed) {
        Write-Host "error: ignis-server is already running (pid $($managed.Id)). make stop, or make restart"
        exit 1
    }
    $stray = @(Get-Process -Name 'ignis-server' -ErrorAction SilentlyContinue)
    if ($stray.Count -gt 0) {
        $ids = ($stray | ForEach-Object { $_.Id }) -join ', '
        Write-Host "error: an ignis-server make did not start is running (pid $ids). make stop FORCE=1 stops it"
        exit 1
    }
    if (-not (Test-Path $Bin)) {
        Write-Host "error: $Bin not found (make build)"
        exit 1
    }
}

# Start the server with its output in $LogFile and record its pid.
# -SameConsole shares this console, so a Ctrl+C here reaches the server too
# (its graceful shutdown); without it the server gets a hidden console of
# its own and outlives this script (`make start`).
function Start-IgnisServer([switch]$SameConsole) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PidFile) | Out-Null
    $startArgs = @{
        FilePath               = $Bin
        WorkingDirectory       = (Get-Location).Path
        RedirectStandardOutput = $LogFile
        RedirectStandardError  = "$LogFile.err"
        PassThru               = $true
    }
    if ($SameConsole) { $startArgs.NoNewWindow = $true } else { $startArgs.WindowStyle = 'Hidden' }
    if ($ServerArgs) { $startArgs.ArgumentList = $ServerArgs }
    $p = Start-Process @startArgs
    # Touching the handle keeps ExitCode readable after the process exits.
    $null = $p.Handle
    Set-Content -Path $PidFile -Value $p.Id -Encoding ascii
    Write-Host "started ignis-server pid $($p.Id): $Bin $ServerArgs"
    return $p
}

# 'ready', 'exited' (log tail shown) or 'timeout'.
function Wait-IgnisReady($Process) {
    Write-Host "log $LogFile ; waiting up to ${Timeout}s for $Url/v1/models"
    $clock = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Ready)) {
        if ($Process.HasExited) {
            Write-Host "error: ignis-server exited with code $($Process.ExitCode) before it was ready"
            Show-LogTail
            Remove-Item $PidFile -ErrorAction SilentlyContinue
            return 'exited'
        }
        if ($clock.Elapsed.TotalSeconds -ge $Timeout) {
            Write-Host "error: not ready after ${Timeout}s (pid $($Process.Id))"
            return 'timeout'
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Host ("ready: {0} (pid {1}, {2:N1}s)" -f $Url, $Process.Id, $clock.Elapsed.TotalSeconds)
    Show-GeneratedKey
    return 'ready'
}

# Give a process $GraceSec to finish a shutdown it was already asked for
# (the console's Ctrl+C), then kill it.
function Stop-Gently($Process, [int]$GraceSec, [string]$Name) {
    if (-not $Process -or $Process.HasExited) { return }
    if ($GraceSec -gt 0 -and $Process.WaitForExit($GraceSec * 1000)) {
        Write-Host "$Name (pid $($Process.Id)) stopped"
        return
    }
    Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    $null = $Process.WaitForExit(15000)
    Write-Host "$Name (pid $($Process.Id)) killed"
}

# A Windows job object with KILL_ON_JOB_CLOSE. Its only handle belongs to
# this PowerShell process, so however this script ends -- a Ctrl+C, a closed
# terminal, make killing its children -- Windows kills every process in the
# job (and their children, which inherit it). Nothing launched for a
# foreground session can outlive it.
function New-KillOnCloseJob {
    if (-not ('IgnisMkJob' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class IgnisMkJob {
    [StructLayout(LayoutKind.Sequential)]
    struct BasicLimits {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IoCounters {
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct ExtendedLimits {
        public BasicLimits Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    const int JobObjectExtendedLimitInformation = 9;
    const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr CreateJobObject(IntPtr attributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits info, uint length);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    public static IntPtr Create() {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception();
        ExtendedLimits limits = new ExtendedLimits();
        limits.Basic.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, ref limits,
                (uint)Marshal.SizeOf(typeof(ExtendedLimits))))
            throw new Win32Exception();
        return job;
    }

    public static void Assign(IntPtr job, IntPtr process) {
        if (!AssignProcessToJobObject(job, process)) throw new Win32Exception();
    }
}
'@
    }
    return [IgnisMkJob]::Create()
}

function Add-ProcessToJob($Job, $Process) {
    [IgnisMkJob]::Assign($Job, $Process.Handle)
}
