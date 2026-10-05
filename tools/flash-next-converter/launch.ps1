# Launches the Flash-Next conversion detached: a hidden PowerShell runs the converter (streams to
# files, exit code to a sentinel), then, when the pass ends with exit 0 and the packer binary exists,
# the packer and the post-pack verify; the shared GPU lock is held under -LockOwner for the whole
# chain and released at its end, whatever happens.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/flash-next-converter/launch.ps1 `
#       [-Out F:/ai/models/Qwen3.8-Flash-Next-ignis] [-LockOwner flash-next-convert] `
#       [-StopFile <Out>/STOP] [-Extra "..."]
#
# A dry run goes to its own -Out (a work tree belongs to one configuration; the converter refuses
# another): -Out F:/ai/models/fn-dryrun -Extra "--layers 2 --table-shards 4 --ckpt-every 1".
#
# Files in -Out: convert.out / convert.err (the converter's streams), convert.log (one line per
# layer), convert.exit (0 done, 75 stopped by the stop file, 2 refused, 3 an acceptance check
# FAILED, 1 error); after an exit 0: pack.out / pack.exit (the packer) and verify.log /
# verify.json / verify.exit (the container decode check).
param(
    [string]$Out = "F:/ai/models/Qwen3.8-Flash-Next-ignis",
    [string]$LockOwner = "flash-next-convert",
    [string]$StopFile = "",
    [string]$Extra = "",
    [string]$Study = "F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03/real",
    [string]$Windows = "F:/ai/opencode/inference/.scratch/kld-2026-09-24",
    [string]$CkptDir = "E:/flash-next-ckpt",
    [string]$CkptSmallDir = "C:/flash-next-ckpt-small",
    [int]$Prefetch = 0,
    [string]$Python = "F:/ai/ngram-venv/Scripts/python.exe",
    [string]$Packer = "",
    [switch]$Detached
)
$ErrorActionPreference = "Stop"
$tool = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $tool "..\..")
$bash = "C:/Program Files/Git/bin/bash.exe"
$lock = "F:/ai/opencode/.inference-qwen-worktrees/.swarm/gpu-lock.sh"
if (-not $Packer) { $Packer = Join-Path $repo "target/release/ignis-artifact-pack.exe" }
$artifact = Join-Path $Out "qwen3_8_flash_next_trellis_a25-v2.ninfer"

function Invoke-Logged([string]$exe, [string[]]$argv, [string]$stdout, [string]$stderr) {
    $p = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $tool -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    return $p.ExitCode
}

if ($Detached) {
    # the chain itself, in the hidden process; the lock is ours until the finally block
    try {
        $env:PYTHONIOENCODING = "utf-8"
        $argv = @("convert.py", "run", "--out", "`"$Out`"", "--lock-owner", $LockOwner,
                  "--ood-dir", "`"$Study/ood`"", "--windows-dir", "`"$Windows`"",
                  "--table-cache", "`"$Study/table_cache`"", "--ckpt-dir", "`"$CkptDir`"",
                  "--ckpt-small-dir", "`"$CkptSmallDir`"", "--prefetch", "$Prefetch")
        if ($StopFile) { $argv += @("--stop-file", "`"$StopFile`"") }
        if ($Extra) { $argv += $Extra.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries) }
        $code = Invoke-Logged $Python $argv "$Out/convert.out" "$Out/convert.err"
        Set-Content -Path "$Out/convert.exit" -Value $code -Encoding ascii
        if ($code -eq 0 -and (Test-Path $Packer)) {
            $pcode = Invoke-Logged $Packer @("--work", "`"$Out/work`"", "--out", "`"$artifact`"") `
                "$Out/pack.out" "$Out/pack.err"
            Set-Content -Path "$Out/pack.exit" -Value $pcode -Encoding ascii
            if ($pcode -eq 0) {
                $vcode = Invoke-Logged $Python @("convert.py", "verify", "--artifact", "`"$artifact`"",
                                                 "--lock-owner", $LockOwner) "$Out/verify.out" "$Out/verify.err"
                Set-Content -Path "$Out/verify.exit" -Value $vcode -Encoding ascii
            }
        }
    } finally {
        & $bash $lock release $LockOwner | Out-Null
    }
    exit 0
}

New-Item -ItemType Directory -Force $Out | Out-Null
& $bash $lock try $LockOwner "Flash-Next conversion, detached ($Out)"
if ($LASTEXITCODE -ne 0) { Write-Output "GPU lock held by someone else: not launched"; exit 2 }
foreach ($f in @("convert.exit", "pack.exit", "verify.exit")) { Remove-Item -ErrorAction SilentlyContinue (Join-Path $Out $f) }
try {
    $self = $MyInvocation.MyCommand.Path
    $argv = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$self`"", "-Detached",
              "-Out", "`"$Out`"", "-LockOwner", $LockOwner, "-Study", "`"$Study`"", "-Windows", "`"$Windows`"",
              "-CkptDir", "`"$CkptDir`"", "-CkptSmallDir", "`"$CkptSmallDir`"", "-Prefetch", "$Prefetch",
              "-Python", "`"$Python`"", "-Packer", "`"$Packer`"")
    if ($StopFile) { $argv += @("-StopFile", "`"$StopFile`"") }
    if ($Extra) { $argv += @("-Extra", "`"$Extra`"") }
    $p = Start-Process -FilePath "powershell.exe" -ArgumentList $argv -WindowStyle Hidden -PassThru
} catch {
    & $bash $lock release $LockOwner | Out-Null
    Write-Output "launch failed, GPU lock released: $_"
    exit 1
}
Write-Output "launched: pid $($p.Id); watch $Out/convert.log; exit codes in $Out/convert.exit (pack.exit, verify.exit)"
