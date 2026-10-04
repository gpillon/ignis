# Launches the Flash-Next conversion detached: hidden window, stdout/stderr to files, an exit-code
# sentinel, and the shared GPU lock held under -LockOwner for exactly the life of the process.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/flash-next-converter/launch.ps1 `
#       [-Out F:/ai/models/Qwen3.8-Flash-Next-ignis] [-LockOwner flash-next-convert] `
#       [-StopFile F:/ai/models/Qwen3.8-Flash-Next-ignis/STOP] [-Extra "--layers 2 --table-shards 4"]
#
# Files in -Out: convert.out / convert.err (the process's streams), convert.log (one line per
# layer), convert.exit (the exit code, written when the process ends: 0 done, 75 stopped by the
# stop file, 2 refused, 3 quality FAIL, 1 error). The lock is released after the exit code is written.
param(
    [string]$Out = "F:/ai/models/Qwen3.8-Flash-Next-ignis",
    [string]$LockOwner = "flash-next-convert",
    [string]$StopFile = "",
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
$tool = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = "F:/ai/ngram-venv/Scripts/python.exe"
$bash = "C:/Program Files/Git/bin/bash.exe"
$lock = "F:/ai/opencode/.inference-qwen-worktrees/.swarm/gpu-lock.sh"
$study = "F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03/real"
$windows = "F:/ai/opencode/inference/.scratch/kld-2026-09-24"

New-Item -ItemType Directory -Force $Out | Out-Null
& $bash $lock try $LockOwner "Flash-Next conversion, detached ($Out)"
if ($LASTEXITCODE -ne 0) { Write-Output "GPU lock held by someone else: not launched"; exit 2 }

$arguments = "convert.py run --out `"$Out`" --lock-owner $LockOwner --ood-dir `"$study/ood`" " +
             "--windows-dir `"$windows`" --table-cache `"$study/table_cache`""
if ($StopFile) { $arguments += " --stop-file `"$StopFile`"" }
if ($Extra) { $arguments += " $Extra" }
$exitFile = Join-Path $Out "convert.exit"
Remove-Item -ErrorAction SilentlyContinue $exitFile

$chain = "set PYTHONIOENCODING=utf-8& cd /d `"$tool`"& `"$py`" $arguments > `"$Out/convert.out`" 2> `"$Out/convert.err`"" +
         "& echo !ERRORLEVEL! > `"$exitFile`"& `"$bash`" `"$lock`" release $LockOwner"
$p = Start-Process -WindowStyle Hidden -FilePath "cmd.exe" -ArgumentList "/v:on /s /c `"$chain`"" -PassThru
Write-Output "launched: cmd pid $($p.Id); watch $Out/convert.log, exit code in $exitFile"
