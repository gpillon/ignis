# `make run-ui` / `make dev-ui` on Windows (the DEV_UI hook, mk/os/windows.mk):
# ignis-server plus the Playground's Vite dev server (hot reload, /v1 proxied
# to the server) as ONE foreground session. Inputs: the IGNIS_MK_* environment
# (common.ps1).
#
# Everything launched here dies with the session:
#   - both children share this console, so Ctrl+C reaches all three at once:
#     the server starts its graceful shutdown, Vite exits, and the finally
#     block below waits for them (then kills what is left);
#   - both children sit in a kill-on-close job object, so if this script is
#     killed instead of interrupted (make terminating its children, a closed
#     terminal window), Windows kills them anyway;
#   - if either child exits on its own, the other is stopped too.
#
# Vite runs as `node vite.js`, not through npm.cmd: a batch file answers
# Ctrl+C with "Terminate batch job (Y/N)?" and would hold the session open.

. (Join-Path $PSScriptRoot 'common.ps1')

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ViteJs = Join-Path $Repo 'web\node_modules\vite\bin\vite.js'
if (-not (Test-Path $ViteJs)) {
    Write-Host "error: $ViteJs not found (make web-install)"
    exit 1
}
$Node = (Get-Command node -ErrorAction SilentlyContinue).Source
if (-not $Node) {
    Write-Host "error: node not found on PATH"
    exit 1
}

Assert-NoServer
$job = New-KillOnCloseJob
$server = $null
$vite = $null
# Set once the session ends by itself; still $false in the finally block
# means Ctrl+C, and the children are already shutting down.
$settled = $false
$code = 0

try {
    $server = Start-IgnisServer -SameConsole
    Add-ProcessToJob $job $server
    if ((Wait-IgnisReady $server) -ne 'ready') {
        $settled = $true
        $code = 1
    }
}  finally {
    if ($settled) {
        Stop-Gently $server 0 'ignis-server'
        Remove-Item $PidFile -ErrorAction SilentlyContinue
    }
}
if ($settled) { exit $code }

try {
    $env:IGNIS_URL = $Url
    $vite = Start-Process -FilePath $Node -ArgumentList "`"$ViteJs`"" `
        -WorkingDirectory (Join-Path $Repo 'web') -NoNewWindow -PassThru
    $null = $vite.Handle
    Add-ProcessToJob $job $vite
    Write-Host ""
    Write-Host "Playground (Vite, hot reload) -> $Url ; server log $LogFile"
    Write-Host "Ctrl+C stops the server and Vite together"
    Write-Host ""

    while ($true) {
        if ($vite.HasExited) {
            Write-Host "vite exited (code $($vite.ExitCode)); stopping the server"
            $code = $vite.ExitCode
            break
        }
        if ($server.HasExited) {
            Write-Host "ignis-server exited (code $($server.ExitCode)); stopping vite"
            Show-LogTail
            $code = 1
            break
        }
        Start-Sleep -Milliseconds 250
    }
    $settled = $true
} finally {
    $grace = if ($settled) { 0 } else { 20 }
    if (-not $settled) { Write-Host "`nCtrl+C: stopping vite and ignis-server" }
    Stop-Gently $vite ([Math]::Min($grace, 5)) 'vite'
    Stop-Gently $server $grace 'ignis-server'
    Remove-Item $PidFile -ErrorAction SilentlyContinue
}
exit $code
