# G5 gate session (GitHub #156). One GPU-exclusive session, one session id.
# Order: GPU profile; ignis spec-off launch (equivalence capture only);
# reference 1, ignis 1, reference 2, ignis 2 (g5 per launch; canary + spec-on
# capture on ignis 2; G4 trace replay last on each engine's launch 2);
# then the equivalence compare and the pooled g5-gate.
param([switch]$SkipProfile)
$ErrorActionPreference = "Stop"
Set-Location F:\ai\opencode\inference

$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"
$Corpus   = "F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids"
$Trace    = "bench\traces\g4-load-trace.jsonl"
$Ref      = "F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe"
$Server   = ".\target\x86_64-pc-windows-msvc\release\ignis-server.exe"
$Bench    = ".\target\x86_64-pc-windows-msvc\release\ignis-bench.exe"
$Out      = ".scratch\g5-run"
$Session  = "g5-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$Session | Set-Content "$Out\session-id.txt"
function Log($m) { $l = "[{0}] {1}" -f (Get-Date -Format o), $m; Write-Host $l; try { Add-Content "$Out\session.log" $l -ErrorAction Stop } catch { Write-Host "(session.log busy: $_)" } }
Log "session $Session"

$RefProfile   = "reference hq-e8-2b, max-context 262144, chunk 1024, kv-capacity 465984, graphs+prefix reuse on, --spec dflash2 --draft-tokens 7"
$IgnisProfile = "ignis hq-e8-2b, max-context 262144, chunk 1024, 4 GiB auto pool, graphs+prefix reuse on, --spec dflash2 --draft-tokens 7"

function Start-Engine($name, $exe, $argv, $port) {
    Log "$name starting: $exe $($argv -join ' ')"
    $p = Start-Process -FilePath $exe -ArgumentList $argv -PassThru -NoNewWindow `
        -RedirectStandardOutput "$Out\$name-server.log" -RedirectStandardError "$Out\$name-server.log.err"
    $deadline = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $deadline) {
        if ($p.HasExited) { throw "$name exited during load (code $($p.ExitCode))" }
        try { Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 5 | Out-Null } catch { Start-Sleep 2; continue }
        Log "$name ready pid=$($p.Id)"
        $kv = Select-String -Path "$Out\$name-server.log*" -Pattern 'ignis.runtime.kv_pool|KV capacity' | Select-Object -First 1
        Log "$name kv: $($kv.Line)"
        return $p
    }
    throw "$name not ready in 5 min"
}
function Stop-Engine($name, $p) {
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
    Start-Sleep 5
    Log "$name stopped"
}
function Bench($name, [string[]]$argv) {
    Log "bench $name : $($argv -join ' ')"
    $ErrorActionPreference = "Continue"   # PS 5.1 wraps native stderr as errors
    & $Bench @argv *> "$Out\$name.log"
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    Log "bench $name exit=$code"
    # A depth record or a capture that failed cannot decide the gate: stop
    # rather than spend the remaining launches (#159).
    if ($code -ne 0 -and $name -match '-g5$|capture$') {
        Get-Process ignis-server, ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
        Log "session aborted: $name exit=$code; engines stopped"
        throw "$name failed (exit $code): session aborted"
    }
}
function Ignis-Args([bool]$spec) {
    $a = @("--artifact", $Artifact, "--bind", "127.0.0.1:8000", "--kv-format", "hq-e8-2b",
           "--max-context", "262144", "--prefill-chunk", "1024", "--request-timeout", "1800")
    if ($spec) { $a += @("--spec", "dflash2", "--draft-tokens", "7") }
    $a
}
$RefArgs = @($Artifact, "--model-id", "qwen3.8-27b-nvfp4full-v2", "--host", "127.0.0.1", "--port", "8080",
             "--kv-dtype", "hq-e8-2b", "--max-context", "262144", "--kv-capacity", "465984",
             "--max-concurrency", "8", "--prefill-chunk", "1024", "--spec", "dflash2", "--draft-tokens", "7")

if (-not $SkipProfile) {
    Log "GPU profile"
    $ErrorActionPreference = "Continue"
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\gpu-profile.ps1 *> "$Out\gpu-profile.log"
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    Log "GPU profile exit=$code"
    if ($code -ne 0) { throw "GPU profile red; the tree under measurement is not green" }
}

# Equivalence, spec-off side: its own launch, before any measured launch.
$p = Start-Engine "ignis-specoff" $Server (Ignis-Args $false) 8000
Bench "ignis-specoff-capture" @("g5-equivalence", "capture", "--endpoint", "http://127.0.0.1:8000", "--artifact", $Artifact, "--side", "spec-off", "--session", $Session, "--out", "$Out\equivalence-spec-off.json")
Stop-Engine "ignis-specoff" $p

foreach ($launch in 1, 2) {
    $p = Start-Engine "ninfer-launch$launch" $Ref $RefArgs 8080
    Bench "ninfer-launch$launch-g5" @("g5", "--endpoint", "http://127.0.0.1:8080", "--artifact", $Artifact, "--corpus", $Corpus,
        "--label", "reference", "--profile", $RefProfile, "--session", $Session, "--out", "$Out\ninfer-launch$launch-g5.json")
    if ($launch -eq 2) {
        Bench "ninfer-launch2-g4trace" @("g4", "--endpoint", "http://127.0.0.1:8080", "--artifact", $Artifact, "--trace", $Trace, "--corpus", $Corpus,
            "--label", "reference", "--profile", $RefProfile, "--session", $Session, "--out", "$Out\ninfer-launch2-g4trace.json")
    }
    Stop-Engine "ninfer-launch$launch" $p

    $p = Start-Engine "ignis-launch$launch" $Server (Ignis-Args $true) 8000
    Bench "ignis-launch$launch-g5" @("g5", "--endpoint", "http://127.0.0.1:8000", "--artifact", $Artifact, "--corpus", $Corpus,
        "--label", "ignis", "--profile", $IgnisProfile, "--session", $Session, "--out", "$Out\ignis-launch$launch-g5.json")
    if ($launch -eq 2) {
        Bench "ignis-launch2-canary" @("canary", "--endpoint", "http://127.0.0.1:8000", "--out", "$Out\ignis-launch2-canary.json")
        Bench "ignis-launch2-capture" @("g5-equivalence", "capture", "--endpoint", "http://127.0.0.1:8000", "--artifact", $Artifact, "--side", "spec-on", "--session", $Session, "--out", "$Out\equivalence-spec-on.json")
        Bench "ignis-launch2-g4trace" @("g4", "--endpoint", "http://127.0.0.1:8000", "--artifact", $Artifact, "--trace", $Trace, "--corpus", $Corpus,
            "--label", "ignis", "--profile", $IgnisProfile, "--session", $Session, "--out", "$Out\ignis-launch2-g4trace.json")
    }
    Stop-Engine "ignis-launch$launch" $p
}

Bench "equivalence" @("g5-equivalence", "--spec-on-capture", "$Out\equivalence-spec-on.json", "--spec-off-capture", "$Out\equivalence-spec-off.json", "--out", "$Out\equivalence.json")
Bench "g5-gate" @("g5-gate", "--ours", "$Out\ignis-launch1-g5.json", "--ours", "$Out\ignis-launch2-g5.json",
    "--ref", "$Out\ninfer-launch1-g5.json", "--ref", "$Out\ninfer-launch2-g5.json",
    "--note", "session $Session, #156", "--out", "$Out\g5-verdict.json")
Log "session done"
