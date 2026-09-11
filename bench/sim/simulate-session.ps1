<#
Simulates a real "1 main agent + N subagents" coding session against
ignis-bench's capture proxy (crates/bench/src/record.rs), to record a G4
load trace (GitHub #118, runtime spec 04 section 6, gate G4).

Why this exists: the G4 trace must come from a REAL agent session against
the reference stack (ADR 0015 / ADR 0021 -- no synthetic fixture is a
reference; see bench/traces/README.md). Standing up a live coding-agent
client (Qwen Code or similar) for ~10 real subagent turns by hand is slow
and hard to reproduce. This script plays that role directly: it POSTs
real, large (whole-file) prompts straight at the capture proxy, launched a
few seconds apart but left to run concurrently -- real coding tools launch
subagents one after another, not in lockstep, but the requests do overlap
in flight, and that overlap is exactly what the gate's per-class cells
measure. The trace this produces is real, not synthetic.

This is NOT a lorem-ipsum load generator. Every request's content is real
repository text you choose via -Config's `files` lists -- "real working
content" is what ADR 0015 requires, and exactly why the resulting trace is
never committed (only its SHA-256 + shape are; see bench/traces/README.md
and bench/sim/README.md). Write a fresh config describing your actual
current task before recording -- do not reuse an old config's task text
for unrelated work; the files it names may also have moved on.

Prerequisites (this script only fires the load; it manages neither
process):
  1. Start the reference engine (ninfer) per its own docs.
  2. Start the capture proxy:
       ignis-bench record --listen 127.0.0.1:8090 --target http://127.0.0.1:8080 `
         --out bench/traces/<load>-trace.jsonl --class first-is-main

Usage:
  powershell -File bench/sim/simulate-session.ps1 `
    -ProxyUrl http://127.0.0.1:8090 `
    -Config bench/sim/example-session.json `
    -RepoRoot F:/ai/opencode/inference

  POST /v1/session/end is sent automatically once every request finishes;
  the proxy prints the recorded trace's path + summary (see
  bench/traces/README.md for what to do with it next: replay, sha256 +
  shape, gate).

-Config is a JSON file (bench/sim/example-session.json is a full worked
example) shaped:
  {
    "model": "<the model id the proxy's target engine reports>",
    "staggerSeconds": 2,
    "requests": [
      {
        "name": "main",
        "maxTokens": 16000,
        "tools": true,
        "system": "<system prompt text>",
        "files": ["crates/core/src/admission.rs", "..."],
        "task": "<the user's task text, appended after the pasted files>"
      },
      ...
    ]
  }
`files` are resolved relative to -RepoRoot and pasted in full as fenced
code blocks ahead of `task`. `tools` (default true) attaches a small
generic read_file/bash/edit_file/spawn_subagent tool set (OpenAI
tool-calling) so the recorded traffic exercises the same request shape a
real tool-calling agent client sends (see GitHub #132).
#>

param(
    [string]$ProxyUrl = "http://127.0.0.1:8090",
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$RepoRoot = $(try { (git rev-parse --show-toplevel 2>$null) } catch { $null }),
    [string]$OutDir = (Join-Path $env:TEMP "ignis-sim-session")
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    throw "could not resolve -RepoRoot (not inside a git repo?) -- pass it explicitly"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$cfg = Get-Content -Raw -Path $Config | ConvertFrom-Json
$stagger = if ($cfg.PSObject.Properties.Name -contains "staggerSeconds") { $cfg.staggerSeconds } else { 2 }
$model = $cfg.model

# A small generic tool set (OpenAI tool-calling, GitHub #132) so a
# recorded request has the same shape a real tool-calling agent client
# sends. Every entry with `tools: true` (the default) gets this set;
# there is no per-request tool customization here on purpose -- the point
# is the prompt content, not the tool schema.
$tools = @(
    @{ type = "function"; function = @{ name = "read_file"; description = "Read a file from the repository"; parameters = @{ type = "object"; properties = @{ path = @{ type = "string" } }; required = @("path") } } },
    @{ type = "function"; function = @{ name = "bash"; description = "Run a shell command in the repo root"; parameters = @{ type = "object"; properties = @{ command = @{ type = "string" } }; required = @("command") } } },
    @{ type = "function"; function = @{ name = "edit_file"; description = "Apply a string replacement edit to a file"; parameters = @{ type = "object"; properties = @{ path = @{ type = "string" }; old = @{ type = "string" }; new = @{ type = "string" } }; required = @("path", "old", "new") } } },
    @{ type = "function"; function = @{ name = "spawn_subagent"; description = "Delegate a scoped task to a subagent"; parameters = @{ type = "object"; properties = @{ task = @{ type = "string" } }; required = @("task") } } }
)

function Build-Body {
    param($req)

    $userText = New-Object System.Text.StringBuilder
    foreach ($f in $req.files) {
        $full = Join-Path $RepoRoot $f
        if (-not (Test-Path $full)) { throw "config references a missing file: $f" }
        # -Encoding UTF8 is required: PowerShell 5.1's Get-Content default
        # (no -Encoding) misreads a UTF-8-without-BOM source file as the
        # system codepage, mangling every non-ASCII byte (e.g. "§"
        # becomes two mojibake characters) -- silently, no error.
        $content = Get-Content -Raw -Encoding UTF8 -Path $full
        [void]$userText.AppendLine("`n--- $f ---")
        [void]$userText.AppendLine('```')
        [void]$userText.AppendLine($content)
        [void]$userText.AppendLine('```')
    }
    [void]$userText.AppendLine("`n" + $req.task)

    $useTools = if ($req.PSObject.Properties.Name -contains "tools") { $req.tools } else { $true }
    $body = [ordered]@{
        model      = $model
        stream     = $true
        max_tokens = $req.maxTokens
        messages   = @(
            @{ role = "system"; content = $req.system }
            @{ role = "user"; content = $userText.ToString() }
        )
    }
    if ($useTools) {
        $body["tool_choice"] = "auto"
        $body["tools"] = $tools
    }
    return ($body | ConvertTo-Json -Depth 12 -Compress)
}

Write-Host "simulate-session: $($cfg.requests.Count) requests, stagger ${stagger}s, proxy $ProxyUrl"

$jobs = @()
foreach ($req in $cfg.requests) {
    $bodyJson = Build-Body $req
    # Written for audit/debugging only -- the actual request below sends
    # $bodyJson's UTF-8 bytes directly (no round trip through this file),
    # because Set-Content's own "utf8" encoding always prepends a BOM in
    # Windows PowerShell 5.1, and a BOM before "{" is invalid JSON (the
    # engine's serde_json parser refuses it with a bare 400, no reason
    # given -- this bit us during development; see bench/sim/README.md).
    $bodyFile = Join-Path $OutDir "$($req.name).json"
    [System.IO.File]::WriteAllBytes($bodyFile, [System.Text.Encoding]::UTF8.GetBytes($bodyJson))
    $outFile = Join-Path $OutDir "$($req.name).out"

    Write-Host "$(Get-Date -Format 'HH:mm:ss') launching $($req.name) (backgrounded, $($bodyJson.Length) bytes, max_tokens=$($req.maxTokens))"
    $jobs += Start-Job -ScriptBlock {
        param($url, $bodyJson, $outFile)
        # Encode to bytes ourselves and post the byte array: a string
        # -Body left to Invoke-WebRequest's own encoding is not guaranteed
        # to be UTF-8 on the wire, and this prompt's pasted file content is
        # never plain ASCII.
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($bodyJson)
        try {
            $resp = Invoke-WebRequest -Uri "$url/v1/chat/completions" -Method Post -ContentType "application/json" -Body $bytes -UseBasicParsing -TimeoutSec 900
            [System.IO.File]::WriteAllBytes($outFile, [System.Text.Encoding]::UTF8.GetBytes($resp.Content))
            "status=$($resp.StatusCode)"
        }
        catch {
            "error=$($_.Exception.Message)"
        }
    } -ArgumentList $ProxyUrl, $bodyJson, $outFile

    Start-Sleep -Seconds $stagger
}

Write-Host "$(Get-Date -Format 'HH:mm:ss') all $($jobs.Count) launched, waiting for completion..."
$results = $jobs | Wait-Job | Receive-Job
$jobs | Remove-Job
for ($i = 0; $i -lt $cfg.requests.Count; $i++) {
    Write-Host "  [$($cfg.requests[$i].name)] $($results[$i])"
}
Write-Host "$(Get-Date -Format 'HH:mm:ss') all requests finished"

Write-Host "ending session"
$end = Invoke-WebRequest -Uri "$ProxyUrl/v1/session/end" -Method Post -UseBasicParsing
Write-Host $end.Content
