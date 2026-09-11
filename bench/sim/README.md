# bench/sim — simulated "1 main + N subagents" coding session

`simulate-session.ps1` plays the role of a real coding-agent client (Qwen
Code or similar) for recording a G4 load trace (GitHub #118, runtime spec
04 section 6, gate G4) when standing up a live agent client by hand is too
slow or hard to reproduce. It POSTs real prompts straight at the capture
proxy (`ignis-bench record`, `crates/bench/src/record.rs`) instead of the
reference engine directly, so the proxy records every request as it
normally would.

**This is not a synthetic-fixture generator.** ADR 0015 / ADR 0021 require
a *real* recorded session — the synthetic fixture
(`crates/bench/tests/fixtures/main_plus_10.jsonl`) is explicitly not a
reference (see `bench/traces/README.md`). Every request this script sends
carries real, current repository content: whole files read off disk at
run time and pasted into the prompt, not lorem-ipsum filler. What makes a
recording usable for the gate is that it is real working content — so a
config's `files` and `task` fields must describe whatever you are actually
working on *right now*, not a frozen scenario replayed forever.
`example-session.json` is a worked example (the session that produced the
first real G4 trace), kept as a template to copy and edit — not a fixture
to run unmodified for unrelated work.

## Why staggered-but-concurrent

A real coding agent launches subagents one after another, a few seconds
apart, not in lockstep — but the subagents then run *concurrently*, and
that overlap (fighting over the reference's single global prefill lane and
its decode lanes) is exactly what the gate's per-class TTFT/tok-s cells
measure. `simulate-session.ps1` reproduces that shape: each request is
launched in its own background job `-StaggerSeconds` after the previous
one (default 2s), but nothing waits for a prior request to *finish* before
the next one launches — the overlap is real, not simulated after the
fact.

## Prerequisites

This script only fires the load — it manages neither the reference engine
nor the capture proxy.

1. Start the reference engine (ninfer) per its own docs.
2. Start the capture proxy:
   ```
   ignis-bench record --listen 127.0.0.1:8090 --target http://127.0.0.1:8080 \
     --out bench/traces/<load>-trace.jsonl --class first-is-main
   ```

## Usage

```powershell
powershell -File bench/sim/simulate-session.ps1 `
  -ProxyUrl http://127.0.0.1:8090 `
  -Config bench/sim/example-session.json `
  -RepoRoot F:/ai/opencode/inference
```

`POST /v1/session/end` is sent automatically once every request finishes;
the proxy prints the recorded trace's path and summary. From there, follow
`bench/traces/README.md`: replay the trace, compute its SHA-256 + shape,
and commit only that summary (never the trace itself — it contains real
working content).

## Config shape

```jsonc
{
  "model": "<the model id the proxy's target engine reports>",
  "staggerSeconds": 2,
  "requests": [
    {
      "name": "main",
      "maxTokens": 16000,
      "tools": true,               // optional, default true
      "system": "<system prompt text>",
      "files": ["crates/core/src/admission.rs", "..."],
      "task": "<the user's task text, appended after the pasted files>"
    }
  ]
}
```

- `files` are resolved relative to `-RepoRoot` and pasted in full as
  fenced code blocks ahead of `task`. There is no size limit imposed by
  this script (the capture proxy's own body limit is 32 MiB); pasting
  whole files rather than snippets is deliberate — it is what makes the
  prompt-length distribution and the prefix-sharing profile realistic.
- `tools` attaches a small generic `read_file` / `bash` / `edit_file` /
  `spawn_subagent` tool set (OpenAI tool-calling, GitHub #132) so the
  recorded traffic exercises the same request shape a real tool-calling
  agent client sends. It is not per-request customizable on purpose — the
  point of varying requests is the prompt content, not the tool schema.
- The first request in `requests` becomes the trace's `main`; the rest
  become `sub` (the proxy's `first-is-main` class policy). Order your
  config accordingly.

## Known pitfalls (already fixed in this script, noted for the next port)

- **Windows PowerShell 5.1's `Get-Content` without `-Encoding UTF8`**
  misreads a UTF-8-without-BOM source file as the system codepage,
  silently mangling every non-ASCII byte (no error) -- e.g. `§4` becomes
  two mojibake characters. Always pass `-Encoding UTF8` when reading repo
  files.
- **`Set-Content -Encoding utf8` always writes a BOM** in Windows
  PowerShell 5.1. A BOM before `{` is invalid JSON, and the engine's
  `serde_json` parser refuses it with a bare 400 and no useful reason.
  This script writes request bodies with `[IO.File]::WriteAllBytes` +
  `[Text.Encoding]::UTF8.GetBytes(...)` instead, and posts the same bytes
  directly rather than round-tripping the body through a file.
- **A `string` `-Body` on `Invoke-WebRequest`** is not guaranteed to hit
  the wire as UTF-8 -- encode to bytes yourself (as above) whenever the
  body can contain non-ASCII content, which a pasted source file always
  can.

## Picking `maxTokens` and `staggerSeconds`

Bump `maxTokens` and shrink `staggerSeconds` to lengthen how long the
session's requests stay concurrently in flight — the reference's single
global prefill lane and bounded decode-lane concurrency mean a denser,
longer-running load matches a real burst more closely than a handful of
short, barely-overlapping ones. There is no single right value: pick
whatever makes the *session's own overlap window* — the time between the
first launch and the last completion — resemble the real burst you are
trying to capture, and check the proxy's own log (`recorded req-NNN
[class] +N ms`) plus each request's `time_total` to see whether it did.
