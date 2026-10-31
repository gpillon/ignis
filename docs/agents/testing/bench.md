# Benches and gate runs

## The G2 measurement instrument (P2-05, GitHub #87)

`ignis-bench ttft` measures time to first token at an **exact** prompt
length against any OpenAI-compatible endpoint, and `ignis-bench g2` turns
two such records into the G2 verdict (ADR 0015). Both engines are measured
by this one instrument; only the runs themselves need the GPU.

Every sample — the warmup included — gets its own deterministically
generated prompt, distinct from the first content token, from the other
cells' prompts too ([every prompt of a run is its own](#every-prompt-of-a-run-is-its-own-github-426-443)),
and reads the engine's own computed-prefill-token count back to prove the
prefix was cold. A sample whose computed prefill is not the prompt's own length is
**void** and fails its cell, and `g2` refuses a verdict over a record with
a void sample, a missing cell, or a session that does not match its
counterpart's.

The gate run (GitHub #88) needs the GPU exclusively and the reference
stopped and restarted around it (ADR 0006). One session, one shared
`--session` value:

```powershell
$Session = "g2-$(Get-Date -Format yyyyMMddTHHmmssZ)"
$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"

# 1. The reference, in the owner's production profile (hq-e8-2b KV, 1024
#    prefill chunk, CUDA graphs) -- the bar actually experienced, measured
#    as it is actually run (ADR 0015).
cargo run -p ignis-bench -- ttft `
  --endpoint http://127.0.0.1:8080 --artifact $Artifact `
  --cells 8192,32768 --label reference --profile "hq-e8-2b KV, 1024 chunk, graphs" `
  --session $Session --out .scratch/g2-reference.json

# 2. Stop the reference (it owns the GPU), start ignis-server on the same
#    artifact, then measure it on the same cells in the same session.
cargo run -p ignis-bench -- ttft `
  --endpoint http://127.0.0.1:8000 --artifact $Artifact `
  --cells 8192,32768 --label ignis --profile "BF16 KV, 1024 chunk, eager" `
  --session $Session --out .scratch/g2-ignis.json

# 3. The verdict. `--note` records what is *not* equal between the two
#    engines, rather than correcting for it.
cargo run -p ignis-bench -- g2 `
  --ours .scratch/g2-ignis.json --ref .scratch/g2-reference.json `
  --note "KV format differs: ignis BF16, reference hq-e8-2b" `
  --out .scratch/g2-verdict.json

# 4. Restart the reference.
```

Each ignis leg of a bench or a gate runs with the loop guard off, and takes the
engine shape it needs as load options; both rules are in
[testing.md](../testing.md#gpu-tests-that-compare-output-or-load-an-option).

## The multimodal TTFT cell (GitHub #181)

`ignis-bench ttft --image <png>` adds one cell to a TTFT record: a fixed image
plus a short question (`--image-question`, default "Describe what this
screenshot shows in one sentence."), sent as content parts — the text first,
then the image as a `data:` URI. An error detector for the whole vision path,
not a gate. With `--image`, `--cells` may be omitted.

The committed image is `crates/bench/tests/fixtures/vision_ttft/screenshot.png`,
a 1280x800 editor screenshot drawn by `tools/vision-ttft/screenshot.py`
(deterministic; re-running it is a no-op). It sits on the 32-pixel grid, so it
expands to 1,000 vision tokens and the cell is 1,027 tokens with the default
question — `crates/bench/tests/ttft_image_real_frontend.rs` pins that against
the real frontend, and the reference counts it the same.

Cold in everything, like a text cell: each prompt (the warmup's too) starts
with its own nonce, and its image is the fixture with the top-left pixel
changed, so neither a prefix cache nor either engine's digest-keyed
preprocessing cache can serve it; the size, and so the length, is unchanged.
The record names the image (SHA-256, size, question), and `g2` refuses to pair
cells of one length over different images, or an image cell with a text cell.

A record carrying the image cell cannot be compared with one that has none, so
the image cell is measured as its own record. The session is
`scripts/vision-ttft-session.sh <out-dir>`, run from the repository root on a
free card after a release build: the reference with `--vision` (the owner's
hq-e8-2b-262k preset), ignis with `--vision-enabled`, and ignis without it, **each
launched twice** (ADR 0021). Every launch writes a text record (1024, 8192 and
32768 tokens, cut from the ninfer corpus) and, with vision, an image record;
`scripts/ttft-pool.py` pools each engine's two launches into one record, and
`g2` compares the pooled records: image against image, text against text, and
ignis's text with `--vision-enabled` against without. The 1.5 threshold `g2` prints is
G2's own; for this cell the ratios are error detectors.

**Measure a launch once.** The prompts are fixed — the same nonces, the same
changed pixels — so a second `ttft`, `g3` or `g5` run against a live engine is
served from its prefix and media caches. What prompt reuse serves, ignis reports as
cached tokens (#284), and those samples come back void; a media-cache hit counts
none, and nothing catches it. Restart the engine instead.

**Watch the card's headroom.** Both ignis loads at 262,144 context with
DFlash2 took ~29.8 GiB of their own when measured, before #215 (retained
state was then sized from what was free after load), so the desktop's own VRAM decides whether the card
oversubscribes; when it does, every cell comes back slower and erratic rather
than failing. The script samples `nvidia-smi` beside every launch — distrust
one whose peak reaches the total
([finding](../../findings/2026-09-16-vision-ttft-live-live.md)).

## The KV-format A/B: 2x2 over engine and format (GitHub #139)

What it answers: how much the hq-e8-2b KV format costs against BF16, whether
that cost is ours or the format's, and what the card is actually doing while it
is paid. The result of the 2026-09-13 run is
[`docs/findings/2026-09-13-hq-vs-bf16-decode-cost.md`](../../findings/2026-09-13-hq-vs-bf16-decode-cost.md);
this is how to redo it.

**The one rule that makes it a comparison.** Give every leg the *same KV token
capacity*. A BF16 token costs 65,536 bytes against hq-e8-2b's 9,216 (7.11x, see
[hq-e8-2b KV capacity](../../findings/2026-09-11-hq-e8-2b-kv-capacity.md)), so the
same byte budget buys 7.11x more pages under hq and a "just change the format"
run confounds the format with its page count. 65,536 tokens is the value to
pick: it is 1,024 pages, the geometry spec 03's ITL table and #114's cap
arithmetic assume, and the ITL fixture's peak of 65,344 tokens fits it with 192
to spare. Under hq that means asking for the smaller pool explicitly
(`--vram-kv-pool-bytes 576M`), because its default 4 GiB buys 465,984 tokens.

For the record: shrinking the hq pool from 7,281 pages to 1,024 moved ITL p50 by
0.04% and p95 by 0.05%, so the page count is not itself a performance term —
but that is a measured result, not an assumption to skip the matching on.

Needs the GPU exclusively (ADR 0006) and `ninfer-serve` stopped for the ignis
legs. Four runs, ~4 minutes each. Launch the legs with the explicit commands
below: `make start` brings up the G5 configuration (262144 context, DFlash2),
a different engine from these legs.

```powershell
$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"
$Corpus   = "F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids"
cargo build --release -p ignis-server -p ignis-bench --features cuda

# Per leg: start the telemetry sampler, start the engine, run the cell, stop both.
# Stop the sampler as soon as ignis-bench exits -- see the pitfall below.
nvidia-smi --format=csv,noheader -l 1 `
  --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu `
  > .scratch/gpu-<leg>.csv

# ignis, hq-e8-2b / BF16 (--model-kv-format bf16 --vram-kv-pool-bytes 4G: 65,536 BF16
# tokens; with no --vram-kv-pool-bytes the pool is the rest of the VRAM budget)
.\target\x86_64-pc-windows-msvc\release\ignis-server.exe --model-artifact $Artifact `
  --server-bind 127.0.0.1:8000 --model-kv-format hq-e8-2b --model-max-context 40960 `
  --vram-kv-pool-bytes 576M --model-prefill-chunk 1024 --server-request-timeout 900 `
  --model-idle-unload 0 --model-loop-guard off

# the reference, same two formats
F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe $Artifact `
  --host 127.0.0.1 --port 8080 --kv-dtype hq-e8-2b --max-context 40960 `
  --kv-capacity 65536 --max-concurrency 8 --prefill-chunk 1024

# the cell, once per leg (port 8000 for ignis, 8080 for the reference)
.\target\x86_64-pc-windows-msvc\release\ignis-bench.exe g3 `
  --endpoint http://127.0.0.1:8000 --artifact $Artifact --corpus $Corpus `
  --label ignis --profile "hq-e8-2b KV, 1024 chunk, max-context 40960, 65536-token pool" `
  --session "kvab-<leg>-$(Get-Date -Format yyyyMMddTHHmmssZ)" `
  --out .scratch/g3-<leg>.json
```

Each engine logs the capacity it resolved — `ignis.runtime.kv_pool`'s
`page_count`, and ninfer's `KV capacity explicit resolved=... tokens pages=...`.
Check both say 1,024 pages / 65,536 tokens before believing the leg.

Then reduce each capture over the ITL cell, which is the last cell a `g3` run
measures and whose length the record itself states
(`window_end_ms - window_start_ms`):

```powershell
python scripts/gpu-telemetry-summary.py .scratch/gpu-<leg>.csv --label "<leg> ITL" --last 90
```

**Pitfall: a sampler left running past the cell dilutes the window.** Idle
samples pull the means down hard — the same hq leg reads 96% util / 306 W over
its ITL cell and 83% / 274 W if eleven seconds of idle tail are included. Stop
the sampler when the bench exits, or pass `--from`/`--to` instead of `--last`.

**Pitfall: a cold machine is not a measurement.** The first `g3` run of a
session came in 14% under the records committed the day before and converged on
them over four runs (C=1 46.6 -> 51.3 -> 51.0 -> 54.2 against 55.6). Discard the
first leg, or run one throwaway before the matrix.

**This is not live/live and cannot decide a gate.** Each leg is its own session,
so `g3-gate` will refuse to build a verdict from these records, by design (ADR
0015, ADR 0021). The A/B answers a question about *where time goes*; a verdict
needs both engines measured back to back under one `--session`.

## G3 against ignis (GitHub #426)

The ITL decode lanes send each engine the one EOS control it is asked by:
`ignore_eos` to an engine whose `GET /v1/models` names ignis as the owner
(`owned_by`), since ignis refuses a non-empty `logit_bias` (#288), and
`logit_bias` to the reference, which accepts it without applying it
([finding](../../findings/2026-09-13-itl-lane-terminator-race.md)). The run says
which on stderr.

## Every prompt of a run is its own (GitHub #426, #443)

A `ttft`, `g3` or `g5` run generates all its prompts before its first request,
numbered across the run: `ttft`'s a set per cell in `--cells` order, the image
cell's last (`small-card`'s TTFT cells are such a run, of the cells that fit);
`g3`'s C=1, C=4, the ITL lanes, then the ITL prefillers; `g5`'s one per depth.
Each prompt starts with its own nonce and seed or, cut from a corpus,
its own window, the run's windows spread evenly over the bank (a `g5` depth
opens in its own third of it); a window that starts with another prompt's
content is passed over for the next rotation. Then the run is proven: no two
prompts share more than the chat template's own tokens, its header and footer
together, nor a whole KV page of them, or it refuses to start, every cell
failing with an error that names the two. That bound is #426's, looser than the
header alone, and harmless: a head short of a whole 64-token page is nothing
prompt reuse can serve. So prompt reuse, on by default, has nothing an earlier
cell left behind to serve a later one from. A one-cell `ttft` run sends the
prompts it always did. The prompts are still fixed:
[measure a launch once](#the-multimodal-ttft-cell-github-181).

The nonces cycle through 26 letters, so a filler run of more than 26 prompts
(three cells at `--samples 8`) starts two of them with the same letter. Where
the tokenizer makes that letter a token of its own, a cell's own check refuses
two such prompts of one cell; the run's check admits two of different cells,
whose shared head stays within that bound.

## The harness's own request deadline (GitHub #138)

Every `ignis-bench` subcommand drives its endpoint through one
`HttpEndpoint`, which sets an **explicit** per-request deadline: 1800 s by
default, and `IGNIS_BENCH_REQUEST_TIMEOUT=<seconds>` overrides it (`0`
removes it entirely). Each run prints the deadline it is using on stderr
before its first request.

This is not a knob a normal run needs — it exists because the *implicit*
one cost a gate session. `reqwest::blocking::Client::new()` carries an
undeclared 30-second total timeout that covers reading the response body,
so a streamed measurement request lives entirely inside it. The G4 needle
cell at 131,072 tokens (~30 s of prefill on this hardware) died on it
against **both** engines, every attempt, reporting `read SSE: error
decoding response body` — which reads like an engine fault and was chased
as one across four launches. The 65,536-token cell, at ~15 s, passed every
time.

Two consequences for anyone measuring here: the deadline is now the
harness's own declared number rather than a library default, and a request
that does hit it says so by name instead of surfacing as a decode error.
Raise it (or set `0`) for a cell whose legitimate wall time could approach
half an hour; never lower it to "make a run finish".
