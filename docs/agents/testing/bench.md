# Benches

`ignis-bench` measures an engine through its OpenAI-compatible endpoint:
ignis, or another engine beside it. No bench decides a gate: an artifact or a
route is admitted by ADR 0082's legs, and the phase gates against the
reference engine, with the ratios `g2`, `g3-gate`, `g4-gate` and `g5-gate`
still print (1.5, 99%), are past work. What is below is what runs today.

## The measuring rule

A number that compares two engines, or two builds of ignis, holds only when
both sides were measured in one session, on cold-prefix samples, each pooled
over at least two independent launches. The rule and its reasons are in the
opening comment of `crates/bench/src/lib.rs`; in a run it means:

- **One session.** Both sides run back to back, on the same machine, under
  one `--session` value. `g2`, `g3-gate`, `g4-gate` and `g5-gate` refuse
  records of different sessions, so a record from another day is never one
  side of a comparison.
- **Cold prefixes.** Nothing to do: `ttft`, `g3` and `g5` generate a prompt
  of their own for every sample and void a sample the engine did not compute
  whole. [Measure a launch once](#the-multimodal-ttft-cell-github-181).
- **Two launches a side.** Stop the server and start it again between the
  launches: one launch picks its own throughput band, up to 17% from another
  launch of the same build
  ([finding](../../findings/2026-09-10-one-process-launch-picks-its-own-throughput-band.md)).
  `g4-gate` and `g5-gate` take every launch (`--ours` and `--ref`, repeated)
  and refuse fewer than two a side; `ttft-pool` pools one side's `ttft`
  records for `g2`; `g3-gate` takes one record a side, so run it once per
  pair of launches and read the two together.
- **Each engine as it is served.** Give `--profile` what the engine was
  launched with, and `--note` what is unequal between the sides: it is
  recorded beside the result, never corrected for.

One process holds the card at a time (ADR 0006,
[who holds the card](../testing.md#who-holds-the-card)), so the sides run one
after the other. Each ignis leg of a bench runs with the loop guard off, and
takes the engine shape it needs as load options; both rules are in
[testing.md](../testing.md#gpu-tests-that-compare-output-or-load-an-option).

## TTFT: `ttft`, `ttft-pool` and `g2` (GitHub #87)

`ignis-bench ttft` measures time to first token at an **exact** prompt
length against any OpenAI-compatible endpoint, `ttft-pool` pools one side's
launches into one record, and `ignis-bench g2` compares two records: the
ratio of the medians per cell.

Every sample — the warmup included — gets its own deterministically
generated prompt, distinct from the first content token, from the other
cells' prompts too ([every prompt of a run is its own](#every-prompt-of-a-run-is-its-own-github-426-443)),
and reads the engine's own computed-prefill-token count back to prove the
prefix was cold. A sample whose computed prefill is not the prompt's own length is
**void** and fails its cell, and `g2` refuses a verdict over a record with
a void sample, a missing cell, or a session that does not match its
counterpart's.

A comparison of two builds of ignis, A and B, in one session:

```powershell
$Session = "ttft-$(Get-Date -Format yyyyMMddTHHmmssZ)"
$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"

# 1. Per side, twice: start the server (make gpu-run, below), measure, stop it.
cargo run -p ignis-bench -- ttft `
  --endpoint http://127.0.0.1:8000 --artifact $Artifact `
  --cells 8192,32768 --label a --profile "hq-e8-2b KV, 1024 chunk" `
  --session $Session --out .scratch/ttft-a-1.json

# 2. Pool each side's launches.
cargo run -p ignis-bench -- ttft-pool `
  --launch .scratch/ttft-a-1.json --launch .scratch/ttft-a-2.json `
  --out .scratch/ttft-a.json

# 3. Compare. `--note` records what is *not* equal between the two sides,
#    rather than correcting for it.
cargo run -p ignis-bench -- g2 `
  --ours .scratch/ttft-b.json --ref .scratch/ttft-a.json `
  --out .scratch/ttft-b-vs-a.json
```

Another engine is a side like any other: its own endpoint, label and profile.
`ttft-pool` refuses launches of different sessions, labels, engines, profiles,
`--max-tokens` or cells, one launch given twice (the same `date`), a record
already pooled, and a launch with a failed cell.

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
the real frontend.

Cold in everything, like a text cell: each prompt (the warmup's too) starts
with its own nonce, and its image is the fixture with the top-left pixel
changed, so neither a prefix cache nor an engine's digest-keyed
preprocessing cache can serve it; the size, and so the length, is unchanged.
The record names the image (SHA-256, size, question), and `g2` refuses to pair
cells of one length over different images, or an image cell with a text cell.

A record carrying the image cell cannot be compared with one that has none, so
the image cell is measured as its own record. The session is
`scripts/vision-ttft-session.sh <out-dir>`, run from the repository root on a
free card after a release build: ignis with `--vision-enabled` and ignis
without it, **each launched twice**. Every launch writes a text record (1024, 8192 and
32768 tokens, cut from the ninfer corpus) and, with vision, an image record;
`ignis-bench ttft-pool` pools each load's two launches into one record, and
`g2` compares the pooled text records, with `--vision-enabled` against without.
The 1.5 threshold `g2` prints decides nothing; the ratio is an error detector.
Another engine is one more `launch` loop of the script, with its own command.

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

## The KV-format A/B (GitHub #139)

What it answers: how much the hq-e8-2b KV format costs against BF16 on ignis,
and what the card is actually doing while it is paid. The 2026-09-13 run, which
also ran both formats on the reference engine to tell ignis's cost from the
format's, is
[`docs/findings/2026-09-13-hq-vs-bf16-decode-cost.md`](../../findings/2026-09-13-hq-vs-bf16-decode-cost.md);
this is how to redo its ignis legs.

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

Needs the GPU exclusively (ADR 0006,
[who holds the card](../testing.md#who-holds-the-card)). Two runs, ~4 minutes each. Launch the legs with the explicit commands
below: `make start` brings up the G5 configuration (262144 context, DFlash2),
a different engine from these legs. Each engine started by hand runs under
`make gpu-run` (GitHub #707): it waits for the GPU lock, runs the guard, and
the server holds the lock until it exits. `CMD` is one sh line, so the paths
take forward slashes and the artifact's path sits in single quotes.
`ignis-bench` is an HTTP client and takes no lock: the server it measures holds
it, and so does one `make start` started.

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
$Flags = "--server-bind 127.0.0.1:8000 --model-kv-format hq-e8-2b --model-max-context 40960 " +
  "--vram-kv-pool-bytes 576M --model-prefill-chunk 1024 --server-request-timeout 900 " +
  "--model-idle-unload 0 --model-loop-guard off"
make gpu-run CMD="./target/x86_64-pc-windows-msvc/release/ignis-server.exe --model-artifact '$Artifact' $Flags"

# the cell, once per leg
.\target\x86_64-pc-windows-msvc\release\ignis-bench.exe g3 `
  --endpoint http://127.0.0.1:8000 --artifact $Artifact --corpus $Corpus `
  --label ignis --profile "hq-e8-2b KV, 1024 chunk, max-context 40960, 65536-token pool" `
  --session "kvab-<leg>-$(Get-Date -Format yyyyMMddTHHmmssZ)" `
  --out .scratch/g3-<leg>.json
```

The engine logs the capacity it resolved, `ignis.runtime.kv_pool`'s
`page_count`. Check it says 1,024 pages / 65,536 tokens before believing the leg.

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

**This is not a comparison by the measuring rule.** Each leg is its own session,
so `g3-gate` will refuse to build a verdict from these records, by design.
The A/B answers a question about *where time goes*; a ratio between two sides
needs them measured back to back under one `--session`.

## G3 against ignis (GitHub #426)

The ITL decode lanes send each engine the one EOS control it is asked by:
`ignore_eos` to an engine whose `GET /v1/models` names ignis as the owner
(`owned_by`), since ignis refuses a non-empty `logit_bias` (#288), and
`logit_bias`, the OpenAI field, to any other engine (the reference engine
accepted it without applying it:
[finding](../../findings/2026-09-13-itl-lane-terminator-race.md)). The run says
which on stderr.

## Every prompt of a run is its own (GitHub #426, #443)

A `ttft`, `g3` or `g5` run generates all its prompts before its first request,
numbered across the run: `ttft`'s a set per cell in `--cells` order, the image
cell's last (`small-card`'s TTFT cells are such a run, of the cells that fit);
`g3`'s C=1, C=4, the ITL lanes, then the ITL prefillers; `g5`'s `--lanes` per
depth (one unset), the depths in order. Each prompt starts with its own nonce
and seed or, cut from a corpus, its own window, the run's windows spread evenly
over the bank (at `g5`'s three gate depths and one lane, each depth opens in its
own third of it); a window that starts with another prompt's
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
