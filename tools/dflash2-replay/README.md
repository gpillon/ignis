# DFlash2 teacher-forced replay of recorded agent turns

The tools behind `docs/findings/2026-09-24-dflash2-sampled-acceptance.md` (spec
`docs/specs/runtime/08-dflash2-sampled-acceptance.md`, GitHub #267), kept as they
were run so that a drafter or verify change can be paired against the same rounds.
GitHub #340 (spec `docs/specs/27b-16gb/05-gates-and-simulation.md`, gate 3) runs the
same turns on the 16 GB artifact's trellis body and on the served NVFP4 one.

A recorded turn is walked through **real verify rounds** on a DFlash2-7 hq-e8-2b
load (full head, 1024-token chunks, rope none, 262,144 context; the trellis body in
`rtx50-16gb`'s 16 GiB plan at two lanes, the NVFP4 one at eight on the card's
derived budget, or at `--lanes N` / `--prefill-chunk N`, which the trellis body
refuses -- `crates/runtime/examples/support/replay_shape.rs`). The load is planned
with the server's plan pieces -- the same lines, `plan_vram`, and the read-back of
the leaf's buffers and pool against the plan -- but it does not adapt a plan that
does not fit (it refuses it) and runs no host plan (ADR 0062; it holds no KV-RAM
arena, KV disk or retained slots): `crates/runtime/examples/support/replay_load.rs`.
The drafter proposes from the lane's window as in serving, and a
teacher-forced round (`step::decode_program_verify_forced`) commits the recorded
text instead of the target's draw -- the leading drafts it agrees with, then the
recorded token. A live round under sampling accepts a one-hot draft exactly when
the token it samples equals it, so these are the rounds a live run that sampled
the recorded text would have made (`crates/core/tests/dflash2_replay_seams_gpu.rs`
holds forced rounds to the live ones bit for bit, greedy and sampled). Per round the
driver also reads the drafter's lattice (`dflash2_readout::propose`) and the verify's
target logits (`dflash2_readout::verify_logits`). Every turn must end at a stop id
(`<|im_end|>`): past a turn's end the driver forces the pad `-1`, which the leaf takes
as a run's successor only after a stop (all 200 of the 2026-09-24 turns end at one).

**Under a draft rule** (GitHub #577, spec #574, ADR 0090). `--draft-rule sampled` replays
the same turns under the sampled draft at its default drafter settings (temperature
factor 0.9, top-p 0.95); without it, today's fixed draft. Each turn's lane then samples as
the recorded traffic did (temperature 1, top-k 20, top-p 0.95) under a seed of the turn's
own. The forced round drafts along the recording -- each recorded token that is one of
its column's 16 candidates -- and takes it with the probability a live round under the
rule takes the token it emits, min(p, q) / p: the live round's own verdict where the
lane's draws emit the recorded token, a coin of its own elsewhere
(`sampled_draft::forced_verdict`, held to the rule's double-precision oracle). So the
rounds a replay walks are distributed as live rounds under the rule are, over the same
text, and the replay seams test holds forced rounds to live ones bit for bit under the
sampled draft too (`dflash2_replay_seams_sampled_gpu.rs`). The record is the same 9,520
bytes; its `drafts` are the lane's draws, and its `agreed` says the forced run took only
recorded tokens among their column's candidates. The rule lands in `OUT_DIR/rule.json`
(none: today's rule, as the 2026-09-24 run); one directory keeps one rule.

| File | What it is |
|---|---|
| `select.py` | Picks the turns from the copy-drafting dataset (`.scratch/copy-drafting-2026-09-24/` in the main checkout: 13,102 rendered, tokenized qwen-code/opencode turns): source x kind strata, joined to their ninfer request-log DFlash2 stats. Writes `manifest.json` and one `<job>.tok` / `<job>.cat` per stream. Fixed seed. |
| `crates/runtime/examples/dflash2_replay.rs` | The GPU driver (in the crate): one 9,520-byte record per round, `<job>.rounds`, plus `turns.jsonl`; resumable per job (`<job>.done`). It prints the load's VRAM plan, line by line, before the first job. |
| `records.py` | The record layout, and the vendored sampler's distribution (top-k 20, top-p inclusive, renormalized) over a column's top 20. `test_records.py` pins the layout to the driver's (`python -m pytest tools/dflash2-replay`, CPU). |
| `validate.py` | Replay against the live ninfer rounds of the same turns: tok/round and P(accepted >= i) per source. |
| `analyze.py` | Per round: the realized coupling walk, the exact expected accepted length from the draft-conditioned verify columns, the greedy verdict, and one-hot/multi-candidate ceilings along the recorded chain; per column and per output kind: p(draft), p(argmax), top-2/4/16 mass; the expected-length lattice walk (lever 2). Writes `analysis.json` and `per_round.npz`. |
| `tree.py` | Lever 3: a best-first token tree of N nodes from the drafter's own lattice, its expected accepted length along the recorded chain. |
| `lever1.py` | Lever 1: adaptive extent at 8 lanes from `per_round.npz` and the width-cost curve (`crates/runtime/examples/verify_width_cost.rs`). |
| `draft_rule.py` | GitHub #577: one or more replay directories, each under its rule (`rule.json`), in the form spec #574's study printed: tok/round realized and expected (the Rao-Blackwellized expectation at the same anchors), traffic-weighted and per cell; the survival P(accepted >= i) and the probability a reached draft column is taken, per column; the same per kind of output; for today's rule also the sampled draft priced offline at its anchors (the study's estimate, 3.738 against 3.461 on the 2026-09-24 rounds, which it reproduces); with `--reference` a run of today's rule turn by turn. `test_draft_rule.py` holds its q to the host reference's closed forms. |

## Running it

```sh
python tools/dflash2-replay/select.py .scratch/replay/input          # CPU, seconds
cargo build -p ignis-runtime --features cuda --example dflash2_replay
target/x86_64-pc-windows-msvc/debug/examples/dflash2_replay.exe \
    F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer \
    .scratch/replay/input/manifest.json .scratch/replay/input .scratch/replay/out [FIRST [LAST]]
python tools/dflash2-replay/validate.py .scratch/replay/out .scratch/replay/input/manifest.json
python tools/dflash2-replay/analyze.py  .scratch/replay/out .scratch/replay/input/manifest.json
python tools/dflash2-replay/tree.py     .scratch/replay/out .scratch/replay/input/manifest.json
# GitHub #577: the same turns under the sampled draft, into a directory of its own
target/x86_64-pc-windows-msvc/debug/examples/dflash2_replay.exe --draft-rule sampled     F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer     .scratch/replay/input/manifest.json .scratch/replay/input .scratch/replay/out-sampled [FIRST [LAST]]
python tools/dflash2-replay/draft_rule.py .scratch/replay/input/manifest.json     .scratch/replay/out .scratch/replay/out-sampled [--reference OUT_DIR_OF_AN_EARLIER_RUN]
target/x86_64-pc-windows-msvc/debug/examples/verify_width_cost.exe ARTIFACT TEXT.bin 4096,16384,32768 30 cost.jsonl
python tools/dflash2-replay/lever1.py   .scratch/replay/out cost.jsonl 32768 .scratch/replay/input/manifest.json
```

The 2026-09-24 inputs (`manifest.json` and the 116 jobs' `.tok`/`.cat`) and that
run's `out/` are in the main checkout's `.scratch/dflash2-replay-267/replay/`
(untracked). The driver writes its records into whatever `OUT_DIR` it is given:
one directory per artifact. The dev profile builds at `opt-level = 2`, so the
debug example is not a slow build.

The 2026-09-24 run: 200 turns, 116 jobs, 6.5 M prefilled and 209 K replayed tokens,
61,692 rounds, ~55 minutes of GPU (20 of them prefill, the rest ~25 ms a round when the CPU is free). Keep the CPU free while it
runs: the round waits on the host's reductions, and under parallel CUDA builds the
replay fell to 1% GPU utilization. NumPy for the scripts (the `tokenizers` package
too for `analyze.py`'s stop ids).
