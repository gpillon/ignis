# Gate 1: KLD against a BF16 checkpoint

The tools behind gate 1 of the 27B's artifacts: an artifact's distributions, dumped by the engine on set A
(16 windows of 4,096 tokens: 6 code, 4 prose, 6 chat), against those of a BF16 checkpoint, held to a
registered limit. #340 (`docs/findings/2026-10-24-27b-16gb-quality-gates.md`) and #422
(`docs/findings/2026-10-30-27b-16gb-abliterated-gates.md`) ran it from `.scratch/`; spec #449 (GitHub #470,
#472) runs it on a **fine-tune**, against the fine-tune's own BF16, with two cross terms that show the gate
measured the fine-tune and not the base.

| File | What it is |
|---|---|
| `ref_kld.py` | The reference side: a BF16 checkpoint run layer by layer under transformers' Qwen3_5 text modules, on the GPU, one shard open at a time, and each `span_logits` dump scored against it per position (KL(ref ‖ engine) over the full vocabulary, top-1, both NLLs) into `<out-dir>/<label>.json`. `--save-hidden REF.pt` keeps the reference's post-norm hidden rows after its forward (0.67 GB for set A); `--hidden REF.pt` scores a later dump against them with the head alone, the same bytes into the same matmul. |
| `gate1.py` | The verdict, from four `ref_kld.py` results (below): the matched pair's KLD against `--limit` with its window-bootstrap 95% CI (#422's, 20,000 resamples, seed 0), the paired ratio to the baseline pair, the cross terms and the controls; `--json` writes the report. Exit 0 PASS, 1 FAIL, 3 when a cross term or a control does not hold (the gate decides nothing: stop and report). |
| `test_ref_kld.py`, `test_gate1.py`, `conftest.py` | The tests (CPU only) on synthetic logits and synthetic results, and a machine-local one: #422's results in the main checkout's `.scratch/` give #422's printed numbers. |
| `ref_top64.py`, `score_top64.py`, `test_top64.py` | The release's sm_86 gate (below): a top-64 reference set from one BF16 pass, its scorer, and their tests on a synthetic three-token set. |

## The four terms

Two dumps, the fine-tune's artifact and the vanilla one of the same layout, each scored against two
references, the fine-tune's BF16 and the base's. Each `gate1.py` term is `NAME=FILE`, a dump's name in a
`ref_kld.py` result:

| term | dump | reference | role |
|---|---|---|---|
| `--matched` | fine-tune | fine-tune BF16 | the gate: KLD ≤ `--limit` |
| `--cross-artifact` | vanilla | fine-tune BF16 | must sit above the matched pair |
| `--cross-reference` | fine-tune | base BF16 | must sit above the matched pair |
| `--baseline-pair` | vanilla | base BF16 | the registered baseline's own measurement; `--reproduce NAME=FILE` compares it per position with a prior run's arm (reported; `--require-reproduction` makes it void the gate) |

**The cross terms.** A cross term holds when it is above the matched pair in every window and the window
bootstrap's 95% CI of its difference from the matched pair is above 0. The spec says "far above" and
registers no number, so the printed ratios and differences go to the owner either way. On #422's data (a 70-matrix change) the cross
terms are 1.07x and 1.09x the matched pair and lower in one window each: under this rule that gate would
have stopped.

**The controls**, each of which voids the gate: the two terms of one dump carry the same engine NLL per
position, the two terms of one reference the same reference NLL; the two references' NLLs differ in every
window, and the two dumps' too.

## Running it

```sh
PY=F:/ai/ngram-venv/Scripts/python.exe
W=F:/ai/opencode/inference/.scratch/kld-2026-09-24/windows.json   # set A; windows_meta.json beside it
# one forward a checkpoint (GPU, ~13 min each); a dump may be scored in the same run
$PY -u tools/kld-gate/ref_kld.py ref-ft --windows-file $W --ckpt H:/models/<fine-tune> --save-hidden H:/kld/ref-ft.pt
$PY -u tools/kld-gate/ref_kld.py ref-base --windows-file $W --ckpt H:/models/Qwen3.8-27B --save-hidden H:/kld/ref-base.pt
# each dump (span_logits, ~32.5 GB for set A) against both, then deleted
$PY -u tools/kld-gate/ref_kld.py ftref-ft --windows-file $W --hidden H:/kld/ref-ft.pt --dump ft=H:/kld/ft --out-dir results
$PY -u tools/kld-gate/ref_kld.py baseref-ft --windows-file $W --hidden H:/kld/ref-base.pt --dump ft=H:/kld/ft --out-dir results
# ... the same for the vanilla dump `van`, then:
$PY tools/kld-gate/gate1.py --limit L --baseline B --matched ft=results/ftref-ft.json \
    --cross-artifact van=results/ftref-van.json --cross-reference ft=results/baseref-ft.json \
    --baseline-pair van=results/baseref-van.json --json gate1.json
```

`F:/ai/ngram-venv` holds what it needs (torch with CUDA, transformers with `Qwen3_5`, safetensors, numpy,
pytest). The reference reads the checkpoint through `tools/27b-16gb-converter/overlay.py`
(`IGNIS_CONVERTER_TOOLS` names another copy), which also lays a `replay.py fetch` overlay over it (`--overlay`,
#422's abliterated BF16).

## The sm_86 gate: top 64 of the BF16 checkpoint

The release's sm_86 gate (spec #474, `docs/agents/testing.md` and `docs/user/contributing/releases.md`, "The sm_86
gate") scores the 27B's 16 GB artifact on a card that cannot hold the 55.6 GB BF16 checkpoint, so it scores against a
**reference set** that holds, at each of set A's 65,536 positions, the BF16 model's 64 most probable next tokens and
their log-probabilities: about 90 MB, the windows included (`windows.json`, what `span_logits` takes).

- `ref_top64.py` builds it on the RTX 5090: `ref_kld.py`'s forward of the BF16 checkpoint, then per position the
  BF16 stream's top 64 and, beside them, a dense `span_logits` dump's (the `q_` stream) argmax, top 64 and
  log-probabilities at the BF16 ids, its full-vocabulary KL, and the top-k sweep (k = 16 ... 256) against that KL.
  The files follow flash-next-layout.md section 10, with `manifest.json` and `full_kld_5090.json` beside them;
  `--gate1` checks the full-vocabulary KL, position by position, against a `ref_kld.py` result of the same dump.
- `score_top64.py` scores a dense dump against the set, with python3 and numpy alone, one window at a time in
  slices of `--rows` rows (about 1 GB of RAM): per kind and over all positions, the top-64 KLD against BF16 (the
  `kl_top64` of `crates/bench/src/flash_next/kld.rs`), top-1 against BF16 and against the set's `q_` stream, and
  the perplexity; `--json` writes the figures. The gate reads the `all` row.

The set stays out of the repo: the release's coordinator carries it to the sm_86 card. To rebuild it, with the BF16
checkpoint (`Qwen/Qwen3.8-27B` at the 16 GB artifact's revision) and the 5090's dump of set A on the serving defaults
(chunk 1024, `hq-e8-2b`, the 8-bit prompt band), 32.5 GB of disk while it lasts:

```sh
cargo run --release -p ignis-core --features cuda --example span_logits -- \
    <models>/qwen3_8_27b_trellis_sc30-v2.ninfer hq-e8-2b <windows.json> <dump-5090> 1024
python -u tools/kld-gate/ref_top64.py --ckpt <Qwen3.8-27B BF16> --windows-file <windows.json> \
    --dump <dump-5090> --out ref27b-top64 --engine "<commit>, RTX 5090, 8-bit band"
```

`<windows.json>` is set A's with its `windows_meta.json` beside it, or a set's own `windows.json`, whose kinds its
`manifest.json` gives. The top-64 KLD is a lower bound of the full-vocabulary KLD: on the 5090's dump it reads 5.8%
under it (0.03801 against gate 1's 0.04034), the same truncation for any engine to first order, so an engine's figure
is held to the 5090's top-64 figure, never to gate 1's full-vocabulary one.

## Tests

```
F:/ai/ngram-venv/Scripts/python.exe -m pytest tools/kld-gate -q
```
