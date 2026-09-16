# Vision TTFT live/live

- Kind: experiment
- Status: current
- Observed: 2026-09-16
- Last verified: 2026-09-16
- Scope: serving / multimodal TTFT, text TTFT with `--vision` loaded, VRAM headroom
- Related: https://github.com/gpillon/ignis/issues/181, [ADR 0015](../adr/0015-g2-live-live-cold-prefix-gate.md), [ADR 0021](../adr/0021-live-live-launch-pooling.md), [testing.md](../agents/testing.md#the-multimodal-ttft-cell-github-181)
- Superseded by: none

## Question

How does ignis's time to first token on a ~1-megapixel screenshot plus a short
question compare with the reference's, both with `--vision`, and does loading
`--vision` move ignis's text TTFT?

## Evidence

One session (`vision-ttft-20260916T185831Z`), `scripts/vision-ttft-session.sh`,
each engine launched twice and pooled (`scripts/ttft-pool.py`), five samples
plus a warmup per cell per launch, every sample cold. Raw records, per-launch
`nvidia-smi` samples and server logs: `.scratch/vision-kv-reuse-run/181/ttft3/`.

- Reference: `ninfer-serve --vision --spec mtp --draft-tokens 3 --lm-head-draft`,
  hq-e8-2b KV, 262,144 context, 4 concurrency, 450,000 KV capacity.
- ignis: hq-e8-2b KV, 262,144 context, 1024 prefill chunk, `--spec dflash2
  --draft-tokens 7`, prompt reuse on, with and without `--vision`.
- The image cell: `crates/bench/tests/fixtures/vision_ttft/screenshot.png`,
  1280x800, 1,000 vision tokens, 1,027 prompt tokens; both engines counted it
  the same.

Pooled medians (ms):

| cell | reference `--vision` | ignis `--vision` | ignis text-only | ignis/reference | `--vision`/text-only |
|---|---|---|---|---|---|
| image, 1,027 | 158.1 | 196.3 | — | 1.24 | — |
| text, 1,024 | 94.8 | 175.3 | 185.2 | 1.85 | 0.95 |
| text, 8,192 | 819.3 | 1,019.1 | 927.6 | 1.24 | 1.10 |
| text, 32,768 | 4,344.6 | 5,046.2 | 4,919.0 | 1.16 | 1.03 |

The two launches of each engine agreed to within 1% on every cell (for example
ignis `--vision` at 8,192: 1,017.7 and 1,020.4; text-only: 927.5 and 930.3).
Peak `memory.used` per launch: reference 27.95 GiB, ignis 31.6–31.8 GiB either
way, with the desktop holding 1.9–2.0 GiB before each launch.

A discarded first session (`vision-ttft-20260916T183353Z`, one launch each,
`.scratch/vision-kv-reuse-run/181/ttft/`) measured ignis `--vision` at 760.7 /
3,632.5 / 22,899.3 ms on the text cells and 463.3 ms on the image, with
samples worsening within a cell (1,024: 293, 503, 761, 1,138, 1,033 ms), one
32K sample at 41 s and a 5,993 ms image warmup. The desktop held 3.1 GiB
before that launch (2.3 GiB before its text-only launch, which measured
normally). Diagnostic single launches with the same flags
(`.scratch/vision-kv-reuse-run/181/ttft-diag/`) measured 214 / 1,424 ms on
the text cells and 196 ms on the image; with `--vision-max-tokens 8192`,
185 / 904 / 194 ms; with `--prompt-reuse off`, 130 / 980 / 199 ms.

## Finding

Observed:

- The image cell is 1.24x the reference, but the vision part of it is not
  where ignis loses. Taking the image cell minus the text cell of about the
  same length: ignis 196.3 − 175.3 = 21 ms, reference 158.1 − 94.8 = 63 ms.
  The ratio comes from the short-prompt text gap (1.85x at 1,024 tokens).
- Loading `--vision` does not slow short or long text prefill (0.95x at 1,024,
  1.03x at 32,768). At 8,192 it is 1.10x, reproduced in both launches — beyond
  noise and unexplained.
- An ignis load at 262,144 context with DFlash2 peaks at ~29.8 GiB of its own
  either way, because the retained pool is derived from what is free after
  load (868 MiB with `--vision`, 2.06 GiB without).

Inferred:

- The discarded session was VRAM oversubscription: ~29.8 GiB of ignis plus
  3.1 GiB of desktop exceeds the card's 32.6 GiB, and Windows pages instead of
  failing, which fits samples that worsen and wander rather than an error.
  Nothing in that launch recorded memory, so this is inferred from the
  arithmetic, the non-reproduction on identical flags, and the
  `--vision-max-tokens 8192` diagnostic returning to parity.
- The 1,024-token gap to the reference is mostly prompt reuse's cost on a
  short prompt: one diagnostic launch without it measured 130 ms against
  175–214 ms with it.

## Implications

- No large vision-specific regression: the vision path (acquisition,
  preprocessing, encode, the image's prefill) costs ignis less than the
  reference on this image.
- A derived retained pool leaves under a gigabyte of headroom on a card that
  also drives a desktop, and the failure mode is silent slowness, not an error.
  Any TTFT or throughput measurement on this machine should sample
  `nvidia-smi` beside the engine.

## Limits and unknowns

- The engines run different drafters (reference MTP3, ignis DFlash2/7), each
  as its operator runs it; a TTFT includes each drafter's prefill-side work.
- ignis reports no cached prompt tokens, so its samples' coldness rests on the
  prompts' construction (a nonce per text part, a changed pixel per image), not
  on the void rule.
- The oversubscription explanation and the prompt-reuse attribution each rest
  on single diagnostic launches.
- One image, one question, `max_tokens` 8, thinking off.

## Follow-ups

- [#202](https://github.com/gpillon/ignis/issues/202): the 8,192-token text
  cell 1.10x with `--vision` loaded.
- [#203](https://github.com/gpillon/ignis/issues/203): prompt reuse's cost on
  short prompts (the 1,024-token cell), which bears on the reuse gate
  ([#191](https://github.com/gpillon/ignis/issues/191)).
- [#204](https://github.com/gpillon/ignis/issues/204): VRAM headroom of the
  derived retained pool.
