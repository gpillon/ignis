# The codebase as an n-gram memory: the summary exists, the injection cannot use it

- Kind: experiment
- Status: current
- Observed: 2026-09-19
- Last verified: 2026-09-19
- Scope: retrieval / conditional memory, hidden-state structure, PyTorch measurement vehicle
- Related: [`.scratch/ngram-study/02-storage.md`](../../.scratch/ngram-study/02-storage.md) (plan), [`.scratch/ngram-study/03-risultati.md`](../../.scratch/ngram-study/03-risultati.md) (full numbers), [`.scratch/ngram-study/00-fatti.md`](../../.scratch/ngram-study/00-fatti.md) (primary sources on Flash-Next / Engram)
- Superseded by: none

## Question

Qwen3.8-Flash-Next and DeepSeek Engram both add an embedding table indexed by a
hash of the local n-gram: static knowledge retrieved with a gather rather than
with FLOPs, cheap enough to live in host RAM. Neither the model ignis serves
nor ninfer has such a module.

The question this experiment answers is not whether ignis could serve one. It
is whether the same *mechanism* — key, table row, add into the residual — can
be pointed at this repository's own source: key = symbol name, value = a hidden
state of the model taken at that symbol's definition, retrieved at O(1) when
the name reappears. The KV cache cannot do this job (9.07 MB of sources is
~2.6 M BPE tokens against a 262 k window, and attention over it would be O(len)
per generated token against the lookup's O(1)), so if the idea works at all it
is a separate mechanism, not a cache tuning.

The hypothesis under test, stated in the plan before measuring: **is the hidden
state at the last token of a function body a usable summary of that body?**

## Evidence

Vehicle: `Qwen/Qwen3.8-27B` at revision `1d4bf0f2…` — the exact revision the
ignis artifact manifest pins — quantized on load to NF4 by bitsandbytes 0.50.2,
16.49 GiB on disk. Two identical forwards give bitwise identical logits, which
is what makes the α=0 control readable. Corpus: this repository, 801 files,
9.07 MB, 2,599,881 BPE tokens, 12,780 symbols from a parser that reports zero
failures. Raw output under `.scratch/ngram-study/results/`.

**Phase 0a — are end-of-definition hidden states distinct?** 200 top-level
definitions, against a control of random spans in the same files. Raw pairwise
cosine is uninterpretable (0.995 between *any* two positions at L2), so the
numbers are mean-centered and read against the control. Effective rank out of a
possible 199:

| layer / pooling | definitions | control |
|---|---|---|
| L2 / last | 18.3 | 35.2 |
| L19 / last | 68.3 | 143.5 |
| L33 / last | 85.0 | 150.2 |
| L47 / last | 82.2 | 149.6 |
| L61 / last | 60.5 | 149.0 |

Not collapsed — a collapse would read ~1 — but about half the spread of an
arbitrary position.

**Phase 0b — does a use site recognise its own definition?** Hidden at the
first use of each symbol in another file, scored against all 200 definition
vectors. Chance is 1/200 = 0.5%.

| layer | recall@1 | gold cosine | other cosine |
|---|---|---|---|
| L2 | 0.011 | +0.928 | +0.928 |
| L5 | 0.011 | +0.951 | +0.951 |
| **L19** | **0.413** | +0.782 | +0.764 |
| L33 | 0.228 | +0.643 | +0.615 |
| L47 | 0.217 | +0.521 | +0.485 |
| L61 | 0.228 | +0.390 | +0.342 |

**Phases 2-4 — injection.** Index of 3,345 rows, 20 held-out files chosen for
using the most symbols defined elsewhere, teacher-forced NLL with and without
injection, primary metric the NLL over the 8 tokens after each match,
bootstrap over files. Sweep: layers {2, 19, 47} x poolings {last, mean} x
α {0, 0.03, 0.1, 0.3, 1.0} x cosine threshold {none, median, high}.

The form the plan specified, `residual += α·‖h_t‖·v/‖v‖` on the raw row, is
damage almost everywhere and monotone in α (+0.668% at L19/last, α=1). That
form is wrong, and Phase 0a says why: measured over the index, ‖μ‖/‖v‖ is
0.953 at L19/last and 0.993 at L2, so a unit-normalised raw row is almost
entirely the layer's shared massive-activation direction — the one `h_t`
already has — and only 0.31 (L19) or 0.12 (L2) of it is what distinguishes one
symbol from another. Centering the rows on the index mean before normalising
makes α scale the distinctive component; the gate cosines then spread from
-0.02 to 0.40 instead of sitting packed in 0.77-0.89.

Centered, the best cell of the whole sweep is -0.012% [-0.048, +0.024] at
L19/last, α=0.03, median threshold. With the rare-key filter (at least two BPE
tokens and contained in no other key) several cells go significantly negative:
-0.280% [-0.412, -0.152] at α=0.1, -0.358% [-0.702, -0.025] at α=0.3.

Two negative controls decide what those numbers mean. Both permute the
key→row pairing, keeping the same keys, positions and norms. Intervals are
bootstrap over held-out files (n=20 at the full key set, n=18 at the rare set,
where two files contain no rare match at all):

| condition | correct row | shuffled row |
|---|---|---|
| all keys, L19, α=0.1, τ none | +0.010 [-0.067, +0.080] | +0.058 [-0.003, +0.113] |
| all keys, L19, α=0.1, τ median | +0.007 [-0.041, +0.056] | +0.062 [+0.012, +0.114] |
| all keys, L19, α=0.3, τ none | +0.051 [-0.133, +0.225] | +0.198 [+0.059, +0.330] |
| rare keys, L19, α=0.1, τ none | -0.280 [-0.412, -0.152] | -0.210 [-0.408, -0.026] |
| rare keys, L19, α=0.3, τ none | -0.358 [-0.702, -0.025] | **-0.502** [-0.769, -0.227] |

At the full key set the correct row's point estimate is 1.3-8.4x lower than the
shuffled one and the sign is the same in all six L19 cells measured, but every
interval overlaps: suggestive, not separated. Separating them would need a
paired bootstrap over per-file deltas, which this run did not store. At the
rare key set the shuffled row does as well or better, and there the overlap
argues in the conservative direction.

**Out of domain.** Two corpora, and the cleaner one is not the one the plan
asked for:

| corpus | keys | match rate, in domain | match rate, out |
|---|---|---|---|
| third-party C (ffmpeg, curl headers under ninfer's build tree) | 3,345 | 3.76% | **2.80%** |
| third-party C | 2,850 rare | 0.28% | 0.17% |
| ninfer's own sources | 3,345 | 3.76% | 3.89% |
| ninfer's own sources | 2,850 rare | 0.28% | 0.86% |

The ninfer row is **not a clean out-of-domain measurement**: ignis carries a
vendored copy of ninfer's kernel sources in `kernel/vendor/`, which is part of
this repository and therefore part of the index — 1,184 of the 3,345 index rows
(35.4%) are defined there, and 11 of the 20 sampled ninfer files have a
same-basename counterpart under it. It measures a partially overlapping corpus,
not a foreign one.

The third-party sample is the cleaner one, but "third-party" overstates it:
of its 20 files, 16 are vendored foreign code (ffmpeg, curl under
`build-ninja/vcpkg_installed/`) and 4 are ninfer's own sources
(`apps/cli/options.cpp`, `src/ops/linear/linear.cpp`,
`src/ops/attn_input_proj/w8/w8_attn_input_plan.cpp`,
`src/product/prompt_input/prompt_input.cpp`), which carry the same shared
provenance the row above describes. It still matches at **2.80%, 74% of the
in-domain rate**, and that is the load-bearing non-selectivity number — with
the contamination pulling it *up*, so a purely foreign corpus would land at or
below 74%, never above. The correction runs in the conservative direction.

Note also that the match rate is a property of the key and the corpus, not of
the layer: the in-domain rate is 3.76% at both L19 and L47. The two rows
therefore compare two *corpora*, not two layers. What is missing is a clean
foreign corpus measured with the L19 index; the 3.89% row is the contaminated
one, and it says that on code sharing provenance the mechanism fires slightly
*more* than at home.

The rare-key filter removes 15% of the index keys (3,345 -> 2,850; of the
corpus's 7,598 distinct names, 6,033 pass the rule, and the index holds only
the subset that is unambiguous and fits the token budget) and 93% of the
matches.

## Finding

Observed:

- The summary exists and is locatable. End-of-definition hidden states are not
  collapsed, and at **L19 of 64** a use site picks its own definition out of
  200 candidates 41.3% of the time, 83x chance.
- It is not at layer 2. At L2 and L5 recall@1 is at chance and the gold and
  non-gold cosines agree to three decimals.
- With the full key set, injecting the *correct* row costs less than injecting
  a random row of the same norm at the same position: point estimates 1.3-8.4x
  lower, the same sign in all six L19 cells, intervals overlapping.
- With the rare-key set, the apparent gain is reproduced — and at α=0.3
  exceeded — by the shuffled pairing.
- On code with no shared provenance, the mechanism still matches at 74% of its
  in-domain rate.

Inferred:

- The key→value match probably carries information about **compatibility**
  (the right vector disturbs less, consistently in direction if not
  separably) but not information the model converts into a **better
  prediction**. An additive injection does not turn the first into the second.
- The small rare-key improvement is a generic perturbation effect at a 0.28%
  match rate, not retrieval. Without the shuffled control it would have been
  reported as a -0.28% gain that does not exist.
- Flash-Next's layer-2 injection point does not transfer to this use. There the
  n-gram describes the current token, and at L2 the residual still is the
  current token; a key that has to recall a summary has nothing to recall yet.

By the criteria fixed before measuring, the outcome is **no signal**: no
(layer, pooling, α, threshold) gives an in-domain ΔNLL below -0.5% that
survives its control.

## Implications

- The fast path sketched in [`.scratch/ngram-study/01-strade.md`](../../.scratch/ngram-study/01-strade.md)
  — a mapped host arena, a UVA gather kernel, device-resident addresses, capture
  inside the decode graph — **should not be built**. Its premise was that the
  signal existed.
- What the plan called "the part that is not free" is narrower and harder than
  it looked. It is not enough to stop the model getting worse with a cosine
  threshold; a learned transform is needed to turn a compatible vector into a
  prediction. That is the Engram Adapter's trained gate, and this experiment
  quantifies how much of the work it has to do: all of it.
- Two measurement lessons that outlive this experiment. **Raw cosine in a
  residual stream does not measure similarity** — mean-center and carry a
  control set, or a massive-activation artifact will be read as structure (in
  either direction). **A retrieval experiment needs a shuffled-value control**;
  correct-versus-random is the only comparison that separates retrieval from
  perturbation.
- Two operational facts for anyone running PyTorch on this box. At 8192 tokens
  a 27B forward peaks at 32.00 GiB of the card's 32.6 and takes 78.85 s; at
  4096 it is 20.58 GiB and 4.11 s. `expandable_segments` prints "not supported
  on this platform" on Windows, so the allocator's reservation drifts into WDDM
  paging across files of differing length unless `torch.cuda.empty_cache()` is
  called between them. Calling the wrapper model rather than the language stack
  runs `lm_head` over every position — 2 GB of logits built and discarded per
  forward — and by itself accounted for 31.9 GiB of reservation against 18.8.

## Limits and unknowns

- One injection form was tested: a single additive vector at the token that
  completes the name, at one layer. A learned projection, a gated residual, a
  cross-attention read, or injection at several layers are all untested.
- Values are end-of-definition and body-mean hidden states. A value trained for
  the purpose — the Engram Adapter recipe — is a different object and this says
  nothing about it.
- The evaluation truncates held-out files at 4096 tokens, a VRAM limit rather
  than a design choice, so the longest files are only partly scored.
- The "correct disturbs less" comparison is unpaired: the bootstrap resamples
  files within each arm separately, so the intervals overlap even though the
  direction is consistent. A paired bootstrap over per-file deltas would settle
  it and needs a rerun that stores them.
- ninfer is a *partially overlapping* corpus, not an out-of-domain one, because
  `kernel/vendor/` is a vendored copy of its kernel sources and is deliberately
  part of ignis's own corpus. Its numbers are reported as same-domain. The
  third-party C sample is the cleaner one, and 16 of its 20 files are foreign;
  the other 4 are ninfer's own, so even it is not a purely foreign corpus. No
  clean foreign corpus was measured against the L19 index.
- NF4 weights, not the NVFP4 the server runs. The measurement is of hidden-state
  structure, which is unlikely to hinge on that, but it is not the served
  numerics.
- The index covers 3,345 of the corpus's 7,598 distinct names: 1,149 are
  ambiguous (defined twice) and 2,945 sit in files over the token budget.
- `recall@1 = 0.413` at L19 is measured against 200 candidates. It says nothing
  about how it would scale against 3,345 or against a monorepo.

## Follow-ups

- If conditional memory is revisited, start from L19 and from a trained gate,
  not from a threshold — and keep the shuffled control as the acceptance test.
- The measurement harness (`.scratch/ngram-study/scripts/`) is reusable for any
  "does the model already know X" question: layer taps by character span, a
  teacher-forced NLL with post-match windows, and the two controls.
