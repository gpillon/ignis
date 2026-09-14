# A refused sequence alloc retries every advance and floods the log

Observed 2026-09-14 driving the Playground (#164) against a real GPU ignis.
Evidence excerpt: `2026-09-14-seq-alloc-exhausted-log-storm.excerpt.txt`
(next to this file; `.txt` because `*.log` is gitignored; the 1.2 GB
original was not kept).

## Setup

`ignis-server` built from `main` 95371fd, `--features cuda`:

```
--artifact qwen3_8_27b_nvfp4full-v2.ninfer --kv-format hq-e8-2b
--max-context 262144 --prefill-chunk 1024 --request-timeout 1800
--spec dflash2 --draft-tokens 7
```

Load reported `page_count 7281` (4 GiB auto pool, 64 tokens/page → 465,984
tokens), speculation `dflash2 draft_tokens=7`.

## What happened

- 16:09:37, six seconds after request 5 finished (`stop`, 2,629 tokens),
  `ignis.runtime.leaf_error {context: "sequence alloc", error:
  "ignis_seq_alloc: sequence pool exhausted (KV pages)"}` starts and never
  stops until the process is killed (~16:14:30).
- 4,135,039 of the log's 4,135,090 lines are that event — about 13,000 per
  second, ERROR severity, each identical.
- Requests kept being served meanwhile (14 admitted / 14 done in total; the
  last, request 14, 2,695 tokens at 82 tok/s), so some request was stuck
  retrying while others went through.
- The owner's conversation was about 10K tokens — far below the context
  limit.

## Facts from the code

- The reservation passed to `ignis_seq_alloc` is not the tokens in use: it
  is `prompt + (max_tokens or max_sequence_tokens)`
  (`crates/core/src/concrete.rs`, the `PrefillJob.context_tokens` build).
  With `max_tokens` unset (the Playground's "engine cap"), a request reserves
  `prompt + 262,144` tokens = 4,097+ pages, so two such sequences (8,194
  pages) cannot coexist in 7,281.
- `RuntimeCompute::prefill_step` (`crates/runtime/src/lib.rs`) maps the
  alloc failure to `RuntimeError::Leaf`; the scheduler leaves a failed
  prefill retryable (`concrete.rs` module docs: "the retry resends…"), and
  `cuda_leaf.rs::leaf_error` logs ERROR on every attempt — no backoff, no
  dedup.

## Open questions for triage

1. Why did admission let the request reach the leaf at all? Core charges
   `kv_pages` on materialization; if its view says the pages are free while
   the leaf's pool says not (a published prefix still holding pages, the
   DFlash2 drafter window, or a released sequence not yet returned), the two
   accountings disagree — that is the real bug.
2. Should a request whose full reservation can never fit while another
   resident holds its pages wait (admission), be refused (503/413), or
   reserve less than `max_context` when `max_tokens` is absent?
3. The failure path must not be a hot loop: one event per request state
   change (or rate-limited), and no busy retry every advance — it also burns
   a CPU core on the model thread.
