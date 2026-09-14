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

(Corrected at triage: the first draft said the storm ran ~5 min at ~13k/s
while other requests were served. The excerpt says otherwise.)

- 16:09:37, six seconds after request 5 finished (`stop`, 2,629 tokens),
  `ignis.runtime.leaf_error {context: "sequence alloc", error:
  "ignis_seq_alloc: sequence pool exhausted (KV pages)"}` starts; the last
  one is at 16:10:17 — about 40 s.
- 4,135,039 of the log's 4,135,090 lines are that event — about 100,000 per
  second (the excerpt's per-second counts), ERROR severity, each identical.
- The stuck request is **request 10**: it never logs `admitted`. No request
  was admitted during the storm; it ends when request 10 goes away (most
  likely the client's cancel), and requests 11–14 are served normally
  afterwards.
- The owner's conversation was about 10K tokens — far below the context
  limit.

## Root cause (triage)

Not a core/leaf page-accounting drift. The leaf caps one sequence's
reservation at `logical_page_capacity = pages_for_tokens(max_context)`
(`kernel/src/seq.cu`, `PagedKVPool::can_reserve`): 4,096 pages at 262,144
tokens. A request without `max_tokens` reserved `prompt + max_context`,
always past that cap, so its alloc failed however empty the pool was — the
"pool exhausted" message is misleading. Core's `Oversized` check only
compared against the whole pool (7,281 pages), so the request was accepted;
the failed prefill was then retried on every advance with no bound. The
Playground omits `max_tokens` when its field is empty; requests 0–5 must
have carried one. Drafter window (separate arena) and prefix pages ruled
out.

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
