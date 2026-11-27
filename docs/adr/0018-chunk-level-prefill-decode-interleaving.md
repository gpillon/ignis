---
status: accepted
---

# Prefill and decode interleave at chunk granularity, driven by the Rust scheduler

Prefill and decode take turns on the one model stream at chunk granularity: the Rust scheduler drives the chunk
loop, handing the leaf one chunk-wide span per call, and one `advance()` runs at most one prefill call and one
decode round. Nothing runs concurrently on the GPU. A prefill call that ran a whole span stopped every decoding
lane for its length, about 3.8 s for a 32K prompt when this was decided against about 0.1 s for one 1,024-token
chunk, and this decision fixes the serving loop's shape: a loop frozen around "a prefill call runs to completion"
and a request state machine around "a request is never observably half-prefilled" could later be undone only by
rewriting the scheduler.

- **No callback from the leaf into Rust.** It would put scheduling below the step boundary, against ADR 0009. The
  leaf keeps its own loop over a long span, which the per-token self-oracle and the GPU tests use.
- **Exactly one request at a time holds multi-chunk prefill progress**, and the others queue. A paused prefill
  holds its KV pages, GDN slot and conv taps, so several at once would multiply that VRAM. Interleaving the chunks
  of several prompts would buy fairness for a short prompt behind a long one, never throughput, and is not built.
- **`Prefilling` is a durable state that carries prefill progress.** Resuming needs no primitive: it is the
  scheduling of the next chunk. **Cancel is abort, not suspend.**
- **A completed chunk boundary is a GDN resumable boundary**, as consistent as a decode round's, and a snapshot
  point for the whole sequence state that exists. A new section of sequence state must be checked against this
  property before it is captured there. A lane-less `Prefilling` request is an eviction candidate at a chunk
  boundary (ADR 0023).
- **A load plans two chunk widths and the scheduler picks between them at every chunk boundary**: the prefill chunk
  while a lane is decode-ready, the solo width while none is, so the width can change within a prompt. The prefill
  scratch is reserved once, at load, for the wider (ADR 0030): serving narrows without allocating anything, never
  runs a chunk wider than the load reserved for, and borrows nothing from the KV pool. The solo width's default
  follows the card class (ADR 0063).
- **The decode share paces a prompt's chunks.** After a chunk that took `t`, the same prompt's next chunk waits
  until the decoding lanes have had `t * s / (1 - s)` of wall time; a share of 0 is one decode round per chunk,
  which held while a chunk took about 0.1 s. Flash-Next's chunk takes seconds, and one round per chunk left its
  other lanes at about 1 tok/s during a long prefill. The total work is the same at any share, only who waits
  changes, and a prompt alone is unaffected. The default share keeps the decoding lanes fluent for a reader, who
  reads about 5 to 8 tok/s, while a long prompt's time to first token under load grows by about a third instead of
  doubling as at an even split.
- **The prefill phase hands its events over mid-tick**, before the lane deal and the decode round, so a prompt's
  first token waits for the prefill alone, and every token is still sent once, in order.

When a lane counts as decode-ready, when the decode share holds nothing, what a cancel releases and what the
hand-off carries are the scheduler's rules, in the opening comment of `crates/core/src/concrete.rs`.

## Considered Options

- **Serialize prefill and decode.** The reference does not overlap them at all, so the gate of the time, which
  compared inter-token latency under a concurrent prefill with the reference's, would have passed with nothing
  built. Rejected for the loop shape it freezes, above.
- **True overlap on separate streams**, prefill and decode resident together and contending for SMs: the largest
  win and the largest unknown. It is not built, on grounds of risk and not because it is unwanted, since it is the
  differentiator the north star names: SM contention has no partitioning story, the workspace arena is not built for
  two writers, and it conflicts with capturing decode CUDA graphs while another stream is live.
- **A count of decode rounds per chunk, K, as the dial.** Per cycle a lane sees `K-1` short gaps and exactly one
  long one, so the long gaps stay the top `1/K` of the distribution and the p95 inter-token latency does not move
  until `K >= 20`, a 240% inflation of the time to first token.
- **A 2,048-token chunk for Flash-Next, to shorten the gap.** It halves the longest gap (1.7 s) but leaves the lanes
  at about 1 tok/s and costs 28.3 to 28.7 s of time to first token on a 34K prompt even with nobody decoding,
  against about 15 s at 8,192 tokens
  ([finding](../findings/2026-10-07-flash-next-prefill-chunk-and-decode-share.md)).

## Consequences

- The longest gap a decoding lane sees under a prefill is one chunk's wall time plus a decode round, and only the
  chunk width moves it. The decode share moves the lanes' rate during a prefill, not that gap.
- The same prompt can end in logits that differ in the last bits, depending on whether lanes were decoding while
  it prefilled. The quality gates use KLD; a test that compares logits bit for bit pins both widths.
- A sequence cannot decode mid-prefill by accident: only a prompt's last chunk leaves it a pending token, and the
  leaf's decode rejects a sequence without one.
- The decode CUDA graphs' staging buffers must be a reservation separate from the prefill scratch. A graph replays
  fixed device addresses, and a prefill chunk landing between two replays would otherwise clobber what the graph
  expects to reread: a silent failure that depends on the batch width.
