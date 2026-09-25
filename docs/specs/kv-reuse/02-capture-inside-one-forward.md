# 02 - capture a reuse boundary inside one forward

GitHub: #272

Every place a request publishes state — a reuse boundary (spec decide/16),
a prompt checkpoint's generation opener — is today a **chunk split**: the
prefill is cut there so the mutable state at the boundary exists as the end
state of a traversal. A traversal costs ~19 ms fixed whatever its token
count, so reuse turns a 1,024-token prompt's one traversal into three and
costs 74.7 ms of TTFT
(`docs/findings/2026-09-18-prompt-reuse-tax-on-short-ttft.md`). Spec
decide/16 adds more boundaries per request, each one more split.

## Problem Statement

A boundary is always a whole number of KV pages, and a KV page is 64 tokens.
The chunked GDN kernel runs in 64-token chunks (`kChunkSize == 64`, asserted
in the vendored `gated_delta_net/chunked` stages), so the recurrent state at
any boundary is a chunk-start state the kernel's state-passing stage already
computes on its way through the prompt. It is thrown away because nothing
asks for it. Two traps make it less free than it looks:

- the stage writes each chunk's state to `h_chunk` in **BF16**, while the
  state it carries is **FP32** (`state_in`/`state_out`). A BF16 copy is not
  the state a split would have captured; vLLM measured the same round trip
  failing ("bfloat16 roundtrips fail (max diff = 2.0)");
- a retained slot holds more than the GDN recurrent state: the conv state,
  the hq-e8-2b residual window (sink and recent ring, ~34 MiB), and on a
  drafter load the drafter's window. Each has to be captured as it stands
  *at the boundary*, not at the end of the forward.

SGLang does this (`_force_track_h`, reading the intermediate state from the
chunked kernel within one extend) and vLLM RFC #52959 proposes it.

## Solution

When a traversal crosses a boundary it was asked to publish, capture every
section of the mutable state at that boundary into the retained slot the
scheduler names, **without ending the traversal there**. The GDN part is the
FP32 state at the boundary's chunk start, written out beside the BF16
`h_chunk` only for the chunks that are boundaries; the other sections are
reconstructed or copied as they stand at that position.

## Implementation Decisions

- Correctness is **bit-exactness against a split capture** at the same
  boundary, section by section, under BF16 and hq-e8-2b. The publisher's own
  answer may differ from its split run at near-ties, as any change of chunking
  may (ADR 0029, Consequences) — recorded, not failed.
- The vendored stage is changed as a recorded, tested patch, the way ADR
  0037 records one.
- Where a section cannot be captured mid-traversal exactly, the boundary
  falls back to today's split for that request, never to an approximate
  capture.

## Acceptance

1. A GPU test captures at a boundary inside one traversal and at the same
   boundary with a split, and every section is bit-identical, under BF16 and
   hq-e8-2b, with and without the drafter.
2. A request with k boundaries runs the same number of traversals as with
   none (asserted on the chunk count the scheduler records).
3. The 1,024-token TTFT with reuse on, measured before and after, loses the
   split tax, recorded as a finding with a README row.
4. Every reuse test (retained prefixes, checkpoints, chains, KV-RAM,
   multimodal) passes unchanged.
5. `cargo test` passes workspace-wide.

## Out of Scope

- Where boundaries are placed (spec decide/16).
- Lossy capture (Tail-Replay, DASC, int8 checkpoints).

## References

- ADR 0029, ADR 0037; spec decide/16 (#270).
- `docs/findings/2026-09-26-prefix-reuse-prior-art-for-decisions.md` §3
  (SGLang, vLLM #52959).
