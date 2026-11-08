---
status: accepted
---

# ADR 0062 — the 27B's token embedding lives in pinned host memory, on every card

## Status

Accepted (2026-10-21, after the owner's review: "accetto tutto"). Proposed the same day. Owner decision in the 16 GB grill: on every card, after an A/B on the 5090.

Related:
- [ADR 0030](0030-device-memory-reserved-at-load.md): device memory is reserved at load;
- [ADR 0060](0060-the-27b-has-a-trellis-body-for-small-cards.md): the 16 GB body;
- [ADR 0061](0061-prefill-attention-reads-the-hq-codec.md): prefill without prompt planes.

## Context

The token embedding, `text/token_embedding` `[248320, 5120]` in W8G32, is 1.35 GB in VRAM. Each token reads one row of
5,440 bytes; the rest of the table stays unread. The drafter embeds through the same table.

What 1.35 GB buys:
- on a 16 GB card, ~0.44 bits per weight of body;
- on the 5090, ~146K tokens of hq KV.

Other engines keep it off the device:
- llama.cpp keeps `token_embd` on the CPU by rule;
- an engine serving this exact model moved it to mapped pinned memory with identical perplexity (kekzl/imp PR #2573).

What ignis already has:
- mapped pinned memory read by kernels (the Flash-Next expert pool);
- host placement in the binder (`retain_tensor_on_host`);
- a gather kernel that takes a raw table pointer, which does not care whether the pointer is device or mapped host
  memory.

## Decision

1. **The 27B's token embedding is bound into mapped pinned host memory**, on every card class.
   - The gather reads it through the mapped device pointer.
   - Every route reads it there: prefill, decode graphs and the drafter.
   - The 32 GB artifact and its W8G32 format are unchanged.
   - The 16 GB artifact's embedding format, W8 or BF16, is open (ADR 0060). In host memory BF16 costs only RAM,
     2.54 GB, and is exact.
2. **The VRAM plan drops the table; the host plan holds it.** The 27B's host plan refuses a start that cannot pin it,
   as Flash-Next's does.
3. **An A/B on the 5090 before it lands** (owner). Proposed by the agent as its acceptance: decode tok/s (1 and 8
   lanes) and cold prefill TTFT (8K, 46K) within noise of today's.

## Considered options

- **Keep it in VRAM.** Rejected, for the bytes above.
- **Copy the needed rows per step** (`cudaMemcpyAsync` of the token ids' rows). Rejected: the ids are on the device,
  inside captured graphs, so the host does not know them in time.

## Consequences

- Each decode step reads one row per lane over PCIe, a latency of microseconds. A 1,024-token prefill chunk gathers
  ~5.6 MB: under 1 ms even on the PCIe Gen 3 host.
- How WDDM treats a 1.35 GB mapped pinned allocation is unverified. The A/B in point 3 includes a run with the desktop
  under load.

**As built** (2026-10-21, GitHub #335), where the decision left a choice:

- **A placement of its own.** The binder places the table in a host-mapped allocation (`cudaHostAllocMapped |
  cudaHostAllocPortable`) and reads its aligned file span straight into it; the bound weight's pointers are the mapped
  device pointer's. The leaf asks the runtime where each bound tensor lives, so `vram_bytes` leaves the table out and
  `host_mapped_bytes` counts it.
- **The 27B's host plan is a whole one.** Beside the table it charges what the 27B already pinned without a check: its
  host retained slots, its KV-RAM arena and KV-disk's staging. A start that cannot hold them all beside
  `model.host_margin_bytes` is now refused (~18 GB available under the make defaults); the refusal names the 27B's own
  flags.
- **Measured** ([finding](../findings/2026-10-21-token-embedding-in-host-memory.md)): the weights line −1,350,860,800
  bytes, ~146K more KV tokens; decode −0.1% at one lane, ~−0.4% at eight; cold prefill −0.15% to −0.7% from 14K to
  105K, −1.9% at 8K; the gather ~1 ms per 1,024-token chunk. WDDM counts the table as shared memory, and a loaded host
  slows main and the branch alike.
