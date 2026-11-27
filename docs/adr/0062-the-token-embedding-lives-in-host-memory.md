---
status: accepted
---

# The 27B's token embedding lives in pinned host memory, on every card

The 27B's token embedding is bound into mapped pinned host memory, on every card class, and the gather reads it
there through the mapped device pointer on every route: prefill, the decode graphs and the drafter, which embeds
through the same table. A token reads one row of a 1.35 GB table and the rest stays unread, while those bytes of
VRAM buy about 0.44 bits per weight of body on a 16 GB card and about 146K tokens of hq KV on the 5090. Other
engines keep the table off the device: llama.cpp keeps it on the CPU by rule, and an engine serving this exact
model moved it to mapped pinned memory with identical perplexity.

- **The VRAM plan drops the table and the host plan holds it** (ADR 0030). The 27B's host plan is a whole one:
  beside the table it charges the host retained slots, the KV-RAM arena and KV-disk's staging, which the 27B pinned
  without a check before. A start that cannot hold them all beside `model.host_margin_bytes` is refused, as
  Flash-Next's is, naming the 27B's own flags.
- **The table has a placement of its own in the binder**: a host-mapped allocation its file span is read straight
  into. The gather takes a raw table pointer, which does not care whether it is device or mapped host memory.
- **The 32 GB artifact and its W8G32 format are unchanged.** The format is the artifact's: for the small-card body
  it was left open between W8 and BF16 (ADR 0060), and in host memory BF16 costs only RAM, 2.54 GB, and is exact.

## Considered Options

- **Keep it in VRAM.** Rejected, for the bytes above.
- **Copy the needed rows per step.** Rejected: the token ids are on the device, inside captured graphs, so the host
  does not know them in time.

## Consequences

- Each decode step reads one row per lane over PCIe, microseconds of latency, and a 1,024-token prefill chunk
  gathers about 5.6 MB in about 1 ms on a PCIe Gen 3 host. Against the table in VRAM, on the 5090
  ([measured](../findings/2026-10-21-token-embedding-in-host-memory.md)): decode -0.1% at one lane and about -0.4%
  at eight, cold prefill -0.15% to -0.7% from 14K to 105K tokens and -1.9% at 8K.
- WDDM counts the table as shared GPU memory, not dedicated, and a loaded host slows a build with the table in VRAM
  and one with it in host memory alike.
