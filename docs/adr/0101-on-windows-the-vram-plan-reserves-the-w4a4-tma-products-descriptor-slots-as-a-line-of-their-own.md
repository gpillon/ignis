---
status: accepted
amends: [ADR-0030]
---

# On Windows the VRAM plan reserves the W4A4 TMA products' descriptor slots as a line of their own

[ADR 0030](0030-device-memory-reserved-at-load.md) reserves at load the device memory a request uses, in the named
lines of `ignis.runtime.vram_plan`. On Windows the NVFP4 body's W4A4 TMA products take their tensor maps through a
pointer, and upstream's launchers allocated a copy of them from the device's memory pool, copied the 512 bytes into it
and freed it before every product: an allocation the plan could not state, whose first one cost the first request
after a load about 110 ms on the RTX 5090 (spec #646, GitHub #650).

- **A fifteenth plan line, `tma_descriptors`, placed after `trellis_dense`**: the device slots that keep the
  products' maps, 512 bytes for each such product of a prefill traversal at every width the route takes, a multiple of
  256 from 1,024 tokens up to the widest chunk the load is handed, the wider of the prefill chunk and the solo width.
  A request's first prefill finds them in place and allocates nothing a later one does not.
- **It is 0** on Linux, which passes the maps by value; on the trellis body, Flash-Next and a card below compute
  capability 12; below a 1,024-token chunk; and under the measurement switch `IGNIS_TMA_DESCRIPTOR_CACHE=0`, which
  keeps upstream's per-launch upload for a comparison.
- **The bytes the products write are unchanged.** The KV pool gives up the line's bytes: 0.6 MB at a 2,048-token
  chunk, 3.7 MB at 8,192.
