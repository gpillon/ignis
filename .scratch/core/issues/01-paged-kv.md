# 01 — paged KV cache + block table (VRAM)

Status: needs-triage
GitHub: #9
Blocked by: #4 (artifact-01)

The paged KV cache (VRAM) + block table for `ignis-core`
(`docs/design/ignis-v1.md` §2):

- The KV pool is **auto-sized from free VRAM** (~12–13 GB budget after
  weights + graphs + runtime).
- The block table maps logical KV blocks to physical VRAM blocks.
- KV tensor geometry / layout come from the artifact materialization
  (`artifact-01`).

## Acceptance

- The KV pool auto-sizes from free VRAM (≈12–13 GB budget).
- The block table maps logical → physical blocks; no OOM under the N=8
  load.
