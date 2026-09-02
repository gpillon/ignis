# 02 — GDN state management (resumable at frontier / checkpoint)

Status: needs-triage
GitHub: #11
Blocked by: #9 (core-01), #5 (kernel-abi-01)

GDN (linear-attention) state management in `ignis-core`
(`docs/design/ignis-v1.md` §2, `CONTEXT.md` "GDN state"):

- The recurrent state of the linear-attention (GDN) layers is resumable
  **only** at checkpoint / frontier boundaries.
- The KV-RAM host tier must respect this boundary — a snapshot mid-prefill
  is invalid for GDN layers.

## Acceptance

- GDN state is resumable at frontier / checkpoint boundaries.
- The host tier honors the boundary (no invalid mid-prefill snapshots for
  GDN layers).
