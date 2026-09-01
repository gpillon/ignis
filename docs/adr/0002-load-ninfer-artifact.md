# ADR 0002 — The engine loads `.ninfer` artifacts directly

## Status

Accepted (2026-09-02, grilling session).

## Context

The model weights (and the frontend/tokenizer objects) live in the native NInfer
`.ninfer` container (1,325 objects: 1,259 base + 66 grafted DFlash2; 281 NVFP4
tensors, 9 BF16 exception tensors, ~19.4 GB). The container format is
code-defined (no external spec); the reference reader is a self-contained C++
module (`reader`, `binder`, `materializer`, `storage_layouts`).

Options:
- (a) port the artifact reader to Rust; the engine loads `.ninfer` files directly
- (b) offline converter `.ninfer` → safetensors; the engine reads the standard format

Both require writing the container reader once; (b) additionally moves it into a
one-off tool and adds a 19 GB conversion step.

## Decision

(a). The artifact stays the source of truth. The reader is ported as the engine's
model-loading layer.

## Consequences

- No conversion step; the existing `qwen3_8_27b_nvfp4full-v2.ninfer` file is the
  deployment artifact from v1 onward.
- The binder's "every object must be consumed at bind time" check becomes a
  first-class correctness gate (a future artifact with missing/extra objects fails
  loudly instead of loading silently).
- Coupling to the NInfer family's container format; format changes upstream are a
  supported input, not a surprise.
- Sidecars (`conversion.json`, `graft.json`) document object provenance and are
  consumed by the loader.