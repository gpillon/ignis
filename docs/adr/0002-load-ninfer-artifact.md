---
status: accepted
---

# The engine loads `.ninfer` artifacts directly

The model's weights and its frontend and tokenizer objects live in NInfer's native `.ninfer` container, a format
defined by its reader's code, with no external spec. The engine loads `.ninfer` files directly: the reference's
artifact reader is ported to Rust as the engine's model-loading layer, and the artifact stays the source of truth.

## Considered Options

- **An offline converter from `.ninfer` to safetensors, the engine reading the standard format**: rejected. Either
  way the container reader is written once; the converter moves it into a one-off tool and adds a conversion step of
  the whole artifact, 19 GB for the 27B.

## Consequences

- There is no conversion step: the `.ninfer` file is the deployment artifact.
- The binder's check that every object is consumed at bind time is a first-class correctness gate: an artifact with
  missing or extra objects fails loudly instead of loading silently.
- ignis is coupled to the NInfer family's container format: a format change upstream is a supported input, not a
  surprise.
- The sidecars (`conversion.json`, `graft.json`) document object provenance and are consumed by the loader.
