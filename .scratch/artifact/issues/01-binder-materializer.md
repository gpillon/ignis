# 01 — binder + materializer (device materialization)

Status: ready-for-agent
GitHub: #4
Blocked by: (none — the `.ninfer` reader is done, kernel-port 02)

Port the reference's per-model **binder** (`binder.h`), **materializer**
(`materializer.h`), and **typed binding** (`typed_binding.h`) into
`ignis-artifact`, on top of the generic reader:

- The binder consumes **every** object at bind time — an unconsumed object
  is a load failure (ADR 0002). It owns the per-model semantic layer the
  generic reader deliberately leaves out (format spec §6.1).
- The materializer places NVFP4 / BF16 tensors into VRAM at the geometry /
  alignment the reader computed (rowsplit / blockscale / rowscale layouts).
- The typed binding exposes the materialized tensors to the engine
  (`ignis-core`).
- Load `qwen3_8_27b_nvfp4full-v2.ninfer` (1,325 objects: 281 NVFP4 + 9 BF16
  + 6 frontend).

## Acceptance

- Load succeeds; the full object manifest is consumed (zero unconsumed
  objects, ADR 0002).
- NVFP4 / BF16 tensors materialize into VRAM at the reader-computed
  geometry.
- The typed binding is consumable by `ignis-core`.
