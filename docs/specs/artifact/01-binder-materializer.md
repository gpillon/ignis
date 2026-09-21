# 01 — binder + materializer (device materialization)

GitHub: #4

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
- Verification: `cargo test -p ignis-artifact` (CPU, default features) +
  `tests/real_artifact.rs` (binds all 1,325 objects when the 19 GB artifact
  is present). The `CudaDevice` / real-artifact VRAM upload runs only with
  the `cuda` feature + `IGNIS_TEST_CUDA=1` + a free GPU (ADR 0006); full
  `CpuDevice` materialization is gated behind `IGNIS_TEST_FULL_MATERIALIZE=1`
  (heavy: multi-GB host alloc) and is not in the default gate by design.

## References

- Design: `docs/design/ignis-v1.md` §2 (Artifact loader), §7 (frontend
  risk).
- ADRs: 0001 (flat C ABI), 0002 (load failure on unconsumed objects), 0006
  (exclusive GPU).
- Reader: `crates/artifact/src/lib.rs` (ticket kernel-port 02, closed).