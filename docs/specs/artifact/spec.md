# Artifact binder / materializer — spec

The generic `.ninfer` container **reader** is done (kernel-port 02, now
closed): `Reader::open` + storage-layout geometry + mmap + 4096-aligned
direct I/O, verified against the real 19 GB `qwen3_8_27b_nvfp4full-v2.ninfer`
(1,325 objects). This feature builds the **semantic / materialization layer**
on top of the reader (ADR 0002): the binder consumes *every* object at bind
time (an unconsumed object is a load failure), the materializer places
NVFP4 / BF16 tensors into VRAM, and the typed binding exposes them to the
engine. Plus: frontend object extraction (tokenizer / chat template) and
tensor checksum validation against the `conversion.json` / `graft.json`
sidecars.

## v1 scope (priority order)

1. Binder + materializer + typed binding (device materialization) — `artifact-01`
2. Frontend object extraction (tokenizer + chat template + generation config) — `artifact-02`
3. Tensor checksum validation against `conversion.json` / `graft.json` sidecars — `artifact-03`

## Acceptance

- The binder consumes the full 1,325-object manifest of
  `qwen3_8_27b_nvfp4full-v2.ninfer`; **zero unconsumed objects** (ADR 0002).
- NVFP4 / BF16 tensors materialize into VRAM at the geometry / alignment the
  reader computed (rowsplit / blockscale / rowscale layouts).
- Frontend objects (`tokenizer.json`, `tokenizer_config.json`,
  `chat_template.jinja`, `generation_config.json`) extracted and round-tripped
  (confirms the §7 risk that the frontend is carried by the container).
- Tensor checksums match the `conversion.json` / `graft.json` sidecars
  (offline verification step).

## References

- Design: `docs/design/ignis-v1.md` §2 (Artifact loader), §7 (frontend risk).
- ADRs: 0002 (load `.ninfer` artifact), 0005/0007 (performance-first, gate).
- Reader: `crates/artifact/src/lib.rs` (ticket kernel-port 02, closed).