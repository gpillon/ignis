# 02 — artifact reader port

Status: in progress

Port the reference stack's `src/artifact/*` module (reader, binder,
materializer, storage layouts) into `ignis-artifact`:

- Parse the `.ninfer` container: header, object index, tensor encoding
  (NVFP4 / BF16 / FP8 layouts, alignment rules).
- The binder consumes **every** object at bind time — an unconsumed object is a
  load failure (ADR 0002).
- Load `qwen3_8_27b_nvfp4full-v2.ninfer` (1,325 objects; 281 NVFP4 + 9 BF16
  tensors + frontend objects).
- Verify the `frontend` object set actually carries tokenizer + chat template
  (open risk in docs/design/ignis-v1.md §7 — confirm during the port).
- Acceptance: load succeeds, object manifest fully consumed, tensor checksums
  match the `conversion.json` / `graft.json` sidecars.

## Progress (2026-09-02)

**Done — generic container reader** (the "reader" proper), in
`crates/artifact`:

- `Reader::open` — 16-byte prefix check (rejects v1 with a migration hint),
  closed-JSON directory parse (exact member sets), per-object geometry /
  ordering / alignment / bounds / duplicate-name validation, name index.
- Storage-layout geometry ported 1:1 (`contiguous-le-v1`, `row-split-k128-v1`,
  `blockscale-k16-m128x4-v1` NVFP4, `row-scale-v1`) with checked arithmetic.
- Payload spans served from an `mmap` of the whole file; 4096-aligned direct
  I/O (`FILE_FLAG_NO_BUFFERING` + `SetFilePointerEx` + `ReadFile` on Windows;
  `O_DIRECT` + `pread` on Unix), mirroring the reference `MappedFile`.
- `ignis-artifact-inspect` CLI: prints identity, object inventory, format
  histogram, and resource entries; `--find NAME` for a single descriptor.
- **Verified against the real 19 GB artifact**
  (`qwen3_8_27b_nvfp4full-v2.ninfer`): 1,325 objects (1,319 tensors +
  6 resources), identity `qwen3.8-27b/nvfp4full`; all 6 frontend resources
  present (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
  `generation_config.json`, `preprocessor_config.json`,
  `video_preprocessor_config.json`) — the §7 risk (tokenizer / chat template
  carried by the container) is **confirmed**. NVFP4 tensors use the blockscale
  layout. 18 unit + integration tests pass; clippy clean.

**Remaining (deferred to the engine / ticket 03 and later):**

- Per-model **binder** (`binder.h`), **materializer** (`materializer.h`), and
  **typed binding** (`typed_binding.h`) — the semantic / device-materialization
  layer. The generic reader is deliberately minimal (per the format spec §6.1);
  these own model semantics and belong to the decode-step work.
- **Tensor checksum verification** against the `conversion.json` /
  `graft.json` sidecars (acceptance item) — offline verification step.