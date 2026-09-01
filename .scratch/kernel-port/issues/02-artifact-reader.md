# 02 — artifact reader port

Status: ready-for-agent

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