---
status: accepted
---

# ADR 0065 — the 27B's BF16 exception tensors can be held in FP8, converted at load

## Status

Proposed (2026-10-21, GitHub #341, spec [prefill-5090/01](../specs/prefill-5090/01-the-prefill-fixed-part.md) lever 1).
Owner, the same day: no conversion at load if it lengthens the load, else the FP8 goes into the artifact. Measured:
it does not (finding [2026-10-21-the-bf16-exception-tensors-in-fp8](../findings/2026-10-21-the-bf16-exception-tensors-in-fp8.md)).

Related:
- [ADR 0030](0030-device-memory-reserved-at-load.md): device memory is a plan line;
- [ADR 0051](0051-kernel-code-is-ours-adapted-or-vendored.md): the kernels extended here are ours (tier 1);
- [ADR 0062](0062-the-token-embedding-lives-in-host-memory.md): a placement of its own in the binder, the same shape.

## Context

The 32 GB artifact keeps nine matrices in BF16 among the NVFP4 body: the fused attention projection
`attention/query_key_gate_value` [14336][5120] of GQA layers 3-23, `attention/output` [5120][6144] of GQA layers 3 and
7, and `gdn/output` [5120][6144] of GDN layer 4. They take 1.07 GB of VRAM, and in prefill the vendored BF16 GEMMs run
them at ~190 TFLOPS, 5.6 µs of every token's ~100. Our FP8 W8A8 GEMM (the image model's, spec image-model/02) runs at
500-535 TFLOPS on the block-scaled `mxf8f6f4` MMA with unit scales, and the FP8 row-scale format already has a GEMV and
a tensor-core route with BF16 activations (`ignis_fp8_linear`, Flash-Next's).

## Decision

1. **The nine are held on the device in FP8 row-scale** (`FP8_E4M3FN_ROW_BF16S` / `row-scale-v1`): per row a BF16
   scale, the least BF16 >= amax / 448, and E4M3 codes rounded to nearest even. The fused projection is four payloads
   one after another at 256-byte offsets, q | k | gate | v in the stored row order, so every route takes a projection
   as its own weight.
2. **Converted at load, on the GPU.** The binder places each in a slot of its conversion's size (`DeviceEncoding::
   Fp8Rows`); the materializer uploads the stored BF16 into one staging buffer and quantizes it into the slot
   (`ignis_fp8_quantize_rows`), the same bytes `ignis_artifact::fp8` writes on the host. The artifact does not change.
3. **The leaf dispatches on the format.** `attn_input_proj` and `linear_add` (ours) gain an FP8 arm: up to 8 tokens, or
   under `A16Only`, BF16 activations (the FP8 linear's GEMV or tensor-core route; the residual add in its epilogue);
   past 8 under `AllowA8`, the engine's policy for these weights, the activations quantized per token to E4M3 in the
   call's workspace, then one W8A8 launch over the four payloads (`ignis_image_gemm_w8a8_grouped`) or the W8A8 residual
   add. A decode round of up to 8 lanes and a one-lane verify round are on the first route; a verify round of two lanes
   or more ((window + 1) x lanes rows) and every prefill chunk on the second.
4. **A config field, `model.exception_weights: bf16 | fp8`**, the 27B's alone, **`bf16` by default** until the owner
   decides: FP8 is faster at the same 4K KLD, but two pre-registered decision fixtures read differently under it
   (`decide_entailment_gpu` case 1, `decide_scalar_gpu`'s negative balance), and the second does with BF16
   activations too, so it is the FP8 weights' own error (2.6%, E4M3's mantissa) and not the activation codes.

## Considered options

- **FP8 in the artifact.** The owner's fallback if the load grew. Costs a new 32 GB artifact users re-download, a new
  content hash, for a conversion the GPU does in milliseconds. Not needed.
- **FP8 in prefill only, BF16 kept for decode.** Either both copies (+0.53 GB) or a per-chunk conversion into a 73 MB
  scratch (~0.5 µs per token at 2,048, the scratch past the vision arena). Decode would stay bit-identical; the VRAM
  would go the wrong way.
- **Block-scaled activations (MXFP8, an E8M0 scale per 32).** Not needed: per-token scales measured within the KLD's
  noise.

## Consequences

- Under `fp8`: the weights line drops 534.6 MB and the KV pool takes it; cold prefill is ~3% faster at 14K-105K;
  decode rounds take as long; the 4K KLD is unchanged; two decision fixtures move (above).
- Greedy decode text changes against a BF16 load (2 of 8 prompts identical): the weights and the prefill's activations
  are other numbers. Tests that bind the 27B directly name `fp8_exceptions: false` and keep today's numerics, as the
  server's default does; `fp8_exceptions_gpu` and `dflash2_round_gpu`'s FP8 test cover the FP8 load.
- The KV-disk blob identity does not name the encoding: by its own rule ([ADR 0029](0029-cross-request-state-reuse.md)) a knob that moves numerics but
  leaves the state's meaning, as the prefill chunk width and the prompt attention's precision do, is not identity.
- The activation quantization takes up to 2,048 x 6,144 codes in the layer scratch: +5.5 MB on a text-only load at
  2,048, nothing with vision on (the encoder sizes the arena); and the verify rounds' in the decode-graph scratch:
  +315,904 B (82,125,920 -> 82,441,824 at the serving shape).
