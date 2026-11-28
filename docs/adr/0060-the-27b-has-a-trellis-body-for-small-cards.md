---
status: accepted
---

# The 27B gets a second body format, trellis, for the 16 GB card class

The 27B's served NVFP4 artifact cannot be loaded on a 16 GiB card at any context: its text scope alone is 16.03 GiB,
the smallest plan when this was decided (no speculation, no vision, 32K, 8 lanes) was 20.1 GB, and its body is
14.51 GB at 4.76 bits per weight, NVFP4 at 4.5 bpw plus nine BF16 exception matrices. The owner's targets were the
27B on 16 GB Blackwell cards, a context of at least 256K and DFlash2 kept, and a trellis code at about 3 bits per
weight fits with room for both. So the 27B gets a second artifact of the same family, for the 16 GB card class,
whose body format is trellis; the 32 GB NVFP4 artifact, its kernels and its numbers do not change. The rest of the
16 GB plan is ADRs 0061, 0062, 0063 and 0064.

On others' suites turboderp's self-calibrated EXL3 at 3.0 bpw reads a KLD of 0.0257, Unsloth's NVFP4 0.0092, EXL3 at
4.0 bpw 0.0082 and a 12 GiB GGUF Q3 0.0209. The served NVFP4 runs W4A4, FP4 activations too, and FP4 activations
account for half or more of an NVFP4's KLD in two sources; a trellis body runs with 16-bit activations.

- **The format.** The code is EXL3's, the one Flash-Next's experts are stored in
  ([ADR 0044](0044-experts-in-trellis-with-a-k-per-expert.md)): the `mul1` codebook, Hadamard rotation with
  `suh`/`svh` scales, out scales on, bit-compatible with exllamav3's records, a K per matrix from 2 to 6, the output
  head included. The GDN small set (a/b, conv1d, A_log, dt_bias) and the norms stay BF16, and the token embedding
  lives in host memory ([ADR 0062](0062-the-token-embedding-lives-in-host-memory.md)).
- **The body is a repack of the public quant** `turboderp/Qwen3.8-27B-exl3`, branch `SC_3.00bpw_H4` (Apache-2.0,
  like the checkpoint), about 3.0 bpw on average. On this repo's text it is closer to BF16 than the served NVFP4,
  weights only, and DFlash2 keeps its tokens per round on it; its weak spot is memorized Python
  ([the finding](../findings/2026-10-21-exl3-3bpw-27b-on-our-text-and-dflash2.md)). exllamav3 quantizes
  `in_proj_qkv` / `in_proj_z` and `q/k/v` as separate matrices with their own scales, so the 27B's fused planes
  cannot be rebuilt by concatenation: the repack serves them split.
- **The gates are the owner's**, on this repo's KLD set and the Flash-Next study's held-out set: a KLD at most 1.2x
  the served NVFP4's on the same suite; MMLU-Pro (n=281) not significantly below BF16 (McNemar p > 0.05); DFlash2
  tokens per round at least 90% of those on the NVFP4 target; a loop rate on the loop set no worse than the served
  NVFP4's. They admitted this body ([the gates](../findings/2026-10-24-27b-16gb-quality-gates.md)); what admits an
  artifact today is [ADR 0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md).
- **The DFlash2 module is grafted as it is**: its NVFP4 objects are copied byte for byte from the served artifact,
  their provenance in the sidecar, by our own converter, since the reference engine's graft tool refuses any base
  that is not its own NVFP4 recipe.
- **The oracles.** exllamav3's reconstruction is the bit-exact oracle of the tile decode, as for Flash-Next, and the
  BF16 checkpoint, run layer by layer, is the oracle of quality. No reference engine runs this artifact: the
  reference parity of ADRs 0002 and 0010 does not apply to it.
- **The kernels are ours**, the dense trellis linear of decode, verify and prefill and the head, adapted from our
  Flash-Next trellis decode under [ADR 0051](0051-kernel-code-is-ours-adapted-or-vendored.md). The layer code picks
  the linear op by the tensor's format.

## Considered Options

Every lever that keeps NVFP4 was costed and fails:

- **NVFP4, squeezed** (the embedding in host memory, the exceptions and the head in NVFP4, one lane, 32K, no DFlash2,
  no vision): it only fits a card with nothing else on it, and decodes 2.5-3x slower than the trellis body.
- **Weight offload over PCIe:** streaming the bits that do not fit costs 2-5x decode at DFlash2's acceptance; the
  offload wins in the literature need a τ of 10-30 or batches of 100 and more.
- **A smaller model:** Qwen3.8 has no dense model below 27B, and the nearest is two generations older and far weaker
  on agentic coding: Qwen3.5-9B scores 65.6 on LiveCodeBench, the 27B 90.3.
- **GGUF k/i-quants:** at equal VRAM about 2.5x the KLD of EXL3 on this exact model (turboderp's chart), and they
  would need a kernel family of their own.
- **Pruning:** every positive result paid 50-480B tokens of distillation.

Among trellis bodies:

- **2.5 bpw, as Flash-Next's experts:** rejected for 16 GB. A dense model degrades faster than routed experts at
  equal bits: EXL3-SC at 2.2 bpw has a KLD of 0.088 on this model, past the point where AIME collapses on its
  predecessor.
- **3.25 bpw:** the fallback, had the 3.0 bpw body failed the gates. It needs
  [ADR 0061](0061-prefill-attention-reads-the-hq-codec.md) to reach 256K, with less room for the second lane.
- **Our own conversion**, exllamav3's quantizer run on the fused planes with our Hessians and a broad, self-generated
  calibration set: the measured alternative, not the plan, worth it only if the gates fail in ignis or if the Python
  weak spot matters.
- **ExLlamaV3 as the serving engine:** rejected for the reasons ADRs 0043 and 0044 recorded: it is an oracle here,
  not the engine.

## Consequences

- Two artifacts of one family to convert, publish and keep in step: two Hugging Face repos, two catalog entries
  ([ADR 0064](0064-a-catalog-entry-states-its-card-class.md)).
- Prefill on GeForce runs on FP16 tensor cores, without an FP4 copy of the weights, so prefix reuse carries the
  agentic load on these cards. The kernels' prefill rate is measured in
  [a finding](../findings/2026-10-23-dense-trellis-kernels-on-the-5090.md), and the decode estimate made from them in
  [another](../findings/2026-10-24-27b-16gb-speed-estimate.md).
- The body's capacity and quality are measured on the RTX 5090. Its decode speed on a 16 GB card is stated as an
  estimate until a real 16 GB card measures it; there is none in the project.
- Two phases: the first is this artifact on the 16 GB card class; the second, Flash-Next on 16 GB, the abliterated
  variant, the speed gate on real hardware and 12 GB; after that, images.
