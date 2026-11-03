# The NVFP4 converter reproduces the served 27B: 831 objects byte-equal, 135 weights equal, 68 of 135 divisors not

- Kind: experiment
- Status: current
- Observed: 2026-11-03
- Last verified: 2026-11-03
- Scope: models / `tools/27b-nvfp4-converter/` run on the base BF16 `Qwen/Qwen3.8-27B` and the DFlash2 drafter,
  packed with `ignis-artifact-pack` and compared object by object with the served `qwen3_8_27b_nvfp4full-v2.ninfer`
- Related: [GitHub #469](https://github.com/gpillon/ignis/issues/469), [GitHub #449](https://github.com/gpillon/ignis/issues/449),
  [GitHub #468](https://github.com/gpillon/ignis/issues/468), [GitHub #470](https://github.com/gpillon/ignis/issues/470),
  [ADR 0074](../adr/0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md),
  [ADR 0078](../adr/0078-the-built-in-catalog-names-each-entrys-license-and-may-ship-restricted-weights.md)
- Superseded by: none

## Question

Before the converter builds the Swift artifact (#470), does it reproduce the served 27B NVFP4 artifact from the base
checkpoint? Ticket #469 asks for three tiers: every object that is neither calibrated nor taken from the third-party
quant must be byte-equal (binding); the 135 parents that were already local are expected byte-equal, and any difference
is reported with its size; the 112 parents the served artifact took from the third-party NVFP4 quant are expected to
differ, and each one's reconstruction error is reported beside the served one's.

## Evidence

**Run.** One machine (the RTX 5090 host), the tool at the 449 integration branch. Base BF16 `H:/models/Qwen3.8-27B`
(Qwen/Qwen3.8-27B at `1d4bf0f2`, pins verified before each step); drafter `incoai/Qwen3.8-27B-DFlash2` at `dedf8df6`
(file list checked, SHA-256 equal to its LFS oid). Weights id `nvfp4full-local-469`, recipe
`qwen3_8_27b_nvfp4full-v2-local`. Nothing from this run ships.

| Step | Time | VRAM peak (allocated / reserved) | Host peak (working set / commit) |
|---|---|---|---|
| pins check, each step | 41-57 s | | |
| calibrate, reference GDN rule | 61 s (prefill of 2,690 tokens 50.9 s) | 5.00 / 6.17 GiB | 5.37 / 17.31 GiB |
| calibrate, `--gdn-kernel installed` | 64 s | 5.00 / 5.62 GiB | 5.27 / 16.62 GiB |
| run (281 NVFP4 parents, tree 19,406,582,553 B) | 196 s | 5.66 / 10.28 GiB | 6.83 / 17.43 GiB |
| pack (debug `ignis-artifact-pack`, 1,325 objects, 19,410,923,780 B) | about 86 s (text unit 77.5 s) | CPU | |
| verify, `--device cuda` | 283 s, VERIFY PASS | 5.66 / 10.28 GiB | 8.10 / 17.43 GiB |
| compare (CPU) | 484 s, TIER 1 PASS | CPU | 5.62 / 9.91 GiB |

Raw material, all under `.scratch/469/`: `calibrate-base.log`, `calibrate-base-installed.log`, `run-base.log`,
`pack-base.log`, `verify-base.log`, `compare-base.log` (each with its `.exit`), `calibration.json`,
`calibration-installed.json`, `verify.json`, `compare.json`, `compare.txt`, `divisors.txt`, `divisors.py`, `step.py`.
The scratch artifact was in `H:/nvfp4-469/`.

**Container size.** The built `.ninfer` is 3,981,312 B larger than the served one (19,406,942,468 B). That is exactly
the difference in payload offset: ignis's packer starts payloads at 4,194,304 B, ninfer's at 212,992 B. The payloads
match.

**Tier 1 (binding).** `compare.txt`: 831 of 831 objects byte-equal (6,023,474,017 bytes): the base-direct objects, the
BF16 exceptions, the embedding, the output head, the draft head, the vision tower, the frontend resources and the 66
DFlash2 objects. The converter's real-artifact test (CPU, from the base and the drafter) binds the same objects:
765 checkpoint objects in 25-30 s (6.6 GiB working set, 10.7 GiB commit) and the 66 DFlash2 objects in 94 s
(8.1 / 11.4 GiB).

**Tier 2 (135 parents that were already local).** The 135 packed weights are byte-equal (0 differing words). Their 135
activation divisors, calibrated with transformers' reference GDN rule: 67 byte-equal, 68 differ. The relative
difference of the 68: median 3.97e-3, maximum 0.2675 (`text/layers/63/mlp/down_projection`, served 6.754 against
local 8.561). Others above 5%: layer 62 mlp down projection +8.2%, layer 63 attention output projection +6.1%. The full
list is `compare.txt` lines 3-71. Because the weights are byte-equal, the weights' reconstruction error against the BF16
is the same on both sides; only the divisors differ.

**The GDN kernel explains part of the divisor difference** (`divisors.txt`). Calibrating with `--gdn-kernel installed`
(the fla kernel, recorded as `gdn_kernel: installed`) instead of transformers' torch reference rule gives 72 of 135
divisors equal, median 0, maximum 8.43e-2 (reference rule: 67 equal, median 3.97e-3, maximum 0.2675).

**Tier 3 (112 parents the served artifact took from the third-party quant: MLP gate_up and down, layers 0-55).**
Relative reconstruction error against the BF16:

| | Mean | Max |
|---|---|---|
| served | 0.108259 | 0.153464 |
| local | 0.094938 | 0.095096 |

The local parent is below the served one on 112 of 112. Their divisors: 0 byte-equal (expected, a different
calibration); |relative difference| median 0.343, maximum 7.87 with the reference rule, 0.342 and 7.82 with the
installed kernel.

## Finding

Observed:

- The converter reproduces every object it does not calibrate and every weight it encodes locally, bit for bit, from the
  base checkpoint and the drafter: tier 1 is 831 of 831, and the 135 local parents' weights are 135 of 135.
- The binding tier passes. The encoder is deterministic on these parents.
- The calibrated activation divisors do not reproduce exactly: 68 of 135 differ with the reference GDN rule, 63 with the
  installed kernel (72 equal). The largest difference is 26.75% (reference) and 8.43% (installed).
- On the 112 third-party parents the local encode has a lower and tighter reconstruction error than the served one
  (mean 0.0949 against 0.1083, max 0.0951 against 0.1535).

Inferred:

- The divisor differences come from the calibration forward, not from the encoder: the weights are equal, and switching
  the GDN implementation alone moves 5 divisors to equal and cuts the maximum from 0.2675 to 0.0843. Part of the rest
  is probably other numerics of the calibration forward (transformers 5.17.0, torch 2.13.0+cu130, causal_conv1d
  falling back to its PyTorch reference); this run did not isolate it.
- The lower reconstruction error on tier 3 is a property of the encode against the BF16, not a measure of model
  quality.

## Implications

- The converter is proven enough to build Swift (#470) on the tier-1 and tier-2-weights results. The Swift build uses
  `--gdn-kernel installed`, the calibration closer to the served artifact's divisors; the owner may choose otherwise,
  and a rebuild costs about 10 minutes of GPU.
- Tier 2's divisor difference goes to the owner as this finding, not as noise, as the ticket asks. The ticket is done
  when the owner has seen it.
- An artifact built this way differs from the served one in 63 to 68 divisors by construction; a reader comparing a
  future build with the served artifact should expect that and compare the weights, not the divisors.

## Limits and unknowns

- One machine, one run per calibration variant. Calibration determinism under the fla kernel was not measured.
- The unexplained remainder of the divisor difference (63 of 135 with the installed kernel) is not attributed to a
  cause.
- The KLD of this scratch artifact against the BF16 was not measured (not asked), so the divisor differences and the
  lower reconstruction error on tier 3 say nothing yet about output quality.
- The scratch artifact is deleted once this finding is in, unless the owner asks to keep it.

## Follow-ups

- The owner's decision on the Swift build's calibration (`--gdn-kernel installed` or the reference rule) and on whether
  tier 2's divisor difference needs more than this report: [GitHub #469](https://github.com/gpillon/ignis/issues/469)
  and [#470](https://github.com/gpillon/ignis/issues/470).
