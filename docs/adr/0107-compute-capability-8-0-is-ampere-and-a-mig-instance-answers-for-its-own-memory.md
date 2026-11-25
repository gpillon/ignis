---
status: proposed
amends: [ADR-0079]
---

# Compute capability 8.0 is Ampere, and a MIG instance answers for its own memory

ignis runs on RTX cards, and its releases are built for `86;120a`. The data-center Ampere cards, the A100 and
the A30, are compute capability 8.0. A plain `80` entry in `IGNIS_CUDA_ARCHITECTURES` gives them machine code
that also runs on every later 8.x, so it covers the RTX 30 and RTX 40 too, and the leaf compiles for it
unchanged: its routes are chosen by the compute capability's major (8 or 12), never by 8.6 itself. Two things
still stood between such a build and those cards, and both are decided here.

- **8.0 is Ampere.** [ADR 0079](0079-a-catalog-entry-states-the-oldest-gpu-generation-it-loads-on.md) maps 8.6
  and 8.7 to Ampere and gives every other capability no generation, which refuses nothing by generation. On an
  A100 the NVFP4 entries (`generation: blackwell`) were then offered and taken, and refused only by the route
  checks at the load, after a switch had torn the loaded model down. `GpuGeneration::of` now maps 8.0 to Ampere
  as well, so they are refused before anything stops (`409 model_above_gpu_generation`), as on an RTX 3090. The
  release list stays `86;120a`; a card the build has no code for is still refused at start (GitHub #457).
- **A MIG instance answers for its own memory.** The leaf reads a device's memory, name and processes from NVML
  before the CUDA context exists (GitHub #210), through the handle of its PCI bus id. With MIG enabled (the A100,
  the A30, the H100 and later data-center cards) that bus id names the physical card: its memory is every
  instance's, and an unprivileged process, as in a container, may not read it at all. The card class, the VRAM
  plan and the context measurement then read a card that is not the one the process runs on. On a card with MIG
  enabled the leaf now asks the instance instead (`kernel/src/nvml_mig.h`): the one whose UUID (`MIG-` and the 16
  bytes the CUDA runtime reports for the device) is the device's; else the only instance the process can see,
  which is what a container is given; else, as before, the physical card. A card without MIG, or a driver
  without the MIG queries, takes the same path as before. A 3g.40gb instance of an A100 reads 39.25 GiB, the
  32 GB class.

## Considered options

- **Read the free memory from `cudaMemGetInfo` on a MIG card.** Rejected: it needs the context the plan is made
  before (GitHub #210), and the card class is read before any context too.
- **Name the generation `datacenter` or give 8.0 a generation of its own.** Rejected: what a catalog entry's
  `generation` protects (FP4 tensor cores, the block-scaled FP8 MMA) is missing on 8.0 exactly as on 8.6, and a
  new name would need every catalog to learn it.
- **Add `80` to the release list.** Not decided here: every architecture costs another device compile of nearly
  every translation unit, and an `80;120a` build would run the RTX 30 and RTX 40 on 8.0 code, which the
  full-rate rule (`mk/full-rate-kernels.txt`) does not read. A build for these cards names the list itself
  (`.github/workflows/image-architectures.yml`).

## Consequences

- The A100 and the A30, whole or as a MIG instance, run a build that carries `80`, on the entries that state
  no generation or `ampere` (the 27B's trellis entries). Their totals fit no class whole (an A100 of 80 GB reads
  79.25 GiB, between the 32 GB and the 96 GB classes), so `--profile auto` refuses them and a profile is named
  (`--profile rtx-24gb`, the Ampere trellis profile); a MIG instance takes its own class.
- No speed is measured on these cards yet: the GPU profile has run on none of them.
