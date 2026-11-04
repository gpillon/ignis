---
title: "Cutting a release"
description: "Version bump, changelog, tag, and what the release workflow publishes."
sidebar:
  order: 6
---

The version lives in four files: `Cargo.toml` and `web/package.json` (the
Playground ships inside the binary) and the two lockfiles, which the next
build would rewrite. The release workflow runs the same `mk/version.sh check`
`make version-check` does, so all four have to agree there too.

```
make version                     # what each of the four declares
make version-bump T=patch        # or minor, major
make version-bump V=1.2.3-rc.1   # or an exact version: x.y.z, -prerelease optional
make version-check               # all four files agree, before you tag
```

`version-bump` edits and stops there: the commit and the tag are printed, not
run, because pushing a `v*` tag publishes a release.

What the release changed is drafted from the range, not written from memory:

```
make changelog                   # since the last v* tag, to HEAD
make changelog FROM=v0.1.1 TO=v0.1.2
```

It reads the issues the range's commits name, the ADRs added or amended in it,
and the commits that name no issue, and prints Markdown for the version-bump
commit and the GitHub release body. It is a draft: the headings are sorted by
what the range can prove, not by what matters most.

`.github/workflows/release.yml` runs the CPU-only test suite on Linux and
Windows, and only then builds the GPU engine for both hosts — on every push to
`main` or a `ci/**` branch, publishing nothing. A `v*` tag — which must match
`workspace.package.version`, or the job refuses it — turns the same run into a
GitHub Release and pushes the container image. The image is built by the same
job as the Linux tarball, out of the same compile, so the two can never carry
different binaries (ADR 0032). Both hosts build the kernel for `86;120a`, so
every artifact runs on RTX 30 and RTX 40 cards as well as on Blackwell
([Building from source](building.md#other-card-architectures)). Neither host
packages a server that is not a CUDA build: `mk/cuda-build-check.sh` reads the
binaries and refuses a server that would fall back to the CPU mock, which
v0.9.0 and v0.9.1 shipped (#408), or any binary that carries no machine code
for 8.6, or none for 12.0 (#482).

## The sm_86 gate

A release built for `86;120a` promises RTX 30 cards a kernel that was only ever
tested on the RTX 5090, so a release whose range touched the leaf or the routes
is checked on an sm_86 card before it ships. The release's coordinator runs it
on that card; CI has none. A release that touched neither skips it, and costs no
card time. Ask the range:

```
git diff --stat "$(git describe --tags --abbrev=0 --match 'v*')".. -- kernel/ crates/server/src/config/arch.rs
```

Any line of output means the gate runs; none means it does not. The gate has
three parts, and a failure of any of them stops the release.

**Quality.** The method is the
[first-light finding's](../../findings/2026-11-03-first-light-on-sm86.md), whose
scripts and reference it names: the 16 windows of 4,096 tokens of #340's gate 1
(`tools/kld-gate/`), dumped by the `span_logits` example at prefill chunk 1024
with the `hq-e8-2b` KV format and `IGNIS_MODEL_PREFILL_ATTENTION=fp16` (the
band a card below compute capability 12.0 runs), and scored against the BF16
checkpoint's 64 most probable next tokens per position (`kl_top64`,
`crates/bench/src/flash_next/kld.rs`). The limits come from the RTX 5090's
figures on the same FP16 band, 0.03825 for the top-64 KLD and 93.42% for top-1:

| | limit |
|---|---|
| top-64 KLD, all windows | at most 0.04016 (1.05 times the 5090's) |
| top-1 | at least 92.92% (within 0.5 points of the 5090's) |

A change that alters the 5090's numerics records the 5090's new FP16-band
figures in a finding first, and the limits follow them.

**The leaf's tests.** Build with `IGNIS_CUDA_ARCHITECTURES=86;120a` and run
`ctest -LE sm120a-only`: every test passes. `ignis_kernel_gdn_gating_proj_tests`
prints `budget` lines and names with `SKIP` the cooperative schedules the card
cannot hold resident (#478): skips there are expected, failures are not.

**Smoke.** Start the server under `--profile auto` (on a 24 GB card that is
`rtx-24gb`) with DFlash2 on, and ask for:

- a greedy chat completion;
- a small image: the smallest the processor accepts, since it upscales every
  image to its 65,536-pixel floor, which is 256 patches at the vision tower and
  64 tokens after the merge. The tower's W8 shapes have no small-token route
  (the SIMT routes serve up to 16 tokens, the merger's up to 4 or 5, and
  `mma_r64_c128` above), so no image reaches a kernel only an sm_120a build
  holds;
- a large image, near the profile's vision item bound (`vision.max_tokens`).

Each must be answered correctly.

The release binary must also carry machine code for 8.6 and 12.0: the release
check above does that in CI.

The gate records correctness only: the two KLD and top-1 figures, pass or fail
for each part, and the small image's patch count. It records no speed and no
description of the machine.

The user guide follows the release. `.github/workflows/docs.yml` builds the
site from the tag and publishes it once the release run has ended, and again
whenever the release is edited. The guide's Changelog section is read from the
release notes, so the notes written with `gh release edit` after publishing are
what its page for the release shows (ADR 0066).

CI cannot run the GPU profile: no runner has an NVIDIA card. Correctness on
the card stays on the development machine (`docs/agents/testing.md`).
