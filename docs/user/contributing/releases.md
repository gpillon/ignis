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
for 8.6, or none for 12.0, which the workflow requires apart from the list the
build is given (#482).

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
[first-light finding's](../../findings/2026-11-03-first-light-on-sm86.md): the
16 windows of 4,096 tokens of #340's gate 1, dumped by the `span_logits`
example at prefill chunk 1024 with the `hq-e8-2b` KV format, on the route a
card below compute capability 12.0 runs when nothing names one: the dense
prefill, the integer band and ignis's own gated delta net prefill. The example
reads the variables rather than the card's default, so the gate names the
route: `IGNIS_MODEL_PREFILL_LINEAR=int8 IGNIS_MODEL_PREFILL_ATTENTION=int8
IGNIS_GDN_PREFILL=ours`. The dump is scored against the BF16 checkpoint's 64
most probable next tokens per position (`kl_top64`,
`crates/bench/src/flash_next/kld.rs`). That reference is a set of about 90 MB,
the windows included, which stays out of the repo: `tools/kld-gate/ref_top64.py`
builds it from the BF16 checkpoint on the RTX 5090
([its README](../../../tools/kld-gate/README.md)), and the coordinator carries
it to the card. `tools/kld-gate/score_top64.py` scores a dump against it. The
limits are derived from the RTX 5090 on the same route: the same dump there,
under the same three variables, with the same windows, chunk and reference
set, scored by the same command. Its figures are TODO(#515 coordinator: the
5090's top-64 KLD on this route) for the top-64 KLD and TODO(#515 coordinator:
the 5090's top-1 on this route) for top-1, and the limits, read on the `all`
row of the scorer's table, follow from them:

| | limit |
|---|---|
| `KLD top64`, all windows | at most 1.05 times the 5090's |
| `top-1` | at least 0.5 points under the 5090's |

```sh
# On the sm_86 card, from the release's checkout, the set ref27b-top64/ beside it
# (tools/kld-gate/README.md) and 32.5 GB free for the dump. ignis-server is the
# release candidate's; on Linux every cargo command here also needs
# CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu.
export IGNIS_CUDA_ARCHITECTURES='86;120a'
ignis-server model download qwen3.8-27b-16g      # into ./models
IGNIS_MODEL_PREFILL_LINEAR=int8 IGNIS_MODEL_PREFILL_ATTENTION=int8 IGNIS_GDN_PREFILL=ours \n    cargo run --release -p ignis-core --features cuda --example span_logits -- \
    models/qwen3_8_27b_trellis_sc30-v2.ninfer hq-e8-2b ref27b-top64/windows.json dump-sm86 1024
python3 tools/kld-gate/score_top64.py ref27b-top64 dump-sm86 --json gate-sm86.json
```

A change that alters the 5090's numerics on this route records the 5090's new
figures in a finding first, from the same dump and score on the 5090, and the
limits follow them. Releases before 0.10.0 gated the FP16 band instead (0.04016
and 92.92%, from the 5090's 0.03825 and 93.42%); naming
`IGNIS_MODEL_PREFILL_LINEAR=fp16 IGNIS_MODEL_PREFILL_ATTENTION=fp16` still runs
that route on the card, and it is no longer the gate.

**The leaf's tests.** Built for `86;120a`, `ctest -LE sm120a-only` passes every
test. `ignis_kernel_gdn_gating_proj_tests` prints `budget` lines and names with
`SKIP` the cooperative schedules the card cannot hold resident (#478): skips
there are expected, failures are not.

```sh
bash kernel/build.sh            # kernel\build.ps1 on Windows: the whole project, tests included
ctest --test-dir kernel/build -LE sm120a-only --output-on-failure
```

**Smoke.** Start the server under `--profile auto` (on a 24 GB card that is
`rtx-24gb`, which turns DFlash2 and vision on). Its `GET /v1/config` reports
`prefill_linear` and `prefill_attention` as `int8`, the card their source: the
smoke runs the route the card's users run. Ask for:

- a greedy chat completion;
- a small image: the smallest the processor accepts, since it upscales every
  image to its 65,536-pixel floor, which is 256 patches at the vision tower and
  64 tokens after the merge (the request's `media.vision_tokens` in the
  server's log). The tower's W8 shapes have no small-token route (the SIMT
  routes serve up to 16 tokens, the merger's up to 4 or 5, and `mma_r64_c128`
  above), so no image reaches a kernel only an sm_120a build holds;
- a large image, at the processor's item bound: 16,384 tokens, 4096 x 4096
  pixels at 32 x 32 a merged token, under the profile's `vision.max_tokens`.

```sh
ignis-server --profile auto      # its ignis.config.source line names rtx-24gb
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"qwen3.8-27b","temperature":0,"enable_thinking":false,"max_tokens":64,"messages":[{"role":"user","content":"What is the capital of France? One word."}]}'
# Two solid-colour PNGs: 32 x 32 red, upscaled to the floor; 4096 x 4096 blue, the item bound.
python3 - <<'EOF'
import struct, zlib
def png(path, side, rgb):
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = (b"\0" + bytes(rgb) * side) * side
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
png("small.png", 32, (220, 30, 30))
png("large.png", 4096, (30, 60, 220))
EOF
for image in small large; do
  printf '{"model":"qwen3.8-27b","temperature":0,"enable_thinking":false,"max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"What colour is this image? One word."},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]}]}' \
    "$(base64 -w0 "$image.png")" > "$image.json"
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d "@$image.json"
done
```

Each must be answered correctly: `Paris`, red, blue.

The release binary must also carry machine code for 8.6 and 12.0: the release
check above does that in CI, whatever list the build was given.

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
