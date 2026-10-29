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
different binaries (ADR 0032). Neither host packages a server that is not a
CUDA build: `mk/cuda-build-check.sh` reads the binary and refuses one that
would fall back to the CPU mock, which v0.9.0 and v0.9.1 shipped (#408).

The user guide follows the release. `.github/workflows/docs.yml` builds the
site from the tag and publishes it once the release run has ended, and again
whenever the release is edited. The guide's Changelog section is read from the
release notes, so the notes written with `gh release edit` after publishing are
what its page for the release shows (ADR 0066).

CI cannot run the GPU profile: no runner has an NVIDIA card. Correctness on
the card stays on the development machine (`docs/agents/testing.md`).
