---
status: accepted
amends: [ADR-0032]
---

# The builds run beside the tests, and nothing is published before they pass

ADR 0032 made the Linux release one pipeline in which nothing is built before
the tests pass (2026-09-21). A push to `main` took 46 to 64 minutes, most of
it one wait after another: the `test` legs (Windows 24 to 34 minutes, the
critical path) and only then the two builds (14 to 23 more). Inside a test leg
~80% was compiling, not testing: rust-cache keeps only the dependencies, so
every workspace crate and every one of the ~330 test binaries was built again
on every push (Linux ~10 minutes of ~12.5, Windows ~21 of ~25). The Linux
build compiled the kernel leaf from nothing on every push, its layer coming
after `COPY . .`. The owner decided on 2026-11-08 (#552) that the builds run
beside the tests and that only what publishes waits for them.

- **`linux` and `windows` need only `version`; whatever publishes needs
  `test`.** The 2026-09-21 rule becomes: nothing is *published* before the
  tests pass. `release` and `prerelease` need `test`, and the image is pushed
  by a job of its own, `image` (needs `test` and `linux`). A push whose tests
  fail still pays for its builds, in runner time a public repository does not
  pay for.
- **The image is still the `linux` job's compile.** When the run will
  publish, that job writes the `runtime` image to an OCI archive, and `image`
  copies it to each tag with `skopeo copy --all --preserve-digests`, its index
  and provenance attestation included, as the registry held them when the
  build step pushed. The tarball and the image cannot carry different builds;
  only the push moved.
- **The `test` job runs as four shards per host** (`mk/test-shard.sh`). A
  shard is `cargo test --workspace` with the target flags of its part, so
  features resolve as for the whole workspace and the shard compiles only its
  part. The parts are read off `cargo metadata` and hold every target `cargo
  test --workspace` runs exactly once; a target the flags cannot select as it
  does is refused (`mk/test-shard-selftest.sh`, run in shard 1).
- **The Linux build compiles the leaf in a layer that only `kernel/`
  invalidates, and the dependencies in one that only the manifests and the
  lock file invalidate** (cargo-chef, pinned), and keeps only the three
  shipped binaries from its cargo step: the target directory, rewritten into
  the 10 GB Actions cache on every push, crowded out the layers that are
  reused.
- **The Windows build's kernel cache restores only on an exact hit, and then
  dates the sources it hashes back**: ninja decides by date, the checkout
  dates every source later than the cached objects, and restored as is the
  leaf was compiled again in full on every push (274 commands, ~12.5 minutes).
  An exact hit proves the sources are the bytes the objects came from; a
  partial restore cannot, so there is none.
- **The test job builds with line tables only**, in place of full debuginfo:
  the compile took about half as long on Linux and a sixth less on Windows,
  with the suite unchanged.
- `crates/server/tests/ci_pipeline.rs` holds the workflow and the
  Containerfile to this shape.
