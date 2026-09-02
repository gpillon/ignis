# 03 — wire checksum verification into the artifact loader path

GitHub: #21

Wire the `verify()` / `is_clean()` call from `crates/core`'s `checksum.rs`
(commit 18043b3) into the artifact loader path in `crates/server`. When the
server loads an artifact, it must verify the checksum report and **fail the
load** if `is_clean()` is false.

## Acceptance

- `Server::load_artifact` (or equivalent) calls `verify()` with the sidecar.
- On `!is_clean()`, the load fails with a descriptive error (no panic).
- On `is_clean()`, the load proceeds normally.
- Unit test: inject a mock `ChecksumReport` with a mismatch → load fails.
- Unit test: inject a clean report → load succeeds.