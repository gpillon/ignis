# 03 — tensor checksum validation against sidecars

GitHub: #8

Offline verification step (carried over from the "Remaining" list of
kernel-port 02): verify the materialized tensor checksums against the
`conversion.json` / `graft.json` sidecars.

## Acceptance

- Tensor checksums match the `conversion.json` / `graft.json` sidecars.
- Any mismatch is reported (load failure / flagged tensor).

Delivered (commit 18043b3): new `checksum.rs` in `crates/artifact` —
`Sidecar::load` (parses the shared fields of both sidecar shapes; an
absent `grafted_from` block is tolerated for the conversion shape) and
`verify(reader, sidecar) -> ChecksumReport` (global invariants +
per-parent checks; never panics, `is_clean()` is the load-failure
surface). 45 lib + 4 integration tests green (6 new checksum unit tests
+ 1 gated real-artifact integration test).

Key design finding: the v2 container's `graft.json` / `conversion.json`
sidecars carry **no per-tensor digests** — the contract requires none
(verified against the reference spec §8 and the real sidecars in
`F:\ai\q38\ninfer-models\`). The per-tensor datum they *do* record is the
NVFP4 `local_nvfp4.parents` table (a float `weight_scale_divisor` +
`relative_frobenius_error`), plus whole-file invariants
(`artifact.bytes`, `objects.count`). The container stores the divisor as
a trailing FP32 word in each NVFP4 blockscale payload, so "checksum
match" is: (a) file size + object count hold, and (b) each recorded
parent resolves to an NVFP4 tensor whose stored FP32 divisor
value-matches the recorded number (compared as promoted `f64` to avoid a
narrowing ULP shift). Mismatches surface as a `ChecksumReport` with a
per-object `matched` / `mismatched` / `missing` list + `flagged` /
`global_flags` (a flagged report, not a panic).

Gated real-artifact run (skips when the files are absent): the 19.4 GB
`qwen3_8_27b_nvfp4full-v2` container verifies clean — 1,325 objects,
19,406,942,468 bytes, 34 parents (all `weight_scale_divisor: null` —
weight-only grafts), 281 NVFP4 tensors (34 covered by the graft, 247
inherited from the v1 base).

Follow-up (tracked, not in this ticket): nothing in the loader path
calls `verify()` / `ChecksumReport::is_clean()` yet — the acceptance line
"any mismatch is reported (load failure / flagged tensor)" is satisfied
as an *inspectable* report; making a sidecar mismatch actually **fail
the load** needs a call site holding both the `Reader` and the sidecar
in the engine/loader path (the `crates/server` territory — the natural
next step for the server actor, see PENDING.md).
