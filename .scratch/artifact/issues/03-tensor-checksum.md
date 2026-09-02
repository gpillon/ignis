# 03 — tensor checksum validation against sidecars

Status: needs-triage
GitHub: #8
Blocked by: #4 (artifact-01)

Offline verification step (carried over from the "Remaining" list of
kernel-port 02): verify the materialized tensor checksums against the
`conversion.json` / `graft.json` sidecars.

## Acceptance

- Tensor checksums match the `conversion.json` / `graft.json` sidecars.
- Any mismatch is reported (load failure / flagged tensor).
