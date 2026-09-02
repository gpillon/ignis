# 06 — KV-RAM host tier (probation / protected eviction)

Status: needs-triage
GitHub: #17
Blocked by: #13 (core-04)

The **KV-RAM host tier** (`docs/design/ignis-v1.md` §2, `CONTEXT.md`
"KV-RAM"):

- Snapshots GPU lanes to host RAM so sibling requests **restore** instead of
  re-prefilling.
- Two-tier eviction: **probation → protected**.
- Pulled into v1 (not v1.1). Must respect the GDN boundary (`core-02`).

## Acceptance

- The host tier evicts / restores GPU lanes to host RAM.
- Evictions are bounded under the N=8 + overflow load; sibling requests
  restore instead of re-prefilling.
