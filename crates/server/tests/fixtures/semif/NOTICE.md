# SemIf's authored decision fixture

`authored144.jsonl` is a verbatim copy of `benchmarks/data/authored144.jsonl`
from the SemIf project (formerly OpenJev), an independent reproduction of
TypeSafe's Jev interface pattern with open models.

- Upstream: https://github.com/TheoLeeCJ/SemIf
- Commit: `ca3ba65f142967030ecb453346e94d6f476a69df`
- License: MIT, Copyright (c) 2026 TheoLeeCJ

The fixture's own `provenance` fields record it as project-authored synthetic
material ("Project authored; no copied external text"), independently model
reviewed rather than human adjudicated.

It is committed here, rather than fetched, because `classify_readout_gpu.rs`
is a GPU-profile test: under `IGNIS_GPU_PROFILE=1` a missing fixture is a hard
failure, so the serialized sweep must never depend on a file somebody
downloaded by hand (ADR 0006, `docs/agents/testing.md`).

It is kept verbatim so that a number measured here is comparable, row for row,
with the numbers SemIf published against it.
