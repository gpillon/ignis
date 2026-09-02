# 04 — scheduler: N=8 decode lanes + batched prefill

Status: resolved (commit 32cb738, 2026-09-02; GitHub #13)
GitHub: #13
Blocked by: #12 (core-03), #5 (kernel-abi-01)

The v1 scheduler (`docs/design/ignis-v1.md` §2):

- **N=8 resident decode lanes** (with host-tier overflow).
- **Batched (concurrent) prefill** to saturate the GPU in prefill and cut
  burst TTFT — *an experiment to verify* (we may be compute-bound, in which
  case it is useless; an experiment, not a guarantee).
- The scheduler drives the kernel C ABI (`kernel-abi-01` prefill + GDN, and
  the decode path from `kernel-port-03`).

Delivered: `ConcreteScheduler` (N=8 resident lanes, batched prefill in one
compute call, lane deal class-priority + FIFO, batched decode, in-flight cap
= N_DECODE_LANES until the host tier) + `MockCompute` (deterministic,
recording `Compute` stand-in, ADR 0006). CPU-tested: `n8_lanes`,
`batched_prefill`, `compute_errors`, `admission_priority` (all green).

Deferred: the GPU-saturation *measurement* of batched prefill (an experiment,
not a guarantee) is deferred to the bench harness (bench-01/02, GitHub
#19/#20) through the 99% gate of ADR 0007 on the exclusive GPU (ADR 0006) —
it runs once the harness `HttpEndpoint` + a reference baseline are in place.

## Acceptance

- The scheduler holds N=8 resident decode lanes.
- Concurrent / batched prefill saturates the GPU in prefill (measured) —
  deferred to the bench harness (see Deferred above).
