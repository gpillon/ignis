# 04 — scheduler: N=8 decode lanes + batched prefill

Status: in-progress (start — contract only, commit 0966eef; concrete N=8
scheduler + batched-prefill experiment pending, blocked on the kernel-abi
CUDA implementations; GitHub #13)
GitHub: #13
Blocked by: #12 (core-03), #5 (kernel-abi-01)

The v1 scheduler (`docs/design/ignis-v1.md` §2):

- **N=8 resident decode lanes** (with host-tier overflow).
- **Batched (concurrent) prefill** to saturate the GPU in prefill and cut
  burst TTFT — *an experiment to verify* (we may be compute-bound, in which
  case it is useless; an experiment, not a guarantee).
- The scheduler drives the kernel C ABI (`kernel-abi-01` prefill + GDN, and
  the decode path from `kernel-port-03`).

## Acceptance

- The scheduler holds N=8 resident decode lanes.
- Concurrent / batched prefill saturates the GPU in prefill (measured).
