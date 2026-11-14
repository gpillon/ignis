---
status: accepted
amends: [ADR-0030]
---

# The CUDA context is measured when it is made, and a reload adds back what the driver reports

ADR 0030's VRAM plan charges the CUDA context as a line, `cuda_context`, and #324 made a later load in the
same process add back what the process already holds. Both took their figures from the WDDM counter on the
RTX 5090 under Windows: `CUDA_CONTEXT_BYTES` (452,595,712 bytes) for the context and
`QWEN27B_PROCESS_STATE_BYTES` (83,894,272) for the state a 27B load leaves, and every card and OS charged and
added back the same two. The RTX 5080 under Linux holds 299,892,736 bytes after a 27B teardown (NVML's figure
for the process, measured in the RTX 5080 fork GitHub #604 links), so its cold start charged 236,597,248 bytes more than the
process holds, and its reload, adding back those constants, reserved 5,242 KV pages against 4,858 cold. The
reload's plan charges again what it adds back, so it was sized to the memory actually free; the plan that
was off is the cold start's. Under WSL a context made through the driver API took 519,270,400 bytes on the
RTX 5090, so a cold start there charged less than its context takes.

The owner's brief for GitHub #604 (2026-11-14) takes each figure from what can measure it:

- **The context is measured when it is made.** The process's first `CudaDevice` reads NVML's free memory
  just before it makes the context and again after: the difference is the context's bytes, kept for the
  process's life, as the context is. NVML can count a fresh allocation late, in 64 MiB pieces, by up to 63 ms
  (finding 2026-10-31, `nvml-counts-a-fresh-allocation-late`), so the reading after is read every 10 ms and
  taken once the readings have held within 1 MiB of one another for 250 ms, the lowest of them; at 2 s the
  lowest reading seen is taken, an over-charge, never short. Every line that charged `CUDA_CONTEXT_BYTES`
  charges it: the 27B's `cuda_context` line and Flash-Next's. A load plans once its context exists:
  Flash-Next, whose weights make its device after the plan, makes and measures the context first.
- **A reading that cannot be read**, or a difference of zero or less (another process freed memory inside
  the window), falls back to `CUDA_CONTEXT_BYTES`, with the WARN `ignis.runtime.cuda_context_fallback`.
  Neither refuses a load.
- **The 27B's per-process state is still the 5090's figure.** It exists only once the 27B's kernels have
  run, after the plan, so `residual` still charges `QWEN27B_PROCESS_STATE_BYTES`, inside
  `LOAD_RESIDUAL_BYTES`.
- **The add-back is the driver's figure for the process, capped at what the plan charges again.** For what
  the process already holds the plan charges again the measured context, and on a 27B plan
  `QWEN27B_PROCESS_STATE_BYTES` too once a 27B load has run; Flash-Next's plan charges again the context
  only. A later load adds back NVML's per-process figure for its PID
  (`nvmlDeviceGetComputeRunningProcesses_v3`, every entry of the PID summed), never more than that. With the
  5080's figure the reload adds back exactly what the process holds and plans its cold start's pool, as long
  as that card's 27B state is no larger than the 5090's; a figure above the cap sizes the pool to the memory
  actually free, and a family's state that the plan does not charge, Flash-Next's or the 27B's under a
  Flash-Next plan, is never added back.
- **Where the driver reports no figure** — a PID absent from its list (under WSL the list is empty), a
  figure it marks not available (every figure under WDDM), a driver without the query, a call that fails —
  the add-back is what the plan charges again. That keeps the reload sized to the memory actually free, as
  #324 had it.
- **The plan events say where each figure came from**: `ignis.runtime.vram_plan` and
  `ignis.runtime.flash_next_plan` carry `cuda_context_bytes` with `cuda_context_source` (`measured` or
  `fallback`), and `process_held_bytes` with `process_held_source` (`none` on a first load, `per_process` or
  `charged_again`).

The context is read from NVML's free memory, not from a per-process figure, because only the free memory is
a reading everywhere the plan runs. On the RTX 5090 under WSL (the podman machine, kernel-mode driver
610.88) NVML's free memory fell by 519,270,400 bytes when a probe made its context and counted a 1 GiB
allocation on top to the byte, while NVML's per-process list was empty with the context held and with the
1 GiB on top; under WDDM the driver marks every per-process figure not available. The free memory moves with
every other process on the card, which is why the window is short and the reading it takes is the lowest.

## Considered Options

- **The fork's fix** (f09c53f2, which GitHub #604 links): on Linux only, add back NVML's per-process figure whole,
  and refuse the load where it cannot be read. It leaves the cold start's over-charge in place and makes the
  reload match it, not the card; under WSL it reads 0 for a PID the list does not carry and adds back
  nothing; and it refuses a load over a figure that only sizes a reload.
- **A per-process figure for the context too**: none exists under WSL or WDDM.
- **The WDDM counter as a per-process figure on Windows**: not taken here. Windows adds back what the plan
  charges again, as #324 did.

## Consequences

- `CUDA_CONTEXT_BYTES` is the fallback only. The process's first device takes 250 ms longer to make, 2 s at
  most.
- The image plan is unchanged: it reads NVML after its context exists and charges none.
- On a card whose 27B state is larger than the 5090's the cold start still charges too little for it, as
  before; measuring that state, and `LOAD_RESIDUAL_BYTES`, on other cards is left out.
- The reading is checked on the card by `nvml_memory_gpu`, which holds the measured context to the process's
  WDDM dedicated usage right after the context is made, within 16 MiB, and prints the driver's figure for the
  process where it reports one, a different reading that no run here has held the context to;
  `vram_reload_gpu` prints the context, the add-back and where it came from, and holds the reload's pool to
  the cold start's within 1%.
