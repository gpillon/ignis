---
status: accepted
---

# The CUDA context is measured when it is made, and a reload adds back what the driver reports

The VRAM plan (ADR 0030) charges the CUDA context as a line, and a later load in the same process adds back what the
process already holds: ignis never resets the context, and a 27B load leaves per-process state beside it. Both
figures were constants read from the WDDM counter on the RTX 5090 under Windows, charged and added back on every card
and OS, so each is now taken from what can measure it. The RTX 5080 under Linux holds 299,892,736 bytes after a 27B
teardown (NVML's figure for the process, measured in the fork GitHub #604 links), so its cold start charged,
with the constants of that day, 236,597,248 bytes more than the process holds, and planned 4,858 KV pages where its
reload planned 5,242; under WSL
a context took 519,270,400 bytes on the RTX 5090, so a cold start there charged less than its context takes.

- **The context is measured when it is made.** The process's first device reads NVML's free memory just before it
  makes the context and again after: the difference is the context's bytes, kept for the process's life, as the
  context is. NVML can count a fresh allocation late
  ([finding](../findings/2026-10-31-nvml-counts-a-fresh-allocation-late.md)), so the reading after is taken once
  the readings have held still, the lowest of them, and at a deadline the lowest seen: an over-charge, never short
  (`measure_context` in `crates/artifact/src/device.rs`). Every line that charged the constant charges the measured
  figure, the 27B's `cuda_context` and Flash-Next's, and a load plans once its context exists: Flash-Next, whose
  weights make its device after the plan, makes and measures the context first.
- **A reading that cannot be read**, or a difference of zero or less (another process freed memory inside the
  window), falls back to `CUDA_CONTEXT_BYTES`, with the WARN `ignis.runtime.cuda_context_fallback`. Neither refuses
  a load.
- **The 27B's per-process state is still the 5090's figure.** It exists only once the 27B's kernels have run, after
  the plan, so `residual` still charges `QWEN27B_PROCESS_STATE_BYTES`, inside `LOAD_RESIDUAL_BYTES`.
- **The add-back is the driver's figure for the process, capped at what the plan charges again**: the measured
  context and, on a 27B plan, the 27B's state once a 27B load has run. Flash-Next's plan, which has no residual line
  and whose own per-process state is not measured, charges again the context only: an under-count costs pool, never
  more memory than the card has. A later load adds back NVML's per-process figure for its PID, never more than the
  cap. With the 5080's figure the reload adds back exactly what the process holds and plans its cold start's pool,
  as long as that card's 27B state is no larger than the 5090's; a figure above the cap sizes the pool to the memory
  actually free, and a family's state that the plan does not charge, Flash-Next's or the 27B's under a Flash-Next
  plan, is never added back.
- **Where the driver reports no figure**, a PID absent from its list (under WSL the list is empty), a figure it
  marks not available (every figure under WDDM), a driver without the query or a call that fails, the add-back is
  what the plan charges again. The reload's plan charges again what it adds back, so it stays sized to the memory
  actually free: with the constants, the plan that was off was the cold start's.
- **The plan events say where each figure came from** (`cuda_context_source`, `process_held_source`).

The context is read from NVML's free memory, not from a per-process figure, because only the free memory is a reading
everywhere the plan runs: on the RTX 5090 under WSL (the podman machine, kernel-mode driver 610.88) it fell by the
context's bytes when a probe made the context and counted a 1 GiB allocation on
top to the byte while NVML's per-process list stayed empty, and under WDDM the driver marks every per-process figure
not available. The free memory moves with every other process on the card, which is why the window is short and the
reading it takes is the lowest.

## Considered Options

- **The fork's fix**: on Linux only, add back NVML's per-process figure whole, and refuse the load where it cannot
  be read. It leaves the cold start's over-charge in place and makes the reload match it, not the card; under WSL it
  reads 0 for a PID the list does not carry and adds back nothing; and it refuses a load over a figure that only
  sizes a reload.
- **A per-process figure for the context too**: none exists under WSL or WDDM.
- **The WDDM counter as a per-process figure on Windows**: not taken here. Windows adds back what the plan charges
  again.

## Consequences

- `CUDA_CONTEXT_BYTES` is the fallback only, and the process's first device takes the measurement's window longer
  to make.
- The image model's plan is unchanged: it reads NVML after its context exists and charges neither a context nor a
  residual line, so what a 27B load left in the process stays outside its budget.
- On a card whose 27B state is larger than the 5090's the cold start still charges too little for it, as before;
  measuring that state, and `LOAD_RESIDUAL_BYTES`, on other cards is left out.
