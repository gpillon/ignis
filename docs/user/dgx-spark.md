# Why not DGX Spark?

Ignis targets one class of card: an **RTX 5090** or an **RTX PRO 6000**, both
`SM120a`. The DGX Spark is the obvious question — a Blackwell GPU with native
NVFP4, 128 GB of memory, on a desk. Why is it not on the list?

## The goal is speed, for coding

Ignis was built for one job: to be a **very fast** local engine for coding
agents. An agent does not read one long answer. It works in loops: read a file,
call a tool, wait for the result, think, call the next tool, and one main agent
often has a handful of subagents doing the same thing on the same card. Every
turn waits on the model, and the waits add up. For this workload the numbers
that matter are how fast each lane decodes and how fast a long prompt is
prefilled. How big a model fits comes second.

Those numbers come from memory bandwidth, and that is where the Spark falls
short of the target:

|                    | RTX 5090      | DGX Spark (GB10)          |
|--------------------|---------------|---------------------------|
| Memory bandwidth   | ~1,792 GB/s   | ~273 GB/s                 |
| Streaming multiprocessors | 170    | 48                        |
| Memory             | 32 GB GDDR7   | 128 GB LPDDR5x, unified   |

Ignis's decode round is weight streaming at the card's bandwidth: on a 5090 it
is 15.81 ms of device time with the backbone GEMM at roofline (see the
[root README](../../README.md#the-decode-round)). A round that is bound by
bandwidth gets slower in step with the bandwidth, so the same round on a Spark
would take roughly **6.5× longer, around 100 ms instead of 16**. Prefill leans
on compute, and with a bit more than a quarter of the SMs it would likely be
**around 3.5× slower**. For an interactive coding loop, that is probably not
enough. It would be a different product, not a slower version of this one.

## But the road is not closed

That is an estimate, not a measurement. Nobody has run Ignis on a Spark, and a
few things argue against dismissing it too quickly:

- **128 GB changes the budget.** On a 5090 the KV cache, the retained
  conversations and the vision workspace all fight over 32 GB. On a Spark they
  would have about four times the room: longer contexts, more retained state,
  and no need for a host-RAM spill tier.
- **Speculation earns more where bandwidth is scarce.** DFlash2 turns one pass
  over the weights into several accepted tokens. The fewer bytes per second a
  machine has, the more each accepted draft token is worth. How much of the
  gap it would close is unknown.
- **Throughput can matter more than latency.** A quiet, always-on box running
  background agents cares more about total tokens across all lanes than about
  how fast any single lane answers.

So the Spark is not a target today, and the argument above says it probably
should not be the main one. Still, it is worth exploring, and the first
measurement on real hardware would settle more than any further reasoning.

## Preventive technical exploration

What follows is a reading of the code at v0.3.1, done without a Spark and
without running anything. It lists what would stop Ignis on a Spark and
roughly what fixing each thing would take. Nothing here has been verified on
hardware.

### What blocks it today

1. **GPU architecture.** The kernel leaf is compiled for `120a` only
   ([`kernel/CMakeLists.txt`](../../kernel/CMakeLists.txt),
   [`kernel/build.sh`](../../kernel/build.sh)). GB10 is compute capability
   12.1. Code built for an `a` target, whether machine code or PTX, runs only on
   that exact capability, so on a Spark the binary would fail to load any
   kernel. The fix is to build for `121a`, or for `120a;121a` in one binary.
   The architecture-specific instructions the kernels use are the block-scaled
   FP4 MMA (`mma.sync…kind::mxf4nvf4.block_scale`,
   [`common/mma.cuh`](../../kernel/vendor/src/ops/common/mma.cuh)), TMA bulk
   tensor copies and `setmaxnreg`
   ([`nvfp4_w4a4_tma.cuh`](../../kernel/vendor/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh)).
   All of them are expected to exist on `121a`, but only a compile can confirm
   `setmaxnreg`.
2. **CPU architecture.** The Spark is ARM (`aarch64`). The release workflow,
   the [`Containerfile`](../../Containerfile) and
   [`mk/os/linux.mk`](../../mk/os/linux.mk) all name
   `x86_64-unknown-linux-gnu`. In the Makefile the triple can be overridden
   (`TARGET_TRIPLE ?=`). The Rust code uses no x86 intrinsics, and the build
   script links from `$CUDA_HOME/lib64`, which ARM installs of CUDA also
   provide, so a native build on the Spark looks plausible. There is no ARM
   release archive or container image.

### What would likely stop it at startup

3. **NVML memory query.** At load time the server reads the free memory
   through `nvmlDeviceGetMemoryInfo` before it creates the CUDA context
   (GitHub #210), and it refuses to start if that call fails
   (`VRAM budget: free memory is unreadable`,
   [`crates/server/src/runtime.rs`](../../crates/server/src/runtime.rs)). On
   the Spark's unified memory, NVML reportedly does not report memory at all,
   and `nvidia-smi` shows it as "Not Supported". A fallback is needed there.
   `cudaMemGetInfo` is the obvious candidate, but on unified memory it is
   imprecise too, because it does not count page cache the OS could reclaim.

### What works but means the wrong thing

4. **The memory plan assumes a discrete card.** The VRAM budget (#210) and the
   KV arena in pinned host RAM (#213) treat device memory and host memory as
   two separate pools. On a Spark they are the same LPDDR5x: moving KV pages
   "to RAM" frees nothing and costs copies. A unified-memory mode would plan
   one pool and switch the spill tier off.
5. **Launch shapes tuned on 170 SMs.** Two vendored launchers read the SM count
   at runtime (the GDN gating projection plan and GQA prefill attention). Other
   grid shapes were tuned on a 5090. On 48 SMs that affects speed, not
   correctness.

### The smallest port

Four pieces: a `121a` build target, an `aarch64` build, an NVML fallback, and a
unified-memory mode in the memory plan. None of them is large, but none of them
can be checked without a Spark to run on.
