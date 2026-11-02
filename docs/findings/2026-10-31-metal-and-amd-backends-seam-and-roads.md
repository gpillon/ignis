# A Metal or AMD backend for the leaf: the roads, a seam at the segment, and trellis decode's ALU budget on an M4 Pro

- Kind: discovery
- Status: current
- Observed: 2026-10-31
- Last verified: 2026-10-31
- Scope: kernel / the leaf's structure (orchestration, layer bodies, kernels) and where a hardware seam can go for a
  Metal backend (the owner's M4 Pro laptop, 48 GB unified memory) and a later AMD consumer backend (RDNA3/4 through
  HIP); serving / the card class and memory plan on unified memory; quality / the reference a non-CUDA backend gates on
- Related: [GitHub #439](https://github.com/gpillon/ignis/issues/439), [ADR 0001](../adr/0001-cpp-kernel-leaf-c-abi.md), [ADR 0009](../adr/0009-step-level-device-resident-abi.md),
  [ADR 0010](../adr/0010-vendored-reference-kernels.md), [ADR 0016](../adr/0016-step-abi-extensible-options.md),
  [ADR 0019](../adr/0019-decode-cuda-graph-slot-indirection.md),
  [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md),
  [ADR 0030](../adr/0030-device-memory-reserved-at-load.md),
  [ADR 0051](../adr/0051-kernel-code-is-ours-adapted-or-vendored.md),
  [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md),
  [ADR 0062](../adr/0062-the-token-embedding-lives-in-host-memory.md),
  [ADR 0063](../adr/0063-the-profile-follows-the-card-class.md),
  [ADR 0075](../adr/0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md),
  [decode round anatomy](2026-09-18-decode-round-anatomy.md),
  [dense trellis kernels on the 5090](2026-10-23-dense-trellis-kernels-on-the-5090.md),
  [Flash-Next decode fusion](2026-10-08-flash-next-decode-fusion.md)
- Superseded by: none

## Question

The owner asked what it would take for ignis to run on Metal, and later on AMD consumer cards, without giving up the
project's performance principle (ADR 0005). Small overheads are acceptable only when a JIT or automatic port layer
costs almost nothing. Three questions follow from that:

1. **Which roads exist** from today's CUDA leaf to Metal (and to HIP), and which of them can keep the performance?
2. **Where does the seam go** between code shared by every backend and code each backend owns? The layer, the op, or
   something else?
3. **What stands in the way** of the first target, independent of the seam? The first target is the 27B on the
   owner's M4 Pro laptop (48 GB), with the trellis body (ADR 0060) as the owner's first choice of format.

The owner set the bar per backend as resource use rather than parity with the 5090: **≥ 95% of the binding
resource, ≥ 90% acceptable, below 90% discussed**. AMD's target is consumer RDNA (wave32, 16-24 GB discrete cards).

This was exploration only. Nothing was built, run, or measured on Apple or AMD hardware, and no Metal code was
written. The exploration ran in one session in four passes:

- the session's own survey of the code and the roads;
- a review by the advisor model;
- an independent review by a fresh agent (Opus, max effort), briefed with the code and the open question but not with
  the session's lean;
- a reconcile pass in which the session and the advisor retracted two positions on the independent review's evidence.

## Evidence

Code read at `main` 1a9f8e1a (2026-10-31); the two later commits on `main` (22501a1b) touch only the glossary and
ADR 0064. The independent review's full report, with every citation, is kept at
`.scratch/metal-port/seam-review.md`.

### E1. The CUDA-specific surface of the leaf

```bash
git ls-files '*.cu' '*.cuh' | wc -l                       # 330 files
git ls-files '*.cu' '*.cuh' | xargs cat | wc -l           # 96,242 lines (tests and vendored ops included)
grep -rhoE "\b(cudaGraph[A-Za-z]*|cudaMallocHost|cudaHostAlloc|cudaHostGetDevicePointer|cp\.async|mma\.sync|wgmma|tcgen05|ldmatrix|__shfl_[a-z_]*sync|cudaLaunchCooperativeKernel|cluster)" \
  kernel/src kernel/vendor/src | sort | uniq -c | sort -rn
```

| construct | occurrences in `kernel/src` + `kernel/vendor/src` |
|---|---:|
| `ldmatrix` | 163 |
| `__shfl_sync` | 97 |
| `cp.async` | 81 |
| `__shfl_xor_sync` | 54 |
| `cluster` | 53 |
| `mma.sync` | 25 |
| `__shfl_down_sync` | 22 |
| `cudaHostAlloc` | 21 |
| `cudaGraphExec` | 13 |
| `__shfl_up_sync` | 12 |
| `cudaGraphLaunch` | 11 |
| `cudaGraphInstantiate` | 8 |
| `cudaHostGetDevicePointer` | 7 |

The hot path is PTX shaped for Blackwell (`ldmatrix`, `cp.async`, `mma.sync`, thread-block clusters), with CUDA graph
capture and pinned and host-mapped memory around it. The C ABI is 33 headers in `kernel/include/`, with about 193
`ignis_*` entry points.

### E2. Orchestration files hold no kernels, but call the CUDA runtime directly

Counted per file in `kernel/src`: `__global__`, `<<<` launches, and `cuda[A-Z]*(` calls.

| file | lines | `__global__` | `<<<` | `cuda*` calls |
|---|---:|---:|---:|---:|
| `model.cu` | 2,206 | 0 | 0 | 27 |
| `seq.cu` | 2,596 | 0 | 0 | 61 |
| `seq_checkpoint.cu` | 423 | 0 | 0 | 2 |
| `seq_prefix.cu` | 528 | 0 | 0 | 5 |
| `step.cu` | 2,840 | 0 | 0 | 144 |
| `decode_graph.cu` | 460 | 0 | 0 | 20 |
| `device.cu` | 403 | 0 | 0 | 29 |
| `paged_kv_budget.cu` | 88 | 0 | 0 | 0 |
| `gdn_layer.cu` | 740 | 0 | 0 | 13 |
| `gqa_layer.cu` | 774 | 0 | 0 | 17 |
| `trellis_body.cu` | 117 | 0 | 0 | 1 |
| `linear.cu` | 201 | 0 | 0 | 0 |
| `residency.cu` | 1,491 | 6 | 9 | 88 |
| `trellis_linear.cu` | 432 | 3 | 8 | 13 |
| `trellis_linear_decode.cu` | 441 | 1 | 0 | 8 |
| `trellis_linear_prefill.cu` | 335 | 1 | 0 | 5 |
| `hq_prompt_attention.cu` | 1,156 | 4 | 7 | 6 |

17 of the 40 `.cu` files in `kernel/src` define a kernel. The orchestration, about 11,000 lines, reaches the kernels
only through launchers.

The independent review read what the `cuda*` calls are:

- `step.cu`: 67 `cudaMemcpyAsync`, 19 `cudaStreamSynchronize`, the chunk profiler's events, 4 `cudaGraphLaunch`;
- `seq*.cu`: copies, memsets and syncs;
- `model.cu` and `device.cu`: allocation, streams, memory info.

All of these are mechanical wrappers. The real coupling is elsewhere: **about 45 distinct device ops** called
directly from `gdn_layer.cu`, `gqa_layer.cu`, `decode_graph.cu`, `step.cu`, `trellis_body.cu`, `dflash2_drafter.cu`,
`dflash2_readout.cu` and `attention_readout.cu`.

- Most are vendored `ninfer::ops` with `cudaStream_t` in the signature (`vendor/include/ninfer/ops/rmsnorm.h:24`,
  `linear_add.h:72-73`).
- They are shaped by NVFP4: `LinearPolicy`, workspace arenas, capacity queries.
- Two reach into vendored internals: `detail::gqa_attention_resolve_route` (`gqa_layer.cu:301`) and
  `detail::gdn_projected_conv_record_launch` (`gdn_layer.cu:544`).

The shared vocabulary types are already free of CUDA. `ninfer::Tensor` and `ninfer::Weight`
(`vendor/src/core/tensor.h`) are plain structs over `void*`. Only `DeviceBuffer`, `DeviceArena` and `HostPinnedArena`
allocate through CUDA (`vendor/src/core/arena.h:20,111`).

### E3. Each layer kind carries three bodies, one per round kind

**GDN, `gdn_layer.cu`** (spot-checked: the three signatures at lines 137, 322 and 469):

| body | lines | serves | state addressing | arena | conv / recurrence form |
|---|---|---|---|---|---|
| `run_gdn_layer` | 137-307 | prefill, per-token route | host `seq->slot` | `model->scratch` | in place |
| `run_gdn_layer_graph` | 322-453 | decode round | whole-pool state, staged `decode_graph_slots` (364-374) | `decode_graph_scratch` | snapshot |
| `run_gdn_layer_record` | 469-600 | verify | records, `valid_columns` (513-519) | `decode_graph_scratch` | replay record |

Each body has its own trellis/NVFP4 branch (lines 85, 115, 533).

**GQA, `gqa_layer.cu`:**

- `run_gqa_layer` (114-374) serves prefill and the per-token route. It reads host-scalar positions (132, 167),
  narrows the block table to the sequence's own row (233-252), runs ignis's hq prompt attention for hq prefill
  (328-338), and carries the readout and the taps.
- `run_gqa_layer_batch` (395-571) serves decode and verify. It reads staged positions and the pool-wide table with
  staged row selectors (509-512), and runs masked A1 for verify.

NVFP4's `LinearPolicyMode` (`layer_internal.h:39-90`) rides every body signature (`layer_internal.h:116-204`), but only
the NVFP4 arm reads it (`gdn_layer.cu:75-78`).

### E4. Where today's code already crosses a layer-shaped seam

**Layer code doing step-level work:**

- **Sync and frontier advance in the layer files.** `ignis_gdn_layer_step_mode` synchronizes the stream and advances
  `seq->gdn_positions` (spot-checked at `gdn_layer.cu:658-672`). `ignis_gqa_layer_step_mode` does the same
  (`gqa_layer.cu:676, 686`). Both serve the per-token prefill route (`step.cu:473-475`).
- **The attention readout** (ADRs 0038-0041) is scored inside the GQA body (`gqa_layer.cu:304-351`) from targets the
  step builds (`step.cu:1084-1159`). Under hq it runs through a callback inside the prompt attention
  (`gqa_layer.cu:333`).
- **Test taps** sit inside the mixer (`gqa_layer.cu:195-201, 315`).
- **The layer builds KV views from pool internals** and checks the pool's format against the load's
  (`gqa_layer.cu:233-252, 596-605`).
- **The layer reads load options** (`model->prefill_attention`, `gqa_layer.cu:330`) and runs a vendored route resolver
  (`gqa_layer.cu:299-303`).

**Step code reaching into layer or state internals:**

- **Per-layer frontiers.** The step advances them with the 27B's topology formulas (`step.cu:1375-1383, 1782-1791,
  2684-2690`; the formulas at `layer_internal.h:241-256`).
- **hq ring invalidation.** The step clears the codec's ring bits at commit and on failure (`step.cu:1805-1826,
  2128-2145`).
- **The GDN fold.** `gdn_replay_fold` runs over every GDN layer (`step.cu:2151-2154`) from records the verify body
  writes (`gdn_layer.cu:518-519, 572-576`).
- **Verify staging.** The step writes it (`step.cu:1965-1996`) and the bodies read it (`gqa_layer.cu:751-761`,
  `gdn_layer.cu:513-519`).
- **The drafter.** Its forward runs inside the verify traversal (`decode_graph.cu:179-181`).
- **Load planning restates every body's intermediates** and every op's workspace query (`model.cu:562-605, 617-666,
  700-713`).

**Clean cuts already in the code:**

- the DFlash2 feature taps, at the residual level (`step.cu:1186-1206`; `decode_graph.cu:209-222`);
- the per-site weight-format dispatch (`trellis_body.h`; `gdn_layer.cu:85, 115`; `gqa_layer.cu:63, 84`).

### E5. The replay contract is already free of CUDA in what it asks

- **The traversal is built to be replayed.** `ignis_decode_graph_run_batch` is "Enqueue-only and free of host-visible
  branching on device data" (`decode_graph.cu:69-72`). Every per-lane input is read from stable staging buffers the
  round refreshes first (`step.cu:2475-2546`; ADR 0019).
- **Capture and replay are thin.** Capture is three calls around the traversal (`decode_graph.cu:321-376`), and
  replay is one (`step.cu:2549`).
- **The eager path runs the same function** (`step.cu:2562`), at a cost measured as 2.2% of a round
  (`step.cu:2466-2470`; [decode round anatomy](2026-09-18-decode-round-anatomy.md)).
- **Verify has the same shape** (`decode_graph.cu:143-273, 280-312`; `step.cu:2003-2014`).

### E6. Prefix reuse and checkpoints move bytes, not layouts

`seq_prefix.cu:14-17`: what each half of the state is made of "comes from the leaf's one state-section table
(ignis_seq_sections.h) ... This file moves bytes, owns lifetimes, and decides what to refuse." Checkpoints work the
same way. KV capture reads the page layout, but it is test-only (`ignis_kv_capture.h`).

### E7. Flash-Next already has the shape a shared program needs

- **One `forward`** for embed and every layer (`flash_next/program.cu:286`, spot-checked).
- **A `Batch`** (`flash_next/flash_next_internal.h:293`, spot-checked) whose device `slots` and `positions` make
  prefill and decode one code path (`flash_next/gdn.h:68`).
- **Cross-stage elementwise fusion through a pending epilogue.** `PendingInject` (`program.cu:297-305`, spot-checked)
  leaves a sublayer's inject pending and has the next mix apply it inside its own first launch.

### E8. What trellis decode spends per weight on CUDA

`trellis_decode.cuh:8-22` defines the mul1 codebook: `prod = state * 0x83DCD12D`, `bsum` = the sum of `prod`'s bytes,
`value = fp16((1024 + bsum) * fp16(0x1EEE) + fp16(0xC931))`. That is "one IMAD, one DP4A ... and half an HFMA2", plus
a funnel shift to extract the 16-bit window. The layout ties step `p` to tensor-core fragments: "lane t of a warp owns
steps 8t .. 8t+7, which are exactly its m16n8k16 B-operand registers".

Each step's value is a pure function of the 16-bit window ending at a known bit, so decode order is free on any
backend. The step-to-element permutation inside a tile is part of the encoding. Tile order across the matrix is
stored as `[in/16][out/16]` words (`ignis_trellis_linear.h:9-11`) and could be repacked per backend at load.

### E9. Numbers this exploration reuses from the repo

- **Trellis body:** 9,136 MB at the public quant's K mix (24.35 G parameters, ~3.0 bpw). Decode is 7,845 µs on the
  5090, 1,165 GB/s, 65.0% of the 1,792 GB/s roofline.
- **Trellis decode under SM caps:** at a 36-SM cap the body moves 316 GB/s, ~8.8 GB/s per SM; uncapped it moves
  ~6.9 GB/s per SM ([dense trellis kernels](2026-10-23-dense-trellis-kernels-on-the-5090.md)).
- **The 27B's NVFP4 artifact (ADR 0060 Context):** text scope 17.21 GB (16.03 GiB); body 14.51 GB at 4.76 bpw.
- **The NVFP4 decode round:** 1,166 nodes, 4.8% idle; host submission ~0.32 µs per node; all memcpy nodes together
  ~0.5% of the round ([decode round anatomy](2026-09-18-decode-round-anatomy.md)).
- **Launch cost:** about 1 µs is the bare cost of a launch in graph replay on the 5090
  ([Flash-Next decode fusion](2026-10-08-flash-next-decode-fusion.md)).

### E10. Dispatches in a trellis decode round, counted from the code

These counts are inferred from the code, not measured; the independent review's Appendix A gives them per op.

| part | launches per layer | layers | total |
|---|---:|---:|---:|
| GDN (`run_gdn_layer_graph`) | 21 | 48 | 1,008 |
| GQA (`run_gqa_layer_batch`) | ~20-22 | 16 | ~336 |
| embed, final norm, head, sample | | | ~6 |
| **round** | | | **≈ 1,350** |

What each GDN launch is:

- rmsnorm (1);
- `in_proj_qkv` and `in_proj_z`: rotate + decode, 2 each;
- conv snapshot (1);
- three qkv split copies (`gdn_layer.cu:406-408`);
- the gating projection: partial + reduce (2);
- recurrence (1);
- gated norm (1);
- residual copy (`gdn_layer.cu:439`, 1);
- out projection (2);
- rmsnorm (1);
- SwiGLU (2);
- down (2).

The GQA count adds the q/gate de-interleave copies (`trellis_body.cu:57-87`) and a conditional workspace memset
(`gqa_layer.cu:539-547`).

Fusing each segment (see the Finding) would bring Metal to about 7 dispatches per GDN layer and 8 per GQA layer,
**≈ 300-470 per round**.

### E11. Hardware facts used but not verified in this exploration

These come from vendor specifications as recalled, and were not checked against a source here.

| | RTX 5090 | M4 Pro | M4 Max (40-core GPU) | M3 Ultra | RDNA3/4 consumer |
|---|---|---|---|---|---|
| Memory bandwidth | 1,792 GB/s (the repo's roofline) | 273 GB/s | 546 GB/s | 819 GB/s | ~576-960 GB/s |
| On-chip memory per block | up to 228 KB | 32 KB threadgroup memory | 32 KB | 32 KB | ~64 KB LDS |
| Async global→on-chip copy | `cp.async`, TMA | none | none | none | limited |
| Matrix unit | `mma`, FP4/FP8 | `simdgroup_matrix` 8×8 | same | same | WMMA |
| SIMD width | warp 32 | simdgroup 32 | 32 | 32 | wave32 (RDNA), wave64 (CDNA) |
| Byte dot product | `dp4a` | none in MSL | none | none | `v_dot4_u32_u8` |
| Step replay | CUDA graph | indirect command buffer, or re-encode | same | same | hipGraph |
| Memory topology | discrete | unified | unified | unified | discrete (APUs unified) |

More details from the same recall:

- The M4 generation has no matrix accelerators inside the GPU; Apple added those with M5, exposed through Metal 4's
  tensor operations.
- An M4 Pro laptop ships with a 16-core or a 20-core GPU. The owner's machine's count is not yet known.
- By default macOS lets the GPU wire roughly 75% of RAM (~36 GB of 48 GB); a sysctl (`iogpu.wired_limit_mb`) raises
  it.

## Finding

### Observed (from the code)

1. **No translator can carry the leaf's hot kernels to Metal.**
   - The kernels' performance comes from Blackwell-specific constructs (E1) that have no Metal counterpart.
   - The tile shapes assume 228 KB of on-chip memory, against Apple's 32 KB.
   - What does port mechanically is the host side: the ~300 `cuda*` calls in the orchestration (E2).
2. **The orchestration is already separated from the kernels by file** (E2), and its types are free of CUDA. The real
   coupling is the ~45 direct op call sites with CUDA- and NVFP4-shaped signatures.
3. **A per-backend layer would duplicate a great deal.** Each layer kind has three bodies, one per round kind (E3), and
   the layer files do step-level work (E4). Rewriting the layer per backend would repeat, in every backend:
   - the three round-kind variants;
   - the readout and tap hooks;
   - the KV views and the record formats;
   - the scratch sizing;
   - the crossings E4 lists.
4. **"The same fusions everywhere" is already false inside CUDA.**
   - **NVFP4** fuses its input projections into one plane (`gdn_input_proj`, `attn_input_proj`) and SwiGLU into one
     op.
   - **Trellis** keeps separate matrices, because the quantizer quantizes them separately (ADR 0060 §4). Its SwiGLU
     sits in a decode epilogue, but prefill takes two passes (`ignis_trellis_linear.h:40-41`).
5. **Step-level state work does not need the state's physical layout.** Prefix reuse and checkpoints move section bytes
   that one table lists (E6). The verify fold and the hq ring invalidation are the only step-level operations on state
   contents (E4).
6. **The replay contract already transfers to Metal** (E5). The traversal is enqueue-only and branch-free on device
   data, with stable staging addresses. A Metal indirect command buffer and a cached re-encode both need exactly that.
7. **The repo already contains the shape a shared, backend-neutral program needs**, in Flash-Next (E7).

### Inferred: the roads

| road | verdict | why |
|---|---|---|
| CUDA → other translators (ZLUDA, chipStar/HIP, CUDA → SPIR-V → MSL through SPIRV-Cross, a commercial CUDA-to-AMD compiler) | **no** | None targets Metal, or they drop the inline PTX. Where something compiles, the tiles, async copies and matrix instructions are wrong for the target, so the cost is multiples, not percent. |
| A portable kernel language (Triton, Slang, WGSL through wgpu, Vulkan through MoltenVK, a proprietary-compiler language) | **no** | It is a rewrite anyway and lands a tier below hand-tuned code. On Apple, llama.cpp's Vulkan route is slower than its Metal route. |
| MLX as the runtime backend (mlx-c from Rust) | **no as a runtime, yes as an oracle and source** | It gets a model running fastest and is close to the state of the art on Apple. But it takes the forward pass out of the leaf (ADR 0009), and its quantized formats are not ignis's. MLX and ggml-metal are MIT-licensed, so they can be the reference implementations and a source of adapted ops (ADR 0051, tier 2). mlx-lm has Metal kernels for Qwen3-Next's gated delta net. |
| **A native Metal backend behind the same step-level C ABI** | **yes** | metal-cpp for the host side. MSL kernels precompiled to a `.metallib`, with function constants where CUDA uses template parameters. Pipelines built at load. Nothing is JIT-compiled at runtime, so the only per-call overhead is the API's own. |

**AMD sits between CUDA and Metal.** On the host it is nearly CUDA: HIP mirrors the runtime API, with hipGraph,
pinned memory and discrete memory. In its kernels it needs native work, like Metal:

- the PTX does not exist there;
- WMMA has its own fragment layouts;
- `__shfl_*_sync` written for 32 lanes breaks on wave64 (CDNA, not the consumer target).

An interface that holds for both CUDA and Metal therefore holds for HIP. Designing the seam against those two
backends, the most different pair, is the strongest test it can get. An interface with one adapter is a hypothesis;
with two it is a real seam.

### Inferred: where the seam goes

Three positions were held during the exploration. Two of them were retracted on the evidence of E3, E4 and E6.

1. **Seam at the layer** (the session's first proposal, endorsed by the advisor; **retracted**).
   - The argument: the CUDA layer files hold no kernels (E2), so they could move into a CUDA backend as they are, and
     each backend would get full fusion freedom.
   - Why it fails: a per-backend layer duplicates the three round-kind bodies, their hooks, scratch sizing and the E4
     crossings in every backend, for as long as the backends exist.
   - What survives: the CUDA side of any refactor can be pure code motion.
2. **Shared layers, backends implement ops only.**
   - It pins Metal at CUDA's op vocabulary (`cudaStream_t`, `LinearPolicy`, `detail::` resolvers) and at CUDA's
     ≈ 1,350 dispatches per round (E10).
   - Its premise, "the same fusions everywhere", is false even inside CUDA (Observed 4).
   - What it gets right: one program, one set of boundaries.
3. **Seam at the segment** (the independent review's proposal; **recommended**).

**The segment and the boundary rule.** A segment is a stretch of a layer's dataflow between two points that no single
kernel can cross, on any backend, without grid-wide synchronization: the output of a projection, which every consumer
needs whole, and a split-K reduction. Metal gives no co-residency guarantee across threadgroups, so it cannot cross
them at all; CUDA and HIP can only with cooperative launches. A cut that fails this test is an op, not a seam.
Elementwise tails that cross a boundary travel as a pending epilogue, as Flash-Next's `PendingInject` does (E7).

**The 27B's layer has five segments:**

1. **project:** the pre-norm and every projection reading the same input;
2. **mix:** the GDN or GQA core, which is per-head work;
3. the output projection and the residual;
4. the pre-norm and gate/up with SwiGLU;
5. down and the residual.

To these add about ten step ops: `embed`, `sample`, `argmax`, `prepare_verify`, `accept`, `select_hidden`,
`gdn_commit`, `kv_commit`, the permit mask, and penalty adjust.

**What fusion means under this seam.** The fusion *boundaries* belong to the model and hold on every backend. What
differs per backend:

- the *implementation*: tile shape, on-chip budget, async copies, cluster reductions, PDL;
- the *penalty for not fusing*:
  - on the 5090 a replayed launch costs ~1 µs (E9), so the ≈ 1,350 launches of a CUDA trellis round matter little;
  - on Metal the per-dispatch cost is unknown. If it is *d* µs, an unfused round costs ~1,350·*d* µs, about 3.7% of a
    ~36 ms step per µs of *d*: exactly the band between 90% and 95%.

**What goes on each side:**

| shared: one source, compiled per backend | per backend |
|---|---|
| C ABI entry points: load and plan, program prefill/decode, seq alloc/release/snapshot/restore/prefix/checkpoint, device ABI | `gpu::` runtime: memory, queue/stream, copies, fills, finish, events, record/run, capabilities, memory info |
| Binding weights by name into weight groups; topology checks | Weight binding, with an optional repack at load (trellis tile order, NVFP4 blockscale layout) |
| Load planning from `seg::scratch_bytes` and capabilities; plan lines (ADR 0030) | Segments: `project`, `gdn_mix`, `gqa_mix` (with readout and tap hooks), each with its `scratch_bytes` |
| State store: pool, slots, page accounting, section table, prefix reuse, checkpoints | The *contents* layout of KV pages and GDN slots, declared through the pool layouts |
| Step drivers: chunk loop, decode round, verify round (staging, run, readback, cut, redirect, penalties, frontiers) | Step ops (above) |
| Traversals: prefill chunk, decode round, verify round, DFlash2 taps | Kernels |
| One layer program per kind, over a `Batch` that covers prefill, decode and verify | CUDA: composed defaults that are today's op sequences, `LinearPolicyMode` included |

**The interface's rough shape** (a sketch from the independent review, not a design):

```cpp
namespace ignis::gpu {                 // one implementation per build, chosen by CMake; no virtual calls
using Addr = std::uint64_t;            // the CUDA/HIP pointer value; Metal: MTLBuffer.gpuAddress + offset
struct Caps {
  bool unified_memory;                 // one budget; nothing staged or host-mapped
  bool recordable;                     // graph or indirect command buffer; false: the traversal re-runs
  uint32_t group_onchip_bytes;         // ~228K / ~64K / 32K
};
Status upload(Stream &, Addr dst, const void *src, size_t bytes);   // stream-ordered, on Metal too
Status record(Stream &, Recording &, FunctionRef<Status(Stream &)> traverse);
Status run(Stream &, Recording &, FunctionRef<Status(Stream &)> traverse);  // replay, or traverse()
}
namespace ignis::seg {
enum class Pass : uint8_t { Prefill, Decode, Verify };
struct Batch { Pass pass; int32_t lanes, columns; gpu::Addr slots, positions, rope_positions, valid_columns;
               uint32_t max_visible; const ignis_seq *single; };     // flash_next::Batch, generalized
enum class Out : uint8_t { Store, AddFrom, SwiGLU };
Status project(const Batch &, Act x, const Norm *prenorm, Act normalized_or_none, const WeightGroup &,
               std::span<const Act> outputs, Out, Act residual, Scratch &, gpu::Stream &);
Status gdn_mix(const Batch &, const GdnLayer &, Act h, Act qkv, Act z, Act mixed, Scratch &, gpu::Stream &);
Status gqa_mix(const Batch &, const GqaLayer &, Act q, Act gate, Act k, Act v, Act mixed,
               const MixHooks *hooks, Scratch &, gpu::Stream &);
size_t scratch_bytes(Segment, const Shape &);                 // replaces model.cu:562-713's restatement
}
```

A shared GDN layer is then five calls:

```text
project(prenorm → qkv, z) → gdn_mix → project(out, AddFrom in) → project(prenorm → gate_up, SwiGLU) → project(down, AddFrom)
```

The backends implement it differently:

- **CUDA** keeps its launch order: `AddFrom` is today's copy plus in-place add.
- **Metal** fuses each segment into one or two dispatches. The norm and the trellis activation rotation go into the
  projection's prologue, and the residual becomes the epilogue.

**The amendments the shared-orchestration premise needs:**

1. **`gpu::` is address-based**, so the shared `void*` arithmetic survives (for example `step.cu:1251`). On Metal
   that means binding by GPU address: argument buffers or argument tables, residency sets, and a map from address to
   `contents()` for host access.
2. **The shared code depends on the recordable-traversal contract (E5), not on CUDA graphs.**
3. **Memory topology is a capability bit, not a fork.**
   - On unified memory these discrete-card mechanisms collapse into policy:
     - ADR 0062's host token embedding and ADR 0075's host-mapped pool (`MappedHostRegion`, `model_internal.h:283-299`;
       `device.cu:355-378`) are read in place;
     - the vision staging ring (`model_internal.h:203-243`) is pointless;
     - the retained host slots and the disk tier's transfer lane (`ignis_seq_internal.h:165-191`) become device slots;
     - Flash-Next's residency degenerates into ADR 0054's resident load.
   - HIP shares CUDA's discrete paths.
4. **Selection is at compile time per backend, plus at runtime per GPU family inside a backend.** Apple M1-M5 differ,
   as do RDNA3 and RDNA4 WMMA.

### Inferred: trellis decode on the M4 Pro is likely ALU-bound

This sits upstream of the seam: it may change which artifact the Apple card class serves, not where the seam goes.
The estimate below rests on two unmeasured lines of reasoning, and on the assumed hardware of E11.

1. **The bandwidth target in weights per second.** The body is 9,136 MB for 24.35 G weights, ~3.0 bpw. At 90% of
   273 GB/s the decode must turn out ~6.6·10¹¹ weights/s.
2. **The ALU budget.** 16-20 GPU cores × 128 lanes × ~1.4-1.6 GHz ≈ 2.9-4.1·10¹² lane-ops/s. That leaves **≈ 4.4-6.3
   instructions per weight for everything**: decode, FMA, loads, index math and reduction.
3. **What a faithful port spends.** CUDA's decode spends ~5 instructions per weight on decode alone (E8). MSL has no
   `dp4a` and no funnel shift, and Apple's 32-bit integer multiply is believed slower than its FMA. A faithful port is
   likely ~9-12 instructions per weight, **≈ 35-60% of the bandwidth target**.
4. **Cross-check from the 5090.** Under the SM cap the trellis decode route moves ~8.8-9.2 GB/s per SM (E9). An Apple
   GPU core with roughly half an SM's issue rate would move ~4.5 GB/s, or ~70-90 GB/s on the whole GPU: **≈ 26-33% of
   273 GB/s**.

**What that means in tokens per second** (body only, ceilings, all estimates):

| format on the M4 Pro | effective read | tok/s ceiling |
|---|---|---:|
| trellis at 90% of 273 GB/s | ~246 GB/s | ~27 |
| trellis ALU-bound at 26-33% | ~70-90 GB/s | **~8-10** |
| the NVFP4 artifact served as W4A16 (14.51 GB body) at 90% | ~246 GB/s | ~17 |

**The trap in the stated target.** A trellis GEMV at 95% ALU use meets the letter of "GPU cores > 95%" and still runs
about three times slower than a 4-bit format that saturates bandwidth. In decode the metric is bytes per second
against the measured read peak. A decode that is ALU-bound is a *format* problem, not a kernel-efficiency problem.

**The owner's options if the spike confirms the ALU bound:**

1. **Accept trellis at about a third of bandwidth.**
2. **Serve the Apple class the NVFP4 artifact as W4A16.**
   - Its 17.21 GB text scope fits in 48 GB.
   - Dequantizing FP4 through a lookup table is cheap.
   - It would be better quality than the served W4A4, since FP4 activations account for half or more of NVFP4's KLD
     (ADR 0060 Context).
   - The Blackwell blockscale layout would need a repack at load.
3. **A 4-bit format friendly to Metal.**
4. **Lean on DFlash2.** One trellis decode serves up to eight verify columns (`ignis_trellis_linear.h:22-25`), so ALU
   cost per generated token falls with acceptance. But the DFlash2 drafter is NVFP4 even on the 16 GB artifact
   (ADR 0060 §3), so speculation brings NVFP4 kernels and the drafter's ~15 ops onto Metal anyway.

**This is Apple's problem, not the format's.** RDNA3/4 have a byte dot product, and their ALU-to-bandwidth ratio is in
the 5090's range. The consumer AMD class can most likely keep the trellis artifact; this is an assumption until
measured on AMD.

### Inferred: replay on Metal

Replay sits below the traversal under any seam (E5). Metal has three ways to do it:

| option | cost | what it needs |
|---|---|---|
| **(a) Re-encode every round** | Host encode per dispatch per round. It can hide behind chunked commits while the GPU runs. | Nothing new: this is today's eager path. |
| **(b) Encode round N+1 on a worker thread while the GPU runs round N** | Little on the critical path. | Staging must be double-buffered, because a host write into a shared buffer is *not* stream-ordered as `cudaMemcpyAsync` is. Today the round syncs before returning (`step.cu:2635`), so there is no race until someone pipelines. |
| **(c) An indirect command buffer per width** | The least host work per round. | A dispatch-only traversal: today's copy and memset nodes (`gdn_layer.cu:406-408, 439`; `gqa_layer.cu:539-547, 556`; `trellis_body.cu:57-87`; `decode_graph.cu:213, 227`) must become kernels or be fused away. Slot indirection through device-read selectors must hold. |

At ~40 ms per token on the M4 Pro, a thousand dispatches plausibly cost 2-5%, exactly the band between 90% and 95%.

### Inferred: quality on a non-CUDA backend

The gate is KLD against BF16, not agreement between engines. The 27B in BF16 is about 54 GB: it fits neither the 48 GB
machine nor the ~36 GB the GPU may wire by default. So:

- BF16 reference logits must be precomputed once, on the 5090 or on the CPU, over a fixed corpus, and carried to the
  Mac.
- Comparing Metal's trellis output with the 5090's trellis output measures engine agreement, not quality.

Per-segment conformance tests can use the f64 layer references that already exist in Rust
(`crates/artifact/src/f64_reference.rs:185, 552, 665`).

## Implications

- **A Metal backend is a native leaf implementation behind today's C ABI**, not a translation. ADR 0009's boundary,
  with Rust above the step, survives unchanged. Rust still needs work:
  - a single memory budget for unified memory (`crates/core/src/residency/plan.rs`);
  - an Apple card class (ADRs 0063/0064);
  - topology reported by the device ABI in place of NVML figures (`ignis_device_nvml_mem_info` in `ignis_device.h`
    is CUDA- and NVML-specific).
- **If a second backend is built, the seam to aim for is the segment**, with the boundary rule, a `Batch` covering the
  three round kinds, and an address-based `gpu::` runtime. CUDA's side is code motion, and it can be accepted on a check
  stronger than KLD:
  - the captured decode and verify graphs come out node-for-node identical (kernel name, grid, order);
  - the canary's logits are bit-identical;
  - then the 99% gate runs once, at the end.

  The risky step is collapsing the three body variants into one program. It comes after the CUDA segments hold the
  variants internally.
- **Order: experiment before structure.** The independent review proposed five steps of CUDA refactoring before any
  Metal code. The session and the advisor disagree with that order, in line with the owner's rules (experiment before
  design, no overengineering):
  1. a throwaway spike first (S1-S4 below), written against the segment signatures as discipline but not kept;
  2. the refactor only after S2 and S4 have answered.
- **The format question comes before the seam.** Whether the Apple class serves trellis or a 4-bit format follows from
  S2. The owner's "trellis directly" stands until S2 contradicts it.
- **The first Metal scope:**
  - in: the 27B's decode round, BF16 KV (ADR 0022's oracle format), no speculation (refused by capability; the load
    options already express "no window");
  - after it: prefill, then verify and DFlash2;
  - outside the first seam: the hq codec, Flash-Next, vision and the image model. The seam must not preclude them:
    `gpu::` will need fork and join for Flash-Next's side branches (`flash_next/program.cu:376-460`).
- **Decisions that would follow, after measurement.** If it proceeds, the work would need ADRs for:
  - the segment seam (amending ADR 0001/0009 on what the leaf's interior looks like);
  - Metal kernels adapted from MLX/ggml-metal under ADR 0051;
  - the Apple card class and the unified-memory budget;
  - the reference-logit oracle for backends that cannot hold BF16.

  None is written yet; this finding is their evidence.

## Limits and unknowns

- **Nothing was run on Apple or AMD hardware.** Every Apple and AMD figure in E11 and in the ALU estimate is recalled
  from vendor specifications or assumed, not measured or sourced here. That covers bandwidths, core counts, clocks,
  lanes per core, the lack of `dp4a` and funnel shift in MSL, and Apple's integer multiply rate.
- **The ALU-bound estimate rests on two unmeasured lines of reasoning** that agree only roughly (26-33% and 35-60%).
  It could be wrong in either direction: Apple's integer throughput, the compiler, and a decode reorganized for
  `simdgroup_matrix` could all move it.
- **The owner's machine is an M4 Pro laptop with 48 GB; its GPU core count (16 or 20) is unknown.** The ALU budget
  scales with it.
- **Metal's per-dispatch cost** (host encode, GPU fixed cost of a dependent dispatch, indirect command buffer replay)
  is unknown. It decides how much the segment seam's fusion is worth on Metal, as opposed to its maintenance value.
- **The trellis round's ≈ 1,350 launches are counted from the code, not from a capture.** The NVFP4 round's 1,166 is
  measured. The Metal figure of ≈ 300-470 assumes each segment fuses into one or two dispatches. S5 and S6 test that,
  including whether a GDN head's 64 KB fp32 state can stream through registers under 32 KB of threadgroup memory.
- **`simdgroup_matrix`'s per-thread fragment layout is not a documented contract**, so the trellis "decode straight
  into fragments" trick (E8) may not carry over; Metal would decode into threadgroup memory or registers in its own
  order.
- **Whether an indirect command buffer can carry the slot indirection** of ADR 0019, and how Metal binds by GPU
  address (argument buffers, argument tables, residency sets), is unverified.
- **Whether a laptop's sustained bandwidth and clocks** match a short microbenchmark's peak is unknown. Thermal
  throttling would move the denominator.
- **macOS's default GPU working-set ceiling** (~75% of RAM) and the sysctl that raises it are recalled, not verified.
- **Whether hosted macOS CI runners expose Metal compute** was not checked. The GPU gates likely need the owner's
  machine, as they need the 5090 today.
- **HIP on Windows** (the owner's development platform) was not examined. Neither were RDNA3/4 WMMA's fragment
  layouts, nor whether consumer AMD keeps trellis.
- **The roads' verdicts draw on recalled knowledge of external projects, not on sources fetched here.** That covers
  what the translators target, llama.cpp's Vulkan route against its Metal route on Apple, MLX's formats, and mlx-lm's
  gated-delta-net kernels. None of these was re-checked during this exploration.
- **The code citations are at `main` 1a9f8e1a.** The independent review read the files whole; the session spot-checked
  only the claims that carry the recommendation (E3's three GDN signatures, E4's sync and frontier advance, E7's
  `forward`, `Batch` and `PendingInject`, E5's eager cost, E6's byte-moving comment, E8's instruction mix).

## Follow-ups

The spike, in this order, with its decision rules (S2 can overturn more than the seam):

| # | measure | decides |
|---|---|---|
| S1 | **The peaks:** streaming-read GB/s, and FP16 `simdgroup_matrix` FLOP/s, on the owner's machine, in the same thermal state as the benchmarks. | The denominators of the 90/95% targets: decode in bytes/s, prefill in FLOP/s. |
| S2 | **Trellis decode rate:** one GEMV at the 27B's shapes, at 1 and 8 columns, against S1. The shapes are GDN qkv 5120×10240 K3, gate/up 5120×2×17408 K3, down 17408×5120 K3, head 5120×248320 K4. Compare an NVFP4-as-W4A16 GEMV and an affine 4-bit GEMV at the same shapes. | The Apple class's artifact. Trellis ≥ ~80% of S1: stay. Trellis < ~60% while a 4-bit format reaches ≥ 90%: the owner chooses. |
| S3 | **Dispatch cost:** (a) the GPU's fixed cost for a dependent trivial dispatch in one serial compute encoder; (b) host encode cost per dispatch with bind-by-address; (c) indirect command buffer replay of the same; (d) a blit against a copy kernel inside the round. | What fusion is worth on Metal. |
| S4 | **One full decode step, two ways:** 64 layers with real weights and state, (i) fused per segment at ≈ 300-470 dispatches, (ii) the same math split at CUDA's op boundaries at ≈ 1,350. Record GPU time, host encode time, and the gap at step start. | (ii) within ~2% of (i): shared fine ops are viable on Metal, and the segment seam keeps only its maintenance case. (ii) ≥ ~5% slower: the segment seam is required. |
| S5 | **Rotation and reduction:** the activation rotation in each projection's prologue against a separate dispatch; RMSNorm recomputed per threadgroup; whether the last-arrival split-K reduction (`trellis_linear_decode.cu:10-21`) can be written with Metal's device-scope atomics and fences. | The dispatch count per segment. |
| S6 | **The GDN mixer as one dispatch per layer:** conv, l2norm, gating, recurrence and gated norm per head, against the state traffic (48 layers × 48 heads × 64 KB × 2 ≈ 302 MB per token, ~1.1 ms at 273 GB/s). | Whether `gdn_mix` is one kernel. |
| S7 | **Replay options:** re-encode with chunked commits, encode-ahead, and an indirect command buffer, including whether its bindings carry the slot indirection. | `gpu::record/run` on Metal. |
| S8 | **Host round trip:** commit-to-completion latency for the decode round's one sync and the verify round's two (`step.cu:2030, 2202`). | Whether a device-resident accept is worth building. |
| S9 | **Sustained bandwidth** on the shared 48 GB pool with the desktop running, and thermal behaviour over a long decode on a laptop. | Whether S1's peak is the honest denominator. |
| S10 | **BF16 attention decode** at 8K, 32K and 128K. BF16 KV is 64 KiB per token, so 2 GiB per step at 32K (~7.9 ms at 273 GB/s against a ~36 ms body). | When KV traffic overtakes weight traffic. |

Prerequisite for any quality claim on Metal: the BF16 reference logits, precomputed off the Mac.

The work is tracked on [GitHub #439](https://github.com/gpillon/ignis/issues/439) (parked as `idea`).
