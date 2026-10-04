# Flash-Next in ignis — the feature, its tickets, and how to run them

GitHub: master #298; specs 01-06 are #299-#304; phase 2 is #305.

The feature-level document the Flash-Next tickets decompose
(`docs/agents/issue-tracker.md`). Read this first, then the ticket's spec.

## What and why

ignis gets a second model, **Qwen3.8-Flash-Next** (125B MoE, 6B active). It is
selected at start and never loaded beside the 27B. It serves the work where the
27B falls short: general knowledge, languages, mathematics, long technical
documents. To fit one RTX 5090:
- **experts:** trellis-coded with a bit width K per expert, 2.5 bits per weight
  on average, all of them in pinned host RAM, with a VRAM expert cache in
  front;
- **other linears:** FP8;
- **n-gram table:** INT4, read from NVMe.

The evidence is the compression study of 2026-10-03/04 in
`F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03/`
(`RISULTATI_3.md`, runs 1-8; `review/` for memory, placement and prefill).

**Decisions:**
- ADR 0043: a second model without a ninfer reference; the checkpoint's own
  modeling code is the oracle.
- ADR 0044: trellis experts with a K per expert, on our own kernels.
- Owner decisions of 2026-10-04:
  - K per expert;
  - MoE kernels written by us;
  - ExLlamaV3 as a reference only;
  - the fast version directly;
  - hq-e8-2b KV in scope;
  - prompt reuse right after serving;
  - the model switch in phase 2.

## The tickets

| spec | what | blocked by | can start without the artifact |
|---|---|---|---|
| 01 | converter + artifact + reference recordings | — | everything; the conversion run itself is the long GPU job |
| 02 | MoE kernels (router, per-K trellis GEMV / grouped GEMM, shared expert, combine) | 01 for the full-shape checks | yes: reduced-shape fixtures come from exllamav3 directly |
| 03 | expert residency (pinned pool, VRAM cache, lookahead, scan-resistant prefill) | 01 (traces, sidecar), 02 (kernels) | CPU policy model, plan arithmetic, miss-path measurement |
| 04 | forward + serving (topology, GDN 2560, QSA 24/2 + indexer, hyper-connections, n-gram, FP8, hq-e8-2b, 3 lanes) | 01, 02, 03 | topology, the GDN layer/head conflation fix, CPU tests, op kernels with recorded fixtures |
| 05 | prompt reuse on Flash-Next state | 04 | section sizes and CPU plan tests |
| 06 | final measurements of what was built | 04, 05 | — |
| phase 2 | runtime model switch: notes only, not for now | — | — |

## Environment

- **Study code and data** (untracked, read by absolute path from any worktree):
  `F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03/`.
  - `real/`: pipeline, run 8, fetcher, routing data, n-gram table cache;
  - `review/`: analyses, `run4_paired.py`;
  - `.scratch/kld-2026-09-24/`: the 27B KLD windows the calibration uses.
- **Python:** `F:/ai/ngram-venv`, with torch 2.13 cu130 and exllamav3 1.5.3,
  whose extension is already built for this card. **Never set
  `TORCH_CUDA_ARCH_LIST`** when importing exllamav3: it rebuilds for ~6
  minutes.
- **Checkpoint:** `Qwen/Qwen3.8-Flash-Next` revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`, by HTTP range requests.
  - Internet and the NAS share WiFi: ~42-50 MB/s.
  - Never stage the checkpoint on Y:.
- **Artifact output:** `F:/ai/models/Qwen3.8-Flash-Next-ignis/`, ~71 GB.
- **Worktrees:** `../.inference-qwen-worktrees/<branch>/`, one per ticket.

## The GPU: one card, one run at a time

- `make gpu-status` before any GPU work.
- Take the shared lock (`../.inference-qwen-worktrees/.swarm/gpu-lock.sh`:
  `try` / `acquire` / `release` / `status`, logged in `gpu.log`).
- A second job on the card kills the first with no diagnostic.
- Sequence for a night:
  1. **Short GPU prerequisites first:**
     - spec 02's reduced-shape exllamav3 fixtures;
     - spec 03's miss-path bandwidth measurement;
     - a 2-layer dry run of the converter.
  2. **Then the full conversion** (spec 01, ~9-10 hours): it holds the lock
     until it ends.
     - Launch it detached: PowerShell `Start-Process` with
       `cmd /v:on /c ... & echo !ERRORLEVEL! > x.exit`. Background Bash
       watchers get killed under memory pressure.
     - Watch it with a recurring heartbeat.
     - It resumes per layer.
  3. **During the conversion:** only CPU work.
     - Rust reader, binder and packer (01);
     - kernels compiled but not GPU-tested (02);
     - CPU policy model (03);
     - topology and conflation fix (04).
- Keep total VRAM ≤ ~28 GB, desktop included.

## Disk

- F: had ~105 GB free on 2026-10-04.
- The conversion needs ~71 GB of output plus about one layer of work files. The
  packer deletes them as it appends.
- Every worktree's cargo target dir also lives on F:. Keep at most two CUDA
  worktrees built at once, and check `df` before launching the conversion.
- The converter refuses below output + 15 GB.
- Never delete the study's n-gram table cache: the owner decides.

## Rules

- **Tests:** every change ships with a test. `cargo test` must pass
  workspace-wide; GPU tests are `--ignored` and run under the GPU profile.
- **Formatting:** never run `cargo fmt`; diffs are semantic only.
- **Merging and publishing:** merge a ticket's branch into local `main` only
  when every acceptance criterion holds. Never push.
- **Closing:** close the ticket with a comment listing the evidence for each
  criterion. Otherwise leave it open with a progress comment saying what holds,
  what does not and why.
- **Vendored files:** never edit them (ADR 0043). New geometries are our code
  beside them.
- **ExLlamaV3 and QTIP:** never copy ExLlamaV3 code into the engine, and never
  read its QTIP-derived GEMV.
- **Calibration data:** only the allowlisted sources; contributed fixtures and
  partner material never enter a corpus or the repo.

## What one night can deliver

Realistically:
- the converter written, dry-run and its full run launched or finished
  (spec 01);
- the MoE kernels written and tested on reduced fixtures (spec 02);
- the CPU sides of 03 and 04.

Specs 03-06 need the artifact and the kernels on the GPU and take further
sessions. A good night ends with:
- 01 done or running;
- 02 green on reduced fixtures;
- progress comments on 03/04.
