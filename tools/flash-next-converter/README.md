# Flash-Next converter

Spec 01 (`docs/specs/flash-next/01-the-flash-next-artifact.md`, GitHub #299): the offline
tool that turns `Qwen/Qwen3.8-Flash-Next` at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` into ignis **work files** (trellis experts with a K
per expert, FP8 non-experts, the INT4 n-gram table), the **reference recordings** and the
**routing traces**, in one layer-streamed GPU pass. The byte contract is
`docs/specs/flash-next/layout.md`; the Rust packer (`ignis-artifact-pack`, `crates/artifact`)
assembles the `.ninfer` container from the work files. Python never writes the container.

The recipe and the pass are the compression study's run 8 (`real/e2e8.py`), ported here.
Nothing under `.scratch` is imported or named by the Python: the study's token files, its
n-gram cache and the coverage sample's parquet files are passed by command-line arguments
(`launch.ps1` holds their default paths) and pinned by `corpus_manifest.json`.

## Files

| File | What it is |
|---|---|
| `convert.py` | The command: `run` (the conversion; preflight, resume, stop file), `verify` (after packing: decode sampled experts from the container) and `fixture` (a reduced work tree for the packer's tests). |
| `pipeline.py` | The pass: three streams per layer (BF16, quantized, FP8-only), Hessians, the four-K sweep, allocation, work files, replay of finished layers, the n-gram table sweep, the kernel recordings. |
| `finish.py` | After the last layer: the head, stored references, the G1 fixture, traces, the verdicts (acceptance 2-4), `converter.json`, `report.txt`. |
| `driver.py` | The layer loop: DONE markers checked on resume, replay, the crash-safe checkpoint, the stop file (exit 75). |
| `layout.py`, `fp8.py`, `table.py` | The byte formats of layout.md: expert records and index, FP8 row-scale payloads, INT4 g32 table rows, the non-expert encoding rule, the DONE protocol. |
| `allocate.py` | Run 8's Lagrangian K allocation, rate = K + fp16 scales. |
| `corpus.py`, `corpus_manifest.json` | The study's run 4-8 calibration/test set in its order; every source checked against the allowlist; token files pinned by sha256; the n-gram cache pinned per shard (size and the sha256 of its first and last MiB, all 128 shards; a full hash of 102 GB is too slow); the coverage sample's parquet files pinned by sha256. |
| `container.py` | Reads expert records back out of a packed container (for `verify`). |
| `fetch.py` | HTTP range fetcher at the pinned revision (the study's, ported). |
| `trellis.py` | The only exllamav3 user: batch quantizer, the decode oracle, the reconstruct checksum. |
| `preflight.py` | GPU lock owner, GPU workloads by name, VRAM in use (the Makefile's gpu-guard rules), VRAM cap, disk. |
| `fixture.py` | The reduced work tree (2 layers, 8 experts in >= 3 K classes, 1,000 table rows). |
| `launch.ps1` | Detached launch: hidden process, streams to files, exit-code sentinels, GPU lock held for the whole chain (pass, packer, verify) and released at its end. |
| `test_*.py` | Unit tests (CPU only). |

## Tests

```
F:/ai/ngram-venv/Scripts/python.exe -m pytest tools/flash-next-converter -q
```

## The full conversion

Check `make gpu-status` and `bash ../.inference-qwen-worktrees/.swarm/gpu-lock.sh status`
first. Then, from the repository root:

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools/flash-next-converter/launch.ps1 `
    -Out F:/ai/models/Qwen3.8-Flash-Next-ignis -LockOwner flash-next-convert `
    -StopFile F:/ai/models/Qwen3.8-Flash-Next-ignis/STOP
```

Defaults: `--prefetch 0` (RAM), the BF16 stream's checkpoint on `E:/flash-next-ckpt`, the
other streams' on `C:/flash-next-ckpt-small`, the packer at `target/release/ignis-artifact-pack.exe`.

- **Watch:** `convert.log` in `-Out` (one line per layer: time, RSS, VRAM peak, MoE dB beside
  run 6 / run 8, rates, K histogram); `convert.exit` appears when the pass ends.
- **Exit codes of the pass:** 0 done; 75 stopped by the stop file; 2 refused (GPU lock, a GPU
  workload or VRAM in use, disk, a work tree of another configuration); 3 done but an
  acceptance check FAILED (rates over `--budget`, the MoE error of acceptance 3, KLD or MMLU of
  acceptance 4, or a work-file re-decode mismatch): the report names the fallback, a
  re-conversion at a 3.0-bit mean into a new `-Out` with `-Extra "--budget 3.0"` (its rate
  check then follows the 3.0 budget); 1 error (traceback in `convert.log`).
- **After an exit 0** the hidden process runs the packer (`pack.exit`) and then `convert.py
  verify` (`verify.exit`, `verify.json`: the sampled expert projections decoded from the
  container's bytes against the sha256 the pass recorded), still under the lock.
- **GPU windows:** create the stop file; the converter checkpoints before its next layer and
  exits 75, releasing the lock. Delete the stop file and launch the same command again.
- **Resume after a crash:** launch the same command. Every finished layer's files are checked
  against its DONE (a torn one is redone); the state comes from the checkpoint, or is replayed
  from the finished layers' work files and re-fetched BF16 weights (~130 s per layer) when the
  checkpoint is missing, torn or past a redone layer.
- **One configuration per tree:** a dry run goes to its own `-Out`, e.g.
  `-Out F:/ai/models/fn-dryrun -Extra "--layers 2 --table-shards 4 --ckpt-every 1"`.

## Departures from the study

- **An expert with no calibration token** is quantized with the layer's Hessian (gate/up: the
  layer H; down: its activations' second moment over 64K calibration tokens), as spec 01
  says, where run 8 left it to exllamav3's uncalibrated fallback. Such an expert still gets
  K = 2 (zero routed energy). The dry run's layer 0 had 6 of them.
- **FP8 scales are bf16** (the container's existing format), rounded up from amax/448.
- **The hot-row list** ranks the calibration set together with the research session's n-gram
  coverage sample (layout.md §7.3).

## Resources

Measured on the 2-layer dry run of 2026-10-05 (`--layers 2 --table-shards 4`), on this machine.

- **Layer 0 reproduces run 8:** MoE error -20.97 dB, rates 2.4982 / 2.5000, K histograms
  gu [189, 163, 156, 4], dn [228, 130, 133, 21], all identical to run 8's layer 0 (before the
  zero-token Hessian fallback, which touches 6 unrouted experts there).
- **Time:** GDN layer ~700 s (K sweep ~635 s, BF16 stream ~30-85 s, quantized and FP8-only
  streams ~17 s); a QSA layer adds ~200 s (the checkpoint's own indexer on the eight 8192-token
  windows, ~12.9 s per window per stream). 36 x 700 + 12 x 900 s, plus start-up, checkpoints,
  the layer fetches of `--prefetch 0` (~45-120 s each) and the head (~70 s): **~11 h**.
- **VRAM:** peak reserved 23.8 GB under the 24 GB cap (`--vram-gb`); ~25.6 GB on the card with
  the desktop.
- **RAM:** peak working set 39.1 GB during layer 0 with `--prefetch 1` (states 20.4 GB, layer
  1's n-gram rows 4.4 GB, the next layer's BF16 weights ~5 GB, the K sweep's records ~3.6 GB);
  `--prefetch 0` (the launcher's default) saves the ~5 GB of the next layer.
- **Disk:** work files ~0.89 GB per layer (experts 0.79), globals 1.28 GB, the INT4 table
  28.8 GB, references and traces ~1 GB: **~74 GB on F:** at the end of the pass; the preflight
  refuses below what is still to write + 15 GB (+ the small checkpoint when it is on the same
  drive). Packing moves the work files into the container and deletes them as it goes (peak
  ~74 GB + one unit).
- **Checkpoint:** one slot, 13.5 GB (BF16 stream) on `E:/flash-next-ckpt` + 6.9 GB (quantized
  and FP8-only streams) on `C:/flash-next-ckpt-small`; written in 83 s, loaded in 38 s; every
  6 layers (`--ckpt-every`) and at the stop file. The old slot's files are removed before the
  new ones are written (two copies do not fit), so the peak is one slot; the slot is marked
  invalid first and a torn slot is never loaded.
- **Resume:** checked on the dry run: a relaunch with the stop file present loads the
  checkpoint and exits 75; a relaunch without it resumes from the checkpoint and writes
  references bit-identical to the uninterrupted run's; with the checkpoint torn, the replay
  from layer 0 writes them bit-identical too.
