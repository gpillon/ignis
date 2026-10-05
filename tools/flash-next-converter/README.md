# Flash-Next converter

Spec 01 (`docs/specs/flash-next/01-the-flash-next-artifact.md`, GitHub #299): the offline
tool that turns `Qwen/Qwen3.8-Flash-Next` at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` into ignis **work files** (trellis experts with a K
per expert, FP8 non-experts, the INT4 n-gram table), the **reference recordings** and the
**routing traces**, in one layer-streamed GPU pass. The byte contract is
`docs/specs/flash-next/layout.md`; the Rust packer in `crates/artifact` assembles the
`.ninfer` container from the work files. Python never writes the container.

The recipe and the pass are the compression study's run 8 (`.scratch/flash-next-compression-2026-10-03`,
`real/e2e8.py`), ported here: nothing under `.scratch` is imported at run time; the study's
token files and n-gram cache are read as data, pinned by sha256 (`corpus_manifest.json`).

## Files

| File | What it is |
|---|---|
| `convert.py` | The command: `run` (the conversion; preflight, resume, stop file) and `fixture` (a reduced work tree for the packer's tests). |
| `pipeline.py` | The pass: three streams per layer (BF16, quantized, FP8-only), Hessians, the four-K sweep, allocation, work files, replay of finished layers, the n-gram table sweep. |
| `finish.py` | After the last layer: the head, stored references, the G1 fixture, traces, KLD / MMLU with McNemar, the self-check, `converter.json`, `report.txt`. |
| `driver.py` | The layer loop: DONE markers, replay, the crash-safe checkpoint, the stop file (exit 75). |
| `layout.py`, `fp8.py`, `table.py` | The byte formats of layout.md: expert records and index, FP8 row-scale payloads, INT4 g32 table rows, the non-expert encoding rule. |
| `allocate.py` | Run 8's Lagrangian K allocation, rate = K + fp16 scales. |
| `corpus.py`, `corpus_manifest.json` | The study's run 4-8 calibration/test set in its order, sha256-pinned, sources checked against the allowlist. |
| `fetch.py` | HTTP range fetcher at the pinned revision (the study's, ported). |
| `trellis.py` | The only exllamav3 user: batch quantizer and the decode oracle. |
| `preflight.py` | GPU lock owner, GPU workloads by name, VRAM in use, VRAM cap, disk. |
| `fixture.py` | The reduced work tree (2 layers, 8 experts in >= 3 K classes, 1,000 table rows). |
| `launch.ps1` | Detached launch: hidden window, streams to files, `convert.exit` sentinel, GPU lock held for the process's life. |
| `test_*.py` | Unit tests (CPU only). |

## Tests

```
F:/ai/ngram-venv/Scripts/python.exe -m pytest tools/flash-next-converter -q
```

## The full conversion

Check `make gpu-status` and `bash ../.inference-qwen-worktrees/.swarm/gpu-lock.sh status` first. Then, from the
repository root:

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools/flash-next-converter/launch.ps1 `
    -Out F:/ai/models/Qwen3.8-Flash-Next-ignis -LockOwner flash-next-convert `
    -StopFile F:/ai/models/Qwen3.8-Flash-Next-ignis/STOP
```

- **Watch:** `F:/ai/models/Qwen3.8-Flash-Next-ignis/convert.log` (one line per layer: time,
  RSS, VRAM peak, MoE dB beside run 6 / run 8, rates, K histogram); `convert.exit` appears when
  the process ends.
- **Exit codes:** 0 done; 75 stopped by the stop file; 2 refused by the preflight (lock, GPU busy,
  disk); 3 done but acceptance 3 or 4 FAILED (the report names the 3.0-bit fallback:
  re-run with `-Extra "--budget 3.0"` into a new `-Out`); 1 error (traceback in the log).
- **GPU windows:** create the stop file; the converter finishes its layer, checkpoints and exits
  75, releasing the lock. Delete the stop file and launch the same command again: it resumes.
- **Resume after a crash:** launch the same command. Finished layers (with `DONE`) are never
  redone; the state comes from the checkpoint, or is replayed from the finished layers' work
  files and re-fetched BF16 weights (~2 min per layer, network-bound) when the checkpoint is
  missing or torn.

## Resources

Measured on the 2-layer dry run of 2026-10-05 (`--layers 2 --table-shards 4`), on this machine.

- **Layer 0 reproduces run 8:** MoE error -20.97 dB, rates 2.4982 / 2.5000, K histograms
  gu [189, 163, 156, 4], dn [228, 130, 133, 21], all identical to run 8's layer 0.
- **Time:** GDN layer ~700 s (K sweep ~635 s, BF16 stream ~30-85 s, quantized and FP8-only
  streams ~17 s); a QSA layer adds ~200 s (the checkpoint's own indexer on the eight 8192-token
  windows, ~12.9 s per window per stream). 36 x 700 + 12 x 900 s, plus start-up, checkpoints
  and the head (~70 s): **~10.3 h** for the full run.
- **VRAM:** peak reserved 23.8 GB under the 24 GB cap (`--vram-gb`); ~25.6 GB on the card with
  the desktop.
- **RAM:** peak working set 39.1 GB during layer 0 (states 20.4 GB, layer 1's n-gram rows
  4.4 GB, the next layer's BF16 weights ~5 GB, the K sweep's records ~3.6 GB); 34.5 GB at the
  end of layer 0, ~35 GB per layer once layer 1's n-gram rows are freed. `--prefetch 0`
  saves ~5 GB and costs the fetch of each layer (~45-120 s at the measured 40-120 MB/s).
- **Disk:** work files ~0.89 GB per layer (experts 0.79), globals 1.28 GB, the INT4 table
  28.8 GB, references ~0.2 GB, traces ~0.8 GB: **~74 GB on F:**; the preflight refuses below
  what is still to write + 15 GB (+ the small checkpoint when it is on the same drive).
- **Checkpoint:** one slot, 13.5 GB (BF16 stream) on `E:/flash-next-ckpt` + 6.9 GB
  (quantized and FP8-only streams) under `work/state/ckpt`; written in 83 s, loaded in 38 s;
  every 6 layers (`--ckpt-every`) and at the stop file. Two copies do not fit on these disks:
  the slot is invalidated before it is rewritten and a torn slot is never loaded. Without a
  usable checkpoint the finished layers replay at ~130 s each (BF16 weights re-fetched).
- **Resume:** checked on the dry run: a relaunch with the stop file present loads the
  checkpoint and exits 75; a relaunch without it resumes from the checkpoint, replays the
  finished layer and writes references bit-identical to the uninterrupted run's.
