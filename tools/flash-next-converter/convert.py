"""Qwen3.8-Flash-Next -> ignis work files, references and traces (spec 01, GitHub #299).

    python convert.py run --lock-owner flash-next-convert --ood-dir ... --windows-dir ... --table-cache ...
    python convert.py fixture --out DIR          # a reduced work tree for the packer's CPU tests

Exit codes: 0 done (PASS or dry run), 75 stopped by the stop file, 2 refused by the
preflight, 3 done but the quality acceptance FAILED (the report names the fallback),
1 error. See README.md for the full command and its disk, RAM and time figures.
"""
import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GB = 1e9


class Log:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, *a):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def expected_output_bytes(n_layers, table_shards):
    """Work files + references + traces, for the disk preflight (README's figures)."""
    import layout
    import pipeline as P
    experts = P.E * (layout.record_bytes("gu", 5) + layout.record_bytes("dn", 5))   # mean ~2.5 bits
    per_layer = experts + 0.08 * GB + 0.017 * GB                                     # non-experts, routing
    table = table_shards * P.SHARD_ROWS * 90
    return n_layers * per_layer + 1.28 * GB + table + 1.2 * GB


def state_bytes(conv):
    big = small = 0
    for stream, sets in conv.stream_sets.items():
        for s in sets:
            n, T = conv.ids[s].shape
            b = n * T * 4 * 2560 * 2
            if stream == "bf16":
                big += b
            else:
                small += b
    return big, small


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def cmd_run(a):
    import driver
    import preflight
    os.makedirs(a.out, exist_ok=True)
    log = Log(os.path.join(a.out, "convert.log"))
    log(f"convert {' '.join(sys.argv[1:])}")
    t_start = time.time()
    try:
        preflight.check_gpu(a.lock_owner)
        out_need = expected_output_bytes(a.layers, a.table_shards) - dir_bytes(a.out)
        small_dir = a.ckpt_small_dir or os.path.join(a.out, "work", "state", "ckpt")
        # state checkpoint sizes are known once the corpus is loaded; reserve them now
        big_ck, small_ck = 13.6 * GB, 6.9 * GB
        same_drive = os.path.splitdrive(os.path.abspath(small_dir))[0].lower() == \
            os.path.splitdrive(os.path.abspath(a.out))[0].lower()
        free = preflight.check_disk(a.out, max(out_need, 0) + (small_ck if same_drive else 0),
                                    a.disk_margin_gb * GB, "output")
        log(f"disk: {free / GB:.1f} GB free on {a.out}; output still to write {max(out_need, 0) / GB:.1f} GB "
            f"+ small checkpoint {small_ck / GB:.1f} GB + margin {a.disk_margin_gb} GB")
        ck_dirs = [a.ckpt_dir, small_dir]
        try:
            preflight.check_disk(a.ckpt_dir, big_ck - dir_bytes(a.ckpt_dir), 1 * GB, "checkpoint")
        except preflight.Refused as e:
            log(f"WARNING {e}: no state checkpoint; a relaunch replays the finished layers from layer 0")
            ck_dirs = None
        import torch
        import trellis
        trellis.check_version()
        frac, total = preflight.cap_vram(a.vram_gb)
        log(f"VRAM cap {a.vram_gb} GB ({frac:.3f} of {total / GB:.1f} GB)")
    except preflight.Refused as e:
        log(f"REFUSED: {e}")
        return driver.EXIT_REFUSED
    try:
        import pipeline
        import finish
        conv = pipeline.Conversion(a, log)
        big, small = state_bytes(conv)
        log(f"state: BF16 stream {big / GB:.1f} GB, quantized + FP8-only {small / GB:.1f} GB; RSS "
            f"{pipeline.rss_gb():.1f} GB")
        ck = driver.Checkpoint(ck_dirs, placement=lambda name: 0 if name.startswith("bf16.") else 1)             if ck_dirs else None
        peek = ck.next_layer(conv.fingerprint()) if ck else None
        conv.start_table(need_gather=not (peek and peek >= 2))
        loop = driver.LayerLoop(a.layers, conv.layer_dir, stop_file=a.stop_file, checkpoint=ck,
                                every=a.ckpt_every, log=log)
        code, state = loop.run(conv)
        if code != driver.EXIT_DONE:
            return code
        if conv.sweep is not None:
            conv.sweep.join()
            if conv.sweep.error:
                raise conv.sweep.error
        return finish.finish(conv, state, t_start)
    except Exception:
        log("ERROR\n" + traceback.format_exc())
        return driver.EXIT_ERROR


def cmd_fixture(a):
    import fixture
    fixture.make(a.out)
    print(f"fixture work tree written to {a.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="convert (resumes by itself)")
    r.add_argument("--out", default="F:/ai/models/Qwen3.8-Flash-Next-ignis")
    r.add_argument("--ood-dir", required=True, help="the study's real/ood (corpus files, pinned by sha256)")
    r.add_argument("--windows-dir", required=True, help="the 27B KLD study's windows directory")
    r.add_argument("--table-cache", required=True, help="the study's BF16 n-gram table cache (read only)")
    r.add_argument("--repo", default=REPO, help="this repository (canary fixture, commit id)")
    r.add_argument("--lock-owner", required=True, help="the name the GPU lock is held under")
    r.add_argument("--layers", type=int, default=48, help="convert layers 0..N-1 (dry runs: 2)")
    r.add_argument("--table-shards", type=int, default=128, help="n-gram shards to convert (dry runs only)")
    r.add_argument("--budget", type=float, default=2.5, help="mean bits per weight incl. scales (fallback 3.0)")
    r.add_argument("--stop-file", default=None, help="checked before each layer: checkpoint and exit 75")
    r.add_argument("--ckpt-dir", default="E:/flash-next-ckpt", help="the BF16 stream's checkpoint (~13.6 GB)")
    r.add_argument("--ckpt-small-dir", default=None, help="the other streams' checkpoint (~6.9 GB)")
    r.add_argument("--ckpt-every", type=int, default=6, help="checkpoint every N layers (0: only on stop)")
    r.add_argument("--vram-gb", type=float, default=24.0)
    r.add_argument("--prefetch", type=int, default=1, choices=(0, 1), help="fetch layer L+1 during L (+5 GB RAM)")
    r.add_argument("--disk-margin-gb", type=float, default=15.0)
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("fixture", help="a reduced work tree (2 layers, 8 experts, 1000 table rows), CPU only")
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fixture)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
