"""Qwen3.8-Flash-Next -> ignis work files, references and traces (spec 01, GitHub #299).

    python convert.py run --lock-owner flash-next-convert --ood-dir ... --windows-dir ... --table-cache ...
    python convert.py verify --lock-owner ... --artifact X.ninfer   # after packing: decode from the container
    python convert.py fixture --out DIR          # a reduced work tree for the packer's tests

Exit codes: 0 done (PASS or dry run); 75 stopped by the stop file; 4 stopped because a
drive filled during the run (checkpointed first when the checkpoint drives have room);
2 refused (GPU lock, GPU busy, disk, a work tree of another configuration); 3 done but an acceptance check
FAILED (rates over --budget, MoE error, KLD/MMLU, or a work-file re-decode mismatch; the
report names the 3.0-bit fallback), or for verify a container decode mismatch; 1 error
(traceback in the log). See README.md for the full command and its figures.
"""
import argparse
import json
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


def preflight_run(a, log):
    """The checks before any GPU or disk work; returns the checkpoint directories (or None)."""
    import preflight
    import trellis
    preflight.check_gpu(a.lock_owner)
    out_need = max(expected_output_bytes(a.layers, a.table_shards) - dir_bytes(a.out), 0)
    small_dir = a.ckpt_small_dir or os.path.join(a.out, "work", "state", "ckpt")
    big_ck, small_ck = 13.6 * GB, 6.9 * GB      # the state sizes of the run 8 corpus (README)
    same_drive = os.path.splitdrive(os.path.abspath(small_dir))[0].lower() == \
        os.path.splitdrive(os.path.abspath(a.out))[0].lower()
    free = preflight.check_disk(a.out, out_need + (small_ck if same_drive else 0), a.disk_margin_gb * GB, "output")
    log(f"disk: {free / GB:.1f} GB free on {a.out}; output still to write {out_need / GB:.1f} GB "
        f"+ {'small checkpoint ' + format(small_ck / GB, '.1f') + ' GB + ' if same_drive else ''}"
        f"margin {a.disk_margin_gb} GB")
    if not same_drive:
        preflight.check_disk(small_dir, small_ck - dir_bytes(small_dir), 1 * GB, "small checkpoint")
    ck_dirs = [a.ckpt_dir, small_dir]
    try:
        preflight.check_disk(a.ckpt_dir, big_ck - dir_bytes(a.ckpt_dir), 1 * GB, "checkpoint")
    except preflight.Refused as e:
        log(f"WARNING {e}: no state checkpoint; a relaunch replays the finished layers from layer 0")
        ck_dirs = None
    trellis.check_version()
    frac, total = preflight.cap_vram(a.vram_gb)
    log(f"VRAM cap {a.vram_gb} GB ({frac:.3f} of {total / GB:.1f} GB)")
    return ck_dirs


def output_space(a, free=None):
    """None while the output drive holds the rest of the run plus the floor, else why not.
    Checked before each layer, so a drive filled by someone else stops the run cleanly."""
    import preflight
    rest = max(expected_output_bytes(a.layers, a.table_shards) - dir_bytes(a.out), 0)
    have = (free or preflight.free_bytes)(a.out)
    if have < rest + a.disk_floor_gb * GB:
        return (f"{a.out}: {have / GB:.1f} GB free, the rest of the run writes {rest / GB:.1f} GB "
                f"(+{a.disk_floor_gb:g} GB floor)")
    return None


def check_run_record(conv):
    """A work tree belongs to one configuration (revision, corpus, layers, table shards, budget):
    returns why this one differs, or None after recording it on first use."""
    import layout
    path = os.path.join(conv.work, "state", "run.json")
    want = json.loads(json.dumps(conv.run_record()))
    if os.path.exists(path):
        have = json.load(open(path))
        if have != want:
            keys = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
            return f"{conv.work} was made by another configuration (differs in {keys}): use a fresh --out"
        return None
    layout.write_json_atomic(path, want)
    return None


def cmd_run(a):
    import driver
    import preflight
    os.makedirs(a.out, exist_ok=True)
    log = Log(os.path.join(a.out, "convert.log"))
    log(f"convert {' '.join(sys.argv[1:])}")
    t_start = time.time()
    try:
        ck_dirs = preflight_run(a, log)
    except preflight.Refused as e:
        log(f"REFUSED: {e}")
        return driver.EXIT_REFUSED
    except Exception:
        log("ERROR in the preflight\n" + traceback.format_exc())
        return driver.EXIT_ERROR
    try:
        import finish
        import pipeline
        conv = pipeline.Conversion(a, log)
        why = check_run_record(conv)
        if why:
            log(f"REFUSED: {why}")
            return driver.EXIT_REFUSED
        big, small = state_bytes(conv)
        log(f"state: BF16 stream {big / GB:.1f} GB, quantized + FP8-only {small / GB:.1f} GB; RSS "
            f"{pipeline.rss_gb():.1f} GB")
        ck = None
        if ck_dirs:
            ck = driver.Checkpoint(ck_dirs, placement=lambda name: 0 if name.startswith("bf16.") else 1)
        peek = ck.next_layer(conv.fingerprint()) if ck else None
        conv.start_table(need_gather=not (peek and peek >= 2))
        loop = driver.LayerLoop(a.layers, conv.layer_dir, stop_file=a.stop_file, checkpoint=ck,
                                every=a.ckpt_every, log=log, space_check=lambda L: output_space(a))
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


def cmd_verify(a):
    """After packing: decode the sampled expert projections from the container's own bytes and
    compare them with the sha256 the pass recorded (spec 01 acceptance 7)."""
    import hashlib
    import container
    import driver
    import layout
    import preflight
    import torch
    import trellis
    sidecar = a.sidecar or a.artifact + ".conversion.json"
    log = Log(os.path.join(os.path.dirname(os.path.abspath(a.artifact)), "verify.log"))
    try:
        preflight.check_gpu(a.lock_owner)
        preflight.cap_vram(a.vram_gb)
        side = json.load(open(sidecar))
        c = container.Container(a.artifact)
        bad = []
        for s in side["decode_sha256"]:
            proj = s["class"].split("-")[0]
            k2, rec = c.expert_record(s["layer"], s["expert"], proj)
            w = trellis.decode(rec, k2, proj, "cuda").to(torch.bfloat16)
            if hashlib.sha256(w.cpu().view(torch.int16).numpy().tobytes()).hexdigest() != s["sha256"]:
                bad.append(f"L{s['layer']} {s['class']} expert {s['expert']}")
        result = {"artifact": a.artifact, "projections_checked": len(side["decode_sha256"]),
                  "bit_identical": not bad, "mismatches": bad}
        layout.write_json_atomic(os.path.join(os.path.dirname(os.path.abspath(a.artifact)), "verify.json"), result)
        log(f"verify: {result['projections_checked']} projections decoded from the container, "
            f"bit-identical to the pass: {not bad}" + (f"; mismatches {bad}" if bad else ""))
        return 0 if not bad else driver.EXIT_QUALITY_FAIL
    except preflight.Refused as e:
        log(f"REFUSED: {e}")
        return driver.EXIT_REFUSED
    except Exception:
        log("ERROR\n" + traceback.format_exc())
        return driver.EXIT_ERROR


def cmd_fixture(a):
    import fixture
    if a.reduced:
        fixture.make_reduced(a.out)
        print(f"reduced-geometry work tree written to {a.out}")
    else:
        fixture.make(a.out)
        print(f"fixture work tree written to {a.out}/work")
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
    r.add_argument("--ckpt-every", type=int, default=3,
                   help="checkpoint every N layers (always after the last layer and on a stop)")
    r.add_argument("--vram-gb", type=float, default=24.0)
    r.add_argument("--prefetch", type=int, default=1, choices=(0, 1), help="fetch layer L+1 during L (+5 GB RAM)")
    r.add_argument("--disk-margin-gb", type=float, default=15.0, help="free space the start requires beyond the output")
    r.add_argument("--disk-floor-gb", type=float, default=5.0,
                   help="free space each layer requires beyond the rest of the output (else exit 4)")
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("verify", help="after packing: decode the sampled projections from the container")
    v.add_argument("--artifact", required=True)
    v.add_argument("--sidecar", default=None, help="default: <artifact>.conversion.json")
    v.add_argument("--lock-owner", required=True)
    v.add_argument("--vram-gb", type=float, default=24.0)
    v.set_defaults(fn=cmd_verify)
    f = sub.add_parser("fixture", help="a reduced work tree (2 layers, 8 experts, 1000 table rows), CPU only")
    f.add_argument("--out", required=True)
    f.add_argument("--reduced", action="store_true",
                   help="the packer's reduced geometry from the HF modeling code; --out is the work root")
    f.set_defaults(fn=cmd_fixture)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
