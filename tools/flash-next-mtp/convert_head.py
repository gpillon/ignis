"""Spec flash-next/07 phase B: the MTP head's companion container (layout.md §13).

    python convert_head.py chunks --ood-dir <real/ood> --windows-dir <kld windows> --out <calib>
    flash_next_mtp_calibration <calib>              (the engine example: the tapped stacks)
    python convert_head.py run --calib <calib> --work <work-mtp> --main <main .ninfer> --lock-owner <name>
    ignis-artifact-pack --family mtp --work <work-mtp> --pair-main <main .ninfer>
    python convert_head.py verify --artifact <companion> --main <main .ninfer> --lock-owner <name>
    python convert_head.py alpha --artifact <companion> --corpus <phase A corpus> --bf16 <phase A alpha.json>
                                 --out <json> --lock-owner <name>

`chunks` writes the trunk's calibration corpus (the converter's own, in its order) as one token
file per chunk, its `valid` tokens only, and `chunks.json`. The engine prefills every chunk with
the residual tap armed and writes its final pre-mixer stacks beside the tokens.

`run` builds the head's entries over those states by spec 07's convention (comb a, norm a), runs
the BF16 head layer causally over each chunk (prototype `mtp.Head`, dense: a chunk is <= 2048
tokens) and keeps the MoE sublayer's inputs and routing. Then it runs the trunk converter's own
expert step on them, unchanged (`pipeline.Conversion._convert_experts`: g^2-weighted Hessians,
shrink, the zero-token fallback, the four-K sweep, the allocation at --budget, the records), writes
the non-experts in the trunk's encodings, decodes the written records (the self-check samples and
the MoE error on the test chunks) and writes converter.json (layout.md §13.5). The packer then
assembles the companion and pins it to the main container.

`verify` decodes the sampled projections from the packed companion's bytes against the sha256 the
run recorded, and checks the main container's whole-file sha256. `alpha` scores the quantized head
on phase A's texts exactly as phase A scored the BF16 one (comb a, norm a, chain a; dense, and the
head's own indexer on the long texts) and pairs its draft-1 hits with phase A's.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import types
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "..", "flash-next-converter")
sys.path.insert(0, CONVERTER)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

import layout  # noqa: E402  (the converter's modules, flat files)

SCHEMA = "flash-next-mtp-converter-v1"
PREFIX = "mtp."
LAYER_PREFIX = "mtp.layers.0."
# The trunk's seed rule (L * 10000 + proj * 1000 + expert) at layer index 48: no MTP seed equals a
# trunk layer's (layout.md §13.4).
MTP_LAYER = 48
WIDTH = 4 * 2560
BF16_ALPHA = [0.828, 0.800, 0.811, 0.821]
HEAD = {"comb": "a", "norm": "a", "chain": "a", "idx": "own"}
EXPERTS = ("layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.down_proj")


# ---------------------------------------------------------------------------------- pure helpers
def split_name(name):
    """(layer-local name, container prefix) of a checkpoint `mtp.*` tensor: the trunk's encoding
    rule (layout.encoding_of) reads the layer-local name, the container keeps the whole one."""
    if not name.startswith(PREFIX):
        raise ValueError(f"{name} is not an MTP tensor")
    if name.startswith(LAYER_PREFIX):
        return name[len(LAYER_PREFIX):], LAYER_PREFIX
    return name[len(PREFIX):], PREFIX


def export_chunks(chunks, out):
    """Writes each chunk's `valid` tokens as `<name>.tokens.u32` and `chunks.json`; returns the
    manifest. Chunk i is `c<i>`, in the corpus order (calibration chunks first)."""
    os.makedirs(out, exist_ok=True)
    rows = []
    for i, c in enumerate(chunks):
        name = f"c{i:03d}"
        ids = np.asarray(c["ids"][:c["valid"]], dtype="<u4")
        ids.tofile(os.path.join(out, f"{name}.tokens.u32"))
        rows.append({"name": name, "tokens": int(len(ids)), "kind": c["kind"], "cal": bool(c["cal"]),
                     "test": bool(c["test"]), "source": c["source"]})
    manifest = {"chunks": rows, "width": WIDTH}
    layout.write_json_atomic(os.path.join(out, "chunks.json"), manifest)
    return manifest


def row_masks(chunks):
    """Per entry row (a chunk of n tokens has n - 1 entries, in chunk order): calibration and
    test masks, each test row's kind index, and the sorted kind names."""
    kinds = sorted({c["kind"] for c in chunks if c["test"]})
    cal, test, kind = [], [], []
    for c in chunks:
        n = c["tokens"] - 1
        cal += [c["cal"]] * n
        test += [c["test"]] * n
        if c["test"]:
            kind += [kinds.index(c["kind"])] * n
    return np.array(cal, bool), np.array(test, bool), np.array(kind, np.int64), kinds


def paired(ok_bf16, ok_q):
    """Mean of (BF16 hit - quantized hit) per position and its 95% half-width."""
    d = ok_bf16.astype(float) - ok_q.astype(float)
    return [float(d.mean()), float(1.96 * d.std(ddof=1) / math.sqrt(len(d)))]


def alphas(counts):
    """alpha_j from summed [n_j, ok_j] pairs."""
    return [k / n if n else 0.0 for n, k in counts]


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------------- chunks
def cmd_chunks(a):
    import corpus
    c = corpus.load(a.ood_dir, a.windows_dir)
    m = export_chunks(c.chunks, a.out)
    n = sum(r["tokens"] for r in m["chunks"])
    print(f"{len(m['chunks'])} chunks ({sum(r['cal'] for r in m['chunks'])} calibration), {n} tokens -> {a.out}")
    return 0


# ---------------------------------------------------------------------------------- run
class _Shim(types.SimpleNamespace):
    """What the trunk converter's expert step reads of its Conversion, for the one MTP layer."""


def _shim(dev, budget, work, cal, test, kind, kinds):
    import torch
    import trellis
    from pipeline import Conversion
    s = _Shim(dev=dev, args=types.SimpleNamespace(budget=budget), exl3_debug=trellis.debug_dir(work),
              cal_mask=torch.from_numpy(cal).to(dev), tmask=torch.from_numpy(test).to(dev),
              test_kind=torch.from_numpy(kind).to(dev), kind_names=kinds)
    for f in ("_convert_experts", "_fallback_hessians", "_decode_experts", "_moe_error"):
        setattr(s, f, types.MethodType(getattr(Conversion, f), s))
    return s


def _moe_inputs(head, calib, chunks, dev, batch=1024):
    """The MoE sublayer's input, router ids and weights over every chunk's entries, in order."""
    import torch
    import phase_a
    cap = {}
    hooks = [head.layer.mlp.register_forward_pre_hook(lambda m, args: cap.__setitem__("x", args[0])),
             head.layer.mlp.gate.register_forward_hook(lambda m, args, o: cap.__setitem__("r", o))]
    xs, ri, rw = [], [], []
    width = head.S * head.H
    try:
        with torch.no_grad():
            for c in chunks:
                base = os.path.join(calib, c["name"])
                tokens = np.fromfile(base + ".tokens.u32", np.uint32).astype(np.int64)
                stacks = np.fromfile(base + ".stacks.bf16", np.uint16)
                if stacks.size != len(tokens) * width:
                    raise RuntimeError(f"{base}.stacks.bf16 holds {stacks.size} values, want {len(tokens)} x {width}")
                S = phase_a.to_bf16(stacks.reshape(len(tokens), width), dev)
                E = phase_a.entries(head, S, torch.from_numpy(tokens).to(dev), "a", "a")
                head.window(E, None, "dense")
                n = len(tokens) - 1
                for b0 in range(0, n, batch):
                    head.first_step(torch.arange(b0, min(b0 + batch, n), device=dev))
                    xs.append(cap["x"].reshape(-1, cap["x"].shape[-1]))
                    ri.append(cap["r"][2])
                    rw.append(cap["r"][1].float())
                    cap.clear()
                del S, E
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(xs), torch.cat(ri), torch.cat(rw)


def cmd_run(a):
    import torch
    import fetch
    import finish
    import nonexpert
    import preflight
    import mtp
    preflight.check_gpu(a.lock_owner)
    preflight.cap_vram(a.vram_gb)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    t0 = time.time()
    tm = defaultdict(float)
    d = os.path.join(a.work, "mtp")
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(a.work, "state"), exist_ok=True)
    chunks = json.load(open(os.path.join(a.calib, "chunks.json")))["chunks"]

    # The main container's whole-file sha256, read beside the GPU work.
    main_sha = {}
    hasher = threading.Thread(target=lambda: main_sha.update(sha=layout.file_digest(a.main)), daemon=True)
    hasher.start()

    ta = time.time()
    src = fetch.Source(a.hub)
    text = src._json_cached("config.json")["text_config"]
    weights = src.load(PREFIX)
    embed = src.get("model.language_model.embed_tokens.weight")
    tm["fetch"] = time.time() - ta
    if len(weights) != 31:
        raise RuntimeError(f"the checkpoint has {len(weights)} mtp.* tensors, the contract 31")

    ta = time.time()
    entries = []
    for name, t in sorted(weights.items()):
        if name not in EXPERTS:
            local, prefix = split_name(PREFIX + name)
            entries.append(nonexpert.write_tensor(d, local, t, prefix=prefix))
    layout.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": entries})
    tm["write"] += time.time() - ta

    # The head reads no logits here: the output head is a one-row stand-in.
    head = mtp.Head(mtp.layer_config(text), weights, embed, embed[:1], device=dev)
    del weights, embed

    ta = time.time()
    cal, test, kind, kinds = row_masks(chunks)
    X, ridx, rw = _moe_inputs(head, a.calib, chunks, dev)
    if X.shape[0] != len(cal):
        raise RuntimeError(f"{X.shape[0]} entries, the chunks give {len(cal)}")
    tm["moe_inputs"] = time.time() - ta
    shim = _shim(dev, a.budget, a.work, cal, test, kind, kinds)
    rec = {"layer": MTP_LAYER}
    moe = shim._convert_experts(MTP_LAYER, head.layer, X, ridx, rw, d, rec, tm)
    del X, ridx, rw
    torch.cuda.empty_cache()

    ta = time.time()
    gq, dq, sample, _ = shim._decode_experts(d, MTP_LAYER, True)
    tm["decode"] = time.time() - ta
    err = {"layer": 0}         # _moe_error looks up the trunk's run 6 / run 8 baselines, none apply here
    shim._moe_error(head.layer.mlp.experts, gq, dq, moe, err)
    del gq, dq, moe
    experts_bin = {"bytes": os.path.getsize(os.path.join(d, "experts.bin")),
                   "sha256": layout.file_digest(os.path.join(d, "experts.bin"))}
    layout.mark_done(d, {"experts.bin": experts_bin["sha256"]})

    hasher.join()
    import importlib.metadata as md
    budget_ok = all(rec["rates"][p] <= a.budget + 1e-9 for p in ("gu", "dn"))
    n_cal, n_test = int(cal.sum()), int(test.sum())
    record = {
        "schema": SCHEMA,
        "status": "complete",
        "verdict": "PASS" if budget_ok else "FAIL",
        "source": {"repo": fetch.REPO, "revision": fetch.REVISION, "tensors": 31, "prefix": PREFIX},
        "pair": {"main": {"file": os.path.basename(a.main), "file_sha256": main_sha["sha"]}},
        "head": HEAD,
        "converter": {"commit": _git("rev-parse", "HEAD"),
                      "dirty": bool(_git("status", "--porcelain", "--", "tools/flash-next-mtp",
                                         "tools/flash-next-converter")),
                      "command": " ".join(sys.argv),
                      "seed_rule": f"exllamav3 seed {MTP_LAYER}*10000 + proj*1000 + expert (proj 0 gate/up, 1 down)"},
        "versions": {"exllamav3": md.version("exllamav3"), "transformers": md.version("transformers"),
                     "torch": torch.__version__, "python": platform.python_version()},
        "quantizer": {"codebook": "mul1", "apply_out_scales": True, "K_set": [k / 2 for k in layout.K2_SET],
                      "budget_bits": a.budget, "batch": 32,
                      "hessian": "the trunk's: g^2-weighted per-expert input metric on calibration entries; gate/up "
                                 "shrunk 5% toward the layer H, down 5% toward I; an expert with no calibration "
                                 "entry takes the layer H (gate/up) and its all-entry activation moment (down)",
                      "hessian_fallback": rec.get("hessian_fallback", [])},
        "calibration": {"source": "engine residual tap", "artifact": os.path.basename(a.main),
                        "kv_format": "hq-e8-2b", "chunks": sum(c["cal"] for c in chunks), "entries": n_cal,
                        "test_chunks": sum(c["test"] for c in chunks), "test_entries": n_test, "steps": [1],
                        "chunks_manifest_sha256": hashlib.sha256(
                            open(os.path.join(a.calib, "chunks.json"), "rb").read()).hexdigest()},
        "k_map": {p: rec["k2"][p] for p in ("gu", "dn")},
        "k_hist": {p: dict(zip(("2", "2.5", "3", "4"), rec["k_hist"][p])) for p in ("gu", "dn")},
        "rates": {**rec["rates"], **{f"{p}_stored": rec["stored_rates"][p] for p in ("gu", "dn")}},
        "k_classes": finish.k_classes([rec]),
        "expert_traffic": rec["expert_traffic"],
        "curve_db": rec["curve_db"],
        "unrouted": rec["unrouted"],
        "moe_error_db": {"db": err["moe_db"], "rel": err["moe_rel_err"], "per_kind_db": err["moe_db_kind"]},
        "experts_bin": experts_bin,
        "decode_sha256": [{k: v for k, v in s.items() if k != "layer"} for s in sample],
        "self_check": {"work_files": "decode_sha256 decoded from experts.bin as written",
                       "container": "convert_head.py verify, after packing"},
        "time_s": {"total": time.time() - t0, **dict(tm)},
    }
    layout.write_json_atomic(os.path.join(a.work, "converter.json"), record)
    print(f"MTP experts: rates gu/dn {rec['rates']['gu']:.4f}/{rec['rates']['dn']:.4f} (budget {a.budget}), "
          f"K hist gu {rec['k_hist']['gu']} dn {rec['k_hist']['dn']}, MoE {err['moe_db']:.2f} dB "
          f"{ {k: round(v, 2) for k, v in err['moe_db_kind'].items()} }, fallback {len(record['quantizer']['hessian_fallback'])}, "
          f"{n_cal} calibration entries, {time.time() - t0:.0f} s")
    return 0 if budget_ok else 3


# ---------------------------------------------------------------------------------- the container
def _payload(c, o):
    with open(c.path, "rb") as f:
        f.seek(c.payload_start + o["offset"])
        return bytearray(f.read(o["bytes"]))


def expert_record(c, expert, proj):
    """(k2, record) of one MTP expert projection, read from the companion's bytes."""
    import container
    o = c.objects[f"{LAYER_PREFIX}mlp.experts.{expert}.{container.PROJ_NAME[proj]}"]
    k2 = container.K2_OF_FORMAT[o["format"]]
    if o["bytes"] != layout.record_bytes(proj, k2):
        raise ValueError(f"{o['name']}: {o['bytes']} bytes, a {proj} K={k2 / 2:g} record is "
                         f"{layout.record_bytes(proj, k2)}")
    entry = layout.IndexEntry(expert, proj, k2, o["bytes"], c.payload_start + o["offset"])
    return k2, layout.read_record(c.path, entry)


def quantized_head_weights(path, dev):
    """The head's weights as the companion stores them, without the `mtp.` prefix: experts
    decoded by the trellis oracle, FP8 decoded (code * scale, rounded to bf16), BF16 copied."""
    import torch
    import container
    import fp8
    import trellis
    c = container.Container(path)
    out = {}
    for name, o in c.objects.items():
        if ".mlp.experts." in name:
            continue
        raw = _payload(c, o)
        if o["format"].startswith("FP8"):
            t = fp8.decode(raw, o["shape"])
        else:
            t = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(o["shape"]).clone()
        out[name[len(PREFIX):]] = t
    n = sum(1 for k in c.objects if ".mlp.experts." in k) // 2
    gu = torch.empty((n, 1280, 2560), dtype=torch.bfloat16, device=dev)
    dn = torch.empty((n, 2560, 640), dtype=torch.bfloat16, device=dev)
    for e in range(n):
        for proj, dst in (("gu", gu), ("dn", dn)):
            k2, rec = expert_record(c, e, proj)
            dst[e] = trellis.decode(rec, k2, proj, dev).to(torch.bfloat16)
    out[EXPERTS[0]], out[EXPERTS[1]] = gu, dn
    return out


def cmd_verify(a):
    import torch
    import container
    import preflight
    import trellis
    preflight.check_gpu(a.lock_owner)
    side = json.load(open(a.artifact + ".conversion.json"))
    c = container.Container(a.artifact)
    bad = []
    for s in side["decode_sha256"]:
        k2, rec = expert_record(c, s["expert"], s["proj"])
        w = trellis.decode(rec, k2, s["proj"], "cuda").to(torch.bfloat16)
        if hashlib.sha256(w.cpu().view(torch.int16).numpy().tobytes()).hexdigest() != s["sha256"]:
            bad.append(f"{s['class']} expert {s['expert']}")
    main_ok = None
    if a.main:
        main_ok = layout.file_digest(a.main) == side["pair"]["main"]["file_sha256"]
    result = {"artifact": a.artifact, "projections_checked": len(side["decode_sha256"]),
              "bit_identical": not bad, "mismatches": bad, "main_file_sha256_matches": main_ok}
    layout.write_json_atomic(a.artifact + ".verify.json", result)
    print(json.dumps(result))
    return 0 if not bad and main_ok is not False else 3


# ---------------------------------------------------------------------------------- acceptance
def cmd_alpha(a):
    import torch
    import fetch
    import mtp
    import phase_a
    import preflight
    preflight.check_gpu(a.lock_owner)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    t0 = time.time()
    weights = quantized_head_weights(a.artifact, dev)
    src = fetch.Source(a.hub)
    text = src._json_cached("config.json")["text_config"]
    embed = src.get("model.language_model.embed_tokens.weight")
    lm_head = src.get("lm_head.weight")
    head = mtp.Head(mtp.layer_config(text), weights, embed, lm_head, device=dev)
    del weights, embed, lm_head
    print(f"quantized head loaded in {time.time() - t0:.0f} s", flush=True)
    bf16 = json.load(open(a.bf16))["texts"]
    results = {"texts": {}}
    for t, tokens, stacks, picks in phase_a.texts(a.corpus):
        name, prompt = t["name"], t["prompt_tokens"]
        long = len(tokens) > 2051
        S = phase_a.to_bf16(stacks, dev)
        runs = [("dense", 0)] + ([("index", 0)] if long else [])
        first = 0 if not long else prompt - 512
        results["texts"][name] = {"kind": t["kind"], "tokens": len(tokens), "runs": {}}
        for mode, off in runs:
            r = phase_a.score_text(head, tokens, S, picks, prompt, "a", "a", ("a",), mode, off, first,
                                   batch=256 if not long else 64)
            key = f"aa-{mode}{off if mode == 'index' else ''}"
            ok = r.pop("gen_ok1")
            r["gen_ok1"] = "".join("1" if x else "0" for x in ok)
            results["texts"][name]["runs"][key] = r
            print(f"{name} {key}: chain a {r['gen']['a']}", flush=True)
        del S
        torch.cuda.empty_cache()
    results["summary"] = summarize(results, bf16)
    layout.write_json_atomic(a.out, results)
    print(json.dumps(results["summary"], indent=1))
    return 0


def summarize(results, bf16):
    """alpha_1..4 of the quantized head (chain a) per text class next to the BF16 head's on the same
    positions (phase A's file), and the paired draft-1 difference."""
    classes = {"all": lambda t: True, "code": lambda t: t["kind"] == "code",
               "prose": lambda t: t["kind"] == "prose", "long": lambda t: t["tokens"] > 2051}
    out = {}
    for key in sorted({k for t in results["texts"].values() for k in t["runs"]}):
        for cname, pred in classes.items():
            names = [n for n, t in results["texts"].items() if pred(t) and key in t["runs"]]
            if not names:
                continue
            q = np.sum([results["texts"][n]["runs"][key]["gen"]["a"] for n in names], axis=0)
            b = np.sum([bf16[n]["runs"][key]["gen"]["a"] for n in names], axis=0)
            ok_q = np.concatenate([np.frombuffer(results["texts"][n]["runs"][key]["gen_ok1"].encode(), np.uint8) - 48
                                   for n in names])
            ok_b = np.concatenate([np.frombuffer(bf16[n]["runs"][key]["gen_ok1"].encode(), np.uint8) - 48
                                   for n in names])
            out.setdefault(key, {})[cname] = {"texts": len(names), "alpha_quantized": alphas(q),
                                              "alpha_bf16": alphas(b), "positions": int(len(ok_q)),
                                              "alpha1_bf16_minus_quantized": paired(ok_b, ok_q)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    hub = "F:/ai/models/Qwen3.8-Flash-Next-ignis/work/state/hub"
    c = sub.add_parser("chunks", help="the calibration corpus as token files")
    c.add_argument("--ood-dir", required=True)
    c.add_argument("--windows-dir", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_chunks)
    r = sub.add_parser("run", help="the conversion: work-mtp/ and converter.json")
    r.add_argument("--calib", required=True, help="chunks + tapped stacks")
    r.add_argument("--work", required=True, help="the head's work tree (work-mtp)")
    r.add_argument("--main", required=True, help="the main container the head belongs to")
    r.add_argument("--lock-owner", required=True)
    r.add_argument("--budget", type=float, default=3.0)
    r.add_argument("--hub", default=hub, help="the fetcher's header cache")
    r.add_argument("--vram-gb", type=float, default=24.0)
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("verify", help="after packing: decode the sampled projections from the companion")
    v.add_argument("--artifact", required=True)
    v.add_argument("--main", default=None, help="also check the main container's whole-file sha256")
    v.add_argument("--lock-owner", required=True)
    v.set_defaults(fn=cmd_verify)
    q = sub.add_parser("alpha", help="the quantized head's acceptance on phase A's texts")
    q.add_argument("--artifact", required=True)
    q.add_argument("--corpus", required=True, help="phase A's corpus (the example's output)")
    q.add_argument("--bf16", required=True, help="phase A's alpha.json (the BF16 head on the same texts)")
    q.add_argument("--out", required=True)
    q.add_argument("--lock-owner", required=True)
    q.add_argument("--hub", default=hub)
    q.set_defaults(fn=cmd_alpha)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
