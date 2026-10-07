"""Spec flash-next/07 phase B: the MTP head's companion container (layout.md §13).

    python convert_head.py chunks --ood-dir <real/ood> --windows-dir <kld windows> --out <calib>
    flash_next_mtp_calibration <calib>              (the engine example: the tapped stacks)
    python convert_head.py run --calib <calib> --work <work-mtp> --main <main .ninfer> --lock-owner <name>
    ignis-artifact-pack --family mtp --work <work-mtp> --pair-main <main .ninfer>
    python convert_head.py verify --artifact <companion> --main <main .ninfer> --lock-owner <name>
    python convert_head.py alpha --artifact <companion> --corpus <phase A corpus> --bf16 <phase A alpha.json>
                                 --out <json> --lock-owner <name> [--trunk bf16|main --main <main .ninfer>]

`chunks` writes the trunk's calibration corpus (the converter's own, in its order) as one token
file per chunk, its `valid` tokens only, and `chunks.json`. The engine prefills every chunk with
the residual tap armed and writes its final pre-mixer stacks beside the tokens.

`run` builds the head's entries over those states by spec 07's convention (comb a, norm a), runs
the BF16 head layer causally over each chunk (prototype `mtp.Head`, dense: a chunk is <= 2048
tokens) and keeps the MoE sublayer's inputs and routing. Then it runs the trunk converter's own
expert step on them, unchanged (`pipeline.Conversion._convert_experts`: g^2-weighted Hessians,
shrink, the zero-token fallback, the four-K sweep, the allocation at --budget, the records), writes
the non-experts in the trunk's encodings, decodes the written records (the sha256 of every decoded
projection and the MoE error on the test chunks) and writes converter.json (layout.md §13.5). Only
the head is fetched: the token embedding is the main container's. The packer then assembles the
companion and pins it to the main container.

`verify` decodes every projection from the packed companion's bytes against the sha256 the run
recorded, and checks the main container's whole-file sha256. `alpha` scores the quantized head on
phase A's texts exactly as phase A scored the BF16 one (comb a, norm a, chain a; dense, and the
head's own indexer on the long texts), pairs its draft-1 hits with phase A's and writes AC2's
verdict into the companion's sidecar.
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
    manifest = {"chunks": rows}
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


def check_inputs(calib, chunks, main):
    """Everything the GPU work reads, before it starts: the main container, the tap's record of
    the container its states came from, and every chunk's stacks at its size. Returns tap.json."""
    if not os.path.isfile(main):
        raise RuntimeError(f"the main container {main} does not exist")
    tap_path = os.path.join(calib, "tap.json")
    if not os.path.isfile(tap_path):
        raise RuntimeError(f"{tap_path} is missing: the engine example has not finished the tap")
    tap = json.load(open(tap_path))
    tapped = os.path.join(tap["model_dir"], tap["artifact"])
    if not os.path.isfile(tapped) or not os.path.samefile(tapped, main):
        raise RuntimeError(f"the states were tapped from {tapped}, the head is paired with {main}")
    for c in chunks:
        path = os.path.join(calib, c["name"] + ".stacks.bf16")
        want = c["tokens"] * WIDTH * 2
        if not os.path.isfile(path) or os.path.getsize(path) != want:
            raise RuntimeError(f"{path} is missing or not {want} bytes")
    return tap


def container_tensor(c, name, device="cpu"):
    """A non-expert tensor of a packed container as the bf16 weight it stands for: FP8 decoded
    (code * scale, rounded to bf16, layout.md §6.1), BF16 copied."""
    import torch
    import fp8
    o = c.objects[name]
    raw = _payload(c, o)
    if o["format"] == "FP8_E4M3FN_ROW_BF16S":
        return fp8.decode(raw, o["shape"], device)
    if o["format"] == "BF16":
        return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(o["shape"]).clone().to(device)
    raise ValueError(f"{name}: {o['format']} is not a non-expert format")


def decode_hashes(gq, dq, k2):
    """sha256 of every decoded projection's bf16 (out, in) bytes, as layout.md §9's samples are."""
    import torch
    out = []
    for e in range(gq.shape[0]):
        for proj, w in (("gu", gq[e]), ("dn", dq[e])):
            out.append({"class": f"{proj}-{k2[proj][e] / 2:g}", "proj": proj, "k2": k2[proj][e], "expert": e,
                        "sha256": hashlib.sha256(w.cpu().view(torch.int16).numpy().tobytes()).hexdigest()})
    return out


def _experts_step(shim, layer, X, ridx, rw, d, tm):
    """The trunk converter's expert step on the head's MoE inputs, then the decode of what it wrote:
    its record, the MoE error and every decoded projection's sha256. Without autograd, as the
    trunk's pass runs it (the layer's parameters require grad)."""
    import torch
    with torch.no_grad():
        rec = {"layer": MTP_LAYER}
        moe = shim._convert_experts(MTP_LAYER, layer, X, ridx, rw, d, rec, tm)
        torch.cuda.empty_cache()
        ta = time.time()
        gq, dq, _, _ = shim._decode_experts(d, MTP_LAYER, False)
        hashes = decode_hashes(gq, dq, rec["k2"])
        tm["decode"] = time.time() - ta
        err = {"layer": 0}     # _moe_error looks up the trunk's run 6 / run 8 baselines, none apply here
        shim._moe_error(layer.mlp.experts, gq, dq, moe, err)
    return rec, err, hashes


def cmd_run(a):
    import torch
    import container
    import fetch
    import finish
    import nonexpert
    import preflight
    import mtp
    chunks = json.load(open(os.path.join(a.calib, "chunks.json")))["chunks"]
    tap = check_inputs(a.calib, chunks, a.main)
    preflight.check_gpu(a.lock_owner)
    preflight.cap_vram(a.vram_gb)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    t0 = time.time()
    tm = defaultdict(float)
    d = os.path.join(a.work, "mtp")
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(a.work, "state"), exist_ok=True)

    # The main container's whole-file sha256, read beside the GPU work.
    main_sha = {}

    def digest():
        try:
            main_sha["sha"] = layout.file_digest(a.main)
        except BaseException as e:
            main_sha["error"] = e
    hasher = threading.Thread(target=digest, daemon=True)
    hasher.start()

    # Only the head is fetched; the token embedding is the main container's, as the engine
    # serves it (FP8 decoded).
    ta = time.time()
    src = fetch.Source(a.hub)
    text = src._json_cached("config.json")["text_config"]
    weights = src.load(PREFIX)
    embed = container_tensor(container.Container(a.main), "embed_tokens.weight")
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
    head.embed = None          # the combine is done; the experts step needs the layer only
    shim = _shim(dev, a.budget, a.work, cal, test, kind, kinds)
    rec, err, hashes = _experts_step(shim, head.layer, X, ridx, rw, d, tm)
    del X, ridx, rw

    hasher.join()
    if "error" in main_sha:
        raise RuntimeError(f"hashing {a.main} failed: {main_sha['error']}")
    experts_bin = {"bytes": os.path.getsize(os.path.join(d, "experts.bin")),
                   "sha256": layout.file_digest(os.path.join(d, "experts.bin"))}
    layout.mark_done(d, {"experts.bin": experts_bin["sha256"]})
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
        "calibration": {"source": "engine residual tap", "artifact": tap["artifact"], "kv_format": tap["kv_format"],
                        "embed_tokens": "the main container's (FP8 decoded)",
                        "chunks": sum(c["cal"] for c in chunks), "entries": n_cal,
                        "test_chunks": sum(c["test"] for c in chunks), "test_entries": n_test, "steps": [1]},
        "k_map": {p: rec["k2"][p] for p in ("gu", "dn")},
        "k_hist": {p: dict(zip(("2", "2.5", "3", "4"), rec["k_hist"][p])) for p in ("gu", "dn")},
        "rates": {**rec["rates"], **{f"{p}_stored": rec["stored_rates"][p] for p in ("gu", "dn")}},
        "k_classes": finish.k_classes([rec]),
        "expert_traffic": rec["expert_traffic"],
        "curve_db": rec["curve_db"],
        "unrouted": rec["unrouted"],
        "moe_error_db": {"db": err["moe_db"], "per_kind_db": err["moe_db_kind"]},
        "experts_bin": experts_bin,
        "decode_sha256": hashes,
        "self_check": {"work_files": "decode_sha256: every projection decoded from experts.bin as written",
                       "container": "convert_head.py verify, after packing"},
        "time_s": {"total": time.time() - t0, **dict(tm)},
    }
    layout.write_json_atomic(os.path.join(a.work, "converter.json"), record)
    print(f"MTP experts: rates gu/dn {rec['rates']['gu']:.4f}/{rec['rates']['dn']:.4f} (budget {a.budget}), "
          f"K hist gu {rec['k_hist']['gu']} dn {rec['k_hist']['dn']}, MoE {err['moe_db']:.2f} dB "
          f"{ {k: round(v, 2) for k, v in err['moe_db_kind'].items()} }, "
          f"fallback {len(record['quantizer']['hessian_fallback'])}, {n_cal} calibration entries, "
          f"{time.time() - t0:.0f} s")
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


def quantized_head_weights(path, dev, decode=None):
    """The head's weights as the companion stores them, without the `mtp.` prefix: experts
    decoded by `decode(record, k2, proj, dev)` (the trellis oracle, layout.md §3) into the
    checkpoint's fused [experts, out, in] tensors, the non-experts by `container_tensor`."""
    import torch
    import container
    import trellis
    decode = decode or trellis.decode
    c = container.Container(path)
    out = {name[len(PREFIX):]: container_tensor(c, name) for name in c.objects if ".mlp.experts." not in name}
    n = sum(".mlp.experts." in name for name in c.objects) // 2
    fused = {}
    for proj in ("gu", "dn"):
        i, o = layout.SHAPES[proj]
        fused[proj] = torch.empty((n, o, i), dtype=torch.bfloat16, device=dev)
    for e in range(n):
        for proj in ("gu", "dn"):
            k2, rec = expert_record(c, e, proj)
            fused[proj][e] = decode(rec, k2, proj, dev).to(torch.bfloat16)
    out[EXPERTS[0]], out[EXPERTS[1]] = fused["gu"], fused["dn"]
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
        if k2 != s["k2"] or hashlib.sha256(w.cpu().view(torch.int16).numpy().tobytes()).hexdigest() != s["sha256"]:
            bad.append(f"{s['class']} expert {s['expert']}")
    main_ok = layout.file_digest(a.main) == side["pair"]["main"]["file_sha256"]
    result = {"artifact": a.artifact, "projections_checked": len(side["decode_sha256"]),
              "bit_identical": not bad, "mismatches": bad, "main_file_sha256_matches": main_ok}
    layout.write_json_atomic(a.artifact + ".verify.json", result)
    print(json.dumps(result))
    return 0 if not bad and main_ok else 3


# ---------------------------------------------------------------------------------- acceptance
def cmd_alpha(a):
    """The quantized head on phase A's texts. --trunk bf16 keeps phase A's BF16 embed_tokens and
    lm_head (fetched), so the difference to phase A is the head's quantization alone (AC2);
    --trunk main takes them from the main container (FP8), as the engine serves them."""
    import torch
    import container
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
    if a.trunk == "bf16":
        embed, lm_head = src.get("model.language_model.embed_tokens.weight"), src.get("lm_head.weight")
    else:
        main = container.Container(a.main)
        embed, lm_head = container_tensor(main, "embed_tokens.weight"), container_tensor(main, "lm_head.weight")
    head = mtp.Head(mtp.layer_config(text), weights, embed, lm_head, device=dev)
    del weights, embed, lm_head
    print(f"quantized head loaded in {time.time() - t0:.0f} s", flush=True)
    bf16 = json.load(open(a.bf16))["texts"]
    results = {"trunk": a.trunk, "texts": {}}
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
    acceptance = verdict(results["summary"], a.trunk)
    sidecar = a.artifact + ".conversion.json"
    side = json.load(open(sidecar))
    side["acceptance" if a.trunk == "bf16" else "acceptance_served"] = acceptance
    layout.write_json_atomic(sidecar, side)
    print(json.dumps(acceptance, indent=1))
    return 0 if acceptance["pass"] else 3


def verdict(summary, trunk, limit=0.03):
    """AC2: the quantized head's alpha_1 within `limit` of the BF16 head's on phase A's windows
    (dense, every text)."""
    s = summary["aa-dense"]["all"]
    delta = s["alpha1_bf16_minus_quantized"]
    return {"trunk": trunk, "alpha_bf16": s["alpha_bf16"], "alpha_quantized": s["alpha_quantized"],
            "delta_alpha1": delta, "limit": limit, "pass": delta[0] <= limit,
            "windows": "spec 07 phase A's 16 texts, comb a, norm a, chain a, dense",
            "by_class": {k: {"alpha_quantized": v["alpha_quantized"], "alpha_bf16": v["alpha_bf16"]}
                         for k, v in summary["aa-dense"].items()},
            "index_long": summary.get("aa-index0", {}).get("all")}


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
    v.add_argument("--main", required=True, help="the main container: its whole-file sha256 is checked")
    v.add_argument("--lock-owner", required=True)
    v.set_defaults(fn=cmd_verify)
    q = sub.add_parser("alpha", help="the quantized head's acceptance on phase A's texts")
    q.add_argument("--artifact", required=True)
    q.add_argument("--corpus", required=True, help="phase A's corpus (the example's output)")
    q.add_argument("--bf16", required=True, help="phase A's alpha.json (the BF16 head on the same texts)")
    q.add_argument("--out", required=True)
    q.add_argument("--lock-owner", required=True)
    q.add_argument("--hub", default=hub)
    q.add_argument("--trunk", choices=("bf16", "main"), default="bf16",
                   help="embed_tokens and lm_head: phase A's BF16 (AC2) or the main container's (served)")
    q.add_argument("--main", default="F:/ai/models/Qwen3.8-Flash-Next-ignis/qwen3_8_flash_next_trellis_a25-v2.ninfer")
    q.set_defaults(fn=cmd_alpha)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
