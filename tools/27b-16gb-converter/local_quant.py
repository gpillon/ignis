"""A fine-tune's EXL3 quant, converted on this machine by exllamav3 at the bit widths of the quant the 16 GB
27B artifact repacks (GitHub #471, spec #449: Swift 1.5's 16 GB artifact).

`run` (the GPU step) converts the fine-tune's BF16 checkpoint with exllamav3's own conversion
(`exllamav3.conversion.convert_model`: its parser, its checks and its loop, in this process so its peaks
are read exactly):

- the recipe is `recipe.derive` of the published quant's `quantization_config.json`: every matrix at the
  K it has in `turboderp/Qwen3.8-27B-exl3` `SC_3.00bpw_H4`, the head at 4, bits 3.0, the `mul1` codebook,
  output scales always, the quant's MTP bits; the vision tower stays unquantized (`-vb 16`), as in the
  quant, whose configuration states no vision bits;
- the calibration data is that quant's own trace, `cal_trace.safetensors`, 250 rows x 2,048;
- both files are the ones downloaded at the revision the default 16 GB artifact's record names
  (`--record`): their download records give the revision, the configuration is the record's by sha256,
  the trace is its LFS oid;
- before the GPU: the fine-tune's checkpoint is pinned by its download records (its revision, each
  shard's LFS oid), and holds every matrix of the recipe at its shape; the output and the work
  directory are empty; the disk holds the output, the work files and exllamav3's checkpoints.

After the conversion every matrix's K and shape are checked against the recipe, the head at 4 among
them, and the configuration as the 16 GB converter checks a quant (`convert.check_quant_config`). Only
then is `<out>.pins.json` written beside the quant: `files`, every file of the quant by bytes and sha256
(the pins `convert.py run --quant-pins` takes, GitHub #465), and `provenance`: the fine-tune's revision,
the quant's repository, branch and revision, the calibration trace's sha256, exllamav3's version and
arguments (paths by role), the recipe (also `<out>.recipe.yaml`), the time and the VRAM and host peaks.

`control` (GPU) measures each matrix's relative reconstruction error against the fine-tune's BF16
(`reconstruction.rel_errors`) beside the default 16 GB artifact's error on the same matrix against the
base BF16 (its record's `trellis.matrices`), and writes both to a JSON file.

  python local_quant.py run --bf16 DIR --bf16-repo REPO --recipe-quant DIR --record R.json --out DIR --work DIR [--check]
  python local_quant.py control --quant DIR --bf16 DIR --bf16-repo REPO --record R.json --out C.json
"""
import argparse
import importlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import convert
import exl3
import plan
import reconstruction
import recipe
import work

SCHEMA = "qwen3.8-27b-local-quant-v1"
CONTROL_SCHEMA = "qwen3.8-27b-local-quant-control-v1"
QUANT_CONFIG = "quantization_config.json"
CAL_TRACE = "cal_trace.safetensors"
INDEX = convert.INDEX
ENTRY = "exllamav3.conversion.convert_model"
# The vision tower as the quant stores it: unquantized (1.5.3 would quantize it at its default 6 bits).
VISION_BITS = 16
# exllamav3 checkpoints its calibration state at most this often (its -cpi); this many seconds never comes.
NO_CHECKPOINTS = 10 ** 9
# A checkpoint rotation holds three copies of the state at once (ckpt, ckpt_old, ckpt_new).
CHECKPOINT_COPIES = 3
# The paths exllamav3 is given, by the role the provenance records them under.
ROLES = ("bf16", "work", "out", "recipe", "cal_data")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def beside(out, suffix):
    """A file beside the quant directory `out`: `<out><suffix>`."""
    return os.path.normpath(out) + suffix


def is_empty(path):
    return not os.path.isdir(path) or not os.listdir(path)


def exllamav3_argv(r, paths, checkpoints=True, resume=False):
    """exllamav3's arguments for the recipe `r`, the paths by role (`ROLES`)."""
    argv = ["-i", paths["bf16"], "-w", paths["work"], "-o", paths["out"], "--recipe", paths["recipe"],
            "-hb", str(r["head_bits"])]
    if r["mtp_bits"] is not None:
        argv += ["-mb", str(r["mtp_bits"])]
    argv += ["-vb", str(VISION_BITS), "-cb", r["codebook"], "--out_scales", r["out_scales"],
             "-cd", paths["cal_data"], "-cr", str(r["calibration"]["rows"]), "-cc", str(r["calibration"]["cols"])]
    if not checkpoints:
        argv += ["-cpi", str(NO_CHECKPOINTS)]
    if resume:
        argv.append("-r")
    return argv


def pin_sources(bf16_root, recipe_quant, record):
    """The sources, pinned before the GPU step: the published quant's two files at the record's revision
    (the configuration the record's by sha256, the trace its LFS oid), the fine-tune's checkpoint by its
    download records. Refused names what does not hold."""
    source = record["source"]
    quant_revision = convert.revision_of(recipe_quant, [QUANT_CONFIG, CAL_TRACE], "the recipe's quant")
    if quant_revision != source["revision"]:
        raise convert.Refused(f"{recipe_quant} was downloaded at {quant_revision}, the record's quant is "
                              f"{source['repo']} {source['branch']} at {source['revision']}")
    config_path = os.path.join(recipe_quant, QUANT_CONFIG)
    config_sha256 = work.file_digest(config_path)
    pinned = (source.get("files") or {}).get(QUANT_CONFIG, {}).get("sha256")
    if config_sha256 != pinned:
        raise convert.Refused(f"{QUANT_CONFIG} hashes to {config_sha256}, the record's is {pinned}")
    trace_path = os.path.join(recipe_quant, CAL_TRACE)
    trace_sha256, trace_oid = work.file_digest(trace_path), convert.download_record(recipe_quant, CAL_TRACE)[1]
    if trace_sha256 != trace_oid:
        raise convert.Refused(f"{CAL_TRACE} hashes to {trace_sha256}, its LFS oid is {trace_oid}")
    shards = convert.bf16_shards(bf16_root)
    bf16_revision = convert.revision_of(bf16_root, shards + [INDEX], "the BF16 checkpoint")
    config = read_json(config_path)
    return {
        "bf16": {"revision": bf16_revision,
                 "index": {"bytes": os.path.getsize(os.path.join(bf16_root, INDEX)),
                           "sha256": work.file_digest(os.path.join(bf16_root, INDEX))},
                 "shards": {s: {"bytes": os.path.getsize(os.path.join(bf16_root, s)),
                                "lfs_oid": convert.download_record(bf16_root, s)[1]} for s in shards}},
        "quant": {"repo": source["repo"], "branch": source["branch"], "revision": quant_revision,
                  "quantization_config": {"bytes": os.path.getsize(config_path), "sha256": config_sha256},
                  "made_with_exllamav3": config.get("version")},
        "calibration": {"file": CAL_TRACE, "repo": source["repo"], "revision": quant_revision,
                        "bytes": os.path.getsize(trace_path), "sha256": trace_sha256},
    }, config


def check_bf16(bf16_root, r):
    """Refused unless the checkpoint holds every matrix of the recipe, `<key>.weight`, at its shape."""
    weight_map = read_json(os.path.join(bf16_root, INDEX))["weight_map"]
    headers = {s: exl3.read_header(os.path.join(bf16_root, s)) for s in sorted(set(weight_map.values()))}
    problems = []
    for key, m in r["matrices"].items():
        name = key + ".weight"
        t = headers[weight_map[name]].get(name) if name in weight_map else None
        if t is None:
            problems.append(f"{name}: not in the checkpoint")
        elif t.shape != m["shape"]:
            problems.append(f"{name}: shape {t.shape}, the recipe's is {m['shape']}")
    if problems:
        raise convert.Refused("the BF16 checkpoint does not hold the recipe's matrices:\n  " + "\n  ".join(problems))


def disk_need(bf16_root, r, record, checkpoints):
    """The bytes the run writes at its peak: the output and exllamav3's work files (`qtensors`), each about
    the published quant's shards, and with checkpoints three copies of the fp32 calibration state."""
    shards = sum(f["bytes"] for name, f in record["source"]["files"].items() if name.endswith(".safetensors"))
    hidden = read_json(os.path.join(bf16_root, "config.json")).get("text_config", {}).get("hidden_size")
    state = r["calibration"]["rows"] * r["calibration"]["cols"] * (hidden or 0) * 4
    return 2 * shards + (CHECKPOINT_COPIES * state if checkpoints else 0)


def exllamav3_convert(argv):
    """exllamav3's own entry point (`python -m exllamav3.conversion.convert_model`): its parser, its
    `prepare` (which checks the job and writes `args.json`) and its `main`."""
    cm = importlib.import_module(ENTRY)
    args = cm.parser.parse_args(argv)
    try:
        in_args, job_state, ok, error = cm.prepare(args)
    except ValueError as e:
        raise convert.Refused(f"exllamav3 refused the job: {e}")
    if not ok:
        raise convert.Refused(f"exllamav3 refused the job: {error}")
    cm.main(in_args, job_state)


def peaks(device=0):
    """This process's peaks: the CUDA allocator's (allocated, reserved) and the host's (working set and
    commit, which Windows keeps per process)."""
    import psutil
    import torch
    mem = psutil.Process().memory_info()
    return {"gpu": torch.cuda.get_device_name(device),
            "vram_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "vram_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "host_peak_working_set_bytes": getattr(mem, "peak_wset", None),
            "host_peak_commit_bytes": getattr(mem, "peak_pagefile", None)}


def check_output(out, r):
    """What keeps the quant `out` from being the recipe's: one line per problem. Every matrix of the
    recipe at its K and shape, none outside it but the MTP layer's at the MTP bits, the configuration
    the 16 GB converter takes."""
    try:
        quant = exl3.Quant(out)
    except (OSError, ValueError) as e:
        return [f"{out} is not an EXL3 quant the 16 GB converter reads: {e}"]
    problems = []
    try:
        convert.check_quant_config(quant.config)
    except convert.Refused as e:
        problems.append(str(e))
    for k in ("calibration", "mtp_bits"):
        if quant.config.get(k) != r[k]:
            problems.append(f"{QUANT_CONFIG} has {k} = {quant.config.get(k)!r}, the recipe's is {r[k]!r}")
    keys = set(quant.keys())
    for key in sorted(keys | set(r["matrices"])):
        want = r["matrices"].get(key)
        if key not in keys:
            problems.append(f"{key}: missing")
            continue
        try:
            m = quant.matrix(key)
        except ValueError as e:
            problems.append(str(e))
            continue
        k = recipe.rate(m.k2 / 2)
        if want is None:
            if not key.startswith("mtp."):
                problems.append(f"{key}: quantized, and not in the recipe")
            elif k != r["mtp_bits"]:
                problems.append(f"{key}: K {k}, the MTP bits are {r['mtp_bits']}")
        elif k != want["K"]:
            problems.append(f"{key}: K {k}, the recipe's is {want['K']}")
        elif m.shape != want["shape"]:
            problems.append(f"{key}: shape {m.shape}, the recipe's is {want['shape']}")
    return problems


def pin_dir(root):
    """{name: {bytes, sha256}} of every file of `root`."""
    names = sorted(n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n)))
    with ThreadPoolExecutor(max_workers=6) as pool:
        digests = {n: pool.submit(work.file_digest, os.path.join(root, n)) for n in names}
        return {n: {"bytes": os.path.getsize(os.path.join(root, n)), "sha256": d.result()} for n, d in digests.items()}


def gib(n):
    return "?" if n is None else f"{n / 2**30:.2f} GiB"


def run(bf16_root, bf16_repo, recipe_quant, record_path, out, work_dir, checkpoints=True, resume=False, check=False,
        log=print, disk_free=None):
    """Converts the checkpoint `bf16_root` into the quant `out` and writes `<out>.recipe.yaml` and
    `<out>.pins.json` beside it (`check` stops before the GPU step and writes nothing). Raises Refused."""
    t0 = time.time()
    if not is_empty(out):
        raise convert.Refused(f"{out} is not empty: remove it to convert again")
    if resume and is_empty(work_dir):
        raise convert.Refused(f"--resume takes exllamav3's work directory of an interrupted job; {work_dir} is empty")
    if not resume and not is_empty(work_dir):
        raise convert.Refused(f"{work_dir} is not empty: remove it, or --resume the job it holds")
    record = read_json(record_path)
    sources, config = pin_sources(bf16_root, recipe_quant, record)
    r = recipe.derive(config)
    check_bf16(bf16_root, r)
    need = disk_need(bf16_root, r, record, checkpoints)
    anchor = os.path.dirname(os.path.abspath(out))
    while not os.path.isdir(anchor):
        anchor = os.path.dirname(anchor)
    free = (disk_free or (lambda p: shutil.disk_usage(p).free))(anchor)
    log(f"recipe: {len(r['matrices'])} matrices {recipe.formats(r)}, head {r['head_bits']}, MTP {r['mtp_bits']}, "
        f"bits {r['bits']}; {bf16_repo} at {sources['bf16']['revision']}, the recipe's quant at "
        f"{sources['quant']['revision']}; disk: {gib(need)} needed, {gib(free)} free")
    if not resume and free < need:
        raise convert.Refused(f"the run writes up to {gib(need)} and {anchor} has {gib(free)} free"
                              + ("" if not checkpoints else "; --no-checkpoints saves exllamav3's checkpoints"))
    ex = recipe.exllamav3(r)
    paths = {"bf16": bf16_root, "work": work_dir, "out": out, "recipe": beside(out, ".recipe.yaml"),
             "cal_data": os.path.join(recipe_quant, CAL_TRACE)}
    argv = exllamav3_argv(r, paths, checkpoints, resume)
    if check:
        log("CHECK PASS")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    recipe.write_yaml(paths["recipe"], ex)
    log(f"exllamav3: {' '.join(argv)}")
    started = time.time()
    exllamav3_convert(argv)
    seconds = round(time.time() - started)
    measured = peaks()
    log(f"exllamav3: {seconds} s; VRAM peak {gib(measured['vram_peak_allocated_bytes'])} allocated, "
        f"{gib(measured['vram_peak_reserved_bytes'])} reserved; host peak {gib(measured['host_peak_working_set_bytes'])} "
        f"working set, {gib(measured['host_peak_commit_bytes'])} commit")

    problems = check_output(out, r)
    if problems:
        raise convert.Refused("the quant is not the recipe's:\n  " + "\n  ".join(problems))
    files = pin_dir(out)
    record_out = {
        "schema": SCHEMA,
        "files": files,
        "provenance": {
            "status": "complete",
            "bf16": {"repo": bf16_repo, **sources["bf16"]},
            "quant": sources["quant"],
            "calibration": dict(sources["calibration"], **r["calibration"]),
            "recipe": {"file": os.path.basename(paths["recipe"]), "sha256": work.file_digest(paths["recipe"]),
                       "formats": recipe.formats(r), "vision_bits": VISION_BITS, "mtp_bits": r["mtp_bits"],
                       "codebook": r["codebook"], "out_scales": r["out_scales"], "exllamav3": ex},
            "exllamav3": {"version": convert.versions()["exllamav3"], "entry": ENTRY,
                          "argv": exllamav3_argv(r, {k: f"<{k}>" for k in ROLES}, checkpoints, resume)},
            "run": dict(measured, seconds=seconds, resumed=resume),
            "converter": convert.tool_commit(),
            "versions": convert.versions(),
        },
    }
    work.write_json_atomic(beside(out, ".pins.json"), record_out)
    log(f"{os.path.basename(beside(out, '.pins.json'))}: {len(files)} files, "
        f"{sum(f['bytes'] for f in files.values()) / 1e9:.2f} GB; {round(time.time() - t0)} s in all")
    return record_out


def main(argv=None, disk_free=None, errors_fn=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="the GPU step: exllamav3's conversion at the recipe, the check, the pins")
    r.add_argument("--bf16", required=True, help="the fine-tune's BF16 checkpoint, with .cache/huggingface/download")
    r.add_argument("--bf16-repo", required=True, help="its repository on the hub, for the provenance")
    r.add_argument("--recipe-quant", required=True, help="the published quant's quantization_config.json and "
                                                         "cal_trace.safetensors, with .cache/huggingface/download")
    r.add_argument("--record", required=True, help="the default 16 GB artifact's .conversion.json")
    r.add_argument("--out", required=True, help="the quant directory to write (absent or empty)")
    r.add_argument("--work", required=True, help="exllamav3's work directory (absent or empty, or the "
                                                 "interrupted job's with --resume)")
    r.add_argument("--resume", action="store_true", help="resume the interrupted job in --work (exllamav3's -r)")
    r.add_argument("--no-checkpoints", action="store_true", help="no checkpoints of the calibration state (saves "
                                                                 "three copies of it on disk; no --resume)")
    r.add_argument("--check", action="store_true", help="check the sources and stop before the GPU")
    c = sub.add_parser("control", help="the GPU step: each matrix's error against the BF16, beside the default's")
    c.add_argument("--quant", required=True, help="the quant `run` wrote")
    c.add_argument("--bf16", required=True, help="the fine-tune's BF16 checkpoint")
    c.add_argument("--bf16-repo", required=True, help="its repository on the hub")
    c.add_argument("--record", required=True, help="the default 16 GB artifact's .conversion.json")
    c.add_argument("--out", required=True, help="the control's JSON file to write")
    a = ap.parse_args(argv)
    log = lambda line: print(line, flush=True)
    try:
        if a.command == "run":
            run(a.bf16, a.bf16_repo, a.recipe_quant, a.record, a.out, a.work, checkpoints=not a.no_checkpoints,
                resume=a.resume, check=a.check, log=log, disk_free=disk_free)
        else:
            control(a.quant, a.bf16, a.bf16_repo, a.record, a.out, errors_fn=errors_fn, log=log)
    except convert.Refused as e:
        print(f"REFUSED: {e}", flush=True)
        return convert.EXIT_REFUSED
    return 0


def summary(values):
    return {"max": max(values), "mean": round(sum(values) / len(values), 6)}


def control(quant_root, bf16_root, bf16_repo, record_path, out_path, errors_fn=None, log=print):
    """Writes `out_path`: each matrix of the quant `quant_root`, under the default 16 GB artifact's object
    name, with its relative reconstruction error against the checkpoint `bf16_root` beside the default
    artifact's error on the same matrix against the base BF16 (its record's `trellis.matrices`). Refused
    when the record does not hold the quant's matrices at their K."""
    record = read_json(record_path)
    quant = exl3.Quant(quant_root)
    matrices, _ = convert.text_matrices(quant)
    default = {m["source"]: (name, m) for name, m in record["trellis"]["matrices"].items()}
    problems = [f"{key}: not in the record" for key in sorted(set(matrices) - set(default))]
    problems += [f"{key}: in the record, not in the quant" for key in sorted(set(default) - set(matrices))]
    for key in sorted(set(matrices) & set(default)):
        name, m = default[key]
        if m["k2"] != matrices[key].k2:
            problems.append(f"{name}: K {matrices[key].k2 / 2:g}, the record's is {m['k2'] / 2:g}")
    if problems:
        raise convert.Refused("the record is not of the quant's matrices:\n  " + "\n  ".join(problems))
    shards = convert.bf16_shards(bf16_root)
    revision = convert.revision_of(bf16_root, shards + [INDEX], "the BF16 checkpoint")
    stand_ins = [plan.Planned("text", default[key][0], "tensor", m.format, exl3.LAYOUT, m.shape, key=key)
                 for key, m in matrices.items()]
    errors = (errors_fn or reconstruction.rel_errors)(quant, bf16_root, stand_ins, log=log)
    rows = {}
    for p in sorted(stand_ins, key=lambda p: p.name):
        rows[p.name] = {"source": p.key, "K": recipe.rate(matrices[p.key].k2 / 2),
                        "rel_error": round(errors[p.name], 6), "default_rel_error": default[p.key][1]["rel_error"]}
    log(f"{'object':<44} {'K':>3} {'rel_error':>10} {'default':>10}")
    for name, row in rows.items():
        log(f"{name:<44} {row['K']:>3} {row['rel_error']:>10.6f} {row['default_rel_error']:>10.6f}")
    pins = beside(quant_root, ".pins.json")
    result = {
        "schema": CONTROL_SCHEMA,
        "quant": {"dir": os.path.basename(os.path.normpath(quant_root)),
                  "pins": {"file": os.path.basename(pins), "sha256": work.file_digest(pins)}
                  if os.path.isfile(pins) else None},
        "reference": {"repo": bf16_repo, "revision": revision},
        "default": {"record": {"file": os.path.basename(record_path), "sha256": work.file_digest(record_path)},
                    "identity": record.get("identity"),
                    "reference": {k: (record.get("reference") or {}).get(k) for k in ("repo", "revision")}},
        "summary": {"matrices": len(rows), "rel_error": summary([r["rel_error"] for r in rows.values()]),
                    "default_rel_error": summary([r["default_rel_error"] for r in rows.values()])},
        "matrices": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    work.write_json_atomic(out_path, result)
    log(f"control: {len(rows)} matrices; rel_error against {bf16_repo} max {result['summary']['rel_error']['max']:.4f}, "
        f"mean {result['summary']['rel_error']['mean']:.4f}; the default's against its BF16 max "
        f"{result['summary']['default_rel_error']['max']:.4f}, mean {result['summary']['default_rel_error']['mean']:.4f}")
    return result


if __name__ == "__main__":
    sys.exit(main())
