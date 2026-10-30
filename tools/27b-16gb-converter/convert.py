"""The 16 GB artifact's converter (spec docs/specs/27b-16gb/01-the-16gb-artifact.md, GitHub #338).

`run` writes the packer's work tree (docs/formats/flash-next-layout.md §2: per unit a
`tensors.json`, one payload file per tensor and a `DONE` written last; `frontend/` holds the
resources) and `converter.json`, the converter's half of the sidecar. The packer
(`ignis-artifact-pack --family qwen3.8-27b-trellis`) assembles the container from it; Python never
writes the container.

- The body and the output head are the public quant's trellis records, repacked bit for bit
  (`exl3.py`); everything else is the served artifact's object, byte for byte (`plan.py`).
- The source is pinned: the quant's repository, branch and revision (from its Hugging Face
  download metadata), every file the converter reads by sha256, each shard's checked against its
  LFS oid; the served artifact and its graft sidecar by sha256.
- Every trellis matrix's reconstruction error against the BF16 checkpoint is recorded, and one
  above `--error-bound` fails the conversion (`reconstruction.py`, the GPU step). The BF16 checkpoint
  it is measured against is pinned like the quant: its revision, every file the step reads by
  sha256, each shard against its LFS oid. `--skip-errors` writes a
  `dry-run` tree instead (the packer packs it only when asked).

`verify` reads a packed artifact back: its identity, its object table against the plan, every
trellis object against its source tensors bit for bit, every copied object's sha256 against the
served one's. With `--baseline`, a variant is also read against the artifact it varies (the
abliterated 16 GB artifact against the default one, GitHub #422): the same directory under another
identity, every payload the baseline's bit for bit but the override's records, which differ. It
prints VERIFY PASS or VERIFY FAILED (exit 3).

  python convert.py run --served S.ninfer --quant DIR --bf16 DIR --work DIR [--skip-errors]
  python convert.py verify --artifact A.ninfer --served S.ninfer --quant DIR [--baseline B.ninfer] [--report R.json]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import container
import exl3
import override as override_mod
import plan
import reconstruction
import work

SCHEMA = "qwen3.8-27b-trellis-converter-v1"
IDENTITY = {"model_id": "qwen3.8-27b", "weights_id": "trellis-sc30"}
SERVED_IDENTITY = {"model_id": "qwen3.8-27b", "weights_id": "nvfp4full"}
REPO = "turboderp/Qwen3.8-27B-exl3"
BRANCH = "SC_3.00bpw_H4"
# The quant's recipe as its quantization_config.json states it (ADR 0060 §1, §4).
QUANT_CONFIG = {"quant_method": "exl3", "codebook": "mul1", "out_scales": "always", "bits": 3.0, "head_bits": 4}
# The quant's files the converter reads, beside its shards.
QUANT_META = ("model.safetensors.index.json", "quantization_config.json", "config.json")
# The override's files beside its shards (it holds no config.json; replay.json is its provenance).
OVERRIDE_META = (override_mod.INDEX, "quantization_config.json", override_mod.REPLAY_RECORD)
BF16_REPO = "Qwen/Qwen3.8-27B"
INDEX = override_mod.INDEX
EXIT_REFUSED = 2
EXIT_VERIFY_FAILED = 3


class Refused(Exception):
    """A conversion that must not produce a work tree."""


def download_record(root, name):
    """(revision, etag) of a file in a Hugging Face download (`.cache/huggingface/download`): an LFS
    file's etag is its sha256."""
    path = os.path.join(root, ".cache", "huggingface", "download", name + ".metadata")
    if not os.path.isfile(path):
        raise Refused(f"{root} has no Hugging Face download record for {name}: the revision cannot be pinned")
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().split("\n")]
    if len(lines) < 2 or not lines[0] or not lines[1]:
        raise Refused(f"{path} is not a Hugging Face download record")
    return lines[0], lines[1]


def revision_of(root, names, what):
    """The one revision the files `names` of `root` were downloaded at."""
    revisions = {download_record(root, name)[0] for name in names}
    if len(revisions) != 1:
        raise Refused(f"{what}'s files come from different revisions: {sorted(revisions)}")
    return revisions.pop()


def pin_files(root, digests, lfs):
    """{name: {bytes, sha256}} from `digests` (name -> future sha256), each file of `lfs` checked
    against its LFS oid."""
    out = {}
    for name, future in digests.items():
        digest = future.result()
        oid = download_record(root, name)[1]
        if name in lfs and digest != oid:
            raise Refused(f"{name} hashes to {digest}, its LFS oid is {oid}")
        out[name] = {"bytes": os.path.getsize(os.path.join(root, name)), "sha256": digest}
    return out


def pin_local(root, digests):
    """{name: {bytes, sha256}} from `digests` (name -> future sha256) for files with no download
    record: the digest is taken at conversion time."""
    return {name: {"bytes": os.path.getsize(os.path.join(root, name)), "sha256": future.result()}
            for name, future in digests.items()}


def bf16_shards(bf16_root):
    """The BF16 checkpoint's shards (the reconstruction step reads its linears from them)."""
    with open(os.path.join(bf16_root, INDEX), encoding="utf-8") as f:
        return sorted(set(json.load(f)["weight_map"].values()))


def check_quant_config(config):
    for k, v in QUANT_CONFIG.items():
        if config.get(k) != v:
            raise Refused(f"the quant's quantization_config.json has {k} = {config.get(k)!r}, not {v!r}")


def text_matrices(quant):
    """(matrices, unused): the quant's text body and head, keyed; the other quantized keys."""
    matrices, unused = {}, []
    for key in quant.keys():
        if key.startswith("model.language_model.") or key == "lm_head":
            matrices[key] = quant.matrix(key)
        else:
            unused.append(key)
    return matrices, unused


def file_name(p):
    """The work file of object `p` inside its unit: the name past the unit, '/' -> '.'."""
    rest = p.name[len(p.unit) + 1:]
    return rest if p.kind == "resource" else rest.replace("/", ".") + ".bin"


def tool_commit():
    """{tool, commit, dirty}: the commit the converter ran from, and whether its files differed from it."""
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"tool": "tools/27b-16gb-converter", "commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--", "."))}
    except (OSError, subprocess.CalledProcessError):
        return {"tool": "tools/27b-16gb-converter", "commit": None, "dirty": None}


def identity_of(weights_id):
    """The artifact's identity; `weights_id` is the packer's option for a variant."""
    return IDENTITY if weights_id is None else IDENTITY | {"weights_id": weights_id}


def refuse_override(override, matrices):
    problems = override.problems(matrices)
    if problems:
        raise Refused("the override cannot replace the quant's records:\n  " + "\n  ".join(problems))


def versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for dist in ("torch", "exllamav3", "safetensors"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def convert(served_path, quant_root, bf16_root, out_work, skip_errors=False, error_bound=reconstruction.ERROR_BOUND,
            errors_fn=None, log=print, override=None, weights_id=None):
    """Writes the work tree `out_work`; returns the converter.json record. Raises Refused.
    `override` (an `override.Override`) swaps its records for the quant's of the same name."""
    t0 = time.time()
    if os.path.isdir(out_work) and os.listdir(out_work):
        raise Refused(f"{out_work} is not empty: remove it to convert again")
    served = container.Container(served_path)
    if served.identity != SERVED_IDENTITY:
        raise Refused(f"{served_path} is {served.identity}, not the served artifact {SERVED_IDENTITY}")
    graft_path = served_path + ".graft.json"
    if not os.path.isfile(graft_path):
        raise Refused(f"{graft_path} is missing: the DFlash2 graft's provenance comes from it")
    with open(graft_path, encoding="utf-8") as f:
        graft = json.load(f)
    quant = exl3.Quant(quant_root)
    check_quant_config(quant.config)

    matrices, unused = text_matrices(quant)
    if override is not None:
        refuse_override(override, matrices)
    objects, dropped = plan.plan(served.objects, matrices)
    log(f"plan: {len(objects)} objects ({sum(p.key is not None for p in objects)} trellis), "
        f"{len(dropped)} served objects dropped, {len(unused)} quant matrices unused")

    # The revisions are read before the GPU step; the hashes (~33 GB, ~88 GB with the BF16
    # reference) run beside it.
    quant_files = list(quant.shards) + list(QUANT_META)
    quant_revision = revision_of(quant_root, quant_files, "the quant")
    bf16_files = [] if skip_errors else bf16_shards(bf16_root) + [INDEX]
    bf16_revision = revision_of(bf16_root, bf16_files, "the BF16 checkpoint") if bf16_files else None
    ov_files, ov_reference = [], None
    if override is not None:
        if override.quant_revision != quant_revision:
            raise Refused(f"the override was made against the quant at {override.quant_revision}, "
                          f"this is {quant_revision}")
        ov_files = list(override.quant.shards) + list(OVERRIDE_META)
        try:
            ov_reference = override.pin_reference()
        except ValueError as e:
            raise Refused(str(e))
    with ThreadPoolExecutor(max_workers=6) as pool:
        def digests(root, names):
            return {name: pool.submit(work.file_digest, os.path.join(root, name)) for name in names}

        quant_digests = digests(quant_root, quant_files)
        bf16_digests = digests(bf16_root, bf16_files)
        ov_digests = digests(override.root, ov_files) if override is not None else {}
        served_digest = pool.submit(work.file_digest, served_path)
        if skip_errors:
            rel = None
        else:
            # The override's matrices are measured against its own reference BF16.
            extra = {} if override is None else {"override": override}
            rel = (errors_fn or reconstruction.rel_errors)(quant, bf16_root, objects, log=log, **extra)
        pinned = pin_files(quant_root, quant_digests, set(quant.shards))
        bf16_pinned = pin_files(bf16_root, bf16_digests, set(bf16_files) - {INDEX})
        if override is not None:
            ov_pinned = pin_local(override.root, ov_digests)
        served_sha256 = served_digest.result()
    if rel is not None:
        over = {n: e for n, e in rel.items() if not e <= error_bound}
        if over:
            raise Refused(f"reconstruction error above {error_bound} (a layout error): {over}")

    trellis, copied = {}, {}
    for unit in plan.UNITS:
        d = os.path.join(out_work, unit)
        os.makedirs(d)
        entries, digests = [], {}
        started = time.time()
        for p in (p for p in objects if p.unit == unit):
            if p.key is not None:
                m = matrices[p.key]
                swapped = override is not None and override.replaces(p.key)
                payload = override.record(p.key) if swapped else quant.record(m)
            else:
                payload = served.read(p.copy)
            digest = hashlib.sha256(payload).hexdigest()
            fname = file_name(p)
            if fname in digests:
                raise ValueError(f"two objects of {unit} share the file name {fname}")
            with open(os.path.join(d, fname), "wb") as f:
                f.write(payload)
            digests[fname] = digest
            if p.key is not None:
                trellis[p.name] = {"source": p.key, "k2": m.k2, "format": p.format,
                                   "shape": p.shape, "bytes": len(payload), "sha256": digest,
                                   "rel_error": None if rel is None else round(rel[p.name], 6)}
                if swapped:
                    trellis[p.name]["override"] = True
            else:
                if served.by_name[p.copy]["bytes"] != len(payload):
                    raise ValueError(f"{p.copy}: read {len(payload)} bytes")
                copied[p.name] = {"bytes": len(payload), "sha256": digest}
            if p.kind == "tensor":
                entries.append({"name": p.name, "file": fname, "format": p.format, "layout": p.layout,
                                "shape": p.shape, "bytes": len(payload)})
        if unit != "frontend":
            work.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": entries})
        work.mark_done(d, digests)
        log(f"{unit}: {len(digests)} objects, {time.time() - started:.1f} s")

    nvfp4_copied = {p.name for p in objects if p.copy is not None and p.format == "NVFP4"}
    local_nvfp4 = dict(graft.get("local_nvfp4") or {})
    local_nvfp4["parents"] = {n: v for n, v in (local_nvfp4.get("parents") or {}).items() if n in nvfp4_copied}
    errors_list = [t["rel_error"] for t in trellis.values() if t["rel_error"] is not None]
    k_hist = {}
    for t in trellis.values():
        k_hist[t["format"]] = k_hist.get(t["format"], 0) + 1
    record = {
        "schema": SCHEMA,
        "status": "dry-run" if skip_errors else "complete",
        "identity": identity_of(weights_id),
        "source": {
            "repo": REPO,
            "branch": BRANCH,
            "revision": quant_revision,
            "license": "apache-2.0",
            "quantization_config": {k: quant.config.get(k) for k in QUANT_CONFIG},
            "files": pinned,
        },
        "served": {
            "file": os.path.basename(served_path),
            "bytes": os.path.getsize(served_path),
            "sha256": served_sha256,
            "identity": served.identity,
            "graft_sidecar": {"file": os.path.basename(graft_path), "sha256": work.file_digest(graft_path)},
        },
        # The checkpoint the reconstruction errors are measured against (none in a dry run).
        "reference": None if skip_errors else {"repo": BF16_REPO, "revision": bf16_revision, "files": bf16_pinned},
        "graft": {
            "how": f"the {sum(n.startswith('dflash2/') for n in copied)} dflash2/* objects copied byte for byte "
                   f"from the served artifact, which grafted them",
            "recipe_id": graft.get("recipe_id"),
            "grafted_from": graft.get("grafted_from"),
            "source": graft.get("source"),
            "objects": graft.get("objects"),
        },
        # The served graft's per-object NVFP4 records of the objects copied here, so
        # `checksum::verify` covers the copied drafter in this artifact as in the served one.
        "local_nvfp4": local_nvfp4,
        "objects": {"tensors": sum(p.kind == "tensor" for p in objects),
                    "resources": sum(p.kind == "resource" for p in objects),
                    "trellis": len(trellis), "copied": len(copied)},
        "trellis": {
            "bytes": sum(t["bytes"] for t in trellis.values()),
            "formats": dict(sorted(k_hist.items())),
            "error_bound": error_bound,
            "rel_error_max": max(errors_list) if errors_list else None,
            "rel_error_mean": round(sum(errors_list) / len(errors_list), 6) if errors_list else None,
            "matrices": trellis,
        },
        "copied": {"bytes": sum(c["bytes"] for c in copied.values()), "objects": copied},
        "dropped": dropped,
        "quant_unused": unused,
        "converter": tool_commit(),
        "versions": versions(),
    }
    if override is not None:
        record["override"] = {
            "repo": override.repo,
            "revision": override.revision,
            "files": ov_pinned,
            "matrices": sorted(n for n, t in trellis.items() if t.get("override")),
            "quant_revision": override.quant_revision,
            # Their errors are measured against the BF16 the override was made from, pinned by tensor.
            "reference": ov_reference,
        }
    work.write_json_atomic(os.path.join(out_work, "converter.json"), record)
    log(f"converter.json: {record['status']}, {len(trellis)} trellis + {len(copied)} copied, "
        f"{time.time() - t0:.0f} s")
    return record


# A directory entry's fields beside its name: two artifacts with the same entries under different
# identities differ in their content hash by the identity alone.
DESCRIPTOR = ("kind", "format", "layout", "shape", "encoding", "offset", "bytes")


def descriptor(o):
    return {k: o.get(k) for k in DESCRIPTOR}


def verify(artifact_path, served_path, quant_root, log=print, override=None, weights_id=None, baseline=None):
    """(ok, report) for a packed artifact against the plan and its sources; with `override`, its
    records stand in for the quant's of the same name. With `baseline` (the artifact this one
    varies), the directory must be the baseline's under another identity, and every payload the
    baseline's bit for bit except the override's records, which must differ."""
    art = container.Container(artifact_path)
    served = container.Container(served_path)
    quant = exl3.Quant(quant_root)
    matrices, _ = text_matrices(quant)
    problems = []
    if override is not None:
        problems += override.problems(matrices)
        if problems:
            return False, {"artifact": os.path.basename(artifact_path), "problems": problems}
    objects, _ = plan.plan(served.objects, matrices)
    want_identity = identity_of(weights_id)
    if art.identity != want_identity:
        problems.append(f"identity {art.identity}, not {want_identity}")
    want = {p.name: p for p in objects}
    extra = sorted(set(art.by_name) - set(want))
    missing = sorted(set(want) - set(art.by_name))
    if extra:
        problems.append(f"objects the plan does not have: {extra[:5]}")
    if missing:
        problems.append(f"objects missing: {missing[:5]}")
    base, same, differ = None, 0, []
    if baseline is not None:
        base = container.Container(baseline)
        if base.identity == art.identity:
            problems.append(f"the baseline's identity is the artifact's, {art.identity['model_id']}/"
                            f"{art.identity['weights_id']}: the two would share a content hash")
        elif base.identity["model_id"] != art.identity["model_id"]:
            problems.append(f"the baseline is model {base.identity['model_id']}, the artifact "
                            f"{art.identity['model_id']}")
        if [o["name"] for o in base.objects] != [o["name"] for o in art.objects]:
            problems.append("the baseline's objects are not the artifact's, in the same order")
        for o in art.objects:
            b = base.by_name.get(o["name"])
            if b is not None and descriptor(b) != descriptor(o):
                problems.append(f"{o['name']}: the baseline's descriptor is {descriptor(b)}, the artifact's "
                                f"{descriptor(o)}")
    checked = {"trellis": 0, "copied": 0}
    for name, p in want.items():
        o = art.by_name.get(name)
        if o is None:
            continue
        if p.kind == "resource":
            if o["kind"] != "resource" or o["encoding"] != p.encoding:
                problems.append(f"{name}: {o['kind']} {o.get('encoding')}, not a {p.encoding} resource")
        elif (o["kind"], o.get("format"), o.get("layout"), o.get("shape")) != ("tensor", p.format, p.layout, p.shape):
            problems.append(f"{name}: {o.get('format')} {o.get('layout')} {o.get('shape')}, "
                            f"not {p.format} {p.layout} {p.shape}")
        data = art.read(name)
        if p.key is not None:
            swapped = override is not None and override.replaces(p.key)
            if data != (override.record(p.key) if swapped else quant.record(matrices[p.key])):
                problems.append(f"{name}: not {p.key}'s {'override ' if swapped else ''}trellis, "
                                f"suh and svh bit for bit")
            checked["trellis"] += 1
        else:
            if hashlib.sha256(data).digest() != hashlib.sha256(served.read(p.copy)).digest():
                problems.append(f"{name}: sha256 differs from the served object's")
            checked["copied"] += 1
        if base is not None and name in base.by_name:
            swapped = override is not None and p.key is not None and override.replaces(p.key)
            equal = base.read(name) == data
            if swapped and equal:
                problems.append(f"{name}: the override's record is the baseline's, bit for bit")
            elif not swapped and not equal:
                problems.append(f"{name}: not the baseline's, bit for bit")
            if equal:
                same += 1
            else:
                differ.append(name)
    report = {"artifact": os.path.basename(artifact_path), "identity": art.identity,
              "objects": len(art.objects), "checked": checked, "problems": problems}
    if base is not None:
        report["baseline"] = {"artifact": os.path.basename(baseline), "identity": base.identity,
                              "same": same, "differ": differ}
        log(f"verify: against the baseline {os.path.basename(baseline)}, {same} objects the same, "
            f"{len(differ)} differ")
    log(f"verify: {checked['trellis']} trellis objects, {checked['copied']} copied, {len(problems)} problems")
    return not problems, report


def add_override_arguments(r):
    r.add_argument("--override", help="a directory `replay.py run` wrote: EXL3 records that replace the quant's "
                                      "of the same name, and replay.json, their provenance")
    r.add_argument("--override-bf16", help="the partial BF16 checkpoint the override was made from, "
                                           "`replay.py fetch`'s directory (with --override)")


def load_override(root, reference_root=None):
    """The `Override` at `root`, with the partial BF16 it was made from when given (`run` needs it,
    `verify` does not); Refused naming what keeps it from loading."""
    try:
        return override_mod.Override(root, reference_root)
    except (ValueError, OSError) as e:
        raise Refused(f"the override: {e}")


def override_of(a):
    """The `Override` the arguments name, or None; Refused when they name it by halves."""
    if not a.override and not a.override_bf16:
        return None
    if not a.override or not a.override_bf16:
        raise Refused(f"--override takes {'--override-bf16' if a.override else '--override'} too")
    return load_override(a.override, a.override_bf16)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="write the packer's work tree and converter.json")
    r.add_argument("--served", required=True, help="the served qwen3_8_27b_nvfp4full-v2.ninfer (its .graft.json beside)")
    r.add_argument("--quant", required=True, help="the SC_3.00bpw_H4 download, with .cache/huggingface/download")
    r.add_argument("--bf16", required=True, help="the BF16 checkpoint the reconstruction errors are measured against")
    r.add_argument("--work", required=True, help="the work tree to write (absent or empty)")
    add_override_arguments(r)
    r.add_argument("--weights-id", help="the weights id of a variant (default %s)" % IDENTITY["weights_id"])
    r.add_argument("--skip-errors", action="store_true", help="skip the GPU step: a dry-run tree")
    r.add_argument("--error-bound", type=float, default=reconstruction.ERROR_BOUND,
                   help="the largest relative reconstruction error (default %(default)s; above it is a layout error)")
    v = sub.add_parser("verify", help="check a packed artifact against the plan and its sources")
    v.add_argument("--artifact", required=True, help="the packed .ninfer")
    v.add_argument("--served", required=True, help="the served artifact it copies from")
    v.add_argument("--quant", required=True, help="the quant it repacks")
    v.add_argument("--override", help="the override directory the artifact was packed with")
    v.add_argument("--weights-id", help="the weights id the artifact was packed with (default %s)" % IDENTITY["weights_id"])
    v.add_argument("--baseline", help="the artifact this variant varies: every object but the override's must be its, "
                                      "bit for bit")
    v.add_argument("--report", help="write the report as JSON here")
    a = ap.parse_args(argv)
    log = lambda line: print(line, flush=True)
    if a.command == "run":
        try:
            convert(a.served, a.quant, a.bf16, a.work, skip_errors=a.skip_errors, error_bound=a.error_bound, log=log,
                    override=override_of(a), weights_id=a.weights_id)
        except Refused as e:
            print(f"REFUSED: {e}", flush=True)
            return EXIT_REFUSED
        return 0
    try:
        ov = load_override(a.override) if a.override else None
    except Refused as e:
        print(f"REFUSED: {e}", flush=True)
        return EXIT_REFUSED
    ok, report = verify(a.artifact, a.served, a.quant, log=log, override=ov, weights_id=a.weights_id,
                        baseline=a.baseline)
    if a.report:
        work.write_json_atomic(a.report, report)
    for p in report["problems"]:
        print(f"  {p}", flush=True)
    print("VERIFY PASS" if ok else "VERIFY FAILED", flush=True)
    return 0 if ok else EXIT_VERIFY_FAILED


if __name__ == "__main__":
    sys.exit(main())
