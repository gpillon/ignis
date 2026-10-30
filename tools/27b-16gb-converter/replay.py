"""The 70 matrices the huihui-ai abliteration changed, re-quantized by replaying exllamav3's own
conversion of them (spec #386, GitHub #421).

The abliterated BF16 is the base BF16 with huihui's 70 changed tensors laid over it. Its changed
matrices (layers 17-51: the MLP down projection, and the GDN output or the attention output
projection by layer type) are re-quantized at the K the `SC_3.00bpw_H4` quant gives each, the way
exllamav3's converter (`exllamav3/conversion/convert_model.py`, 1.5.3) made the quant's records:

- the calibration rows are the quant's own trace (`cal_trace.safetensors`, 250 rows x 2,048);
- the state entering layer L comes from layers 0..L-1 as quantized: the quant's records, with the
  re-quantized matrices in their place from layer 17 on;
- the Hessian is captured on layer L of the abliterated BF16, run whole;
- each matrix takes the seed the converter gives its layer, the module index L+1 (module 0 is
  the token embedding);
- each matrix is quantized standalone by the converter's per-matrix call, with the `mul1` codebook
  and output scales.

The output is a directory of EXL3 records for exactly those matrices, under the quant's names, in
the quant's layout (index, shards, `quantization_config.json`), which the converter's override
reads. turboderp's quant was made with exllamav3 1.4.2, so its bytes are not reproduced bit for bit.

`plan` is the pure step (no torch); `fetch` takes huihui's changed tensors by HTTP range; `run` is
the GPU step and the only exllamav3 user.
"""
import hashlib
import http.client
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import plan as plan_mod
import work
from override import FETCH_RECORD, INDEX, REPLAY_RECORD, mismatch, read_json

HF = "https://huggingface.co"
# The abliteration the 32 GB artifact was made from, at the revision it used.
HUIHUI_REPO = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"
HUIHUI_REVISION = "739e3c5b89849f6c238ce1e5b70008612ae42cdd"
SCHEMA = "qwen3.8-27b-trellis-replay-v1"
# The quant's calibration rows: its own trace, shipped in its repository.
CAL_TRACE = "cal_trace.safetensors"
# The converter's arguments the per-matrix call reads (convert_model.py's defaults and the quant's
# recipe): output scales always, the mul1 codebook, the output-side Hessian regularization.
QUANT_ARGS = {"apply_out_scales": True, "codebook": "mul1", "hessians_reg": 0.025}
# The paths a command names, recorded by their role: the record carries no local path.
PATH_ARGS = {"--quant": "<quant>", "--bf16": "<bf16>", "--abliterated": "<abliterated>", "--out": "<out>"}
FIRST_LAYER, LAST_LAYER = 17, 51
# A layer's output projection, by layer type: the quant holds exactly one of them per layer.
OUTPUTS = ("linear_attn.out_proj", "self_attn.o_proj")
DOWN = "mlp.down_proj"


class Replayed:
    """One matrix of the replay: the quant's record `key`, its layer, its seed, `2·K` and shape."""

    def __init__(self, key, layer, k2, rows, columns):
        self.key, self.layer, self.k2, self.rows, self.columns = key, layer, k2, rows, columns

    @property
    def seed(self):
        """The converter seeds a layer's matrices with its module index, L+1."""
        return self.layer + 1

    @property
    def shape(self):
        return [self.rows, self.columns]


def plan(quant, first=FIRST_LAYER, last=LAST_LAYER):
    """The matrices to re-quantize, layer by layer: each layer's output projection, then its down
    projection, with the K the quant (`exl3.Quant`) gives each."""
    out = []
    for layer in range(first, last + 1):
        prefix = plan_mod.QUANT_LAYER.format(layer)
        outputs = [prefix + m for m in OUTPUTS if prefix + m + ".trellis" in quant.tensors]
        if len(outputs) != 1:
            raise ValueError(f"layer {layer} has {'two' if outputs else 'no'} output projection"
                             f"{'s' if outputs else ''} in the quant: {OUTPUTS}")
        for key in outputs + [prefix + DOWN]:
            m = quant.matrix(key)
            out.append(Replayed(key, layer, m.k2, m.rows, m.columns))
    return out


def write_layer(out, layer, tensors):
    """One layer's records (name -> numpy array: `.trellis`, `.suh`, `.svh`, `.mul1` per matrix) as
    a shard of `out`; returns (shard, its tensor names)."""
    from safetensors.numpy import save_file
    os.makedirs(out, exist_ok=True)
    shard = f"replay-layer-{layer:02d}.safetensors"
    save_file(dict(tensors), os.path.join(out, shard))
    return shard, sorted(tensors)


def write_index(out, quant, written, version):
    """The index of the shards `written` (shard -> names) and `quantization_config.json`: the
    quant's recipe (`exl3.Quant` config), less its tensor table, at exllamav3 `version`."""
    weight_map = {name: shard for shard, names in sorted(written.items()) for name in names}
    work.write_json_atomic(os.path.join(out, INDEX), {"metadata": {}, "weight_map": weight_map})
    config = {k: v for k, v in quant.config.items() if k != "tensor_storage"} | {"version": version}
    work.write_json_atomic(os.path.join(out, "quantization_config.json"), config)


def check_records(out, planned):
    """What keeps `out` from being exactly the plan's records, each at its K and shape: one line
    per record."""
    import exl3
    records = exl3.Quant(out)
    keys = set(records.keys())
    problems = []
    for m in planned:
        if m.key not in keys:
            problems.append(f"{m.key}: missing")
            continue
        unlike = mismatch(m.key, records.matrix(m.key), m, "the plan's")
        if unlike:
            problems.append(unlike)
    problems += [f"{key}: not in the plan" for key in sorted(keys - {m.key for m in planned})]
    return problems


def http_get(url, start=None, end=None, tries=4):
    """The bytes of `url`, or of its range [start, end), retried on a network error."""
    import urllib.request
    headers = {"User-Agent": "ignis-27b-16gb-converter"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end - 1}"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r:
                if start is not None and r.status != 206:
                    raise ValueError(f"{url}: HTTP {r.status} to a range request")
                return r.read()
        except (OSError, http.client.HTTPException):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch(planned, out, repo=HUIHUI_REPO, revision=HUIHUI_REVISION, get=http_get, workers=4, log=print):
    """Takes the planned matrices' BF16 weights from `repo` at `revision` by HTTP range, never
    whole shards, into `out`: each source shard's planned tensors in a file of the shard's name,
    an index, and `fetch.json` (the revision, each shard's LFS oid, each tensor's source range and
    sha256), written last. A tensor whose dtype or shape is not the plan's, or whose bytes are not
    its header's length, is refused. `get(url, start=None, end=None)` reads a URL or a range of it."""
    if os.path.isdir(out) and os.listdir(out):
        raise ValueError(f"{out} is not empty: remove it to fetch again")

    def url(name):
        return f"{HF}/{repo}/resolve/{revision}/{name}"

    index = get(url(INDEX))
    weight_map = json.loads(index)["weight_map"]
    listing = {f["path"]: f for f in json.loads(get(f"{HF}/api/models/{repo}/tree/{revision}"))}
    wanted = {m.key + ".weight": m.shape for m in planned}
    missing = sorted(set(wanted) - set(weight_map))
    if missing:
        raise ValueError(f"{repo}@{revision} has no {missing[:5]}")
    by_shard = {}
    for name in sorted(wanted):
        by_shard.setdefault(weight_map[name], []).append(name)

    os.makedirs(out, exist_ok=True)
    jobs, tensors, shards = [], {}, {}
    for shard, names in sorted(by_shard.items()):
        (n,) = struct.unpack("<Q", get(url(shard), 0, 8))
        header = json.loads(get(url(shard), 8, 8 + n))
        written, size = {}, 0
        for name in names:
            h = header.get(name)
            if h is None:
                raise ValueError(f"{shard} has no {name}, which the index puts there")
            rows, columns = wanted[name]
            start, end = h["data_offsets"]
            if h["dtype"] != "BF16" or h["shape"] != [rows, columns] or end - start != 2 * rows * columns:
                raise ValueError(f"{name}: {h['dtype']} {h['shape']}, {end - start} bytes in {shard}, "
                                 f"not BF16 [{rows}, {columns}]")
            written[name] = {"dtype": "BF16", "shape": h["shape"], "data_offsets": [size, size + end - start]}
            tensors[name] = {"shard": shard, "source_range": [8 + n + start, 8 + n + end], "bytes": end - start,
                             "dtype": "BF16", "shape": h["shape"]}
            size += end - start
        raw = json.dumps(written).encode()
        raw += b" " * (-len(raw) % 8)
        with open(os.path.join(out, shard), "wb") as f:
            f.write(struct.pack("<Q", len(raw)) + raw)
            f.truncate(8 + len(raw) + size)
        for name in names:
            jobs.append((shard, name, 8 + len(raw) + written[name]["data_offsets"][0]))
        listed = listing.get(shard) or {}
        shards[shard] = {"bytes": listed.get("size"), "lfs_oid": (listed.get("lfs") or {}).get("oid")}

    def take(job):
        shard, name, at = job
        start, end = tensors[name]["source_range"]
        t0 = time.time()
        data = get(url(shard), start, end)
        if len(data) != end - start:
            raise ValueError(f"{name}: {len(data)} bytes read from {shard}, its header says {end - start}")
        with open(os.path.join(out, shard), "r+b") as f:
            f.seek(at)
            f.write(data)
        log(f"{name}: {len(data) / 1e6:.0f} MB in {time.time() - t0:.1f} s")
        return name, hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, digest in pool.map(take, jobs):
            tensors[name]["sha256"] = digest
    work.write_json_atomic(os.path.join(out, INDEX), {
        "metadata": {"source": f"{repo}@{revision}"}, "weight_map": {n: t["shard"] for n, t in sorted(tensors.items())}})
    record = {"repo": repo, "revision": revision, "index_sha256": hashlib.sha256(index).hexdigest(),
              "shards": shards, "tensors": dict(sorted(tensors.items()))}
    work.write_json_atomic(os.path.join(out, FETCH_RECORD), record)
    log(f"fetched {len(tensors)} tensors, {sum(t['bytes'] for t in tensors.values()) / 1e9:.2f} GB, "
        f"from {len(shards)} shards of {repo}@{revision}")
    return record


def bits(m):
    """The K exllamav3's strategy gives `m`: an int when whole, as `allocation._as_rate` gives it."""
    return m.k2 // 2 if m.k2 % 2 == 0 else m.k2 / 2


def public_command(argv):
    """The command line `argv` with each path argument replaced by its role."""
    out, role = ["python", "tools/27b-16gb-converter/replay.py"], None
    for a in argv[1:]:
        name, eq, _ = a.partition("=")
        if role is not None:
            out.append(role)
            role = None
        elif eq and name in PATH_ARGS:
            out.append(f"{name}={PATH_ARGS[name]}")
        else:
            out.append(a)
            role = PATH_ARGS.get(a)
    return " ".join(out)


def prepare(quant_root, bf16_root, abliterated, out, first=FIRST_LAYER, last=LAST_LAYER):
    """What the GPU step checks before it touches the GPU: an empty `out`, the quant's recipe, the
    plan, the sources' revisions, the calibration trace against its LFS oid, the abliterated tensors
    `fetch` wrote. Returns (quant, planned, pins, fetched); raises ValueError."""
    import convert
    import exl3

    if os.path.isdir(out) and os.listdir(out):
        raise ValueError(f"{out} is not empty: remove it to replay again")
    quant = exl3.Quant(quant_root)
    try:
        convert.check_quant_config(quant.config)
        planned = plan(quant, first, last)
        quant_revision = convert.revision_of(
            quant_root, list(quant.shards) + list(convert.QUANT_META) + [CAL_TRACE], "the quant")
        bf16_revision = convert.revision_of(bf16_root, convert.bf16_shards(bf16_root) + [INDEX], "the base BF16")
        cal_oid = convert.download_record(quant_root, CAL_TRACE)[1]
    except convert.Refused as e:
        raise ValueError(str(e))
    cal_path = os.path.join(quant_root, CAL_TRACE)
    cal_sha256 = work.file_digest(cal_path)
    if cal_sha256 != cal_oid:
        raise ValueError(f"{CAL_TRACE} hashes to {cal_sha256}, its LFS oid is {cal_oid}")
    fetch_path = os.path.join(abliterated, FETCH_RECORD)
    if not os.path.isfile(fetch_path):
        raise ValueError(f"{abliterated} has no {FETCH_RECORD}: fetch the abliterated tensors first")
    fetched = read_json(fetch_path)
    missing = [m.key + ".weight" for m in planned if m.key + ".weight" not in fetched["tensors"]]
    if missing:
        raise ValueError(f"{abliterated} does not hold {missing[:5]}")
    pins = {"quant_revision": quant_revision, "bf16_revision": bf16_revision, "cal_path": cal_path,
            "cal_sha256": cal_sha256}
    return quant, planned, pins, fetched


def run(quant_root, bf16_root, abliterated, out, first=FIRST_LAYER, last=LAST_LAYER, device_index=0, argv=(),
        log=print):
    """The GPU step: re-quantizes the planned matrices by replaying exllamav3's conversion, writes
    their records into `out`, then measures each against the abliterated BF16 beside the quant's own
    against the base BF16 (the control) and writes `replay.json`, the provenance, last."""
    import convert
    import exl3
    import reconstruction

    t0 = time.time()
    quant, planned, pins, fetched = prepare(quant_root, bf16_root, abliterated, out, first, last)
    quant_revision, bf16_revision = pins["quant_revision"], pins["bf16_revision"]
    cal_path, cal_sha256 = pins["cal_path"], pins["cal_sha256"]
    by_layer = {}
    for m in planned:
        by_layer.setdefault(m.layer, []).append(m)

    import torch
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.conversion.calibration_data import get_file_calibration
    from exllamav3.conversion.convert_model import check_bad_rows, get_H_data, make_quant_args, num_ref_states
    from exllamav3.modules import Linear
    from exllamav3.modules.quant import LinearEXL3, LinearFP16
    from exllamav3.modules.quant.exl3_lib.quantize import get_temp_buffers
    from exllamav3.util.memory import free_mem
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    device = torch.device(device_index)
    os.makedirs(out, exist_ok=True)
    qargs = dict(QUANT_ARGS, work_dir=out)

    # The abliterated BF16: the base, huihui's tensors laid over it (the collection maps a key to
    # the last file that holds it).
    config = Config.from_directory(bf16_root)
    config.stc.add_tensor_files(abliterated)
    for m in planned:
        held = config.stc.tensor_file_map.get(m.key + ".weight")
        if held is None or os.path.normcase(os.path.dirname(os.path.abspath(held))) != \
                os.path.normcase(os.path.abspath(abliterated)):
            raise ValueError(f"{m.key}.weight is read from {held}, not from the abliterated tensors")
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    calibration = quant.config["calibration"]
    state = get_file_calibration({"cal_data": cal_path, "cal_rows": calibration["rows"],
                                  "cal_cols": calibration["cols"]}, tokenizer)
    original_input_ids = state.copy()
    preserves = [{} for _ in state]
    bad_rows = set()
    log(f"calibration: {len(state)} rows x {state[0].shape[-1]} tokens from {CAL_TRACE}")

    def get_preserve(i, params):
        params.update(preserves[i])
        params["quant_preserve"] = preserves[i]

    def put_preserve(i, params):
        preserves[i] = params["quant_preserve"]

    q_handles = {}

    def quant_layer(layer):
        """The quant's tensors of `layer`: the records its state advances through."""
        prefix = plan_mod.QUANT_LAYER.format(layer)
        tensors = {}
        for name, t in quant.tensors.items():
            if name.startswith(prefix):
                if t.shard not in q_handles:
                    q_handles[t.shard] = safe_open(os.path.join(quant_root, t.shard), framework="pt", device="cpu")
                tensors[name] = q_handles[t.shard].get_tensor(name)
        return tensors

    def capture(module, load_device):
        """convert_model's capture: layer L of the abliterated BF16, run whole over every row."""
        defer = module.can_defer_load()
        if defer:
            config.stc.begin_deferred_load()
        try:
            module.load(load_device, keep_source_weights=True)
        finally:
            if defer:
                config.stc.end_deferred_load()
        config.stc.close()
        capture_H = {}
        for i in range(len(state)):
            if i in bad_rows:
                continue
            params = {"attn_mode": "flash_attn_nc", "capture": capture_H,
                      "activate_all_experts": model.calibration_all_experts, "input_ids": original_input_ids[i]}
            get_preserve(i, params)
            rs = module.prepare_for_device(state[i], params)
            rs = module.forward(rs, params)
            put_preserve(i, params)
            if i < num_ref_states and not torch.isfinite(rs).all().item():
                bad_rows.add(i)
                log(f" !! Non-finite reference state in calibration row {i}, excluding row")
            rs = None
        for k, v in capture_H.items():
            infs, nans = v["inf_nan"][0].item(), v["inf_nan"][1].item()
            if infs or nans:
                log(f" !! Warning: {k} state has {infs:,} inf values and {nans:,} NaN values")
            v["H_swap_device"] = v["H"].device
            v["H"] = v["H"].cpu()
        return capture_H

    results = {}

    def quantize(module, idx, matrices, capture_H):
        """convert_model's per-matrix call for each planned matrix of the layer, at the quant's K."""
        keys = {p.key for p in matrices}
        linears = {m.key: m for m in module if isinstance(m, Linear) and m.key in keys}
        if set(linears) != keys:
            raise ValueError(f"module {module.key} has no {sorted(keys - set(linears))}")
        tensors = {}
        for p in matrices:
            linear = linears[p.key]
            if not isinstance(linear.inner, LinearFP16):
                raise ValueError(f"{p.key} did not load from the BF16 checkpoint")
            linear.inner.swap_cpu()
            quant_args = make_quant_args(qargs, idx, bits(p), [device_index])
            H_data, quant_args["H_out"] = get_H_data(qargs, linear, capture_H, state)
            started = time.time()
            proxy_err = linear.convert_exl3(H_data, quant_args=quant_args, verbose=False, save_reg=False,
                                            override_swap_device=device_index)
            assert isinstance(linear.inner, LinearEXL3)
            linear.inner.swap_cpu()
            got = {k: v.cpu() for k, v in linear.get_tensors().items()}
            if set(got) != {f"{p.key}.{s}" for s in ("trellis", "suh", "svh", "mul1")}:
                raise ValueError(f"{p.key}: the quantizer returned {sorted(got)}")
            tensors.update(got)
            g_scale = quant_args.get("g_scale")
            proxy_err = float(proxy_err)
            results[p.key] = {"proxy_err": proxy_err, "g_scale": None if g_scale is None else float(g_scale),
                              "q_fallback": bool(quant_args.get("q_fallback")),
                              "apply_out_scales": bool(quant_args.get("apply_out_scales")),
                              "seconds": round(time.time() - started, 1)}
            log(f" -- {p.key}: K {bits(p)}, seed {quant_args['seed']}, proxy_err {proxy_err:.6f}, "
                f"{results[p.key]['seconds']} s")
        return tensors

    def advance(module):
        """convert_model's forward pass: every row through the layer as quantized."""
        for i in range(len(state)):
            if i in bad_rows:
                continue
            params = {"attn_mode": "flash_attn_nc", "input_ids": original_input_ids[i]}
            state[i] = module.prepare_for_device(state[i], params)
            get_preserve(i, params)
            rs = module.forward(state[i], params)
            if not torch.isfinite(rs).all().item():
                bad_rows.add(i)
                log(f" !! Non-finite hidden state in calibration row {i}, excluding row")
            state[i] = rs.cpu()
            put_preserve(i, params)
        check_bad_rows(bad_rows, len(state))

    written = {}
    with torch.inference_mode():
        # Module 0 is the token embedding, module L+1 layer L: the module index is the seed.
        for idx in range(last + 2):
            get_temp_buffers.cache_clear()
            free_mem()
            started = time.time()
            module, layer = model.modules[idx], idx - 1
            if idx > 0 and module.key != f"model.language_model.layers.{layer}":
                raise ValueError(f"module {idx} is {module.key}, not layer {layer}: its seed would not be {idx}")
            if module.num_slices != 1:
                raise ValueError(f"{module.key} is converted in {module.num_slices} slices, which the replay "
                                 f"does not do")
            load_device = torch.device("cpu") if module.caps.get("prefer_cpu") else device
            new = {}
            if layer in by_layer:
                capture_H = capture(module, load_device)
                new = quantize(module, idx, by_layer[layer], capture_H)
                capture_H = None
                module.unload()
                config.stc.close()
                shard, names = write_layer(out, layer, {k: v.numpy() for k, v in new.items()})
                written[shard] = names
            if layer == last:
                break
            # The layer as quantized: the quant's records, the re-quantized ones in their place.
            q_tensors = quant_layer(layer) if idx > 0 else {}
            q_tensors.update(new)
            if q_tensors:
                config.stc.set_new_tensors(q_tensors)
            module.load(load_device, source=q_tensors, keep_source_weights=True)
            config.stc.set_new_tensors(None)
            advance(module)
            module.unload()
            config.stc.close()
            del q_tensors, new
            log(f" -- {module.key}: {time.time() - started:.1f} s, "
                f"VRAM peak {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    state = original_input_ids = None
    q_handles.clear()
    free_mem()

    version = convert.versions()["exllamav3"]
    write_index(out, quant, written, version)
    problems = check_records(out, planned)
    if problems:
        raise ValueError("the records are not the plan's:\n  " + "\n  ".join(problems))

    # The control: each record against the abliterated BF16, beside the quant's own against the
    # base BF16 (and against the abliterated BF16: what keeping the quant's record would cost).
    stand_ins = [plan_mod.Planned("text", m.key, "tensor", exl3.FORMATS[m.k2], exl3.LAYOUT, m.shape, key=m.key)
                 for m in planned]
    records = exl3.Quant(out)
    cuda = f"cuda:{device_index}"
    replayed = reconstruction.rel_errors(records, abliterated, stand_ins, device=cuda, log=log)
    quant_base = reconstruction.rel_errors(quant, bf16_root, stand_ins, device=cuda, log=log)
    quant_abliterated = reconstruction.rel_errors(quant, abliterated, stand_ins, device=cuda, log=log)
    matrices = {}
    log(f"{'matrix':<40} {'K':>2} {'proxy_err':>10} {'replay/abl':>10} {'quant/base':>10} {'quant/abl':>10}")
    for m in planned:
        matrices[m.key] = {
            "layer": m.layer, "seed": m.seed, "K": bits(m), "format": exl3.FORMATS[m.k2], "shape": m.shape,
            **results[m.key],
            "rel_error": round(replayed[m.key], 6),
            "quant_rel_error_base": round(quant_base[m.key], 6),
            "quant_rel_error_abliterated": round(quant_abliterated[m.key], 6),
            "record_sha256": hashlib.sha256(records.record(records.matrix(m.key))).hexdigest(),
        }
        log(f"{m.key[len('model.language_model.'):]:<40} {bits(m):>2} {results[m.key]['proxy_err']:>10.6f} "
            f"{replayed[m.key]:>10.6f} {quant_base[m.key]:>10.6f} {quant_abliterated[m.key]:>10.6f}")

    def summary(name):
        values = [v[name] for v in matrices.values()]
        return {"max": max(values), "mean": round(sum(values) / len(values), 6)}

    record = {
        "schema": SCHEMA,
        "status": "complete" if (first, last) == (FIRST_LAYER, LAST_LAYER) else "partial",
        "layers": [first, last],
        "sources": {
            "quant": {"repo": convert.REPO, "branch": convert.BRANCH, "revision": quant_revision,
                      "made_with_exllamav3": quant.config.get("version")},
            "base": {"repo": convert.BF16_REPO, "revision": bf16_revision},
            "abliterated": {"repo": fetched["repo"], "revision": fetched["revision"],
                            "how": "the base BF16 with these tensors laid over it, taken by HTTP range",
                            "tensors": {n: t["sha256"] for n, t in fetched["tensors"].items()}},
        },
        "calibration": {"file": CAL_TRACE, "repo": convert.REPO, "revision": quant_revision, "sha256": cal_sha256,
                        "rows": calibration["rows"], "cols": calibration["cols"], "excluded_rows": sorted(bad_rows)},
        "quantizer": {"exllamav3": version,
                      "call": "exllamav3.conversion.convert_model's make_quant_args, get_H_data and "
                              "Linear.convert_exl3, per matrix, on one device",
                      "args": QUANT_ARGS, "seed": "the layer's module index, L+1",
                      "capture": "layer L of the abliterated BF16, run whole, on the state through layers "
                                 "0..L-1 as quantized"},
        "control": {"rel_error": summary("rel_error"), "quant_rel_error_base": summary("quant_rel_error_base"),
                    "quant_rel_error_abliterated": summary("quant_rel_error_abliterated")},
        "matrices": matrices,
        "command": public_command(argv),
        "converter": convert.tool_commit(),
        "versions": convert.versions(),
        "gpu": {"name": torch.cuda.get_device_name(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
        "seconds": round(time.time() - t0),
    }
    work.write_json_atomic(os.path.join(out, REPLAY_RECORD), record)
    log(f"{REPLAY_RECORD}: {len(matrices)} records, rel_error max {record['control']['rel_error']['max']:.4f} "
        f"(the quant's against the base BF16: max {record['control']['quant_rel_error_base']['max']:.4f}), "
        f"{record['seconds']} s")
    return record


def main(argv=None):
    import argparse
    import sys

    import exl3

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="take the planned matrices' abliterated BF16 weights by HTTP range")
    f.add_argument("--quant", required=True, help="the SC_3.00bpw_H4 download (the plan reads its headers)")
    f.add_argument("--out", required=True, help="the directory to write (absent or empty)")
    f.add_argument("--repo", default=HUIHUI_REPO)
    f.add_argument("--revision", default=HUIHUI_REVISION)
    r = sub.add_parser("run", help="the GPU step: replay the conversion of the planned matrices")
    r.add_argument("--quant", required=True, help="the SC_3.00bpw_H4 download, with .cache/huggingface/download")
    r.add_argument("--bf16", required=True, help="the base BF16 checkpoint, with .cache/huggingface/download")
    r.add_argument("--abliterated", required=True, help="the directory `fetch` wrote")
    r.add_argument("--out", required=True, help="the records directory to write (absent or empty)")
    r.add_argument("--first", type=int, default=FIRST_LAYER, help="the first layer (default %(default)s)")
    r.add_argument("--last", type=int, default=LAST_LAYER,
                   help="the last layer (default %(default)s; an earlier one is a smoke run)")
    r.add_argument("--device", type=int, default=0, help="the CUDA device (default %(default)s)")
    r.add_argument("--check", action="store_true", help="check the sources and stop before the GPU")
    a = ap.parse_args(argv)
    log = lambda line: print(line, flush=True)
    try:
        if a.command == "fetch":
            planned = plan(exl3.Quant(a.quant))
            log(f"plan: {len(planned)} matrices, layers {planned[0].layer}-{planned[-1].layer}")
            fetch(planned, a.out, repo=a.repo, revision=a.revision, log=log)
        elif a.check:
            _, planned, pins, fetched = prepare(a.quant, a.bf16, a.abliterated, a.out, a.first, a.last)
            log(f"plan: {len(planned)} matrices, layers {a.first}-{a.last}; quant {pins['quant_revision']}, "
                f"base {pins['bf16_revision']}, abliterated {fetched['repo']}@{fetched['revision']}, "
                f"{CAL_TRACE} {pins['cal_sha256']}")
            log("CHECK PASS")
        else:
            run(a.quant, a.bf16, a.abliterated, a.out, first=a.first, last=a.last, device_index=a.device,
                argv=sys.argv if argv is None else ["replay.py"] + list(argv), log=log)
    except ValueError as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
