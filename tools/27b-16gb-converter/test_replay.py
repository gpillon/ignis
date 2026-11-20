import hashlib
import json
import os
import re
import struct

import pytest

import exl3
import replay
from conftest import body_k, make_sources, write_safetensors

P = "model.language_model.layers."


def _linear(tensors, key, rows, columns, k2):
    tensors[f"{key}.trellis"] = ("I16", [columns // 16, rows // 16, k2 * 8], bytes(rows * columns // 16 * k2))
    tensors[f"{key}.suh"] = ("F16", [columns], bytes(2 * columns))
    tensors[f"{key}.svh"] = ("F16", [rows], bytes(2 * rows))
    tensors[f"{key}.mul1"] = ("I32", [], struct.pack("<I", exl3.MUL1))


def _quant(root, tensors):
    os.makedirs(root, exist_ok=True)
    write_safetensors(os.path.join(root, "model-00001-of-00001.safetensors"), tensors)
    with open(os.path.join(root, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"weight_map": {n: "model-00001-of-00001.safetensors" for n in tensors}}, f)
    with open(os.path.join(root, "quantization_config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"quant_method": "exl3", "codebook": "mul1"}, f)
    return exl3.Quant(root)


def _quant_27b_layers(root, k2=None):
    """The 27B's 64 layers (full attention at every fourth, from layer 3), reduced to each layer's
    output projection and down projection; `k2` maps a key to its 2·K (default 6, K 3)."""
    k2 = k2 or {}
    tensors = {}
    for layer in range(64):
        out = "self_attn.o_proj" if (layer + 1) % 4 == 0 else "linear_attn.out_proj"
        for module, rows, columns in ((out, 128, 128), ("mlp.down_proj", 128, 256)):
            key = f"{P}{layer}.{module}"
            _linear(tensors, key, rows, columns, k2.get(key, 6))
    return _quant(root, tensors)


def test_the_plan_is_each_layers_output_projection_and_down_projection_at_the_quants_k(tmp_path):
    quant = exl3.Quant(make_sources(str(tmp_path))["quant"])
    planned = replay.plan(quant, first=1, last=3)
    # The synthetic quant: layers 1-2 GDN (out_proj K4), layer 3 GQA (o_proj K4), down_proj K3.
    assert [(m.key, m.layer, m.seed, m.k2, m.shape) for m in planned] == [
        (P + "1.linear_attn.out_proj", 1, 2, 8, [128, 128]),
        (P + "1.mlp.down_proj", 1, 2, 6, [128, 256]),
        (P + "2.linear_attn.out_proj", 2, 3, 8, [128, 128]),
        (P + "2.mlp.down_proj", 2, 3, 6, [128, 256]),
        (P + "3.self_attn.o_proj", 3, 4, 8, [128, 128]),
        (P + "3.mlp.down_proj", 3, 4, 6, [128, 256]),
    ]


def test_on_the_27b_the_plan_is_the_70_matrices_of_layers_17_to_51(tmp_path):
    quant = _quant_27b_layers(str(tmp_path), k2={P + "23.self_attn.o_proj": 8})
    planned = replay.plan(quant)
    assert len(planned) == 70
    assert sorted({m.layer for m in planned}) == list(range(17, 52))
    attention = [m.layer for m in planned if m.key.endswith(".self_attn.o_proj")]
    assert attention == [19, 23, 27, 31, 35, 39, 43, 47, 51]
    assert sum(m.key.endswith(".linear_attn.out_proj") for m in planned) == 26
    assert sum(m.key.endswith(".mlp.down_proj") for m in planned) == 35
    assert all(m.seed == m.layer + 1 for m in planned)
    assert {m.key: m.k2 / 2 for m in planned if m.k2 != 6} == {P + "23.self_attn.o_proj": 4}


@pytest.mark.parametrize("change,message", [
    ("neither", "layer 20 has no output projection"),
    ("both", "layer 20 has two output projections"),
    ("no down", "no model.language_model.layers.20.mlp.down_proj"),
])
def test_a_layer_the_plan_cannot_read_is_refused(tmp_path, change, message):
    tensors = {}
    for layer in (19, 20):
        out = "self_attn.o_proj" if layer == 19 else "linear_attn.out_proj"
        _linear(tensors, f"{P}{layer}.{out}", 128, 128, 6)
        _linear(tensors, f"{P}{layer}.mlp.down_proj", 128, 256, 6)
    if change == "neither":
        tensors = {n: v for n, v in tensors.items() if ".20.linear_attn" not in n}
    elif change == "both":
        _linear(tensors, f"{P}20.self_attn.o_proj", 128, 128, 6)
    else:
        tensors = {n: v for n, v in tensors.items() if ".20.mlp" not in n}
    with pytest.raises(ValueError, match=message):
        replay.plan(_quant(str(tmp_path), tensors), first=19, last=20)


class Remote:
    """A model repository at one revision, served the way `replay.fetch` reads Hugging Face: the
    index whole, the file listing, shard bytes by range. `cut` shortens one tensor's range."""

    def __init__(self, root, shards, cut=None):
        self.root, self.cut, self.requests = root, cut, []
        os.makedirs(root)
        weight_map = {}
        for shard, tensors in shards.items():
            write_safetensors(os.path.join(root, shard), tensors)
            weight_map.update({n: shard for n in tensors})
        self.headers = {shard: exl3.read_header(os.path.join(root, shard)) for shard in shards}
        self.index = json.dumps({"metadata": {}, "weight_map": weight_map}).encode()
        self.tree = json.dumps([{"path": s, "size": os.path.getsize(os.path.join(root, s)),
                                 "lfs": {"oid": f"oid-{s}"}} for s in shards]).encode()

    def get(self, url, start=None, end=None):
        self.requests.append((url, start, end))
        if url == f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}":
            return self.tree
        head = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
        assert url.startswith(head), url
        name = url[len(head):]
        if name == "model.safetensors.index.json":
            return self.index
        with open(os.path.join(self.root, name), "rb") as f:
            data = f.read()[start:end]
        cut = self.cut and self.headers[name].get(self.cut)
        return data[:-1] if cut and start == cut.offset else data


REPO, REVISION = "someone/abliterated", "f" * 40


def _bf16(rows, columns, fill):
    return ("BF16", [rows, columns], bytes([fill]) * 2 * rows * columns)


def _remote_shards():
    return {
        "model-00001-of-00002.safetensors": {P + "1.linear_attn.out_proj.weight": _bf16(128, 128, 1),
                                              P + "1.mlp.down_proj.weight": _bf16(128, 256, 2),
                                              P + "1.mlp.up_proj.weight": _bf16(256, 128, 3)},
        "model-00002-of-00002.safetensors": {P + "2.linear_attn.out_proj.weight": _bf16(128, 128, 4),
                                              P + "2.mlp.down_proj.weight": _bf16(128, 256, 5)},
    }


def _planned(tmp_path):
    return replay.plan(exl3.Quant(make_sources(str(tmp_path / "src"))["quant"]), first=1, last=2)


def test_fetch_takes_only_the_planned_tensors_by_range_under_their_names(tmp_path):
    shards = _remote_shards()
    remote = Remote(str(tmp_path / "remote"), shards)
    out = str(tmp_path / "abliterated")
    replay.fetch(_planned(tmp_path), out, repo=REPO, revision=REVISION, get=remote.get, log=lambda _: None)

    with open(os.path.join(out, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    wanted = {n: s for s, tensors in shards.items() for n in tensors if not n.endswith("up_proj.weight")}
    assert weight_map == wanted
    for name, shard in wanted.items():
        t = exl3.read_header(os.path.join(out, shard))[name]
        with open(os.path.join(out, shard), "rb") as f:
            f.seek(t.offset)
            assert (t.dtype, t.shape, f.read(t.size)) == shards[shard][name]
    with open(os.path.join(out, "fetch.json"), encoding="utf-8") as f:
        record = json.load(f)
    assert (record["repo"], record["revision"]) == (REPO, REVISION)
    assert record["shards"]["model-00002-of-00002.safetensors"]["lfs_oid"] == "oid-model-00002-of-00002.safetensors"
    assert {n: t["sha256"] for n, t in record["tensors"].items()} == {
        n: hashlib.sha256(shards[s][n][2]).hexdigest() for n, s in wanted.items()}
    # the unplanned tensor's bytes were never requested
    up = remote.headers["model-00001-of-00002.safetensors"][P + "1.mlp.up_proj.weight"]
    assert all(start is None or end <= up.offset or start >= up.offset + up.size for _, start, end in remote.requests)


@pytest.mark.parametrize("change,message", [
    ("short", "down_proj.weight: 65535 bytes read from model-00002-of-00002.safetensors, its header says 65536"),
    ("shape", r"2\.linear_attn\.out_proj\.weight: BF16 \[128, 256\], 65536 bytes .*not BF16 \[128, 128\]"),
])
def test_fetch_refuses_a_tensor_not_of_its_headers_length_or_the_plans_shape(tmp_path, change, message):
    shards = _remote_shards()
    if change == "shape":
        shards["model-00002-of-00002.safetensors"][P + "2.linear_attn.out_proj.weight"] = _bf16(128, 256, 4)
    cut = P + "2.mlp.down_proj.weight" if change == "short" else None
    remote = Remote(str(tmp_path / "remote"), shards, cut=cut)
    with pytest.raises(ValueError, match=message):
        replay.fetch(_planned(tmp_path), str(tmp_path / "abliterated"), repo=REPO, revision=REVISION,
                     get=remote.get, log=lambda _: None)


def _records(planned, k2=None):
    """Each planned matrix's EXL3 tensors, as the GPU step hands them to `write_layer`, by layer."""
    import numpy as np
    k2 = k2 or {}
    layers = {}
    for m in planned:
        words = k2.get(m.key, m.k2) * 8
        layers.setdefault(m.layer, {}).update({
            f"{m.key}.trellis": np.full((m.columns // 16, m.rows // 16, words), m.layer, dtype=np.int16),
            f"{m.key}.suh": np.ones(m.columns, dtype=np.float16),
            f"{m.key}.svh": np.ones(m.rows, dtype=np.float16),
            f"{m.key}.mul1": np.array(exl3.MUL1, dtype=np.uint32).view(np.int32),
        })
    return layers


def _write(out, quant, layers):
    written = dict(replay.write_layer(out, layer, tensors) for layer, tensors in layers.items())
    replay.write_index(out, quant, written, version="1.5.3")


def test_the_records_are_an_exl3_directory_of_the_plans_matrices_at_the_quants_k_and_shape(tmp_path):
    quant = exl3.Quant(make_sources(str(tmp_path / "src"))["quant"])
    planned = replay.plan(quant, first=1, last=3)
    out = str(tmp_path / "records")
    _write(out, quant, _records(planned))

    records = exl3.Quant(out)
    assert records.keys() == sorted(m.key for m in planned)
    for m in planned:
        r, q = records.matrix(m.key), quant.matrix(m.key)
        assert (r.k2, r.shape) == (q.k2, q.shape)
        assert records.record(r)[:2] == bytes([m.layer, 0])
    assert records.config["quant_method"] == "exl3" and records.config["codebook"] == "mul1"
    assert records.config["version"] == "1.5.3"
    assert replay.check_records(out, planned) == []
    # the converter's override takes them like for like
    import convert
    import override
    with open(os.path.join(out, "replay.json"), "w", encoding="utf-8") as f:
        json.dump({"status": "complete", "sources": {"quant": {"revision": "q"}, "abliterated": {
            "repo": "r", "revision": "v", "tensors": {"t": "0"}}},
            "matrices": {m.key: {"record_sha256": hashlib.sha256(records.record(records.matrix(m.key))).hexdigest()}
                         for m in planned}}, f)
    assert override.Override(out).problems(convert.text_matrices(quant)[0]) == []


# --- the replay's GPU step on the CPU (#703) ---

from conftest import (BF16_REVISION, CAL_TRACE, OVERRIDE_REPO, OVERRIDE_REVISION, REVISION, ReplayExllamav3,  # noqa: E402
                      frozen, make_replay_sources, replay_argv)


def _timeless(record):
    """The record without its clock: the run's seconds and each matrix's."""
    out = {k: v for k, v in record.items() if k != "seconds"}
    out["matrices"] = {k: {f: v for f, v in m.items() if f != "seconds"} for k, m in record["matrices"].items()}
    return out


def replayed(tmp_path, monkeypatch, local=False, name="src"):
    """`replay.py run` on the reduced sources (downloaded, or `local` at its recipe with its pins and trace),
    on the CPU: (sources, stub, exit code, the records directory)."""
    src = make_replay_sources(str(tmp_path / name), local=local)
    stub = ReplayExllamav3(src).install(monkeypatch)
    out = str(tmp_path / name / "override")
    extra = ("--recipe", src["recipe"], "--quant-pins", src["pins"], "--trace", src["trace"]) if local else ()
    code = replay.main(replay_argv(src, out, *extra), recipes=src["recipes"])
    return src, stub, code, out


def test_a_downloaded_quants_replay_writes_the_record_it_wrote_before(tmp_path, monkeypatch, capsys):
    src, stub, code, out = replayed(tmp_path, monkeypatch)
    assert code == 0, capsys.readouterr().out
    with open(os.path.join(out, "replay.json"), encoding="utf-8") as f:
        record = json.load(f)
    # key for key, value for value, in the same order: only the clock differs
    assert json.dumps(_timeless(record)) == json.dumps(_timeless(frozen("replay.json")))
    assert stub.calibration["cal_data"] == os.path.join(src["quant"], CAL_TRACE)
    assert [(k[len(P):], seed, K) for k, seed, K in stub.quantized] == [
        ("1.linear_attn.out_proj", 2, 4), ("1.mlp.down_proj", 2, 3), ("2.linear_attn.out_proj", 3, 4),
        ("2.mlp.down_proj", 3, 3), ("3.self_attn.o_proj", 4, 4), ("3.mlp.down_proj", 4, 3)]


def test_a_local_quants_replay_names_it_by_its_pins_file_and_the_trace_by_its_recipes_pin(tmp_path, monkeypatch,
                                                                                          capsys):
    import convert
    src, stub, code, out = replayed(tmp_path, monkeypatch, local=True)
    assert code == 0, capsys.readouterr().out
    with open(os.path.join(out, "replay.json"), encoding="utf-8") as f:
        record = json.load(f)
    trace = os.path.join(src["trace"], CAL_TRACE)
    assert record["sources"]["quant"] == {
        "recipe": {"name": "ig28a", "sha256": _sha256(os.path.join(src["recipes"], "ig28a.json"))}, "revision": None,
        "pins": {"file": "quant.pins.json", "bytes": os.path.getsize(src["pins"]), "sha256": _sha256(src["pins"])},
        "made_with_exllamav3": "1.5.3"}
    assert record["calibration"] == {"file": CAL_TRACE, "recipe": "ig28a", "bytes": os.path.getsize(trace),
                                     "sha256": _sha256(trace), "rows": 4, "cols": 16, "excluded_rows": []}
    assert stub.calibration["cal_data"] == trace
    # the command names every path by its role; nothing names the downloaded quant
    assert record["command"] == ("python tools/27b-16gb-converter/replay.py run --quant <quant> --bf16 <bf16> "
                                 "--abliterated <abliterated> --out <out> --recipe ig28a --quant-pins <quant-pins> "
                                 "--trace <trace>")
    text = json.dumps(record)
    assert convert.REPO not in text and convert.BRANCH not in text and str(tmp_path) not in text
    # the rest is the downloaded quant's record: the same matrices at the same K, records, errors and sources
    def rest(r):
        return {k: v for k, v in _timeless(r).items() if k not in ("calibration", "command")} | {
            "sources": {k: v for k, v in r["sources"].items() if k != "quant"}}
    assert rest(record) == rest(frozen("replay.json"))
    with open(os.path.join(out, "quantization_config.json"), encoding="utf-8") as f:
        assert json.load(f)["bits"] == 2.79


def _local(src, *extra):
    return ("--quant-pins", src["pins"], "--trace", src["trace"], *extra)


def _said(capsys):
    return capsys.readouterr().out.splitlines()


# The reduced model's abliterated layers, for a command that runs without the stub (which sets them).
LAYERS = ("--first", "1", "--last", "3")


def test_the_check_takes_a_local_quant_at_its_recipe_with_its_pins_and_the_trace_its_recipe_pins(tmp_path, capsys):
    src = make_replay_sources(str(tmp_path), local=True)
    out = str(tmp_path / "override")
    # at the default recipe, sc30, the local quant is refused as today: its configuration is not sc30's
    assert replay.main(replay_argv(src, out, "--check", *LAYERS, *_local(src)), recipes=src["recipes"]) == 2
    assert _said(capsys)[-1] == "REFUSED: the quant's quantization_config.json has bits = 2.79, not 3.0"
    assert replay.main(replay_argv(src, out, "--check", *LAYERS, *_local(src, "--recipe", "ig28a")), recipes=src["recipes"]) == 0
    said = _said(capsys)
    assert said[-3] == ("plan: 6 matrices, layers 1-3: linear_attn.out_proj K4 x2; mlp.down_proj K3 x3; "
                        "self_attn.o_proj K4 x1")
    assert said[-2] == (f"sources: quant converted locally at the recipe ig28a, pinned by quant.pins.json "
                        f"(sha256 {_sha256(src['pins'])}), base {BF16_REVISION}, abliterated "
                        f"{OVERRIDE_REPO}@{OVERRIDE_REVISION}, {CAL_TRACE} {_sha256(os.path.join(src['trace'], CAL_TRACE))} "
                        f"(the recipe ig28a's pin)")
    assert said[-1] == "CHECK PASS"
    assert not os.path.exists(out)


def test_the_check_of_a_downloaded_quant_names_its_revision_and_the_plans_k(tmp_path, capsys):
    src = make_replay_sources(str(tmp_path))
    assert replay.main(replay_argv(src, str(tmp_path / "override"), "--check", *LAYERS), recipes=src["recipes"]) == 0
    said = _said(capsys)
    assert said[-3].startswith("plan: 6 matrices, layers 1-3: linear_attn.out_proj K4 x2;")
    assert said[-2] == (f"sources: quant {REVISION}, base {BF16_REVISION}, abliterated {OVERRIDE_REPO}@{OVERRIDE_REVISION}, "
                        f"{CAL_TRACE} {_sha256(os.path.join(src['quant'], CAL_TRACE))} (its LFS oid)")
    assert said[-1] == "CHECK PASS"


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _flip_last_byte(path):
    with open(path, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 1]))


def _edit(path, change):
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    change(value)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f)


def _recipe(src, **fields):
    from conftest import IG28A_CONFIG, pin_dir, write_recipe
    write_recipe(os.path.join(src["recipes"], "ig28a.json"), **({"weights_id": "trellis-ig28a",
                 "quantization_config": IG28A_CONFIG, "files": pin_dir(src["trace"])} | fields))


SHARD2 = "model-00002-of-00002.safetensors"
TRACE_SHA = r"[0-9a-f]{64}"


@pytest.mark.parametrize("change,message", [
    # a quant file whose bytes or sha256 are not its pin's, or that its pins do not cover
    (lambda s: _flip_last_byte(os.path.join(s["quant"], SHARD2)), f"{SHARD2} hashes to {TRACE_SHA}, its pin is {TRACE_SHA}$"),
    (lambda s: _edit(s["pins"], lambda p: p["files"]["model-00001-of-00002.safetensors"].update(bytes=1)),
     r"model-00001-of-00002\.safetensors is \d+ bytes, its pin is 1$"),
    (lambda s: _edit(s["pins"], lambda p: p["files"].pop("config.json")),
     r"config\.json: .*quant\.pins\.json gives no bytes and sha256 for it$"),
    (lambda s: _edit(s["pins"], lambda p: p.pop("files")), r"quant\.pins\.json has no `files`"),
    # a trace whose bytes or sha256 are not its recipe's pin, or that the recipe does not pin
    (lambda s: _flip_last_byte(os.path.join(s["trace"], CAL_TRACE)),
     f"cal_trace.safetensors hashes to {TRACE_SHA}, the recipe ig28a's pin is {TRACE_SHA}$"),
    (lambda s: open(os.path.join(s["trace"], CAL_TRACE), "ab").write(b"\0"),
     r"cal_trace\.safetensors is \d+ bytes, its pin is \d+$"),
    (lambda s: os.remove(os.path.join(s["trace"], CAL_TRACE)), r"cal_trace\.safetensors is pinned, and missing from "),
    (lambda s: _recipe(s, files={}), r"cal_trace\.safetensors: the recipe ig28a gives no bytes and sha256 for it$"),
    # a quant that is not the named recipe's: its configuration, a matrix's K
    (lambda s: _edit(os.path.join(s["quant"], "quantization_config.json"), lambda c: c.update(out_scales="auto")),
     "the quant's quantization_config.json has out_scales = 'auto', not 'always'$"),
    (lambda s: _recipe(s, tensors=body_k() | {P + "0.mlp.down_proj": 2}),
     r"the quant is not the recipe ig28a's:\n  model\.language_model\.layers\.0\.mlp\.down_proj: K 3, the recipe's is 2$"),
])
def test_a_local_quant_whose_pins_trace_or_recipe_do_not_hold_is_refused_naming_the_file_or_field(
        tmp_path, capsys, change, message):
    src = make_replay_sources(str(tmp_path), local=True)
    change(src)
    out = tmp_path / "override"
    for check in (("--check",), ()):     # the GPU step checks what the check does, before the GPU
        argv = replay_argv(src, str(out), *check, *LAYERS, *_local(src, "--recipe", "ig28a"))
        assert replay.main(argv, recipes=src["recipes"]) == 2
        said = capsys.readouterr().out
        assert said.startswith("REFUSED: ") and re.search(message, said.strip(), re.M), said
    assert not out.exists()


@pytest.mark.parametrize("local,argv,message", [
    (True, ("--recipe", "ig28a", "--quant-pins", "{pins}"), "a quant converted locally .--quant-pins. takes --trace"),
    (True, ("--recipe", "ig28a", "--trace", "{trace}"), "--trace is a quant converted locally's, with --quant-pins"),
    # without its pins a local quant is refused as a quant with no download record
    (True, ("--recipe", "sc30",), "bits = 2.79, not 3.0"),
    (False, ("--recipe", "ig28a"), "a downloaded quant is named turboderp/Qwen3.8-27B-exl3 SC_3.00bpw_H4, the recipe "
                                   "sc30's: the recipe ig28a is replayed from a quant converted locally"),
    (False, ("--recipe", "ig28c"), "no recipe 'ig28c' in .* .its recipes: ig28a, sc30."),
])
def test_a_quant_named_as_the_other_kind_or_at_no_recipe_is_refused(tmp_path, capsys, local, argv, message):
    src = make_replay_sources(str(tmp_path), local=local)
    if not local:
        _recipe(dict(src, trace=src["quant"]))
    argv = [a.format(**src) for a in argv]
    assert replay.main(replay_argv(src, str(tmp_path / "override"), "--check", *LAYERS, *argv),
                       recipes=src["recipes"]) == 2
    said = capsys.readouterr().out
    assert said.startswith("REFUSED: ") and re.search(message, said), said


def test_records_not_the_plans_are_named(tmp_path):
    quant = exl3.Quant(make_sources(str(tmp_path / "src"))["quant"])
    planned = replay.plan(quant, first=1, last=3)
    layers = _records(planned, k2={P + "2.mlp.down_proj": 8})
    del layers[3][P + "3.self_attn.o_proj.trellis"]
    for s in ("suh", "svh", "mul1"):
        del layers[3][P + f"3.self_attn.o_proj.{s}"]
    out = str(tmp_path / "records")
    _write(out, quant, layers)
    assert replay.check_records(out, planned) == [
        f"{P}2.mlp.down_proj: K 4, the plan's is 3",
        f"{P}3.self_attn.o_proj: missing",
    ]
