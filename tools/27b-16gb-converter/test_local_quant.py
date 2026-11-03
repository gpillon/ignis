"""The fine-tune's local quant (GitHub #471): `local_quant.py run` and `control` on the CPU.

exllamav3 and torch are stubs: the stub conversion writes, into the output directory, what exllamav3's
compile writes (the shards, the index, `config.json`, `quantization_config.json`, the copied files),
holding the reduced quant of `conftest.make_sources`, so the 16 GB converter's own `convert` can take
the result through `--quant-pins`.
"""
import hashlib
import json
import os
import shutil
import sys
import types

import pytest
import yaml

import convert
import exl3
import local_quant
from conftest import REVISION, download_records, make_sources, write_safetensors

SWIFT_REVISION = "b4c84d42903a8646b25857eb2827288d92ed85a4"
SWIFT_REPO = "someone/fine-tune"
TRACE = "cal_trace.safetensors"


def _storage(quant):
    """The quant's tensor_storage as exllamav3 lists it: the main model's modules (not the MTP layer)."""
    out = {}
    for key in quant.keys():
        if key.startswith("mtp."):
            continue
        m = quant.matrix(key)
        out[key] = {"stored_tensors": {f"{key}.{s}": {"shape": quant.tensors[f"{key}.{s}"].shape}
                                       for s in ("trellis", "suh", "svh", "mul1")},
                    "quant_format": "exl3", "bits_per_weight": local_quant.recipe.rate(m.k2 / 2)}
    out["model.language_model.layers.0.input_layernorm"] = {"stored_tensors": {}}
    return out


def _sources(tmp_path):
    """The run's inputs: the published quant's two files at its revision (its config with the quant's
    tensor_storage), the fine-tune's BF16 checkpoint (every matrix's weight, its own revision), and the
    default 16 GB artifact's conversion record, made by the 16 GB converter from the reduced sources."""
    src = make_sources(str(tmp_path / "src"))
    quant = exl3.Quant(src["quant"])
    config = dict(quant.config, calibration={"rows": 250, "cols": 2048}, mtp_bits=3, tensor_storage=_storage(quant))

    # The quant states its tensor_storage, as the published one does; its download record follows.
    published = str(tmp_path / "sc30")
    for root in (src["quant"], published):
        os.makedirs(os.path.join(root, ".cache", "huggingface", "download"), exist_ok=True)
        with open(os.path.join(root, "quantization_config.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(config, f)
    download_records(src["quant"], ["quantization_config.json"], set(), REVISION)
    write_safetensors(os.path.join(published, TRACE), {"input_ids": ("I64", [4, 16], bytes(8 * 64))})
    download_records(published, ["quantization_config.json", TRACE], {TRACE}, REVISION)

    swift = str(tmp_path / "swift")
    os.makedirs(os.path.join(swift, ".cache", "huggingface", "download"))
    weights = {f"{key}.weight": ("BF16", quant.matrix(key).shape, bytes(2 * quant.matrix(key).rows *
                                                                         quant.matrix(key).columns))
               for key in quant.keys()}
    names = sorted(weights)
    shards = {"model-00001-of-00002.safetensors": names[:len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for shard, part in shards.items():
        write_safetensors(os.path.join(swift, shard), {n: weights[n] for n in part})
        weight_map.update({n: shard for n in part})
    with open(os.path.join(swift, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    with open(os.path.join(swift, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"hidden_size": 128}}, f)
    download_records(swift, list(shards) + ["model.safetensors.index.json", "config.json"], set(shards),
                     SWIFT_REVISION)

    # The default artifact's record: the 16 GB converter's own, with an error per matrix.
    errors = {}

    def measured(q, bf16, objects, log=print):
        for i, p in enumerate(o for o in objects if o.key is not None):
            errors[p.name] = round(0.1 + i / 1000, 6)
        return dict(errors)

    record = convert.convert(src["served"], src["quant"], src["bf16"], str(tmp_path / "default-work"),
                             errors_fn=measured, log=lambda _: None)
    record_path = str(tmp_path / "qwen3_8_27b_trellis_sc30-v2.ninfer.conversion.json")
    with open(record_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f)
    return {"src": src, "published": published, "swift": swift, "record": record_path, "quant": quant,
            "config": config, "default_errors": errors,
            "out": str(tmp_path / "out" / "swift-exl3"), "work": str(tmp_path / "work")}


class Exllamav3:
    """The stub of exllamav3's `convert_model`: its parser and `prepare` record the arguments, its `main`
    writes the output exllamav3's compile writes, holding the reduced quant. `change(out)` alters it."""

    def __init__(self, s, change=None):
        self.s, self.change, self.argv, self.calls = s, change, None, []

    def install(self, monkeypatch):
        cm = types.ModuleType("exllamav3.conversion.convert_model")
        cm.parser = types.SimpleNamespace(parse_args=self.parse_args)
        cm.prepare = self.prepare
        cm.main = self.main
        conversion = types.ModuleType("exllamav3.conversion")
        conversion.convert_model = cm
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(max_memory_allocated=lambda d: 11 << 30, max_memory_reserved=lambda d: 14 << 30,
                                           get_device_name=lambda d: "a test GPU")
        for name, module in {"exllamav3": types.ModuleType("exllamav3"), "exllamav3.conversion": conversion,
                             "exllamav3.conversion.convert_model": cm, "torch": torch}.items():
            monkeypatch.setitem(sys.modules, name, module)
        return self

    def parse_args(self, argv):
        self.argv = list(argv)
        self.calls.append("parse_args")
        return types.SimpleNamespace(argv=list(argv))

    def prepare(self, args):
        self.calls.append("prepare")
        a = args.argv
        self.recipe = a[a.index("--recipe") + 1]
        with open(self.recipe, encoding="utf-8") as f:
            self.recipe_read = yaml.safe_load(f)
        return {"out_dir": a[a.index("-o") + 1]}, {"next_module_idx": 0}, True, None

    def main(self, in_args, job_state):
        self.calls.append("main")
        out, src = in_args["out_dir"], self.s["src"]["quant"]
        os.makedirs(out)
        for name in os.listdir(src):
            if name.endswith(".safetensors") or name == "model.safetensors.index.json":
                shutil.copy(os.path.join(src, name), os.path.join(out, name))
        qcfg = {"quant_method": "exl3", "version": "1.5.3", "bits": 3.0, "head_bits": 4,
                "calibration": {"rows": 250, "cols": 2048}, "out_scales": "always", "codebook": "mul1",
                "mtp_bits": 3}
        with open(os.path.join(out, "config.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "quantization_config": qcfg}, f)
        with open(os.path.join(out, "quantization_config.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(qcfg, tensor_storage=self.s["config"]["tensor_storage"]), f)
        with open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write("{}")
        if self.change:
            self.change(out)


def _run(s, *extra, free=1 << 50):
    return local_quant.main(["run", "--bf16", s["swift"], "--bf16-repo", SWIFT_REPO, "--recipe-quant", s["published"],
                             "--record", s["record"], "--out", s["out"], "--work", s["work"], *extra],
                            disk_free=lambda path: free)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_run_converts_the_fine_tune_at_the_quants_recipe_with_its_calibration_trace(tmp_path, monkeypatch):
    s = _sources(tmp_path)
    ex = Exllamav3(s).install(monkeypatch)
    assert _run(s) == 0
    assert ex.calls == ["parse_args", "prepare", "main"]
    recipe_path = s["out"] + ".recipe.yaml"
    assert ex.argv == ["-i", s["swift"], "-w", s["work"], "-o", s["out"], "--recipe", recipe_path,
                       "-hb", "4", "-mb", "3", "-vb", "16", "-cb", "mul1", "--out_scales", "always",
                       "-cd", os.path.join(s["published"], TRACE), "-cr", "250", "-cc", "2048"]
    # the recipe exllamav3 read: the body at the quant's K, the head at head_bits 4, bits 3.0
    body = {k: s["quant"].matrix(k).k2 // 2 for k in s["quant"].keys() if k.startswith("model.language_model.")}
    assert ex.recipe == recipe_path
    assert ex.recipe_read == {"target_bpw": 3.0, "head_bits": 4, "tensors": body}
    assert s["quant"].matrix("lm_head").k2 == 8


def test_the_pins_cover_every_file_of_the_quant_and_the_16gb_converter_takes_them(tmp_path, monkeypatch):
    s = _sources(tmp_path)
    Exllamav3(s).install(monkeypatch)
    assert _run(s) == 0
    pins = _read(s["out"] + ".pins.json")
    assert pins["files"] == {name: {"bytes": os.path.getsize(os.path.join(s["out"], name)),
                                    "sha256": _sha256(os.path.join(s["out"], name))}
                             for name in sorted(os.listdir(s["out"]))}
    assert {"config.json", "quantization_config.json", "model.safetensors.index.json", "tokenizer.json"} <= set(pins["files"])
    # #465's contract: the 16 GB converter takes the local quant by these pins
    record = convert.convert(s["src"]["served"], s["out"], s["src"]["bf16"], str(tmp_path / "work16"),
                             skip_errors=True, log=lambda _: None, quant_pins=s["out"] + ".pins.json",
                             labels={"quant_repo": SWIFT_REPO, "quant_branch": "local"})
    assert record["source"]["revision"] is None
    assert record["source"]["pins"]["sha256"] == _sha256(s["out"] + ".pins.json")


def test_the_provenance_carries_what_a_rebuild_from_public_sources_needs(tmp_path, monkeypatch):
    s = _sources(tmp_path)
    Exllamav3(s).install(monkeypatch)
    assert _run(s) == 0
    p = _read(s["out"] + ".pins.json")["provenance"]
    assert p["bf16"]["repo"] == SWIFT_REPO and p["bf16"]["revision"] == SWIFT_REVISION
    assert set(p["bf16"]["shards"]) == {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
    assert p["quant"] == {"repo": "turboderp/Qwen3.8-27B-exl3", "branch": "SC_3.00bpw_H4", "revision": REVISION,
                          "quantization_config": {"bytes": os.path.getsize(os.path.join(s["published"], "quantization_config.json")),
                                                  "sha256": _sha256(os.path.join(s["published"], "quantization_config.json"))},
                          "made_with_exllamav3": "1.4.2"}
    trace = os.path.join(s["published"], TRACE)
    assert p["calibration"] == {"file": TRACE, "repo": "turboderp/Qwen3.8-27B-exl3", "revision": REVISION,
                                "bytes": os.path.getsize(trace), "sha256": _sha256(trace), "rows": 250, "cols": 2048}
    assert p["recipe"]["sha256"] == _sha256(s["out"] + ".recipe.yaml")
    assert p["recipe"]["exllamav3"] == yaml.safe_load(open(s["out"] + ".recipe.yaml", encoding="utf-8"))
    assert p["recipe"]["formats"] == {"TRELLIS_MUL1_K2": 8, "TRELLIS_MUL1_K3": 8, "TRELLIS_MUL1_K4": 9,
                                      "TRELLIS_MUL1_K5": 1}
    assert p["exllamav3"]["version"] == convert.versions()["exllamav3"]
    assert p["exllamav3"]["argv"][:6] == ["-i", "<bf16>", "-w", "<work>", "-o", "<out>"]
    run = p["run"]
    assert (run["gpu"], run["vram_peak_allocated_bytes"], run["vram_peak_reserved_bytes"]) == ("a test GPU", 11 << 30, 14 << 30)
    assert {"seconds", "host_peak_working_set_bytes", "host_peak_commit_bytes", "resumed"} <= set(run)
    assert str(tmp_path) not in json.dumps(p) and "H:/" not in json.dumps(p)


def _rewrite_shard(root, key, k2=None, drop=False):
    """Rewrites the shard of `root` holding `key`'s records: its trellis at `k2`, or the key dropped (and
    taken out of the index)."""
    index_path = os.path.join(root, "model.safetensors.index.json")
    index = _read(index_path)
    shard = index["weight_map"][f"{key}.trellis"]
    path = os.path.join(root, shard)
    tensors = {}
    with open(path, "rb") as f:
        data = f.read()
    for name, t in exl3.read_header(path).items():
        tensors[name] = (t.dtype, t.shape, data[t.offset:t.offset + t.size])
    for s in ("trellis", "suh", "svh", "mul1"):
        if drop:
            del tensors[f"{key}.{s}"]
            del index["weight_map"][f"{key}.{s}"]
    if k2 is not None:
        dtype, shape, _ = tensors[f"{key}.trellis"]
        tensors[f"{key}.trellis"] = (dtype, shape[:2] + [k2 * 8], bytes(shape[0] * shape[1] * 16 * k2))
    write_safetensors(path, tensors)
    with open(index_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f)


L1 = "model.language_model.layers.1."


def _another_revision(s):
    download_records(s["published"], ["quantization_config.json", TRACE], {TRACE}, "f" * 40)


def _edit_config(s):
    with open(os.path.join(s["published"], "quantization_config.json"), "a", encoding="utf-8") as f:
        f.write(" ")


def _edit_trace(s):
    with open(os.path.join(s["published"], TRACE), "ab") as f:
        f.write(b"\0")


def _drop_weight(s):
    index = _read(os.path.join(s["swift"], "model.safetensors.index.json"))
    del index["weight_map"][L1 + "mlp.down_proj.weight"]
    with open(os.path.join(s["swift"], "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)


def _reshape_weight(s):
    shard = _read(os.path.join(s["swift"], "model.safetensors.index.json"))["weight_map"][L1 + "mlp.up_proj.weight"]
    path = os.path.join(s["swift"], shard)
    tensors = {n: (t.dtype, t.shape, bytes(t.size)) for n, t in exl3.read_header(path).items()}
    tensors[L1 + "mlp.up_proj.weight"] = ("BF16", [128, 128], bytes(2 * 128 * 128))
    write_safetensors(path, tensors)


def _fill(path):
    os.makedirs(path)
    with open(os.path.join(path, "x"), "w") as f:
        f.write("x")


@pytest.mark.parametrize("change,message", [
    (_another_revision, f"was downloaded at {'f' * 40}, the record's quant is turboderp/Qwen3.8-27B-exl3 "
                        f"SC_3.00bpw_H4 at {REVISION}"),
    (_edit_config, "quantization_config.json hashes to [0-9a-f]{64}, the record's is"),
    (_edit_trace, "cal_trace.safetensors hashes to [0-9a-f]{64}, its LFS oid is"),
    (_drop_weight, L1 + r"mlp\.down_proj\.weight: not in the checkpoint"),
    (_reshape_weight, L1 + r"mlp\.up_proj\.weight: shape \[128, 128\], the recipe's is \[256, 128\]"),
    (lambda s: _fill(s["out"]), "swift-exl3 is not empty"),
    (lambda s: _fill(s["work"]), r"work is not empty: remove it, or --resume the job it holds"),
])
def test_a_source_that_does_not_hold_is_refused_before_the_gpu_and_nothing_is_written(
        tmp_path, monkeypatch, capsys, change, message):
    s = _sources(tmp_path)
    ex = Exllamav3(s).install(monkeypatch)
    change(s)
    assert _run(s) == convert.EXIT_REFUSED
    import re
    assert re.search(message, capsys.readouterr().out)
    assert ex.calls == []
    assert not os.path.exists(s["out"] + ".recipe.yaml") and not os.path.exists(s["out"] + ".pins.json")


def test_check_stops_before_the_gpu_and_writes_nothing(tmp_path, monkeypatch, capsys):
    s = _sources(tmp_path)
    ex = Exllamav3(s).install(monkeypatch)
    assert _run(s, "--check") == 0
    out = capsys.readouterr().out
    assert "recipe: 26 matrices" in out and "CHECK PASS" in out
    assert ex.calls == [] and not os.path.exists(os.path.dirname(s["out"]))


def test_a_disk_that_cannot_hold_the_checkpoints_is_refused_and_no_checkpoints_fits(tmp_path, monkeypatch, capsys):
    s = _sources(tmp_path)
    ex = Exllamav3(s).install(monkeypatch)
    # three copies of the fp32 state (250 x 2,048 x 128 x 4 bytes) do not fit in 100 MB; the outputs do
    assert _run(s, free=100 << 20) == convert.EXIT_REFUSED
    assert "--no-checkpoints saves exllamav3's checkpoints" in capsys.readouterr().out
    assert _run(s, "--no-checkpoints", free=100 << 20) == 0
    assert ex.argv[-2:] == ["-cpi", str(10 ** 9)]
    assert _read(s["out"] + ".pins.json")["provenance"]["exllamav3"]["argv"][-2:] == ["-cpi", str(10 ** 9)]


def test_resume_takes_the_interrupted_jobs_work_directory(tmp_path, monkeypatch, capsys):
    s = _sources(tmp_path)
    ex = Exllamav3(s).install(monkeypatch)
    assert _run(s, "--resume") == convert.EXIT_REFUSED
    assert "--resume takes exllamav3's work directory of an interrupted job" in capsys.readouterr().out
    _fill(s["work"])
    assert _run(s, "--resume") == 0
    assert ex.argv[-1] == "-r"
    assert _read(s["out"] + ".pins.json")["provenance"]["run"]["resumed"] is True


@pytest.mark.parametrize("change,message", [
    (lambda out: _rewrite_shard(out, L1 + "mlp.down_proj", k2=8), L1 + "mlp.down_proj: K 4, the recipe's is 3"),
    (lambda out: _rewrite_shard(out, L1 + "linear_attn.in_proj_z", drop=True), L1 + "linear_attn.in_proj_z: missing"),
    (lambda out: _rewrite_shard(out, "mtp.fc", k2=8), "mtp.fc: K 4, the MTP bits are 3"),
    (lambda out: _edit_json(os.path.join(out, "quantization_config.json"), head_bits=6), "head_bits = 6, not 4"),
    (lambda out: _edit_json(os.path.join(out, "quantization_config.json"), calibration={"rows": 100, "cols": 2048}),
     "calibration = {'rows': 100, 'cols': 2048}, the recipe's is {'rows': 250, 'cols': 2048}"),
])
def test_a_quant_that_is_not_the_recipes_is_refused_naming_the_matrix_and_gets_no_pins(
        tmp_path, monkeypatch, capsys, change, message):
    s = _sources(tmp_path)
    Exllamav3(s, change=change).install(monkeypatch)
    assert _run(s) == convert.EXIT_REFUSED
    assert message in capsys.readouterr().out
    assert not os.path.exists(s["out"] + ".pins.json")


def _edit_json(path, **fields):
    value = _read(path)
    value.update(fields)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f)


def _control(s, out, errors_fn):
    return local_quant.main(["control", "--quant", s["out"], "--bf16", s["swift"], "--bf16-repo", SWIFT_REPO,
                             "--record", s["record"], "--out", out], errors_fn=errors_fn)


def test_the_control_puts_each_matrixs_error_against_the_fine_tune_beside_the_defaults(tmp_path, monkeypatch):
    s = _sources(tmp_path)
    Exllamav3(s).install(monkeypatch)
    assert _run(s) == 0
    measured = []

    def errors(quant, bf16_root, planned, log=print):
        measured.append((quant.root, bf16_root, [(p.name, p.key, p.format, p.shape) for p in planned]))
        return {p.name: 0.3 for p in planned} | {"text/layers/1/mlp/down": 0.25}

    out = str(tmp_path / "scratch" / "471" / "control.json")
    assert _control(s, out, errors) == 0
    # every matrix of the quant, under the default artifact's object name, measured against the fine-tune's BF16
    (quant_root, bf16_root, planned), = measured
    assert (quant_root, bf16_root) == (s["out"], s["swift"])
    default = _read(s["record"])["trellis"]["matrices"]
    assert sorted(planned) == sorted((name, m["source"], m["format"], m["shape"]) for name, m in default.items())
    c = _read(out)
    assert c["matrices"]["text/layers/1/mlp/down"] == {
        "source": L1 + "mlp.down_proj", "K": 3, "rel_error": 0.25,
        "default_rel_error": s["default_errors"]["text/layers/1/mlp/down"]}
    assert c["matrices"]["text/output_head"]["K"] == 4
    assert len(c["matrices"]) == 26
    assert c["reference"] == {"repo": SWIFT_REPO, "revision": SWIFT_REVISION}
    assert c["default"]["record"] == {"file": os.path.basename(s["record"]), "sha256": _sha256(s["record"])}
    assert c["default"]["reference"] == {"repo": "Qwen/Qwen3.8-27B", "revision": "0123456789abcdef0123456789abcdef01234567"}
    assert c["quant"]["pins"] == {"file": os.path.basename(s["out"]) + ".pins.json",
                                  "sha256": _sha256(s["out"] + ".pins.json")}
    assert c["summary"]["rel_error"] == {"max": 0.3, "mean": round((25 * 0.3 + 0.25) / 26, 6)}
    assert c["summary"]["default_rel_error"]["max"] == max(s["default_errors"].values())


def test_a_control_whose_record_does_not_hold_the_quants_matrices_is_refused(tmp_path, monkeypatch, capsys):
    s = _sources(tmp_path)
    Exllamav3(s).install(monkeypatch)
    assert _run(s) == 0
    record = _read(s["record"])
    del record["trellis"]["matrices"]["text/layers/1/mlp/down"]
    record["trellis"]["matrices"]["text/layers/2/mlp/down"]["k2"] = 8
    with open(s["record"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f)
    out = str(tmp_path / "control.json")
    assert _control(s, out, lambda *a, **k: pytest.fail("measured")) == convert.EXIT_REFUSED
    printed = capsys.readouterr().out
    assert f"{L1}mlp.down_proj: not in the record" in printed
    assert "text/layers/2/mlp/down: K 3, the record's is 4" in printed
    assert not os.path.exists(out)
