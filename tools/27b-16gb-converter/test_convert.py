import hashlib
import json
import os

import pytest

import container
import convert
import exl3
from conftest import BF16_REVISION, REVISION, make_sources, write_container


def _errors(value=0.25):
    def fn(quant, bf16_root, objects, log=print):
        return {p.name: value for p in objects if p.key is not None}
    return fn


def _run(src, out, **kw):
    kw.setdefault("errors_fn", _errors())
    return convert.convert(src["served"], src["quant"], src["bf16"], out, log=lambda _: None, **kw)


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _never(*args, **kw):
    raise AssertionError("the GPU step ran before a refusal that needs none")


def pack(work, artifact, as_resource=(), identity=convert.IDENTITY):
    """What the packer does with a work tree, for `verify`'s tests: the units in order, frontend
    files by DONE's (sorted) names, tensors in tensors.json order; the tensors named in
    `as_resource` written as resources instead (a broken artifact)."""
    objects = []
    for unit in ("frontend", "text", "mtp", "vision", "dflash2"):
        d = os.path.join(work, unit)
        if unit == "frontend":
            for name in sorted(_json(os.path.join(d, "DONE"))["files"]):
                with open(os.path.join(d, name), "rb") as f:
                    objects.append(({"name": f"frontend/{name}", "kind": "resource", "encoding": "raw-bytes-v1"},
                                    f.read()))
            continue
        for t in _json(os.path.join(d, "tensors.json"))["tensors"]:
            if t["name"] in as_resource:
                desc = {"name": t["name"], "kind": "resource", "encoding": "raw-bytes-v1"}
            else:
                desc = {k: t[k] for k in ("name", "format", "layout", "shape")} | {"kind": "tensor"}
            with open(os.path.join(d, t["file"]), "rb") as f:
                objects.append((desc, f.read()))
    write_container(artifact, identity, objects)


def test_run_writes_a_work_tree_the_packer_takes(tmp_path):
    src = make_sources(str(tmp_path))
    work = str(tmp_path / "work")
    record = _run(src, work)
    assert sorted(os.listdir(work)) == ["converter.json", "dflash2", "frontend", "mtp", "text", "vision"]
    served = container.Container(src["served"])
    quant = exl3.Quant(src["quant"])
    for unit in ("frontend", "text", "mtp", "vision", "dflash2"):
        d = os.path.join(work, unit)
        done = _json(os.path.join(d, "DONE"))["files"]
        for name, entry in done.items():
            assert entry == {"bytes": os.path.getsize(os.path.join(d, name)), "sha256": _sha(os.path.join(d, name))}
        if unit == "frontend":
            assert sorted(done) == ["chat_template.jinja", "tokenizer.json"]
            continue
        for t in _json(os.path.join(d, "tensors.json"))["tensors"]:
            assert t["name"].startswith(unit + "/") and t["bytes"] == done[t["file"]]["bytes"]
            with open(os.path.join(d, t["file"]), "rb") as f:
                data = f.read()
            source = record["trellis"]["matrices"].get(t["name"])
            if source:
                assert t["layout"] == "trellis-tile16-v1"
                assert data == quant.record(quant.matrix(source["source"]))
            else:
                assert data == served.read(t["name"])
    assert record == _json(os.path.join(work, "converter.json"))


def test_converter_json_carries_the_provenance(tmp_path):
    src = make_sources(str(tmp_path))
    record = _run(src, str(tmp_path / "work"))
    assert record["status"] == "complete"
    assert record["identity"] == {"model_id": "qwen3.8-27b", "weights_id": "trellis-sc30"}
    s = record["source"]
    assert (s["repo"], s["branch"], s["revision"]) == ("turboderp/Qwen3.8-27B-exl3", "SC_3.00bpw_H4", REVISION)
    for name, pin in s["files"].items():
        assert pin == {"bytes": os.path.getsize(os.path.join(src["quant"], name)),
                       "sha256": _sha(os.path.join(src["quant"], name))}
    assert sorted(s["files"]) == ["config.json", "model-00001-of-00002.safetensors",
                                  "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                                  "quantization_config.json"]
    assert record["served"]["sha256"] == _sha(src["served"])
    ref = record["reference"]
    assert (ref["repo"], ref["revision"]) == ("Qwen/Qwen3.8-27B", BF16_REVISION)
    for name, pin in ref["files"].items():
        assert pin == {"bytes": os.path.getsize(os.path.join(src["bf16"], name)),
                       "sha256": _sha(os.path.join(src["bf16"], name))}
    assert sorted(ref["files"]) == ["model-00001-of-00001.safetensors", "model.safetensors.index.json"]
    assert record["graft"]["how"].startswith("the 2 dflash2/* objects")
    assert record["served"]["graft_sidecar"]["sha256"] == _sha(src["served"] + ".graft.json")
    assert record["graft"]["grafted_from"]["repository"] == "someone/base"
    # only the copied drafter's NVFP4 record survives: the body plane it named is gone
    assert list(record["local_nvfp4"]["parents"]) == ["dflash2/feature_projection"]
    m = record["trellis"]["matrices"]
    assert len(m) == 3 * 6 + 1 * 7 + 1 and record["objects"]["trellis"] == len(m)
    assert m["text/layers/3/attention/v_proj"]["k2"] == 10
    assert m["text/output_head"] == m["text/output_head"] | {"source": "lm_head", "k2": 8, "rel_error": 0.25}
    assert all(c["sha256"] for c in record["copied"]["objects"].values())
    assert record["quant_unused"] == ["mtp.fc"]
    assert all(n.endswith("input_scale_divisor") for n in record["dropped"])
    assert record["trellis"]["rel_error_max"] == 0.25
    assert sorted(record["converter"]) == ["commit", "dirty", "tool"]
    assert record["converter"]["tool"] == "tools/27b-16gb-converter"
    assert record["versions"]["safetensors"]


def test_the_same_inputs_give_the_same_work_tree(tmp_path):
    src = make_sources(str(tmp_path))
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run(src, a)
    _run(src, b)
    for unit in ("frontend", "text", "mtp", "vision", "dflash2"):
        assert _sha(os.path.join(a, unit, "DONE")) == _sha(os.path.join(b, unit, "DONE"))
    assert _sha(os.path.join(a, "converter.json")) == _sha(os.path.join(b, "converter.json"))


def test_a_reconstruction_error_above_the_bound_fails_the_conversion(tmp_path):
    src = make_sources(str(tmp_path))
    work = tmp_path / "work"
    with pytest.raises(convert.Refused, match="layout error"):
        _run(src, str(work), errors_fn=_errors(0.61))
    assert not (work / "converter.json").exists() and not work.exists()


def test_skip_errors_writes_a_dry_run(tmp_path):
    src = make_sources(str(tmp_path))
    record = _run(src, str(tmp_path / "work"), skip_errors=True, errors_fn=None)
    assert record["status"] == "dry-run"
    assert all(t["rel_error"] is None for t in record["trellis"]["matrices"].values())
    assert record["reference"] is None, "a dry run reads no BF16 checkpoint"


@pytest.mark.parametrize("root,shard", [("quant", "model-00002-of-00002.safetensors"),
                                        ("bf16", "model-00001-of-00001.safetensors")])
def test_a_shard_that_is_not_its_lfs_oid_is_refused(tmp_path, root, shard):
    src = make_sources(str(tmp_path))
    path = os.path.join(src[root], shard)
    with open(path, "r+b") as f:
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 1]))
    with pytest.raises(convert.Refused, match="LFS oid"):
        _run(src, str(tmp_path / "work"))


@pytest.mark.parametrize("root,name,content,message", [
    ("quant", "config.json", None, "no Hugging Face download record"),
    ("quant", "model-00001-of-00002.safetensors", "86b95530f9d365d90dd24d01a997a8de0cdf6c55", "not a Hugging Face"),
    ("bf16", "model.safetensors.index.json", "f" * 40 + "\n" + "0" * 40, "different revisions"),
])
def test_a_source_that_cannot_be_pinned_is_refused_before_the_gpu_step(tmp_path, root, name, content, message):
    src = make_sources(str(tmp_path))
    record = os.path.join(src[root], ".cache", "huggingface", "download", name + ".metadata")
    if content is None:
        os.remove(record)
    else:
        with open(record, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    with pytest.raises(convert.Refused, match=message):
        _run(src, str(tmp_path / "work"), errors_fn=_never)


def test_a_nonempty_work_dir_is_refused(tmp_path):
    src = make_sources(str(tmp_path))
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "x").write_text("x")
    with pytest.raises(convert.Refused, match="not empty"):
        _run(src, str(tmp_path / "work"))


def test_another_quant_recipe_is_refused(tmp_path):
    src = make_sources(str(tmp_path))
    path = os.path.join(src["quant"], "quantization_config.json")
    config = _json(path) | {"codebook": "mcg"}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f)
    with pytest.raises(convert.Refused, match="codebook"):
        _run(src, str(tmp_path / "work"))


def test_verify_passes_the_packed_artifact_and_names_a_changed_byte(tmp_path):
    src = make_sources(str(tmp_path))
    work = str(tmp_path / "work")
    _run(src, work)
    artifact = str(tmp_path / "a.ninfer")
    pack(work, artifact)
    ok, report = convert.verify(artifact, src["served"], src["quant"], log=lambda _: None)
    assert ok, report["problems"]
    assert report["checked"] == {"trellis": 26, "copied": len(container.Container(artifact).objects) - 26}

    art = container.Container(artifact)
    for name in ("text/layers/2/mlp/gate_proj", "vision/patch_embedding"):
        offset, _ = art.span(name)
        with open(artifact, "r+b") as f:
            f.seek(offset + 3)
            b = f.read(1)
            f.seek(offset + 3)
            f.write(bytes([b[0] ^ 0x40]))
    ok, report = convert.verify(artifact, src["served"], src["quant"], log=lambda _: None)
    assert not ok
    assert any(p.startswith("text/layers/2/mlp/gate_proj: not") for p in report["problems"])
    assert any(p.startswith("vision/patch_embedding: sha256") for p in report["problems"])


def test_verify_names_an_object_of_the_wrong_kind(tmp_path):
    src = make_sources(str(tmp_path))
    work = str(tmp_path / "work")
    _run(src, work)
    artifact = str(tmp_path / "a.ninfer")
    pack(work, artifact, as_resource={"text/layers/0/mlp/down"})
    ok, report = convert.verify(artifact, src["served"], src["quant"], log=lambda _: None)
    assert not ok
    assert any(p.startswith("text/layers/0/mlp/down: None None None, not TRELLIS_MUL1_K3") for p in report["problems"])


# --- the override (#420) ---

import argparse  # noqa: E402

import override as override_mod  # noqa: E402
from conftest import OVERRIDE_REPO, OVERRIDE_REVISION, PREFIX, REVISION as QUANT_REVISION, make_override  # noqa: E402

OV_NAMES = ["text/layers/2/mlp/gate_proj", "text/layers/3/attention/v_proj"]
OV_FILES = ["layers.2.mlp.gate_proj.bin", "layers.3.attention.v_proj.bin"]


def _override(ov):
    return override_mod.Override(ov["override"], ov["bf16"])


def _ov_errors(quant, bf16_root, objects, log=print, override=None):
    return {p.name: 0.25 for p in objects if p.key is not None}


def _text_file(work, name):
    entry = next(t for t in _json(os.path.join(work, "text", "tensors.json"))["tensors"] if t["name"] == name)
    with open(os.path.join(work, "text", entry["file"]), "rb") as f:
        return f.read()


def test_an_overrides_records_land_and_every_other_record_is_the_quants(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    base, swapped = str(tmp_path / "base"), str(tmp_path / "swapped")
    _run(src, base)
    record = _run(src, swapped, override=_override(ov), errors_fn=_ov_errors)
    over = exl3.Quant(ov["override"])
    for key, name in zip(ov["keys"], OV_NAMES):
        assert _text_file(swapped, name) == over.record(over.matrix(key))
        assert _text_file(swapped, name) != _text_file(base, name)
    # every other file of every unit is the base conversion's, bit for bit
    for unit in ("frontend", "text", "mtp", "vision", "dflash2"):
        a = _json(os.path.join(base, unit, "DONE"))["files"]
        b = _json(os.path.join(swapped, unit, "DONE"))["files"]
        assert sorted(a) == sorted(b)
        assert sorted(n for n in a if a[n] != b[n]) == (OV_FILES if unit == "text" else [])
    assert sorted(n for n, t in record["trellis"]["matrices"].items() if t.get("override")) == OV_NAMES


@pytest.mark.parametrize("records,message", [
    ({PREFIX + "9.mlp.gate_proj": (256, 128, 4)}, "layers.9.mlp.gate_proj: the quant has no record of that name"),
    ({"mtp.fc": (128, 256, 6)}, "mtp.fc: the quant has no record of that name"),
    ({PREFIX + "2.mlp.gate_proj": (256, 128, 6)}, r"layers.2.mlp.gate_proj: K 3, the quant's is 2\b"),
    ({PREFIX + "2.mlp.gate_proj": (384, 128, 4)},
     r"layers.2.mlp.gate_proj: shape \[384, 128\], the quant's is \[256, 128\]"),
])
def test_an_override_that_cannot_swap_like_for_like_is_refused(tmp_path, records, message):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src, records=records)
    work = tmp_path / "work"
    with pytest.raises(convert.Refused, match=message):
        _run(src, str(work), override=_override(ov), errors_fn=_never)
    assert not work.exists()


def test_the_conversion_record_names_the_override_and_its_reference(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    seen = {}

    def measure(quant, bf16_root, objects, log=print, override=None):
        seen["override"] = override
        # a distinct error for the override's matrices: they are measured against its own source
        return {p.name: (0.125 if override is not None and p.key in override.matrices else 0.25)
                for p in objects if p.key is not None}

    record = _run(src, str(tmp_path / "work"), override=_override(ov), errors_fn=measure,
                  weights_id="trellis-sc30-abliterated")
    o = record["override"]
    # the source repository and revision are the replay's, the files pinned by what they hold now
    assert (o["repo"], o["revision"]) == (OVERRIDE_REPO, OVERRIDE_REVISION)
    assert o["quant_revision"] == QUANT_REVISION
    assert o["matrices"] == OV_NAMES
    assert sorted(o["files"]) == ["model-00001-of-00001.safetensors", "model.safetensors.index.json",
                                  "quantization_config.json", "replay.json"]
    for name, pin in o["files"].items():
        assert pin == {"bytes": os.path.getsize(os.path.join(ov["override"], name)),
                       "sha256": _sha(os.path.join(ov["override"], name))}
    ref = o["reference"]
    assert (ref["repo"], ref["revision"]) == (OVERRIDE_REPO, OVERRIDE_REVISION)
    assert sorted(ref["files"]) == ["fetch.json", "model.safetensors.index.json"]
    # each tensor is pinned by its own sha256, its shard by the LFS oid of the whole file
    assert sorted(ref["tensors"]) == [k + ".weight" for k in ov["keys"]]
    assert all(t["shard"] == ov["ref_shard"] and len(t["sha256"]) == 64 for t in ref["tensors"].values())
    assert ref["shards"] == {ov["ref_shard"]: {"lfs_oid": "2" * 64}}
    # the step is handed the override, whose reference BF16 it reads the override matrices' from
    assert seen["override"].reference_root == ov["bf16"]
    m = record["trellis"]["matrices"]
    assert [m[n]["rel_error"] for n in OV_NAMES] == [0.125, 0.125]
    assert m["text/layers/0/mlp/down"]["rel_error"] == 0.25
    assert record["reference"]["repo"] == "Qwen/Qwen3.8-27B"
    assert record["identity"] == {"model_id": "qwen3.8-27b", "weights_id": "trellis-sc30-abliterated"}


def test_a_dry_run_with_an_override_pins_its_reference_and_measures_nothing(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    record = _run(src, str(tmp_path / "work"), override=_override(ov), skip_errors=True, errors_fn=None)
    assert record["override"]["revision"] == OVERRIDE_REVISION
    assert sorted(record["override"]["reference"]["tensors"]) == [k + ".weight" for k in ov["keys"]]
    assert record["reference"] is None


def _edit_json(path, change):
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    change(value)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f)


def _refused(tmp_path, src, ov, message, **kw):
    work = tmp_path / "work"
    with pytest.raises(convert.Refused, match=message):
        _run(src, str(work), override=_override(ov), errors_fn=_never, **kw)
    assert not work.exists()


def test_an_override_with_no_provenance_is_refused_naming_what_is_missing(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    os.remove(os.path.join(ov["override"], "replay.json"))
    a = argparse.Namespace(override=ov["override"], override_bf16=ov["bf16"])
    with pytest.raises(convert.Refused, match="has no replay.json: the override's source repository and revision"):
        convert.override_of(a)


def test_both_commands_refuse_an_override_they_cannot_load_alike(tmp_path, capsys):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    os.remove(os.path.join(ov["override"], "replay.json"))
    commands = {
        "run": ["run", "--served", src["served"], "--quant", src["quant"], "--bf16", src["bf16"],
                "--work", str(tmp_path / "work"), "--override", ov["override"], "--override-bf16", ov["bf16"]],
        "verify": ["verify", "--artifact", str(tmp_path / "a.ninfer"), "--served", src["served"],
                   "--quant", src["quant"], "--override", ov["override"]],
    }
    for name, argv in commands.items():
        assert convert.main(argv) == convert.EXIT_REFUSED, name
        said = capsys.readouterr().out
        assert said.startswith(f"REFUSED: the override: {ov['override']} has no replay.json"), (name, said)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update(status="partial"), "status complete .it is 'partial'."),
    (lambda r: r["sources"]["abliterated"].pop("revision"), "sources.abliterated.revision"),
    (lambda r: r["sources"].pop("quant"), "sources.quant.revision"),
    (lambda r: r["matrices"][next(iter(r["matrices"]))].pop("record_sha256"), r"matrices\.\*\.record_sha256"),
])
def test_a_provenance_missing_what_the_pin_needs_is_refused(tmp_path, change, message):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    _edit_json(os.path.join(ov["override"], "replay.json"), change)
    with pytest.raises(ValueError, match="replay.json lacks .*" + message):
        _override(ov)


def test_a_record_that_is_not_the_one_the_replay_made_is_refused(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    _edit_json(os.path.join(ov["override"], "replay.json"),
               lambda r: r["matrices"][ov["keys"][0]].update(record_sha256="0" * 64))
    _refused(tmp_path, src, ov, r"layers.2.mlp.gate_proj: the record's sha256 is not replay.json's")


def test_an_override_made_against_another_quant_revision_is_refused(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    _edit_json(os.path.join(ov["override"], "replay.json"),
               lambda r: r["sources"]["quant"].update(revision="b" * 40))
    _refused(tmp_path, src, ov, "made against the quant at b{40}")


def test_a_tampered_tensor_in_the_partial_reference_is_refused(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    shard = os.path.join(ov["bf16"], ov["ref_shard"])
    with open(shard, "r+b") as f:
        f.seek(-1, os.SEEK_END)          # the last byte of the last tensor
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 1]))
    for skip in (False, True):           # a dry run checks it too: it needs no GPU
        _refused(tmp_path, src, ov, "layers.3.self_attn.v_proj.weight: hashes to",
                 skip_errors=skip)


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update(revision="c" * 40), "was made from someone/abliterated@fedcba"),
    (lambda r: r["tensors"].pop(next(iter(r["tensors"]))), "weight: not in the reference"),
    (lambda r: r["tensors"][next(iter(r["tensors"]))].update(sha256="0" * 64), "fetch.json's sha256 is not replay.json's"),
])
def test_a_reference_that_is_not_the_one_the_replay_used_is_refused(tmp_path, change, message):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    _edit_json(os.path.join(ov["bf16"], "fetch.json"), change)
    _refused(tmp_path, src, ov, message)


def test_a_reference_with_no_fetch_record_is_refused(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    os.remove(os.path.join(ov["bf16"], "fetch.json"))
    _refused(tmp_path, src, ov, "has no fetch.json: its tensors cannot be pinned")


def test_without_an_override_the_record_is_as_before(tmp_path):
    src = make_sources(str(tmp_path))
    record = _run(src, str(tmp_path / "work"))
    assert "override" not in record
    assert not any("override" in t for t in record["trellis"]["matrices"].values())
    assert record["identity"] == convert.IDENTITY


def test_verify_passes_an_artifact_packed_with_the_override_and_names_a_differing_record(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    work = str(tmp_path / "work")
    _run(src, work, override=_override(ov), errors_fn=_ov_errors)
    artifact = str(tmp_path / "a.ninfer")
    pack(work, artifact)
    kw = {"log": lambda _: None, "override": _override(ov)}
    ok, report = convert.verify(artifact, src["served"], src["quant"], **kw)
    assert ok, report["problems"]
    assert report["checked"]["trellis"] == 26
    # without the override the same artifact fails on exactly the swapped records
    ok, report = convert.verify(artifact, src["served"], src["quant"], log=lambda _: None)
    assert not ok and sorted(p.split(":")[0] for p in report["problems"]) == OV_NAMES
    # an artifact holding the quant's record where the override has its own: the record is named
    base = str(tmp_path / "base")
    _run(src, base)
    pack(base, str(tmp_path / "b.ninfer"))
    ok, report = convert.verify(str(tmp_path / "b.ninfer"), src["served"], src["quant"], **kw)
    assert not ok
    assert any(p.startswith(OV_NAMES[0] + ": not") and "override" in p for p in report["problems"])


def test_verify_refuses_an_override_that_cannot_swap(tmp_path):
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src, records={PREFIX + "2.mlp.gate_proj": (256, 128, 6)})
    work = str(tmp_path / "work")
    _run(src, work)
    pack(work, str(tmp_path / "a.ninfer"))
    ok, report = convert.verify(str(tmp_path / "a.ninfer"), src["served"], src["quant"], log=lambda _: None,
                                override=_override(ov))
    assert not ok and "layers.2.mlp.gate_proj: K 3" in report["problems"][0]


def test_verify_takes_the_variants_weights_id(tmp_path):
    src = make_sources(str(tmp_path))
    work = str(tmp_path / "work")
    _run(src, work)
    pack(work, str(tmp_path / "a.ninfer"))
    ok, report = convert.verify(str(tmp_path / "a.ninfer"), src["served"], src["quant"], log=lambda _: None,
                                weights_id="trellis-sc30-abliterated")
    assert not ok and "identity" in report["problems"][0]


def test_an_override_given_by_halves_is_refused():
    a = argparse.Namespace(override="x", override_bf16=None)
    with pytest.raises(convert.Refused, match="--override takes --override-bf16 too"):
        convert.override_of(a)
    a = argparse.Namespace(override=None, override_bf16="x")
    with pytest.raises(convert.Refused, match="--override takes --override too"):
        convert.override_of(a)


# --- the variant against the artifact it varies (#422) ---

VARIANT = "trellis-sc30-abliterated"


def _pair(tmp_path):
    """The default artifact and the variant packed with the override and its own weights id."""
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    base, variant = str(tmp_path / "base.ninfer"), str(tmp_path / "variant.ninfer")
    _run(src, str(tmp_path / "base"))
    pack(str(tmp_path / "base"), base)
    _run(src, str(tmp_path / "variant"), override=_override(ov), errors_fn=_ov_errors, weights_id=VARIANT)
    pack(str(tmp_path / "variant"), variant, identity=convert.identity_of(VARIANT))
    return src, ov, base, variant


def _verify_variant(src, ov, variant, baseline):
    return convert.verify(variant, src["served"], src["quant"], log=lambda _: None, override=_override(ov),
                          weights_id=VARIANT, baseline=baseline)


def test_verify_against_the_baseline_finds_only_the_overrides_records_differ(tmp_path):
    src, ov, base, variant = _pair(tmp_path)
    ok, report = _verify_variant(src, ov, variant, base)
    assert ok, report["problems"]
    b = report["baseline"]
    assert b["artifact"] == "base.ninfer" and b["identity"] == convert.IDENTITY
    assert b["differ"] == OV_NAMES
    assert b["same"] == len(container.Container(variant).objects) - len(OV_NAMES)


def _flip(path, name):
    offset, _ = container.Container(path).span(name)
    with open(path, "r+b") as f:
        f.seek(offset + 3)
        b = f.read(1)
        f.seek(offset + 3)
        f.write(bytes([b[0] ^ 0x40]))


def test_verify_names_a_record_outside_the_override_that_is_not_the_baselines(tmp_path):
    src, ov, base, variant = _pair(tmp_path)
    _flip(base, "vision/patch_embedding")
    ok, report = _verify_variant(src, ov, variant, base)
    assert not ok
    assert report["problems"] == ["vision/patch_embedding: not the baseline's, bit for bit"]


def test_verify_names_an_override_record_the_baseline_already_holds(tmp_path):
    src, ov, _, variant = _pair(tmp_path)
    # a baseline packed from the variant's own work tree under the default identity
    same = str(tmp_path / "same.ninfer")
    work = str(tmp_path / "variant-again")
    _run(src, work, override=_override(ov), errors_fn=_ov_errors, weights_id=VARIANT)
    pack(work, same)
    ok, report = _verify_variant(src, ov, variant, same)
    assert not ok
    assert report["problems"] == [f"{n}: the override's record is the baseline's, bit for bit" for n in OV_NAMES]


def test_verify_refuses_a_baseline_with_the_artifacts_identity_or_another_directory(tmp_path):
    src, ov, base, variant = _pair(tmp_path)
    ok, report = _verify_variant(src, ov, variant, variant)
    assert not ok
    assert report["problems"][0] == ("the baseline's identity is the artifact's, qwen3.8-27b/trellis-sc30-abliterated: "
                                     "the two would share a content hash")
    # the default work tree packed with one tensor written as a resource: another directory
    other = str(tmp_path / "other.ninfer")
    _run(src, str(tmp_path / "base-again"))
    pack(str(tmp_path / "base-again"), other, as_resource={"text/layers/0/mlp/down"})
    ok, report = _verify_variant(src, ov, variant, other)
    assert not ok
    assert report["problems"][0].startswith("text/layers/0/mlp/down: the baseline's descriptor is ")


def test_the_verify_command_takes_a_baseline(tmp_path):
    src, ov, base, variant = _pair(tmp_path)
    out = str(tmp_path / "report.json")
    code = convert.main(["verify", "--artifact", variant, "--served", src["served"], "--quant", src["quant"],
                         "--override", ov["override"], "--weights-id", VARIANT, "--baseline", base, "--report", out])
    assert code == 0
    assert _json(out)["baseline"]["differ"] == OV_NAMES
