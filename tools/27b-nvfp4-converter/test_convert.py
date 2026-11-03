"""The converter's commands on the reduced synthetic checkpoint (conftest.make_sources), on the CPU."""
import json
import os
import shutil
import struct

import pytest
import torch

import calibrate
import checkpoint
import container
import convert
import inventory
import recipe
from conftest import (ALL_FILES, ARTIFACT, G, LICENSE, REPO, REVISION, WEIGHTS_ID, bf16_parent, calibrate_args,
                      checkpoint_tensors, corrupt, decode_nvfp4, drafter_config_json, make_checkpoint, make_drafter,
                      run_args)


def test_the_calibration_is_the_whole_models_forward_measured_one_layer_at_a_time(built):
    """Every site's amax is what the text model, loaded whole, sees over the corpus."""
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5Model

    with open(built["calibration"], encoding="utf-8") as f:
        record = json.load(f)
    sites = record["measured_sites"]
    assert len(sites) == 247
    assert record["corpus"]["documents"] == 10

    calibrate.gdn_reference()
    config = AutoConfig.from_pretrained(built["checkpoint"], dtype=torch.bfloat16)
    model = Qwen3_5Model(config).eval()
    tensors = checkpoint_tensors(built["checkpoint"])
    state = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    assert not missing and not unexpected
    modules = dict(model.named_modules())
    seen = {}
    for name, entry in sites.items():
        def hook(module, args, kwargs, name=name):
            seen[name] = max(seen.get(name, 0.0), args[0].abs().amax().item())
        modules[entry["module"]].register_forward_pre_hook(hook, with_kwargs=True)
    ids = AutoTokenizer.from_pretrained(built["checkpoint"])(calibrate.corpus())["input_ids"]
    assert record["corpus"]["tokens"] == len(ids)
    with torch.inference_mode():
        model(torch.tensor([ids]), use_cache=False)
    for name, entry in sites.items():
        assert entry["amax"] == seen[name], name
        assert entry["input_scale_divisor"] == struct.unpack("<f", struct.pack("<f", 2688.0 / seen[name]))[0], name


def read_tree(work_dir):
    """name -> (tensors.json entry or None for a resource, payload bytes), and converter.json."""
    out = {}
    for unit in inventory.UNITS:
        d = os.path.join(work_dir, unit)
        with open(os.path.join(d, "DONE"), encoding="utf-8") as f:
            done = json.load(f)["files"]
        if unit == "frontend":
            for name in done:
                with open(os.path.join(d, name), "rb") as f:
                    out[f"frontend/{name}"] = (None, f.read())
            continue
        with open(os.path.join(d, "tensors.json"), encoding="utf-8") as f:
            entries = json.load(f)["tensors"]
        assert set(done) == {e["file"] for e in entries} | {"tensors.json"}
        for e in entries:
            with open(os.path.join(d, e["file"]), "rb") as f:
                out[e["name"]] = (e, f.read())
    with open(os.path.join(work_dir, "converter.json"), encoding="utf-8") as f:
        return out, json.load(f)


def test_the_work_tree_holds_the_inventory_under_the_weights_id(built):
    tree, record = read_tree(built["work"])
    expected = inventory.objects(G)
    assert set(tree) == {o.name for o in expected}
    for o in expected:
        entry, payload = tree[o.name]
        if o.kind == "tensor":
            assert (entry["format"], entry["layout"], tuple(entry["shape"]), entry["bytes"]) == \
                (o.format, o.layout, o.shape, len(payload)), o.name
    assert record["status"] == "complete"
    assert record["identity"] == {"model_id": "qwen3.8-27b", "weights_id": WEIGHTS_ID}
    assert record["recipe_id"] == "qwen3_8_27b_nvfp4full-v2-local"
    assert {k: record["source"][k] for k in ("repo", "revision", "license")} == \
        {"repo": REPO, "revision": REVISION, "license": LICENSE}
    with open(os.path.join(built["checkpoint"], "tokenizer.json"), "rb") as f:
        assert tree["frontend/tokenizer.json"][1] == f.read()
    assert built["checkpoint"] not in json.dumps(record) and built["drafter"] not in json.dumps(record)


def test_every_nvfp4_parent_is_encoded_here_beside_its_calibrated_divisor(built):
    tree, record = read_tree(built["work"])
    with open(built["calibration"], encoding="utf-8") as f:
        sites = json.load(f)["measured_sites"]
    tensors = checkpoint_tensors(built["checkpoint"])
    parents = record["local_nvfp4"]["parents"]
    nvfp4 = [o for o in inventory.objects(G) if o.format == "NVFP4"]
    assert set(parents) == {o.name for o in nvfp4} and len(parents) == 281
    assert record["recipe"]["local_nvfp4"] == 247 and record["recipe"]["source_nvfp4_parents"] == 0
    for o in nvfp4:
        values, divisor = decode_nvfp4(tree[o.name][1], *o.shape)
        if o.unit == "dflash2":
            assert parents[o.name]["weight_scale_divisor"] is None
            continue
        parent = bf16_parent(tensors, o.name).float()
        assert divisor == struct.unpack("<f", struct.pack("<f", 2688.0 / parent.abs().max().item()))[0], o.name
        assert parents[o.name]["weight_scale_divisor"] == divisor
        assert ((values - parent).norm() / parent.norm()).item() < 0.15, o.name
        site = inventory.divisor_name(o.name)
        assert tree[site][1] == struct.pack("<f", sites[site]["input_scale_divisor"]), site


def test_the_packed_container_carries_a_sidecar_the_loader_reads(packed):
    """The packer took the tree (it checks every work file against the inventory and every recorded
    divisor against its file's trailing word, `docs/formats/27b-nvfp4-work-tree.md` §7) and wrote the
    fields the loader's checksum verification reads (`crates/artifact/src/checksum.rs`)."""
    with open(packed + ".conversion.json", encoding="utf-8") as f:
        sidecar = json.load(f)
    assert sidecar["recipe_id"] == "qwen3_8_27b_nvfp4full-v2-local"
    assert sidecar["artifact"]["bytes"] == os.path.getsize(packed)
    assert sidecar["objects"]["count"] == 1325
    assert len(sidecar["local_nvfp4"]["parents"]) == 281
    found = container.Container(packed)
    assert found.identity == {"model_id": "qwen3.8-27b", "weights_id": WEIGHTS_ID}


def verify_args(src, artifact, weights_id=WEIGHTS_ID):
    return ["verify", "--artifact", artifact, "--checkpoint", src["checkpoint"], "--dflash2", src["drafter"],
            "--ranking", src["ranking"], "--calibration", src["calibration"], "--weights-id", weights_id,
            "--device", "cpu", "--geometry", "fixture"]


def test_verify_passes_on_the_packed_artifact(built, packed, capsys, tmp_path):
    report = str(tmp_path / "report.json")
    assert convert.main(verify_args(built, packed) + ["--report", report]) == 0
    assert capsys.readouterr().out.strip().endswith("VERIFY PASS")
    with open(report, encoding="utf-8") as f:
        checked = json.load(f)["checked"]
    assert checked["objects"] == 1325 and checked["local_nvfp4"] == 247 and checked["input_divisor"] == 247


def test_verify_fails_naming_a_parent_that_does_not_re_encode(built, packed, capsys, tmp_path):
    copy = corrupt(packed, tmp_path, "text/layers/9/mlp/down")
    assert convert.main(verify_args(built, copy)) == convert.EXIT_VERIFY_FAILED
    out = capsys.readouterr().out
    assert out.strip().endswith("VERIFY FAILED")
    assert "text/layers/9/mlp/down: does not re-encode byte-equal from its source (1 of" in out


def test_verify_fails_naming_a_divisor_that_is_not_the_calibrations(built, packed, capsys, tmp_path):
    name = "text/layers/30/gdn/input_projection/input_scale_divisor"
    copy = corrupt(packed, tmp_path, name, at=1)
    assert convert.main(verify_args(built, copy)) == convert.EXIT_VERIFY_FAILED
    out = capsys.readouterr().out
    assert f"{name}: stores divisor" in out and "the calibration's is" in out


def test_verify_fails_on_another_weights_id(built, packed, capsys):
    assert convert.main(verify_args(built, packed, weights_id="nvfp4full")) == convert.EXIT_VERIFY_FAILED
    assert "the identity is" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------
# Refusals: each names what failed, and nothing is written
# ---------------------------------------------------------------------------------------------------

def refused(capsys, argv, work_dir, **kwargs):
    """Runs `argv`, which must be refused before anything is written; returns the message."""
    assert convert.main(argv, **kwargs) == convert.EXIT_REFUSED
    assert not os.path.exists(work_dir)
    return capsys.readouterr().out


def copy_sources(built, tmp_path, *names):
    out = dict(built)
    for name in names:
        out[name] = str(tmp_path / name)
        shutil.copytree(built[name], out[name])
    return out


def rewrite_record(root, name, revision):
    path = os.path.join(root, ".cache", "huggingface", "download", name + ".metadata")
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    lines[0] = revision
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def test_a_non_empty_work_tree_is_refused(built, capsys, tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "left.bin").write_bytes(b"x")
    assert convert.main(run_args(built, built["calibration"], str(work_dir))) == convert.EXIT_REFUSED
    assert "is not empty" in capsys.readouterr().out
    assert os.listdir(work_dir) == ["left.bin"]


def test_the_served_artifacts_weights_id_is_refused(built, capsys, tmp_path):
    out = refused(capsys, run_args(built, built["calibration"], str(tmp_path / "work"), weights_id="nvfp4full"),
                  str(tmp_path / "work"))
    assert "--weights-id nvfp4full is the served artifact's" in out


def tensors_but(change):
    tensors = dict(checkpoint.checkpoint_tensors(G))
    change(tensors)
    return tensors


def test_a_checkpoint_without_a_tensor_of_the_27b_is_refused(built, capsys, tmp_path):
    name = "model.language_model.layers.17.linear_attn.in_proj_z.weight"
    src = dict(built, checkpoint=make_checkpoint(str(tmp_path / "ckpt"), tensors=tensors_but(lambda t: t.pop(name))))
    out = refused(capsys, run_args(src, built["calibration"], str(tmp_path / "work")), str(tmp_path / "work"))
    assert f"the checkpoint has no {name}" in out


def test_a_checkpoint_whose_tensor_has_another_shape_is_refused(built, capsys, tmp_path):
    name = "model.language_model.layers.3.self_attn.k_proj.weight"
    src = dict(built, checkpoint=make_checkpoint(str(tmp_path / "ckpt"),
                                                 tensors=tensors_but(lambda t: t.update({name: (32, G.hidden)}))))
    out = refused(capsys, run_args(src, built["calibration"], str(tmp_path / "work")), str(tmp_path / "work"))
    assert f"the checkpoint's {name} is [32, 128], the 27B's is [64, 128]" in out


def test_a_shard_that_is_not_its_lfs_oid_is_refused(built, capsys, tmp_path):
    src = copy_sources(built, tmp_path, "checkpoint")
    shard = os.path.join(src["checkpoint"], "model-00002-of-00003.safetensors")
    with open(shard, "r+b") as f:
        f.seek(-3, os.SEEK_END)
        f.write(b"\x00\x01\x02")
    out = refused(capsys, run_args(src, built["calibration"], str(tmp_path / "work")), str(tmp_path / "work"))
    assert "the checkpoint's model-00002-of-00003.safetensors hashes to" in out and "its LFS oid is" in out


def test_files_from_two_revisions_are_refused(built, capsys, tmp_path):
    src = copy_sources(built, tmp_path, "checkpoint")
    rewrite_record(src["checkpoint"], "model-00003-of-00003.safetensors", "0" * 40)
    out = refused(capsys, run_args(src, built["calibration"], str(tmp_path / "work")), str(tmp_path / "work"))
    assert "the checkpoint's files come from two revisions" in out and "0" * 40 in out


def test_a_checkpoint_at_another_revision_than_the_one_named_is_refused(built, capsys, tmp_path):
    argv = run_args(built, built["calibration"], str(tmp_path / "work"))
    argv[argv.index("--source-revision") + 1] = "f" * 40
    out = refused(capsys, argv, str(tmp_path / "work"))
    assert f"the checkpoint was downloaded at revision {REVISION}, not {'f' * 40}" in out


def test_a_drafter_whose_configuration_differs_from_the_bases_is_refused(built, capsys, tmp_path):
    config = drafter_config_json()
    config["dflash_config"] = dict(config["dflash_config"], target_layer_ids=[1, 2, 3, 4, 5])
    src = dict(built, drafter=make_drafter(str(tmp_path / "drafter"), config=config))
    out = refused(capsys, run_args(src, built["calibration"], str(tmp_path / "work")), str(tmp_path / "work"))
    assert "the drafter's config.json is not the base's drafter's" in out
    assert "dflash_config.target_layer_ids is [1, 2, 3, 4, 5], not [5, 19, 33, 47, 61]" in out


def test_a_recipe_with_other_counts_is_refused(built, capsys, tmp_path, monkeypatch):
    """A recipe that takes one parent from a quantized source instead of encoding it here."""
    local_only = recipe.recipes

    def one_from_a_source(g):
        table = local_only(g)
        table["text/layers/0/mlp/down"] = recipe.Recipe("source_nvfp4")
        return table

    monkeypatch.setattr(recipe, "recipes", one_from_a_source)
    out = refused(capsys, run_args(built, built["calibration"], str(tmp_path / "work")), str(tmp_path / "work"))
    assert "the recipe is not the local-only one" in out
    assert "246 objects by route local_nvfp4, the local-only recipe has 247" in out
    assert "1 objects by route source_nvfp4, the local-only recipe has 0" in out


def test_a_calibration_of_another_checkpoint_is_refused(built, capsys, tmp_path):
    """A base calibration never feeds a fine-tune's build: the same revision, other shards."""
    src = dict(built, checkpoint=make_checkpoint(str(tmp_path / "ckpt"), seed=1))
    out = refused(capsys, run_args(src, built["calibration"], str(tmp_path / "work")), str(tmp_path / "work"))
    assert "the calibration was measured on another checkpoint" in out


def test_a_disk_without_room_for_the_work_tree_is_refused(built, capsys, tmp_path):
    out = refused(capsys, run_args(built, built["calibration"], str(tmp_path / "work")), str(tmp_path / "work"),
                  disk_free=1000)
    assert "the work tree takes 5712904 bytes" in out


# ---------------------------------------------------------------------------------------------------
# One shard open at a time
# ---------------------------------------------------------------------------------------------------




def test_no_two_shards_are_open_at_once_in_calibration(built, handles, tmp_path):
    assert convert.main(calibrate_args(built, str(tmp_path / "calibration.json"))) == 0
    assert handles["peak"] == 1 and handles["open"] == 0
    assert set(handles["opened"]) == ALL_FILES - {"model.safetensors"}
    # the layers' shards were opened again and again, never two at once
    assert len(handles["opened"]) > 3


def test_no_two_shards_are_open_at_once_in_conversion(built, handles, tmp_path):
    assert convert.main(run_args(built, built["calibration"], str(tmp_path / "work"))) == 0
    assert handles["peak"] == 1 and handles["open"] == 0
    assert set(handles["opened"]) == ALL_FILES


def test_no_two_shards_are_open_at_once_in_verify(built, packed, handles):
    assert convert.main(verify_args(built, packed)) == 0
    assert handles["peak"] == 1 and handles["open"] == 0
    assert set(handles["opened"]) == ALL_FILES
