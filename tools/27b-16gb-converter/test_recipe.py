import collections
import hashlib
import json

import pytest
import yaml

import convert
import recipe

P = "model.language_model.layers."


def test_the_16gb_artifacts_recipe_is_the_default_and_names_its_k_head_ceiling_and_pins():
    r = recipe.load(recipe.DEFAULT)
    assert (r["name"], r["weights_id"], r["ceiling_bytes"]) == ("sc30", "trellis-sc30", 9_830_000_000)
    # SC_3.00bpw_H4's allocation: K2 34, K3 297, K4 66, K5 4 over its 401 matrices, the head at K4 among them
    assert len(r["tensors"]) == 400 and all(k.startswith(P) for k in r["tensors"])
    assert collections.Counter(r["tensors"].values()) == {2: 34, 3: 297, 4: 65, 5: 4}
    assert r["quantization_config"] == convert.QUANT_CONFIG
    assert r["job"] == {"mtp_bits": 3, "calibration": {"rows": 250, "cols": 2048}}
    # the shipped artifact's record pins the quant's files; the calibration trace is its LFS oid
    assert r["files"]["quantization_config.json"] == {
        "bytes": 629042, "sha256": "e7c692d6330f098f6e5c4f002e88e686cbd7500df5ab5cbf48d44583b194133c"}
    assert r["files"]["model-00001-of-00002.safetensors"]["sha256"] == (
        "904c0498d25d8ed5833ac7b706a8d6a8f3de5fa9521a126008e073cdd05aab3c")
    assert r["files"]["cal_trace.safetensors"] == {
        "bytes": 4096088, "sha256": "a034bee4c7920b4ac2e62b9d881a9b118488c414a81cf89aa64bd9274c47d685"}
    assert sorted(r["files"]) == ["cal_trace.safetensors", "config.json", "model-00001-of-00002.safetensors",
                                  "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                                  "quantization_config.json"]


def test_the_public_2_5_bit_quants_recipe_names_its_k_config_ceiling_and_pins():
    r = recipe.load("tandem25")
    assert (r["name"], r["weights_id"], r["ceiling_bytes"], r["job"]) == ("tandem25", "trellis-tandem25", 8_304_000_000, None)
    # 0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-2.5 at a7fef26e: K2 192 and K3 208 over the body, the head at K4
    assert len(r["tensors"]) == 400 and all(k.startswith(P) for k in r["tensors"])
    assert collections.Counter(r["tensors"].values()) == {2: 192, 3: 208}
    assert r["quantization_config"] == {"quant_method": "exl3", "codebook": "mul1", "out_scales": "auto", "bits": 2.5,
                                        "head_bits": 4}
    # the shards are pinned by their LFS oid, as its download records give it
    assert r["files"]["model-00001-of-00002.safetensors"]["sha256"] == (
        "6f744805a09cf267d246fc2e4badf06d8f63acf89514ed155c27d04b997c71e1")
    assert r["files"]["model-00002-of-00002.safetensors"]["sha256"] == (
        "a83396393c028ae2e30321716f5272bce59664070927edc1412e8f49286142fb")
    assert sorted(r["files"]) == ["config.json", "model-00001-of-00002.safetensors",
                                  "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                                  "quantization_config.json"]


def test_the_12gb_classs_first_candidate_is_the_optimizers_recipe_converted_as_sc30s():
    r, sc30 = recipe.load("ig28a"), recipe.load(recipe.DEFAULT)
    assert (r["name"], r["weights_id"], r["ceiling_bytes"]) == ("ig28a", "trellis-ig28a", 9_200_000_000)
    # sc_optimize.py at 2.7942 bits, K 2 to 5: K2 72, K3 287, K4 35, K5 6 over the body, the head at K4
    assert sorted(r["tensors"]) == sorted(sc30["tensors"])
    assert collections.Counter(r["tensors"].values()) == {2: 72, 3: 287, 4: 35, 5: 6}
    # exllamav3 writes `bits` as the body's average rounded to two places
    assert r["quantization_config"] == dict(sc30["quantization_config"], bits=2.79)
    assert r["job"] == sc30["job"]
    assert r["files"] == {n: sc30["files"][n] for n in ("cal_trace.safetensors", "quantization_config.json")}


def test_the_12gb_classs_second_candidate_is_the_gguf_allocation_converted_as_sc30s():
    r, sc30 = recipe.load("ig28b"), recipe.load(recipe.DEFAULT)
    assert (r["name"], r["weights_id"], r["ceiling_bytes"]) == ("ig28b", "trellis-ig28b", 9_200_000_000)
    # gguf_recipe.py on Qwen3.8-27B-UD-Q2_K_XL.gguf at 4af51b35: K2 106, K3 226, K4 46, K5 22 over the body, the head at K4
    assert sorted(r["tensors"]) == sorted(sc30["tensors"])
    assert collections.Counter(r["tensors"].values()) == {2: 106, 3: 226, 4: 46, 5: 22}
    assert r["quantization_config"] == dict(sc30["quantization_config"], bits=2.79)
    assert r["job"] == sc30["job"]
    assert r["files"] == {n: sc30["files"][n] for n in ("cal_trace.safetensors", "quantization_config.json")}


def _named(tmp_path, change=None, name="mine"):
    """A recipe `name` in a recipes directory of its own: two body matrices, the head at 4, one pin."""
    data = {"schema": recipe.SCHEMA, "weights_id": "trellis-mine", "ceiling_bytes": 1 << 30,
            "quantization_config": {"quant_method": "exl3", "codebook": "mul1", "out_scales": "auto", "bits": 2.5,
                                    "head_bits": 4},
            "tensors": {P + "0.mlp.down_proj": 2, P + "0.mlp.up_proj": 3},
            "files": {"config.json": {"bytes": 12, "sha256": "ab" * 32}}}
    if change:
        change(data)
    (tmp_path / "recipes").mkdir(exist_ok=True)
    (tmp_path / "recipes" / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    return str(tmp_path / "recipes")


def test_a_recipe_is_named_data_its_k_ints_and_its_file_hashed(tmp_path):
    root = _named(tmp_path, lambda d: d["tensors"].update({P + "0.mlp.up_proj": 3.0}))
    r = recipe.load("mine", root)
    assert r["tensors"] == {P + "0.mlp.down_proj": 2, P + "0.mlp.up_proj": 3}
    assert type(r["tensors"][P + "0.mlp.up_proj"]) is int
    assert (r["name"], r["weights_id"], r["job"]) == ("mine", "trellis-mine", None)
    with open(tmp_path / "recipes" / "mine.json", "rb") as f:
        assert r["sha256"] == hashlib.sha256(f.read()).hexdigest()


@pytest.mark.parametrize("change,message", [
    (lambda d: d["tensors"].update({P + "0.mlp.down_proj": 1}), P + r"0\.mlp\.down_proj: K 1, outside 2 to 5"),
    (lambda d: d["tensors"].update({P + "0.mlp.down_proj": 6}), P + r"0\.mlp\.down_proj: K 6, outside 2 to 5"),
    (lambda d: d["tensors"].update({P + "0.mlp.down_proj": 2.5}),
     P + r"0\.mlp\.down_proj: K 2\.5, a fractional K: the load binds integer K 2 to 5"),
    (lambda d: d["quantization_config"].update(head_bits=6),
     r"the head at 6 bits \(quantization_config\.head_bits\): the load binds the output head at 4"),
    (lambda d: d["quantization_config"].pop("head_bits"), r"the head at None bits"),
    (lambda d: d["tensors"].update(lm_head=4), r"lm_head: not a matrix of the body"),
    (lambda d: d.update(ceiling_bytes=9.2e9), r"`ceiling_bytes` 9200000000\.0: not a number of bytes"),
    (lambda d: d["files"].update({"a.safetensors": {"bytes": 1}}), r"a\.safetensors: its pin is not bytes and a sha256"),
    (lambda d: d.pop("files"), r"no `files`"),
    (lambda d: d.update(celing_bytes=1), r"`celing_bytes`: not a field of a recipe"),
])
def test_a_recipe_the_load_cannot_bind_is_refused_naming_its_cause(tmp_path, change, message):
    with pytest.raises(convert.Refused, match=r"the recipe mine:\n(.|\n)*" + message):
        recipe.load("mine", _named(tmp_path, change))


def test_a_recipe_that_is_not_there_is_refused_naming_the_ones_that_are(tmp_path):
    root = _named(tmp_path)
    for name in ("sc31", "../mine"):
        with pytest.raises(convert.Refused, match=r"no recipe '.*' in .*recipes \(its recipes: mine\)"):
            recipe.load(name, root)


# A reduced 27B's matrices as a conversion record gives their shapes, and a recipe of them.
SHAPES = {P + "0.linear_attn.in_proj_qkv": [256, 128], P + "0.linear_attn.in_proj_z": [128, 128],
          P + "0.mlp.down_proj": [128, 256], P + "3.self_attn.v_proj": [128, 128], "lm_head": [512, 128]}
KS = {P + "0.linear_attn.in_proj_qkv": 3, P + "0.linear_attn.in_proj_z": 2, P + "0.mlp.down_proj": 5,
      P + "3.self_attn.v_proj": 4}


def _job(tmp_path, change=None):
    def recipe_of(d):
        d.update(tensors=dict(KS), quantization_config=dict(convert.QUANT_CONFIG),
                 job={"mtp_bits": 3, "calibration": {"rows": 250, "cols": 2048}})
        if change:
            change(d)
    return recipe.load("mine", _named(tmp_path, recipe_of))


def test_a_local_conversion_takes_every_matrix_at_the_recipes_k_and_the_records_shape(tmp_path):
    r = recipe.job(_job(tmp_path), SHAPES)
    assert r["matrices"] == {
        P + "0.linear_attn.in_proj_qkv": {"K": 3, "shape": [256, 128]},
        P + "0.linear_attn.in_proj_z": {"K": 2, "shape": [128, 128]},
        P + "0.mlp.down_proj": {"K": 5, "shape": [128, 256]},
        P + "3.self_attn.v_proj": {"K": 4, "shape": [128, 128]},
        "lm_head": {"K": 4, "shape": [512, 128]},
    }
    assert (r["head_bits"], r["mtp_bits"], r["bits"]) == (4, 3, 3.0)
    assert (r["codebook"], r["out_scales"], r["calibration"]) == ("mul1", "always", {"rows": 250, "cols": 2048})
    assert recipe.formats(r) == {"TRELLIS_MUL1_K2": 1, "TRELLIS_MUL1_K3": 1, "TRELLIS_MUL1_K4": 2,
                                 "TRELLIS_MUL1_K5": 1}


def test_exllamav3s_recipe_is_the_body_at_its_k_and_the_head_at_head_bits(tmp_path):
    ex = recipe.exllamav3(recipe.job(_job(tmp_path, lambda d: d["quantization_config"].update(bits=2.8)), SHAPES))
    # exllamav3 takes the head from head_bits: a `tensors` entry for it would be ignored, so there is none.
    assert ex == {"target_bpw": 2.8, "head_bits": 4, "tensors": KS}


def test_the_yaml_reads_back_as_exllamav3_reads_it(tmp_path):
    path = tmp_path / "recipe.yaml"
    ex = recipe.exllamav3(recipe.job(_job(tmp_path), SHAPES))
    recipe.write_yaml(str(path), ex)
    with open(path, encoding="utf-8") as f:
        back = yaml.safe_load(f)
    assert back == ex
    # convert_model.prepare takes integer rates 1-8 (16 unquantized) and 1.5 / 2.5 / 3.5 with mul1
    assert all(type(v) is int and 1 <= v <= 8 for v in back["tensors"].values())


@pytest.mark.parametrize("change,shapes,message", [
    (lambda d: d.update(job=None), SHAPES, r"the recipe mine states no exllamav3 job"),
    (lambda d: d["quantization_config"].pop("codebook"), SHAPES, r"the recipe mine's quantization_config states no codebook"),
    (None, dict(SHAPES, **{P + "1.mlp.down_proj": [128, 256]}), P + r"1\.mlp\.down_proj: in the record, not in the recipe"),
    (lambda d: d["tensors"].update({P + "9.mlp.down_proj": 3}), SHAPES, P + r"9\.mlp\.down_proj: in the recipe, not in the record"),
])
def test_a_recipe_that_is_not_converted_here_is_refused_naming_why(tmp_path, change, shapes, message):
    with pytest.raises(convert.Refused, match=message):
        recipe.job(_job(tmp_path, change), shapes)


def test_bits_per_weight_is_the_bodys_average_k_by_weights_the_head_apart():
    # (3 x 32,768 + 2 x 16,384 + 5 x 32,768) / 81,920 weights; the head's 4 bits count for nothing
    matrices = {P + "0.a": {"K": 3, "shape": [256, 128]}, P + "0.b": {"K": 2, "shape": [128, 128]},
                P + "0.c": {"K": 5, "shape": [128, 256]}, "lm_head": {"K": 4, "shape": [512, 128]}}
    assert recipe.bpw(matrices) == 3.6


def test_the_text_scopes_device_bytes_are_its_trellis_records_and_its_other_device_objects():
    # a K3 [256, 128] record: 12,288 bytes of trellis, 256 of suh and 512 of svh, padded to 16,384; beside it
    # one norm on the device, the token embedding and the draft head not, nor anything outside text/
    copied = {"text/final_norm": 256, "text/token_embedding": 9999, "text/draft_head": 7777,
              "text/draft_head_token_ids": 5, "vision/patch_embedding": 1000}
    assert recipe.text_scope_bytes({P + "0.a": {"K": 3, "shape": [256, 128]}}, copied) == 16384 + 256
