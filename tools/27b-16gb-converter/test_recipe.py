import pytest
import yaml

import convert
import recipe

P = "model.language_model.layers."


def _storage(key, rows, columns, words, bits=None):
    """A tensor_storage entry as exllamav3's compile writes it: `words` is the trellis's last dimension, 16·K."""
    return {"stored_tensors": {f"{key}.trellis": {"shape": [columns // 16, rows // 16, words], "dtype": "torch.int16"},
                               f"{key}.suh": {"shape": [columns]}, f"{key}.svh": {"shape": [rows]},
                               f"{key}.mul1": {"shape": []}},
            "quant_format": "exl3",
            "bits_per_weight": (words // 16 if words % 16 == 0 else words / 16) if bits is None else bits,
            "mul1_multiplier": 2212286765}


def _config(**storage):
    """A synthetic quantization_config.json: two GDN layers and one attention layer of a reduced 27B, the head at 4."""
    tensors = {
        P + "0.linear_attn.in_proj_qkv": _storage(P + "0.linear_attn.in_proj_qkv", 256, 128, 48),
        P + "0.linear_attn.in_proj_z": _storage(P + "0.linear_attn.in_proj_z", 128, 128, 32),
        P + "0.linear_attn.out_proj": _storage(P + "0.linear_attn.out_proj", 128, 128, 64),
        P + "0.linear_attn.in_proj_a": {"stored_tensors": {P + "0.linear_attn.in_proj_a.weight": {"shape": [4, 128]}}},
        P + "0.input_layernorm": {"stored_tensors": {P + "0.input_layernorm.weight": {"shape": [128]}}},
        P + "0.mlp.down_proj": _storage(P + "0.mlp.down_proj", 128, 256, 80),
        P + "3.self_attn.q_proj": _storage(P + "3.self_attn.q_proj", 256, 128, 48),
        P + "3.self_attn.v_proj": _storage(P + "3.self_attn.v_proj", 128, 128, 80),
        "model.language_model.embed_tokens": {"stored_tensors": {}},
        "lm_head": _storage("lm_head", 512, 128, 64),
    }
    tensors.update(storage)
    return {"quant_method": "exl3", "version": "1.4.2", "bits": 3.0, "head_bits": 4,
            "calibration": {"rows": 250, "cols": 2048}, "out_scales": "always", "codebook": "mul1", "mtp_bits": 3,
            "tensor_storage": tensors}


def test_every_matrix_takes_the_quants_k_and_shape_and_the_head_is_4():
    r = recipe.derive(_config())
    assert r["matrices"] == {
        P + "0.linear_attn.in_proj_qkv": {"K": 3, "shape": [256, 128]},
        P + "0.linear_attn.in_proj_z": {"K": 2, "shape": [128, 128]},
        P + "0.linear_attn.out_proj": {"K": 4, "shape": [128, 128]},
        P + "0.mlp.down_proj": {"K": 5, "shape": [128, 256]},
        P + "3.self_attn.q_proj": {"K": 3, "shape": [256, 128]},
        P + "3.self_attn.v_proj": {"K": 5, "shape": [128, 128]},
        "lm_head": {"K": 4, "shape": [512, 128]},
    }
    assert (r["head_bits"], r["mtp_bits"], r["bits"]) == (4, 3, 3.0)
    assert (r["codebook"], r["out_scales"], r["calibration"]) == ("mul1", "always", {"rows": 250, "cols": 2048})
    assert recipe.formats(r) == {"TRELLIS_MUL1_K2": 1, "TRELLIS_MUL1_K3": 2, "TRELLIS_MUL1_K4": 2,
                                 "TRELLIS_MUL1_K5": 2}


def test_exllamav3s_recipe_is_the_body_at_its_k_and_the_head_at_head_bits():
    ex = recipe.exllamav3(recipe.derive(_config()))
    # exllamav3 takes the head from head_bits: a `tensors` entry for it would be ignored, so there is none.
    assert ex == {"target_bpw": 3.0, "head_bits": 4, "tensors": {
        P + "0.linear_attn.in_proj_qkv": 3, P + "0.linear_attn.in_proj_z": 2, P + "0.linear_attn.out_proj": 4,
        P + "0.mlp.down_proj": 5, P + "3.self_attn.q_proj": 3, P + "3.self_attn.v_proj": 5}}


def test_the_yaml_reads_back_as_exllamav3_reads_it(tmp_path):
    path = tmp_path / "recipe.yaml"
    recipe.write_yaml(str(path), recipe.exllamav3(recipe.derive(_config())))
    with open(path, encoding="utf-8") as f:
        back = yaml.safe_load(f)
    assert back == recipe.exllamav3(recipe.derive(_config()))
    # convert_model.prepare takes integer rates 1-8 (16 unquantized) and 1.5 / 2.5 / 3.5 with mul1
    assert all(type(v) is int and 1 <= v <= 8 for v in back["tensors"].values())


def test_a_half_rate_is_kept_as_a_half():
    r = recipe.derive(_config(**{P + "0.mlp.down_proj": _storage(P + "0.mlp.down_proj", 128, 256, 40)}))
    assert r["matrices"][P + "0.mlp.down_proj"]["K"] == 2.5
    assert recipe.exllamav3(r)["tensors"][P + "0.mlp.down_proj"] == 2.5


@pytest.mark.parametrize("change,message", [
    ({P + "0.mlp.down_proj": _storage(P + "0.mlp.down_proj", 128, 256, 48, bits=4)},
     P + r"0\.mlp\.down_proj: bits_per_weight 4, its trellis stores K 3"),
    ({P + "0.mlp.down_proj": _storage(P + "0.mlp.down_proj", 128, 256, 128)},
     P + r"0\.mlp\.down_proj: K 8, which the 16 GB artifact cannot store"),
    ({"lm_head": _storage("lm_head", 512, 128, 96)}, r"lm_head: K 6, the head is quantized at head_bits 4"),
    ({"model.visual.blocks.0.attn.qkv": _storage("model.visual.blocks.0.attn.qkv", 128, 128, 48)},
     r"model\.visual\.blocks\.0\.attn\.qkv: a quantized matrix outside the body and the head"),
])
def test_a_configuration_the_recipe_cannot_take_is_refused_naming_the_matrix(change, message):
    with pytest.raises(convert.Refused, match=message):
        recipe.derive(_config(**change))


def test_a_configuration_with_no_head_is_refused():
    config = _config()
    del config["tensor_storage"]["lm_head"]
    with pytest.raises(convert.Refused, match="no quantized lm_head"):
        recipe.derive(config)


def test_another_recipe_is_refused_as_the_converter_refuses_it():
    with pytest.raises(convert.Refused, match="bits = 2.5"):
        recipe.derive(dict(_config(), bits=2.5))
