import collections
import json
import struct

import pytest

import exl3
import gguf_recipe
import recipe

P = "model.language_model.layers."


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def gguf(tensors, metadata=(), data=4096):
    """A GGUF v3 file's bytes: `metadata` [(key, value type, value)], `tensors` [(name, dims, ggml type)], then
    `data` bytes of tensor data; and the header's length."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, t, v in metadata:
        out += _s(key) + struct.pack("<I", t)
        if t == 8:
            out += _s(v)
        elif t == 4:
            out += struct.pack("<I", v)
        else:  # an array: (element type, items), strings or int32
            et, items = v
            out += struct.pack("<IQ", et, len(items))
            out += b"".join(_s(i) if et == 8 else struct.pack("<i", i) for i in items)
    for i, (name, dims, t) in enumerate(tensors):
        out += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", t, i * 4096)
    return out + bytes(-len(out) % 32) + bytes(data), len(out)


class Ranges:
    """A file served by ranges, every range asked for recorded."""

    def __init__(self, raw):
        self.raw, self.asked = raw, []

    def __call__(self, start, n):
        self.asked.append((start, n))
        return self.raw[start:start + n]


def test_the_header_is_read_by_ranges_and_no_further_than_one_chunk_past_its_end():
    raw, header = gguf([("output.weight", [256, 512], 12), ("blk.0.ffn_up.weight", [256, 512], 22)],
                       [("general.architecture", 8, "qwen35"), ("tokenizer.ggml.tokens", 9, (8, ["a", "bc", "d"])),
                        ("tokenizer.ggml.token_type", 9, (5, [1, 2, 3])), ("qwen35.block_count", 4, 1)],
                       data=1 << 20)
    src = Ranges(raw)
    h = gguf_recipe.read_header(src, chunk=64)
    assert h["header_bytes"] == header
    assert h["metadata"]["general.architecture"] == "qwen35" and h["metadata"]["qwen35.block_count"] == 1
    assert h["metadata"]["tokenizer.ggml.tokens"] == {"array": 8, "count": 3}
    assert h["tensors"] == [{"name": "output.weight", "dims": [256, 512], "type": 12, "offset": 0},
                            {"name": "blk.0.ffn_up.weight", "dims": [256, 512], "type": 22, "offset": 4096}]
    # the ranges tile the file from its start, and stop within one chunk of the header's end
    assert [s for s, _ in src.asked] == list(range(0, 64 * len(src.asked), 64))
    assert h["fetched_bytes"] == sum(n for _, n in src.asked) < header + 64


def test_a_file_that_is_not_gguf_is_refused():
    with pytest.raises(recipe.Refused, match="not a GGUF file"):
        gguf_recipe.read_header(Ranges(b"PK\x03\x04" + bytes(4096)), chunk=64)


# A reduced 27B: two GDN layers and a GQA layer, hidden 256.
SHAPES = {P + "0.linear_attn.in_proj_qkv": [512, 256], P + "0.linear_attn.in_proj_z": [256, 256],
          P + "0.linear_attn.out_proj": [256, 256], P + "0.mlp.gate_proj": [512, 256], P + "0.mlp.up_proj": [512, 256],
          P + "0.mlp.down_proj": [256, 512],
          P + "1.self_attn.q_proj": [512, 256], P + "1.self_attn.k_proj": [128, 256], P + "1.self_attn.v_proj": [128, 256],
          P + "1.self_attn.o_proj": [256, 256], P + "1.mlp.gate_proj": [512, 256], P + "1.mlp.up_proj": [512, 256],
          P + "1.mlp.down_proj": [256, 512],
          P + "2.linear_attn.in_proj_qkv": [512, 256], P + "2.linear_attn.in_proj_z": [256, 256],
          P + "2.linear_attn.out_proj": [256, 256], P + "2.mlp.gate_proj": [512, 256], P + "2.mlp.up_proj": [512, 256],
          P + "2.mlp.down_proj": [256, 512],
          "lm_head": [1024, 256]}
GGUF_NAME = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
             "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
             "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
             "mlp.down_proj": "ffn_down"}
# ggml types: IQ1_S 19, IQ2_XXS 16, IQ2_XS 17, IQ2_S 22, Q2_K 10, IQ3_XXS 18, IQ3_S 21, Q3_K 11, IQ4_XS 23, Q4_K 12,
# Q5_K 13, Q6_K 14
TYPES = {P + "0.linear_attn.in_proj_qkv": 22, P + "0.linear_attn.in_proj_z": 18, P + "0.linear_attn.out_proj": 23,
         P + "0.mlp.gate_proj": 19, P + "0.mlp.up_proj": 16, P + "0.mlp.down_proj": 17,
         P + "1.self_attn.q_proj": 10, P + "1.self_attn.k_proj": 21, P + "1.self_attn.v_proj": 12,
         P + "1.self_attn.o_proj": 14, P + "1.mlp.gate_proj": 18, P + "1.mlp.up_proj": 18, P + "1.mlp.down_proj": 21,
         P + "2.linear_attn.in_proj_qkv": 18, P + "2.linear_attn.in_proj_z": 11, P + "2.linear_attn.out_proj": 13,
         P + "2.mlp.gate_proj": 22, P + "2.mlp.up_proj": 22, P + "2.mlp.down_proj": 18,
         "lm_head": 12}
# Every other text object the text scope counts, and two it leaves out.
COPIED = {"text/final_norm": 512, "text/layers/0/input_norm": 512, "text/token_embedding": 1 << 20,
          "text/draft_head": 1 << 20, "vision/x": 4096}


def infos(types=TYPES, shapes=SHAPES):
    """The GGUF tensor infos of `types`: dims [in, out], the MTP layer's and the norms beside them."""
    out = [{"name": "output_norm.weight", "dims": [256], "type": 0, "offset": 0},
           {"name": "blk.3.ffn_up.weight", "dims": [256, 512], "type": 14, "offset": 0},
           {"name": "blk.0.ssm_alpha.weight", "dims": [256, 48], "type": 8, "offset": 0}]
    for key, t in types.items():
        name = "output.weight" if key == "lm_head" else (
            f"blk.{key[len(P):].split('.')[0]}.{GGUF_NAME[key[len(P):].split('.', 1)[1]]}.weight")
        out.append({"name": name, "dims": list(reversed(shapes[key])), "type": t, "offset": 0})
    return out


def test_each_matrix_takes_its_types_bits_per_weight_block_scales_included():
    a = gguf_recipe.allocation(infos(), SHAPES)
    assert set(a) == set(SHAPES)
    assert a[P + "0.mlp.gate_proj"] == {"type": "IQ1_S", "bits": 1.5625}
    assert a[P + "0.linear_attn.in_proj_qkv"] == {"type": "IQ2_S", "bits": 2.5625}
    assert a[P + "1.self_attn.k_proj"] == {"type": "IQ3_S", "bits": 3.4375}
    assert a[P + "2.linear_attn.in_proj_z"] == {"type": "Q3_K", "bits": 3.4375}
    assert a[P + "1.self_attn.o_proj"] == {"type": "Q6_K", "bits": 6.5625}
    assert a["lm_head"] == {"type": "Q4_K", "bits": 4.5}


@pytest.mark.parametrize("change,message", [
    (lambda t: [x for x in t if x["name"] != "blk.1.attn_k.weight"], P + r"1\.self_attn\.k_proj: no GGUF tensor"),
    (lambda t: [dict(x, dims=[256, 512]) if x["name"] == "blk.0.ffn_down.weight" else x for x in t],
     P + r"0\.mlp\.down_proj: GGUF dims \[256, 512\] \(\[in, out\]\), the record's shape is \[256, 512\] \(\[out, in\]\)"),
    (lambda t: [dict(x, type=39) if x["name"] == "blk.0.ffn_up.weight" else x for x in t],
     P + r"0\.mlp\.up_proj: ggml type 39, not one of the table's"),
])
def test_a_header_that_does_not_hold_the_records_matrices_is_refused(change, message):
    with pytest.raises(recipe.Refused, match=message):
        gguf_recipe.allocation(change(infos()), SHAPES)


def _fit(ceiling, types=TYPES):
    return gguf_recipe.fit(gguf_recipe.allocation(infos(types), SHAPES), SHAPES, COPIED, ceiling)


def _bytes(ks):
    return recipe.text_scope_bytes({k: {"K": K, "shape": SHAPES[k]} for k, K in ks.items()}, COPIED)


def _at(K):
    return _bytes({k: (4 if k == "lm_head" else K) for k in SHAPES})


CEILINGS = sorted({_at(2), _at(2) + 50_000, _at(3) - 1, _at(3), _at(3) + 30_000, _at(4) - 70_000, _at(4), _at(5) - 1,
                   _at(5), _at(5) + 1})


@pytest.mark.parametrize("ceiling", CEILINGS)
def test_the_fit_keeps_the_order_of_the_bits_and_the_ceiling_and_stops_at_the_first_matrix_that_would_cross_it(ceiling):
    a = gguf_recipe.allocation(infos(), SHAPES)
    f = gguf_recipe.fit(a, SHAPES, COPIED, ceiling)
    ks = f["K"]
    assert ks["lm_head"] == 4 and set(ks) == set(SHAPES)
    assert all(2 <= K <= 5 for K in ks.values())
    assert f["text_scope_bytes"] == _bytes(ks) <= ceiling
    body = [k for k in SHAPES if k != "lm_head"]
    # more bits never take a lower K (the head is outside the order: the load binds it at 4)
    assert all(ks[i] >= ks[j] for i in body for j in body if a[i]["bits"] > a[j]["bits"])
    # the next class to cross is promoted in rank order, a prefix: its first matrix left out does not fit
    if f["left"]:
        nxt = dict(ks, **{f["left"][0]: ks[f["left"][0]] + 1})
        assert _bytes(nxt) > ceiling
    else:
        assert all(ks[k] == min(5, max(2, round(a[k]["bits"] + 10))) for k in body)


def test_the_fit_clamps_to_k_2_and_k_5():
    assert _fit(_at(2))["K"][P + "0.mlp.gate_proj"] == 2  # IQ1_S, 1.5625 bits: K2 even with nothing to spare
    f = _fit(_at(5) + 1)
    assert set(f["K"][k] for k in SHAPES if k != "lm_head") == {5}  # Q6_K, 6.5625 bits: K5
    assert f["delta"] is not None and f["left"] == []


def test_with_the_shift_at_zero_each_matrix_takes_the_integer_nearest_its_bits():
    # every class of TYPES rounds the same way at a shift just above zero (Q4_K's 4.5 at 5)
    nearest = {k: min(5, max(2, int(b["bits"] + 0.5))) for k, b in gguf_recipe.allocation(infos(), SHAPES).items()}
    nearest["lm_head"] = 4
    f = _fit(_bytes(nearest))
    assert f["K"] == nearest and f["delta"] == "0" and f["left"] and f["promoted"] == []


def test_ties_are_promoted_by_the_layers_mean_bits_then_by_the_key():
    nearest = {k: min(5, max(2, int(b["bits"] + 0.5))) for k, b in gguf_recipe.allocation(infos(), SHAPES).items()}
    nearest["lm_head"] = 4
    f = _fit(_bytes(nearest))
    # the next class to cross above the nearest integers: IQ3_S and Q3_K, tied at 3.4375 bits (shift 1/16); layer 1's
    # matrices average 3.49 bits and layer 2's 3.14, so layer 1's go first, and within it the keys in order
    assert f["next_delta"] == "1/16"
    assert f["left"] == [P + "1.mlp.down_proj", P + "1.self_attn.k_proj", P + "2.linear_attn.in_proj_z"]
    one = exl3.record_bytes(8, 256, 512) - exl3.record_bytes(6, 256, 512)
    f = _fit(_bytes(nearest) + one)
    assert f["promoted"] == [P + "1.mlp.down_proj"] and f["K"][P + "1.mlp.down_proj"] == 4
    assert f["left"] == [P + "1.self_attn.k_proj", P + "2.linear_attn.in_proj_z"]


def test_a_ceiling_below_every_matrix_at_k_2_is_refused():
    with pytest.raises(recipe.Refused, match=f"at K2 the text scope is {_at(2):,} bytes, above the ceiling"):
        _fit(_at(2) - 1)


def test_the_bits_are_exllamav3s_its_body_average_k_to_two_decimals():
    ks = {k: 3 for k in SHAPES}
    ks[P + "0.mlp.down_proj"] = 2
    ks["lm_head"] = 4
    body = [k for k in SHAPES if k != "lm_head"]
    numel = {k: SHAPES[k][0] * SHAPES[k][1] for k in body}
    want = round(sum(numel[k] * ks[k] for k in body) / sum(numel.values()), 2)
    assert gguf_recipe.exllamav3_bits(ks, SHAPES) == want == 2.93


def _record(tmp_path):
    rec = {"trellis": {"matrices": {f"m{i}": {"source": k, "shape": s} for i, (k, s) in enumerate(SHAPES.items())}},
           "copied": {"objects": {n: {"bytes": b} for n, b in COPIED.items()}}}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    return str(path)


def _table(tmp_path):
    table = {"source": {"repo": "someone/Model-GGUF", "file": "Model-UD-Q2_K_XL.gguf", "revision": "ab" * 20,
                        "lfs_oid": "cd" * 32, "bytes": 123456},
             "header_bytes": 999, "fetched_bytes": 1024, "metadata": {}, "tensors": infos()}
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return str(path)


def test_the_recipe_is_written_as_the_bases_with_its_k_its_bits_and_the_calibrations_pins(tmp_path, capsys):
    out = tmp_path / "recipes"
    out.mkdir()
    ceiling = _at(3)
    assert gguf_recipe.main(["recipe", "--table", _table(tmp_path), "--record", _record(tmp_path), "--name", "ig28x",
                             "--weights-id", "trellis-ig28x", "--ceiling", str(ceiling), "--recipes", str(out)]) == 0
    r = recipe.load("ig28x", str(out))
    base = recipe.load("sc30")
    assert (r["weights_id"], r["ceiling_bytes"], r["job"]) == ("trellis-ig28x", ceiling, base["job"])
    assert r["files"] == {n: base["files"][n] for n in ("cal_trace.safetensors", "quantization_config.json")}
    f = _fit(ceiling)
    assert recipe.ks(r) == f["K"]
    assert r["quantization_config"] == dict(base["quantization_config"], bits=gguf_recipe.exllamav3_bits(f["K"], SHAPES))
    data = json.loads((out / "ig28x.json").read_text(encoding="utf-8"))
    assert list(data) == ["schema", "about", "weights_id", "ceiling_bytes", "quantization_config", "job", "files",
                          "tensors"]
    assert "someone/Model-GGUF Model-UD-Q2_K_XL.gguf at revision " + "ab" * 20 in data["about"]
    said = capsys.readouterr().out
    assert f"text scope {f['text_scope_bytes']:,} bytes of {ceiling:,}" in said
    assert "K2 " in said and collections.Counter(f["K"].values())[2] > 0


def test_a_recipe_that_exists_is_not_overwritten(tmp_path):
    out = tmp_path / "recipes"
    out.mkdir()
    (out / "ig28x.json").write_text("{}", encoding="utf-8")
    assert gguf_recipe.main(["recipe", "--table", _table(tmp_path), "--record", _record(tmp_path), "--name", "ig28x",
                             "--weights-id", "trellis-ig28x", "--ceiling", str(_at(3)), "--recipes", str(out)]) == 2
    assert (out / "ig28x.json").read_text(encoding="utf-8") == "{}"
