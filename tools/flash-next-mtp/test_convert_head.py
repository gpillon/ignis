"""CPU tests of the MTP head converter's pure parts (layout.md §13)."""
import json
import os

import numpy as np
import pytest
import torch

import convert_head as ch
import layout
import nonexpert

# The checkpoint's 31 mtp.* tensors (revision de4b8e4), the two fused expert tensors last.
CHECKPOINT = {
    "mtp.hyper_connection_mixer.hc_norm.weight": [10240],
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": [320, 10240],
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": [4, 10240],
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": [10240],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": [320, 10240],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.mlp.gate.weight": [512, 2560],
    "mtp.layers.0.mlp.shared_expert_gate.weight": [1, 2560],
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": [640, 2560],
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": [640, 2560],
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": [2560, 640],
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": [4, 10240],
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": [10240],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": [320, 10240],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": [640, 2560],
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": [128],
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": [128],
    "mtp.layers.0.self_attn.k_norm.weight": [256],
    "mtp.layers.0.self_attn.o_proj.weight": [2560, 6144],
    "mtp.layers.0.self_attn.k_proj.weight": [512, 2560],
    "mtp.layers.0.self_attn.q_proj.weight": [12288, 2560],
    "mtp.layers.0.self_attn.v_proj.weight": [512, 2560],
    "mtp.layers.0.self_attn.q_norm.weight": [256],
    "mtp.pre_fc_norm_embedding.weight": [2560],
    "mtp.pre_fc_norm_hidden.weight": [10240],
    "mtp.fc_embedding.weight": [2560, 2560],
    "mtp.fc_hidden.weight": [2560, 2560],
    "mtp.layers.0.mlp.experts.gate_up_proj": [512, 1280, 2560],
    "mtp.layers.0.mlp.experts.down_proj": [512, 2560, 640],
}
# The Rust inventory's FP8 set (crates/artifact/src/flash_next.rs, mtp_entries' test).
FP8 = {"mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
       "mtp.hyper_connection_mixer.input_mix_weight_down.weight", "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
       "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
       "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
       "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
       "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
       "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight",
       "mtp.layers.0.self_attn.v_proj.weight", "mtp.layers.0.self_attn.o_proj.weight",
       "mtp.layers.0.self_attn.indexer.index_qk_proj.weight", "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
       "mtp.layers.0.mlp.shared_expert.up_proj.weight", "mtp.layers.0.mlp.shared_expert.down_proj.weight"}


def test_the_trunk_rule_on_layer_local_names_gives_the_contracts_formats():
    encodings = {}
    for name, shape in CHECKPOINT.items():
        local, prefix = ch.split_name(name)
        assert prefix + local == name
        encodings[name] = layout.encoding_of(local, tuple(shape))
    assert {n for n, e in encodings.items() if e == "expert"} == {"mtp." + n for n in ch.EXPERTS}
    assert {n for n, e in encodings.items() if e == "fp8"} == FP8
    assert sum(e == "bf16" for e in encodings.values()) == 13
    with pytest.raises(ValueError):
        ch.split_name("model.language_model.lm_head.weight")


def test_a_tensor_file_carries_the_whole_checkpoint_name(tmp_path):
    torch.manual_seed(0)
    out = []
    for name, shape in (("mtp.fc_hidden.weight", (32, 32)), ("mtp.layers.0.mlp.gate.weight", (8, 32)),
                        ("mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", (4, 64))):
        local, prefix = ch.split_name(name)
        out.append(nonexpert.write_tensor(str(tmp_path), local, torch.randn(shape).bfloat16(), prefix=prefix))
    assert [(e["name"], e["format"]) for e in out] == [
        ("mtp.fc_hidden.weight", "FP8_E4M3FN_ROW_BF16S"), ("mtp.layers.0.mlp.gate.weight", "BF16"),
        ("mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", "BF16")]
    assert all(os.path.getsize(tmp_path / e["file"]) == e["bytes"] for e in out)


def _chunks():
    return [{"ids": list(range(100, 110)), "valid": 10, "kind": "code", "cal": True, "test": False, "source": "a"},
            {"ids": list(range(200, 210)), "valid": 6, "kind": "mmlu", "cal": True, "test": False, "source": "b"},
            {"ids": list(range(300, 310)), "valid": 10, "kind": "prose", "cal": False, "test": True, "source": "c"},
            {"ids": list(range(400, 410)), "valid": 4, "kind": "code", "cal": False, "test": True, "source": "d"}]


def test_the_chunks_are_written_valid_tokens_only_in_corpus_order(tmp_path):
    m = ch.export_chunks(_chunks(), str(tmp_path))
    assert [c["name"] for c in m["chunks"]] == ["c000", "c001", "c002", "c003"]
    assert [c["tokens"] for c in m["chunks"]] == [10, 6, 10, 4]
    assert np.fromfile(tmp_path / "c001.tokens.u32", np.uint32).tolist() == list(range(200, 206))
    assert json.load(open(tmp_path / "chunks.json")) == m

    cal, test, kind, kinds = ch.row_masks(m["chunks"])
    # a chunk of n tokens gives n - 1 entries: entry p is built from (stack p, token p + 1)
    assert len(cal) == 9 + 5 + 9 + 3
    assert cal.tolist() == [True] * 14 + [False] * 12
    assert test.tolist() == [False] * 14 + [True] * 12
    assert kinds == ["code", "prose"]
    assert kind.tolist() == [1] * 9 + [0] * 3


def test_the_shim_runs_the_trunks_zero_token_fallback(tmp_path):
    """The trunk converter's expert step reads its Conversion through these attributes; an expert
    no calibration entry reached gets the layer H (gate/up) and the all-entry moment (down)."""
    cal = np.array([True] * 6 + [False] * 2)
    shim = ch._shim("cpu", 3.0, str(tmp_path), cal, ~cal, np.zeros(2, np.int64), ["code"])
    torch.manual_seed(0)
    X = torch.randn(8, 2560).bfloat16()
    Hl = torch.eye(2560)
    Wgu = torch.randn(2, 1280, 2560).bfloat16() * 0.02
    Hg = torch.zeros(2, 2560, 2560)
    Hg[1] = 2 * torch.eye(2560)
    Hd = torch.zeros(2, 640, 640)
    rec = {}
    Qg, Qd = shim._fallback_hessians(X, Hl, Wgu, Hg, Hd, 0, rec)
    assert rec["hessian_fallback"] == [0]
    assert torch.equal(Qg[0], Hl) and torch.equal(Qg[1], Hg[1])
    y = X[:6].float() @ Wgu[0].float().T
    a = torch.nn.functional.silu(y[:, :640]) * y[:, 640:]
    torch.testing.assert_close(Qd[0], a.T @ a / 6)


def test_the_summary_pairs_the_quantized_hits_with_phase_as():
    def run(gen, ok):
        return {"runs": {"aa-dense": {"gen": {"a": gen}, "gen_ok1": ok}}}
    q = {"texts": {"t1": {"kind": "code", "tokens": 2048, **run([[4, 3], [3, 2], [2, 2], [2, 1]], "1101")},
                   "t2": {"kind": "prose", "tokens": 2048, **run([[4, 2], [2, 1], [1, 1], [1, 0]], "0110")}}}
    bf16 = {"t1": run([[4, 4], [4, 3], [3, 3], [3, 2]], "1111"), "t2": run([[4, 2], [2, 2], [2, 1], [1, 1]], "0110")}
    s = ch.summarize(q, bf16)["aa-dense"]
    assert s["all"]["alpha_quantized"] == [5 / 8, 3 / 5, 3 / 3, 1 / 3]
    assert s["all"]["alpha_bf16"] == [6 / 8, 5 / 6, 4 / 5, 3 / 4]
    assert s["code"]["texts"] == 1 and s["all"]["positions"] == 8
    mean, half = s["all"]["alpha1_bf16_minus_quantized"]
    assert mean == pytest.approx(1 / 8) and half > 0


def test_the_moe_inputs_are_each_entrys_own_in_chunk_order(tmp_path):
    """Entry p of a chunk is built from (stack p, token p + 1); its MoE input and routing are the
    head layer's over the chunk causally, whatever the batch split, chunk after chunk."""
    import mtp
    import phase_a
    import test_mtp
    cfg, ref, head32 = test_mtp.build()
    weights = {"layers.0." + k: v for k, v in ref.state_dict().items()}
    weights.update({"hyper_connection_mixer." + k: v for k, v in head32.mixer.state_dict().items()})
    weights.update({"fc_hidden.weight": head32.fc_hidden, "fc_embedding.weight": head32.fc_embedding,
                    "pre_fc_norm_hidden.weight": head32.norm_hidden,
                    "pre_fc_norm_embedding.weight": head32.norm_embedding})
    head = mtp.Head(cfg, weights, head32.embed, head32.lm_head, dtype=torch.bfloat16)
    np.random.seed(1)
    torch.manual_seed(1)
    width = cfg.hc_count * cfg.hidden_size
    chunks = []
    for name, n in (("c000", 7), ("c001", 5)):
        np.random.randint(0, cfg.vocab_size, n).astype("<u4").tofile(tmp_path / f"{name}.tokens.u32")
        torch.randn(n, width).bfloat16().view(torch.int16).numpy().tofile(tmp_path / f"{name}.stacks.bf16")
        chunks.append({"name": name, "tokens": n})
    X, ridx, rw = ch._moe_inputs(head, str(tmp_path), chunks, "cpu", batch=4)
    assert X.shape == (6 + 4, cfg.hidden_size)
    assert ridx.shape == rw.shape == (10, cfg.num_experts_per_tok)

    want, want_r = [], []
    cap = {}
    hooks = [head.layer.mlp.register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0])),
             head.layer.mlp.gate.register_forward_hook(lambda m, a, o: cap.__setitem__("r", o))]
    with torch.no_grad():
        for c in chunks:
            tok = torch.from_numpy(np.fromfile(tmp_path / f"{c['name']}.tokens.u32", np.uint32).astype(np.int64))
            S = phase_a.to_bf16(np.fromfile(tmp_path / f"{c['name']}.stacks.bf16", np.uint16).reshape(-1, width), "cpu")
            Xe = head.combine(S[:-1], tok[1:], "a", "a")
            head.window(head.entries(Xe, torch.arange(len(tok) - 1)), None, "dense")
            head.first_step(torch.arange(len(tok) - 1))
            want.append(cap["x"][0])
            want_r.append(cap["r"])
    for h in hooks:
        h.remove()
    assert torch.equal(X, torch.cat(want))
    assert torch.equal(ridx, torch.cat([r[2] for r in want_r]))
    assert torch.equal(rw, torch.cat([r[1].float() for r in want_r]))


def test_the_shim_carries_every_attribute_the_trunks_expert_step_reads(tmp_path):
    """A later change to the trunk converter's expert step that reads another attribute of its
    Conversion fails here, not mid-run on the GPU."""
    import inspect
    import re
    from pipeline import Conversion
    cal = np.array([True, False])
    shim = ch._shim("cpu", 3.0, str(tmp_path), cal, ~cal, np.zeros(1, np.int64), ["code"])
    for f in ("_convert_experts", "_fallback_hessians", "_decode_experts", "_moe_error"):
        for attr in set(re.findall(r"self\.(\w+)", inspect.getsource(getattr(Conversion, f)))):
            assert hasattr(shim, attr), f"{f} reads self.{attr}"
    assert shim.args.budget == 3.0


def test_the_inputs_are_checked_before_any_gpu_work(tmp_path):
    main = tmp_path / "model" / "main-v2.ninfer"
    main.parent.mkdir()
    main.write_bytes(b"main")
    calib = tmp_path / "calib"
    m = ch.export_chunks(_chunks()[:2], str(calib))
    with pytest.raises(RuntimeError, match="tap.json is missing"):
        ch.check_inputs(str(calib), m["chunks"], str(main))
    tap = {"model_dir": str(main.parent), "artifact": "other-v2.ninfer", "kv_format": "hq-e8-2b", "chunks": 2}
    (calib / "tap.json").write_text(json.dumps(tap))
    with pytest.raises(RuntimeError, match="tapped from"):
        ch.check_inputs(str(calib), m["chunks"], str(main))
    (calib / "tap.json").write_text(json.dumps({**tap, "artifact": main.name}))
    with pytest.raises(RuntimeError, match="c000.stacks.bf16 is missing"):
        ch.check_inputs(str(calib), m["chunks"], str(main))
    for c in m["chunks"]:
        (calib / f"{c['name']}.stacks.bf16").write_bytes(bytes(c["tokens"] * ch.WIDTH * 2))
    assert ch.check_inputs(str(calib), m["chunks"], str(main))["kv_format"] == "hq-e8-2b"
    with pytest.raises(RuntimeError, match="does not exist"):
        ch.check_inputs(str(calib), m["chunks"], str(tmp_path / "gone.ninfer"))


def _write_container(path, tensors, records):
    """A minimal v2 container (the reader's framing): tensors {name: (format, shape, bytes)} then
    expert records {(expert, proj): (k2, record)} under the companion's names."""
    import struct
    import container
    objects, payload = [], b""

    def put(name, fmt, shape, layout_name, body):
        nonlocal payload
        payload += bytes(-len(payload) % 4096)
        objects.append({"name": name, "kind": "tensor", "shape": list(shape), "format": fmt, "layout": layout_name,
                        "offset": len(payload), "bytes": len(body)})
        payload += body
    for name, (fmt, shape, body) in tensors.items():
        put(name, fmt, shape, "row-scale-v1" if fmt.startswith("FP8") else "contiguous-le-v1", body)
    for (e, p), (k2, rec) in records.items():
        i, o = layout.SHAPES[p]
        fmt = {v: k for k, v in container.K2_OF_FORMAT.items()}[k2]
        put(f"mtp.layers.0.mlp.experts.{e}.{container.PROJ_NAME[p]}", fmt, (o, i), "trellis-tile16-v1",
            layout.record_payload(p, k2, rec))
    js = json.dumps({"identity": {"model_id": "qwen3.8-flash-next-mtp", "weights_id": "w"}, "objects": objects}).encode()
    head = container.MAGIC + struct.pack("<Q", len(js)) + js
    path.write_bytes(head + bytes(-len(head) % 4096) + payload)


def test_the_companions_tensors_and_records_come_back_from_its_bytes(tmp_path):
    import container
    import fp8
    from test_layout import fake_record
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    norm = torch.randn(64).bfloat16()
    recs = {(0, "gu"): (6, fake_record("gu", 6, 1)), (0, "dn"): (5, fake_record("dn", 5, 2))}
    _write_container(tmp_path / "c.ninfer", {
        "mtp.fc_hidden.weight": ("FP8_E4M3FN_ROW_BF16S", (32, 64), fp8.encode(w)),
        "mtp.pre_fc_norm_embedding.weight": ("BF16", (64,), norm.view(torch.int16).numpy().tobytes())}, recs)
    c = container.Container(str(tmp_path / "c.ninfer"))
    assert torch.equal(ch.container_tensor(c, "mtp.fc_hidden.weight"), fp8.decode(fp8.encode(w), (32, 64)))
    assert torch.equal(ch.container_tensor(c, "mtp.pre_fc_norm_embedding.weight"), norm)
    for (e, p), (k2, want) in recs.items():
        got_k2, got = ch.expert_record(c, e, p)
        assert got_k2 == k2
        for name in ("trellis", "suh", "svh"):
            assert np.array_equal(got[name].view(np.uint8), want[name].view(np.uint8))


def test_every_decoded_projection_is_hashed_by_class():
    gq, dq = torch.zeros(2, 4, 3, dtype=torch.bfloat16), torch.ones(2, 3, 2, dtype=torch.bfloat16)
    h = ch.decode_hashes(gq, dq, {"gu": [5, 8], "dn": [4, 6]})
    assert [(x["expert"], x["class"]) for x in h] == [(0, "gu-2.5"), (0, "dn-2"), (1, "gu-4"), (1, "dn-3")]
    assert h[0]["sha256"] == h[2]["sha256"] != h[1]["sha256"]


def test_ac2_passes_within_three_hundredths_of_alpha_1():
    def summary(delta):
        return {"aa-dense": {"all": {"alpha_bf16": [0.83], "alpha_quantized": [0.83 - delta],
                                     "alpha1_bf16_minus_quantized": [delta, 0.01]}}}
    assert ch.verdict(summary(0.02), "bf16")["pass"]
    assert not ch.verdict(summary(0.031), "bf16")["pass"]
