"""Record Flash-Next's three-axis positions and rotations over prompts with images (GitHub #558,
spec #555, ADR 0087).

Run from the repository root, CPU only (no GPU; only the checkpoint's config.json is fetched):

    F:/ai/ngram-venv/Scripts/python.exe kernel/tests/fixtures/flash_next_mrope/record_mrope.py

The oracle is the checkpoint's own transformers code (ADR 0043), models/qwen4_exp (transformers
5.17), whose multimodal rope index is Qwen3.5's:

- positions: `Qwen4ExpModel.get_rope_index` over each prompt's token types and image grids (text
  advances all three axes; an image's tokens take its start plus their row and column on the merged
  grid, and the text after it resumes past the larger side), and for every decode step
  `Qwen4ExpModel.compute_3d_position_ids` (the position plus the prompt's rope delta). Both are
  called on a stand-in `self` holding the checkpoint's `vision_config.spatial_merge_size`, read from
  its config.json at the pinned revision, so no model is built;
- rotations: one QSA layer, `Qwen4ExpTextAttention` (layer 3, with its `Qwen4ExpTextQSAIndexer`),
  at Flash-Next's geometry from the same config (24 query heads and 2 key heads of 256, rotary 64,
  the indexer's 4 query heads and one key head of 128, compress 4), its rotary table
  `Qwen4ExpTextRotaryEmbedding` over the prompt's positions, and the cache the text model builds
  (`DynamicCache(config=...)`, which keeps the indexer's raw keys), run as transformers runs a
  prompt: the whole prompt in one call, then one call per decode step. The projections are replaced
  by synthetic outputs (the layer's own q_proj, k_proj, v_proj and index_qk_proj are the only
  thing replaced), so the recorded values are the module's own norms and rotations of known inputs.
  `apply_rotary_pos_emb`, which the module calls for its indexer queries, for every pooled block key
  (at the block's FIRST token's three axes) and for the attention's q and k, is wrapped to read
  what it returns.

The synthetic inputs sit on a half-integer grid ({+-0.5, +-1, +-1.5, +-2}, counter hash
`lowbias32(i * 0x9E3779B9 + stream * 0x85EBCA6B)`, kernel/tests/moe_fixture.h), so every sum of
squares and every pooled mean is exact in fp32 whatever the order: the norms come out the same
in any summation order, and what a kernel can differ in is the rotation's own rounding. The test
regenerates them from the streams below and checks their checksums against the recorded ones.

Cases (`calls` are where kernel/tests/test_flash_next_mrope.cu cuts each prompt into prefill
calls -- the reference runs it whole -- so that a block straddles a call boundary):
  text                 23 text tokens.
  one_image            5 text, a 3 x 4 merged image, 9 text.
  two_images           3 text, 2 x 2, 2 text, 4 x 3, 7 text: block [12, 16) starts on the second
                       image's (1, 0) token and straddles the call boundary at 13.
  image_across_blocks  6 text, a 3 x 3 image, 9 text: blocks [8, 12) and [12, 16) start on image
                       tokens (0, 2) and (2, 0), and each straddles a call boundary.
  decode_after_image   5 text, a 2 x 3 image ending the prompt, then 6 decode steps: the first step
                       completes block [8, 12), which starts on the image's (1, 0) token.

Writes, next to itself: mrope_ref.bin (IGNFX001, kernel/tests/moe_fixture.h reads it) and
mrope_provenance.json; and crates/artifact/tests/fixtures/flash_next_mrope/positions.json, the
positions alone, which crates/artifact/tests/flash_next_mrope_positions.rs checks
`assign_positions` against bit for bit.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import os
import struct
import sys
import types

import httpx
import torch
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
POSITIONS_JSON = os.path.join(REPO_ROOT, "crates", "artifact", "tests", "fixtures", "flash_next_mrope",
                              "positions.json")
REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
torch.set_num_threads(4)

TEXT_ID = 1000  # any text token: the positions read the token types, not the ids
LAYER = 3       # a QSA layer of the checkpoint's layer_types
# The attention query heads recorded (the kernel rotates all 24 alike): one per KV group.
Q_HEADS = (0, 13)

# (name, runs, prefill calls of the kernel test, decode steps). A run is ("T", n) or ("I", h, w) with
# h x w the MERGED grid (grid_thw = (1, merge h, merge w) in patches).
CASES = [
    ("text", [("T", 23)], [(0, 10), (10, 23)], 0),
    ("one_image", [("T", 5), ("I", 3, 4), ("T", 9)], [(0, 7), (7, 26)], 0),
    ("two_images", [("T", 3), ("I", 2, 2), ("T", 2), ("I", 4, 3), ("T", 7)], [(0, 13), (13, 28)], 0),
    ("image_across_blocks", [("T", 6), ("I", 3, 3), ("T", 9)], [(0, 9), (9, 13), (13, 24)], 0),
    ("decode_after_image", [("T", 5), ("I", 2, 3)], [(0, 11)], 6),
]

# Counter-hash streams, shared with kernel/tests/test_flash_next_mrope.cu: per case c the
# attention's q_proj output, its k_proj output and the indexer's index_qk_proj output; one stream
# per norm weight.
def streams(case):
    base = 0x5580 + 16 * case
    return base, base + 1, base + 2


S_Q_NORM, S_K_NORM, S_IX_Q_NORM, S_IX_K_NORM = 0x55F0, 0x55F1, 0x55F2, 0x55F3

M32 = 0xFFFFFFFF


def lowbias32(x: torch.Tensor) -> torch.Tensor:
    x = x & M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32
    x = x ^ (x >> 16)
    return x


def hash_u32(stream: int, i: torch.Tensor) -> torch.Tensor:
    return lowbias32(((i & M32) * 0x9E3779B9 + stream * 0x85EBCA6B) & M32)


def grid(stream: int, rows: int, width: int) -> torch.Tensor:
    """BF16 [rows][width]: element i = r * width + j is 0.5 (1 + h % 4), negative when bit 2 of h is set."""
    h = hash_u32(stream, torch.arange(rows * width, dtype=torch.int64))
    v = 0.5 * (1 + (h & 3)).double()
    v = torch.where(((h >> 2) & 1) == 1, -v, v)
    return v.view(rows, width).to(torch.bfloat16)


def norm_weight(stream: int, n: int) -> torch.Tensor:
    """(1 + w) norm weights w = (h % 65 - 32) / 128, BF16-exact."""
    return (((hash_u32(stream, torch.arange(n, dtype=torch.int64)) % 65) - 32).double() / 128.0).to(torch.bfloat16)


def checksum(t: torch.Tensor) -> int:
    """sum over i of bits(t[i]) * (i + 1): the test's own checksum of what it regenerates."""
    bits = t.contiguous().view(torch.int16).flatten().to(torch.int64) & 0xFFFF
    idx = torch.arange(1, bits.numel() + 1, dtype=torch.int64)
    return int((bits * idx).sum().item())


def checkpoint_config():
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/config.json"
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content, json.loads(r.content)


def text_config(cfg):
    tc = cfg["text_config"]
    keys = ["hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "indexer_n_heads",
            "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio", "rms_norm_eps",
            "max_position_embeddings", "rope_parameters"]
    return Qwen4ExpTextConfig(**{k: tc[k] for k in keys if k in tc})


def prompt(runs, merge, image_id):
    ids, types_, grids = [], [], []
    for run in runs:
        if run[0] == "T":
            ids += [TEXT_ID] * run[1]
            types_ += [0] * run[1]
        else:
            _, h, w = run
            ids += [image_id] * (h * w)
            types_ += [1] * (h * w)
            grids.append([1, merge * h, merge * w])
    return ids, types_, grids


def stand_in(merge):
    """`self` for Qwen4ExpModel's position methods: the checkpoint's merge size and nothing else."""
    s = types.SimpleNamespace()
    s.config = types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=merge))
    s.get_vision_position_ids = types.MethodType(M.Qwen4ExpModel.get_vision_position_ids, s)
    s.rope_deltas = None
    return s


class Rows(torch.nn.Module):
    """A projection replaced by its known output: the rows [at, at + n) of `out` for an n-row call."""

    def __init__(self, out):
        super().__init__()
        self.out = out
        self.at = 0

    def forward(self, h):
        n = h.shape[1]
        return self.out[self.at:self.at + n][None]


class Taps:
    """What apply_rotary_pos_emb returns during one module call, by role."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        original = M.apply_rotary_pos_emb

        def wrapped(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
            out = original(q, k, cos=cos, sin=sin, unsqueeze_dim=unsqueeze_dim)
            self.calls.append((k is not None, unsqueeze_dim, out))
            return out

        self.original = original
        M.apply_rotary_pos_emb = wrapped
        return self

    def __exit__(self, *exc):
        M.apply_rotary_pos_emb = self.original

    def split(self):
        """The indexer's queries [n][4][128], its block keys at the call's last row (or None), the
        attention's q [n][24][256] and k [n][2][256]."""
        assert self.calls[0][0] is False and self.calls[0][1] == 2, "the indexer's queries come first"
        ix_q = self.calls[0][2][0]
        keys = [c[2].squeeze(1) for c in self.calls[1:-1]]
        assert all(c[0] is False and c[1] == 1 for c in self.calls[1:-1]), "then a block-key rotation per row"
        for a, b in zip(keys, keys[1:]):
            assert torch.equal(a, b[: a.shape[0]]), "a row's block keys are the next row's first blocks"
        assert self.calls[-1][0] is True, "the attention's q and k come last"
        q, k = self.calls[-1][2]
        return ix_q, (keys[-1] if keys else None), q[0].transpose(0, 1), k[0].transpose(0, 1)


DT = {torch.int32: 3, torch.float32: 6, torch.int64: 8}


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(b"IGNFX001")

    def put(self, name, t):
        t = t.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            t, code = t.view(torch.int16), 5
        else:
            code = DT[t.dtype]
        nb = name.encode("utf-8")
        payload = t.numpy().tobytes()
        self.f.write(struct.pack("<I", len(nb)) + nb + struct.pack("<II", code, t.dim()))
        for d in t.shape:
            self.f.write(struct.pack("<Q", d))
        self.f.write(struct.pack("<Q", len(payload)) + payload)

    def close(self):
        self.f.close()


def record_case(index, name, runs, calls, decode, cfg, merge, image_id, rot, norms):
    ids, types_, grids = prompt(runs, merge, image_id)
    length = len(ids)
    assert calls[0][0] == 0 and calls[-1][1] == length and all(a[1] == b[0] for a, b in zip(calls, calls[1:]))
    s = stand_in(merge)
    input_ids = torch.tensor([ids])
    pos, deltas = M.Qwen4ExpModel.get_rope_index(
        s, input_ids, torch.tensor([types_]), image_grid_thw=torch.tensor(grids) if grids else None)
    s.rope_deltas = deltas
    positions = [pos[:, 0]]  # [3][length]

    class SeenLength:
        def __init__(self, n):
            self.n = n

        def get_seq_length(self):
            return self.n

    for step in range(decode):
        p = M.Qwen4ExpModel.compute_3d_position_ids(
            s, input_ids=torch.tensor([[TEXT_ID]]), inputs_embeds=torch.zeros(1, 1, cfg.hidden_size),
            past_key_values=SeenLength(length + step))
        positions.append(p[:, 0])
    positions = torch.cat(positions, dim=1).to(torch.int64)  # [3][length + decode]
    total = length + decode

    attn = M.Qwen4ExpTextAttention(cfg, layer_idx=LAYER).to(torch.bfloat16).eval()
    s_qg, s_k, s_qk = streams(index)
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    ix_heads, ix_hd = cfg.indexer_n_heads, cfg.indexer_head_dim
    qg = grid(s_qg, total, heads * 2 * hd)
    kk = grid(s_k, total, kv_heads * hd)
    qk = grid(s_qk, total, (ix_heads + cfg.indexer_kv_heads) * ix_hd)
    attn.q_proj, attn.k_proj, attn.indexer.index_qk_proj = Rows(qg), Rows(kk), Rows(qk)
    attn.v_proj = Rows(grid(s_k ^ 0xFFFF, total, kv_heads * hd))
    with torch.no_grad():
        attn.q_norm.weight.copy_(norms["q"].float())
        attn.k_norm.weight.copy_(norms["k"].float())
        attn.indexer.q_layernorm.weight.copy_(norms["ix_q"].float())
        attn.indexer.k_layernorm.weight.copy_(norms["ix_k"].float())
    cos, sin = rot(torch.zeros(1, total, 8, dtype=torch.bfloat16), positions[:, None, :])

    cache = DynamicCache(config=cfg)
    out = {"ix_q": [], "q": [], "k": []}
    block_keys = None
    steps = [(0, length)] + [(length + i, length + i + 1) for i in range(decode)]
    for begin, end in steps:
        n = end - begin
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.indexer.index_qk_proj):
            proj.at = begin
        mask = torch.full((n, end), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
        for r in range(n):
            mask[r, : begin + r + 1] = 0
        with torch.no_grad(), Taps() as taps:
            attn(torch.zeros(1, n, cfg.hidden_size, dtype=torch.bfloat16), (cos[:, :end], sin[:, :end]),
                 mask[None, None], cache)
        ix_q, keys, q, k = taps.split()
        out["ix_q"].append(ix_q)
        out["q"].append(q[:, list(Q_HEADS)])
        out["k"].append(k)
        if keys is not None:
            assert block_keys is None or torch.equal(block_keys, keys[: block_keys.shape[0]]), \
                "a later call's block keys extend the earlier ones'"
            block_keys = keys
    assert block_keys.shape[0] == total // 4
    return {
        "ids": ids, "types": types_, "grids": grids, "calls": calls, "decode": decode, "length": length,
        "positions": positions, "delta": int(deltas.flatten()[0]),
        "ix_q": torch.cat(out["ix_q"]), "q": torch.cat(out["q"]), "k": torch.cat(out["k"]),
        "block_keys": block_keys, "sums": [checksum(qg), checksum(kk), checksum(qk)],
    }


def main():
    raw, config = checkpoint_config()
    merge = config["vision_config"]["spatial_merge_size"]
    image_id = config["image_token_id"]
    cfg = text_config(config)
    rot = M.Qwen4ExpTextRotaryEmbedding(cfg)
    assert rot.attention_scaling == 1.0, "the checkpoint's own table has no attention factor"
    norms = {"q": norm_weight(S_Q_NORM, cfg.head_dim), "k": norm_weight(S_K_NORM, cfg.head_dim),
             "ix_q": norm_weight(S_IX_Q_NORM, cfg.indexer_head_dim),
             "ix_k": norm_weight(S_IX_K_NORM, cfg.indexer_head_dim)}

    wr = Writer(os.path.join(HERE, "mrope_ref.bin"))
    wr.put("inv_freq", rot.inv_freq.float())
    wr.put("q_heads", torch.tensor(Q_HEADS, dtype=torch.int32))
    for key, w in norms.items():
        wr.put(f"norm.{key}", w)
    wr.put("cases", torch.tensor([len(CASES)], dtype=torch.int32))
    positions_json = {
        "recorded_by": "kernel/tests/fixtures/flash_next_mrope/record_mrope.py",
        "oracle": "transformers models/qwen4_exp Qwen4ExpModel.get_rope_index (prompts) and "
                  "compute_3d_position_ids (decode steps)",
        "checkpoint": {"repo": REPO, "revision": REVISION,
                       "config_sha256": hashlib.sha256(raw).hexdigest()},
        "spatial_merge_size": merge, "image_token_id": image_id, "cases": [],
    }
    prov_cases = {}
    for index, (name, runs, calls, decode) in enumerate(CASES):
        c = record_case(index, name, runs, calls, decode, cfg, merge, image_id, rot, norms)
        at = f"c{index}"
        wr.put(f"{at}.length", torch.tensor([c["length"]], dtype=torch.int32))
        wr.put(f"{at}.decode", torch.tensor([decode], dtype=torch.int32))
        wr.put(f"{at}.calls", torch.tensor(calls, dtype=torch.int32))
        wr.put(f"{at}.positions", c["positions"].to(torch.int32))
        wr.put(f"{at}.delta", torch.tensor([c["delta"]], dtype=torch.int32))
        wr.put(f"{at}.sums", torch.tensor(c["sums"], dtype=torch.int64))
        wr.put(f"{at}.attn_q", c["q"])
        wr.put(f"{at}.attn_k", c["k"])
        wr.put(f"{at}.ix_q", c["ix_q"])
        wr.put(f"{at}.block_keys", c["block_keys"])
        length = c["length"]
        positions_json["cases"].append({
            "name": name, "types": c["types"], "grids": c["grids"],
            "positions": c["positions"][:, :length].tolist(), "rope_delta": c["delta"],
            "decode": {"first": length, "positions": c["positions"][:, length:].tolist()},
        })
        prov_cases[name] = {"index": index, "tokens": length, "decode_steps": decode, "calls": calls,
                            "grids_thw": c["grids"], "rope_delta": c["delta"],
                            "streams": dict(zip(("q_proj", "k_proj", "index_qk_proj"), streams(index))),
                            "input_checksums": c["sums"], "blocks": int(c["block_keys"].shape[0])}
        print(f"{name}: {length} tokens + {decode} decode, delta {c['delta']}, "
              f"{c['block_keys'].shape[0]} blocks")
    wr.close()
    os.makedirs(os.path.dirname(POSITIONS_JSON), exist_ok=True)
    with open(POSITIONS_JSON, "w", newline="\n") as f:
        json.dump(positions_json, f, indent=1)
        f.write("\n")
    prov = {
        "recorded_by": "kernel/tests/fixtures/flash_next_mrope/record_mrope.py",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "torch": torch.__version__, "python": sys.version.split()[0], "transformers": transformers.__version__,
        "modeling_sha256": hashlib.sha256(inspect.getsource(M).encode("utf-8")).hexdigest(),
        "oracle": "transformers models/qwen4_exp: Qwen4ExpModel.get_rope_index and compute_3d_position_ids "
                  "(positions); Qwen4ExpTextAttention layer 3 with its Qwen4ExpTextQSAIndexer, "
                  "Qwen4ExpTextRotaryEmbedding and DynamicCache (rotations), CPU, bf16",
        "checkpoint": positions_json["checkpoint"],
        "geometry": {"hidden": cfg.hidden_size, "q_heads": cfg.num_attention_heads,
                     "kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
                     "indexer_heads": cfg.indexer_n_heads, "indexer_head_dim": cfg.indexer_head_dim,
                     "compress": cfg.indexer_compress_ratio, "rope_parameters": cfg.rope_parameters,
                     "spatial_merge_size": merge},
        "recorded_q_heads": list(Q_HEADS),
        "norm_streams": {"q_norm": S_Q_NORM, "k_norm": S_K_NORM, "ix_q_norm": S_IX_Q_NORM, "ix_k_norm": S_IX_K_NORM},
        "positions_json": "crates/artifact/tests/fixtures/flash_next_mrope/positions.json",
        "cases": prov_cases,
    }
    with open(os.path.join(HERE, "mrope_provenance.json"), "w", newline="\n") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
