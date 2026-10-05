"""Record the Flash-Next QSA indexer fixture (spec flash-next/04, GitHub #302, slice S3).

Run from the repository root, CPU only (no GPU, no download):

    F:/ai/ngram-venv/Scripts/python.exe kernel/tests/fixtures/flash_next_indexer/record_indexer.py

The oracle is the checkpoint's own modeling code (ADR 0043): transformers'
`Qwen4ExpTextQSAIndexer` (models/qwen4_exp/modeling_qwen4_exp.py, transformers 5.17) at Flash-Next's
real geometry -- hidden 2560, 4 query heads of 128 and one key head, compress ratio 4, budget 2048
(512 blocks), rotary 64 of the head's 128 at theta 1e7 -- with synthetic weights. The script runs
the module itself, re-derives its scores with the same torch ops, and asserts that the module's
selected-token mask equals the re-derivation's for every row before writing anything.

Exactness by construction. The weights are FP8 E4M3FN codes of magnitude 0.125..1.875 with a
power-of-two row scale, so the BF16 weight the module runs is the FP8 dequantization exactly, and
the inputs sit on a coarse grid (multiples of 1/16 up to 1/2, or of 1/256 for the one engineered
token), so every partial sum of a projection row is exact in fp32 (asserted below: the sum of
|terms| of every row stays under 2^24 grid units). The projection a kernel computes in any
summation order is then bit-identical to the module's, and what remains to compare is the
indexer's own arithmetic.

Two sequences ("lanes") of 4100 tokens:
  lane 0  every token a counter-hash grid vector (stream 0x1D10): ~6% of blocks score exactly 0.
  lane 1  the TIE sequence. Three blocks in four repeat one engineered token `x_rep` whose pooled,
          normed key has (almost) no rotary part and a negative dot product with every head of
          the query token `x_q` (placed at the recorded rows): those blocks score exactly 0 for
          those rows at any position. With fewer than 512 positive blocks the selection is
          decided among exact zero ties, i.e. by the tie rule alone.

The tie rule. `scores.topk(k)` keeps the k largest and leaves ties at the k-th value to the
implementation. torch on CPU does not keep the lowest index (measured here, see provenance);
torch on CUDA -- what the converter's reference ran -- writes the values strictly above the k-th
and then the equal ones in index order ("choose the first seen set", TensorTopK.cu). The fixture's
expected selection is that rule stated explicitly: stable descending sort, LOWEST BLOCK INDEX
FIRST among equal scores (check_topk_ties_cuda.py verifies torch CUDA against it on a GPU). The
CPU `topk` selection's agreement with it is recorded per row in the provenance.

Writes, next to itself: indexer_ref.bin (IGNFX001, kernel/tests/moe_fixture.h reads it) and
indexer_provenance.json.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import struct
import sys

import torch

from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)

# ---------------------------------------------------------------------------------------------
# Geometry (Qwen/Qwen3.8-Flash-Next @ de4b8e4d config.json, the fields the indexer reads).

HIDDEN = 2560
HEADS = 4
HEAD_DIM = 128
COMPRESS = 4
BUDGET = 2048
BLOCK_TOPK = BUDGET // COMPRESS  # 512
ROTARY = 64  # head_dim 256 * partial_rotary_factor 0.25
THETA = 10_000_000
EPS = 1e-6
QK_ROWS = (HEADS + 1) * HEAD_DIM  # 640
T = 4100

CONFIG = dict(
    hidden_size=HIDDEN, num_attention_heads=24, num_key_value_heads=2, head_dim=256,
    indexer_n_heads=HEADS, indexer_kv_heads=1, indexer_head_dim=HEAD_DIM,
    indexer_budget=BUDGET, indexer_compress_ratio=COMPRESS, rms_norm_eps=EPS,
    max_position_embeddings=262144,
    rope_parameters={"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                     "partial_rotary_factor": 0.25, "rope_theta": THETA, "rope_type": "default"},
)

# Counter-hash streams, shared with kernel/tests/test_flash_next_indexer.cu.
S_CODES, S_SCALE, S_QNORM, S_KNORM = 0x1D00, 0x1D01, 0x1D02, 0x1D03
S_X0, S_X1, S_XQ = 0x1D10, 0x1D11, 0x1D12

LANE0_ROWS = [2049, 2050, 2051, 2052, 2053, 2054, 2055, 2100, 2500, 3071, 3072, 3584,
              4095, 4096, 4097, 4098, 4099]
LANE1_ROWS = [2400, 2603, 3299, 3800, 4096, 4097, 4098, 4099]  # every one is an x_q token

# ---------------------------------------------------------------------------------------------
# Counter hash (kernel/tests/moe_fixture.h): lowbias32(i * 0x9E3779B9 + stream * 0x85EBCA6B).

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


def e4m3_value(code: torch.Tensor) -> torch.Tensor:
    sign = (code >> 7) & 1
    e = (code >> 3) & 15
    m = code & 7
    v = (1.0 + m.double() / 8.0) * torch.pow(2.0, (e - 7).double())
    return torch.where(sign == 1, -v, v)


def weights():
    """FP8 codes [640][2560] (exponent field 4..7: |value| in [0.125, 1.875]), power-of-two BF16
    row scales 2^-(2 + h % 4), and the two (1 + w) norm weights w = (h % 65 - 32) / 128."""
    i = torch.arange(QK_ROWS * HIDDEN, dtype=torch.int64)
    h = hash_u32(S_CODES, i)
    code = (((h >> 5) & 1) << 7) | ((4 + (h & 3)) << 3) | ((h >> 2) & 7)
    code = code.view(QK_ROWS, HIDDEN)
    r = torch.arange(QK_ROWS, dtype=torch.int64)
    scale = torch.pow(2.0, -(2 + (hash_u32(S_SCALE, r) % 4)).double())
    d = torch.arange(HEAD_DIM, dtype=torch.int64)
    qn = ((hash_u32(S_QNORM, d) % 65) - 32).double() / 128.0
    kn = ((hash_u32(S_KNORM, d) % 65) - 32).double() / 128.0
    w = e4m3_value(code) * scale[:, None]
    assert torch.equal(w.to(torch.bfloat16).double(), w), "FP8 dequantization must be BF16-exact"
    return code.to(torch.uint8), scale, w, qn, kn


def grid_tokens(stream: int, first: int, n: int) -> torch.Tensor:
    """x[t][c] = (h % 17 - 8) / 16 for t in [first, first + n): BF16-exact, |x| <= 1/2."""
    i = (torch.arange(n, dtype=torch.int64)[:, None] + first) * HIDDEN + torch.arange(HIDDEN)[None, :]
    return ((hash_u32(stream, i) % 17) - 8).double() / 16.0


# ---------------------------------------------------------------------------------------------
# The module and its transparent re-derivation.


def build(w, qn, kn):
    cfg = Qwen4ExpTextConfig(**CONFIG)
    idx = M.Qwen4ExpTextQSAIndexer(cfg, layer_idx=3)
    idx.load_state_dict({"index_qk_proj.weight": w.to(torch.bfloat16),
                         "q_layernorm.weight": qn.to(torch.bfloat16),
                         "k_layernorm.weight": kn.to(torch.bfloat16)})
    idx = idx.to(torch.bfloat16).eval()
    rot = M.Qwen4ExpTextRotaryEmbedding(cfg)
    pos = torch.arange(T).view(1, 1, T).expand(3, 1, T)
    cos, sin = rot(torch.zeros(1, T, 8, dtype=torch.bfloat16), pos)
    return idx, rot, cos, sin


def exact_projection_check(code: torch.Tensor, x: torch.Tensor):
    """Every partial sum of every projection row, in any order, is exact in fp32: a code is a
    multiple of 2^-6 and an input of 2^-4 (or 2^-8), so a product is a multiple of the unit u =
    2^-10 (2^-14), and any partial sum is bounded by the sum of |terms|, which stays under 2^24 u.
    The power-of-two row scale multiplies an exact sum exactly."""
    unit = 2.0 ** -10 if torch.equal(x * 16, torch.round(x * 16)) else 2.0 ** -14
    assert torch.equal(x / unit * 2.0 ** -6, torch.round(x / unit * 2.0 ** -6)), "input off the grid"
    worst = (e4m3_value(code.to(torch.int64)).abs() @ x.abs().T).max().item()
    assert worst < (2 ** 24) * unit, f"projection sums not exact: {worst} >= {(2 ** 24) * unit}"
    return worst


def rederive(idx, x, cos, sin, rows):
    """The module's indexer math, op for op, exposing q, the block keys and the scores."""
    with torch.no_grad():
        h = x.to(torch.bfloat16)[None]
        qk = idx.index_qk_proj(h)
        q, tk = torch.split(qk, [HEADS * HEAD_DIM, HEAD_DIM], dim=-1)
        q = q.reshape(1, T, -1, HEAD_DIM)
        raw = tk.reshape(1, T, -1, HEAD_DIM).squeeze(2)
        q = idx.q_layernorm(q)
        q = M.apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)

        def block_keys(n):
            bti = torch.arange(n * COMPRESS).view(n, COMPRESS)
            groups = raw[0].index_select(0, bti.flatten()).view(n, COMPRESS, HEAD_DIM)
            pooled = idx.k_layernorm(groups.float().mean(dim=1).to(raw.dtype))
            starts = bti[:, 0]
            return M.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[0].index_select(0, starts),
                                          sin=sin[0].index_select(0, starts)).squeeze(1)

        all_keys = block_keys(T // COMPRESS)
        out = {}
        for r in rows:
            n = (r + 1) // COMPRESS
            bk = block_keys(n)
            assert torch.equal(bk, all_keys[:n]), f"row {r}: block keys depend on the prefix length"
            s = torch.matmul(q[0, r].float(), bk.float().transpose(-1, -2)).transpose(-1, -2)
            s = torch.relu(s).sum(dim=-1) / math.sqrt(HEAD_DIM)
            out[r] = (q[0, r].clone(), s, torch.matmul(q[0, r].float(), bk.float().T))
        return qk[0], raw[0], all_keys, out


def rule_select(s: torch.Tensor, k: int) -> torch.Tensor:
    """The documented rule: the k largest, ties at the k-th value by lowest block index."""
    order = torch.sort(s, descending=True, stable=True).indices
    return torch.sort(order[:k]).values


def module_blocks(mask_row: torch.Tensor, r: int):
    sel = torch.nonzero(mask_row[: r + 1]).flatten()
    n = (r + 1) // COMPRESS
    tail = list(range(n * COMPRESS, r + 1))
    body = sel[sel < n * COMPRESS]
    assert all(t in sel.tolist() for t in tail), f"row {r}: tail tokens not selected"
    blocks = torch.unique(body // COMPRESS)
    assert body.numel() == blocks.numel() * COMPRESS, f"row {r}: partial block selected"
    return blocks


# ---------------------------------------------------------------------------------------------
# The engineered tie token.


def make_x_rep(w, kn, x_q, idx):
    """A token whose pooled key (4 copies) after k_layernorm points along d: no rotary part and
    d . q_h = -c for every head h of the query token x_q, so its blocks score relu(<0) = 0."""
    with torch.no_grad():
        qk = idx.index_qk_proj(x_q.to(torch.bfloat16)[None, None])
        q = idx.q_layernorm(qk[..., : HEADS * HEAD_DIM].reshape(HEADS, HEAD_DIM)).double()
    qn = q[:, ROTARY:]  # [4, 64], untouched by rope
    d = torch.zeros(HEAD_DIM, dtype=torch.float64)
    d[ROTARY:] = -qn.T @ torch.linalg.solve(qn @ qn.T, torch.ones(HEADS, dtype=torch.float64))
    d *= math.sqrt(HEAD_DIM) / d.norm()
    r = d / (1.0 + kn)
    wk = w[HEADS * HEAD_DIM:]
    x = wk.T @ torch.linalg.solve(wk @ wk.T, r)
    x *= 0.45 / x.abs().max()
    return torch.clamp(torch.round(x * 256.0) / 256.0, -0.5, 0.5)


# ---------------------------------------------------------------------------------------------
# IGNFX001 writer (as fixtures/flash_next/record.py's).

DT = {torch.uint8: 0, torch.int16: 1, torch.int32: 3, torch.float16: 4, torch.float32: 6,
      torch.float64: 7}


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(b"IGNFX001")

    def put(self, name, t, code=None):
        t = t.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            t, code = t.view(torch.int16), 5
        code = DT[t.dtype] if code is None else code
        nb = name.encode("utf-8")
        payload = t.numpy().tobytes()
        self.f.write(struct.pack("<I", len(nb)) + nb + struct.pack("<II", code, t.dim()))
        for d in t.shape:
            self.f.write(struct.pack("<Q", d))
        self.f.write(struct.pack("<Q", len(payload)) + payload)

    def close(self):
        self.f.close()


def main():
    code, scale, w, qn, kn = weights()
    idx, rot, cos, sin = build(w, qn, kn)
    inv = rot.inv_freq.clone()
    replica = torch.tensor([1.0 / torch.tensor(THETA ** (2.0 * i / ROTARY), dtype=torch.float32).item()
                            for i in range(ROTARY // 2)], dtype=torch.float32)
    assert torch.equal(inv, replica), "inv_freq != 1.0f / (float)pow(theta, 2i/rotary)"

    x0 = grid_tokens(S_X0, 0, T)
    x_q = grid_tokens(S_XQ, 0, 1)[0]
    x_rep = make_x_rep(w, kn, x_q, idx)
    x1 = grid_tokens(S_X1, 0, T)
    rep_tokens = [t for t in range(T) if (t // COMPRESS) % 4 != 0]
    x1[rep_tokens] = x_rep
    x1[LANE1_ROWS] = x_q
    for x in (x0, grid_tokens(S_X1, 0, T), x_q[None]):
        exact_projection_check(code, x)
    exact_projection_check(code, x_rep[None])

    prov = {"recorded_by": "kernel/tests/fixtures/flash_next_indexer/record_indexer.py",
            "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "torch": torch.__version__, "python": sys.version.split()[0],
            "transformers": __import__("transformers").__version__,
            "oracle": "transformers models/qwen4_exp Qwen4ExpTextQSAIndexer.forward (CPU, bf16)",
            "geometry": {"hidden": HIDDEN, "heads": HEADS, "head_dim": HEAD_DIM, "compress": COMPRESS,
                         "budget": BUDGET, "block_topk": BLOCK_TOPK, "rotary": ROTARY, "theta": THETA,
                         "tokens": T},
            "streams": {"codes": S_CODES, "scale": S_SCALE, "q_norm": S_QNORM, "k_norm": S_KNORM,
                        "x_lane0": S_X0, "x_lane1": S_X1, "x_q": S_XQ},
            "tie_rule": "k largest; at the k-th value, lowest block index first (torch CUDA topk's "
                        "gather order); CPU topk differs (cpu_vs_rule_differing_blocks per row)",
            "rows": {}}

    wr = Writer(os.path.join(HERE, "indexer_ref.bin"))
    wr.put("inv_freq", inv)
    wr.put("x_rep", x_rep.to(torch.bfloat16))
    for lane, (x, rows) in enumerate(((x0, LANE0_ROWS), (x1, LANE1_ROWS))):
        with torch.no_grad():
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))[None, None]
            sel_mask = idx(x.to(torch.bfloat16)[None], (cos, sin), mask, None)[0, 0]
        qk, raw, all_keys, out = rederive(idx, x, cos, sin, rows)
        exact = (x @ w.T).to(torch.bfloat16)  # fp64 sum, one rounding: what any exact fp32 sum gives
        assert torch.equal(qk, exact), f"lane {lane}: the module's projection is not the exact sum"
        wr.put(f"lane{lane}.rows", torch.tensor(rows, dtype=torch.int32))
        if lane == 0:  # every block key of one lane localizes a key-path error
            wr.put(f"lane{lane}.block_keys", all_keys)
        for r in rows:
            q, s, dots = out[r]
            n = s.numel()
            k = min(BLOCK_TOPK, n)
            cpu = torch.sort(s.topk(k, dim=0).indices).values
            mod = module_blocks(sel_mask[r], r)
            assert torch.equal(mod, cpu), f"lane {lane} row {r}: module mask != re-derived topk"
            rule = rule_select(s, k)
            kth = torch.sort(s, descending=True).values[k - 1].item()
            prov["rows"][f"lane{lane}.{r}"] = {
                "blocks": n, "k": k, "kth_score": kth,
                "above_kth": int((s > kth).sum()), "equal_kth": int((s == kth).sum()),
                "zero_scores": int((s == 0).sum()),
                "cpu_topk_equals_rule": bool(torch.equal(cpu, rule)),
                "cpu_vs_rule_differing_blocks": int(k - len(set(cpu.tolist()) & set(rule.tolist()))),
                "max_dot_over_zero_blocks": float(dots[:, s == 0].max()) if bool((s == 0).any()) else None,
            }
            wr.put(f"lane{lane}.row{r}.q", q)
            wr.put(f"lane{lane}.row{r}.scores", s.float())
            wr.put(f"lane{lane}.row{r}.select", rule.to(torch.int32))
        print(f"lane {lane}: module mask == re-derivation on {len(rows)} rows")
    wr.close()
    with open(os.path.join(HERE, "indexer_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
        f.write("\n")
    for key, v in prov["rows"].items():
        print(key, v)


if __name__ == "__main__":
    main()
