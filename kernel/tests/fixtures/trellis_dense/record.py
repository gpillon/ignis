"""Record the dense trellis fixtures (spec 27b-16gb/02, GitHub #337).

Run from the repository root, on a free GPU (it imports exllamav3's CUDA extension), with the
exllamav3 1.5.3 venv and TORCH_CUDA_ARCH_LIST unset:

    F:/ai/ngram-venv/Scripts/python.exe kernel/tests/fixtures/trellis_dense/record.py

It writes, next to itself:

  trellis_k2.bin, trellis_k2p5.bin, trellis_k3.bin, trellis_k4.bin, trellis_k5.bin, trellis_k6.bin
      One projection at reduced shape (256 inputs x 128 outputs) per K, encoded by exllamav3's own
      quantizer (quantize_exl3_batch, mul1, out scales on, the 27B quant's settings) from a
      synthetic weight and Hessian: its trellis tensor bit for bit (exllamav3's int16
      [in/16][out/16][16 K] layout), the fp16 channel scales suh/svh, `reconstruct`'s fp16 output
      -- the oracle the kernel's tile decode must equal bit for bit -- and three tokens of
      activations with their fp64 product through the full weight diag(suh) H inner H diag(svh),
      the rotation semantics checked against exllamav3's own forward before writing.
  trellis_checksums.bin
      Bit-exactness at the 27B's real shapes at no fixture cost: for every K and every shape the
      27B serves (spec 02's table, fused and split), a trellis tensor of counter-hash words (any
      bit pattern is a valid encoding) and a 64-bit checksum of `reconstruct`'s output. The CTest
      regenerates the same words and checksums its own decode.
  provenance.json
      Versions, seeds, and what this script measured about the reference.

ExLlamaV3 is an oracle only (ADR 0060, ADR 0051): this script calls its quantizer, its
`reconstruct` and its Python forward, and reads nothing of its CUDA sources. The decoder below
(`decode_inner`) is the format's definition, the one kernel/tests/fixtures/flash_next/record.py
states for K <= 4 generalized to every K, asserted bit-exact against `reconstruct` before
anything is written.

Container format (IGNFX001), little-endian: the 8-byte magic, then records until EOF, each
  u32 name_len, name (utf-8), u32 dtype, u32 ndim, u64 dims[ndim], u64 nbytes, payload.
dtype codes: 0 u8, 1 i16, 2 u16, 3 i32, 4 f16, 5 bf16 (as u16 bits), 6 f32, 7 f64, 8 u32, 9 u64.
kernel/tests/moe_fixture.h reads it.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import struct

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda:0")
KS = (2, 2.5, 3, 4, 5, 6)

# ---------------------------------------------------------------------------------------------
# Counter hash shared with the C++ tests (kernel/tests/moe_fixture.h): lowbias32 of
# (i * 0x9E3779B9 + stream * 0x85EBCA6B) mod 2^32. Vectorized in int64 with explicit masking.

M32 = 0xFFFFFFFF


def lowbias32(x: torch.Tensor) -> torch.Tensor:
    x = x & M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32
    x = x ^ (x >> 16)
    return x


def hash_u32(stream: int, start: int, n: int, device=None) -> torch.Tensor:
    i = torch.arange(start, start + n, dtype=torch.int64, device=device)
    return lowbias32((i * 0x9E3779B9 + stream * 0x85EBCA6B) & M32)


def hash_uniform(stream: int, n: int, amplitude: float) -> torch.Tensor:
    """float32 in [-amplitude, amplitude): ((h >> 8) * 2^-23 - 1) * amplitude, each step exact or
    one IEEE float32 rounding, as the C++ side computes it."""
    h = hash_u32(stream, 0, n)
    u = (h >> 8).to(torch.float32) * (2.0 ** -23) - 1.0
    return u * torch.tensor(amplitude, dtype=torch.float32)


# ---------------------------------------------------------------------------------------------
# The trellis format, from its definition (flash_next/record.py), for every K.
#
# A tile is a bitstream of 256 trellis steps; step p contributes k_p new bits (K; for K = 2.5,
# 3 bits at odd p and 2 at even p), 16 K int16 words. The stream is big-endian in 32-bit units.
# Step p's 16-bit state is the 16 stream bits ending at bit end_p - 1, wrapping around the tile
# (tail-biting); its value is the mul1 codebook; it lands at tile element TC_PERM[p].

MUL1 = 0x83DCD12D


def tc_perm() -> list[int]:
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        rows = [r0, r0 + 1, r0 + 8, r0 + 9]
        for j in range(8):
            r = rows[j % 4]
            c = c0 + (8 if j >= 4 else 0)
            perm[t * 8 + j] = r * 16 + c
    return perm


def steps(K: float) -> list[int]:
    if K == 2.5:
        return [3 if p % 2 == 1 else 2 for p in range(256)]
    assert K in (2, 3, 4, 5, 6)
    return [int(K)] * 256


def words_per_tile(K: float) -> int:
    return int(16 * K)


def mul1_value(states: torch.Tensor) -> torch.Tensor:
    prod = (states.to(torch.int64) * MUL1) & M32
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    k_inv = torch.tensor([0x1EEE], dtype=torch.int16).view(torch.float16).float()
    k_bias = torch.tensor([0xC931 - 0x10000], dtype=torch.int16).view(torch.float16).float()
    return ((1024 + bsum).to(torch.float32) * k_inv.to(states.device) + k_bias.to(states.device)).to(torch.float16)


def decode_states(trellis: torch.Tensor, K: float) -> torch.Tensor:
    tk, tn, W = trellis.shape
    assert W == words_per_tile(K)
    w16 = trellis.view(torch.int16).to(torch.int64) & 0xFFFF
    u32 = w16[..., 0::2] | (w16[..., 1::2] << 16)
    L = W * 16
    b = torch.arange(L, device=trellis.device)
    bits = (u32[..., b // 32] >> (31 - (b % 32))) & 1
    ks = steps(K)
    ends = torch.tensor([sum(ks[: p + 1]) for p in range(256)], device=trellis.device)
    j = torch.arange(16, device=trellis.device)
    idx = (ends.unsqueeze(1) - 16 + j.unsqueeze(0)) % L
    return (bits[..., idx] << (15 - j)).sum(-1)


def decode_inner(trellis: torch.Tensor, K: float) -> torch.Tensor:
    tk, tn, _ = trellis.shape
    vals = mul1_value(decode_states(trellis, K))
    perm = torch.tensor(tc_perm(), device=trellis.device)
    tile = torch.empty_like(vals)
    tile[..., perm] = vals
    return tile.view(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(tk * 16, tn * 16)


def hadamard128_f64() -> torch.Tensor:
    i = torch.arange(128)
    x = i.unsqueeze(1) & i.unsqueeze(0)
    pc = torch.zeros_like(x)
    for bit in range(7):
        pc += (x >> bit) & 1
    return (1 - 2 * (pc % 2)).to(torch.float64) / math.sqrt(128.0)


def full_weight_f64(inner: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """diag(suh) Hb inner Hb diag(svh), Hb = block-diagonal 128-wide Sylvester / sqrt(128)."""
    k, n = inner.shape
    H = hadamard128_f64().to(inner.device)
    w = inner.to(torch.float64)
    w = (H @ w.view(k // 128, 128, n)).view(k, n)
    w = w * suh.to(torch.float64).unsqueeze(1)
    w = (w.view(k, n // 128, 128) @ H).view(k, n)
    return w * svh.to(torch.float64).unsqueeze(0)


# ---------------------------------------------------------------------------------------------
# Container writer.

DT = {torch.uint8: 0, torch.int16: 1, torch.float16: 4, torch.int32: 3, torch.float32: 6, torch.float64: 7}


class Writer:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "wb")
        self.f.write(b"IGNFX001")

    def put(self, name: str, t: torch.Tensor, dtype_code: int | None = None):
        t = t.detach().cpu().contiguous()
        code = dtype_code if dtype_code is not None else DT[t.dtype]
        self.put_raw(name, code, list(t.shape), t.numpy().tobytes())

    def put_raw(self, name: str, code: int, dims: list[int], payload: bytes):
        nb = name.encode("utf-8")
        self.f.write(struct.pack("<I", len(nb)) + nb)
        self.f.write(struct.pack("<II", code, len(dims)))
        for d in dims:
            self.f.write(struct.pack("<Q", d))
        self.f.write(struct.pack("<Q", len(payload)))
        self.f.write(payload)

    def close(self):
        self.f.close()


def kname(K: float) -> str:
    return "k2p5" if K == 2.5 else f"k{int(K)}"


# ---------------------------------------------------------------------------------------------
# Fixture 1: reduced-shape projections encoded by exllamav3's quantizer.

RED_IN, RED_OUT = 256, 128
TOKENS = 3


def record_trellis(K: float, prov: dict):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.modules.quant.exl3_lib import quantize as xq

    g = torch.Generator(device="cpu").manual_seed(337 + int(K * 10))
    # A weight with per-row and per-column magnitude structure, and a correlated Hessian, so the
    # quantizer's regularization, LDLQ and scale refit all do real work.
    W = torch.randn(RED_IN, RED_OUT, generator=g) * 0.02
    W *= torch.exp(torch.randn(RED_IN, 1, generator=g) * 0.5)
    W *= torch.exp(torch.randn(1, RED_OUT, generator=g) * 0.3)
    X = torch.randn(4 * RED_IN, RED_IN, generator=g) @ (torch.eye(RED_IN) + 0.1 * torch.randn(RED_IN, RED_IN, generator=g))
    H = (X.T @ X) / X.shape[0]
    hd = {"H": H.to(DEV), "count": 1, "finalized": False, "device": DEV, "L": None, "first_key": f"dense.{kname(K)}"}
    qa = {"K": K, "devices": [0], "mul1": True, "apply_out_scales": True, "seed": 3370 + int(K * 10)}
    res = xq.quantize_exl3_batch([W.to(DEV)], [hd], [qa])
    ot = res[0][1]
    trellis, suh, svh = ot["trellis"].to(DEV), ot["suh"].to(DEV), ot["svh"].to(DEV)
    assert trellis.dtype == torch.int16 and tuple(trellis.shape) == (RED_IN // 16, RED_OUT // 16, words_per_tile(K))

    recon = torch.empty((RED_IN, RED_OUT), dtype=torch.half, device=DEV)
    ext.reconstruct(recon, trellis, K, False, True)
    mine = decode_inner(trellis, K)
    assert torch.equal(recon.view(torch.int16), mine.view(torch.int16)), f"K={K}: decoder != reconstruct"

    # Activations: BF16-representable values from the counter hash, three tokens.
    x = hash_uniform(stream=7337 + int(K * 10), n=TOKENS * RED_IN, amplitude=2.0).to(torch.bfloat16).float().view(TOKENS, RED_IN)
    w64 = full_weight_f64(recon, suh, svh)
    y64 = x.to(torch.float64).to(DEV) @ w64

    # The rotation semantics, checked against exllamav3's own forward (fp16 internally).
    mul1 = torch.tensor(xq.codebook_mul1_mult, dtype=torch.uint32).view(torch.int).to(DEV)
    lin = LinearEXL3(None, RED_IN, RED_OUT, suh=suh, svh=svh, trellis=trellis, mul1=mul1)
    y_exl = lin.forward(x.half().to(DEV), {"reconstruct": True}).double()
    rel = ((y_exl - y64).abs().max() / y64.abs().max()).item()
    assert rel < 3e-3, f"K={K}: fp64 semantics vs exllamav3 forward rel {rel}"

    path = os.path.join(HERE, f"trellis_{kname(K)}.bin")
    w = Writer(path)
    w.put("K_times_2", torch.tensor([int(K * 2)], dtype=torch.int32))
    w.put("trellis", trellis)
    w.put("suh", suh)
    w.put("svh", svh)
    w.put("reconstruct", recon)
    w.put("x", x)
    w.put("y64", y64)
    w.close()
    prov["trellis"][kname(K)] = {"file": os.path.basename(path), "shape_in_out": [RED_IN, RED_OUT],
                                  "y64_vs_exllamav3_forward_max_rel": rel}
    print(f"K={K}: wrote {path}; y64 vs exllamav3 forward max rel {rel:.2e}")


# ---------------------------------------------------------------------------------------------
# Fixture 2: real-geometry checksums over counter-hash trellis words.

# (in, out) of every matrix the 27B serves (spec 02's table, and the split parts a repack of the
# public quant arrives as).
SHAPES = {
    "gate_up": (5120, 34816), "gate": (5120, 17408), "down": (17408, 5120),
    "qkvz": (5120, 16384), "qkv": (5120, 10240), "z": (5120, 6144), "gdn_out": (6144, 5120),
    "qkgv": (5120, 14336), "q": (5120, 12288), "kv": (5120, 1024),
    "head": (5120, 248320),
}
CHUNK = 1 << 24


def checksum_u16(values16: torch.Tensor) -> int:
    """sum_i u16[i] * (lowbias32(i) | 1) mod 2^64 over the row-major (in, out) fp16 bits."""
    v = values16.reshape(-1).view(torch.int16)
    total = 0
    for start in range(0, v.numel(), CHUNK):
        part = v[start:start + CHUNK].to(torch.int64) & 0xFFFF
        mult = lowbias32(torch.arange(start, start + part.numel(), dtype=torch.int64, device=part.device)) | 1
        total = (total + int((part * mult).sum().item())) % (1 << 64)  # int64 wraps mod 2^64
    return total


def trellis_words(stream: int, tiles_k: int, tiles_n: int, K: float) -> torch.Tensor:
    n = tiles_k * tiles_n * words_per_tile(K)
    out = torch.empty(n, dtype=torch.int16, device=DEV)
    for start in range(0, n, CHUNK):
        m = min(CHUNK, n - start)
        h = hash_u32(stream, start, m, device=DEV) & 0xFFFF
        out[start:start + m] = torch.where(h >= 32768, h - 65536, h).to(torch.int16)
    return out.view(tiles_k, tiles_n, words_per_tile(K))


def stream_of(K: float, shape_index: int) -> int:
    return 33700 + int(K * 2) * 100 + shape_index


def record_checksums(prov: dict):
    from exllamav3.ext import exllamav3_ext as ext

    w = Writer(os.path.join(HERE, "trellis_checksums.bin"))
    prov["checksums"] = {}
    for K in KS:
        for si, (name, (k, n)) in enumerate(SHAPES.items()):
            stream = stream_of(K, si)
            words = trellis_words(stream, k // 16, n // 16, K)
            recon = torch.empty((k, n), dtype=torch.half, device=DEV)
            ext.reconstruct(recon, words, K, False, True)
            if k * n <= 5120 * 1024:  # the definition too, where its bit tensor fits
                assert torch.equal(recon.view(torch.int16), decode_inner(words, K).view(torch.int16)), f"{name} K={K}"
            cs = checksum_u16(recon)
            key = f"{name}_{kname(K)}"
            w.put_raw(key, 9, [4], struct.pack("<QQQQ", stream, cs, k, n))
            prov["checksums"][key] = {"stream": stream, "checksum": f"{cs:016x}", "in_out": [k, n]}
            print(f"{key}: stream {stream} checksum {cs:016x}")
            del words, recon
            torch.cuda.empty_cache()
    w.close()


def main():
    assert "TORCH_CUDA_ARCH_LIST" not in os.environ, "unset TORCH_CUDA_ARCH_LIST (it rebuilds exllamav3)"
    from exllamav3.version import __version__ as exl3_version

    prov = {
        "recorded_by": "kernel/tests/fixtures/trellis_dense/record.py",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exllamav3": exl3_version, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "trellis": {},
    }
    for K in KS:
        record_trellis(K, prov)
    record_checksums(prov)
    with open(os.path.join(HERE, "provenance.json"), "w", newline="\n") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
