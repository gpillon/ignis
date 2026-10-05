"""Record the Flash-Next n-gram embedding's device-side fixture (spec flash-next/04, slice S4 of
GitHub #302): the checkpoint's own `Qwen4ExpTextPLELayer.forward` (transformers, the oracle, ADR
0043) at the real geometry, CPU only.

Run from the repository root with the study's venv (no GPU is touched):

    CUDA_VISIBLE_DEVICES= F:/ai/ngram-venv/Scripts/python.exe kernel/tests/fixtures/flash_next_ngram/record.py

It writes, next to itself:

  ngram_ref.bin (IGNFX001, kernel/tests/moe_fixture.h reads it)
      rows     u8   [T][16][90]   each token's gathered INT4 table rows (layout.md 7.1), from the
                                  counter hash
      ple_out  bf16 [T][10240]    the PLE layer's output for those rows and the hidden input below,
                                  one sequence from an empty conv state -- what the decoder adds to
                                  every stream before layer 1's attention mix
  provenance.json
      versions and the generation constants.

The weights and the hidden input are not stored: they come from the counter hash
(kernel/tests/moe_fixture.h's hash_u32 / hash_uniform, restated below), and the CTest
(kernel/tests/test_flash_next_ngram.cu) regenerates them with the same constants. The FP8
projections are FP8_E4M3FN_ROW_BF16S payloads built exactly as fp8_test_common.h's make_fp8, and
the reference runs the bf16 weight each payload stands for (the converter's fp8.decode: code times
scale in fp32, rounded to bf16), as the quantized reference does. The gathered rows are
dequantized by the converter's own table.decode_rows, rounded to bf16, as the conversion feeds
the quantized stream.
"""

from __future__ import annotations

import datetime
import json
import os
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "flash-next-converter"))

import fp8  # noqa: E402  (the converter's FP8 payload rules)
import table  # noqa: E402  (the converter's INT4 row rules)

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer  # noqa: E402

# ---- the geometry (the checkpoint's text config, revision de4b8e4d) --------------------------
HIDDEN = 2560
STREAMS = 4
EMBED = 2560  # ple_embed_dim: 16 heads x 160
HEADS = 16
ROW_BYTES = 90
CONV_KERNEL = 4
NGRAM_SIZE = 3  # also the conv's dilation
EPS = 1e-6
T = 16  # tokens: more than the conv's 9 past columns

# ---- generation constants, mirrored in test_flash_next_ngram.cu -----------------------------
S_KEY, S_VALUE = 4101, 4201
S_NORM_KEY, S_NORM_QUERY, S_NORM_CONV, S_CONV = 4301, 4302, 4303, 4304
S_HIDDEN, S_ROW_CODES, S_ROW_SCALES = 4401, 4501, 4502
KEY_SCALE_MAG, VALUE_SCALE_MAG = 0.0004, 0.0008
NORM_AMP, CONV_AMP, HIDDEN_AMP = 0.5, 0.5, 0.05
ROW_SCALE_MAG = 0.02

M32 = 0xFFFFFFFF


def lowbias32(x: torch.Tensor) -> torch.Tensor:
    x = x & M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32
    x = x ^ (x >> 16)
    return x


def hash_u32(stream: int, n: int) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.int64)
    return lowbias32((i * 0x9E3779B9 + stream * 0x85EBCA6B) & M32)


def hash_uniform(stream: int, n: int, amplitude: float) -> torch.Tensor:
    h = hash_u32(stream, n)
    u = (h >> 8).to(torch.float32) * (2.0 ** -23) - 1.0
    return u * torch.tensor(amplitude, dtype=torch.float32)


def make_fp8(stream: int, rows: int, cols: int, scale_mag: float) -> bytes:
    """fp8_test_common.h's make_fp8: hashed E4M3FN codes (NaN codes flipped), padding to 256, BF16
    scales scale_mag * (0.5 + |u|)."""
    codes = (hash_u32(stream, rows * cols) & 0xFF).to(torch.uint8)
    nan = (codes & 0x7F) == 0x7F
    codes = torch.where(nan, codes ^ 1, codes)
    mag = torch.tensor(scale_mag, dtype=torch.float32)
    scale = (mag * (0.5 + hash_uniform(stream + 1, rows, 1.0).abs())).to(torch.bfloat16)
    raw = codes.numpy().tobytes()
    raw += bytes(-len(raw) % 256)
    return raw + scale.view(torch.int16).numpy().tobytes()


def bf16_vector(stream: int, n: int, amplitude: float) -> torch.Tensor:
    return hash_uniform(stream, n, amplitude).to(torch.bfloat16)


def gathered_rows() -> torch.Tensor:
    """[T][16][90]: hashed code bytes, then five fp16 scales ROW_SCALE_MAG * (0.5 + |u|)."""
    n = T * HEADS
    codes = (hash_u32(S_ROW_CODES, n * 80) & 0xFF).to(torch.uint8).view(n, 80)
    mag = torch.tensor(ROW_SCALE_MAG, dtype=torch.float32)
    scales = (mag * (0.5 + hash_uniform(S_ROW_SCALES, n * 5, 1.0).abs())).to(torch.float16)
    raw = torch.cat([codes, scales.view(n, 5).view(torch.uint8)], dim=1)
    return raw.view(T, HEADS, ROW_BYTES)


class Rows(torch.nn.Module):
    """Stands in for the n-gram embedding: the hashing and the table gather are the host's (and
    tested there); the device starts from the gathered rows."""

    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, input_ids, past_key_values):
        return self.embeddings


def main():
    torch.manual_seed(0)
    cfg = Qwen4ExpTextConfig(
        hidden_size=HIDDEN,
        hc_count=STREAMS,
        ple_embed_dim=EMBED,
        ple_conv_kernel_size=CONV_KERNEL,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS // (NGRAM_SIZE - 1),
        rms_norm_eps=EPS,
        num_hidden_layers=2,
        layer_types=["linear_attention", "linear_attention"],
        ple_layer_ids=[1],
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        output_gate_type="sigmoid",
        eos_token_id=248044,
        vocab_size=248320,
    )
    with torch.device("meta"):
        layer = Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=0)

    rows = gathered_rows()
    embeddings = torch.from_numpy(table.decode_rows(rows.view(-1, ROW_BYTES).numpy()).numpy())
    embeddings = embeddings.to(torch.bfloat16).view(1, T, EMBED)
    layer.ple_embedding = Rows(embeddings)

    width = STREAMS * HIDDEN
    weights = {
        "key_proj.weight": fp8.decode(make_fp8(S_KEY, width, EMBED, KEY_SCALE_MAG), (width, EMBED)),
        "value_proj.weight": fp8.decode(make_fp8(S_VALUE, HIDDEN, EMBED, VALUE_SCALE_MAG), (HIDDEN, EMBED)),
        "norm_key.weight": bf16_vector(S_NORM_KEY, width, NORM_AMP),
        "norm_query.weight": bf16_vector(S_NORM_QUERY, width, NORM_AMP),
        "norm_conv.weight": bf16_vector(S_NORM_CONV, width, NORM_AMP),
        "conv1d.weight": bf16_vector(S_CONV, width * CONV_KERNEL, CONV_AMP).view(width, 1, CONV_KERNEL),
    }
    layer = layer.to_empty(device="cpu")
    missing, unexpected = layer.load_state_dict(weights, strict=False, assign=True)
    missing = [m for m in missing if not m.startswith("ple_embedding.")]
    if missing or unexpected:
        raise RuntimeError(f"PLE weights: missing {missing}, unexpected {unexpected}")
    layer = layer.to(torch.bfloat16).eval()
    layer.ple_embedding = Rows(embeddings)

    hidden = bf16_vector(S_HIDDEN, T * width, HIDDEN_AMP).view(1, T, width)
    with torch.no_grad():
        out = layer(hidden, input_ids=None, past_key_values=None)
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (1, T, width):
        raise RuntimeError(f"PLE output {out.dtype} {tuple(out.shape)}")

    path = os.path.join(HERE, "ngram_ref.bin")
    with open(path, "wb") as f:
        f.write(b"IGNFX001")

        def put(name, code, dims, payload):
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<II", code, len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<Q", len(payload)))
            f.write(payload)

        put("rows", 0, [T, HEADS, ROW_BYTES], rows.contiguous().numpy().tobytes())
        put("ple_out", 5, [T, width], out.view(T, width).contiguous().view(torch.int16).numpy().tobytes())

    import transformers

    provenance = {
        "recorded": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "oracle": "transformers Qwen4ExpTextPLELayer.forward, bf16, CPU, past_key_values=None",
        "tokens": T,
        "streams": {
            "key_proj": S_KEY, "value_proj": S_VALUE, "norm_key": S_NORM_KEY, "norm_query": S_NORM_QUERY,
            "norm_conv": S_NORM_CONV, "conv": S_CONV, "hidden": S_HIDDEN,
            "row_codes": S_ROW_CODES, "row_scales": S_ROW_SCALES,
        },
        "magnitudes": {
            "key_scale": KEY_SCALE_MAG, "value_scale": VALUE_SCALE_MAG, "norm": NORM_AMP,
            "conv": CONV_AMP, "hidden": HIDDEN_AMP, "row_scale": ROW_SCALE_MAG,
        },
        "ple_out_abs_max": float(out.float().abs().max()),
        "ple_out_abs_mean": float(out.float().abs().mean()),
    }
    with open(os.path.join(HERE, "provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(provenance, f, indent=1)
        f.write("\n")
    print(f"wrote {path}: {T} tokens, |ple_out| max {provenance['ple_out_abs_max']:.4f} "
          f"mean {provenance['ple_out_abs_mean']:.4f}")


if __name__ == "__main__":
    main()
