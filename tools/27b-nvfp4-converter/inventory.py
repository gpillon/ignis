"""The 27B NVFP4 container's objects, generated from a geometry (docs/formats/27b-nvfp4-work-tree.md §4, §5).

The authority is the packer's `ignis_artifact::nvfp4_27b::inventory`; this module states the same list
for the converter, which writes one work file per object. `Geometry.qwen3_8_27b()` is the 27B's widths,
`Geometry.fixture()` the reduced synthetic checkpoint the tests build (the packer's `--geometry fixture`).
Both have every object of the served `qwen3_8_27b_nvfp4full-v2.ninfer`, with the same names, formats,
layouts and order; only the widths differ. The layer counts and the positional BF16 exceptions are the
27B's at both.
"""
from dataclasses import dataclass

MODEL_ID = "qwen3.8-27b"
UNITS = ("frontend", "text", "mtp", "vision", "dflash2")
FRONTEND_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                  "preprocessor_config.json", "video_preprocessor_config.json")
TEXT_LAYERS = 64
VISION_LAYERS = 27
DFLASH2_LAYERS = 5
# The positional BF16 exceptions (the Qwen3.6 pattern the served recipe applies; ninfer
# `inventory_nvfp4full.py`, ignis `crates/artifact/src/inventory.rs`).
QKGV_BF16_LAYERS = (3, 7, 11, 15, 19, 23)
ATTENTION_OUT_BF16_LAYERS = (3, 7)
GDN_OUT_BF16_LAYERS = (4,)

BF16, FP32, I32 = "BF16", "FP32", "I32"
W8, Q4, Q5, Q6 = "W8G32_F16S", "Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S"
NVFP4 = "NVFP4"
CONTIGUOUS = "contiguous-le-v1"
ROW_SPLIT = "row-split-k128-v1"
BLOCK_SCALE = "blockscale-k16-m128x4-v1"
RAW_BYTES = "raw-bytes-v1"
LAYOUT_OF = {BF16: CONTIGUOUS, FP32: CONTIGUOUS, I32: CONTIGUOUS,
             W8: ROW_SPLIT, Q4: ROW_SPLIT, Q5: ROW_SPLIT, Q6: ROW_SPLIT, NVFP4: BLOCK_SCALE}


@dataclass(frozen=True)
class Geometry:
    """Every width the inventory is generated from (the Rust `Geometry`'s fields)."""
    name: str
    hidden: int
    vocab: int
    intermediate: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_head_dim: int
    gdn_conv_kernel: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    draft_head_rows: int
    vision_hidden: int
    vision_intermediate: int
    vision_patch_size: int
    vision_positions: int
    dflash2_heads: int
    dflash2_kv_heads: int
    dflash2_head_dim: int
    dflash2_taps: int
    dflash2_conv_rows: int
    dflash2_selector_rank: int

    @staticmethod
    def qwen3_8_27b():
        return Geometry("qwen3.8-27b", hidden=5120, vocab=248320, intermediate=17408, gdn_key_heads=16,
                        gdn_value_heads=48, gdn_head_dim=128, gdn_conv_kernel=4, attention_heads=24, kv_heads=4,
                        head_dim=256, draft_head_rows=131072, vision_hidden=1152, vision_intermediate=4304,
                        vision_patch_size=16, vision_positions=48 * 48, dflash2_heads=32, dflash2_kv_heads=8,
                        dflash2_head_dim=128, dflash2_taps=5, dflash2_conv_rows=1280, dflash2_selector_rank=256)

    @staticmethod
    def fixture():
        return Geometry("fixture", hidden=128, vocab=256, intermediate=64, gdn_key_heads=1, gdn_value_heads=3,
                        gdn_head_dim=64, gdn_conv_kernel=4, attention_heads=4, kv_heads=2, head_dim=32,
                        draft_head_rows=128, vision_hidden=32, vision_intermediate=64, vision_patch_size=4,
                        vision_positions=4 * 4, dflash2_heads=2, dflash2_kv_heads=1, dflash2_head_dim=32,
                        dflash2_taps=5, dflash2_conv_rows=128, dflash2_selector_rank=128)

    @staticmethod
    def named(name):
        for g in (Geometry.qwen3_8_27b(), Geometry.fixture()):
            if g.name == name:
                return g
        raise ValueError(f"no geometry {name!r}: qwen3.8-27b or fixture")

    @property
    def vision_patch(self):
        """The patch embedding's columns: channels x temporal patch x patch²."""
        return 3 * 2 * self.vision_patch_size ** 2

    @property
    def gdn_conv_channels(self):
        return 2 * self.gdn_key_heads * self.gdn_head_dim + self.gdn_value_heads * self.gdn_head_dim

    @property
    def gdn_value_width(self):
        return self.gdn_value_heads * self.gdn_head_dim

    @property
    def qkgv_rows(self):
        return 2 * self.attention_heads * self.head_dim + 2 * self.kv_heads * self.head_dim

    @property
    def qkvz_rows(self):
        return self.gdn_conv_channels + self.gdn_value_width


def is_gqa(layer):
    return (layer + 1) % 4 == 0


@dataclass(frozen=True)
class Obj:
    """One container object: a resource (format and layout None, shape None) or a tensor."""
    name: str
    format: str = None
    shape: tuple = None

    @property
    def unit(self):
        return self.name.split("/", 1)[0]

    @property
    def kind(self):
        return "resource" if self.format is None else "tensor"

    @property
    def layout(self):
        return RAW_BYTES if self.format is None else LAYOUT_OF[self.format]


def divisor_name(parent):
    """The activation divisor of a text NVFP4 parent (`<site>/input_scale_divisor`)."""
    prefix, suffix = parent.rsplit("/", 1)
    site = {"query_key_gate_value": "input_projection", "query_key_value_z": "input_projection",
            "output": "output_projection", "gate_up": "gate_up_projection", "down": "down_projection"}[suffix]
    return f"{prefix}/{site}/input_scale_divisor"


def objects(g):
    """Every object of the container, in its order."""
    out = [Obj(f"frontend/{f}") for f in FRONTEND_FILES]
    H, I = g.hidden, g.intermediate

    def text(name, fmt, shape):
        out.append(Obj(name, fmt, tuple(shape)))
        if fmt == NVFP4:
            out.append(Obj(divisor_name(name), FP32, ()))

    text("text/token_embedding", W8, (g.vocab, H))
    for i in range(TEXT_LAYERS):
        p = f"text/layers/{i}/"
        text(p + "input_norm", BF16, (H,))
        if is_gqa(i):
            text(p + "attention/query_key_gate_value", BF16 if i in QKGV_BF16_LAYERS else NVFP4, (g.qkgv_rows, H))
            text(p + "attention/query_norm", BF16, (g.head_dim,))
            text(p + "attention/key_norm", BF16, (g.head_dim,))
            text(p + "attention/output", BF16 if i in ATTENTION_OUT_BF16_LAYERS else NVFP4,
                 (H, g.attention_heads * g.head_dim))
        else:
            text(p + "gdn/a_log", FP32, (g.gdn_value_heads,))
            text(p + "gdn/dt_bias", FP32, (g.gdn_value_heads,))
            text(p + "gdn/convolution", BF16, (g.gdn_conv_kernel, g.gdn_conv_channels))
            text(p + "gdn/a_b_projection", BF16, (2 * g.gdn_value_heads, H))
            text(p + "gdn/query_key_value_z", NVFP4, (g.qkvz_rows, H))
            text(p + "gdn/norm", BF16, (g.gdn_head_dim,))
            text(p + "gdn/output", BF16 if i in GDN_OUT_BF16_LAYERS else NVFP4, (H, g.gdn_value_width))
        text(p + "post_attention_norm", BF16, (H,))
        text(p + "mlp/gate_up", NVFP4, (2 * I, H))
        text(p + "mlp/down", NVFP4, (H, I))
    text("text/final_norm", BF16, (H,))
    text("text/output_head", W8, (g.vocab, H))
    text("text/draft_head", Q4, (g.draft_head_rows, H))
    text("text/draft_head_token_ids", I32, (g.draft_head_rows,))

    A, D, Kv = g.attention_heads, g.head_dim, g.kv_heads
    for name, fmt, shape in (
            ("mtp/input_projection", W8, (H, 2 * H)),
            ("mtp/embedding_norm", BF16, (H,)),
            ("mtp/hidden_norm", BF16, (H,)),
            ("mtp/layer/input_norm", BF16, (H,)),
            ("mtp/layer/attention/query_key_gate_value", W8, (g.qkgv_rows, H)),
            ("mtp/layer/attention/query_norm", BF16, (D,)),
            ("mtp/layer/attention/key_norm", BF16, (D,)),
            ("mtp/layer/attention/output", W8, (H, A * D)),
            ("mtp/layer/post_attention_norm", BF16, (H,)),
            ("mtp/layer/mlp/gate_up", W8, (2 * I, H)),
            ("mtp/layer/mlp/down", W8, (H, I)),
            ("mtp/final_norm", BF16, (H,))):
        out.append(Obj(name, fmt, shape))

    vH, vI = g.vision_hidden, g.vision_intermediate
    out += [Obj("vision/patch_embedding", Q6, (vH, g.vision_patch)),
            Obj("vision/patch_embedding_bias", BF16, (vH,)),
            Obj("vision/position_embedding", BF16, (g.vision_positions, vH))]
    for b in range(VISION_LAYERS):
        p = f"vision/layers/{b}/"
        out += [Obj(p + "attention/qkv", Q4, (3 * vH, vH)), Obj(p + "attention/qkv_bias", BF16, (3 * vH,)),
                Obj(p + "attention/output", Q5, (vH, vH)), Obj(p + "attention/output_bias", BF16, (vH,)),
                Obj(p + "mlp/fc1", Q4, (vI, vH)), Obj(p + "mlp/fc1_bias", BF16, (vI,)),
                Obj(p + "mlp/fc2", Q5, (vH, vI)), Obj(p + "mlp/fc2_bias", BF16, (vH,))]
        out += [Obj(p + n, BF16, (vH,)) for n in ("norm1/weight", "norm1/bias", "norm2/weight", "norm2/bias")]
    out += [Obj("vision/merger/fc1", W8, (4 * vH, 4 * vH)), Obj("vision/merger/fc1_bias", BF16, (4 * vH,)),
            Obj("vision/merger/fc2", W8, (H, 4 * vH)), Obj("vision/merger/fc2_bias", BF16, (H,)),
            Obj("vision/merger/norm/weight", BF16, (vH,)), Obj("vision/merger/norm/bias", BF16, (vH,))]

    dA, dKv, dD, dC = g.dflash2_heads, g.dflash2_kv_heads, g.dflash2_head_dim, g.dflash2_conv_rows
    out += [Obj("dflash2/feature_projection", NVFP4, (H, g.dflash2_taps * H)),
            Obj("dflash2/context_norm", BF16, (H,))]
    for l in range(DFLASH2_LAYERS):
        p = f"dflash2/layers/{l}/"
        out += [Obj(p + "input_norm", BF16, (H,)),
                Obj(p + "attention/query_key_value", NVFP4, ((dA + 2 * dKv) * dD, H)),
                Obj(p + "attention/query_norm", BF16, (dD,)), Obj(p + "attention/key_norm", BF16, (dD,)),
                Obj(p + "attention/output", NVFP4, (H, dA * dD)),
                Obj(p + "attention/conv_base", BF16, (2, 2, H)), Obj(p + "attention/conv_proj", NVFP4, (dC, H)),
                Obj(p + "post_attention_norm", BF16, (H,)),
                Obj(p + "mlp/gate_up", NVFP4, (2 * I, H)), Obj(p + "mlp/down", NVFP4, (H, I)),
                Obj(p + "mlp/conv_base", BF16, (2, 2, H)), Obj(p + "mlp/conv_proj", NVFP4, (dC, H))]
    out += [Obj("dflash2/final_norm", BF16, (H,)),
            Obj("dflash2/selector/hidden", NVFP4, (g.dflash2_selector_rank, H)),
            Obj("dflash2/selector/predecessor", NVFP4, (g.vocab, g.dflash2_selector_rank)),
            Obj("dflash2/selector/successor", NVFP4, (g.vocab, g.dflash2_selector_rank))]
    return out


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


GROUP = {W8: (32, 8), Q4: (64, 4), Q5: (64, 5), Q6: (64, 6)}
WORD = {BF16: 2, FP32: 4, I32: 4}


def row_split_geometry(fmt, shape):
    """(k_pad, groups per row, base, high and scale row bytes, high and scale plane offsets, payload bytes)."""
    n, k = shape
    group, bits = GROUP[fmt]
    k_pad = align_up(k, 128)
    groups = k_pad // group
    base_row = groups * (group if bits == 8 else group // 2)
    high_row = groups * (0 if bits in (4, 8) else group * (bits - 4) // 8)
    scale_row = groups * 2
    high_offset = align_up(n * base_row, 256)
    scale_offset = high_offset + align_up(n * high_row, 256)
    return k_pad, groups, base_row, high_row, scale_row, high_offset, scale_offset, scale_offset + n * scale_row


def block_scale_geometry(shape):
    """(scale plane offset, weight divisor offset, payload bytes) of an NVFP4 `[N, K]` payload."""
    n, k = shape
    if n % 128 or k % 64:
        raise ValueError(f"NVFP4 {list(shape)} is not whole 128 x 64 blocks")
    scale_offset = align_up(n * k // 2, 256)
    divisor_offset = scale_offset + n * k // 16
    return scale_offset, divisor_offset, divisor_offset + 4


def encoded_size(obj):
    """The bytes the container stores for a tensor object (`ignis_artifact::tensor_encoded_size`)."""
    if obj.format in WORD:
        size = WORD[obj.format]
        for d in obj.shape:
            size *= d
        return size
    if obj.format == NVFP4:
        return block_scale_geometry(obj.shape)[2]
    return row_split_geometry(obj.format, obj.shape)[-1]
