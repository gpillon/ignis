"""pytest: the tool is a flat file beside the tests; a synthetic checkpoint and a stand-in main container.

No test touches the GPU or the network. CUDA is hidden from this process before torch is imported, and the
builder reads the synthetic checkpoint through `LocalHub`, which answers the reads the hub answers (a small
file, a byte range of a shard) from a directory and counts them.

`make_checkpoint` writes, in a temporary directory, a checkpoint with Flash-Next's layout at the fixture
tower's widths (the packer's `--geometry fixture`, `ignis_artifact::flash_next::vision::fixture_geometry`):
the 333 `model.visual.*` tensors, BF16, in the first of two shards between a text tensor and an MTP one, its
index and its `config.json`. `write_main` writes a stand-in main container: a valid `.ninfer` v2 file of one
resource, which is all the pair pin reads of a main container.
"""
import hashlib
import json
import os
import struct
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import convert  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
# The fixture tower: the packer's `--geometry fixture`.
VH, VI, PATCH, POSITIONS, OUT = 32, 64, 4, 16, 64


def visual_shapes():
    """The checkpoint's `model.visual.*` tensors at the fixture widths, as the checkpoint stores them."""
    v = "model.visual."
    shapes = {v + "patch_embed.proj.weight": (VH, 3, 2, PATCH, PATCH), v + "patch_embed.proj.bias": (VH,),
              v + "pos_embed.weight": (POSITIONS, VH)}
    for b in range(27):
        p = f"{v}blocks.{b}."
        shapes.update({p + "attn.qkv.weight": (3 * VH, VH), p + "attn.qkv.bias": (3 * VH,),
                       p + "attn.proj.weight": (VH, VH), p + "attn.proj.bias": (VH,),
                       p + "mlp.linear_fc1.weight": (VI, VH), p + "mlp.linear_fc1.bias": (VI,),
                       p + "mlp.linear_fc2.weight": (VH, VI), p + "mlp.linear_fc2.bias": (VH,),
                       p + "norm1.weight": (VH,), p + "norm1.bias": (VH,), p + "norm2.weight": (VH,),
                       p + "norm2.bias": (VH,)})
    shapes.update({v + "merger.linear_fc1.weight": (4 * VH, 4 * VH), v + "merger.linear_fc1.bias": (4 * VH,),
                   v + "merger.linear_fc2.weight": (OUT, 4 * VH), v + "merger.linear_fc2.bias": (OUT,),
                   v + "merger.norm.weight": (VH,), v + "merger.norm.bias": (VH,)})
    return shapes


def visual_tensors(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {name: (torch.randn(shape, generator=g) * 0.05).to(torch.bfloat16)
            for name, shape in sorted(visual_shapes().items())}


def config(**vision):
    v = {"depth": 27, "hidden_size": VH, "intermediate_size": VI, "in_channels": 3, "patch_size": PATCH,
         "temporal_patch_size": 2, "spatial_merge_size": 2, "num_position_embeddings": POSITIONS,
         "out_hidden_size": OUT, "num_heads": 4}
    v.update(vision)
    return {"model_type": "fixture", "language_model_only": False, "text_config": {"hidden_size": OUT},
            "vision_config": v}


def make_checkpoint(root, tensors=None, cfg=None, visual_shard=0):
    """A checkpoint in `root`: the visual tensors in shard `visual_shard` between two others; returns them."""
    os.makedirs(root, exist_ok=True)
    tensors = visual_tensors() if tensors is None else tensors
    first = {"model.language_model.embed_tokens.weight": torch.ones((8, OUT), dtype=torch.bfloat16),
             "mtp.fc.weight": torch.full((4, 4), 2.0, dtype=torch.bfloat16)}
    second = {"model.language_model.norm.weight": torch.ones((OUT,), dtype=torch.bfloat16)}
    (first if visual_shard == 0 else second).update(tensors)
    weight_map = {}
    for shard, part in zip(SHARDS, (first, second)):
        save_file(part, os.path.join(root, shard))
        weight_map.update({name: shard for name in part})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(config() if cfg is None else cfg, f)
    return tensors


def write_main(path, weights_id="fixture-main"):
    """A stand-in main container: one 4096-byte resource under a Flash-Next identity."""
    directory = json.dumps({"identity": {"model_id": "qwen3.8-flash-next", "weights_id": weights_id},
                            "objects": [{"name": "frontend/config.json", "kind": "resource",
                                         "encoding": "raw-bytes-v1", "offset": 0, "bytes": 4096}]}).encode()
    start = (16 + len(directory) + 4095) // 4096 * 4096
    with open(path, "wb") as f:
        f.write(b"NINFER\x00\x02" + struct.pack("<Q", len(directory)) + directory)
        f.write(b"\x00" * (start - 16 - len(directory)))
        f.write(bytes(range(256)) * 16)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def packer():
    """The packer binary: IGNIS_ARTIFACT_PACK, else this checkout's build of it."""
    if os.environ.get("IGNIS_ARTIFACT_PACK"):
        return os.environ["IGNIS_ARTIFACT_PACK"]
    for build in ("debug", "release"):
        for target in (os.path.join("x86_64-pc-windows-msvc", build), build):
            for exe in ("ignis-artifact-pack.exe", "ignis-artifact-pack"):
                path = os.path.join(REPO_ROOT, "target", target, exe)
                if os.path.isfile(path):
                    return path
    return None


@pytest.fixture
def pack_exe():
    exe = packer()
    if exe is None:
        pytest.skip("no ignis-artifact-pack: set IGNIS_ARTIFACT_PACK or build it "
                    "(cargo build -p ignis-artifact --bin ignis-artifact-pack)")
    return exe
