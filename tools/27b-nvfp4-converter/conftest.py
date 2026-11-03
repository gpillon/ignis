"""pytest: the tool's modules are flat files beside the tests; the synthetic sources they share.

No test touches the GPU: CUDA is hidden from this process before torch is imported (ADR 0006), and every
command runs with `--device cpu`. `CUDA_VISIBLE_DEVICES` is `-1`, not empty: Windows drops an empty variable,
and flash-linear-attention, which transformers' Qwen3.5 model imports, initializes CUDA on import.

`make_sources` builds, in a temporary directory, a reduced 27B checkpoint at the fixture geometry
(`inventory.Geometry.fixture()`, the packer's `--geometry fixture`): every tensor of the 27B's checkpoint
with its name, BF16, at the fixture's widths, in three shards whose split cuts through layers, with its
index, its configuration (a model transformers builds and runs on the CPU), the six frontend files (a
word-level tokenizer of 256 ids), and Hugging Face download records at one revision; the base's drafter at
those widths, at the revision the fixture pins; and a ranking of 256 ids.

The session's `built` (those sources, their calibration and the work tree converted from them) and `packed`
(that tree packed by the real packer) are shared by the converter's tests and the comparison's, with the
oracles they check against (`decode_nvfp4`, `bf16_parent`) and the handle counter (`handles`).
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import checkpoint  # noqa: E402
import container  # noqa: E402
import convert  # noqa: E402
import inventory  # noqa: E402
import shards  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def no_gpu():
    yield
    assert not torch.cuda.is_initialized(), "a test initialized CUDA"

REVISION = "b4c84d42903a8646b25857eb2827288d92ed85a4"
REPO = "someone/fine-tune"
LICENSE = "swift-open-license-1.0"
WEIGHTS_ID = "nvfp4full-swift-1.5"
G = inventory.Geometry.fixture()
SHARDS = 3


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download_records(root, names, revision):
    """The Hugging Face download records of `names` under `root`: a safetensors file or the tokenizer is
    an LFS file, whose etag is its sha256; any other's is its git blob id."""
    meta = os.path.join(root, ".cache", "huggingface", "download")
    os.makedirs(meta, exist_ok=True)
    for name in names:
        with open(os.path.join(root, name), "rb") as f:
            data = f.read()
        lfs = name.endswith(".safetensors") or name == "tokenizer.json"
        etag = hashlib.sha256(data).hexdigest() if lfs else git_blob(data)
        with open(os.path.join(meta, name + ".metadata"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{revision}\n{etag}\n1792534371.28\n")


def write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False)


def _values(name, shape, rng):
    if name.endswith("A_log"):
        t = torch.rand(shape, generator=rng)
    elif "norm" in name and len(shape) == 1:
        t = 1.0 + 0.1 * torch.randn(shape, generator=rng)
    else:
        t = 0.08 * torch.randn(shape, generator=rng)
    return t.to(torch.bfloat16)


def checkpoint_config():
    """A config.json transformers builds at the fixture's widths, with the 27B's members."""
    expected = checkpoint.checkpoint_config(G)
    text = dict(expected["text_config"], model_type="qwen3_5_text", hidden_act="silu", partial_rotary_factor=0.25,
                dtype="bfloat16", bos_token_id=1, eos_token_id=1, pad_token_id=None, attention_bias=False,
                attention_dropout=0.0, output_gate_type="swish", initializer_range=0.02, use_cache=True)
    text["rope_parameters"] = dict(text["rope_parameters"], mrope_interleaved=True, partial_rotary_factor=0.25,
                                   rope_type="default")
    vision = dict(expected["vision_config"], model_type="qwen3_5", deepstack_visual_indexes=[],
                  hidden_act="gelu_pytorch_tanh", initializer_range=0.02)
    return dict({k: v for k, v in expected.items() if not isinstance(v, dict)}, language_model_only=False,
                image_token_id=252, video_token_id=253, vision_start_token_id=250, vision_end_token_id=251,
                text_config=text, vision_config=vision, transformers_version="5.17.0")


def tokenizer_files():
    """A word-level tokenizer of 256 ids: [UNK], the corpus's 249 most frequent words and 6 special tokens."""
    import calibrate
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = {}
    for w in pre_tokenizers.Whitespace().pre_tokenize_str(calibrate.corpus()):
        words[w[0]] = words.get(w[0], 0) + 1
    common = sorted(words, key=lambda w: (-words[w], w))[:249]
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(common)}}
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(specials)
    config = {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]", "eos_token": "<|im_end|>",
              "added_tokens_decoder": {str(250 + i): {"content": s, "special": True} for i, s in enumerate(specials)}}
    return tok.to_str(), config


def make_checkpoint(root, seed=468, revision=REVISION, tensors=None):
    """The reduced checkpoint under `root`. `tensors` (name -> shape) replaces the 27B's names and shapes."""
    os.makedirs(root)
    rng = torch.Generator().manual_seed(seed)
    tensors = checkpoint.checkpoint_tensors(G) if tensors is None else tensors
    names = sorted(tensors)
    per = -(-len(names) // SHARDS)
    weight_map = {}
    for s in range(SHARDS):
        shard = f"model-{s + 1:05d}-of-{SHARDS:05d}.safetensors"
        part = names[s * per:(s + 1) * per]
        save_file({n: _values(n, tensors[n], rng) for n in part}, os.path.join(root, shard))
        weight_map.update({n: shard for n in part})
    write_json(os.path.join(root, "model.safetensors.index.json"), {"metadata": {}, "weight_map": weight_map})
    write_json(os.path.join(root, "config.json"), checkpoint_config())
    tokenizer, tokenizer_config = tokenizer_files()
    with open(os.path.join(root, "tokenizer.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(tokenizer)
    write_json(os.path.join(root, "tokenizer_config.json"), tokenizer_config)
    with open(os.path.join(root, "chat_template.jinja"), "w", encoding="utf-8", newline="\n") as f:
        f.write("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}")
    write_json(os.path.join(root, "generation_config.json"), {"eos_token_id": 252, "temperature": 0.6})
    write_json(os.path.join(root, "preprocessor_config.json"), {"patch_size": G.vision_patch_size})
    write_json(os.path.join(root, "video_preprocessor_config.json"), {"patch_size": G.vision_patch_size})
    download_records(root, sorted(f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))), revision)
    return root


def drafter_config_json():
    return dict(checkpoint.drafter_config(G), model_type="qwen3", dtype="bfloat16", hidden_act="silu",
                attention_bias=False, use_sliding_window=True, max_window_layers=5)


def make_drafter(root, seed=467, config=None):
    os.makedirs(root)
    rng = torch.Generator().manual_seed(seed)
    tensors = checkpoint.drafter_tensors(G)
    save_file({n: _values(n, s, rng) for n, s in sorted(tensors.items())}, os.path.join(root, "model.safetensors"))
    write_json(os.path.join(root, "config.json"), config or drafter_config_json())
    download_records(root, ["config.json", "model.safetensors"], checkpoint.PINNED["fixture"]["drafter"][1])
    return root


def make_ranking(path, seed=466):
    rng = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 1 << 20, (6, G.vocab), generator=rng, dtype=torch.int64)
    counts.numpy().astype("<i8").tofile(path)
    return path


def make_sources(root):
    """{checkpoint, drafter, ranking}: the synthetic sources under `root`."""
    return {"checkpoint": make_checkpoint(os.path.join(root, "checkpoint")),
            "drafter": make_drafter(os.path.join(root, "drafter")),
            "ranking": make_ranking(os.path.join(root, "ranking.train.counts.i64"))}


def calibrate_args(src, out):
    return ["calibrate", "--checkpoint", src["checkpoint"], "--out", out, "--device", "cpu", "--geometry", "fixture"]


def run_args(src, calibration, work_dir, weights_id=WEIGHTS_ID):
    return ["run", "--checkpoint", src["checkpoint"], "--source-repo", REPO, "--source-revision", REVISION,
            "--license", LICENSE, "--dflash2", src["drafter"], "--ranking", src["ranking"],
            "--calibration", calibration, "--weights-id", weights_id, "--work", work_dir,
            "--device", "cpu", "--geometry", "fixture"]


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """The sources, their calibration and a work tree converted from them."""
    root = str(tmp_path_factory.mktemp("built"))
    src = make_sources(root)
    src["calibration"] = os.path.join(root, "calibration.json")
    assert convert.main(calibrate_args(src, src["calibration"])) == 0
    src["work"] = os.path.join(root, "work")
    assert convert.main(run_args(src, src["calibration"], src["work"])) == 0
    return src


def checkpoint_tensors(root):
    out = {}
    for name in sorted(os.listdir(root)):
        if name.endswith(".safetensors"):
            out.update(load_file(os.path.join(root, name)))
    return out


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def decode_nvfp4(payload, n, k):
    """The values of a `blockscale-k16-m128x4-v1` payload (contract §3), and its weight divisor."""
    codes = torch.frombuffer(bytearray(payload[: n * k // 2]), dtype=torch.uint8).reshape(n, k // 2).to(torch.int64)
    nibbles = torch.stack((codes & 0xF, codes >> 4), dim=2).reshape(n, k)
    offset = -(-(n * k // 2) // 256) * 256
    stored = torch.frombuffer(bytearray(payload[offset: offset + n * k // 16]), dtype=torch.uint8)
    scales = stored.reshape(n // 128, k // 64, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(n, k // 16)
    (divisor,) = struct.unpack("<f", payload[-4:])
    values = torch.tensor(E2M1)[nibbles].reshape(n, k // 16, 16)
    return (values * scales.view(torch.float8_e4m3fn).float().unsqueeze(-1) / divisor).reshape(n, k), divisor


def bf16_parent(tensors, name):
    """A text NVFP4 parent's BF16 rows, fused as the contract's §4.2 states."""
    _, _, layer, block, suffix = name.split("/")
    p = f"model.language_model.layers.{layer}."
    if block == "mlp":
        return torch.cat([tensors[p + "mlp.gate_proj.weight"], tensors[p + "mlp.up_proj.weight"]]) \
            if suffix == "gate_up" else tensors[p + "mlp.down_proj.weight"]
    if block == "gdn":
        return torch.cat([tensors[p + "linear_attn.in_proj_qkv.weight"], tensors[p + "linear_attn.in_proj_z.weight"]]) \
            if suffix == "query_key_value_z" else tensors[p + "linear_attn.out_proj.weight"]
    if suffix == "output":
        return tensors[p + "self_attn.o_proj.weight"]
    q = tensors[p + "self_attn.q_proj.weight"].reshape(G.attention_heads, 2, G.head_dim, G.hidden)
    return torch.cat([q[:, 0].reshape(-1, G.hidden), tensors[p + "self_attn.k_proj.weight"],
                      q[:, 1].reshape(-1, G.hidden), tensors[p + "self_attn.v_proj.weight"]])


ARTIFACT = "qwen3_8_27b_nvfp4full-v2-ukisai-swift-1.5.ninfer"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


@pytest.fixture(scope="session")
def packed(built, tmp_path_factory):
    """The work tree packed by the `qwen3.8-27b-nvfp4` family at the fixture geometry."""
    exe = packer()
    if exe is None:
        pytest.skip("no ignis-artifact-pack: set IGNIS_ARTIFACT_PACK or build it "
                    "(cargo build -p ignis-artifact --bin ignis-artifact-pack)")
    root = str(tmp_path_factory.mktemp("packed"))
    work_dir = os.path.join(root, "work")
    shutil.copytree(built["work"], work_dir)
    out = os.path.join(root, ARTIFACT)
    done = subprocess.run([exe, "--family", "qwen3.8-27b-nvfp4", "--work", work_dir, "--out", out,
                           "--weights-id", WEIGHTS_ID, "--geometry", "fixture"], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return out


def corrupt(artifact, tmp_path, name, at=0):
    """A copy of `artifact` with one byte of `name`'s payload flipped."""
    copy = str(tmp_path / ARTIFACT)
    shutil.copyfile(artifact, copy)
    offset, _ = container.Container(copy).span(name)
    with open(copy, "r+b") as f:
        f.seek(offset + at)
        byte = f.read(1)
        f.seek(offset + at)
        f.write(bytes([byte[0] ^ 0x01]))
    return copy


@pytest.fixture
def handles(monkeypatch):
    """Every safetensors handle the tool opens, counted: the most open at once and the files opened."""
    real = shards.safe_open
    state = {"open": 0, "peak": 0, "opened": []}

    class Counted:
        def __init__(self, path, **kwargs):
            self.inner = real(path, **kwargs)
            self.path = path

        def __enter__(self):
            state["open"] += 1
            state["peak"] = max(state["peak"], state["open"])
            state["opened"].append(os.path.basename(self.path))
            return self.inner.__enter__()

        def __exit__(self, *exc):
            state["open"] -= 1
            return self.inner.__exit__(*exc)

    monkeypatch.setattr(shards, "safe_open", Counted)
    return state


ALL_FILES = {"model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
             "model-00003-of-00003.safetensors", "model.safetensors"}
