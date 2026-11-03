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
"""
import hashlib
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import checkpoint  # noqa: E402
import inventory  # noqa: E402


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
