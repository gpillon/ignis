"""Record the n-gram ids Flash-Next's own hashing code gives fixed token streams.

The fixture `crates/core/tests/fixtures/flash_next/ngram_ids.json` is what
`ignis_core::ngram` is checked against bit for bit (spec flash-next/04,
"n-gram hashing bit-exact against a fixture of ids recorded from the
checkpoint's hashing code on fixed token streams").

What runs is the checkpoint's modeling code as transformers ships it
(`Qwen4ExpTextNGramEmbedding`, the module the PLE layer of decoder layer 1
calls), on CPU, with:
- the hash buffers the checkpoint itself stores (`layer_multipliers`,
  `ngram_heads_vocab_sizes`, `ngram_heads_offsets` of
  `model.language_model.layers.1.ple.ple_embedding`), read by HTTP range
  requests at the pinned revision -- 280 bytes, nothing else of the
  checkpoint is downloaded. The buffers the config would rebuild are recorded
  beside them, with whether the two agree;
- the embedding table replaced by an identity that returns the row ids, so
  the 51 GB table is never built (the module is constructed on the meta
  device, as the compression study did);
- every stream hashed whole, then again chunk by chunk through the real
  `DynamicCache` the text model builds (`DynamicCache(config=text_config)`),
  which is the decode path: the ids must not depend on how a stream is cut.

Run (CPU only, no GPU, no exllamav3):
    F:/ai/ngram-venv/Scripts/python.exe tools/flash-next-fixtures/record_ngram_ids.py
"""
import json
import os
import random
import struct
import sys

import httpx
import torch
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as mq
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PREFIX = "model.language_model.layers.1.ple.ple_embedding."
# Where the checkpoint's index puts each buffer (`model.safetensors.index.json`
# at the revision); the offsets are read from each file's own header below.
BUFFERS = {
    "layer_multipliers": "model-00005-of-00131.safetensors",
    "ngram_heads_vocab_sizes": "model-00037-of-00131.safetensors",
    "ngram_heads_offsets": "model-00037-of-00131.safetensors",
}
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO_ROOT, "crates", "core", "tests", "fixtures", "flash_next", "ngram_ids.json")
CHUNK_SIZES = (1, 3, 5)


def url(filename):
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}"


def read_range(client, filename, start, end):
    r = client.get(url(filename), headers={"Range": f"bytes={start}-{end - 1}"})
    r.raise_for_status()
    if len(r.content) != end - start:
        raise IOError(f"{filename} [{start}, {end}): got {len(r.content)} bytes")
    return r.content


def stored_buffers():
    """The three hash buffers as the checkpoint stores them, at REVISION."""
    out = {}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        headers = {}
        for name, filename in BUFFERS.items():
            if filename not in headers:
                n = struct.unpack("<Q", read_range(client, filename, 0, 8))[0]
                headers[filename] = (8 + n, json.loads(read_range(client, filename, 8, 8 + n)))
            base, header = headers[filename]
            entry = header[PREFIX + name]
            if entry["dtype"] != "I64":
                raise ValueError(f"{name}: dtype {entry['dtype']}, expected I64")
            start, end = entry["data_offsets"]
            raw = read_range(client, filename, base + start, base + end)
            out[name] = list(struct.unpack(f"<{(end - start) // 8}q", raw))
    return out


class RowIds(torch.nn.Module):
    """Stands in for the embedding table: returns the row ids it is asked for."""

    def __init__(self):
        super().__init__()
        self.weight = torch.empty(0)

    def forward(self, ids):
        return ids[..., None]


def text_config():
    """The checkpoint's text config: the `text_config` of its `config.json` at REVISION."""
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        r = client.get(url("config.json"))
        r.raise_for_status()
        return Qwen4ExpTextConfig(**r.json()["text_config"])


def hashing_module(cfg, stored):
    with torch.device("meta"):
        module = mq.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx=1, ple_layer_index=0)
    computed = {
        "layer_multipliers": mq._build_layer_multipliers(
            cfg.vocab_size, cfg.ngram_size, module.ple_layer_index, cfg.seed
        ).tolist(),
        "ngram_heads_vocab_sizes": list(module.head_vocab_sizes),
        "ngram_heads_offsets": list(module.head_offsets),
    }
    for name, values in stored.items():
        module._buffers[name] = torch.tensor(values, dtype=torch.long)
    module.ngram_embedding = RowIds()
    return module, computed


def ids_whole(module, tokens):
    return module(torch.tensor([tokens], dtype=torch.long), None)[0].tolist()


def ids_chunked(module, cfg, tokens, size):
    cache = DynamicCache(config=cfg)
    out = []
    for start in range(0, len(tokens), size):
        chunk = torch.tensor([tokens[start:start + size]], dtype=torch.long)
        out.extend(module(chunk, cache)[0].tolist())
    return out


def streams(vocab, eos):
    rng = random.Random(302)
    plain = [rng.randrange(vocab) for _ in range(48)]
    out = [
        ("no-eos", plain),
        ("extreme-ids", [0, vocab - 1, 0, 1, vocab - 1, vocab - 2, 0, 0, vocab - 1, 7]),
        ("one-token", [plain[0]]),
        ("two-tokens", plain[:2]),
        ("eos-first", [eos] + plain[:12]),
        ("eos-middle", plain[:9] + [eos] + plain[9:20]),
        ("eos-last", plain[:11] + [eos]),
        ("eos-twice", plain[:6] + [eos, eos] + plain[6:14]),
        ("eos-alternating", [t for pair in zip(plain[:8], [eos] * 8) for t in pair]),
        ("eos-only", [eos] * 5),
    ]
    long = []
    for _ in range(320):
        long.append(eos if rng.random() < 0.05 else rng.randrange(vocab))
    out.append(("long-mixed", long))
    return out


def render(fixture):
    """JSON with one line per header field and one line per token's ids."""
    def compact(value):
        return json.dumps(value, separators=(",", ":"))

    lines = ["{"]
    for key, value in fixture.items():
        if key != "streams":
            lines.append(f" {json.dumps(key)}: {compact(value)},")
    lines.append(' "streams": [')
    for i, stream in enumerate(fixture["streams"]):
        rows = ",\n   ".join(compact(row) for row in stream["ids"])
        lines.append("  {")
        lines.append(f'   "name": {json.dumps(stream["name"])},')
        lines.append(f'   "tokens": {compact(stream["tokens"])},')
        lines.append(f'   "chunk_sizes_equal_to_whole": {compact(stream["chunk_sizes_equal_to_whole"])},')
        lines.append(f'   "ids": [\n   {rows}\n   ]')
        lines.append("  }" + ("," if i + 1 < len(fixture["streams"]) else ""))
    lines.append(" ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    cfg = text_config()
    eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
    stored = stored_buffers()
    module, computed = hashing_module(cfg, stored)
    divisor = cfg.make_ngram_vocab_size_divisible_by
    records = []
    for name, tokens in streams(cfg.vocab_size, eos):
        whole = ids_whole(module, tokens)
        equal = []
        for size in CHUNK_SIZES:
            chunked = ids_chunked(module, cfg, tokens, size)
            if chunked != whole:
                sys.exit(f"{name}: chunks of {size} hash differently from the whole stream")
            equal.append(size)
        records.append({"name": name, "tokens": tokens, "chunk_sizes_equal_to_whole": equal, "ids": whole})
    fixture = {
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "buffers": {name: f"{BUFFERS[name]}:{PREFIX}{name}" for name in BUFFERS},
            "modeling_code": f"transformers {transformers.__version__} "
                             "models/qwen4_exp/modeling_qwen4_exp.py Qwen4ExpTextNGramEmbedding",
            "recorder": "tools/flash-next-fixtures/record_ngram_ids.py",
        },
        "layout": "ids[t] holds token t's 16 row ids: heads 0-7 hash its 2-gram, heads 8-15 its "
                  "3-gram; each id is (hash mod ngram_heads_vocab_sizes[h]) + ngram_heads_offsets[h]",
        "config": {
            "vocab_size": cfg.vocab_size,
            "ngram_size": cfg.ngram_size,
            "heads_per_ngram": cfg.heads_per_ngram,
            "ngram_vocab_size_base": cfg.ngram_vocab_size_base,
            "make_ngram_vocab_size_divisible_by": cfg.make_ngram_vocab_size_divisible_by,
            "seed": cfg.seed,
            "eos_token_id": eos,
            "ple_layer_index": module.ple_layer_index,
        },
        "stored": stored,
        "computed": computed,
        "stored_equals_computed": stored == computed,
        "table_rows": module.total_vocab_size,
        "padded_table_rows": -(-module.total_vocab_size // divisor) * divisor,
        "streams": records,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="\n") as f:
        f.write(render(fixture))
    print(f"wrote {OUT}: {len(records)} streams, {sum(len(r['tokens']) for r in records)} tokens, "
          f"stored buffers {'equal' if fixture['stored_equals_computed'] else 'DIFFER from'} the computed ones")


if __name__ == "__main__":
    main()
