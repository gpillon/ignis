"""The sources and their checks: a BF16 checkpoint of the 27B, the base's DFlash2 drafter, the frequency
ranking. Every check here runs before anything is written, and a failure is `Refused`, naming what failed.

- **The 27B's checkpoint**: exactly its tensors (names, shapes, BF16), and its configuration's members that
  the recipe and the calibration depend on (ninfer `qwen3_6_27b/convert.py` `validate_config`, by geometry).
- **The drafter**: `incoai/Qwen3.8-27B-DFlash2` at `dedf8df6`, trained for the base and used unchanged for
  every fine-tune: its revision, its configuration (ninfer `convert_nvfp4full.py` `validate_dflash2_config`)
  and its tensors.
- **Pins**: every file read comes with its Hugging Face download record (`.cache/huggingface/download`), all
  at one revision; a file whose record holds an LFS oid must hash to it (every shard does), and one whose
  record holds a git blob id (the small JSON and template files, the frontend's among them) must be that
  blob. The pattern is the 16 GB converter's `download_record` and `revision_of`, with the blob check added.
- **The ranking**: ninfer's committed `ranking.train.counts.i64`, pinned by its sha256, whose total row
  picks the draft head's rows (ninfer `qwen3_6/common/draft_head.py`, ported here).
"""
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import inventory
import shards

LFS_OID = re.compile(r"^[0-9a-f]{64}$")
GIT_BLOB = re.compile(r"^[0-9a-f]{40}$")
CONFIG = "config.json"
# What each geometry pins beyond its widths: the configuration members the widths do not give, the
# drafter's repository and revision, the ranking's sha256 and the tokenizer's id count.
PINNED = {
    "qwen3.8-27b": {
        "mrope_section": [11, 11, 10], "vision_heads": 16,
        "drafter": ("incoai/Qwen3.8-27B-DFlash2", "dedf8df68adfb1afeaf7b7480c0a0243108177b4"),
        "ranking_sha256": "c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5",
        "tokenizer_ids": 248077,
    },
    "fixture": {
        "mrope_section": [2, 1, 1], "vision_heads": 2,
        "drafter": ("someone/drafter", "d1f1a5b2d1f1a5b2d1f1a5b2d1f1a5b2d1f1a5b2"),
        "ranking_sha256": None, "tokenizer_ids": None,
    },
}
RANKING_ORIGIN = "ninfer tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64 (row 0: the totals)"
TEXT = "model.language_model."


class Refused(Exception):
    """A source the converter must not use; nothing has been written."""


# ---------------------------------------------------------------------------------------------------
# The expected tensors and configurations
# ---------------------------------------------------------------------------------------------------

def checkpoint_tensors(g):
    """The 27B checkpoint's tensors at geometry `g`: name -> shape (all BF16)."""
    H, I, V = g.hidden, g.intermediate, g.vocab
    A, Kv, D = g.attention_heads, g.kv_heads, g.head_dim
    out = {TEXT + "embed_tokens.weight": (V, H), TEXT + "norm.weight": (H,), "lm_head.weight": (V, H)}

    def attention(p):
        out.update({p + "self_attn.q_proj.weight": (2 * A * D, H), p + "self_attn.k_proj.weight": (Kv * D, H),
                    p + "self_attn.v_proj.weight": (Kv * D, H), p + "self_attn.o_proj.weight": (H, A * D),
                    p + "self_attn.q_norm.weight": (D,), p + "self_attn.k_norm.weight": (D,)})

    def mlp_and_norms(p):
        out.update({p + "input_layernorm.weight": (H,), p + "post_attention_layernorm.weight": (H,),
                    p + "mlp.gate_proj.weight": (I, H), p + "mlp.up_proj.weight": (I, H),
                    p + "mlp.down_proj.weight": (H, I)})

    for i in range(inventory.TEXT_LAYERS):
        p = f"{TEXT}layers.{i}."
        mlp_and_norms(p)
        if inventory.is_gqa(i):
            attention(p)
        else:
            q = p + "linear_attn."
            out.update({q + "A_log": (g.gdn_value_heads,), q + "dt_bias": (g.gdn_value_heads,),
                        q + "conv1d.weight": (g.gdn_conv_channels, 1, g.gdn_conv_kernel),
                        q + "in_proj_a.weight": (g.gdn_value_heads, H), q + "in_proj_b.weight": (g.gdn_value_heads, H),
                        q + "in_proj_qkv.weight": (g.gdn_conv_channels, H), q + "in_proj_z.weight": (g.gdn_value_width, H),
                        q + "norm.weight": (g.gdn_head_dim,), q + "out_proj.weight": (H, g.gdn_value_width)})
    out.update({"mtp.fc.weight": (H, 2 * H), "mtp.pre_fc_norm_embedding.weight": (H,),
                "mtp.pre_fc_norm_hidden.weight": (H,), "mtp.norm.weight": (H,)})
    mlp_and_norms("mtp.layers.0.")
    attention("mtp.layers.0.")
    vH, vI, ps = g.vision_hidden, g.vision_intermediate, g.vision_patch_size
    v = "model.visual."
    out.update({v + "patch_embed.proj.weight": (vH, 3, 2, ps, ps), v + "patch_embed.proj.bias": (vH,),
                v + "pos_embed.weight": (g.vision_positions, vH)})
    for b in range(inventory.VISION_LAYERS):
        p = f"{v}blocks.{b}."
        out.update({p + "attn.qkv.weight": (3 * vH, vH), p + "attn.qkv.bias": (3 * vH,),
                    p + "attn.proj.weight": (vH, vH), p + "attn.proj.bias": (vH,),
                    p + "mlp.linear_fc1.weight": (vI, vH), p + "mlp.linear_fc1.bias": (vI,),
                    p + "mlp.linear_fc2.weight": (vH, vI), p + "mlp.linear_fc2.bias": (vH,),
                    p + "norm1.weight": (vH,), p + "norm1.bias": (vH,), p + "norm2.weight": (vH,), p + "norm2.bias": (vH,)})
    out.update({v + "merger.linear_fc1.weight": (4 * vH, 4 * vH), v + "merger.linear_fc1.bias": (4 * vH,),
                v + "merger.linear_fc2.weight": (H, 4 * vH), v + "merger.linear_fc2.bias": (H,),
                v + "merger.norm.weight": (vH,), v + "merger.norm.bias": (vH,)})
    return out


def drafter_tensors(g):
    """The DFlash2 drafter's tensors at geometry `g`: name -> shape (all BF16)."""
    H, I = g.hidden, g.intermediate
    dA, dKv, dD, dC, dS = (g.dflash2_heads, g.dflash2_kv_heads, g.dflash2_head_dim, g.dflash2_conv_rows,
                           g.dflash2_selector_rank)
    out = {"fc.weight": (H, g.dflash2_taps * H), "hidden_norm.weight": (H,), "norm.weight": (H,),
           "candidate_selector.hidden_projection.weight": (dS, H),
           "candidate_selector.predecessor_codebook": (g.vocab, dS),
           "candidate_selector.successor_codebook": (g.vocab, dS)}
    for l in range(inventory.DFLASH2_LAYERS):
        p = f"layers.{l}."
        out.update({p + "input_layernorm.weight": (H,), p + "post_attention_layernorm.weight": (H,),
                    p + "self_attn.q_proj.weight": (dA * dD, H), p + "self_attn.k_proj.weight": (dKv * dD, H),
                    p + "self_attn.v_proj.weight": (dKv * dD, H), p + "self_attn.o_proj.weight": (H, dA * dD),
                    p + "self_attn.q_norm.weight": (dD,), p + "self_attn.k_norm.weight": (dD,),
                    p + "attention_conv.base_kernel": (2, 2, H), p + "attention_conv.kernel_projection.weight": (dC, H),
                    p + "mlp.gate_proj.weight": (I, H), p + "mlp.up_proj.weight": (I, H), p + "mlp.down_proj.weight": (H, I),
                    p + "mlp_conv.base_kernel": (2, 2, H), p + "mlp_conv.kernel_projection.weight": (dC, H)})
    return out


def checkpoint_config(g):
    """The members of the checkpoint's config.json the conversion and the calibration depend on."""
    pinned = PINNED[g.name]
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": False,
        "text_config": {
            "num_hidden_layers": inventory.TEXT_LAYERS, "full_attention_interval": 4,
            "layer_types": ["full_attention" if inventory.is_gqa(i) else "linear_attention"
                            for i in range(inventory.TEXT_LAYERS)],
            "hidden_size": g.hidden, "intermediate_size": g.intermediate, "vocab_size": g.vocab,
            "num_attention_heads": g.attention_heads, "num_key_value_heads": g.kv_heads, "head_dim": g.head_dim,
            "attn_output_gate": True, "linear_num_key_heads": g.gdn_key_heads,
            "linear_num_value_heads": g.gdn_value_heads, "linear_key_head_dim": g.gdn_head_dim,
            "linear_value_head_dim": g.gdn_head_dim, "linear_conv_kernel_dim": g.gdn_conv_kernel,
            "mamba_ssm_dtype": "float32", "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
            "tie_word_embeddings": False, "max_position_embeddings": 262144, "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000000, "mrope_section": pinned["mrope_section"]},
        },
        "vision_config": {
            "depth": inventory.VISION_LAYERS, "hidden_size": g.vision_hidden,
            "intermediate_size": g.vision_intermediate, "out_hidden_size": g.hidden,
            "num_heads": pinned["vision_heads"], "in_channels": 3, "patch_size": g.vision_patch_size,
            "temporal_patch_size": 2, "spatial_merge_size": 2, "num_position_embeddings": g.vision_positions,
        },
    }


def drafter_config(g):
    """The base's DFlash2 drafter's configuration members at geometry `g`."""
    return {
        "architectures": ["DFlash2DraftModel"], "hidden_size": g.hidden, "intermediate_size": g.intermediate,
        "num_attention_heads": g.dflash2_heads, "num_key_value_heads": g.dflash2_kv_heads,
        "head_dim": g.dflash2_head_dim, "num_hidden_layers": inventory.DFLASH2_LAYERS,
        "num_target_layers": inventory.TEXT_LAYERS, "sliding_window": 2048, "max_position_embeddings": 262144,
        "is_causal": False, "vocab_size": g.vocab, "rms_norm_eps": 1e-6,
        "layer_types": ["sliding_attention"] * inventory.DFLASH2_LAYERS,
        "dflash_config": {"block_size": 8, "conv_kernel_size": 2, "conv_group_size": 16,
                          "selector_rank": g.dflash2_selector_rank, "selector_top_k": 16, "mask_token_id": 248070,
                          "target_layer_ids": [5, 19, 33, 47, 61]},
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    }


def config_mismatches(actual, expected, scope=""):
    """Every member of `expected` that `actual` does not hold: `scope.member is X, not Y`."""
    out = []
    for key, want in expected.items():
        got = actual.get(key) if isinstance(actual, dict) else None
        if isinstance(want, dict):
            out += config_mismatches(got if isinstance(got, dict) else {}, want, f"{scope}{key}.")
        elif got != want:
            out.append(f"{scope}{key} is {got!r}, not {want!r}")
    return out


def signature_problems(actual, expected, what):
    """The tensors `actual` (name -> (dtype, shape)) holds that `expected` (name -> shape) does not, or
    holds otherwise, and those it lacks."""
    out = [f"{what} has no {name}" for name in sorted(set(expected) - set(actual))]
    out += [f"{what} has {name}, which the 27B's has not" for name in sorted(set(actual) - set(expected))]
    for name in sorted(set(actual) & set(expected)):
        dtype, shape = actual[name]
        if tuple(shape) != tuple(expected[name]):
            out.append(f"{what}'s {name} is {list(shape)}, the 27B's is {list(expected[name])}")
        elif dtype != "BF16":
            out.append(f"{what}'s {name} is {dtype}, the 27B's is BF16")
    return out


def refuse_if(problems, intro):
    if problems:
        shown = problems[:8] + ([f"... and {len(problems) - 8} more"] if len(problems) > 8 else [])
        raise Refused(intro + ":\n  " + "\n  ".join(shown))


def read_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise Refused(f"{what} cannot be read: {e}")


# ---------------------------------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------------------------------

def download_record(root, name):
    """(revision, etag) of a downloaded file: an LFS file's etag is its sha256, another's its git blob id."""
    path = os.path.join(root, ".cache", "huggingface", "download", name + ".metadata")
    if not os.path.isfile(path):
        raise Refused(f"{root} has no Hugging Face download record for {name}: it cannot be pinned")
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().split("\n")]
    if len(lines) < 2 or not lines[0] or not lines[1]:
        raise Refused(f"{path} is not a Hugging Face download record")
    return lines[0], lines[1]


def records(root, names, lfs, what, revision=None):
    """(revision, name -> etag) of the files `names`, all from one revision (`revision` when given); each
    of `lfs` must have an LFS oid. Reads no file but the records."""
    etags, revisions = {}, {}
    for name in names:
        if not os.path.isfile(os.path.join(root, name)):
            raise Refused(f"{what} has no {name}")
        rev, etag = download_record(root, name)
        revisions.setdefault(rev, []).append(name)
        etags[name] = etag
        if name in lfs and not LFS_OID.match(etag):
            raise Refused(f"{what}'s {name} has no LFS oid in its download record ({etag})")
    if len(revisions) != 1:
        raise Refused(f"{what}'s files come from two revisions: "
                      + "; ".join(f"{r} ({len(n)} files, {n[0]} ...)" for r, n in sorted(revisions.items())))
    (rev,) = revisions
    if revision is not None and rev != revision:
        raise Refused(f"{what} was downloaded at revision {rev}, not {revision}")
    return rev, etags


def _digest(path, git):
    sha = hashlib.sha256()
    blob = None
    if git:
        blob = hashlib.sha1()
        blob.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        while True:
            chunk = f.read(64 << 20)
            if not chunk:
                break
            sha.update(chunk)
            if blob:
                blob.update(chunk)
    return sha.hexdigest(), blob.hexdigest() if blob else None


def hash_pins(root, etags, what, workers=4):
    """name -> {bytes, sha256}, each file checked against its record: its sha256 against an LFS oid, its
    git blob id against a blob id."""
    names = sorted(etags)
    with ThreadPoolExecutor(workers) as pool:
        digests = dict(zip(names, pool.map(
            lambda n: _digest(os.path.join(root, n), bool(GIT_BLOB.match(etags[n]))), names)))
    out = {}
    for name in names:
        sha, blob = digests[name]
        etag = etags[name]
        if LFS_OID.match(etag) and sha != etag:
            raise Refused(f"{what}'s {name} hashes to {sha}, its LFS oid is {etag}")
        if GIT_BLOB.match(etag) and blob != etag:
            raise Refused(f"{what}'s {name} is git blob {blob}, its download record says {etag}")
        if not LFS_OID.match(etag) and not GIT_BLOB.match(etag):
            raise Refused(f"{what}'s {name} has neither an LFS oid nor a git blob id in its record ({etag})")
        out[name] = {"bytes": os.path.getsize(os.path.join(root, name)), "sha256": sha}
    return out


def file_sha256(path):
    return _digest(path, False)[0]


# ---------------------------------------------------------------------------------------------------
# The draft head's rows (ninfer qwen3_6/common/draft_head.py)
# ---------------------------------------------------------------------------------------------------

def tokenizer_ids(root):
    """The tokenizer's id count: its largest id, of the model's vocabulary and the added tokens, plus one."""
    t = read_json(os.path.join(root, "tokenizer.json"), "the checkpoint's tokenizer.json")
    ids = list(t.get("model", {}).get("vocab", {}).values()) + [a["id"] for a in t.get("added_tokens", [])]
    return max(ids) + 1


def special_ids(root):
    """The special ids of `tokenizer_config.json`'s `added_tokens_decoder`, sorted."""
    config = read_json(os.path.join(root, "tokenizer_config.json"), "the checkpoint's tokenizer_config.json")
    return tuple(sorted({int(i) for i, m in config.get("added_tokens_decoder", {}).items()
                         if isinstance(m, dict) and m.get("special", False)}))


def ranking_totals(path, vocab):
    """The ranking's total row: `vocab` little-endian i64 counts (the file is `[rows, vocab]`)."""
    size = os.path.getsize(path)
    if size == 0 or size % (vocab * 8):
        raise Refused(f"the ranking is {size} bytes, not a whole number of {vocab}-entry i64 rows")
    return np.fromfile(path, dtype="<i8", count=vocab)


def shortlist(totals, n, forced):
    """Exactly `n` ids in stable descending-frequency order, the special ids forced in."""
    counts = np.asarray(totals, dtype=np.int64)
    vocab = counts.size
    if not 0 < n <= vocab:
        raise Refused(f"a shortlist of {n} rows from {vocab} ids")
    order = np.argsort(-counts, kind="stable")
    forced = np.array(sorted({int(t) for t in forced if 0 <= int(t) < vocab}), dtype=np.int64)
    if forced.size >= n:
        return np.ascontiguousarray(forced[np.argsort(-counts[forced], kind="stable")][:n], dtype=np.int64)
    forced_set = set(forced.tolist())
    picked = []
    for token in order.tolist():
        if token not in forced_set:
            picked.append(token)
            if len(picked) == n - forced.size:
                break
    selected = np.concatenate((np.asarray(picked, dtype=np.int64), forced))
    selected = selected[np.argsort(-counts[selected], kind="stable")]
    if selected.size != n or np.unique(selected).size != n:
        raise Refused("the shortlist did not give unique rows")
    return np.ascontiguousarray(selected, dtype=np.int64)


def draft_ids(ranking, checkpoint_root, g):
    """(ids, record): the draft head's rows picked from the ranking's totals over the tokenizer's ids."""
    domain = tokenizer_ids(checkpoint_root)
    expected = PINNED[g.name]["tokenizer_ids"]
    if expected is not None and domain != expected:
        raise Refused(f"the checkpoint's tokenizer has {domain} ids, the 27B's has {expected}")
    if domain > g.vocab:
        raise Refused(f"the checkpoint's tokenizer has {domain} ids, more than the vocabulary's {g.vocab}")
    forced = special_ids(checkpoint_root)
    ids = shortlist(ranking_totals(ranking, g.vocab)[:domain], g.draft_head_rows, forced)
    return ids, {"tokenizer_ids": domain, "forced_special_ids": len(forced)}


# ---------------------------------------------------------------------------------------------------
# The sources, checked
# ---------------------------------------------------------------------------------------------------

class Sources:
    """The sources of a conversion, checked: the checkpoint (its revision, its files' records, its
    tensors' shard map), the drafter (its files' records) and the ranking (its pin). `hash` checks every
    file against its record and pins it (`files`, `drafter_files`): the one long check, made last."""

    def __init__(self, g, checkpoint, revision, etags, weight_map, drafter=None, drafter_etags=None,
                 ranking=None, ranking_pin=None):
        self.geometry = g
        self.checkpoint = checkpoint
        self.revision = revision
        self.etags = etags
        self.weight_map = weight_map
        self.drafter = drafter
        self.drafter_etags = drafter_etags
        self.ranking = ranking
        self.ranking_pin = ranking_pin
        self.files = self.drafter_files = None

    @property
    def shard_oids(self):
        return {name: etag for name, etag in self.etags.items() if name.endswith(".safetensors")}

    def hash(self):
        self.files = hash_pins(self.checkpoint, self.etags, "the checkpoint")
        if self.drafter is not None:
            self.drafter_files = hash_pins(self.drafter, self.drafter_etags, "the drafter")
        return self


def check_checkpoint(root, g, revision=None):
    """(revision, name -> etag, weight map) of a checkpoint whose tensors and configuration are the 27B's:
    the cheap checks, before any file is hashed."""
    what = "the checkpoint"
    if not os.path.isfile(os.path.join(root, shards.INDEX)):
        raise Refused(f"{what} has no {shards.INDEX} ({root})")
    wmap = shards.weight_map(root)
    shard_files = sorted(set(wmap.values()))
    names = [shards.INDEX, CONFIG] + list(inventory.FRONTEND_FILES) + shard_files
    rev, etags = records(root, names, set(shard_files), what, revision)
    refuse_if(config_mismatches(read_json(os.path.join(root, CONFIG), "the checkpoint's config.json"),
                                checkpoint_config(g)), "the checkpoint's config.json is not the 27B's")
    actual = {}
    for shard in shard_files:
        for name, sig in shards.header(os.path.join(root, shard)).items():
            if wmap.get(name) != shard:
                raise Refused(f"{what}'s {shard} holds {name}, which its index puts in {wmap.get(name)}")
            actual[name] = sig
    missing = sorted(set(wmap) - set(actual))
    if missing:
        raise Refused(f"{what}'s index names {missing[0]}, which no shard holds")
    refuse_if(signature_problems(actual, checkpoint_tensors(g), what), f"{what}'s tensors are not the 27B's")
    return rev, etags, wmap


DRAFTER_FILE = "model.safetensors"


def check_drafter(root, g):
    """name -> etag of the drafter's files, its revision the pinned one, its configuration and tensors the
    base's drafter's: the cheap checks."""
    what = "the drafter"
    repo, revision = PINNED[g.name]["drafter"]
    _, etags = records(root, [CONFIG, DRAFTER_FILE], {DRAFTER_FILE}, f"{what} ({repo})", revision)
    refuse_if(config_mismatches(read_json(os.path.join(root, CONFIG), "the drafter's config.json"), drafter_config(g)),
              f"the drafter's config.json is not the base's drafter's ({repo} at {revision})")
    refuse_if(signature_problems(shards.header(os.path.join(root, DRAFTER_FILE)), drafter_tensors(g), what),
              f"the drafter's tensors are not the base's drafter's ({repo})")
    return etags


def check_ranking(path, g):
    if not os.path.isfile(path):
        raise Refused(f"the ranking {path} does not exist")
    digest = file_sha256(path)
    expected = PINNED[g.name]["ranking_sha256"]
    if expected is not None and digest != expected:
        raise Refused(f"the ranking hashes to {digest}, the committed ranking's sha256 is {expected}")
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": digest, "origin": RANKING_ORIGIN}


def check_sources(g, checkpoint, revision=None, drafter=None, ranking=None):
    """The sources, through every check but the hashes (`Sources.hash`)."""
    rev, etags, wmap = check_checkpoint(checkpoint, g, revision)
    drafter_etags = check_drafter(drafter, g) if drafter is not None else None
    ranking_pin = check_ranking(ranking, g) if ranking is not None else None
    return Sources(g, checkpoint, rev, etags, wmap, drafter, drafter_etags, ranking, ranking_pin)
