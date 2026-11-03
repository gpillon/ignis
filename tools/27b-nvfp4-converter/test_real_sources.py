"""Machine-local: the 27B geometry's expectations against the real sources, when they are on this machine
(each test skips otherwise): headers, configurations, records and the tokenizer, and a few objects of the base
checkpoint against the served artifact's, encoded on the CPU. No shard is hashed, nothing runs on the GPU."""
import os

import pytest
import torch

import calibrate
import checkpoint
import container
import encode
import inventory
import recipe
import shards

G = inventory.Geometry.qwen3_8_27b()
CHECKPOINTS = {"Qwen/Qwen3.8-27B": ("H:/models/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
               "ukisai/Swift-1.5-Qwen3.8-27b": ("H:/models/Swift-1.5-Qwen3.8-27b",
                                                "b4c84d42903a8646b25857eb2827288d92ed85a4")}
RANKING = "F:/ai/q38/ninfer/tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"


@pytest.mark.parametrize("repo", sorted(CHECKPOINTS))
def test_the_real_checkpoint_passes_the_cheap_checks(repo):
    root, revision = CHECKPOINTS[repo]
    if not os.path.isfile(os.path.join(root, "model.safetensors.index.json")):
        pytest.skip(f"{repo} is not on this machine ({root})")
    rev, etags, weight_map = checkpoint.check_checkpoint(root, G, revision)
    assert rev == revision
    assert len(weight_map) == 1199
    assert sum(name.endswith(".safetensors") for name in etags) == 18
    assert checkpoint.tokenizer_ids(root) == 248077


def test_the_corpus_is_2690_tokens_of_the_27bs_tokenizer():
    root = CHECKPOINTS["Qwen/Qwen3.8-27B"][0]
    if not os.path.isfile(os.path.join(root, "tokenizer.json")):
        pytest.skip(f"the 27B's tokenizer is not on this machine ({root})")
    from transformers import AutoTokenizer
    assert len(AutoTokenizer.from_pretrained(root)(calibrate.corpus())["input_ids"]) == 2690


@pytest.mark.skipif(not os.path.isfile(RANKING), reason=f"ninfer's ranking is not on this machine ({RANKING})")
def test_ninfers_ranking_is_the_pinned_one():
    assert checkpoint.check_ranking(RANKING, G)["sha256"] == checkpoint.PINNED["qwen3.8-27b"]["ranking_sha256"]


SERVED = "F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer"
# Objects the served artifact made from the base checkpoint, each route and encoder once, small enough for the
# CPU: the draft head's rows (the shortlist), a resource, the GDN small set (the convolution's orientation, the
# FP32 cast, the a/b order), a norm, a BF16 exception (the interleaved q_proj split at 24 heads), one of the 135
# parents ninfer already encoded locally (NVFP4), and the grouped encoders on the vision tower and the MTP layer.
SPOT_CHECK = ("text/draft_head_token_ids", "frontend/chat_template.jinja", "text/layers/0/gdn/convolution",
              "text/layers/0/gdn/a_log", "text/layers/0/gdn/a_b_projection", "text/layers/0/input_norm",
              "text/layers/3/attention/query_key_gate_value", "text/layers/0/gdn/query_key_value_z",
              "vision/patch_embedding", "vision/layers/0/attention/qkv", "vision/layers/0/attention/output",
              "vision/merger/fc1", "mtp/layer/attention/output")


def test_the_served_artifacts_objects_are_what_the_recipe_makes_of_the_base():
    """A preview of #469's byte-for-byte comparison, on the CPU: each object of SPOT_CHECK is the served one."""
    root, revision = CHECKPOINTS["Qwen/Qwen3.8-27B"]
    for path in (SERVED, RANKING, os.path.join(root, "model.safetensors.index.json")):
        if not os.path.isfile(path):
            pytest.skip(f"{path} is not on this machine")
    served = container.Container(SERVED)
    table = recipe.recipes(G)
    objs = {o.name: o for o in inventory.objects(G)}
    ids, _ = checkpoint.draft_ids(RANKING, root, G)
    with shards.ShardReader.checkpoint(root) as reader:
        for name in SPOT_CHECK:
            o, r = objs[name], table[name]
            if r.route == "resource":
                with open(os.path.join(root, r.parts[0][0]), "rb") as f:
                    payload = f.read()
            elif r.transform == "ids":
                payload = encode.direct(torch.from_numpy(ids).to(torch.int32), o.format)
            else:
                payload, _ = encode.encode(o, recipe.materialize(r, o, reader, ids), "cpu")
            assert payload == served.read(name), name
