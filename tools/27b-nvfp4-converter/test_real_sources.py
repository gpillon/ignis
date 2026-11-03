"""Machine-local: the 27B geometry's expectations against the real sources, when they are on this machine
(each test skips otherwise). Headers, configurations, records and the tokenizer only: no shard is hashed or
read, and nothing runs on the GPU."""
import os

import pytest

import calibrate
import checkpoint
import inventory

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
