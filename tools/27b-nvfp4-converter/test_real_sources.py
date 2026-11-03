"""Machine-local: the 27B geometry's expectations against the real sources, when they are on this machine
(each test skips otherwise): headers, configurations, records and the tokenizer, and #469's tier 1, every
object of the served artifact neither calibrated nor taken from the third-party quant, made of the base
checkpoint and its drafter on the CPU. No shard is hashed (the drafter's file is), nothing runs on the GPU."""
import os
import time

import pytest

import calibrate
import checkpoint
import compare
import container
import convert
import inventory
import recipe

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
DRAFTER = "H:/models/Qwen3.8-27B-DFlash2"


def base_sources(drafter=None):
    """The base checkpoint's sources through the cheap checks (no shard hashed), or a skip naming what is
    absent."""
    root, revision = CHECKPOINTS["Qwen/Qwen3.8-27B"]
    for path in (SERVED, RANKING, os.path.join(root, "model.safetensors.index.json")):
        if not os.path.isfile(path):
            pytest.skip(f"{path} is not on this machine")
    return checkpoint.check_sources(G, root, revision, drafter, RANKING)


def costed(rebuild, *args):
    """`rebuild(*args)`, with its wall time and this process's peaks so far printed (`pytest -s`)."""
    t0 = time.time()
    done = rebuild(*args)
    measured = convert.peaks(convert.torch.device("cpu"))
    gib = lambda n: "-" if n is None else f"{n / 2 ** 30:.2f} GiB"  # noqa: E731
    print(f"\n{done['objects']} objects, {done['bytes']} bytes in {time.time() - t0:.0f} s; process peak "
          f"{gib(measured['host_peak_working_set_bytes'])} working set, {gib(measured['host_peak_commit_bytes'])} "
          f"commit")
    return done


def test_tier_1_of_the_base_checkpoint_is_the_served_artifacts_byte_for_byte():
    """#469's binding tier on the CPU: the 765 tier-1 objects the recipe makes of the base checkpoint (the
    frontend resources, the norms and the GDN small set, the BF16 exceptions, the W8 embedding and output head,
    the Q4 draft head and its ids, the MTP layer and the vision tower) are the served artifact's byte for byte."""
    sources = base_sources()
    ids, _ = checkpoint.draft_ids(RANKING, sources.checkpoint, G)
    wanted = {n for n in compare.tier_names(G, 1) if not n.startswith("dflash2/")}
    done = costed(compare.rebuild, container.Container(SERVED), sources, recipe.recipes(G), ids, wanted, "cpu")
    assert done["differ"] == []
    assert done["objects"] == 765


def test_a_parent_the_served_artifact_encoded_itself_re_encodes_byte_equal():
    """A tier-2 preview: one of the 135 parents ninfer encoded locally, from the base on the CPU."""
    name = "text/layers/0/gdn/query_key_value_z"
    done = compare.rebuild(container.Container(SERVED), base_sources(), recipe.recipes(G), None, {name}, "cpu")
    assert (done["objects"], done["differ"]) == (1, [])


def drafter_or_skip():
    """The drafter's directory when its file is on this machine and hashes to its LFS oid; a skip otherwise."""
    path = os.path.join(DRAFTER, checkpoint.DRAFTER_FILE)
    if not os.path.isfile(path):
        pytest.skip(f"the DFlash2 drafter is not on this machine ({path})")
    try:
        etags = checkpoint.check_drafter(DRAFTER, G)
        checkpoint.hash_pins(DRAFTER, {checkpoint.DRAFTER_FILE: etags[checkpoint.DRAFTER_FILE]}, "the drafter")
    except checkpoint.Refused as e:
        pytest.skip(f"the DFlash2 drafter on this machine is not the pinned one: {e}")
    return DRAFTER


def test_the_dflash2_objects_of_the_bases_drafter_are_the_served_artifacts_byte_for_byte():
    """The rest of tier 1: the 66 DFlash2 objects the recipe makes of the base's drafter (dedf8df6), its 34
    NVFP4 matrices weight-only, are the served artifact's byte for byte."""
    sources = base_sources(drafter_or_skip())
    wanted = {n for n in compare.tier_names(G, 1) if n.startswith("dflash2/")}
    done = costed(compare.rebuild, container.Container(SERVED), sources, recipe.recipes(G), None, wanted, "cpu")
    assert done["differ"] == []
    assert done["objects"] == 66
