r"""Tests for the match builder and the NLL bookkeeping.

No model is loaded: these cover the parts that decide *where* an injection
lands and *which* tokens its effect is measured over, which is where an error
would silently change every number in the sweep without failing anything.

The tokenizer is real, because the point of matching on text rather than on
token ids is that `decode_round` and ` decode_round` tokenize differently —
a fake tokenizer would test the wrong thing.

Run with:  F:/ai/ngram-venv/Scripts/python.exe -m pytest -q test_nll.py
"""

import os

import pytest
import torch

from nll import build_matches, window_mean, bootstrap_ci

TOKENIZER = os.environ.get("NGRAM_TOKENIZER", "Y:/models/Qwen3.8-27B")


@pytest.fixture(scope="module")
def tok():
    if not os.path.isdir(TOKENIZER):
        pytest.skip("tokenizer not available at %s" % TOKENIZER)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)


def _match(tok, text, keys):
    rows = {k: i for i, k in enumerate(keys)}
    ids, offsets, pos, row_ids, names = build_matches(tok, text, keys, rows)
    return ids, offsets, pos, names


def test_match_is_found_with_and_without_a_leading_space(tok):
    text = "call decode_round(x);\nlet y=decode_round(z);\n"
    ids, offsets, pos, names = _match(tok, text, ["decode_round"])
    assert names == ["decode_round", "decode_round"]
    assert len(pos) == 2 and pos[0] < pos[1]


def test_injection_lands_on_the_last_token_of_the_name(tok):
    text = "fn use() { decode_round(1); }\n"
    ids, offsets, pos, _ = _match(tok, text, ["decode_round"])
    a, b = offsets[pos[0]]
    # the token that completes the name ends where the name ends
    assert b == text.index("decode_round") + len("decode_round")
    assert text[a:b] in "decode_round"


def test_partial_identifier_is_not_a_match(tok):
    text = "let a = kv_pages + 1;\nlet b = my_kv;\n"
    _, _, pos, names = _match(tok, text, ["kv"])
    assert names == [] and pos == []


def test_whole_identifier_is_a_match(tok):
    text = "let a = kv + 1;\n"
    _, _, pos, names = _match(tok, text, ["kv"])
    assert names == ["kv"]


def test_two_keys_never_share_a_position(tok):
    # `Foo` and `FooBar` cannot both claim the same completing token
    text = "use Foo; use FooBar;\n"
    _, _, pos, names = _match(tok, text, ["Foo", "FooBar"])
    assert len(pos) == len(set(pos))
    assert sorted(names) == ["Foo", "FooBar"]


def test_positions_come_back_sorted(tok):
    text = "alpha(); zulu(); alpha();\n"
    _, _, pos, _ = _match(tok, text, ["zulu", "alpha"])
    assert pos == sorted(pos)


def test_matches_past_the_truncation_point_are_dropped(tok):
    text = "x = 1;\n" * 400 + "decode_round();\n"
    rows = {"decode_round": 0}
    _, _, pos, _, names = build_matches(tok, text, ["decode_round"], rows,
                                        max_tokens=64)
    assert names == []


# ------------------------------------------------------------- windowing ---

def test_window_covers_the_tokens_after_the_match():
    nll = torch.arange(20, dtype=torch.float32)
    mean, n = window_mean(nll, [5], width=4)
    assert n == 4 and mean == pytest.approx((5 + 6 + 7 + 8) / 4)


def test_overlapping_windows_are_unioned_not_double_counted():
    nll = torch.ones(20, dtype=torch.float32)
    _, n = window_mean(nll, [5, 6], width=4)
    assert n == 5                      # 5..9, not 8


def test_window_is_clipped_at_the_end_of_the_sequence():
    nll = torch.ones(10, dtype=torch.float32)
    _, n = window_mean(nll, [8], width=32)
    assert n == 2


def test_no_match_gives_no_window():
    nll = torch.ones(10, dtype=torch.float32)
    mean, n = window_mean(nll, [], width=8)
    assert mean is None and n == 0


# ------------------------------------------------------------- bootstrap ---

def test_bootstrap_brackets_the_mean():
    ci = bootstrap_ci([-2.0, -1.0, -3.0, -2.5, -1.5], iters=500, seed=1)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["n_files"] == 5


def test_bootstrap_is_reproducible():
    a = bootstrap_ci([1.0, -1.0, 2.0], iters=200, seed=7)
    b = bootstrap_ci([1.0, -1.0, 2.0], iters=200, seed=7)
    assert a == b


def test_bootstrap_of_nothing_is_none():
    assert bootstrap_ci([]) is None
