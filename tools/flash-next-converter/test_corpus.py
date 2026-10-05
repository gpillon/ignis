"""The calibration corpus: allowlisted, pinned by sha256, assembled in the study's order."""
import copy
import hashlib
import json

import pytest

import corpus

CH = corpus.CH


def chunk(kind, test, valid=CH, mmlu=(), cal=None, tok=1):
    c = {"ids": [tok] * CH, "kind": kind, "test": test, "valid": valid, "mmlu": list(mmlu)}
    if cal is not None:
        c["cal"] = cal
    return c


def write_tree(tmp_path):
    """A miniature of the study's files with the same structure."""
    (tmp_path / "windows").mkdir()
    (tmp_path / "ood").mkdir()
    files = {
        # 5 windows of 4096 -> 5 pairs; pair 4 is a test pair (p % 5 == 4)
        "windows.json": [[10 + w] * 4096 for w in range(5)],
        "windows_meta.json": [{"kind": "code", "source": "a", "part": w} for w in range(5)],
        # long windows: index 3 is 32768 code, 5 is 65536 prose (the long8192 picks)
        "long_windows.json": [[20 + w] * (65536 if w >= 4 else 32768) for w in range(6)],
        "long_windows_meta.json": [{"kind": ["code", "code", "prose", "code", "code", "prose"][w], "source": "r",
                                    "len": 0, "part": 0} for w in range(6)],
        "run4_chunks.json": {"chunks": [chunk("en", False, cal=True), chunk("en", True, cal=False),
                                        chunk("de", True, cal=False)]},
        "chunks.json": {"chunks": [chunk("en", False), chunk("mmlu", True, valid=100, mmlu=[[50, 1, 4, "law"]])],
                        "letter_ids": list(range(10))},
        "calib_mmlu.json": {"chunks": [chunk("mmlu", False, valid=90)]},
    }
    man = copy.deepcopy(json.load(open(corpus.MANIFEST)))
    for name, obj in files.items():
        d = man["files"][name]["dir"]
        raw = json.dumps(obj).encode()
        (tmp_path / d / name).write_bytes(raw)
        man["files"][name]["sha256"] = hashlib.sha256(raw).hexdigest()
    return man


def test_chunks_follow_the_study_order(tmp_path):
    man = write_tree(tmp_path)
    c = corpus.load(tmp_path / "ood", tmp_path / "windows", manifest=man)
    kinds = [(x["kind"], x["cal"], x["test"]) for x in c.chunks]
    # calibration: in-domain train chunks, run4 calibration, MMLU calibration; then the
    # held-out ones: in-domain test chunks, run4 tests, MMLU tests
    n_cal = sum(1 for x in c.chunks if x["cal"])
    assert all(x["cal"] for x in c.chunks[:n_cal]) and not any(x["cal"] for x in c.chunks[n_cal:])
    assert kinds[n_cal - 2:n_cal] == [("en", True, False), ("mmlu", True, False)]
    assert [k for k, _, _ in kinds[n_cal:]][-3:] == ["en", "de", "mmlu"]
    # 5 + 8 + 8 + 8 + 16 + 16 = 61 window pairs; pairs 4, 9, 14, ... are held out
    assert sum(1 for x in c.chunks if x["kind"] in ("code", "prose") and x["test"]) == 8
    assert c.letter_ids == list(range(10))
    assert c.mmlu_questions == 1
    assert [(w["kind"], len(w["ids"])) for w in c.long] == [("code", 8192)] * 4 + [("prose", 8192)] * 4
    assert c.long[1]["ids"][0] == 23 and c.long[4]["ids"][0] == 25


def test_a_file_changed_since_the_manifest_is_refused(tmp_path):
    man = write_tree(tmp_path)
    (tmp_path / "ood" / "calib_mmlu.json").write_text("{}")
    with pytest.raises(corpus.CorpusError, match="sha256"):
        corpus.load(tmp_path / "ood", tmp_path / "windows", manifest=man)


def test_a_source_outside_the_allowlist_is_refused(tmp_path):
    man = write_tree(tmp_path)
    man["files"]["calib_mmlu.json"]["sources"].append("partner-mail-export")
    with pytest.raises(corpus.CorpusError, match="allowlist"):
        corpus.load(tmp_path / "ood", tmp_path / "windows", manifest=man)


def test_the_committed_manifest_only_names_allowlisted_sources():
    corpus.check_allowlist(json.load(open(corpus.MANIFEST)))


def test_a_table_cache_shard_that_moved_is_refused(tmp_path):
    (tmp_path / "shard_0.weight.pt").write_bytes(b"a" * 3000)
    size, head, tail = corpus._head_tail(tmp_path / "shard_0.weight.pt")
    man = {"table_cache": {"files": [{"file": "shard_0.weight.pt", "bytes": size, "head_sha256": head,
                                      "tail_sha256": tail}]}}
    corpus.check_table_cache(tmp_path, 1, man)
    (tmp_path / "shard_0.weight.pt").write_bytes(b"a" * 2999 + b"b")
    with pytest.raises(corpus.CorpusError, match="first/last"):
        corpus.check_table_cache(tmp_path, 1, man)


def test_the_committed_manifest_pins_all_128_table_shards():
    man = json.load(open(corpus.MANIFEST))
    assert [f["file"] for f in man["table_cache"]["files"]] == [f"shard_{n}.weight.pt" for n in range(128)]


def test_a_hot_sample_file_that_changed_is_refused(tmp_path):
    man = copy.deepcopy(json.load(open(corpus.MANIFEST)))
    man["hot_sample"]["corpora"] = [c for c in man["hot_sample"]["corpora"] if c["kind"] == "parquet"][:1]
    f = man["hot_sample"]["corpora"][0]["files"][0]
    (tmp_path / f["file"]).parent.mkdir(parents=True)
    (tmp_path / f["file"]).write_bytes(b"not the pinned parquet")
    with pytest.raises(corpus.CorpusError, match="sha256"):
        list(corpus.hot_sample_corpora(tmp_path, tmp_path, man))


def test_licences_name_each_used_source():
    man = json.load(open(corpus.MANIFEST))
    lic = corpus.licences(man, ["tiger-lab-mmlu-pro", "wikitext-103", "tiger-lab-mmlu-pro"])
    assert lic == {"tiger-lab-mmlu-pro": man["allowlist"]["tiger-lab-mmlu-pro"],
                   "wikitext-103": man["allowlist"]["wikitext-103"]}
