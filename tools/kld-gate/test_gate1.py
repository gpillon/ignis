"""Gate 1's verdict on synthetic `ref_kld.py` results: four dumps' worth of per-position KL, two references."""
import json
import os

import pytest

import gate1

KINDS = ["code", "code", "prose", "chat"]
LENGTHS = [2, 6, 4, 4]


def result(path, dump, kl, engine_nll, ref_nll):
    """A `ref_kld.py` result holding one dump: `kl` is a value a window, the others a list a window."""
    windows = []
    for w, (kind, n) in enumerate(zip(KINDS, LENGTHS)):
        windows.append({"window": w, "kind": kind, "ref_nll": ref_nll[w][:n - 1],
                        dump: {"kl": [kl[w]] * n, "top1": [1.0] * n, "nll": engine_nll[w][:n - 1]}})
    with open(path, "w") as f:
        json.dump({"label": path.stem, "dumps": [dump], "windows": windows}, f)
    return f"{dump}={path}"


def nll(seed):
    return [[seed + w + i / 100 for i in range(n)] for w, n in enumerate(LENGTHS)]


def world(tmp_path, matched, cross_artifact=(0.3,) * 4, cross_reference=(0.3,) * 4, baseline_pair=(0.04,) * 4,
          base_ref=2, cross_reference_dump=3):
    """The gate's four terms, each a file of its own as one `ref_kld.py` run a dump writes them: the fine-tune's
    dump `ft` and the vanilla one `van`, against the fine-tune's BF16 and the base's. The NLLs are a dump's or a
    reference's own (seeds: the fine-tune's BF16 1, the base's `base_ref`, the dumps 3 and 4)."""
    ft_ref, base, ft_dump, van_dump = nll(1), nll(base_ref), nll(3), nll(4)
    return ["--matched", result(tmp_path / "ftref-ft.json", "ft", matched, ft_dump, ft_ref),
            "--cross-artifact", result(tmp_path / "ftref-van.json", "van", cross_artifact, van_dump, ft_ref),
            "--cross-reference", result(tmp_path / "baseref-ft.json", "ft", cross_reference,
                                        nll(cross_reference_dump), base),
            "--baseline-pair", result(tmp_path / "baseref-van.json", "van", baseline_pair, van_dump, base)]


def run(tmp_path, terms, *extra):
    out = tmp_path / "gate1.json"
    code = gate1.main(terms + ["--json", str(out), *extra])
    return code, json.loads(out.read_text())


def test_the_gate_holds_the_position_weighted_kld_to_the_limit(tmp_path):
    # 2 x 0.02 + 6 x 0.06 + 4 x 0.05 + 4 x 0.03 over 16 positions (the windows' mean would be 0.04)
    terms = world(tmp_path, matched=(0.02, 0.06, 0.05, 0.03))

    code, report = run(tmp_path, terms, "--limit", "0.0450", "--baseline", "0.0375")
    assert (code, report["verdict"]) == (0, "PASS")
    assert report["kld"] == pytest.approx(0.045)
    lo, hi = report["ci95"]
    assert 0.02 <= lo < 0.045 < hi <= 0.06

    code, report = run(tmp_path, terms, "--limit", "0.0449", "--baseline", "0.0374")
    assert (code, report["verdict"]) == (1, "FAIL")


def test_a_cross_term_not_above_the_matched_pair_in_every_window_stops_the_gate(tmp_path):
    matched = (0.02, 0.06, 0.05, 0.03)

    code, report = run(tmp_path, world(tmp_path, matched), "--limit", "0.0590", "--baseline", "0.04916")
    assert (code, report["verdict"], report["stop"]) == (0, "PASS", False)
    assert report["cross"]["cross_artifact"]["above_windows"] == 4
    assert report["cross"]["cross_reference"]["ratio"] == pytest.approx(0.3 / 0.045)
    assert report["cross_rule"] == "above in every window, the CI of the difference above 0"

    # The vanilla artifact sits below the matched pair in window 2: the gate may have measured the base.
    code, report = run(tmp_path, world(tmp_path, matched, cross_artifact=(0.3, 0.3, 0.03, 0.3)),
                       "--limit", "0.0590", "--baseline", "0.04916")
    assert (code, report["verdict"], report["stop"]) == (3, "PASS", True)
    assert report["cross"]["cross_artifact"]["above_windows"] == 3
    assert report["cross"]["cross_reference"]["above"] is True


def test_the_gate_is_void_unless_its_terms_are_two_dumps_against_two_references(tmp_path):
    matched = (0.02, 0.06, 0.05, 0.03)
    code, report = run(tmp_path, world(tmp_path, matched), "--limit", "0.0590", "--baseline", "0.04916")
    assert (code, report["valid"]) == (0, True)

    # The "base" reference is the fine-tune's BF16 again: nothing tells the two apart.
    code, report = run(tmp_path, world(tmp_path, matched, base_ref=1), "--limit", "0.0590", "--baseline", "0.04916")
    assert (code, report["verdict"], report["valid"]) == (3, "PASS", False)
    assert report["controls"]["references_differ"] is False

    # The cross-reference term scored the vanilla dump, not the matched pair's.
    code, report = run(tmp_path, world(tmp_path, matched, cross_reference_dump=4),
                       "--limit", "0.0590", "--baseline", "0.04916")
    assert (code, report["valid"]) == (3, False)
    assert report["controls"]["matched_and_cross_reference_one_dump"] is False
    assert report["controls"]["cross_artifact_and_baseline_pair_one_dump"] is True


def test_a_prior_run_of_the_baseline_pair_is_reproduced_per_position_or_reported_as_drifted(tmp_path):
    matched = (0.02, 0.06, 0.05, 0.03)
    terms = world(tmp_path, matched)
    same = result(tmp_path / "prior-same.json", "old_arm", (0.04,) * 4, nll(4), nll(2))
    drifted = result(tmp_path / "prior-drifted.json", "old_arm", (0.04, 0.04, 0.041, 0.04), nll(4), nll(2))
    args = ["--limit", "0.0590", "--baseline", "0.04916"]

    code, report = run(tmp_path, terms, *args, "--reproduce", same)
    assert (code, report["reproduction"]["identical_windows"], report["reproduction"]["reproduces"]) == (0, 4, True)

    # A newer build may move the baseline pair: by default that is reported, not a reason to void the gate.
    code, report = run(tmp_path, terms, *args, "--reproduce", drifted)
    assert (code, report["valid"]) == (0, True)
    assert (report["reproduction"]["identical_windows"], report["reproduction"]["reproduces"]) == (3, False)
    assert report["reproduction"]["max_abs_diff"] == pytest.approx(0.001)

    code, report = run(tmp_path, terms, *args, "--reproduce", drifted, "--require-reproduction")
    assert (code, report["valid"], report["controls"]["reproduces"]) == (3, False, False)


SCRATCH = "F:/ai/opencode/inference/.scratch"
G422 = f"{SCRATCH}/386/422/results"
G340 = f"{SCRATCH}/kld-2026-09-24/results/g340-gate1.json"


def test_machine_local_it_reproduces_422s_gate_1(tmp_path):
    """#422's results on this machine, read the way its own gate1.py read them: the same numbers it printed
    (`.scratch/386/422/gate1.log`). Its two artifacts differ in 70 matrices, so its cross terms sit only
    7-9% above the matched pair, and not in every window: under #449's rule the gate would have stopped."""
    paths = [f"{G422}/g422-abl.json", f"{G422}/g422-base.json", G340]
    if not all(map(os.path.isfile, paths)):
        pytest.skip(f"#422's results are not on this machine ({G422})")
    terms = ["--matched", f"abliterated={G422}/g422-abl.json", "--cross-artifact", f"default={G422}/g422-abl.json",
             "--cross-reference", f"abliterated={G422}/g422-base.json",
             "--baseline-pair", f"default={G422}/g422-base.json"]

    code, report = run(tmp_path, terms, "--limit", "0.0484", "--baseline", "0.04034",
                       "--reproduce", f"trellis={G340}", "--require-reproduction")

    assert (report["verdict"], report["valid"]) == ("PASS", True)
    assert round(report["kld"], 5) == 0.04088
    assert [round(x, 5) for x in report["ci95"]] == [0.03508, 0.04672]
    assert round(report["ratio_to_baseline_pair"], 4) == 1.0133
    assert [round(x, 4) for x in report["ratio_ci95"]] == [0.9983, 1.0308]
    assert report["lower_windows"] == 7
    assert (report["reproduction"]["identical_windows"], report["reproduction"]["max_abs_diff"]) == (16, 0.0)
    assert round(report["cross"]["cross_artifact"]["kld"], 5) == 0.04362
    assert round(report["cross"]["cross_reference"]["kld"], 5) == 0.04453
    assert [report["cross"][k]["above_windows"] for k in ("cross_artifact", "cross_reference")] == [15, 15]
    assert (code, report["stop"]) == (3, True)
