"""The end-of-run verdicts and records (spec 01 acceptance 2-4, story 19: fail loudly)."""
import torch

import finish
import layout
import pipeline as P


def layer(L, db, gu=(4, 5), dn=(4, 5), traffic=(3, 1)):
    k2 = {"gu": list(gu), "dn": list(dn)}
    return {"layer": L, "moe_db": db, "k2": k2, "expert_traffic": list(traffic),
            "rates": {p: sum(layout.rate(p, k) for k in k2[p]) / len(k2[p]) for p in k2}}


def summary(q_kld, f8_kld=0.005):
    row = {"kld": q_kld, "kld_top64": q_kld, "top1": 0.9}
    return {"q": {"code": dict(row), "chat": dict(row)},
            "f8": {"code": dict(row, kld=f8_kld), "chat": dict(row, kld=f8_kld)}, "head_fp8_only": {}}


def mmlu(ref_ok=200, q_ok=200, n=281):
    return {"ref": [(1, 1)] * ref_ok + [(0, 1)] * (n - ref_ok), "q": [(1, 1)] * q_ok + [(0, 1)] * (n - q_ok),
            "f8": [(1, 1)] * ref_ok + [(0, 1)] * (n - ref_ok)}


def full_layers(db=None):
    """48 layers at `db`, or by default at run 6's dB or -17, whichever is lower (inside every bound)."""
    return [layer(L, db if db is not None else min(P.RUN6_DB[L], -17.0)) for L in range(48)]


def test_a_conversion_inside_every_bound_passes_with_exit_0():
    v, code = finish.verdict(full_layers(), summary(0.10), mmlu(), True, 2.5, full=True)
    assert (v["verdict"], code) == ("PASS", 0)
    assert v["acceptance2"]["pass"] and v["acceptance3"]["pass"] and v["acceptance4"]["pass"]


def test_kld_over_run6_times_1_1_fails_and_names_the_fallback():
    v, code = finish.verdict(full_layers(), summary(0.119 * 1.1 + 1e-4), mmlu(), True, 2.5, full=True)
    assert (v["verdict"], code) == ("FAIL", 3)
    assert not v["acceptance4"]["kld_pass"] and "3.0" in v["acceptance4"]["fallback"]
    assert v["kld"]["quantized"]["chat"]["limit"] is None        # chat is reported, not judged


def test_mmlu_below_the_floor_or_significantly_below_bf16_fails():
    v, _ = finish.verdict(full_layers(), summary(0.1), mmlu(ref_ok=199, q_ok=199), True, 2.5, full=True)
    assert not v["acceptance4"]["mmlu_pass"]                     # 199/281 = 70.8% < 71%
    v, _ = finish.verdict(full_layers(), summary(0.1), mmlu(ref_ok=230, q_ok=205), True, 2.5, full=True)
    assert v["mmlu"]["mcnemar"]["quantized"]["lost"] == 25 and not v["acceptance4"]["mmlu_pass"]


def test_layers_1_to_5_must_average_within_half_a_db_of_run8_and_bad_layers_are_named():
    ls = full_layers(-16.2)                                      # -16.2 > -16.8 + 0.5
    v, code = finish.verdict(ls, summary(0.1), mmlu(), True, 2.5, full=True)
    assert not v["acceptance3"]["pass"] and code == 3
    ls = full_layers()
    ls[7]["moe_db"] = P.RUN6_DB[7] + 1.5
    v, _ = finish.verdict(ls, summary(0.1), mmlu(), True, 2.5, full=True)
    assert v["acceptance3"]["pass"] and v["acceptance3"]["named_layers_worse_than_run6_by_1db"] == [7]


def test_the_rate_bound_follows_the_budget():
    ls = [layer(L, -17.0, gu=(6, 6), dn=(6, 6)) for L in range(48)]   # 3 bits + scales
    assert not finish.verdict(ls, summary(0.1), mmlu(), True, 2.5, full=True)[0]["acceptance2"]["pass"]
    assert finish.verdict(ls, summary(0.1), mmlu(), True, 3.05, full=True)[0]["acceptance2"]["pass"]


def test_fp8_only_over_0_01_is_flagged_not_failed():
    v, code = finish.verdict(full_layers(), summary(0.1, f8_kld=0.02), mmlu(), True, 2.5, full=True)
    assert code == 0 and len(v["flags"]) == 2 and "FP8-only" in v["flags"][0]


def test_a_work_file_redecode_mismatch_fails_and_a_dry_run_never_fails():
    assert finish.verdict(full_layers(), summary(0.1), mmlu(), False, 2.5, full=True)[1] == 3
    v, code = finish.verdict(full_layers()[:2], summary(9.0), mmlu(q_ok=0), False, 2.5, full=False)
    assert (v["verdict"], code) == ("DRY-RUN", 0)


def test_summarize_means_per_domain():
    acc = finish._accumulator()
    acc["q"]["kl"]["code"] += [torch.tensor([0.1, 0.3])]
    acc["q"]["kl64"]["code"] += [torch.tensor([0.1, 0.1])]
    acc["q"]["top1"]["code"] += [torch.tensor([1.0, 0.0])]
    acc["q"]["nll"]["code"] += [torch.tensor([0.0, 0.0])]
    acc["ref"]["nll"]["code"] += [torch.tensor([0.0, 0.0])]
    s = finish.summarize(acc)
    assert abs(s["q"]["code"]["kld"] - 0.2) < 1e-6 and s["q"]["code"]["top1"] == 0.5 and s["q"]["code"]["ppl"] == 1.0
    assert s["f8"] == {}


def test_k_classes_count_projections_selections_and_bytes():
    ls = [layer(0, -17, gu=(4, 6), dn=(4, 4), traffic=(3, 1))]
    c = {x["class"]: x for x in finish.k_classes(ls)}
    assert c["gu-2"]["projections"] == 1 and c["gu-2"]["selections"] == 3 and c["gu-3"]["selections"] == 1
    assert c["dn-2"]["projections"] == 2 and c["dn-2"]["selections"] == 4
    assert c["gu-2"]["record_bytes"] == 827392 and c["gu-4"]["projections"] == 0
    assert abs(c["gu-2"]["traffic_share"] - 0.75) < 1e-12


def test_the_report_names_the_verdict_and_every_domain():
    ls = [dict(layer(L, -17.0), run6_db=P.RUN6_DB[L], run8_db=None, k_hist={"gu": [1, 0, 1, 0], "dn": [2, 0, 0, 0]},
               time_s=700.0) for L in range(2)]
    v, _ = finish.verdict(ls, summary(0.1), mmlu(), True, 2.5, full=False)
    r = {"verdict": v["verdict"], "status": "dry-run", "flags": ["FLAGGED"], "source": {"repo": "r", "revision": "x"},
         "converter": {"commit": "c", "dirty": False}, "versions": {"exllamav3": "1.5.3", "transformers": "5.17.0"},
         "rates": {"mean": {"gu": 2.5, "dn": 2.5, "gu_stored": 2.52, "dn_stored": 2.54}},
         "acceptance2": v["acceptance2"], "acceptance3": v["acceptance3"], "kld": v["kld"],
         "kld_head_fp8_only": {"code": 0.001}, "mmlu": v["mmlu"], "acceptance4": v["acceptance4"],
         "self_check": {"work_files": {"projections_checked": 4, "bit_identical": True}}, "time_s": {"total": 3600}}
    text = finish.render_report(r, ls)
    assert text.startswith("Flash-Next conversion: DRY-RUN") and "FLAG FLAGGED" in text
    assert "  code " in text and "  chat " in text and "time: 1.00 h" in text
