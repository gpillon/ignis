"""Gate 1 for a fine-tune's artifact: its KLD against the fine-tune's own BF16 on set A, held to a limit.

  python gate1.py --limit 0.0590 --baseline 0.04916 \
      --matched swift=results/s470-swift.json --cross-artifact served=results/s470-served.json \
      --cross-reference swift=results/b470-swift.json --baseline-pair served=results/b470-served.json \
      [--reproduce nvfp4_c2048=.../g340-gate1.json] [--require-reproduction] [--json gate1.json]

Every term is NAME=FILE: a `ref_kld.py` result and the dump scored in it. Four terms, two dumps (the
fine-tune's artifact, the vanilla one) against two references (the fine-tune's BF16, the base's):

- matched: the fine-tune's artifact against the fine-tune's BF16, the number the gate holds to `--limit`;
- cross-artifact: the vanilla artifact against the fine-tune's BF16;
- cross-reference: the fine-tune's artifact against the base BF16;
- baseline pair: the vanilla artifact against the base BF16, the registered baseline's own measurement
  (`--baseline`, the number the limit is 1.2 x of), beside which the matched pair's paired ratio is printed.

The KLD is the mean over every position of every window; its 95% CI is the window bootstrap of #340's and
#422's gate 1 (the windows resampled 20,000 times, each weighted by its positions, as the mean is).

The cross terms (spec #449): each must sit above the matched pair, or the gate measured the base and not
the fine-tune. One holds when it is above the matched pair in every window and the 95% CI of its difference
from the matched pair (same resamples) is above 0; `--cross-margin R` also asks the CI of its ratio to the
matched pair to be at or above R. The ratios and differences are printed whatever the rule says.

The controls, each of which voids the gate: the two terms of one dump carry the same engine NLL per position
and the two terms of one reference the same reference NLL (the files are wired as stated); the references'
NLLs differ in every window, and the dumps' too. With `--require-reproduction`, the baseline pair must equal
`--reproduce`'s arm per position, as #422 required of #340's trellis arm; without it the comparison is
reported only, since a newer build may move the baseline pair and the registered limit is absolute.

Exit status: 0 PASS; 1 FAIL; 3 a cross term or a control does not hold, PASS or FAIL alike: the work stops
and the numbers go to the owner. #422's `.scratch/386/422/gate1.py` was this script with its limit, baseline,
dump names and controls written in; `test_gate1.py` reproduces its printed numbers from #422's results.
"""
import argparse
import json
import sys

import numpy as np

TERMS = ("matched", "cross_artifact", "cross_reference", "baseline_pair")
TITLES = {"matched": "fine-tune artifact | fine-tune BF16", "cross_artifact": "vanilla artifact | fine-tune BF16",
          "cross_reference": "fine-tune artifact | base BF16", "baseline_pair": "vanilla artifact | base BF16"}


class Term:
    """One dump's rows in one `ref_kld.py` result."""

    def __init__(self, spec, files):
        self.spec = spec
        self.dump, sep, self.path = spec.partition("=")
        if not sep or not self.dump or not self.path:
            raise SystemExit(f"{spec}: a term is NAME=FILE, the dump's name in a ref_kld.py result")
        if self.path not in files:
            with open(self.path) as f:
                files[self.path] = json.load(f)
        result = files[self.path]
        if self.dump not in result["dumps"]:
            raise SystemExit(f"{self.path} scores no dump {self.dump!r} (it scores {result['dumps']})")
        self.windows = result["windows"]
        self.kl = [np.asarray(r[self.dump]["kl"], dtype=np.float64) for r in self.windows]
        self.sums = np.array([k.sum() for k in self.kl])

    def field(self, name):
        return [np.asarray(r[self.dump][name], dtype=np.float64) for r in self.windows]

    def ref_nll(self):
        return [np.asarray(r["ref_nll"], dtype=np.float64) for r in self.windows]


def boot_ci(num, den, idx):
    """Percentile 95% CI of sum(num) / sum(den) over the resampled windows."""
    b = num[idx].sum(1) / den[idx].sum(1)
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for term in TERMS:
        ap.add_argument("--" + term.replace("_", "-"), required=True, metavar="NAME=FILE", help=TITLES[term])
    ap.add_argument("--limit", type=float, required=True, help="the registered limit on the matched pair's KLD")
    ap.add_argument("--baseline", type=float, required=True,
                    help="the registered baseline the limit is 1.2 x of (the vanilla artifact's KLD, base BF16)")
    ap.add_argument("--reproduce", metavar="NAME=FILE",
                    help="a prior run's arm of the baseline pair (e.g. #340's), compared with it per position")
    ap.add_argument("--require-reproduction", action="store_true",
                    help="void the gate unless the baseline pair reproduces --reproduce per position")
    ap.add_argument("--cross-margin", type=float, default=None,
                    help="also ask each cross term's ratio to the matched pair to have its 95%% CI at or above this")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args(argv)
    files = {}
    t = {term: Term(getattr(a, term), files) for term in TERMS}
    windows = t["matched"].windows
    for term in TERMS[1:]:
        other = t[term].windows
        if [(r["kind"], len(r[t[term].dump]["kl"])) for r in other] != \
                [(r["kind"], len(r[t["matched"].dump]["kl"])) for r in windows]:
            raise SystemExit(f"{t[term].spec}: its windows are not the matched pair's (kinds and lengths)")
    n = np.array([len(k) for k in t["matched"].kl], dtype=np.float64)
    nw = len(windows)
    kinds = ["all"] + sorted(set(r["kind"] for r in windows))
    report = {"inputs": {term: t[term].spec for term in TERMS}, "limit": a.limit, "baseline": a.baseline,
              "windows": nw, "table": {}}

    print(f"{'term':16s} {'kind':6s} {'KLD':>8s} {'median':>7s} {'p99':>6s} {'top-1':>7s} {'NLL':>7s} {'ppl':>6s}"
          f"   ({' / '.join(f'{k}: {v}' for k, v in TITLES.items())})")
    for term in TERMS:
        top1, nlls = t[term].field("top1"), t[term].field("nll")
        for kind in kinds:
            sel = [i for i, r in enumerate(windows) if kind == "all" or r["kind"] == kind]
            kl = np.concatenate([t[term].kl[i] for i in sel])
            t1 = np.concatenate([top1[i] for i in sel]).mean()
            nll = np.concatenate([nlls[i] for i in sel]).mean()
            report["table"][f"{term}/{kind}"] = {"kld": float(kl.mean()), "top1": float(t1), "nll": float(nll)}
            print(f"{term:16s} {kind:6s} {kl.mean():8.5f} {np.median(kl):7.4f} {np.percentile(kl, 99):6.3f} "
                  f"{t1 * 100:6.2f}% {nll:7.4f} {np.exp(nll):6.3f}")

    rng = np.random.default_rng(0)
    idx = rng.integers(0, nw, size=(a.boot, nw))
    m, b = t["matched"].sums, t["baseline_pair"].sums
    kld = float(m.sum() / n.sum())
    ci = boot_ci(m, n, idx)
    base_kld = float(b.sum() / n.sum())
    ratio, ratio_ci = float(m.sum() / b.sum()), boot_ci(m, b, idx)
    lower = int(np.sum(m < b))
    print(f"\ngate: KLD({TITLES['matched']}) = {kld:.5f}, 95% CI [{ci[0]:.5f}, {ci[1]:.5f}] "
          f"(window bootstrap, {a.boot}); limit {a.limit} (1.2 x {a.baseline} = {1.2 * a.baseline:.6f})")
    print(f"paired: / KLD({TITLES['baseline_pair']}) {base_kld:.5f} = {ratio:.4f}, 95% CI [{ratio_ci[0]:.4f}, "
          f"{ratio_ci[1]:.4f}]; the matched pair lower in {lower}/{nw} windows")
    for kind in kinds[1:]:
        g, bb = report["table"][f"matched/{kind}"]["kld"], report["table"][f"baseline_pair/{kind}"]["kld"]
        print(f"    {kind:6s} {g:.5f} / {bb:.5f} = {g / bb:.4f}")
    if 1.2 * a.baseline >= kld > a.limit:
        print(f"NOTE: the KLD lies between the registered {a.limit} and 1.2 x {a.baseline}; the registered "
              f"{a.limit} decides")

    print("\nper window KLD:")
    print("  w  kind  " + " ".join(f"{term:>15s}" for term in TERMS))
    for i in range(nw):
        print(f"  {i:2d} {windows[i]['kind']:5s} " + " ".join(f"{np.mean(t[term].kl[i]):15.5f}" for term in TERMS))

    print("\ncross terms (each must sit above the matched pair, or the gate measured the base, not the fine-tune):")
    cross = {}
    for term in ("cross_artifact", "cross_reference"):
        c = t[term].sums
        above = int(sum(ck.mean() > mk.mean() for ck, mk in zip(t[term].kl, t["matched"].kl)))
        diff_b = (c[idx].sum(1) - m[idx].sum(1)) / n[idx].sum(1)
        x = {"kld": float(c.sum() / n.sum()), "ci95": boot_ci(c, n, idx), "ratio": float(c.sum() / m.sum()),
             "ratio_ci95": boot_ci(c, m, idx), "diff": float((c.sum() - m.sum()) / n.sum()),
             "diff_ci95": [float(v) for v in np.percentile(diff_b, [2.5, 97.5])], "above_windows": above}
        x["above"] = above == nw and x["diff_ci95"][0] > 0 and \
            (a.cross_margin is None or x["ratio_ci95"][0] >= a.cross_margin)
        cross[term] = x
        print(f"  {term:16s} KLD({TITLES[term]}) = {x['kld']:.5f}, CI [{x['ci95'][0]:.5f}, {x['ci95'][1]:.5f}]; "
              f"/ matched {x['ratio']:.3f}, CI [{x['ratio_ci95'][0]:.3f}, {x['ratio_ci95'][1]:.3f}]; - matched "
              f"{x['diff']:.5f}, CI [{x['diff_ci95'][0]:.5f}, {x['diff_ci95'][1]:.5f}]; above in {above}/{nw} "
              f"windows: {'yes' if x['above'] else 'NO'}")
    stop = not all(x["above"] for x in cross.values())
    rule = "above in every window, the CI of the difference above 0" + \
        ("" if a.cross_margin is None else f", the CI of the ratio at or above {a.cross_margin}")
    print(f"  the rule: {rule}")

    print("\ncontrols (each voids the gate if it fails):")
    controls = {}

    def identical(x, y):
        return sum(np.array_equal(p, q) for p, q in zip(x, y))

    for key, (x, y), what in (
            ("matched_and_cross_reference_one_dump", ("matched", "cross_reference"), "the fine-tune's dump"),
            ("cross_artifact_and_baseline_pair_one_dump", ("cross_artifact", "baseline_pair"), "the vanilla dump")):
        same = identical(t[x].field("nll"), t[y].field("nll"))
        controls[key] = same == nw
        print(f"  {x} and {y} score one dump, {what}: its NLL identical per position in {same}/{nw} windows")
    for key, (x, y), what in (
            ("matched_and_cross_artifact_one_reference", ("matched", "cross_artifact"), "the fine-tune's BF16"),
            ("cross_reference_and_baseline_pair_one_reference", ("cross_reference", "baseline_pair"), "the base BF16")):
        same = identical(t[x].ref_nll(), t[y].ref_nll())
        controls[key] = same == nw
        print(f"  {x} and {y} score against one reference, {what}: its NLL identical per position in "
              f"{same}/{nw} windows")
    differ = nw - identical(t["matched"].ref_nll(), t["cross_reference"].ref_nll())
    controls["references_differ"] = differ == nw
    print(f"  the fine-tune's BF16 and the base's: their NLLs differ in {differ}/{nw} windows")
    differ = nw - identical(t["matched"].field("nll"), t["cross_artifact"].field("nll"))
    controls["artifacts_differ"] = differ == nw
    print(f"  the fine-tune's artifact and the vanilla one: their dumps' NLLs differ in {differ}/{nw} windows")
    if a.reproduce:
        prior = Term(a.reproduce, files)
        if [(r["kind"], len(k)) for r, k in zip(prior.windows, prior.kl)] != \
                [(r["kind"], len(k)) for r, k in zip(windows, t["baseline_pair"].kl)]:
            raise SystemExit(f"{a.reproduce}: its windows are not the matched pair's (kinds and lengths)")
        same = identical(t["baseline_pair"].kl, prior.kl)
        ref_same = identical(t["baseline_pair"].ref_nll(), prior.ref_nll()) == nw
        worst = max(float(np.abs(x - y).max()) for x, y in zip(t["baseline_pair"].kl, prior.kl))
        prior_kld = float(prior.sums.sum() / n.sum())
        reproduction = {"prior": a.reproduce, "prior_kld": prior_kld, "identical_windows": same,
                        "max_abs_diff": worst, "reference_identical": ref_same,
                        "reproduces": same == nw and ref_same}
        report["reproduction"] = reproduction
        if a.require_reproduction:
            controls["reproduces"] = reproduction["reproduces"]
        print(f"  {'' if a.require_reproduction else '(reported) '}baseline pair vs {a.reproduce}: {base_kld:.5f} "
              f"vs {prior_kld:.5f}, identical per position in {same}/{nw} windows (max |diff| {worst:.3g}); "
              f"base reference NLL identical: {ref_same}")
    valid = all(controls.values())

    verdict = "PASS" if kld <= a.limit else "FAIL"
    report.update({"kld": kld, "ci95": ci, "baseline_pair_kld": base_kld, "ratio_to_baseline_pair": ratio,
                   "ratio_ci95": ratio_ci, "lower_windows": lower, "cross": cross, "cross_rule": rule,
                   "stop": stop, "controls": controls, "valid": valid, "verdict": verdict})
    print(f"\nGATE 1: {verdict} (KLD {kld:.5f} {'<=' if kld <= a.limit else '>'} {a.limit}, "
          f"95% CI [{ci[0]:.5f}, {ci[1]:.5f}])")
    if not valid:
        print(f"INVALID: a control failed {[k for k, v in controls.items() if not v]}: the terms are not two dumps "
              "against two references; the gate decides nothing")
    if stop:
        print("STOP: a cross term does not sit above the matched pair, so the gate may have measured the base, "
              "not the fine-tune: the work stops and the numbers go to the owner")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
    return 3 if stop or not valid else 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
