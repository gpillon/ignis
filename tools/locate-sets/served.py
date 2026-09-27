"""Spec 18 phase B's acceptance: `locate` through `/v1/decide` (GitHub #275).

The pre-registered check (spec 18 acceptance 8, as its "PHASE B REVIVED"
banner and the #275 issue restate it): on a **fresh** set -- set D is spent
by spec 19's check -- asked through the served endpoint on the served
artifact, hq-e8-2b with the residual window, each family's top-1 is **at or
above D's floor rate** (spec 18's rule 4, computed on D):

    logs >= 41/43, records >= 43/45, prose >= 58/67

counted on the family's present questions a `locate` serves -- the ones
whose target is within `LOCATE_MAX_KEYS`, which the endpoint refuses past
(`locate_too_long`) rather than answer unmeasured. Reported beside it: top-3,
the present/absent AUC of `confidence`, the refusals, and the wall time
(against the labelled route when `labelled.py` ran on the same set).

`ask` puts every question of a set to a running server (`make start`), one
`locate` per request, and records what came back; `judge` applies the rule.

    python served.py ask --set .scratch/locate/F --url http://127.0.0.1:8000 --out F-served.json
    python served.py judge --served F-served.json [--labelled F-labelled.json] --out F-acceptance.json
"""

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request

# D's rule-4 floors (docs/findings/2026-09-27-locate-by-head-vote-go.md):
# (hits, questions) per family; a family passes at this rate or above.
FLOORS = {"logs": (41, 43), "records": (43, 45), "prose": (58, 67)}
TOO_LONG = "locate_too_long"


def post(url, body, timeout):
    """(status, payload, milliseconds) of one `/v1/decide` request; a 422 is
    an answer here, not an exception."""
    request = urllib.request.Request(url + "/v1/decide", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, payload = response.status, json.load(response)
    except urllib.error.HTTPError as error:
        status, payload = error.code, json.load(error)
    return status, payload, (time.perf_counter() - started) * 1e3


def ask(args):
    with open(os.path.join(args.set, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    rows = []
    for q in manifest["questions"][:args.limit]:
        body = {"state": q["state"], "questions": {"q": {"type": "locate", "instructions": q["instruction"]}}}
        status, payload, ms = post(args.url, body, args.timeout)
        row = {"id": q["id"], "family": q["family"], "split": q["split"], "absent": q["absent"],
               "segments": q["segments"], "targets": q["targets"], "status": status, "ms": ms}
        if status == 200:
            answer = payload["answers"]["q"]
            row["answer"] = answer
            row["input_tokens"] = payload["usage"]["input_tokens"]
        else:
            row["code"] = payload.get("error", {}).get("code")
        rows.append(row)
        a = row.get("answer", {})
        print(f"  {q['id']} {q['family']:7} {q['split']:10} {'absent ' if q['absent'] else 'present'} "
              f"{status} {a.get('type', row.get('code'))} -> {a.get('segment')} targets {q['targets']} "
              f"conf {a.get('confidence')} ({ms:.0f} ms)", flush=True)
    out = {"seed": manifest.get("seed"), "url": args.url, "questions": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def auc(pos, neg):
    """P(a present question's confidence > an absent one's), ties half."""
    if not pos or not neg:
        return None
    greater = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return greater / (len(pos) * len(neg))


def judge(rows, labelled=None):
    """The rule on `ask`'s rows: per family, the served present questions'
    top-1 against D's floor rate; the verdict is every family at or above
    it."""
    def located(r):
        return r["status"] == 200 and r.get("answer", {}).get("type") == "locate"

    def hit(r):
        return located(r) and not r["absent"] and r["answer"]["segment"] in r["targets"]

    def hit3(r):
        ranked = [entry["segment"] for entry in r["answer"]["ranking"][:3]] if located(r) else []
        return not r["absent"] and bool(set(ranked) & set(r["targets"]))

    refused = [r for r in rows if r["status"] == 422 and r.get("code") == TOO_LONG]
    served = [r for r in rows if not (r["status"] == 422 and r.get("code") == TOO_LONG)]
    present = [r for r in served if not r["absent"]]
    families = {}
    for family, (floor_hits, floor_n) in FLOORS.items():
        sub = [r for r in present if r["family"] == family]
        hits = sum(hit(r) for r in sub)
        families[family] = {
            "hits": hits, "of": len(sub), "top1": 100.0 * hits / len(sub) if sub else None,
            "floor": f"{floor_hits}/{floor_n}", "floor_rate": 100.0 * floor_hits / floor_n,
            # hits / n >= floor_hits / floor_n, in integers.
            "pass": bool(sub) and hits * floor_n >= floor_hits * len(sub),
        }
    splits = {}
    for split in sorted({r["split"] for r in present}):
        sub = [r for r in present if r["split"] == split]
        splits[split] = {"hits": sum(hit(r) for r in sub), "of": len(sub)}
    confidence = lambda r: r["answer"]["confidence"]
    report = {
        "rule": "every family's top-1 at or above D's floor rate, on its present questions within LOCATE_MAX_KEYS",
        "pass": all(f["pass"] for f in families.values()),
        "families": families,
        "splits": splits,
        "top1": {"hits": sum(hit(r) for r in present), "of": len(present)},
        "top3": {"hits": sum(hit3(r) for r in present), "of": len(present)},
        "auc_present_absent": auc([confidence(r) for r in present if located(r)],
                                  [confidence(r) for r in served if r["absent"] and located(r)]),
        "refused_too_long": {"present": sum(not r["absent"] for r in refused), "absent": sum(r["absent"] for r in refused)},
        "errors": [{"id": r["id"], "status": r["status"], "code": r.get("code") or r.get("answer", {}).get("code")}
                   for r in served if not located(r)],
        "wall_ms_median": statistics.median([r["ms"] for r in served if located(r)]) if served else None,
        "input_tokens_median": statistics.median([r["input_tokens"] for r in served if located(r)]) if served else None,
    }
    if labelled:
        by_id = {q["id"]: q for q in labelled["questions"]}
        both = [r for r in present if r["id"] in by_id]
        report["labelled"] = {
            "n": len(both),
            "locate": sum(hit(r) for r in both),
            "labelled": sum(by_id[r["id"]]["hit"] for r in both),
            "locate_ms_median": statistics.median([r["ms"] for r in both]) if both else None,
            "labelled_ms_median": statistics.median([by_id[r["id"]]["choice_ms"] for r in both]) if both else None,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="put a set's questions to a running server as `locate`s")
    a.add_argument("--set", required=True, help="the set directory (manifest.json)")
    a.add_argument("--url", default="http://127.0.0.1:8000")
    a.add_argument("--out", required=True)
    a.add_argument("--timeout", type=float, default=600)
    a.add_argument("--limit", type=int)
    j = sub.add_parser("judge", help="apply the pre-registered rule to `ask`'s output")
    j.add_argument("--served", required=True)
    j.add_argument("--labelled", help="labelled.py's results on the same set")
    j.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "ask":
        ask(args)
        return
    with open(args.served, encoding="utf-8") as f:
        rows = json.load(f)["questions"]
    labelled = None
    if args.labelled:
        with open(args.labelled, encoding="utf-8") as f:
            labelled = json.load(f)
    report = judge(rows, labelled)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for family, f in report["families"].items():
        print(f"{family:8} {f['hits']}/{f['of']} ({f['top1']:.1f}%) floor {f['floor']} ({f['floor_rate']:.1f}%) "
              f"{'PASS' if f['pass'] else 'FAIL'}")
    print(f"splits {json.dumps(report['splits'])}; top-3 {report['top3']}; AUC {report['auc_present_absent']}; "
          f"refused as too long {report['refused_too_long']}; errors {len(report['errors'])}")
    print(f"acceptance: {'PASS' if report['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
