"""Spec 22's judge (GitHub #278), committed with the builders before any
route is asked on sets R3, P3 or J3
(`docs/specs/decide/22-locate-by-copy-over-a-folded-state.md` § The runs,
§ Scoring, § The rules).

Inputs: `accept22.py run`'s records of runs 1, 3, 4, 5 and 6, run 2's
(`folded_locate.py --route choice --values` over R3's present questions),
set F's recorded vote (`F-served.json`) and the sets' manifests (P3's for
its paragraphs).

**Scoring.** A present question is **right** when the answer names its
target (for prose, a gold sentence) **and** `found` >= 0.5 where the route
carries `found`. An absent question is **flagged** when `found` < 0.5. The
**pick** is the answer's segment, or — under 0.5 — its ranking's first.

    python r3_judge.py --r3 R3-defaults.json --r3-choice R3-choice.json \
        --p3 P3-defaults.json --p3abs P3abs-defaults.json --p3-manifest <P3>/manifest.json \
        --j3 J3-defaults.json --f-vote F-vote.json --f-defaults F-defaults.json \
        --f-recorded F-served.json --out r3-judge.json
"""

import argparse
import json
import statistics

import numpy as np

FOUND = 0.5
HEADS = 32
# Rule 11: the recorded vote's right answers per family on set F, less five
# points (spec 22 § The rules).
SHORT_FLOORS = {"logs": (39, 43), "records": (44, 46), "prose": (56, 67)}


def load(path):
    return json.load(open(path, encoding="utf-8"))["questions"]


def found_of(row):
    return (row.get("answer") or {}).get("found")


def pick_of(row):
    a = row.get("answer") or {}
    if a.get("segment") is not None:
        return a["segment"]
    ranking = a.get("ranking") or []
    return ranking[0]["segment"] if ranking else None


def keeps(row):
    """Whether the answer names a segment: `found` at or past 0.5, or no
    `found` on its route."""
    f = found_of(row)
    return f is None or f >= FOUND


def right(row):
    return row.get("status") == 200 and pick_of(row) in row["targets"] and keeps(row)


def flagged(row):
    f = found_of(row)
    return f is not None and f < FOUND


def rate(hits, n):
    return {"hits": hits, "n": n, "rate": round(hits / n, 4) if n else None}


def auc(pos, neg):
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def paragraphs(lines):
    """Each line's paragraph index (lines between empty lines)."""
    out, p = [], 0
    for i, line in enumerate(lines):
        if line == "" and i > 0:
            p += 1
        out.append(p)
    return out


def paragraph_f1(row, lines):
    a = row.get("answer") or {}
    pointers = [p["segment"] for p in a.get("pointers") or []]
    if not pointers:
        return 0.0
    of = paragraphs(lines)
    got, gold = {of[s] for s in pointers}, {of[t] for t in row["targets"]}
    both = len(got & gold)
    if not both:
        return 0.0
    precision, recall = both / len(got), both / len(gold)
    return 2 * precision * recall / (precision + recall)


def sentence_f1(row):
    a = row.get("answer") or {}
    got, gold = {p["segment"] for p in a.get("pointers") or []}, set(row["targets"])
    both = len(got & gold)
    if not got or not both:
        return 0.0
    precision, recall = both / len(got), both / len(gold)
    return 2 * precision * recall / (precision + recall)


def wall(rows):
    ms = sorted(r["ms"] for r in rows)
    if not ms:
        return None
    return {"median_s": round(statistics.median(ms) / 1e3, 3), "p90_s": round(ms[int(0.9 * (len(ms) - 1))] / 1e3, 3),
            "n": len(ms)}


def by(rows, key, fn):
    groups = {}
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r)
    return {k: rate(sum(fn(r) for r in v), len(v)) for k, v in sorted(groups.items())}


def located(paths):
    """Every `ignis.decide.located` line of the server's JSON logs, by the
    question id the runner asked it under (the last one wins on a rerun)."""
    out = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "ignis.decide.located" not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                a = event.get("attributes") or {}
                if a.get("question"):
                    out[a["question"]] = a
    return out


def target_template(lines, target):
    """The template a fold of the window's non-blank lines puts the target
    in, as the served fold numbers them (first seen)."""
    import compress as C
    content = [i for i, line in enumerate(lines) if line.strip()]
    folded = C.fold([lines[i] for i in content], C.SIM, values=True)
    at = content.index(target)
    return next(n for n, c in enumerate(folded.clusters) if at in c.members)


def from_log(args, r3, p3, j3):
    """What spec 22 reports beside the rules that only the request log holds:
    level-1 accuracy, shortlist recall, windows read, and the host time of the
    plan (the fold included) and of `auto`."""
    log = located(args.log)
    rep = {"log_lines": len(log)}
    windows = json.load(open(args.r3_manifest, encoding="utf-8"))["windows"] if args.r3_manifest else {}

    def last(a):
        lists = (a.get("candidates") or "").split(";")
        return [int(x) for x in lists[-1].split(",") if x] if lists and lists[-1] else []

    for name, rows in (("R3", [r for r in r3 if r.get("variant") == "present"]), ("P3", p3),
                       ("J3", [r for r in j3 if not r["absent"]])):
        seen = [(r, log[r["id"]]) for r in rows if r["id"] in log]
        if not seen:
            continue
        recall = sum(any(t in last(a) for t in r["targets"]) for r, a in seen)
        rep[f"{name} shortlist recall"] = rate(recall, len(seen))
        rep[f"{name} windows read"] = {str(k): v for k, v in sorted(
            {w: sum(1 for _, a in seen if a.get("windows") == w) for w in {a.get("windows") for _, a in seen}}.items(),
            key=lambda kv: str(kv[0]))}
        plans = sorted(a["plan_ms"] for _, a in seen if a.get("plan_ms") is not None)
        if plans:
            rep[f"{name} plan host ms"] = {"median": round(statistics.median(plans), 1), "max": round(plans[-1], 1)}
        autos = sorted(a["auto_ms"] for _, a in seen if a.get("auto_ms") is not None)
        if autos:
            rep[f"{name} auto host ms"] = {"median": round(statistics.median(autos), 1), "max": round(autos[-1], 1)}
        if name == "R3" and windows:
            level1 = [(r, a) for r, a in seen if a.get("template") is not None]
            hits = sum(a["template"] == target_template(windows[r["window"]], r["targets"][0]) for r, a in level1)
            rep["R3 level-1 accuracy"] = rate(hits, len(level1))
            longest = max(level1, key=lambda ra: ra[0].get("tier", 0), default=None)
            if longest:
                rep["R3 longest window's plan host ms"] = longest[1].get("plan_ms")
    return rep


def main():
    ap = argparse.ArgumentParser()
    for name in ("r3", "r3-choice", "p3", "p3abs", "p3-manifest", "j3", "f-vote", "f-defaults", "f-recorded", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--r3-manifest", help="R3's manifest, for the level-1 accuracy the log is read against")
    ap.add_argument("--log", nargs="*", default=[], help="the server's JSON request logs of the runs")
    args = ap.parse_args()
    out = {"rules": {}, "reported": {}}
    rules = out["rules"]

    # ---------------------------------------------------------------- R3
    r3 = load(args.r3)
    present = [r for r in r3 if r.get("variant") == "present"]
    removed = [r for r in r3 if r.get("variant") == "removed"]
    authored = [r for r in r3 if r.get("variant") == "authored"]
    right_r3 = sum(right(r) for r in present)
    rules["1 logs"] = dict(rate(right_r3, len(present)), floor=0.85, pass_=right_r3 >= 0.85 * len(present))
    choice_rows = {r["id"]: r for r in load(args.r3_choice)}
    top1 = sum(pick_of(r) in r["targets"] for r in present)
    choice_hits = sum(bool(choice_rows.get(r["id"], {}).get("hit")) for r in present)
    rules["2 the heads' part"] = {"defaults_top1": top1, "choice_alone": choice_hits, "n": len(present),
                                  "missing_from_run2": sum(r["id"] not in choice_rows for r in present),
                                  "pass_": top1 >= choice_hits}
    firsts = [r for r in r3 if r.get("tier") == 100_000 and r.get("first_of_state") and r.get("variant") != "removed"]
    median = statistics.median(r["ms"] for r in firsts) / 1e3 if firsts else None
    rules["3 latency"] = {"median_s": median, "windows": len(firsts), "ceiling_s": 3.0,
                          "pass_": median is not None and median <= 3.0}
    picked = [r for r in present if pick_of(r) in r["targets"]]
    kept = sum(keeps(r) for r in picked)
    flag_removed = sum(flagged(r) for r in removed)
    flag_authored = sum(flagged(r) for r in authored)
    rules["6 not found, logs"] = {
        "present_kept": dict(rate(kept, len(picked)), floor=0.95),
        "removed_flagged": dict(rate(flag_removed, len(removed)), floor=0.60),
        "authored_flagged": dict(rate(flag_authored, len(authored)), floor=0.65),
        "pass_": kept >= 0.95 * len(picked) and flag_removed >= 0.60 * len(removed)
        and flag_authored >= 0.65 * len(authored),
    }

    # ---------------------------------------------------------------- P3
    p3 = load(args.p3)
    p3abs = load(args.p3abs)
    states = {q["id"]: q["state"].split("\n") for q in json.load(open(args.p3_manifest, encoding="utf-8"))["questions"]}
    upto = [r for r in p3 if r.get("segments", 0) <= 200_000]
    right_p3 = sum(right(r) for r in upto)
    f1 = [paragraph_f1(r, states[r["id"]]) for r in p3]
    mean_f1 = sum(f1) / len(f1) if f1 else 0.0
    rules["4 prose"] = {"right_upto_200k": dict(rate(right_p3, len(upto)), floor=0.85),
                        "paragraph_pointer_f1": {"mean": round(mean_f1, 4), "n": len(f1), "floor": 0.75},
                        "pass_": right_p3 >= 0.85 * len(upto) and mean_f1 >= 0.75}
    gold_picked = [r for r in upto if pick_of(r) in r["targets"]]
    kept_p3 = sum(keeps(r) for r in gold_picked)
    deleted = [r for r in p3abs if r.get("variant") == "deleted" or "~del" in r["id"]]
    cross = [r for r in p3abs if r not in deleted]
    flag_deleted, flag_cross = sum(flagged(r) for r in deleted), sum(flagged(r) for r in cross)
    rules["7 not found, prose"] = {
        "present_kept": dict(rate(kept_p3, len(gold_picked)), floor=0.95),
        "gold_removed_flagged": dict(rate(flag_deleted, len(deleted)), floor=0.50),
        "other_window_flagged": dict(rate(flag_cross, len(cross)), floor=0.90),
        "pass_": kept_p3 >= 0.95 * len(gold_picked) and flag_deleted >= 0.50 * len(deleted)
        and flag_cross >= 0.90 * len(cross),
    }

    # ---------------------------------------------------------------- J3
    j3 = load(args.j3)
    j_present = [r for r in j3 if not r["absent"]]
    j_absent = [r for r in j3 if r["absent"]]
    j_10k = [r for r in j_present if r.get("segments") == 10_000]
    right_j3, right_10k = sum(right(r) for r in j_present), sum(right(r) for r in j_10k)
    rules["5 records"] = {"right": dict(rate(right_j3, len(j_present)), floor=0.90),
                          "right_10000": dict(rate(right_10k, len(j_10k)), floor=0.85),
                          "pass_": right_j3 >= 0.90 * len(j_present) and right_10k >= 0.85 * len(j_10k)}
    j_picked = [r for r in j_present if pick_of(r) in r["targets"]]
    kept_j3, flag_j3 = sum(keeps(r) for r in j_picked), sum(flagged(r) for r in j_absent)
    rules["8 not found, records"] = {"present_kept": dict(rate(kept_j3, len(j_picked)), floor=0.95),
                                     "absent_flagged": dict(rate(flag_j3, len(j_absent)), floor=0.90),
                                     "pass_": kept_j3 >= 0.95 * len(j_picked) and flag_j3 >= 0.90 * len(j_absent)}

    # ---------------------------------------------------------------- F
    f_vote, f_def, f_rec = load(args.f_vote), load(args.f_defaults), load(args.f_recorded)
    kinds = {
        "R3 -> log": [r for r in r3],
        "P3 -> prose": p3 + p3abs,
        "J3 -> records": j3,
        "F records -> records": [r for r in f_def if r.get("family") == "records"],
    }
    wrong, failed = {}, {}
    for name, rows in kinds.items():
        want = name.split("-> ")[1]
        # A question that failed says nothing of `auto`: counted apart.
        answered = [r for r in rows if (r.get("answer") or {}).get("type") == "locate"]
        miss = [r["id"] for r in answered if r["answer"].get("kind") != want]
        if miss:
            wrong[name] = miss[:20] + ([f"+{len(miss) - 20}"] if len(miss) > 20 else [])
        if len(answered) < len(rows):
            failed[name] = [r["id"] for r in rows if r not in answered][:20]
    rules["9 auto"] = {"wrong": wrong, "not_answered": failed, "pass_": not wrong}

    recorded = {r["id"]: r for r in f_rec}
    vote = {r["id"]: r for r in f_vote}
    diffs = []
    for qid, rec in recorded.items():
        now = vote.get(qid)
        if now is None:
            diffs.append((qid, "not asked"))
            continue
        rec_refused = rec.get("status") != 200
        if rec_refused or now.get("status") != 200:
            if (rec.get("code"), rec_refused) != (now.get("code"), now.get("status") != 200):
                diffs.append((qid, f"refusal {rec.get('code')} vs {now.get('code')}"))
            continue
        ranking = rec["answer"]["ranking"]
        votes = [round(e["share"] * HEADS) for e in ranking]
        lead = votes[0] - (votes[1] if len(votes) > 1 else 0)
        segment = now["answer"].get("segment")
        allowed = [ranking[0]["segment"]] if lead > 1 else [e["segment"] for e in ranking[:2]]
        if segment not in allowed:
            diffs.append((qid, f"segment {segment}, recorded {ranking[0]['segment']} by {lead}"))
    rules["10 the vote unchanged"] = {"questions": len(recorded), "differences": diffs, "pass_": not diffs}

    served = {qid for qid, rec in recorded.items() if rec.get("status") == 200 and not rec["absent"]}
    short = {}
    ok = True
    for family, (floor, registered) in SHORT_FLOORS.items():
        rows = [r for r in f_def if r["id"] in served and r.get("family") == family]
        hits = sum(right(r) for r in rows)
        short[family] = {"hits": hits, "n": len(rows), "floor": floor, "registered_n": registered}
        ok = ok and hits >= floor
    rules["11 short states"] = dict(short, pass_=ok)

    # ---------------------------------------------------------- reported
    rep = out["reported"]
    rep["R3 top-1 by source"] = by(present, "source", lambda r: pick_of(r) in r["targets"])
    rep["R3 right by tier"] = by(present, "tier", right)
    rep["R3 right by sibling bin"] = by(present, "bin", right)
    rep["R3 wall, present"] = wall(present)
    rep["R3 wall by tier, first of window"] = {
        str(t): wall([r for r in r3 if r.get("tier") == t and r.get("first_of_state") and r.get("variant") != "removed"])
        for t in sorted({r.get("tier") for r in r3})}
    rep["R3 prompt tokens, median"] = statistics.median(r.get("input_tokens", 0) for r in present) if present else None
    rep["R3 found AUC"] = {"removed": auc([found_of(r) for r in picked], [found_of(r) for r in removed]),
                           "authored": auc([found_of(r) for r in picked], [found_of(r) for r in authored])}
    rep["R3 confidence and found, right vs wrong"] = {
        "right": [(r["answer"].get("confidence"), found_of(r)) for r in present if right(r)][:200],
        "wrong": [(r["answer"].get("confidence"), found_of(r), r["id"]) for r in present if r.get("status") == 200 and not right(r)]}
    rep["R3 run 2 by tier"] = by([choice_rows[r["id"]] | {"tier": r.get("tier")} for r in present if r["id"] in choice_rows],
                                 "tier", lambda r: bool(r.get("hit")))
    rep["P3 right by tier"] = by(p3, "segments", right)
    rep["P3 1M group"] = rate(sum(right(r) for r in p3 if r["segments"] > 200_000), sum(r["segments"] > 200_000 for r in p3))
    rep["P3 sentence pointer F1"] = round(sum(sentence_f1(r) for r in p3) / max(1, len(p3)), 4)
    rep["P3 paragraph pointer F1 by tier"] = {
        str(t): round(float(np.mean([paragraph_f1(r, states[r["id"]]) for r in p3 if r["segments"] == t])), 4)
        for t in sorted({r["segments"] for r in p3})}
    rep["P3 wall by tier"] = {str(t): wall([r for r in p3 if r["segments"] == t]) for t in sorted({r["segments"] for r in p3})}
    rep["P3 found AUC"] = {"deleted": auc([found_of(r) for r in gold_picked], [found_of(r) for r in deleted]),
                           "cross": auc([found_of(r) for r in gold_picked], [found_of(r) for r in cross])}
    rep["J3 right by length"] = by(j_present, "segments", right)
    rep["J3 right by split"] = by(j_present, "split", right)
    rep["J3 wall by length"] = {str(t): wall([r for r in j3 if r["segments"] == t]) for t in sorted({r["segments"] for r in j3})}
    rep["J3 found AUC"] = auc([found_of(r) for r in j_picked], [found_of(r) for r in j_absent])
    f_absent = [r for r in f_def if r["absent"]]
    rep["F absent flagged by family (defaults)"] = by(f_absent, "family", flagged)
    rep["F right by family (defaults, all present)"] = by([r for r in f_def if not r["absent"]], "family", right)

    if args.log:
        rep.update(from_log(args, r3, p3, j3))

    out["all_pass"] = all(rule["pass_"] for rule in rules.values())
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for name, rule in rules.items():
        print(f"{name:26s} {'PASS' if rule['pass_'] else 'FAIL'}  {json.dumps({k: v for k, v in rule.items() if k != 'pass_'})[:300]}")
    print("all rules pass" if out["all_pass"] else "a rule FAILED: reported as failed; the owner decides")


if __name__ == "__main__":
    main()
