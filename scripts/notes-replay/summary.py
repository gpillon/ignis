"""Sums up one replay of replay.py (GitHub #429): prints JSON and writes <dir>/summary.json.

    python scripts/notes-replay/summary.py .scratch/notes-replay/<label>
"""
import json, re, statistics as st, sys
from pathlib import Path

d = Path(sys.argv[1])
rows = {w: [json.loads(l) for l in (d / f"rows.{w}.jsonl").read_text(encoding="utf-8").splitlines()]
        for w in ("R", "A", "C") if (d / f"rows.{w}.jsonl").exists()}
meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    q = sorted(xs)
    return {"mean": round(st.mean(xs), 2), "median": round(st.median(xs), 2), "p95": round(q[min(len(q) - 1, int(0.95 * len(q)))], 2),
            "max": round(max(xs), 2), "total": round(sum(xs), 1)}


out = {"meta": meta}
for way in ("R", "A"):
    if way not in rows:
        continue
    up = [r for r in rows[way] if r["way"] == way]
    lk = [r for r in rows[way] if r["way"] == "look"]
    o = {"updates": len(up), "secs": stats([r["secs"] for r in up]), "decode": stats([r["completion"] for r in up]),
         "looks": len(lk), "look_secs": stats([r["secs"] for r in lk])}
    prev_lines = [0] + [r["lines"] for r in up[:-1]]
    kept = [r["kept"] / p for r, p in zip(up, prev_lines) if p]
    o["kept_word_for_word_of_previous"] = stats(kept)
    o["lines_end"] = up[-1]["lines"]
    if way == "A":
        o["prompt_tokens"] = stats([r["prompt"] for r in up])
    else:
        o["rounds"] = stats([r["rounds"] for r in up])
        o["round_cap_hit"] = sum(r["capped"] for r in up)
        o["restarts"] = sum(r["restart"] for r in up)
        o["thread_tokens"] = stats([r["thread_tokens"] for r in up])
        o["reuse_first_round"] = stats([r["cached0"] / r["prompt0"] for r in up if r["prompt0"] and r["cached0"] is not None and not r["restart"] and r is not up[0]])
        o["calls"] = sum(r["calls"] for r in up)
        o["noop"] = sum(r["noop"] for r in up)
        o["refused"] = sum(r["refused"] for r in up)
        o["bad_json"] = sum(r["bad_json"] for r in up)
        o["text_instead"] = sum(r["text_instead"] for r in up)
        reasons = {}
        for r in up:
            for t in r["refusals"]:
                k = re.sub(r"N\d+|\d+|'[^']*'", "#", t)[:70]
                reasons[k] = reasons.get(k, 0) + 1
        o["refusal_kinds"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
        o["guards"] = {k: sum(r["guards"][k] for r in up) for k in up[0]["guards"]}
        secs = list(up[0]["per_section"])
        o["sections_over_time"] = {s: [r["per_section"][s] for r in up] for s in secs}
        o["sections_filled_updates"] = {s: sum(1 for r in up if r["per_section"][s] > 0) for s in secs}
        snaps = json.loads((d / "running.snapshots.json").read_text(encoding="utf-8"))
        people = next((s for s in secs if s.startswith("People") or s.startswith("Players")), None)
        if people:
            last = snaps[-1]["notes"][people]
            heads = [re.split(r"[:,(—–-]", t, maxsplit=1)[0].strip().casefold() for t in last]
            o["people_end"] = last
            o["people_duplicate_heads"] = sorted({h for h in heads if heads.count(h) > 1})
    out[way] = o
if "C" in rows:
    c = rows["C"][0]
    out["C"] = {"secs": c["secs"], "prompt": c["prompt"], "decode": c["completion"]}
(d / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
