"""EXPERIMENT (branch locate-long-context, exploratory): long JSON arrays of
records as a `locate` kind of their own.

`build`: arrays of spec 18's record kinds (employees by city, tickets by
title, products by name) at lengths past spec 18's 300, each holding several
targets whose selecting value is unique in the array; per target a lexical
and a paraphrase question (spec 18's checks, against every other record,
other targets included), and per array absent questions (a lexical and a
paraphrase one for a value no record holds).

`heads`: the served heads and the end heads read each question over the
**native array** (segments = records, the product's own segmentation),
reduced to key features. `rank`: sum / end / both readings, per head set.

`ask`: over a ranking's first 16 records (array order, compact JSON, labelled)
the plain `choice`, the `choice` with a "none" option and a yes/no, as
`zd_notfound.py` asks them. `logpipe`: the `log` pipeline over the records as
compact JSON lines (what `auto` sends records to), same three questions.

    python zd_records.py build --out J
    python zd_records.py heads --set J --heads served.json endheads.json --out Jw
    python zd_records.py rank --set J --dir Jw --out J-rank.json
    python zd_records.py ask --set J --rank J-rank.json --reading endheads:end --out J-ask-end.json
    python zd_records.py logpipe --set J --heads endheads.json --out J-log.json
    python zd_records.py report ...
"""

import argparse
import glob
import json
import os
import random

import numpy as np

import long_rows as LR
import records as RC
from common import paraphrase_clean, rare_shared
from zd_cache import key_features
from zd_notfound import NONE, FOUND, auc
from zd_notfound import ask as ask_lines
from zd_notfound import one as log_one
from zd_offline import zsum
from zd_qtok import post

KINDS = ("employees", "tickets", "products")


def line(record):
    return json.dumps(record, ensure_ascii=False)


def build_array(r, kind, length, targets, absent):
    if kind == "employees":
        build, pool, filler = RC._employee, sorted(RC.CAPITALS), lambda: r.choice(RC.CITIES)
    elif kind == "tickets":
        build, pool, filler = RC._ticket, sorted(RC.TICKETS), lambda: f"{r.choice(RC.TICKET_AREAS)} {r.choice(RC.TICKET_SYMPTOMS)}"
    else:
        build, pool, filler = RC._product, sorted(RC.PRODUCTS), lambda: f"{r.choice(RC.PRODUCT_ADJ)} {r.choice(RC.PRODUCT_NOUNS)}"
    values = r.sample(pool, targets + absent)
    where = sorted(r.sample(range(length), targets))
    at = dict(zip(where, values[:targets]))
    records = [build(r, i, at.get(i) or filler()) for i in range(length)]
    return records, [(i, at[i]) for i in where], values[targets:]


def questions_for(kind, value):
    if kind == "employees":
        return f"Which employee works from {value}?", f"Which employee works from the capital of {RC.CAPITALS[value]}?"
    if kind == "tickets":
        return f"Which ticket is titled \"{value}\"?", f"Which ticket is about {RC.TICKETS[value]}?"
    return f"Which product is the {value.lower()}?", f"Which product {RC.PRODUCTS[value]}?"


def build(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    r = random.Random(args.seed)
    out, windows = [], []
    for length in args.lengths:
        for kind in KINDS:
            wid = f"j{length:05d}-{kind}"
            for _ in range(50):
                records, targets, missing = build_array(r, kind, length, args.targets, args.absent)
                texts = [RC._text(x) for x in records]
                ok = True
                qs = []
                for i, value in targets:
                    lexical, paraphrase = questions_for(kind, value)
                    rest = [t for j, t in enumerate(texts) if j != i]
                    if not rare_shared(lexical, texts[i], rest) or not paraphrase_clean(paraphrase, texts[i]):
                        ok = False
                        break
                    qs += [("lexical", lexical, [i]), ("paraphrase", paraphrase, [i])]
                for value in missing:
                    lexical, paraphrase = questions_for(kind, value)
                    qs += [("lexical", lexical, []), ("paraphrase", paraphrase, [])]
                if ok:
                    break
            else:
                raise SystemExit(f"{wid}: word checks failed 50 times")
            tokens = len(tok.encode(json.dumps(records, ensure_ascii=False)).ids)
            windows.append({"id": wid, "kind": kind, "records": length, "tokens": tokens})
            for n, (split, instruction, t) in enumerate(qs):
                out.append({"id": f"{wid}-{n:02d}", "family": "records", "kind": kind, "split": split,
                            "absent": not t, "segments": length, "window": wid, "tokens": tokens,
                            "state": records, "instruction": instruction, "targets": t})
            print(wid, length, "records", tokens, "tokens", f"{tokens / length:.1f}/record", len(qs), "questions", flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"seed": args.seed, "windows": windows, "questions": out},
              open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"))


def heads(args):
    sets = [(os.path.splitext(os.path.basename(h))[0], json.load(open(h, encoding="utf-8"))["heads"]) for h in args.heads]
    qs = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    os.makedirs(args.out, exist_ok=True)
    for q in qs:
        for name, hs in sets:
            path = os.path.join(args.out, f"{q['id']}.{name}.npz")
            if os.path.exists(path):
                continue
            with open(args.control, "w", encoding="utf-8") as f:
                json.dump({"heads": hs, "tag": name}, f)
            before = set(glob.glob(os.path.join(args.raw, "*.json")))
            body = {"state": q["state"], "questions": {"q": {"type": "locate", "instructions": q["instruction"]}}}
            status, payload, ms = post(args.url, body)
            with open(args.control, "w", encoding="utf-8") as f:
                json.dump({}, f)
            new = sorted(set(glob.glob(os.path.join(args.raw, "*.json"))) - before)
            if len(new) != 1:
                print(q["id"], name, "no dump", status, str(payload)[:300], flush=True)
                continue
            base = os.path.splitext(new[0])[0]
            meta, qr, nar = LR.load(os.path.dirname(base), os.path.basename(base))
            keys = [tuple(k) if k else None for k in meta["keys"]]
            fq, fn = key_features(qr, keys, meta["span"]), key_features(nar, keys, meta["span"])
            arrays = {f"{k}/q": fq[k].astype(np.float32) for k in ("sum", "last", "next1", "sep")}
            arrays.update({f"{k}/na": fn[k].astype(np.float32) for k in ("sum", "last", "next1", "sep")})
            arrays["meta"] = np.array(json.dumps({"ms": ms, "segments": len(keys)}))
            np.savez_compressed(path, **arrays)
            for ext in (".json", ".bin"):
                os.remove(base + ext)
            print(f"{q['id']} {name} {ms / 1e3:.1f} s", flush=True)


def cut(records, tok, budget):
    """[(first, end)] record ranges of at most `budget` tokens each."""
    out, start, total = [], 0, 0
    for i, x in enumerate(records):
        cost = len(tok.encode(line(x)).ids) + 1
        if total + cost > budget and i > start:
            out.append((start, i))
            start, total = i, 0
        total += cost
    out.append((start, len(records)))
    return out


def heads_windows(args):
    """`heads` past one window: the array cut at record boundaries into
    sub-arrays of at most `--window` tokens, windows outer and questions
    inner (every question after the first claims the window's prefix)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    sets = [(os.path.splitext(os.path.basename(h))[0], json.load(open(h, encoding="utf-8"))["heads"]) for h in args.heads]
    qs = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    by = {}
    for q in qs:
        by.setdefault(q["window"], []).append(q)
    os.makedirs(args.out, exist_ok=True)
    for wid, group in by.items():
        records = group[0]["state"]
        for si, (a, b) in enumerate(cut(records, tok, args.window)):
            for q in group:
                for name, hs in sets:
                    path = os.path.join(args.out, f"{q['id']}.s{si:02d}.{name}.npz")
                    if os.path.exists(path):
                        continue
                    with open(args.control, "w", encoding="utf-8") as f:
                        json.dump({"heads": hs, "tag": name}, f)
                    before = set(glob.glob(os.path.join(args.raw, "*.json")))
                    body = {"state": records[a:b], "questions": {"q": {"type": "locate", "instructions": q["instruction"]}}}
                    status, payload, ms = post(args.url, body)
                    with open(args.control, "w", encoding="utf-8") as f:
                        json.dump({}, f)
                    new = sorted(set(glob.glob(os.path.join(args.raw, "*.json"))) - before)
                    if len(new) != 1:
                        print(q["id"], si, name, "no dump", status, str(payload)[:300], flush=True)
                        continue
                    base = os.path.splitext(new[0])[0]
                    meta, qr, nar = LR.load(os.path.dirname(base), os.path.basename(base))
                    keys = [tuple(k) if k else None for k in meta["keys"]]
                    fq, fn = key_features(qr, keys, meta["span"]), key_features(nar, keys, meta["span"])
                    arrays = {f"{k}/q": fq[k].astype(np.float32) for k in ("sum", "last", "next1", "sep")}
                    arrays.update({f"{k}/na": fn[k].astype(np.float32) for k in ("sum", "last", "next1", "sep")})
                    arrays["meta"] = np.array(json.dumps({"ms": ms, "first": a, "end": b, "sub": si}))
                    np.savez_compressed(path, **arrays)
                    for ext in (".json", ".bin"):
                        os.remove(base + ext)
                    print(f"{q['id']} s{si} {name} {ms / 1e3:.1f} s", flush=True)


def rank_windows(args):
    """Each window's readings standardized within it, then merged."""
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    merged = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "*.npz"))):
        qid, sub, name = os.path.basename(path)[:-4].rsplit(".", 2)
        q = qs[qid]
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        f = {k: (z[f"{k}/q"].astype(float), z[f"{k}/na"].astype(float)) for k in ("sum", "last", "next1", "sep")}
        end = [f["last"][i] + f["sep"][i] + f["next1"][i] for i in (0, 1)]
        vals = {"sum": zsum(f["sum"][0] - f["sum"][1]).sum(0), "end": zsum(end[0] - end[1]).sum(0)}
        vals["both"] = vals["sum"] + vals["end"]
        for reading, v in vals.items():
            v = np.where(np.isfinite(v), v, -1e9)
            v = (v - v.mean()) / (v.std() + 1e-12)
            m = merged.setdefault((qid, f"{name}:{reading}"), np.full(len(q["state"]), -1e9))
            m[meta["first"]:meta["end"]] = v
    out, table = {}, {}
    for (qid, key), v in merged.items():
        q = qs[qid]
        order = [int(i) for i in np.argsort(-v, kind="stable")]
        out.setdefault(qid, {})[key] = order[:64]
        if q["targets"]:
            pos = order.index(q["targets"][0])
            for g in ("all", q["split"]):
                table.setdefault((key, g), []).append(pos)
    json.dump(out, open(args.out, "w"))
    for (key, g), pos in sorted(table.items(), key=lambda x: (x[0][1], x[0][0])):
        pos = np.array(pos)
        print(f"{key:18s} {g:10s} n {len(pos):3d}  @1 {100 * np.mean(pos < 1):5.1f}  @4 {100 * np.mean(pos < 4):5.1f}"
              f"  @16 {100 * np.mean(pos < 16):5.1f}  @64 {100 * np.mean(pos < 64):5.1f}")


def rank(args):
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    out, table = {}, {}
    for path in sorted(glob.glob(os.path.join(args.dir, "*.npz"))):
        qid, name = os.path.basename(path)[:-4].rsplit(".", 1)
        q = qs[qid]
        z = np.load(path, allow_pickle=True)
        f = {k: (z[f"{k}/q"].astype(float), z[f"{k}/na"].astype(float)) for k in ("sum", "last", "next1", "sep")}
        end = [f["last"][i] + f["sep"][i] + f["next1"][i] for i in (0, 1)]
        s = zsum(f["sum"][0] - f["sum"][1]).sum(0)
        e = zsum(end[0] - end[1]).sum(0)
        for reading, v in (("sum", s), ("end", e), ("both", s + e)):
            v = np.where(np.isfinite(v), v, -1e9)
            order = [int(i) for i in np.argsort(-v, kind="stable")]
            key = f"{name}:{reading}"
            out.setdefault(qid, {})[key] = order[:64]
            if q["targets"]:
                pos = order.index(q["targets"][0])
                for g in ("all", f"{q['segments']}", q["split"]):
                    row = table.setdefault((key, g), [])
                    row.append(pos)
    json.dump(out, open(args.out, "w"))
    for (key, g), pos in sorted(table.items(), key=lambda x: (x[0][1], x[0][0])):
        pos = np.array(pos)
        print(f"{key:18s} {g:10s} n {len(pos):3d}  @1 {100 * np.mean(pos < 1):5.1f}  @4 {100 * np.mean(pos < 4):5.1f}"
              f"  @16 {100 * np.mean(pos < 16):5.1f}  @64 {100 * np.mean(pos < 64):5.1f}")


def ask(args):
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    ranks = json.load(open(args.rank, encoding="utf-8"))
    out = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    for qid, r in sorted(ranks.items()):
        if qid in out:
            continue
        q = qs[qid]
        cand = sorted(r[args.reading][:args.k])
        rec = ask_lines(args.url, [line(q["state"][i]) for i in cand], q["instruction"])
        pick = cand[rec["plain"][0][0]]
        out[qid] = dict(rec, cand=cand, pick=pick, targets=q["targets"], segments=q["segments"], split=q["split"],
                        absent=q["absent"], hit=pick in q["targets"])
        json.dump(out, open(args.out, "w", encoding="utf-8"))
        print(qid, "absent" if q["absent"] else ("hit" if pick in q["targets"] else "miss"),
              f"none {rec['p_none']:.3f} found {rec['found']:.3f} {rec['ms']:.0f} ms", flush=True)


def logpipe(args):
    hs = json.load(open(args.heads, encoding="utf-8"))["heads"]
    qs = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    out = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    for q in qs:
        if q["id"] in out:
            continue
        lines = [line(x) for x in q["state"]]
        rec = log_one(args, hs, lines, q["instruction"], q["targets"][0] if q["targets"] else None)
        rec.update(segments=q["segments"], split=q["split"], absent=q["absent"], targets=q["targets"])
        out[q["id"]] = rec
        json.dump(out, open(args.out, "w", encoding="utf-8"))
        print(q["id"], "absent" if q["absent"] else ("hit" if rec["hit"] else "miss"), "clusters", rec["clusters"],
              "tc", rec["target_cluster"], "l1 pick", rec["l1_pick"],
              f"L2 none {rec['l2']['p_none']:.3f} found {rec['l2']['found']:.3f}", flush=True)


def verify(args):
    """The answer checked alone (as `zd_verify.py` checks a log line): the
    picked record, a `choice` between it (A) and "none", and a yes/no."""
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    data = json.load(open(args.answers, encoding="utf-8"))
    out = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    for qid, rec in data.items():
        if qid in out:
            continue
        q = qs[qid]
        pick = rec["pick"] if "pick" in rec else rec["answer_lines"][0]
        body = {"state": f"A: {line(q['state'][pick])}", "questions": {
            "none": {"type": "choice", "instructions": q["instruction"],
                     "criteria": {"A": None, "none": "No record of the evidence answers the criterion"}},
            "yes": {"type": "noul", "instructions": f"The record in the evidence answers this question: {q['instruction']}"}}}
        status, payload, ms = post(args.url, body)
        ans = payload.get("answers", {}) if status == 200 else {}
        if ans.get("none", {}).get("type") != "choice":
            print(qid, "error", str(payload)[:200])
            continue
        out[qid] = {"absent": q["absent"], "split": q["split"], "segments": q["segments"], "hit": rec["hit"],
                    "p_none": ans["none"]["probabilities"].get("none", 0.0), "yes": ans["yes"].get("noul"), "ms": ms}
        json.dump(out, open(args.out, "w", encoding="utf-8"))
    pres = [r for r in out.values() if not r["absent"]]
    absn = [r for r in out.values() if r["absent"]]
    f = lambda r: (r["p_none"] + 1 - r["yes"]) / 2
    print(f"{os.path.basename(args.answers)} verify: present {len(pres)} (hit {sum(r['hit'] for r in pres)}), absent {len(absn)}")
    for rn, g in (("p_none", lambda r: r["p_none"]), ("1-yes", lambda r: 1 - r["yes"]), ("mean", f)):
        print(f"  {rn:7s} AUC {auc([g(r) for r in absn], [g(r) for r in pres]):.3f}  > 0.5: present right "
              f"{sum(r['hit'] and g(r) <= 0.5 for r in pres)}/{len(pres)}, absent flagged {sum(g(r) > 0.5 for r in absn)}/{len(absn)}")


def report(args):
    for path in args.files:
        data = json.load(open(path, encoding="utf-8"))
        items = list(data.values())
        lvl = (lambda r: r["l2"]) if "l2" in items[0] else (lambda r: r)
        print(f"== {os.path.basename(path)}")
        groups = sorted({str(r["segments"]) for r in items}) + ["all"]
        for g in groups:
            sub = [r for r in items if g == "all" or str(r["segments"]) == g]
            pres = [r for r in sub if not r["absent"]]
            absn = [r for r in sub if r["absent"]]
            line_ = f"  {g:6s}"
            for split in ("lexical", "paraphrase"):
                p = [r for r in pres if r["split"] == split]
                line_ += f"  {split} {sum(r['hit'] for r in p)}/{len(p)}"
            says = lambda r: lvl(r)["none_best"] == "none"
            line_ += (f"  | none argmax: present right {sum(r['hit'] and not says(r) for r in pres)}/{len(pres)},"
                      f" absent flagged {sum(says(r) for r in absn)}/{len(absn)}")
            if absn and pres:
                line_ += (f" | AUC p_none {auc([lvl(r)['p_none'] for r in absn], [lvl(r)['p_none'] for r in pres]):.3f}"
                          f" 1-found {auc([1 - lvl(r)['found'] for r in absn], [1 - lvl(r)['found'] for r in pres]):.3f}")
            print(line_)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=20261130)
    b.add_argument("--lengths", type=int, nargs="+", default=[1000, 2500])
    b.add_argument("--targets", type=int, default=5)
    b.add_argument("--absent", type=int, default=1)
    b.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    h = sub.add_parser("heads")
    h.add_argument("--set", required=True)
    h.add_argument("--heads", nargs="+", required=True)
    h.add_argument("--out", required=True)
    for p in (h,):
        p.add_argument("--raw", default="F:/ai/opencode/inference/.scratch/locate/zd/raw")
        p.add_argument("--control", default="F:/ai/opencode/inference/.scratch/locate/zd/control.json")
        p.add_argument("--url", default="http://127.0.0.1:8000")
    k = sub.add_parser("rank")
    k.add_argument("--set", required=True)
    k.add_argument("--dir", required=True)
    k.add_argument("--out", required=True)
    a = sub.add_parser("ask")
    a.add_argument("--set", required=True)
    a.add_argument("--rank", required=True)
    a.add_argument("--reading", default="endheads:end")
    a.add_argument("--k", type=int, default=16)
    a.add_argument("--out", required=True)
    a.add_argument("--url", default="http://127.0.0.1:8000")
    l = sub.add_parser("logpipe")
    l.add_argument("--set", required=True)
    l.add_argument("--heads", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--raw", default="F:/ai/opencode/inference/.scratch/locate/zd/raw")
    l.add_argument("--control", default="F:/ai/opencode/inference/.scratch/locate/zd/control.json")
    l.add_argument("--url", default="http://127.0.0.1:8000")
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    hw = sub.add_parser("heads-windows")
    hw.add_argument("--set", required=True)
    hw.add_argument("--heads", nargs="+", required=True)
    hw.add_argument("--out", required=True)
    hw.add_argument("--window", type=int, default=200_000)
    hw.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    hw.add_argument("--raw", default="F:/ai/opencode/inference/.scratch/locate/zd/raw")
    hw.add_argument("--control", default="F:/ai/opencode/inference/.scratch/locate/zd/control.json")
    hw.add_argument("--url", default="http://127.0.0.1:8000")
    rw = sub.add_parser("rank-windows")
    rw.add_argument("--set", required=True)
    rw.add_argument("--dir", required=True)
    rw.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--set", required=True)
    v.add_argument("--answers", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    {"build": build, "heads": heads, "rank": rank, "ask": ask, "logpipe": logpipe, "report": report,
     "verify": verify, "heads-windows": heads_windows, "rank-windows": rank_windows}[args.cmd](args)


if __name__ == "__main__":
    main()
