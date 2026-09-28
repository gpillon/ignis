"""EXPERIMENT (branch locate-long-context): `locate` over a folded log
(`compress.py`) — level 1 picks the template, level 2 the instance by its
values, the map gives the original line.

For each question of a `long.py`-shaped set: fold its state (cached per
state), ask a `locate` over the level-1 text; if the chosen cluster holds
more than one level-2 row, ask a second `locate` over its values; unfold.
Recorded beside it: whether level 1 chose the target's cluster, and the
top-3 clusters. The server is this branch's (its ceiling lifted for
level-1 texts past 4,554 keys).

    python folded_locate.py --set <R> --out R-folded.json [--render values|lines]
"""

import argparse
import re
import json
import os
import time
import urllib.error
import urllib.request

import compress as C


def post(url, body, timeout=1800):
    """One `/v1/decide` request; an answer turned away with `engine_full`
    (another client holds every lane) is asked again, up to six times."""
    for attempt in range(6):
        request = urllib.request.Request(url + "/v1/decide", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, payload = response.status, json.load(response)
        except urllib.error.HTTPError as error:
            status, payload = error.code, json.load(error)
        full = any(isinstance(a, dict) and a.get("code") == "engine_full"
                   for a in (payload.get("answers") or {}).values())
        if not full:
            break
        time.sleep(5 * (attempt + 1))
    return status, payload, (time.perf_counter() - started) * 1e3


ROUTE = "vote"
KIND_LINE = ("Find the one line of the evidence that the instruction asks for. Answer with only "
             "a JSON object {\"quote\": \"<that line, copied exactly>\"}.")


def generate(url, lines, instruction):
    """The generation route over `lines`: the model writes the line out,
    greedy; the answer is the line it wrote (exact, else case- and
    space-folded, else the line sharing the longest prefix)."""
    from generation import parse
    body = {"model": "qwen3.8-27b", "temperature": 0, "max_tokens": 300, "enable_thinking": False,
            "messages": [{"role": "system", "content": "{\"evidence\":" + json.dumps("\n".join(lines), ensure_ascii=False, separators=(",", ":")) + "}"},
                         {"role": "user", "content": KIND_LINE + "\n\n{\"instruction\":" + json.dumps(instruction, ensure_ascii=False, separators=(",", ":")) + "}"}]}
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        reply = json.load(response)
    ms = (time.perf_counter() - started) * 1e3
    content = reply["choices"][0]["message"].get("content") or ""
    quotes = parse(content, False)
    quote = quotes[0] if quotes else re.sub(r'^\s*\{\s*"quote"\s*:\s*"?|"?\s*\}\s*$', "", content)
    exact = [i for i, line in enumerate(lines) if quote == line]
    if not exact:
        exact = [i for i, line in enumerate(lines) if quote and quote in line]
    if exact:
        seg, how = exact[0], "exact"
    else:
        # a folded line shows `{a|b|c}` where the model writes one value:
        # the line whose words the quote covers best
        words = set(re.findall(r"[A-Za-z0-9_.:/-]+", quote))

        def overlap(line):
            mine = set(re.findall(r"[A-Za-z0-9_.:/-]+", line))
            return len(words & mine) / max(1, len(words))
        seg, how = max(range(len(lines)), key=lambda i: overlap(lines[i])), "words"
    return {"status": 200, "segment": seg, "ranking": [{"segment": seg, "share": 1.0}], "ms": ms, "match": how,
            "content": content[:400]}


ALPHABET = json.load(open("F:/ai/opencode/inference/.scratch/locate/dumps/C-hq.json", encoding="utf-8"))["alphabet"]
MAX_OPTIONS = 256


def choice_once(url, lines, instruction):
    """One labelled `choice` over at most 256 lines (spec 18's route)."""
    labels = ALPHABET[:len(lines)]
    state = "\n".join(f"{label}: {line}" for label, line in zip(labels, lines))
    body = {"state": state, "questions": {"q": {"type": "choice", "instructions": instruction,
                                                "criteria": {label: None for label in labels}}}}
    status, payload, ms = post(url, body)
    a = payload.get("answers", {}).get("q", {}) if status == 200 else {}
    if a.get("type") != "choice":
        return {"status": status, "error": payload.get("error") or a, "ms": ms}
    ranking = sorted(((labels.index(k), p) for k, p in a["probabilities"].items()), key=lambda x: -x[1])
    return {"status": status, "segment": labels.index(a["choice"]), "confidence": a["confidence"],
            "ranking": [{"segment": s, "share": p} for s, p in ranking], "ms": ms}


def choice(url, lines, instruction):
    """A labelled `choice`; past 256 lines, one per chunk of 256 and a final
    one over the chunks' winners (still no token generated)."""
    if len(lines) <= MAX_OPTIONS:
        return choice_once(url, lines, instruction)
    winners, ms = [], 0.0
    for start in range(0, len(lines), MAX_OPTIONS):
        one = choice_once(url, lines[start:start + MAX_OPTIONS], instruction)
        ms += one["ms"]
        if "segment" in one:
            winners.append(start + one["segment"])
    final = choice_once(url, [lines[w] for w in winners], instruction)
    if "segment" not in final:
        return dict(final, ms=ms + final["ms"])
    return {"status": 200, "segment": winners[final["segment"]], "confidence": final["confidence"],
            "ranking": [{"segment": winners[e["segment"]], "share": e["share"]} for e in final["ranking"]],
            "ms": ms + final["ms"], "chunks": len(winners), "winners": winners}


def locate(url, lines, instruction):
    if len(lines) < 2:
        return {"segment": 0, "ranking": [{"segment": 0, "share": 1.0}], "ms": 0.0, "status": 200}
    if ROUTE == "generate":
        return generate(url, lines, instruction)
    if ROUTE == "choice":
        return choice(url, lines, instruction)
    body ={"state": "\n".join(lines), "questions": {"q": {"type": "locate", "instructions": instruction}}}
    status, payload, ms = post(url, body)
    if status != 200 or payload["answers"]["q"].get("type") != "locate":
        return {"status": status, "error": payload.get("error") or payload["answers"]["q"], "ms": ms}
    a = payload["answers"]["q"]
    return {"status": status, "segment": a["segment"], "ranking": a["ranking"], "confidence": a["confidence"], "ms": ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--render", choices=("values", "lines"), default="values")
    ap.add_argument("--sim", type=float, default=C.SIM)
    ap.add_argument("--values", action="store_true", help="level 1 shows each slot's distinct values")
    ap.add_argument("--top", type=int, default=1, help="level 2 over the union of the first `top` level-1 clusters")
    ap.add_argument("--route", choices=("vote", "generate", "choice"), default="vote")
    ap.add_argument("--tournament", action="store_true",
                    help="with --top: level 2 inside each top cluster, then one pass over the winners")
    args = ap.parse_args()
    global ROUTE
    ROUTE = args.route
    with open(os.path.join(args.set, "manifest.json"), encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    folds, rows = {}, []
    for q in questions:
        lines = q["state"].split("\n")
        key = hash(q["state"])
        if key not in folds:
            folds[key] = C.fold(lines, args.sim, values=args.values)
        f = folds[key]
        one = locate(args.url, f.level1, q["instruction"])
        if args.tournament and "segment" in one:
            # each of the level-1 ranking's first `top` clusters picks its own
            # row at level 2; a last pass chooses among those rows' original
            # lines
            chosen = [e["segment"] for e in one["ranking"][:args.top]]
            finalists, ms = [], one.get("ms", 0)
            for ci in chosen:
                t2, m2 = C.level2(f, ci)
                two = locate(args.url, t2, q["instruction"])
                ms += two.get("ms", 0)
                if "segment" in two:
                    finalists.append(m2[two["segment"]])
            last = locate(args.url, [lines[m[0]] for m in finalists], q["instruction"])
            ms += last.get("ms", 0)
            answer = finalists[last["segment"]] if "segment" in last else None
            tc = next((i for i, c in enumerate(f.clusters) if q["targets"] and q["targets"][0] in c.members), None)
            row = {"id": q["id"], "absent": q["absent"], "segments": q["segments"], "targets": q["targets"],
                   "split": q.get("split"), "clusters": len(f.clusters), "level1": one, "target_cluster": tc,
                   "level1_hit": tc in chosen, "finalists": finalists, "answer_lines": answer, "ms": ms,
                   "finalist_hit": any(q["targets"] and q["targets"][0] in m for m in finalists),
                   "hit": bool(answer and q["targets"] and q["targets"][0] in answer)}
            rows.append(row)
            print(f"  {q['id']} {'absent ' if q['absent'] else 'present'} L1 top{args.top} {row['level1_hit']} "
                  f"finalists {row['finalist_hit']} -> {'HIT' if row['hit'] else 'miss'} | {ms:.0f} ms", flush=True)
            continue
        if args.top > 1 and "segment" in one:
            # level 2 over the union of the level-1 ranking's first `top`
            # clusters: each row tagged with its cluster's template head
            chosen = [e["segment"] for e in one["ranking"][:args.top]]
            texts, members = [], []
            for ci in chosen:
                t2, m2 = C.level2(f, ci)
                head = " ".join(t for t in f.clusters[ci].template[:6] if t not in ("<*>", "<v>"))
                texts += [f"{f.clusters[ci].label} {head} :: {t}" for t in t2]
                members += m2
            two = locate(args.url, texts, q["instruction"])
            row = {"id": q["id"], "absent": q["absent"], "segments": q["segments"], "targets": q["targets"],
                   "split": q.get("split"), "clusters": len(f.clusters), "level1": one, "level2": dict(two, rows=len(texts))}
            tc = next((i for i, c in enumerate(f.clusters) if q["targets"] and q["targets"][0] in c.members), None)
            row["target_cluster"] = tc
            row["level1_hit"] = tc in chosen
            answer = members[two["segment"]] if "segment" in two else None
            row["answer_lines"] = answer
            row["hit"] = bool(answer and q["targets"] and q["targets"][0] in answer)
            rows.append(row)
            print(f"  {q['id']} {'absent ' if q['absent'] else 'present'} L1 top{args.top} has target {row['level1_hit']} "
                  f"| L2 rows {len(texts)} -> {'HIT' if row['hit'] else 'miss'}", flush=True)
            continue
        row = {"id": q["id"], "absent": q["absent"], "segments": q["segments"], "targets": q["targets"],
               "split": q.get("split"), "clusters": len(f.clusters), "level1": one}
        target_cluster = next((i for i, c in enumerate(f.clusters) if q["targets"] and q["targets"][0] in c.members), None)
        row["target_cluster"] = target_cluster
        row["target_cluster_size"] = len(f.clusters[target_cluster].members) if target_cluster is not None else None
        answer = None
        if "segment" in one:
            ci = one["segment"]
            if args.render == "values":
                texts, members = C.level2(f, ci)
            else:
                members = [[m] for m in f.clusters[ci].members]
                texts = [lines[m[0]] for m in members]
            two = locate(args.url, texts, q["instruction"])
            row["level2"] = dict(two, rows=len(texts))
            if "segment" in two:
                answer = members[two["segment"]]
        row["answer_lines"] = answer
        row["hit"] = bool(answer and q["targets"] and q["targets"][0] in answer)
        row["level1_hit"] = target_cluster is not None and one.get("segment") == target_cluster
        rows.append(row)
        print(f"  {q['id']} {'absent ' if q['absent'] else 'present'} clusters {len(f.clusters):4d} "
              f"target cluster {target_cluster} (size {row['target_cluster_size']}) | L1 -> {one.get('segment')} "
              f"{'ok' if row['level1_hit'] else '--'} | L2 rows {row.get('level2', {}).get('rows')} -> "
              f"{'HIT' if row['hit'] else 'miss'} | {one.get('ms', 0) + row.get('level2', {}).get('ms', 0):.0f} ms", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"render": args.render, "sim": args.sim, "questions": rows}, f, indent=1)
    present = [r for r in rows if not r["absent"]]
    by = {}
    for r in present:
        b = by.setdefault(r["segments"], [0, 0, 0])
        b[0] += 1
        b[1] += r["level1_hit"]
        b[2] += r["hit"]
    for tier, (n, l1, hit) in sorted(by.items()):
        print(f"{tier:>7}: n {n}  level-1 cluster {l1}  final {hit}")
    print(f"all: {sum(r['hit'] for r in present)}/{len(present)} (level 1 {sum(r['level1_hit'] for r in present)})")


if __name__ == "__main__":
    main()
