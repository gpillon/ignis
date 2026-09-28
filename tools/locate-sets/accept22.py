"""Spec 22's acceptance runs (GitHub #278): every question of a registered
set put to a running server (`make start`), **one `locate` per request**
through `/v1/decide`, and what came back recorded for `r3_judge.py`.

A set is one of: R3 (`r3set.py build`: windows held once, each question's
state its window, less its target line for a `removed` one), P3
(`prosehay.py`), P3abs (`zd_prose_nf.py build`), J3 (`zd_records.py
build`), F (spec 18's set F). Questions over one state are asked one after
another, so a state's prefix is kept while its questions run; each row says
whether it was its state's first question (rule 3 times those).

`--fields` adds fields to every question (`'{"method":"vote","compression":"none"}'`
for run 5); none are added otherwise — the runs of the defaults send none.

    python accept22.py run --set R3 --manifest <R3>/manifest.json --out R3-defaults.json
    python accept22.py run --set F --manifest <F>/manifest.json --fields '{"method":"vote","compression":"none"}' --out F-vote.json

The output is rewritten after every question; a rerun skips the questions
already recorded (after a crash, not to redo one).

`inline` writes R3's present questions with their states inline, the shape
`folded_locate.py` reads, for run 2 (the labelled `choice` alone over the
fold):

    python accept22.py inline --manifest <R3>/manifest.json --out <R3present>
    python folded_locate.py --set <R3present> --route choice --values --out R3-choice.json
"""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request


def post(url, body, timeout):
    """(status, payload, milliseconds) of one request; a 422 is an answer
    here, and an `engine_full` answer is asked again, up to six times."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for attempt in range(6):
        request = urllib.request.Request(url + "/v1/decide", data=data, headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, payload = response.status, json.load(response)
        except urllib.error.HTTPError as error:
            status, payload = error.code, json.load(error)
        ms = (time.perf_counter() - started) * 1e3
        full = any(isinstance(a, dict) and a.get("code") == "engine_full" for a in (payload.get("answers") or {}).values())
        if not full:
            return status, payload, ms
        time.sleep(5 * (attempt + 1))
    return status, payload, ms


def questions_of(set_name, manifest):
    """(question, state) in asking order: every state's questions together,
    in the manifest's order of first appearance."""
    rows = manifest["questions"]
    if set_name == "R3":
        windows = manifest["windows"]
        out = []
        for q in rows:
            lines = windows[q["window"]]
            if q.get("remove") is not None:
                lines = lines[:q["remove"]] + lines[q["remove"] + 1:]
            out.append((q, "\n".join(lines)))
    else:
        out = [(q, q["state"]) for q in rows]
    keyed = {}
    order = []
    for q, state in out:
        key = hashlib.sha256(json.dumps(state, ensure_ascii=False).encode("utf-8")).hexdigest()
        if key not in keyed:
            keyed[key] = []
            order.append(key)
        keyed[key].append((q, state))
    return [(q, state, n == 0, key) for key in order for n, (q, state) in enumerate(keyed[key])]


def run(args):
    with open(args.manifest, encoding="utf-8") as f:
        text = f.read()
    manifest = json.loads(text)
    fields = json.loads(args.fields) if args.fields else {}
    done = {}
    if os.path.exists(args.out):
        done = {r["id"]: r for r in json.load(open(args.out, encoding="utf-8"))["questions"]}
    out = {"set": args.set, "manifest_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "fields": fields,
           "url": args.url, "questions": list(done.values())}
    for q, state, first, key in questions_of(args.set, manifest):
        if q["id"] in done:
            continue
        instruction = q["instruction"]
        body = {"state": state, "questions": {"q": dict({"type": "locate", "instructions": instruction}, **fields)}}
        status, payload, ms = post(args.url, body, args.timeout)
        row = {"id": q["id"], "status": status, "ms": ms, "first_of_state": first, "state": key[:16],
               "targets": q.get("targets", []), "absent": q.get("absent", False)}
        for k in ("family", "split", "segments", "tier", "variant", "window", "source", "siblings", "bin", "kind"):
            if k in q:
                row[k] = q[k]
        if status == 200:
            row["answer"] = payload["answers"]["q"]
            row["input_tokens"] = payload["usage"]["input_tokens"]
        else:
            row["code"] = payload.get("error", {}).get("code")
            row["message"] = payload.get("error", {}).get("message")
        out["questions"].append(row)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f)
        a = row.get("answer", {})
        print(f"{q['id']:34s} {status} {a.get('kind', row.get('code'))} -> {a.get('segment')} targets {row['targets']} "
              f"found {a.get('found')} ({ms / 1e3:.1f} s{' first' if first else ''})", flush=True)


def inline(args):
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    rows = []
    for q, state, _, _ in questions_of("R3", manifest):
        if q["variant"] != "present":
            continue
        rows.append(dict({k: v for k, v in q.items() if k != "remove"}, state=state))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": manifest["seed"], "questions": rows}, f, ensure_ascii=False)
    print(len(rows), "present questions ->", args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inline")
    i.add_argument("--manifest", required=True)
    i.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True, choices=["R3", "P3", "P3abs", "J3", "F"])
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--fields")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--timeout", type=float, default=3600)
    args = ap.parse_args()
    {"run": run, "inline": inline}[args.cmd](args)


if __name__ == "__main__":
    main()
