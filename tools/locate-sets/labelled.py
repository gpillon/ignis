"""The labelled route: spec 18's baseline, measured and never shipped.

Jev's "line search": every segment of the state gets an answer label written
in front of it, and one `choice` over those labels asks which. This asks a
set's questions that way through a running server's `/v1/decide`, wherever a
question has at most 256 segments that own content (the endpoint's measured
option ceiling), and writes what the scorer compares the attention reading
with (`score.py check --labelled`).

- A string state's content lines become `<label>: <line>`; empty lines stay
  empty and unlabelled. An array becomes an object from label to element, in
  order.
- The labels are the endpoint's own answer alphabet in the order it assigns
  them (A-Z, a-z, 0-9, then the admitted bigrams), read from the harness
  dump's metadata, and the options are declared under those names -- so the
  letter the model is shown for an option is the label its segment carries.
- The instruction is the question's own, as the criterion.

Beside each choice it times a `noul` over the **unlabelled** state with the
same instruction: one prefill of the same state and one readout, the cost
class a `locate` is in. The server is used as it runs (`make start`), one
request at a time.

    python labelled.py --set .scratch/locate/C --alphabet <C-hq.json> \
        --url http://127.0.0.1:8000 --out C-labelled.json
"""

import argparse
import json
import os
import time
import urllib.request

MAX_OPTIONS = 256


def post(url, body, timeout):
    request = urllib.request.Request(url + "/v1/decide", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload, (time.perf_counter() - started) * 1e3


def labelled_state(state, alphabet):
    """The state with its content segments labelled, and the segment index
    each label stands for."""
    if isinstance(state, str):
        lines, mapping = [], {}
        for index, line in enumerate(state.split("\n")):
            if line.strip():
                label = alphabet[len(mapping)]
                mapping[label] = index
                lines.append(f"{label}: {line}")
            else:
                lines.append(line)
        return "\n".join(lines), mapping
    mapping = {alphabet[index]: index for index in range(len(state))}
    return {label: state[index] for label, index in mapping.items()}, mapping


def owning(state):
    if isinstance(state, str):
        return sum(1 for line in state.split("\n") if line.strip())
    return len(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="the set directory (manifest.json)")
    ap.add_argument("--alphabet", required=True, help="the harness dump's .json (its `alphabet`)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    with open(args.alphabet, encoding="utf-8") as f:
        alphabet = json.load(f)["alphabet"]
    assert len(alphabet) >= MAX_OPTIONS, f"the alphabet names {len(alphabet)} options"
    with open(os.path.join(args.set, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    results = []
    questions = [q for q in manifest["questions"] if owning(q["state"]) <= MAX_OPTIONS]
    for q in questions[:args.limit]:
        state, mapping = labelled_state(q["state"], alphabet)
        body = {"state": state, "questions": {"q": {
            "type": "choice", "instructions": q["instruction"],
            "criteria": {label: None for label in mapping}}}}
        answer, choice_ms = post(args.url, body, args.timeout)
        a = answer["answers"]["q"]
        assert a["type"] == "choice", f"{q['id']}: {a}"
        picked = mapping[a["choice"]]
        top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        plain = {"state": q["state"], "questions": {"q": {"type": "noul", "instructions": q["instruction"]}}}
        noul, noul_ms = post(args.url, plain, args.timeout)
        results.append({
            "id": q["id"], "family": q["family"], "split": q["split"], "absent": q["absent"],
            "owning": len(mapping), "winner": picked, "targets": q["targets"],
            "hit": (not q["absent"]) and picked in q["targets"],
            "confidence": a["confidence"], "top3": [[mapping[k], p] for k, p in top],
            "choice_ms": choice_ms, "choice_input_tokens": answer["usage"]["input_tokens"],
            "noul_ms": noul_ms, "noul_input_tokens": noul["usage"]["input_tokens"],
        })
        r = results[-1]
        print(f"  {q['id']} {q['family']:7} {q['split']:10} {'absent ' if q['absent'] else 'present'} "
              f"{len(mapping):4} labels -> {picked} {'hit' if r['hit'] else '   '} targets {q['targets']} "
              f"({choice_ms:.0f} ms, noul {noul_ms:.0f} ms)", flush=True)
    present = [r for r in results if not r["absent"]]
    out = {"set": manifest.get("seed"), "url": args.url, "asked": len(results),
           "hits": sum(r["hit"] for r in present), "present": len(present), "questions": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"labelled route: {out['hits']}/{out['present']} present questions right")


if __name__ == "__main__":
    main()
