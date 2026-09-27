"""The generation route: spec 19's comparator for spans (phase 1, GitHub #276).

What a caller does today to get a position in a text: ask the model for the
words that answer, then find them in the state. Every question of a span set
(`spans.py`) goes through a running server's `/v1/chat/completions` with the
render the attention harness measures -- the state as `{"evidence": …}` in
the system message, the span kind text and `{"instruction": …}` in the user
message -- greedily, thinking off, and the answer is searched for in the
state's segments (a line, or an element's compact JSON):

- **exact**: the quote occurs verbatim; every occurrence is kept, the first
  is the route's answer, and more than one is counted **ambiguous**;
- **lenient**: failing that, the same search case-folded with whitespace
  collapsed (reported apart, never mixed into exact);
- **none**: the model answered words that are not in the state, or no quote.

For the several-span families (`logmulti`, `hotfacts`) the kind text asks for
every part (`{"quotes": [...]}`), which is the comparator spec 19 names for
several spans. Timing and token counts are kept per question.

    python generation.py --set .scratch/locate/E1 --url http://127.0.0.1:8000 --out E1-generation.json
"""

import argparse
import json
import os
import re
import time
import urllib.request

SPAN_KIND = ("Find the exact answer to the instruction in the evidence. Answer with only a JSON "
             "object {\"quote\": \"<the answer, copied exactly from the evidence>\"}.")
MULTI_KIND = ("Find every part of the evidence that answers the instruction. Answer with only a JSON "
              "object {\"quotes\": [\"<each part, copied exactly>\"]}, an empty list if there is none.")
MULTI_FAMILIES = ("logmulti", "hotfacts")
MAX_TOKENS = 400


def segment_texts(state):
    if isinstance(state, str):
        return state.split("\n")
    return [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in state]


def user_text(kind_text, instruction):
    """The kind text, a blank line, then the instruction as `/v1/decide`
    writes one (`support/locate.rs::user_text`)."""
    return f"{kind_text}\n\n{{\"instruction\":{json.dumps(instruction, ensure_ascii=False, separators=(',', ':'))}}}"


def evidence_text(state):
    return "{\"evidence\":" + json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "}"


def parse(content, multi):
    """The quote(s) a reply carries, or None when it carries none."""
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text)
    except ValueError:
        found = re.search(r"\{.*\}", text, re.S)
        if not found:
            return None
        try:
            value = json.loads(found.group(0))
        except ValueError:
            return None
    if not isinstance(value, dict):
        return None
    if multi:
        quotes = value.get("quotes")
        return [q for q in quotes if isinstance(q, str) and q] if isinstance(quotes, list) else None
    quote = value.get("quote")
    return [quote] if isinstance(quote, str) and quote else None


def fold(text):
    return " ".join(text.split()).casefold()


def occurrences(texts, quote):
    """Every (segment, start, end) where `quote` occurs verbatim, then -- only
    if there is none -- the lenient ones, in folded coordinates mapped back
    to the segment's own characters."""
    exact = []
    for j, text in enumerate(texts):
        at = text.find(quote)
        while at >= 0:
            exact.append({"segment": j, "start": at, "end": at + len(quote)})
            at = text.find(quote, at + 1)
    if exact:
        return "exact", exact
    needle = fold(quote)
    lenient = []
    if needle:
        for j, text in enumerate(texts):
            # map folded offsets back through a per-character index
            chars, index = [], []
            for i, c in enumerate(text):
                if c.isspace():
                    if chars and chars[-1] != " ":
                        chars.append(" ")
                        index.append(i)
                    continue
                for f in c.casefold():
                    chars.append(f)
                    index.append(i)
            folded = "".join(chars)
            at = folded.find(needle)
            while at >= 0:
                end = index[at + len(needle) - 1] + 1
                lenient.append({"segment": j, "start": index[at], "end": end})
                at = folded.find(needle, at + 1)
    return ("lenient", lenient) if lenient else ("none", [])


def ask(url, question, timeout):
    multi = question["family"] in MULTI_FAMILIES
    body = {
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "system", "content": evidence_text(question["state"])},
            {"role": "user", "content": user_text(MULTI_KIND if multi else SPAN_KIND, question["instruction"])},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "enable_thinking": False,
    }
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    ms = (time.perf_counter() - started) * 1e3
    content = reply["choices"][0]["message"].get("content") or ""
    usage = reply.get("usage", {})
    quotes = parse(content, multi)
    texts = segment_texts(question["state"])
    found = [dict(zip(("match", "spans"), occurrences(texts, q)), quote=q) for q in quotes or []]
    return {"id": question["id"], "family": question["family"], "split": question["split"],
            "absent": question["absent"], "content": content, "quotes": quotes, "found": found,
            "ms": ms, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="a span set directory (spans.py)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    with open(os.path.join(args.set, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    questions = manifest["questions"][:args.limit] if args.limit else manifest["questions"]
    rows = []
    for q in questions:
        row = ask(args.url, q, args.timeout)
        rows.append(row)
        match = row["found"][0]["match"] if row["found"] else "no quote"
        print(f"  {q['id']:14} {q['split']:10} {'absent' if q['absent'] else 'present':7} {match:8} "
              f"{row['ms']:6.0f} ms {row['completion_tokens']} tok  {(row['quotes'] or [''])[0][:60]!r}", flush=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        json.dump({"set": manifest.get("seed"), "span_kind": SPAN_KIND, "multi_kind": MULTI_KIND,
                   "questions": rows}, f, ensure_ascii=False, indent=1)
    print(f"generation route: {len(rows)} questions -> {args.out}")


if __name__ == "__main__":
    main()
