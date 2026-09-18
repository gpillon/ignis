"""The agentic shape, modelled correctly this time.

`Request::publish_point` anchors the retained prefix at the end of the SYSTEM
block (crates/core/src/request.rs:294-305), so reuse is keyed on a shared
system message, not on any common prefix. The first attempt put the shared
text inside one user message and collected nothing -- that was the test being
wrong, not the engine.

  python reuse_system.py <scratch dir>
"""
import json
import os
import statistics
import sys
import time
import urllib.request

SP = sys.argv[1]
URL = "http://127.0.0.1:8000/v1/chat/completions"

corpus = []
for directory, _, files in os.walk("crates"):
    for name in sorted(files):
        if name.endswith(".rs"):
            with open(os.path.join(directory, name), encoding="utf-8", errors="ignore") as handle:
                corpus.append(handle.read())
text = "".join(corpus)
SYSTEM = text[:4000]      # ~960 tokens of shared instructions
TAIL = 260                # ~64 tokens of per-turn user text


def ask(messages):
    payload = json.dumps({
        "model": "qwen3.8-27b", "messages": messages,
        "max_tokens": 1, "temperature": 0.0, "stream": False,
    }).encode()
    request = urllib.request.Request(URL, data=payload,
                                     headers={"Content-Type": "application/json"})
    begin = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()
    return (time.perf_counter() - begin) * 1000.0


def turn(i):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": text[4000 * 13 + i * TAIL:4000 * 13 + (i + 1) * TAIL]}]


def cold(i):
    return [{"role": "system", "content": text[4000 * (20 + i):4000 * (21 + i)]},
            {"role": "user", "content": text[4000 * 13 + i * TAIL:4000 * 13 + (i + 1) * TAIL]}]


ask([{"role": "user", "content": "hello"}])

c = [ask(cold(i)) for i in range(6)]
print("  cold system    %s ms   median %.1f" % (" ".join("%.0f" % t for t in c), statistics.median(c)))
s = [ask(turn(i)) for i in range(8)]
print("  shared system  %s ms" % " ".join("%.0f" % t for t in s))
print("  shared system: first %.1f ms, rest median %.1f ms" % (s[0], statistics.median(s[1:])))

with open(os.path.join(SP, "reuse-system.json"), "w", encoding="utf-8") as handle:
    json.dump({"cold": c, "shared_system": s}, handle)
