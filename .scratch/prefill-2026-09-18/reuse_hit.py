"""The other side of the reuse bet: what a claimant saves when it hits.

Three request shapes against one running server, timed client-side:

  cold      a fresh 1,024-token prompt every time, nothing to claim
  repeat    the same 1,024-token prompt over and over -- a full-prompt match
  agentic   a shared ~960-token prefix with a different tail each time, which
            is the shape the north star actually runs

  python reuse_hit.py <scratch dir> <label>
"""
import json
import os
import statistics
import sys
import time
import urllib.request

SP = sys.argv[1]
LABEL = sys.argv[2]
URL = "http://127.0.0.1:8000/v1/chat/completions"

corpus = []
for directory, _, files in os.walk("crates"):
    for name in sorted(files):
        if name.endswith(".rs"):
            with open(os.path.join(directory, name), encoding="utf-8", errors="ignore") as handle:
                corpus.append(handle.read())
text = "".join(corpus)
BODY = 4000   # ~960 tokens of code
TAIL = 260    # ~64 tokens


def ask(content):
    payload = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    request = urllib.request.Request(URL, data=payload,
                                     headers={"Content-Type": "application/json"})
    begin = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()
    return (time.perf_counter() - begin) * 1000.0


def run(name, bodies):
    times = [ask(b) for b in bodies]
    # The first of a series pays for the series; report it apart.
    print("  %-9s first %7.1f ms   rest median %7.1f ms   (n=%d)  %s"
          % (name, times[0], statistics.median(times[1:]) if len(times) > 1 else float("nan"),
             len(times), " ".join("%.0f" % t for t in times)))
    return times


print("warming the server")
ask("hello")

SHARED = text[:BODY]
cold = run("cold", [text[i * BODY * 3:i * BODY * 3 + BODY + TAIL] for i in range(1, 7)])
repeat = run("repeat", [SHARED + text[BODY:BODY + TAIL]] * 6)
agentic = run("agentic", [SHARED + text[BODY * 7 + i * TAIL:BODY * 7 + (i + 1) * TAIL]
                          for i in range(6)])

with open(os.path.join(SP, "reuse-hit-%s.json" % LABEL), "w", encoding="utf-8") as handle:
    json.dump({"cold": cold, "repeat": repeat, "agentic": agentic}, handle)
print("wrote reuse-hit-%s.json" % LABEL)
