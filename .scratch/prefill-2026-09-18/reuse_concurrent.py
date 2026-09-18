"""Does the shared-prefix path need live siblings to publish?

The sequential agentic arm collected nothing. The retained-prefix mechanism
(#126) publishes from a request so a sibling can claim the head; if that only
happens when the sibling is in flight, then firing the shared-prefix requests
concurrently should hit where firing them one after another did not.

  python reuse_concurrent.py <scratch dir>
"""
import json
import os
import statistics
import sys
import threading
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
BODY, TAIL = 4000, 260
SHARED = text[:BODY]


def ask(content):
    payload = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1, "temperature": 0.0, "stream": False,
    }).encode()
    request = urllib.request.Request(URL, data=payload,
                                     headers={"Content-Type": "application/json"})
    begin = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()
    return (time.perf_counter() - begin) * 1000.0


ask("hello")

# Four siblings at once: same ~960-token head, four different tails.
results = [0.0] * 4
def worker(i):
    results[i] = ask(SHARED + text[BODY * 9 + i * TAIL:BODY * 9 + (i + 1) * TAIL])

threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
begin = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join()
wall = (time.perf_counter() - begin) * 1000.0
print("  concurrent x4  wall %7.1f ms   per request %s ms"
      % (wall, " ".join("%.0f" % r for r in results)))

# Then a fifth, alone, after the four have published whatever they publish.
later = [ask(SHARED + text[BODY * 11 + i * TAIL:BODY * 11 + (i + 1) * TAIL]) for i in range(4)]
print("  sequential after them   %s ms   median %.1f"
      % (" ".join("%.0f" % t for t in later), statistics.median(later)))

with open(os.path.join(SP, "reuse-concurrent.json"), "w", encoding="utf-8") as handle:
    json.dump({"concurrent": results, "wall": wall, "after": later}, handle)
