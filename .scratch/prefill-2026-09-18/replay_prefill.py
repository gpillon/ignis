"""Replay a real captured agent trace, prefill only, in conversation order.

The synthetic arms in the finding used 81-token tails, where the reuse cuts
were most of the work. A real trace shares a median 78% of each prompt but
still leaves a ~5,200-token tail, so the cuts should be a much smaller
fraction. This sends the trace's own request bodies with `max_tokens` forced
to 1 -- what is being measured is the prefill, not the generation -- one at a
time, in the order they were recorded, so the reuse machinery sees the real
conversation.

  python replay_prefill.py <trace.jsonl> <out.json> [count]
"""
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

TRACE, OUT = sys.argv[1], sys.argv[2]
COUNT = int(sys.argv[3]) if len(sys.argv) > 3 else 60
URL = "http://127.0.0.1:8000/v1/chat/completions"

rows = []
with open(TRACE, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
rows = rows[:COUNT]

results = []
for n, row in enumerate(rows):
    body = json.loads(row["prompt"])
    body["max_tokens"] = 1
    body["stream"] = False
    payload = json.dumps(body).encode()
    request = urllib.request.Request(URL, data=payload,
                                     headers={"Content-Type": "application/json"})
    begin = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.loads(response.read())
        elapsed = (time.perf_counter() - begin) * 1000.0
        prompt_tokens = answer.get("usage", {}).get("prompt_tokens", 0)
        results.append({"i": n, "class": row.get("class"), "ms": elapsed,
                        "prompt_tokens": prompt_tokens})
        print("  %3d  %-5s %8.0f ms  %7d prompt tokens"
              % (n, row.get("class"), elapsed, prompt_tokens), flush=True)
    except urllib.error.HTTPError as error:
        detail = error.read()[:160].decode("utf-8", "replace")
        print("  %3d  HTTP %s %s" % (n, error.code, detail), flush=True)
        results.append({"i": n, "class": row.get("class"), "ms": None, "error": detail})
    except Exception as error:  # noqa: BLE001 - a replay should report, not abort
        print("  %3d  %s" % (n, error), flush=True)
        results.append({"i": n, "class": row.get("class"), "ms": None, "error": str(error)})

ok = [r["ms"] for r in results if r.get("ms")]
if ok:
    print("\n  %d of %d answered, median %.0f ms, total %.1f s"
          % (len(ok), len(results), statistics.median(ok), sum(ok) / 1000.0))
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(results, handle)
