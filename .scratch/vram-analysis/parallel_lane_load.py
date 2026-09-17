"""Parallel multi-turn load on one lane, beside the trace replay (VRAM analysis).

Each worker takes a distinct real @agent body from the trace (system + tools),
then runs a conversation whose history grows every turn.
"""
import json, sys, threading, time, urllib.request

TRACE, ENDPOINT, LANE, WORKERS, TURNS, OUT = sys.argv[1:7]
WORKERS, TURNS = int(WORKERS), int(TURNS)

bodies = []
for line in open(TRACE, encoding="utf-8"):
    b = json.loads(json.loads(line)["prompt"])
    if b.get("model", "").endswith("@agent"):
        bodies.append(b)
bodies.sort(key=lambda b: len(json.dumps(b)), reverse=True)
seeds, seen = [], set()
for b in bodies:
    sig = json.dumps(b["messages"][:2])[:4000]
    if sig not in seen:
        seen.add(sig); seeds.append(b)
lock = threading.Lock()
log = open(OUT, "a", encoding="utf-8")

def post(body):
    req = urllib.request.Request(ENDPOINT + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())

def worker(i):
    base = seeds[i % len(seeds)]
    msgs = list(base["messages"])
    for turn in range(TURNS):
        msgs.append({"role": "user", "content": f"[worker {i} turn {turn}] Continua l'analisi: riassumi in 10 punti lo stato e proponi il prossimo passo concreto."})
        body = {k: v for k, v in base.items() if k not in ("messages", "stream", "stream_options")}
        body.update(model="qwen3.8-27b@" + LANE, messages=msgs, max_tokens=400, stream=False)
        t0 = time.time()
        try:
            resp = post(body)
            content = resp["choices"][0]["message"].get("content") or ""
            usage = resp.get("usage", {})
            msgs.append({"role": "assistant", "content": content})
            status = "ok"
        except Exception as e:  # keep going: we measure memory, not correctness
            usage, status = {}, f"err {e}"
            msgs.pop()
        with lock:
            log.write(json.dumps({"t": time.strftime("%H:%M:%S"), "worker": i, "turn": turn, "status": status,
                                  "secs": round(time.time() - t0, 2), "usage": usage}) + "\n"); log.flush()

threads = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
for t in threads: t.start()
for t in threads: t.join()
print("done")
