"""A synthetic multi-agent load: N coding agents, each a multi-turn conversation.

    python scripts/agent-swarm.py run --out <dir> [--agents 8] [--turns 8] ...
    python scripts/agent-swarm.py report <dir> [<dir> ...]
    python scripts/agent-swarm.py selftest

Why it exists: a recorded client trace replays a human's pauses and one
session's concurrency. This load is generated, deterministic for a seed, and
shaped like an agent swarm instead:

- every agent shares one system block (the head a burst of subagents shares),
  then its own task line, so the head is a retained prefix and each
  conversation leaves prompt checkpoints behind (ADR 0029);
- each turn appends a tool result cut from the repository's own files and
  asks for a short answer, and the reply goes back into the history, so the
  next turn claims the checkpoint the previous one left;
- between turns an agent waits a simulated tool time drawn from the seed --
  never a human pause -- so the lanes stay busy and the run is short.

With greedy decoding (the default) two runs of the same seed send the same
conversations, so an A/B of a server knob compares like with like.

`run` writes `run.json` (the configuration and the plan) and
`requests.jsonl` (one line per request: timings, token arrival times,
usage). `report` summarizes one or more run directories side by side, and
joins the server's own log and a before/after `/metrics` scrape when the
directory holds them (`serve/ignis-server.log`, `metrics-before.prom`,
`metrics-after.prom`, as `scripts/swarm-ab.sh` leaves them).

Standard library only, so it runs wherever Python does.
"""

import argparse
import hashlib
import http.client
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.parse

CHARS_PER_TOKEN = 4

# ── the plan ────────────────────────────────────────────────────────────────


def load_corpus(root, globs=(".md", ".rs", ".cu", ".h")):
    """Every text file under `root` with one of the suffixes, sorted, so a
    seed picks the same slices on every machine that has the same tree."""
    texts = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("target", "node_modules", "vendor"))
        for name in sorted(files):
            if name.endswith(tuple(globs)):
                path = os.path.join(base, name)
                try:
                    with open(path, encoding="utf-8") as f:
                        text = f.read()
                except (UnicodeDecodeError, OSError):
                    continue
                if len(text) > 2000:
                    texts.append((os.path.relpath(path, root).replace("\\", "/"), text))
    if not texts:
        raise SystemExit(f"no corpus files under {root}")
    return texts


def slice_of(rng, corpus, chars):
    """A `chars`-long run of one corpus file, starting at a line."""
    path, text = corpus[rng.randrange(len(corpus))]
    if len(text) <= chars:
        return path, text
    start = rng.randrange(len(text) - chars)
    start = text.rfind("\n", 0, start) + 1
    return path, text[start : start + chars]


def system_block(corpus, tokens):
    """The one head every agent shares: an agent preamble, then reference
    text up to about `tokens` tokens. Seeded apart from the turns, so the
    head is the same whatever --agents and --turns are."""
    rng = random.Random("system")
    parts = [
        "You are a coding agent working in a Rust + CUDA inference engine repository. "
        "You read files with tools and answer briefly and precisely. "
        "Below is reference material from the repository.\n"
    ]
    size = len(parts[0])
    while size < tokens * CHARS_PER_TOKEN:
        path, text = slice_of(rng, corpus, 3000)
        part = f"\n--- {path} ---\n{text}\n"
        parts.append(part)
        size += len(part)
    return "".join(parts)[: tokens * CHARS_PER_TOKEN]


def plan(corpus, agents, turns, tool_tokens, think_ms, seed):
    """Every agent's task line, and per turn its tool result and the
    simulated tool time before it is sent."""
    out = []
    for agent in range(agents):
        rng = random.Random(f"{seed}/{agent}")
        path, _ = corpus[rng.randrange(len(corpus))]
        task = (
            f"You are agent {agent}. Your task: review `{path}` and the code around it "
            f"for correctness problems. Each turn you get one tool result."
        )
        steps = []
        for turn in range(turns):
            path, text = slice_of(rng, corpus, tool_tokens * CHARS_PER_TOKEN)
            wait = rng.uniform(*think_ms) if turn else 0.0
            steps.append({"path": path, "text": text, "think_ms": round(wait, 1)})
        out.append({"agent": agent, "task": task, "turns": steps})
    return out


def plan_digest(system, agents):
    """One hash over every byte the plan sends before any reply: two runs
    with the same digest put the same conversations to the server."""
    h = hashlib.sha256(system.encode("utf-8"))
    for spec in agents:
        h.update(spec["task"].encode("utf-8"))
        for step in spec["turns"]:
            h.update(step["text"].encode("utf-8"))
            h.update(str(step["think_ms"]).encode("utf-8"))
    return h.hexdigest()


def user_message(task, step, first):
    head = f"{task}\n\n" if first else ""
    return (
        f"{head}Tool `read_file` returned `{step['path']}`:\n```\n{step['text']}\n```\n"
        "In at most three sentences: what does this do, and what is one thing worth checking next?"
    )


# ── the client ──────────────────────────────────────────────────────────────


def parse_sse_line(line):
    """The JSON payload of one SSE `data:` line, `"[DONE]"`, or None."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return data
    return json.loads(data)


def stream_chat(endpoint, api_key, body, clock, timeout):
    """POST one streaming chat completion. Returns the record's timing part:
    send and first-token times, every content chunk's arrival (seconds on
    `clock`), the reply text, usage, the finish reason and the server's
    request id."""
    url = urllib.parse.urlparse(endpoint)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    record = {"t_send": clock(), "t_first": None, "arrivals": [], "text": "", "usage": None,
              "finish_reason": None, "id": None, "error": None}
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), headers)
        resp = conn.getresponse()
        if resp.status != 200:
            record["error"] = f"HTTP {resp.status}: {resp.read()[:500]!r}"
            return record
        parts = []
        while True:
            raw = resp.readline()
            if not raw:
                break
            event = parse_sse_line(raw.decode("utf-8", "replace"))
            if event is None:
                continue
            if event == "[DONE]":
                break
            now = clock()
            record["id"] = record["id"] or event.get("id")
            if event.get("usage"):
                record["usage"] = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    if record["t_first"] is None:
                        record["t_first"] = now
                    record["arrivals"].append(now)
                    if delta.get("content"):
                        parts.append(delta["content"])
                if choice.get("finish_reason"):
                    record["finish_reason"] = choice["finish_reason"]
        record["text"] = "".join(parts)
    except Exception as e:  # a failed request is a result, not a crash
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    return record


def resolve_model(endpoint, api_key):
    url = urllib.parse.urlparse(endpoint)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=30)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    conn.request("GET", "/v1/models", headers=headers)
    models = json.loads(conn.getresponse().read())
    conn.close()
    return models["data"][0]["id"]


def run(args):
    corpus = load_corpus(args.corpus)
    think = tuple(float(x) for x in args.think_ms.split("-"))
    system = system_block(corpus, args.system_tokens)
    agents = plan(corpus, args.agents, args.turns, args.tool_tokens, think, args.seed)
    model = args.model or resolve_model(args.endpoint, args.api_key)
    os.makedirs(args.out, exist_ok=True)

    def body(messages, max_tokens):
        return {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens,
                "temperature": args.temperature, "enable_thinking": args.thinking,
                "stream_options": {"include_usage": True}, "class": args.cls}

    if args.warmup:
        stream_chat(args.endpoint, args.api_key,
                    body([{"role": "user", "content": "Say hello."}], 8), time.perf_counter, args.timeout)

    # perf_counter, not monotonic: on Windows monotonic ticks every ~15.6 ms,
    # coarser than the gaps between decode rounds this load exists to see.
    start = time.perf_counter()
    clock = lambda: time.perf_counter() - start
    lock = threading.Lock()
    out = open(os.path.join(args.out, "requests.jsonl"), "w", encoding="utf-8")
    failures = []

    def agent_loop(spec):
        time.sleep(spec["agent"] * args.stagger_ms / 1000)
        messages = [{"role": "system", "content": system}]
        for turn, step in enumerate(spec["turns"]):
            time.sleep(step["think_ms"] / 1000)
            messages.append({"role": "user", "content": user_message(spec["task"], step, turn == 0)})
            rec = stream_chat(args.endpoint, args.api_key, body(messages, args.max_tokens), clock, args.timeout)
            rec.update(agent=spec["agent"], turn=turn)
            with lock:
                out.write(json.dumps({k: v for k, v in rec.items() if k != "text"}) + "\n")
                out.flush()
                if rec["error"]:
                    failures.append(rec)
            if rec["error"]:
                print(f"  agent {spec['agent']} turn {turn}: {rec['error']}", file=sys.stderr)
                return
            messages.append({"role": "assistant", "content": rec["text"]})
            ttft = (rec["t_first"] - rec["t_send"]) * 1000 if rec["t_first"] else float("nan")
            print(f"  agent {spec['agent']} turn {turn}: ttft {ttft:7.0f} ms, "
                  f"{len(rec['arrivals'])} tokens, prompt {(rec['usage'] or {}).get('prompt_tokens')}")

    threads = [threading.Thread(target=agent_loop, args=(spec,)) for spec in agents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out.close()
    wall = clock()
    config = {k: v for k, v in vars(args).items() if k not in ("func", "api_key")}
    config.update(model=model, wall_s=round(wall, 3), system_chars=len(system),
                  plan_sha256=plan_digest(system, agents),
                  plan=[{"agent": a["agent"], "turns": [
                      {"path": s["path"], "think_ms": s["think_ms"]} for s in a["turns"]]} for a in agents])
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"done in {wall:.1f} s, {len(failures)} failed request(s) -> {args.out}")
    return 1 if failures else 0


# ── the report ──────────────────────────────────────────────────────────────


def percentile(values, q):
    """Nearest-rank percentile, q in [0, 100]; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))
    return ordered[int(rank) - 1]


def gaps_of(arrivals, burst_s=0.001):
    """Gaps between distinct arrivals. A speculative round lands several
    tokens at one instant; gaps under `burst_s` are inside a round and are
    not inter-token latency anyone waits for."""
    return [b - a for a, b in zip(arrivals, arrivals[1:]) if b - a >= burst_s]


def stall_windows(records, threshold_s, min_streams):
    """Time the whole server stood still: the union of intervals during
    which at least `min_streams` streams were each inside a gap longer than
    `threshold_s`. One lane's slow round is not a stall; every lane going
    quiet at once is what a synchronous copy looks like from outside."""
    events = []
    for rec in records:
        arr = rec["arrivals"]
        for a, b in zip(arr, arr[1:]):
            if b - a > threshold_s:
                events.append((a, 1))
                events.append((b, -1))
    # At one instant a gap opening sorts before one closing, so two gaps
    # that touch are one quiet stretch, not two.
    events.sort(key=lambda e: (e[0], -e[1]))
    total, count, active, since = 0.0, 0, 0, None
    for t, step in events:
        before = active
        active += step
        if before < min_streams <= active:
            since = t
        elif before >= min_streams > active and since is not None:
            total += t - since
            count += 1
            since = None
    return count, total


def read_requests(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_server_log(path):
    """request_id -> the server's admitted/done attributes."""
    by_id = {}
    if not os.path.exists(path):
        return by_id
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"ignis.request.' not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            attrs = event.get("attributes") or {}
            rid = attrs.get("request_id")
            if rid is None:
                continue
            name = event.get("event_name", "").rsplit(".", 1)[-1]
            by_id.setdefault(rid, {})[name] = attrs
    return by_id


def read_prom(path):
    """A Prometheus text scrape as {series: value}; empty when absent."""
    series = {}
    if not os.path.exists(path):
        return series
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            name, _, value = line.rpartition(" ")
            try:
                series[name.strip()] = float(value)
            except ValueError:
                pass
    return series


def summarize(run_dir, stall_ms=50.0):
    recs = read_requests(os.path.join(run_dir, "requests.jsonl"))
    ok = [r for r in recs if not r["error"] and r["t_first"] is not None]
    config = {}
    if os.path.exists(os.path.join(run_dir, "run.json")):
        with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
            config = json.load(f)
    agents = config.get("agents") or len({r["agent"] for r in recs}) or 1
    ttft = lambda rs: [(r["t_first"] - r["t_send"]) * 1000 for r in rs]
    gaps = [g * 1000 for r in ok for g in gaps_of(r["arrivals"])]
    tokens = sum(len(r["arrivals"]) for r in ok)
    wall = config.get("wall_s") or (max(r["arrivals"][-1] for r in ok) if ok else 0)
    stalls, stall_s = stall_windows(ok, stall_ms / 1000, max(2, min(3, agents - 1)))
    s = {
        "dir": run_dir,
        "requests": len(recs),
        "failed": len(recs) - len(ok),
        "wall_s": wall,
        "tokens": tokens,
        "agg_tok_s": tokens / wall if wall else None,
        "ttft_first_p50": percentile(ttft([r for r in ok if r["turn"] == 0]), 50),
        "ttft_later_p50": percentile(ttft([r for r in ok if r["turn"] > 0]), 50),
        "ttft_later_p95": percentile(ttft([r for r in ok if r["turn"] > 0]), 95),
        "gap_p50": percentile(gaps, 50),
        "gap_p95": percentile(gaps, 95),
        "gap_p99": percentile(gaps, 99),
        "gap_max": max(gaps) if gaps else None,
        "gaps_over_30ms": sum(1 for g in gaps if g > 30),
        "gaps_over_50ms": sum(1 for g in gaps if g > 50),
        "gaps_over_100ms": sum(1 for g in gaps if g > 100),
        f"global_stalls_{int(stall_ms)}ms": stalls,
        f"global_stall_s_{int(stall_ms)}ms": stall_s,
    }
    # The admitted line's `prefilled_tokens` is the position the prefill
    # reached, reuse included; what it did not compute shows in the chunk
    # count, and the exact reused tokens in the /metrics counters.
    log = read_server_log(os.path.join(run_dir, "serve", "ignis-server.log"))
    if log:
        ids = {int(r["id"].rsplit("-", 1)[-1]) for r in ok if r.get("id")}
        s["prefill_chunks"] = sum(log.get(rid, {}).get("admitted", {}).get("prefill_chunks_consumed", 0) for rid in ids)
    prompt = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in ok)
    s["prompt_tokens"] = prompt
    before = read_prom(os.path.join(run_dir, "metrics-before.prom"))
    after = read_prom(os.path.join(run_dir, "metrics-after.prom"))
    deltas = {name: value - before.get(name, 0.0) for name, value in sorted(after.items())
              if name.startswith(("ignis_retained_", "ignis_prefix_reused")) and "_total" in name}
    if after:
        reused = sum(v for k, v in deltas.items() if k.startswith(("ignis_retained_reused_tokens_total",
                                                                  "ignis_prefix_reused_tokens_total")))
        s["reused_share"] = reused / prompt if prompt else None
    s.update({k: v for k, v in deltas.items() if v})
    return s


def report(args):
    rows = [summarize(d, args.stall_ms) for d in args.dirs]
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    width = max(len(k) for k in keys)
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else
                                           f"{v:.1f}" if isinstance(v, float) else str(v))
    cols = [max(12, max(len(fmt(r.get(k))) for k in keys)) for r in rows]
    for k in keys:
        print(f"{k:<{width}}  " + "  ".join(f"{fmt(r.get(k)):>{c}}" for r, c in zip(rows, cols)))
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def selftest(_args):
    assert percentile([], 50) is None
    assert percentile([5, 1, 3], 50) == 3
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([7], 99) == 7
    # A round of four tokens at one instant is one arrival, not three gaps.
    assert [round(g, 4) for g in gaps_of([0.0, 0.010, 0.0101, 0.0102, 0.030])] == [0.010, 0.0198]
    assert parse_sse_line("data: [DONE]") == "[DONE]"
    assert parse_sse_line(": keep-alive") is None
    assert parse_sse_line('data: {"id":"chatcmpl-7"}')["id"] == "chatcmpl-7"
    # Three streams quiet over [1.0, 1.2]: one window of 0.15 s where all
    # three overlap; a lone slow stream elsewhere is not a stall.
    recs = [{"arrivals": [0.9, 1.0, 1.3]}, {"arrivals": [0.97, 1.05, 1.2]}, {"arrivals": [1.0, 1.02, 1.25]},
            {"arrivals": [2.0, 2.5]}]
    count, total = stall_windows(recs, 0.1, 3)
    assert count == 1 and abs(total - (1.2 - 1.05)) < 1e-9, (count, total)
    assert stall_windows(recs, 0.1, 5) == (0, 0.0)
    # The plan is a function of the seed alone.
    corpus = [("a.md", "line\n" * 1000), ("b.rs", "fn x() {}\n" * 800)]
    p1 = plan(corpus, 3, 4, 100, (500, 1500), 1)
    p2 = plan(corpus, 3, 4, 100, (500, 1500), 1)
    p3 = plan(corpus, 3, 4, 100, (500, 1500), 2)
    assert p1 == p2 and p1 != p3
    assert plan_digest("s", p1) == plan_digest("s", p2) != plan_digest("s", p3)
    assert all(0 <= len(s["text"]) <= 400 for a in p1 for s in a["turns"])
    assert p1[0]["turns"][0]["think_ms"] == 0.0, "an agent's first turn goes out at its stagger"
    assert system_block(corpus, 200) == system_block(corpus, 200)
    assert len(system_block(corpus, 200)) == 200 * CHARS_PER_TOKEN
    selftest_end_to_end()
    print("selftest ok")
    return 0


def selftest_end_to_end():
    """`run` then `report` against a stand-in server on a free local port:
    every turn sends the history the previous turns built, and the report
    reads what `run` wrote."""
    import http.server
    import tempfile

    bodies = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_GET(self):
            payload = json.dumps({"data": [{"id": "stand-in"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            # Chunk-framed, as axum streams SSE, so the client's readline()
            # is tested over the framing the real server sends.
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send(event):
                data = f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            rid = len(bodies)
            for piece in ("ok", " fine", "."):
                send({"id": f"chatcmpl-{rid}", "choices": [{"delta": {"content": piece}}]})
                time.sleep(0.002)
            send({"id": f"chatcmpl-{rid}", "choices": [{"delta": {}, "finish_reason": "stop"}]})
            send({"id": f"chatcmpl-{rid}", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
            send("[DONE]")
            self.wfile.write(b"0\r\n\r\n")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = os.path.join(tmp, "corpus")
            os.makedirs(corpus_dir)
            with open(os.path.join(corpus_dir, "a.md"), "w", encoding="utf-8") as f:
                f.write("some text\n" * 500)
            out = os.path.join(tmp, "run")
            args = argparse.Namespace(
                out=out, endpoint=f"http://127.0.0.1:{server.server_address[1]}", api_key=None, model=None,
                agents=2, turns=3, system_tokens=100, tool_tokens=50, max_tokens=8, think_ms="0-5",
                stagger_ms=0, seed=1, temperature=0.0, thinking=False, cls="agent", corpus=corpus_dir,
                timeout=30, warmup=True)
            assert run(args) == 0
            recs = read_requests(os.path.join(out, "requests.jsonl"))
            assert len(recs) == 6 and all(len(r["arrivals"]) == 3 and r["t_first"] is not None for r in recs), recs
            assert all(r["finish_reason"] == "stop" and r["usage"]["prompt_tokens"] == 10 for r in recs)
            chats = [b for b in bodies if len(b["messages"]) > 1]
            # Agent history: system, user, then (assistant, user) per later turn.
            assert sorted(len(b["messages"]) for b in chats) == [2, 2, 4, 4, 6, 6], [len(b["messages"]) for b in chats]
            assert all(b["messages"][0]["content"] == chats[0]["messages"][0]["content"] for b in chats), \
                "every agent shares one system block"
            assert all(b["messages"][2]["content"] == "ok fine." for b in chats if len(b["messages"]) > 2), \
                "a reply goes back into the history as sent"
            summary = summarize(out)
            assert summary["requests"] == 6 and summary["failed"] == 0 and summary["tokens"] == 18, summary
            assert json.load(open(os.path.join(out, "run.json"), encoding="utf-8"))["plan_sha256"]
    finally:
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(required=True)

    r = sub.add_parser("run", help="drive the load against a running server")
    r.add_argument("--out", required=True, help="run directory (created)")
    r.add_argument("--endpoint", default="http://127.0.0.1:8000")
    r.add_argument("--api-key", default=os.environ.get("IGNIS_API_KEY"))
    r.add_argument("--model", help="default: the first of /v1/models")
    r.add_argument("--agents", type=int, default=8)
    r.add_argument("--turns", type=int, default=8)
    r.add_argument("--system-tokens", type=int, default=12000,
                   help="the shared head, approximate (qwen-code's agent prompt with tools is ~10-15K)")
    r.add_argument("--tool-tokens", type=int, default=1500, help="each turn's tool result, approximate")
    r.add_argument("--max-tokens", type=int, default=192)
    r.add_argument("--think-ms", default="500-2000", help="simulated tool time between turns, uniform")
    r.add_argument("--stagger-ms", type=float, default=300, help="between agent starts")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--thinking", action="store_true", help="enable_thinking (off: bounded, comparable replies)")
    r.add_argument("--class", dest="cls", default="agent", choices=("agent", "interactive"))
    r.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
                   help="where tool results are cut from (default: this repository)")
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--no-warmup", dest="warmup", action="store_false")
    r.set_defaults(func=run)

    p = sub.add_parser("report", help="summarize run directories side by side")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--stall-ms", type=float, default=50.0,
                   help="a gap longer than this, in several lanes at once, is a global stall "
                        "(a synchronous 222 MiB PCIe copy is ~19 ms; a whole-history spill 30-50 ms)")
    p.set_defaults(func=report)

    t = sub.add_parser("selftest", help="check the pure parts offline")
    t.set_defaults(func=selftest)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
