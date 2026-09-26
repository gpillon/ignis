"""GitHub #270 live acceptance (spec docs/specs/decide/16-reuse-boundaries.md,
acceptance 7): does /v1/decide reuse a `state` given as content parts?

The issue's repro, kept in the repository and extended with a reuse-marker
case. Standard library only.

    IGNIS_API_KEY=... python scripts/decide-reuse-repro.py [http://127.0.0.1:8000] [http://127.0.0.1:9464]

Run it against a server started with the issue's flags (`--vision --spec
dflash2 --metrics`, `--prompt-reuse` left on) and nothing else using it. Each
case sends N requests one after another (nothing concurrent), so only
cross-request reuse and a fan-out's own head can help. Every case has its own
text and its own picture, so no case is served by what an earlier one left
behind: the "first" column is what that case costs cold.

Reported per case: the first request, the second, the median of requests
2..N, and the deltas of the retained-state counters from the metrics listener.
Then the spec's inequalities, each PASS or FAIL.
"""
import base64
import json
import os
import random
import statistics
import struct
import sys
import time
import urllib.request
import zlib

API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
METRICS = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:9464"
KEY = os.environ.get("IGNIS_API_KEY", "")
N = 5


def long_text(case):
    """~1,500 tokens of static instructions, identical in every request of a
    case (the shape of a game agent's mission prompt) and unlike any other
    case's."""
    return f"Mission {case}. " + " ".join(
        f"Rule {i}: when you see a door, a switch, a lift, a key or a monster, consider what it means for reaching "
        f"the exit of the level, and prefer actions that explore unseen space and keep you alive."
        for i in range(1, 60))


SHORT = "A first-person view from a video game."


def png(seed, w=320, h=208):
    """A noise image, as a data URI (the same bytes for the same seed)."""
    rnd = random.Random(seed)
    raw = b"".join(b"\x00" + bytes(rnd.randrange(256) for _ in range(w * 3)) for _ in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


def text(value, marked=False):
    part = {"type": "text", "text": value}
    if marked:
        part["cache_control"] = {"type": "ephemeral"}
    return part


def image(seed):
    return {"type": "image_url", "image_url": {"url": png(seed)}}


Q1 = {"q": {"type": "noul", "instructions": "Is there a monster in the view?"}}
Q4 = {f"q{i}": {"type": "noul", "instructions": f"Question {i}: is there a monster in the view?"} for i in range(4)}


def post(path, body):
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t) * 1000


COUNTERS = ("ignis_retained_state_hits_total", "ignis_retained_reused_tokens_total",
            "ignis_prefix_reused_tokens_total", "ignis_retained_state_misses_total")


def counters():
    try:
        text_ = urllib.request.urlopen(METRICS + "/metrics", timeout=10).read().decode()
    except OSError:
        return {}
    out = {}
    for line in text_.splitlines():
        if line.startswith(COUNTERS) or line.startswith('ignis_retained_slots{state="in_use"}'):
            k, v = line.rsplit(" ", 1)
            out[k] = float(v)
    return out


RESULTS = {}


def case(name, label, state, questions):
    before = counters()
    ms = []
    for _ in range(N):
        ans, t = post("/v1/decide", {"state": state, "questions": questions})
        assert all(a.get("type") != "error" for a in ans["answers"].values()), ans
        ms.append(t)
    after = counters()
    delta = {k.replace("ignis_", ""): int(after[k] - before.get(k, 0))
             for k in after if after[k] != before.get(k, 0)}
    RESULTS[name] = {"first": ms[0], "second": ms[1], "rest": statistics.median(ms[1:]), "delta": delta}
    print(f"{name}  {label:44s} first {ms[0]:6.0f}  second {ms[1]:6.0f}  median 2..{N} {RESULTS[name]['rest']:6.0f} ms"
          f"   {delta}", flush=True)


print(f"server {API}, {N} sequential requests per case\n")
case("A", "JSON state {mission: LONG}", {"mission": long_text("A")}, Q1)
case("B", "parts [LONG]", [text(long_text("B"))], Q1)
case("C", "parts [SHORT, image]", [text(SHORT), image(3)], Q1)
case("D", "parts [LONG, image]", [text(long_text("D")), image(4)], Q1)
case("E", "parts [LONG, image], 4 questions", [text(long_text("E")), image(5)], Q4)
case("F", "parts [SHORT, image], 4 questions", [text(SHORT), image(6)], Q4)
case("G", "parts [LONG + marker, image]", [text(long_text("G"), marked=True), image(7)], Q1)

r = RESULTS


def prefix_reused(name):
    return sum(v for k, v in r[name]["delta"].items()
               if k.startswith("retained_reused_tokens_total") and 'kind="prefix"' in k)


checks = [
    ("B <= 1.3 x A", r["B"]["rest"] <= 1.3 * r["A"]["rest"]),
    ("D <= C + 30 ms", r["D"]["rest"] <= r["C"]["rest"] + 30),
    ("E <= 2 x D", r["E"]["rest"] <= 2 * r["D"]["rest"]),
    ("E first <= 1.5 x D first", r["E"]["first"] <= 1.5 * r["D"]["first"]),
    ("G second <= C + 30 ms", r["G"]["second"] <= r["C"]["rest"] + 30),
    ("prefix reuse grows in B", prefix_reused("B") > 0),
    ("prefix reuse grows in D", prefix_reused("D") > 0),
    ("prefix reuse grows in E", prefix_reused("E") > 0),
]
print()
for label, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
sys.exit(0 if all(ok for _, ok in checks) else 1)
