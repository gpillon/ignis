"""Spec 22's set R3 (GitHub #278): real logs for the acceptance of `locate`'s
defaults, registered before any route is asked on it
(`docs/specs/decide/22-locate-by-copy-over-a-folded-state.md` § Set R3).

Two sources, neither ever committed (`.scratch/`):

- the owner's cluster: a fresh `kubectl logs --since=6h --timestamps`
  capture, read only, merged by `prodset.py timeline`;
- the **full** LogHub logs (Zenodo record 8196385, not the 2k samples):
  `loghub` fetches each system's archive and keeps the first 8 MB of its log
  (the large ones — Spark, Thunderbird, Windows — streamed and cut there),
  every line of the system's 2k sample left out, so no window shares a line
  with it.

`windows` cuts, seeded (20261050): cluster windows of 16K / 50K / 100K /
200K tokens (two each) and two of ~1M, 4 targets each; per LogHub system one
window of 100K tokens and, where its stream holds it, one of 200K, 3 targets
each. Targets by `r2set.py`'s rule: siblings are the lines whose word sets —
times, numbers, ids, hashes removed — have a Jaccard of at least 0.5 with
it; a line whose normalized text occurs twice is not a target; targets are
drawn round-robin over the sibling bins 0, 1-5, 6-50, > 50. Past 100K tokens
siblings are counted for 400 sampled eligible lines (against every line),
as `z23_r4.py` sampled 120: every line's count is quadratic in the window.

`show` prints each target with its three nearest lines, for writing a
question that singles it out. `build` checks each question by
`r2set.py build`'s rules — a lexical or combo question's shared content
words, all together, in no other line; a paraphrase shares none and only on
a target without siblings — and **drops** (never replaces) a target no
question can single out; an authored absent question's content words
together in no line of its window. The manifest holds each window's lines
once and three kinds of question: `present`, `removed` (a present question
over its window without its target line — `zd_notfound.py --deleted`'s
construction) and `authored` (absent).

    python r3set.py loghub --samples <loghub 2k dir> --cache <downloads> --out <full>
    python r3set.py windows --timeline timeline4.txt --loghub <full> --out r3-windows.json
    python r3set.py show --windows r3-windows.json [--window ID ...]
    python r3set.py build --windows r3-windows.json --questions r3-questions.json --out <R3>
"""

import argparse
import hashlib
import io
import json
import os
import random
import tarfile
import urllib.request
import zipfile

from common import content_stems, paraphrase_clean, rare_shared
from r2set import BINS, STRIP, WORD, cost_of, cut

SEED = 20261050
ZENODO = "https://zenodo.org/records/8196385/files/{}?download=1"
STREAM_BYTES = 8 << 20
# System -> (archive, the members that make its log, in archive order).
SYSTEMS = {
    "Android": ("Android_v1.zip", ["Android.log"]),
    "Apache": ("Apache.tar.gz", ["Apache.log"]),
    "BGL": ("BGL.zip", ["BGL.log"]),
    "Hadoop": ("Hadoop.zip", None),
    "HDFS": ("HDFS_v1.zip", ["HDFS.log"]),
    "HealthApp": ("HealthApp.tar.gz", ["HealthApp.log"]),
    "HPC": ("HPC.zip", ["HPC.log"]),
    "Linux": ("Linux.tar.gz", ["Linux.log"]),
    "Mac": ("Mac.tar.gz", ["Mac.log"]),
    "OpenSSH": ("SSH.tar.gz", ["SSH.log"]),
    "OpenStack": ("OpenStack.tar.gz", ["openstack_normal1.log", "openstack_normal2.log", "openstack_abnormal.log"]),
    "Proxifier": ("Proxifier.tar.gz", ["Proxifier.log"]),
    "Spark": ("Spark.tar.gz", None),
    "Thunderbird": ("Thunderbird.tar.gz", ["Thunderbird.log"]),
    "Windows": ("Windows.tar.gz", ["Windows.log"]),
    "Zookeeper": ("Zookeeper.tar.gz", ["Zookeeper.log"]),
}
CLUSTER_TIERS = ((16_000, 2), (50_000, 2), (100_000, 2), (200_000, 2), (1_000_000, 2))
LOGHUB_TIERS = (100_000, 200_000)
SAMPLED_PAST = 100_000


def _members(names, wanted):
    """The members a system's log is made of: the named ones, or every
    `.log` member in archive order."""
    if wanted is not None:
        return [n for n in wanted if n in names]
    return [n for n in names if n.endswith(".log")]


def loghub(args):
    """Each system's stream: the first `STREAM_BYTES` of its log, lines of
    its 2k sample left out."""
    os.makedirs(args.out, exist_ok=True)
    for system, (archive, wanted) in SYSTEMS.items():
        sample_path = os.path.join(args.samples, f"{system}_2k.log")
        with open(sample_path, encoding="utf-8", errors="replace") as f:
            sample = set(f.read().split("\n"))
        cached = os.path.join(args.cache, archive)
        data = bytearray()
        if archive.endswith(".zip"):
            if not os.path.exists(cached):
                urllib.request.urlretrieve(ZENODO.format(archive), cached)
            with zipfile.ZipFile(cached) as z:
                for name in _members(z.namelist(), wanted):
                    with z.open(name) as member:
                        data += member.read(STREAM_BYTES - len(data))
                    if len(data) >= STREAM_BYTES:
                        break
        else:
            source = open(cached, "rb") if os.path.exists(cached) else urllib.request.urlopen(ZENODO.format(archive))
            with source, tarfile.open(fileobj=source, mode="r|gz") as t:
                for member in t:
                    if not member.isfile() or os.path.basename(member.name) not in _members(
                            [os.path.basename(member.name)], wanted):
                        continue
                    data += t.extractfile(member).read(STREAM_BYTES - len(data))
                    if len(data) >= STREAM_BYTES:
                        break
        text = data.decode("utf-8", errors="replace")
        lines = text.split("\n")
        if len(data) >= STREAM_BYTES:
            lines = lines[:-1]  # a cut line is no line
        kept = [line.rstrip("\r") for line in lines if line.strip() and line.rstrip("\r") not in sample]
        with open(os.path.join(args.out, f"{system}.log"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(kept) + "\n")
        print(f"{system}: {len(kept)} lines kept ({len(lines) - len(kept)} blank or in the 2k sample)", flush=True)


def stream_of(path, source):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().rstrip("\n").split("\n")
    if source == "cluster":
        return [line for line in lines if len(line) <= 1200]
    return [line for line in lines if line.strip()]


def pick_targets(lines, want, r, sampled):
    """`want` targets by `r2set.py`'s rule; siblings over a sample of 400
    eligible lines when `sampled`."""
    norm = [STRIP.sub(" ", line) for line in lines]
    seen = {}
    for n in norm:
        seen[n] = seen.get(n, 0) + 1
    eligible = [i for i in range(len(lines)) if seen[norm[i]] == 1 and 20 <= len(lines[i]) <= 600]
    sets = [set(WORD.findall(n.lower())) for n in norm]
    if sampled:
        r.shuffle(eligible)
        eligible = sorted(eligible[:400])
    sib = {}
    for i in eligible:
        a = sets[i]
        sib[i] = sum(1 for j, b in enumerate(sets) if j != i and a and len(a & b) / len(a | b) >= 0.5)
    by_bin = [[i for i in eligible if lo <= sib[i] <= hi] for lo, hi in BINS]
    for pool in by_bin:
        r.shuffle(pool)
    picks = []
    while len(picks) < want and any(by_bin):
        for b, pool in enumerate(by_bin):
            if pool and len(picks) < want:
                i = pool.pop()
                picks.append({"line": i, "siblings": sib[i], "bin": b})
    return picks


def windows(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    r = random.Random(SEED)
    out = []
    stream = stream_of(args.timeline, "cluster")
    cost = cost_of(tok, stream)
    for tier, count in CLUSTER_TIERS:
        for w in range(count):
            got = cut(stream, cost, tier, r)
            if got is None:
                raise SystemExit(f"the capture holds no window of {tier} tokens")
            out.append({"id": f"c{tier // 1000:04}k-{w}", "source": "cluster", "file": args.timeline,
                        "tier": tier, "start": got[0], "end": got[1], "tokens": got[2], "want": 4})
    for system in sorted(SYSTEMS):
        path = os.path.join(args.loghub, f"{system}.log")
        lines = stream_of(path, "loghub")
        c = cost_of(tok, lines)
        for tier in LOGHUB_TIERS:
            if sum(c) < tier:
                print(f"{system}: its stream holds no window of {tier} tokens ({sum(c)})")
                continue
            got = cut(lines, c, tier, r)
            if got is None:
                print(f"{system}: no window of {tier} tokens found")
                continue
            out.append({"id": f"h-{system.lower()}-{tier // 1000:03}k", "source": "loghub", "file": path,
                        "tier": tier, "start": got[0], "end": got[1], "tokens": got[2], "want": 3})
    for w in out:
        lines = window_lines(w)
        w["lines"] = len(lines)
        w["targets"] = pick_targets(lines, w.pop("want"), r, w["tier"] > SAMPLED_PAST)
        print(f"{w['id']:22s} {w['tier']:>8} tokens {w['tokens']:>8} lines {len(lines):>6} targets "
              + " ".join(f"{p['line']}(s{p['siblings']})" for p in w["targets"]), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "windows": out}, f, indent=1)
    at100 = sum(1 for w in out if w["tier"] == 100_000)
    print(f"{len(out)} windows, {sum(len(w['targets']) for w in out)} targets, {at100} at 100K tokens")


def window_lines(w):
    return stream_of(w["file"], w["source"])[w["start"]:w["end"]]


def show(args):
    meta = json.load(open(args.windows, encoding="utf-8"))
    for w in meta["windows"]:
        if args.window and w["id"] not in args.window:
            continue
        lines = window_lines(w)
        sets = [set(WORD.findall(STRIP.sub(" ", line).lower())) for line in lines]
        print(f"=== {w['id']} ({w['lines']} lines, {w['source']})")
        for p in w["targets"]:
            i = p["line"]
            near = sorted(((len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])), j)
                           for j in range(len(lines)) if j != i), reverse=True)[:3]
            print(f"  T {i} (siblings {p['siblings']}): {lines[i][:args.width]}")
            for s, j in near:
                print(f"     ~{s:.2f} {j}: {lines[j][:args.width]}")


def check(q, lines, t, sib):
    """`r2set.py build`'s rules for one present question; the problem, or
    None."""
    target, rest = lines[t], lines[:t] + lines[t + 1:]
    if q["split"] == "lexical" and not rare_shared(q["instruction"], target, rest):
        return "lexical question shares no rare word"
    if q["split"] == "paraphrase":
        if not paraphrase_clean(q["instruction"], target):
            return "paraphrase shares a content word"
        if sib:
            return f"paraphrase on a target with {sib} siblings"
    if q["split"] in ("combo", "lexical"):
        shared = content_stems(q["instruction"]) & content_stems(target)
        also = [i for i, line in enumerate(lines) if i != t and shared <= content_stems(line)]
        if not shared or also:
            return f"{q['split']}: shared {sorted(shared)} also in {also[:5]}"
    return None


def build(args):
    meta = {w["id"]: w for w in json.load(open(args.windows, encoding="utf-8"))["windows"]}
    asked = json.load(open(args.questions, encoding="utf-8"))
    texts, rows, dropped = {}, [], []
    for q in asked["questions"]:
        w = meta[q["window"]]
        lines = texts.setdefault(w["id"], window_lines(w))
        t = q["target"]
        target = next((p for p in w["targets"] if p["line"] == t), None)
        if target is None:
            raise SystemExit(f"{q['window']}:{t} is not one of the window's drawn targets")
        problem = check(q, lines, t, target["siblings"])
        if problem:
            dropped.append(f"{q['window']}:{t} {problem}")
            continue
        base = {"source": w["source"], "split": q["split"], "segments": w["tier"], "tier": w["tier"],
                "siblings": target["siblings"], "bin": target["bin"], "window": w["id"],
                "instruction": q["instruction"], "depth": t / len(lines)}
        rows.append(dict(base, id=f"r3-{w['id']}-{t}", variant="present", absent=False, remove=None, targets=[t]))
        rows.append(dict(base, id=f"r3-{w['id']}-{t}~removed", variant="removed", absent=True, remove=t, targets=[]))
    for n, q in enumerate(asked["absent"]):
        w = meta[q["window"]]
        lines = texts.setdefault(w["id"], window_lines(w))
        stems = content_stems(q["instruction"])
        holders = [i for i, line in enumerate(lines) if stems and stems <= content_stems(line)]
        if not stems or holders:
            dropped.append(f"{q['window']} authored absent: its words together in {holders[:5]}")
            continue
        rows.append({"id": f"r3-{w['id']}-absent{n}", "source": w["source"], "split": "authored",
                     "segments": w["tier"], "tier": w["tier"], "window": w["id"], "instruction": q["instruction"],
                     "variant": "authored", "absent": True, "remove": None, "targets": []})
    for d in dropped:
        print("DROP", d)
    present = sum(r["variant"] == "present" for r in rows)
    at100 = len({r["window"] for r in rows if r["tier"] == 100_000 and r["variant"] == "present"})
    print(f"{present} present, {sum(r['variant'] == 'authored' for r in rows)} authored absent, "
          f"{sum(r['variant'] == 'removed' for r in rows)} target-removed; {at100} windows at 100K tokens; "
          f"{len(dropped)} dropped")
    os.makedirs(args.out, exist_ok=True)
    manifest = {"seed": SEED, "windows": texts, "questions": rows}
    body = json.dumps(manifest, ensure_ascii=False)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print("manifest sha256", hashlib.sha256(body.encode("utf-8")).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("loghub")
    h.add_argument("--samples", required=True, help="the LogHub 2k samples")
    h.add_argument("--cache", required=True, help="where archives already downloaded are, and go")
    h.add_argument("--out", required=True)
    w = sub.add_parser("windows")
    w.add_argument("--timeline", required=True)
    w.add_argument("--loghub", required=True)
    w.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    w.add_argument("--out", required=True)
    s = sub.add_parser("show")
    s.add_argument("--windows", required=True)
    s.add_argument("--window", nargs="*")
    s.add_argument("--width", type=int, default=240)
    b = sub.add_parser("build")
    b.add_argument("--windows", required=True)
    b.add_argument("--questions", required=True)
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    {"loghub": loghub, "windows": windows, "show": show, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
