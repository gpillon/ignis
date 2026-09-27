"""Score spec 19's span dumps: where in the text, to the key (phase 2, GitHub #276).

`crates/server/tests/attention_span_locate_gpu.rs` dumps, per question of a
span set (`spans.py`), every head's scores over the full row at the copy
scaffold's last token with the instruction (`q`) and without it (`q-na`),
the instruction's own queries' mean weights, and -- with the first gold
forced after the scaffold -- the span rows at the quote's first, middle and
last tokens. This reads them on the key grid of the state:

- **gold keys**: each gold character span (segment, start, end in the
  segment's plain text) goes to evidence bytes through the evidence writer
  and onto the keys whose byte ranges it overlaps (`key_bytes` in the dump);
- **readings** (`read`), chosen on the development sets by cross-validation
  over questions (`score.fold_of`'s folds): one head's argmax key; a vote of
  the K best heads' argmax keys, each counting within `w` keys; the argmax
  of the K heads' summed z-scored profiles; and the segment the line vote
  names, then its best key -- each over the lift (`q` less `q-na`) or the
  weight alone. A span grows from the answer key while the summed profile
  stays within `delta` of its peak, inside the answer's segment;
- **metrics**: token hit (the answer key inside a gold span), key-level span
  F1 and exact match, by family / split / length; the vote agreement's
  present/absent AUC;
- **the generation route** (`generation.py`) on the same grid: its first
  quote's first occurrence mapped onto keys, scored the same way;
- **Q4** (`regions`): each head's softmax mass over the full row by region
  (evidence, kind text, instruction, template, scaffold), with and without
  the instruction, lexical against paraphrase;
- **forced** (`forced`): where each head's argmax key sits against the gold
  span at the quote's first, middle and last tokens.

    python spanscore.py extract --dump <E1-hq.json> --manifest <E1/manifest.json> --out <dir>
    python spanscore.py report --sets <dir>/E1 <dir>/E2 --dumps <E1-hq.json> <E2-hq.json> --generation <E1-generation.json> <E2-generation.json> --out spans.json

Needs NumPy.
"""

import argparse
import json
import os

import numpy as np

import profiles
import score
from common import rare_shared, stem

H = score.N_HEADS
REGIONS = ("evidence", "kind", "instruction", "template", "scaffold")
KS = (1, 3, 5, 8, 16, 32)
WINDOWS = (0, 1, 2)
DELTAS = (0.5, 1.0, 2.0, 4.0)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(value, f, indent=1)


# ---------------------------------------------------------------------------
# Gold character spans onto keys
# ---------------------------------------------------------------------------

def plain_to_evidence_bytes(state):
    """For each segment, a function from a plain-text character offset to a
    byte offset in the evidence text (`profiles.evidence`), through the JSON
    escaping of a line; an array element is written verbatim."""
    text, segments = profiles.evidence(state)
    prefix = np.concatenate([[0], np.cumsum([len(c.encode("utf-8")) for c in text])])
    maps = []
    for j, (start, end, _) in enumerate(segments):
        if isinstance(state, str):
            line = state.split("\n")[j]
            widths = [len(json.dumps(c, ensure_ascii=False)) - 2 for c in line]
            escaped = np.concatenate([[0], np.cumsum(widths)]).astype(int) + start
        else:
            escaped = np.arange(start, end + 1)
        maps.append(lambda i, e=escaped: int(prefix[e[i]]))
    return maps


def span_keys(state, spans, key_bytes):
    """[S] bool: the span keys any of `spans` overlaps, and the per-span key
    index lists."""
    maps = plain_to_evidence_bytes(state)
    kb = np.asarray(key_bytes)
    mask = np.zeros(len(kb), dtype=bool)
    each = []
    for s in spans:
        b0, b1 = maps[s["segment"]](s["start"]), maps[s["segment"]](s["end"])
        hit = (kb[:, 0] < b1) & (kb[:, 1] > b0)
        mask |= hit
        each.append(np.flatnonzero(hit))
    return mask, each


# ---------------------------------------------------------------------------
# extract: one pass over a span dump
# ---------------------------------------------------------------------------

def block(raw, offset, rows, cols):
    return np.asarray(raw[offset:offset + rows * cols], dtype=np.float32).reshape(rows, cols)


def softmax_mass(scores, labels, n_labels):
    """[H, T] scores -> [H, n_labels] softmax mass per label (labels [T])."""
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    e /= e.sum(axis=1, keepdims=True)
    out = np.zeros((scores.shape[0], n_labels), dtype=np.float32)
    for k in range(n_labels):
        out[:, k] = e[:, labels == k].sum(axis=1)
    return out


def region_labels(row, total, which="q"):
    labels = np.full(total, REGIONS.index("template"), dtype=np.int32)
    runs = row["regions"] if which == "q" else row["q-na"]["regions"]
    for a, b, name in runs:
        labels[a:b] = REGIONS.index(name)
    s0, s1 = row["scaffold"] if which == "q" else (total - (row["scaffold"][1] - row["scaffold"][0]), total)
    labels[s0:s1] = REGIONS.index("scaffold")
    return labels


def instruction_rare_positions(row, question, tokenizer):
    """The prompt positions of the instruction's tokens that carry a word it
    shares with its gold segment and with no other segment (a lexical
    question's rare word, `common.rare_shared`), or None when the question
    has none or the instruction's tokens cannot be matched to its region.
    The instruction is tokenized as the render writes it, inside
    `{"instruction":"…"}`, and its tokens are held to the region's count."""
    if tokenizer is None or not question["spans"]:
        return None
    texts = profiles.segment_texts(question["state"])
    gold = {s["segment"] for s in question["spans"]}
    others = [t for j, t in enumerate(texts) if j not in gold]
    rare = set()
    for g in gold:
        rare |= rare_shared(question["instruction"], texts[g], others)
    if not rare:
        return None
    region = next((a, b) for a, b, name in row["regions"] if name == "instruction")
    wrapper = '{"instruction":' + json.dumps(question["instruction"], ensure_ascii=False, separators=(",", ":")) + "}"
    enc = tokenizer.encode(wrapper, add_special_tokens=False)
    base = len('{"instruction":"')
    inside = [i for i, (a, b) in enumerate(enc.offsets) if a >= base and a < len(wrapper) - 2]
    if len(inside) != region[1] - region[0]:
        return None
    escaped = wrapper[base:len(wrapper) - 2]
    positions = []
    for m in profiles.WORD_SPAN.finditer(escaped.lower()):
        if stem(m.group()) in rare:
            a, b = base + m.start(), base + m.end()
            positions += [region[0] + k for k, i in enumerate(inside) if enc.offsets[i][0] < b and enc.offsets[i][1] > a]
    return sorted(set(positions)) or None


def question_stats(raw, row, question, tokenizer=None):
    s0, S = row["span"]
    T, Tn = row["q"]["keys"], row["q-na"]["keys"]
    q = block(raw, row["q"]["scores_offset"], H, T)
    na = block(raw, row["q-na"]["scores_offset"], H, Tn)
    qs, nas = q[:, s0:s0 + S], na[:, s0:s0 + S]
    lift = qs - nas
    gold, each = span_keys(question["state"], question["spans"], row["key_bytes"])
    out = {
        "argmax": np.stack([qs.argmax(axis=1), lift.argmax(axis=1), nas.argmax(axis=1)]).astype(np.int32),
        "gold": gold,
        "gold_first": np.array([e[0] for e in each if len(e)], dtype=np.int32),
        "gold_last": np.array([e[-1] for e in each if len(e)], dtype=np.int32),
    }
    # each head's five best segments by the lift of its shares (line level)
    shares_q, _ = score.features(qs, row["keys"])
    shares_na, _ = score.features(nas, row["keys"])
    own = np.array([k is not None for k in row["keys"]])
    seg_lift = np.where(own[None, :], shares_q.astype(np.float64) - shares_na, -np.inf)
    out["seg_top"] = np.argsort(-seg_lift, axis=1, kind="stable")[:, :5].astype(np.int32)
    out["gold_segments"] = np.array(sorted({s["segment"] for s in question["spans"]}), dtype=np.int32)
    if gold.any() and not gold.all():
        auc, ap = profiles.auroc_ap(np.concatenate([qs, lift]), gold)
        out["auc"], out["ap"] = auc.reshape(2, H).astype(np.float32), ap.reshape(2, H).astype(np.float32)
    # Q4: where the full row's mass goes, with and without the instruction.
    labels = region_labels(row, T)
    out["mass_q"] = softmax_mass(q, labels, len(REGIONS))
    out["mass_na"] = softmax_mass(na, region_labels(row, Tn, "na"), len(REGIONS))
    gl = np.zeros(T, dtype=np.int32)
    gl[s0:s0 + S][gold] = 1
    out["mass_gold"] = softmax_mass(q, gl, 2)[:, 1]
    # H1: the scaffold's mass on the instruction's copy of the rare word
    rare = instruction_rare_positions(row, question, tokenizer)
    if rare:
        rl = np.zeros(T, dtype=np.int32)
        rl[rare] = 1
        out["mass_rare_instruction"] = softmax_mass(q, rl, 2)[:, 1]
        out["rare_instruction_tokens"] = len(rare)
    # the instruction's own queries (ICR's direction): mass on the state,
    # and their argmax key inside the span
    W = row["q"]["weight_keys"]
    w = block(raw, row["q"]["weights_offset"], H, W)
    out["icr_state"] = w[:, s0:s0 + S].sum(axis=1).astype(np.float32)
    out["icr_argmax"] = w[:, s0:s0 + S].argmax(axis=1).astype(np.int32)
    # the forced quote's queries
    f = row.get("q-forced")
    if f:
        out["forced_argmax"] = np.stack([block(raw, off, H, S).argmax(axis=1)
                                         for off in f["scores_offsets"]]).astype(np.int32)
        out["forced_queries"] = np.array(f["queries"], dtype=np.int32) - (row["scaffold"][1] - 1)
        out["forced_n"] = int(f["quote_tokens"])
    return out


def extract(dump_path, manifest_path, out_dir, limit=None, tokenizer_path=None):
    meta = read_json(dump_path)
    base = os.path.dirname(dump_path)
    with open(os.path.join(base, meta["rows_file"]), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    questions = {q["id"]: q for q in read_json(manifest_path)["questions"]}
    raw = np.memmap(os.path.join(base, meta["bin_file"]), dtype="<f2", mode="r")
    rows = rows[:limit] if limit else rows
    tokenizer = None
    if tokenizer_path:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
    stats = []
    for i, row in enumerate(rows):
        stats.append(question_stats(raw, row, questions[row["id"]], tokenizer))
        if (i + 1) % 25 == 0:
            print(f"{meta['set']}: {i + 1}/{len(rows)}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{meta['set']}.npz")
    np.savez_compressed(path, ids=np.array([r["id"] for r in rows]), stats=np.array(stats, dtype=object))
    print(f"{meta['set']}: {len(rows)} questions -> {path}")


# ---------------------------------------------------------------------------
# report: readings chosen by cross-validation
# ---------------------------------------------------------------------------

def load_all(set_paths, dumps, manifests, generations=None):
    """The questions of every set, with their extract, the raw dump for the
    combined readings, and -- given one `generation.py` file per set, in the
    same order -- the generation route's answer (ids repeat across sets)."""
    rows = []
    generations = generations or [None] * len(set_paths)
    for path, dump, manifest, generation in zip(set_paths, dumps, manifests, generations):
        gen = {g["id"]: g for g in read_json(generation)["questions"]} if generation else {}
        meta = read_json(dump)
        with open(os.path.join(os.path.dirname(dump), meta["rows_file"]), encoding="utf-8") as f:
            dumped = {r["id"]: r for r in (json.loads(line) for line in f)}
        stored = np.load(path + ".npz", allow_pickle=True)
        raw = np.memmap(os.path.join(os.path.dirname(dump), meta["bin_file"]), dtype="<f2", mode="r")
        texts = {q["id"]: q for q in read_json(manifest)["questions"]}
        for qid, st in zip(stored["ids"], stored["stats"]):
            q = texts[str(qid)]
            r = dumped[str(qid)]
            rows.append({"id": str(qid), "set": meta["set"], "family": q["family"], "split": q["split"],
                         "absent": q["absent"], "state": q["state"], "spans": q["spans"],
                         "keys": r["keys"], "key_bytes": r["key_bytes"], "x": st, "tier": tier_of(r),
                         "raw": raw, "at": (r["span"][0], r["span"][1], r["q"]["scores_offset"], r["q"]["keys"],
                                            r["q-na"]["scores_offset"], r["q-na"]["keys"]),
                         "gen": gen.get(str(qid))})
    return rows


def tier_of(row):
    n = row["span"][1]
    return 0 if n <= 2500 else (1 if n <= 6000 else 2)


def key_segment(row):
    """[S] the segment of each span key (-1 for a separator)."""
    owner = np.full(len(row["key_bytes"]), -1, dtype=np.int32)
    for j, k in enumerate(row["keys"]):
        if k is not None:
            owner[k[0]:k[1]] = j
    return owner


def token_hits(rows, o):
    """[Q, H] whether each head's argmax key (ordinate o: 0 weight, 1 lift)
    is a gold key."""
    return np.stack([r["x"]["gold"][r["x"]["argmax"][o]] for r in rows])


def best_heads(train, o, k):
    hits = token_hits(train, o).sum(axis=0)
    return [int(h) for h in np.argsort(-hits, kind="stable")[:k]]


def zprof(r, heads, o):
    """[len(heads), S] z-scored per-key profiles of `heads`, read from the
    dump: the weight's scores (o = 0) or the lift (o = 1)."""
    s0, S, qo, T, no, Tn = r["at"]
    raw = r["raw"]
    rows = []
    for h in heads:
        y = np.asarray(raw[qo + h * T + s0:qo + h * T + s0 + S], dtype=np.float32)
        if o == 1:
            y = y - np.asarray(raw[no + h * Tn + s0:no + h * Tn + s0 + S], dtype=np.float32)
        rows.append(y)
    p = np.stack(rows)
    return (p - p.mean(axis=1, keepdims=True)) / (p.std(axis=1, keepdims=True) + 1e-6)


def answer_key(r, heads, method, o, w):
    arg = r["x"]["argmax"][o][heads]
    if method == "head":
        return int(arg[0]), 1.0
    if method == "vote":
        S = len(r["x"]["gold"])
        counts = np.zeros(S)
        for a in arg:
            counts[max(0, a - w):a + w + 1] += 1
        best = counts.max()
        tied = set(np.flatnonzero(counts == best).tolist())
        return int(next((a for a in arg if a in tied), min(tied))), best / len(heads)
    z = zprof(r, heads, o)
    y = z.sum(axis=0)
    if method == "sum":
        return int(y.argmax()), float(np.mean(np.abs(arg - y.argmax()) <= 2))
    # segment first: the most-voted segment of the heads' argmax keys, then
    # the summed profile's best key inside it
    seg = key_segment(r)
    votes = [seg[a] for a in arg if seg[a] >= 0]
    if not votes:
        return int(y.argmax()), 0.0
    vals, counts = np.unique(votes, return_counts=True)
    top = vals[counts == counts.max()]
    s = next((v for v in votes if v in top), top[0])
    inside = np.flatnonzero(seg == s)
    return int(inside[y[inside].argmax()]), counts.max() / len(heads)


def grow(r, heads, o, key, delta):
    """The span around `key`: neighbours inside its segment whose summed
    profile is within `delta` of the answer key's."""
    z = zprof(r, heads, o)
    y = z.sum(axis=0) if len(z) else np.zeros(len(r["x"]["gold"]))
    seg = key_segment(r)
    lo = hi = key
    while lo - 1 >= 0 and seg[lo - 1] == seg[key] and y[lo - 1] >= y[key] - delta:
        lo -= 1
    while hi + 1 < len(y) and seg[hi + 1] == seg[key] and y[hi + 1] >= y[key] - delta:
        hi += 1
    return lo, hi + 1


def span_f1(r, lo, hi):
    """Key-level F1 and exact match of [lo, hi) against the best gold span."""
    best, em = 0.0, False
    gold = r["x"]["gold"]
    for first, last in zip(r["x"]["gold_first"], r["x"]["gold_last"]):
        g = set(range(first, last + 1))
        p = set(range(lo, hi))
        ov = len(g & p)
        if ov:
            best = max(best, 2 * ov / (len(g) + len(p)))
        em |= g == p
    return best, em, bool(gold[lo:hi].any())


CONFIGS = [(m, o, k, w) for m in ("head", "vote", "sum", "segment") for o in (0, 1)
           for k in (KS if m != "head" else (1,)) for w in (WINDOWS if m == "vote" else (0,))]


def cv_readings(rows):
    """Every configuration's 5-fold token hit, on present questions."""
    present = [r for r in rows if not r["absent"] and r["x"]["gold"].any()]
    out = {}
    for m, o, k, w in CONFIGS:
        hits, by = 0, {}
        for fold in range(score.FOLDS):
            train = [r for r in present if score.fold_of(r) != fold]
            test = [r for r in present if score.fold_of(r) == fold]
            heads = best_heads(train, o, k)
            for r in test:
                key, _ = answer_key(r, heads, m, o, w)
                g = bool(r["x"]["gold"][key])
                hits += g
                for kk in ("all", r["family"], r["split"], f"tier{r['tier']}"):
                    by.setdefault(kk, []).append(g)
        name = f"{m} {'lift' if o else 'q'} K={k}" + (f" w={w}" if m == "vote" else "")
        out[name] = {"method": m, "ordinate": o, "k": k, "w": w, "hits": int(hits), "of": len(present),
                     "hit": 100.0 * hits / len(present),
                     "by": {kk: {"n": len(v), "hit": 100 * float(np.mean(v))} for kk, v in sorted(by.items())}}
    return out


def order(item):
    name, c = item
    return (-c["hits"], c["k"], -c["ordinate"], name)


def config_hits(rows, config, folds, fold_of):
    m, o, k, w = config
    got = []
    for fold in range(folds):
        train = [r for r in rows if fold_of(r) != fold]
        test = [r for r in rows if fold_of(r) == fold]
        heads = best_heads(train, o, k)
        got += [(r, bool(r["x"]["gold"][answer_key(r, heads, m, o, w)[0]])) for r in test]
    return got


def nested(rows):
    """The whole choice under nested CV: in each outer fold the configuration
    with the most inner-CV hits (3 folds) is applied to the held-out fold."""
    present = [r for r in rows if not r["absent"] and r["x"]["gold"].any()]
    hits, by, chosen = 0, {}, []
    for fold in range(score.FOLDS):
        train = [r for r in present if score.fold_of(r) != fold]
        test = [r for r in present if score.fold_of(r) == fold]
        inner = {c: sum(g for _, g in config_hits(train, c, 3, lambda r: score.fold_of(r) % 3)) for c in CONFIGS}
        best = sorted(inner.items(), key=lambda kv: (-kv[1], kv[0][2], -kv[0][1], kv[0][0]))[0][0]
        chosen.append(list(best))
        m, o, k, w = best
        heads = best_heads(train, o, k)
        for r in test:
            g = bool(r["x"]["gold"][answer_key(r, heads, m, o, w)[0]])
            hits += g
            for kk in ("all", r["family"], r["split"]):
                by.setdefault(kk, []).append(g)
    return {"hits": int(hits), "of": len(present), "hit": 100.0 * hits / len(present), "chosen": chosen,
            "by": {kk: 100 * float(np.mean(v)) for kk, v in sorted(by.items())}}


def span_choice(rows, cv):
    """The best configuration by CV token hit (ties: fewer heads, the lift
    first) and its span, `delta` reported per value."""
    name, c = sorted(cv.items(), key=order)[0]
    present = [r for r in rows if not r["absent"] and r["x"]["gold"].any()]
    per_delta = {}
    for delta in DELTAS:
        f1s, ems, by = [], [], {}
        for fold in range(score.FOLDS):
            train = [r for r in present if score.fold_of(r) != fold]
            test = [r for r in present if score.fold_of(r) == fold]
            heads = best_heads(train, c["ordinate"], max(c["k"], 1))
            for r in test:
                key, _ = answer_key(r, heads, c["method"], c["ordinate"], c["w"])
                lo, hi = grow(r, heads, c["ordinate"], key, delta)
                f1, em, _ = span_f1(r, lo, hi)
                f1s.append(f1)
                ems.append(em)
                for kk in ("all", r["family"], r["split"]):
                    by.setdefault(kk, []).append(f1)
        per_delta[delta] = {"f1": 100 * float(np.mean(f1s)), "em": 100 * float(np.mean(ems)),
                            "by": {kk: 100 * float(np.mean(v)) for kk, v in sorted(by.items())}}
    return {"reading": name, "config": c, "span_by_delta": per_delta}


def confidence_auc(rows, c):
    pos, neg = [], []
    for fold in range(score.FOLDS):
        train = [r for r in rows if not r["absent"] and r["x"]["gold"].any() and score.fold_of(r) != fold]
        heads = best_heads(train, c["ordinate"], max(c["k"], 1))
        for r in rows:
            if score.fold_of(r) != fold:
                continue
            _, conf = answer_key(r, heads, c["method"], c["ordinate"], c["w"])
            (neg if r["absent"] else pos).append(conf)
    return score.auc(pos, neg)


def generation_on_keys(rows):
    """The generation route's first answer on the key grid."""
    out, by = [], {}
    for r in rows:
        g = r["gen"]
        if g is None or r["absent"] or not r["x"]["gold"].any():
            continue
        found = g["found"][0] if g["found"] else None
        if not found or found["match"] == "none":
            res = (0.0, False, False)
        else:
            mask, each = span_keys(r["state"], found["spans"][:1], r["key_bytes"])
            keys = each[0] if each else []
            res = span_f1(r, int(keys[0]), int(keys[-1]) + 1) if len(keys) else (0.0, False, False)
        out.append(res)
        for kk in ("all", r["family"], r["split"]):
            by.setdefault(kk, []).append(res)
    def summ(v):
        return {"n": len(v), "hit": 100 * float(np.mean([x[2] for x in v])),
                "f1": 100 * float(np.mean([x[0] for x in v])), "em": 100 * float(np.mean([x[1] for x in v]))}
    return {kk: summ(v) for kk, v in sorted(by.items())}


def regions_table(rows, heads):
    """Mean softmax mass by region for `heads`, lexical against paraphrase,
    with and without the instruction."""
    out = {}
    for split in ("lexical", "paraphrase"):
        sub = [r for r in rows if r["split"] == split and not r["absent"]]
        if not sub:
            continue
        mq = np.stack([r["x"]["mass_q"][heads] for r in sub]).mean(axis=(0, 1))
        mn = np.stack([r["x"]["mass_na"][heads] for r in sub]).mean(axis=(0, 1))
        gold = float(np.mean([r["x"]["mass_gold"][heads].mean() for r in sub]))
        out[split] = {"q": dict(zip(REGIONS, np.round(mq, 4).tolist())),
                      "na": dict(zip(REGIONS, np.round(mn, 4).tolist())), "gold": round(gold, 4), "n": len(sub)}
        rare = [r["x"]["mass_rare_instruction"][heads].mean() for r in sub if "mass_rare_instruction" in r["x"]]
        if rare:
            out[split]["rare_instruction"] = {"n": len(rare), "mass": round(float(np.mean(rare)), 4)}
    return out


def forced_label(p, n):
    """A forced query by its place: the scaffold's last token (p = 0), the
    quote's last token (p = n, about to close the quote), its first (p = 1),
    or one in the middle."""
    if p == 0:
        return "scaffold"
    if p == n:
        return "last"
    return "first" if p == 1 else "middle"


def forced_table(rows):
    """Per forced query place, the heads whose argmax key lands most often on
    the gold's first key, anywhere on the gold, on its last key, and on the
    two keys just after it."""
    out = {}
    for name in ("scaffold", "first", "middle", "last"):
        counts = {c: np.zeros(H) for c in ("gold_first", "gold_any", "gold_last", "after_last")}
        n = 0
        for r in rows:
            fa = r["x"].get("forced_argmax")
            if fa is None or not len(r["x"]["gold_first"]):
                continue
            for a, p in zip(fa, r["x"]["forced_queries"]):
                if forced_label(int(p), r["x"]["forced_n"]) != name:
                    continue
                n += 1
                first, last = r["x"]["gold_first"][0], r["x"]["gold_last"][0]
                counts["gold_first"] += a == first
                counts["gold_any"] += r["x"]["gold"][a]
                counts["gold_last"] += a == last
                counts["after_last"] += (a > last) & (a <= last + 2)
        out[name] = {"n": n, **{c: profiles.head_table(v / max(1, n), top=5) for c, v in counts.items()}}
    return out


MULTI = ("logmulti", "hotfacts")
MULTI_KS = (8, 16, 32)
MULTI_BARS = (0.1, 0.2, 0.3, 0.4, 0.5)


def line_hits(rows):
    """[Q, H]: each head's best segment (by share lift) is a gold segment."""
    return np.stack([np.isin(r["x"]["seg_top"][:, 0], r["x"]["gold_segments"]) for r in rows])


def multi_answer(r, heads, bar):
    """The segments named by at least `bar` of the heads' best segments:
    several, one, or none."""
    votes = r["x"]["seg_top"][heads, 0]
    vals, counts = np.unique(votes, return_counts=True)
    return set(vals[counts >= bar * len(heads)].tolist())


def set_scores(pred, gold):
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, pred == gold


def multi_table(rows):
    """Several golds or none (Q7): the heads with the most line hits on the
    training folds' single-gold questions of every family vote their best
    segment, and every segment with at least `bar` of the votes is an
    answer. K and the bar are chosen by CV on the several-gold families'
    set F1; the empty answer is how an absent question is read."""
    multi = [r for r in rows if r["family"] in MULTI]
    single = [r for r in rows if r["family"] not in MULTI and not r["absent"] and len(r["x"]["gold_segments"])]
    out = {}
    for k in MULTI_KS:
        for bar in MULTI_BARS:
            f1s, exact, empty_right, by = [], [], [], {}
            for fold in range(score.FOLDS):
                train = [r for r in single if score.fold_of(r) != fold]
                heads = np.argsort(-line_hits(train).sum(axis=0), kind="stable")[:k]
                for r in (r for r in multi if score.fold_of(r) == fold):
                    gold = set(r["x"]["gold_segments"].tolist())
                    p, rc, f1, ex = set_scores(multi_answer(r, heads, bar), gold)
                    f1s.append(f1)
                    exact.append(ex)
                    if not gold:
                        empty_right.append(ex)
                    by.setdefault(r["family"], []).append(f1)
            out[f"K={k} bar={bar}"] = {"f1": 100 * float(np.mean(f1s)), "exact": 100 * float(np.mean(exact)),
                                       "absent_right": 100 * float(np.mean(empty_right)) if empty_right else None,
                                       "by": {f: 100 * float(np.mean(v)) for f, v in sorted(by.items())}}
    return out


def report(set_paths, dumps, manifests, generations, out_path):
    rows = load_all(set_paths, dumps, manifests, generations or None)
    print(f"{len(rows)} questions, {sum(not r['absent'] for r in rows)} present", flush=True)
    rep = {"n": len(rows)}
    cv = cv_readings(rows)
    rep["cv"] = cv
    rep["nested"] = nested(rows)
    rep["choice"] = span_choice(rows, cv)
    rep["confidence_auc"] = confidence_auc(rows, rep["choice"]["config"])
    if generations:
        rep["generation"] = generation_on_keys(rows)
    vote_heads = best_heads([r for r in rows if not r["absent"] and r["x"]["gold"].any()], 1, 8)
    rep["regions"] = {"vote_heads": [score.head_name(h) for h in vote_heads],
                      "table": regions_table(rows, vote_heads),
                      "all_heads": regions_table(rows, list(range(H)))}
    rep["forced"] = forced_table(rows)
    rep["multi"] = multi_table(rows)
    write_json(out_path, rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dump", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int)
    e.add_argument("--tokenizer", help="the artifact's tokenizer.json, to find the instruction's rare-word tokens")
    r = sub.add_parser("report")
    r.add_argument("--sets", nargs="+", required=True)
    r.add_argument("--dumps", nargs="+", required=True)
    r.add_argument("--manifests", nargs="+", required=True)
    r.add_argument("--generation", nargs="*", default=[], help="generation.py's file per set, in --sets order")
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.dump, args.manifest, args.out, args.limit, args.tokenizer)
    else:
        report(args.sets, args.dumps, args.manifests, args.generation, args.out)


if __name__ == "__main__":
    main()
