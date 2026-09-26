"""Read spec 18's dumps as profiles: spec 19 phase 0 (GitHub #276).

Spec 19 (`docs/specs/decide/19-a-span-read-from-attention.md`) treats every
signal one prefill gives per state token as a *profile* of the input and asks
which are worth reading. Phase 0 is what spec 18's dumps can already say, with
no GPU: every GQA head's pre-softmax scores at the scaffold's last token over
the state's keys, for the index scaffold (`s1`), the copy scaffold (`s2`), and
each with the content-free instruction (`-na`).

Two steps:

- `extract` walks one set's dump once and keeps, per question and head, what
  the report needs from the raw scores -- a few tens of MB instead of 3.5 GB:
  the argmax key of six ordinates, their token-level AUROC and average
  precision for "key inside a gold segment", the span's log-normaliser, the
  best score on the target and on the whole span, and the event-related
  windows around the target's first and last key. Given the tokenizer, it
  also finds the target's *rare* keys (the tokens of the word only the target
  shares with a lexical question, `common.rare_shared`) and keeps the rank
  each head gives them.
- `report` reads the development sets' extracts beside `score.py`'s
  per-segment features and the manifests, and writes phase 0's tables:
  ordinate quality, where each head's peak key sits relative to the target,
  span-aligned averages, line offsets, head combinations, the top-k ceiling
  and where the lexical misses land.

The six ordinates, per head:

- `s1`, `s2`: the attention weight (a head's scores order its keys the way
  its weights do);
- `lift1`, `lift2`: `log a_q - log a_NA`, what the instruction adds to the
  content-free prefill's weight on the same key;
- `na1`, `na2`: the content-free weight alone -- the prior every question
  over that state shares, the control the others are read against.

    python profiles.py extract --dump <A-hq.json> --manifest <A/manifest.json> --tokenizer <tokenizer.json> --out <dir>
    python profiles.py report --sets <dir>/A <dir>/B --dumps <A-hq.json> <B-hq.json> --manifests <A/manifest.json> <B/manifest.json> --out phase0.json

Needs NumPy and SciPy (and `tokenizers` for the rare keys).
"""

import argparse
import json
import os
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import score
from common import content_stems, rare_shared, stem

H = score.N_HEADS
ORDS = ("s1", "s2", "lift1", "lift2", "na1", "na2")
ERA_ORDS = ("s1", "s2", "lift1", "lift2")     # log(a * N) for s*, the lift for lift*
WINDOW = 16
L39_H10 = score.POINTING
L39_H12 = 9 * score.HEADS + 12
OFFSETS = tuple(range(-3, 4))
CATEGORIES = ("first", "inside", "last", "sep_after", "next_first", "next_other",
              "sep_before", "prev_last", "prev_other", "elsewhere")
LAMBDAS = (0.01, 0.1, 1.0)
INNER_FOLDS = 3
ANCHOR_KS = (8, 16, 32)
TOPK = (1, 2, 3, 5)
WORD_SPAN = re.compile(r"[a-z0-9]+")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(value, f, indent=1)


# ---------------------------------------------------------------------------
# The evidence as the harness writes it (support/locate.rs), for the rare keys
# ---------------------------------------------------------------------------

def evidence(state):
    """`{"evidence": state}` and each segment's character range in it, as
    `support/locate.rs` writes them: a string's lines without the `\\n`
    escapes, an array's elements without commas (and a string element
    without its quotes). Returns (text, [(start, end, owns)])."""
    text = '{"evidence":'
    segments = []
    if isinstance(state, str):
        text += '"'
        for i, line in enumerate(state.split("\n")):
            if i:
                text += "\\n"
            start = len(text)
            text += json.dumps(line, ensure_ascii=False)[1:-1]
            segments.append((start, len(text), bool(line.strip())))
        text += '"'
    else:
        text += "["
        for i, item in enumerate(state):
            if i:
                text += ","
            start = len(text)
            text += json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            if isinstance(item, str):
                segments.append((start + 1, len(text) - 1, bool(item.strip())))
            else:
                segments.append((start, len(text), True))
        text += "]"
    return text + "}", segments


def segment_texts(state):
    """Each segment's plain text: a line, or an element's JSON."""
    if isinstance(state, str):
        return state.split("\n")
    return [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in state]


def map_keys(offsets, segments, base):
    """`support/locate.rs`'s owners, key_span and segment_keys over one
    tokenisation: (span [start, length], keys per segment or None). Tokens
    and segments are both in order, so each token starts from the first
    segment the previous one could still reach."""
    owners, first_seg = [], 0
    for a, b in offsets:
        while first_seg < len(segments) and segments[first_seg][1] + base <= a:
            first_seg += 1
        best = None
        for j in range(first_seg, len(segments)):
            sa, sb, owns = segments[j]
            sa, sb = sa + base, sb + base
            if sa >= b:
                break
            if not owns:
                continue
            overlap = min(b, sb) - max(a, sa)
            if overlap > 0 and (best is None or overlap > best[1]):
                best = (j, overlap)
        owners.append(best[0] if best else None)
    owned = [i for i, o in enumerate(owners) if o is not None]
    first, last = owned[0], owned[-1]
    keys = [None] * len(segments)
    for p in range(first, last + 1):
        o = owners[p]
        if o is None:
            continue
        at = p - first
        keys[o] = [at, at + 1] if keys[o] is None else [keys[o][0], at + 1]
    return [first, last - first + 1], keys


class RareKeys:
    """The target's rare keys: the state's tokens that carry a word the
    question shares with the target and with no other segment. Found by
    tokenizing the evidence again and checked against the dump's own span and
    segment keys -- a question whose keys do not match gets none."""

    SYSTEM = "<|im_start|>system\n"

    def __init__(self, tokenizer_path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def find(self, question, row):
        """(rare key positions relative to the span, or None when the question
        is not lexical or its tokenisation does not match the dump)."""
        if row["absent"] or question["split"] != "lexical":
            return None
        text, segments = evidence(question["state"])
        # The render opens with the same system header and closes the
        # evidence with the end of the system turn, a special token that no
        # merge crosses: token i here is prompt position i.
        enc = self.tokenizer.encode(self.SYSTEM + text + "<|im_end|>", add_special_tokens=False)
        span, keys = map_keys(enc.offsets, segments, len(self.SYSTEM))
        if span != row["span"] or keys != row["keys"]:
            return None
        plain = segment_texts(question["state"])
        found = set()
        for t in row["targets"]:
            others = [p for j, p in enumerate(plain) if j != t]
            rare = rare_shared(question["instruction"], plain[t], others)
            sa, sb, _ = segments[t]
            base = len(self.SYSTEM) + sa
            for m in WORD_SPAN.finditer(text[sa:sb].lower()):
                if stem(m.group()) not in rare:
                    continue
                a, b = base + m.start(), base + m.end()
                found |= {i - span[0] for i, (ta, tb) in enumerate(enc.offsets) if ta < b and tb > a}
        return sorted(found) or None


# ---------------------------------------------------------------------------
# extract: one pass over a dump
# ---------------------------------------------------------------------------

def ordinate_values(block):
    """{variant: [H, N] scores} -> [6, H, N], each ordinate's values up to a
    per-head constant (which changes no order inside a question)."""
    return np.stack([block["s1"], block["s2"],
                     block["s1"] - block["s1-na"], block["s2"] - block["s2-na"],
                     block["s1-na"], block["s2-na"]])


def gold_mask(row, n):
    mask = np.zeros(n, dtype=bool)
    for t in row["targets"]:
        k = row["keys"][t]
        if k is not None:
            mask[k[0]:k[1]] = True
    return mask


def auroc_ap(y, pos, seed=0):
    """Per-row AUROC and average precision of the values `y` [R, N] for the
    labels `pos` [N], from one sort. The dump's scores are f16, so keys tie;
    a relative jitter of 1e-5 -- twenty times finer than f16's resolution,
    so it reorders no two distinct values -- breaks ties at random instead
    of in favour of the earlier key."""
    p = int(pos.sum())
    n = y.shape[1]
    u = np.random.default_rng(seed).uniform(-1.0, 1.0, size=y.shape).astype(np.float32)
    order = np.argsort(-(y + (np.abs(y) * 1e-5 + 1e-9) * u), axis=1)
    lab = pos[order]
    hits = np.cumsum(lab, axis=1)
    ap = (hits / np.arange(1, n + 1) * lab).sum(axis=1) / p
    # negatives ranked above each positive: its place less the positives above it
    above = ((np.arange(n)[None, :] - (hits - 1)) * lab).sum(axis=1)
    auc = 1.0 - above / (p * (n - p))
    return auc, ap


def windows(values, at, n):
    """values [R, H, N] -> [R, H, 2W+1] at positions at-W..at+W (NaN outside)."""
    out = np.full(values.shape[:2] + (2 * WINDOW + 1,), np.nan, dtype=np.float32)
    lo, hi = max(0, at - WINDOW), min(n, at + WINDOW + 1)
    out[:, :, lo - (at - WINDOW):hi - (at - WINDOW)] = values[:, :, lo:hi]
    return out


def question_stats(block, row, rare=None):
    """Everything `report` needs from one question's raw scores."""
    n = row["span"][1]
    logz = np.stack([logsumexp(block[v], axis=1) for v in score.VARIANTS])       # [4, H]
    smax = np.stack([block[v].max(axis=1) for v in score.VARIANTS])
    vals = ordinate_values(block)                                                 # [6, H, N]
    out = {
        "argmax": vals.argmax(axis=2).astype(np.int32),
        "logz": logz.astype(np.float32), "smax": smax.astype(np.float32),
        "auc": np.full((len(ORDS), H), np.nan, dtype=np.float32),
        "ap": np.full((len(ORDS), H), np.nan, dtype=np.float32),
        "tmax": np.full((len(score.VARIANTS), H), np.nan, dtype=np.float32),
        "era": np.full((len(ERA_ORDS), 2, H, 2 * WINDOW + 1), np.nan, dtype=np.float16),
        "rare_rank": np.zeros((len(ORDS), H), dtype=np.int32),
        "rare_count": 0,
    }
    if row["absent"]:
        return out
    pos = gold_mask(row, n)
    if pos.any() and not pos.all():
        auc, ap = auroc_ap(vals.reshape(-1, n), pos)
        out["auc"] = auc.reshape(len(ORDS), H).astype(np.float32)
        out["ap"] = ap.reshape(len(ORDS), H).astype(np.float32)
        out["tmax"] = np.stack([block[v][:, pos].max(axis=1) for v in score.VARIANTS]).astype(np.float32)
    # log(a * N) and the lift, for the span-aligned averages.
    loga = {v: block[v] - logz[i][:, None] for i, v in enumerate(score.VARIANTS)}
    era_vals = np.stack([loga["s1"] + np.log(n), loga["s2"] + np.log(n),
                         loga["s1"] - loga["s1-na"], loga["s2"] - loga["s2-na"]])
    first = row["targets"][0]
    k = row["keys"][first]
    if k is not None:
        out["era"][:, 0] = windows(era_vals, k[0], n)
        out["era"][:, 1] = windows(era_vals, k[1] - 1, n)
    if rare:
        # The best rank (1 = the head's argmax) any rare key gets: one more
        # than the keys scored strictly above it.
        best = vals[:, :, rare].max(axis=2)
        out["rare_rank"] = (1 + (vals > best[:, :, None]).sum(axis=2)).astype(np.int32)
        out["rare_count"] = len(rare)
    return out


def extract(dump_path, manifest_path, tokenizer_path, out_dir, limit=None):
    meta = read_json(dump_path)
    base = os.path.dirname(dump_path)
    with open(os.path.join(base, meta["rows_file"]), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    questions = {q["id"]: q for q in read_json(manifest_path)["questions"]} if manifest_path else {}
    rare_keys = RareKeys(tokenizer_path) if tokenizer_path else None
    raw = np.memmap(os.path.join(base, meta["bin_file"]), dtype="<f2", mode="r")
    rows = rows[:limit] if limit else rows
    stats, matched = [], 0
    for i, row in enumerate(rows):
        span = row["span"][1]
        block = {}
        for v in score.VARIANTS:
            off = row["variants"][v]["offset"]
            block[v] = np.asarray(raw[off:off + H * span], dtype=np.float32).reshape(H, span)
        rare = None
        if rare_keys and row["id"] in questions:
            q = dict(questions[row["id"]], split=row["split"])
            rare = rare_keys.find(q, row)
            matched += rare is not None
        stats.append(question_stats(block, row, rare))
        if (i + 1) % 20 == 0:
            print(f"{meta['set']}: {i + 1}/{len(rows)}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    arrays = {key: np.stack([s[key] for s in stats]) for key in stats[0] if key != "rare_count"}
    arrays["rare_count"] = np.array([s["rare_count"] for s in stats], dtype=np.int32)
    np.savez_compressed(os.path.join(out_dir, f"{meta['set']}.npz"), ids=np.array([r["id"] for r in rows]), **arrays)
    print(f"{meta['set']}: {len(rows)} questions, rare keys found on {matched}")


# ---------------------------------------------------------------------------
# report: the tables
# ---------------------------------------------------------------------------

def load_all(set_dirs, dumps, manifests):
    """The development questions: `score.load`'s rows (per-segment features)
    with the extract's arrays and the manifest's texts attached."""
    rows = []
    for extract_path, dump, manifest in zip(set_dirs, dumps, manifests):
        _, part = score.load(dump)
        with np.load(extract_path + ".npz") as npz:
            stored = {key: npz[key] for key in npz.files}
        index = {q: i for i, q in enumerate(stored["ids"])}
        texts = {q["id"]: q for q in read_json(manifest)["questions"]}
        for row in part:
            i = index[row["id"]]
            row["x"] = {key: value[i] for key, value in stored.items() if key != "ids"}
            q = texts[row["id"]]
            row["instruction"] = q["instruction"]
            row["texts"] = segment_texts(q["state"])
            rows.append(row)
    return rows


def split_keys(row):
    return (row["family"], row["split"], f"tier{score.tier(row)}")


def tally(by, row, value):
    for key in ("all",) + split_keys(row):
        by.setdefault(key, []).append(value)


def summarise(by, fn=np.mean):
    return {k: {"n": len(v), "value": float(fn(v))} for k, v in sorted(by.items())}


def category(key, row):
    """Where a key sits relative to the (first) target: its first key, inside,
    its last key, the separator after it, the next segment, ..."""
    t = row["targets"][0]
    a, b = row["keys"][t]
    if key == a:
        return "first"
    if key == b - 1:
        return "last"
    if a < key < b - 1:
        return "inside"
    nxt = next((row["keys"][j] for j in range(t + 1, len(row["keys"])) if row["keys"][j]), None)
    prv = next((row["keys"][j] for j in range(t - 1, -1, -1) if row["keys"][j]), None)
    if nxt and b <= key < nxt[0]:
        return "sep_after"
    if nxt and key == nxt[0]:
        return "next_first"
    if nxt and nxt[0] < key < nxt[1]:
        return "next_other"
    if prv and prv[1] <= key < a:
        return "sep_before"
    if prv and key == prv[1] - 1:
        return "prev_last"
    if prv and prv[0] <= key < prv[1] - 1:
        return "prev_other"
    return "elsewhere"


def head_table(values, heads=None, top=10, reverse=True):
    """The `top` heads by `values` [H] (NaN last), as (name, value)."""
    v = np.where(np.isfinite(values), values, -np.inf if reverse else np.inf)
    order = np.argsort(-v if reverse else v, kind="stable")[:top]
    return [(score.head_name(int(h)), round(float(values[h]), 4)) for h in order]


def quality(present):
    """Ordinate quality at token level: per ordinate and head, the mean AUROC
    and AP of 'key inside a gold segment', the token hit (argmax inside), by
    split / family / tier; and the token hit of one head chosen per ordinate
    in cross-validation."""
    out = {}
    for o, name in enumerate(ORDS):
        auc = np.stack([r["x"]["auc"][o] for r in present])
        ap = np.stack([r["x"]["ap"][o] for r in present])
        hit = np.stack([gold_mask(r, r["span"][1])[r["x"]["argmax"][o]] for r in present])
        best_ap = int(np.nanargmax(np.nanmean(ap, axis=0)))
        best_hit = int(np.argmax(hit.mean(axis=0)))
        by = {}
        for r, a, p, h in zip(present, auc, ap, hit):
            tally(by, r, (a[best_ap], p[best_ap], h[best_hit]))
        cv_hits, cv_by = 0, {}
        for fold in range(score.FOLDS):
            train = [i for i, r in enumerate(present) if score.fold_of(r) != fold]
            test = [i for i, r in enumerate(present) if score.fold_of(r) == fold]
            h = int(np.argmax(hit[train].mean(axis=0)))
            for i in test:
                cv_hits += hit[i, h]
                tally(cv_by, present[i], float(hit[i, h]))
        out[name] = {
            "auc_top": head_table(np.nanmean(auc, axis=0)),
            "ap_top": head_table(np.nanmean(ap, axis=0)),
            "hit_top": head_table(hit.mean(axis=0)),
            "best_ap_head": score.head_name(best_ap), "best_hit_head": score.head_name(best_hit),
            "by": {k: {"n": len(v), "auc": float(np.mean([x[0] for x in v])),
                       "ap": float(np.mean([x[1] for x in v])), "hit": float(np.mean([x[2] for x in v]))}
                   for k, v in sorted(by.items())},
            "cv_token_hit": {"hits": int(cv_hits), "of": len(present),
                             "by": {k: {"n": len(v), "hit": float(np.mean(v))} for k, v in sorted(cv_by.items())}},
            "l39_h10": {"auc": float(np.nanmean(auc[:, L39_H10])), "ap": float(np.nanmean(ap[:, L39_H10])),
                        "hit": float(hit[:, L39_H10].mean())},
            "l39_h12": {"auc": float(np.nanmean(auc[:, L39_H12])), "ap": float(np.nanmean(ap[:, L39_H12])),
                        "hit": float(hit[:, L39_H12].mean())},
        }
    return out


def peak_places(present):
    """Where each head's argmax key sits relative to the target (logs and
    records: one target each), per ordinate: the share of questions in each
    category, and the heads most consistent in each."""
    single = [r for r in present if r["family"] in ("logs", "records") and len(r["targets"]) == 1]
    out = {}
    for o, name in enumerate(ORDS[:4]):
        counts = np.zeros((H, len(CATEGORIES)))
        for r in single:
            for h, key in enumerate(r["x"]["argmax"][o]):
                counts[h, CATEGORIES.index(category(int(key), r))] += 1
        share = counts / max(1, len(single))
        out[name] = {
            "n": len(single),
            "top": {c: head_table(share[:, i], top=6) for i, c in enumerate(CATEGORIES)},
            "l39_h10": dict(zip(CATEGORIES, np.round(share[L39_H10], 3).tolist())),
            "l39_h12": dict(zip(CATEGORIES, np.round(share[L39_H12], 3).tolist())),
        }
        for fam in ("logs", "records"):
            sub = [r for r in single if r["family"] == fam]
            c = np.zeros((H, len(CATEGORIES)))
            for r in sub:
                for h, key in enumerate(r["x"]["argmax"][o]):
                    c[h, CATEGORIES.index(category(int(key), r))] += 1
            c /= max(1, len(sub))
            out[name][fam] = {cat: head_table(c[:, i], top=4) for i, cat in enumerate(CATEGORIES)
                              if cat in ("first", "last", "sep_after", "next_first")}
            out[name][fam]["l39_h10"] = dict(zip(CATEGORIES, np.round(c[L39_H10], 3).tolist()))
    return out


def span_aligned(present):
    """Event-related averages: per ordinate and head, the mean profile
    around the target's first key and its last key (-W..+W), over logs and
    records; the heads whose start-aligned average peaks at the first key
    (initiator candidates) and whose end-aligned average peaks after the last
    (terminator candidates)."""
    out = {}
    for fams in (("logs", "records"), ("prose",)):
        sub = [r for r in present if r["family"] in fams]
        if not sub:
            continue
        era = np.stack([r["x"]["era"].astype(np.float32) for r in sub])        # [Q, 4, 2, H, 33]
        mean = np.nanmean(era, axis=0)                                           # [4, 2, H, 33]
        block = {}
        for o, name in enumerate(ERA_ORDS):
            start, end = mean[o, 0], mean[o, 1]
            # contrast: the value at an offset minus the window's median
            def contrast(w, at):
                return w[:, WINDOW + at] - np.nanmedian(w, axis=1)
            block[name] = {
                "initiators": head_table(contrast(start, 0)),
                "before_start": head_table(contrast(start, -1)),
                "enders": head_table(contrast(end, 0)),
                "terminators": head_table(np.nanmax(np.stack([contrast(end, d) for d in (1, 2, 3)]), axis=0)),
                "start_peak_offset": {score.head_name(h): int(np.nanargmax(start[h]) - WINDOW)
                                      for h in (L39_H10, L39_H12)},
                "end_peak_offset": {score.head_name(h): int(np.nanargmax(end[h]) - WINDOW)
                                    for h in (L39_H10, L39_H12)},
                "l39_h10_start": np.round(start[L39_H10], 2).tolist(),
                "l39_h10_end": np.round(end[L39_H10], 2).tolist(),
            }
        out["+".join(fams)] = {"n": len(sub), "ordinates": block, "_mean": mean}
    return out


# --- line level ------------------------------------------------------------

def seg_scores(row, scaffold, baseline):
    """[H, segments] per-segment scores of every head (score.py's R1 shares,
    minus the -na shares with a baseline), -inf where no key is owned."""
    s = row["feat"][scaffold][0].astype(np.float64)
    if baseline:
        s = s - row["feat"][scaffold + "-na"][0]
    return np.where(score.owning(row)[None, :], s, -np.inf)


def shifted_winner(s, d):
    """The segment a head's reading names when its winner sits `d` segments
    after the target: winner - d, clipped into the state."""
    return np.clip(s.argmax(axis=1) - d, 0, s.shape[1] - 1)


def line_offsets(present):
    """Per head and scaffold: the distribution of (winner - target) on logs
    and records, the heads with a stable non-zero offset, and R1 with a
    per-head offset chosen in cross-validation."""
    single = [r for r in present if r["family"] in ("logs", "records") and len(r["targets"]) == 1]
    out = {}
    for scaffold in ("s1", "s2"):
        for baseline in (False, True):
            name = score.config_name("r1", scaffold, baseline)
            diffs = np.stack([seg_scores(r, scaffold, baseline).argmax(axis=1) - r["targets"][0] for r in single])
            stable = []
            for h in range(H):
                vals, counts = np.unique(diffs[:, h], return_counts=True)
                i = int(np.argmax(counts))
                if vals[i] != 0 and counts[i] >= 0.3 * len(single):
                    stable.append((score.head_name(h), int(vals[i]), round(counts[i] / len(single), 3)))
            stable.sort(key=lambda x: -x[2])
            # CV over every present question (prose too): (head, d) by training hits.
            hits_all = {}
            for r in present:
                s = seg_scores(r, scaffold, baseline)
                hits_all[id(r)] = np.stack([np.isin(shifted_winner(s, d), r["targets"]) for d in OFFSETS])
            cv, chosen, by = 0, [], {}
            for fold in range(score.FOLDS):
                train = [r for r in present if score.fold_of(r) != fold]
                test = [r for r in present if score.fold_of(r) == fold]
                total = sum(hits_all[id(r)] for r in train)                          # [offsets, H]
                d_i, h = np.unravel_index(int(np.argmax(total)), total.shape)
                chosen.append((score.head_name(int(h)), OFFSETS[d_i]))
                for r in test:
                    g = bool(hits_all[id(r)][d_i, h])
                    cv += g
                    tally(by, r, float(g))
            l39 = np.stack([np.isin(shifted_winner(seg_scores(r, scaffold, baseline)[[L39_H10]], d), r["targets"])[0]
                            for r in present for d in OFFSETS]).reshape(len(present), len(OFFSETS)).sum(axis=0)
            out[name] = {"stable_nonzero": stable[:12], "cv_hits": int(cv), "of": len(present),
                         "cv_top1": 100.0 * cv / len(present), "chosen": chosen,
                         "by": {k: {"n": len(v), "top1": 100 * float(np.mean(v))} for k, v in sorted(by.items())},
                         "l39_h10_by_offset": dict(zip(map(str, OFFSETS), l39.tolist()))}
    return out


def features(row, kind):
    """Per-segment feature rows [segments, d] for a combination over heads:
    `lift2` the S2 lift per segment, `lift12` both scaffolds' lifts,
    `logs` log shares of S2 and S2-na, `all` lifts and log shares of both."""
    eps = 1e-12
    def ls(v):
        return np.log(row["feat"][v][0].astype(np.float64).T + eps)
    lift1, lift2 = ls("s1") - ls("s1-na"), ls("s2") - ls("s2-na")
    if kind == "lift2":
        return lift2
    if kind == "lift12":
        return np.hstack([lift1, lift2])
    if kind == "logs":
        return np.hstack([ls("s2"), ls("s2-na")])
    return np.hstack([lift1, lift2, ls("s1"), ls("s2")])


class ConditionalLogit:
    """Per-head weights over a question's segments: p(segment) = softmax of
    w . x over the segments that own keys, fitted to maximise the probability
    of the gold segments (any of them), with an L2 penalty. AT2's shape at the
    line level."""

    def __init__(self, lam):
        self.lam = lam

    def fit(self, xs, golds):
        x = np.vstack(xs)
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0) + 1e-6
        x = ((x - self.mu) / self.sd).astype(np.float32)
        bounds = np.cumsum([0] + [len(a) for a in xs])
        gold = np.zeros(len(x), dtype=bool)
        for i, g in enumerate(golds):
            gold[bounds[i] + np.asarray(g, dtype=int)] = True
        starts = bounds[:-1]
        n = len(xs)

        def loss(w):
            z = (x @ w.astype(np.float32)).astype(np.float64)
            m = np.maximum.reduceat(z, starts)
            e = np.exp(z - np.repeat(m, np.diff(bounds)))
            tot = np.add.reduceat(e, starts)
            eg = np.where(gold, e, 0.0)
            gtot = np.add.reduceat(eg, starts)
            val = np.sum(np.log(tot) - np.log(gtot)) / n + self.lam * w @ w / 2
            p = e / np.repeat(tot, np.diff(bounds))
            pg = eg / np.repeat(gtot, np.diff(bounds))
            grad = (x.T @ (p - pg).astype(np.float32)).astype(np.float64) / n + self.lam * w
            return val, grad

        res = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 300})
        self.w = res.x
        return self

    def scores(self, x):
        return ((x - self.mu) / self.sd) @ self.w


def question_xy(row, kind):
    own = np.flatnonzero(score.owning(row))
    x = features(row, kind)[own]
    gold = [int(np.flatnonzero(own == t)[0]) for t in row["targets"] if t in own]
    return own, x, gold


def fit_logit(train, kind, lam, cache):
    xs, golds = [], []
    for r in train:
        _, x, g = cache[(id(r), kind)]
        if g:
            xs.append(x)
            golds.append(g)
    return ConditionalLogit(lam).fit(xs, golds)


def rank_of_gold(scores, gold):
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return int(min(ranks[g] for g in gold))


def combinations(present, kinds=("lift2", "lift12", "all")):
    """5-fold CV of the conditional logit over every head (lambda by an inner
    3-fold CV on the training folds), per feature set; top-k recall."""
    cache = {}
    for r in present:
        for kind in kinds:
            cache[(id(r), kind)] = question_xy(r, kind)
    out = {}
    for kind in kinds:
        ranks, by, lams = {}, {}, []
        for fold in range(score.FOLDS):
            train = [r for r in present if score.fold_of(r) != fold]
            test = [r for r in present if score.fold_of(r) == fold]
            best = None
            for lam in LAMBDAS:
                inner = 0
                for f2 in range(INNER_FOLDS):
                    tr = [r for i, r in enumerate(train) if i % INNER_FOLDS != f2]
                    te = [r for i, r in enumerate(train) if i % INNER_FOLDS == f2]
                    model = fit_logit(tr, kind, lam, cache)
                    for r in te:
                        _, x, g = cache[(id(r), kind)]
                        inner += bool(g) and rank_of_gold(model.scores(x), g) == 1
                if best is None or inner > best[0]:
                    best = (inner, lam)
            lams.append(best[1])
            model = fit_logit(train, kind, best[1], cache)
            for r in test:
                _, x, g = cache[(id(r), kind)]
                rank = rank_of_gold(model.scores(x), g) if g else 10 ** 6
                ranks[r["id"] + r["set"]] = rank
                tally(by, r, float(rank == 1))
        allr = np.array(list(ranks.values()))
        out[kind] = {"cv_top1": 100.0 * float(np.mean(allr == 1)), "hits": int(np.sum(allr == 1)), "of": len(allr),
                     "lambdas": lams, "recall": {k: 100.0 * float(np.mean(allr <= k)) for k in TOPK},
                     "by": {k: {"n": len(v), "top1": 100 * float(np.mean(v))} for k, v in sorted(by.items())},
                     "ranks": ranks}
    return out


def r1_cv_ranks(present, scaffold="s2", baseline=True):
    """R1 as score.py chooses it (per fold), with the gold's rank per question."""
    ranks = {}
    for fold in range(score.FOLDS):
        train = [r for r in present if score.fold_of(r) != fold]
        test = [r for r in present if score.fold_of(r) == fold]
        method = score.select(train, "r1", scaffold, baseline)
        for r in test:
            s = score.reading(r, method, scaffold, baseline)
            ranks[r["id"] + r["set"]] = (rank_of_gold(np.where(np.isfinite(s), s, -1e30), r["targets"]), method[1])
    return ranks


VOTE_KS = (1, 3, 5, 8, 16, 32)
VOTE_CONFIGS = [(sc, b, k) for sc in ("s1", "s2") for b in (False, True) for k in VOTE_KS]


def vote_answer(winners, heads):
    """The most-voted segment among `heads`' winners; a tie goes to the tied
    segment named by the best-ranked head (`heads` is best first)."""
    votes = winners[heads]
    vals, counts = np.unique(votes, return_counts=True)
    best = set(vals[counts == counts.max()].tolist())
    return int(next(v for v in votes if v in best))


def vote_heads(train, scaffold, baseline, k):
    """The K heads with the most training hits as R1 (ties: score.py's order,
    larger mean target share), best first."""
    hits = score.r1_hits(train, scaffold, baseline)
    mean = np.mean([score.target_share(r, scaffold) for r in train], axis=0)
    return np.lexsort((-mean, -hits))[:k]


def votes(present):
    """A head vote at the line level: the K heads with the most training
    hits each name their winner segment (per-segment shares, less the -na
    shares with a baseline) and the most-voted segment is the answer. The
    scaffold, the baseline and K are chosen on the training folds by an
    inner 4-fold CV (ties: fewer heads, no baseline, S1 -- score.py's
    order), so the outer 5-fold estimate is of the whole procedure; every
    configuration's plain 5-fold CV is reported beside it."""
    winners = {(id(r), sc, b): seg_scores(r, sc, b).argmax(axis=1)
               for r in present for sc in ("s1", "s2") for b in (False, True)}

    def cv(rows, config, folds, fold_of):
        sc, b, k = config
        hits = []
        for fold in range(folds):
            train = [r for r in rows if fold_of(r) != fold]
            test = [r for r in rows if fold_of(r) == fold]
            heads = vote_heads(train, sc, b, k)
            hits += [(r, vote_answer(winners[(id(r), sc, b)], heads) in r["targets"]) for r in test]
        return hits

    def order(item):
        (sc, b, k), n = item
        return (-n, k, b, sc != "s1")

    plain = {}
    for config in VOTE_CONFIGS:
        got = cv(present, config, score.FOLDS, score.fold_of)
        by = {}
        for r, g in got:
            tally(by, r, float(g))
        plain[f"{config[0]}{' +base' if config[1] else ''} K={config[2]}"] = {
            "hits": sum(g for _, g in got), "of": len(got),
            "top1": 100.0 * sum(g for _, g in got) / len(got),
            "by": {kk: {"n": len(v), "top1": 100 * float(np.mean(v))} for kk, v in sorted(by.items())}}
    nested, chosen, by = 0, [], {}
    for fold in range(score.FOLDS):
        train = [r for r in present if score.fold_of(r) != fold]
        test = [r for r in present if score.fold_of(r) == fold]
        inner = {c: sum(g for _, g in cv(train, c, 4, lambda r: int(score.fold_of(r) * 7 + len(r["id"])) % 4))
                 for c in VOTE_CONFIGS}
        best = sorted(inner.items(), key=order)[0][0]
        chosen.append(list(best))
        heads = vote_heads(train, *best)
        for r in test:
            g = vote_answer(winners[(id(r), best[0], best[1])], heads) in r["targets"]
            nested += g
            tally(by, r, float(g))
    # The reading fitted on every development question, and its heads.
    inner = {c: sum(g for _, g in cv(present, c, score.FOLDS, score.fold_of)) for c in VOTE_CONFIGS}
    final = sorted(inner.items(), key=order)[0][0]
    return {"nested_top1": 100.0 * nested / len(present), "nested_hits": int(nested), "of": len(present),
            "nested_by": {kk: {"n": len(v), "top1": 100 * float(np.mean(v))} for kk, v in sorted(by.items())},
            "chosen_per_fold": chosen, "plain": plain,
            "final": {"scaffold": final[0], "baseline": final[1], "k": final[2],
                      "heads": [score.head_name(int(h)) for h in vote_heads(present, *final)],
                      "head_ids": [int(h) for h in vote_heads(present, *final)]}}


def anchored(present, r1ranks):
    """Spec 14's anchored set in one dimension, at the line level: the K
    heads with the most training hits vote their winner segment (S2, with the
    baseline); votes further than twice the median distance from the anchor
    head's winner are dropped; the answer is the most-voted segment left
    (ties to the anchor's)."""
    out = {}
    for k in ANCHOR_KS:
        hits, by = 0, {}
        for fold in range(score.FOLDS):
            train = [r for r in present if score.fold_of(r) != fold]
            test = [r for r in present if score.fold_of(r) == fold]
            order = np.argsort(-score.r1_hits(train, "s2", True), kind="stable")
            heads, anchor = order[:k], order[0]
            for r in test:
                w = seg_scores(r, "s2", True).argmax(axis=1)
                a = w[anchor]
                votes = w[heads]
                dist = np.abs(votes - a)
                keep = votes[dist <= 2 * max(1.0, float(np.median(dist)))]
                vals, counts = np.unique(keep, return_counts=True)
                best = vals[counts == counts.max()]
                answer = a if a in best else int(best[0])
                g = answer in r["targets"]
                hits += g
                tally(by, r, float(g))
        out[k] = {"cv_top1": 100.0 * hits / len(present), "hits": int(hits),
                  "by": {kk: {"n": len(v), "top1": 100 * float(np.mean(v))} for kk, v in sorted(by.items())}}
    return out


# --- the lexical misses ----------------------------------------------------

def lexical_misses(present, r1ranks):
    """Where the chosen reading's misses land, lexical against paraphrase:
    how far, how early, on a distractor, on a segment sharing a content word
    with the instruction, and whether the baseline moved the winner; and, on
    lexical questions, the rank each reading gives the target's rare keys."""
    out = {}
    for split in ("lexical", "paraphrase"):
        rows = [r for r in present if r["split"] == split]
        per = []
        for r in rows:
            rank, head = r1ranks[r["id"] + r["set"]]
            s = seg_scores(r, "s2", True)[head]
            w = int(np.argmax(s))
            w0 = int(np.argmax(seg_scores(r, "s2", False)[head]))
            nseg = len(r["keys"])
            instr = content_stems(r["instruction"])
            per.append({
                "family": r["family"], "hit": rank == 1, "rank": rank,
                "distance": min(abs(w - t) for t in r["targets"]),
                "position": w / max(1, nseg - 1), "target_position": r["targets"][0] / max(1, nseg - 1),
                "distractor": w in r.get("distractors", []),
                "shares_word": bool(content_stems(r["texts"][w]) & instr) if w < len(r["texts"]) else False,
                "target_shares_word": bool(content_stems(r["texts"][r["targets"][0]]) & instr),
                "baseline_moved": w != w0, "no_baseline_hit": w0 in r["targets"],
            })
        misses = [p for p in per if not p["hit"]]
        hitsl = [p for p in per if p["hit"]]
        def frac(sub, key):
            return round(float(np.mean([p[key] for p in sub])), 3) if sub else None
        out[split] = {
            "n": len(per), "hits": len(hitsl), "misses": len(misses),
            "miss_far_gt2": sum(p["distance"] > 2 for p in misses),
            "miss_position_median": frac(misses, "position") and float(np.median([p["position"] for p in misses])),
            "miss_in_first_decile": sum(p["position"] < 0.1 for p in misses),
            "target_position_median_of_misses": float(np.median([p["target_position"] for p in misses])) if misses else None,
            "miss_on_distractor": sum(p["distractor"] for p in misses),
            "miss_winner_shares_instruction_word": sum(p["shares_word"] for p in misses),
            "hit_rank_median_of_misses": float(np.median([p["rank"] for p in misses])) if misses else None,
            "baseline_moved_on_miss": sum(p["baseline_moved"] for p in misses),
            "no_baseline_would_hit": sum(p["no_baseline_hit"] for p in misses),
            "by_family": {f: {"n": sum(p["family"] == f for p in per), "misses": sum(p["family"] == f for p in misses),
                              "far": sum(p["family"] == f and p["distance"] > 2 for p in misses)}
                          for f in ("logs", "records", "prose")},
        }
    # The rare keys, on lexical questions whose tokenisation matched.
    lex = [r for r in present if r["split"] == "lexical" and r["x"]["rare_count"] > 0]
    rare = {}
    for o, name in enumerate(ORDS):
        rr = np.stack([r["x"]["rare_rank"][o] for r in lex]) if lex else np.zeros((0, H))
        rare[name] = {"n": len(lex),
                      "rare_is_argmax_top": head_table((rr == 1).mean(axis=0)) if lex else [],
                      "l39_h12_rank_median": float(np.median(rr[:, L39_H12])) if lex else None,
                      "l39_h12_rare_argmax": float((rr[:, L39_H12] == 1).mean()) if lex else None}
    # the span's normaliser: does the instruction move mass off the whole span?
    dz = {}
    for split in ("lexical", "paraphrase"):
        sub = [r for r in present if r["split"] == split]
        # logz rows: s1, s1-na, s2, s2-na
        d2 = np.stack([r["x"]["logz"][2] - r["x"]["logz"][3] for r in sub])
        t2 = np.stack([r["x"]["tmax"][2] - r["x"]["smax"][2] for r in sub])
        dz[split] = {"dlogz_s2_mean_l39h12": float(d2[:, L39_H12].mean()),
                     "dlogz_s2_mean_all_heads": float(d2.mean()),
                     "target_gap_s2_median_l39h12": float(np.median(t2[:, L39_H12]))}
    out["rare_keys"] = rare
    out["span_normaliser"] = dz
    return out


PARTS = ("quality", "peaks", "span_aligned", "offsets", "votes", "anchored", "combinations", "lexical")


def report(set_dirs, dumps, manifests, out_path, parts=PARTS):
    rows = load_all(set_dirs, dumps, manifests)
    present = [r for r in rows if not r["absent"]]
    print(f"{len(rows)} questions, {len(present)} present", flush=True)
    rep = {"n": len(present)}
    r1ranks = r1_cv_ranks(present)
    rr = np.array([v[0] for v in r1ranks.values()])
    rep["r1"] = {"cv_top1": 100 * float(np.mean(rr == 1)), "recall": {k: 100 * float(np.mean(rr <= k)) for k in TOPK},
                 "heads": sorted({score.head_name(v[1]) for v in r1ranks.values()})}
    steps = {
        "quality": lambda: quality(present),
        "peaks": lambda: peak_places(present),
        "span_aligned": lambda: span_aligned(present),
        "offsets": lambda: line_offsets(present),
        "votes": lambda: votes(present),
        "anchored": lambda: anchored(present, r1ranks),
        "combinations": lambda: combinations(present),
        "lexical": lambda: lexical_misses(present, r1ranks),
    }
    for part in parts:
        rep[part] = steps[part]()
        if part == "span_aligned":
            np.savez_compressed(os.path.splitext(out_path)[0] + ".era.npz",
                                **{k.replace("+", "_"): v.pop("_mean") for k, v in rep[part].items()})
        print(f"{part} done", flush=True)
    write_json(out_path, rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dump", required=True)
    e.add_argument("--manifest")
    e.add_argument("--tokenizer")
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int)
    r = sub.add_parser("report")
    r.add_argument("--sets", nargs="+", required=True, help="<out>/<set> of each extract (without .npz)")
    r.add_argument("--dumps", nargs="+", required=True)
    r.add_argument("--manifests", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--parts", default=",".join(PARTS), help="which tables, comma-separated")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.dump, args.manifest, args.tokenizer, args.out, args.limit)
    else:
        report(args.sets, args.dumps, args.manifests, args.out, args.parts.split(","))


if __name__ == "__main__":
    main()
