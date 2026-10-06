"""Spec flash-next/07 phase A: the MTP head's acceptance on the engine's own states, and the
pre-registered GO / NO-GO.

    python phase_a.py run --corpus <dir> --out <results.json>
    python phase_a.py project --alpha <results.json> --cost <dir>/cost.json

`run` reads what the engine's example `flash_next_mtp_phase_a corpus` wrote (per text: the
tokens, the tapped pre-mixer stacks, the engine's prefill pick after every position), fetches the
checkpoint's BF16 head into RAM by HTTP range at the converter's pinned revision (the `mtp.*`
tensors, the trunk's embed_tokens, lm_head and final mixer: ~7.8 GB, nothing written to disk
but the converter's header cache), and on the GPU:

1. checks the tap: the trunk's own mixer and lm_head over a tapped stack give the engine's pick;
2. holds the prototype to transformers' layer on the real weights (a causal pass and a chain);
3. scores every candidate convention on the same positions. Draft j at entry i (built from the
   stack at i and token i+1) is accepted when it equals the trunk's greedy token i+1+j. On the
   generated part of a text the trunk's greedy tokens are the text itself (the decode route's
   picks, what a verify accepts), so a chain fed the text is exact: alpha_j is the share of
   draft j accepted among the positions whose drafts 1..j-1 were. On the prompt part (held-out
   text) only alpha_1 is scored, against the engine's prefill pick.

`project` turns alpha_1..4 and the round costs into the projected speedups, the pre-registered
one-lane verdict (c = 3.8 ms, R1 = 15.3 ms, d = 0.9 ms: GO at >= 1.20x for some k <= 3) and the
re-projection with the measured round costs at 1, 2 and 3 lanes.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import mtp

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "..", "flash-next-converter")
HUB_CACHE = "F:/ai/models/Qwen3.8-Flash-Next-ignis/work/state/hub"
DRAFTS = 4

# The pre-registered projection (spec 07, Acceptance 1).
R1_MS, C_MS, D_MS, GO = 15.3, 3.8, 0.9, 1.20


# ---------------------------------------------------------------------------------- inputs
def fetch_head():
    """The BF16 head and the trunk tensors it needs, into RAM, plus the text config."""
    sys.path.insert(0, CONVERTER)
    from fetch import Source
    src = Source(HUB_CACHE)
    text = src._json_cached("config.json")["text_config"]
    began = time.time()
    weights = src.load("mtp.")
    trunk = {
        "embed": src.get("model.language_model.embed_tokens.weight"),
        "lm_head": src.get("lm_head.weight"),
        "mixer": src.load("model.language_model.hyper_connection_mixer."),
    }
    size = sum(t.numel() * t.element_size() for t in weights.values()) + sum(
        trunk[k].numel() * trunk[k].element_size() for k in ("embed", "lm_head"))
    print(f"fetched {size / 1e9:.2f} GB in {time.time() - began:.0f} s", flush=True)
    return text, weights, trunk


def texts(corpus):
    manifest = json.load(open(os.path.join(corpus, "manifest.json")))
    for t in manifest["texts"]:
        base = os.path.join(corpus, t["name"])
        tokens = np.fromfile(base + ".tokens.u32", np.uint32).astype(np.int64)
        stacks = np.fromfile(base + ".stacks.bf16", np.uint16).reshape(len(tokens), t["width"])
        picks = np.fromfile(base + ".argmax.u32", np.uint32).astype(np.int64)
        yield t, tokens, stacks, picks


def to_bf16(bits, device):
    return torch.from_numpy(bits.view(np.int16)).to(device).view(torch.bfloat16)


# ---------------------------------------------------------------------------------- scoring
def entries(head, stacks, tokens, comb, norm, chunk=4096):
    """The window's entries: entry i is built from the stack at i and token i+1."""
    parts = []
    for a in range(0, len(tokens) - 1, chunk):
        b = min(a + chunk, len(tokens) - 1)
        X = head.combine(stacks[a:b], tokens[a + 1:b + 1], comb, norm)
        parts.append(head.entries(X, torch.arange(a, b, device=head.device)))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def argmax(head, mixed):
    return head.logits(mixed).argmax(-1)


@torch.no_grad()
def score_text(head, tokens, stacks, picks, prompt, comb, norm, chains, mode, off, first, batch):
    """Counts of drafts accepted at entries i >= first. Returns
    {"prompt": [n, ok] (alpha_1 vs the prefill pick, held-out text),
     "gen_prefill": [n, ok] (alpha_1 on generated text vs the prefill pick),
     "gen": {chain: [[n_j, ok_j] for j in 1..4]}, "gen_ok1": per-position 0/1 of draft 1}."""
    dev = head.device
    T = len(tokens)
    tok = torch.from_numpy(tokens).to(dev)
    pick = torch.from_numpy(picks).to(dev)
    E = entries(head, stacks, tok, comb, norm)
    head.window(E, head.trunk_blocks(E, off) if mode == "index" else None, mode)
    idx_all = torch.arange(first, T - 2, device=dev)             # draft 1 at i targets token i+2
    d1, st1, mx1 = [], [], []
    for a in range(0, len(idx_all), batch):
        idx = idx_all[a:a + batch]
        stack, mixed = head.first_step(idx)
        d1.append(argmax(head, mixed))
        st1.append(stack)
        mx1.append(mixed)
    d1, st1, mx1 = torch.cat(d1), torch.cat(st1), torch.cat(mx1)
    gen = idx_all + 2 >= prompt                                   # the drafted token was generated
    out = {}
    held = ~gen
    out["prompt"] = [int(held.sum()), int((d1[held] == pick[idx_all[held] + 1]).sum())]
    out["gen_prefill"] = [int(gen.sum()), int((d1[gen] == pick[idx_all[gen] + 1]).sum())]
    ok1 = (d1 == tok[idx_all + 2]) & gen
    out["gen_ok1"] = ok1[gen].to(torch.uint8).cpu().numpy()
    out["gen"] = {}
    for chain in chains:
        counts = [[int(gen.sum()), int(ok1.sum())]]
        alive = ok1.clone()
        stack, mixed = st1, mx1
        caches = None
        sel = torch.arange(len(idx_all), device=dev)
        for j in range(2, DRAFTS + 1):
            keep = alive[sel] & (idx_all[sel] + 1 + j <= T - 1)
            sel, stack, mixed = sel[keep], stack[keep], mixed[keep]
            if caches is not None:
                caches = tuple(c[keep] for c in caches)
            if len(sel) == 0:
                counts.append([0, 0])
                continue
            i = idx_all[sel]
            new_stack, new_mixed, new_caches, ok = [], [], [], []
            for a in range(0, len(sel), batch):
                s = slice(a, a + batch)
                X = head.combine(head.chain_stack(stack[s], mixed[s], chain), tok[i[s] + j], comb, norm)
                prev = (None, None, None) if caches is None else tuple(c[s] for c in caches)
                st, mx, cache = head.chain_step(X, i[s], *prev)
                ok.append(argmax(head, mx) == tok[i[s] + 1 + j])
                new_stack.append(st)
                new_mixed.append(mx)
                new_caches.append(cache)
            ok = torch.cat(ok)
            counts.append([len(sel), int(ok.sum())])
            stack, mixed = torch.cat(new_stack), torch.cat(new_mixed)
            caches = tuple(torch.cat([c[k] for c in new_caches]) for k in range(3))
            alive = torch.zeros_like(alive)
            alive[sel] = ok
        out["gen"][chain] = counts
    return out


def wilson(n, k):
    if n == 0:
        return [0.0, 0.0, 0.0]
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]


# ---------------------------------------------------------------------------------- checks
@torch.no_grad()
def check_tap(head, trunk_mixer, stacks, picks, rows):
    """The trunk's mixer + lm_head over tapped stacks vs the engine's picks at `rows`."""
    agree, n, worst = 0, 0, []
    for a in range(0, len(rows), 256):
        r = rows[a:a + 256]
        S = to_bf16(stacks[r], head.device)
        logits = head.logits(trunk_mixer(S)).float()
        top2 = logits.topk(2, -1)
        mine = top2.indices[:, 0].cpu().numpy()
        same = mine == picks[r]
        agree += int(same.sum())
        n += len(r)
        gap = (top2.values[:, 0] - top2.values[:, 1]).cpu().numpy()
        worst.extend(gap[~same].tolist())
    return {"rows": n, "agree": agree, "disagree_margin_max": max(worst) if worst else 0.0}


@torch.no_grad()
def check_against_hf(head, stacks, tokens, n=300, lasts=(150, 151, 152, 153)):
    """The prototype vs transformers' layer on the real weights: a causal pass over n entries,
    and three chained steps after a few entries, as the HF layer over one sequence."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
    dev = head.device
    tok = torch.from_numpy(tokens[: n + 1]).to(dev)
    X = head.combine(to_bf16(stacks[:n], dev), tok[1:n + 1], "a", "a")
    cos, sin = head.rope(torch.arange(n, device=dev))

    def layer(seq):
        m = seq.shape[0]
        mask = torch.ones(m, m, dtype=torch.bool, device=dev).tril()[None, None]
        return head.layer(seq[None], position_embeddings=(cos[None, :m], sin[None, :m]), attention_mask=mask)[0]

    E = head.entries(X, torch.arange(n, device=dev))
    head.window(E, None, "dense")
    stack, mixed = head.first_step(torch.arange(n, device=dev))
    want = layer(X)
    rel = float((stack.float() - want.float()).norm() / want.float().norm())
    same = float((argmax(head, mixed) == argmax(head, head.mixer(want))).float().mean())
    chain_rel = []
    for last in lasts:
        k = v = r = None
        steps = []
        s1, m1 = head.first_step(torch.tensor([last], device=dev))
        prev_s, prev_m = s1, m1
        for j in range(2, 5):
            Xj = head.combine(prev_s, tok[last + j][None], "a", "a")
            steps.append(Xj)
            prev_s, prev_m, (k, v, r) = head.chain_step(Xj, torch.tensor([last], device=dev), k, v, r)
        seq = torch.cat([X[: last + 1]] + steps)
        ref = layer(seq)[-1]
        chain_rel.append(float((prev_s[0].float() - ref.float()).norm() / ref.float().norm()))
    return {"causal_rel_l2": rel, "causal_draft_agreement": same, "chain_rel_l2": chain_rel}


# ---------------------------------------------------------------------------------- run
def run(args):
    dev = "cuda"
    text, weights, trunk = fetch_head()
    cfg = mtp.layer_config(text)
    head = mtp.Head(cfg, weights, trunk["embed"], trunk["lm_head"], device=dev)
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
    trunk_mixer = hf.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    trunk_mixer.load_state_dict(trunk["mixer"])
    trunk_mixer = trunk_mixer.to(dev, torch.bfloat16).eval()
    del weights, trunk
    results = {"checks": {}, "texts": {}}
    corpus = list(texts(args.corpus))

    t0, tok0, st0, pk0 = corpus[0]
    results["checks"]["tap"] = check_tap(head, trunk_mixer, st0, pk0, np.arange(len(tok0)))
    tl, tokl, stl, pkl = max(corpus, key=lambda c: len(c[1]))
    results["checks"]["tap_long"] = check_tap(head, trunk_mixer, stl, pkl, np.arange(len(tokl) - 1024, len(tokl)))
    results["checks"]["hf"] = check_against_hf(head, st0, tok0)
    print(json.dumps(results["checks"]), flush=True)

    variants = [(c, n) for c in mtp.COMBS for n in mtp.NORMS]
    for t, tokens, stacks, picks in corpus:
        name, prompt = t["name"], t["prompt_tokens"]
        long = len(tokens) > 2051
        S = to_bf16(stacks, dev)
        runs = []
        if not long:
            runs = [(c, n, "dense", 0) for c, n in variants]
        else:
            # Long windows: the conventions again, at their prompt's last 512 entries and the
            # continuation, dense; then the indexer at two block offsets for the spec's prior.
            runs = [(c, n, "dense", 0) for c, n in variants] + [("a", "a", "index", 0), ("a", "a", "index", 1)]
        first = 0 if not long else prompt - 512
        results["texts"][name] = {"kind": t["kind"], "tokens": len(tokens), "prompt": prompt, "runs": {}}
        for comb, norm, mode, off in runs:
            began = time.time()
            r = score_text(head, tokens, S, picks, prompt, comb, norm, mtp.CHAINS, mode, off, first,
                           batch=256 if not long else 64)
            key = f"{comb}{norm}-{mode}{off if mode == 'index' else ''}"
            ok1 = r.pop("gen_ok1")
            r["gen_ok1"] = "".join("1" if x else "0" for x in ok1)
            results["texts"][name]["runs"][key] = r
            print(f"{name} {key}: a1 gen {r['gen']['a'][0]} prompt {r['prompt']} chain a {r['gen']['a'][1:]} "
                  f"b {r['gen']['b'][1:]} ({time.time() - began:.0f} s)", flush=True)
        del S
        torch.cuda.empty_cache()
    json.dump(results, open(args.out, "w"), indent=1)
    summarize(results)


def summarize(results):
    """alpha per run key over every text (and per class), with Wilson intervals, and the paired
    alpha_1 difference of every convention against the best."""
    keys = sorted({k for t in results["texts"].values() for k in t["runs"]})
    classes = {"all": lambda t: True, "short": lambda t: t["tokens"] <= 2051, "long": lambda t: t["tokens"] > 2051,
               "code": lambda t: t["kind"] == "code", "prose": lambda t: t["kind"] == "prose"}
    table = {}
    for key in keys:
        for cname, pred in classes.items():
            ts = [t for t in results["texts"].values() if pred(t) and key in t["runs"]]
            if not ts:
                continue
            row = {"texts": len(ts)}
            for chain in mtp.CHAINS:
                counts = np.sum([t["runs"][key]["gen"][chain] for t in ts], axis=0)
                row[f"chain_{chain}"] = [wilson(int(n), int(k)) + [int(n)] for n, k in counts]
            pn, pk = np.sum([t["runs"][key]["prompt"] for t in ts], axis=0)
            gn, gk = np.sum([t["runs"][key]["gen_prefill"] for t in ts], axis=0)
            row["alpha1_prompt_vs_prefill"] = wilson(int(pn), int(pk)) + [int(pn)]
            row["alpha1_gen_vs_prefill"] = wilson(int(gn), int(gk)) + [int(gn)]
            table.setdefault(key, {})[cname] = row
    # Paired alpha_1 differences on the generated positions every short text scored with all keys.
    dense = [k for k in keys if k.endswith("dense")]
    ok = {k: np.concatenate([np.frombuffer(t["runs"][k]["gen_ok1"].encode(), np.uint8) - 48
                             for t in results["texts"].values() if k in t["runs"]]) for k in dense}
    best = max(dense, key=lambda k: ok[k].mean())
    paired = {}
    for k in dense:
        d = ok[best].astype(float) - ok[k].astype(float)
        paired[k] = [float(d.mean()), float(1.96 * d.std(ddof=1) / math.sqrt(len(d)))]
    results["summary"] = {"table": table, "best_alpha1": best, "paired_alpha1_diff_vs_best": paired}
    for key, rows in table.items():
        r = rows["all"]
        print(key, "a1..a4 chain a:", [round(x[0], 3) for x in r["chain_a"]],
              "chain b:", [round(x[0], 3) for x in r["chain_b"]],
              "a1 prompt:", round(r["alpha1_prompt_vs_prefill"][0], 3))
    print("best", best, "paired diff", {k: [round(x, 4) for x in v] for k, v in paired.items()})
    return results


# ---------------------------------------------------------------------------------- projection
def tau(alphas, k):
    """Expected tokens per round with k drafts: 1 + a1 + a1 a2 + ... (a_j conditional)."""
    total, run = 1.0, 1.0
    for a in alphas[:k]:
        run *= a
        total += run
    return total


def round_table(cost):
    """{(lanes, width): round ms} from the example's cost.json (one consecutive cell per shape)
    or cost_repeat.json (text groups, widths interleaved, width 1 first and last): there a
    width's round is its lane count's mean width-1 round plus the mean over groups of the
    width's increment over its own group's width-1 rounds."""
    cells = cost["cells"]
    if "shape" in cells[0]:
        return {(c["lanes"], c["width"]): c["time"]["median_ms"] for c in cells if c["shape"] == "consecutive"}
    groups = {}
    for c in cells:
        groups.setdefault(tuple(c["texts"]), []).append(c)
    base, inc = {}, {}
    for texts, cs in groups.items():
        b = float(np.mean([c["time"]["median_ms"] for c in cs if c["width"] == 1]))
        base.setdefault(len(texts), []).append(b)
        for c in cs:
            if c["width"] > 1:
                inc.setdefault((len(texts), c["width"]), []).append(c["time"]["median_ms"] - b)
    table = {(L, 1): float(np.mean(b)) for L, b in base.items()}
    table.update({(L, w): table[(L, 1)] + float(np.mean(d)) for (L, w), d in inc.items()})
    return table


def project(alphas, cost):
    """The pre-registered one-lane verdict, and the speedup at 1/2/3 lanes from measured rounds."""
    pre = {k: tau(alphas, k) * R1_MS / (R1_MS + k * C_MS + k * D_MS) for k in (1, 2, 3)}
    rounds = round_table(cost)
    lanes = {}
    for L in (1, 2, 3):
        base = rounds[(L, 1)]
        cells = {}
        for k in range(1, 8):
            if (L, k + 1) not in rounds:
                continue
            c = (rounds[(L, k + 1)] - base) / k
            cells[k] = {"round_ms": rounds[(L, k + 1)], "c_ms_per_column": c,
                        "speedup": tau(alphas, k) * base / (rounds[(L, k + 1)] + k * D_MS)}
        lanes[L] = {"spec_off_round_ms": base, "k": cells}
    verdict = "GO" if max(pre.values()) >= GO else "NO-GO"
    return {"pre_registered": pre, "verdict": verdict, "measured": lanes}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--corpus", required=True)
    r.add_argument("--out", required=True)
    q = sub.add_parser("project")
    q.add_argument("--alpha", required=True)
    q.add_argument("--key", default=None, help="run key whose alphas project (default: the best alpha_1)")
    q.add_argument("--chain", default="a")
    q.add_argument("--cost", required=True)
    args = p.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        results = summarize(json.load(open(args.alpha)))
        key = args.key or results["summary"]["best_alpha1"]
        alphas = [x[0] for x in results["summary"]["table"][key]["all"][f"chain_{args.chain}"]]
        out = project(alphas, json.load(open(args.cost)))
        print(json.dumps({"key": key, "chain": args.chain, "alphas": alphas, **out}, indent=1))


if __name__ == "__main__":
    main()
