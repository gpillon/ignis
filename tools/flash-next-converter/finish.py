"""After the last layer: the head over the three streams, the stored references, the G1
fixture, the routing traces, the self-check, converter.json and the plain report.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import fp8
import layout
import pipeline as P
import scoring
import trellis

SLICE = 512


def _git(repo, *args):
    try:
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=60).stdout.strip()
    except OSError:
        return ""


class Head:
    """mixer -> lm_head, in BF16 (the checkpoint) or as the artifact holds it (FP8)."""

    def __init__(self, conv, quantized):
        mq, src, dev = conv.mq, conv.src, conv.dev
        g = os.path.join(conv.work, "global")
        sd = src.load(P.PFX + "hyper_connection_mixer.")
        lm = src.get("lm_head.weight")
        if quantized:
            w = conv._fp8_weights(g, "cpu")
            sd = {k: (w["hyper_connection_mixer." + k] if "hyper_connection_mixer." + k in w else v) for k, v in sd.items()}
            lm = w["lm_head.weight"]
        with torch.device("meta"):
            self.mixer = mq.Qwen4ExpTextGatedResidual(conv.cfg, use_combine=False)
        self.mixer.load_state_dict(sd, assign=True)
        self.mixer = self.mixer.to(dev, torch.bfloat16).eval()
        self.lm = lm.to(dev, torch.bfloat16)

    def logits(self, h):
        """(T, 4*2560) bf16 states -> (T, vocab) fp32 logits (bf16 matmul, upcast)."""
        return (self.mixer(h[None])[0] @ self.lm.T).float()


class RefWriter:
    """layout.md §10: one reference set's files."""

    def __init__(self, root, name):
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.parts = defaultdict(list)
        self.windows = []
        self.first = 0

    def window(self, meta, tokens, valid):
        self.windows.append(dict(meta, index=len(self.windows), length=len(tokens), valid=valid,
                                 first_position=self.first))
        self.parts["tokens"].append(np.asarray(tokens, dtype="<u4"))
        self.first += valid

    def add(self, **arrays):
        for k, v in arrays.items():
            self.parts[k].append(v.cpu().numpy())

    def close(self):
        names = {"tokens": "tokens.u32", "r_ids": "bf16_top64_ids.i32", "r_lp": "bf16_top64_lp.f32",
                 "r_lse": "bf16_lse.f32", "q_arg": "q_argmax.i32", "q_ids": "q_top64_ids.i32", "q_lp": "q_top64_lp.f32",
                 "q_lse": "q_lse.f32"}
        dt = {"tokens": "<u4", "r_ids": "<i4", "r_lp": "<f4", "r_lse": "<f4", "q_arg": "<i4", "q_ids": "<i4",
              "q_lp": "<f4", "q_lse": "<f4"}
        for k, fname in names.items():
            np.concatenate(self.parts[k]).astype(dt[k]).tofile(os.path.join(self.dir, fname))
        layout.write_json_atomic(os.path.join(self.dir, "manifest.json"),
                                 {"windows": self.windows, "positions": self.first, "top": scoring.TOP,
                                  "position_rule": "row p of a window is the distribution after tokens [0..p]"})


def _score_window(heads, states, tokens, valid, writer, acc, kind, mm=(), letters=None, attrib=True):
    """One window through the heads: stores its references, accumulates its metrics.
    `mm`: MMLU marks (position before the answer letter, gold, options, category)."""
    hb, hq = heads["bf16"], heads["q"]
    ref_h, q_h = states["bf16"], states["q"]
    f8_h = states.get("f8")
    tgt_all = torch.tensor(tokens[1:valid], device=ref_h.device)
    letters_t = torch.tensor(letters, device=ref_h.device) if mm else None
    argmax_q, argmax_r = [], []
    for s in range(0, valid, SLICE):
        e = min(s + SLICE, valid)
        lr_raw = hb.logits(ref_h[s:e])
        lse_r = torch.logsumexp(lr_raw, -1)
        lr = lr_raw - lse_r[:, None]
        del lr_raw
        lq_raw = hq.logits(q_h[s:e])
        lse_q = torch.logsumexp(lq_raw, -1)
        lq = lq_raw - lse_q[:, None]
        del lq_raw
        r_ids, r_lp, q_ids, q_lp = [], [], [], []
        for t in range(0, e - s, 128):
            a, b = scoring.top_k(lr[t:t + 128], scoring.TOP)
            r_ids.append(a), r_lp.append(b)
            a, b = scoring.top_k(lq[t:t + 128], scoring.TOP)
            q_ids.append(a), q_lp.append(b)
        r_ids, r_lp, q_ids, q_lp = map(torch.cat, (r_ids, r_lp, q_ids, q_lp))
        aq, ar = scoring.argmax(lq), scoring.argmax(lr)
        writer.add(r_ids=r_ids.int(), r_lp=r_lp, r_lse=lse_r, q_arg=aq.int(), q_ids=q_ids.int(), q_lp=q_lp,
                   q_lse=lse_q)
        argmax_q.append(aq.cpu()), argmax_r.append(ar.cpu())
        n = min(e, valid - 1) - s            # positions with a next token
        lf = None
        if n > 0:
            tgt = tgt_all[s:s + n]
            acc["q"]["kl"][kind].append(scoring.kl_exact(lr[:n], lq[:n]).cpu())
            acc["q"]["kl64"][kind].append(scoring.kl_top64(r_ids[:n], r_lp[:n], lq[:n]).cpu())
            acc["q"]["top1"][kind].append((ar[:n] == aq[:n]).float().cpu())
            acc["q"]["nll"][kind].append(-lq[:n].gather(1, tgt[:, None])[:, 0].cpu())
            acc["ref"]["nll"][kind].append(-lr[:n].gather(1, tgt[:, None])[:, 0].cpu())
            if f8_h is not None:
                lf = torch.log_softmax(hq.logits(f8_h[s:s + n]), -1)
                acc["f8"]["kl"][kind].append(scoring.kl_exact(lr[:n], lf).cpu())
                acc["f8"]["kl64"][kind].append(scoring.kl_top64(r_ids[:n], r_lp[:n], lf).cpu())
                acc["f8"]["top1"][kind].append((ar[:n] == scoring.argmax(lf)).float().cpu())
            if attrib:
                lh = torch.log_softmax(hq.logits(ref_h[s:s + n]), -1)
                acc["head"]["kl"][kind].append(scoring.kl_exact(lr[:n], lh).cpu())
                del lh
        for pos, gold, nopt, _cat in mm:
            if s <= pos < s + max(n, 0):
                for name, lp in (("ref", lr), ("q", lq), ("f8", lf)):
                    if lp is not None:
                        sc = lp[pos - s][letters_t]
                        acc["mmlu"][name].append((int(sc[:nopt].argmax()), gold))
        del lr, lq, lf
    return torch.cat(argmax_q), torch.cat(argmax_r)


def head_pass(conv, state, refs_root):
    dev = conv.dev
    heads = {"bf16": Head(conv, quantized=False), "q": Head(conv, quantized=True)}
    acc = {"q": defaultdict(lambda: defaultdict(list)), "f8": defaultdict(lambda: defaultdict(list)),
           "ref": defaultdict(lambda: defaultdict(list)), "head": defaultdict(lambda: defaultdict(list)),
           "mmlu": defaultdict(list)}
    letters = conv.corpus.letter_ids
    with torch.no_grad():
        w = RefWriter(refs_root, "test2048")
        for j, ci in enumerate(conv.test_sel):
            c = conv.chunks[ci]
            states = {"bf16": state["bf16.chunks"][ci].to(dev), "q": state["q.test"][j].to(dev),
                      "f8": state["f8.test"][j].to(dev)}
            w.window({"kind": c["kind"], "source": c["source"]}, c["ids"], c["valid"])
            marks = [(m[0], m[1], m[2], m[3] if len(m) > 3 else "") for m in c["mmlu"]]
            _score_window(heads, states, c["ids"], c["valid"], w, acc, c["kind"], mm=marks, letters=letters)
            del states
        w.close()
        long_acc = {"q": defaultdict(lambda: defaultdict(list)), "f8": defaultdict(lambda: defaultdict(list)),
                    "ref": defaultdict(lambda: defaultdict(list)), "head": defaultdict(lambda: defaultdict(list)),
                    "mmlu": defaultdict(list)}
        w = RefWriter(refs_root, "long8192")
        for j, x in enumerate(conv.corpus.long):
            states = {"bf16": state["bf16.long"][j].to(dev), "q": state["q.long"][j].to(dev)}
            w.window({"kind": x["kind"], "source": x["source"]}, x["ids"], 8192)
            _score_window(heads, states, x["ids"], 8192, w, long_acc, x["kind"], attrib=False)
        w.close()
        g1 = []
        w = RefWriter(refs_root, "canary")
        canary_acc = {"q": defaultdict(lambda: defaultdict(list)), "f8": defaultdict(lambda: defaultdict(list)),
                      "ref": defaultdict(lambda: defaultdict(list)), "head": defaultdict(lambda: defaultdict(list)),
                      "mmlu": defaultdict(list)}
        for j, x in enumerate(conv.canary):
            seq = x["prompt_ids"] + x["token_ids"]
            states = {"bf16": state["bf16.canary"][j].to(dev), "q": state["q.canary"][j].to(dev)}
            w.window({"kind": "canary", "source": x["id"]}, seq, len(seq))
            aq, ar = _score_window(heads, states, seq, len(seq), w, canary_acc, "canary", attrib=False)
            lp = len(x["prompt_ids"])
            n = len(x["token_ids"])
            g1.append({"id": x["id"], "prompt": x["prompt"], "text": x["text"], "token_ids": x["token_ids"],
                       "prompt_token_ids": x["prompt_ids"],
                       "expected_argmax": aq[lp - 1:lp - 1 + n].tolist(),
                       "bf16_argmax": ar[lp - 1:lp - 1 + n].tolist()})
        w.close()
    fixture = {"model": "qwen3.8-flash-next-ignis", "max_tokens": 32, "prompts": g1,
               "render": {"template": f"chat_template.jinja@{P.fetch.REVISION}", "enable_thinking": False,
                          "add_generation_prompt": True},
               "reference": "quantized", "canary_source_model": conv.canary_fixture_model}
    layout.write_json_atomic(os.path.join(refs_root, "g1_flash_next.json"), fixture)
    del heads
    torch.cuda.empty_cache()
    return acc, long_acc, canary_acc, fixture


def _mean(parts):
    return float(torch.cat(parts).mean()) if parts else float("nan")


def summarize(acc):
    out = {}
    for v in ("q", "f8"):
        out[v] = {}
        for kind in sorted(acc[v]["kl"]):
            out[v][kind] = {"kld": _mean(acc[v]["kl"][kind]), "kld_top64": _mean(acc[v]["kl64"][kind]),
                            "top1": _mean(acc[v]["top1"][kind])}
            if v == "q":
                out[v][kind]["ppl"] = float(np.exp(_mean(acc["q"]["nll"][kind])))
                out[v][kind]["ppl_bf16"] = float(np.exp(_mean(acc["ref"]["nll"][kind])))
    out["head_fp8_only"] = {k: _mean(v) for k, v in acc["head"]["kl"].items()}
    return out


def traces(conv, root):
    os.makedirs(root, exist_ok=True)
    out = []
    groups = defaultdict(list)
    for j, ci in enumerate(conv.test_sel):
        groups[conv.chunks[ci]["kind"]].append(j)
    sets = [(k, "test", js, P.CH, [conv.chunks[conv.test_sel[j]]["valid"] for j in js]) for k, js in groups.items()]
    sets.append(("long8192", "long", list(range(len(conv.corpus.long))), 8192, [8192] * len(conv.corpus.long)))
    for name, s, js, T, valids in sets:
        rows = np.concatenate([j * T + np.arange(v) for j, v in zip(js, valids)])
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        for what, dt, width in (("experts", np.int16, 10), ("weights", np.float16, 10), ("lookahead", np.int16, 20)):
            arr = np.empty((rows.size, conv.n_layers, width), dtype=dt)
            for L in range(conv.n_layers):
                arr[:, L] = np.load(os.path.join(conv.layer_dir(L), f"route_{s}_{what}.npy"), mmap_mode="r")[rows]
            arr.astype({"experts": "<i2", "weights": "<f2", "lookahead": "<i2"}[what]).tofile(
                os.path.join(d, {"experts": "experts.i16", "weights": "weights.f16", "lookahead": "lookahead.i16"}[what]))
        layout.write_json_atomic(os.path.join(d, "manifest.json"), {
            "tokens": int(rows.size), "layers": conv.n_layers, "set": s, "chunks": js, "valid": valids,
            "stream": "quantized", "order": "token-major (N, layers, k)"})
        out.append({"domain": name, "tokens": int(rows.size)})
    return out


def self_check(conv):
    n, bad = 0, []
    for L in range(conv.n_layers):
        d = conv.layer_dir(L)
        rec = json.load(open(os.path.join(d, "layer.json")))
        index = {(e.expert, e.proj): e for e in layout.read_index(os.path.join(d, "experts.idx"))}
        for cls, s in rec["decode_sha256"].items():
            proj = cls.split("-")[0]
            en = index[(s["expert"], proj)]
            w = trellis.decode(layout.read_record(os.path.join(d, "experts.bin"), en), en.k2, proj, conv.dev)
            got = hashlib.sha256(w.to(torch.bfloat16).cpu().view(torch.int16).numpy().tobytes()).hexdigest()
            n += 1
            if got != s["sha256"]:
                bad.append(f"L{L} {cls} expert {s['expert']}")
    return {"projections_checked": n, "bit_identical": not bad, "mismatches": bad}


def finish(conv, state, t_start):
    log = conv.log
    refs_root = os.path.join(conv.out, "references")
    t0 = time.time()
    acc, long_acc, canary_acc, fixture = head_pass(conv, state, refs_root)
    log(f"head and references: {time.time() - t0:.0f}s")
    state.clear()
    summary = summarize(acc)
    long_summary = summarize(long_acc)
    tr = traces(conv, os.path.join(conv.out, "traces"))
    sc = self_check(conv)
    layers = [json.load(open(os.path.join(conv.layer_dir(L), "layer.json"))) for L in range(conv.n_layers)]
    full = conv.n_layers == P.N_LAYERS

    # acceptance 2
    rates = {p: float(np.mean([l["rates"][p] for l in layers])) for p in ("gu", "dn")}
    stored = {p: float(np.mean([l["stored_rates"][p] for l in layers])) for p in ("gu", "dn")}
    a2 = all(rates[p] <= 2.5 + 1e-12 for p in rates) and all(
        set(l["k2"][p]) <= set(layout.K2_SET) for l in layers for p in ("gu", "dn"))
    # acceptance 3
    dbs = [l["moe_db"] for l in layers]
    mean15 = float(np.mean(dbs[1:6])) if len(dbs) >= 6 else None
    worse = [l["layer"] for l in layers if l["moe_db"] > P.RUN6_DB[l["layer"]] + 1.0]
    a3 = mean15 is not None and mean15 <= P.RUN8_MEAN_1_5 + 0.5
    # acceptance 4
    kld = {"quantized": {}, "fp8_only": {}}
    a4_kld = True
    for kind, s in summary["q"].items():
        lim = 1.1 * P.RUN6_KLD[kind] if kind in P.RUN6_KLD else None
        ok = lim is None or s["kld"] <= lim
        a4_kld &= ok
        kld["quantized"][kind] = dict(s, run6=P.RUN6_KLD.get(kind, P.RUN6_KLD_INFO.get(kind)), limit=lim, pass_=ok)
    for kind, s in summary["f8"].items():
        kld["fp8_only"][kind] = dict(s, run6_fp8=P.RUN6_FP8_KLD.get(kind))
    mm = acc["mmlu"]
    ok_ref = [a == g for a, g in mm["ref"]]
    mmlu = {"n": len(ok_ref), "bf16": float(np.mean(ok_ref)) if ok_ref else None, "mcnemar": {}}
    a4_mmlu = True
    for v, name in (("q", "quantized"), ("f8", "fp8_only")):
        okv = [a == g for a, g in mm[v]]
        mmlu[name] = float(np.mean(okv)) if okv else None
        lost, gained, p = scoring.paired(ok_ref, okv)
        mmlu["mcnemar"][name] = {"lost": lost, "gained": gained, "p": p}
        if v == "q":
            sig_below = lost > gained and p <= 0.05
            a4_mmlu = mmlu[name] is not None and mmlu[name] >= P.MMLU_FLOOR and not sig_below
    a4 = a4_kld and a4_mmlu
    verdict = "PASS" if (a2 and a3 and a4 and sc["bit_identical"]) else "FAIL"
    if not full:
        verdict = "DRY-RUN"
    total = time.time() - t_start
    hot = json.load(open(os.path.join(conv.work, "ngram", "hot_rows.json")))
    repo = conv.args.repo
    import importlib.metadata as md
    record = {
        "schema": P.CONVERTER_SCHEMA,
        "status": "complete" if full else "dry-run",
        "verdict": verdict,
        "source": {"repo": P.fetch.REPO, "revision": P.fetch.REVISION},
        "converter": {"commit": _git(repo, "rev-parse", "HEAD"),
                      "dirty": bool(_git(repo, "status", "--porcelain", "--", "tools/flash-next-converter")),
                      "seed_rule": "exllamav3 seed L*10000 + proj*1000 + expert (proj 0 gate/up, 1 down)",
                      "command": " ".join(sys.argv)},
        "versions": {"exllamav3": md.version("exllamav3"), "transformers": md.version("transformers"),
                     "torch": torch.__version__, "python": platform.python_version()},
        "quantizer": {"codebook": "mul1", "apply_out_scales": True, "K_set": [k / 2 for k in layout.K2_SET],
                      "hessian": "g^2-weighted per-expert input metric on calibration tokens; gate/up shrunk 5% "
                                 "toward the layer H, down 5% toward I", "budget_bits": conv.args.budget,
                      "batch": P.BQ},
        "corpus": {"manifest": conv.corpus.manifest,
                   "calibration_chunks": sum(1 for c in conv.chunks if c["cal"]),
                   "test_chunks": len(conv.test_sel),
                   "tokens_per_domain": {
                       "calibration": _tokens(conv.chunks, True), "test": _tokens(conv.chunks, False)},
                   "long8192": [x["source"] for x in conv.corpus.long],
                   "canary": [x["id"] for x in conv.canary]},
        "k_map": {"layers": [{"layer": l["layer"], "gu": l["k2"]["gu"], "dn": l["k2"]["dn"]} for l in layers]},
        "k_hist": [{"layer": l["layer"], **{p: dict(zip(["2", "2.5", "3", "4"], l["k_hist"][p]))
                                            for p in ("gu", "dn")}} for l in layers],
        "rates": {"per_layer": [{"layer": l["layer"], "gu": l["rates"]["gu"], "dn": l["rates"]["dn"],
                                 "gu_stored": l["stored_rates"]["gu"], "dn_stored": l["stored_rates"]["dn"]}
                                for l in layers],
                  "mean": {"gu": rates["gu"], "dn": rates["dn"], "gu_stored": stored["gu"], "dn_stored": stored["dn"]}},
        "k_classes": _k_classes(layers),
        "expert_traffic": {"layers": [{"layer": l["layer"], "counts": l["expert_traffic"]} for l in layers]},
        "moe_error_db": [{"layer": l["layer"], "db": l["moe_db"], "run6_db": l["run6_db"], "run8_db": l["run8_db"],
                          "per_kind_db": l["moe_db_kind"]} for l in layers],
        "acceptance2": {"pass": a2},
        "acceptance3": {"mean_db_layers_1_5": mean15, "run8_mean_db_layers_1_5": P.RUN8_MEAN_1_5,
                        "named_layers_worse_than_run6_by_1db": worse, "pass": a3},
        "kld": kld,
        "kld_long8192": long_summary,
        "kld_head_fp8_only": summary["head_fp8_only"],
        "mmlu": mmlu,
        "acceptance4": {"pass": a4, "kld_pass": a4_kld, "mmlu_pass": a4_mmlu,
                        "fallback": "re-convert at a 3.0-bit mean: --budget 3.0 (45.3 GB pinned)"},
        "self_check": sc,
        "ngram": {"rows": P.TABLE_SHARDS * P.SHARD_ROWS, "row_bytes": 90, "hot_rows": hot["rows"],
                  "hot_rows_bytes": hot["bytes"], "shards_converted": conv.table_shards},
        "references": {"dir": "references", "sets": ["test2048", "long8192", "canary"], "g1": "g1_flash_next.json"},
        "traces": {"dir": "traces", "domains": tr},
        "time_s": {"total": total, "per_layer": [l["time_s"] for l in layers]},
    }
    layout.write_json_atomic(os.path.join(conv.work, "converter.json"), record)
    report = render_report(record, layers)
    with open(os.path.join(conv.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    log(report)
    if verdict == "FAIL":
        return 3
    return 0


def _tokens(chunks, cal):
    out = defaultdict(int)
    for c in chunks:
        if bool(c["cal"]) == cal and (cal or c["test"]):
            out[c["kind"]] += c["valid"]
    return dict(out)


def _k_classes(layers):
    counts = defaultdict(int)
    traffic = defaultdict(float)
    total = 0.0
    for l in layers:
        tr = l["expert_traffic"]
        for p in ("gu", "dn"):
            for e, k2 in enumerate(l["k2"][p]):
                counts[(p, k2)] += 1
                traffic[(p, k2)] += tr[e]
        total += sum(tr)
    out = []
    for p in ("gu", "dn"):
        for k2 in layout.K2_SET:
            out.append({"class": f"{p}-{k2 / 2:g}", "k2": k2, "projections": counts[(p, k2)],
                        "record_bytes": layout.record_bytes(p, k2),
                        "traffic_share": traffic[(p, k2)] / total if total else 0.0})
    return out


def render_report(r, layers):
    L = []
    L.append(f"Flash-Next conversion: {r['verdict']} ({r['status']})")
    L.append(f"source {r['source']['repo']}@{r['source']['revision']}  converter {r['converter']['commit']}"
             f"{' (dirty)' if r['converter']['dirty'] else ''}  exllamav3 {r['versions']['exllamav3']}  "
             f"transformers {r['versions']['transformers']}")
    m = r["rates"]["mean"]
    L.append(f"rate (K + scales, b/w): gate/up {m['gu']:.4f}  down {m['dn']:.4f}   stored incl. 4 KiB padding: "
             f"{m['gu_stored']:.4f} / {m['dn_stored']:.4f}   acceptance 2 {'PASS' if r['acceptance2']['pass'] else 'FAIL'}")
    L.append("")
    L.append("layer | MoE dB | run 6 | run 8 | K hist gu [2,2.5,3,4] | K hist dn | time s")
    for l in layers:
        r8 = f"{l['run8_db']:.2f}" if l["run8_db"] is not None else "  -  "
        L.append(f"{l['layer']:5d} | {l['moe_db']:6.2f} | {l['run6_db']:6.2f} | {r8:>6} | {l['k_hist']['gu']} | "
                 f"{l['k_hist']['dn']} | {l['time_s']:.0f}")
    a3 = r["acceptance3"]
    L.append(f"mean dB layers 1-5 {a3['mean_db_layers_1_5']} (run 8 {a3['run8_mean_db_layers_1_5']}, limit +0.5): "
             f"{'PASS' if a3['pass'] else 'FAIL'}; layers worse than run 6 by > 1 dB: {a3['named_layers_worse_than_run6_by_1db'] or 'none'}")
    L.append("")
    L.append("KLD vs BF16 (test chunks) | quantized exact / top-64 / top-1 | limit (run 6 x 1.1) | FP8-only | run 6 FP8-only")
    for kind, s in r["kld"]["quantized"].items():
        f = r["kld"]["fp8_only"].get(kind, {})
        lim = f"{s['limit']:.3f}" if s["limit"] is not None else "  -  "
        L.append(f"  {kind:6s} {s['kld']:.4f} / {s['kld_top64']:.4f} / {s['top1'] * 100:.1f}% | {lim} "
                 f"{'ok' if s['pass_'] else 'OVER'} | {f.get('kld', float('nan')):.4f} | {f.get('run6_fp8')}")
    L.append("head alone in FP8 (BF16 stream, KLD): " + ", ".join(
        f"{k} {v:.4f}" for k, v in r["kld_head_fp8_only"].items()))
    mm = r["mmlu"]
    L.append(f"MMLU-Pro proxy (n={mm['n']}): BF16 {mm['bf16']}, quantized {mm['quantized']} "
             f"(lost/gained/p {mm['mcnemar']['quantized']}), FP8-only {mm['fp8_only']} "
             f"({mm['mcnemar']['fp8_only']}); floor {P.MMLU_FLOOR}")
    a4 = r["acceptance4"]
    L.append(f"acceptance 4: {'PASS' if a4['pass'] else 'FAIL — fallback: ' + a4['fallback']}")
    sc = r["self_check"]
    L.append(f"self-check: {sc['projections_checked']} projections re-decoded, bit-identical {sc['bit_identical']}")
    L.append(f"time: {r['time_s']['total'] / 3600:.2f} h")
    return "\n".join(L) + "\n"
