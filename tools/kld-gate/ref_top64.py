"""A portable top-64 KLD reference set for the 27B, from one BF16 pass: what the sm_86 gate scores against.

The reference is `ref_kld.py`'s: a BF16 checkpoint (`--ckpt`) run layer by layer under transformers' own
Qwen3_5 text modules, every window's hidden state on the GPU (`ref_kld.forward`). Then, per window and
512-position slice of the head, beside one dense `span_logits` dump (`--dump DIR`, `[len][vocab]` BF16, the
`q_` stream):

  (a) the full-vocabulary KL(ref || dump), top-1 agreement and NLL, with ref_kld.py's own arithmetic
      (`--gate1` checks it against a ref_kld.py result of the same dump, position by position);
  (b) the BF16 stream's 64 most probable next tokens (i32, descending, ties by the lower id) and their
      log-probabilities (f32 nats, fp32 log-softmax over the full vocabulary), plus its logsumexp;
  (c) the same for the dump, its argmax, and its log-probabilities at the BF16 row's 64 ids.

The set's files follow flash-next-layout.md section 10 (the names
`crates/bench/src/flash_next/references.rs` reads), so a Rust reader loads it as is. Beside them:
`bf16_next_lp.f32`, the BF16 log-prob of each position's actual next token (NaN at a window's last
position), `full_kl_5090.f32` (the full-vocabulary KL of the dump per position) and `full_kld_5090.json`
(summaries, the top-k sweep, the identity check), named for the RTX 5090's dump the release gate's set
holds; `windows.json`, the fed windows, so the set is all `span_logits` and `score_top64.py` need.

The top-k sweep scores the dump with `kl_top64`'s formula (bench `flash_next/kld.rs`, f64) for k in KS, so
the error of the truncation is measured against the exact KL in the same pass.

  python ref_top64.py --ckpt BF16 --windows-file W.json --dump DIR --out SET [--gate1 JSON] [--windows N]

`--windows-file` takes its windows' kinds from `<name>_meta.json` beside it, else from a set's
`manifest.json` there: a set rebuilds from its own `windows.json`. Needs the GPU, torch with CUDA,
transformers with `Qwen3_5`, safetensors and numpy, as `ref_kld.py` does.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import time

import numpy as np
import torch

import ref_kld

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "..", "..", "crates", "server", "src", "download", "catalog.yaml")
VOCAB = ref_kld.VOCAB
SLICE = ref_kld.SLICE
TOP = 64
KS = (16, 32, 64, 128, 256)
TAIL_EPS = 1e-12  # bench flash_next/kld.rs TAIL_EPS


def order(logits, k):
    """The k most probable ids per row, descending, ties by the lower id: a
    stable sort keeps equal logits in id order (bf16 logits tie often)."""
    return torch.sort(logits, dim=-1, descending=True, stable=True).indices[:, :k]


def kl_topk(r, c):
    """kl_top64's formula (f64) per row: reference log-probs r and the
    candidate's log-probs c at the same ids, the reference's remaining mass
    one bucket against the candidate's mass outside those ids."""
    r, c = r.double(), c.double()
    p = r.exp()
    head = (p * (r - c)).sum(-1)
    ref_tail = (1.0 - p.sum(-1)).clamp(min=0.0)
    cand_tail = (1.0 - c.exp().sum(-1)).clamp(min=TAIL_EPS)
    tail = ref_tail * (ref_tail.clamp(min=TAIL_EPS).log() - cand_tail.log())
    return head + torch.where(ref_tail < TAIL_EPS, torch.zeros_like(tail), tail)


def score_slice(ref_logits_bf16, q_logits, nxt, ks=KS, top=TOP):
    """One slice of rows. `ref_logits_bf16` is the head's bf16 output,
    `q_logits` the dump's rows as f32, `nxt` each row's next token."""
    lf = ref_logits_bf16.float()
    # (a) exactly ref_kld.py's arithmetic
    ref = torch.log_softmax(lf, dim=-1)
    ref_p = ref.exp()
    rt = ref.argmax(-1)
    q = torch.log_softmax(q_logits, dim=-1)
    out = {
        "kl": (ref_p * (ref - q)).sum(-1),
        "top1": q.argmax(-1) == rt,
        "nll": -q.gather(1, nxt[:, None])[:, 0],
        "ref_nll": -ref.gather(1, nxt[:, None])[:, 0],
    }
    del ref_p
    # (b) the BF16 stream
    kmax = max(max(ks), top)
    r_ids = order(lf, kmax)
    out["bf16_ids"] = r_ids[:, :top].to(torch.int32)
    out["bf16_lp"] = ref.gather(1, r_ids[:, :top])
    out["bf16_lse"] = torch.logsumexp(lf, dim=-1)
    for k in ks:
        out[f"kl_top{k}"] = kl_topk(ref.gather(1, r_ids[:, :k]), q.gather(1, r_ids[:, :k]))
    # (c) the dump's stream
    q_ids = order(q_logits, top)
    out["q_ids"] = q_ids.to(torch.int32)
    out["q_lp"] = q.gather(1, q_ids)
    out["q_lse"] = torch.logsumexp(q_logits, dim=-1)
    out["q_argmax"] = q_ids[:, 0].to(torch.int32)
    out["q_argmax_mismatch"] = (q_ids[:, 0] != q.argmax(-1)).sum()
    out["q_at_bf16"] = q.gather(1, out["bf16_ids"].long())
    out["nonfinite_q"] = (~torch.isfinite(q_logits)).any(-1).sum()
    return out


def summarize(values):
    return ref_kld.summarize(values)


def write(path, array, dtype):
    a = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    with open(path, "wb") as f:
        f.write(a.tobytes())
    return int(a.nbytes)


def catalog_entry(path, entry_id):
    """The catalog entry's repo, revision, artifact and the artifact file's
    bytes and sha256, read from catalog.yaml without a YAML dependency."""
    text = open(path, encoding="utf-8").read()
    start = text.index(f"- id: {entry_id}\n")
    end = text.find("\n  - id: ", start + 1)
    block = text[start:end if end > 0 else len(text)]
    get = lambda key: re.search(rf"^\s+{key}: \"?([^\"\n]+)\"?$", block, re.M).group(1)
    artifact = get("artifact")
    f = re.search(rf"name: {re.escape(artifact)}, bytes: (\d+), sha256: \"?([0-9a-f]+)\"?", block)
    return {"catalog_id": entry_id, "repo": get("repo"), "revision": get("revision"), "file": artifact,
            "bytes": int(f.group(1)), "sha256": f.group(2)}


def checkpoint_revision(ckpt):
    """The hub commit the checkpoint was downloaded at (huggingface_hub's
    download metadata, first line); None when it was not downloaded that way."""
    meta = os.path.join(ckpt, ".cache", "huggingface", "download", "config.json.metadata")
    return open(meta).readline().strip() if os.path.exists(meta) else None


def windows_meta(windows_file):
    """Each window's kind, source and part: `<name>_meta.json` beside the windows, else the `manifest.json`
    of the set they are in."""
    beside = windows_file[:-len(".json")] + "_meta.json" if windows_file.endswith(".json") else None
    if beside and os.path.exists(beside):
        return json.load(open(beside))
    manifest = os.path.join(os.path.dirname(os.path.abspath(windows_file)), "manifest.json")
    if os.path.exists(manifest):
        return [{"kind": w["kind"], "source": w["source"], "part": w.get("part")}
                for w in json.load(open(manifest))["windows"]]
    raise SystemExit(f"{windows_file}: no {beside or '<name>_meta.json'} and no manifest.json beside it "
                     "to give the windows' kinds")


FILES = {
    "tokens.u32": ("u32", "(sum length,)", "the fed token ids, each window's length tokens, in window order"),
    "bf16_top64_ids.i32": ("i32", "(P, 64)", "BF16 stream: the 64 most probable next tokens, descending, ties by the lower id"),
    "bf16_top64_lp.f32": ("f32", "(P, 64)", "their log-probabilities (nats, fp32 log-softmax over the full vocabulary)"),
    "bf16_lse.f32": ("f32", "(P,)", "BF16 stream: logsumexp of the position's logits"),
    "q_argmax.i32": ("i32", "(P,)", "q stream (the --dump): argmax, ties by the lower id"),
    "q_top64_ids.i32": ("i32", "(P, 64)", "q stream: the 64 most probable next tokens, descending, ties by the lower id"),
    "q_top64_lp.f32": ("f32", "(P, 64)", "q stream: their log-probabilities"),
    "q_lse.f32": ("f32", "(P,)", "q stream: logsumexp of the position's logits"),
    "q_lp_at_bf16_ids.f32": ("f32", "(P, 64)", "q stream: log-probabilities at the BF16 row's 64 ids (its top-64 KLD from the set alone)"),
    "bf16_next_lp.f32": ("f32", "(P,)", "BF16 log-probability of the actual next token; NaN at each window's last position"),
    "full_kl_5090.f32": ("f32", "(P,)", "full-vocabulary KL(BF16 || q) per position, ref_kld.py's arithmetic (fp32)"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="the BF16 checkpoint (Qwen/Qwen3.8-27B at the artifact's revision)")
    ap.add_argument("--windows-file", required=True,
                    help="the windows' token ids (JSON), as span_logits takes them; kinds in <name>_meta.json "
                         "beside it, or a set's manifest.json")
    ap.add_argument("--dump", required=True, help="the dense span_logits dump dir of those windows (the q_ stream)")
    ap.add_argument("--out", required=True, help="the reference set dir")
    ap.add_argument("--gate1", default="",
                    help="a ref_kld.py result of the same dump, for the per-position identity check (none: skipped)")
    ap.add_argument("--gate1-dump", default="trellis", help="the dump's name in the --gate1 result")
    ap.add_argument("--work", default="",
                    help="also write the per-position top-k sweep, NLL and top-1 of the dump here (none: skipped)")
    ap.add_argument("--windows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--catalog", default=CATALOG, help="catalog.yaml (default: the checkout's)")
    ap.add_argument("--catalog-id", default="qwen3.8-27b-16g", help="the entry whose artifact the dump is of")
    ap.add_argument("--chunk", type=int, default=1024, help="the dump's prefill chunk")
    ap.add_argument("--kv", default="hq-e8-2b", help="the dump's KV format")
    ap.add_argument("--engine", default="", help="what produced the dump (exe, commit, card, prompt band)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    artifact = catalog_entry(a.catalog, a.catalog_id)
    revision = checkpoint_revision(a.ckpt)

    windows = json.load(open(a.windows_file))
    meta = windows_meta(a.windows_file)
    if a.windows:
        windows, meta = windows[:a.windows], meta[:a.windows]
    for w, ids in enumerate(windows):
        path = os.path.join(a.dump, f"w{w:03d}.bin")
        if os.path.getsize(path) != len(ids) * VOCAB * 2:
            raise SystemExit(f"{path}: {os.path.getsize(path)} bytes, expected {len(ids) * VOCAB * 2}")
    os.makedirs(a.out, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    hidden, head, reference = ref_kld.forward(argparse.Namespace(ckpt=a.ckpt, overlay=None, smoke=False), windows)
    forward_s = time.time() - t0
    dev = head.device

    t2 = time.time()
    keys = ["kl", "top1", "nll", "ref_nll", "bf16_ids", "bf16_lp", "bf16_lse", "q_ids", "q_lp", "q_lse",
            "q_argmax", "q_at_bf16"] + [f"kl_top{k}" for k in KS]
    cols = {k: [] for k in keys}
    q_argmax_mismatch = nonfinite_q = 0
    rows = []
    first = 0
    for w, (ids, m) in enumerate(zip(windows, meta)):
        n = len(ids)
        h = hidden[w]
        nxt = torch.tensor(ids[1:] + [ids[-1]], device=dev)
        per = {k: [] for k in keys}
        with torch.inference_mode():
            for s in range(0, n, SLICE):
                e = min(n, s + SLICE)
                qrows = ref_kld.bf16_file_rows(os.path.join(a.dump, f"w{w:03d}.bin"), s, e).to(dev)
                o = score_slice(h[s:e] @ head.T, qrows, nxt[s:e])
                q_argmax_mismatch += int(o["q_argmax_mismatch"])
                nonfinite_q += int(o["nonfinite_q"])
                for k in keys:
                    per[k].append(o[k].cpu())
                del o, qrows
        hidden[w] = None
        per = {k: torch.cat(v).numpy() for k, v in per.items()}
        per["ref_nll"][-1] = np.nan  # a window's last position has no next token
        per["nll"][-1] = np.nan
        for k in keys:
            cols[k].append(per[k])
        row = {"window": w, "kind": m["kind"], "source": m["source"], "part": m.get("part"),
               "length": n, "first_position": first,
               "kld": float(per["kl"].mean()), "top1": float(per["top1"].mean()),
               "ppl": float(np.exp(np.nanmean(per["nll"]))), "ref_ppl": float(np.exp(np.nanmean(per["ref_nll"]))),
               "kld_top": {str(k): float(per[f"kl_top{k}"].mean()) for k in KS}}
        rows.append(row)
        first += n
        print(f"w{w:02d} {m['kind']:5s} KLD {row['kld']:.5f} top64 {row['kld_top']['64']:.5f} "
              f"top1 {row['top1'] * 100:.2f}% ppl {row['ppl']:.3f} ref ppl {row['ref_ppl']:.3f}", flush=True)
    score_s = time.time() - t2
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    kinds = np.array([r["kind"] for r in rows for _ in range(r["length"])])
    P = int(cat["kl"].size)

    # the set (flash-next-layout.md section 10 names)
    out_dir = a.out
    sizes = {}
    tokens = np.concatenate([np.asarray(ids, dtype=np.int64) for ids in windows])
    assert tokens.min() >= 0 and tokens.max() < VOCAB
    sizes["tokens.u32"] = write(os.path.join(out_dir, "tokens.u32"), tokens, "<u4")
    sizes["bf16_top64_ids.i32"] = write(os.path.join(out_dir, "bf16_top64_ids.i32"), cat["bf16_ids"], "<i4")
    sizes["bf16_top64_lp.f32"] = write(os.path.join(out_dir, "bf16_top64_lp.f32"), cat["bf16_lp"], "<f4")
    sizes["bf16_lse.f32"] = write(os.path.join(out_dir, "bf16_lse.f32"), cat["bf16_lse"], "<f4")
    sizes["q_argmax.i32"] = write(os.path.join(out_dir, "q_argmax.i32"), cat["q_argmax"], "<i4")
    sizes["q_top64_ids.i32"] = write(os.path.join(out_dir, "q_top64_ids.i32"), cat["q_ids"], "<i4")
    sizes["q_top64_lp.f32"] = write(os.path.join(out_dir, "q_top64_lp.f32"), cat["q_lp"], "<f4")
    sizes["q_lse.f32"] = write(os.path.join(out_dir, "q_lse.f32"), cat["q_lse"], "<f4")
    sizes["q_lp_at_bf16_ids.f32"] = write(os.path.join(out_dir, "q_lp_at_bf16_ids.f32"), cat["q_at_bf16"], "<f4")
    sizes["bf16_next_lp.f32"] = write(os.path.join(out_dir, "bf16_next_lp.f32"), -cat["ref_nll"], "<f4")
    sizes["full_kl_5090.f32"] = write(os.path.join(out_dir, "full_kl_5090.f32"), cat["kl"], "<f4")
    if a.work:
        # per-position sweep values, for a check of the scorer; not part of the set
        os.makedirs(a.work, exist_ok=True)
        for k in KS:
            write(os.path.join(a.work, f"kl_top{k}_5090.f64"), cat[f"kl_top{k}"], "<f8")
        write(os.path.join(a.work, "nll_5090.f32"), cat["nll"], "<f4")
        write(os.path.join(a.work, "top1_5090.u8"), cat["top1"], "u1")

    def figures(mask):
        kl = cat["kl"][mask]
        s = summarize(kl)
        s["top1"] = float(cat["top1"][mask].mean())
        s["nll"] = float(np.nanmean(cat["nll"][mask]))
        s["ppl"] = float(np.exp(s["nll"]))
        s["ref_nll"] = float(np.nanmean(cat["ref_nll"][mask]))
        s["ref_ppl"] = float(np.exp(s["ref_nll"]))
        s["kld_top"] = {}
        for k in KS:
            t = float(cat[f"kl_top{k}"][mask].mean())
            s["kld_top"][str(k)] = {"mean": t, "rel_err": (t - s["mean"]) / s["mean"]}
        return s

    summary = {"all": figures(np.ones(P, dtype=bool))}
    for kind in sorted(set(kinds.tolist())):
        summary[kind] = figures(kinds == kind)

    identity = None
    if a.gate1:
        g = json.load(open(a.gate1))
        gk = np.concatenate([np.asarray(r[a.gate1_dump]["kl"]) for r in g["windows"][:len(windows)]])
        gn = np.concatenate([np.asarray(r["ref_nll"]) for r in g["windows"][:len(windows)]])
        gt = np.concatenate([np.asarray(r[a.gate1_dump]["top1"]) for r in g["windows"][:len(windows)]])
        new_rn = np.concatenate([c[:-1] for c in cols["ref_nll"]])
        dk = np.abs(cat["kl"].astype(np.float64) - gk)
        dn = np.abs(new_rn.astype(np.float64) - gn)
        identity = {
            "gate1_file": os.path.basename(a.gate1), "gate1_dump": a.gate1_dump,
            "gate1_kld_all": float(gk.mean()), "new_kld_all": float(cat["kl"].mean()),
            "gate1_top1_all": float(gt.mean()), "new_top1_all": float(cat["top1"].mean()),
            "kl_max_abs_diff": float(dk.max()), "kl_positions_identical": int((dk == 0).sum()),
            "ref_nll_max_abs_diff": float(dn.max()), "ref_nll_positions_identical": int((dn == 0).sum()),
            "positions": P,
            "per_kind": {kind: {"gate1": float(gk[kinds == kind].mean()),
                                "new": float(cat["kl"][kinds == kind].mean())}
                         for kind in sorted(set(kinds.tolist()))},
            "per_window": [{"window": r["window"], "gate1": float(gk[r["first_position"]:r["first_position"] + r["length"]].mean()),
                            "new": r["kld"]} for r in rows],
        }

    result = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "positions": P, "vocab": VOCAB,
              "top": TOP, "ks": list(KS), "forward_s": round(forward_s, 1), "score_s": round(score_s, 1),
              "q_argmax_mismatch": q_argmax_mismatch, "nonfinite_q_rows": nonfinite_q,
              "summary": summary, "windows": rows, "identity_gate1": identity, "files": sizes}
    with open(os.path.join(out_dir, "full_kld_5090.json"), "w", newline="\n") as f:
        json.dump(result, f, indent=1)
    if os.path.abspath(a.windows_file) != os.path.abspath(os.path.join(out_dir, "windows.json")):
        shutil.copyfile(a.windows_file, os.path.join(out_dir, "windows.json"))

    import transformers
    manifest = {
        "set": os.path.basename(os.path.abspath(out_dir)),
        "layout": "flash-next-layout.md section 10 (crates/bench/src/flash_next/references.rs reads it); "
                  "the q_ stream is the --dump",
        "created": result["created"],
        "windows": [{"index": r["window"], "kind": r["kind"], "source": r["source"], "part": r["part"],
                     "length": r["length"], "valid": r["length"], "first_position": r["first_position"]}
                    for r in rows],
        "positions": P,
        "top": TOP,
        "vocab": VOCAB,
        "chunk": a.chunk,
        "kv": a.kv,
        "checkpoint": {"repo": "Qwen/Qwen3.8-27B", "revision": revision,
                       "revision_source": "the checkpoint's .cache/huggingface/download metadata; the same "
                                          "revision as the artifact's conversion.json reference block",
                       "runner": f"transformers {transformers.__version__} Qwen3_5 text modules, BF16, layer by "
                                 f"layer, torch {torch.__version__}, TF32 off"},
        "artifact": dict(artifact, sha256_source=f"{a.catalog_id} in crates/server/src/download/catalog.yaml"),
        "streams": {
            "bf16": "the checkpoint's head output (bf16 matmul) upcast to fp32; log-probabilities and lse in fp32",
            "q": f"a dense span_logits dump of the artifact (chunk {a.chunk}, kv {a.kv}): bf16 logits upcast to "
                 "fp32; log-probabilities and lse in fp32. An engine-vs-engine bug detector, not a quality reference",
        },
        "dump": {"engine": a.engine},
        "conventions": [
            "position p of a window is the distribution after consuming tokens [0..p] (predicting token p+1); "
            "every position 0..length-1 is stored (valid = length), so a window's last position has no next token "
            "in the set: KL and top-1 count it, NLL does not",
            "ties in top-64 and argmax resolve to the lower token id",
            "all files little-endian, row-major, rows in window order (first_position is a window's first row)",
        ],
        "files": {name: {"dtype": t, "shape": s, "bytes": sizes[name], "content": c}
                  for name, (t, s, c) in FILES.items()},
        "extra_files": {"windows.json": "the fed windows as span_logits takes them (a JSON array of token-id arrays)",
                        "full_kld_5090.json": "the full-vocabulary and top-k KLD of the q stream, per kind and window"},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: result[k] for k in ("summary", "identity_gate1", "q_argmax_mismatch", "nonfinite_q_rows")},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
