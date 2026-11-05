"""KLD of ignis's distributions against a BF16 checkpoint's, per position: gate 1's reference side.

The reference is a BF16 checkpoint of the 27B (`--ckpt`: the base's, or a fine-tune's of the same
architecture) under transformers' own Qwen3_5 text modules. It is run **layer by layer**: every window's
hidden state stays on the GPU (16 x 4096 x 5120 BF16 = 0.67 GB) and each decoder layer is read from the
checkpoint once, run over every window, and freed -- no CPU offload, which 64 GB of host RAM does not
hold beside the desktop. Each window is its own sequence (no cache), exactly the teacher-forced forward a
single call would do.

Each engine dump (`--dump NAME=DIR`, from `span_logits`) is scored beside the reference in 512-position
slices of the head (`score_windows`). Metrics per position: KL(ref || engine) in nats, top-1 agreement,
and the NLL each assigns the actual next token. Writes <out-dir>/<label>.json, which `gate1.py` reads.

  python ref_kld.py LABEL --windows-file W.json [--ckpt BF16] [--overlay DIR] --dump NAME=DIR ...
  python ref_kld.py LABEL --windows-file W.json [--ckpt BF16] --save-hidden REF.pt [--dump NAME=DIR ...]
  python ref_kld.py LABEL --windows-file W.json --hidden REF.pt --dump NAME=DIR ...

`--save-hidden` keeps the reference's post-norm hidden rows (0.67 GB for set A) after its forward, and
`--hidden` scores a dump made later against them with the head alone, the same bytes into the same
matmul: one forward a checkpoint serves every dump, and a ~30 GiB dump need not wait on disk for the
next one (spec #449's gates).

The harness of #340's and #422's gate 1 (`.scratch/kld-2026-09-24/ref_kld.py`, then #422's copy with
`--ckpt` and `--overlay`, a `replay.py fetch` directory laid over the checkpoint through
`tools/27b-16gb-converter/overlay.py`), brought into the repo for spec #449 (GitHub #470, #472). What
differs from #422's copy: the scoring loop is `score_windows`, the head's vocabulary its own; the
checkpoint keeps one shard open at a time; `--save-hidden` and `--hidden`; `--windows-file` is a path
as given and required, `--out-dir` replaces the script's own `results/`. The numbers are the same.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("IGNIS_KLD_CKPT", "H:/models/Qwen3.8-27B")  # Y: is too slow (owner, 2026-10-21)
# The 16 GB converter's tools beside this one: overlay.py (GitHub #422) and what it imports.
TOOLS = os.environ.get("IGNIS_CONVERTER_TOOLS", os.path.join(HERE, "..", "27b-16gb-converter"))
VOCAB = 248320
SLICE = 512
HIDDEN_FORMAT = "ignis-kld-reference-hidden-v1"


def bf16_file_rows(path, start, stop, vocab=VOCAB):
    """Rows [start, stop) of a `[len][vocab]` little-endian BF16 dump, as f32."""
    with open(path, "rb") as f:
        f.seek(start * vocab * 2)
        raw = np.frombuffer(f.read((stop - start) * vocab * 2), dtype=np.uint16)
    return torch.from_numpy(raw.astype(np.int32) << 16).view(torch.float32).reshape(stop - start, vocab)


def score_windows(windows, meta, hidden, head, dumps, tail=0):
    """Each engine dump of `dumps` (name -> a `span_logits` directory) scored against the reference whose
    post-norm hidden rows are `hidden` (one `[len][hidden]` tensor a window) and whose output head is `head`
    (`[vocab][hidden]`), in SLICE-position slices: per position KL(ref || engine) in nats, top-1 agreement,
    and the NLL each assigns the next token (none at a window's last position). One row a window."""
    vocab, dev = head.shape[0], head.device
    rows = []
    for w, (ids, m) in enumerate(zip(windows, meta)):
        n = len(ids)
        first = n - tail if tail else 0
        h = hidden[w]
        nxt = torch.tensor(ids[1:] + [ids[-1]], device=dev)
        per = {name: {"kl": [], "top1": [], "nll": []} for name in dumps}
        ref_nll = []
        with torch.inference_mode():
            for s in range(first, n, SLICE):
                e = min(n, s + SLICE)
                ref = torch.log_softmax((h[s:e] @ head.T).float(), dim=-1)
                ref_p = ref.exp()
                rt = ref.argmax(-1)
                ref_nll.append((-ref.gather(1, nxt[s:e, None])[:, 0]).cpu())
                for name, d in dumps.items():
                    rows_q = bf16_file_rows(os.path.join(d, f"w{w:03d}.bin"), s - first, e - first, vocab)
                    q = torch.log_softmax(rows_q.to(dev), dim=-1)
                    per[name]["kl"].append((ref_p * (ref - q)).sum(-1).cpu())
                    per[name]["top1"].append((q.argmax(-1) == rt).cpu())
                    per[name]["nll"].append((-q.gather(1, nxt[s:e, None])[:, 0]).cpu())
                    del q
                del ref, ref_p
        rn = torch.cat(ref_nll)[:-1]
        row = {"window": w, **m, "ref_nll": rn.tolist()}
        msg = [f"w{w:02d} {m['kind']:5s}"]
        for name in dumps:
            kl = torch.cat(per[name]["kl"])
            top1 = torch.cat(per[name]["top1"]).float()
            nll = torch.cat(per[name]["nll"])[:-1]
            row[name] = {"kl": kl.tolist(), "top1": top1.tolist(), "nll": nll.tolist()}
            msg.append(f"{name}: KLD {kl.mean():.4f} top1 {top1.mean() * 100:.1f}% ppl {nll.mean().exp():.3f}")
        msg.append(f"ref ppl {rn.mean().exp():.3f}")
        print(" | ".join(msg), flush=True)
        rows.append(row)
    return rows


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "p999": float(np.percentile(a, 99.9)), "max": float(a.max())}


class Checkpoint:
    """`root`'s tensors, with `overlay`'s laid over them when given (overlay.Reference checks it).

    One shard is open at a time, closed before the next opens: on Windows each open handle costs its
    file's size in commit charge, and the 27B's 18 shards open at once are 55.6 GB (GitHub #422)."""

    def __init__(self, root, overlay=None):
        sys.path.insert(0, TOOLS)
        from overlay import Reference
        self.reference = Reference(root, overlay)
        self.files = self.reference.files

    def tensors(self, prefix):
        by_file = {}
        for name, file in self.files.items():
            if name.startswith(prefix):
                # keyed by path: the overlay's partial shards carry the base shards' names
                by_file.setdefault(file, []).append(name)
        out = {}
        for file, names in by_file.items():
            # safetensors copies a tensor out of the file, so it outlives the handle
            with safe_open(file, framework="pt", device="cpu") as handle:
                for name in names:
                    out[name[len(prefix):]] = handle.get_tensor(name)
        return out


def windows_sha256(windows):
    return hashlib.sha256(json.dumps(windows).encode()).hexdigest()


def save_hidden(path, hidden, windows, reference):
    """A reference's post-norm hidden rows of `windows` (one `[len][hidden]` tensor a window; set A's are
    16 x 4,096 x 5,120 BF16, 0.67 GB) and what the reference is, so that a dump made later is scored against
    it with the head alone (`load_hidden`), bit for bit as the forward that saved them would score it."""
    torch.save({"format": HIDDEN_FORMAT, "windows_sha256": windows_sha256(windows), "reference": reference,
                "hidden": [h.detach().cpu() for h in hidden]}, path)


def load_hidden(path, windows):
    """The hidden rows and the reference `save_hidden` wrote to `path`, refused for windows other than `windows`."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format") != HIDDEN_FORMAT:
        raise ValueError(f"{path} is not a reference saved by --save-hidden ({HIDDEN_FORMAT})")
    if saved["windows_sha256"] != windows_sha256(windows):
        raise ValueError(f"{path} holds the reference of other windows (sha256 {saved['windows_sha256']}; "
                         f"these windows' is {windows_sha256(windows)})")
    return saved["hidden"], saved["reference"]


def read_head(reference):
    """The output head (`lm_head.weight`) of a reference `save_hidden` described, for `--hidden`.

    The head is read from the checkpoint alone, opening only its index and the shard holding `lm_head.*`,
    unless the reference's overlay replaced it (`replaced_tensors` names an `lm_head.` tensor): then the
    overlay is laid over the checkpoint as the forward did. Either way the same bytes the reference's
    forward used go into the same matmul, so the scores do not change; the overlay directory and the
    shards it replaces are not needed otherwise (a trimmed checkpoint is enough)."""
    root = reference["ckpt"]
    if any(name.startswith("lm_head.") for name in reference.get("replaced_tensors") or []):
        return Checkpoint(root, reference.get("overlay")).tensors("lm_head.")["weight"]
    index = os.path.join(root, "model.safetensors.index.json")
    if not os.path.isfile(index):
        raise FileNotFoundError(f"the reference's checkpoint {root} has no index {index}, which the output head needs")
    with open(index) as f:
        shard = json.load(f)["weight_map"]["lm_head.weight"]
    path = os.path.join(root, shard)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"the output head is in the shard {shard} of the reference's checkpoint {root} "
                                f"(index {index}), which is missing: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor("lm_head.weight")


def forward(a, windows):
    """The reference's forward over `windows`, layer by layer: its post-norm hidden rows, its output head on
    the device and what it is; None for `--smoke`, which prints the first argmaxes instead."""
    ckpt = a.ckpt or CKPT
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    from transformers import AttentionInterface

    def blocked_causal(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        """Causal attention in 512-query blocks: Windows torch has no SDPA flash
        kernel, and at 64K the math fallback wants the whole T x T score matrix
        per head (96 GiB). Same semantics as sdpa is_causal."""
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        t = query.shape[2]
        out = torch.empty_like(query)
        for q0 in range(0, t, 512):
            q1 = min(t, q0 + 512)
            cols = torch.arange(q1, device=query.device)
            rows = torch.arange(q0, q1, device=query.device)[:, None]
            mask = cols[None, :] <= rows
            out[:, :, q0:q1] = torch.nn.functional.scaled_dot_product_attention(
                query[:, :, q0:q1], key[:, :, :q1], value[:, :, :q1], attn_mask=mask, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("blocked_causal", blocked_causal)
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    cfg._attn_implementation = "blocked_causal"
    ck = Checkpoint(ckpt, a.overlay)
    r = ck.reference
    reference = {"ckpt": ckpt, "overlay": a.overlay, "replaced": len(r.replaced), "repo": r.repo,
                 "revision": r.revision, "replaced_tensors": r.replaced}
    print(f"reference: {ckpt}" + (f" + {a.overlay}: {len(r.replaced)} tensors of {r.repo}@{r.revision}"
                                     if a.overlay else ", no overlay"), flush=True)
    dev = "cpu" if a.smoke else "cuda"
    rotary = mq.Qwen3_5TextRotaryEmbedding(cfg, device=dev)

    t0 = time.time()
    embed = ck.tensors("model.language_model.embed_tokens.")["weight"].to(dev)
    hidden = [embed[torch.tensor(ids, device=dev)][None] for ids in windows]
    del embed
    torch.cuda.empty_cache()
    per_len = {}

    def geometry(w):
        n = hidden[w].shape[1]
        if n not in per_len:
            pos = torch.arange(n, device=dev).view(1, 1, -1).expand(3, 1, -1)
            text_pos = torch.arange(n, device=dev)[None]
            mask_kwargs = {"config": cfg, "inputs_embeds": hidden[w], "attention_mask": None,
                           "past_key_values": None, "position_ids": text_pos}
            per_len[n] = (rotary(hidden[w], pos), text_pos,
                          # None, not create_causal_mask's explicit mask: sdpa then runs
                          # is_causal on its flash kernel instead of materializing
                          # T x T scores per head (96 GiB at 64K).
                          {"full_attention": None,
                           "linear_attention": mq.create_recurrent_attention_mask(**mask_kwargs)})
        return per_len[n]

    for i in range(4 if a.smoke else cfg.num_hidden_layers):
        t1 = time.time()
        with torch.device("meta"):
            layer = mq.Qwen3_5DecoderLayer(cfg, i)
        layer.load_state_dict(ck.tensors(f"model.language_model.layers.{i}."), assign=True, strict=True)
        layer.to(dev).eval()
        load_s = time.time() - t1
        with torch.inference_mode():
            for w in range(len(hidden)):
                pe, text_pos, masks = geometry(w)
                out = layer(hidden[w], position_embeddings=pe, attention_mask=masks[cfg.layer_types[i]],
                            position_ids=text_pos, past_key_values=None)
                hidden[w] = out[0] if isinstance(out, tuple) else out
        del layer
        torch.cuda.empty_cache()
        print(f"layer {i:2d} {cfg.layer_types[i]:17s} load {load_s:5.1f}s run {time.time() - t1 - load_s:5.1f}s",
              flush=True)

    norm_w = ck.tensors("model.language_model.norm.")
    with torch.device("meta"):
        norm = mq.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.load_state_dict(norm_w, assign=True)
    norm.to(dev)
    head = ck.tensors("lm_head.")["weight"].to(dev)
    print(f"reference forward done in {time.time() - t0:.0f}s", flush=True)
    if a.smoke:
        h = norm(hidden[0][0])
        print("smoke: logits", (h @ head.T).float().log_softmax(-1).argmax(-1)[:8].tolist())
        return None
    with torch.inference_mode():
        for w in range(len(hidden)):
            hidden[w] = norm(hidden[w][0])
    return hidden, head, reference



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--dump", action="append", default=[], help="NAME=DIR, a span_logits dump to score")
    ap.add_argument("--windows", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="CPU, 1 window of 32 tokens, 4 layers, no scoring")
    ap.add_argument("--windows-file", required=True,
                    help="the windows' token ids (JSON); their kinds are in <name>_meta.json beside it")
    ap.add_argument("--tail", type=int, default=0, help="dumps hold only each window's last TAIL rows")
    ap.add_argument("--ckpt", help=f"the BF16 checkpoint (default IGNIS_KLD_CKPT, else {CKPT})")
    ap.add_argument("--overlay", help="a `replay.py fetch` directory laid over --ckpt (GitHub #422)")
    ap.add_argument("--save-hidden", metavar="FILE",
                    help="after the forward, save the reference's post-norm hidden rows to FILE (0.67 GB for set A)")
    ap.add_argument("--hidden", metavar="FILE",
                    help="score against the reference --save-hidden saved to FILE, with its head alone: no forward")
    ap.add_argument("--out-dir", default="results",
                    help="where <label>.json goes (default %(default)s, under the working directory)")
    a = ap.parse_args(argv)
    dumps = dict(d.split("=", 1) for d in a.dump)
    if a.hidden and (a.ckpt or a.overlay or a.save_hidden or a.smoke):
        ap.error("--hidden names the reference it was saved from: it takes no --ckpt, --overlay, --save-hidden or --smoke")
    if not dumps and not (a.smoke or a.save_hidden):
        ap.error("nothing to score: give --dump NAME=DIR")

    windows = json.load(open(a.windows_file))
    meta = json.load(open(a.windows_file.replace(".json", "_meta.json")))
    if a.windows:
        windows, meta = windows[:a.windows], meta[:a.windows]
    if a.smoke:
        windows, meta = [windows[0][:32]], meta[:1]

    torch.backends.cuda.matmul.allow_tf32 = False
    if a.hidden:
        hidden, reference = load_hidden(a.hidden, windows)
        print(f"reference: {a.hidden}, saved from {reference['ckpt']}"
              + (f" + {reference['overlay']}" if reference.get("overlay") else ", no overlay"), flush=True)
        dev = "cuda"
        hidden = [h.to(dev) for h in hidden]
        head = read_head(reference).to(dev)
        reference = dict(reference, hidden=a.hidden)
    else:
        done = forward(a, windows)
        if done is None:
            return
        hidden, head, reference = done
        if a.save_hidden:
            save_hidden(a.save_hidden, hidden, windows, reference)
            print(f"saved the reference's hidden rows to {a.save_hidden}", flush=True)
        if not dumps:
            return
    out = {"label": a.label, "dumps": list(dumps), "reference": reference,
           "windows": score_windows(windows, meta, hidden, head, dumps, a.tail)}

    summary = {}
    buckets = [] if a.tail else [(0, 512), (512, 1024), (1024, 2048), (2048, 4096)]
    for name in dumps:
        s = {}
        rows = out["windows"]
        s["all"] = summarize([v for r in rows for v in r[name]["kl"]])
        s["all"]["top1"] = float(np.mean([v for r in rows for v in r[name]["top1"]]))
        s["all"]["ppl"] = float(np.exp(np.mean([v for r in rows for v in r[name]["nll"]])))
        for kind in sorted(set(r["kind"] for r in rows)):
            kr = [r for r in rows if r["kind"] == kind]
            s[kind] = summarize([v for r in kr for v in r[name]["kl"]])
            s[kind]["top1"] = float(np.mean([v for r in kr for v in r[name]["top1"]]))
            s[kind]["ppl"] = float(np.exp(np.mean([v for r in kr for v in r[name]["nll"]])))
            s[kind]["ref_ppl"] = float(np.exp(np.mean([v for r in kr for v in r["ref_nll"]])))
        for lo, hi in buckets:
            s[f"pos {lo}-{hi}"] = summarize([v for r in rows for v in r[name]["kl"][lo:hi]])
        summary[name] = s
    summary["ref_ppl"] = float(np.exp(np.mean([v for r in out["windows"] for v in r["ref_nll"]])))
    out["summary"] = summary
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.label}.json"), "w") as f:
        json.dump(out, f)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
