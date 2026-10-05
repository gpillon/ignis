"""The conversion pass: Qwen3.8-Flash-Next, layer by layer, three streams, one pass.

Per decoder layer (the study's run 8 pipeline, `real/e2e8.py`, extended to the artifact):
1. the **BF16** stream (the checkpoint) runs every window and captures the MoE inputs
   and routing of the 2048-token chunks;
2. Hessians from the calibration tokens; every expert projection encoded at every K of
   {2, 2.5, 3, 4} by exllamav3; Lagrangian allocation; the chosen records written;
3. the non-expert tensors written (FP8 row scale or BF16, layout.md §6);
4. the **quantized** stream runs the test chunks, the long windows and the canaries on
   what was written (experts decoded from experts.bin, FP8 decoded from its payloads,
   the INT4 n-gram rows), recording its routing;
5. the **FP8-only** stream (BF16 experts, FP8 non-experts, BF16 table) runs the test chunks.

A finished layer is replayed instead (steps 1, 4, 5 from the work files). After the last
layer the head reads all three streams: stored references, KLD, MMLU proxy, traces,
the self-check and the report.
"""
import hashlib
import json
import os
import threading
import time
from collections import defaultdict

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import allocate
import corpus as corpus_mod
import fetch
import fp8
import layout
import nonexpert
import scoring
import table
import trellis

E, I, HID, HC = 512, 640, 2560, 4
CH = 2048
N_LAYERS = 48
BQ = 32
PFX = "model.language_model."
FRONTEND = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
            "preprocessor_config.json", "video_preprocessor_config.json", "config.json")
TABLE_SHARDS, SHARD_ROWS = 128, 2_500_012
HOT_CAP_ROWS = (2 << 30) // table.ROW_BYTES
CONVERTER_SCHEMA = "flash-next-converter-v1"
# kernel-level recordings for spec 02 (layout.md §10): whole-MoE-block inputs/outputs and
# full-shape reconstruct checksums at three depths
MOE_BLOCK_LAYERS, MOE_BLOCK_TOKENS, MOE_BLOCK_FIRST = (2, 24, 46), 64, 1024
CHECKSUM_LAYERS, CHECKSUMS_PER_CLASS = (0, 24, 47), 2
FALLBACK_TOKENS = 65536

# the study's per-layer MoE error (dB, held-out tokens): run 6 dA250f_fp8_t4, run 8 exl3_a25
RUN6_DB = [-19.96, -17.28, -16.26, -16.61, -15.67, -12.71, -12.91, -14.84, -14.66, -14.39, -14.42, -14.23, -13.91,
           -15.2, -13.69, -13.8, -12.97, -13.15, -13.38, -12.51, -12.53, -12.18, -12.73, -11.87, -12.9, -13.0, -12.97,
           -13.17, -11.8, -13.79, -13.32, -12.98, -12.31, -12.57, -12.47, -12.62, -13.31, -12.58, -13.47, -12.93,
           -12.35, -11.83, -12.05, -11.98, -11.64, -12.32, -13.45, -22.82]
RUN8_DB = [-20.97, -18.38, -16.86, -17.37, -16.68, -14.53]
RUN8_MEAN_1_5 = -16.8
# run 6 KLD per domain: the 2.5-bit recipe (acceptance 4's base) and the FP8-only variant
RUN6_KLD = {"code": 0.119, "prose": 0.177, "en": 0.114, "it": 0.056, "zh": 0.089, "math": 0.044, "py": 0.028,
            "de": 0.224, "ja": 0.211}
RUN6_KLD_INFO = {"chat": 0.0587, "mmlu": 0.1322}
RUN6_FP8_KLD = {"chat": 0.026, "code": 0.058, "prose": 0.084, "en": 0.029, "it": 0.015, "zh": 0.025, "math": 0.015,
                "py": 0.009, "de": 0.024, "ja": 0.022, "mmlu": 0.065}
MMLU_FLOOR = 0.71


def gb(x):
    return x / 1e9


# ---------------------------------------------------------------- study helpers (real/e2e*.py)

def expert_hessians(X, idx, w, Wgu, Wdn, experts):
    """g^2-weighted input metrics of gate/up and down for a list of experts (calibration tokens)."""
    Hg, Hd = [], []
    for e in experts:
        t, s = (idx == e).nonzero(as_tuple=True)
        g = w[t, s]
        Xe = X[t].float()
        Hg.append((Xe * g[:, None] ** 2).T @ Xe)
        y = Xe @ Wgu[e].float().T
        a = F.silu(y[:, :I]) * y[:, I:]
        Hd.append((a * g[:, None] ** 2).T @ a)
    return torch.stack(Hg), torch.stack(Hd)


def layer_H(X, mask, block=32768):
    idx = mask.nonzero()[:, 0]
    H = torch.zeros(HID, HID, device=X.device)
    for s in range(0, idx.numel(), block):
        xb = X[idx[s:s + block]].float()
        H += xb.T @ xb
    return H / max(idx.numel(), 1)


def moe_out(experts, X, idx, w, block=32768):
    return torch.cat([experts(X[s:s + block], idx[s:s + block], w[s:s + block].to(X.dtype))
                      for s in range(0, X.shape[0], block)])


def sq_norms(y, block=65536):
    return torch.cat([y[s:s + block].float().pow(2).sum(-1) for s in range(0, y.shape[0], block)])


class PLEStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cur = None

    def forward(self, input_ids, past_key_values):
        return self.cur


class IdCap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.empty(0)

    def forward(self, ids):
        return ids[..., None]


def rss_gb():
    return gb(psutil.Process().memory_info().rss)


# ---------------------------------------------------------------- the n-gram table sweep

class TableSweep(threading.Thread):
    """One pass over the table's 128 BF16 shards (the study's local cache): writes the INT4
    shard files and gathers the rows every stream needs for layer 1's PLE input."""

    def __init__(self, conv, gather):
        super().__init__(daemon=True)
        self.conv, self.gather = conv, gather
        self.error = None
        self.ple_bf16, self.ple_q = {}, {}

    def run(self):
        try:
            self._run()
        except BaseException as e:  # surfaced by join()
            self.error = e

    def _run(self):
        c = self.conv
        out_dir = os.path.join(c.work, "ngram", "table")
        os.makedirs(out_dir, exist_ok=True)
        flats, q_flats = {}, {}
        if self.gather:
            for name, ng in c.ngram_ids.items():
                flats[name] = ng.reshape(-1)
                self.ple_bf16[name] = torch.zeros(flats[name].numel(), table.DIM, dtype=torch.bfloat16)
            for name, (src, rows) in c.q_ple_sets.items():
                q_flats[name] = c.ngram_ids[src][rows].reshape(-1)
                self.ple_q[name] = torch.zeros(q_flats[name].numel(), table.DIM, dtype=torch.bfloat16)
        t0 = time.time()
        for n in range(c.table_shards):
            path = os.path.join(out_dir, f"shard_{n:03d}.int4")
            need_write = not (os.path.exists(path) and os.path.getsize(path) == SHARD_ROWS * table.ROW_BYTES)
            if not need_write and not self.gather:
                continue
            src = torch.load(os.path.join(c.args.table_cache, f"shard_{n}.weight.pt"), mmap=True, weights_only=True)
            if tuple(src.shape) != (SHARD_ROWS, table.DIM) or src.dtype != torch.bfloat16:
                raise RuntimeError(f"table shard {n}: {src.dtype} {tuple(src.shape)}")
            if need_write:
                enc = np.empty((SHARD_ROWS, table.ROW_BYTES), dtype=np.uint8)
                for s in range(0, SHARD_ROWS, 1 << 18):
                    enc[s:s + (1 << 18)] = table.encode_rows(src[s:s + (1 << 18)])
                enc.tofile(path + ".tmp")
                os.replace(path + ".tmp", path)
            else:
                enc = np.fromfile(path, dtype=np.uint8).reshape(SHARD_ROWS, table.ROW_BYTES)
            if self.gather:
                off = n * SHARD_ROWS
                for name, flat in flats.items():
                    m = (flat >= off) & (flat < off + SHARD_ROWS)
                    if m.any():
                        self.ple_bf16[name][m] = src[flat[m] - off]
                for name, flat in q_flats.items():
                    m = (flat >= off) & (flat < off + SHARD_ROWS)
                    if m.any():
                        local = (flat[m] - off).numpy()
                        self.ple_q[name][m] = table.decode_rows(enc[local]).to(torch.bfloat16)
            del src, enc
            if n % 16 == 15:
                c.log(f"table: {n + 1}/{c.table_shards} shards, {time.time() - t0:.0f}s, RSS {rss_gb():.1f} GB")
        if self.gather:
            self.ple_bf16 = {k: v.view(*c.ngram_ids[k].shape[:2], HID) for k, v in self.ple_bf16.items()}
            self.ple_q = {k: v.view(*c.ngram_ids[src][rows].shape[:2], HID)
                          for (k, v), (src, rows) in zip(self.ple_q.items(), c.q_ple_sets.values())}
        c.finish_ngram_dir()


# ---------------------------------------------------------------- the conversion

class Conversion:
    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.out = args.out
        self.work = os.path.join(args.out, "work")
        self.n_layers = args.layers
        self.table_shards = args.table_shards
        self.dev = "cuda"
        for d in ("state", "frontend", "global", "ngram", "layers"):
            os.makedirs(os.path.join(self.work, d), exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.src = fetch.Source(os.path.join(self.work, "state", "hub"), log=log)
        self._frontend()
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen4_exp import modeling_qwen4_exp as mq
        self.mq = mq
        self.cfg = AutoConfig.from_pretrained(os.path.join(self.work, "frontend")).text_config
        self.cfg._attn_implementation = "sdpa"
        self.tok = AutoTokenizer.from_pretrained(os.path.join(self.work, "frontend"))
        self._patch_indexer()
        self.corpus = corpus_mod.load(args.ood_dir, args.windows_dir)
        self._sets()
        self.routers = torch.stack([self.src.get(f"{PFX}layers.{L}.mlp.gate.weight")
                                    for L in range(N_LAYERS)]).to(self.dev)
        self._pe, self._masks = {}, {}
        self.prefetch = fetch.Prefetch(self._load_layer, depth=args.prefetch)
        self.sweep = None
        self.ple_bf16 = self.ple_q = None
        self.exl3_debug = trellis.debug_dir(self.work)

    # ------------------------------------------------------------ setup
    def _frontend(self):
        d = os.path.join(self.work, "frontend")
        if layout.is_done(d):
            return
        for f in FRONTEND:
            raw = self.src.file_bytes(f)
            with open(os.path.join(d, f), "wb") as out:
                out.write(raw)
        layout.mark_done(d)

    def _patch_indexer(self):
        """Up to the indexer's budget (2048) QSA selects every visible token, so the
        all-visible mask is exact there (the study's patch, checked bit-exact); longer
        windows run the checkpoint's own indexer."""
        mq = self.mq
        orig = getattr(mq.Qwen4ExpTextQSAIndexer, "_converter_orig", mq.Qwen4ExpTextQSAIndexer.forward)
        mq.Qwen4ExpTextQSAIndexer._converter_orig = orig

        def forward(self_, hs, pe_, am, pkv):
            if pkv is None and hs.shape[1] <= self_.token_budget:
                return torch.ones_like(am) if am.dtype == torch.bool else torch.zeros_like(am)
            return orig(self_, hs, pe_, am, pkv)
        mq.Qwen4ExpTextQSAIndexer.forward = forward

    def _sets(self):
        c = self.corpus
        self.chunks = c.chunks
        self.ids = {"chunks": torch.tensor([x["ids"] for x in c.chunks]),
                    "long": torch.tensor([x["ids"] for x in c.long])}
        self.test_sel = [i for i, x in enumerate(c.chunks) if x["test"]]
        self.test_idx = torch.tensor(self.test_sel)
        self.ids["test"] = self.ids["chunks"][self.test_idx]
        # canaries: the 27B fixture's prompts + completions, rendered with Flash-Next's template
        fixture = json.load(open(os.path.join(self.args.repo, json.load(open(corpus_mod.MANIFEST))["canary"]["file"]),
                                 encoding="utf-8"))
        self.canary = []
        for p in fixture["prompts"]:
            text = self.tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
            prompt_ids = self.tok(text, add_special_tokens=False)["input_ids"]
            comp_ids = self.tok(p["text"], add_special_tokens=False)["input_ids"]
            self.canary.append({"id": p["id"], "prompt": p["prompt"], "text": p["text"], "prompt_ids": prompt_ids,
                                "token_ids": comp_ids})
        self.canary_fixture_model = fixture["model"]
        tc = max(len(x["prompt_ids"]) + len(x["token_ids"]) for x in self.canary)
        tc = -(-tc // 64) * 64
        eos = self.cfg.eos_token_id if not isinstance(self.cfg.eos_token_id, list) else self.cfg.eos_token_id[0]
        self.eos = eos
        rows = [x["prompt_ids"] + x["token_ids"] for x in self.canary]
        self.ids["canary"] = torch.tensor([r + [eos] * (tc - len(r)) for r in rows])
        # stream -> sets it carries; set -> batch size
        self.stream_sets = {"bf16": ("chunks", "long", "canary"), "q": ("test", "long", "canary"), "f8": ("test",)}
        self.batch = {"chunks": 8, "test": 8, "long": 1, "canary": len(self.canary)}
        pos = torch.arange(CH)
        self.cal_mask = torch.cat([torch.full((CH,), bool(x["cal"])) & (pos < x["valid"]) for x in c.chunks]).to(self.dev)
        is_test = torch.zeros(len(c.chunks), dtype=torch.bool)
        is_test[self.test_idx] = True
        self.tmask = is_test.repeat_interleave(CH).to(self.dev)
        self.kind_names = sorted({c.chunks[i]["kind"] for i in self.test_sel})
        self.test_kind = torch.tensor([self.kind_names.index(c.chunks[i]["kind"]) for i in self.test_sel],
                                      device=self.dev).repeat_interleave(CH)

    def fingerprint(self):
        h = hashlib.sha256()
        h.update(json.dumps(self.run_record(), sort_keys=True).encode())
        return h.hexdigest()

    def run_record(self):
        """Everything the work tree depends on: a tree made with another value is refused."""
        man = json.load(open(corpus_mod.MANIFEST))
        return {"schema": CONVERTER_SCHEMA, "revision": fetch.REVISION, "corpus": self.corpus.manifest,
                "canary": [x["prompt_ids"] + x["token_ids"] for x in self.canary], "layers": self.n_layers,
                "table_shards": self.table_shards, "budget": self.args.budget,
                "hot_sample": man["hot_sample"], "long8192": man["long8192"]}

    def layer_dir(self, L):
        return os.path.join(self.work, "layers", f"L{L:02d}")

    def _load_layer(self, L):
        return self.src.load(f"{PFX}layers.{L}.", skip=("ple.ple_embedding.ngram_embedding.shard_",))

    # ------------------------------------------------------------ the n-gram table and globals
    def _ngram_ids(self):
        """(n, T, 16) table rows of every set, from the checkpoint's own hashing."""
        mq, cfg = self.mq, self.cfg
        pre = f"{PFX}layers.1.ple.ple_embedding."
        bufs = self.src.load(pre, skip=("ngram_embedding.shard_",))
        with torch.device("meta"):
            ng = mq.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, 1, 0)
        for k, v in bufs.items():
            ng._buffers[k] = v
        ng.ngram_embedding = IdCap()
        self._ng = ng
        out = {}
        for name in ("chunks", "long", "canary"):
            ids = self.ids[name]
            out[name] = torch.cat([ng(ids[i:i + 1], None) for i in range(ids.shape[0])])
        return bufs, out

    def finish_ngram_dir(self):
        d = os.path.join(self.work, "ngram")
        if layout.is_done(d):
            return
        for k, v in self.ngram_bufs.items():
            v.numpy().astype("<i8").tofile(os.path.join(d, f"{k}.i64"))
        cal = torch.cat([self.ngram_ids["chunks"][i, :x["valid"]] for i, x in enumerate(self.chunks)
                         if x["cal"]]).numpy().ravel()
        sample, sample_tokens = self._hot_sample_rows()
        ids = np.concatenate([cal, sample])
        hot = table.hot_rows(ids, HOT_CAP_ROWS)
        hot.astype("<u4").tofile(os.path.join(d, "hot_rows.u32"))
        counts = np.unique(ids, return_counts=True)[1]
        layout.write_json_atomic(os.path.join(d, "hot_rows.json"), {
            "rows": int(hot.size), "bytes": int(hot.size * table.ROW_BYTES), "cap_rows": HOT_CAP_ROWS,
            "calibration_lookups": int(cal.size), "sample_lookups": int(sample.size), "sample_tokens": sample_tokens,
            "lookups_covered": int(np.sort(counts)[::-1][:hot.size].sum()),
            "table_rows": TABLE_SHARDS * SHARD_ROWS, "row_bytes": table.ROW_BYTES, "shards": self.table_shards,
            "complete": self.table_shards == TABLE_SHARDS})
        layout.mark_done(os.path.join(d, "table"))
        layout.mark_done(d)

    def _hot_sample_rows(self):
        """Table rows of the n-gram coverage sample's train documents (review/ngram_coverage.py):
        each corpus up to its token budget, every 4th document held out, at most 8192 tokens each."""
        budget = json.load(open(corpus_mod.MANIFEST))["hot_sample"]["tokens_per_corpus"]
        rows, tokens = [], {}
        for name, docs in corpus_mod.hot_sample_corpora(self.args.repo, self.args.ood_dir):
            n = 0
            for i, text in enumerate(docs):
                ids = self.tok(text, add_special_tokens=False)["input_ids"][:8192]
                if len(ids) < 8:
                    continue
                if i % 4 != 3:
                    rows.append(self._ng(torch.tensor(ids)[None], None)[0].numpy().ravel())
                n += len(ids)
                if n >= budget:
                    break
            tokens[name] = n
        return np.concatenate(rows), tokens

    def _verify_table_cache(self):
        """The study's cache (fetched from `main`) must match the manifest's pins, and its first
        MiB of shards 0 and 127 the pinned revision's bytes."""
        corpus_mod.check_table_cache(self.args.table_cache, self.table_shards)
        for n in (0, TABLE_SHARDS - 1):
            name = f"{PFX}layers.1.ple.ple_embedding.ngram_embedding.shard_{n}.weight"
            f = self.src.index[name]
            s0 = self.src.meta(name)["data_offsets"][0] + self.src.headers[f]["base"]
            hub = self.src._range(f, s0, s0 + (1 << 20))
            cache = torch.load(os.path.join(self.args.table_cache, f"shard_{n}.weight.pt"), mmap=True, weights_only=True)
            if cache.view(-1)[:1 << 19].view(torch.int16).numpy().tobytes() != hub:
                raise RuntimeError(f"table cache shard {n} differs from revision {fetch.REVISION}")

    def start_table(self, need_gather):
        if need_gather or not layout.is_done(os.path.join(self.work, "ngram")):
            self._verify_table_cache()
            self.ngram_bufs, self.ngram_ids = self._ngram_ids()
            # quantized stream's PLE rows: test chunks, long windows, canaries
            self.q_ple_sets = {"test": ("chunks", self.test_idx), "long": ("long", slice(None)),
                               "canary": ("canary", slice(None))}
            self.sweep = TableSweep(self, need_gather)
            self.sweep.start()

    def _join_table(self):
        if self.sweep is None or not self.sweep.gather:
            if self.sweep is not None:
                self.sweep.join()
            self.start_table(need_gather=True)
        self.sweep.join()
        if self.sweep.error:
            raise self.sweep.error
        self.ple_bf16, self.ple_q = self.sweep.ple_bf16, self.sweep.ple_q

    def _write_globals(self):
        d = os.path.join(self.work, "global")
        if layout.is_done(d):
            return
        entries = []
        for name in ("embed_tokens.weight", "lm_head.weight", "hyper_connection_mixer.hc_norm.weight",
                     "hyper_connection_mixer.input_mix_weight_down.weight",
                     "hyper_connection_mixer.input_mix_weight_up.weight"):
            full = name if name.startswith("lm_head") else PFX + name
            t = self.src.get(full)
            entries.append(nonexpert.write_tensor(d, name, t))
            del t
        layout.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": entries})
        layout.mark_done(d)

    # ------------------------------------------------------------ initial state
    def initial_state(self):
        self._write_globals()
        g = os.path.join(self.work, "global")
        embed = self.src.get(PFX + "embed_tokens.weight")
        embed_q = fp8.decode(bytearray(open(os.path.join(g, "embed_tokens.weight.bin"), "rb").read()),
                             tuple(embed.shape))
        state = {}
        for stream, sets in self.stream_sets.items():
            src = embed if stream == "bf16" else embed_q
            for s in sets:
                state[f"{stream}.{s}"] = src[self.ids[s]].repeat(1, 1, HC)
        del embed, embed_q
        return state

    # ------------------------------------------------------------ forward helpers
    def _pe_mask(self, n, T):
        key = (n, T)
        if key not in self._pe:
            mq, cfg = self.mq, self.cfg
            dummy = torch.zeros(n, T, HID, device=self.dev, dtype=torch.bfloat16)
            p_ = torch.arange(T, device=self.dev).view(1, 1, -1).expand(4, n, -1)
            rotary = mq.Qwen4ExpTextRotaryEmbedding(cfg, device=self.dev)
            self._pe[key] = rotary(dummy, p_[1:])
            kw = dict(config=cfg, inputs_embeds=dummy, attention_mask=None, past_key_values=None,
                      position_ids=p_[0], allow_is_causal_skip=False)
            self._masks[key] = (mq.create_causal_mask(**kw), mq.create_recurrent_attention_mask(**kw))
        return self._pe[key], self._masks[key]

    def _ple(self, stream, s, b):
        if stream == "q":
            return self.ple_q[s][b]
        if s == "test":
            return self.ple_bf16["chunks"][self.test_idx[b]]
        return self.ple_bf16[s][b]

    def _run(self, layer, state, stream, s, capture=False, route=False, L=None):
        """Runs `layer` in place over a stream's set; optionally returns the MoE inputs and
        routing (capture) or the routing records (route)."""
        states = state[f"{stream}.{s}"]
        ids = self.ids[s]
        n, T = ids.shape
        bs = self.batch[s]
        cap, hooks = {}, []
        xs, ri, rw, rec = [], [], [], defaultdict(list)
        if capture or route:
            hooks.append(layer.mlp.register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0])))
            hooks.append(layer.mlp.gate.register_forward_hook(lambda m, a, o: cap.__setitem__("r", o)))
        try:
            for s0 in range(0, n, bs):
                b = list(range(s0, min(s0 + bs, n)))
                h = states[b[0]:b[-1] + 1].to(self.dev)
                if layer.ple is not None:
                    layer.ple.ple_embedding.cur = self._ple(stream, s, b).to(self.dev)
                pe, mk = self._pe_mask(len(b), T)
                out = layer(h, position_embeddings=pe, attention_mask=mk[0], conv_mask=mk[1],
                            past_key_values=None, ple_input_ids=ids[b[0]:b[-1] + 1].to(self.dev))
                states[b[0]:b[-1] + 1] = out.to("cpu")
                del h, out
                if capture:
                    xs.append(cap["x"].reshape(-1, HID))
                    ri.append(cap["r"][2])
                    rw.append(cap["r"][1].float())
                if route:
                    x = cap["x"].reshape(-1, HID)
                    rec["experts"].append(cap["r"][2].to(torch.int16).cpu())
                    rec["weights"].append(cap["r"][1].to(torch.float16).cpu())
                    if L + 1 < N_LAYERS:
                        la = torch.topk(F.linear(x, self.routers[L + 1]), 20, dim=-1).indices.to(torch.int16)
                    else:
                        la = torch.full((x.shape[0], 20), -1, dtype=torch.int16, device=x.device)
                    rec["lookahead"].append(la.cpu())
                cap.clear()
        finally:
            for hk in hooks:
                hk.remove()
        if capture:
            return torch.cat(xs), torch.cat(ri), torch.cat(rw)
        if route:
            return {k: torch.cat(v).numpy() for k, v in rec.items()}

    def _build(self, L, sd):
        mq = self.mq
        with torch.device("meta"):
            layer = mq.Qwen4ExpTextDecoderLayer(self.cfg, L)
        if layer.ple is not None:
            layer.ple.ple_embedding = PLEStub()
            sd = {k: v for k, v in sd.items() if not k.startswith("ple.ple_embedding.")}
            self._join_table()
        missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"layer {L}: missing {missing} unexpected {unexpected}")
        return layer.to(self.dev, torch.bfloat16).eval()

    def _swap_fp8(self, layer, weights):
        """Puts the FP8-decoded weights in place; returns a restore function."""
        saved = []
        for name, w in weights.items():
            mod = layer.get_submodule(name[:-len(".weight")])
            saved.append((mod, mod.weight.data))
            mod.weight.data = w

        def restore():
            for mod, w0 in saved:
                mod.weight.data = w0
        return restore

    def _decode_experts(self, d, L, convert):
        """The layer's expert weights as decoded from its experts.bin (bf16, HF layout). When
        converting, also the self-check samples (first expert of each K class: sha256 of its
        decoded bf16 weight) and, at CHECKSUM_LAYERS, kern's reconstruct checksums."""
        index = layout.read_index(os.path.join(d, "experts.idx"))
        path = os.path.join(d, "experts.bin")
        gq = torch.empty((E, 2 * I, HID), dtype=torch.bfloat16, device=self.dev)
        dq = torch.empty((E, HID, I), dtype=torch.bfloat16, device=self.dev)
        sample, checks = [], []
        for en in index:
            rec = layout.read_record(path, en)
            w = trellis.decode(rec, en.k2, en.proj, self.dev).to(torch.bfloat16)
            (gq if en.proj == "gu" else dq)[en.expert] = w
            if not convert:
                continue
            cls = f"{en.proj}-{en.k2 / 2:g}"
            if not any(x["class"] == cls for x in sample):
                sample.append({"layer": L, "class": cls, "proj": en.proj, "k2": en.k2, "expert": en.expert,
                               "sha256": hashlib.sha256(w.cpu().view(torch.int16).numpy().tobytes()).hexdigest()})
            same = sum(x["proj"] == en.proj and x["k2"] == en.k2 for x in checks)
            if L in CHECKSUM_LAYERS and same < CHECKSUMS_PER_CLASS:
                raw = trellis.reconstruct_raw(rec, en.k2, en.proj, self.dev)
                checks.append({"layer": L, "expert": en.expert, "proj": en.proj, "k2": en.k2,
                               "checksum": f"{trellis.checksum_u16(raw):016x}"})
        return gq, dq, sample, checks

    # ------------------------------------------------------------ one layer
    def process(self, L, state):
        self._layer(L, state, convert=True)

    def replay(self, L, state):
        self._layer(L, state, convert=False)

    def _layer(self, L, state, convert):
        t0 = time.time()
        tm = defaultdict(float)
        torch.cuda.reset_peak_memory_stats()
        sd = self.prefetch.get(L, self.n_layers)
        tm["fetch_wait"] = time.time() - t0
        layer = self._build(L, sd)
        del sd
        d = self.layer_dir(L)
        os.makedirs(d, exist_ok=True)
        experts = layer.mlp.experts
        Wgu0, Wdn0 = experts.gate_up_proj.data, experts.down_proj.data
        rec = {"layer": L, "layer_type": layer.layer_type}
        with torch.no_grad():
            ta = time.time()
            cap = self._run(layer, state, "bf16", "chunks", capture=convert)
            for s in self.stream_sets["bf16"][1:]:
                self._run(layer, state, "bf16", s)
            tm["fwd_bf16"] = time.time() - ta
            if convert:
                X, ridx, rw = cap
                del cap
                moe = self._convert_experts(L, layer, X, ridx, rw, d, rec, tm)
                del X, ridx, rw
                torch.cuda.empty_cache()
                ta = time.time()
                entries = [nonexpert.write_tensor(d, name, t, prefix=f"layers.{L}.") for name, t in layer.state_dict().items()
                           if layout.encoding_of(name, tuple(t.shape)) != "expert"]
                layout.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": entries})
                tm["write"] += time.time() - ta
            ta = time.time()
            gq, dq, sample, checks = self._decode_experts(d, L, convert)
            fp8w = nonexpert.fp8_weights(d, self.dev)
            tm["decode"] = time.time() - ta
            if convert:
                self._moe_error(experts, gq, dq, moe, rec)
                del moe
                rec["decode_sha256"] = sample
                rec["trellis_checksums"] = checks
            # quantized stream
            ta = time.time()
            experts.gate_up_proj.data, experts.down_proj.data = gq, dq
            restore = self._swap_fp8(layer, fp8w)
            routes = {}
            if convert and L in MOE_BLOCK_LAYERS:
                self._record_moe_block(layer, state, L, d)
            for s in self.stream_sets["q"]:
                r = self._run(layer, state, "q", s, route=convert and s in ("test", "long"), L=L)
                if r is not None:
                    routes[s] = r
            # FP8-only stream: BF16 experts, FP8 non-experts, BF16 n-gram rows
            experts.gate_up_proj.data, experts.down_proj.data = Wgu0, Wdn0
            del gq, dq
            for s in self.stream_sets["f8"]:
                self._run(layer, state, "f8", s)
            restore()
            tm["fwd_q_f8"] = time.time() - ta
        if convert:
            for s, r in routes.items():
                for k, v in r.items():
                    np.save(os.path.join(d, f"route_{s}_{k}.npy"), v)
        if L == 1 and self.ple_bf16 is not None:
            self.ple_bf16 = self.ple_q = None    # the n-gram rows feed layer 1 only
            if self.sweep is not None:
                self.sweep.ple_bf16 = self.sweep.ple_q = None
        del layer, experts, Wgu0, Wdn0, fp8w
        torch.cuda.empty_cache()
        rec["time_s"] = time.time() - t0
        rec["time_split_s"] = dict(tm)
        rec["rss_gb"] = rss_gb()
        rec["vram_peak_gb"] = gb(torch.cuda.max_memory_reserved())
        if convert:
            rec["experts_bin"] = {"bytes": os.path.getsize(os.path.join(d, "experts.bin")),
                                  "sha256": layout.file_digest(os.path.join(d, "experts.bin"))}
            layout.write_json_atomic(os.path.join(d, "layer.json"), rec)
            layout.mark_done(d, {"experts.bin": rec["experts_bin"]["sha256"]})
            self.log(self._layer_line(rec))
        else:
            self.log(f"layer {L} replayed in {rec['time_s']:.0f}s | RSS {rec['rss_gb']:.1f} GB | "
                     f"VRAM peak {rec['vram_peak_gb']:.1f} GB")

    def _layer_line(self, r):
        L = r["layer"]
        ref6 = RUN6_DB[L]
        ref8 = f"{RUN8_DB[L]:.2f}" if L < len(RUN8_DB) else "-"
        return (f"layer {L} {r['time_s']:.0f}s | RSS {r['rss_gb']:.1f} GB | VRAM peak {r['vram_peak_gb']:.1f} GB | "
                f"MoE {r['moe_db']:.2f} dB (run6 {ref6:.2f}, run8 {ref8}) | rate gu/dn {r['rates']['gu']:.4f}/"
                f"{r['rates']['dn']:.4f} | K hist gu {r['k_hist']['gu']} dn {r['k_hist']['dn']} | "
                + " ".join(f"{k} {v:.0f}" for k, v in r["time_split_s"].items()))

    def _convert_experts(self, L, layer, X, ridx, rw, d, rec, tm):
        experts = layer.mlp.experts
        Wgu0, Wdn0 = experts.gate_up_proj.data, experts.down_proj.data
        ta = time.time()
        Xte, ite, wte = X[self.tmask], ridx[self.tmask], rw[self.tmask]
        idx_cal = ridx.clone()
        idx_cal[~self.cal_mask] = -1
        Hl = layer_H(X, self.cal_mask)
        y_ref = moe_out(experts, Xte, ite, wte)
        r2 = sq_norms(y_ref)
        rec["expert_traffic"] = torch.bincount(ridx[self.cal_mask].reshape(-1), minlength=E).tolist()
        tm["hess"] += time.time() - ta
        store = {p: [[None] * E for _ in layout.K2_SET] for p in ("gu", "dn")}
        Dv = {p: torch.empty(E, len(layout.K2_SET), device=self.dev) for p in ("gu", "dn")}
        en = {p: torch.empty(E, device=self.dev) for p in ("gu", "dn")}
        eye = torch.eye(I, device=self.dev)
        for b0 in range(0, E, BQ):
            sl = slice(b0, b0 + BQ)
            ta = time.time()
            Hg, Hd = expert_hessians(X, idx_cal, rw, Wgu0, Wdn0, list(range(b0, b0 + BQ)))
            Qg, Qd = self._fallback_hessians(X, Hl, Wgu0, Hg, Hd, b0, rec)
            mats = {"gu": (Wgu0[sl].float(), Hg, trellis.shrink(Qg, Hl.expand(BQ, -1, -1))),
                    "dn": (Wdn0[sl].float(), Hd, trellis.shrink(Qd, eye.expand(BQ, -1, -1)))}
            tm["hess"] += time.time() - ta
            for mi, (p, (W, H, Hs)) in enumerate(mats.items()):
                en[p][sl] = torch.einsum("bmi,bij,bmj->b", W, H, W)
                hds = trellis.hessian_data(Hs, [f"L{L}.{p}.{b0 + b}" for b in range(BQ)], self.dev)
                for ki, k2 in enumerate(layout.K2_SET):
                    torch.cuda.synchronize()
                    ta = time.time()
                    recs = trellis.quantize_batch(W, hds, k2, [L * 10000 + mi * 1000 + b0 + b for b in range(BQ)],
                                                  self.exl3_debug, self.dev)
                    torch.cuda.synchronize()
                    tm["quant"] += time.time() - ta
                    ta = time.time()
                    for b in range(BQ):
                        dW = W[b] - trellis.decode(recs[b], k2, p, self.dev)
                        Dv[p][b0 + b, ki] = (dW @ H[b] * dW).sum() / en[p][b0 + b].clamp(min=1e-30)
                        store[p][ki][b0 + b] = recs[b]
                    tm["dec"] += time.time() - ta
                del hds
            del mats, Hg, Hd
        picks = {p: allocate.allocate(en[p], Dv[p], p, self.args.budget) for p in ("gu", "dn")}
        ta = time.time()
        _, digests = layout.write_experts(
            d, E, lambda e, p: (picks[p][e], store[p][layout.K2_SET.index(picks[p][e])][e]))
        tm["write"] += time.time() - ta
        del store
        rec["k2"] = picks
        rec["k_hist"] = {p: [picks[p].count(k) for k in layout.K2_SET] for p in ("gu", "dn")}
        rec["rates"] = {p: sum(layout.rate(p, k) for k in picks[p]) / E for p in ("gu", "dn")}
        rec["stored_rates"] = {p: sum(layout.stored_rate(p, k) for k in picks[p]) / E for p in ("gu", "dn")}
        rec["curve_db"] = {p: (10 * Dv[p][en[p] > 0].log10().mean(0)).tolist() for p in ("gu", "dn")}
        rec["unrouted"] = {p: int((en[p] <= 0).sum()) for p in ("gu", "dn")}
        rec["record_sha256"] = digests
        return {"Xte": Xte, "ite": ite, "wte": wte, "y_ref": y_ref, "r2": r2}

    def _record_moe_block(self, layer, state, L, d):
        """Spec 02's whole-MoE-block reference: MOE_BLOCK_TOKENS tokens of the first test chunk
        through this layer's quantized MoE block (layout.md §10, moe_block/). Runs the layer on a
        copy of that chunk's state, so the stream itself is untouched."""
        cap = {}
        hooks = [layer.mlp.register_forward_hook(lambda m, a, o: cap.update(x=a[0], y=o)),
                 layer.mlp.gate.register_forward_hook(lambda m, a, o: cap.__setitem__("r", o))]
        try:
            h = state["q.test"][0:1].to(self.dev)
            if layer.ple is not None:
                layer.ple.ple_embedding.cur = self._ple("q", "test", [0]).to(self.dev)
            pe, mk = self._pe_mask(1, CH)
            layer(h, position_embeddings=pe, attention_mask=mk[0], conv_mask=mk[1], past_key_values=None,
                  ple_input_ids=self.ids["test"][0:1].to(self.dev))
        finally:
            for hk in hooks:
                hk.remove()
        sl = slice(MOE_BLOCK_FIRST, MOE_BLOCK_FIRST + MOE_BLOCK_TOKENS)
        x = cap["x"][0, sl]
        mlp = layer.mlp
        shared = torch.sigmoid(mlp.shared_expert_gate(x)) * mlp.shared_expert(x)
        T = cap["x"].shape[1]
        rows = slice(MOE_BLOCK_FIRST, MOE_BLOCK_FIRST + MOE_BLOCK_TOKENS)   # router outputs are (tokens, k)
        out = os.path.join(d, "moe_block")
        os.makedirs(out, exist_ok=True)
        x.contiguous().view(torch.int16).cpu().numpy().astype("<i2").tofile(os.path.join(out, "x.bf16"))
        cap["y"][0, sl].float().cpu().numpy().astype("<f4").tofile(os.path.join(out, "y.f32"))
        assert cap["r"][2].shape[0] == T
        cap["r"][2][rows].to(torch.int32).cpu().numpy().astype("<i4").tofile(os.path.join(out, "ids.i32"))
        cap["r"][1][rows].float().cpu().numpy().astype("<f4").tofile(os.path.join(out, "weights.f32"))
        shared.float().cpu().numpy().astype("<f4").tofile(os.path.join(out, "shared.f32"))
        c = self.chunks[self.test_sel[0]]
        layout.write_json_atomic(os.path.join(out, "manifest.json"), {
            "layer": L, "tokens": MOE_BLOCK_TOKENS, "first_position": MOE_BLOCK_FIRST, "test_chunk": 0,
            "kind": c["kind"], "source": c["source"], "revision": fetch.REVISION, "stream": "quantized",
            "files": {"x.bf16": "(N, 2560) bf16, the block's input", "y.f32": "(N, 2560) block output, bf16 upcast",
                      "ids.i32": "(N, 10) router top-10, descending", "weights.f32": "(N, 10) routing weights",
                      "shared.f32": "(N, 2560) sigmoid(shared_expert_gate(x)) * shared_expert(x)"}})

    def _fallback_hessians(self, X, Hl, Wgu, Hg, Hd, b0, rec):
        """Spec 01: an expert no calibration token was routed to is quantized with the layer's
        Hessian, not left to exllamav3's uncalibrated fallback (the study's run 8 did the latter).
        Gate/up takes the layer H; down takes its activations' second moment over the first
        FALLBACK_TOKENS calibration tokens as if they were all routed to it. The allocation still
        sees zero routed energy, so such an expert gets K = 2 as in run 8."""
        empty = [b for b in range(Hg.shape[0]) if not torch.any(Hg[b])]
        if not empty:
            return Hg, Hd
        Qg, Qd = Hg.clone(), Hd.clone()
        xs = X[self.cal_mask.nonzero()[:FALLBACK_TOKENS, 0]].float()
        for b in empty:
            y = xs @ Wgu[b0 + b].float().T
            a = F.silu(y[:, :I]) * y[:, I:]
            Qg[b] = Hl
            Qd[b] = a.T @ a / a.shape[0]
            rec.setdefault("hessian_fallback", []).append(b0 + b)
        return Qg, Qd

    def _moe_error(self, experts, gq, dq, m, rec):
        """Run 8's MoE error on held-out tokens, with the experts decoded from the written file."""
        W0 = experts.gate_up_proj.data, experts.down_proj.data
        experts.gate_up_proj.data, experts.down_proj.data = gq, dq
        y = moe_out(experts, m["Xte"], m["ite"], m["wte"])
        experts.gate_up_proj.data, experts.down_proj.data = W0
        e2 = torch.cat([(y[s:s + 65536].float() - m["y_ref"][s:s + 65536].float()).pow(2).sum(-1)
                        for s in range(0, y.shape[0], 65536)])
        rel = float(e2.sum() / m["r2"].sum())
        rec["moe_rel_err"] = rel
        rec["moe_db"] = scoring.db(rel)
        rec["moe_db_kind"] = {k: scoring.db(float(e2[self.test_kind == i].sum() / m["r2"][self.test_kind == i].sum()))
                              for i, k in enumerate(self.kind_names)}
        L = rec["layer"]
        rec["run6_db"] = RUN6_DB[L]
        rec["run8_db"] = RUN8_DB[L] if L < len(RUN8_DB) else None
