"""Qwen3.8-Flash-Next's MTP head as a measurement prototype (spec flash-next/07 phase A).

Everything but the input combine and the chaining is the checkpoint's own math, run by
transformers' own modules: the MTP block is `Qwen4ExpTextDecoderLayer(cfg', 0)` (cfg' = the
text config with one full-attention layer), the mixer `Qwen4ExpTextGatedResidual(cfg,
use_combine=False)`, the head the trunk's `lm_head`. What this file writes by hand:

- the combine of the trunk's pre-mixer stack `S_p` [streams * hidden] with the next token's
  embedding, in the spec's candidate conventions (C-comb a/b x C-norm a/b);
- the layer's attention split in two, so one pass over a window can serve many queries and a
  chained draft can attend to the window's entries plus its own: `entries` projects every entry
  (q, gate, k, v, the indexer's q and raw key), `finish` attends for a subset of queries and
  runs the rest of the layer. The QSA indexer is the HF indexer's rule (pooled blocks of 4
  visible entries, ReLU head-sum scores, top budget/4 blocks plus the tail), vectorized, with
  the block boundary offset as a parameter (the spec's position row).

`test_mtp.py` holds this to the HF layer itself on a tiny random config: a causal pass and a
chained step are the HF layer over the same sequence.
"""
import math

import torch
import torch.nn.functional as F
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

COMBS = ("a", "b")    # a: fc_hidden per normed stream, embedding broadcast; b: streams averaged
NORMS = ("a", "b")    # a: RMSNorm grouped per stream; b: one RMSNorm over all streams
CHAINS = ("a", "b")   # a: next stack = the block's own pre-mixer stack; b: post-mixer state x4
MODES = ("dense", "index")  # attention over every visible entry, or the indexer's selection


def layer_config(text_config, attn="sdpa"):
    """cfg': the text config with one full-attention layer (the MTP block's shape)."""
    # No PLE: the MTP block has none (layer_idx + 1 = 1 is not a PLE layer of the trunk).
    cfg = Qwen4ExpTextConfig(**{**text_config, "layer_types": ["full_attention"], "num_hidden_layers": 1,
                                "ple_layer_ids": []})
    cfg._attn_implementation = attn
    cfg._experts_implementation = "eager"
    return cfg


def rms(x, w, eps, group=None):
    """The family's RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w) in fp32, back to x's dtype."""
    y = x.float()
    if group is not None:
        y = y.reshape(*y.shape[:-1], -1, group)
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    if group is not None:
        y = y.flatten(-2)
    return (y * (1.0 + w.float())).to(x.dtype)


class Head:
    """The MTP head on `weights` (the checkpoint's `mtp.*` tensors without the prefix), the
    trunk's `embed` [vocab, hidden] and `lm_head` [vocab, hidden]."""

    def __init__(self, cfg, weights, embed, lm_head, device="cpu", dtype=torch.bfloat16):
        self.cfg = cfg
        self.H = cfg.hidden_size
        self.S = cfg.hc_count
        self.eps = cfg.rms_norm_eps
        # Built on the meta device and given the checkpoint's tensors, so the 5 GB of experts
        # are never allocated a second time at the default dtype.
        with torch.device("meta"):
            layer = hf.Qwen4ExpTextDecoderLayer(cfg, 0)
            mixer = hf.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        layer.load_state_dict({k[len("layers.0."):]: v for k, v in weights.items() if k.startswith("layers.0.")},
                              assign=True)
        mixer.load_state_dict({k[len("hyper_connection_mixer."):]: v for k, v in weights.items()
                               if k.startswith("hyper_connection_mixer.")}, assign=True)
        self.layer = layer.to(device=device, dtype=dtype).eval()
        self.mixer = mixer.to(device=device, dtype=dtype).eval()
        self.rotary = hf.Qwen4ExpTextRotaryEmbedding(cfg).to(device)
        put = lambda t: t.to(device=device, dtype=dtype)
        self.fc_hidden = put(weights["fc_hidden.weight"])
        self.fc_embedding = put(weights["fc_embedding.weight"])
        self.norm_hidden = put(weights["pre_fc_norm_hidden.weight"])
        self.norm_embedding = put(weights["pre_fc_norm_embedding.weight"])
        self.embed = put(embed)
        self.lm_head = put(lm_head)
        self.device = device
        self.dtype = dtype
        attn = self.layer.self_attn
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = attn.head_dim
        self.scale = attn.scaling
        ix = attn.indexer
        self.ix_heads, self.ix_dim = ix.index_n_heads, ix.index_head_dim
        self.ix_ratio, self.ix_blocks = ix.compress_ratio, ix.block_topk

    # ------------------------------------------------------------------ combine
    def combine(self, stack, tokens, comb, norm):
        """The block's input for entries built from `stack` [N, S*H] and `tokens` [N]."""
        e = rms(self.embed[tokens], self.norm_embedding, self.eps)
        e = F.linear(e, self.fc_embedding).float()
        n = rms(stack, self.norm_hidden, self.eps, group=self.H if norm == "a" else None)
        n = n.view(-1, self.S, self.H)
        if comb == "a":
            x = F.linear(n, self.fc_hidden).float() + e[:, None]
        else:
            x = (F.linear(n.float().mean(1).to(self.dtype), self.fc_hidden).float() + e)[:, None].expand(-1, self.S, -1)
        return x.to(self.dtype).reshape(-1, self.S * self.H)

    def chain_stack(self, stack, mixed, chain):
        """The next draft step's hidden input after a step that left `stack` / `mixed`."""
        return stack if chain == "a" else mixed.repeat(1, self.S)

    # ------------------------------------------------------------------ entries
    def rope(self, positions):
        pos = positions.to(self.device).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self.rotary(self.lm_head, pos)
        return cos[0], sin[0]

    def entries(self, X, positions):
        """Every projection of entries X [N, S*H] at `positions` [N] the attention needs."""
        attn = self.layer.self_attn
        x, Xres, inj = self.layer.attn_hyper_connection(X)
        n = X.shape[0]
        cos, sin = self.rope(positions)
        q, gate = torch.chunk(attn.q_proj(x).view(n, self.heads, 2 * self.head_dim), 2, dim=-1)
        q = attn.q_norm(q)
        k = attn.k_norm(attn.k_proj(x).view(n, self.kv_heads, self.head_dim))
        v = attn.v_proj(x).view(n, self.kv_heads, self.head_dim)
        q, k = hf.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        ix = attn.indexer
        iq, raw = torch.split(ix.index_qk_proj(x), [self.ix_heads * self.ix_dim, self.ix_dim], dim=-1)
        iq = hf.apply_rotary_pos_emb(ix.q_layernorm(iq.reshape(n, self.ix_heads, self.ix_dim)), None, cos, sin,
                                     unsqueeze_dim=1)
        return dict(Xres=Xres, inj=inj, q=q, gate=gate.reshape(n, -1), k=k, v=v, iq=iq, raw=raw,
                    pos=positions.to(self.device))

    # ------------------------------------------------------------------ indexer
    def trunk_blocks(self, E, off):
        """Pooled keys of the window's blocks of entries [4b - off, 4b + 3 - off] (the first one
        shorter when off > 0), and each entry's block id."""
        n = E["raw"].shape[0]
        r = self.ix_ratio
        block = (torch.arange(n, device=self.device) + off) // r
        nb = int(block[-1]) + 1
        firsts = torch.clamp(torch.arange(nb, device=self.device) * r - off, min=0)
        sums = torch.zeros(nb, self.ix_dim, device=self.device).index_add_(0, block, E["raw"].float())
        counts = torch.bincount(block, minlength=nb).float()[:, None]
        k = self.layer.self_attn.indexer.k_layernorm((sums / counts).to(self.dtype))
        cos, sin = self.rope(E["pos"][firsts])
        return dict(keys=hf.apply_rotary_pos_emb(k[:, None], None, cos, sin, unsqueeze_dim=1)[:, 0],
                    block=block, off=off)

    def select(self, iq, last, blocks, chain_raw):
        """Which trunk entries [B, N] and chain entries [B, c] each query's QSA reads, by the
        indexer's rule: the top budget/4 complete blocks of visible entries, plus the tail.

        Query b sees trunk entries 0..last[b] and its own chain entries (storage last[b]+1..)."""
        r, off = self.ix_ratio, blocks["off"]
        B = iq.shape[0]
        c = 0 if chain_raw is None else chain_raw.shape[1]
        n = blocks["block"].shape[0]
        nbt = (last + 1 + off) // r                       # complete trunk-only blocks
        nb = (last + 1 + c + off) // r                    # complete blocks, chain entries included
        keys = blocks["keys"]
        scores = torch.relu(torch.einsum("bhd,kd->bhk", iq.float(), keys.float())).sum(1) / math.sqrt(self.ix_dim)
        nk = keys.shape[0]
        scores = scores.masked_fill(torch.arange(nk, device=self.device)[None] >= nbt[:, None], -math.inf)
        mixed = nb > nbt                                  # one complete block holds chain entries
        if c:
            first = nbt * r - off                         # the mixed block's first entry
            idx = first[:, None] + torch.arange(r, device=self.device)[None]          # [B, r]
            trunk_part = idx <= last[:, None]
            raw_t = self._raw_trunk[idx.clamp(0, n - 1)]                               # [B, r, d]
            raw_c = chain_raw.gather(1, (idx - last[:, None] - 1).clamp(0, c - 1)[..., None].expand(-1, -1, self.ix_dim))
            raw_m = torch.where(trunk_part[..., None], raw_t, raw_c)
            valid = idx >= 0
            mean = (raw_m.float() * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
            km = self.layer.self_attn.indexer.k_layernorm(mean.to(self.dtype))
            cos, sin = self.rope(first.clamp(min=0))       # the block's first real entry
            km = hf.apply_rotary_pos_emb(km[:, None], None, cos, sin, unsqueeze_dim=1)[:, 0]
            s_m = torch.relu(torch.einsum("bhd,bd->bh", iq.float(), km.float())).sum(1) / math.sqrt(self.ix_dim)
            s_m = s_m.masked_fill(~mixed, -math.inf)
            scores = torch.cat([scores, s_m[:, None]], dim=1)
        k = min(self.ix_blocks, scores.shape[1])
        top = scores.topk(k, dim=1)
        chosen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, top.indices, top.values > -math.inf)
        e = torch.arange(n, device=self.device)
        eb = blocks["block"]
        trunk = chosen[:, :nk].gather(1, eb[None].expand(B, -1)) & (eb[None] < nbt[:, None])
        mixed_chosen = chosen[:, nk] if c else torch.zeros(B, dtype=torch.bool, device=self.device)
        # Entries past the trunk-only blocks: the mixed block's (if complete and chosen) or the tail.
        rest = (eb[None] >= nbt[:, None]) & (eb[None] < nb[:, None])
        tail = eb[None] >= nb[:, None]
        trunk = trunk | (rest & mixed_chosen[:, None]) | tail
        trunk &= e[None] <= last[:, None]
        if not c:
            return trunk, None
        s = last[:, None] + 1 + torch.arange(c, device=self.device)[None]
        cb = (s + off) // r
        chain = ((cb < nb[:, None]) & mixed_chosen[:, None]) | (cb >= nb[:, None])
        return trunk, chain

    # ------------------------------------------------------------------ attention + rest of the layer
    def attend(self, q, gate, E, trunk_mask, ck=None, cv=None, chain_mask=None):
        """QSA output [B, H] for queries q [B, heads, D] over the trunk entries `trunk_mask` and
        the query's own chain entries ck/cv [B, c, kv_heads, D]."""
        B = q.shape[0]
        g = self.heads // self.kv_heads
        qg = q.float().view(B, self.kv_heads, g, self.head_dim)
        s = torch.einsum("bkgd,nkd->bkgn", qg, E["k"].float()) * self.scale
        s = s.masked_fill(~trunk_mask[:, None, None], -math.inf)
        if ck is not None:
            sc = torch.einsum("bkgd,bckd->bkgc", qg, ck.float()) * self.scale
            sc = sc.masked_fill(~chain_mask[:, None, None], -math.inf)
            s = torch.cat([s, sc], dim=-1)
        p = torch.softmax(s, dim=-1)
        n = E["k"].shape[0]
        out = torch.einsum("bkgn,nkd->bkgd", p[..., :n], E["v"].float())
        if ck is not None:
            out = out + torch.einsum("bkgc,bckd->bkgd", p[..., n:], cv.float())
        out = out.reshape(B, -1).to(self.dtype) * torch.sigmoid(gate)
        return self.layer.self_attn.o_proj(out)

    def finish(self, rows, attn_out):
        """The rest of the block after the attention, for entries `rows` (a dict of [B]-row
        slices of `entries`): the inject, the MoE sublayer, the mixer. Returns the block's
        pre-mixer stack and the mixed state."""
        X = rows["Xres"] + (attn_out[:, None, :] * rows["inj"][..., None]).flatten(-2)
        x, Xres, inj = self.layer.mlp_hyper_connection(X)
        y = self.layer.mlp(x[None])[0]
        stack = Xres + (y[:, None, :] * inj[..., None]).flatten(-2)
        return stack, self.mixer(stack)

    def logits(self, mixed):
        return F.linear(mixed, self.lm_head)

    # ------------------------------------------------------------------ one window
    def window(self, E, blocks, mode):
        """Keep the window's entries and blocks for `step`."""
        self._E, self._blocks, self._mode = E, blocks, mode
        self._raw_trunk = E["raw"]

    def first_step(self, idx):
        """Draft 1 for entries `idx` [B] of the window (the causal pass)."""
        E = self._E
        rows = {k: v[idx] for k, v in E.items()}
        if self._mode == "dense":
            trunk = torch.arange(E["k"].shape[0], device=self.device)[None] <= idx[:, None]
        else:
            trunk, _ = self.select(rows["iq"], idx, self._blocks, None)
        a = self.attend(rows["q"], rows["gate"], E, trunk)
        return self.finish(rows, a)

    def chain_step(self, X, last, chain_k, chain_v, chain_raw):
        """A chained draft: entries X [B, S*H] stored after `last` [B] and the chain entries the
        query's earlier steps wrote (chain_* [B, c-1, ...] or None). Returns the stack, the mixed
        state and this step's own (k, v, raw) to append to the chain."""
        c_prev = 0 if chain_k is None else chain_k.shape[1]
        pos = last + 1 + c_prev
        rows = self.entries(X, pos)
        ck = rows["k"][:, None] if chain_k is None else torch.cat([chain_k, rows["k"][:, None]], 1)
        cv = rows["v"][:, None] if chain_v is None else torch.cat([chain_v, rows["v"][:, None]], 1)
        cr = rows["raw"][:, None] if chain_raw is None else torch.cat([chain_raw, rows["raw"][:, None]], 1)
        E = self._E
        if self._mode == "dense":
            trunk = torch.arange(E["k"].shape[0], device=self.device)[None] <= last[:, None]
            chain = torch.ones(ck.shape[:2], dtype=torch.bool, device=self.device)
        else:
            trunk, chain = self.select(rows["iq"], last, self._blocks, cr)
        a = self.attend(rows["q"], rows["gate"], E, trunk, ck, cv, chain)
        stack, mixed = self.finish(rows, a)
        return stack, mixed, (ck, cv, cr)
