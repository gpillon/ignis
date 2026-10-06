"""The prototype's split layer is transformers' own layer, on a tiny random config (CPU, fp32).

A causal pass of `first_step` over a window, and a chain of `chain_step`s after an entry, are
both the HF `Qwen4ExpTextDecoderLayer` over one sequence: the window's entries up to the last
one, then the chain's. The window is long enough for the indexer to select (budget 8 tokens,
blocks of 4), so its vectorized rule is held to HF's per-query loop, a chain entry inside a
complete block included.
"""
import torch
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

import mtp

TEXT = dict(
    hidden_size=32, hc_count=4, hc_lowrank=8, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=4,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=16,
    vocab_size=64, ple_layer_ids=[2], rms_norm_eps=1e-6,
)


def build(seed=0):
    torch.manual_seed(seed)
    cfg = mtp.layer_config(TEXT)
    ref = hf.Qwen4ExpTextDecoderLayer(cfg, 0)
    mixer = hf.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    with torch.no_grad():
        for module in (ref, mixer):
            for p in module.parameters():
                p.copy_(torch.randn_like(p) * 0.2)
    H, S, V = cfg.hidden_size, cfg.hc_count, cfg.vocab_size
    weights = {"layers.0." + k: v for k, v in ref.state_dict().items()}
    weights.update({"hyper_connection_mixer." + k: v for k, v in mixer.state_dict().items()})
    weights.update({"fc_hidden.weight": torch.randn(H, H) * 0.2, "fc_embedding.weight": torch.randn(H, H) * 0.2,
                    "pre_fc_norm_hidden.weight": torch.randn(S * H) * 0.1,
                    "pre_fc_norm_embedding.weight": torch.randn(H) * 0.1})
    head = mtp.Head(cfg, weights, torch.randn(V, H), torch.randn(V, H), dtype=torch.float32)
    return cfg, ref.eval(), head


def hf_layer(ref, head, X):
    n = X.shape[0]
    cos, sin = head.rope(torch.arange(n))
    mask = torch.ones(n, n, dtype=torch.bool).tril()[None, None]
    with torch.no_grad():
        return ref(X[None], position_embeddings=(cos[None], sin[None]), attention_mask=mask)[0]


def window(head, X, mode, off=0):
    E = head.entries(X, torch.arange(X.shape[0]))
    head.window(E, head.trunk_blocks(E, off), mode)


@torch.no_grad()
def test_a_causal_pass_is_the_hf_layer_over_the_window():
    cfg, ref, head = build()
    S, H = cfg.hc_count, cfg.hidden_size
    for n, mode in [(9, "dense"), (9, "index"), (30, "index")]:
        X = torch.randn(n, S * H)
        window(head, X, mode)
        stack, mixed = head.first_step(torch.arange(n))
        want = hf_layer(ref, head, X)
        torch.testing.assert_close(stack, want, atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(mixed, head.mixer(want), atol=2e-5, rtol=1e-4)


@torch.no_grad()
def test_a_chain_is_the_hf_layer_over_the_window_then_the_chain():
    cfg, ref, head = build(1)
    S, H = cfg.hc_count, cfg.hidden_size
    X = torch.randn(30, S * H)
    chain = torch.randn(3, S * H)
    # Dense attention is HF's below budget + 3 = 11 visible entries; past that HF selects.
    for mode, lasts in [("dense", range(3, 8)), ("index", range(13, 19))]:
        window(head, X, mode)
        for last in lasts:              # every offset of the chain against the blocks of 4
            want = hf_layer(ref, head, torch.cat([X[: last + 1], chain]))[last + 1:]
            k = v = r = None
            for j in range(3):
                stack, mixed, (k, v, r) = head.chain_step(chain[j][None], torch.tensor([last]), k, v, r)
                torch.testing.assert_close(stack[0], want[j], atol=2e-5, rtol=1e-4)


@torch.no_grad()
def test_a_batch_of_chains_is_each_chain_alone():
    cfg, ref, head = build(2)
    S, H = cfg.hc_count, cfg.hidden_size
    X = torch.randn(30, S * H)
    window(head, X, "index")
    last = torch.tensor([13, 17, 22])
    steps = torch.randn(2, 3, S * H)
    k = v = r = None
    batched = []
    for j in range(2):
        stack, _, (k, v, r) = head.chain_step(steps[j], last, k, v, r)
        batched.append(stack)
    for b in range(3):
        k = v = r = None
        for j in range(2):
            stack, _, (k, v, r) = head.chain_step(steps[j, b][None], last[b:b + 1], k, v, r)
            torch.testing.assert_close(stack[0], batched[j][b], atol=2e-5, rtol=1e-4)


@torch.no_grad()
def test_the_combines_agree_where_they_must():
    cfg, _, head = build()
    S, H = cfg.hc_count, cfg.hidden_size
    tokens = torch.tensor([3, 7])
    stream = torch.randn(2, H)
    same = stream.repeat(1, S)
    head.norm_hidden = torch.randn(H).repeat(S) * 0.1    # one weight for every stream
    a = head.combine(same, tokens, "a", "a")
    b = head.combine(same, tokens, "b", "a")
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    # b is one state broadcast to every stream; a is per stream.
    mixed = torch.randn(2, S * H)
    assert torch.equal(head.combine(mixed, tokens, "b", "b").view(2, S, H)[:, 0],
                       head.combine(mixed, tokens, "b", "b").view(2, S, H)[:, 3])
    m = torch.randn(2, H)
    assert torch.equal(head.chain_stack(None, m, "b"), m.repeat(1, S))


@torch.no_grad()
def test_blocks_shift_with_the_offset():
    cfg, _, head = build()
    X = torch.randn(10, cfg.hc_count * cfg.hidden_size)
    E = head.entries(X, torch.arange(10))
    assert head.trunk_blocks(E, 0)["block"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert head.trunk_blocks(E, 1)["block"].tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
