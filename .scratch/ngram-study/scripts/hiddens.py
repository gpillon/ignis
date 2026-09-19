"""Read hidden states at chosen layers, for chosen character spans.

`output_hidden_states=True` would keep every layer for every position: on the
27B that is 65 x T x 5120 x 2 B, about 3.3 GB at T=5k, on top of the weights.
So this registers forward hooks on just the layers being swept and keeps only
the positions asked for.

The unit of interest is a character span in the source file (a symbol name, a
definition body), which has to become token positions.  `offset_mapping` from
the fast tokenizer is what bridges the two, so every span here is expressed in
characters and converted once.
"""

import torch

from modeladapter import decoder_layers, layer_output


class LayerTap:
    """Capture the output of the given decoder layers during one forward pass.

    Only the positions in `keep` are retained, so a whole file can be run while
    holding a few hundred vectors rather than a few thousand.
    """

    def __init__(self, model, layers, keep=None, dtype=torch.float32):
        self.layers = list(layers)
        self.keep = keep
        self.dtype = dtype
        self.captured = {}
        self._handles = []
        self._model_layers = decoder_layers(model)

    def __enter__(self):
        for idx in self.layers:
            self._handles.append(
                self._model_layers[idx].register_forward_hook(self._make(idx)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def _make(self, idx):
        def hook(_module, _args, out):
            h = layer_output(out)[0]               # batch of one
            if self.keep is not None:
                h = h.index_select(0, self.keep)
            self.captured[idx] = h.detach().to(self.dtype).cpu()
        return hook


def token_spans(offsets, char_span):
    """Token indices whose character range overlaps `char_span`.

    A BPE token straddles character boundaries (` decode_round` is one token
    that starts at the space), so overlap is the right relation, not
    containment: the last token of a name is the last token that overlaps it.
    """
    start, end = char_span
    out = []
    for i, (a, b) in enumerate(offsets):
        if a == b:                                 # special tokens carry (0, 0)
            continue
        if a < end and b > start:
            out.append(i)
    return out


def encode_file(tokenizer, text, max_tokens=None):
    """Tokenize a whole source file, keeping the character offsets."""
    enc = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True, truncation=max_tokens is not None,
                    max_length=max_tokens)
    return enc["input_ids"], enc["offset_mapping"]


def pool(vectors, how):
    """`02-storage.md` §7 Fase 3 sweeps two poolings.

    'last' is the hidden at the final token of the definition, the one a causal
    model computed having seen the whole body.  'mean' averages the body, which
    is the escape route if 'last' turns out to be collapsed onto "predict the
    next blank line" (§3).
    """
    if how == "last":
        return vectors[-1]
    if how == "mean":
        return vectors.mean(dim=0)
    raise ValueError("unknown pooling %r" % how)


@torch.no_grad()
def hiddens_for_spans(model, tokenizer, text, layers, spans, poolings,
                      device=None, max_tokens=None):
    """One forward pass over `text`; returns {(layer, pooling): tensor[len(spans), H]}.

    `spans` are character ranges.  A span with no tokens (an empty body, or one
    truncated away) yields a row of NaN so the caller can drop it explicitly
    rather than silently shifting the index.
    """
    ids, offsets = encode_file(tokenizer, text, max_tokens=max_tokens)
    per_span = [token_spans(offsets, s) for s in spans]
    wanted = sorted({t for ts in per_span for t in ts})
    if not wanted:
        return {}, per_span
    index = {t: i for i, t in enumerate(wanted)}
    device = device or next(model.parameters()).device
    keep = torch.tensor(wanted, dtype=torch.long, device=device)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # `model.model`, not `model`: the language-model stack alone.  Calling the
    # wrapper would run `lm_head` over every position, and at 4096 positions
    # against a 248320 vocabulary that is a 2 GB logits tensor per forward,
    # allocated and thrown away — which is both wasted time and the allocator
    # drift that pushes the reservation into WDDM paging territory.  Nothing
    # here ever reads a logit.
    with LayerTap(model, layers, keep=keep) as tap:
        model.model(input_ids=input_ids, use_cache=False)

    # Windows has no `expandable_segments`, so the caching allocator cannot
    # reshape a segment when the next file has a different length: the
    # reservation drifts from a ~19 GiB working set up past 32 GiB, and the
    # card starts paging under WDDM.  Returning the segments after each file
    # costs microseconds against a 2 s forward and keeps the reservation flat.
    torch.cuda.empty_cache()

    out = {}
    for layer, h in tap.captured.items():
        for how in poolings:
            rows = torch.full((len(spans), h.shape[-1]), float("nan"))
            for i, ts in enumerate(per_span):
                if ts:
                    rows[i] = pool(h.index_select(
                        0, torch.tensor([index[t] for t in ts])), how)
            out[(layer, how)] = rows
    return out, per_span
