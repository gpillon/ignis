"""Sum an index row into the residual stream, at the positions that matched.

`02-storage.md` §7 Fase 2 fixes the form:

    residual += alpha * ||h_t|| * v / ||v||

and says why it is not `alpha * v`: a mean-pooled row has a fraction of the
norm of a single hidden, and residual norms grow with depth, so a bare alpha
would make the sweep measure norms instead of the effect.  Scaled this way,
alpha is a declared fraction of the local residual amplitude and is comparable
across layers and poolings.

The gate (§6) has no trained parameters here.  Selectivity is the cosine
between the local hidden and the row, thresholded — the "occupancy tracking as
a lightweight selectivity prior" escape route — plus the rare-key filter
applied when the match set is built.
"""

import torch

from modeladapter import decoder_layers, layer_output, rewrap_layer_output


class Injector:
    """Adds rows at fixed token positions in the output of one decoder layer.

    `positions` and `vectors` line up: position[i] receives vectors[i].  The
    same position may appear twice only if two keys matched there, which the
    match builder does not produce.
    """

    def __init__(self, model, layer, positions, vectors, alpha,
                 cos_threshold=None, force=False, center=None):
        self.layer = decoder_layers(model)[layer]
        self.positions = positions
        self.vectors = vectors
        self.alpha = alpha
        self.cos_threshold = cos_threshold
        # `center` is the mean of the index rows.  Without it the injection is
        # mostly not the symbol: Fase 0a measured a raw cosine of 0.931 between
        # *any* two L19 hidden states, so a unit-normalised row is 0.965 along
        # the layer's shared (massive-activation) direction and only 0.26 along
        # the part that distinguishes one symbol from another.  Normalising the
        # raw row therefore spends almost all of alpha amplifying a direction
        # `h_t` already has.  Subtracting the mean first makes alpha a fraction
        # of the *distinctive* component, which is what §7's formula meant.
        self.center = center
        # `force` runs the hook even at alpha = 0.  That is the §7 Fase 3
        # control: a zero-amplitude injection must reproduce the baseline
        # exactly, which only tests anything if the clone-and-index_add path
        # actually runs.  It is also how the cosines are collected.
        self.force = force
        self.injected = 0
        self.cosines = None
        self._handle = None

    def __enter__(self):
        if (self.alpha != 0.0 or self.force) and len(self.positions):
            self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def _hook(self, _module, _args, out):
        h = layer_output(out)
        local = h[0].index_select(0, self.positions).float()
        v = self.vectors.float()
        local_norm = local.norm(dim=1, keepdim=True)

        if self.center is None:
            v_dir = v
            q = local
        else:
            mu = self.center.to(local.device).float()
            v_dir = v - mu
            q = local - mu          # the gate has to compare like with like
        v_unit = v_dir / v_dir.norm(dim=1, keepdim=True).clamp_min(1e-9)
        cos = (q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)
               * v_unit).sum(dim=1)
        self.cosines = cos.detach().cpu()

        delta = local_norm * v_unit * self.alpha
        if self.cos_threshold is not None:
            delta = delta * (cos >= self.cos_threshold).unsqueeze(1)
            self.injected = int((cos >= self.cos_threshold).sum())
        else:
            self.injected = int(len(self.positions))

        h = h.clone()
        h[0] = h[0].index_add(0, self.positions, delta.to(h.dtype))
        return rewrap_layer_output(out, h)


@torch.no_grad()
def cosine_probe(model, layer, positions, vectors, input_ids, center=None):
    """Cosines between the local hidden and the matched rows, with no injection
    — what the median and high thresholds of the sweep are derived from."""
    with Injector(model, layer, positions, vectors, alpha=0.0,
                  force=True, center=center) as probe:
        model.model(input_ids=input_ids, use_cache=False)
    return probe.cosines
