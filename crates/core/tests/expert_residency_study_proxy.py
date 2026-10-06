"""Build the machine-local study proxy that `expert_residency_study_replay.rs` replays.

Spec flash-next/03 (GitHub #301) replays the converter's routing traces through the
residency policy model. Until the converter has run, the compression study's run-3
routing (BF16 path, `real/e2e3/routing.pt`) stands in for them. This script writes it
in the shape `docs/specs/flash-next/layout.md` gives the real artifact, so one reader
serves both:

    <out>/work/converter.json          k_map, k_classes (record bytes), expert_traffic
    <out>/traces/<domain>/manifest.json  finish.py's fields: tokens, layers, set, chunks, valid
    <out>/traces/<domain>/experts.i16   (N, 48, 10) the router's top-10 per token, layer
    <out>/traces/<domain>/lookahead.i16 (N, 48, 20) router L+1 on layer L's MoE input

The study has no K map: it allocated entropy-coded rates. The proxy allocates K in
{2, 2.5, 3, 4} per projection the way the converter will (a Lagrangian on the
distortion curves at a 2.5-bit mean including scales, gate/up and down separately),
reading the run-3 curves at those four rates by log-linear interpolation. That makes it
a stand-in for the converter's K map, not the K map itself; expert_traffic counts the
run-3 calibration tokens' selections.

CPU only, read-only on the study. Run with the study's venv:

    CUDA_VISIBLE_DEVICES= F:/ai/ngram-venv/Scripts/python.exe \
        crates/core/tests/expert_residency_study_proxy.py [--out DIR]

Default DIR: F:/ai/models/flash-next-residency-study-proxy (about 350 MB, never
committed).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

STUDY = "F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03"
LAYERS, EXPERTS, TOP_K, LOOKAHEAD = 48, 512, 10, 20
HIDDEN, INTER = 2560, 640
K = np.array([2.0, 2.5, 3.0, 4.0])
K2 = [4, 5, 6, 8]
# layout.md §3: 4 KiB-aligned records, tiles + suh + svh.
RECORD = {"gu": [827392, 1032192, 1236992, 1646592], "dn": [417792, 520192, 622592, 827392]}
# Scale bits per weight: fp16 suh + svh over the plane's weights.
OVERHEAD = {"gu": (HIDDEN + 2 * INTER) * 16 / (HIDDEN * 2 * INTER),
            "dn": (INTER + HIDDEN) * 16 / (INTER * HIDDEN)}


def allocate(r, d, en, budget):
    """K index per expert: min sum(en * D(K)) s.t. mean rate <= budget (Lagrangian)."""
    rate = np.broadcast_to(K, (r.shape[0], 4))
    dist = np.empty((r.shape[0], 4))
    for e in range(r.shape[0]):
        order = np.argsort(r[e])
        dist[e] = np.exp(np.interp(K, r[e][order], np.log(np.maximum(d[e][order], 1e-30))))
    cost_d = en[:, None] * dist

    def pick(mu):
        k = np.argmin(cost_d + mu * rate, axis=1)
        return k, rate[np.arange(len(k)), k].mean()

    lo, hi = 1e-20, 1e20
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        _, bits = pick(mid)
        lo, hi = (mid, hi) if bits > budget else (lo, mid)
    return pick(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="F:/ai/models/flash-next-residency-study-proxy")
    a = ap.parse_args()
    R = torch.load(f"{STUDY}/real/e2e3/routing.pt", weights_only=False, map_location="cpu")
    st = torch.load(f"{STUDY}/real/results/stats_run3.pt", weights_only=False, map_location="cpu")
    test = R["test"].numpy()
    kinds = R["kind_names"]
    test_kind = R["test_kind"].numpy()
    chunk_of = R["chunk_of_token"].numpy()

    k_layers, traffic, means = [], [], {"gu": [], "dn": []}
    for L in range(LAYERS):
        c = st["layers"][L]["curves"]
        row = {"layer": L}
        for proj, tag in (("gu", "g"), ("dn", "d")):
            k, mean = allocate(c[f"r_{tag}"].numpy().astype(np.float64),
                               c[f"d_{tag}"].numpy().astype(np.float64),
                               c[f"en_{tag}"].numpy().astype(np.float64),
                               2.5 - OVERHEAD[proj])
            row[proj] = [K2[i] for i in k]
            means[proj].append(mean + OVERHEAD[proj])
        k_layers.append(row)
        idx = R["idx"][L].numpy()
        traffic.append({"layer": L, "counts": np.bincount(idx[~test].ravel().astype(np.int64),
                                                          minlength=EXPERTS).tolist()})
        print(f"layer {L}: gu {means['gu'][-1]:.3f} dn {means['dn'][-1]:.3f} b/w", flush=True)

    k_classes = []
    for proj in ("gu", "dn"):
        for i, k in enumerate(K):
            n = sum(row[proj].count(K2[i]) for row in k_layers)
            k_classes.append({"class": f"{proj}-{k:g}", "projections": n,
                              "record_bytes": RECORD[proj][i], "traffic_share": None})
    os.makedirs(f"{a.out}/work", exist_ok=True)
    with open(f"{a.out}/work/converter.json", "w", encoding="utf-8") as f:
        json.dump({"schema": "flash-next-converter-v1",
                   "status": "study-proxy",
                   "proxy": {"routing": "study run 3, BF16 path, real/e2e3/routing.pt",
                             "k_map": "Lagrangian over K in {2, 2.5, 3, 4} at 2.5 b/w incl. "
                                      "scales on run-3 curves, log-linear interpolation",
                             "mean_bits": {p: float(np.mean(v)) for p, v in means.items()}},
                   "k_map": {"layers": k_layers},
                   "k_classes": k_classes,
                   "expert_traffic": {"layers": traffic}}, f)

    test_tok = np.nonzero(test)[0]
    experts = np.stack([R["idx"][L].numpy()[test_tok] for L in range(LAYERS)], axis=1)
    look = np.full((len(test_tok), LAYERS, LOOKAHEAD), -1, dtype=np.int16)
    for L in range(LAYERS - 1):
        look[:, L, :] = R["pred"][L][1].numpy()[:, :LOOKAHEAD]
    for kid, kind in enumerate(kinds):
        rows = np.nonzero(test_kind == kid)[0]
        if rows.size == 0:
            continue
        chunks = sorted(set(chunk_of[test_tok[rows]].tolist()))
        valid = [int((chunk_of[test_tok[rows]] == ch).sum()) for ch in chunks]
        out = f"{a.out}/traces/{kind}"
        os.makedirs(out, exist_ok=True)
        experts[rows].astype("<i2").tofile(f"{out}/experts.i16")
        look[rows].astype("<i2").tofile(f"{out}/lookahead.i16")
        with open(f"{out}/manifest.json", "w", encoding="utf-8") as f:
            # finish.py's manifest, field for field (layout.md §12).
            json.dump({"tokens": int(rows.size), "layers": LAYERS, "set": "test",
                       "chunks": chunks, "valid": valid,
                       "stream": "bf16 (study run 3, proxy)",
                       "order": "token-major (N, layers, k)"}, f)
        print(f"{kind}: {rows.size} tokens, {len(chunks)} chunks", flush=True)
    print("mean b/w incl. scales:", {p: round(float(np.mean(v)), 4) for p, v in means.items()})


if __name__ == "__main__":
    sys.exit(main())
