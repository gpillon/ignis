"""Record the fixture that holds the Rust port of the converter's scorers to the converter's own
Python (spec flash-next/04 acceptance 5, 6 and 8; `crates/bench/src/flash_next/`).

Run from the repository root with the study's venv, CPU only:

    CUDA_VISIBLE_DEVICES= F:/ai/ngram-venv/Scripts/python.exe crates/bench/tests/fixtures/flash_next_kld/record.py

The converter's functions are imported from `tools/flash-next-converter/scoring.py`, never
restated. Each case is one position, built the way `finish.py` `_score_window` builds it:

- the reference's BF16 logits (V columns) upcast to fp32, `lse = logsumexp`, `lr = raw - lse`,
  then `scoring.top_k(lr, 64)`: the stored `ref_ids` / `ref_lp` (layout.md §10);
- the candidate's BF16 logits (the engine's row), `lc = raw - logsumexp(raw)` in fp32;
- `kl_f32 = scoring.kl_top64(ref_ids, ref_lp, lc)`: the converter's figure, in its fp32;
- `kl_f64`: the same function on the same stored inputs with `lc` in float64 (the Rust port
  computes in f64, so this one it must match to rounding);
- `cand_argmax = scoring.argmax(lc)`, `ref_argmax = scoring.argmax(lr)`.

The shapes cover a peaked reference whose tail mass is below 1e-12 (the tail term dropped), a
moderate and a flat one, a Zipf-like one, candidates near and far, and a candidate with an exact
BF16 tie at its maximum (the lowest id wins). It also records `scoring.mcnemar_p` on a few counts.

Writes, next to itself: `cases.json`, `cand_bf16.bin` (cases x V little-endian BF16 bit patterns,
in case order) and `provenance.json`. Deterministic: a rerun writes the same bytes.
"""
import hashlib
import json
import os
import platform
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "flash-next-converter"))
import scoring  # noqa: E402

V = 2048
SEED = 20261005


def bf16(x):
    return x.to(torch.bfloat16)


def case_logits(g):
    """(name, reference logits fp32, candidate logits fp32) before BF16 rounding."""
    out = []
    shapes = {
        "peaked": lambda: torch.randn(V, generator=g) + torch.nn.functional.one_hot(torch.tensor(7), V) * 45.0
        + torch.nn.functional.one_hot(torch.tensor(9), V) * 44.0,
        "confident": lambda: torch.randn(V, generator=g) * 2 + torch.nn.functional.one_hot(torch.tensor(300), V) * 14.0,
        "moderate": lambda: torch.randn(V, generator=g) * 3,
        "flat": lambda: torch.randn(V, generator=g),
        "zipf": lambda: -1.1 * torch.log(torch.randperm(V, generator=g).float() + 1) * 4,
    }
    for name, make in shapes.items():
        for sigma in (0.05, 0.3, 1.0):
            ref = make()
            out.append((f"{name}-near{sigma}", ref, ref + sigma * torch.randn(V, generator=g)))
        ref = make()
        out.append((f"{name}-independent", ref, make()))
    ref = shapes["moderate"]()
    cand = ref + 0.3 * torch.randn(V, generator=g)
    top = int(torch.argmax(cand))
    tie = top // 2                                   # a lower id holding the same maximum: it must win
    cand[tie] = cand[top]
    out.append(("moderate-tied-max", ref, cand))
    return out


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    cases, rows = [], []
    for name, ref_raw, cand_raw in case_logits(g):
        rb, cb = bf16(ref_raw)[None], bf16(cand_raw)[None]
        lr_raw = rb.float()
        lr = lr_raw - torch.logsumexp(lr_raw, -1)[:, None]
        ref_ids, ref_lp = scoring.top_k(lr, scoring.TOP)
        lc_raw = cb.float()
        lc = lc_raw - torch.logsumexp(lc_raw, -1)[:, None]
        lc64_raw = cb.double()
        lc64 = lc64_raw - torch.logsumexp(lc64_raw, -1)[:, None]
        kl_f32 = float(scoring.kl_top64(ref_ids, ref_lp, lc)[0])
        kl_f64 = float(scoring.kl_top64(ref_ids, ref_lp.double(), lc64)[0])
        tail = float(1 - ref_lp.double().exp().sum())
        cases.append({"name": name, "ref_ids": ref_ids[0].tolist(), "ref_lp": [float(v) for v in ref_lp[0]],
                      "kl_f32": kl_f32, "kl_f64": kl_f64, "ref_tail_mass": tail,
                      "cand_argmax": int(scoring.argmax(lc)[0]), "ref_argmax": int(scoring.argmax(lr)[0])})
        rows.append(cb.view(torch.int16).numpy().astype("<i2").tobytes())
    counts = [(0, 0), (10, 0), (3, 5), (25, 0), (20, 30), (12, 13), (40, 9)]
    record = {"vocab": V, "top": scoring.TOP, "cases": cases,
              "mcnemar": [{"lost": b, "gained": c, "p": scoring.mcnemar_p(b, c)} for b, c in counts]}
    with open(os.path.join(HERE, "cases.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    blob = b"".join(rows)
    with open(os.path.join(HERE, "cand_bf16.bin"), "wb") as f:
        f.write(blob)
    scoring_src = open(os.path.join(REPO, "tools", "flash-next-converter", "scoring.py"), "rb").read()
    provenance = {"generator": "crates/bench/tests/fixtures/flash_next_kld/record.py", "seed": SEED, "vocab": V,
                  "cases": len(cases), "scoring_py_sha256": hashlib.sha256(scoring_src).hexdigest(),
                  "cand_bf16_sha256": hashlib.sha256(blob).hexdigest(),
                  "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}
    with open(os.path.join(HERE, "provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(provenance, f, indent=1)
        f.write("\n")
    worst = max(abs(c["kl_f32"] - c["kl_f64"]) for c in cases)
    print(f"{len(cases)} cases, V={V}; |kl_f32 - kl_f64| max {worst:.3e}")
    for c in cases:
        print(f"  {c['name']:24s} kl_f64 {c['kl_f64']:.6e} kl_f32 {c['kl_f32']:.6e} tail {c['ref_tail_mass']:.3e}"
              f" argmax ref {c['ref_argmax']} cand {c['cand_argmax']}")


if __name__ == "__main__":
    main()
