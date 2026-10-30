"""Each trellis matrix's relative reconstruction error against the BF16 checkpoint (the only GPU
work, and the only exllamav3 user).

The weight a record stands for is exllamav3's own reconstruction (`LinearEXL3.get_weight_tensor`,
the oracle of ADR 0060 §5 and of M1's harness), `[in, out]` fp16; the error is
`‖W − W_bf16‖_F / ‖W_bf16‖_F` with `W_bf16` the checkpoint's `<key>.weight`, summed in fp64 over
row blocks so the 248,320-row head never needs a second full-size copy. A matrix above the bound
(0.6, M1's) is a layout error, never a quantization one: SC_3.00bpw_H4's worst is 0.34.
"""
import json
import os

ERROR_BOUND = 0.6
ROW_BLOCK = 16384


def rel_errors(quant, bf16_root, planned, device="cuda", log=print, override=None):
    """{object name: relative error} for every planned object built from a quant matrix. A matrix
    the `override` replaces is read from the override and measured against its reference BF16."""
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from safetensors import safe_open

    def weight_map(root):
        with open(os.path.join(root, "model.safetensors.index.json"), encoding="utf-8") as f:
            return json.load(f)["weight_map"]

    maps = {bf16_root: weight_map(bf16_root)}
    if override is not None:
        maps[override.reference_root] = weight_map(override.reference_root)
    q_handles, b_handles = {}, {}

    def tensor(handles, root, shard, name):
        if (root, shard) not in handles:     # an override's shards may share the quant's names
            handles[root, shard] = safe_open(os.path.join(root, shard), framework="pt", device="cpu")
        return handles[root, shard].get_tensor(name)

    errors = {}
    for p in planned:
        if p.key is None:
            continue
        swapped = override is not None and p.key in override.matrices
        source, reference_root = (override.quant, override.reference_root) if swapped else (quant, bf16_root)
        bf16_map = maps[reference_root]
        # The head alone peaks at 13.4 GB (18.5 GB reserved): with the blocks the matrices before it
        # left cached, the allocator ran short and the process died on 2026-10-30 (#422).
        torch.cuda.empty_cache()
        q = {s: tensor(q_handles, source.root, source.weight_map[f"{p.key}.{s}"], f"{p.key}.{s}").to(device)
             for s in ("trellis", "suh", "svh", "mul1")}
        rows, columns = p.shape
        lin = LinearEXL3(None, columns, rows, suh=q["suh"], svh=q["svh"], trellis=q["trellis"], mul1=q["mul1"],
                         key=p.key)
        w = lin.get_weight_tensor()                                   # [in, out] fp16
        ref_name = f"{p.key}.weight"
        ref = tensor(b_handles, reference_root, bf16_map[ref_name], ref_name)  # [out, in] bf16
        if list(ref.shape) != [rows, columns] or list(w.shape) != [columns, rows]:
            raise ValueError(f"{p.key}: reconstruction {list(w.shape)} / checkpoint {list(ref.shape)} "
                             f"do not match [{rows}, {columns}]")
        diff2 = norm2 = 0.0
        for r0 in range(0, rows, ROW_BLOCK):
            r = ref[r0:r0 + ROW_BLOCK].to(device).float()
            d = w[:, r0:r0 + ROW_BLOCK].t().float() - r
            diff2 += float(d.double().pow(2).sum())
            norm2 += float(r.double().pow(2).sum())
        errors[p.name] = (diff2 / norm2) ** 0.5
        del q, lin, w, ref
        log(f"{p.name}: K {p.format[len('TRELLIS_MUL1_K'):]}, rel error {errors[p.name]:.4f}")
    torch.cuda.empty_cache()
    return errors
