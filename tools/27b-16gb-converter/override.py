"""An override directory (GitHub #420, spec #386): EXL3 records that replace the quant's records of
the same name, so a variant made from another BF16 (the abliterated one) is built like the quant.

The directory is an EXL3 checkpoint in the quant's layout (index, shards, `quantization_config.json`)
holding only the matrices it replaces. A record swaps like for like: its name must be one of the
quant's text matrices, with the same K and the same shape. Its reconstruction error is measured
against `reference`, the BF16 checkpoint the override was made from, not against the base BF16.
"""
import exl3


class Override:
    """`root`: the override's EXL3 directory, from repository `repo`; `reference_root`: the BF16
    checkpoint it was made from, from repository `reference_repo`."""

    def __init__(self, root, repo, reference_root, reference_repo):
        self.root, self.repo = str(root), repo
        self.reference_root, self.reference_repo = reference_root, reference_repo
        self.quant = exl3.Quant(self.root)
        self.matrices = {key: self.quant.matrix(key) for key in self.quant.keys()}

    def problems(self, base):
        """What keeps this override from swapping into the quant's matrices `base` (key -> Matrix):
        one line per record, naming it and what differs."""
        out = []
        if not self.matrices:
            out.append("the override holds no EXL3 record")
        for key, m in sorted(self.matrices.items()):
            b = base.get(key)
            if b is None:
                out.append(f"{key}: the quant has no record of that name")
            elif m.k2 != b.k2:
                out.append(f"{key}: K {m.k2 / 2:g}, the quant's is {b.k2 / 2:g}")
            elif m.shape != b.shape:
                out.append(f"{key}: shape {m.shape}, the quant's is {b.shape}")
        return out

    def record(self, key):
        return self.quant.record(self.matrices[key])
